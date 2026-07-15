/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Acceleration.hpp"
#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/DuDt.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/SmoothingLength.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/sph/EquationOfState.hpp"
#include "spearhed/sph/SphKernel.hpp"
#include "spmacc/particles/algorithms/InteractParticles.hpp"
#include "spmacc/particles/algorithms/InteractionContext.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace spearhed
{
    namespace tags
    {
        //! Derived neighbour quantity P_j / rho_j^2, staged once per neighbour by HydroInteraction.
        DEFINE_TAG(pOverRho2);
    } // namespace tags

    /**
     * Per-particle functor: zero dvdt and dudt before the pairwise accumulation pass.
     */
    struct ZeroDerivatives
    {
        HDINLINE constexpr void operator()(auto& /*worker*/, auto& particle) const
        {
            using namespace spearhed::tags;

            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { *particle[dvdt][tag] = typename CS::T_Axis{0}; });
            *particle[dudt] = typename CS::T_Axis{0};
        }
    };

    /**
     * Pairwise interaction functor: symmetric SPH pressure-gradient and P*dV energy.
     *
     * No viscosity. Accumulates into ownParticle's dvdt and dudt:
     *   dv_i/dt -= m_j * (P_i/rho_i^2 * gradW(r_ij, h_i) + P_j/rho_j^2 * gradW(r_ij, h_j))
     *   du_i/dt += P_i/rho_i^2 * m_j * dot(v_i - v_j, gradW(r_ij, h_i))
     *
     * The symmetric gradient is a scalar multiple of r_ij, so both gradients are evaluated in
     * scalar form via KernelT::gradWScalar (gradW(r_ij, h) == gradWScalar(r, invR, h) * r_ij); the
     * per-pair vector temporaries and the reconstruction division 1/r are thereby removed.
     *
     * Two neighbour-independent hoists:
     *   - prepare() computes the own-side invariant P_i/rho_i^2 once per own particle.
     *   - stage() computes the neighbour-side invariant P_j/rho_j^2 (one pressure() and one
     *     division) once per staged neighbour, into the derived StagedRecord, instead of once per
     *     pair. The pressure work is thus hoisted out of the O(N*M) inner loop on both sides.
     *
     * omega factors (grad-h correction) are all 1 here (constant h).
     *
     * The caller is responsible for zeroing dvdt/dudt before this pass (ZeroDerivatives).
     *   - neighbourReads: m_j, h_j, rho_j, u_j, v_j (the globals stage() reads).
     *   - ownReads:       h_i, rho_i, u_i, v_i.
     *   - ownAccumulate:  dvdt, dudt.
     */
    template<SphKernel KernelT>
    struct HydroInteraction
    {
        typename CS::T_Axis gamma;

        static constexpr auto neighbourReads
            = ll::makeSet(tags::mass, tags::smoothingLength, tags::density, tags::internalEnergy, tags::vel);
        static constexpr auto ownReads
            = ll::makeSet(tags::smoothingLength, tags::density, tags::internalEnergy, tags::vel);
        static constexpr auto ownAccumulate = ll::makeSet(tags::dvdt, tags::dudt);

        //! Derived neighbour record staged once per neighbour: P_j/rho_j^2, m_j, h_j, v_j.
        using StagedRecord = ll::Record<
            ll::Field<tags::pOverRho2_t, typename CS::T_Axis>,
            tags::massField<typename CS::T_Axis>,
            tags::smoothingLengthField<typename CS::T_Axis>,
            tags::velField<CS>>;

        //! Own-side invariants, computed once per particle by prepare().
        struct Prepared
        {
            //! P_i / rho_i^2, reused for both the pressure-gradient and the P*dV energy terms.
            typename CS::T_Axis pOverRho2;
        };

        //! Hoist the neighbour-independent own-side pressure work out of the pairwise loop.
        HDINLINE constexpr Prepared prepare(auto const& ownRead) const
        {
            using namespace spearhed::tags;
            using T = typename CS::T_Axis;

            T const rho_i = *ownRead[density];
            T const u_i = *ownRead[internalEnergy];
            T const P_i = pressure(gamma, rho_i, u_i);
            return Prepared{P_i / (rho_i * rho_i)};
        }

        //! Hoist the neighbour-independent P_j/rho_j^2 into SMEM, once per staged neighbour.
        HDINLINE constexpr void stage(auto const& nParticle, auto staged) const
        {
            using namespace spearhed::tags;
            using T = typename CS::T_Axis;

            T const rho_j = *nParticle[density];
            T const u_j = *nParticle[internalEnergy];
            T const P_j = pressure(gamma, rho_j, u_j);

            *staged[pOverRho2] = P_j / (rho_j * rho_j);
            *staged[mass] = *nParticle[mass];
            *staged[smoothingLength] = *nParticle[smoothingLength];
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { *staged[vel][tag] = *nParticle[vel][tag]; });
        }

        HDINLINE constexpr void operator()(
            auto& /*worker*/,
            auto const& ownRead,
            Prepared const& prep,
            auto const& nb,
            pmacc::spearhed::PairContext<CS> const& ctx,
            auto& acc) const
        {
            using namespace spearhed::tags;
            using T = typename CS::T_Axis;

            if(ctx.isSelf) [[unlikely]]
                return;

            T const h_i = *ownRead[smoothingLength];
            T const h_j = *nb[smoothingLength];

            // Scalar gradient factors: gW_i == s_i * rVec, gW_j == s_j * rVec (single rsqrt in ctx).
            T const s_i = KernelT::gradWScalar(ctx.r, ctx.invR, h_i);
            T const s_j = KernelT::gradWScalar(ctx.r, ctx.invR, h_j);

            T const m_j = *nb[mass];
            T const pOverRho2_j = *nb[pOverRho2];

            // Symmetric pressure gradient: dv_i -= m_j*(P_i/rho_i^2*gW_i + P_j/rho_j^2*gW_j)
            T const term_i = m_j * prep.pOverRho2;
            T const term_j = m_j * pOverRho2_j;

            pmacc::spearhed::for_each_tag<CS>([&](auto tag)
                                              { *acc[dvdt][tag] -= (term_i * s_i + term_j * s_j) * ctx.rVec[tag]; });

            // P*dV energy: du_i += P_i/rho_i^2 * m_j * dot(v_i - v_j, gW_i), gW_i = s_i * rVec.
            T dot_v_rVec{0};
            pmacc::spearhed::for_each_tag<CS>([&](auto tag)
                                              { dot_v_rVec += (*ownRead[vel][tag] - *nb[vel][tag]) * ctx.rVec[tag]; });
            *acc[dudt] += term_i * s_i * dot_v_rVec;
        }
    };

    /**
     * Stage functor: full pressure-gradient and P*dV energy pass.
     *
     * Zeros dvdt/dudt, then accumulates pairwise pressure forces and energy exchange.
     */
    template<SphKernel KernelT>
    struct UpdateHydroForces
    {
        typename CS::T_Axis gamma;

        /** Requires a caller-built FrameIndexBuffer for the target, which can be cached across passes
         *  and timesteps while the frame-list topology is unchanged. The same index also drives the
         *  zeroing launch below, so no separate index build/scan is paid for that pass either.
         *
         *  Asynchronous: returns the combined EventTask of both enqueued launches; the bundle, target
         *  and index must outlive kernel completion (see interact()'s lifetime contract). */
        [[nodiscard]] pmacc::EventTask operator()(
            pmacc::spearhed::IsNeighbourBundle auto&& neighbourBundle,
            auto& target,
            auto& index,
            typename CS::T_Axis h0) const
        {
            // PMacc transaction ordering runs this zeroing kernel before the interaction kernels
            // enqueued by interact() below on the device queue, so no host wait is needed between
            // them -- only the caller's eventual wait on the combined event.
            auto zeroDone
                = pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, target, index, ZeroDerivatives{});

            auto sources = neighbourBundle.template selectByRole<pmacc::spearhed::roles::Source>();
            return zeroDone
                   + pmacc::spearhed::interact(
                       sources,
                       target,
                       index,
                       static_cast<typename CS::T_Axis>(KernelT::supportRadius) * h0,
                       HydroInteraction<KernelT>{gamma});
        }
    };

} // namespace spearhed
