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
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/SmoothingLength.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/sph/EquationOfState.hpp"
#include "spearhed/sph/SphKernel.hpp"
#include "spmacc/particles/algorithms/InteractionContext.hpp"
#include "spmacc/topology/Vec.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace spearhed
{
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
     * No viscosity (that is Phase 5). Accumulates into ownParticle's dvdt and dudt:
     *   dv_i/dt -= m_j * (P_i/rho_i^2 * gradW(r_ij, h_i) + P_j/rho_j^2 * gradW(r_ij, h_j))
     *   du_i/dt += P_i/rho_i^2 * m_j * dot(v_i - v_j, gradW(r_ij, h_i))
     *
     * omega factors (Phase 6 grad-h correction) are all 1 here (constant h).
     *
     * The caller is responsible for zeroing dvdt/dudt before this pass (ZeroDerivatives).
     */
    template<SphKernel KernelT>
    struct AccumulateMomentumAndEnergy
    {
        typename CS::T_Axis gamma;

        HDINLINE constexpr void operator()(
            auto& /*worker*/,
            auto& ownParticle,
            auto const& ownVolume,
            auto& neighbourParticle,
            auto const& neighbourVolume) const
        {
            using namespace spearhed::tags;
            using namespace pmacc::spearhed::tags;
            using T = typename CS::T_Axis;
            using Vec = pmacc::spearhed::Vec<CS, pmacc::spearhed::ValueStorage<CS>>;

            if(*ownParticle[particleId] == *neighbourParticle[particleId])
                return;

            auto const ownAbsPos = ownVolume.getPosition(ownParticle[relativePos].get());
            auto const neighAbsPos = neighbourVolume.getPosition(neighbourParticle[relativePos].get());

            Vec r_vec;
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { r_vec[tag] = ownAbsPos[tag] - neighAbsPos[tag]; });

            T r{0};
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { r += r_vec[tag] * r_vec[tag]; });
            r = static_cast<T>(std::sqrt(static_cast<double>(r)));

            T const h_i = *ownParticle[smoothingLength];
            T const h_j = *neighbourParticle[smoothingLength];

            Vec const gW_i = KernelT::gradW(ctx.r_vec, r, h_i);
            Vec const gW_j = KernelT::gradW(ctx.r_vec, r, h_j);

            T const rho_i = *ownParticle[density];
            T const rho_j = *neighbourParticle[density];
            T const u_i = *ownParticle[internalEnergy];
            T const u_j = *neighbourParticle[internalEnergy];
            T const m_j = *neighbourParticle[mass];

            T const P_i = pressure(gamma, rho_i, u_i);
            T const P_j = pressure(gamma, rho_j, u_j);

            T const rho_i2 = rho_i * rho_i;
            T const rho_j2 = rho_j * rho_j;

            // Symmetric pressure gradient: dv_i -= m_j*(P_i/rho_i^2*gW_i + P_j/rho_j^2*gW_j)
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                { *ownParticle[dvdt][tag] -= m_j * (P_i / rho_i2 * gW_i[tag] + P_j / rho_j2 * gW_j[tag]); });

            // P*dV energy: du_i += P_i/rho_i^2 * m_j * dot(v_i - v_j, gW_i)
            T dot_v_gW{0};
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    T const dv = *ownParticle[vel][tag] - *neighbourParticle[vel][tag];
                    dot_v_gW += dv * gW_i[tag];
                });
            *ownParticle[dudt] += P_i / rho_i2 * m_j * dot_v_gW;
        }
    };

} // namespace spearhed
