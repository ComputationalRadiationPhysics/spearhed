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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/SmoothingLength.hpp"
#include "spearhed/sph/SphKernel.hpp"
#include "spmacc/particles/algorithms/InteractParticles.hpp"
#include "spmacc/particles/algorithms/InteractionContext.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <llamaLite/tag/TagPath.hpp>

namespace spearhed
{
    /**
     * Pairwise interaction functor for SPH density summation.
     *
     * Accumulates the neighbour contribution into ownParticle's density:
     *   rho_i += m_j * W(r, h_i)
     *
     * r and isSelf are provided in the PairContext; the self-contribution is seeded by
     * DensityInitSelf before the pairwise pass and resumed by the framework accumulator.
     *   - neighbourReads: only the neighbour mass m_j (h comes from the own side).
     *   - ownReads:       the own smoothing length h_i.
     *   - ownAccumulate:  the density accumulator rho_i.
     */
    template<SphKernel KernelT>
    struct AccumulateDensity
    {
        static constexpr auto neighbourReads = ll::makeSet(tags::mass);
        static constexpr auto ownReads = ll::makeSet(tags::smoothingLength);
        static constexpr auto ownAccumulate = ll::makeSet(tags::density);

        HDINLINE constexpr void operator()(
            auto& /*worker*/,
            auto const& ownRead,
            auto const& nb,
            pmacc::spearhed::PairContext<CS> const& ctx,
            auto& acc) const
        {
            using namespace spearhed::tags;

            if(ctx.isSelf) [[unlikely]]
                return;

            typename CS::T_Axis const h = ownRead[smoothingLength];
            typename CS::T_Axis const m_j = nb[mass];

            acc[density] += m_j * KernelT::W(ctx.r, h);
        }
    };

    /**
     * Per-particle functor: resets density to zero then adds the self-contribution
     *   rho_i = m_i * W(0, h_i).
     *
     * Run this BEFORE the pairwise pass so that densities start from the self term.
     */
    template<SphKernel KernelT>
    struct DensityInitSelf
    {
        HDINLINE constexpr void operator()(auto& /*worker*/, auto& particle) const
        {
            using namespace spearhed::tags;

            typename CS::T_Axis const h = particle[smoothingLength];
            typename CS::T_Axis const m_i = particle[mass];

            particle[density] = m_i * KernelT::W(typename CS::T_Axis{0}, h);
        }
    };

    /**
     * Stage functor: full density summation pass.
     *
     * Seeds each particle with its self-contribution, then accumulates
     * neighbour contributions via the pairwise pass.
     *
     * @tparam KernelT    SPH smoothing kernel.
     */
    template<SphKernel KernelT>
    struct UpdateDensity
    {
        /** Requires a caller-built FrameIndexBuffer for the target, which can be cached across passes
         *  and timesteps while the frame-list topology is unchanged. The same index also drives the
         *  self-init launch below, so no separate index build/scan is paid for that pass either.
         *
         *  Returns a lazy composed sender. The bundle, target, index, and any device arguments must
         *  outlive sender completion (see interact()'s lifetime contract). */
        [[nodiscard]] auto operator()(
            pmacc::spearhed::IsNeighbourBundle auto&& neighbourBundle,
            auto& target,
            auto& index,
            typename CS::T_Axis h0) const
        {
            // Sequence on the native queue: pairwise accumulation must observe self density.
            auto zeroDone = pmacc::spearhed::launchForEach(
                pmacc::spearhed::levels::particle,
                target,
                index,
                DensityInitSelf<KernelT>{});

            // Preserve the self-init when interact() is a no-op for an empty source bundle. Native
            // sequencing keeps dependent work ordered without a host-side synchronization.
            auto sources = neighbourBundle.template selectByRole<pmacc::spearhed::roles::Source>();
            return std::move(zeroDone)
                   | caravan::alpaka::sequence(
                       pmacc::spearhed::interact(
                           sources,
                           target,
                           index,
                           static_cast<typename CS::T_Axis>(KernelT::supportRadius) * h0,
                           AccumulateDensity<KernelT>{}));
        }
    };

} // namespace spearhed
