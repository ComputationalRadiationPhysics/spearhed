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
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
#include "spmacc/particles/algorithms/InteractionContext.hpp"
#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"

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
     * r and is_self are provided by InteractParticles; self-contribution
     * is seeded by DensityInitSelf before the pairwise pass.
     */
    template<SphKernel KernelT>
    struct AccumulateDensity
    {
        using RequiredSharedTags = ll::TagList<tags::mass, tags::smoothingLength>;
        using RequiredOwnTags = ll::TagList<tags::smoothingLength, tags::density>;

        HDINLINE constexpr void operator()(
            auto& /*worker*/,
            auto& ownParticle,
            auto& neighbourParticle,
            pmacc::spearhed::InteractionContext<CS> const& ctx) const
        {
            using namespace spearhed::tags;

            if(ctx.is_self) [[unlikely]]
                return;

            typename CS::T_Axis const h = *ownParticle[smoothingLength];
            typename CS::T_Axis const m_j = *neighbourParticle[mass];

            *ownParticle[density] += m_j * KernelT::W(ctx.r(), h);
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

            typename CS::T_Axis const h = *particle[smoothingLength];
            typename CS::T_Axis const m_i = *particle[mass];

            *particle[density] = m_i * KernelT::W(typename CS::T_Axis{0}, h);
        }
    };

    /**
     * Stage functor: full density summation pass.
     *
     * Seeds each particle with its self-contribution, then accumulates
     * neighbour contributions via the pairwise pass.
     */
    template<SphKernel KernelT>
    struct UpdateDensity
    {
        void operator()(auto& targetPRBuf, auto&& sourcePRBufTuple, auto&& neighbourListsTuple, typename CS::T_Axis h0)
            const
        {
            pmacc::spearhed::ForEachParticleInPRBuf{}(targetPRBuf, DensityInitSelf<KernelT>{});
            pmacc::spearhed::InteractParticles{}(
                targetPRBuf,
                std::forward<decltype(sourcePRBufTuple)>(sourcePRBufTuple),
                std::forward<decltype(neighbourListsTuple)>(neighbourListsTuple),
                static_cast<typename CS::T_Axis>(KernelT::supportRadius) * h0,
                AccumulateDensity<KernelT>{});
        }
    };

} // namespace spearhed
