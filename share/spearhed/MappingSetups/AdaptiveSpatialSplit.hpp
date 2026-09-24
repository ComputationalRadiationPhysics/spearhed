/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spearhed/param/setup.hpp"
#include "spmacc/particles/regions/mapping/AdaptiveSplit/Policies.hpp"

namespace spearhed::mapping_setups
{
    struct SpatialSplitTrigger
    {
        auto operator()(auto const&) const
        {
            return pmacc::spearhed::ParticleCountExceeds{128u};
        }
    };

    struct SpatialSplitPartition
    {
        auto operator()(auto const&) const
        {
            return pmacc::spearhed::WorldAxisMidpoint<CS, pmacc::spearhed::tags::x_t>{};
        }
    };

    /** @brief Particle-count-triggered binary split along the world x axis. */
    struct AdaptiveSpatialSplitSetup : DefaultSetup
    {
        using DecompositionGroups = std::tuple<pmacc::spearhed::AdaptiveSplitDecompositionGroup<
            SpatialSplitTrigger,
            SpatialSplitPartition,
            pmacc::spearhed::species::Default,
            pmacc::spearhed::species::Boundary,
            pmacc::spearhed::species::Tracer>>;
    };
} // namespace spearhed::mapping_setups
