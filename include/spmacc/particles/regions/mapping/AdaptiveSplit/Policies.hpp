/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cstdint>

namespace pmacc::spearhed
{
    /** @brief Select regions whose packed live-particle count exceeds @c threshold. */
    struct ParticleCountExceeds
    {
        uint32_t threshold;

        DINLINE bool shouldSplitRegion(auto const&, auto const& region) const
        {
            return region.particleFrameList.getNumParticles() > threshold;
        }
    };

    /** @brief Binary partition at the current occupancy midpoint of one world-space axis. */
    template<CoordinateSystem CS, typename T_AxisTag>
    struct WorldAxisMidpoint
    {
        static constexpr uint32_t partitionCount = 2u;

        HDINLINE uint32_t operator()(auto particle, auto const& metadata) const
        {
            auto const worldPosition = metadata.chart.toWorld(particle[tags::relativePos].get());
            auto const midpoint = (metadata.occupancy.min[T_AxisTag{}] + metadata.occupancy.max[T_AxisTag{}]) *
                                  typename CS::T_Axis{0.5};
            return worldPosition[T_AxisTag{}] < midpoint ? 0u : 1u;
        }
    };
} // namespace pmacc::spearhed
