/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/RegionChart.hpp"
#include "spmacc/particles/regions/WorldAABB.hpp"

namespace pmacc::spearhed
{
    /** @brief Material-bucket spatial metadata: a chart and its world-space occupancy. */
    template<CoordinateSystem CS>
    struct MaterialRegionMetadata
    {
        using Chart = RegionChart<CS>;
        using Bounds = WorldAABB<CS>;
        using Pnt = typename Bounds::Pnt;
        using Vec = typename Bounds::Vec;

        constexpr MaterialRegionMetadata() = default;

        constexpr MaterialRegionMetadata(Chart const& chart, Bounds const& occupancy)
            : chart(chart)
            , occupancy(occupancy)
        {
        }

        /** @brief Compatibility conversion for setup's legacy chart-relative AABB. */
        constexpr MaterialRegionMetadata(AABB<CS> const& legacy)
            : chart(legacy.origin)
            , occupancy(chart.toWorld(legacy.min), chart.toWorld(legacy.max))
        {
        }

        [[nodiscard]] constexpr Vec localMin() const
        {
            return chart.toLocal(occupancy.min);
        }

        [[nodiscard]] constexpr Vec localMax() const
        {
            return chart.toLocal(occupancy.max);
        }

        Chart chart{};
        Bounds occupancy{};
    };
} // namespace pmacc::spearhed
