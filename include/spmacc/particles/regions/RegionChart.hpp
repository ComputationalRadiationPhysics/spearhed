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

#include "spmacc/topology/CartesianStorage.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Point.hpp"

namespace pmacc::spearhed
{
    /**
     * @brief Translation-only coordinate frame for the positions in one particle bucket.
     *
     * Particle @c relativePos values are expressed in this chart. The chart deliberately has no
     * ownership or occupancy geometry; those belong to a decomposition.
     */
    template<CoordinateSystem CS>
    struct RegionChart
    {
        using Pnt = Point<CS, ValueStorage<CS>>;
        using Vec = spearhed::Vec<CS, ValueStorage<CS>>;

        constexpr RegionChart() = default;

        constexpr explicit RegionChart(Pnt const& origin) : origin(origin)
        {
        }

        template<typename Storage>
        [[nodiscard]] constexpr Pnt toWorld(spearhed::Vec<CS, Storage> const& relativePosition) const
        {
            return origin + relativePosition;
        }

        template<typename Storage>
        [[nodiscard]] constexpr Vec toLocal(Point<CS, Storage> const& worldPosition) const
        {
            return worldPosition - origin;
        }

        Pnt origin{typename CS::T_Axis{0}};
    };
} // namespace pmacc::spearhed
