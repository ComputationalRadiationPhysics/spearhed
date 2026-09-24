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

#include <limits>

namespace pmacc::spearhed
{
    /** @brief Axis-aligned bounds whose corners are always world-space points. */
    template<CoordinateSystem CS>
    struct WorldAABB
    {
        using TAxis = CS::T_Axis;
        using Pnt = Point<CS, ValueStorage<CS>>;
        using Vec = spearhed::Vec<CS, ValueStorage<CS>>;

        constexpr WorldAABB() = default;

        constexpr WorldAABB(Pnt const& min, Pnt const& max) : min(min), max(max)
        {
        }

        [[nodiscard]] constexpr bool empty() const
        {
            return pmacc::spearhed::any_of_tag<CS>([&](auto tag) { return min[tag] > max[tag]; });
        }

        constexpr void extend(Pnt const& point)
        {
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    if(point[tag] < min[tag])
                        min[tag] = point[tag];
                    if(point[tag] > max[tag])
                        max[tag] = point[tag];
                });
        }

        constexpr void extend(WorldAABB const& other)
        {
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    if(other.min[tag] < min[tag])
                        min[tag] = other.min[tag];
                    if(other.max[tag] > max[tag])
                        max[tag] = other.max[tag];
                });
        }

        [[nodiscard]] constexpr WorldAABB expand(TAxis margin) const
        {
            WorldAABB result = *this;
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    result.min[tag] -= margin;
                    result.max[tag] += margin;
                });
            return result;
        }

        friend constexpr bool intersects(WorldAABB const& a, WorldAABB const& b)
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag)
                                                   { return !(a.min[tag] > b.max[tag] || a.max[tag] < b.min[tag]); });
        }

        Pnt min{std::numeric_limits<TAxis>::max()};
        Pnt max{std::numeric_limits<TAxis>::lowest()};
    };

    template<CoordinateSystem CS>
    constexpr auto computeVolume(WorldAABB<CS> const& aabb)
    {
        typename WorldAABB<CS>::TAxis volume{1};
        if(aabb.empty())
            return typename WorldAABB<CS>::TAxis{0};
        for_each_tag<CS>([&](auto tag) { volume *= aabb.max[tag] - aabb.min[tag]; });
        return volume;
    }
} // namespace pmacc::spearhed
