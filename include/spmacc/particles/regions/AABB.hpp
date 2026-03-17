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

#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Point.hpp"
#include "spmacc/topology/PointStorage.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/math/vector/Vector.hpp>

namespace pmacc::spearhed
{
    /**
     * Axis aligned bounding box
     */
    template<typename TAxis, unsigned DIM>
    struct AABB
    {
        using CS = Cartesian<TAxis, DIM>;
        using Pnt = spearhed::Point<CS, PointValueStorage<CS>>;

        constexpr void reset()
        {
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    min[tag] = std::numeric_limits<TAxis>::max();
                    max[tag] = std::numeric_limits<TAxis>::lowest();
                });
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

        constexpr void extend(AABB const& other)
        {
            for(unsigned i = 0; i < DIM; ++i)
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
        }

        /**
         * Returns a new AABB expanded by a margin in all directions
         */
        constexpr AABB expand(TAxis margin) const
        {
            AABB result = *this;
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    result.min[tag] -= margin;
                    result.max[tag] += margin;
                });

            return result;
        }

        // [[nodiscard]] constexpr AABB shuffle_down(auto worker, unsigned delta, int width) const
        // {
        //     AABB result;
        //     //  mask assumes all threads in warp are active (standard for reduction)
        //     // constexpr auto active_mask = 0xffff'ffff;
        //     for(unsigned i = 0; i < DIM; ++i)
        //     {
        //         result.min[i] = alpaka::warp::shfl_down(worker.getAcc(), min[i], delta, width);
        //         result.max[i] = alpaka::warp::shfl_down(worker.getAcc(), max[i], delta, width);
        //     }
        //     return result;
        // }

        // This may need to be optimized later
        friend constexpr bool intersects(const AABB& a, const AABB& b)
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag)
                                                   { return !(a.min[tag] > b.max[tag] || a.max[tag] < b.min[tag]); });
        }

        Pnt min;
        Pnt max;
    };
} // namespace pmacc::spearhed
