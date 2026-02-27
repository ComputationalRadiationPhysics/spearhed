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

#include <pmacc/math/vector/Vector.hpp>

namespace pmacc::spearhed
{
    /**
     * Axis aligned bounding box
     */
    template<typename TAxis, unsigned DIM>
    struct AABB
    {
        using Vec = math::Vector<TAxis, DIM>;

        // This may need to be optimized later
        friend constexpr bool intersects(const AABB& a, const AABB& b)
        {
            for(unsigned i = 0; i < DIM; ++i)
            {
                // Check for separation along axis i
                if(a.min[i] > b.max[i] || a.max[i] < b.min[i])
                {
                    return false;
                }
            }
            return true;
        }

    private:
        Vec min;
        Vec max;
    };
} // namespace pmacc::spearhed
