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

#include "spmacc/topology/Point.hpp"

#include <pmacc/math/functions/Root.hpp>

namespace pmacc::spearhed
{
    // template<typename CS, typename StorageA, typename StorageB>
    // auto distance(Point<CS, StorageA>, Point<CS, StorageB>)
    // {
    // }

    template<typename T_Distance, T_Dim Dim, typename StorageA, typename StorageB>
    [[nodiscard]] constexpr auto distance(
        Point<pmacc::spearhed::Cartesian<T_Distance, Dim>, StorageA> pointA,
        Point<pmacc::spearhed::Cartesian<T_Distance, Dim>, StorageB> pointB) noexcept -> T_Distance
    {
        auto const dx = pointA[tags::x] - pointB[tags::x];
        auto const dy = pointA[tags::y] - pointB[tags::y];
        auto const dz = pointA[tags::z] - pointB[tags::z];

        if constexpr(Dim == 1)
        {
            return math::sqrt(dx * dx);
        }
        else if constexpr(Dim == 2)
        {
            return math::sqrt(dx * dx + dy * dy);
        }
        else if constexpr(Dim == 3)
        {
            return math::sqrt(dx * dx + dy * dy + dz * dz);
        }
        else
        {
            static_assert(Dim <= 3, "Unsupported dimension");
        }
    }

} // namespace pmacc::spearhed
