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

#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

namespace pmacc::spearhed
{
    template<CoordinateSystem CS>
    struct PointValueStorage;

    template<typename T>
    struct PointValueStorage<Cartesian<T, 3>>
    {
        T x, y, z;

        constexpr PointValueStorage() = default;

        constexpr PointValueStorage(T x_, T y_, T z_) : x(x_), y(y_), z(z_)
        {
        }

        [[nodiscard]] constexpr T get_x() const
        {
            return x;
        }

        [[nodiscard]] constexpr T get_y() const
        {
            return y;
        }

        [[nodiscard]] constexpr T get_z() const
        {
            return z;
        }

        [[nodiscard]] constexpr T& get_x()
        {
            return x;
        }

        [[nodiscard]] constexpr T& get_y()
        {
            return y;
        }

        [[nodiscard]] constexpr T& get_z()
        {
            return z;
        }
    };

    // requires that pointViewType supports the correct accessors for CS
    template<CoordinateSystem CS, typename PointViewType>
    struct PointViewStorage;

    template<typename T, typename PointViewType>
    struct PointViewStorage<Cartesian<T, 3>, PointViewType>
    {
        PointViewType pointView;

        constexpr PointViewStorage(PointViewType pV) : pointView(pV)
        {
        }

        [[nodiscard]] constexpr T& get_x() const
        {
            return *pointView[tags::x];
        }

        [[nodiscard]] constexpr T& get_y() const
        {
            return *pointView[tags::y];
        }

        [[nodiscard]] constexpr T& get_z() const
        {
            return *pointView[tags::z];
        }
    };

    template<typename T, typename PointViewType>
    struct PointViewStorage<Cartesian<T, 2>, PointViewType>
    {
        PointViewType pointView;

        constexpr PointViewStorage(PointViewType pV) : pointView(pV)
        {
        }

        [[nodiscard]] constexpr T& get_x() const
        {
            return *pointView[tags::x];
        }

        [[nodiscard]] constexpr T& get_y() const
        {
            return *pointView[tags::y];
        }
    };

} // namespace pmacc::spearhed
