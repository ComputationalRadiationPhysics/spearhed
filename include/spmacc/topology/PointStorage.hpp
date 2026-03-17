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

#include "spmacc/topology/Cartesian.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <array>

namespace pmacc::spearhed
{
    template<CoordinateSystem CS>
    struct PointValueStorage;

    template<typename T, T_Dim Dim>
    struct PointValueStorage<Cartesian<T, Dim>> : public std::array<T, Dim>
    {
        using Base = std::array<T, Dim>;
        using CS = Cartesian<T, Dim>;

        template<CartesianTag Tag>
        [[nodiscard]] constexpr T& operator[](Tag) noexcept
        {
            return std::get<index_of<CS, Tag>()>(*this);
        }

        template<CartesianTag Tag>
        [[nodiscard]] constexpr T operator[](Tag) const noexcept
        {
            return std::get<index_of<CS, Tag>()>(*this);
        }
    };

    // requires that pointViewType supports the correct accessors for CS
    template<CoordinateSystem CS, typename PointViewType>
    struct PointViewStorage;

    template<typename T, T_Dim Dim, typename PointViewType>
    struct PointViewStorage<Cartesian<T, Dim>, PointViewType>
    {
        PointViewType pointView;
        using CS = Cartesian<T, Dim>;

        constexpr PointViewStorage(PointViewType pV) : pointView(pV)
        {
        }

        template<CartesianTag Tag>
        [[nodiscard]] constexpr T& operator[](Tag t) noexcept
        {
            return *pointView[t];
        }

        template<CartesianTag Tag>
        [[nodiscard]] constexpr T operator[](Tag t) const noexcept
        {
            return *pointView[t];
        }

        template<std::size_t I>
        [[nodiscard]] constexpr T& get() noexcept
        {
            return *pointView[tag_of<CS, I>{}];
        }

        template<std::size_t I>
        [[nodiscard]] constexpr T get() const noexcept
        {
            return *pointView[tag_of<CS, I>{}];
        }
    };

} // namespace pmacc::spearhed
