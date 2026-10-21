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

#include "spmacc/topology/Cartesian.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <array>

namespace pmacc::spearhed
{
    template<CoordinateSystem CS>
    struct ValueStorage;

    template<typename T, T_Dim Dim>
    struct ValueStorage<Cartesian<T, Dim>> : public std::array<T, Dim>
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

    // requires that ViewType supports the correct accessors for CS
    template<CoordinateSystem CS, typename ViewType>
    struct ViewStorage;

    template<typename T, T_Dim Dim, typename ViewType>
    struct ViewStorage<Cartesian<T, Dim>, ViewType>
    {
        ViewType view;
        using CS = Cartesian<T, Dim>;

        constexpr ViewStorage(ViewType v) : view(v)
        {
        }

        template<CartesianTag Tag>
        [[nodiscard]] constexpr T& operator[](Tag t) noexcept
        {
            return view[t];
        }

        template<CartesianTag Tag>
        [[nodiscard]] constexpr T operator[](Tag t) const noexcept
        {
            return view[t];
        }

        template<std::size_t I>
        [[nodiscard]] constexpr T& get() noexcept
        {
            return view[tag_of<CS, I>{}];
        }

        template<std::size_t I>
        [[nodiscard]] constexpr T get() const noexcept
        {
            return view[tag_of<CS, I>{}];
        }
    };

} // namespace pmacc::spearhed
