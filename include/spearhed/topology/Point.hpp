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

#include "spearhed/topology/CoordinateSystem.hpp"

#include <concepts>

namespace spearhed
{

    // Think about alignment and memory laybout
    template<CoordinateSystem CS, typename Storage>
    struct Point;

    // {
    //     using Scalar = typename CS::Scalar;

    //     static constexpr T_Dim dim = CS::dimension;

    //     using Storage::Storage;

    //     // friend constexpr VectorType operator-(Point const& lhs, Point const& rhs) noexcept
    //     // {
    //     //     VectorType v;
    //     //     for(T_Dim i = 0; i < dim; ++i)
    //     //         v.data_[i] = lhs.data_[i] - rhs.data_[i];
    //     //     return v;
    //     // }

    //     // friend constexpr Point operator+(Point p, VectorType const& v) noexcept
    //     // {
    //     //     for(T_Dim i = 0; i < dim; ++i)
    //     //         p.data_[i] += v.data_[i];
    //     //     return p;
    //     // }
    // };

    template<typename T, T_Dim Dim, typename Storage>
    struct Point<Cartesian<T, Dim>, Storage> : public Storage
    {
        using CS = Cartesian<T, Dim>;
        using Scalar = typename CS::Scalar;
        static constexpr T_Dim dim = CS::dimension;

        using Storage::Storage;

        template<CoordinateSystem OtherCS, typename OtherStorage>
        constexpr explicit(false) Point(Point<OtherCS, OtherStorage> const& other) noexcept
        {
            transformPoint(other, this);
        }

        // Assignment operator for View = Value (Scatter) or View = View
        template<CoordinateSystem OtherCS, typename OtherStorage>
        constexpr Point& operator=(Point<OtherCS, OtherStorage> const& other) noexcept
        {
        }
    };

    template<typename T, T_Dim Dim, typename Storage>
    struct Point<Polar<T, Dim>, Storage> : public Storage
    {
        using CS = Polar<T, Dim>;
        using Scalar = typename CS::Scalar;
        static constexpr T_Dim dim = CS::dimension;

        using Storage::Storage;
    };

    template<CoordinateSystem From, CoordinateSystem To, typename StorageFrom, typename StorageTo>
    void transformPoint(Point<From, StorageFrom> const& x, Point<To, StorageTo>& y)
    {
        // Careful. Do read copy write.
        // x and y might be the same particle, so we dont want to overwrite componenets prematurely
        if constexpr(!std::same_as<From, To>)
        {
            // if constexpr(std::same_as<To, Cartesian>)
            // {
            // OtherCS::from_spherical(a, b, c, this->get_x(), this->template get<1>(), this->template get<2>());
            // }
            // elif so on
        }
    }

} // namespace spearhed
