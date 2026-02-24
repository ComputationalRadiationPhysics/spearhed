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

#include "spmacc/topology/CoordinateSystem.hpp"

#include <concepts>
#include <iostream>

namespace pmacc::spearhed
{

    // Think about alignment and memory laybout
    template<CoordinateSystem CS, typename Storage>
    struct Point;

    namespace detail
    {
        template<typename T>
        static constexpr T abs_diff(T a, T b) noexcept
        {
            T const diff = a - b;
            return diff < T{0} ? -diff : diff;
        }
    } // namespace detail

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

        // template<CoordinateSystem OtherCS, typename OtherStorage>
        // constexpr explicit(false) Point(Point<OtherCS, OtherStorage> const& other) noexcept
        // {
        //     transformPoint(other, this);
        // }

        // Assignment operator for View = Value (Scatter) or View = View
        // template<CoordinateSystem OtherCS, typename OtherStorage>
        // constexpr Point& operator=(Point<OtherCS, OtherStorage> const& other) noexcept
        // {
        // }

        template<typename OtherStorage>
        requires(dim == 3 && std::constructible_from<Storage, T, T, T>)
        constexpr Point(Point<CS, OtherStorage> const& other) noexcept
            : Storage(other.get_x(), other.get_y(), other.get_z())
        {
        }

        template<typename OtherStorage>
        requires(dim == 2 && std::constructible_from<Storage, T, T>)
        constexpr Point(Point<CS, OtherStorage> const& other) noexcept : Storage(other.get_x(), other.get_y())
        {
        }

        template<typename OtherStorage>
        requires(dim == 1 && std::constructible_from<Storage, T>)
        constexpr Point(Point<CS, OtherStorage> const& other) noexcept : Storage(other.get_x())
        {
        }

        template<typename OtherStorage>
        constexpr Point& operator=(Point<CS, OtherStorage> const& other) noexcept
        {
            if constexpr(dim == 3)
            {
                this->get_x() = other.get_x();
                this->get_y() = other.get_y();
                this->get_z() = other.get_z();
            }
            else if constexpr(dim == 2)
            {
                this->get_x() = other.get_x();
                this->get_y() = other.get_y();
            }
            else if constexpr(dim == 1)
            {
                this->get_x() = other.get_x();
            }
            return *this;
        }

        [[nodiscard]] friend constexpr bool operator==(Point const& p, Scalar const val) noexcept
        {
            if constexpr(dim == 3)
            {
                return p.get_x() == val && p.get_y() == val && p.get_z() == val;
            }
            else if constexpr(dim == 2)
            {
                return p.get_x() == val && p.get_y() == val;
            }
            else
            {
                return p.get_x() == val;
            }
        }

        [[nodiscard]] constexpr bool isApprox(Point const& other, Scalar eps = std::numeric_limits<Scalar>::epsilon())
            const noexcept
        {
            if constexpr(dim == 3)
            {
                return detail::abs_diff(this->get_x(), other.get_x()) <= eps
                       && detail::abs_diff(this->get_y(), other.get_y()) <= eps
                       && detail::abs_diff(this->get_z(), other.get_z()) <= eps;
            }
            else if constexpr(dim == 2)
            {
                return detail::abs_diff(this->get_x(), other.get_x()) <= eps
                       && detail::abs_diff(this->get_y(), other.get_y()) <= eps;
            }
            else
            {
                return detail::abs_diff(this->get_x(), other.get_x()) <= eps;
            }
        }

        // Check if all components of this Point are approximately equal to a Scalar value
        [[nodiscard]] constexpr bool isApprox(Scalar val, Scalar eps = std::numeric_limits<Scalar>::epsilon())
            const noexcept
        {
            if constexpr(dim == 3)
            {
                return detail::abs_diff(this->get_x(), val) <= eps && detail::abs_diff(this->get_y(), val) <= eps
                       && detail::abs_diff(this->get_z(), val) <= eps;
            }
            else if constexpr(dim == 2)
            {
                return detail::abs_diff(this->get_x(), val) <= eps && detail::abs_diff(this->get_y(), val) <= eps;
            }
            else
            {
                return detail::abs_diff(this->get_x(), val) <= eps;
            }
        }

        friend std::ostream& operator<<(std::ostream& os, Point const& p)
        {
            if constexpr(dim == 3)
            {
                return os << "(" << p.get_x() << ", " << p.get_y() << ", " << p.get_z() << ")";
            }
            else if constexpr(dim == 2)
            {
                return os << "(" << p.get_x() << ", " << p.get_y() << ")";
            }
            else
            {
                return os << "(" << p.get_x() << ")";
            }
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

    // template<CoordinateSystem From, CoordinateSystem To, typename StorageFrom, typename StorageTo>
    // void transformPoint(Point<From, StorageFrom> const& x, Point<To, StorageTo>& y)
    // {
    //     // Careful. Do read copy write.
    //     // x and y might be the same particle, so we dont want to overwrite componenets prematurely
    //     if constexpr(!std::same_as<From, To>)
    //     {
    //         // if constexpr(std::same_as<To, Cartesian>)
    //         // {
    //         // OtherCS::from_spherical(a, b, c, this->get_x(), this->template get<1>(), this->template get<2>());
    //         // }
    //         // elif so on
    //     }
    // }

} // namespace pmacc::spearhed
