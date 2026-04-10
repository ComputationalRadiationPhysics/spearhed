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
#include "spmacc/topology/CartesianStorage.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Vec.hpp"

#include <concepts>
#include <iostream>

namespace pmacc::spearhed
{

    // Think about alignment and memory laybout
    template<CoordinateSystem CS, typename Storage>
    struct Point;

    template<typename T, T_Dim Dim, typename Storage>
    struct Point<Cartesian<T, Dim>, Storage> : public Storage
    {
        using CS = Cartesian<T, Dim>;
        using Scalar = typename CS::T_Axis;
        static constexpr T_Dim dim = CS::dimension;

        using Storage::Storage;

    private:
        template<std::size_t... Is>
        constexpr Point(Scalar val, std::index_sequence<Is...>) noexcept : Storage{(static_cast<void>(Is), val)...}
        {
        }

    public:
        // template<CoordinateSystem OtherCS, typename OtherStorage>
        // constexpr explicit(false) Point(Point<OtherCS, OtherStorage> const& other) noexcept
        // {
        //     transformPoint(other, this);
        // }

        template<std::convertible_to<Scalar>... Args>
        requires(sizeof...(Args) == dim)
        constexpr Point(Args... args) noexcept : Storage{static_cast<Scalar>(args)...}
        {
        }

        // broadcast
        constexpr explicit Point(Scalar val) noexcept requires(dim > 1)
            : Point(val, std::make_index_sequence<dim>{})
        {
        }

        // Using `requires(T v) { Storage{v, v, v}; })` instead of `std::constructible_from` because the latter checks
        // for paren() init support and this disables brace elison for Storage and thus returns false
        template<typename OtherStorage>
        requires(dim == 3 && requires(T v) { Storage{v, v, v}; })
        constexpr Point(Point<CS, OtherStorage> const& other) noexcept
            : Storage({other[tags::x], other[tags::y], other[tags::z]})
        {
        }

        template<typename OtherStorage>
        requires(dim == 2 && requires(T v) { Storage{v, v}; })
        constexpr Point(Point<CS, OtherStorage> const& other) noexcept : Storage({other[tags::x], other[tags::y]})
        {
        }

        template<typename OtherStorage>
        requires(dim == 1 && requires(T v) { Storage{v}; })
        constexpr Point(Point<CS, OtherStorage> const& other) noexcept : Storage({other[tags::x]})
        {
        }

        template<typename OtherStorage>
        constexpr Point& operator=(Point<CS, OtherStorage> const& other) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] = other[tag]; });
            return *this;
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr bool operator==(Point const& lhs, Point<CS, OtherStorage> const& rhs) noexcept
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag) { return lhs[tag] == rhs[tag]; });
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr bool operator!=(Point const& lhs, Point<CS, OtherStorage> const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        template<typename VecStorage>
        constexpr Point& operator+=(Vec<CS, VecStorage> const& v) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] += v[tag]; });
            return *this;
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator-(
            Point lhs,
            Point<CS, OtherStorage> const& rhs) noexcept
        {
            Vec<CS, ValueStorage<CS>> result;
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { result[tag] = lhs[tag] - rhs[tag]; });
            return result;
        }

        template<typename VecStorage>
        [[nodiscard]] friend constexpr Point<CS, ValueStorage<CS>> operator+(
            Point lhs,
            Vec<CS, VecStorage> const& v) noexcept
        {
            Point<CS, ValueStorage<CS>> result = lhs;
            return result += v;
        }

        template<typename VecStorage>
        [[nodiscard]] friend constexpr Point<CS, ValueStorage<CS>> operator+(
            Vec<CS, VecStorage> const& v,
            Point rhs) noexcept
        {
            Point<CS, ValueStorage<CS>> result = rhs;
            return result += v;
        }

        template<typename OtherStorage>
        [[nodiscard]] constexpr bool isApprox(
            Point<CS, OtherStorage> const& other,
            Scalar eps = std::numeric_limits<Scalar>::epsilon()) const noexcept
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag)
                                                   { return detail::abs_diff((*this)[tag], other[tag]) <= eps; });
        }

        friend std::ostream& operator<<(std::ostream& os, Point const& p)
        {
            if constexpr(dim == 3)
            {
                return os << "(" << p[tags::x] << ", " << p[tags::y] << ", " << p[tags::z] << ")";
            }
            else if constexpr(dim == 2)
            {
                return os << "(" << p[tags::x] << ", " << p[tags::y] << ")";
            }
            else
            {
                return os << "(" << p[tags::x] << ")";
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
    //         // OtherCS::from_spherical(a, b, c, (*this)[tags::x], this->template get<1>(), this->template
    //         get<2>());
    //         // }
    //         // elif so on
    //     }
    // }

} // namespace pmacc::spearhed
