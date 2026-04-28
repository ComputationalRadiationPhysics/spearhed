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

#include <pmacc/math/functions/Abs.hpp>
#include <pmacc/math/functions/Pow.hpp>
#include <pmacc/math/functions/Root.hpp>

#include <concepts>
#include <iostream>
#include <limits>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<typename T>
        static constexpr T abs_diff(T a, T b) noexcept
        {
            T const diff = a - b;
            return diff < T{0} ? -diff : diff;
        }
    } // namespace detail

    template<CoordinateSystem CS, typename Storage>
    struct Vec;

    template<typename T, T_Dim Dim, typename Storage>
    struct Vec<Cartesian<T, Dim>, Storage> : public Storage
    {
        using CS = Cartesian<T, Dim>;
        using Scalar = typename CS::T_Axis;
        static constexpr T_Dim dim = CS::dimension;

        using Storage::Storage;

    private:
        template<std::size_t... Is>
        constexpr Vec(Scalar val, std::index_sequence<Is...>) noexcept : Storage{(static_cast<void>(Is), val)...}
        {
        }

    public:
        template<std::convertible_to<Scalar>... Args>
        requires(sizeof...(Args) == dim)
        constexpr Vec(Args... args) noexcept : Storage{static_cast<Scalar>(args)...}
        {
        }

        // broadcast
        constexpr explicit Vec(Scalar val) noexcept requires(dim > 1)
            : Vec(val, std::make_index_sequence<dim>{})
        {
        }

        template<typename OtherStorage>
        requires(dim == 3 && requires(T v) { Storage{v, v, v}; })
        constexpr Vec(Vec<CS, OtherStorage> const& other) noexcept
            : Storage({other[tags::x], other[tags::y], other[tags::z]})
        {
        }

        template<typename OtherStorage>
        requires(dim == 2 && requires(T v) { Storage{v, v}; })
        constexpr Vec(Vec<CS, OtherStorage> const& other) noexcept : Storage({other[tags::x], other[tags::y]})
        {
        }

        template<typename OtherStorage>
        requires(dim == 1 && requires(T v) { Storage{v}; })
        constexpr Vec(Vec<CS, OtherStorage> const& other) noexcept : Storage({other[tags::x]})
        {
        }

        template<typename OtherStorage>
        constexpr Vec& operator=(Vec<CS, OtherStorage> const& other) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] = other[tag]; });
            return *this;
        }

        template<typename OtherStorage>
        constexpr Vec& operator+=(Vec<CS, OtherStorage> const& other) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] += other[tag]; });
            return *this;
        }

        template<typename OtherStorage>
        constexpr Vec& operator-=(Vec<CS, OtherStorage> const& other) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] -= other[tag]; });
            return *this;
        }

        constexpr Vec& operator*=(Scalar const val) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] *= val; });
            return *this;
        }

        constexpr Vec& operator/=(Scalar const val) noexcept
        {
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { (*this)[tag] /= val; });
            return *this;
        }

        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator-(Vec const& v) noexcept
        {
            Vec<CS, ValueStorage<CS>> result = v;
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { result[tag] = -result[tag]; });
            return result;
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator+(
            Vec lhs,
            Vec<CS, OtherStorage> const& rhs) noexcept
        {
            Vec<CS, ValueStorage<CS>> result = lhs;
            return result += rhs;
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator-(
            Vec lhs,
            Vec<CS, OtherStorage> const& rhs) noexcept
        {
            Vec<CS, ValueStorage<CS>> result = lhs;
            return result -= rhs;
        }

        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator*(Vec lhs, Scalar const val) noexcept
        {
            Vec<CS, ValueStorage<CS>> result = lhs;
            return result *= val;
        }

        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator*(Scalar const val, Vec rhs) noexcept
        {
            Vec<CS, ValueStorage<CS>> result = rhs;
            return result *= val;
        }

        [[nodiscard]] friend constexpr Vec<CS, ValueStorage<CS>> operator/(Vec lhs, Scalar const val) noexcept
        {
            Vec<CS, ValueStorage<CS>> result = lhs;
            return result /= val;
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr bool operator==(Vec const& lhs, Vec<CS, OtherStorage> const& rhs) noexcept
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag) { return lhs[tag] == rhs[tag]; });
        }

        template<typename OtherStorage>
        [[nodiscard]] friend constexpr bool operator!=(Vec const& lhs, Vec<CS, OtherStorage> const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        template<typename OtherStorage>
        [[nodiscard]] constexpr bool isApprox(
            Vec<CS, OtherStorage> const& other,
            Scalar eps = std::numeric_limits<Scalar>::epsilon()) const noexcept
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag)
                                                   { return detail::abs_diff((*this)[tag], other[tag]) <= eps; });
        }

        friend std::ostream& operator<<(std::ostream& os, Vec const& v)
        {
            if constexpr(dim == 3)
            {
                return os << "[" << v[tags::x] << ", " << v[tags::y] << ", " << v[tags::z] << "]";
            }
            else if constexpr(dim == 2)
            {
                return os << "[" << v[tags::x] << ", " << v[tags::y] << "]";
            }
            else
            {
                return os << "[" << v[tags::x] << "]";
            }
        }
    };

    /** L1 (Manhattan) norm: sum of absolute component values. */
    template<CoordinateSystem CS, typename Storage>
    [[nodiscard]] constexpr typename CS::T_Axis norm1(Vec<CS, Storage> const& v) noexcept
    {
        typename CS::T_Axis result{0};
        for_each_tag<CS>([&](auto tag) { result += pmacc::math::abs(v[tag]); });
        return result;
    }

    /** L2 (Euclidean) norm: sqrt of sum of squared components. */
    template<CoordinateSystem CS, typename Storage>
    [[nodiscard]] constexpr typename CS::T_Axis norm2(Vec<CS, Storage> const& v) noexcept
    {
        typename CS::T_Axis r2{0};
        for_each_tag<CS>([&](auto tag) { r2 += v[tag] * v[tag]; });
        return pmacc::math::sqrt(r2);
    }

    /** L-infinity (Chebyshev) norm: maximum absolute component value. */
    template<CoordinateSystem CS, typename Storage>
    [[nodiscard]] constexpr typename CS::T_Axis normInf(Vec<CS, Storage> const& v) noexcept
    {
        typename CS::T_Axis result{0};
        for_each_tag<CS>(
            [&](auto tag)
            {
                auto const a = pmacc::math::abs(v[tag]);
                result = result < a ? a : result;
            });
        return result;
    }

    /** General Lp norm: pow(sum(|x_i|^p), 1/p). */
    template<CoordinateSystem CS, typename Storage>
    [[nodiscard]] typename CS::T_Axis normp(Vec<CS, Storage> const& v, typename CS::T_Axis p) noexcept
    {
        typename CS::T_Axis result{0};
        for_each_tag<CS>([&](auto tag) { result += pmacc::math::pow(pmacc::math::abs(v[tag]), p); });
        return pmacc::math::pow(result, typename CS::T_Axis{1} / p);
    }

} // namespace pmacc::spearhed
