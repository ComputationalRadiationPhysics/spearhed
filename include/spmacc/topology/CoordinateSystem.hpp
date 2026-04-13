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

#include <concepts>
#include <cstdint>
#include <tuple>

namespace pmacc::spearhed
{
    // type which holds the number of dimensions
    using T_Dim = uint32_t;

    // Currently assumes constant metrics
    enum class MetricKind
    {
        // Dense metric tensor
        General,
        // Diagonal metric tensor
        Orthogonal,
        // Identity metric tensor
        Orthonormal
    };

    template<typename CS>
    concept CoordinateSystem = requires {
        // type in which the coordinates are stored
        typename CS::T_Axis;
        typename CS::tags;
        { CS::dimension } -> std::convertible_to<std::size_t>;
        { CS::metricKind } -> std::convertible_to<MetricKind>;
    };

    // Compile time mapping from Index to Tag type
    template<CoordinateSystem CS, std::size_t I>
    requires(I < std::tuple_size_v<typename CS::tags>)
    using tag_of = std::tuple_element_t<I, typename CS::tags>;

    // Compile time mapping from Tag to Index using a generalized helper
    template<CoordinateSystem CS, typename T>
    [[nodiscard]] consteval std::size_t index_of() noexcept
    {
        constexpr auto find_index = []<std::size_t... Is>(std::index_sequence<Is...>)
        {
            std::size_t match = -1;
            [[maybe_unused]] bool _
                = ((std::same_as<T, std::tuple_element_t<Is, typename CS::tags>> ? (match = Is, true) : false) || ...);
            return match;
        };

        constexpr std::size_t idx = find_index(std::make_index_sequence<std::tuple_size_v<typename CS::tags>>{});
        static_assert(idx != static_cast<std::size_t>(-1), "Tag not found in coordinate system.");

        return idx;
    }

    template<auto Start, auto End, typename F>
    constexpr void constexpr_for(F&& f)
    {
        [&]<auto... Is>(std::integer_sequence<decltype(Start), Is...>)
        {
            (f(std::integral_constant<decltype(Start), Start + Is>{}), ...);
        }(std::make_integer_sequence<decltype(Start), End - Start>{});
    }

    template<CoordinateSystem CS, typename F>
    constexpr void for_each_index(F&& f) noexcept
    {
        constexpr std::size_t N = std::tuple_size_v<typename CS::tags>;
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        { (f(std::integral_constant<std::size_t, Is>{}), ...); }(std::make_index_sequence<N>{});
    }

    template<CoordinateSystem CS, typename F>
    constexpr void for_each_enum_tag(F&& f) noexcept
    {
        constexpr std::size_t N = std::tuple_size_v<typename CS::tags>;
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            (f(std::integral_constant<std::size_t, Is>{}, pmacc::spearhed::tag_of<CS, Is>{}), ...);
        }(std::make_index_sequence<N>{});
    }

    template<CoordinateSystem CS, typename F>
    constexpr void for_each_tag(F&& f) noexcept
    {
        constexpr std::size_t N = std::tuple_size_v<typename CS::tags>;
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        { (f(pmacc::spearhed::tag_of<CS, Is>{}), ...); }(std::make_index_sequence<N>{});
    }

    template<CoordinateSystem CS, typename F>
    constexpr bool all_of_tag(F&& f) noexcept
    {
        constexpr std::size_t N = std::tuple_size_v<typename CS::tags>;
        return [&]<std::size_t... Is>(std::index_sequence<Is...>)
        { return (f(pmacc::spearhed::tag_of<CS, Is>{}) && ...); }(std::make_index_sequence<N>{});
    }

    template<CoordinateSystem CS, typename F>
    constexpr bool any_of_tag(F&& f) noexcept
    {
        constexpr std::size_t N = std::tuple_size_v<typename CS::tags>;
        return [&]<std::size_t... Is>(std::index_sequence<Is...>)
        { return (f(pmacc::spearhed::tag_of<CS, Is>{}) || ...); }(std::make_index_sequence<N>{});
    }


} // namespace pmacc::spearhed
