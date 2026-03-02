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

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace pmacc::spearhed::meta
{
    template<template<typename> typename... Templates>
    struct ComponentList
    {
        static constexpr std::size_t size = sizeof...(Templates);
    };

    namespace detail
    {
        template<template<typename> typename T1, template<typename> typename T2>
        struct is_same_template : std::false_type
        {
        };

        template<template<typename> typename T>
        struct is_same_template<T, T> : std::true_type
        {
        };

        template<template<typename> typename T1, template<typename> typename T2>
        constexpr bool is_same_template_v = is_same_template<T1, T2>::value;

    } // namespace detail

    template<template<typename> typename Target, template<typename> typename... List>
    constexpr bool contains_template = (detail::is_same_template_v<Target, List> || ...);

    template<typename T>
    struct IsUniqueComp : std::true_type
    {
    };

    template<template<typename> typename... Ts>
    struct IsUniqueComp<ComponentList<Ts...>>
    {
        static constexpr bool check()
        {
            if constexpr(sizeof...(Ts) == 0)
                return true;
            else
                return check_impl<Ts...>();
        }

        template<template<typename> typename Head, template<typename> typename... Tail>
        static constexpr bool check_impl()
        {
            if constexpr(contains_template<Head, Tail...>)
                return false;
            else if constexpr(sizeof...(Tail) == 0)
                return true;
            else
                return check_impl<Tail...>();
        }

        static constexpr bool value = check();
    };

    template<typename T>
    constexpr bool isUniqueComp_v = IsUniqueComp<T>::value;

    template<typename Derived, typename T_List>
    struct InheritComponentsFrom;

    template<typename Derived, template<typename> typename... Ts>
    struct InheritComponentsFrom<Derived, ComponentList<Ts...>> : public Ts<Derived>...
    {
    };

    namespace detail
    {
        template<typename Derived, typename T_List>
        struct ComponentsAsTuple;

        template<typename Derived, template<typename> typename... Ts>
        struct ComponentsAsTuple<Derived, ComponentList<Ts...>>
        {
            using type = std::tuple<Ts<Derived>...>;
        };
    } // namespace detail

    template<typename Derived, typename T_List>
    using ComponentsAsTuple_t = typename detail::ComponentsAsTuple<Derived, T_List>::type;
} // namespace pmacc::spearhed::meta
