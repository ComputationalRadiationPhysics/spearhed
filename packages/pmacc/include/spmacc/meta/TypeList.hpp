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
    template<typename... Ts>
    requires(std::is_class_v<Ts> && ...)
    struct TypeList
    {
        static constexpr std::size_t size = sizeof...(Ts);
    };

    namespace detail
    {
        template<typename T_Type>
        struct ToTypeList
        {
            using type = TypeList<T_Type>;
        };

        template<typename... Ts>
        struct ToTypeList<TypeList<Ts...>>
        {
            using type = TypeList<Ts...>;
        };
    } // namespace detail

    /** If T_Type is an TypeList, return it. Otherwise wrap it in an TypeList.
     */
    template<typename T_Type>
    using ToTypeList_t = typename detail::ToTypeList<T_Type>::type;


    template<typename T, typename... Ts>
    constexpr bool contains_type = (std::is_same_v<T, Ts> || ...);

    template<typename T>
    struct IsUnique : std::true_type
    {
    };

    template<typename... Ts>
    struct IsUnique<TypeList<Ts...>>
    {
        static constexpr bool check()
        {
            if constexpr(sizeof...(Ts) == 0)
                return true;
            else
                return check_impl<Ts...>();
        }

        template<typename Head, typename... Tail>
        static constexpr bool check_impl()
        {
            if constexpr(contains_type<Head, Tail...>)
                return false;
            else if constexpr(sizeof...(Tail) == 0)
                return true;
            else
                return check_impl<Tail...>();
        }

        static constexpr bool value = check();
    };

    template<typename T>
    constexpr bool isUnique_v = IsUnique<T>::value;


    template<typename T_List>
    struct InheritFrom;

    template<typename... Ts>
    struct InheritFrom<TypeList<Ts...>> : public Ts...
    {
    };

    namespace detail
    {
        template<typename T_List>
        struct AsTuple;

        template<typename... Ts>
        struct AsTuple<TypeList<Ts...>>
        {
            using type = std::tuple<Ts...>;
        };
    } // namespace detail

    template<typename T_Type>
    using AsTuple_t = typename detail::AsTuple<T_Type>::type;

} // namespace pmacc::spearhed::meta
