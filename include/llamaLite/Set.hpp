// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/utility.hpp"

#include <cstddef>
#include <type_traits>

namespace llama_lite
{
    /**
     * A compile-time set of unique types with zero runtime overhead.
     *
     * Element identity is type-identity: every empty tag/TagPath type has exactly one
     * value, so a Set works both for single tags and for multi-level TagPaths. Prefer
     * makeSet() over spelling out Set<...> directly, so callers only ever pass values.
     */
    template<typename... Ts>
    struct Set
    {
        static_assert(
            ((countTypeOccurrences<Ts, Ts...>() == 1) && ...),
            "Duplicate elements detected in Set. All elements must be unique.");

        static constexpr std::size_t size = sizeof...(Ts);
        static constexpr bool empty = (size == 0);

        template<typename T>
        [[nodiscard]] static consteval bool contains(T = {})
        {
            return (std::is_same_v<T, Ts> || ...);
        }

        template<typename T>
        [[nodiscard]] consteval bool operator()(T tag) const
        {
            return contains(tag);
        }

        /// Set equality: same elements, order independent.
        template<typename... Us>
        [[nodiscard]] consteval bool operator==(Set<Us...>) const
        {
            if constexpr(sizeof...(Ts) != sizeof...(Us))
                return false;
            else
                return (Set<Us...>::contains(Ts{}) && ...);
        }

        /// Subset-or-equal: every element of this Set is contained in @p other.
        template<typename... Us>
        [[nodiscard]] consteval bool operator<=(Set<Us...>) const
        {
            return (Set<Us...>::contains(Ts{}) && ...);
        }

        /// Superset-or-equal: this Set contains every element of @p other.
        template<typename... Us>
        [[nodiscard]] consteval bool operator>=(Set<Us...>) const
        {
            return (contains(Us{}) && ...);
        }
    };

    template<typename T>
    concept IsSet = isSpecializationOf_v<T, Set>;

    namespace detail
    {
        // Insert T into Acc unless Acc already contains it.
        template<typename Acc, typename T, bool = Acc::template contains<T>()>
        struct InsertIfAbsent
        {
            using type = Acc;
        };

        template<typename... AccTs, typename T>
        struct InsertIfAbsent<Set<AccTs...>, T, false>
        {
            using type = Set<AccTs..., T>;
        };

        template<typename Acc, typename T>
        using insert_if_absent_t = typename InsertIfAbsent<Acc, T>::type;

        // Fold Bs... into Acc, one insert-if-absent at a time.
        template<typename Acc, typename... Bs>
        struct UnionFold
        {
            using type = Acc;
        };

        template<typename Acc, typename B0, typename... Rest>
        struct UnionFold<Acc, B0, Rest...>
        {
            using type = typename UnionFold<insert_if_absent_t<Acc, B0>, Rest...>::type;
        };

        // Keep elements of As... whose membership in BSet matches KeepIfPresent.
        template<typename BSet, bool KeepIfPresent, typename Acc, typename... As>
        struct FilterByMembership
        {
            using type = Acc;
        };

        template<typename BSet, bool KeepIfPresent, typename... AccTs, typename A0, typename... Rest>
        struct FilterByMembership<BSet, KeepIfPresent, Set<AccTs...>, A0, Rest...>
        {
            using NextAcc
                = std::conditional_t<BSet::template contains<A0>() == KeepIfPresent, Set<AccTs..., A0>, Set<AccTs...>>;
            using type = typename FilterByMembership<BSet, KeepIfPresent, NextAcc, Rest...>::type;
        };
    } // namespace detail

    /// Deduces element types from @p objs and builds a deduplicated Set.
    template<typename... Objs>
    [[nodiscard]] consteval auto makeSet(Objs... /*objs*/)
    {
        return typename detail::UnionFold<Set<>, Objs...>::type{};
    }

    /// Union: elements present in either Set, deduplicated.
    template<typename... As, typename... Bs>
    [[nodiscard]] consteval auto operator|(Set<As...>, Set<Bs...>)
    {
        return typename detail::UnionFold<Set<As...>, Bs...>::type{};
    }

    /// Intersection: elements present in both Sets.
    template<typename... As, typename... Bs>
    [[nodiscard]] consteval auto operator&(Set<As...>, Set<Bs...>)
    {
        return typename detail::FilterByMembership<Set<Bs...>, true, Set<>, As...>::type{};
    }

    /// Difference: elements of the left Set that are not in the right Set.
    template<typename... As, typename... Bs>
    [[nodiscard]] consteval auto operator-(Set<As...>, Set<Bs...>)
    {
        return typename detail::FilterByMembership<Set<Bs...>, false, Set<>, As...>::type{};
    }

    /// Symmetric difference: elements present in exactly one of the two Sets.
    template<typename... As, typename... Bs>
    [[nodiscard]] consteval auto operator^(Set<As...> a, Set<Bs...> b)
    {
        return (a - b) | (b - a);
    }

    /// Adds @p t to @p s. Ill-formed if @p t is already present.
    template<typename... Ts, typename T>
    [[nodiscard]] consteval auto add(Set<Ts...>, T)
    {
        static_assert(!Set<Ts...>::template contains<T>(), "Element already present in Set; use | for a union.");
        return Set<Ts..., T>{};
    }

    /// Removes @p t from @p s. A no-op if @p t is not present.
    template<typename... Ts, typename T>
    [[nodiscard]] consteval auto remove(Set<Ts...>, T)
    {
        return typename detail::FilterByMembership<Set<T>, false, Set<>, Ts...>::type{};
    }

    /// True if every element of @p a is contained in @p b.
    template<typename... As, typename... Bs>
    [[nodiscard]] consteval bool is_subset(Set<As...> a, Set<Bs...> b)
    {
        return a <= b;
    }

    /// Invokes fn(Ts{})... for every element of the Set, in declaration order.
    template<typename... Ts, typename F>
    constexpr void forEach(Set<Ts...>, F&& fn)
    {
        (fn(Ts{}), ...);
    }

} // namespace llama_lite
