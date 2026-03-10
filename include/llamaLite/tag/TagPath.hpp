// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Tuple.hpp"

#include <concepts>
#include <cstddef>

namespace llama_lite
{

    struct TagBase
    {
    };

    template<typename T>
    concept IsTag = std::derived_from<std::remove_cvref_t<T>, TagBase> && std::is_empty_v<T>;


    template<IsTag... Tags>
    struct TagPath;

    template<typename T>
    concept IsTagPath = requires {
        typename std::remove_cvref_t<T>::TagsTuple;
        { std::remove_cvref_t<T>::depth } -> std::convertible_to<size_t>;
    };

    template<typename T>
    concept IsRecordAccess = IsTag<std::remove_cvref_t<T>> || IsTagPath<std::remove_cvref_t<T>>;

    // ToPath: Normalize Tag or TagPath to TagPath
    template<IsRecordAccess RA, bool = IsTagPath<RA>>
    struct ToPath;

    template<IsRecordAccess RA>
    struct ToPath<RA, false>
    {
        using type = TagPath<std::remove_cvref_t<RA>>;
    };

    template<IsRecordAccess RA>
    struct ToPath<RA, true>
    {
        using type = std::remove_cvref_t<RA>;
    };

    template<IsRecordAccess RA>
    using to_path_t = typename ToPath<RA>::type;

    // namespace detail
    // {
    //     template<IsTag... Tags>
    //     struct PathHeadTail
    //     {
    //         using Head = void;
    //         using Tail = void;
    //         constexpr bool operator==(PathHeadTail const&) const = default;
    //     };

    //     template<IsTag H, IsTag... T>
    //     struct PathHeadTail<H, T...>
    //     {
    //         using Head = H;
    //         using Tail = TagPath<T...>;
    //         constexpr bool operator==(PathHeadTail const&) const = default;
    //     };

    // } // namespace detail

    template<IsTag... Tags>
    struct TagPath
    {
        static constexpr size_t depth = sizeof...(Tags);

        // using HeadTag = typename detail::PathHeadTail<Tags...>::Head;
        // using TailPath = typename detail::PathHeadTail<Tags...>::Tail;

        constexpr bool operator==(TagPath const&) const = default;

        using TagsTuple = Tuple<Tags...>;

        // Element Access
        template<size_t I>
        requires(I < sizeof...(Tags))
        using tag_at = std::tuple_element_t<I, TagsTuple>;

        // Path Manipulation
        template<size_t... I>
        static consteval auto take_first_helper(std::index_sequence<I...>)
        {
            return TagPath<tag_at<I>...>{};
        }

        template<size_t N>
        requires(N <= sizeof...(Tags) && sizeof...(Tags) > 0)
        using take_first = decltype(take_first_helper(std::make_index_sequence<N>{}));

        template<size_t N, size_t... I>
        static consteval auto drop_first_helper(std::index_sequence<I...>)
        {
            return TagPath<tag_at<I + N>...>{};
        }

        template<size_t N>
        requires(N <= sizeof...(Tags) && sizeof...(Tags) > 0)
        using drop_first = decltype(drop_first_helper<N>(std::make_index_sequence<sizeof...(Tags) - N>{}));

        // Path Comparisons
        static consteval auto head() requires(sizeof...(Tags) > 0)
        {
            return tag_at<0>{};
        }

        static consteval auto tail() requires(sizeof...(Tags) > 0)
        {
            return drop_first<1>{};
        }

        /// Exact equality
        template<IsRecordAccess RA>
        [[nodiscard]] static consteval bool isSame()
        {
            using Other = to_path_t<RA>;
            return (depth == Other::depth) && (commonPrefixLength<RA>() == depth);
        }

        // We define that each node is its own ancestor
        // TODO reconsider this
        /// This path is an ancestor of Other (or equal)
        template<IsRecordAccess RA>
        [[nodiscard]] static consteval bool isAncestorOf()
        {
            // Empty path is an ancestor for everything
            // TODO reconsider this based on whats useful for algorithms
            using Other = to_path_t<RA>;
            return commonPrefixLength<Other>() == depth;
        }

        /// This path is a strict ancestor of Other (not equal)
        template<IsRecordAccess RA>
        [[nodiscard]] static consteval bool isStrictAncestorOf()
        {
            using Other = to_path_t<RA>;
            return (depth < Other::depth) && isAncestorOf<RA>();
        }

        /// This path is a descendant of Other (or equal)
        template<IsRecordAccess RA>
        [[nodiscard]] static consteval bool isDescendantOf()
        {
            return to_path_t<RA>::template isAncestorOf<TagPath>();
        }

        /// This path is a strict descendant of Other (not equal)
        template<IsRecordAccess RA>
        [[nodiscard]] static consteval bool isStrictDescendantOf()
        {
            return to_path_t<RA>::template isStrictAncestorOf<TagPath>();
        }

        /// Returns the length of the common prefix
        template<IsRecordAccess RA>
        [[nodiscard]] static consteval size_t commonPrefixLength()
        {
            using Other = to_path_t<RA>;
            constexpr size_t minDepth = (depth < Other::depth) ? depth : Other::depth;

            if constexpr(minDepth == 0)
                return 0;
            else
            {
                size_t len = 0;
                auto check = [&]<size_t I>()
                {
                    if constexpr(std::same_as<tag_at<I>, typename Other::template tag_at<I>>)
                        ++len;
                    else
                        return false;
                    return true;
                };

                [&]<size_t... I>(std::index_sequence<I...>)
                { (check.template operator()<I>() && ...); }(std::make_index_sequence<minDepth>{});

                return len;
            }
        }
    };

    // Path Concatenation
    template<IsRecordAccess RA1, IsRecordAccess RA2>
    struct Append
    {
        using LHS = to_path_t<RA1>;
        using RHS = to_path_t<RA2>;

        using type = decltype([]<IsTag... L, IsTag... R>(TagPath<L...>, TagPath<R...>)
                              { return TagPath<L..., R...>{}; }(LHS{}, RHS{}));
    };

    template<IsRecordAccess RA1, IsRecordAccess RA2>
    using append_t = typename Append<RA1, RA2>::type;

    template<IsRecordAccess LHS, IsRecordAccess RHS>
    [[nodiscard]] consteval auto operator/(LHS, RHS) noexcept
    {
        return append_t<LHS, RHS>{};
    }

    // Redundancy Check (for Access types)
    namespace detail
    {
        template<IsTagPath QueryPath, IsRecordAccess... OtherRAs>
        consteval bool isStrictAncestorOfAny()
        {
            return (QueryPath::template isStrictAncestorOf<OtherRAs>() || ...);
        }
    } // namespace detail

    /// Returns true if any path in RAs is a strict ancestor of another.
    template<IsRecordAccess... RAs>
    consteval bool hasRedundantPaths()
    {
        if constexpr(sizeof...(RAs) <= 1)
            return false;
        else
            return (detail::isStrictAncestorOfAny<to_path_t<RAs>, RAs...>() || ...);
    }

} // namespace llama_lite
