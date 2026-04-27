// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Record.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "traits.hpp"
#include "utility.hpp"

#include <utility>

namespace llama_lite
{

    // Iteration Interface
    namespace selectors
    {
        // Policy: Iterate everything (default)
        struct SelectAll
        {
            template<typename Path>
            static consteval bool allow(Path)
            {
                return true;
            }
        };

        // Policy: Iterate only specific paths (and their children)
        // - Continues traversal if 'Path' is a prefix of a target (to reach it).
        // - Visits 'Path' if it is a descendant of a target (inside the match).
        // TODO require Targets is a set and doesnt have descendents of other targets
        template<IsRecordAccess... Targets>
        struct Include
        {
            template<typename Path>
            static consteval bool allow(Path)
            {
                // Path is allowed if it leads to a target (ancestor)
                // OR if it is already inside a target (descendant)
                return ((Path::template isAncestorOf<Targets>() || Path::template isDescendantOf<Targets>()) || ...);
            }
        };

        // Policy: Iterate everything EXCEPT specific paths
        // - Prunes traversal if 'Path' matches or is inside a target.
        template<IsRecordAccess... Targets>
        struct Exclude
        {
            template<typename Path>
            static consteval bool allow(Path)
            {
                // Stop if Path is a descendant of (or equal to) any target
                return !((Path::template isDescendantOf<Targets>()) || ...);
            }
        };
    } // namespace selectors

    namespace detail
    {
        template<IsRecord R, typename CurrentPath, typename Selector, template<typename> typename VisitorTrait>
        constexpr void iterate_recursive(auto&& view, auto&&... args)
        {
            using fields_tuple_type = typename R::fields_tuple_type;

            // if CurrentPath is terminal
            // process current path. getNode

            // Compile-time loop over fields
            auto process_field = [&]<size_t I>()
            {
                using Field = std::tuple_element_t<I, fields_tuple_type>;
                using FieldTag = typename Field::tag_type;
                using NextPath = append_t<CurrentPath, FieldTag>;

                //  Should we enter this branch?
                if constexpr(Selector::allow(NextPath{}))
                {
                    using Val = typename Field::value_type;

                    if constexpr(traits::IsTraitSpecialized<VisitorTrait, Field>::value)
                    {
                        VisitorTrait<Field>{}(view[FieldTag{}], args...);
                    }
                    else
                    {
                        // Access the field instance
                        // auto& field_instance = std::get<I>(fields);
                        if constexpr(IsRecord<Val>)
                        {
                            // Recursively iterate sub-record
                            // Assumes field_instance is the sub-record or convertible to it
                            iterate_recursive<Val, NextPath, Selector, VisitorTrait>(view[FieldTag{}], args...);
                        }
                        else
                        {
                            // Visit leaf
                            VisitorTrait<Field>{}(view[FieldTag{}], args...);
                        }
                    }
                }
            };

            // Fold expression to unroll the loop
            [&]<size_t... Is>(std::index_sequence<Is...>) { (process_field.template operator()<Is>(), ...); }(
                std::make_index_sequence<std::tuple_size_v<fields_tuple_type>>{});
        }

        template<IsRecord R, typename CurrentPath, typename Selector, template<typename> typename VisitorTrait>
        constexpr void iterate_path_recursive(auto&&... args)
        {
            using fields_tuple_type = typename R::fields_tuple_type;

            auto process_field = [&]<size_t I>()
            {
                using Field = std::tuple_element_t<I, fields_tuple_type>;
                using FieldTag = typename Field::tag_type;
                using NextPath = append_t<CurrentPath, FieldTag>;

                if constexpr(Selector::allow(NextPath{}))
                {
                    using Val = typename Field::value_type;

                    if constexpr(IsRecord<Val>)
                    {
                        iterate_path_recursive<Val, NextPath, Selector, VisitorTrait>(args...);
                    }
                    else
                    {
                        VisitorTrait<NextPath>{}(args...);
                    }
                }
            };

            [&]<size_t... Is>(std::index_sequence<Is...>) { (process_field.template operator()<Is>(), ...); }(
                std::make_index_sequence<std::tuple_size_v<fields_tuple_type>>{});
        }
    } // namespace detail

    /**
     * @brief Iterates over the record tree with a compile-time selector policy and calls a trait as a functor
     *
     * TODO link the record and view. This iteration should actually be over an access set. And views should have an
     * access set
     * @tparam R Record
     * @tparam Selector The filtering policy (SelectAll, Include<...>, Exclude<...>)
     * @tparam visitor Trait for Fields types, which is callable with signature void(FieldView&, value)
     * @param view View of record R at its root. Requires record_type
     * If the trait is not specialized for a field, recursively iterates if the value type is a record, deafult
     * initializes leaf fields
     * To check if a trait is
     */
    template<
        IsRecord R,
        typename Selector = llama_lite::selectors::SelectAll,
        template<typename> typename VisitorTrait>
    constexpr void iterate(auto&& view, auto&&... args)
    {
        detail::iterate_recursive<R, TagPath<>, Selector, VisitorTrait>(LL_FORWARD(view), LL_FORWARD(args)...);
    }

    // Helper aliases for cleaner syntax (optional)
    template<IsRecord R, template<typename> typename VisitorTrait, IsRecordAccess auto... Paths, typename... Args>
    constexpr void iterate_only(auto&& view, Args&&... args)
    {
        iterate<R, selectors::Include<decltype(Paths)...>, VisitorTrait>(LL_FORWARD(view), LL_FORWARD(args)...);
    }

    template<IsRecord R, template<typename> typename VisitorTrait, IsRecordAccess auto... Paths, typename... Args>
    constexpr void iterate_except(auto&& view, Args&&... args)
    {
        iterate<R, selectors::Exclude<decltype(Paths)...>, VisitorTrait>(LL_FORWARD(view), LL_FORWARD(args)...);
    }

    /**
     * @brief Path-based iteration: traverses the record tree and calls Functor with the full TagPath as a
     * template parameter. The original view and args are passed unchanged. The functor navigates to the
     * field via the path itself.
     *
     * Functor must provide: template<typename Path> void operator()(auto&& view, auto&&... args)
     *
     * @tparam R Record
     * @tparam Selector The filtering policy (SelectAll, Include<...>, Exclude<...>)
     * @tparam Functor Callable with template<typename Path> operator()(view, args...)
     */
    template<
        IsRecord R,
        typename Selector = llama_lite::selectors::SelectAll,
        template<typename> typename VisitorTrait>
    constexpr void iterate_path(auto&&... args)
    {
        detail::iterate_path_recursive<R, TagPath<>, Selector, VisitorTrait>(LL_FORWARD(args)...);
    }

    template<IsRecord R, template<typename> typename VisitorTrait, IsRecordAccess auto... Paths, typename... Args>
    constexpr void iterate_path_only(Args&&... args)
    {
        iterate_path<R, selectors::Include<decltype(Paths)...>, VisitorTrait>(LL_FORWARD(args)...);
    }

    template<IsRecord R, template<typename> typename VisitorTrait, IsRecordAccess auto... Paths, typename... Args>
    constexpr void iterate_path_except(Args&&... args)
    {
        iterate_path<R, selectors::Exclude<decltype(Paths)...>, VisitorTrait>(LL_FORWARD(args)...);
    }

} // namespace llama_lite
