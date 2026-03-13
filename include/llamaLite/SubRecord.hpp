// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Record.hpp"

#include <type_traits>

namespace llama_lite
{
    namespace detail
    {

        // Extracts the tail of a TagPath if the head matches the QueryTag
        template<IsTag QueryTag, IsTagPath Path>
        struct MatchingTail
        {
            using type = Tuple<>;
        };

        template<IsTag QueryTag, IsTag... Rest>
        struct MatchingTail<QueryTag, TagPath<QueryTag, Rest...>>
        {
            using type = std::conditional_t<sizeof...(Rest) == 0, Tuple<>, Tuple<TagPath<Rest...>>>;
        };

        // Identifies exact Tag matches
        template<IsTag QueryTag, IsTagPath Path>
        struct IsExactMatch : std::false_type
        {
        };

        template<IsTag QueryTag>
        struct IsExactMatch<QueryTag, TagPath<QueryTag>> : std::true_type
        {
        };

        // Forward declarations for recursion
        template<typename TupleObj, IsRecord Rec>
        struct SubRecordFromList;

        template<typename FieldsTuple, IsTagPath... Paths>
        struct ProcessAllFields;

        // Evaluates a single field against all requested paths
        template<IsField F, IsTagPath... Paths>
        struct ProcessField
        {
            using Tag = typename F::tag_type;
            using Value = typename F::value_type;

            // Does any path explicitly terminate at this Tag?
            static constexpr bool exact_match = (IsExactMatch<Tag, Paths>::value || ...);

            // Gather all nested paths that continue past this Tag
            using TailsList = typename ConcatTuples<typename MatchingTail<Tag, Paths>::type...>::type;

            static consteval auto evaluate()
            {
                if constexpr(exact_match)
                {
                    return std::type_identity<Tuple<F>>{};
                }
                else if constexpr(std::is_same_v<TailsList, Tuple<>>)
                {
                    return std::type_identity<Tuple<>>{};
                }
                else
                {
                    static_assert(IsRecord<Value>, "Sub-record path continues, but requested field is a leaf.");
                    using SubRec = typename SubRecordFromList<TailsList, Value>::type;
                    return std::type_identity<Tuple<Field<Tag, SubRec>>>{};
                }
            }

            using type = typename decltype(evaluate())::type;
        };

        // Maps ProcessField over the Tuple of fields
        template<IsField... Fs, IsTagPath... Paths>
        struct ProcessAllFields<Tuple<Fs...>, Paths...>
        {
            using KeptFieldsList = typename ConcatTuples<typename ProcessField<Fs, Paths...>::type...>::type;

            template<typename List>
            struct ToRecord;

            template<typename... KeptFs>
            struct ToRecord<Tuple<KeptFs...>>
            {
                using type = Record<KeptFs...>;
            };

            using type = typename ToRecord<KeptFieldsList>::type;
        };

        // Unpacks a Tuple of Paths to recursively process inner records
        template<IsTagPath... Paths, IsRecord Rec>
        struct SubRecordFromList<Tuple<Paths...>, Rec>
        {
            using type = typename ProcessAllFields<typename Rec::fields_tuple_type, Paths...>::type;
        };
    } // namespace detail

    template<IsRecord Rec, IsRecordAccess auto... RAs>
    struct SubRecord
    {
        // Ensure requested paths are valid upfront
        static_assert(
            (Rec::hasPath(RAs) && ...),
            "One or more paths provided to SubRecord do not exist in the source Record.");

        using type =
            typename detail::ProcessAllFields<typename Rec::fields_tuple_type, to_path_t<decltype(RAs)>...>::type;
    };

    template<IsRecord Rec, IsRecordAccess auto... RAs>
    using sub_record_t = typename SubRecord<Rec, RAs...>::type;


} // namespace llama_lite
