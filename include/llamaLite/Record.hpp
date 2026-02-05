#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace llama_lite
{
    template<IsField... Fs>
    struct Record;

    template<typename T>
    concept IsRecord = isSpecializationOf_v<T, Record> && requires { typename T::fields_tuple_type; };

    template<IsField... Fs>
    struct Record
    {
        template<IsTag QueryTag, typename... SearchTags>
        static consteval uint32_t countTagOccurrences()
        {
            return ((std::is_same_v<QueryTag, SearchTags> ? 1 : 0) + ... + 0);
        }

        static_assert(
            ((countTagOccurrences<typename Fs::tag_type, typename Fs::tag_type...>() == 1) && ...),
            "Duplicate tags detected in Record definition. All fields must be unique.");

        using fields_tuple_type = Tuple<Fs...>;

        static constexpr std::size_t leaf_count = []() consteval
        {
            std::size_t sum = 0;
            auto add_field = [&]<IsField Field>()
            {
                using Val = typename Field::value_type;
                if constexpr(IsRecord<Val>)
                {
                    sum += Val::leaf_count;
                }
                else
                {
                    sum += 1;
                }
            };
            (add_field.template operator()<Fs>(), ...);
            return sum;
        }();

        template<IsTag QueryTag>
        [[nodiscard]] static consteval uint32_t getIndex()
        {
            static_assert((std::is_same_v<typename Fs::tag_type, QueryTag> || ...), "Tag not found in Record");

            constexpr bool matches[] = {std::is_same_v<typename Fs::tag_type, QueryTag>...};
            for(uint32_t i = 0; i < sizeof...(Fs); ++i)
            {
                if(matches[i])
                {
                    return i;
                }
            }
            // Unreachable due to static_assert
            return -1u;
        }

        // check if tag exists in the top level of the fields in the record
        template<IsTag QueryTag>
        [[nodiscard]] static consteval bool hasTag()
        {
            return (std::is_same_v<typename Fs::tag_type, QueryTag> || ...);
        }

        template<IsRecordAccess Query>
        [[nodiscard]] static consteval bool hasPath()
        {
            using Path = typename ToPath<Query>::type;

            // empty path exists in all records
            if constexpr(Path::depth == 0)
                return true;

            using Head = typename Path::HeadTag;
            if constexpr(!hasTag<Head>())
            {
                return false;
            }
            else
            {
                if constexpr(Path::depth == 1)
                {
                    return true;
                }
                else
                {
                    // Check recursively
                    constexpr size_t idx = getIndex<Head>();
                    using FieldType = std::tuple_element_t<idx, fields_tuple_type>::value_type;

                    if constexpr(IsRecord<FieldType>)
                    {
                        return FieldType::template hasPath<typename Path::TailPath>();
                    }
                    else
                    {
                        return false; // Path continues but field is a leaf
                    }
                }
            }
        }

        template<IsRecordAccess Query>
        [[nodiscard]] static consteval auto resolvePathToField()
        {
            using Path = typename ToPath<Query>::type;

            constexpr std::size_t idx = getIndex<typename Path::HeadTag>();
            using CurrentField = std::tuple_element_t<idx, fields_tuple_type>;

            if constexpr(Path::depth == 1)
            {
                return CurrentField{};
            }
            else
            {
                static_assert(
                    IsRecord<typename CurrentField::value_type>,
                    "TagPath continues but the field at this level is not a Record.");

                using Tail = typename Path::TailPath;
                using Result = typename CurrentField::value_type::template field_for<Tail>;
                return Result{};
            }
        }

        template<IsRecordAccess RA>
        using field_for = decltype(resolvePathToField<RA>());

        template<IsRecordAccess RA>
        using value_type_for = field_for<RA>::value_type;

        template<IsRecordAccess RA>
        static consteval bool isLeaf()
        {
            return !IsRecord<value_type_for<RA>>;
        }

        // // Tuple of accessor types for all fields
        // using AccessorTupleType = Tuple<accessor_t<Fields>...>;

        // // Get accessor for the Record itself
        // using accessor_for_record = accessor_t<Record>;

        // // Get accessor for a specific field by tag
        // template<typename QueryTag>
        // using accessor_for_field = accessor_t<Field<QueryTag, field_for<QueryTag>>>;
    };


} // namespace llama_lite
