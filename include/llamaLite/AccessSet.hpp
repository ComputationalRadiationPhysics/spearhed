#pragma once

#include "llamaLite/Record.hpp"
#include "llamaLite/tag/TagPath.hpp"

#include <concepts>

namespace llama_lite
{
    /**
     * Note: there is no check in RAs for sub paths. For example { pos, pos/x } is a valid RAs list.
     * Even though including pos implies including pos/x
     */
    template<IsRecord R, IsRecordAccess... RAs>
    requires(R::template hasPath<RAs>() && ...) && ((countTypeOccurrences<RAs, RAs...>() == 1) && ...)
    struct AccessList
    {
        using record_type = R;
        using access_tuple_type = Tuple<RAs...>;
        static constexpr size_t size = sizeof...(RAs);
        static constexpr bool supportsDirectAccess = (sizeof...(RAs) == 1);

        template<IsRecordAccess Query>
        [[nodiscard]] static consteval bool contains()
        {
            return (std::is_same_v<typename ToPath<Query>::type, typename ToPath<RAs>::type> || ...);
        }

        template<IsRecord R2>
        [[nodiscard]] static consteval bool validFor(R2 = {})
        {
            return (R2::template hasPath<RAs> && ...);
        }
    };

    /**
     * Note: there is no check in excluded RAs for sub paths. For example { pos, pos/x } is a valid excluded RAs list.
     * Even though excluding pos implies excluding pos/x
     */
    template<IsRecord R, IsRecordAccess... ExcludedRAs>
    requires(R::template hasPath<ExcludedRAs>() && ...)
            && ((countTypeOccurrences<ExcludedRAs, ExcludedRAs...>() == 1) && ...)
    struct AccessWithExclusions
    {
    private:
        /**
         * Recursively counts leaves in Rec that are NOT under an excluded path.
         */
        template<typename Rec, typename PathSoFar = TagPath<>>
        static consteval size_t countValidLeaves()
        {
            using Fields = typename Rec::fields_tuple_type;

            return []<std::size_t... I>(std::index_sequence<I...>)
            {
                auto count_field = []<std::size_t Idx>() -> size_t
                {
                    using Field = std::tuple_element_t<Idx, Fields>;
                    using FieldTag = typename Field::tag_type;

                    using CurrentPath = append_t<PathSoFar, TagPath<FieldTag>>;

                    // If this specific node is explicitly excluded, prune this entire branch.
                    if constexpr(isExcluded<CurrentPath>())
                    {
                        return 0;
                    }
                    else
                    {
                        using Val = typename Field::value_type;
                        if constexpr(IsRecord<Val>)
                        {
                            // It is a Record, recurse deeper
                            return countValidLeaves<Val, CurrentPath>();
                        }
                        else
                        {
                            // It is a Leaf and not excluded
                            return 1;
                        }
                    }
                };

                // Sum up valid leaves from all fields
                return (count_field.template operator()<I>() + ... + 0);
            }(std::make_index_sequence<std::tuple_size_v<Fields>>{});
        }

        template<IsRecordAccess Query>
        [[nodiscard]] static consteval bool isExcluded()
        {
            return (std::is_same_v<typename ToPath<Query>::type, typename ToPath<ExcludedRAs>::type> || ...);
        }

    public:
        static constexpr size_t size = countValidLeaves<R>();

        // AccessWithRecord always supports direct access
        static constexpr bool supportsDirectAccess = true;

        using record_type = R;
        using excluded_tuple_type = Tuple<ExcludedRAs...>;

        template<IsRecordAccess Query>
        [[nodiscard]] static consteval bool contains(Query = {})
        {
            return !isExcluded<Query>() && R::template hasPath<Query>();
        }

        template<IsRecord QueryRecord>
        [[nodiscard]] static consteval bool validFor(QueryRecord = {})
        {
            if constexpr(std::is_same_v<QueryRecord, R>)
            {
                return true;
            }
            else
            {
                return checkStructuralCompatibility<QueryRecord, TagPath<>>();
            }
        }

    private:
        template<typename QueryRec, typename PathSoFar = TagPath<>>
        static consteval bool checkStructuralCompatibility()
        {
            using Fields = typename R::fields_tuple_type;

            return []<std::size_t... I>(std::index_sequence<I...>)
            {
                auto check_field = []<std::size_t Idx>()
                {
                    using Field = std::tuple_element_t<Idx, Fields>;
                    using FieldTag = typename Field::tag_type;

                    // Construct absolute path to this field
                    using CurrentPath = append_t<PathSoFar, FieldTag>;

                    // If this specific path is excluded, we ignore it (and its children)
                    if constexpr(isExcluded<CurrentPath>())
                    {
                        return true;
                    }
                    else
                    {
                        // Path is included. Query must have it.
                        if constexpr(!QueryRec::template hasPath<CurrentPath>())
                        {
                            return false;
                        }
                        else
                        {
                            // If it's a nested record, recurse
                            using Val = typename Field::value_type;
                            if constexpr(IsRecord<Val>)
                            {
                                return checkStructuralCompatibility<Val, QueryRec, CurrentPath>();
                            }
                            else
                            {
                                // It's a leaf and we confirmed hasPath above.
                                return true;
                            }
                        }
                    }
                };

                return (check_field.template operator()<I>() && ...);
            }(std::make_index_sequence<std::tuple_size_v<Fields>>{});
        }
    };

    namespace detail
    {

        template<typename T>
        inline constexpr bool is_access_list_v = false;

        template<IsRecordAccess... RAs>
        inline constexpr bool is_access_list_v<AccessList<RAs...>> = true;

        template<typename T>
        inline constexpr bool is_access_with_exclusions_v = false;

        template<IsRecord R, IsRecordAccess... RAs>
        inline constexpr bool is_access_with_exclusions_v<AccessWithExclusions<R, RAs...>> = true;

    } // namespace detail

    template<typename T>
    concept AccessSet = (detail::is_access_list_v<T> || detail::is_access_with_exclusions_v<T>) && requires {
        { T::size } -> std::convertible_to<size_t>;
        { T::supportsDirectAccess } -> std::same_as<bool>;
        // { T::template contains<>() } -> std::convertible_to<bool>;
        // { T::template validFor<>() } -> std::convertible_to<bool>;
    };

} // namespace llama_lite
