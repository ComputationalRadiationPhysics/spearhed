#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/tag/TagPath.hpp"

#include <cstdint>

namespace llama_lite
{

    namespace tranform
    {
        template<IsField F>
        struct OneTransform;

        template<typename... Ts>
        auto transform_record_one(Tuple<Ts...>) -> Tuple<typename OneTransform<Ts>::type...>;

        template<typename Record>
        using transform_record_one_t
            = decltype(transform_record_one(std::declval<typename Record::fields_tuple_type>()));

        template<IsField F>
        requires IsRecord<typename F::value_type>
        struct OneTransform<F>
        {
            using record_type = F::value_type;

            using type = transform_record_one_t<record_type>;
        };

        template<IsField F>
        requires(!IsRecord<typename F::value_type>)
        struct OneTransform<F>
        {
            using type = typename F::value_type;
        };

    } // namespace tranform

    namespace detail
    {
        template<IsRecordAccess RA, typename CurrentRecord, typename CurrentStorage>
        static constexpr auto resolveLeaf(CurrentStorage& storage)
        {
            using Path = typename ToPath<RA>::type;

            constexpr uint32_t idx = CurrentRecord::template getIndex<typename Path::HeadTag>();
            using ValueT = typename CurrentRecord::template value_type_for<typename Path::HeadTag>;

            auto& child = tuple::get<idx>(storage);

            if constexpr(Path::depth == 1)
            {
                if constexpr(IsRecord<ValueT>)
                {
                    static_assert(
                        Path::depth != 1,
                        "TagPath refers to a Node (Record), but a Leaf Field was expected. Path is too short.");
                }
                else
                {
                    return child;
                }
            }
            else
            {
                if constexpr(!IsRecord<ValueT>)
                {
                    static_assert(
                        IsRecord<ValueT>,
                        "TagPath continues, but a Leaf Field was encountered. Path is too long/invalid.");
                }
                else
                {
                    return resolveLeaf<typename Path::TailPath, ValueT>(child);
                }
            }
        }

    } // namespace detail

    /**
     * Recursive Structure-of-Arrays (SoA) container for a Record
     *
     * Stores hierarchical records by flattening them into a nested tuple of arrays.
     * This allows logical grouping of components while maintaining contiguous memory
     * storage for individual fields.
     */
    template<IsRecord R>
    struct One
    {
    public:
        using record_type = R;
        static constexpr size_t size = 1;

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf()
        {
            return detail::resolveLeaf<RA, R>(storage);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf() const
        {
            return detail::resolveLeaf<RA, R>(storage);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA)
        {
            using Path = typename ToPath<RA>::type;
            return OneView(*this, Path{});
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA) const
        {
            using Path = typename ToPath<RA>::type;
            return OneView(*this, Path{});
        }

    private:
        tranform::transform_record_one_t<R> storage;
    };

} // namespace llama_lite
