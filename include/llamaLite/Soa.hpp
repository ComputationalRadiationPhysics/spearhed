// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/SoAView.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/tag/TagPath.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace llama_lite
{

    namespace tranform
    {
        template<IsField F, size_t Size>
        struct SoATransform;

        template<size_t Size, typename... Ts>
        auto transform_types_soa(Tuple<Ts...>) -> Tuple<typename SoATransform<Ts, Size>::type...>;

        template<typename Record, size_t Size>
        using transform_record_soa_t
            = decltype(transform_types_soa<Size>(std::declval<typename Record::fields_tuple_type>()));

        template<IsField F, size_t Size>
        requires IsRecord<typename F::value_type>
        struct SoATransform<F, Size>
        {
            static constexpr size_t size = Size;
            using record_type = F::value_type;

            using type = transform_record_soa_t<record_type, Size>;
        };

        template<IsField F, size_t Size>
        requires(!IsRecord<typename F::value_type>)
        struct SoATransform<F, Size>
        {
            static constexpr size_t size = Size;

            struct alignas(128) AlignedArray : public std::array<typename F::value_type, Size>
            {
            };

            using type = AlignedArray;
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
                    return std::span{child};
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
    template<IsRecord R, uint32_t Size>
    struct SoA
    {
    public:
        using size_type = uint32_t;
        using record_type = R;
        static constexpr size_t size = Size;

        template<IsRecordAccess... Tags>
        using View = SoAView<SoA, Tags...>;

        template<IsRecordAccess... Tags>
        using IndexedView = SoAIndexedView<SoA, Tags...>;

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf()
        {
            return detail::resolveLeaf<RA, R>(channels_);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf() const
        {
            return detail::resolveLeaf<RA, R>(channels_);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA)
        {
            using Path = to_path_t<RA>;
            return SoAView(*this, Path{});
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA) const
        {
            using Path = to_path_t<RA>;
            return SoAView(*this, Path{});
        }

        [[nodiscard]] auto operator[](size_type idx)
        {
            return SoAIndexedView(*this, idx);
        }

        [[nodiscard]] auto operator[](size_type idx) const
        {
            return SoAIndexedView(*this, idx);
        }

    private:
        alignas(128) tranform::transform_record_soa_t<R, Size> channels_;
    };

    // // Push back requires decomposing the input tuple
    // void push_back(typename R::TupleType const& val)
    // {
    //     [&]<std::size_t... I>(std::index_sequence<I...>)
    //     {
    //         (tuple::get<I>(channels_).push_back(std::get<I>(val)), ...);
    //     }(std::make_index_sequence<std::tuple_size_v<typename R::TupleType>>{});
    // }

} // namespace llama_lite
