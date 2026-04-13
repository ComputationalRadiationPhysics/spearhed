// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/ResolveLeaf.hpp"
#include "llamaLite/SoAView.hpp"
#include "llamaLite/Transform.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "utility.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace llama_lite
{

    namespace transform
    {
        template<size_t Size>
        struct PolicySoA
        {
            template<typename T>
            struct Apply
            {
                struct alignas(128) type : public std::array<T, Size>
                {
                };
            };
        };

        template<typename Record, size_t Size>
        using transform_record_soa_t = transform_record_t<Record, PolicySoA<Size>::template Apply>;

    } // namespace transform

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
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf() const
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... tags)
        {
            return SoAView<SoA, to_path_t<RAs>...>(*this, to_path_t<RAs>{}...);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... tags) const
        {
            return SoAView<SoA const, to_path_t<RAs>...>(*this, to_path_t<RAs>{}...);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto operator[](RA tag)
        {
            return view(tag);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto operator[](RA tag) const
        {
            return view(tag);
        }

        [[nodiscard]] constexpr auto operator[](size_type idx)
        {
            return SoAIndexedView(*this, idx);
        }

        [[nodiscard]] constexpr auto operator[](size_type idx) const
        {
            return SoAIndexedView(*this, idx);
        }

    private:
        alignas(128) transform::transform_record_soa_t<R, Size> channels_;
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
