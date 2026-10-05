// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/ResolveLeaf.hpp"
#include "llamaLite/Set.hpp"
#include "llamaLite/Transform.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/View.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace llama_lite
{

    template<size_t Bytes>
    struct ColumnAlignment
    {
        static_assert(Bytes == 0 || (Bytes & (Bytes - 1)) == 0, "Column alignment must be zero or a power of two");

        template<typename T>
        static constexpr size_t value = Bytes < alignof(T) ? alignof(T) : Bytes;
    };

    using CompactAlignment = ColumnAlignment<0>;

    namespace transform
    {
        template<size_t Size, typename Alignment = ColumnAlignment<128>>
        struct PolicySoA
        {
            template<typename T>
            struct Apply
            {
                struct alignas(Alignment::template value<T>) type : public std::array<T, Size>
                {
                };
            };
        };

        template<typename Record, size_t Size, typename Alignment = ColumnAlignment<128>>
        using transform_record_soa_t = transform_record_t<Record, PolicySoA<Size, Alignment>::template Apply>;

    } // namespace transform

    /**
     * Recursive Structure-of-Arrays (SoA) container for a Record
     *
     * Stores hierarchical records by flattening them into a nested tuple of arrays.
     * This allows logical grouping of components while maintaining contiguous memory
     * storage for individual fields.
     */
    template<IsRecord R, uint32_t Size, typename Alignment = ColumnAlignment<128>>
    struct SoA
    {
    public:
        using size_type = uint32_t;
        using record_type = R;
        static constexpr size_t size = Size;

        template<IsRecordAccess... Tags>
        using view_type = View<SoA, access_set_t<Tags...>>;

        template<IsRecordAccess... Tags>
        using indexed_view_type = ViewIndexed<SoA, access_set_t<Tags...>>;

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf(RA /*tag*/)
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf(RA /*tag*/) const
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... /*tags*/)
        {
            return View<SoA, access_set_t<RAs...>>(*this);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... /*tags*/) const
        {
            return View<SoA const, access_set_t<RAs...>>(*this);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto operator[](RA tag)
        {
            return detail::resolveIfLeaf(view(tag));
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto operator[](RA tag) const
        {
            return detail::resolveIfLeaf(view(tag));
        }

        [[nodiscard]] constexpr auto operator[](size_type idx)
        {
            return ViewIndexed<SoA, Set<>>(*this, idx);
        }

        [[nodiscard]] constexpr auto operator[](size_type idx) const
        {
            return ViewIndexed<SoA const, Set<>>(*this, idx);
        }

    private:
        transform::transform_record_soa_t<R, Size, Alignment> channels_;
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
