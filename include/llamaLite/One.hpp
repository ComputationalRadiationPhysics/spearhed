// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/ResolveLeaf.hpp"
#include "llamaLite/Transform.hpp"
#include "llamaLite/View.hpp"
#include "llamaLite/tag/TagPath.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace llama_lite
{

    namespace transform
    {
        template<typename T>
        struct PolicyOne
        {
            using type = T;
        };

        template<typename Record>
        using transform_record_one_t = transform_record_t<Record, PolicyOne>;

    } // namespace transform

    /**
     * Single-element SoA-compatible container for a Record.
     *
     * Stores one instance of each leaf field as a plain scalar. getLeaf() returns
     * std::span<T, 1> so that View/ViewIndexed work unchanged (always use
     * index 0).
     */
    template<IsRecord R>
    struct One
    {
    public:
        using record_type = R;
        static constexpr size_t size = 1;

        template<IsRecordAccess... Tags>
        using view_type = View<One, Tags...>;

        template<IsRecordAccess... Tags>
        using indexed_view_type = ViewIndexed<One, Tags...>;

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf()
        {
            auto& leaf = resolveLeaf<RA, R>(storage);
            return std::span<std::remove_reference_t<decltype(leaf)>, 1>(&leaf, 1);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf() const
        {
            auto const& leaf = resolveLeaf<RA, R>(storage);
            return std::span<std::remove_reference_t<decltype(leaf)> const, 1>(&leaf, 1);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... tags)
        {
            return View<One, to_path_t<RAs>...>(*this, to_path_t<RAs>{}...);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... tags) const
        {
            return View<One const, to_path_t<RAs>...>(*this, to_path_t<RAs>{}...);
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

        [[nodiscard]] constexpr auto operator[](uint32_t idx)
        {
            return ViewIndexed(*this, idx);
        }

        [[nodiscard]] constexpr auto operator[](uint32_t idx) const
        {
            return ViewIndexed(*this, idx);
        }

    private:
        transform::transform_record_one_t<R> storage;
    };

} // namespace llama_lite
