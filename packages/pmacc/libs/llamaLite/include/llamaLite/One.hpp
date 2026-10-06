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
#include "llamaLite/View.hpp"
#include "llamaLite/tag/TagPath.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

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

        constexpr One() = default;
        constexpr One(One const&) = default;
        constexpr One(One&&) = default;
        constexpr One& operator=(One const&) = default;
        constexpr One& operator=(One&&) = default;

        template<typename TSoA, IsAccessSet Leaves, IsTagPath Root>
        requires requires(One& self, ViewIndexed<TSoA, Leaves, Root> const& view) {
            copy_values(self[uint32_t{0}], view);
        }
        constexpr One(ViewIndexed<TSoA, Leaves, Root> const& view)
        {
            copy_values((*this)[uint32_t{0}], view);
        }

        template<typename TSoA, IsAccessSet Leaves, IsTagPath Root>
        requires requires(One& self, ViewIndexed<TSoA, Leaves, Root> const& view) {
            copy_values(self[uint32_t{0}], view);
        }
        constexpr One& operator=(ViewIndexed<TSoA, Leaves, Root> const& view)
        {
            copy_values((*this)[uint32_t{0}], view);
            return *this;
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf(RA /*tag*/)
        {
            auto& leaf = resolveLeaf<RA, R>(storage);
            return std::span<std::remove_reference_t<decltype(leaf)>, 1>(&leaf, 1);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf(RA /*tag*/) const
        {
            auto const& leaf = resolveLeaf<RA, R>(storage);
            return std::span<std::remove_reference_t<decltype(leaf)> const, 1>(&leaf, 1);
        }

        [[nodiscard]] constexpr auto view()
        {
            return View<One>(*this);
        }

        [[nodiscard]] constexpr auto view() const
        {
            return View<One const>(*this);
        }

        template<IsRecordAccess RA>
        requires requires(View<One, record_leaf_set_t<R>> root, RA tag) { root.view(tag); }
        [[nodiscard]] constexpr auto view(RA tag)
        {
            return view().view(tag);
        }

        template<IsRecordAccess RA>
        requires requires(View<One const, record_leaf_set_t<R>> root, RA tag) { root.view(tag); }
        [[nodiscard]] constexpr auto view(RA tag) const
        {
            return view().view(tag);
        }

        template<IsRecordAccess... RAs>
        requires requires(View<One, record_leaf_set_t<R>> root, RAs... tags) { root.select(tags...); }
        [[nodiscard]] constexpr auto select(RAs... tags)
        {
            return view().select(tags...);
        }

        template<IsRecordAccess... RAs>
        requires requires(View<One const, record_leaf_set_t<R>> root, RAs... tags) { root.select(tags...); }
        [[nodiscard]] constexpr auto select(RAs... tags) const
        {
            return view().select(tags...);
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

        [[nodiscard]] constexpr auto operator[](uint32_t idx)
        {
            return view()[idx];
        }

        [[nodiscard]] constexpr auto operator[](uint32_t idx) const
        {
            return view()[idx];
        }

        template<IsRecordAccess... Tags>
        requires(sizeof...(Tags) <= 1)
        using view_type = decltype(std::declval<View<One, record_leaf_set_t<R>>>().view(std::declval<Tags>()...));

        template<IsRecordAccess... Tags>
        requires(sizeof...(Tags) <= 1)
        using indexed_view_type = decltype(std::declval<view_type<Tags...>>()[uint32_t{0}]);

        template<IsRecordAccess... Tags>
        using selection_view_type
            = decltype(std::declval<View<One, record_leaf_set_t<R>>>().select(std::declval<Tags>()...));

        template<IsRecordAccess... Tags>
        using indexed_selection_view_type = decltype(std::declval<selection_view_type<Tags...>>()[uint32_t{0}]);

    private:
        transform::transform_record_one_t<R> storage;
    };

} // namespace llama_lite
