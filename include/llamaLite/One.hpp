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

        template<typename TSoA, IsAccessSet S>
        constexpr One(ViewIndexed<TSoA, S> const& view)
        {
            (*this)[uint32_t{0}].deepCopyFrom(view);
        }

        template<typename TSoA, IsAccessSet S>
        constexpr One& operator=(ViewIndexed<TSoA, S> const& view)
        {
            (*this)[uint32_t{0}].deepCopyFrom(view);
            return *this;
        }

        template<IsRecordAccess... Tags>
        using view_type = View<One, access_set_t<Tags...>>;

        template<IsRecordAccess... Tags>
        using indexed_view_type = ViewIndexed<One, access_set_t<Tags...>>;

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
        [[nodiscard]] constexpr auto view(RAs... /*tags*/)
        {
            return View<One, access_set_t<RAs...>>(*this);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto view(RAs... /*tags*/) const
        {
            return View<One const, access_set_t<RAs...>>(*this);
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
            return ViewIndexed<One, Set<>>(*this, idx);
        }

        [[nodiscard]] constexpr auto operator[](uint32_t idx) const
        {
            return ViewIndexed<One const, Set<>>(*this, idx);
        }

    private:
        transform::transform_record_one_t<R> storage;
    };

} // namespace llama_lite
