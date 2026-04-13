// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Record.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/tag/TagPath.hpp"

namespace llama_lite
{
    /**
     * Recursively resolves a TagPath through a nested tuple storage, returning
     * a reference over the leaf storage element.
     *
     * This function only requires that nested record levels are navigable via
     * tuple::get<idx>. It is entirely agnostic to the type of the leaf node,
     * working seamlessly across SoA (std::array leaves), DynSoA (std::vector
     * leaves), and One (scalar leaves).
     *
     * @tparam RA IsRecordAccess - the tag path to resolve
     * @tparam CurrentRecord  the Record type at the current recursion level
     * @tparam CurrentStorage the storage tuple at the current recursion level
     * @param  storage reference to the current storage tuple
     * @return reference to the leaf data
     */
    template<IsRecordAccess RA, typename CurrentRecord, typename CurrentStorage>
    [[nodiscard]] static constexpr decltype(auto) resolveLeaf(CurrentStorage& storage)
    {
        using Path = to_path_t<RA>;

        constexpr uint32_t idx = CurrentRecord::getIndex(Path::head());
        using ValueT = typename CurrentRecord::template value_type_for<decltype(Path::head())>;

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
                return resolveLeaf<decltype(Path::tail()), ValueT>(child);
            }
        }
    }

} // namespace llama_lite
