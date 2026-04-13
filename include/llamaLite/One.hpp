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
#include "llamaLite/tag/TagPath.hpp"

#include <cstddef>

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
            return resolveLeaf<RA, R>(storage);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto getLeaf() const
        {
            return resolveLeaf<RA, R>(storage);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA)
        {
            using Path = to_path_t<RA>;
            return OneView(*this, Path{});
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA) const
        {
            using Path = to_path_t<RA>;
            return OneView(*this, Path{});
        }

    private:
        transform::transform_record_one_t<R> storage;
    };

} // namespace llama_lite
