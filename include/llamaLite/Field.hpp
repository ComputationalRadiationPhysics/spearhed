// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/utility.hpp"
#include "tag/TagPath.hpp"

namespace llama_lite
{

    template<IsTag Tag, typename T>
    struct Field
    {
        using tag_type = Tag;
        using value_type = T;
    };

    template<typename T>
    concept IsField = isSpecializationOf_v<T, Field>;

} // namespace llama_lite
