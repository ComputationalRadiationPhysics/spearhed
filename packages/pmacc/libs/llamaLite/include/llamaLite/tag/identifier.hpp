// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/tag/TagPath.hpp"

#include <string_view>

/** create an identifier (identifier with arbitrary code as second parameter
 * !! second parameter is optional and can be any C++ code one can add inside a class
 *
 * example: identifier(varname); //create type varname
 * example: identifier(varname,typedef int type;); //create type varname,
 *          later its possible to use: typedef varname::type type;
 *
 * to create an instance of this identifier you can use:
 *      varname();   or varname_
 */
#define DEFINE_TAG(TagName)                                                                                           \
    struct TagName##_t : llama_lite::TagBase                                                                          \
    {                                                                                                                 \
        static constexpr std::string_view name = #TagName;                                                            \
    };                                                                                                                \
    inline constexpr TagName##_t TagName                                                                              \
    {                                                                                                                 \
    }
