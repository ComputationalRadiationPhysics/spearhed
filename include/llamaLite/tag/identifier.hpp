#pragma once

#include "llamaLite/tag/TagPath.hpp"

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
    };                                                                                                                \
    inline constexpr TagName##_t TagName                                                                              \
    {                                                                                                                 \
    }
