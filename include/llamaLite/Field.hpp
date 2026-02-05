#pragma once

#include "llamaLite/utility.hpp"

namespace llama_lite
{

    template<typename Tag, typename T>
    struct Field
    {
        using tag_type = Tag;
        using value_type = T;
    };

    template<typename T>
    concept IsField = isSpecializationOf_v<T, Field>;

} // namespace llama_lite
