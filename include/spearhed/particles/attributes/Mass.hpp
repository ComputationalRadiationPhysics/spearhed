#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/traits.hpp"

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(mass);

        using massField = ll::Field<mass_t, float>;
    } // namespace tags


} // namespace spearhed
