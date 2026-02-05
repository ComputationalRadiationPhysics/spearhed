#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/traits.hpp"

namespace pmacc::spearhed
{
    namespace tags
    {
        DEFINE_TAG(multiMask);
        using MultiMaskField = ll::Field<multiMask_t, uint8_t>;
    } // namespace tags

    template<>
    struct InitValue<tags::MultiMaskField>
    {
        constexpr void operator()(auto multiMaskView, uint8_t val) const
        {
            *multiMaskView = val;
        }
    };

} // namespace pmacc::spearhed
