#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/traits.hpp"

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(particleId);

        using idField = ll::Field<particleId_t, uint64_t>;
    } // namespace tags


} // namespace spearhed

namespace pmacc::spearhed
{
    template<>
    struct Init<::spearhed::tags::idField>
    {
        constexpr void operator()(auto idView, auto worker, auto idGen) const
        {
            *idView = idGen.fetchInc(worker);
        }
    };

} // namespace pmacc::spearhed
