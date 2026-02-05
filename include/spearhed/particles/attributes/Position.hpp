#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/traits.hpp"

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(x);
        DEFINE_TAG(y);
        DEFINE_TAG(z);
        DEFINE_TAG(pos);

        using posField
            = ll::Field<pos_t, ll::Record<ll::Field<x_t, float>, ll::Field<y_t, float>, ll::Field<z_t, float>>>;
    } // namespace tags


} // namespace spearhed

namespace pmacc::spearhed
{
    template<>
    struct InitValue<::spearhed::tags::posField>
    {
        constexpr void operator()(auto posView, float val) const
        {
            *posView[::spearhed::tags::x] = val;
            *posView[::spearhed::tags::y] = val;
            *posView[::spearhed::tags::z] = val;
        }
    };

    template<>
    struct InitZero<::spearhed::tags::posField>
    {
        constexpr void operator()(auto posView) const
        {
            *posView[::spearhed::tags::x] = {0.f};
            *posView[::spearhed::tags::y] = {0.f};
            *posView[::spearhed::tags::z] = {0.f};
        }
    };

} // namespace pmacc::spearhed
