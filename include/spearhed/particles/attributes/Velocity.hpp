#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spearhed/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/traits.hpp"

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(vel);

        using velField
            = ll::Field<vel_t, ll::Record<ll::Field<x_t, float>, ll::Field<y_t, float>, ll::Field<z_t, float>>>;
    } // namespace tags


} // namespace spearhed

namespace pmacc::spearhed
{
    template<>
    struct InitValue<::spearhed::tags::velField>
    {
        constexpr void operator()(auto velView, float val) const
        {
            *velView[::spearhed::tags::x] = val;
            *velView[::spearhed::tags::y] = val;
            *velView[::spearhed::tags::z] = val;
        }
    };

    template<>
    struct InitZero<::spearhed::tags::velField>
    {
        constexpr void operator()(auto velView) const
        {
            *velView[::spearhed::tags::x] = {0.f};
            *velView[::spearhed::tags::y] = {0.f};
            *velView[::spearhed::tags::z] = {0.f};
        }
    };

} // namespace pmacc::spearhed
