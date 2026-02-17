#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/traits.hpp"

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(vel);

        using velField = ll::Field<
            vel_t,
            ll::Record<
                ll::Field<pmacc::spearhed::tags::x_t, float>,
                ll::Field<pmacc::spearhed::tags::y_t, float>,
                ll::Field<pmacc::spearhed::tags::z_t, float>>>;
    } // namespace tags


} // namespace spearhed

namespace pmacc::spearhed
{
    template<>
    struct InitValue<::spearhed::tags::velField>
    {
        constexpr void operator()(auto velView, float val) const
        {
            *velView[tags::x] = val;
            *velView[tags::y] = val;
            *velView[tags::z] = val;
        }
    };

    template<>
    struct InitZero<::spearhed::tags::velField>
    {
        constexpr void operator()(auto velView) const
        {
            *velView[tags::x] = {0.f};
            *velView[tags::y] = {0.f};
            *velView[tags::z] = {0.f};
        }
    };

} // namespace pmacc::spearhed
