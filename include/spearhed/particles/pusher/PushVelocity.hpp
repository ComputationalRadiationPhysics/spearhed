#pragma once

#include "spearhed/ParticleView.hpp"
#include "spearhed/param/dimension.param"
#include "spearhed/particles/attributes/Position.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace spearhed
{
    struct PushVelocity
    {
        HDINLINE constexpr void operator()(auto worker, ParticleView<pos, vel> view, T_dt delt) const
        {
            *view[pos][x] += *view[vel][x] * delt;
            *view[pos][y] += *view[vel][y] * delt;
            *view[pos][z] += *view[vel][z] * delt;
        }
    };

} // namespace spearhed
