/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/traits.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace pmacc::spearhed
{
    namespace tags
    {
        DEFINE_TAG(relativePos);

        using relativePosField = ll::
            Field<relativePos_t, ll::Record<ll::Field<x_t, float>, ll::Field<y_t, float>, ll::Field<z_t, float>>>;
    } // namespace tags

    template<>
    struct InitValue<tags::relativePosField>
    {
        HDINLINE constexpr void operator()(auto relativePosView, float val) const
        {
            *relativePosView[tags::x] = val;
            *relativePosView[tags::y] = val;
            *relativePosView[tags::z] = val;
        }
    };

    template<>
    struct InitZero<tags::relativePosField>
    {
        HDINLINE constexpr void operator()(auto relativePosView) const
        {
            *relativePosView[tags::x] = {0.f};
            *relativePosView[tags::y] = {0.f};
            *relativePosView[tags::z] = {0.f};
        }
    };

} // namespace pmacc::spearhed
