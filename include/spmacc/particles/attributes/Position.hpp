/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/traits.hpp"

namespace pmacc::spearhed
{
    namespace tags
    {
        DEFINE_TAG(pos);

        using posField
            = ll::Field<pos_t, ll::Record<ll::Field<x_t, float>, ll::Field<y_t, float>, ll::Field<z_t, float>>>;
    } // namespace tags

    template<>
    struct InitValue<tags::posField>
    {
        constexpr void operator()(auto posView, float val) const
        {
            *posView[tags::x] = val;
            *posView[tags::y] = val;
            *posView[tags::z] = val;
        }
    };

    template<>
    struct InitZero<tags::posField>
    {
        constexpr void operator()(auto posView) const
        {
            *posView[tags::x] = {0.f};
            *posView[tags::y] = {0.f};
            *posView[tags::z] = {0.f};
        }
    };

} // namespace pmacc::spearhed
