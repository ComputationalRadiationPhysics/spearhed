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
