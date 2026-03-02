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
