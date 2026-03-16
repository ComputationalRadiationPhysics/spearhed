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

#include "spearhed/ParticleView.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace spearhed
{
    struct PushVelocity
    {
        HDINLINE constexpr void operator()(auto worker, ParticleView<relativePos, vel> view, T_dt delt) const
        {
            *view[relativePos][x] += *view[vel][x] * delt;
            *view[relativePos][y] += *view[vel][y] * delt;
            *view[relativePos][z] += *view[vel][z] * delt;
        }
    };

} // namespace spearhed
