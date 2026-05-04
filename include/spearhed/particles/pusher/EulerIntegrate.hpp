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
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Acceleration.hpp"
#include "spearhed/particles/attributes/DuDt.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace spearhed
{
    /**
     * Per-particle Euler update for velocity and internal energy.
     *
     *   v_i  += dvdt_i * dt
     *   u_i  += dudt_i * dt
     *
     * Position update (r += v*dt) is handled separately by PushVelocity, called
     * at the start of the following step.
     */
    struct EulerIntegrate
    {
        HDINLINE constexpr void operator()(auto& /*worker*/, auto& particle, T_dt delt) const
        {
            using namespace spearhed::tags;

            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { *particle[vel][tag] += *particle[dvdt][tag] * delt; });
            *particle[internalEnergy] += *particle[dudt] * delt;
        }
    };

} // namespace spearhed
