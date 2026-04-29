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

#include "spearhed/param.hpp"

#include <concepts>

namespace spearhed
{
    /**
     * Concept describing the SPH kernel interface used throughout the code.
     *
     * A kernel type @c K is a stateless tag struct exposing the following
     * static, host-and-device-callable members for a given CoordinateSystem CS:
     *   - W<CS>(r, h)     -> T_Axis    : the kernel value
     *   - dWdr<CS>(r, h)  -> T_Axis    : the radial derivative
     *   - gradW<CS>(rVec, r, h)        : the vector gradient
     *   - supportRadius                : integer multiplier on h beyond which W = 0
     *   - name                         : short identifier used by the CLI / logging
     *
     * gradW is intentionally not pinned in the concept (its argument type would
     * require pulling in Vec headers transitively); a use-site failure to provide
     * it produces a regular template error.
     */
    template<typename K>
    concept SphKernel = requires(typename CS::T_Axis r, typename CS::T_Axis h) {
        { K::W(r, h) } -> std::same_as<typename CS::T_Axis>;
        { K::dWdr(r, h) } -> std::same_as<typename CS::T_Axis>;
        { K::supportRadius } -> std::convertible_to<int>;
    };
} // namespace spearhed
