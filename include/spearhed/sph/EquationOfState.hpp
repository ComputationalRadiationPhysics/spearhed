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

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cmath>

namespace spearhed
{
    /**
     * Ideal-gas pressure: P = (gamma - 1) * rho * u.
     */
    template<typename T>
    HDINLINE T pressure(T gamma, T rho, T u) noexcept
    {
        return (gamma - T{1}) * rho * u;
    }

    /**
     * Ideal-gas sound speed: c_s = sqrt(gamma * P / rho).
     *
     * Caller must ensure rho > 0.
     */
    template<typename T>
    HDINLINE T soundSpeed(T gamma, T p, T rho) noexcept
    {
        return static_cast<T>(std::sqrt(static_cast<double>(gamma * p / rho)));
    }

} // namespace spearhed
