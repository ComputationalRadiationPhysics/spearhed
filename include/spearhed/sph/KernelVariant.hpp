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

#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spearhed/sph/QuinticSplineKernel.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace spearhed
{
    /**
     * Runtime selector for the SPH kernel.
     *
     * the simulation stores a KernelVariant that std::visit dispatches into a fully-templated device path
     */
    enum class KernelType
    {
        CubicSpline,
        QuinticSpline
    };

    /// std::variant of all available kernel tag types.
    using KernelVariant = std::variant<CubicSplineKernel, QuinticSplineKernel>;

    inline KernelVariant makeKernel(KernelType t)
    {
        switch(t)
        {
        case KernelType::CubicSpline:
            return CubicSplineKernel{};
        case KernelType::QuinticSpline:
            return QuinticSplineKernel{};
        }
        throw std::invalid_argument{"unknown KernelType"};
    }

} // namespace spearhed
