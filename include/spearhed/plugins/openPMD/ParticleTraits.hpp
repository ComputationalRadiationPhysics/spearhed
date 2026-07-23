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

#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/plugins/openPMD/Position.hpp"

#include <array>
#include <string_view>

namespace spearhed::output
{
    /**
     * openPMD metadata for a particle attribute.
     *
     * Primary template is intentionally undefined - a missing specialisation
     * causes a clear compile error rather than silent wrong metadata.
     *
     * unitDimension follows the SI 7-vector convention:
     *   [length, mass, time, electric current, temperature, amount, luminosity]
     */
    template<typename Tag>
    struct OpenPMDTrait;

    template<>
    struct OpenPMDTrait<spearhed::tags::particleId_t>
    {
        static constexpr std::string_view record = "id";
        static constexpr std::array<double, 7> unitDimension = {0, 0, 0, 0, 0, 0, 0};
        static constexpr double unitSI = 1.0;
    };

    template<>
    struct OpenPMDTrait<position_t>
    {
        static constexpr std::string_view record = "position";
        // SI: metres
        static constexpr std::array<double, 7> unitDimension = {1, 0, 0, 0, 0, 0, 0};
        static constexpr double unitSI = 1.0;
    };

    template<>
    struct OpenPMDTrait<spearhed::tags::mass_t>
    {
        static constexpr std::string_view record = "mass";
        // SI: kilogram
        static constexpr std::array<double, 7> unitDimension = {0, 1, 0, 0, 0, 0, 0};
        static constexpr double unitSI = 1.0;
    };

    template<>
    struct OpenPMDTrait<spearhed::tags::vel_t>
    {
        static constexpr std::string_view record = "velocity";
        // SI: metres per second
        static constexpr std::array<double, 7> unitDimension = {1, 0, -1, 0, 0, 0, 0};
        static constexpr double unitSI = 1.0;
    };

    template<>
    struct OpenPMDTrait<spearhed::tags::density_t>
    {
        static constexpr std::string_view record = "mass_density";
        // SI: kg / m^3
        static constexpr std::array<double, 7> unitDimension = {-3, 1, 0, 0, 0, 0, 0};
        static constexpr double unitSI = 1.0;
    };

    template<>
    struct OpenPMDTrait<spearhed::tags::internalEnergy_t>
    {
        static constexpr std::string_view record = "specific_internal_energy";
        // SI: J / kg = m^2 / s^2
        static constexpr std::array<double, 7> unitDimension = {2, 0, -2, 0, 0, 0, 0};
        static constexpr double unitSI = 1.0;
    };

} // namespace spearhed::output
