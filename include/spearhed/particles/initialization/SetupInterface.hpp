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

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

namespace spearhed
{
    // @TODO investigate a way to define setups and their parameters in a text file, which can be read at runtime
    // Interior is used as proxy role to check per-role template requirements.
    template<typename T>
    concept SetupInterface = requires(
        T a,
        pmacc::spearhed::ParticleRegionBuffer<PRType, pmacc::spearhed::roles::Interior>& prBuf,
        DeviceHeap const& deviceHeap) {
        typename T::Roles;
        { a.template setupRegions<pmacc::spearhed::roles::Interior>(prBuf, deviceHeap) } -> std::same_as<void>;
        { a.domain } -> std::convertible_to<pmacc::spearhed::AABB<CS>>;
        typename T::template NumParticlesToCreate<pmacc::spearhed::roles::Interior>;
        // This is currently a std::tuple unpacked on the host side
        // TODO switch to a device friendly compile time dictionary
        { a.template numParticlesToCreateArgs<pmacc::spearhed::roles::Interior>() };
        typename T::template PlaceParticle<pmacc::spearhed::roles::Interior>;
        { a.template placeParticleArgs<pmacc::spearhed::roles::Interior>() };
    };

} // namespace spearhed
