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
    // "Temporary" definition of the interface for simulation setups
    // Per-role block interface: satisfied by any single-role setup or block sub-object.
    // @TODO investigate a way to define setups and their parameters in a text file, which can be read at runtime
    template<typename T>
    concept SetupInterface
        = requires(T a, pmacc::spearhed::ParticleRegionBuffer<PRType>& prBuf, DeviceHeap const& deviceHeap) {
              { a.setupRegions(prBuf, deviceHeap) } -> std::same_as<void>;
              { a.domain } -> std::convertible_to<pmacc::spearhed::AABB<CS>>;
              typename T::NumParticlesToCreate;
              // This is currently a std::tuple unpacked on the host side
              // TODO switch to a device friendly compile time dictionary
              { a.numParticlesToCreateArgs() };
              typename T::PlaceParticle;
              { a.placeParticleArgs() };
          };

    // Multi-role setup: exposes Roles as a std::tuple of role tags and a block<Role>() accessor
    // returning an object satisfying SetupInterface for each role.
    template<typename T>
    concept MultiRoleSetup = requires { typename T::Roles; } && !SetupInterface<T>;

} // namespace spearhed
