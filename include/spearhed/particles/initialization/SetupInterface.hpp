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
#include "spmacc/particles/regions/RegionRole.hpp"

#include <type_traits>
#include <utility>

namespace spearhed
{
    // A setup is composed of one "block" per role. A block describes how a single role's
    // particles are counted, placed, and how its regions are created. Single-role setups can
    // act as their own block; multi-role setups return a distinct block object per role.
    //
    // @TODO investigate a way to define setups and their parameters in a text file, which can be read at runtime
    template<typename B>
    concept SetupBlock = requires(
        B b,
        pmacc::spearhed::ParticleRegionBuffer<PRType, pmacc::spearhed::roles::Interior>& prBuf,
        DeviceHeap const& deviceHeap) {
        { b.setupRegions(prBuf, deviceHeap) } -> std::same_as<void>;
        // Functor type deciding how many particles a region creates.
        typename B::NumParticlesToCreate;
        // Host-side args forwarded into NumParticlesToCreate.
        // This is currently a std::tuple unpacked on the host side
        // TODO switch to a device friendly compile time dictionary
        { b.numParticlesToCreateArgs() };
        // Functor type placing each particle's attributes.
        typename B::PlaceParticle;
        // Host-side args forwarded into PlaceParticle.
        { b.placeParticleArgs() };
    };

    // The block type a setup exposes for a given role.
    template<typename T, typename Role>
    using SetupBlockOf = std::remove_cvref_t<decltype(std::declval<T const&>().template block<Role>())>;

    template<typename T>
    concept SetupInterface = requires(T a) {
        // Compile-time list (std::tuple) of the roles this setup defines.
        typename T::Roles;
        { a.domain } -> std::convertible_to<pmacc::spearhed::AABB<CS>>;
    } && SetupBlock<SetupBlockOf<T, pmacc::spearhed::roles::Interior>>;

} // namespace spearhed
