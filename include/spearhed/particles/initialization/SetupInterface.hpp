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
    // @TODO investigate a way to define setups and their parameters in a text file, which can be read at runtime
    template<typename T>
    concept SetupInterface
        = requires(T a, pmacc::spearhed::ParticleRegionBuffer<PRType>& prBuf, DeviceHeap const& deviceHeap) {
              // Requires a method named setupRegions with matching arguments that returns void
              { a.setupRegions(prBuf, deviceHeap) } -> std::same_as<void>;
              // Requires a domain member describing the full simulation domain
              { a.domain } -> std::convertible_to<pmacc::spearhed::AABB<CS>>;
          };


} // namespace spearhed
