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
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

/**
 * Temporary file which holds the currently simulated setup
 * NOTE: This is not a param file and thus is not included in spearhed/param.hpp
 * An index of setups can be found in share/spearhed , which holds the source of truth for this setup
 */
namespace spearhed
{
    struct SodShockTube
    {
        void setupRegions(pmacc::spearhed::ParticleRegionBuffer<PRType>& prBuf, DeviceHeap const& deviceHeap) const
        {
            PRType boundedParticles{deviceHeap.getAllocatorHandle()};

            prBuf.create(2);

            auto deviceHeapHandle = deviceHeap.getAllocatorHandle();

            // Define the left region
            pmacc::spearhed::AABB<CS> leftVolume{{0, 0, 0}, {0.0, 0.0, 0.0}, {0.5, 1.0, 1.0}};
            auto leftRegion = PRType{deviceHeapHandle, leftVolume};

            // Define the right region
            auto rightRegion = PRType{deviceHeapHandle, {{0, 0, 0}, {0.5, 0.0, 0.0}, {1.0, 1.0, 1.0}}};

            // Add regions to the particleRegions buffer
            prBuf.pushBack(leftRegion);
            prBuf.pushBack(rightRegion);
        }
    };
} // namespace spearhed
