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
#include "spearhed/memory.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spmacc/particles/algorithms/HierarchyForEach.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <cstdint>

/** Computes the sum of all particle IDs in the simulation.
 *
 * Host-side verification sweep: after all kernels are done, mirror the region metadata and the
 * device heap to the host, then walk every live particle with the hierarchy iterator.
 *
 * @return Sum of all particle IDs.
 */
struct ComputeParticleIdSum
{
    auto operator()() const -> uint64_t
    {
        namespace sp = pmacc::spearhed;

        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>(
            pmacc::spearhed::prBufId(pmacc::spearhed::species::default_));

        // Sync order matters on GPU backends: region metadata first, then the device heap.
        prBuf.synchronize();
        auto const heapOffset = spearhed::syncHeapToHost();

        uint64_t totalSum = 0;
        sp::forEach(
            sp::levels::particle,
            sp::hostHeap(heapOffset),
            sp::hostSpecies(prBuf),
            [&](auto particle) { totalSum += particle[spearhed::particleId]; });

        return totalSum;
    }
};
