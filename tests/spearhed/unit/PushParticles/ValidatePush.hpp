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
#include "spearhed/param.hpp"
#include "spmacc/particles/algorithms/HierarchyForEach.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

/** Validates that every particle was pushed to the expected position.
 *
 * Host-side verification sweep: after all kernels are done, mirror the region metadata and the
 * device heap to the host, then count particles whose relativePos is not approx Vec(1.f).
 */
struct ValidatePush
{
    auto operator()() const -> void
    {
        namespace sp = pmacc::spearhed;

        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>(
            pmacc::spearhed::prBufId(pmacc::spearhed::species::default_));

        // Sync order matters on GPU backends: region metadata first, then the device heap.
        prBuf.synchronize();
        auto const heapOffset = spearhed::syncHeapToHost();

        int errorCount = 0;
        sp::forEach(
            sp::levels::particle,
            sp::hostHeap(heapOffset),
            sp::hostSpecies(prBuf),
            [&](auto particle)
            {
                if(!particle[spearhed::relativePos].get().isApprox(
                       pmacc::spearhed::Vec<spearhed::CS, pmacc::spearhed::ValueStorage<spearhed::CS>>(1.f)))
                    ++errorCount;
            });

        INFO("Number of particles with incorrect positions: " << errorCount);
        REQUIRE(errorCount == 0);
    }
};
