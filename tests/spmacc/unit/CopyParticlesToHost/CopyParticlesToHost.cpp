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

#include "TestSetup.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/CopyParticlesToDynSoA.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

struct OffsetParticleIds
{
    HDINLINE constexpr void operator()(auto&, auto& particle) const
    {
        particle[spearhed::particleId] += uint64_t{1000};
    }
};

TEST_CASE_METHOD(ParticleFixture, "CopyParticlesToDynSoA correctness", "[integration][particles][copy]")
{
    // 2 regions: region 0 gets 130 particles, region 1 gets 260 particles.
    // With numFrameSlots=64: region 0 has 3 frames (2 full + 1 partial),
    //                        region 1 has 5 frames (4 full + 1 partial).
    constexpr uint64_t numRegions = 2;
    auto setup = spearhed::EmptyNRegions<numRegions>{};
    setup.baseNumParticlesToCreate = 130u;
    spearhed::InitRegions{}(*deviceHeap, setup);

    spearhed::InitParticles{}(setup);
    // InitParticles syncs via its own explicit waitForAllTasks() call in initBlockSlice.

    // Sync region metadata to host.
    prBuf->synchronize();

    // Count expected total from host metadata.
    uint64_t totalParticles = 0;
    {
        auto hostBox = prBuf->buffer->getHostBuffer().getDataBox();
        for(int r = 0; r < prBuf->size; ++r)
            totalParticles += hostBox[r].particleFrameList.getNumParticles();
    }

    // Sync device heap to host and obtain the pointer offset for frame translation.
    auto heapOffset = spearhed::syncHeapToHost();

    // only serialize a subset of the tags
    // vel isnt used but still copied to check if the iterative path traversal based copy is working
    using OutputRecord
        = ll::sub_record_t<spearhed::FrameType::ParticleRecord, spearhed::tags::particleId, spearhed::tags::vel>;

    ll::DynSoA<OutputRecord> dynSoa;
    pmacc::spearhed::CopyParticlesToDynSoA{}(*prBuf, dynSoa, heapOffset);

    REQUIRE(dynSoa.size() == totalParticles);

    // Collect all particle IDs and verify they form the contiguous range [0, totalParticles).
    auto idSpan = dynSoa.getLeaf(spearhed::tags::particleId);
    std::vector<uint64_t> ids(idSpan.begin(), idSpan.end());
    std::ranges::sort(ids);

    std::vector<uint64_t> expected(totalParticles);
    std::iota(expected.begin(), expected.end(), uint64_t{0});

    REQUIRE(ids == expected);

    // Update heap-resident frame data and ensure the next snapshot copies those changes to the host.
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, OffsetParticleIds{});
    heapOffset = spearhed::syncHeapToHost();

    ll::DynSoA<OutputRecord> updatedSoa;
    pmacc::spearhed::CopyParticlesToDynSoA{}(*prBuf, updatedSoa, heapOffset);
    auto updatedIdSpan = updatedSoa.getLeaf(spearhed::tags::particleId);
    std::vector<uint64_t> updatedIds(updatedIdSpan.begin(), updatedIdSpan.end());
    std::ranges::sort(updatedIds);

    for(auto& id : expected)
        id += 1000;
    REQUIRE(updatedIds == expected);
}
