/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/mapping/AdaptiveSplit/Decomposition.hpp"
#include "spmacc/particles/regions/mapping/AdaptiveSplit/Policies.hpp"

#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    using CS = spearhed::CS;
    using Fixture = spearhed::test::SpearhedParticleFixture<spearhed::simDim>;

    struct AllLeft
    {
        static constexpr uint32_t partitionCount = 2u;

        HDINLINE uint32_t operator()(auto, auto const&) const
        {
            return 0u;
        }
    };

    struct SeedTwoParticles
    {
        DINLINE void operator()(auto const& worker, auto regions) const
        {
            using namespace pmacc::spearhed::tags;
            using namespace spearhed::tags;
            using Frame = typename spearhed::PRType::FrameType;
            auto& frameList = regions[worker.blockDomIdx()].particleFrameList;
            PMACC_SMEM(worker, frame, pmacc::spearhed::memory::FramePointer<Frame>);
            pmacc::lockstep::makeMaster(worker)(
                [&]
                {
                    frameList.setNumParticles(2u);
                    frame = frameList.getEmptyFrame(worker);
                    frame->liveParticles = 2u;
                });
            worker.sync();
            pmacc::lockstep::makeForEach<Frame::frameSize>(worker)(
                [&](uint32_t slot)
                {
                    auto particle = frame[slot];
                    particle[multiMask] = slot < 2u ? 1u : 0u;
                    if(slot >= 2u)
                        return;
                    particle[relativePos][x] = slot == 0u ? 0.25f : 0.75f;
                    particle[relativePos][y] = 0.5f;
                    particle[relativePos][z] = 0.5f;
                    particle[particleId] = slot == 0u ? 7u : 42u;
                    particle[mass] = slot == 0u ? 3.0f : 5.0f;
                });
        }
    };

    void seedTwoParticles(Fixture& fixture)
    {
        auto const allocator = fixture.deviceHeap->getAllocatorHandle();
        fixture.prBuf->create(1u);
        pmacc::spearhed::MaterialRegionMetadata<CS> metadata;
        metadata.chart.origin = {0.0f, 0.0f, 0.0f};
        fixture.prBuf->pushBack(spearhed::PRType{allocator, metadata});
        fixture.prBuf->buffer->hostToDevice();
        PMACC_LOCKSTEP_KERNEL(SeedTwoParticles{})
            .template config<64u>(pmacc::DataSpace<DIM1>(1u))(fixture.prBuf->getDeviceDataBox());
        ++fixture.prBuf->topologyVersion;
    }
} // namespace

TEST_CASE_METHOD(Fixture, "adaptive binary split preserves records and topology", "[spatial][adaptive-split]")
{
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    seedTwoParticles(*this);

    using Partition = pmacc::spearhed::WorldAxisMidpoint<CS, pmacc::spearhed::tags::x_t>;
    pmacc::spearhed::AdaptiveSplitDecomposition decomposition{
        pmacc::spearhed::ParticleCountExceeds{1u},
        Partition{},
        *prBuf};
    auto const topologyBefore = prBuf->topologyVersion;
    decomposition.prepareAfterMotion();
    REQUIRE(prBuf->topologyVersion == topologyBefore + 1u);
    REQUIRE(decomposition.preparedFor(*prBuf).generation() == 1u);

    prBuf->buffer->deviceToHost();
    int64_t const heapOffset = spearhed::syncHeapToHost();
    auto regions = prBuf->buffer->getHostBuffer().getDataBox();
    REQUIRE(prBuf->size == 2);
    REQUIRE(regions[0].spatial.chart.origin[x] == 0.0f);
    REQUIRE(regions[1].spatial.chart.origin[x] == 0.0f);
    std::vector<uint64_t> ids;
    std::vector<float> worldX;
    float totalMass = 0.0f;
    for(int region = 0; region < prBuf->size; ++region)
    {
        auto isLive = [](auto particle) { return static_cast<bool>(particle[multiMask]); };
        REQUIRE(regions[region].particleFrameList.isPacked(isLive));
        REQUIRE(regions[region].particleFrameList.getNumParticles() == 1u);
        auto& frame = *regions[region].particleFrameList.hostIterable(heapOffset).begin();
        ids.push_back(frame[0][particleId]);
        worldX.push_back(regions[region].spatial.chart.toWorld(frame[0][relativePos].get())[x]);
        totalMass += frame[0][mass];
        REQUIRE(regions[region].spatial.occupancy.min[x] == worldX.back());
        REQUIRE(regions[region].spatial.occupancy.max[x] == worldX.back());
    }
    std::ranges::sort(ids);
    std::ranges::sort(worldX);
    REQUIRE(ids == std::vector<uint64_t>{7u, 42u});
    REQUIRE(worldX == std::vector<float>{0.25f, 0.75f});
    REQUIRE(totalMass == 8.0f);

    decomposition.prepareAfterMotion();
    REQUIRE(prBuf->topologyVersion == topologyBefore + 1u);
    REQUIRE(decomposition.preparedFor(*prBuf).generation() == 2u);
}

TEST_CASE_METHOD(Fixture, "adaptive preparation advances generation without a split", "[spatial][adaptive-split]")
{
    seedTwoParticles(*this);
    using Partition = pmacc::spearhed::WorldAxisMidpoint<CS, pmacc::spearhed::tags::x_t>;
    pmacc::spearhed::AdaptiveSplitDecomposition decomposition{
        pmacc::spearhed::ParticleCountExceeds{2u},
        Partition{},
        *prBuf};

    auto const topologyBefore = prBuf->topologyVersion;
    decomposition.prepareAfterMotion();
    REQUIRE(prBuf->topologyVersion == topologyBefore);
    REQUIRE(prBuf->size == 1);
    REQUIRE(decomposition.preparedFor(*prBuf).generation() == 1u);
    decomposition.prepareAfterMotion();
    REQUIRE(prBuf->topologyVersion == topologyBefore);
    REQUIRE(decomposition.preparedFor(*prBuf).generation() == 2u);
}

TEST_CASE_METHOD(Fixture, "adaptive split skips an empty partition", "[spatial][adaptive-split]")
{
    seedTwoParticles(*this);
    pmacc::spearhed::AdaptiveSplitDecomposition decomposition{
        pmacc::spearhed::ParticleCountExceeds{1u},
        AllLeft{},
        *prBuf};

    auto const topologyBefore = prBuf->topologyVersion;
    decomposition.prepareAfterMotion();
    REQUIRE(prBuf->topologyVersion == topologyBefore);
    REQUIRE(prBuf->size == 1);
    REQUIRE(decomposition.preparedFor(*prBuf).generation() == 1u);
}
