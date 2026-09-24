/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/Decomposition.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/Relocation.hpp"
#include "spmacc/particles/regions/mapping/constant/Decomposition.hpp"

#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    using CS = spearhed::CS;
    using Grid = pmacc::spearhed::FixedCartesianGrid<CS>;
    using Fixture = spearhed::test::SpearhedParticleFixture<spearhed::simDim>;

    struct SeedPackedGrid
    {
        DINLINE void operator()(auto const& worker, auto regions) const
        {
            using namespace pmacc::spearhed::tags;
            using namespace spearhed::tags;

            uint32_t const regionSlot = worker.blockDomIdx();
            auto& frameList = regions[regionSlot].particleFrameList;
            using Frame = typename std::remove_cvref_t<decltype(frameList)>::FrameType;

            PMACC_SMEM(worker, frame, pmacc::spearhed::memory::FramePointer<Frame>);
            pmacc::lockstep::makeMaster(worker)(
                [&]
                {
                    frameList.setNumParticles(1u);
                    frame = frameList.getEmptyFrame(worker);
                    frame->liveParticles = 1u;
                });
            worker.sync();

            pmacc::lockstep::makeForEach<Frame::frameSize>(worker)(
                [&](uint32_t slot)
                {
                    auto particle = frame[slot];
                    particle[multiMask] = slot == 0u ? 1u : 0u;
                    if(slot != 0u)
                        return;

                    // Region zero deliberately owns a particle already across its x boundary.
                    // The other particle remains in region one; relocation must pack both records
                    // into region one's single frame without a gap.
                    particle[relativePos][x] = regionSlot == 0u ? 2.5f : 0.75f;
                    particle[relativePos][y] = 0.5f;
                    particle[relativePos][z] = 0.5f;
                    particle[particleId] = regionSlot == 0u ? 7u : 42u;
                    particle[mass] = regionSlot == 0u ? 3.0f : 5.0f;
                });
        }
    };

    struct MoveFirstParticleOutsideGrid
    {
        DINLINE void operator()(auto const& worker, auto regions) const
        {
            using namespace pmacc::spearhed::tags;
            uint32_t const regionSlot = worker.blockDomIdx();
            pmacc::lockstep::makeForEach<64u>(worker)(
                [&](uint32_t slot)
                {
                    if(regionSlot == 0u && slot == 0u)
                        regions[regionSlot].particleFrameList.begin()->operator[](slot)[relativePos][x] = 4.5f;
                });
        }
    };

    template<typename T>
    concept HasMaterializedCandidateArrays = requires(T provider) { provider.neighbourRegions; };

    Grid makeGrid()
    {
        Grid::Point const min{0.0f, 0.0f, 0.0f};
        Grid::Point const max{4.0f, 1.0f, 1.0f};
        return Grid{Grid::Bounds{min, max}, {2u, 1u, 1u}};
    }
} // namespace

TEST_CASE_METHOD(Fixture, "fixed Cartesian initial classification rebases setup particles", "[spatial][fixed-grid]")
{
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    auto const grid = makeGrid();
    auto const allocator = deviceHeap->getAllocatorHandle();
    prBuf->create(1u);
    pmacc::spearhed::MaterialRegionMetadata<CS> materialMetadata;
    materialMetadata.chart.origin = {0.0f, 0.0f, 0.0f};
    prBuf->pushBack(spearhed::PRType{allocator, materialMetadata});
    prBuf->buffer->hostToDevice();

    PMACC_LOCKSTEP_KERNEL(SeedPackedGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(1u))(prBuf->getDeviceDataBox());
    ++prBuf->topologyVersion;

    using Store = std::remove_reference_t<decltype(*prBuf)>;
    pmacc::spearhed::FixedCartesianDecomposition<CS, Store> decomposition{grid, *prBuf};
    auto const topologyBeforeClassification = prBuf->topologyVersion;
    decomposition.initializeFromMaterial(*prBuf, allocator);
    REQUIRE(prBuf->topologyVersion == topologyBeforeClassification + 1u);
    REQUIRE(decomposition.preparedFor(*prBuf).generation() == 1u);

    prBuf->buffer->deviceToHost();
    int64_t const heapOffset = spearhed::syncHeapToHost();
    auto regions = prBuf->buffer->getHostBuffer().getDataBox();
    REQUIRE(prBuf->size == 2);
    auto isLive = [](auto particle) { return static_cast<bool>(particle[multiMask]); };
    REQUIRE(regions[0].particleFrameList.isPacked(isLive));
    REQUIRE(regions[1].particleFrameList.isPacked(isLive));
    REQUIRE(regions[0].particleFrameList.getNumParticles() == 0u);
    auto& frame = *regions[1].particleFrameList.hostIterable(heapOffset).begin();
    REQUIRE(frame.liveParticles == 1u);
    REQUIRE(frame[0][particleId] == 7u);
    REQUIRE(frame[0][mass] == 3.0f);
    REQUIRE(grid.toWorld(1u, frame[0][relativePos].get())[x] == 2.5f);
}

TEST_CASE_METHOD(Fixture, "fixed Cartesian relocation preserves records and packed frames", "[spatial][fixed-grid]")
{
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    auto const grid = makeGrid();
    auto const allocator = deviceHeap->getAllocatorHandle();
    prBuf->create(grid.bucketCount());
    for(uint32_t slot = 0u; slot < grid.bucketCount(); ++slot)
    {
        pmacc::spearhed::MaterialRegionMetadata<CS> metadata;
        metadata.chart = grid.chart(slot);
        prBuf->pushBack(typename spearhed::PRType{allocator, metadata});
    }
    prBuf->buffer->hostToDevice();

    PMACC_LOCKSTEP_KERNEL(SeedPackedGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(grid.bucketCount()))(prBuf->getDeviceDataBox());
    ++prBuf->topologyVersion;

    auto const topologyBeforeRelocation = prBuf->topologyVersion;
    pmacc::spearhed::FixedCartesianRelocator<CS>{grid}.relocate(*prBuf);
    REQUIRE(prBuf->topologyVersion == topologyBeforeRelocation + 1u);

    // Same-grid plans retain no CSR candidate arrays: each source entry derives its
    // conservative cell stencil directly from the fixed grid at interaction time.
    using Store = std::remove_reference_t<decltype(*prBuf)>;
    pmacc::spearhed::FixedCartesianDecomposition<CS, Store> decomposition{grid, *prBuf};
    decomposition.prepareAfterMotion();
    auto prepared = decomposition.preparedFor(*prBuf);
    auto plan = pmacc::spearhed::makeInteractionPlan(prepared, pmacc::spearhed::InteractionQuery{1.0f}, prepared);
    auto& entry = plan.bySpecies(pmacc::spearhed::species::default_);
    STATIC_REQUIRE(pmacc::spearhed::CandidateProvider<decltype(entry.candidateProvider.deviceView())>);
    STATIC_REQUIRE_FALSE(HasMaterializedCandidateArrays<decltype(entry.candidateProvider)>);

    // Cross-mapping plans use the broad-phase CSR fallback in both directions.
    auto materialSource = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>();
    materialSource->create(1u);
    materialSource->pushBack(spearhed::PRType{allocator, pmacc::spearhed::MaterialRegionMetadata<CS>{}});
    materialSource->buffer->hostToDevice();
    PMACC_LOCKSTEP_KERNEL(SeedPackedGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(1u))(materialSource->getDeviceDataBox());
    ++materialSource->topologyVersion;

    pmacc::spearhed::MaterialAabbDecomposition material{*materialSource};
    material.prepareAfterMotion();
    auto fixedMaterialPlan = pmacc::spearhed::makeInteractionPlan(
        prepared,
        pmacc::spearhed::InteractionQuery{1.0f},
        material.preparedFor(*materialSource));
    auto materialFixedPlan = pmacc::spearhed::makeInteractionPlan(
        material.preparedFor(*materialSource),
        pmacc::spearhed::InteractionQuery{1.0f},
        prepared);
    auto& fixedMaterialEntry = fixedMaterialPlan.bySpecies(pmacc::spearhed::species::default_);
    auto& materialFixedEntry = materialFixedPlan.bySpecies(pmacc::spearhed::species::default_);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(fixedMaterialEntry.candidateProvider)>);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(materialFixedEntry.candidateProvider)>);
    fixedMaterialEntry.candidateProvider.regionOffsets.deviceToHost();
    materialFixedEntry.candidateProvider.regionOffsets.deviceToHost();
    auto const fixedMaterialOffsets = fixedMaterialEntry.candidateProvider.regionOffsets.getHostBuffer().getDataBox();
    auto const materialFixedOffsets = materialFixedEntry.candidateProvider.regionOffsets.getHostBuffer().getDataBox();
    REQUIRE(fixedMaterialOffsets[0] == 0u);
    REQUIRE(fixedMaterialOffsets[2] == 2u);
    REQUIRE(materialFixedOffsets[0] == 0u);
    REQUIRE(materialFixedOffsets[1] == 2u);
    REQUIRE(material.preparedFor(*materialSource).chart(0u).origin[x] - prepared.chart(1u).origin[x] == -2.0f);

    auto fixedSource = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>();
    auto const sourceGrid = Grid{grid.domain, {1u, 1u, 1u}};
    fixedSource->create(sourceGrid.bucketCount());
    pmacc::spearhed::MaterialRegionMetadata<CS> fixedSourceMetadata;
    fixedSourceMetadata.chart = sourceGrid.chart(0u);
    fixedSource->pushBack(spearhed::PRType{allocator, fixedSourceMetadata});
    fixedSource->buffer->hostToDevice();
    PMACC_LOCKSTEP_KERNEL(SeedPackedGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(1u))(fixedSource->getDeviceDataBox());
    ++fixedSource->topologyVersion;
    using FixedSourceStore = std::remove_reference_t<decltype(*fixedSource)>;
    pmacc::spearhed::FixedCartesianDecomposition<CS, FixedSourceStore> fixedSourceDecomposition{
        sourceGrid,
        *fixedSource};
    fixedSourceDecomposition.prepareAfterMotion();
    auto fixedFixedPlan = pmacc::spearhed::makeInteractionPlan(
        prepared,
        pmacc::spearhed::InteractionQuery{1.0f},
        fixedSourceDecomposition.preparedFor(*fixedSource));
    auto& fixedFixedEntry = fixedFixedPlan.bySpecies(pmacc::spearhed::species::default_);
    STATIC_REQUIRE_FALSE(HasMaterializedCandidateArrays<decltype(fixedFixedEntry.candidateProvider)>);
    REQUIRE(fixedFixedEntry.candidateProvider.targetGrid.bucketCount() == 2u);
    REQUIRE(fixedFixedEntry.candidateProvider.sourceGrid.bucketCount() == 1u);
    REQUIRE(
        fixedFixedEntry.candidateProvider.sourceGrid.chart(0u).origin[x]
            - fixedFixedEntry.candidateProvider.targetGrid.chart(1u).origin[x]
        == -2.0f);

    auto emptyMaterialSource = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>();
    emptyMaterialSource->create(0u);
    pmacc::spearhed::MaterialAabbDecomposition emptyMaterial{*emptyMaterialSource};
    emptyMaterial.prepareAfterMotion();
    auto emptyPlan = pmacc::spearhed::makeInteractionPlan(
        prepared,
        pmacc::spearhed::InteractionQuery{1.0f},
        emptyMaterial.preparedFor(*emptyMaterialSource));
    auto& emptyEntry = emptyPlan.bySpecies(pmacc::spearhed::species::default_);
    emptyEntry.candidateProvider.regionOffsets.deviceToHost();
    auto const emptyOffsets = emptyEntry.candidateProvider.regionOffsets.getHostBuffer().getDataBox();
    REQUIRE(emptyOffsets[0] == 0u);
    REQUIRE(emptyOffsets[2] == 0u);

    prBuf->buffer->deviceToHost();
    int64_t const heapOffset = spearhed::syncHeapToHost();
    auto regions = prBuf->buffer->getHostBuffer().getDataBox();

    auto isLive = [](auto particle) { return static_cast<bool>(particle[multiMask]); };
    REQUIRE(regions[0].particleFrameList.isPacked(isLive));
    REQUIRE(regions[1].particleFrameList.isPacked(isLive));
    REQUIRE(regions[0].particleFrameList.getNumParticles() == 0u);
    REQUIRE(regions[0].particleFrameList.numFrames() == 0u);

    auto& destinationFrames = regions[1].particleFrameList;
    REQUIRE(destinationFrames.getNumParticles() == 2u);
    REQUIRE(destinationFrames.numFrames() == 1u);
    auto& frame = *destinationFrames.hostIterable(heapOffset).begin();
    REQUIRE(frame.liveParticles == 2u);
    REQUIRE(frame[0][multiMask] == 1u);
    REQUIRE(frame[1][multiMask] == 1u);
    for(uint32_t slot = 2u; slot < spearhed::numFrameSlots; ++slot)
        REQUIRE(frame[slot][multiMask] == 0u);

    std::vector<uint64_t> ids{frame[0][particleId], frame[1][particleId]};
    std::ranges::sort(ids);
    REQUIRE(ids == std::vector<uint64_t>{7u, 42u});
    REQUIRE(frame[0][mass] + frame[1][mass] == 8.0f);

    std::vector<float> worldX{
        grid.toWorld(1u, frame[0][relativePos].get())[x],
        grid.toWorld(1u, frame[1][relativePos].get())[x]};
    std::ranges::sort(worldX);
    REQUIRE(worldX == std::vector<float>{2.5f, 2.75f});
}

TEST_CASE_METHOD(Fixture, "fixed Cartesian relocation publishes two consecutive replacements", "[spatial][fixed-grid]")
{
    using namespace pmacc::spearhed::tags;

    auto const grid = makeGrid();
    auto const allocator = deviceHeap->getAllocatorHandle();
    prBuf->create(grid.bucketCount());
    for(uint32_t slot = 0u; slot < grid.bucketCount(); ++slot)
    {
        pmacc::spearhed::MaterialRegionMetadata<CS> metadata;
        metadata.chart = grid.chart(slot);
        prBuf->pushBack(typename spearhed::PRType{allocator, metadata});
    }
    prBuf->buffer->hostToDevice();
    PMACC_LOCKSTEP_KERNEL(SeedPackedGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(grid.bucketCount()))(prBuf->getDeviceDataBox());
    ++prBuf->topologyVersion;

    auto const topologyBefore = prBuf->topologyVersion;
    pmacc::spearhed::FixedCartesianRelocator<CS>{grid}.relocate(*prBuf);
    pmacc::spearhed::FixedCartesianRelocator<CS>{grid}.relocate(*prBuf);
    REQUIRE(prBuf->topologyVersion == topologyBefore + 2u);

    prBuf->buffer->deviceToHost();
    int64_t const heapOffset = spearhed::syncHeapToHost();
    auto regions = prBuf->buffer->getHostBuffer().getDataBox();
    auto isLive = [](auto particle) { return static_cast<bool>(particle[multiMask]); };
    REQUIRE(regions[0].particleFrameList.isPacked(isLive));
    REQUIRE(regions[1].particleFrameList.isPacked(isLive));
    REQUIRE(regions[1].particleFrameList.getNumParticles() == 2u);
    auto& frame = *regions[1].particleFrameList.hostIterable(heapOffset).begin();
    REQUIRE(frame[0][spearhed::tags::particleId] != frame[1][spearhed::tags::particleId]);
}

TEST_CASE_METHOD(Fixture, "failed fixed Cartesian relocation preserves source storage", "[spatial][fixed-grid]")
{
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    auto const grid = makeGrid();
    auto const allocator = deviceHeap->getAllocatorHandle();
    prBuf->create(grid.bucketCount());
    for(uint32_t slot = 0u; slot < grid.bucketCount(); ++slot)
    {
        pmacc::spearhed::MaterialRegionMetadata<CS> metadata;
        metadata.chart = grid.chart(slot);
        prBuf->pushBack(typename spearhed::PRType{allocator, metadata});
    }
    prBuf->buffer->hostToDevice();
    PMACC_LOCKSTEP_KERNEL(SeedPackedGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(grid.bucketCount()))(prBuf->getDeviceDataBox());
    PMACC_LOCKSTEP_KERNEL(MoveFirstParticleOutsideGrid{})
        .template config<64u>(pmacc::DataSpace<DIM1>(grid.bucketCount()))(prBuf->getDeviceDataBox());
    ++prBuf->topologyVersion;

    auto const topologyBefore = prBuf->topologyVersion;
    REQUIRE_THROWS_AS(pmacc::spearhed::FixedCartesianRelocator<CS>{grid}.relocate(*prBuf), std::out_of_range);
    REQUIRE(prBuf->topologyVersion == topologyBefore);

    prBuf->buffer->deviceToHost();
    int64_t const heapOffset = spearhed::syncHeapToHost();
    auto regions = prBuf->buffer->getHostBuffer().getDataBox();
    auto isLive = [](auto particle) { return static_cast<bool>(particle[multiMask]); };
    REQUIRE(regions[0].particleFrameList.isPacked(isLive));
    REQUIRE(regions[1].particleFrameList.isPacked(isLive));
    REQUIRE(regions[0].particleFrameList.getNumParticles() == 1u);
    REQUIRE(regions[1].particleFrameList.getNumParticles() == 1u);

    auto& firstFrame = *regions[0].particleFrameList.hostIterable(heapOffset).begin();
    auto& secondFrame = *regions[1].particleFrameList.hostIterable(heapOffset).begin();
    REQUIRE(firstFrame[0][particleId] == 7u);
    REQUIRE(firstFrame[0][mass] == 3.0f);
    REQUIRE(secondFrame[0][particleId] == 42u);
    REQUIRE(secondFrame[0][mass] == 5.0f);
    REQUIRE(grid.toWorld(0u, firstFrame[0][relativePos].get())[x] == 4.5f);
}
