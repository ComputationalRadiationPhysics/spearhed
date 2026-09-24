/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/regions/mapping/AdaptiveSplit/Decomposition.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/Decomposition.hpp"
#include "spmacc/particles/regions/mapping/constant/Decomposition.hpp"
#include "spmacc/particles/spatial/BroadPhaseView.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    using CS = spearhed::CS;
    using Grid = pmacc::spearhed::FixedCartesianGrid<CS>;
    using Metadata = pmacc::spearhed::MaterialRegionMetadata<CS>;
    using Buffer = pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>;
    using Fixture = spearhed::test::SpearhedParticleFixture<spearhed::simDim>;

    struct DummyStore
    {
    };

    template<typename T_Tag, typename T_View, typename T_Store>
    struct PreparedGeometry
    {
        using MappingTag = T_Tag;
        using Store = T_Store;

        uint64_t generation() const
        {
            return 1u;
        }

        uint32_t bucketCount() const
        {
            return view.bucketCount();
        }

        auto chart(uint32_t slot) const
        {
            return view.chart(slot);
        }

        uint32_t deviceView() const
        {
            return 0u;
        }

        T_Store& store() const
        {
            return *sourceStore;
        }

        void assertCurrent() const
        {
        }

        T_View broadPhaseView() const
        {
            return view;
        }

        T_Store* sourceStore;
        T_View view;
    };

    template<typename T>
    concept HasMaterializedCandidateArrays = requires(T provider) { provider.neighbourRegions; };

    template<typename T_Provider>
    std::vector<unsigned int> offsets(T_Provider& provider)
    {
        provider.regionOffsets.deviceToHost();
        auto const data = provider.regionOffsets.getHostBuffer().getDataBox();
        std::vector<unsigned int> result(provider.regionOffsets.getHostBuffer().capacityND().productOfComponents());
        for(size_t i = 0u; i < result.size(); ++i)
            result[i] = data[static_cast<int>(i)];
        return result;
    }

    auto materialPrepared(Buffer& buffer)
    {
        using View = pmacc::spearhed::MaterialAabbBroadPhaseView<decltype(buffer.getDeviceDataBox())>;
        return PreparedGeometry<pmacc::spearhed::MaterialAabbMappingTag, View, Buffer>{
            &buffer,
            {buffer.getDeviceDataBox(), static_cast<uint32_t>(buffer.size)}};
    }

    auto adaptivePrepared(Buffer& buffer)
    {
        using View = pmacc::spearhed::MaterialAabbBroadPhaseView<decltype(buffer.getDeviceDataBox())>;
        return PreparedGeometry<pmacc::spearhed::AdaptiveSplitMappingTag, View, Buffer>{
            &buffer,
            {buffer.getDeviceDataBox(), static_cast<uint32_t>(buffer.size)}};
    }

} // namespace

TEST_CASE_METHOD(Fixture, "material pair selection uses the CSR fallback", "[spatial][pair-provider]")
{
    prBuf->create(1u);
    Metadata metadata;
    metadata.chart.origin = {3.0f, 0.0f, 0.0f};
    metadata.occupancy = {{3.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}};
    prBuf->pushBack(spearhed::PRType{deviceHeap->getAllocatorHandle(), metadata});
    prBuf->buffer->hostToDevice();

    auto prepared = materialPrepared(*prBuf);
    auto provider
        = pmacc::spearhed::makeCandidateProvider(prepared, pmacc::spearhed::InteractionQuery{0.0f}, prepared);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(provider)>);
    REQUIRE(offsets(provider) == std::vector<unsigned int>{0u, 1u});
}

TEST_CASE_METHOD(Fixture, "mixed mapping pairs use the CSR fallback", "[spatial][pair-provider]")
{
    prBuf->create(1u);
    Metadata metadata;
    metadata.chart.origin = {3.0f, 0.0f, 0.0f};
    metadata.occupancy = {{3.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}};
    prBuf->pushBack(spearhed::PRType{deviceHeap->getAllocatorHandle(), metadata});
    prBuf->buffer->hostToDevice();
    auto materialPreparedSet = materialPrepared(*prBuf);

    Grid::Bounds const bounds{{2.0f, -1.0f, -1.0f}, {4.0f, 1.0f, 1.0f}};
    Grid const grid{bounds, {2u, 1u, 1u}};
    DummyStore fixedStore;
    auto fixedPreparedSet = PreparedGeometry<
        pmacc::spearhed::FixedCartesianMappingTag,
        pmacc::spearhed::FixedCartesianBroadPhaseView<CS>,
        DummyStore>{&fixedStore, {grid}};
    auto const query = pmacc::spearhed::InteractionQuery{0.0f};

    auto materialFixed = pmacc::spearhed::makeCandidateProvider(materialPreparedSet, query, fixedPreparedSet);
    auto fixedMaterial = pmacc::spearhed::makeCandidateProvider(fixedPreparedSet, query, materialPreparedSet);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(materialFixed)>);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(fixedMaterial)>);
    REQUIRE(offsets(materialFixed) == std::vector<unsigned int>{0u, 2u});
    REQUIRE(offsets(fixedMaterial) == std::vector<unsigned int>{0u, 1u, 2u});

    Grid const periodicGrid{bounds, {2u, 1u, 1u}, {true, false, false}};
    auto periodicFixed = PreparedGeometry<
        pmacc::spearhed::FixedCartesianMappingTag,
        pmacc::spearhed::FixedCartesianBroadPhaseView<CS>,
        DummyStore>{&fixedStore, {periodicGrid}};
    REQUIRE_THROWS_AS(
        pmacc::spearhed::makeCandidateProvider(materialPreparedSet, query, periodicFixed),
        std::invalid_argument);
}

TEST_CASE_METHOD(Fixture, "adaptive pairs use the CSR fallback", "[spatial][pair-provider]")
{
    prBuf->create(1u);
    Metadata metadata;
    metadata.chart.origin = {3.0f, 0.0f, 0.0f};
    metadata.occupancy = {{3.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}};
    prBuf->pushBack(spearhed::PRType{deviceHeap->getAllocatorHandle(), metadata});
    prBuf->buffer->hostToDevice();

    auto material = materialPrepared(*prBuf);
    auto adaptive = adaptivePrepared(*prBuf);
    Grid const grid{{{2.0f, -1.0f, -1.0f}, {4.0f, 1.0f, 1.0f}}, {2u, 1u, 1u}};
    DummyStore fixedStore;
    auto fixed = PreparedGeometry<
        pmacc::spearhed::FixedCartesianMappingTag,
        pmacc::spearhed::FixedCartesianBroadPhaseView<CS>,
        DummyStore>{&fixedStore, {grid}};
    auto const query = pmacc::spearhed::InteractionQuery{0.0f};

    auto adaptiveMaterial = pmacc::spearhed::makeCandidateProvider(adaptive, query, material);
    auto materialAdaptive = pmacc::spearhed::makeCandidateProvider(material, query, adaptive);
    auto adaptiveFixed = pmacc::spearhed::makeCandidateProvider(adaptive, query, fixed);
    auto fixedAdaptive = pmacc::spearhed::makeCandidateProvider(fixed, query, adaptive);
    auto adaptiveAdaptive = pmacc::spearhed::makeCandidateProvider(adaptive, query, adaptive);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(adaptiveMaterial)>);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(materialAdaptive)>);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(adaptiveFixed)>);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(fixedAdaptive)>);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(adaptiveAdaptive)>);
    REQUIRE(offsets(adaptiveMaterial) == std::vector<unsigned int>{0u, 1u});
    REQUIRE(offsets(materialAdaptive) == std::vector<unsigned int>{0u, 1u});
    REQUIRE(offsets(adaptiveFixed) == std::vector<unsigned int>{0u, 2u});
    REQUIRE(offsets(fixedAdaptive) == std::vector<unsigned int>{0u, 1u, 2u});
    REQUIRE(offsets(adaptiveAdaptive) == std::vector<unsigned int>{0u, 1u});
}

TEST_CASE_METHOD(Fixture, "fixed pairs use the implicit provider", "[spatial][pair-provider]")
{
    Grid::Bounds const bounds{{2.0f, -1.0f, -1.0f}, {4.0f, 1.0f, 1.0f}};
    Grid const targetGrid{bounds, {2u, 1u, 1u}};
    Grid const sourceGrid{bounds, {1u, 1u, 1u}};
    Buffer targetStore;
    Buffer sourceStore;
    pmacc::spearhed::FixedCartesianDecomposition target{targetGrid, targetStore};
    pmacc::spearhed::FixedCartesianDecomposition source{sourceGrid, sourceStore};
    target.prepareAfterMotion();
    source.prepareAfterMotion();

    auto provider = pmacc::spearhed::makeCandidateProvider(
        target.preparedFor(targetStore),
        pmacc::spearhed::InteractionQuery{0.0f},
        source.preparedFor(sourceStore));
    STATIC_REQUIRE_FALSE(HasMaterializedCandidateArrays<decltype(provider)>);
    auto const view = provider.deviceView();
    for(uint32_t targetSlot = 0u; targetSlot < targetGrid.bucketCount(); ++targetSlot)
    {
        std::vector<uint32_t> candidates;
        view.forEachCandidate(targetSlot, [&](auto candidate) { candidates.push_back(candidate.sourceBucketSlot); });
        REQUIRE(candidates == std::vector<uint32_t>{0u});
    }
}

TEST_CASE_METHOD(Fixture, "empty mixed source has no candidates", "[spatial][pair-provider]")
{
    auto emptyStore = std::make_shared<Buffer>();
    emptyStore->create(0u);
    auto emptyPrepared = materialPrepared(*emptyStore);

    Grid::Bounds const bounds{{2.0f, -1.0f, -1.0f}, {4.0f, 1.0f, 1.0f}};
    Grid const grid{bounds, {2u, 1u, 1u}};
    DummyStore fixedStore;
    auto fixedPrepared = PreparedGeometry<
        pmacc::spearhed::FixedCartesianMappingTag,
        pmacc::spearhed::FixedCartesianBroadPhaseView<CS>,
        DummyStore>{&fixedStore, {grid}};
    auto provider = pmacc::spearhed::makeCandidateProvider(
        fixedPrepared,
        pmacc::spearhed::InteractionQuery{0.0f},
        emptyPrepared);
    STATIC_REQUIRE(HasMaterializedCandidateArrays<decltype(provider)>);
    REQUIRE(offsets(provider) == std::vector<unsigned int>{0u, 0u, 0u});
}
