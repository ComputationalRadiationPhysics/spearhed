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
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/algorithms/InteractParticles.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Positioning.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

//! Default (no stage(), no prepare()) functor: counts every non-self pair.
struct InteractionCountFunc
{
    static constexpr auto neighbourReads = ll::makeSet();
    static constexpr auto ownReads = ll::makeSet();
    static constexpr auto ownAccumulate = ll::makeSet();

    HDINLINE constexpr void operator()(
        auto& worker,
        auto const& /*ownRead*/,
        auto const& /*nb*/,
        auto const& ctx,
        auto& /*acc*/,
        auto count_db) const
    {
        if(ctx.isSelf) [[unlikely]]
            return;

        alpaka::atomicAdd(worker.getAcc(), &count_db(0), static_cast<uint64_t>(1), ::alpaka::hierarchy::Blocks{});
    }
};

namespace stage_test
{
    //! Brand-new derived-quantity tag, declared next to the functor that stages it.
    DEFINE_TAG(derivedId);
} // namespace stage_test

/**
 * Functor exercising the stage() hook: at staging time each live neighbour's global id is
 * transformed into a derived field (derivedId = id_j + 1) held in the functor's own StagedRecord.
 * The pairwise sweep then reads only the derived field and sums it over all non-self pairs, so a
 * correct total proves stage() ran once per staged neighbour, the derived SMEM record was used as
 * the neighbour cache, self-pairs were excluded, and dead slots were skipped (their garbage cache
 * is never read).
 */
struct StageDerivedFunc
{
    // Read contract: the global attributes stage() may read.
    static constexpr auto neighbourReads = ll::makeSet(spearhed::particleId);
    static constexpr auto ownReads = ll::makeSet();
    static constexpr auto ownAccumulate = ll::makeSet();

    //! Derived neighbour record staged once per neighbour.
    using StagedRecord = ll::Record<ll::Field<stage_test::derivedId_t, uint64_t>>;

    HDINLINE void stage(auto const& nParticle, auto staged) const
    {
        staged[stage_test::derivedId] = nParticle[spearhed::particleId] + uint64_t{1};
    }

    HDINLINE void operator()(
        auto& worker,
        auto const& /*ownRead*/,
        auto const& nb,
        auto const& ctx,
        auto& /*acc*/,
        auto sum_db) const
    {
        if(ctx.isSelf) [[unlikely]]
            return;

        alpaka::atomicAdd(
            worker.getAcc(),
            &sum_db(0),
            static_cast<uint64_t>(nb[stage_test::derivedId]),
            ::alpaka::hierarchy::Blocks{});
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

/**
 * Helper: build an all-to-all NeighbourEntry for @p prBuf with @p numRegions regions.
 * Every target region neighbours every source region, useful for analytic validation.
 */
auto makeAllToAllEntry(auto* prBuf, int numRegions)
{
    using PRBufType = std::remove_pointer_t<decltype(prBuf)>;
    int const totalNeighbours = numRegions * numRegions;

    pmacc::HostDeviceBuffer<unsigned int, 1> neighbourRegions(totalNeighbours);
    auto h_neighbours = neighbourRegions.getHostBuffer().data();
    pmacc::HostDeviceBuffer<unsigned int, 1> regionOffsets(numRegions + 1);
    auto h_offsets = regionOffsets.getHostBuffer().data();

    for(int i = 0; i < numRegions; ++i)
    {
        h_offsets[i] = static_cast<unsigned int>(i * numRegions);
        for(int j = 0; j < numRegions; ++j)
            h_neighbours[i * numRegions + j] = static_cast<unsigned int>(j);
    }
    h_offsets[numRegions] = static_cast<unsigned int>(totalNeighbours);
    neighbourRegions.hostToDevice();
    regionOffsets.hostToDevice();

    return pmacc::spearhed::NeighbourEntry<PRBufType>{prBuf, std::move(neighbourRegions), std::move(regionOffsets)};
}

TEST_CASE_METHOD(ParticleFixture, "InteractParticles validation", "[integration][particles][interaction]")
{
    SECTION("InteractParticles Validation")
    {
        constexpr int numRegions = 2;

        auto setup = spearhed::EmptyNRegions<numRegions>{};
        spearhed::InitRegions{}(*deviceHeap, setup);
        spearhed::InitParticles{}(setup);

        // Setup Neighbour Graph (All-to-All mapping for analytic validation)
        int const totalNeighbours = numRegions * numRegions;
        pmacc::HostDeviceBuffer<unsigned int, 1> neighbourRegions(totalNeighbours);
        auto h_neighbours = neighbourRegions.getHostBuffer().data();

        // Offsets point to the start of each region's neighbour list in h_neighbours.
        pmacc::HostDeviceBuffer<unsigned int, 1> regionOffsets(numRegions + 1);
        auto h_offsets = regionOffsets.getHostBuffer().data();

        for(int i = 0; i < numRegions; ++i)
        {
            h_offsets[i] = static_cast<unsigned int>(i * numRegions);
            for(int j = 0; j < numRegions; ++j)
            {
                h_neighbours[i * numRegions + j] = static_cast<unsigned int>(j);
            }
        }
        // Set the final boundary offset
        h_offsets[numRegions] = static_cast<unsigned int>(totalNeighbours);

        neighbourRegions.hostToDevice();
        regionOffsets.hostToDevice();

        using T_Count = uint64_t;
        pmacc::HostDeviceBuffer<T_Count, 1> countBuffer(1u);
        countBuffer.getHostBuffer().setValue(0);
        countBuffer.hostToDevice();

        auto d_count = countBuffer.getDeviceBuffer().getDataBox();

        // Use an excessively large interaction radius so the distance check always passes
        constexpr double interactionRadius = 1e9;

        using PRBufType = pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>;
        auto bundle = pmacc::spearhed::makeNeighbourBundle(
            pmacc::spearhed::NeighbourEntry<PRBufType>{
                prBuf.get(),
                std::move(neighbourRegions),
                std::move(regionOffsets)});

        auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
        pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> index{*prBuf};
        pmacc::spearhed::interact(sources, *prBuf, index, interactionRadius, InteractionCountFunc{}, d_count)
            .waitForFinished();

        countBuffer.deviceToHost();
        T_Count const h_count = countBuffer.getHostBuffer().data()[0];

        // Calculate expected interactions analytically based on InitParticles logic
        uint64_t totalParticles = 0;
        for(uint64_t i = 0; i < numRegions; ++i)
        {
            totalParticles += setup.baseNumParticlesToCreate * (i + 1);
        }

        // Expected valid interactions: all particles interact with all other particles exactly once.
        // Self-interactions are excluded by the is_self flag. P(N, 2) = N * (N - 1)
        uint64_t const expectedInteractions = totalParticles * (totalParticles - 1);

        INFO("Total Particles: " << totalParticles);
        INFO("Actual Interactions (Device): " << h_count);
        INFO("Expected Interactions: " << expectedInteractions);

        REQUIRE(h_count == expectedInteractions);
    }


    SECTION("InteractParticles stage() hook")
    {
        constexpr int numRegions = 2;

        auto setup = spearhed::EmptyNRegions<numRegions>{};
        spearhed::InitRegions{}(*deviceHeap, setup);
        spearhed::InitParticles{}(setup);

        // All-to-all neighbour graph, identical to the counting test.
        int const totalNeighbours = numRegions * numRegions;
        pmacc::HostDeviceBuffer<unsigned int, 1> neighbourRegions(totalNeighbours);
        auto h_neighbours = neighbourRegions.getHostBuffer().data();
        pmacc::HostDeviceBuffer<unsigned int, 1> regionOffsets(numRegions + 1);
        auto h_offsets = regionOffsets.getHostBuffer().data();
        for(int i = 0; i < numRegions; ++i)
        {
            h_offsets[i] = static_cast<unsigned int>(i * numRegions);
            for(int j = 0; j < numRegions; ++j)
                h_neighbours[i * numRegions + j] = static_cast<unsigned int>(j);
        }
        h_offsets[numRegions] = static_cast<unsigned int>(totalNeighbours);
        neighbourRegions.hostToDevice();
        regionOffsets.hostToDevice();

        using T_Sum = uint64_t;
        pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);
        sumBuffer.getHostBuffer().setValue(0);
        sumBuffer.hostToDevice();
        auto d_sum = sumBuffer.getDeviceBuffer().getDataBox();

        constexpr double interactionRadius = 1e9;

        using PRBufType = pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>;
        auto bundle = pmacc::spearhed::makeNeighbourBundle(
            pmacc::spearhed::NeighbourEntry<PRBufType>{
                prBuf.get(),
                std::move(neighbourRegions),
                std::move(regionOffsets)});

        auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
        pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> index{*prBuf};
        pmacc::spearhed::interact(sources, *prBuf, index, interactionRadius, StageDerivedFunc{}, d_sum)
            .waitForFinished();

        sumBuffer.deviceToHost();
        T_Sum const h_sum = sumBuffer.getHostBuffer().data()[0];

        // InitParticles assigns particle ids 0..N-1 across all regions (see HierarchyForEach). Each own
        // particle interacts once with every other particle, so with derivedId_j = id_j + 1 the total is
        //   sum_{i != j} (id_j + 1) = (N - 1) * (sum_j id_j + N),  sum_j id_j = N(N-1)/2.
        uint64_t totalParticles = 0;
        for(uint64_t i = 0; i < numRegions; ++i)
            totalParticles += setup.baseNumParticlesToCreate * (i + 1);

        uint64_t const sumIds = totalParticles * (totalParticles - 1) / 2;
        uint64_t const expectedSum = (totalParticles - 1) * (sumIds + totalParticles);

        INFO("Total Particles: " << totalParticles);
        INFO("Actual staged sum (Device): " << h_sum);
        INFO("Expected staged sum: " << expectedSum);

        REQUIRE(h_sum == expectedSum);
    }


    /**
     * Multi-entry bundle -> exercises UnifiedFrameInteractionKernel (size >= 2 branch).
     * Two identical all-to-all entries double the per-pair contributions, proving the
     * unified kernel correctly folds multiple sources in a single launch.
     */
    SECTION("InteractParticles multi-entry bundle (unified path)")
    {
        constexpr int numRegions = 2;

        auto setup = spearhed::EmptyNRegions<numRegions>{};
        spearhed::InitRegions{}(*deviceHeap, setup);
        spearhed::InitParticles{}(setup);

        // Two independent entries, both all-to-all.
        auto entryA = makeAllToAllEntry(prBuf.get(), numRegions);
        auto entryB = makeAllToAllEntry(prBuf.get(), numRegions);

        auto bundle = pmacc::spearhed::makeNeighbourBundle(std::move(entryA), std::move(entryB));
        // Verify bundle arity triggers the unified path.
        STATIC_REQUIRE(decltype(bundle)::size() == 2);

        using T_Count = uint64_t;
        pmacc::HostDeviceBuffer<T_Count, 1> countBuffer(1u);
        countBuffer.getHostBuffer().setValue(0);
        countBuffer.hostToDevice();
        auto d_count = countBuffer.getDeviceBuffer().getDataBox();

        constexpr double interactionRadius = 1e9;

        auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
        pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> index{*prBuf};
        pmacc::spearhed::interact(sources, *prBuf, index, interactionRadius, InteractionCountFunc{}, d_count)
            .waitForFinished();

        countBuffer.deviceToHost();
        T_Count const h_count = countBuffer.getHostBuffer().data()[0];

        uint64_t totalParticles = 0;
        for(uint64_t i = 0; i < numRegions; ++i)
            totalParticles += setup.baseNumParticlesToCreate * (i + 1);

        // Each non-self pair is counted twice (once per source entry).
        uint64_t const singleEntryExpected = totalParticles * (totalParticles - 1);
        uint64_t const expectedInteractions = 2 * singleEntryExpected;

        INFO("Total Particles: " << totalParticles);
        INFO("Actual Interactions (Device): " << h_count);
        INFO("Expected Interactions (2x single-entry): " << expectedInteractions);

        REQUIRE(h_count == expectedInteractions);
    }


    /**
     * Explicit perSource policy on a multi-entry bundle -> forces FrameInteractionKernel
     * launches instead of the unified path. The result is the same (2x single-entry count),
     * but the dispatch path is different.
     */
    SECTION("InteractParticles explicit perSource policy")
    {
        constexpr int numRegions = 2;

        auto setup = spearhed::EmptyNRegions<numRegions>{};
        spearhed::InitRegions{}(*deviceHeap, setup);
        spearhed::InitParticles{}(setup);

        auto entryA = makeAllToAllEntry(prBuf.get(), numRegions);
        auto entryB = makeAllToAllEntry(prBuf.get(), numRegions);

        auto bundle = pmacc::spearhed::makeNeighbourBundle(std::move(entryA), std::move(entryB));

        using T_Count = uint64_t;
        pmacc::HostDeviceBuffer<T_Count, 1> countBuffer(1u);
        countBuffer.getHostBuffer().setValue(0);
        countBuffer.hostToDevice();
        auto d_count = countBuffer.getDeviceBuffer().getDataBox();

        constexpr double interactionRadius = 1e9;

        auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
        pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> index{*prBuf};

        // Force per-source launches even though bundle size == 2.
        pmacc::spearhed::interact(
            pmacc::spearhed::perSource,
            sources,
            *prBuf,
            index,
            interactionRadius,
            InteractionCountFunc{},
            d_count)
            .waitForFinished();

        countBuffer.deviceToHost();
        T_Count const h_count = countBuffer.getHostBuffer().data()[0];

        uint64_t totalParticles = 0;
        for(uint64_t i = 0; i < numRegions; ++i)
            totalParticles += setup.baseNumParticlesToCreate * (i + 1);

        uint64_t const expectedInteractions = 2 * totalParticles * (totalParticles - 1);

        INFO("Total Particles: " << totalParticles);
        INFO("Actual Interactions (Device): " << h_count);
        INFO("Expected Interactions: " << expectedInteractions);

        REQUIRE(h_count == expectedInteractions);
    }


    /**
     * Explicit unified policy on a single-entry bundle -> forces UnifiedFrameInteractionKernel
     * even though automatic dispatch would pick FrameInteractionKernel. The result must match
     * the automatic single-entry path.
     */
    SECTION("InteractParticles explicit unified policy on single entry")
    {
        constexpr int numRegions = 2;

        auto setup = spearhed::EmptyNRegions<numRegions>{};
        spearhed::InitRegions{}(*deviceHeap, setup);
        spearhed::InitParticles{}(setup);

        auto entry = makeAllToAllEntry(prBuf.get(), numRegions);

        auto bundle = pmacc::spearhed::makeNeighbourBundle(std::move(entry));

        using T_Count = uint64_t;
        pmacc::HostDeviceBuffer<T_Count, 1> countBuffer(1u);
        countBuffer.getHostBuffer().setValue(0);
        countBuffer.hostToDevice();
        auto d_count = countBuffer.getDeviceBuffer().getDataBox();

        constexpr double interactionRadius = 1e9;

        auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
        pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> index{*prBuf};

        // Force unified kernel even though bundle size == 1.
        pmacc::spearhed::interact(
            pmacc::spearhed::unified,
            sources,
            *prBuf,
            index,
            interactionRadius,
            InteractionCountFunc{},
            d_count)
            .waitForFinished();

        countBuffer.deviceToHost();
        T_Count const h_count = countBuffer.getHostBuffer().data()[0];

        uint64_t totalParticles = 0;
        for(uint64_t i = 0; i < numRegions; ++i)
            totalParticles += setup.baseNumParticlesToCreate * (i + 1);

        uint64_t const expectedInteractions = totalParticles * (totalParticles - 1);

        INFO("Total Particles: " << totalParticles);
        INFO("Actual Interactions (Device): " << h_count);
        INFO("Expected Interactions: " << expectedInteractions);

        REQUIRE(h_count == expectedInteractions);
    }
}
