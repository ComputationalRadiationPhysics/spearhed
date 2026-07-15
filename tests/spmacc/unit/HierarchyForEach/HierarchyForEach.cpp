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

#include "spmacc/particles/algorithms/HierarchyForEach.hpp"

#include "TestSetup.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/eventSystem/waitForAllTasks.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

namespace sp = pmacc::spearhed;

using T_Sum = uint64_t;

// Device kernel: sum particle IDs by walking a whole view (species or multi-species bundle) with
// the composable forEach. One block; parents (species, regions, frames) are walked sequentially by
// the iterator, then the frame's slots are distributed explicitly with lockstepForEachParticle --
// the canonical iterator + slot-combinator composition.
struct HierarchySumKernel
{
    DINLINE void operator()(auto const& worker, auto view, auto sumBox) const
    {
        sp::forEach(
            sp::levels::frame,
            sp::deviceHeap,
            view,
            [&](auto frame)
            {
                sp::lockstepForEachParticle(
                    worker,
                    frame,
                    [&](auto particle)
                    {
                        alpaka::atomicAdd(
                            worker.getAcc(),
                            &sumBox(0),
                            static_cast<T_Sum>(particle[spearhed::particleId]),
                            ::alpaka::hierarchy::Blocks{});
                    });
            });
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(
    ParticleFixture,
    "Hierarchy forEach spans all levels on host and device",
    "[integration][particles][hierarchy]")
{
    constexpr uint64_t numRegions = 3;
    auto setup = spearhed::EmptyNRegions<numRegions>{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    // Expected sum of IDs 0..N-1 (InitParticles assigns region r baseNumParticlesToCreate*(r+1) particles).
    uint64_t totalParticles = 0;
    for(uint64_t r = 0; r < numRegions; ++r)
        totalParticles += setup.baseNumParticlesToCreate * (r + 1);
    uint64_t const expectedSum = (totalParticles * (totalParticles - 1)) / 2;

    // Host-side metadata for frame/region cross-checks.
    prBuf->synchronize();
    auto const heapOffset = spearhed::syncHeapToHost();
    auto hostBox = prBuf->buffer->getHostBuffer().getDataBox();
    uint64_t expectedFrames = 0;
    for(int r = 0; r < prBuf->size; ++r)
        expectedFrames += hostBox[r].particleFrameList.numFrames();

    SECTION("device: forEachParticle over a species (lockstep leaf)")
    {
        pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);
        sumBuffer.getHostBuffer().setValue(0);
        sumBuffer.hostToDevice();

        constexpr uint32_t blockThreads = spearhed::FrameType::frameSize;
        PMACC_LOCKSTEP_KERNEL(HierarchySumKernel{})
            .config<blockThreads>(
                pmacc::DataSpace<DIM1>(1))(sp::deviceSpecies(*prBuf), sumBuffer.getDeviceBuffer().getDataBox());
        pmacc::eventSystem::waitForAllTasks();

        sumBuffer.deviceToHost();
        REQUIRE(sumBuffer.getHostBuffer().data()[0] == expectedSum);
    }

    SECTION("host: particles over a species (serial leaf) matches device")
    {
        T_Sum sum = 0;
        sp::forEach(
            sp::levels::particle,
            sp::hostHeap(heapOffset),
            sp::hostSpecies(*prBuf),
            [&](auto particle) { sum += static_cast<T_Sum>(particle[spearhed::particleId]); });
        REQUIRE(sum == expectedSum);
    }

    SECTION("host: compose regions-in-species then particles-in-region")
    {
        // "Iterate all regions in a species, and for each iterate its particles."
        // One heap-access value threads through both nesting levels.
        auto const access = sp::hostHeap(heapOffset);
        T_Sum total = 0;
        int seenRegions = 0;
        sp::forEach(
            sp::levels::region,
            access,
            sp::hostSpecies(*prBuf),
            [&](auto region)
            {
                ++seenRegions;
                sp::forEach(
                    sp::levels::particle,
                    access,
                    region,
                    [&](auto particle) { total += static_cast<T_Sum>(particle[spearhed::particleId]); });
            });
        REQUIRE(seenRegions == static_cast<int>(numRegions));
        REQUIRE(total == expectedSum);
    }

    SECTION("host: frames over a species visits every frame")
    {
        uint64_t frames = 0;
        sp::forEach(sp::levels::frame, sp::hostHeap(heapOffset), sp::hostSpecies(*prBuf), [&](auto) { ++frames; });
        REQUIRE(frames == expectedFrames);
    }

    SECTION("device: launchForEach over particles (scan-driven decomposition) matches single block")
    {
        pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);
        sumBuffer.getHostBuffer().setValue(0);
        sumBuffer.hostToDevice();

        // The accumulator box is forwarded as a kernel argument (not captured): a captured box would
        // be const inside the const kernel body, so &box(0) could not feed atomicAdd's T*.
        sp::launchForEach(
            sp::levels::particle,
            *prBuf,
            [](auto const& worker, auto particle, auto sumBox) constexpr
            {
                alpaka::atomicAdd(
                    worker.getAcc(),
                    &sumBox(0),
                    static_cast<T_Sum>(particle[spearhed::particleId]),
                    ::alpaka::hierarchy::Blocks{});
            },
            sumBuffer.getDeviceBuffer().getDataBox());
        pmacc::eventSystem::waitForAllTasks();

        sumBuffer.deviceToHost();
        REQUIRE(sumBuffer.getHostBuffer().data()[0] == expectedSum);
    }

    SECTION("device: launchForEach over frames (scan-driven) visits every frame")
    {
        pmacc::HostDeviceBuffer<T_Sum, 1> frameCount(1u);
        frameCount.getHostBuffer().setValue(0);
        frameCount.hostToDevice();

        // Contiguous schedule over a bounded grid: same result, different decomposition policy.
        sp::launchForEach(
            sp::forEachConfig(sp::contiguous, sp::fixedGrid<8>),
            sp::levels::frame,
            *prBuf,
            [](auto const& worker, auto /*frameView*/, auto countBox) constexpr
            {
                pmacc::lockstep::makeMaster(worker)(
                    [&]()
                    { alpaka::atomicAdd(worker.getAcc(), &countBox(0), T_Sum{1}, ::alpaka::hierarchy::Blocks{}); });
            },
            frameCount.getDeviceBuffer().getDataBox());
        pmacc::eventSystem::waitForAllTasks();

        frameCount.deviceToHost();
        REQUIRE(frameCount.getHostBuffer().data()[0] == expectedFrames);
    }

    SECTION("host: particles over a multi-species bundle (compile-time species fold)")
    {
        T_Sum sum = 0;
        sp::forEach(
            sp::levels::particle,
            sp::hostHeap(heapOffset),
            sp::hostMultiSpecies(*prBuf),
            [&](auto particle) { sum += static_cast<T_Sum>(particle[spearhed::particleId]); });
        REQUIRE(sum == expectedSum);
    }

    SECTION("device: forEach over a multi-species bundle (device-capable species fold)")
    {
        // Same generic kernel body as the single-species device section: the view argument is
        // auto, so passing a MultiSpeciesView exercises the pmacc-tuple fold end-to-end on device.
        pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);
        sumBuffer.getHostBuffer().setValue(0);
        sumBuffer.hostToDevice();

        constexpr uint32_t blockThreads = spearhed::FrameType::frameSize;
        PMACC_LOCKSTEP_KERNEL(HierarchySumKernel{})
            .config<blockThreads>(
                pmacc::DataSpace<DIM1>(1))(sp::deviceMultiSpecies(*prBuf), sumBuffer.getDeviceBuffer().getDataBox());
        pmacc::eventSystem::waitForAllTasks();

        sumBuffer.deviceToHost();
        REQUIRE(sumBuffer.getHostBuffer().data()[0] == expectedSum);
    }
}

// Functor: atomically add particle IDs to the sum.
struct SumFunc
{
    HDINLINE constexpr void operator()(auto& worker, auto& particle, auto sumBox) const
    {
        alpaka::atomicAdd(worker.getAcc(), &sumBox(0), particle[spearhed::particleId], ::alpaka::hierarchy::Blocks{});
    }
};

TEST_CASE_METHOD(
    ParticleFixture,
    "launchForEach visits every particle once under every schedule x grid config",
    "[integration][particles][foreach]")
{
    constexpr uint64_t numRegions = 2;
    auto setup = spearhed::EmptyNRegions<numRegions>{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    // Runs the SumFunc reduction with a given value-based ForEachConfig and returns the sum. The
    // accumulator box is forwarded as a kernel argument, not captured: a captured box would be
    // const inside the const kernel body, so &box(0) could not feed atomicAdd's T*.
    auto runSum = [&](auto cfg) -> T_Sum
    {
        pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);
        sumBuffer.getHostBuffer().setValue(0);
        sumBuffer.hostToDevice();

        sp::launchForEach(cfg, sp::levels::particle, *prBuf, SumFunc{}, sumBuffer.getDeviceBuffer().getDataBox());

        sumBuffer.deviceToHost();
        return sumBuffer.getHostBuffer().data()[0];
    };

    // Calculate expected sum analytically based on InitParticles logic
    uint64_t totalParticles = 0;
    for(uint64_t i = 0; i < numRegions; ++i)
        totalParticles += setup.baseNumParticlesToCreate * (i + 1);

    // Sum of arithmetic progression: n*(n-1)/2 because IDs start at 0
    uint64_t const expectedSum = (totalParticles * (totalParticles - 1)) / 2;

    INFO("Total Particles: " << totalParticles);
    INFO("Expected Sum: " << expectedSum);

    // Every execution policy must visit each live particle exactly once, so all value-based config
    // combinations of frame schedule x grid sizing must reproduce the same reduction.
    SECTION("default config (grid-stride, one block per frame)")
    {
        REQUIRE(runSum(sp::defaultForEach) == expectedSum);
    }
    SECTION("one-to-one schedule")
    {
        REQUIRE(runSum(sp::forEachConfig(sp::oneToOne, sp::oneBlockPerFrame)) == expectedSum);
    }
    SECTION("grid-stride over a fixed grid")
    {
        REQUIRE(runSum(sp::forEachConfig(sp::gridStride, sp::fixedGrid<3>)) == expectedSum);
    }
    SECTION("contiguous chunks over a fixed grid")
    {
        REQUIRE(runSum(sp::forEachConfig(sp::contiguous, sp::fixedGrid<3>)) == expectedSum);
    }
    SECTION("contiguous chunks, one block per frame, custom thread count")
    {
        REQUIRE(runSum(sp::forEachConfig<64>(sp::contiguous, sp::oneBlockPerFrame)) == expectedSum);
    }
}
