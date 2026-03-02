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

#include "spmacc/particles/algorithms/ForEachParticle.hpp"

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/dimension.param"
#include "spearhed/param/mallocMC.param"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spmacc/ParticleRegion.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Positioning.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
constexpr auto testHeapSize = 256ull * 1024 * 1024;

// Functor: Atomically add particle IDs to the sum
struct SumFunc
{
    constexpr void operator()(auto& worker, auto& particle, auto sum_db) const
    {
        alpaka::atomicAdd(worker.getAcc(), &sum_db(0), *particle[spearhed::particleId], ::alpaka::hierarchy::Blocks{});
    }
};

TEST_CASE("ForEachParticleInPRBuf Validation", "[integration][particles][foreach]")
{
    // Environment & Allocator Setup
    pmacc::test::PMaccFixture<TEST_DIM> fixture;
    auto& env = pmacc::Environment<TEST_DIM>::get();

    uint64_t maxRanks = env.GridController().getGpuNodes().productOfComponents();
    uint64_t rank = env.GridController().getScalarPosition();

    auto& dc = pmacc::Environment<>::get().DataConnector();
    auto idProvider = std::make_shared<pmacc::IdProvider>("globalId", rank, maxRanks);
    dc.share(idProvider);

    std::shared_ptr<spearhed::DeviceHeap> deviceHeap;

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
    auto& deviceManager = pmacc::manager::Device<pmacc::ComputeDevice>::get();
    auto alpakaDevice = deviceManager.current();
    auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(pmacc::ITask::TASK_DEVICE)->getAlpakaQueue();

    // Create and resize heap
    deviceHeap = std::make_shared<DeviceHeap>(alpakaDevice, alpakaQueue, 0u);
    alpaka::wait(alpakaQueue);

    deviceHeap->destructiveResize(alpakaDevice, alpakaQueue, testHeapSize);
    alpaka::wait(alpakaQueue);

    auto mallocMCBuffer = std::make_unique<pmacc::MallocMCBuffer<DeviceHeap>>(deviceHeap);
    dc.consume(std::move(mallocMCBuffer));
#endif

    dc.get<pmacc::IdProvider>("globalId")->reset();

    //  Particle Region Setup
    using PRType = spearhed::PRType;
    auto prBuf = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<PRType>>();
    dc.share(prBuf);

    prBuf->create(2);
    PRType boundedParticles{deviceHeap->getAllocatorHandle()};

    // Push two regions
    prBuf->pushBack(boundedParticles);
    prBuf->pushBack(boundedParticles);
    prBuf->buffer->hostToDevice();

    // Initialize particles
    spearhed::InitParticles{}();


    // Execute ForEachParticleInPRBuf Test
    // Allocate memory for reduction sum

    using T_Sum = uint64_t;
    pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);

    // Initialize on host and sync to device
    sumBuffer.getHostBuffer().setValue(0);
    sumBuffer.hostToDevice();

    auto d_sum = sumBuffer.getDeviceBuffer().getDataBox();

    // Execute the utility
    pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, SumFunc{}, d_sum);

    // Sync result back to host
    sumBuffer.deviceToHost();
    T_Sum h_sum = sumBuffer.getHostBuffer().data()[0];


    //  Validation
    uint64_t totalParticles = 0;
    constexpr uint64_t numRegions = 2;

    // Calculate expected sum analytically based on InitParticles logic
    for(uint64_t i = 0; i < numRegions; ++i)
    {
        totalParticles += spearhed::init::detail::baseNumParticlesToCreate * (i + 1);
    }

    // Sum of arithmetic progression: n*(n-1)/2 because IDs start at 0
    uint64_t const expectedSum = (totalParticles * (totalParticles - 1)) / 2;

    INFO("Total Particles: " << totalParticles);
    INFO("Actual Sum (Device): " << h_sum);
    INFO("Expected Sum: " << expectedSum);

    REQUIRE(h_sum == expectedSum);

    env.finalize();
}
