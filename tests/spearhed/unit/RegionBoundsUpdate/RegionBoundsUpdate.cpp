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

#include "spmacc/RegionBoundsUpdate.hpp"

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/dimension.param"
#include "spearhed/param/mallocMC.param"
#include "spearhed/param/speciesTraits.param"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spmacc/ParticleRegion.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Point.hpp"
#include "spmacc/topology/PointStorage.hpp"

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <cstdio>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
constexpr auto testHeapSize = 256ull * 1024 * 1024;


// Define expected bounds
// We use a float value that can be exactly represented to avoid precision issues in comparison
using CS = pmacc::spearhed::Cartesian<float, TEST_DIM>;
using PosType = pmacc::spearhed::Point<CS, pmacc::spearhed::PointValueStorage<CS>>;
constexpr PosType expectedMin{0.125f, 0.125f, 0.125f};
constexpr PosType expectedMax{0.875f, 0.875f, 0.875f};
constexpr PosType defaultPos{0.5f, 0.5f, 0.5f};

// Functor to set specific particle positions:
// - ID 0 -> Min corner
// - ID 1 -> Max corner
// - Others -> Center
struct SetPosFunctor
{
    void operator()(auto& worker, auto& particle)
    {
        using namespace spearhed;

        // Reset all to center first
        particle[spearhed::pos].get() = defaultPos;
        // Set outliers to define the bounding box
        if(*particle[particleId] == 0)
        {
            particle[spearhed::pos].get() = expectedMin;
        }
        else if(*particle[particleId] == 1)
        {
            particle[spearhed::pos].get() = expectedMax;
        }
    }
};

TEST_CASE("UpdateRegionBounds Validation", "[integration][particles][bounds]")
{
    //  Environment Setup
    pmacc::test::PMaccFixture<TEST_DIM> fixture;
    auto& env = pmacc::Environment<TEST_DIM>::get();
    auto& dc = env.DataConnector();

    // Setup ID Provider
    uint64_t maxRanks = env.GridController().getGpuNodes().productOfComponents();
    uint64_t rank = env.GridController().getScalarPosition();
    auto idProvider = std::make_shared<pmacc::IdProvider>("globalId", rank, maxRanks);
    dc.share(idProvider);

    // Setup Device Heap
    std::shared_ptr<spearhed::DeviceHeap> deviceHeap;
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
    auto& deviceManager = pmacc::manager::Device<pmacc::ComputeDevice>::get();
    auto alpakaDevice = deviceManager.current();
    auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(pmacc::ITask::TASK_DEVICE)->getAlpakaQueue();

    deviceHeap = std::make_shared<DeviceHeap>(alpakaDevice, alpakaQueue, 0u);
    alpaka::wait(alpakaQueue);
    deviceHeap->destructiveResize(alpakaDevice, alpakaQueue, testHeapSize);
    alpaka::wait(alpakaQueue);

    auto mallocMCBuffer = std::make_unique<pmacc::MallocMCBuffer<DeviceHeap>>(deviceHeap);
    dc.consume(std::move(mallocMCBuffer));
#endif
    dc.get<pmacc::IdProvider>("globalId")->reset();


    // Create Region and Particles
    using PRType = spearhed::PRType;
    auto prBuf = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<PRType>>();
    dc.share(prBuf);

    prBuf->create(1); // Create 1 region
    PRType boundedParticles{deviceHeap->getAllocatorHandle()};

    // Initialize Region Metadata
    boundedParticles.volume.reset();

    prBuf->pushBack(boundedParticles);
    prBuf->buffer->hostToDevice();

    // Create default particles
    spearhed::InitParticles{}();


    //  Setup Test Scenario (Modify Positions)

    // Apply positions on device
    pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, SetPosFunctor{});


    //  Execute UpdateRegionBounds

    // Call the code under test
    pmacc::spearhed::UpdateVolumes<PRType>{}();


    // Validation

    // Sync region metadata back to host
    prBuf->buffer->deviceToHost();

    // Get the first (and only) region
    auto const& db = prBuf->buffer->getHostBuffer().getDataBox();
    auto const& region = db(0);

    INFO("Region Min: " << region.volume.min << ", Expected: " << expectedMin);
    INFO("Region Max: " << region.volume.max << ", Expected: " << expectedMax);

    // Verify bounds for each dimension
    for(unsigned d = 0; d < TEST_DIM; ++d)
    {
        REQUIRE(region.volume.min[d] == expectedMin[d]);
        REQUIRE(region.volume.max[d] == expectedMax[d]);
    }

    env.finalize();
}
