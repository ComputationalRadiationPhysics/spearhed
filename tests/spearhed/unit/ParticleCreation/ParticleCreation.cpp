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

#include "ValidateIdSum.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/dimension.param"
#include "spearhed/param/mallocMC.param"
#include "spearhed/particles/initialization/InitParticles.hpp"

#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

TEST_CASE("Particle Creation and ID Sum Validation", "[integration][particles]")
{
    pmacc::test::PMaccFixture<TEST_DIM> fixture;
    uint64_t maxRanks = pmacc::Environment<TEST_DIM>::get().GridController().getGpuNodes().productOfComponents();
    uint64_t rank = pmacc::Environment<TEST_DIM>::get().GridController().getScalarPosition();

    auto& dc = pmacc::Environment<>::get().DataConnector();
    auto idProvider = std::make_shared<pmacc::IdProvider>("globalId", rank, maxRanks);
    dc.share(idProvider);

    // Setup Environment
    std::shared_ptr<spearhed::DeviceHeap> deviceHeap;

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
    constexpr auto testHeapSize = 256ull * 1024 * 1024;
    auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(pmacc::ITask::TASK_DEVICE)->getAlpakaQueue();
    auto alpakaDevice = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();

    // Create initial empty allocator
    deviceHeap = std::make_shared<spearhed::DeviceHeap>(alpakaDevice, alpakaQueue, 0u);
    alpaka::wait(alpakaQueue);

    // We assume sufficient memory is availabe
    deviceHeap->destructiveResize(alpakaDevice, alpakaQueue, testHeapSize);
    alpaka::wait(alpakaQueue);

    auto mallocMCBuffer = std::make_unique<pmacc::MallocMCBuffer<spearhed::DeviceHeap>>(deviceHeap);
    dc.consume(std::move(mallocMCBuffer));
#endif

    dc.get<pmacc::IdProvider>("globalId")->reset();

    using PRType = spearhed::PRType;
    auto prBuf = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<PRType>>();
    dc.share(prBuf);

    prBuf->create(2);

    PRType boundedParticles{deviceHeap->getAllocatorHandle()};

    prBuf->pushBack(boundedParticles);
    prBuf->pushBack(boundedParticles);

    prBuf->buffer->hostToDevice();

    spearhed::InitParticles{}();

    uint64_t const actualSum = ComputeParticleIdSum{}();

    // Calculate expected sum analytically
    // Logic matches InitParticles::NumParticlesToCreate: Base * (index + 1)
    uint64_t totalParticles = 0;
    constexpr uint64_t numRegions = 2; // We pushed 2 regions

    for(uint64_t i = 0; i < numRegions; ++i)
    {
        totalParticles += spearhed::init::detail::baseNumParticlesToCreate * (i + 1);
    }

    uint64_t const expectedSum = (totalParticles * (totalParticles - 1)) / 2;

    INFO("Total Particles Created: " << totalParticles);
    REQUIRE(actualSum == expectedSum);
    dc.clean();
}
