#include "ValidateIdSum.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/dimension.param"
#include "spearhed/param/mallocMC.param"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spmacc/ParticleRegion.hpp"

#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
static pmacc::test::PMaccFixture<TEST_DIM> fixture;

constexpr auto testHeapSize = 256ull * 1024 * 1024; // 256 MiB for testing (vs 2GB in production)

TEST_CASE("Particle Creation and ID Sum Validation", "[integration][particles]")
{
    uint64_t maxRanks = pmacc::Environment<TEST_DIM>::get().GridController().getGpuNodes().productOfComponents();
    uint64_t rank = pmacc::Environment<TEST_DIM>::get().GridController().getScalarPosition();

    auto& dc = pmacc::Environment<>::get().DataConnector();
    auto idProvider = std::make_shared<pmacc::IdProvider>("globalId", rank, maxRanks);
    dc.share(idProvider);

    // Setup Environment
    std::shared_ptr<spearhed::DeviceHeap> deviceHeap;

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
    auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(pmacc::ITask::TASK_DEVICE)->getAlpakaQueue();
    auto alpakaDevice = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();

    // Create initial empty allocator
    deviceHeap = std::make_shared<DeviceHeap>(alpakaDevice, alpakaQueue, 0u);
    alpaka::wait(alpakaQueue);

    // We assume sufficient memory is availabe
    deviceHeap->destructiveResize(alpakaDevice, alpakaQueue, TestHeapSize);
    alpaka::wait(alpakaQueue);

    auto mallocMCBuffer = std::make_unique<pmacc::MallocMCBuffer<DeviceHeap>>(deviceHeap);
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
    pmacc::Environment<TEST_DIM>::get().finalize();
}
