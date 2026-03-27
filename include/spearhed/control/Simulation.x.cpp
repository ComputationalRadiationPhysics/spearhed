/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED, derived from PIConGPU.
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

#include "spearhed/control/Simulation.hpp"

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/param/setup.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/debug/PMaccVerbose.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>

#include <iostream>
#include <optional>
#include <sstream>

namespace spearhed
{
    Simulation::Simulation() = default;

    Simulation::~Simulation() = default;

    void Simulation::pluginRegisterHelp(pmacc::po::options_description& desc)
    {
        BaseType::pluginRegisterHelp(desc);
        // clang-format off
        desc.add_options()(
            "versionOnce", pmacc::po::value<bool>(&showVersionOnce)->zero_tokens(),
            "print version information once and start")
            ("no-start-simulation", pmacc::po::bool_switch(&skipSimulation)->default_value(false), "Do not actually run the simulation but initialise everything, skip simulation and finalise.")
            ("devices,d", pmacc::po::value<std::vector<uint32_t>>(&devices)->multitoken(),
             "number of devices in each dimension")
            ("numRanksPerDevice,r", pmacc::po::value<uint32_t>(&numRanksPerDevice)->default_value(1u),
             "set the number of MPI ranks using a single device together");
        // clang-format on
    }

    std::string Simulation::pluginGetName() const
    {
        return "SPH";
    }

    void Simulation::pluginLoad()
    {
        // fill periodic with 0
        while(periodic.size() < 3)
            periodic.push_back(0);


        PMACC_VERIFY_MSG(
            devices.size() >= 2 && devices.size() <= 3,
            "Invalid number of devices.\nuse [-d dx=1 dy=1 dz=1]");

        // check on correct number of devices. fill with default value 1 for missing dimensions
        while(devices.size() < 3)
            devices.push_back(1);

        // check for request of > 1 device in z for a 2d simulation, this is probably a user's mistake
        if((simDim == 2) && (devices[2] > 1))
            std::cerr << "Warning: " << devices[2] << " devices requested for z in a 2d simulation, this parameter "
                      << "will be reset to 1. Number of MPI ranks must be equal to the number of devices in x * y\n";


        pmacc::DataSpace<simDim> gpus;
        pmacc::DataSpace<simDim> isPeriodic;

        for(uint32_t i = 0; i < simDim; ++i)
        {
            gpus[i] = devices[i];
            isPeriodic[i] = periodic[i];
        }

        pmacc::Environment<simDim>::get().initDevices(gpus, isPeriodic);
        pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();

        if(gc.getGlobalRank() == 0)
        {
            if(showVersionOnce)
            {
                std::cout << "Alpha development version of SPMacc" << std::endl;
            }
        }

        BaseType::pluginLoad();
    }

    void Simulation::pluginUnload()
    {
        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

        BaseType::pluginUnload();

        /** unshare all registered ISimulationData sets
         *
         * @todo can be removed as soon as our Environment learns to shutdown in
         *       a distinct order, e.g. DataConnector before CUDA context
         */
        dc.clean();
    }

    void Simulation::notify(uint32_t)
    {
    }

    void Simulation::startSimulation()
    {
        if(!skipSimulation)
            BaseType::startSimulation();
    }

    void Simulation::runOneStep(uint32_t currentStep)
    {
        ParticlePush{}(currentStep);
        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<PRType>>("PRBuf");
        auto [neighbourRegions, regionOffsets] = pmacc::spearhed::CalculateNeighbourRegions{}(prBuf, 5);
    }

    void Simulation::init()
    {
        // Allocate and initialize particle species with all left-over memory below
        // meta::ForEach<VectorAllSpecies, particles::CreateSpecies<boost::mpl::_1>> createSpeciesMemory;
        // createSpeciesMemory(deviceHeap, cellDescription.get());

        size_t freeGpuMem = freeDeviceMemory();
        if(freeGpuMem < reservedGpuMemorySize)
        {
            pmacc::log<pmacc::PMaccVerbose::MEMORY>("%1% MiB free memory < %2% MiB required reserved memory")
                % (freeGpuMem / 1024 / 1024) % (reservedGpuMemorySize / 1024 / 1024);
            std::stringstream msg;
            msg << "Cannot reserve " << (reservedGpuMemorySize / 1024 / 1024) << " MiB as there is only "
                << (freeGpuMem / 1024 / 1024) << " MiB free device memory left";
            throw std::runtime_error(msg.str());
        }

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
        size_t heapSize = freeGpuMem - reservedGpuMemorySize;
        pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
        if(pmacc::Environment<>::get().MemoryInfo().isSharedMemoryPool(
               numRanksPerDevice,
               gc.getCommunicator().getMPIComm()))
        {
            heapSize /= 2u;
            pmacc::log<pmacc::PMaccVerbose::MEMORY>(
                "Shared RAM between GPU and host detected - using only half of the 'device' memory.");
        }
        else
            pmacc::log<pmacc::PMaccVerbose::MEMORY>("Device RAM is NOT shared between GPU and host.");

        // initializing the heap for particles
        // TODO use heapsize instead of the hard coded small heap
        auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(pmacc::ITask::TASK_DEVICE)->getAlpakaQueue();
        auto alpakaDevice = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();

        size_t small_heap{2ull * 1024 * 1024 * 1024};
        deviceHeap.emplace(alpakaDevice, alpakaQueue, small_heap);
        alpaka::wait(alpakaQueue);

        auto mallocMCBuffer = std::make_unique<pmacc::MallocMCBuffer<DeviceHeap>>(*deviceHeap);
        auto& dc = pmacc::Environment<>::get().DataConnector();
        dc.consume(std::move(mallocMCBuffer));

#endif

        // meta::ForEach<VectorAllSpecies, particles::LogMemoryStatisticsForSpecies<boost::mpl::_1>>
        //     logMemoryStatisticsForSpecies;
        // logMemoryStatisticsForSpecies(deviceHeap);

        if(pmacc::PMaccVerbose::MEMORY::lvl)
        {
            freeGpuMem = freeDeviceMemory();
            pmacc::log<pmacc::PMaccVerbose::MEMORY>("free mem after all mem is allocated %1% MiB")
                % (freeGpuMem / 1024 / 1024);
        }

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
        /* add CUDA streams to the QueueController for concurrent execution */
        pmacc::Environment<>::get().QueueController().addQueues(6);
#endif
    }

    /**
     * Fill simulation with initial data
     *
     * @return starting step number (0 for new simulation)
     */
    uint32_t Simulation::fillSimulation()
    {
        // set up boundary (and initial) conditions
        // Setup particle distributions
        // Initialize fields

        // load density description from param file. How is this independent from the domain size?
        //
        auto setup = Setup{};

        std::cout << "hello SPH! domain min: " << setup.domain.min << " max: " << setup.domain.max << std::endl;

        InitRegions{}(*deviceHeap, setup);
        InitParticles{}(setup);

        return 0u;
    }

    void Simulation::resetAll(uint32_t currentStep)
    {
    }

    void Simulation::movingWindowCheck(uint32_t currentStep)
    {
    }

    size_t Simulation::freeDeviceMemory() const
    {
        bool const isDeviceSharedBetweenRanks = numRanksPerDevice >= 2u;
        pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();
        if(isDeviceSharedBetweenRanks)
        {
            // Synchronize to guarantee that all other MPI process on the same device allocated there memory.
            MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));
        }

        // free memory reported by the driver
        size_t freeDeviceMemory = 0u;
        size_t totalAvailableMemory = 0u;

        pmacc::Environment<>::get().MemoryInfo().getMemoryInfo(&freeDeviceMemory, &totalAvailableMemory);

        // amount of memory we reduce the allocation in the case if the test allocation later is failing
        size_t stepSize = 16llu * 1024 * 1024;
        // free memory is by default reduced to keep always a few bytes memory for the driver free.
        if(freeDeviceMemory >= stepSize)
            freeDeviceMemory -= stepSize;

        if(isDeviceSharedBetweenRanks)
        {
            // each MPI rank on the GPU gets the same amount of memory from a GPU
            freeDeviceMemory /= numRanksPerDevice;
            // Synchronize to guarantee that all other MPI process on the same device see the same amount of free
            // memory.
            MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));
        }

        size_t allocatableMemory = freeDeviceMemory;
        bool memAlloced = false;
        // tmpBuffer avoids that the memory is freed before all other MPI ranks created there test buffer
        std::optional<::alpaka::Buf<pmacc::ComputeDevice, std::byte, pmacc::AlpakaDim<1>, size_t>> tmpBuffer{};

        // Check how much memory can be allocated with a single allocation call.
        do
        {
            try
            {
                auto testBuffer = alpaka::allocBuf<std::byte, size_t>(
                    pmacc::manager::Device<pmacc::ComputeDevice>::get().current(),
                    allocatableMemory);
                tmpBuffer = testBuffer;
                memAlloced = true;
            }
            catch(...)
            {
                // reduce step size if left over memory is too small to be reduced
                if(allocatableMemory < stepSize)
                    stepSize = std::min(allocatableMemory, stepSize / 2u);
                // reduce memory to test for the next iteration
                allocatableMemory -= stepSize;
                memAlloced = false;
            }
        } while(!memAlloced && allocatableMemory != 0u);

        if(allocatableMemory < freeDeviceMemory)
        {
            pmacc::log<pmacc::PMaccVerbose::MEMORY>(
                "WARNING (not critical): Reported free memory by the driver %1% byte can not be allocated, "
                "reducing free memory to %2% byte.")
                % freeDeviceMemory % allocatableMemory;
        }

        if(isDeviceSharedBetweenRanks)
        {
            // Wait that all MPI processes had checked the available/allocatable memory.
            MPI_CHECK(MPI_Barrier(gc.getCommunicator().getMPIComm()));
        }

        return allocatableMemory;
    }

} // namespace spearhed
