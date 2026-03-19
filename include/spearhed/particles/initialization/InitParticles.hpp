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

#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/ParticleView.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/eventSystem/waitForAllTasks.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/conversion/ResolveAndRemoveFromSeq.hpp>
#include <pmacc/particles/IdProvider.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/operations/InitValueIdentifier.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <unistd.h>

namespace spearhed
{
    namespace init::detail
    {

        constexpr auto baseNumParticlesToCreate = 400u;

        // calculate how many particles we need to make in this system
        struct NumParticlesToCreate
        {
            constexpr auto operator()([[maybe_unused]] auto prDeviceBox, std::integral auto index) const
            {
                return baseNumParticlesToCreate * (index + 1);
            };
        };

        /** Kernel which calculates number of frames needed per particle region
         * One block per particle region
         * Results stored as inclusive prefix sum in framesPerParticleRegionScan
         */
        struct CalculateFramesPerRegion
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numParticleRegions,
                auto framesPerParticleRegionBox) const
            {
                auto const blockIdx = worker.blockDomIdx();

                if(blockIdx >= numParticleRegions)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);

                onlyMaster(
                    [&]()
                    {
                        auto& frameList = prDeviceBox[blockIdx].particleFrameList;

                        constexpr uint32_t frameSize = std::remove_cvref_t<decltype(frameList)>::FrameType::frameSize;
                        uint32_t numParticles = NumParticlesToCreate{}(prDeviceBox, blockIdx);
                        uint32_t const numFrames = alpaka::core::divCeil(numParticles, frameSize);

                        frameList.setNumParticles(numParticles);
                        framesPerParticleRegionBox[blockIdx] = numFrames;
                    });
                // TODO do scan on device
            }
        };

        /** Kernel which allocates a frame, creates particles and adds it to the frame list for the particle region
         * Assumes that the base particleFrameList was empty initially
         */
        struct CreateParticlesInFrame
        {
            /**
             * @param particleFrameList frameList to which we add our frame
             * @param numParticlesToCreate
             */
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto& particleFrameList,
                uint32_t numParticlesToCreate,
                pmacc::IdGenerator& idGen /** init args */) const
            {
                using TFrameList = std::remove_cvref_t<decltype(particleFrameList)>;
                using FrameType = TFrameList::FrameType;

                PMACC_SMEM(worker, framePtr, pmacc::spearhed::memory::FramePointer<FrameType>);

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        // allocate the frame for this block and get a pointer to it
                        // This is filled with junk values
                        framePtr = particleFrameList.getEmptyFrame(worker);
                    });

                worker.sync();

                auto forEachSlotInFrame = pmacc::lockstep::makeForEach<FrameType::frameSize>(worker);

                // fill frames in parallel
                forEachSlotInFrame(
                    [&](uint32_t const idx)
                    {
                        auto particle = framePtr[idx];
                        // First set the multimask to make sure particles which dont exist are disabled
                        setMultiMask(particle, idx < numParticlesToCreate ? 1 : 0);

                        if(idx < numParticlesToCreate)
                        {
                            ll::iterate_except<
                                typename decltype(particle)::record_type,
                                pmacc::spearhed::InitZero,
                                multiMask,
                                particleId,
                                vel>(particle);

                            pmacc::spearhed::Init<idField>{}(particle[particleId], worker, idGen);
                            pmacc::spearhed::InitValue<velField>{}(particle[vel], 100.f);
                        }
                    });
            }

            DINLINE void setMultiMask(ParticleView<multiMask> multiMaskView, uint8_t state) const
            {
                *multiMaskView = state;
            }
        };

        /** Kernel which goes over all particle regions and creates particles
         * Each block creates a frame and fills it
         * Kernel is independent of the number of threads and blocks it is started with
         */
        struct InitParticleRegions
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numParticleRegions,
                auto framesPerParticleRegionScan,
                pmacc::IdGenerator idGen /** init args */) const
            {
                // Use the scan array to find which particle region this block is responsible for
                // framesPerParticleRegionScan holds the inlcusive prefix sum of frames per region
                // We need to find the region where: scan[region] <= blockIdx < scan[region+1]
                auto const blockIdx = worker.blockDomIdx();
                int particleRegionIdx = -1;
                int framesOffset = 0;
                int framesInRegion = 0;

                // Linear search for the particle region where this block falls in the inclusive scan
                // TODO use binary search
                for(int i = 0; i < numParticleRegions; ++i)
                {
                    int endFrameIdx = framesPerParticleRegionScan[i];
                    int startFrameIdx = (i == 0) ? 0 : (framesPerParticleRegionScan[i - 1]);

                    if(blockIdx >= startFrameIdx && blockIdx < endFrameIdx)
                    {
                        particleRegionIdx = i;
                        framesOffset = blockIdx - startFrameIdx;
                        framesInRegion = endFrameIdx - startFrameIdx;
                        break;
                    }
                }

                PMACC_ASSERT(particleRegionIdx != -1);
                PMACC_ASSERT(framesOffset < framesInRegion);

                auto& particleRegion = prDeviceBox[particleRegionIdx];

                auto& frameList = particleRegion.particleFrameList;

                constexpr uint32_t frameSize = std::remove_cvref_t<decltype(frameList)>::FrameType::frameSize;

                // Distribute particles across the frames assigned to this region
                // Calculate how many particles this block should create
                // frame is full unless it is the last one
                // If this is the last frame for the region, adjust particle count

                // precondition: this is non zero. Should be guaranteed when calculating the scan
                uint32_t particlesInLastFrame = frameList.getSizeLastFrame();
                PMACC_ASSERT(particlesInLastFrame != 0);

                uint32_t particlesInThisFrame
                    = (framesOffset == framesInRegion - 1) ? particlesInLastFrame : frameSize;

                // Create particles in this frame
                detail::CreateParticlesInFrame{}(worker, frameList, particlesInThisFrame, idGen);
            }
        };
    } // namespace init::detail

    /** Host function which fills the simulation with particles
     * goes over all the ParticleRegions and then initializes them using the densities
     * uses a parallelisation strategy of having one block working per frame
     */
    // do we do boundaries here?
    struct InitParticles
    {
        /**
         * @param data box holding all particle regions on the device
         * @param size number of particle regions in the data box (data box extent)
         */
        auto operator()()
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();
            auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<PRType>>("PRBuf");
            constexpr uint32_t threadsPerBlock = 32;

            /**
             * for each particle system we want to assign as many blocks as it needs for its work
             *
             * - calculate number of particles to be created for each system. Based on the density functors etc
             * - calculate number of blocks needed for each system from the numParticles and frame size
             * - do a scan, and create a mapping from blocks to frame idx
             * - we need some natural order of particle initialization, which can be split by number of frame slots
             * so that particle init can be independent across blocks and threads
             */

            // Launch a kernel to calculate num particles & num frames to create for each PR
            // Uses one block for each PR to calculate these 2 numbers.
            // TODO this is very wasteful. Use threads in a block to deal with particle regions and do a on device scan
            // Stores the num Frames in a scan/ prefix sum
            // stores the num particles in a frame list
            pmacc::HostDeviceBuffer<unsigned int, DIM1> framesPerParticleRegion(pmacc::DataSpace<DIM1>{prBuf.size});

            PMACC_LOCKSTEP_KERNEL(init::detail::CalculateFramesPerRegion{})
                .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(prBuf.size))(
                    prBuf.getDeviceDataBox(),
                    prBuf.size,
                    framesPerParticleRegion.getDeviceBuffer().getDataBox());

            framesPerParticleRegion.deviceToHost();

            auto hostData = framesPerParticleRegion.getHostBuffer().getDataBox();
            for(int i = 1; i < prBuf.size; ++i)
            {
                hostData[i] += hostData[i - 1];
            }
            uint32_t const totalBlocks = hostData[prBuf.size - 1];

            framesPerParticleRegion.hostToDevice();

            // Launch a kernel to init particles.
            // Launched with max blocks and num threads per block that we can possible use.
            // use the prefix sum of num frames for mapping blocks to PRs and use numParticles to assign exact work
            // to each block.
            if(totalBlocks > 0)
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto idProvider = dc.get<pmacc::IdProvider>("globalId");

                PMACC_LOCKSTEP_KERNEL(init::detail::InitParticleRegions{})
                    .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(totalBlocks))(
                        prBuf.getDeviceDataBox(),
                        prBuf.size,
                        framesPerParticleRegion.getDeviceBuffer().getDataBox(),
                        idProvider->getDeviceGenerator());

                // wait because otherwise kernel args (framesPerParticleRegion) go out of scope
                pmacc::eventSystem::waitForAllTasks();
            }
        }
    };


} // namespace spearhed
