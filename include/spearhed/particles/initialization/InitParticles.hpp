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

#include "Algorithm.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/ParticleView.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/particles/initialization/SetupInterface.hpp"
#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/algorithms/FrameDispatch.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/eventSystem/waitForAllTasks.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>
#include <pmacc/memory/tuple/utility.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/conversion/ResolveAndRemoveFromSeq.hpp>
#include <pmacc/particles/IdProvider.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/operations/InitValueIdentifier.hpp>

#include <concepts>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <unistd.h>

namespace spearhed
{
    namespace init::detail
    {

        /** Kernel which calculates number of frames needed per particle region
         * One block per particle region
         * Results stored as inclusive prefix sum in framesPerParticleRegionScan
         */
        template<typename TNumParticlesToCreate>
        struct CalculateFramesPerRegion
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numParticleRegions,
                auto framesPerParticleRegionBox,
                auto numParticlesToCreateArgsTuple) const
            {
                auto const blockIdx = worker.blockDomIdx();

                if(blockIdx >= numParticleRegions)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);

                onlyMaster(
                    [&]()
                    {
                        auto& particleRegion = prDeviceBox[blockIdx];
                        auto& frameList = particleRegion.particleFrameList;

                        uint32_t numParticles = pmacc::memory::tuple::apply(
                            [&](auto&&... args) { return TNumParticlesToCreate{}(worker, particleRegion, args...); },
                            numParticlesToCreateArgsTuple);

                        frameList.setNumParticles(numParticles);
                        framesPerParticleRegionBox[blockIdx] = frameList.numFrames();
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
             * @param particleRegion particle region this frame belongs to
             * @param numParticlesToCreate number of particles to create in this frame
             * @param frameOffset index of this frame within its particle region (for computing global particle idx)
             * @param idGen id generator
             * @param placeParticle callable that places a single particle
             * @param placeParticleArgs extra args forwarded to placeParticle
             */
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto& particleFrameList,
                auto const& particleRegion,
                uint32_t numParticlesToCreate,
                uint32_t frameOffset,
                pmacc::IdGenerator& idGen,
                auto placeParticle,
                auto placeParticleArgsTuple) const
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
                        framePtr->liveParticles = numParticlesToCreate;
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
                                particleId>(particle);

                            pmacc::spearhed::Init<idField>{}(particle[particleId], worker, idGen);

                            pmacc::memory::tuple::apply(
                                [&](auto&&... args)
                                {
                                    placeParticle(
                                        worker,
                                        particle,
                                        particleRegion,
                                        frameOffset * FrameType::frameSize + idx,
                                        args...);
                                },
                                placeParticleArgsTuple);
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
                pmacc::IdGenerator idGen,
                auto placeParticle,
                auto placeParticleArgsTuple) const
            {
                auto const blockIdx = worker.blockDomIdx();
                auto const loc = pmacc::spearhed::detail::findFrameLocation(
                    blockIdx,
                    framesPerParticleRegionScan,
                    numParticleRegions);

                PMACC_ASSERT(loc.localFrameIdx < loc.framesInRegion);

                auto& particleRegion = prDeviceBox[loc.regionIdx];

                auto& frameList = particleRegion.particleFrameList;

                constexpr uint32_t frameSize = std::remove_cvref_t<decltype(frameList)>::FrameType::frameSize;

                // Distribute particles across the frames assigned to this region
                // Calculate how many particles this block should create
                // frame is full unless it is the last one
                // If this is the last frame for the region, adjust particle count

                // precondition: this is non zero. Should be guaranteed when calculating the scan
                uint32_t particlesInLastFrame = frameList.getSizeLastFrame();
                PMACC_ASSERT(particlesInLastFrame != 0);

                uint32_t const particlesInThisFrame
                    = (loc.localFrameIdx == loc.framesInRegion - 1) ? particlesInLastFrame : frameSize;

                // Create particles in this frame
                detail::CreateParticlesInFrame{}(
                    worker,
                    frameList,
                    particleRegion,
                    particlesInThisFrame,
                    static_cast<uint32_t>(loc.localFrameIdx),
                    idGen,
                    placeParticle,
                    placeParticleArgsTuple);
            }
        };
    } // namespace init::detail

    /** Host function which fills the simulation with particles
     * goes over all the ParticleRegions and then initializes them using the densities
     * uses a parallelisation strategy of having one block working per frame
     */
    struct InitParticles
    {
        // iterates over TSetup::Roles and initialises each role's PRBuf.
        template<SetupInterface TSetup>
        auto operator()(TSetup const& setup)
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();
            using Roles = typename TSetup::Roles;

            auto initOne = [&]<typename Role>()
            {
                auto& prBuf
                    = *dc.get<pmacc::spearhed::ParticleRegionBuffer<PRType, Role>>(pmacc::spearhed::prBufId<Role>());
                initSingleBlock<Role>(prBuf, setup);
            };

            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                (initOne.template operator()<std::tuple_element_t<I, Roles>>(), ...);
            }(std::make_index_sequence<std::tuple_size_v<Roles>>{});
        }

    private:
        template<typename Role, SetupInterface TSetup>
        static void initSingleBlock(auto& prBuf, TSetup const& setup)
        {
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
            auto argsForNumParticles
                = pmacc::memory::tuple::fromStlTuple(setup.template numParticlesToCreateArgs<Role>());
            auto placeParticle = typename TSetup::template PlaceParticle<Role>{};
            auto argsForPlaceParticle = pmacc::memory::tuple::fromStlTuple(setup.template placeParticleArgs<Role>());

            // Launch a kernel to calculate num particles & num frames to create for each PR
            // Uses one block for each PR to calculate these 2 numbers.
            // TODO this is very wasteful. Use threads in a block to deal with particle regions and do a on device scan
            // Stores the num Frames in a scan/ prefix sum
            // stores the num particles in a frame list
            pmacc::HostDeviceBuffer<unsigned int, DIM1> framesPerParticleRegion(pmacc::DataSpace<DIM1>{prBuf.size});
            PMACC_LOCKSTEP_KERNEL(
                init::detail::CalculateFramesPerRegion<typename TSetup::template NumParticlesToCreate<Role>>{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(prBuf.size))(
                    prBuf.getDeviceDataBox(),
                    prBuf.size,
                    framesPerParticleRegion.getDeviceBuffer().getDataBox(),
                    argsForNumParticles);

            uint32_t const totalBlocks = pmacc::spearhed::inclusiveScanOnHost(framesPerParticleRegion, prBuf.size);

            // Launch a kernel to init particles.
            // Launched with max blocks and num threads per block that we can possible use.
            // use the prefix sum of num frames for mapping blocks to PRs and use numParticles to assign exact work
            // to each block.
            if(totalBlocks > 0)
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto idProvider = dc.get<pmacc::IdProvider>("globalId");

                PMACC_LOCKSTEP_KERNEL(init::detail::InitParticleRegions{})
                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(totalBlocks))(
                        prBuf.getDeviceDataBox(),
                        prBuf.size,
                        framesPerParticleRegion.getDeviceBuffer().getDataBox(),
                        idProvider->getDeviceGenerator(),
                        placeParticle,
                        argsForPlaceParticle);

                // wait because otherwise kernel args (framesPerParticleRegion) go out of scope
                pmacc::eventSystem::waitForAllTasks();
            }
        }
    };


} // namespace spearhed
