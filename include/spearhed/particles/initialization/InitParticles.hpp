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
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <unistd.h>

namespace spearhed
{
    namespace init::detail
    {
        /**
         * Kernel which establishes the particle count for each region. The
         * following creation kernel derives the number of frames directly from
         * that count and appends them in list order.
         */
        template<typename TNumParticlesToCreate>
        struct CalculateFramesPerRegion
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numParticleRegions,
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
                    });
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
                using Species = typename FrameType::ParticleDescription::Species;

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
                        setMultiMask<Species>(particle, idx < numParticlesToCreate ? 1 : 0);

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

            template<pmacc::spearhed::SpeciesTag S>
            HDINLINE void setMultiMask(ParticleView<S, multiMask> multiMaskView, uint8_t state) const
            {
                *multiMaskView = state;
            }
        };

        /**
         * @brief Initialise all frames of one region in list order.
         *
         * A region owns one block for the whole operation.  This deliberately
         * trades initialisation parallelism across frames for the storage
         * invariant used everywhere else: every frame before the final frame is
         * full, and the final frame contains a contiguous prefix of live slots.
         * The former one-block-per-frame launch appended frames concurrently, so
         * mallocMC allocation order could put the partial frame in the middle of
         * a list.  PIConGPU repairs that situation with fillAllGaps(); we avoid
         * creating it in the first place.
         */
        struct InitParticleRegions
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numParticleRegions,
                pmacc::IdGenerator idGen,
                auto placeParticle,
                auto placeParticleArgsTuple) const
            {
                auto const regionIdx = worker.blockDomIdx();
                if(regionIdx >= numParticleRegions)
                    return;

                auto& particleRegion = prDeviceBox[regionIdx];
                auto& frameList = particleRegion.particleFrameList;
                using FrameType = typename std::remove_cvref_t<decltype(frameList)>::FrameType;
                constexpr uint32_t frameSize = FrameType::frameSize;

                uint32_t const numFrames = frameList.numFrames();
                if(numFrames == 0u)
                    return;

                uint32_t const particlesInLastFrame = frameList.getSizeLastFrame();
                PMACC_ASSERT(particlesInLastFrame != 0u);

                for(uint32_t frameOffset = 0u; frameOffset < numFrames; ++frameOffset)
                {
                    uint32_t const particlesInThisFrame
                        = (frameOffset + 1u == numFrames) ? particlesInLastFrame : frameSize;
                    detail::CreateParticlesInFrame{}(
                        worker,
                        frameList,
                        particleRegion,
                        particlesInThisFrame,
                        frameOffset,
                        idGen,
                        placeParticle,
                        placeParticleArgsTuple);
                    // CreateParticlesInFrame reuses one shared frame pointer. Ensure every
                    // slot write completed before the master retargets it for the next frame.
                    worker.sync();
                }
            }
        };
    } // namespace init::detail

    /** Host function which fills the simulation with particles
     * goes over all the ParticleRegions and then initializes them using the densities
     * uses a parallelisation strategy of having one block working per frame
     */
    struct InitParticles
    {
        // For each species used by the setup, fill its buffer block by block. A species may be fed by
        // several blocks, so each block initialises only its own contiguous slice of regions using its
        // own recipe; the slices are laid out in the same block order InitRegions used.
        template<SetupInterface TSetup>
        auto operator()(TSetup const& setup)
        {
            pmacc::spearhed::forEachSpecies(
                pmacc::spearhed::species::all,
                [&](auto species) { initSpecies<std::remove_cvref_t<decltype(species)>>(setup); });
        }

    private:
        template<typename Species, SetupInterface TSetup>
        static void initSpecies(TSetup const& setup)
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();

            using PRBuf = pmacc::spearhed::ParticleRegionBuffer<PRTypeFor<Species>>;
            auto const id = pmacc::spearhed::prBufId(Species{});
            if(!dc.hasId(id))
                return; // species not used by this setup
            auto& prBuf = *dc.get<PRBuf>(id);

            // Walk blocks in the same order as InitRegions; each block owns [regionOffset, +numRegions).
            uint32_t regionOffset = 0;
            std::apply(
                [&](auto const&... block)
                {
                    (
                        [&]
                        {
                            using Block = std::remove_cvref_t<decltype(block)>;
                            if constexpr(blockTargets<Block, Species>)
                            {
                                std::vector<pmacc::spearhed::AABB<CS>> volumes;
                                block.template addRegions<Species>(volumes);
                                auto const numRegions = static_cast<uint32_t>(volumes.size());
                                if(numRegions > 0)
                                    initBlockSlice<Species>(dc, prBuf, regionOffset, numRegions, block);
                                regionOffset += numRegions;
                            }
                        }(),
                        ...);
                },
                setup.blocks());

            // The init kernels above (CreateParticlesInFrame, via getEmptyFrame) allocated frames in
            // prBuf's frame lists, so any existing FrameIndexBuffer built over this buffer is now
            // stale; bump the topology version once all blocks have run so the buffer's own
            // bookkeeping reflects the mutation.
            ++prBuf.topologyVersion;
        }

        template<typename Species, typename Block>
        static void initBlockSlice(
            pmacc::DataConnector& dc,
            auto& prBuf,
            uint32_t regionBegin,
            uint32_t numRegions,
            Block const& block)
        {
            constexpr uint32_t threadsPerBlock = 32;

            /**
             * Count particles per region first.  The creation kernel then owns one
             * block per region and appends that region's frames serially, preserving
             * the packed-frame invariant without a later compaction pass.
             */
            auto argsForNumParticles = pmacc::memory::tuple::fromStlTuple(block.numParticlesToCreateArgs());
            auto placeParticle = typename Block::PlaceParticle{};
            auto argsForPlaceParticle = pmacc::memory::tuple::fromStlTuple(block.placeParticleArgs());

            // Restrict the kernels to this block's slice by shifting the device box to its first region.
            // Region-local indices (frame offsets, global particle idx) are unchanged by the shift.
            auto slicedBox = prBuf.getDeviceDataBox().shift(pmacc::DataSpace<DIM1>{static_cast<int>(regionBegin)});
            auto const numRegionsI = static_cast<int>(numRegions);

            // One block per region establishes its total particle count. Frame allocation and
            // placement are transaction-ordered behind this launch.
            PMACC_LOCKSTEP_KERNEL(init::detail::CalculateFramesPerRegion<typename Block::NumParticlesToCreate>{})
                .template config<threadsPerBlock>(
                    pmacc::DataSpace<DIM1>(numRegionsI))(slicedBox, numRegionsI, argsForNumParticles);

            auto idProvider = dc.get<pmacc::IdProvider>("globalId");

            auto event = PMACC_LOCKSTEP_KERNEL(init::detail::InitParticleRegions{})
                             .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numRegionsI))(
                                 slicedBox,
                                 numRegionsI,
                                 idProvider->getDeviceGenerator(),
                                 placeParticle,
                                 argsForPlaceParticle);

            event.waitForFinished();
        }
    };


} // namespace spearhed
