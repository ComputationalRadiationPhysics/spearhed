/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/View.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/Position.hpp"
#include "spmacc/topology/Distance.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        // self interaction must be dealt with by the user in interact Fn
        template<typename ValidParticlePredicate>
        struct FrameInteractionKernel
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numRegions,
                auto framesScanBox,
                auto neighbourRegionsBox,
                auto regionOffsetsBox,
                auto interactionRadius,
                auto fn,
                auto... args) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= static_cast<int>(framesScanBox[numRegions - 1]))
                    return;

                // Binary Search to find the region for this block
                int left = 0;
                int right = numRegions;

                while(left < right)
                {
                    int const mid = left + (right - left) / 2;
                    if(static_cast<int>(framesScanBox[mid]) <= blockIdx)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }

                int const regionIdx = left;
                auto& region = prDeviceBox[regionIdx];
                auto& frameList = region.particleFrameList;
                using FrameType = typename std::remove_reference_t<decltype(frameList)>::FrameType;
                using RecordType = typename FrameType::ParticleRecord;
                constexpr uint32_t frameSize = FrameType::frameSize;

                // We currently load the selected properties into shared memory for one full frame size.
                // We can think of changing (increasing/decreasing) the number of particles cached
                using CachedType = ll::SoA<ll::sub_record_t<RecordType, tags::pos, tags::multiMask>, frameSize>;

                auto const startFrame = (regionIdx == 0) ? 0 : framesScanBox[regionIdx - 1];
                auto const localFrameIdx = blockIdx - startFrame;

                // SMEM Allocations
                PMACC_SMEM(worker, ownFramePtr, pmacc::spearhed::memory::FramePointer<FrameType>);
                PMACC_SMEM(worker, neighbourFramePtr, pmacc::spearhed::memory::FramePointer<FrameType>);

                // Cache array for neighbour attributes to reduce global memory reads
                PMACC_SMEM(worker, smemCache, CachedType);

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        auto itr = frameList.begin();
                        for(int i = 0; i < static_cast<int>(localFrameIdx); i++)
                        {
                            ++itr;
                        }
                        ownFramePtr = memory::FramePointer{&(*itr)};
                    });

                worker.sync();

                auto forEachSlot = pmacc::lockstep::makeForEach<frameSize>(worker);

                int const startNeighbour = regionOffsetsBox[regionIdx];
                int const endNeighbour = regionOffsetsBox[regionIdx + 1];

                for(int n = startNeighbour; n < endNeighbour; ++n)
                {
                    int const neighbourRegionIdx = neighbourRegionsBox[n];
                    auto& neighbourRegion = prDeviceBox[neighbourRegionIdx];
                    auto& neighbourFrameList = neighbourRegion.particleFrameList;

                    for(auto it = neighbourFrameList.begin(); it != neighbourFrameList.end(); ++it)
                    {
                        onlyMaster([&]() { neighbourFramePtr = memory::FramePointer{&*it}; });
                        worker.sync();

                        // Cooperatively load neighbour attributes into shared memory
                        forEachSlot(
                            [&](uint32_t const idx)
                            {
                                auto nParticle = neighbourFramePtr[idx];
                                bool const isValid = ValidParticlePredicate{}(nParticle);
                                *smemCache[idx][tags::multiMask] = isValid;
                                if(isValid)
                                {
                                    smemCache[idx] = nParticle;
                                }
                            });
                        worker.sync();


                        // Interact own particles against the populated SMEM buffer
                        forEachSlot(
                            [&](uint32_t const myIdx)
                            {
                                auto ownParticle = ownFramePtr[myIdx];
                                if(ValidParticlePredicate{}(ownParticle))
                                {
                                    for(uint32_t j = 0; j < frameSize; ++j)
                                    {
                                        auto cachedNeighbourParticle = smemCache[j];
                                        if(*cachedNeighbourParticle[tags::multiMask])
                                        {
                                            //  can add a check for self interaction, neighbourRegionIdx == regionIdx
                                            //  && ownFrameRawPtr == neighbourFrameRawPtr && (myIdx == j);
                                            if(distance(
                                                   ownParticle[tags::pos].get(),
                                                   cachedNeighbourParticle[tags::pos].get())
                                               < interactionRadius)
                                            {
                                                auto neighbourParticle = neighbourFramePtr[j];
                                                fn(worker, ownParticle, neighbourParticle, args...);
                                            }
                                        }
                                    }
                                }
                            });
                        // Guard against overwriting SMEM before all threads finish
                        worker.sync();
                    }
                }
            }
        };
    } // namespace detail

    /**
     * Host Helper to launch pair-wise interactions using the precalculated neighbour list.
     */
    struct InteractParticles
    {
        /**
         * @param fn Functor with the particle interaction logic between `ownParticle` and the cached
         * payload.
         */
        void operator()(
            auto& prBuf,
            auto const& neighbourRegions,
            auto const& regionOffsets,
            auto interactionRadius,
            auto fn,
            auto&&... args) const
        {
            if(prBuf.size == 0)
                return;

            constexpr uint32_t threadsPerBlock = 128;

            pmacc::HostDeviceBuffer<unsigned int, DIM1> framesPerRegion(pmacc::DataSpace<DIM1>{prBuf.size});

            PMACC_LOCKSTEP_KERNEL(detail::CountFramesKernel{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(
                    prBuf.size))(prBuf.getDeviceDataBox(), prBuf.size, framesPerRegion.getDeviceBuffer().getDataBox());

            framesPerRegion.deviceToHost();
            auto hostData = framesPerRegion.getHostBuffer().getDataBox();
            for(int i = 1; i < prBuf.size; ++i)
            {
                hostData[i] += hostData[i - 1];
            }
            uint32_t const totalBlocks = hostData[prBuf.size - 1];
            framesPerRegion.hostToDevice();

            if(totalBlocks > 0)
            {
                PMACC_LOCKSTEP_KERNEL((detail::FrameInteractionKernel<detail::OccupiedSlot>{}))
                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(totalBlocks))(
                        prBuf.getDeviceDataBox(),
                        prBuf.size,
                        framesPerRegion.getDeviceBuffer().getDataBox(),
                        neighbourRegions.getDeviceBuffer().getDataBox(),
                        regionOffsets.getDeviceBuffer().getDataBox(),
                        interactionRadius,
                        fn,
                        std::forward<decltype(args)>(args)...);

                pmacc::eventSystem::waitForAllTasks();
            }
        }
    };

} // namespace pmacc::spearhed
