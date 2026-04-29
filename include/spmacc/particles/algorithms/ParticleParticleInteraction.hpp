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

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/View.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
#include "spmacc/particles/algorithms/FrameDispatch.hpp"
#include "spmacc/particles/algorithms/InteractionContext.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<typename Record, typename TagContainer>
        struct CacheTypeBuilder;

        template<typename Record, ll::IsRecordAccess auto... KernelTags>
        struct CacheTypeBuilder<Record, ll::TagList<KernelTags...>>
        {
            // multiMask and relativePos are mandatory for geometry and validity checks
            using type = ll::sub_record_t<Record, tags::relativePos, tags::multiMask, KernelTags...>;
        };

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

                auto const loc = findFrameLocation(blockIdx, framesScanBox, numRegions);

                auto& region = prDeviceBox[loc.regionIdx];
                auto& frameList = region.particleFrameList;
                using FrameType = typename std::remove_reference_t<decltype(frameList)>::FrameType;
                using VolumeType = typename std::remove_reference_t<decltype(region.volume)>;

                using RecordType = typename FrameType::ParticleRecord;
                constexpr uint32_t frameSize = FrameType::frameSize;

                // We currently load the selected properties into shared memory for one full frame size.
                // We can think of changing (increasing/decreasing) the number of particles cached
                // Dynamically deduce the required shared memory tags from the C++20 kernel definition
                using FnType = std::remove_cvref_t<decltype(fn)>;
                using SubRecord = typename CacheTypeBuilder<RecordType, typename FnType::RequiredSharedTags>::type;
                // Create the SoA caching only the exact fields needed by the kernel + geometry
                using CachedType = ll::SoA<SubRecord, frameSize>;

                // Cache array for neighbour attributes to reduce global memory reads
                PMACC_SMEM(worker, smemCache, CachedType);

                auto itr = frameList.begin();
                std::advance(itr, loc.localFrameIdx);
                memory::FramePointer const ownFramePtr{&*itr};
                VolumeType const ownVolume = region.volume;

                auto forEachSlot = pmacc::lockstep::makeForEach<frameSize>(worker);

                int const startNeighbour = regionOffsetsBox[loc.regionIdx];
                int const endNeighbour = regionOffsetsBox[loc.regionIdx + 1];

                for(int n = startNeighbour; n < endNeighbour; ++n)
                {
                    int const neighbourRegionIdx = neighbourRegionsBox[n];
                    auto& neighbourRegion = prDeviceBox[neighbourRegionIdx];
                    auto& neighbourFrameList = neighbourRegion.particleFrameList;

                    for(auto it = neighbourFrameList.begin(); it != neighbourFrameList.end(); ++it)
                    {
                        memory::FramePointer const neighbourFramePtr{&*it};
                        VolumeType const neighbourVolume = neighbourRegion.volume;
                        // Cooperatively load neighbour attributes into shared memory
                        forEachSlot(
                            [&, neighbourFramePtr](uint32_t const idx)
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
                            [&, ownFramePtr, ownVolume, neighbourFramePtr, neighbourVolume](uint32_t const myIdx)
                            {
                                auto ownParticle = ownFramePtr[myIdx];
                                if(ValidParticlePredicate{}(ownParticle))
                                {
                                    auto const ownAbsPos = ownVolume.getPosition(ownParticle[tags::relativePos].get());
                                    for(uint32_t j = 0; j < frameSize; ++j)
                                    {
                                        auto cachedNeighbourParticle = smemCache[j];
                                        if(*cachedNeighbourParticle[tags::multiMask])
                                        {
                                            auto const neighAbsPos = neighbourVolume.getPosition(
                                                cachedNeighbourParticle[tags::relativePos].get());
                                            auto const r_vec = ownAbsPos - neighAbsPos;
                                            auto const r = norm2(r_vec);
                                            if(r < interactionRadius)
                                            {
                                                auto neighbourParticle = neighbourFramePtr[j];
                                                bool const is_self = (neighbourRegionIdx == loc.regionIdx)
                                                                     && (ownFramePtr == neighbourFramePtr)
                                                                     && (j == myIdx);
                                                using RVecType = std::decay_t<decltype(r_vec)>;
                                                fn(worker,
                                                   ownParticle,
                                                   neighbourParticle,
                                                   InteractionContext<typename RVecType::CS>{r_vec, is_self},
                                                   args...);
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
            ForEachFrameInPRBuf<32, 128>{}(
                prBuf,
                detail::FrameInteractionKernel<detail::OccupiedSlot>{},
                neighbourRegions.getDeviceBuffer().getDataBox(),
                regionOffsets.getDeviceBuffer().getDataBox(),
                interactionRadius,
                fn,
                std::forward<decltype(args)>(args)...);
        }
    };

} // namespace pmacc::spearhed
