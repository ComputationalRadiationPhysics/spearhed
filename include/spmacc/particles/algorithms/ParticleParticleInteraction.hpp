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
#include "spmacc/particles/regions/NeighbourBundle.hpp"

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

        template<typename Record, typename TagContainer>
        struct OwnCacheTypeBuilder;

        template<typename Record, ll::IsRecordAccess auto... KernelTags>
        struct OwnCacheTypeBuilder<Record, ll::TagList<KernelTags...>>
        {
            // relativePos is mandatory for the own-particle position calculation
            using type = ll::sub_record_t<Record, tags::relativePos, KernelTags...>;
        };

        /**
         * @brief Cooperatively load one neighbour frame into the shared-memory cache.
         *
         * Each slot's multiMask records validity; valid slots are deep-copied into the
         * cache. Caller must worker.sync() before reading the cache. Shared by the
         * single-launch and per-source interaction kernels.
         */
        template<typename ValidParticlePredicate>
        DINLINE void loadNeighbourCacheToSmem(auto& forEachSlot, auto& smemCache, auto neighbourFramePtr)
        {
            forEachSlot(
                [&, neighbourFramePtr](uint32_t const idx)
                {
                    auto nParticle = neighbourFramePtr[idx];
                    bool const isValid = ValidParticlePredicate{}(nParticle);
                    *smemCache[idx][tags::multiMask] = isValid;
                    if(isValid)
                        smemCache[idx].deepCopyFrom(nParticle);
                });
        }

        /**
         * @brief Interact one own particle against every valid neighbour in the SMEM cache.
         *
         * Calls fn(worker, ownView, neighbourParticle, InteractionContext) for each pair
         * within interactionRadius. @p isSelfFrame must be true only when the cached frame
         * is the own particle's own frame, so the j == myIdx pair is flagged as the self
         * interaction. Shared by the single-launch and per-source interaction kernels.
         *
         * @tparam frameSize Number of slots per frame (compile-time loop bound).
         */
        template<uint32_t frameSize>
        DINLINE void interactOwnWithCache(
            auto const& worker,
            auto& smemCache,
            uint32_t myIdx,
            auto& ownView,
            auto const& ownAbsPos,
            bool isSelfFrame,
            auto neighbourFramePtr,
            auto const& neighbourVolume,
            auto interactionRadius,
            auto& fn,
            auto&... args)
        {
            for(uint32_t j = 0; j < frameSize; ++j)
            {
                auto cachedNeighbourParticle = smemCache[j];
                if(*cachedNeighbourParticle[tags::multiMask])
                {
                    auto const neighAbsPos
                        = neighbourVolume.getPosition(cachedNeighbourParticle[tags::relativePos].get());
                    auto const r_vec = ownAbsPos - neighAbsPos;
                    // TODO think about using r squared to avoiud the square root
                    auto const r = norm2(r_vec);
                    if(r < interactionRadius)
                    {
                        auto neighbourParticle = neighbourFramePtr[j];
                        bool const is_self = isSelfFrame && (j == myIdx);
                        using RVecType = std::decay_t<decltype(r_vec)>;
                        fn(worker,
                           ownView,
                           neighbourParticle,
                           InteractionContext<typename RVecType::CS>{r_vec, is_self},
                           args...);
                    }
                }
            }
        }

        // self interaction must be dealt with by the user in interact Fn
        template<typename ValidParticlePredicate>
        struct FrameInteractionKernel
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto targetPRDeviceBox,
                int numTargetRegions,
                auto framesScanBox,
                auto sourceView,
                auto interactionRadius,
                auto fn,
                auto... args) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= static_cast<int>(framesScanBox[numTargetRegions - 1]))
                    return;

                auto const loc = findFrameLocation(blockIdx, framesScanBox, numTargetRegions);

                auto& region = targetPRDeviceBox[loc.regionIdx];
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
                // Cache array for neighbour particles for attributes needed by the kernel + geometry
                using CachedType = ll::SoA<SubRecord, frameSize>;

                using OwnSubRecord = typename OwnCacheTypeBuilder<RecordType, typename FnType::RequiredOwnTags>::type;

                PMACC_SMEM(worker, smemCache, CachedType);

                auto itr = frameList.begin();
                std::advance(itr, loc.localFrameIdx);
                memory::FramePointer const ownFramePtr{&*itr};
                VolumeType const ownVolume = region.volume;

                auto forEachSlot = pmacc::lockstep::makeForEach<frameSize>(worker);

                int const startNeighbour = sourceView.regionOffsetsBox[loc.regionIdx];
                int const endNeighbour = sourceView.regionOffsetsBox[loc.regionIdx + 1];

                for(int n = startNeighbour; n < endNeighbour; ++n)
                {
                    int const neighbourRegionIdx = sourceView.neighbourRegionsBox[n];
                    auto& neighbourRegion = sourceView.sourcePRDeviceBox[neighbourRegionIdx];
                    auto& neighbourFrameList = neighbourRegion.particleFrameList;

                    for(auto it = neighbourFrameList.begin(); it != neighbourFrameList.end(); ++it)
                    {
                        memory::FramePointer const neighbourFramePtr{&*it};
                        VolumeType const neighbourVolume = neighbourRegion.volume;
                        // Each live frame is a unique heap allocation belonging to exactly one
                        // region of one buffer, so equal frame pointers already identify the same
                        // frame (same region, same buffer).
                        bool const isSelfFrame = (ownFramePtr == neighbourFramePtr);

                        // Cooperatively load neighbour attributes into shared memory
                        loadNeighbourCacheToSmem<ValidParticlePredicate>(forEachSlot, smemCache, neighbourFramePtr);
                        worker.sync();

                        // Interact own particles against the populated SMEM buffer.
                        // Own particle attributes are cached in per-thread registers via ll::One.
                        forEachSlot(
                            [&, ownFramePtr, ownVolume, neighbourFramePtr, neighbourVolume, isSelfFrame](
                                uint32_t const myIdx)
                            {
                                auto ownParticle = ownFramePtr[myIdx];
                                if(!ValidParticlePredicate{}(ownParticle))
                                    return;

                                ll::One<OwnSubRecord> ownCache{ownParticle};
                                auto ownView = ownCache[uint32_t{0}];
                                auto const ownAbsPos = ownVolume.getPosition(ownView[tags::relativePos].get());

                                interactOwnWithCache<frameSize>(
                                    worker,
                                    smemCache,
                                    myIdx,
                                    ownView,
                                    ownAbsPos,
                                    isSelfFrame,
                                    neighbourFramePtr,
                                    neighbourVolume,
                                    interactionRadius,
                                    fn,
                                    args...);

                                ownView.deepCopyTo(ownParticle);
                            });
                        // Guard against overwriting SMEM before all threads finish
                        worker.sync();
                    }
                }
            }
        };
    } // namespace detail

    /**
     * Host Helper to launch pair-wise interactions using the precalculated neighbour lists.
     * Launches one kernel pass per source entry in the bundle.
     *
     * @param neighbourBundle  NeighbourBundle (or NeighbourBundleView) holding the target buffer
     *                         and the neighbour lists for each source. Call bundle.subset<Roles...>()
     *                         to restrict which sources are iterated.
     * @param fn Functor with the particle interaction logic between `ownParticle` and
     *           `neighbourParticle`. Must expose:
     *             using RequiredSharedTags = ll::TagList<...>;  // cached in neighbour smem
     *             using RequiredOwnTags    = ll::TagList<...>;  // cached in own registers
     *           Must accumulate contributions additively
     *           on the target particle: each source buffer is processed in a separate kernel pass that
     *           writes independently to the same target attributes.
     */
    struct InteractParticles
    {
        void operator()(IsNeighbourBundle auto&& neighbourBundle, auto interactionRadius, auto fn, auto&&... args)
            const
        {
            neighbourBundle.forEachDeviceView(
                [&](auto sourceView)
                {
                    ForEachFrameInPRBuf<32, 128>{}(
                        neighbourBundle.target(),
                        detail::FrameInteractionKernel<detail::OccupiedSlot>{},
                        sourceView,
                        interactionRadius,
                        fn,
                        std::forward<decltype(args)>(args)...);
                });
        }
    };

} // namespace pmacc::spearhed
