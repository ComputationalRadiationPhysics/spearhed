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
#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        /**
         * @brief Persistence-builder for own-particle record: includes
         *        relativePos, multiMask, and the functor's RequiredOwnTags.
         *
         * Unlike OwnCacheTypeBuilder, this variant also stores multiMask so
         * validity can be checked from the persistent copy across source
         * iterations.
         */
        template<typename Record, typename TagContainer>
        struct PersistentOwnCacheTypeBuilder;

        template<typename Record, ll::IsRecordAccess auto... KernelTags>
        struct PersistentOwnCacheTypeBuilder<Record, ll::TagList<KernelTags...>>
        {
            using type = ll::sub_record_t<Record, tags::relativePos, tags::multiMask, KernelTags...>;
        };

        /**
         * @brief Single-launch GPU kernel that iterates over all source views
         *        in one kernel invocation, amortising target-side work.
         *
         * Own-particle state is persisted in shared memory across source
         * iterations and written back to the target frame only once at the end.
         *
         * @tparam ValidParticlePredicate  Predicate for live-particle check.
         */
        template<typename ValidParticlePredicate>
        struct UnifiedFrameInteractionKernel
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto targetPRDeviceBox,
                int numTargetRegions,
                auto framesScanBox,
                auto sourceViewTuple,
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

                using FnType = std::remove_cvref_t<decltype(fn)>;

                // Neighbour SMEM cache: geometry fields + the functor's RequiredSharedTags
                using SubRecord = typename CacheTypeBuilder<RecordType, typename FnType::RequiredSharedTags>::type;
                using CachedType = ll::SoA<SubRecord, frameSize>;

                // Persistent own-particle state: geometry fields, mask, and RequiredOwnTags
                using PersistentOwnRec =
                    typename PersistentOwnCacheTypeBuilder<RecordType, typename FnType::RequiredOwnTags>::type;
                using OwnPersistentType = ll::SoA<PersistentOwnRec, frameSize>;

                PMACC_SMEM(worker, smemCache, CachedType);
                PMACC_SMEM(worker, ownSmem, OwnPersistentType);

                auto itr = frameList.begin();
                std::advance(itr, loc.localFrameIdx);
                memory::FramePointer const ownFramePtr{&*itr};
                VolumeType const ownVolume = region.volume;

                auto forEachSlot = pmacc::lockstep::makeForEach<frameSize>(worker);

                // --- Stage 1: copy own-particle state into persistent SMEM ---
                forEachSlot(
                    [&](uint32_t const myIdx)
                    {
                        auto ownParticle = ownFramePtr[myIdx];
                        if(ValidParticlePredicate{}(ownParticle))
                            ownSmem[myIdx].deepCopyFrom(ownParticle);
                        else
                            *ownSmem[myIdx][tags::multiMask] = false;
                    });
                worker.sync();

                // --- Stage 2: iterate over all source views; accumulate into ownSmem ---
                [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    auto processSource = [&](auto const& sourceView)
                    {
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
                                // Equal frame pointers identifies the frames as the same
                                bool const isSelfFrame = (ownFramePtr == neighbourFramePtr);

                                // Cooperatively load neighbour attributes into SMEM
                                loadNeighbourCacheToSmem<ValidParticlePredicate>(
                                    forEachSlot,
                                    smemCache,
                                    neighbourFramePtr);
                                worker.sync();

                                // Interact using persistent own-particle state
                                forEachSlot(
                                    [&, neighbourFramePtr, neighbourVolume, isSelfFrame](uint32_t const myIdx)
                                    {
                                        if(!*ownSmem[myIdx][tags::multiMask])
                                            return;

                                        auto ownView = ownSmem[myIdx];
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
                                        // no deepCopyTo here - accumulated state is in ownSmem
                                    });
                                worker.sync();
                            }
                        }
                    };
                    (processSource(pmacc::memory::tuple::get<Is>(sourceViewTuple)), ...);
                }(std::make_index_sequence<
                    pmacc::memory::tuple::tuple_size_v<std::remove_cvref_t<decltype(sourceViewTuple)>>>{});

                // --- Stage 3: write accumulated state back to target frame (once) ---
                forEachSlot(
                    [&](uint32_t const myIdx)
                    {
                        auto ownParticle = ownFramePtr[myIdx];
                        if(ValidParticlePredicate{}(ownParticle))
                            ownSmem[myIdx].deepCopyTo(ownParticle);
                    });
            }
        };

    } // namespace detail

    /**
     * @brief Host helper that launches a single kernel to perform
     *        pairwise particle interactions across all source entries.
     *
     * Unlike InteractParticles (one kernel launch per source entry),
     * InteractParticlesUnified launches exactly one kernel. Source
     * entries are compile-time iterated inside the kernel body.
     * Own-particle state persists in shared memory across source
     * iterations; deepCopyTo runs once per target particle.
     *
     * The functor must satisfy the same interface as for InteractParticles:
     *   - using RequiredSharedTags = ll::TagList<...>;
     *   - using RequiredOwnTags    = ll::TagList<...>;
     *
     * @param neighbourBundle  NeighbourBundle or NeighbourBundleView.
     * @param interactionRadius Max pairwise distance.
     * @param fn                Interaction functor.
     * @param args              Additional forwarded arguments to fn.
     */
    struct InteractParticlesUnified
    {
        void operator()(IsNeighbourBundle auto&& neighbourBundle, auto interactionRadius, auto fn, auto&&... args)
            const
        {
            if(neighbourBundle.target().size == 0)
                return;

            auto sourceViews = neighbourBundle.makeDeviceViewTuple();

            ForEachFrameInPRBuf<32, 128>{}(
                neighbourBundle.target(),
                detail::UnifiedFrameInteractionKernel<detail::OccupiedSlot>{},
                std::move(sourceViews),
                interactionRadius,
                fn,
                std::forward<decltype(args)>(args)...);
        }
    };

} // namespace pmacc::spearhed
