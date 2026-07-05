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
#include "spmacc/particles/algorithms/FrameDispatch.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/algorithms/InteractionContext.hpp"
#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Variable.hpp>
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
         * @brief Single-launch GPU kernel that folds every source view into one kernel invocation,
         *        amortising the target-side work.
         *
         * The own-particle registers (position, ownReads, ownAccumulate, prepare result) are loaded
         * once, kept in per-virtual-worker lockstep context variables across all sources, and written
         * back to the target frame exactly once at the end. Each neighbour frame -- of every source --
         * is processed by the shared interactWithNeighbourFrame path (two barrier-separated phases:
         * per-slot staging with a sentinel for dead slots, then a compute sweep resolving the geometry
         * with a single reciprocal square root), so the physics, stage()/prepare() hooks and self-pair
         * semantics are identical to the per-source InteractParticles launch. Unlike the previous
         * design, no persistent own-particle SMEM is needed: registers replace it.
         *
         * @tparam ValidParticlePredicate  Predicate for the live-particle check.
         */
        template<typename ValidParticlePredicate>
        struct UnifiedFrameInteractionKernel
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto targetPRDeviceBox,
                auto framePtrsBox,
                auto regionIdxBox,
                uint32_t totalFrames,
                auto sourceViewTuple,
                auto interactionRadius,
                auto fn,
                auto... args) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= static_cast<int>(totalFrames))
                    return;

                // The prebuilt frame index maps this block straight to its (region, frame) -- no
                // per-block list walk (std::advance) and no findFrameLocation binary search.
                int const rIdx = static_cast<int>(regionIdxBox[blockIdx]);

                auto& region = targetPRDeviceBox[rIdx];
                auto& frameList = region.particleFrameList;
                using FrameType = typename std::remove_reference_t<decltype(frameList)>::FrameType;
                using VolumeType = typename std::remove_reference_t<decltype(region.volume)>;
                using RecordType = typename FrameType::ParticleRecord;
                using CS = typename VolumeType::Vec::CS;
                constexpr uint32_t frameSize = FrameType::frameSize;

                // Derive the SMEM cache and register records from the functor's declared tag sets.
                using FnType = std::remove_cvref_t<decltype(fn)>;
                using NeighbourReadsSet = std::remove_cvref_t<decltype(FnType::neighbourReads)>;
                using OwnReadsSet = std::remove_cvref_t<decltype(FnType::ownReads)>;
                using OwnAccumulateSet = std::remove_cvref_t<decltype(FnType::ownAccumulate)>;

                using NbRecord = nb_cache_record_t<FnType, RecordType, NeighbourReadsSet>;
                using PosRecord = position_record_t<RecordType>;
                using OwnReadsRecord = ll::sub_record_from_set_t<RecordType, OwnReadsSet>;
                using OwnAccumulateRecord = ll::sub_record_from_set_t<RecordType, OwnAccumulateSet>;

                using NbCacheType = ll::SoA<NbRecord, frameSize>;
                using PosCacheType = ll::SoA<PosRecord, frameSize>;
                using OwnReadsOne = ll::One<OwnReadsRecord>;
                using OwnAccumulateOne = ll::One<OwnAccumulateRecord>;
                using OwnReadsView = decltype(std::declval<OwnReadsOne&>()[uint32_t{0}]);

                constexpr bool hasPrepare = HasPrepareHook<FnType, OwnReadsView>;
                using PrepareType = typename PrepareResult<hasPrepare, FnType, OwnReadsView>::type;

                PMACC_SMEM(worker, nbCache, NbCacheType);
                PMACC_SMEM(worker, posCache, PosCacheType);

                memory::FramePointer const ownFramePtr{reinterpret_cast<FrameType*>(framePtrsBox[blockIdx])};
                VolumeType const ownVolume = region.volume;

                auto forEachSlot = pmacc::lockstep::makeForEach<frameSize>(worker);

                // Per-virtual-worker registers persisting across all sources (own SMEM is gone).
                auto ownRelVar = pmacc::lockstep::makeVar<Vec<CS, ValueStorage<CS>>>(forEachSlot);
                auto ownReadsVar = pmacc::lockstep::makeVar<OwnReadsOne>(forEachSlot);
                auto ownAccVar = pmacc::lockstep::makeVar<OwnAccumulateOne>(forEachSlot);
                auto validVar = pmacc::lockstep::makeVar<bool>(forEachSlot);
                auto prepVar = pmacc::lockstep::makeVar<PrepareType>(forEachSlot);

                auto const radius2 = static_cast<typename CS::T_Axis>(interactionRadius)
                                     * static_cast<typename CS::T_Axis>(interactionRadius);

                // --- Load own-particle registers once, then accumulate across all sources ---
                loadOwnRegisters<ValidParticlePredicate, hasPrepare>(
                    forEachSlot,
                    ownFramePtr,
                    fn,
                    ownRelVar,
                    ownReadsVar,
                    ownAccVar,
                    validVar,
                    prepVar);

                [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    auto processSource = [&](auto const& sourceView)
                    {
                        int const startNeighbour = sourceView.regionOffsetsBox[rIdx];
                        int const endNeighbour = sourceView.regionOffsetsBox[rIdx + 1];

                        for(int n = startNeighbour; n < endNeighbour; ++n)
                        {
                            int const neighbourRegionIdx = sourceView.neighbourRegionsBox[n];
                            auto& neighbourRegion = sourceView.sourcePRDeviceBox[neighbourRegionIdx];
                            auto& neighbourFrameList = neighbourRegion.particleFrameList;

                            for(auto it = neighbourFrameList.begin(); it != neighbourFrameList.end(); ++it)
                            {
                                memory::FramePointer const neighbourFramePtr{&*it};
                                VolumeType const neighbourVolume = neighbourRegion.volume;
                                // Equal frame pointers identify the frames as the same.
                                bool const isSelfFrame
                                    = (static_cast<void const*>(ownFramePtr.operator->())
                                       == static_cast<void const*>(neighbourFramePtr.operator->()));

                                interactWithNeighbourFrame<frameSize, ValidParticlePredicate, hasPrepare>(
                                    worker,
                                    forEachSlot,
                                    posCache,
                                    nbCache,
                                    ownVolume,
                                    neighbourFramePtr,
                                    neighbourVolume,
                                    isSelfFrame,
                                    radius2,
                                    fn,
                                    ownRelVar,
                                    ownReadsVar,
                                    ownAccVar,
                                    validVar,
                                    prepVar,
                                    args...);
                            }
                        }
                    };
                    (processSource(pmacc::memory::tuple::get<Is>(sourceViewTuple)), ...);
                }(std::make_index_sequence<
                    pmacc::memory::tuple::tuple_size_v<std::remove_cvref_t<decltype(sourceViewTuple)>>>{});

                // --- Write the accumulated state back to the target frame (once) ---
                storeOwnAccumulators<ValidParticlePredicate>(forEachSlot, ownFramePtr, validVar, ownAccVar);
            }
        };

    } // namespace detail

    /**
     * @brief Host helper that performs all pairwise interactions in a single kernel launch.
     *
     * Unlike InteractParticles (one launch per source entry), InteractParticlesUnified launches
     * exactly one kernel and compile-time iterates the source entries inside the kernel body. The
     * own-particle registers (position, ownReads, ownAccumulate, prepare result) persist across every
     * source; the accumulators are seeded once and flushed once. It uses the same functor contract as
     * InteractParticles: neighbourReads / ownReads / ownAccumulate sets plus the optional prepare()
     * and stage() hooks (see InteractParticles), and the call signature
     *   fn(worker, ownRead, [prep,] nb, PairContext<CS> const&, acc, args...),
     * where PairContext carries rVec, r2, r, invR (= rsqrt(r2)) and isSelf.
     *
     * @param neighbourBundle   NeighbourBundle or NeighbourBundleView.
     * @param interactionRadius Maximum pairwise interaction distance.
     * @param fn                Interaction functor (see InteractParticles for the contract).
     * @param args              Additional forwarded arguments to fn.
     */
    struct InteractParticlesUnified
    {
        void operator()(IsNeighbourBundle auto&& neighbourBundle, auto interactionRadius, auto fn, auto&&... args)
            const
        {
            auto& target = neighbourBundle.target();
            if(target.size == 0)
                return;

            auto sourceViews = neighbourBundle.makeDeviceViewTuple();

            // A single launch folds every source, so one index build serves the whole pass. The index
            // must outlive the launch, so it lives on the stack here and we synchronise once at the end.
            // TODO: persist the index across passes within a timestep (currently rebuilt per call).
            using TargetPRType = typename std::remove_reference_t<decltype(target)>::ParticleRegionType;
            FrameIndexBuffer<TargetPRType> index{target};

            launchForEachFrameInBlockIndexed(
                launchConfig<64>(oneBlockPerFrame),
                target,
                index,
                detail::UnifiedFrameInteractionKernel<pred::Occupied>{},
                std::move(sourceViews),
                interactionRadius,
                fn,
                std::forward<decltype(args)>(args)...);

            // Keep the index's device buffers alive until the launched kernel has finished.
            // TODO fix this
            pmacc::eventSystem::waitForAllTasks();
        }
    };

} // namespace pmacc::spearhed
