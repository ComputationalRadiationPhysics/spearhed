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
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/NeighbourEntry.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/eventSystem/Manager.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Variable.hpp>
#include <pmacc/math/functions/Root.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<typename T>
        using ElementView = decltype((*static_cast<T*>(nullptr))[uint32_t{0}]);

        //! Empty placeholder used as the register type when a functor has no prepare() hook.
        struct NoPrepare
        {
        };

        //! True if @p Fn exposes prepare(ownReadsView) for the given own-reads view type.
        template<typename Fn, typename OwnReadsView>
        concept HasPrepareHook = requires(Fn const& fn, OwnReadsView view) { fn.prepare(view); };

        /**
         * @brief True if @p Fn opts into the neighbour-side stage() hook.
         *
         * The symmetric twin of prepare(): a functor that declares a member type @c StagedRecord
         * (an ll::Record of derived neighbour quantities) transforms each neighbour's global
         * attributes into that record once, at staging time, instead of once per pair. It must
         * also provide a matching stage(neighbourParticleView, stagedView) method. Detection keys
         * on the member type; a missing stage() then produces a regular template error (mirroring
         * the deliberately unpinned vector gradW).
         */
        template<typename Fn>
        concept HasStageHook = requires { typename Fn::StagedRecord; };

        //! Resolves the per-particle register type produced by prepare(); NoPrepare when absent.
        template<bool HasPrepare, typename Fn, typename OwnReadsView>
        struct PrepareResult
        {
            using type = NoPrepare;
        };

        template<typename Fn, typename OwnReadsView>
        struct PrepareResult<true, Fn, OwnReadsView>
        {
            using type = std::decay_t<decltype(std::declval<Fn const&>().prepare(std::declval<OwnReadsView>()))>;
        };

        /**
         * @brief Load the own-particle register state for one target frame, once per pass.
         *
         * For every live slot this materialises, into stable per-virtual-worker lockstep context
         * variables that persist across the whole neighbour sweep:
         *   - relativePos (own-region coordinates) as a value Vec,
         *   - the functor's ownReads attributes (const registers),
         *   - the functor's ownAccumulate attributes, seeded from the particle's CURRENT values
         *     (resume/RMW-once semantics, preserving a separate seeding pre-pass),
         *   - the optional prepare() result.
         * The liveness flag is recorded so dead slots are skipped without re-reading global memory.
         */
        template<typename ValidParticlePredicate, bool HasPrepare>
        DINLINE void loadOwnRegisters(
            auto const& forEachSlot,
            auto ownFramePtr,
            auto& fn,
            auto& ownRelVar,
            auto& ownReadsVar,
            auto& ownAccVar,
            auto& validVar,
            auto& prepVar)
        {
            forEachSlot(
                [&](pmacc::lockstep::Idx const idx)
                {
                    uint32_t const slot = idx;
                    auto ownParticle = ownFramePtr[slot];
                    bool const valid = ValidParticlePredicate{}(ownParticle);
                    validVar[idx] = valid;
                    if(!valid)
                        return;

                    // Const registers and resumed accumulators (copies only the requested fields).
                    ownReadsVar[idx] = ownParticle;
                    ownAccVar[idx] = ownParticle;
                    // Materialise the own position so the sweep needs no global re-read.
                    ownRelVar[idx] = ownParticle[tags::relativePos].get();

                    if constexpr(HasPrepare)
                        prepVar[idx] = fn.prepare(ownReadsVar[idx][uint32_t{0}]);
                });
        }

        /**
         * @brief Write the accumulated own-particle state back to the target frame, once per pass.
         *
         * Flushes only the ownAccumulate sub-record; all other fields are untouched.
         */
        template<typename ValidParticlePredicate>
        DINLINE void storeOwnAccumulators(auto const& forEachSlot, auto ownFramePtr, auto& validVar, auto& ownAccVar)
        {
            forEachSlot(
                [&](pmacc::lockstep::Idx const idx)
                {
                    if(!validVar[idx])
                        return;
                    uint32_t const slot = idx;
                    auto ownParticle = ownFramePtr[slot];
                    ownAccVar[idx][uint32_t{0}].deepCopyTo(ownParticle);
                });
        }

        /**
         * @brief Interact every live own particle against one neighbour frame.
         *
         * Two cooperative phases, each closed by a barrier so the shared caches can be reused for
         * the next frame (there is no serial compaction pass):
         *   1. Staging: each worker stages its OWN slot @c s directly at cache index @c s -- no
         *      compaction, so cacheIndex == slot throughout. For a live neighbour it stages the
         *      functor-facing attributes (either the raw @p neighbourReads sub-record, or, when the
         *      functor opts into the stage() hook, a derived StagedRecord computed once here) plus
         *      the pre-shifted position (rel_j + originShift) into a geometry SoA, so the inner loop
         *      reconstructs no coordinates. For a dead neighbour it writes a large finite sentinel
         *      position; its neighbour cache stays garbage (never read past the cull).
         *   2. Compute: each worker sweeps the full [0, frameSize) range, culls on squared distance,
         *      and for accepted pairs computes a single reciprocal square root (invR = rsqrt(r2),
         *      r = r2 * invR) before calling the functor with SMEM views. The dead-slot sentinel
         *      makes d*d overflow to +inf, so r2 < radius2 is false and the slot is culled for free.
         *      The self pair is identified purely by index: isSelfFrame && (j == mySlot), where
         *      mySlot is the worker's own slot (staging is 1:1, so the own particle sits at its own
         *      index on the self frame). All accumulation lands in the register accumulator.
         *
         * @tparam frameSize             Number of slots per frame (compile-time loop bound).
         * @tparam ValidParticlePredicate Live-particle predicate.
         * @tparam HasPrepare            Whether the functor exposes prepare().
         */
        template<uint32_t frameSize, typename ValidParticlePredicate, bool HasPrepare>
        DINLINE void interactWithNeighbourFrame(
            auto const& worker,
            auto const& forEachSlot,
            auto& posCache,
            auto& nbCache,
            auto const& ownVolume,
            auto neighbourFramePtr,
            auto const& neighbourVolume,
            bool isSelfFrame,
            auto radius2,
            auto& fn,
            auto& ownRelVar,
            auto& ownReadsVar,
            auto& ownAccVar,
            auto& validVar,
            auto& prepVar,
            auto&... args)
        {
            using CS = typename std::remove_cvref_t<decltype(ownVolume)>::Vec::CS;
            using Axis = typename CS::T_Axis;
            using FnType = std::remove_cvref_t<decltype(fn)>;
            constexpr bool hasStage = HasStageHook<FnType>;

            // Phase 1: cooperatively stage neighbours (one slot per worker, no compaction)
            // Origin offset baked into every staged position so the inner loop skips getPosition().
            Vec<CS, ValueStorage<CS>> const originShift = neighbourVolume.origin - ownVolume.origin;
            // Large finite sentinel: sentinel*sentinel overflows to +inf so the cull rejects it.
            constexpr Axis sentinel = std::numeric_limits<Axis>::max() / Axis{4};
            forEachSlot(
                [&](pmacc::lockstep::Idx const idx)
                {
                    uint32_t const slot = idx;
                    auto nParticle = neighbourFramePtr[slot];
                    if(ValidParticlePredicate{}(nParticle))
                    {
                        // Functor-facing neighbour attributes: derived StagedRecord or raw sub-record.
                        if constexpr(hasStage)
                            fn.stage(nParticle, nbCache[slot]);
                        else
                            nbCache[slot].deepCopyFrom(nParticle);
                        // Pre-shifted geometry: shiftedPos = rel_j + (origin_neigh - origin_own).
                        auto const relView = nParticle[tags::relativePos].get();
                        pmacc::spearhed::for_each_tag<CS>(
                            [&](auto tag)
                            { posCache[slot][tags::relativePos][tag] = relView[tag] + originShift[tag]; });
                    }
                    else
                    {
                        // Dead slot: sentinel position; nbCache[slot] left garbage (never read).
                        pmacc::spearhed::for_each_tag<CS>([&](auto tag)
                                                          { posCache[slot][tags::relativePos][tag] = sentinel; });
                    }
                });
            worker.sync();

            // Phase 2: sweep the staged neighbours and accumulate into registers
            forEachSlot(
                [&](pmacc::lockstep::Idx const idx)
                {
                    if(!validVar[idx])
                        return;

                    uint32_t const mySlot = idx;
                    auto const& ownRel = ownRelVar[idx];
                    auto ownReadsView = ownReadsVar[idx][uint32_t{0}];
                    auto ownAccView = ownAccVar[idx][uint32_t{0}];

                    for(uint32_t j = 0; j < frameSize; ++j)
                    {
                        Vec<CS, ValueStorage<CS>> rVec;
                        Axis r2{0};
                        pmacc::spearhed::for_each_tag<CS>(
                            [&](auto tag)
                            {
                                Axis const d = ownRel[tag] - posCache[j][tags::relativePos][tag];
                                rVec[tag] = d;
                                r2 += d * d;
                            });

                        if(r2 < radius2)
                        {
                            Axis const invR = pmacc::math::rsqrt(r2);
                            Axis const r = r2 * invR;
                            bool const isSelf = isSelfFrame && (j == mySlot);
                            auto nbView = nbCache[j];
                            PairContext<CS> const ctx{rVec, r2, r, invR, isSelf};

                            if constexpr(HasPrepare)
                                fn(worker, ownReadsView, prepVar[idx], nbView, ctx, ownAccView, args...);
                            else
                                fn(worker, ownReadsView, nbView, ctx, ownAccView, args...);
                        }
                    }
                });
            worker.sync();
        }

        /**
         * @brief Compile-time helpers deriving the per-role SMEM/register records from the functor.
         */
        template<typename Record, typename Set>
        using neighbour_reads_record_t = ll::sub_record_from_set_t<Record, Set>;

        /**
         * @brief Record type staged into the neighbour SMEM cache.
         *
         * Without the stage() hook this is the sub-record of the functor's @c neighbourReads global
         * attributes (staged verbatim via deepCopyFrom). With the hook the functor's own
         * @c StagedRecord of derived quantities is used instead, populated by stage(). Written once
         * as a shared trait so FrameInteractionKernel and UnifiedFrameInteractionKernel stay in
         * lockstep.
         */
        template<typename Fn, typename Record, typename NeighbourReadsSet, bool = HasStageHook<Fn>>
        struct NbCacheRecord
        {
            using type = neighbour_reads_record_t<Record, NeighbourReadsSet>;
        };

        template<typename Fn, typename Record, typename NeighbourReadsSet>
        struct NbCacheRecord<Fn, Record, NeighbourReadsSet, true>
        {
            using type = typename Fn::StagedRecord;
        };

        template<typename Fn, typename Record, typename NeighbourReadsSet>
        using nb_cache_record_t = typename NbCacheRecord<Fn, Record, NeighbourReadsSet>::type;

        template<typename Record>
        using position_record_t = ll::sub_record_from_set_t<Record, decltype(ll::makeSet(tags::relativePos))>;

        // self interaction must be dealt with by the user in interact Fn
        template<typename ValidParticlePredicate>
        struct FrameInteractionKernel
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto targetPRDeviceBox,
                auto framePtrsBox,
                auto regionIdxBox,
                uint32_t totalFrames,
                auto sourceView,
                auto interactionRadius,
                auto fn,
                auto... args) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= static_cast<int>(totalFrames))
                    return;

                // The prebuilt frame index maps this block straight to its (region, frame) -- no
                // per-block list walk (std::advance) and no inclusive-scan binary search.
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
                using OwnReadsView = detail::ElementView<OwnReadsOne>;

                constexpr bool hasPrepare = HasPrepareHook<FnType, OwnReadsView>;
                using PrepareType = typename PrepareResult<hasPrepare, FnType, OwnReadsView>::type;

                PMACC_SMEM(worker, nbCache, NbCacheType);
                PMACC_SMEM(worker, posCache, PosCacheType);

                memory::FramePointer const ownFramePtr{framePtrsBox[blockIdx]};
                VolumeType const ownVolume = region.volume;

                auto forEachSlot = pmacc::lockstep::makeForEach<frameSize>(worker);

                // Per-virtual-worker registers that persist across the whole neighbour sweep.
                auto ownRelVar = pmacc::lockstep::makeVar<Vec<CS, ValueStorage<CS>>>(forEachSlot);
                auto ownReadsVar = pmacc::lockstep::makeVar<OwnReadsOne>(forEachSlot);
                auto ownAccVar = pmacc::lockstep::makeVar<OwnAccumulateOne>(forEachSlot);
                auto validVar = pmacc::lockstep::makeVar<bool>(forEachSlot);
                auto prepVar = pmacc::lockstep::makeVar<PrepareType>(forEachSlot);

                auto const radius2 = static_cast<typename CS::T_Axis>(interactionRadius)
                                     * static_cast<typename CS::T_Axis>(interactionRadius);

                loadOwnRegisters<ValidParticlePredicate, hasPrepare>(
                    forEachSlot,
                    ownFramePtr,
                    fn,
                    ownRelVar,
                    ownReadsVar,
                    ownAccVar,
                    validVar,
                    prepVar);

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
                        // Each live frame is a unique heap allocation belonging to exactly one
                        // region of one buffer, so equal frame pointers already identify the same
                        // frame (same region, same buffer).
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

                storeOwnAccumulators<ValidParticlePredicate>(forEachSlot, ownFramePtr, validVar, ownAccVar);
            }
        };

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
         * semantics are identical to the per-source FrameInteractionKernel launch.
         * No persistent own-particle SMEM is needed, registers replace it.
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
                // per-block list walk (std::advance) and no inclusive-scan binary search.
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
                using OwnReadsView = detail::ElementView<OwnReadsOne>;
                ;

                constexpr bool hasPrepare = HasPrepareHook<FnType, OwnReadsView>;
                using PrepareType = typename PrepareResult<hasPrepare, FnType, OwnReadsView>::type;

                PMACC_SMEM(worker, nbCache, NbCacheType);
                PMACC_SMEM(worker, posCache, PosCacheType);

                memory::FramePointer const ownFramePtr{framePtrsBox[blockIdx]};
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

                // Load own-particle registers once, then accumulate across all sources
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

                // Write the accumulated state back to the target frame (once)
                storeOwnAccumulators<ValidParticlePredicate>(forEachSlot, ownFramePtr, validVar, ownAccVar);
            }
        };
    } // namespace detail

} // namespace pmacc::spearhed
