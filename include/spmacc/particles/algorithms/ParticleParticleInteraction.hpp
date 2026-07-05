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
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Variable.hpp>
#include <pmacc/math/functions/Root.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
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

            // --- Phase 1: cooperatively stage neighbours (one slot per worker, no compaction) ---
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
                            { *posCache[slot][tags::relativePos][tag] = relView[tag] + originShift[tag]; });
                    }
                    else
                    {
                        // Dead slot: sentinel position; nbCache[slot] left garbage (never read).
                        pmacc::spearhed::for_each_tag<CS>([&](auto tag)
                                                          { *posCache[slot][tags::relativePos][tag] = sentinel; });
                    }
                });
            worker.sync();

            // --- Phase 2: sweep the staged neighbours and accumulate into registers ---
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
                                Axis const d = ownRel[tag] - *posCache[j][tags::relativePos][tag];
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
    } // namespace detail

    /**
     * @brief Host helper: launch pairwise interactions using the precalculated neighbour lists.
     *
     * Launches one kernel pass per source entry in the bundle. Each pass loads the target frame's
     * own state once, sweeps every neighbour frame of that source, and writes the accumulators back
     * once, so a target attribute is read-modify-written exactly once per source pass.
     *
     * Per neighbour frame the kernel runs two barrier-separated phases (no serial compaction):
     *   1. Staging -- each worker stages its own slot verbatim (cacheIndex == slot). Live slots get
     *      their functor-facing neighbour attributes plus a pre-shifted position; dead slots get a
     *      large finite sentinel position whose squared distance overflows to +inf, so the radius
     *      cull rejects them for free.
     *   2. Compute -- each worker sweeps the full frame, culls on squared distance, and for accepted
     *      pairs resolves the geometry with a single reciprocal square root before invoking @p fn.
     * The self pair is detected by index (own particle sits at its own slot on the self frame).
     *
     * The functor receives a PairContext<CS> carrying rVec, r2, r, invR (= 1/r via rsqrt) and
     * isSelf. In the coincident r2 == 0 case invR is +inf and r is NaN, so functors must guard it
     * (self pairs are likewise expected to be short-circuited on ctx.isSelf).
     *
     * @param neighbourBundle  NeighbourBundle (or NeighbourBundleView) holding the target buffer and
     *                         the neighbour lists for each source. Call bundle.byRole(role) (or
     *                         bySpecies(species)) to restrict which sources are iterated.
     * @param interactionRadius Maximum pairwise interaction distance.
     * @param fn Interaction functor for one @c ownParticle / @c neighbourParticle pair. It must
     *           declare three attribute sets and may expose two optional hooks:
     *             static constexpr auto neighbourReads = ll::makeSet(...); // neighbour read contract:
     *                 the global fields stage() / the default staging path may read
     *             static constexpr auto ownReads       = ll::makeSet(...); // const own registers
     *             static constexpr auto ownAccumulate  = ll::makeSet(...); // register accumulators,
     *                 seeded from the current attribute values and written back once per pass
     *             // optional, detected via requires-expressions:
     *             DINLINE auto prepare(ownReadsView) const;                // once per own particle
     *             using StagedRecord = ll::Record<...>;                    // derived neighbour record
     *             DINLINE void stage(neighbourParticleView, stagedView) const; // once per staged nb
     *           When StagedRecord is present the neighbour SMEM cache holds that derived record and
     *           stage() populates it once per staged neighbour (so per-pair neighbour work is
     *           hoisted, the symmetric twin of prepare()); otherwise the cache holds the raw
     *           @c neighbourReads sub-record staged verbatim, and the functor sees ONLY those fields.
     *           The call operator has the signature
     *             fn(worker, ownRead, [prep,] nb, PairContext<CS> const&, acc, args...)
     *           where @c prep is present only when prepare() exists, @c nb is a view of StagedRecord
     *           (with the hook) or of the neighbourReads sub-record (without), @c ownRead / @c acc are
     *           register views restricted to the corresponding set, and accumulation goes into
     *           @c acc. Contributions must be additive so per-source passes compose.
     */
    struct InteractParticles
    {
        void operator()(IsNeighbourBundle auto&& neighbourBundle, auto interactionRadius, auto fn, auto&&... args)
            const
        {
            auto& target = neighbourBundle.target();
            if(target.size == 0)
                return;

            // Build the frame index ONCE for the target buffer: InteractParticles launches one kernel
            // per source, so a single build amortises over every source pass. The index must outlive
            // all those launches, so it lives on the stack here and we synchronise once at the end.
            // TODO: persist the index across passes within a timestep (currently rebuilt per call).
            using TargetPRType = typename std::remove_reference_t<decltype(target)>::ParticleRegionType;
            FrameIndexBuffer<TargetPRType> index{target};

            neighbourBundle.forEachDeviceView(
                [&](auto sourceView)
                {
                    launchForEachFrameInBlockIndexed(
                        launchConfig<64>(oneBlockPerFrame),
                        target,
                        index,
                        detail::FrameInteractionKernel<pred::Occupied>{},
                        sourceView,
                        interactionRadius,
                        fn,
                        std::forward<decltype(args)>(args)...);
                });

            // Keep the index's device buffers alive until every launched kernel has finished.
            // TODO fix this
            pmacc::eventSystem::waitForAllTasks();
        }
    };

} // namespace pmacc::spearhed
