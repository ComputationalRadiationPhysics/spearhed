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

#include "spmacc/particles/algorithms/FrameDispatch.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/algorithms/FrameSchedule.hpp"
#include "spmacc/particles/algorithms/HierarchyForEach.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/eventSystem/events/EventTask.hpp>

#include <type_traits>
#include <utility>

/*
 * Launch distribution for the particle hierarchy: the host entry that dispatches a GPU kernel.
 *
 * The hierarchy iterator (HierarchyForEach.hpp) is sequential for the calling thread; choosing
 * *which* block processes *which* frame is a launch-time decision, so the "better algorithm" --
 * index-driven block-per-frame dispatch (see FrameIndex.hpp / FrameDispatch.hpp), grid-stride,
 * contiguous chunks -- lives here, expressed with the same value-based policies as
 * FrameDispatch.hpp / FrameSchedule.hpp.
 *
 * The layers compose: the launch policy hands each block a frame; the frame -> particle descent is
 * the explicit lockstepForEachParticle slot combinator. Only levels::particle and levels::frame are
 * supported, since both map naturally onto a per-frame grid -- region/species sweeps are a
 * whole-tree concern handled by forEach with deviceHeap (in-kernel) / hostHeap (host-side).
 */

namespace pmacc::spearhed
{
    namespace detail
    {
        //! Per-block kernel: map this block's frame(s) via @p schedule -- reading the block-to-frame
        //! mapping straight from a prebuilt FrameIndexBuffer's device boxes -- then descend to T_Target.
        template<typename T_Target>
        struct HierarchyForEachKernel
        {
            DINLINE void operator()(
                auto const& worker,
                auto prDeviceBox,
                auto framePtrsBox,
                auto regionIdxBox,
                uint32_t totalFrames,
                auto schedule,
                auto body,
                auto... args) const
            {
                forEachBlockFrame(
                    worker,
                    prDeviceBox,
                    framePtrsBox,
                    regionIdxBox,
                    static_cast<int>(totalFrames),
                    schedule,
                    [&](auto frameRef)
                    {
                        FrameView const frameView{frameRef.framePtr};
                        if constexpr(T_Target::rank == levels::Frame::rank)
                            // Frame target: hand the whole frame to the body once per frame. The
                            // body owns intra-frame parallelism, hence the worker.
                            body(worker, frameView, args...);
                        else
                            // Particle target: distribute the frame's live slots across the block,
                            // one call per live particle. The worker is forwarded so the body can
                            // do atomics / reductions (there is no ambient worker to capture on
                            // the host side).
                            lockstepForEachParticle(
                                worker,
                                frameView,
                                [&](auto particle) { body(worker, particle, args...); });
                    });
            }
        };
    } // namespace detail

    /**
     * @brief Host entry point: launch a GPU kernel iterating every element at level @p target, using
     *        the frame decomposition selected by @p cfg and a caller-built FrameIndexBuffer @p index.
     *
     * The decomposition ("which block gets which frame") is a swappable value object, while the
     * per-element work composes the hierarchy iterator with the lockstep slot combinator.
     *
     * Asynchronous: does NOT synchronise -- the kernel is merely enqueued and this function returns
     * immediately. Every device allocation reachable from the launch -- @p prBuf, @p index, and any
     * buffer viewed by @p args -- must outlive kernel COMPLETION, not just this call (PMacc buffer
     * destructors do not wait for in-flight kernels). The caller synchronises via the returned event
     * at its natural sync point (see launchForEachFrameInBlockIndexed in FrameDispatch.hpp).
     *
     * @param cfg    A constexpr ForEachConfig (schedule + grid + thread count); see forEachConfig /
     *               defaultForEach in FrameSchedule.hpp. This is the "better algorithm" knob, e.g.
     *               forEachConfig(contiguous, fixedGrid<2048>).
     * @param prBuf  The ParticleRegionBuffer to traverse.
     * @param index  A rebuilt FrameIndexBuffer for @p prBuf (see FrameIndex.hpp); share one index
     *               across several passes with no intervening frame-list mutation.
     * @param body   levels::particle -> body(worker, particle, args...): one live particle at a time.
     *               levels::frame -> body(worker, frameView, args...): one frame at a time. Either way
     *               the worker is supplied so the body can run atomics / reductions or its own
     *               intra-frame lockstep + shared memory.
     * @param args   Extra kernel arguments forwarded by value to @p body. Prefer this over lambda
     *               capture for device data boxes: a captured box is const inside the (const)
     *               kernel body, whereas a forwarded argument arrives as a mutable parameter.
     * @return EventTask for the enqueued kernel. Wait on it with waitForFinished() before destroying
     *         any buffer the kernel touches.
     */
    template<
        typename T_Cfg,
        typename T_Target,
        typename T_PRBuf,
        typename T_Index,
        typename T_Body,
        typename... T_Args>
    requires(!IsHierarchyLevel<T_Cfg> && IsFrameIndexBuffer<T_Index>)
    [[nodiscard]] pmacc::EventTask launchForEach(
        T_Cfg cfg,
        T_Target /*target*/,
        T_PRBuf& prBuf,
        T_Index& index,
        T_Body body,
        T_Args&&... args)
    {
        static_assert(
            T_Target::rank <= levels::Frame::rank,
            "launchForEach maps blocks onto frames, so it only supports levels::particle or "
            "levels::frame; use forEach with deviceHeap (in-kernel) or hostHeap (host-side) for "
            "region or species traversal.");
        return launchForEachFrameInBlockIndexed(
            launchConfig<T_Cfg::processThreads>(cfg.grid),
            prBuf,
            index,
            detail::HierarchyForEachKernel<T_Target>{},
            cfg.schedule,
            body,
            std::forward<T_Args>(args)...);
    }

    //! Convenience overload using the default decomposition (grid-stride over one block per frame),
    //! with a caller-built FrameIndexBuffer. Same asynchrony and lifetime contract as above.
    template<IsHierarchyLevel T_Target, typename T_PRBuf, typename T_Index, typename T_Body, typename... T_Args>
    requires IsFrameIndexBuffer<T_Index>
    [[nodiscard]] pmacc::EventTask launchForEach(
        T_Target target,
        T_PRBuf& prBuf,
        T_Index& index,
        T_Body body,
        T_Args&&... args)
    {
        return launchForEach(defaultForEach, target, prBuf, index, body, std::forward<T_Args>(args)...);
    }

    /**
     * @brief Host entry point (synchronous): as the indexed overload above, but builds and discards
     *        a temporary FrameIndexBuffer for @p prBuf instead of taking a caller-built one.
     *
     * A fresh index build costs exactly what the old scan-driven dispatch's count-kernel + host
     * inclusive-scan prologue paid on every call, so routing through here is not a regression versus
     * that pre-index code path. Callers that run several passes with no intervening frame-list
     * mutation should build one FrameIndexBuffer explicitly and use the index-taking overloads above
     * instead, to pay that cost once (see Simulation::stepWithKernel).
     *
     * The index is local to this call and is destroyed at scope exit, so waiting for the launch here
     * is mandatory, not a style choice: without it, the index's device buffers -- which the
     * still-in-flight kernel reads via framePtrsBox/regionIdxBox -- would be freed while the kernel
     * is still running.
     */
    template<typename T_Cfg, typename T_Target, typename T_PRBuf, typename T_Body, typename... T_Args>
    requires(!IsHierarchyLevel<T_Cfg>)
    void launchForEach(T_Cfg cfg, T_Target target, T_PRBuf& prBuf, T_Body body, T_Args&&... args)
    {
        if(prBuf.size == 0)
            return;
        FrameIndexBuffer<typename std::remove_reference_t<decltype(prBuf)>::ParticleRegionType> index{prBuf};
        // Mandatory: index dies at the end of this scope, so the kernel must finish before that.
        launchForEach(cfg, target, prBuf, index, body, std::forward<T_Args>(args)...).waitForFinished();
    }

    //! Convenience overload using the default decomposition (grid-stride over one block per frame)
    //! and a freshly built, discarded FrameIndexBuffer. Synchronous (see the cfg overload above).
    template<IsHierarchyLevel T_Target, typename T_PRBuf, typename T_Body, typename... T_Args>
    void launchForEach(T_Target target, T_PRBuf& prBuf, T_Body body, T_Args&&... args)
    {
        launchForEach(defaultForEach, target, prBuf, body, std::forward<T_Args>(args)...);
    }

    //! Multi-species launch: one kernel per species (a host-side fold over the buffers). Each
    //! species is an independent launch, so @p body must be generic over the differing record types.
    template<typename T_Cfg, typename T_Target, typename T_Body, typename... T_PRBuf>
    void launchForEachAllSpecies(T_Cfg cfg, T_Target target, T_Body body, T_PRBuf&... prBufs)
    {
        (launchForEach(cfg, target, prBufs, body), ...);
    }

} // namespace pmacc::spearhed
