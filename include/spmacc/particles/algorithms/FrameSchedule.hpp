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
#include "spmacc/particles/algorithms/FrameDispatch.hpp"
#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    namespace detail
    {
        /**
         * @brief Device-side driver: invoke @p frameBody(FrameRef) for every frame this block
         *        owns under @p schedule, reading the block-to-frame mapping straight from a
         *        prebuilt FrameIndexBuffer's device boxes (see FrameIndex.hpp).
         *
         * Derives FrameType from the PRBox's region type (Region -> particleFrameList ->
         * FrameListType::FrameType) and builds a local frameAt(g) accessor that resolves global
         * frame slot g to a FrameRef{region, framePtr} in O(1) -- a direct index into
         * framePtrsBox / regionIdxBox, no list walk and no binary search. The schedule decides
         * which slots g the calling block visits; frameAt is handed to it as a factory.
         *
         * @param schedule Frame-to-block mapping policy value (e.g. gridStride, contiguous, oneToOne).
         */
        DINLINE void forEachBlockFrame(
            auto const& worker,
            auto prBox,
            auto framePtrsBox,
            auto regionIdxBox,
            int totalFrames,
            auto schedule,
            auto frameBody)
        {
            if(totalFrames <= 0)
                return;

            using Region = std::remove_reference_t<decltype(prBox[0])>;
            using FrameListType = std::remove_cvref_t<decltype(prBox[0].particleFrameList)>;
            using FrameType = typename FrameListType::FrameType;
            using FramePtr = pmacc::spearhed::memory::FramePointer<FrameType>;

            /**
             * @brief What frameAt(g) yields: the owning region (for volume and other
             *        region-level data) plus a pointer to the frame at global slot g.
             */
            struct FrameRef
            {
                using FrameType = typename FramePtr::type;
                Region& region;
                FramePtr framePtr;
            };

            auto frameAt = [&](int g)
            {
                int const rIdx = static_cast<int>(regionIdxBox[g]);
                Region& region = prBox[rIdx];
                return FrameRef{region, FramePtr{framePtrsBox[g]}};
            };

            schedule(worker, totalFrames, frameAt, frameBody);
        }
    } // namespace detail

    /**
     * @brief Frame-to-block mapping (schedule) policies.
     *
     * Each policy is the device-side half of an execution policy: given the launched grid
     * (worker.gridDomSize()) and the total frame count, it drives @p frameAt (an O(1) index into
     * the prebuilt FrameIndexBuffer's device boxes, see detail::forEachBlockFrame) over the frames
     * the calling block must process. Orthogonal to the host-side grid-sizing policy in
     * FrameDispatch.hpp (OneBlockPerFrame, FixedGrid), subject to the constraint that OneToOne
     * is only valid when the grid was sized one-block-per-frame.
     */

    //! One block per frame: block i processes frame i. Requires a OneBlockPerFrame grid.
    struct OneToOne
    {
        DINLINE void operator()(auto const& worker, int total, auto frameAt, auto frameBody) const
        {
            int const f = static_cast<int>(worker.blockDomIdx());
            if(f < total)
                frameBody(frameAt(f));
        }
    };

    //! Grid-stride: block i processes frames i, i + gridDim, i + 2*gridDim, ... Valid for any grid.
    struct GridStride
    {
        DINLINE void operator()(auto const& worker, int total, auto frameAt, auto frameBody) const
        {
            int const stride = static_cast<int>(worker.gridDomSize());
            for(int f = static_cast<int>(worker.blockDomIdx()); f < total; f += stride)
                frameBody(frameAt(f));
        }
    };

    //! Static contiguous chunks: block i processes frames [i*chunk, (i+1)*chunk). With the index,
    //! frameAt is O(1) per call just like GridStride's access, so Contiguous now differs from
    //! GridStride only in which frames land in the same block (contiguous slot range vs. strided),
    //! not in per-frame access cost -- there is no more cursor to reuse across the chunk. Valid for
    //! any grid.
    struct Contiguous
    {
        DINLINE void operator()(auto const& worker, int total, auto frameAt, auto frameBody) const
        {
            int const numBlocks = static_cast<int>(worker.gridDomSize());
            int const chunk = (total + numBlocks - 1) / numBlocks;
            int const first = static_cast<int>(worker.blockDomIdx()) * chunk;
            int const last = (first + chunk < total) ? first + chunk : total;
            for(int f = first; f < last; ++f)
                frameBody(frameAt(f));
        }
    };

    //! Named schedule policy instances for value-based (constexpr object) configuration.
    inline constexpr OneToOne oneToOne{};
    inline constexpr GridStride gridStride{};
    inline constexpr Contiguous contiguous{};

    /**
     * @brief Distribute one frame's live slots across the block (lockstep). The slot-to-thread
     *        distribution counterpart to the sequential particle leaf of the hierarchy iterator
     *        (see HierarchyForEach.hpp); callers compose it with the iterator, the iterator never
     *        invokes it.
     *
     * @tparam T_Frame Any frame handle with a static frameSize and operator[](slot) (e.g. FrameView).
     * @param worker The alpaka worker of the enclosing lockstep kernel.
     * @param frame  The frame whose live slots are distributed.
     * @param body   Invoked as body(particle) for each live particle in the frame.
     */
    template<typename T_Frame>
    DINLINE void lockstepForEachParticle(auto const& worker, T_Frame frame, auto body)
    {
        pmacc::lockstep::makeForEach<std::remove_cvref_t<T_Frame>::frameSize>(worker)(
            [&](uint32_t const slot)
            {
                auto particle = frame[slot];
                // Construct the predicate: odr-using the namespace-scope constexpr instance
                // (pred::occupied) from device code is ill-formed under nvcc.
                if(pred::Occupied{}(particle))
                    body(particle);
            });
    }

    /**
     * @brief Bundled compile-time configuration for a frame-based for-each, as a single value object.
     *
     * The behavioural policies (schedule, grid) are stateless constexpr sub-objects. The thread
     * count -- which must be a compile-time constant for the lockstep kernel launch -- lives in the
     * type as a static constexpr member, so it survives being passed by value and is recovered at
     * the call site via decltype(cfg)::processThreads. This lets callers configure everything with
     * one constexpr value (see forEachConfig / defaultForEach) instead of template arguments. There
     * is no count-kernel thread count here: the index (FrameIndexBuffer, see FrameIndex.hpp) owns
     * its own kernel thread count for its (re)build.
     */
    template<typename T_Schedule, typename T_Grid, uint32_t T_ProcessThreads = 32>
    struct ForEachConfig
    {
        T_Schedule schedule;
        T_Grid grid;
        static constexpr uint32_t processThreads = T_ProcessThreads;
    };

    /**
     * @brief Build a ForEachConfig from constexpr policy objects.
     *
     * @tparam T_ProcessThreads Threads per block for the per-frame processing kernel.
     */
    template<uint32_t T_ProcessThreads = 32>
    constexpr auto forEachConfig(auto schedule, auto grid)
    {
        return ForEachConfig<decltype(schedule), decltype(grid), T_ProcessThreads>{schedule, grid};
    }

    //! Default configuration: grid-stride schedule over a one-block-per-frame grid (original behaviour).
    inline constexpr auto defaultForEach = forEachConfig(gridStride, oneBlockPerFrame);

} // namespace pmacc::spearhed
