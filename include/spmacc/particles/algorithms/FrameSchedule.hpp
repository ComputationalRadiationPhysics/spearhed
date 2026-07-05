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
         * @brief Forward cursor over every frame in a ParticleRegion device box, presented
         *        as one flat sequence (the regions' frame lists concatenated).
         *
         * operator++ follows the frame list's next pointer (O(1)); region boundaries -- and
         * empty regions -- are skipped transparently. Random access to a global frame index
         * is provided by makeFlatFrameCursor via the inclusive-scan index (one binary search
         * plus a bounded forward walk), so a contiguous traversal pays the seek once and then
         * advances with ++, whereas a strided traversal re-seeks per step.
         *
         * The cursor is a per-thread register object: every worker thread in a block holds its
         * own copy. There is no shared-memory hand-off, matching the FrameInteractionKernel
         * pattern, so no extra sync is needed to read the frame pointer.
         *
         * @tparam PRBox   ParticleRegion device data box type (indexable by region index).
         * @tparam ScanBox Inclusive prefix-sum data box type (frame counts per region).
         */
        template<typename PRBox, typename ScanBox>
        struct FlatFrameCursor
        {
            using Region = std::remove_reference_t<decltype(std::declval<PRBox&>()[0])>;
            using FrameListType = std::remove_cvref_t<decltype(std::declval<Region&>().particleFrameList)>;
            using FrameType = typename FrameListType::FrameType;
            using FrameIt = decltype(std::declval<Region&>().particleFrameList.begin());
            using FramePtr = pmacc::spearhed::memory::FramePointer<FrameType>;

            /**
             * @brief What dereferencing the cursor yields: the owning region (for volume and
             *        other region-level data) plus a pointer to the current frame.
             */
            struct FrameRef
            {
                using FrameType = FlatFrameCursor::FrameType;
                Region& region;
                FramePtr framePtr;
            };

            PRBox prBox;
            ScanBox scanBox;
            int numRegions;
            int regionIdx;
            int localFrameIdx;
            FrameIt frameIt;

            DINLINE FrameRef operator*()
            {
                Region& region = prBox[regionIdx];
                return FrameRef{region, FramePtr{&(*frameIt)}};
            }

            DINLINE FlatFrameCursor& operator++()
            {
                ++frameIt;
                ++localFrameIdx;
                // Roll forward over exhausted/empty regions until we land on a frame
                // or run past the last region (in which case the cursor is "end").
                while(regionIdx < numRegions && frameIt == prBox[regionIdx].particleFrameList.end())
                {
                    ++regionIdx;
                    localFrameIdx = 0;
                    if(regionIdx < numRegions)
                        frameIt = prBox[regionIdx].particleFrameList.begin();
                }
                return *this;
            }
        };

        /**
         * @brief Build a FlatFrameCursor positioned at a global frame index.
         *
         * Resolves the index to (region, localFrame) via findFrameLocation (binary search on
         * the inclusive-scan array) and walks the region's forward list to that frame.
         */
        template<typename PRBox, typename ScanBox>
        DINLINE auto makeFlatFrameCursor(PRBox prBox, ScanBox scanBox, int numRegions, int globalFrameIdx)
        {
            auto const loc = findFrameLocation(globalFrameIdx, scanBox, numRegions);
            auto frameIt = prBox[loc.regionIdx].particleFrameList.begin();
            for(int i = 0; i < loc.localFrameIdx; ++i)
                ++frameIt;
            return FlatFrameCursor<PRBox, ScanBox>{
                prBox,
                scanBox,
                numRegions,
                loc.regionIdx,
                loc.localFrameIdx,
                frameIt};
        }

        /**
         * @brief Device-side driver: invoke @p frameBody(FrameRef) for every frame this block
         *        owns under @p Schedule.
         *
         * Computes the total frame count from the scan array and hands a cursor factory to the
         * schedule, which decides the (start, stride, count) of frames for the calling block.
         * The body receives one frame at a time and is responsible for its own intra-frame work
         * (lockstep over slots) and any shared-memory synchronisation.
         *
         * @param schedule Frame-to-block mapping policy value (e.g. gridStride, contiguous, oneToOne).
         */
        DINLINE void forEachBlockFrame(
            auto const& worker,
            auto prBox,
            int numRegions,
            auto scanBox,
            auto schedule,
            auto frameBody)
        {
            int const total = static_cast<int>(scanBox[numRegions - 1]);
            if(total <= 0)
                return;
            schedule(
                worker,
                total,
                [&](int globalFrameIdx) { return makeFlatFrameCursor(prBox, scanBox, numRegions, globalFrameIdx); },
                frameBody);
        }
    } // namespace detail

    /**
     * @brief Frame-to-block mapping (schedule) policies.
     *
     * Each policy is the device-side half of an execution policy: given the launched grid
     * (worker.gridDomSize()) and the total frame count, it drives a cursor over the frames the
     * calling block must process. Orthogonal to the host-side grid-sizing policy in
     * FrameDispatch.hpp (OneBlockPerFrame, FixedGrid), subject to the constraint that OneToOne
     * is only valid when the grid was sized one-block-per-frame.
     */

    //! One block per frame: block i processes frame i. Requires a OneBlockPerFrame grid.
    struct OneToOne
    {
        DINLINE void operator()(auto const& worker, int total, auto makeCursor, auto frameBody) const
        {
            int const f = static_cast<int>(worker.blockDomIdx());
            if(f < total)
                frameBody(*makeCursor(f));
        }
    };

    //! Grid-stride: block i processes frames i, i + gridDim, i + 2*gridDim, ... Valid for any grid.
    //! Re-seeks the cursor per step, so it does not rely on the forward list's cheap ++.
    struct GridStride
    {
        DINLINE void operator()(auto const& worker, int total, auto makeCursor, auto frameBody) const
        {
            int const stride = static_cast<int>(worker.gridDomSize());
            for(int f = static_cast<int>(worker.blockDomIdx()); f < total; f += stride)
                frameBody(*makeCursor(f));
        }
    };

    //! Static contiguous chunks: block i processes frames [i*chunk, (i+1)*chunk). Seeks once to
    //! the chunk start, then advances with the forward list's O(1) ++. Valid for any grid.
    struct Contiguous
    {
        DINLINE void operator()(auto const& worker, int total, auto makeCursor, auto frameBody) const
        {
            int const numBlocks = static_cast<int>(worker.gridDomSize());
            int const chunk = (total + numBlocks - 1) / numBlocks;
            int const first = static_cast<int>(worker.blockDomIdx()) * chunk;
            int const last = (first + chunk < total) ? first + chunk : total;
            if(first >= last)
                return;
            auto cursor = makeCursor(first);
            for(int f = first; f < last; ++f)
            {
                frameBody(*cursor);
                ++cursor;
            }
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
                if(pred::occupied(particle))
                    body(particle);
            });
    }

    /**
     * @brief Bundled compile-time configuration for a frame-based for-each, as a single value object.
     *
     * The behavioural policies (schedule, grid) are stateless constexpr sub-objects. The thread
     * counts -- which must be compile-time constants for the lockstep kernel launch -- live in the
     * type as static constexpr members, so they survive being passed by value and are recovered at
     * the call site via decltype(cfg)::processThreads. This lets callers configure everything with
     * one constexpr value (see forEachConfig / defaultForEach) instead of template arguments.
     */
    template<typename T_Schedule, typename T_Grid, uint32_t T_ProcessThreads = 32, uint32_t T_CountThreads = 32>
    struct ForEachConfig
    {
        T_Schedule schedule;
        T_Grid grid;
        static constexpr uint32_t processThreads = T_ProcessThreads;
        static constexpr uint32_t countThreads = T_CountThreads;
    };

    /**
     * @brief Build a ForEachConfig from constexpr policy objects.
     *
     * @tparam T_ProcessThreads Threads per block for the per-frame processing kernel.
     * @tparam T_CountThreads   Threads per block for the frame-counting kernel.
     */
    template<uint32_t T_ProcessThreads = 32, uint32_t T_CountThreads = 32>
    constexpr auto forEachConfig(auto schedule, auto grid)
    {
        return ForEachConfig<decltype(schedule), decltype(grid), T_ProcessThreads, T_CountThreads>{schedule, grid};
    }

    //! Default configuration: grid-stride schedule over a one-block-per-frame grid (original behaviour).
    inline constexpr auto defaultForEach = forEachConfig(gridStride, oneBlockPerFrame);

} // namespace pmacc::spearhed
