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

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>
#include <optional>

namespace pmacc::spearhed
{
    namespace detail
    {
        /**
         * @brief Fills the frame index: for each region, records every frame's device address and
         *        owning region index into the flat, region-contiguous global slot layout.
         *
         * One block per region, master thread only: walking a singly-linked frame list is inherently
         * serial pointer chasing, so the only available parallelism is across regions. This performs
         * exactly ONE list traversal per region total -- precisely the walk the scan-based interaction
         * prologue used to repeat once per block (i.e. once per frame). The base offset of a region's
         * frames is the exclusive prefix sum, read from the inclusive-scan array @p framesScanBox.
         *
         * The kernel runs on the device, so @c &frame is a device heap address -- exactly what the
         * interaction kernels previously obtained by walking the list themselves. The host never
         * dereferences the stored pointers.
         */
        struct BuildFrameIndexKernel
        {
            template<typename RegionBox, typename ScanBox, typename FramePtrBox, typename RegionIdxBox>
            DINLINE constexpr auto operator()(
                auto const& worker,
                RegionBox prDeviceBox,
                int numRegions,
                ScanBox framesScanBox,
                FramePtrBox framePtrsBox,
                RegionIdxBox regionIdxBox) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= numRegions)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        // Exclusive prefix sum: base slot of this region within the flat layout.
                        uint32_t const offset = (blockIdx == 0) ? 0u : framesScanBox[blockIdx - 1];
                        auto& region = prDeviceBox[blockIdx];
                        auto& frameList = region.particleFrameList;

                        uint32_t i = 0;
                        for(auto it = frameList.begin(); it != frameList.end(); ++it)
                        {
                            int const slot = static_cast<int>(offset + i);
                            // Store the device heap address as an integer (see FrameIndexBuffer::FramePtr
                            // for why the box element is uintptr_t rather than a raw pointer).
                            framePtrsBox[slot] = reinterpret_cast<std::uintptr_t>(&*it);
                            regionIdxBox[slot] = static_cast<uint32_t>(blockIdx);
                            ++i;
                        }
                    });
            }
        };
    } // namespace detail

    /**
     * @brief Derived, rebuildable frame-pointer index over a ParticleRegionBuffer's frame lists.
     *
     * The singly-linked frame lists (mallocMC device heap) remain the OWNING structure: frames are
     * added/removed dynamically as particles migrate and are created. This index is a flat, derived
     * view that maps a global block index directly to (owning region, frame device pointer), so the
     * one-block-per-frame interaction kernels no longer walk the list per block (O(localFrameIdx)
     * std::advance) nor binary-search an inclusive scan (findFrameLocation) in their prologue.
     *
     * Layout: frames are laid out region-contiguously. For global slot g, @c regionIdxPerFrame[g] is
     * the owning region index and @c framePtrs[g] is that frame's device heap address. rebuild()
     * reuses detail::CountFramesKernel + inclusiveScanOnHost to compute the per-region base offsets,
     * then detail::BuildFrameIndexKernel populates both arrays with one list traversal per region.
     *
     * Invalidation contract:
     *   - The index is DERIVED from the frame lists. Allocating a new frame or removing ANY frame in
     *     the buffer (particle migration/creation that adds/frees frames, region add/remove) changes
     *     the set or ordering of frames and INVALIDATES the index -- rebuild() must be called before
     *     the next use.
     *   - Frame heap addresses are STABLE under mallocMC: allocated frames are never relocated or
     *     compacted. Therefore writing particle data into existing frames (positions, accumulators,
     *     even the live/dead multiMask of a slot) does NOT invalidate the index -- only the frame-list
     *     topology matters.
     *
     * @note Follow-up: the interaction host helpers rebuild the index once per operator() call. The
     *       index is not yet persisted across passes within a timestep; a caller that runs several
     *       interaction passes with no intervening frame-list mutation could build it once and reuse
     *       it across all of them. That lifetime-management is an explicit non-goal here.
     *
     * @tparam T_PRType The ParticleRegion type stored in the buffer (supplies FrameType).
     */
    template<typename T_PRType>
    struct FrameIndexBuffer
    {
        using FrameType = typename T_PRType::FrameType;
        //! Element type of framePtrs: the frame's device heap address as an integer. A raw @c FrameType*
        //! element cannot be used because PMacc's buffer set-value kernel (KernelSetValue) special-cases
        //! @c std::is_pointer_v element types to mean "the fill value is passed by pointer" and emits
        //! @c memBox(idx) = *value, which fails to type-check when the element itself is a pointer. A
        //! uintptr_t sidesteps that (it takes the trivially-copyable-by-value path, like uint32_t) and
        //! is reinterpret_cast back to @c FrameType* inside the interaction kernels. The host never
        //! dereferences these device addresses.
        using FramePtr = std::uintptr_t;

        //! Threads per block for the (master-only) count and fill kernels; matches CountFramesKernel.
        static constexpr uint32_t kIndexThreads = 32;

        //! Per global frame slot: device address of the frame. Sized frameCapacity.
        std::optional<pmacc::HostDeviceBuffer<FramePtr, DIM1>> framePtrs;
        //! Per global frame slot: owning region index. Sized frameCapacity.
        std::optional<pmacc::HostDeviceBuffer<uint32_t, DIM1>> regionIdxPerFrame;
        //! Count/scan scratch (one entry per region). Sized regionCapacity.
        std::optional<pmacc::HostDeviceBuffer<uint32_t, DIM1>> framesPerRegion;

        //! Number of live frames in the current build (valid frame slots are [0, totalFrames)).
        uint32_t totalFrames = 0;
        //! Allocated capacity of framePtrs / regionIdxPerFrame.
        uint32_t frameCapacity = 0;
        //! Allocated capacity of framesPerRegion.
        uint32_t regionCapacity = 0;

        //! Constructs an empty index and performs the initial build for @p prBuf.
        explicit FrameIndexBuffer(auto& prBuf)
        {
            rebuild(prBuf);
        }

        //! Device data box of frame pointers (valid entries [0, totalFrames)).
        auto framePtrsBox()
        {
            return framePtrs->getDeviceBuffer().getDataBox();
        }

        //! Device data box of owning region indices (valid entries [0, totalFrames)).
        auto regionIdxBox()
        {
            return regionIdxPerFrame->getDeviceBuffer().getDataBox();
        }

        /**
         * @brief (Re)builds the index from the current frame lists of @p prBuf.
         *
         * Must be called after any frame-list mutation (see the invalidation contract). Growth is
         * handled by constructing fresh HostDeviceBuffers of the new size (the codebase's resize
         * idiom, cf. ParticleRegionBuffer::create); the fill kernel overwrites every valid slot, so
         * previous contents need not be preserved.
         */
        void rebuild(auto& prBuf)
        {
            totalFrames = 0;
            if(prBuf.size == 0)
                return;

            ensureRegionCapacity(static_cast<uint32_t>(prBuf.size));

            // Count frames per region, then inclusive-scan on the host to get base offsets + total.
            PMACC_LOCKSTEP_KERNEL(detail::CountFramesKernel{})
                .template config<kIndexThreads>(pmacc::DataSpace<DIM1>(prBuf.size))(
                    prBuf.getDeviceDataBox(),
                    prBuf.size,
                    framesPerRegion->getDeviceBuffer().getDataBox());

            totalFrames = inclusiveScanOnHost(*framesPerRegion, prBuf.size);
            if(totalFrames == 0)
                return;

            ensureFrameCapacity(totalFrames);

            // One list traversal per region: record each frame's device address and owning region.
            PMACC_LOCKSTEP_KERNEL(detail::BuildFrameIndexKernel{})
                .template config<kIndexThreads>(pmacc::DataSpace<DIM1>(prBuf.size))(
                    prBuf.getDeviceDataBox(),
                    prBuf.size,
                    framesPerRegion->getDeviceBuffer().getDataBox(),
                    framePtrs->getDeviceBuffer().getDataBox(),
                    regionIdxPerFrame->getDeviceBuffer().getDataBox());
        }

    private:
        //! Grow framesPerRegion to hold at least @p needed regions (exact fit).
        void ensureRegionCapacity(uint32_t needed)
        {
            if(framesPerRegion && needed <= regionCapacity)
                return;
            regionCapacity = needed;
            framesPerRegion.emplace(pmacc::DataSpace<DIM1>{static_cast<int>(regionCapacity)});
        }

        //! Grow framePtrs / regionIdxPerFrame to hold at least @p needed frames, amortized doubling.
        void ensureFrameCapacity(uint32_t needed)
        {
            if(framePtrs && needed <= frameCapacity)
                return;
            uint32_t newCapacity = frameCapacity == 0 ? needed : frameCapacity;
            while(newCapacity < needed)
                newCapacity *= 2u;
            frameCapacity = newCapacity;
            framePtrs.emplace(pmacc::DataSpace<DIM1>{static_cast<int>(frameCapacity)});
            regionIdxPerFrame.emplace(pmacc::DataSpace<DIM1>{static_cast<int>(frameCapacity)});
        }
    };

} // namespace pmacc::spearhed
