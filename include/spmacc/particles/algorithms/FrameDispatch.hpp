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

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/eventSystem/waitForAllTasks.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <cstdint>

namespace pmacc::spearhed
{
    namespace detail
    {
        /**
         * Computes frame counts per region.
         * One block per region; only the master thread writes.
         */
        struct CountFramesKernel
        {
            template<typename RegionBox, typename ScanBox>
            DINLINE constexpr auto operator()(
                auto const& worker,
                RegionBox prDeviceBox,
                int numRegions,
                ScanBox framesPerRegion) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= numRegions)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        auto& region = prDeviceBox[blockIdx];
                        auto& frameList = region.particleFrameList;
                        framesPerRegion[blockIdx] = frameList.size();
                    });
            }
        };

        /**
         * Result of mapping a block index to a frame within the inclusive-scan layout.
         */
        struct FrameLocation
        {
            int regionIdx;
            // index of the frame within its region
            int localFrameIdx;
            int framesInRegion;
        };

        /**
         * @brief Maps a block index to a frame location using binary search on the inclusive-scan array.
         *
         * @param blockIdx   Global block index (== worker.blockDomIdx() at the call site)
         * @param scanBox    Inclusive prefix-sum of frame counts: scanBox[i] = sum of frames in regions [0, i]
         * @param numRegions Number of regions (== length of scanBox)
         * @return           FrameLocation with regionIdx, localFrameIdx, framesInRegion
         */
        DINLINE FrameLocation findFrameLocation(int blockIdx, auto const& scanBox, int numRegions)
        {
            // Upper-bound binary search: find smallest regionIdx s.t. scanBox[regionIdx] > blockIdx
            int left = 0;
            int right = numRegions;
            while(left < right)
            {
                int const mid = std::midpoint(left, right);
                if(static_cast<int>(scanBox[mid]) <= blockIdx)
                    left = mid + 1;
                else
                    right = mid;
            }
            int const regionIdx = left;
            int const start = (regionIdx == 0) ? 0 : static_cast<int>(scanBox[regionIdx - 1]);
            int const end = static_cast<int>(scanBox[regionIdx]);
            return {.regionIdx = regionIdx, .localFrameIdx = blockIdx - start, .framesInRegion = end - start};
            ;
        }

    } // namespace detail

    /**
     * @brief In-place inclusive prefix sum on a HostDeviceBuffer<uint32_t> via the host.
     *
     * Copies the buffer device to host, computes arr[i] += arr[i-1] for i in [1, size),
     * then copies back host to device.
     *
     * @param buf   Buffer populated by a device kernel.
     * @param size  Number of elements to scan (must be <= buf capacity).
     * @return      arr[size-1] after the scan (sum of all original counts).
     */
    [[nodiscard]] inline uint32_t inclusiveScanOnHost(pmacc::HostDeviceBuffer<uint32_t, DIM1>& buf, int size)
    {
        buf.deviceToHost();
        auto data = buf.getHostBuffer().getDataBox();
        for(int i = 1; i < size; ++i)
            data[i] += data[i - 1];
        uint32_t const total = data[size - 1];
        buf.hostToDevice();
        return total;
    }

    /**
     * @brief Host helper that dispatches one GPU block per frame across all regions.
     *
     * Launches CountFramesKernel to count frames per region, performs an inclusive prefix-sum
     * on the host, then launches @p processKernel with exactly totalFrames blocks.
     *
     * The process kernel receives:
     *   (worker, prDeviceBox, numRegions, framesScanBox, args...)
     * and should call detail::findFrameLocation(worker.blockDomIdx(), framesScanBox, numRegions)
     * to locate its assigned frame.
     *
     * @tparam TCountThreads   Threads per block for CountFramesKernel.
     * @tparam TProcessThreads Threads per block for processKernel.
     */
    template<uint32_t TCountThreads, uint32_t TProcessThreads>
    struct ForEachFrameInPRBuf
    {
        void operator()(auto& prBuf, auto processKernel, auto&&... args) const
        {
            if(prBuf.size == 0)
                return;

            pmacc::HostDeviceBuffer<uint32_t, DIM1> framesPerRegion(pmacc::DataSpace<DIM1>{prBuf.size});

            PMACC_LOCKSTEP_KERNEL(detail::CountFramesKernel{})
                .template config<TCountThreads>(pmacc::DataSpace<DIM1>(
                    prBuf.size))(prBuf.getDeviceDataBox(), prBuf.size, framesPerRegion.getDeviceBuffer().getDataBox());

            uint32_t const totalBlocks = inclusiveScanOnHost(framesPerRegion, prBuf.size);

            if(totalBlocks > 0)
            {
                PMACC_LOCKSTEP_KERNEL(processKernel)
                    .template config<TProcessThreads>(pmacc::DataSpace<DIM1>(totalBlocks))(
                        prBuf.getDeviceDataBox(),
                        prBuf.size,
                        framesPerRegion.getDeviceBuffer().getDataBox(),
                        std::forward<decltype(args)>(args)...);

                // Ensure framesPerRegion stays alive until the kernel finishes
                pmacc::eventSystem::waitForAllTasks();
            }
        }
    };

} // namespace pmacc::spearhed
