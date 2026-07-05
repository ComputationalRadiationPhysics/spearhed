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
     * @brief Host-side grid-sizing (launch) policies: decide how many blocks
     *        forEachFrame launches for a given total frame count.
     *
     * This is the host half of an execution policy; it is orthogonal to the device-side frame
     * Schedule (see FrameSchedule.hpp), subject to the constraint that the OneToOne schedule is
     * only valid with OneBlockPerFrame.
     */

    //! One block per frame: grid size == total frames. The only launch valid with the OneToOne
    //! schedule; with a grid-stride or contiguous schedule the per-block loop simply runs once.
    struct OneBlockPerFrame
    {
        static constexpr uint32_t numBlocks(uint32_t totalFrames) noexcept
        {
            return totalFrames;
        }
    };

    //! Fixed grid of at most T_NumBlocks blocks (clamped so we never launch idle blocks). Requires
    //! a schedule that covers multiple frames per block (GridStride or Contiguous).
    template<uint32_t T_NumBlocks>
    struct FixedGrid
    {
        static constexpr uint32_t numBlocks(uint32_t totalFrames) noexcept
        {
            return T_NumBlocks < totalFrames ? T_NumBlocks : totalFrames;
        }
    };

    //! Named grid-sizing policy instances for value-based (constexpr object) configuration.
    inline constexpr OneBlockPerFrame oneBlockPerFrame{};
    template<uint32_t T_NumBlocks>
    inline constexpr FixedGrid<T_NumBlocks> fixedGrid{};

    /**
     * @brief Value-based launch configuration for the generic frame for-each.
     *
     * Carries the grid-sizing policy as a stateless constexpr sub-object and the two thread counts
     * -- which must be compile-time constants for the lockstep launch -- as static constexpr members
     * of the type, so the whole thing survives being passed by value (recovered at the call site via
     * decltype(cfg)::processThreads). Build one with launchConfig(...) instead of template arguments.
     */
    template<typename T_Grid, uint32_t T_ProcessThreads = 32, uint32_t T_CountThreads = 32>
    struct LaunchConfig
    {
        T_Grid grid;
        static constexpr uint32_t processThreads = T_ProcessThreads;
        static constexpr uint32_t countThreads = T_CountThreads;
    };

    /**
     * @brief Build a LaunchConfig from a constexpr grid-sizing object.
     *
     * @tparam T_ProcessThreads Threads per block for the process kernel.
     * @tparam T_CountThreads   Threads per block for CountFramesKernel.
     */
    template<uint32_t T_ProcessThreads = 32, uint32_t T_CountThreads = 32>
    constexpr auto launchConfig(auto grid)
    {
        return LaunchConfig<decltype(grid), T_ProcessThreads, T_CountThreads>{grid};
    }

    //! Default launch: one block per frame, 32 threads for both kernels.
    inline constexpr auto defaultLaunch = launchConfig(oneBlockPerFrame);

    /**
     * @brief Generic host for-each over frames: dispatches GPU blocks across all frames in all regions.
     *
     * Launches CountFramesKernel to count frames per region, performs an inclusive prefix-sum on the
     * host, then launches @p processKernel with a grid sized by @p launchCfg.grid. Configuration is a
     * value object (see launchConfig / defaultLaunch) rather than template arguments.
     *
     * The process kernel receives:
     *   (worker, prDeviceBox, numRegions, framesScanBox, args...)
     * and should map its block(s) to frames via a frame Schedule (see FrameSchedule.hpp) or, for the
     * one-block-per-frame kernels, directly via detail::findFrameLocation. The default OneBlockPerFrame
     * grid keeps both styles equivalent.
     *
     * @param launchCfg     A constexpr LaunchConfig value.
     * @param prBuf         The particle region buffer.
     * @param processKernel The per-block kernel functor.
     */
    void launchForEachFrameInBlock(auto launchCfg, auto& prBuf, auto processKernel, auto&&... args)
    {
        if(prBuf.size == 0)
            return;

        using Cfg = decltype(launchCfg);

        pmacc::HostDeviceBuffer<uint32_t, DIM1> framesPerRegion(pmacc::DataSpace<DIM1>{prBuf.size});

        PMACC_LOCKSTEP_KERNEL(detail::CountFramesKernel{})
            .template config<Cfg::countThreads>(pmacc::DataSpace<DIM1>(
                prBuf.size))(prBuf.getDeviceDataBox(), prBuf.size, framesPerRegion.getDeviceBuffer().getDataBox());

        uint32_t const totalFrames = inclusiveScanOnHost(framesPerRegion, prBuf.size);

        if(totalFrames > 0)
        {
            uint32_t const gridSize = launchCfg.grid.numBlocks(totalFrames);

            PMACC_LOCKSTEP_KERNEL(processKernel)
                .template config<Cfg::processThreads>(pmacc::DataSpace<DIM1>(gridSize))(
                    prBuf.getDeviceDataBox(),
                    prBuf.size,
                    framesPerRegion.getDeviceBuffer().getDataBox(),
                    std::forward<decltype(args)>(args)...);

            // Ensure framesPerRegion stays alive until the kernel finishes
            // TODO fix this
            pmacc::eventSystem::waitForAllTasks();
        }
    }

    /**
     * @brief Index-driven counterpart to launchForEachFrameInBlock: no per-launch count kernel and no
     *        host inclusive scan -- the block-to-frame mapping is read straight from a prebuilt index.
     *
     * The caller owns a FrameIndexBuffer (see FrameIndex.hpp) rebuilt from @p prBuf's frame lists. This
     * launcher merely sizes the grid from @p index.totalFrames and hands each block its region index and
     * frame device pointer via the index's device boxes. The process kernel receives:
     *   (worker, prDeviceBox, framePtrsBox, regionIdxBox, totalFrames, args...)
     * and maps its block directly: rIdx = regionIdxBox[blockIdx]; ownFramePtr = framePtrsBox[blockIdx].
     *
     * Unlike launchForEachFrameInBlock this does NOT waitForAllTasks -- the index's device buffers must
     * outlive the launched kernels, which is the caller's responsibility (it keeps the FrameIndexBuffer
     * alive and synchronises once after all launches; see InteractParticles / InteractParticlesUnified).
     *
     * @param launchCfg     A constexpr LaunchConfig value (only its grid-sizing policy is used here).
     * @param prBuf         The particle region buffer (supplies the region device box + region volumes).
     * @param index         A rebuilt FrameIndexBuffer for @p prBuf.
     * @param processKernel The per-block kernel functor.
     */
    void launchForEachFrameInBlockIndexed(auto launchCfg, auto& prBuf, auto& index, auto processKernel, auto&&... args)
    {
        if(index.totalFrames == 0)
            return;

        using Cfg = decltype(launchCfg);

        uint32_t const gridSize = launchCfg.grid.numBlocks(index.totalFrames);

        PMACC_LOCKSTEP_KERNEL(processKernel)
            .template config<Cfg::processThreads>(pmacc::DataSpace<DIM1>(gridSize))(
                prBuf.getDeviceDataBox(),
                index.framePtrsBox(),
                index.regionIdxBox(),
                index.totalFrames,
                std::forward<decltype(args)>(args)...);
    }

} // namespace pmacc::spearhed
