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

#include <pmacc/assert.hpp>
#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/eventSystem/eventSystem.hpp>
#include <pmacc/eventSystem/events/EventTask.hpp>
#include <pmacc/eventSystem/tasks/TaskKernel.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <cstdint>

/*
 * Building blocks for dispatching GPU blocks across a ParticleRegionBuffer's frames.
 *
 * This file holds: the frame-count-per-region kernel (used to build a FrameIndexBuffer, see
 * FrameIndex.hpp) and the host-side inclusive-scan helper it shares with InitParticles.hpp's own
 * scan-based dispatch, the host-side grid-sizing policies (OneBlockPerFrame / FixedGrid), and the
 * single indexed launcher launchForEachFrameInBlockIndexed, which dispatches from a prebuilt
 * FrameIndexBuffer instead of scanning frame counts on every call. See LaunchForEach.hpp for the
 * higher-level entry points built on top of the indexed launcher.
 */

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

    } // namespace detail

    /**
     * @brief In-place inclusive prefix sum on a HostDeviceBuffer<uint32_t> via the host.
     *
     * Copies the buffer device to host, computes arr[i] += arr[i-1] for i in [1, size),
     * then copies non-zero scans back to the device. A zero sum leaves the already-zero device
     * buffer unchanged and avoids queuing a redundant asynchronous copy.
     *
     * @param buf   Buffer populated by a device kernel.
     * @param size  Number of elements to scan (must be <= buf capacity).
     * @return      arr[size-1] after the scan (sum of all original counts).
     */
    [[nodiscard]] inline uint32_t inclusiveScanOnHost(pmacc::HostDeviceBuffer<uint32_t, DIM1>& buf, int size)
    {
        buf.deviceToHost();
        pmacc::eventSystem::getTransactionEvent().waitForFinished();
        auto data = buf.getHostBuffer().getDataBox();
        for(int i = 1; i < size; ++i)
            data[i] += data[i - 1];
        uint32_t const total = data[size - 1];
        if(total != 0u)
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
     * @brief Value-based launch configuration for the indexed frame for-each.
     *
     * Carries the grid-sizing policy as a stateless constexpr sub-object and the process thread
     * count -- which must be a compile-time constant for the lockstep launch -- as a static
     * constexpr member of the type, so the whole thing survives being passed by value (recovered at
     * the call site via decltype(cfg)::processThreads). Build one with launchConfig(...) instead of
     * template arguments. There is no count-kernel thread count here: index (re)builds own their
     * kernel thread count (see FrameIndexBuffer::kIndexThreads in FrameIndex.hpp).
     */
    template<typename T_Grid, uint32_t T_ProcessThreads = 32>
    struct LaunchConfig
    {
        T_Grid grid;
        static constexpr uint32_t processThreads = T_ProcessThreads;
    };

    /**
     * @brief Build a LaunchConfig from a constexpr grid-sizing object.
     *
     * @tparam T_ProcessThreads Threads per block for the process kernel.
     */
    template<uint32_t T_ProcessThreads = 32>
    constexpr auto launchConfig(auto grid)
    {
        return LaunchConfig<decltype(grid), T_ProcessThreads>{grid};
    }

    //! Default launch: one block per frame, 32 threads for the process kernel.
    inline constexpr auto defaultLaunch = launchConfig(oneBlockPerFrame);

    /**
     * @brief Index-driven host for-each over frames: no per-launch count kernel and no host
     *        inclusive scan -- the block-to-frame mapping is read straight from a prebuilt index.
     *
     * The caller owns a FrameIndexBuffer (see FrameIndex.hpp) rebuilt from @p prBuf's frame lists. This
     * launcher merely sizes the grid from @p index.totalFrames and hands each block its region index and
     * frame device pointer via the index's device boxes. The process kernel receives:
     *   (worker, prDeviceBox, framePtrsBox, regionIdxBox, totalFrames, args...)
     * and maps its block directly: rIdx = regionIdxBox[blockIdx]; ownFramePtr = framePtrsBox[blockIdx].
     *
     * This does NOT synchronise: the kernel is merely enqueued and this function returns immediately.
     * Every device allocation reachable from the launch -- @p prBuf, @p index, and any buffer viewed
     * by @p args -- must outlive kernel COMPLETION, not just this call (PMacc buffer destructors do
     * not wait for in-flight kernels). The caller synchronises via the returned event at its natural
     * sync point; see interact() in NeighbourRegions.hpp.
     *
     * @param launchCfg     A constexpr LaunchConfig value (only its grid-sizing policy is used here).
     * @param prBuf         The particle region buffer (supplies the region device box + region volumes).
     * @param index         A rebuilt FrameIndexBuffer for @p prBuf.
     * @param processKernel The per-block kernel functor.
     * @return EventTask for the enqueued kernel (an empty, already-finished event if totalFrames == 0).
     *         Wait on it with waitForFinished() before destroying any buffer the kernel touches.
     */
    [[nodiscard]] pmacc::EventTask launchForEachFrameInBlockIndexed(
        auto launchCfg,
        auto& prBuf,
        auto& index,
        auto processKernel,
        auto&&... args)
    {
        if(index.totalFrames == 0)
            return {};

        // A stale index (built against an older topology) holds dangling device frame pointers.
        PMACC_ASSERT(index.builtVersion == prBuf.topologyVersion);

        using Cfg = decltype(launchCfg);

        uint32_t const gridSize = launchCfg.grid.numBlocks(index.totalFrames);

        return PMACC_LOCKSTEP_KERNEL(processKernel)
            .template config<Cfg::processThreads>(pmacc::DataSpace<DIM1>(gridSize))(
                prBuf.getDeviceDataBox(),
                index.framePtrsBox(),
                index.regionIdxBox(),
                index.totalFrames,
                std::forward<decltype(args)>(args)...);
    }

} // namespace pmacc::spearhed
