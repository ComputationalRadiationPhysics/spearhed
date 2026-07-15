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
#include "spmacc/particles/regions/NeighbourBundle.hpp"
#include "spmacc/particles/regions/NeighbourEntry.hpp"

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>
#include <utility>

namespace pmacc::spearhed
{

    namespace detail
    {

        enum class OpMode
        {
            Count,
            Write
        };

        template<OpMode mode>
        struct FindNeighbourRegionsFunctor
        {
            static constexpr unsigned int kMaxThreads = 64;
            unsigned int totalPairs = 0; // only meaningful in Write mode (bounds assertion)

            DINLINE void operator()(
                auto const& worker,
                auto targetPRDeviceBox,
                int numTargetRegions,
                auto sourcePRDeviceBox,
                int numSourceRegions,
                auto regionOffsetsBox,
                auto neighbourRegionsBox,
                auto smoothingLength) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= numTargetRegions)
                    return;

                auto const threadIdx = worker.workerIdx();
                auto const numWorkers = worker.numWorkers();

                auto const& region = targetPRDeviceBox[blockIdx];
                auto const searchVolume = region.volume.expand(smoothingLength);

                if constexpr(mode == OpMode::Count)
                {
                    unsigned int localCount = 0;
                    for(int otherIdx = threadIdx; otherIdx < numSourceRegions; otherIdx += numWorkers)
                    {
                        if(intersects(searchVolume, sourcePRDeviceBox[otherIdx].volume))
                            localCount++;
                    }

                    PMACC_SMEM(worker, s_counts, unsigned int[kMaxThreads]);
                    s_counts[threadIdx] = localCount;
                    worker.sync();

                    for(unsigned int s = numWorkers / 2; s > 0; s >>= 1)
                    {
                        if(threadIdx < s)
                            s_counts[threadIdx] += s_counts[threadIdx + s];
                        worker.sync();
                    }

                    if(threadIdx == 0)
                        regionOffsetsBox[blockIdx + 1] = s_counts[0];
                }
                else
                {
                    PMACC_SMEM(worker, s_writePtr, unsigned int);
                    if(threadIdx == 0)
                        s_writePtr = regionOffsetsBox[blockIdx];
                    worker.sync();

                    for(int otherIdx = threadIdx; otherIdx < numSourceRegions; otherIdx += numWorkers)
                    {
                        if(intersects(searchVolume, sourcePRDeviceBox[otherIdx].volume))
                        {
                            unsigned int pos
                                = alpaka::atomicAdd(worker.getAcc(), &s_writePtr, 1u, ::alpaka::hierarchy::Threads{});
                            neighbourRegionsBox[pos] = static_cast<unsigned int>(otherIdx);
                        }
                    }

                    worker.sync();

                    if(threadIdx == 0)
                    {
                        PMACC_DEVICE_ASSERT_MSG(
                            s_writePtr <= regionOffsetsBox[blockIdx] + totalPairs,
                            "NeighbourRegion write overflow: block %u allocated %u pairs",
                            blockIdx,
                            totalPairs);
                    }
                }
            }
        };
    } // namespace detail

    /**
     * @brief Compute neighbour-region lists for every source and return an owning bundle.
     *
     * @param target  The target ParticleRegionBuffer.
     * @param h       Smoothing length (scalar) expanding each region's AABB.
     * @param sources One or more source ParticleRegionBuffer objects.
     * @return NeighbourBundle<true, NeighbourEntry<Sources>...>
     */
    template<typename Target, typename SmoothingLength, typename... Sources>
    auto calculateNeighbours(Target& target, SmoothingLength h, Sources&... sources)
    {
        int const numTargetRegions = target.size;
        static constexpr uint32_t threadsPerBlock = 32;

        auto computeOneEntry = [&](auto& sourcePRBuf)
        {
            using SrcType = std::remove_reference_t<decltype(sourcePRBuf)>;
            int const numSourceRegions = sourcePRBuf.size;

            pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets{pmacc::DataSpace<DIM1>{numTargetRegions + 1}};
            regionOffsets.getHostBuffer().setValue(0);
            regionOffsets.hostToDevice();

            PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Count>{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numTargetRegions))(
                    target.getDeviceDataBox(),
                    numTargetRegions,
                    sourcePRBuf.getDeviceDataBox(),
                    numSourceRegions,
                    regionOffsets.getDeviceBuffer().getDataBox(),
                    nullptr,
                    h);

            uint32_t const totalPairs = inclusiveScanOnHost(regionOffsets, numTargetRegions + 1);

            pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions(pmacc::DataSpace<DIM1>{totalPairs});

            if(totalPairs > 0)
            {
                auto writeKernel = detail::FindNeighbourRegionsFunctor<detail::OpMode::Write>{};
                writeKernel.totalPairs = totalPairs;
                PMACC_LOCKSTEP_KERNEL(writeKernel)
                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numTargetRegions))(
                        target.getDeviceDataBox(),
                        numTargetRegions,
                        sourcePRBuf.getDeviceDataBox(),
                        numSourceRegions,
                        regionOffsets.getDeviceBuffer().getDataBox(),
                        neighbourRegions.getDeviceBuffer().getDataBox(),
                        h);
            }

            return NeighbourEntry<SrcType>{&sourcePRBuf, std::move(neighbourRegions), std::move(regionOffsets)};
        };

        return makeNeighbourBundle(computeOneEntry(sources)...);
    }

} // namespace pmacc::spearhed
