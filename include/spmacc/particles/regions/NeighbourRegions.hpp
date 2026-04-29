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

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>

namespace pmacc::spearhed
{

    namespace detail
    {

        enum class OpMode
        {
            Count,
            Write
        };

        /**
         * Functor to find neighbouring volumes.
         * @tparam OpMode is count, only counts the number of neighbours, else the neighbour ids are written
         * into the neighbourRegionsBox
         */
        template<OpMode mode>
        struct FindNeighbourRegionsFunctor
        {
            // TODO Consider iterating otherIdx = blockIdx + 1; otherIdx < numRegions. The tradeoff is that populating
            // the neighbours bidirectionally requires atomicAdd for the offsets and writing, but halves the arithmetic
            // bounds-checking cost.
            DINLINE void operator()(
                auto const& worker,
                auto prDeviceBox,
                int numRegions,
                auto regionOffsetsBox,
                auto neighbourRegionsBox,
                auto smoothingLength) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= numRegions)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        auto const& region = prDeviceBox[blockIdx];
                        auto const searchVolume = region.volume.expand(smoothingLength);

                        unsigned int count = 0;
                        // Brute force check against all other volumes
                        // Use a BVH/Grid here instead of a linear loop
                        for(int otherIdx = 0; otherIdx < numRegions; ++otherIdx)
                        {
                            if(intersects(searchVolume, prDeviceBox[otherIdx].volume))
                            {
                                if constexpr(mode == OpMode::Write)
                                {
                                    // In populate mode, we start writing at our pre-scanned offset
                                    unsigned int writePtr = regionOffsetsBox[blockIdx];
                                    neighbourRegionsBox[writePtr + count] = static_cast<unsigned int>(otherIdx);
                                }
                                count++;
                            }
                        }

                        // If in counting mode, store the count in the offset array. Do the scan on host
                        if constexpr(mode == OpMode::Count)
                        {
                            regionOffsetsBox[blockIdx + 1] = count;
                        }
                    });
            }
        };
    } // namespace detail

    // returns mapping of regions to their neighbours {neighbourRegions, regionOffsets}
    // regionOffsets is an exclusive scan and neighbourRegions holds the list of neighbours
    // regionOffsets[myRegionIdx] and regionOffsets[myRegionIdx+1] defines the way to index into neighbourRegions of
    // myRegionIdx
    struct CalculateNeighbourRegions
    {
        auto operator()(auto& prBuf, auto smoothingLength)
        {
            auto numRegions = prBuf.size;
            pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets{pmacc::DataSpace<DIM1>{numRegions + 1}};
            regionOffsets.getHostBuffer().setValue(0);
            regionOffsets.hostToDevice();

            static constexpr uint32_t threadsPerBlock = 32;

            // Count neighbours per volume
            // We reuse the regionOffsets array to store temporary counts
            PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Count>{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numRegions))(
                    prBuf.getDeviceDataBox(),
                    numRegions,
                    regionOffsets.getDeviceBuffer().getDataBox(),
                    nullptr,
                    smoothingLength);

            uint32_t const totalPairs = inclusiveScanOnHost(regionOffsets, numRegions + 1);

            pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions(pmacc::DataSpace<DIM1>{totalPairs});

            // Populate the neighbour IDs
            if(totalPairs > 0)
            {
                PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Write>{})
                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numRegions))(
                        prBuf.getDeviceDataBox(),
                        numRegions,
                        regionOffsets.getDeviceBuffer().getDataBox(),
                        neighbourRegions.getDeviceBuffer().getDataBox(),
                        smoothingLength);
            }

            return std::pair{std::move(neighbourRegions), std::move(regionOffsets)};
        }
    };

} // namespace pmacc::spearhed
