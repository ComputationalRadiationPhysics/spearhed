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

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        auto const& region = targetPRDeviceBox[blockIdx];
                        auto const searchVolume = region.volume.expand(smoothingLength);

                        unsigned int count = 0;
                        // Brute force check against all source volumes
                        // Use a BVH/Grid here instead of a linear loop
                        for(int otherIdx = 0; otherIdx < numSourceRegions; ++otherIdx)
                        {
                            if(intersects(searchVolume, sourcePRDeviceBox[otherIdx].volume))
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

    /**
     * @brief Computes neighbour-region lists for all sources and returns them as a NeighbourBundle.
     *
     * regionOffsets[targetIdx] .. regionOffsets[targetIdx+1] indexes into neighbourRegions for targetIdx.
     * neighbourRegions holds source-region indices (into the source PRBuf, not the target).
     *
     * Call bundle.subset<Roles...>() to get a narrower view for kernels that only need certain sources.
     */
    struct CalculateNeighbourRegions
    {
        auto operator()(auto& targetPRBuf, auto smoothingLength, auto&... sourcePRBufs)
        {
            int const numTargetRegions = targetPRBuf.size;
            static constexpr uint32_t threadsPerBlock = 32;

            auto computeOneEntry = [&](auto& sourcePRBuf)
            {
                using SrcType = std::remove_reference_t<decltype(sourcePRBuf)>;
                int const numSourceRegions = sourcePRBuf.size;

                pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets{
                    pmacc::DataSpace<DIM1>{numTargetRegions + 1}};
                regionOffsets.getHostBuffer().setValue(0);
                regionOffsets.hostToDevice();

                PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Count>{})
                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numTargetRegions))(
                        targetPRBuf.getDeviceDataBox(),
                        numTargetRegions,
                        sourcePRBuf.getDeviceDataBox(),
                        numSourceRegions,
                        regionOffsets.getDeviceBuffer().getDataBox(),
                        nullptr,
                        smoothingLength);

                uint32_t const totalPairs = inclusiveScanOnHost(regionOffsets, numTargetRegions + 1);

                pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions(pmacc::DataSpace<DIM1>{totalPairs});

                if(totalPairs > 0)
                {
                    PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Write>{})
                        .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numTargetRegions))(
                            targetPRBuf.getDeviceDataBox(),
                            numTargetRegions,
                            sourcePRBuf.getDeviceDataBox(),
                            numSourceRegions,
                            regionOffsets.getDeviceBuffer().getDataBox(),
                            neighbourRegions.getDeviceBuffer().getDataBox(),
                            smoothingLength);
                }

                return NeighbourEntry<SrcType>{&sourcePRBuf, std::move(neighbourRegions), std::move(regionOffsets)};
            };

            using TargetType = std::remove_reference_t<decltype(targetPRBuf)>;
            auto entryTuple = std::make_tuple(computeOneEntry(sourcePRBufs)...);
            using TupleType = decltype(entryTuple);
            return [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                return NeighbourBundle<TargetType, std::tuple_element_t<I, TupleType>...>{
                    &targetPRBuf,
                    std::move(entryTuple)};
            }(std::make_index_sequence<sizeof...(sourcePRBufs)>{});
        }
    };

} // namespace pmacc::spearhed
