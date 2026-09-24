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
#include "spmacc/particles/spatial/BroadPhaseView.hpp"
#include "spmacc/particles/spatial/CsrCandidateProvider.hpp"
#include "spmacc/particles/spatial/InteractionEntry.hpp"

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/eventSystem/eventSystem.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>
#include <type_traits>
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

            DINLINE void operator()(
                auto const& worker,
                auto targetBroadPhase,
                int numTargetRegions,
                auto sourceBroadPhase,
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

                auto const searchVolume
                    = targetBroadPhase.bounds(static_cast<uint32_t>(blockIdx)).expand(smoothingLength);

                if constexpr(mode == OpMode::Count)
                {
                    unsigned int localCount = 0;
                    for(int otherIdx = threadIdx; otherIdx < numSourceRegions; otherIdx += numWorkers)
                    {
                        if(intersects(searchVolume, sourceBroadPhase.bounds(static_cast<uint32_t>(otherIdx))))
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
                        if(intersects(searchVolume, sourceBroadPhase.bounds(static_cast<uint32_t>(otherIdx))))
                        {
                            unsigned int const pos
                                = alpaka::atomicAdd(worker.getAcc(), &s_writePtr, 1u, ::alpaka::hierarchy::Threads{});
                            PMACC_ASSERT(pos < regionOffsetsBox[blockIdx + 1]);
                            if(pos < regionOffsetsBox[blockIdx + 1])
                                neighbourRegionsBox[pos] = static_cast<unsigned int>(otherIdx);
                        }
                    }

                    worker.sync();

                    if(threadIdx == 0)
                    {
                        PMACC_DEVICE_ASSERT_MSG(
                            s_writePtr == regionOffsetsBox[blockIdx + 1],
                            "NeighbourRegion write count mismatch: block index %u wrote to region %u or beyond",
                            blockIdx,
                            regionOffsetsBox[blockIdx + 1]);
                    }
                }
            }
        };
    } // namespace detail

    /** @brief Owning CSR arrays shared by concrete and generic candidate providers. */
    struct MaterializedNeighbourCsr
    {
        pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets;
    };

    /** @brief Materialize broad-phase overlaps into one CSR candidate list. */
    template<BroadPhaseView T_Target, BroadPhaseView T_Source, typename T_SmoothingLength>
    [[nodiscard]] auto materializeNeighbourCsr(
        T_Target const& target,
        T_Source const& source,
        T_SmoothingLength smoothingLength)
    {
        int const numTargetRegions = static_cast<int>(target.bucketCount());
        int const numSourceRegions = static_cast<int>(source.bucketCount());

        static constexpr uint32_t threadsPerBlock = 32;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets{pmacc::DataSpace<DIM1>{numTargetRegions + 1}};
        regionOffsets.getHostBuffer().setValue(0);
        regionOffsets.hostToDevice();
        if(numSourceRegions == 0)
        {
            pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions{pmacc::DataSpace<DIM1>{1}};
            return MaterializedNeighbourCsr{std::move(neighbourRegions), std::move(regionOffsets)};
        }
        if(numTargetRegions > 0)
        {
            PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Count>{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numTargetRegions))(
                    target,
                    numTargetRegions,
                    source,
                    numSourceRegions,
                    regionOffsets.getDeviceBuffer().getDataBox(),
                    nullptr,
                    smoothingLength);
        }

        uint32_t const totalPairs = inclusiveScanOnHost(regionOffsets, numTargetRegions + 1);
        pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions{
            pmacc::DataSpace<DIM1>{totalPairs == 0u ? 1 : static_cast<int>(totalPairs)}};
        if(totalPairs > 0)
        {
            PMACC_LOCKSTEP_KERNEL(detail::FindNeighbourRegionsFunctor<detail::OpMode::Write>{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numTargetRegions))(
                    target,
                    numTargetRegions,
                    source,
                    numSourceRegions,
                    regionOffsets.getDeviceBuffer().getDataBox(),
                    neighbourRegions.getDeviceBuffer().getDataBox(),
                    smoothingLength);
        }
        pmacc::eventSystem::getTransactionEvent().waitForFinished();
        return MaterializedNeighbourCsr{std::move(neighbourRegions), std::move(regionOffsets)};
    }

    /**
     * @brief Return a valid empty candidate bundle when a query has no sources.
     *
     * This permits runtime source selection to produce an empty interaction plan;
     * @c interact treats that plan as a no-op.
     */
    template<typename Target, typename SmoothingLength>
    auto calculateNeighbours(Target&, SmoothingLength)
    {
        return makeNeighbourBundle();
    }

    /**
     * @brief Compute neighbour-region lists for every source and return an owning bundle.
     *
     * @param target  The target ParticleRegionBuffer.
     * @param h       Smoothing length (scalar) expanding each region's AABB.
     * @param sources One or more source ParticleRegionBuffer objects.
     * @return A CSR-backed owning NeighbourBundle with one InteractionEntry per source.
     */
    template<typename Target, typename SmoothingLength, typename... Sources>
    auto calculateNeighbours(Target& target, SmoothingLength h, Sources&... sources) requires(sizeof...(Sources) > 0)
    {
        auto const targetView
            = MaterialAabbBroadPhaseView{target.getDeviceDataBox(), static_cast<uint32_t>(target.size)};

        auto computeOneEntry = [&](auto& sourcePRBuf)
        {
            using SrcType = std::remove_reference_t<decltype(sourcePRBuf)>;
            auto const sourceView
                = MaterialAabbBroadPhaseView{sourcePRBuf.getDeviceDataBox(), static_cast<uint32_t>(sourcePRBuf.size)};
            auto data = materializeNeighbourCsr(targetView, sourceView, h);

            using TargetType = std::remove_reference_t<Target>;
            using Provider = CsrCandidateProvider<TargetType, SrcType>;
            return InteractionEntry{
                &sourcePRBuf,
                Provider{&target, &sourcePRBuf, std::move(data.neighbourRegions), std::move(data.regionOffsets)}};
        };

        return makeNeighbourBundle(computeOneEntry(sources)...);
    }

} // namespace pmacc::spearhed
