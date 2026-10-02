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

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/alpaka.hpp>

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
                            unsigned int pos = ::alpaka::onAcc::atomicAdd(
                                worker.getAcc(),
                                &s_writePtr,
                                1u,
                                ::alpaka::onAcc::scope::Block{});
                            neighbourRegionsBox[pos] = static_cast<unsigned int>(otherIdx);
                        }
                    }

                    worker.sync();

                    if(threadIdx == 0)
                    {
                        PMACC_DEVICE_ASSERT_MSG(
                            s_writePtr <= regionOffsetsBox[blockIdx + 1],
                            "NeighbourRegion write overflow: block index %u wrote to region %u or beyond",
                            blockIdx,
                            regionOffsetsBox[blockIdx + 1]);
                    }
                }
            }
        };
    } // namespace detail

    namespace detail
    {
        template<typename Target, typename Source, typename SmoothingLength>
        [[nodiscard]] auto calculateNeighbourEntrySender(Target& target, Source& source, SmoothingLength h)
        {
            struct State
            {
                Target* target;
                Source* source;
                SmoothingLength smoothingLength;
                int numTargetRegions;
                int numSourceRegions;
                uint32_t totalPairs = 0u;
                pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets;
                std::optional<pmacc::HostDeviceBuffer<unsigned int, DIM1>> neighbourRegions;

                State(Target& targetRef, Source& sourceRef, SmoothingLength hValue)
                    : target(&targetRef)
                    , source(&sourceRef)
                    , smoothingLength(hValue)
                    , numTargetRegions(targetRef.size)
                    , numSourceRegions(sourceRef.size)
                    , regionOffsets(pmacc::DataSpace<DIM1>{numTargetRegions + 1})
                {
                    regionOffsets.getHostBuffer().setValue(0u);
                }
            };

            auto state = std::make_shared<State>(target, source, h);
            constexpr uint32_t threadsPerBlock = 32;
            auto countKernel
                = PMACC_LOCKSTEP_KERNEL(FindNeighbourRegionsFunctor<OpMode::Count>{})
                      .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(std::max(state->numTargetRegions, 1)));
            auto count = caravan::alpaka::submit(
                [state, countKernel](auto& queue) mutable
                {
                    if(state->numTargetRegions > 0 && state->numSourceRegions > 0)
                        countKernel.enqueueNative(
                            queue,
                            state->target->getDeviceDataBox(),
                            state->numTargetRegions,
                            state->source->getDeviceDataBox(),
                            state->numSourceRegions,
                            state->regionOffsets.getDeviceBuffer().getDataBox(),
                            nullptr,
                            state->smoothingLength);
                });

            return state->regionOffsets.hostToDevice() | caravan::alpaka::sequence(std::move(count))
                   | caravan::alpaka::sequence(state->regionOffsets.deviceToHost())
                   | caravan::letValue(
                       [state]()
                       {
                           state->totalPairs = inclusiveScanHost(state->regionOffsets, state->numTargetRegions + 1);
                           state->neighbourRegions.emplace(
                               pmacc::DataSpace<DIM1>{static_cast<int>(state->totalPairs)});
                           auto writeKernel = PMACC_LOCKSTEP_KERNEL(FindNeighbourRegionsFunctor<OpMode::Write>{})
                                                  .template config<threadsPerBlock>(
                                                      pmacc::DataSpace<DIM1>(std::max(state->numTargetRegions, 1)));
                           auto write = caravan::alpaka::submit(
                               [state, writeKernel](auto& queue) mutable
                               {
                                   if(state->numTargetRegions > 0 && state->totalPairs > 0u)
                                       writeKernel.enqueueNative(
                                           queue,
                                           state->target->getDeviceDataBox(),
                                           state->numTargetRegions,
                                           state->source->getDeviceDataBox(),
                                           state->numSourceRegions,
                                           state->regionOffsets.getDeviceBuffer().getDataBox(),
                                           state->neighbourRegions->getDeviceBuffer().getDataBox(),
                                           state->smoothingLength);
                               });
                           return state->regionOffsets.hostToDevice() | caravan::alpaka::sequence(std::move(write))
                                  | caravan::then(
                                      [state]()
                                      {
                                          using Entry = NeighbourEntry<Source>;
                                          return Entry{
                                              state->source,
                                              std::move(*state->neighbourRegions),
                                              std::move(state->regionOffsets)};
                                      });
                       });
        }
    } // namespace detail

    /**
     * @brief Lazily compute neighbour-region lists and deliver an owning bundle when ready.
     *
     * The sender borrows @p target and @p sources. Their objects, region counts, and backing storage
     * must remain valid and unchanged through completion; counts are captured when this function is
     * called. Region contents are read when the sender starts, so callers may update them before
     * execution provided those updates are complete before the sender is started.
     */
    template<typename Target, typename SmoothingLength, typename... Sources>
    [[nodiscard]] auto calculateNeighboursSender(Target& target, SmoothingLength h, Sources&... sources)
    {
        if constexpr(sizeof...(Sources) == 0u)
        {
            return caravan::alpaka::submit([](auto&) {}) | caravan::then([] { return makeNeighbourBundle(); });
        }
        else
            return caravan::whenAll(detail::calculateNeighbourEntrySender(target, sources, h)...)
                   | caravan::then([](auto&&... entries)
                                   { return makeNeighbourBundle(std::forward<decltype(entries)>(entries)...); });
    }

    /** Compute neighbour-region lists synchronously for callers requiring an immediate bundle. */
    template<typename Target, typename SmoothingLength, typename... Sources>
    [[nodiscard]] auto calculateNeighbours(Target& target, SmoothingLength h, Sources&... sources)
    {
        using Bundle
            = decltype(makeNeighbourBundle(std::declval<NeighbourEntry<std::remove_reference_t<Sources>>>()...));
        auto& device = pmacc::Environment<>::get().DeviceContext();
        return caravan::syncWait<Bundle>(
            caravan::alpaka::withDevice(device, calculateNeighboursSender(target, h, sources...)));
    }

} // namespace pmacc::spearhed
