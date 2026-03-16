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

#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <cstdio>

namespace pmacc::spearhed
{
    struct UpdateRegionBounds
    {
        // Size of shared memory buffer
        // must match or exceed threadsPerBlock
        static constexpr int MaxBlockSize = 512;

        // //(Max threads / 32 warps) e.g., with a warp size of 32, for 512 threads, we need 16 slots.
        // static constexpr int MaxWarps = 32;

        HDINLINE constexpr void operator()(auto const& worker, auto prDeviceBox, int numParticleRegions) const
        {
            auto const blockIdx = worker.blockDomIdx();
            auto const threadIdx = worker.workerIdx();
            auto const numWorkers = worker.numWorkers();


            for(int regionIdx = blockIdx; regionIdx < numParticleRegions; regionIdx += worker.gridDomSize())
            {
                auto& region = prDeviceBox[regionIdx];
                using VolumeType = std::remove_cvref_t<decltype(region.volume)>;
                using FrameType = typename std::remove_cvref_t<decltype(region.particleFrameList)>::FrameType;

                // Thread-Local Accumulation
                VolumeType localBounds;
                localBounds.reset();

                auto& frameList = region.particleFrameList;
                for(auto frameItr = frameList.begin(); frameItr != frameList.end(); ++frameItr)
                {
                    auto forEachSlot = pmacc::lockstep::makeForEach<FrameType::frameSize>(worker);
                    forEachSlot(
                        [&](uint32_t const idx)
                        {
                            auto particle = (*frameItr)[idx];
                            if(*particle[tags::multiMask])
                            {
                                localBounds.extend(particle[tags::relativePos].get());
                            }
                        });
                }

                // Block-Wide Reduction
                // Allocate shared memory for the reduction tree
                PMACC_SMEM(worker, s_bounds, VolumeType[MaxBlockSize]);

                // Load thread-local results into shared memory
                if(threadIdx < MaxBlockSize)
                {
                    s_bounds[threadIdx] = localBounds;
                }
                worker.sync();

                // Perform tree reduction in shared memory
                for(unsigned int s = numWorkers / 2; s > 0; s >>= 1)
                {
                    if(threadIdx < s)
                    {
                        // it follows that (threadIdx + s) < numWorkers
                        s_bounds[threadIdx].extend(s_bounds[threadIdx + s]);
                    }
                    worker.sync();
                }

                if(threadIdx == 0)
                {
                    region.volume = s_bounds[0];
                }

                // Ensure write visibility before next iteration
                worker.sync();

                // Reduction using warp (can be used instead of block wide reduction)

                // std::int32_t warpSize = alpaka::warp::getSize(worker.getAcc());
                // // Calculate lane and warp indices
                // auto const laneIdx = threadIdx % warpSize;
                // auto const warpIdx = threadIdx / warpSize;

                // // Warp-Level reduction using shuffle
                // // Each thread combines its result with a neighbour down the warp
                // for(int offset = warpSize / 2; offset > 0; offset /= 2)
                // {
                //     localBounds.extend(localBounds.shuffle_down(worker, offset));
                // }

                // // Store warp results to shared memory
                // // Only the first thread of each warp writes the result
                // PMACC_SMEM(worker, s_warp_bounds, VolumeType[MaxWarps]);

                // if(laneIdx == 0)
                // {
                //     s_warp_bounds[warpIdx] = localBounds;
                // }

                // worker.sync();

                // // Final Block Reduction (Performed by the first warp)
                // if(warpIdx == 0)
                // {
                //     // Read warp results from shared memory
                //     // Check if the warp existed in the block
                //     if(laneIdx < (blockSize / warpSize))
                //     {
                //         localBounds = s_warp_bounds[laneIdx];
                //     }
                //     else
                //     {
                //         localBounds.reset();
                //     }

                //     // Reduce the warp results again within the first warp
                //     for(int offset = warpSize / 2; offset > 0; offset /= 2)
                //     {
                //         localBounds.extend(localBounds.shuffle_down(offset));
                //     }

                //     // Write final result to global memory
                //     if(laneIdx == 0)
                //     {
                //         region.volume = localBounds;
                //     }
                // }

                // // Ensure write visibility before next iteration
                // worker.sync();}
            }
        }
    };

    template<typename T_ParticleRegion>
    struct UpdateVolumes
    {
        // Allow customizing the buffer name if needed
        void operator()() const
        {
            // Tuning constants
            constexpr uint32_t threadsPerBlock = 256;
            // Maximum blocks to launch (prevents kernel launch overhead on small GPUs)
            constexpr int maxBlocks = 1024;

            auto& dc = pmacc::Environment<>::get().DataConnector();

            // Note: Ensure PRType is defined in this scope or passed as a template
            using BufferType = pmacc::spearhed::ParticleRegionBuffer<T_ParticleRegion>;

            auto& prBuf = *dc.get<BufferType>("PRBuf");

            // Dynamic Grid Sizing:
            // Calculate enough blocks to cover the regions, capped at maxBlocks.
            // Since the kernel uses a grid-stride loop, this ensures high occupancy
            // without launching unnecessary empty blocks for small problems.
            int numBlocks = (prBuf.size + threadsPerBlock - 1) / threadsPerBlock;
            if(numBlocks > maxBlocks)
                numBlocks = maxBlocks;
            if(numBlocks == 0)
                numBlocks = 1;

            PMACC_LOCKSTEP_KERNEL(UpdateRegionBounds{})
                .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numBlocks))(prBuf.getDeviceDataBox(), prBuf.size);
        }
    };
} // namespace pmacc::spearhed
