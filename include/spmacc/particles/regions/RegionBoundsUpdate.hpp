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

#include "spmacc/particles/algorithms/FrameSchedule.hpp"
#include "spmacc/particles/algorithms/HierarchyForEach.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

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
                using BoundsType = std::remove_cvref_t<decltype(region.spatial.occupancy)>;

                // Particle coordinates are chart-local, while occupancy is always world-space.
                // Keep the chart immutable: material membership does not rebase particles after motion.
                BoundsType localBounds;

                // DeviceHeapAccess{} instead of the deviceHeap instance: odr-using the
                // namespace-scope constexpr variable from device code is ill-formed under nvcc.
                forEach(
                    levels::frame,
                    DeviceHeapAccess{},
                    makeRegionView(region),
                    [&](auto frame)
                    {
                        lockstepForEachParticle(
                            worker,
                            frame,
                            [&](auto particle)
                            { localBounds.extend(region.spatial.chart.toWorld(particle[tags::relativePos].get())); });
                    });

                // Block-Wide Reduction
                // Allocate shared memory for the reduction tree
                PMACC_SMEM(worker, s_bounds, BoundsType[MaxBlockSize]);

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
                    region.spatial.occupancy = s_bounds[0];
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
                //         region.spatial.occupancy = localBounds;
                //     }
                // }

                // // Ensure write visibility before next iteration
                // worker.sync();}
            }
        }
    };

    /**
     * @brief Reduce chart-relative particle positions into material occupancy bounds.
     *
     * This is the low-level compatibility operation. Spatial lifecycle users
     * should call MaterialAabbDecomposition::prepareAfterMotion() instead.
     */
    template<typename T_ParticleRegion>
    void updateMaterialAabbBounds(ParticleRegionBuffer<T_ParticleRegion>& prBuf)
    {
        constexpr uint32_t threadsPerBlock = 256;
        constexpr int maxBlocks = 1024;

        int numBlocks = (prBuf.size + static_cast<int>(threadsPerBlock) - 1) / static_cast<int>(threadsPerBlock);
        if(numBlocks > maxBlocks)
            numBlocks = maxBlocks;
        if(numBlocks == 0)
            numBlocks = 1;

        PMACC_LOCKSTEP_KERNEL(UpdateRegionBounds{})
            .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numBlocks))(prBuf.getDeviceDataBox(), prBuf.size);
    }

    // Compatibility wrapper for low-level tests and callers not yet migrated to a decomposition.
    template<typename T_ParticleRegion>
    struct UpdateVolumes
    {
        void operator()() const
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();
            using BufferType = pmacc::spearhed::ParticleRegionBuffer<T_ParticleRegion>;
            using Species = typename T_ParticleRegion::Species;
            updateMaterialAabbBounds(*dc.get<BufferType>(prBufId(Species{})));
        }
    };
} // namespace pmacc::spearhed
