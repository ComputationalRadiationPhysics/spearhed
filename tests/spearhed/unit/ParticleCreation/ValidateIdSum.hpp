/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/Identifier.hpp>

#include <cstdint>

namespace reduce::detail
{
    /** Kernel that sums particle IDs from all frames in assigned particle regions
     * Each block processes all frames in a region and outputs partial sum
     */
    struct SumParticleIds
    {
        template<typename T_Worker, typename T_PRBox, typename T_OutputBox>
        HDINLINE constexpr auto operator()(
            T_Worker const& worker,
            T_PRBox prDeviceBox,
            int numParticleRegions,
            T_OutputBox partialSums) const
        {
            using IdType = uint64_t;

            auto const blockIdx = worker.blockDomIdx();

            // Shared memory for block-level reduction
            PMACC_SMEM(worker, blockSum, IdType);

            auto onlyMaster = pmacc::lockstep::makeMaster(worker);
            onlyMaster([&]() { blockSum = 0; });

            worker.sync();

            // Process all particle regions
            // TODO optimize work distribution for load balancing
            for(int regionIdx = blockIdx; regionIdx < numParticleRegions; regionIdx += worker.gridDomSize())
            {
                auto& frameList = prDeviceBox[regionIdx].particleFrameList;

                using FrameType = typename std::remove_cvref_t<decltype(frameList)>::FrameType;
                constexpr uint32_t frameSize = FrameType::frameSize;
                // Iterate through all frames
                for(auto frameItr = frameList.begin(); frameItr != frameList.end(); ++frameItr)
                {
                    IdType threadLocalSum = 0;

                    auto forEachSlotInFrame = pmacc::lockstep::makeForEach<frameSize>(worker);

                    // Each thread processes particles in the frame
                    // TODO unroll last iteration and remove if multimask from the others
                    // use getNumParLastFrame instead of multimask
                    forEachSlotInFrame(
                        [&](uint32_t const idx)
                        {
                            auto particle = (*frameItr)[idx];

                            // Only sum valid particles
                            if(*particle[spearhed::multiMask])
                            {
                                threadLocalSum += *particle[spearhed::particleId];
                            }
                        });

                    // Reduce thread-local sums to block sum
                    // TODO consider warp-level primitives for better performance
                    if(threadLocalSum != 0)
                    {
                        alpaka::atomicAdd(worker.getAcc(), &blockSum, threadLocalSum, ::alpaka::hierarchy::Threads{});
                    }

                    worker.sync();
                }
            }

            onlyMaster([&]() { partialSums[blockIdx] = blockSum; });
        }
    };

} // namespace reduce::detail

/** Computes sum of all particle IDs in the simulation
 * Uses a parallelisation strategy of using one block per particle region
 *
 * @param prBuf Buffer containing all particle regions
 * @return Sum of all particle IDs
 */
struct ComputeParticleIdSum
{
    auto operator()() const -> uint64_t
    {
        constexpr uint32_t threadsPerBlock = 256;

        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>(
            pmacc::spearhed::prBufId<pmacc::spearhed::roles::Interior>());
        pmacc::HostDeviceBuffer<uint64_t, DIM1> partialSums(prBuf.size);

        PMACC_LOCKSTEP_KERNEL(reduce::detail::SumParticleIds{})
            .config<threadsPerBlock>(
                prBuf.size)(prBuf.getDeviceDataBox(), prBuf.size, partialSums.getDeviceBuffer().getDataBox());

        partialSums.deviceToHost();

        auto hostData = partialSums.getHostBuffer().getDataBox();
        uint64_t totalSum = 0;

        for(int i = 0; i < prBuf.size; ++i)
        {
            totalSum += hostData[i];
        }

        return totalSum;
    }
};
