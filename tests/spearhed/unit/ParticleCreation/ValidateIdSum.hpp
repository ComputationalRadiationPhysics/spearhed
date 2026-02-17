#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spmacc/ParticleRegionBuffer.hpp"

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
        constexpr int numBlocks = 256;
        constexpr uint32_t threadsPerBlock = 256;

        pmacc::HostDeviceBuffer<uint64_t, DIM1> partialSums(pmacc::DataSpace<DIM1>{numBlocks});

        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>("PRBuf");

        PMACC_LOCKSTEP_KERNEL(reduce::detail::SumParticleIds{})
            .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(
                numBlocks))(prBuf.getDeviceDataBox(), prBuf.size, partialSums.getDeviceBuffer().getDataBox());

        partialSums.deviceToHost();

        auto hostData = partialSums.getHostBuffer().getDataBox();
        uint64_t totalSum = 0;

        for(int i = 0; i < numBlocks; ++i)
        {
            totalSum += hostData[i];
        }

        return totalSum;
    }
};
