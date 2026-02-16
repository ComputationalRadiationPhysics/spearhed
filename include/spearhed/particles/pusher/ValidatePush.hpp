#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/speciesTraits.param"
#include "spearhed/particles/attributes/Position.hpp"
#include "spmacc/ParticleRegionBuffer.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/Identifier.hpp>

#include <cstdint>

namespace spearhed
{
    /** Kernel that checks postion is 1 from all frames in assigned particle regions
     * Each block processes all frames in a region and outputs partial sum
     */
    struct CheckParticlePos
    {
        template<typename T_Worker, typename T_PRBox>
        HDINLINE constexpr auto operator()(T_Worker const& worker, T_PRBox prDeviceBox, int numParticleRegions) const
        {
            auto const blockIdx = worker.blockDomIdx();
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
                    auto forEachSlotInFrame = pmacc::lockstep::makeForEach<frameSize>(worker);

                    // Each thread processes particles in the frame
                    // TODO unroll last iteration and remove if multimask from the others, if we assume no gaps
                    // use getNumParLastFrame instead of multimask
                    forEachSlotInFrame(
                        [&](uint32_t const idx)
                        {
                            auto particle = (*frameItr)[idx];

                            // Only sum valid particles
                            if(*particle[multiMask])
                            {
                                if(!particle[pos].get().isApprox(1.f))
                                {
                                    printf(
                                        "particle position incorrect. Got pos x %g vel x %g\n",
                                        particle[pos].get().get_x(),
                                        *particle[vel][x]);
                                }
                                else
                                {
                                    printf("true\n");
                                }
                            }
                        });
                }
            }
        }
    };

    /** Computes sum of all particle IDs in the simulation
     * Uses a parallelisation strategy of using one block per particle region
     *
     * @param prBuf Buffer containing all particle regions
     * @return Sum of all particle IDs
     */
    struct ValidatePush
    {
        auto operator()() const -> void
        {
            constexpr int numBlocks = 256;
            constexpr uint32_t threadsPerBlock = 256;

            auto& dc = pmacc::Environment<>::get().DataConnector();
            auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<PRType>>("PRBuf");

            PMACC_LOCKSTEP_KERNEL(CheckParticlePos{})
                .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(numBlocks))(prBuf.getDeviceDataBox(), prBuf.size);
        }
    };

} // namespace spearhed
