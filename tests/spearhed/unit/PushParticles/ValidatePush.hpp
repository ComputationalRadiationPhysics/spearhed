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

#include "catch2/catch_test_macros.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spmacc/ParticleRegionBuffer.hpp"
#include "spmacc/particles/attributes/Position.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/Identifier.hpp>

#include <cstdint>

/** Kernel that checks postion is 1 from all frames in assigned particle regions
 * Each block processes all frames in a region and outputs partial sum
 */
struct CheckParticlePos
{
    template<typename T_Worker, typename T_PRBox>
    HDINLINE constexpr auto operator()(
        T_Worker const& worker,
        T_PRBox prDeviceBox,
        int numParticleRegions,
        int* d_errorCount) const
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
                        if(*particle[spearhed::multiMask])
                        {
                            if(!particle[spearhed::pos].get().isApprox(1.f))
                            {
                                // Increment error counter on device
                                alpaka::atomicAdd(worker.getAcc(), d_errorCount, 1, alpaka::hierarchy::Blocks{});
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
        auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>>("PRBuf");

        pmacc::HostDeviceBuffer<int, 1> errorBuffer(1u);
        errorBuffer.getHostBuffer().setValue(0);
        errorBuffer.hostToDevice();

        PMACC_LOCKSTEP_KERNEL(CheckParticlePos{})
            .config<threadsPerBlock>(pmacc::DataSpace<DIM1>(
                numBlocks))(prBuf.getDeviceDataBox(), prBuf.size, errorBuffer.getDeviceBuffer().data());

        errorBuffer.deviceToHost();
        int const totalErrors = errorBuffer.getHostBuffer().data()[0];

        INFO("Number of particles with incorrect positions: " << totalErrors);
        REQUIRE(totalErrors == 0);
    }
};
