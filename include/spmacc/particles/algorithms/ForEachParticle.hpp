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

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/View.hpp"
#include "spmacc/particles/algorithms/FrameDispatch.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<typename ValidParticlePredicate>
        struct ProcessSlotsKernel
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto prDeviceBox,
                int numRegions,
                auto framesScanBox,
                auto processFn,
                auto... args) const
            {
                auto const blockIdx = worker.blockDomIdx();
                // This kernel should only be called with exactly as many blocks as total frames
                PMACC_ASSERT(blockIdx < static_cast<int>(framesScanBox[numRegions - 1]));

                auto const loc = findFrameLocation(blockIdx, framesScanBox, numRegions);

                PMACC_ASSERT(loc.regionIdx < numRegions);

                auto& region = prDeviceBox[loc.regionIdx];
                auto& frameList = region.particleFrameList;
                using FrameType = typename std::remove_reference_t<decltype(frameList)>::FrameType;

                PMACC_SMEM(worker, framePtr, pmacc::spearhed::memory::FramePointer<FrameType>);
                auto onlyMaster = pmacc::lockstep::makeMaster(worker);

                onlyMaster(
                    [&]()
                    {
                        auto itr = frameList.begin();
                        for(int i = 0; i < loc.localFrameIdx; i++)
                        {
                            ++itr;
                        }
                        framePtr = memory::FramePointer{&(*itr)};
                    });

                worker.sync();

                auto forEachSlot = pmacc::lockstep::makeForEach<FrameType::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t const idx)
                    {
                        auto particle = framePtr[idx];
                        bool const isValid = ValidParticlePredicate{}(particle);
                        if(isValid)
                        {
                            // Call user functor
                            processFn(worker, particle, args...);
                        }
                    });
            }
        };

        struct OccupiedSlot
        {
            constexpr bool operator()(auto const& particle)
            {
                return *particle[tags::multiMask];
            }
        };

    } // namespace detail

    /**
     * Host Helper for to launch a kernel calling a functor for each particle
     * launches as many blocks as frames, by first doing a scan of particle regions
     */
    struct ForEachParticleInPRBuf
    {
        /**
         * @param prBuf The particle region buffer
         * @param fn Functor to call for each particle
         */
        void operator()(auto& prBuf, auto fn, auto&&... args) const
        {
            ForEachFrameInPRBuf<32, 32>{}(
                prBuf,
                detail::ProcessSlotsKernel<detail::OccupiedSlot>{},
                fn,
                std::forward<decltype(args)>(args)...);
        }
    };
} // namespace pmacc::spearhed
