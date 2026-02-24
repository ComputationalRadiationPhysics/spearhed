#pragma once

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/View.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        /**
         * Computes frame counts per region.
         */
        struct CountFramesKernel
        {
            template<typename RegionBox, typename ScanBox>
            DINLINE constexpr auto operator()(
                auto const& worker,
                RegionBox prDeviceBox,
                int numRegions,
                ScanBox framesPerRegion) const
            {
                auto const blockIdx = worker.blockDomIdx();
                if(blockIdx >= numRegions)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]()
                    {
                        auto& region = prDeviceBox[blockIdx];
                        auto& frameList = region.particleFrameList;
                        framesPerRegion[blockIdx] = frameList.size();
                    });
            }
        };

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
                PMACC_ASSERT(blockIdx < framesScanBox[numRegions - 1]);

                // Binary Search to find the region for this block
                int left = 0;
                int right = numRegions;

                while(left < right)
                {
                    int const mid = left + (right - left) / 2;
                    if(static_cast<int>(framesScanBox[mid]) <= blockIdx)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }

                int const regionIdx = left;

                PMACC_ASSERT(regionIdx < numRegions);

                auto& region = prDeviceBox[regionIdx];
                auto& frameList = region.particleFrameList;
                using FrameType = typename std::remove_reference_t<decltype(frameList)>::FrameType;

                auto const startFrame = (regionIdx == 0) ? 0 : framesScanBox[regionIdx - 1];
                auto const localFrameIdx = blockIdx - startFrame;

                PMACC_SMEM(worker, framePtr, pmacc::spearhed::memory::FramePointer<FrameType>);
                auto onlyMaster = pmacc::lockstep::makeMaster(worker);

                onlyMaster(
                    [&]()
                    {
                        auto itr = frameList.begin();
                        for(int i = 0; i < static_cast<int>(localFrameIdx); i++)
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
            if(prBuf.size == 0)
                return;

            constexpr uint32_t threadsPerBlock = 32;

            // Calculate Counts
            pmacc::HostDeviceBuffer<unsigned int, DIM1> framesPerRegion(pmacc::DataSpace<DIM1>{prBuf.size});

            PMACC_LOCKSTEP_KERNEL(detail::CountFramesKernel{})
                .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(
                    prBuf.size))(prBuf.getDeviceDataBox(), prBuf.size, framesPerRegion.getDeviceBuffer().getDataBox());

            // Host Scan (Inclusive Prefix Sum)
            // Note: Could be replaced with an on device scan
            framesPerRegion.deviceToHost();
            auto hostData = framesPerRegion.getHostBuffer().getDataBox();
            for(int i = 1; i < prBuf.size; ++i)
            {
                hostData[i] += hostData[i - 1];
            }
            uint32_t const totalBlocks = hostData[prBuf.size - 1];
            framesPerRegion.hostToDevice();

            // Call Functor with a block for each
            if(totalBlocks > 0)
            {
                PMACC_LOCKSTEP_KERNEL(detail::ProcessSlotsKernel<detail::OccupiedSlot>{})
                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(totalBlocks))(
                        prBuf.getDeviceDataBox(),
                        prBuf.size,
                        framesPerRegion.getDeviceBuffer().getDataBox(),
                        fn,
                        args...);

                // to ensure framesPerRegion doesnt go out of scope before processSlots is finished
                pmacc::eventSystem::waitForAllTasks();
            }
        }
    };
} // namespace pmacc::spearhed
