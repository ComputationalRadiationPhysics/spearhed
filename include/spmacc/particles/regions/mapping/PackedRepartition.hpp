/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/eventSystem/eventSystem.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace pmacc::spearhed::packed_repartition
{
    namespace detail
    {
        /** Prepare independent, packed destination chains without touching source storage. */
        template<typename T_Frame>
        struct AllocatePackedFrameLists
        {
            DINLINE void operator()(
                auto const& worker,
                auto regions,
                uint32_t regionCount,
                auto destinationCounts,
                auto allocationFailed) const
            {
                uint32_t const regionSlot = worker.blockDomIdx();
                if(regionSlot >= regionCount)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]
                    {
                        auto& frameList = regions[regionSlot].particleFrameList;
                        // The replacement can be copied from a source region to retain its
                        // allocator handle. Detaching this copy must never affect the source.
                        frameList.detachAllFrames();

                        uint32_t const particleCount = destinationCounts[regionSlot];
                        frameList.setNumParticles(particleCount);
                        uint32_t const frameCount = frameList.numFrames();
                        for(uint32_t frameSlot = 0u; frameSlot < frameCount; ++frameSlot)
                        {
                            auto frame = frameList.getEmptyFrame(worker);
                            if(!frame.operator->())
                            {
                                alpaka::atomicAdd(
                                    worker.getAcc(),
                                    &allocationFailed[0],
                                    1u,
                                    ::alpaka::hierarchy::Blocks{});
                                frameList.setNumParticles(frameSlot * T_Frame::frameSize);
                                break;
                            }
                            uint32_t const remaining = particleCount - frameSlot * T_Frame::frameSize;
                            frame->liveParticles = remaining < T_Frame::frameSize ? remaining : T_Frame::frameSize;
                        }
                    });
            }
        };

        struct DestroyAllFrames
        {
            DINLINE void operator()(auto const& worker, auto regions, uint32_t regionCount) const
            {
                uint32_t const regionSlot = worker.blockDomIdx();
                if(regionSlot >= regionCount)
                    return;
                pmacc::lockstep::makeMaster(worker)(
                    [&] { regions[regionSlot].particleFrameList.destroyAllFrames(worker); });
            }
        };

    } // namespace detail

    inline void validateDestinationCursors(
        pmacc::HostDeviceBuffer<uint32_t, DIM1>& destinationCursors,
        pmacc::HostDeviceBuffer<uint32_t, DIM1>& destinationCounts,
        uint32_t bucketCount,
        char const* failureMessage)
    {
        destinationCursors.deviceToHost();
        destinationCounts.deviceToHost();
        auto const cursors = destinationCursors.getHostBuffer().getDataBox();
        auto const counts = destinationCounts.getHostBuffer().getDataBox();
        for(uint32_t slot = 0u; slot < bucketCount; ++slot)
            if(cursors[slot] != counts[slot])
                throw std::runtime_error(failureMessage);
    }

    /**
     * @brief Transactional frame-storage operations shared by bulk repartition mappings.
     *
     * Callers build and scatter into a separate @c Replacement. Source frames remain owned by
     * @p source until @c publish succeeds; failure cleanup only touches the replacement.
     */
    template<typename T_Region>
    class PackedRepartition
    {
    public:
        using Store = ParticleRegionBuffer<T_Region>;
        using Frame = typename T_Region::FrameType;
        using Replacement = Store;

        static void allocate(Replacement& replacement, pmacc::HostDeviceBuffer<uint32_t, DIM1>& destinationCounts)
        {
            if(replacement.size == 0)
                return;

            pmacc::HostDeviceBuffer<uint32_t, DIM1> allocationFailed{pmacc::DataSpace<DIM1>{1}};
            allocationFailed.getHostBuffer().setValue(0u);
            allocationFailed.hostToDevice();
            auto allocateDone = PMACC_LOCKSTEP_KERNEL(detail::AllocatePackedFrameLists<Frame>{})
                                    .template config<1u>(pmacc::DataSpace<DIM1>(replacement.size))(
                                        replacement.getDeviceDataBox(),
                                        static_cast<uint32_t>(replacement.size),
                                        destinationCounts.getDeviceBuffer().getDataBox(),
                                        allocationFailed.getDeviceBuffer().getDataBox());
            allocateDone.waitForFinished();

            allocationFailed.deviceToHost();
            if(allocationFailed.getHostBuffer().getDataBox()[0] != 0u)
            {
                discard(replacement);
                throw std::runtime_error("packed repartition could not allocate replacement frames");
            }
        }

        /** Release replacement frames after any post-allocation failure. */
        static void discard(Replacement& replacement)
        {
            if(replacement.size == 0)
                return;
            auto clearDone = PMACC_LOCKSTEP_KERNEL(detail::DestroyAllFrames{})
                                 .template config<1u>(pmacc::DataSpace<DIM1>(replacement.size))(
                                     replacement.getDeviceDataBox(),
                                     static_cast<uint32_t>(replacement.size));
            clearDone.waitForFinished();
        }

        /**
         * @brief Publish a fully populated replacement, then release the old owning lists.
         *
         * Source frames are released only after replacement allocation and scatter validation.
         * Draining PMacc tasks before the buffer swap leaves no task referring to the old storage.
         */
        static void publish(Store& source, Replacement& replacement)
        {
            pmacc::eventSystem::waitForAllTasks();
            std::swap(source.buffer, replacement.buffer);
            std::swap(source.size, replacement.size);
            ++source.topologyVersion;
            source.buffer->deviceToHost();
            pmacc::eventSystem::getTransactionEvent().waitForFinished();
            discard(replacement);
        }
    };
} // namespace pmacc::spearhed::packed_repartition
