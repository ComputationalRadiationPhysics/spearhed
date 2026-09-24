/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"
#include "spmacc/particles/regions/mapping/PackedRepartition.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/eventSystem/eventSystem.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace pmacc::spearhed::adaptive_split
{
    namespace detail
    {
        template<typename T_Trigger>
        struct SelectRegions
        {
            DINLINE void operator()(
                auto const& worker,
                auto regions,
                uint32_t regionCount,
                T_Trigger trigger,
                auto selected) const
            {
                uint32_t const regionSlot = worker.blockDomIdx();
                if(regionSlot >= regionCount)
                    return;
                bool const split = trigger.shouldSplitRegion(worker, regions[regionSlot]);
                pmacc::lockstep::makeMaster(worker)([&] { selected[regionSlot] = split ? 1u : 0u; });
            }
        };

        template<typename T_Frame, typename T_Partition>
        struct CountLabels
        {
            DINLINE void operator()(
                auto const& worker,
                auto sourceFramePtrs,
                auto sourceRegionIndices,
                uint32_t totalSourceFrames,
                auto regions,
                T_Partition partition,
                auto selected,
                auto labelCounts,
                auto invalidLabels) const
            {
                uint32_t const frameIndex = worker.blockDomIdx();
                if(frameIndex >= totalSourceFrames)
                    return;

                uint32_t const parent = sourceRegionIndices[frameIndex];
                if(!selected[parent])
                    return;
                pmacc::spearhed::memory::FramePointer<T_Frame> const frame{sourceFramePtrs[frameIndex]};
                auto const forEachSlot = pmacc::lockstep::makeForEach<T_Frame::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t slot)
                    {
                        auto particle = frame[slot];
                        if(!particle[tags::multiMask])
                            return;
                        uint32_t const label = partition(particle, regions[parent].spatial);
                        PMACC_ASSERT(label < T_Partition::partitionCount);
                        if(label >= T_Partition::partitionCount)
                        {
                            alpaka::atomicAdd(worker.getAcc(), &invalidLabels[0], 1u, ::alpaka::hierarchy::Blocks{});
                            return;
                        }
                        alpaka::atomicAdd(
                            worker.getAcc(),
                            &labelCounts[parent * T_Partition::partitionCount + label],
                            1u,
                            ::alpaka::hierarchy::Blocks{});
                    });
            }
        };

        template<typename T_Frame, typename T_Partition>
        struct Scatter
        {
            DINLINE void operator()(
                auto const& worker,
                auto sourceFramePtrs,
                auto sourceRegionIndices,
                uint32_t totalSourceFrames,
                auto sourceRegions,
                T_Partition partition,
                auto effective,
                auto parentLabelDestination,
                auto destinationFramePtrs,
                auto destinationFrameScan,
                auto destinationCursors) const
            {
                uint32_t const frameIndex = worker.blockDomIdx();
                if(frameIndex >= totalSourceFrames)
                    return;

                uint32_t const parent = sourceRegionIndices[frameIndex];
                pmacc::spearhed::memory::FramePointer<T_Frame> const sourceFrame{sourceFramePtrs[frameIndex]};
                auto const forEachSlot = pmacc::lockstep::makeForEach<T_Frame::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t sourceSlot)
                    {
                        auto sourceParticle = sourceFrame[sourceSlot];
                        if(!sourceParticle[tags::multiMask])
                            return;

                        uint32_t label = 0u;
                        if(effective[parent])
                        {
                            label = partition(sourceParticle, sourceRegions[parent].spatial);
                            PMACC_ASSERT(label < T_Partition::partitionCount);
                            if(label >= T_Partition::partitionCount)
                                return;
                        }
                        uint32_t const destination
                            = parentLabelDestination[parent * T_Partition::partitionCount + label];
                        uint32_t const particleIndex = alpaka::atomicAdd(
                            worker.getAcc(),
                            &destinationCursors[destination],
                            1u,
                            ::alpaka::hierarchy::Blocks{});
                        uint32_t const firstFrame = destination == 0u ? 0u : destinationFrameScan[destination - 1u];
                        pmacc::spearhed::memory::FramePointer<T_Frame> const destinationFrame{
                            destinationFramePtrs[firstFrame + particleIndex / T_Frame::frameSize]};
                        auto destinationParticle = destinationFrame[particleIndex % T_Frame::frameSize];
                        sourceParticle.deepCopyTo(destinationParticle);
                        destinationParticle[tags::multiMask] = 1u;
                    });
            }
        };

        struct Plan
        {
            std::vector<uint32_t> destinationCounts;
            std::vector<uint32_t> destinationParents;
            std::vector<uint32_t> parentLabelDestination;
            std::vector<uint8_t> effective;
        };

        inline Plan makePlan(
            std::vector<uint8_t> const& selected,
            std::vector<uint32_t> const& parentCounts,
            std::vector<uint32_t> const& labelCounts)
        {
            Plan plan;
            uint32_t const parentCount = static_cast<uint32_t>(selected.size());
            plan.parentLabelDestination.resize(parentCount * 2u);
            plan.effective.resize(parentCount);
            for(uint32_t parent = 0u; parent < parentCount; ++parent)
            {
                bool const split
                    = selected[parent] && labelCounts[parent * 2u] != 0u && labelCounts[parent * 2u + 1u] != 0u;
                plan.effective[parent] = split ? 1u : 0u;
                uint32_t const firstDestination = static_cast<uint32_t>(plan.destinationCounts.size());
                plan.parentLabelDestination[parent * 2u] = firstDestination;
                plan.destinationParents.push_back(parent);
                if(split)
                {
                    plan.destinationCounts.push_back(labelCounts[parent * 2u]);
                    plan.parentLabelDestination[parent * 2u + 1u] = firstDestination + 1u;
                    plan.destinationParents.push_back(parent);
                    plan.destinationCounts.push_back(labelCounts[parent * 2u + 1u]);
                }
                else
                {
                    plan.parentLabelDestination[parent * 2u + 1u] = firstDestination;
                    plan.destinationCounts.push_back(parentCounts[parent]);
                }
            }
            return plan;
        }

        template<typename T>
        pmacc::HostDeviceBuffer<T, DIM1> deviceBuffer(std::vector<T> const& values)
        {
            if(values.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::overflow_error("adaptive split buffer exceeds DataSpace index capacity");
            pmacc::HostDeviceBuffer<T, DIM1> buffer{pmacc::DataSpace<DIM1>{static_cast<int>(values.size())}};
            auto host = buffer.getHostBuffer().getDataBox();
            for(size_t i = 0u; i < values.size(); ++i)
                host[static_cast<int>(i)] = values[i];
            buffer.hostToDevice();
            return buffer;
        }

        template<typename T_Region>
        [[nodiscard]] ParticleRegionBuffer<T_Region> makeReplacement(
            ParticleRegionBuffer<T_Region>& source,
            std::vector<uint32_t> const& destinationParents)
        {
            ParticleRegionBuffer<T_Region> replacement;
            replacement.create(destinationParents.size());
            source.buffer->deviceToHost();
            pmacc::eventSystem::getTransactionEvent().waitForFinished();
            auto const parents = source.buffer->getHostBuffer().getDataBox();
            for(uint32_t const parent : destinationParents)
                replacement.pushBack(parents[static_cast<int>(parent)]);
            replacement.buffer->hostToDevice();
            return replacement;
        }
    } // namespace detail

    /** @brief Bulk binary split maintenance with transactional frame replacement. */
    template<typename T_Trigger, typename T_Partition>
    class BinarySplitter
    {
    public:
        static_assert(
            T_Partition::partitionCount == 2u,
            "AdaptiveSplit currently supports binary partition policies only");

        BinarySplitter(T_Trigger trigger, T_Partition partition) : m_trigger(trigger), m_partition(partition)
        {
        }

        template<typename T_Region>
        bool split(ParticleRegionBuffer<T_Region>& store) const
        {
            using Frame = typename T_Region::FrameType;
            using Repartition = packed_repartition::PackedRepartition<T_Region>;
            constexpr uint32_t threadsPerBlock = Frame::frameSize;
            if(store.size < 0)
                throw std::overflow_error("adaptive split received a negative region count");
            uint32_t const parentCount = static_cast<uint32_t>(store.size);
            if(parentCount == 0u)
                return false;
            if(parentCount > std::numeric_limits<uint32_t>::max() / 2u)
                throw std::overflow_error("adaptive split parent count exceeds binary label table capacity");

            updateMaterialAabbBounds(store);
            pmacc::HostDeviceBuffer<uint8_t, DIM1> selected{pmacc::DataSpace<DIM1>{static_cast<int>(parentCount)}};
            selected.getHostBuffer().setValue(0u);
            selected.hostToDevice();
            auto selectedDone = PMACC_LOCKSTEP_KERNEL(detail::SelectRegions<T_Trigger>{})
                                    .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(parentCount))(
                                        store.getDeviceDataBox(),
                                        parentCount,
                                        m_trigger,
                                        selected.getDeviceBuffer().getDataBox());
            selectedDone.waitForFinished();

            FrameIndexBuffer<T_Region> sourceIndex{store};
            pmacc::HostDeviceBuffer<uint32_t, DIM1> labelCounts{
                pmacc::DataSpace<DIM1>{static_cast<int>(parentCount * 2u)}};
            labelCounts.getHostBuffer().setValue(0u);
            labelCounts.hostToDevice();
            pmacc::HostDeviceBuffer<uint32_t, DIM1> invalidLabels{pmacc::DataSpace<DIM1>{1}};
            invalidLabels.getHostBuffer().setValue(0u);
            invalidLabels.hostToDevice();
            if(sourceIndex.totalFrames > 0u)
            {
                auto counted = PMACC_LOCKSTEP_KERNEL(detail::CountLabels<Frame, T_Partition>{})
                                   .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                       sourceIndex.framePtrsBox(),
                                       sourceIndex.regionIdxBox(),
                                       sourceIndex.totalFrames,
                                       store.getDeviceDataBox(),
                                       m_partition,
                                       selected.getDeviceBuffer().getDataBox(),
                                       labelCounts.getDeviceBuffer().getDataBox(),
                                       invalidLabels.getDeviceBuffer().getDataBox());
                counted.waitForFinished();
            }

            invalidLabels.deviceToHost();
            selected.deviceToHost();
            labelCounts.deviceToHost();
            store.buffer->deviceToHost();
            pmacc::eventSystem::getTransactionEvent().waitForFinished();
            if(invalidLabels.getHostBuffer().getDataBox()[0] != 0u)
                throw std::invalid_argument("adaptive partition returned an out-of-range label");

            std::vector<uint8_t> hostSelected(parentCount);
            std::vector<uint32_t> parentCounts(parentCount);
            auto const selectedHost = selected.getHostBuffer().getDataBox();
            auto const parentRegions = store.buffer->getHostBuffer().getDataBox();
            for(uint32_t parent = 0u; parent < parentCount; ++parent)
            {
                hostSelected[parent] = selectedHost[parent];
                parentCounts[parent] = parentRegions[static_cast<int>(parent)].particleFrameList.getNumParticles();
            }
            std::vector<uint32_t> hostLabelCounts(parentCount * 2u);
            auto const labelCountsHost = labelCounts.getHostBuffer().getDataBox();
            for(uint32_t i = 0u; i < parentCount * 2u; ++i)
                hostLabelCounts[i] = labelCountsHost[i];

            auto plan = detail::makePlan(hostSelected, parentCounts, hostLabelCounts);
            bool const hasEffectiveSplit
                = std::find(plan.effective.begin(), plan.effective.end(), uint8_t{1u}) != plan.effective.end();
            if(!hasEffectiveSplit)
                return false;

            auto replacement = detail::makeReplacement(store, plan.destinationParents);
            auto destinationCounts = detail::deviceBuffer(plan.destinationCounts);
            Repartition::allocate(replacement, destinationCounts);
            try
            {
                FrameIndexBuffer<T_Region> destinationIndex{replacement};
                auto effective = detail::deviceBuffer(plan.effective);
                auto parentLabelDestination = detail::deviceBuffer(plan.parentLabelDestination);
                pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCursors{
                    pmacc::DataSpace<DIM1>{static_cast<int>(plan.destinationCounts.size())}};
                destinationCursors.getHostBuffer().setValue(0u);
                destinationCursors.hostToDevice();
                if(sourceIndex.totalFrames > 0u)
                {
                    auto scattered
                        = PMACC_LOCKSTEP_KERNEL(detail::Scatter<Frame, T_Partition>{})
                              .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                  sourceIndex.framePtrsBox(),
                                  sourceIndex.regionIdxBox(),
                                  sourceIndex.totalFrames,
                                  store.getDeviceDataBox(),
                                  m_partition,
                                  effective.getDeviceBuffer().getDataBox(),
                                  parentLabelDestination.getDeviceBuffer().getDataBox(),
                                  destinationIndex.framePtrsBox(),
                                  destinationIndex.scanBox(),
                                  destinationCursors.getDeviceBuffer().getDataBox());
                    scattered.waitForFinished();
                }
                packed_repartition::validateDestinationCursors(
                    destinationCursors,
                    destinationCounts,
                    static_cast<uint32_t>(plan.destinationCounts.size()),
                    "adaptive split lost or duplicated a particle");
            }
            catch(...)
            {
                Repartition::discard(replacement);
                throw;
            }
            Repartition::publish(store, replacement);
            updateMaterialAabbBounds(store);
            return true;
        }

    private:
        T_Trigger m_trigger;
        T_Partition m_partition;
    };
} // namespace pmacc::spearhed::adaptive_split
