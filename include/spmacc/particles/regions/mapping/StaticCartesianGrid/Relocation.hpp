/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/mapping/PackedRepartition.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/eventSystem/eventSystem.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace pmacc::spearhed
{
    namespace detail
    {
        /** Translate a particle position from the chart implied by an existing fixed-grid bucket. */
        template<CoordinateSystem CS>
        struct FixedGridSourceChart
        {
            HDINLINE constexpr auto toWorld(
                FixedCartesianGrid<CS> const& grid,
                uint32_t sourceSlot,
                auto const& relativePosition) const
            {
                return grid.toWorld(sourceSlot, relativePosition);
            }
        };

        /** Translate a particle position from a setup material bucket's saved chart. */
        template<typename T_SourceCharts>
        struct MaterialSourceChart
        {
            T_SourceCharts charts;

            template<CoordinateSystem CS>
            HDINLINE constexpr auto toWorld(
                FixedCartesianGrid<CS> const& /* grid */,
                uint32_t sourceSlot,
                auto const& relativePosition) const
            {
                return charts[sourceSlot].toWorld(relativePosition);
            }
        };

        /** Count the destination cell of every live source particle without mutating storage. */
        template<typename T_Frame, CoordinateSystem CS, typename T_SourceChart>
        struct CountFixedCartesianDestinations
        {
            DINLINE void operator()(
                auto const& worker,
                auto sourceFramePtrs,
                auto sourceRegionIndices,
                uint32_t totalSourceFrames,
                T_SourceChart sourceChart,
                FixedCartesianGrid<CS> grid,
                auto destinationCounts,
                auto invalidDestination) const
            {
                uint32_t const frameIndex = worker.blockDomIdx();
                if(frameIndex >= totalSourceFrames)
                    return;

                pmacc::spearhed::memory::FramePointer<T_Frame> const sourceFrame{sourceFramePtrs[frameIndex]};
                uint32_t const sourceSlot = sourceRegionIndices[frameIndex];
                auto const forEachSlot = pmacc::lockstep::makeForEach<T_Frame::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t slot)
                    {
                        auto particle = sourceFrame[slot];
                        if(!particle[tags::multiMask])
                            return;

                        typename FixedCartesianGrid<CS>::Location location{};
                        auto const worldPosition
                            = sourceChart.toWorld(grid, sourceSlot, particle[tags::relativePos].get());
                        if(!grid.locate(worldPosition, location))
                        {
                            alpaka::atomicAdd(
                                worker.getAcc(),
                                &invalidDestination[0],
                                1u,
                                ::alpaka::hierarchy::Blocks{});
                            return;
                        }
                        alpaka::atomicAdd(
                            worker.getAcc(),
                            &destinationCounts[location.slot],
                            1u,
                            ::alpaka::hierarchy::Blocks{});
                    });
            }
        };

        /** Copy whole particle records into packed destination slots and rebase positions. */
        template<typename T_Frame, CoordinateSystem CS, typename T_SourceChart>
        struct ScatterFixedCartesianParticles
        {
            DINLINE void operator()(
                auto const& worker,
                auto sourceFramePtrs,
                auto sourceRegionIndices,
                uint32_t totalSourceFrames,
                T_SourceChart sourceChart,
                FixedCartesianGrid<CS> grid,
                auto destinationFramePtrs,
                auto destinationFrameScan,
                auto destinationCursors) const
            {
                uint32_t const sourceFrameIndex = worker.blockDomIdx();
                if(sourceFrameIndex >= totalSourceFrames)
                    return;

                pmacc::spearhed::memory::FramePointer<T_Frame> const sourceFrame{sourceFramePtrs[sourceFrameIndex]};
                uint32_t const sourceSlot = sourceRegionIndices[sourceFrameIndex];
                auto const forEachSlot = pmacc::lockstep::makeForEach<T_Frame::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t sourceParticleSlot)
                    {
                        auto sourceParticle = sourceFrame[sourceParticleSlot];
                        if(!sourceParticle[tags::multiMask])
                            return;

                        typename FixedCartesianGrid<CS>::Location location{};
                        auto const worldPosition
                            = sourceChart.toWorld(grid, sourceSlot, sourceParticle[tags::relativePos].get());
                        bool const located = grid.locate(worldPosition, location);
                        PMACC_ASSERT(located);
                        if(!located)
                            return;

                        uint32_t const destinationParticleIndex = alpaka::atomicAdd(
                            worker.getAcc(),
                            &destinationCursors[location.slot],
                            1u,
                            ::alpaka::hierarchy::Blocks{});
                        uint32_t const firstFrame
                            = location.slot == 0u ? 0u : destinationFrameScan[location.slot - 1u];
                        pmacc::spearhed::memory::FramePointer<T_Frame> const destinationFrame{
                            destinationFramePtrs[firstFrame + destinationParticleIndex / T_Frame::frameSize]};
                        auto destinationParticle = destinationFrame[destinationParticleIndex % T_Frame::frameSize];

                        sourceParticle.deepCopyTo(destinationParticle);
                        pmacc::spearhed::for_each_tag<CS>(
                            [&](auto tag)
                            { destinationParticle[tags::relativePos][tag] = location.localPosition[tag]; });
                        destinationParticle[tags::multiMask] = 1u;
                    });
            }
        };

        template<typename T_Region>
        [[nodiscard]] ParticleRegionBuffer<T_Region> copyRegionsForReplacement(ParticleRegionBuffer<T_Region>& source)
        {
            ParticleRegionBuffer<T_Region> replacement;
            replacement.create(static_cast<size_t>(source.size));
            source.buffer->deviceToHost();
            pmacc::eventSystem::getTransactionEvent().waitForFinished();
            auto const regions = source.buffer->getHostBuffer().getDataBox();
            for(int slot = 0; slot < source.size; ++slot)
                replacement.pushBack(regions[slot]);
            replacement.buffer->hostToDevice();
            return replacement;
        }

    } // namespace detail

    /**
     * @brief Bulk single-accelerator relocation for a fixed dense Cartesian grid.
     *
     * The source store remains intact while a copied-metadata replacement receives exactly packed
     * frame chains and all scattered records. The replacement is published only after cursor
     * validation, so allocation or scatter failure cannot discard source particles.
     */
    template<CoordinateSystem CS>
    class FixedCartesianRelocator
    {
    public:
        explicit FixedCartesianRelocator(FixedCartesianGrid<CS> grid) : m_grid(grid)
        {
        }

        template<typename T_Region>
        void relocate(ParticleRegionBuffer<T_Region>& store) const
        {
            using Frame = typename T_Region::FrameType;
            using Repartition = packed_repartition::PackedRepartition<T_Region>;
            constexpr uint32_t threadsPerBlock = Frame::frameSize;

            if(store.size == 0)
                return;
            if(static_cast<uint32_t>(store.size) != m_grid.bucketCount())
                throw std::invalid_argument("fixed Cartesian relocation requires one dense bucket per grid cell");

            FrameIndexBuffer<T_Region> sourceIndex{store};
            if(sourceIndex.totalFrames == 0u)
                return;

            pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCounts{
                pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
            destinationCounts.getHostBuffer().setValue(0u);
            destinationCounts.hostToDevice();
            pmacc::HostDeviceBuffer<uint32_t, DIM1> invalidDestination{pmacc::DataSpace<DIM1>{1}};
            invalidDestination.getHostBuffer().setValue(0u);
            invalidDestination.hostToDevice();

            using SourceChart = detail::FixedGridSourceChart<CS>;
            auto countDone = PMACC_LOCKSTEP_KERNEL(detail::CountFixedCartesianDestinations<Frame, CS, SourceChart>{})
                                 .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                     sourceIndex.framePtrsBox(),
                                     sourceIndex.regionIdxBox(),
                                     sourceIndex.totalFrames,
                                     SourceChart{},
                                     m_grid,
                                     destinationCounts.getDeviceBuffer().getDataBox(),
                                     invalidDestination.getDeviceBuffer().getDataBox());
            countDone.waitForFinished();
            invalidDestination.deviceToHost();
            if(invalidDestination.getHostBuffer().getDataBox()[0] != 0u)
                throw std::out_of_range("particle left a non-periodic fixed Cartesian domain");

            auto replacement = detail::copyRegionsForReplacement(store);
            Repartition::allocate(replacement, destinationCounts);
            try
            {
                FrameIndexBuffer<T_Region> destinationIndex{replacement};
                pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCursors{
                    pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
                destinationCursors.getHostBuffer().setValue(0u);
                destinationCursors.hostToDevice();

                auto scatterDone
                    = PMACC_LOCKSTEP_KERNEL(detail::ScatterFixedCartesianParticles<Frame, CS, SourceChart>{})
                          .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                              sourceIndex.framePtrsBox(),
                              sourceIndex.regionIdxBox(),
                              sourceIndex.totalFrames,
                              SourceChart{},
                              m_grid,
                              destinationIndex.framePtrsBox(),
                              destinationIndex.scanBox(),
                              destinationCursors.getDeviceBuffer().getDataBox());
                scatterDone.waitForFinished();
                packed_repartition::validateDestinationCursors(
                    destinationCursors,
                    destinationCounts,
                    m_grid.bucketCount(),
                    "fixed Cartesian relocation lost or duplicated a particle");
            }
            catch(...)
            {
                Repartition::discard(replacement);
                throw;
            }
            Repartition::publish(store, replacement);
        }

    private:
        FixedCartesianGrid<CS> m_grid;
    };

    /** Convert setup-created material buckets into a dense fixed grid transactionally. */
    template<CoordinateSystem CS>
    class FixedCartesianInitialClassifier
    {
    public:
        explicit FixedCartesianInitialClassifier(FixedCartesianGrid<CS> grid) : m_grid(grid)
        {
        }

        template<typename T_Region, typename T_AllocatorHandle>
        void classify(ParticleRegionBuffer<T_Region>& store, T_AllocatorHandle const& allocatorHandle) const
        {
            using Frame = typename T_Region::FrameType;
            using Chart = RegionChart<CS>;
            using Repartition = packed_repartition::PackedRepartition<T_Region>;
            constexpr uint32_t threadsPerBlock = Frame::frameSize;

            FrameIndexBuffer<T_Region> sourceIndex{store};
            uint32_t const oldRegionCount = static_cast<uint32_t>(store.size);
            pmacc::HostDeviceBuffer<Chart, DIM1> sourceCharts{
                pmacc::DataSpace<DIM1>{static_cast<int>(oldRegionCount)}};
            if(oldRegionCount > 0u)
            {
                store.buffer->deviceToHost();
                pmacc::eventSystem::getTransactionEvent().waitForFinished();
                auto const oldRegions = store.buffer->getHostBuffer().getDataBox();
                auto charts = sourceCharts.getHostBuffer().getDataBox();
                for(uint32_t slot = 0u; slot < oldRegionCount; ++slot)
                    charts[slot] = oldRegions[static_cast<int>(slot)].spatial.chart;
                sourceCharts.hostToDevice();
            }
            auto sourceChart = detail::MaterialSourceChart{sourceCharts.getDeviceBuffer().getDataBox()};

            pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCounts{
                pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
            destinationCounts.getHostBuffer().setValue(0u);
            destinationCounts.hostToDevice();
            pmacc::HostDeviceBuffer<uint32_t, DIM1> invalidDestination{pmacc::DataSpace<DIM1>{1}};
            invalidDestination.getHostBuffer().setValue(0u);
            invalidDestination.hostToDevice();

            if(sourceIndex.totalFrames > 0u)
            {
                auto countDone
                    = PMACC_LOCKSTEP_KERNEL(
                          detail::CountFixedCartesianDestinations<Frame, CS, decltype(sourceChart)>{})
                          .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                              sourceIndex.framePtrsBox(),
                              sourceIndex.regionIdxBox(),
                              sourceIndex.totalFrames,
                              sourceChart,
                              m_grid,
                              destinationCounts.getDeviceBuffer().getDataBox(),
                              invalidDestination.getDeviceBuffer().getDataBox());
                countDone.waitForFinished();
                invalidDestination.deviceToHost();
                if(invalidDestination.getHostBuffer().getDataBox()[0] != 0u)
                    throw std::out_of_range("initial particle lies outside a non-periodic fixed Cartesian domain");
            }

            typename Repartition::Replacement replacement;
            replacement.create(m_grid.bucketCount());
            for(uint32_t slot = 0u; slot < m_grid.bucketCount(); ++slot)
            {
                typename T_Region::VolumeType spatial{};
                spatial.chart = m_grid.chart(slot);
                replacement.pushBack(T_Region{allocatorHandle, spatial});
            }
            replacement.buffer->hostToDevice();

            Repartition::allocate(replacement, destinationCounts);
            try
            {
                if(sourceIndex.totalFrames > 0u)
                {
                    FrameIndexBuffer<T_Region> destinationIndex{replacement};
                    pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCursors{
                        pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
                    destinationCursors.getHostBuffer().setValue(0u);
                    destinationCursors.hostToDevice();
                    auto scatterDone
                        = PMACC_LOCKSTEP_KERNEL(
                              detail::ScatterFixedCartesianParticles<Frame, CS, decltype(sourceChart)>{})
                              .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                  sourceIndex.framePtrsBox(),
                                  sourceIndex.regionIdxBox(),
                                  sourceIndex.totalFrames,
                                  sourceChart,
                                  m_grid,
                                  destinationIndex.framePtrsBox(),
                                  destinationIndex.scanBox(),
                                  destinationCursors.getDeviceBuffer().getDataBox());
                    scatterDone.waitForFinished();
                    packed_repartition::validateDestinationCursors(
                        destinationCursors,
                        destinationCounts,
                        m_grid.bucketCount(),
                        "fixed Cartesian initial classification lost or duplicated a particle");
                }
            }
            catch(...)
            {
                Repartition::discard(replacement);
                throw;
            }
            Repartition::publish(store, replacement);
        }

    private:
        FixedCartesianGrid<CS> m_grid;
    };
} // namespace pmacc::spearhed
