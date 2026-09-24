/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spearhed/param/setup.hpp"
#include "spmacc/particles/algorithms/FrameSchedule.hpp"
#include "spmacc/particles/algorithms/HierarchyForEach.hpp"
#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"

#include <pmacc/memory/shared/Allocate.hpp>

namespace spearhed::mapping_setups
{
    /** @brief Block-uniform region trigger based on the maximum particle speed. */
    struct MaximumSpeedExceeds
    {
        Real threshold;

        DINLINE bool shouldSplitRegion(auto const& worker, auto const& region) const
        {
            using namespace pmacc::spearhed;
            PMACC_ASSERT(worker.numWorkers() <= UpdateRegionBounds::MaxBlockSize);
            Real localMaximum = Real{0};
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
                        {
                            Real speedSquared = Real{0};
                            for_each_tag<CS>(
                                [&](auto tag)
                                {
                                    speedSquared
                                        += particle[spearhed::tags::vel][tag] * particle[spearhed::tags::vel][tag];
                                });
                            if(speedSquared > localMaximum)
                                localMaximum = speedSquared;
                        });
                });

            PMACC_SMEM(worker, maxima, Real[UpdateRegionBounds::MaxBlockSize]);
            maxima[worker.workerIdx()] = localMaximum;
            worker.sync();
            for(uint32_t stride = worker.numWorkers() / 2u; stride > 0u; stride >>= 1u)
            {
                if(worker.workerIdx() < stride && maxima[worker.workerIdx() + stride] > maxima[worker.workerIdx()])
                    maxima[worker.workerIdx()] = maxima[worker.workerIdx() + stride];
                worker.sync();
            }
            return maxima[0] > threshold * threshold;
        }
    };

    struct FastSlowPartition
    {
        Real threshold;
        static constexpr uint32_t partitionCount = 2u;

        HDINLINE uint32_t operator()(auto particle, auto const&) const
        {
            Real speedSquared = Real{0};
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                { speedSquared += particle[spearhed::tags::vel][tag] * particle[spearhed::tags::vel][tag]; });
            return speedSquared < threshold * threshold ? 0u : 1u;
        }
    };

    struct VelocitySplitTrigger
    {
        auto operator()(auto const&) const
        {
            return MaximumSpeedExceeds{Real{0.1f}};
        }
    };

    struct VelocitySplitPartition
    {
        auto operator()(auto const&) const
        {
            return FastSlowPartition{Real{0.1f}};
        }
    };

    /** @brief Maximum-speed-triggered binary partition into slow and fast particles. */
    struct AdaptiveVelocitySplitSetup : DefaultSetup
    {
        using DecompositionGroups = std::tuple<pmacc::spearhed::AdaptiveSplitDecompositionGroup<
            VelocitySplitTrigger,
            VelocitySplitPartition,
            pmacc::spearhed::species::Default,
            pmacc::spearhed::species::Boundary,
            pmacc::spearhed::species::Tracer>>;
    };
} // namespace spearhed::mapping_setups
