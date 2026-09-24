/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/spatial/BroadPhaseView.hpp"
#include "spmacc/particles/spatial/CandidateProvider.hpp"
#include "spmacc/particles/spatial/InteractionQuery.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace pmacc::spearhed
{
    /** @brief Device view of CSR candidates backed by prepared broad-phase geometry. */
    template<typename T_TargetView, typename T_SourceView, typename T_IndexBox>
    struct MaterializedCsrCandidateProviderView
    {
        T_TargetView target;
        T_SourceView source;
        T_IndexBox neighbourRegions;
        T_IndexBox regionOffsets;

        template<typename Fn>
        DINLINE void forEachCandidate(uint32_t targetBucketSlot, Fn&& fn) const
        {
            uint32_t const begin = regionOffsets[targetBucketSlot];
            uint32_t const end = regionOffsets[targetBucketSlot + 1u];
            auto const targetOrigin = target.chart(targetBucketSlot).origin;
            for(uint32_t index = begin; index < end; ++index)
            {
                uint32_t const sourceBucketSlot = neighbourRegions[index];
                fn(RegionCandidate{sourceBucketSlot, source.chart(sourceBucketSlot).origin - targetOrigin, true});
            }
        }
    };

    /** @brief Owning CSR fallback for any pair of broad-phase prepared mappings. */
    template<BroadPhaseView T_TargetView, BroadPhaseView T_SourceView>
    struct MaterializedCsrCandidateProvider
    {
        T_TargetView target;
        T_SourceView source;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets;

        [[nodiscard]] CandidateProvider auto deviceView() const
        {
            return MaterializedCsrCandidateProviderView{
                target,
                source,
                neighbourRegions.getDeviceBuffer().getDataBox(),
                regionOffsets.getDeviceBuffer().getDataBox()};
        }
    };

    /**
     * @brief Generic fallback selected when a mapping pair exposes broad-phase geometry.
     *
     * Mapping-specific overloads use their concrete tag pair and are preferred over
     * this unconstrained tag fallback.
     */
    template<
        typename T_TargetTag,
        typename T_SourceTag,
        BroadPhasePreparedRegionSet T_Target,
        typename T_Radius,
        BroadPhasePreparedRegionSet T_Source>
    [[nodiscard]] auto makeCandidateProviderImpl(
        T_TargetTag,
        T_SourceTag,
        T_Target const& target,
        InteractionQuery<T_Radius> const& query,
        T_Source const& source)
    {
        auto targetView = target.broadPhaseView();
        auto sourceView = source.broadPhaseView();
        // Fixed/fixed providers materialize periodic images implicitly. The generic CSR fallback
        // cannot yet do so, so reject mixed periodic sources rather than omit candidates.
        if constexpr(requires { sourceView.hasPeriodicImages(); })
            if(sourceView.hasPeriodicImages())
                throw std::invalid_argument("mixed periodic mappings require periodic CSR image support");
        auto data = materializeNeighbourCsr(targetView, sourceView, query.radius);
        return MaterializedCsrCandidateProvider{
            targetView,
            sourceView,
            std::move(data.neighbourRegions),
            std::move(data.regionOffsets)};
    }
} // namespace pmacc::spearhed
