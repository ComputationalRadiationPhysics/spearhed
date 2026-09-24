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

#include "spmacc/particles/spatial/CandidateProvider.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>

namespace pmacc::spearhed
{
    /** @brief Trivially-copyable device view of CSR candidate data. */
    template<typename T_TargetBox, typename T_SourceBox, typename T_IndexBox>
    struct CsrCandidateProviderView
    {
        T_TargetBox targetRegions;
        T_SourceBox sourceRegions;
        T_IndexBox neighbourRegions;
        T_IndexBox regionOffsets;

        template<typename Fn>
        DINLINE void forEachCandidate(uint32_t targetBucketSlot, Fn&& fn) const
        {
            uint32_t const begin = regionOffsets[targetBucketSlot];
            uint32_t const end = regionOffsets[targetBucketSlot + 1u];
            auto const targetOrigin = targetRegions[targetBucketSlot].spatial.chart.origin;

            for(uint32_t idx = begin; idx < end; ++idx)
            {
                uint32_t const sourceBucketSlot = neighbourRegions[idx];
                auto const sourceOrigin = sourceRegions[sourceBucketSlot].spatial.chart.origin;
                fn(RegionCandidate{sourceBucketSlot, sourceOrigin - targetOrigin, true});
            }
        }
    };

    /**
     * @brief Owning host-side CSR candidate provider.
     *
     * Region buffers are non-owning references and must outlive queued interaction
     * kernels. The two CSR buffers are owned by this provider.
     */
    template<typename T_TargetStore, typename T_SourceStore>
    struct CsrCandidateProvider
    {
        T_TargetStore* targetStore;
        T_SourceStore* sourceStore;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets;

        CandidateProvider auto deviceView() const
        {
            return CsrCandidateProviderView{
                targetStore->getDeviceDataBox(),
                sourceStore->getDeviceDataBox(),
                neighbourRegions.getDeviceBuffer().getDataBox(),
                regionOffsets.getDeviceBuffer().getDataBox()};
        }
    };
} // namespace pmacc::spearhed
