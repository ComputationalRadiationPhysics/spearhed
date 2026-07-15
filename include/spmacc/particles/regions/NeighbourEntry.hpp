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

#include "spmacc/particles/regions/RegionRole.hpp"

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

namespace pmacc::spearhed
{
    /**
     * @brief Compact device-side view of one source's neighbour data.
     *
     * Passed directly as a single kernel argument, replacing the three separate
     * device boxes (sourcePRDeviceBox, neighbourRegionsBox, regionOffsetsBox).
     */
    template<typename T_SourceDeviceBox, typename T_IdxBox>
    struct SourceView
    {
        T_SourceDeviceBox sourcePRDeviceBox;
        T_IdxBox neighbourRegionsBox;
        T_IdxBox regionOffsetsBox;
        int numSourceRegions;
    };

    /**
     * @brief Host-side record for one source buffer and its pre-computed neighbour lists.
     *
     * A *source* buffer contributes particle data to a *target* region. For a given target
     * region index i, regionOffsets[i]..regionOffsets[i+1] indexes into neighbourRegions,
     * which holds the source-region indices that neighbour target region i.
     *
     * Owns the two neighbour-index buffers; holds a non-owning pointer to the source
     * ParticleRegionBuffer. Species is inherited from the source buffer's Species typedef;
     * the species in turn carries the roles algorithms select on.
     */
    template<typename T_SourcePRBuf>
    struct NeighbourEntry
    {
        using Species = typename T_SourcePRBuf::Species;

        T_SourcePRBuf* sourceBufPtr;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets;

        auto deviceView()
        {
            using SrcBox = decltype(sourceBufPtr->getDeviceDataBox());
            using IdxBox = decltype(neighbourRegions.getDeviceBuffer().getDataBox());
            return SourceView<SrcBox, IdxBox>{
                sourceBufPtr->getDeviceDataBox(),
                neighbourRegions.getDeviceBuffer().getDataBox(),
                regionOffsets.getDeviceBuffer().getDataBox(),
                sourceBufPtr->size};
        }
    };

} // namespace pmacc::spearhed
