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

#include "spmacc/particles/spatial/CsrCandidateProvider.hpp"
#include "spmacc/particles/spatial/InteractionEntry.hpp"

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <utility>

namespace pmacc::spearhed
{
    /**
     * @brief Compatibility entry for a self-store CSR interaction.
     *
     * New code should use InteractionEntry with an explicit provider. This wrapper
     * preserves the former three-argument construction used by low-level tests;
     * its target and source stores are necessarily the same.
     */
    template<typename T_SourceStore>
    struct NeighbourEntry : InteractionEntry<T_SourceStore, CsrCandidateProvider<T_SourceStore, T_SourceStore>>
    {
        using Provider = CsrCandidateProvider<T_SourceStore, T_SourceStore>;
        using Base = InteractionEntry<T_SourceStore, Provider>;
        using Species = typename Base::Species;

        NeighbourEntry(
            T_SourceStore* sourceStore,
            pmacc::HostDeviceBuffer<unsigned int, DIM1>&& neighbourRegions,
            pmacc::HostDeviceBuffer<unsigned int, DIM1>&& regionOffsets)
            : Base{
                  sourceStore,
                  Provider{sourceStore, sourceStore, std::move(neighbourRegions), std::move(regionOffsets)}}
        {
        }
    };
} // namespace pmacc::spearhed
