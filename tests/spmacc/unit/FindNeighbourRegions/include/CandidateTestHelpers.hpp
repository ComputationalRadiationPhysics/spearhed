/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pmacc::spearhed::test
{
    /**
     * @brief Collect candidate source bucket slots without exposing their storage to tests.
     *
     * This is the compatibility point for spatial correctness tests. The current implementation
     * reads the CSR provider's buffers; other providers can add an overload without changing
     * assertions that consume the returned per-target slot lists.
     *
     * Candidate order is deliberately normalised because provider ordering is not a spatial
     * correctness requirement. The vector-of-vectors result is a test-only snapshot chosen for
     * convenient assertions; production plans should retain flat storage or enumerate candidates
     * through a provider to avoid per-target allocations and pointer chasing.
     */
    template<typename Entry>
    auto collectCandidateSourceSlots(Entry& entry, uint32_t targetBucketCount)
    {
        auto& provider = entry.candidateProvider;
        provider.neighbourRegions.deviceToHost();
        provider.regionOffsets.deviceToHost();

        auto const neighbours = provider.neighbourRegions.getHostBuffer().getDataBox();
        auto const offsets = provider.regionOffsets.getHostBuffer().getDataBox();

        std::vector<std::vector<uint32_t>> result(targetBucketCount);
        for(uint32_t targetSlot = 0; targetSlot < targetBucketCount; ++targetSlot)
        {
            auto& candidates = result[targetSlot];
            for(uint32_t edge = offsets[targetSlot]; edge < offsets[targetSlot + 1u]; ++edge)
                candidates.push_back(neighbours[edge]);
            std::ranges::sort(candidates);
        }
        return result;
    }

    /** @brief Count all directed target-to-source candidate region pairs. */
    inline uint64_t countCandidateRegionPairs(std::vector<std::vector<uint32_t>> const& candidates)
    {
        uint64_t result = 0u;
        for(auto const& perTarget : candidates)
            result += perTarget.size();
        return result;
    }
} // namespace pmacc::spearhed::test
