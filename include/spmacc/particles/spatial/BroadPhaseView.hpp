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

#include "spmacc/particles/spatial/PreparedRegionSet.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace pmacc::spearhed
{
    /** @brief Device-copyable world-space geometry of prepared mapping buckets. */
    template<typename T>
    concept BroadPhaseView = std::is_trivially_copyable_v<T> && requires(T const& view, uint32_t slot) {
        { view.bucketCount() } -> std::convertible_to<uint32_t>;
        view.chart(slot);
        view.bounds(slot);
    };

    /** @brief A prepared mapping whose buckets expose broad-phase geometry. */
    template<typename T>
    concept BroadPhasePreparedRegionSet = PreparedRegionSet<T> && requires(T const& prepared) {
        { prepared.broadPhaseView() } -> BroadPhaseView;
    };

    /** @brief Broad-phase adapter for material-style region storage. */
    template<typename T_RegionBox>
    struct MaterialAabbBroadPhaseView
    {
        T_RegionBox regions;
        uint32_t count;

        [[nodiscard]] HDINLINE uint32_t bucketCount() const
        {
            return count;
        }

        [[nodiscard]] HDINLINE decltype(auto) chart(uint32_t slot) const
        {
            return (regions[static_cast<int>(slot)].spatial.chart);
        }

        [[nodiscard]] HDINLINE decltype(auto) bounds(uint32_t slot) const
        {
            return (regions[static_cast<int>(slot)].spatial.occupancy);
        }
    };
} // namespace pmacc::spearhed
