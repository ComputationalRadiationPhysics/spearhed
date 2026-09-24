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

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace pmacc::spearhed
{
    /**
     * @brief Immutable mapping handle returned by decomposition preparation.
     *
     * A prepared region set is a lightweight handle, not a copy of particle
     * storage. It identifies one mapping generation and keeps the metadata that
     * generation needs alive until dependent interaction plans have completed.
     */
    template<typename T>
    concept PreparedRegionSet = requires(std::remove_cvref_t<T> const& prepared, uint32_t localBucketSlot) {
        typename std::remove_cvref_t<T>::MappingTag;
        typename std::remove_cvref_t<T>::Store;
        { prepared.generation() } -> std::convertible_to<uint64_t>;
        { prepared.bucketCount() } -> std::convertible_to<uint32_t>;
        prepared.chart(localBucketSlot);
        { prepared.deviceView() };
        { prepared.store() } -> std::same_as<typename std::remove_cvref_t<T>::Store&>;
        prepared.assertCurrent();
        requires std::is_trivially_copyable_v<decltype(prepared.deviceView())>;
    };

    /** @brief A decomposition lifecycle object usable with one of its attached stores. */
    template<typename D, typename Store>
    concept SpatialDecompositionFor = requires(D& decomposition, Store& store) {
        decomposition.prepareAfterMotion();
        { decomposition.preparedFor(store) } -> PreparedRegionSet;
    };
} // namespace pmacc::spearhed
