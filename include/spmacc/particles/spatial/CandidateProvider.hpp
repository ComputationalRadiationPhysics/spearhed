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

#include "spmacc/particles/regions/RegionCandidate.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace pmacc::spearhed
{
    /** @brief Structural contract for values yielded by a candidate provider. */
    template<typename T>
    concept RegionCandidateValue = requires(T const& candidate) {
        { candidate.sourceBucketSlot } -> std::convertible_to<uint32_t>;
        candidate.sourceChartOffsetInTarget;
        { candidate.isUnshiftedImage } -> std::convertible_to<bool>;
    };

    namespace detail
    {
        struct CandidateProbe
        {
            template<RegionCandidateValue Candidate>
            HDINLINE void operator()(Candidate const&) const
            {
            }
        };
    } // namespace detail

    /**
     * @brief Compact device-side candidate enumerator.
     *
     * Every worker in a block must invoke forEachCandidate with the same target
     * slot. Implementations must enumerate candidates in block-uniform control
     * flow because a callback may contain block barriers.
     */
    template<typename T>
    concept CandidateProvider = std::is_trivially_copyable_v<T> && requires(T const& provider, uint32_t targetSlot) {
        provider.forEachCandidate(targetSlot, detail::CandidateProbe{});
    };
} // namespace pmacc::spearhed
