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

#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"
#include "spmacc/particles/spatial/CandidateProvider.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <array>
#include <cstdint>

namespace pmacc::spearhed
{
    /** @brief Implicit fixed-grid candidate enumerator for one grid pair. */
    template<CoordinateSystem CS>
    struct FixedGridCandidateProviderView
    {
        using Axis = typename CS::T_Axis;
        using Grid = FixedCartesianGrid<CS>;
        using Vec = typename Grid::Vec;

        Grid targetGrid;
        Grid sourceGrid;
        Axis cutoff;
        std::array<uint32_t, CS::dimension> imageRadius;

        template<typename Fn>
        HDINLINE void forEachCandidate(uint32_t targetBucketSlot, Fn&& fn) const
        {
            auto const targetSearchBounds = targetGrid.ownershipBounds(targetBucketSlot).expand(cutoff);
            auto const targetOrigin = targetGrid.chart(targetBucketSlot).origin;
            uint32_t const numSources = sourceGrid.bucketCount();
            uint32_t const numImages = imageCount();

            // Loop bounds depend only on the target slot and the provider value. Every worker in
            // an interaction block therefore follows identical control flow, as CandidateProvider
            // requires for callbacks containing block barriers.
            for(uint32_t sourceBucketSlot = 0u; sourceBucketSlot < numSources; ++sourceBucketSlot)
            {
                auto const sourceBounds = sourceGrid.ownershipBounds(sourceBucketSlot);
                auto const sourceOrigin = sourceGrid.chart(sourceBucketSlot).origin;
                for(uint32_t image = 0u; image < numImages; ++image)
                {
                    auto const translation = imageTranslation(image);
                    auto translatedBounds = sourceBounds;
                    translatedBounds.min = translatedBounds.min + translation;
                    translatedBounds.max = translatedBounds.max + translation;
                    if(intersects(targetSearchBounds, translatedBounds))
                    {
                        bool const isUnshiftedImage = isZero(translation);
                        fn(RegionCandidate{
                            sourceBucketSlot,
                            sourceOrigin + translation - targetOrigin,
                            isUnshiftedImage});
                    }
                }
            }
        }

    private:
        [[nodiscard]] HDINLINE uint32_t imageCount() const
        {
            uint32_t count = 1u;
            for(uint32_t axis = 0u; axis < CS::dimension; ++axis)
                count *= sourceGrid.periodic[axis] ? (2u * imageRadius[axis] + 1u) : 1u;
            return count;
        }

        [[nodiscard]] HDINLINE Vec imageTranslation(uint32_t image) const
        {
            Vec translation{Axis{0}};
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    if(!sourceGrid.periodic[i.value])
                        return;
                    uint32_t const width = 2u * imageRadius[i.value] + 1u;
                    int const coordinate = static_cast<int>(image % width) - static_cast<int>(imageRadius[i.value]);
                    image /= width;
                    translation[tag]
                        = static_cast<Axis>(coordinate) * (sourceGrid.domain.max[tag] - sourceGrid.domain.min[tag]);
                });
            return translation;
        }

        [[nodiscard]] HDINLINE bool isZero(Vec const& value) const
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag) { return value[tag] == Axis{0}; });
        }
    };

    /** @brief Host value for an implicit fixed-grid candidate provider. */
    template<CoordinateSystem CS>
    struct FixedGridCandidateProvider
    {
        using Axis = typename CS::T_Axis;
        using Grid = FixedCartesianGrid<CS>;

        FixedGridCandidateProvider(Grid const& targetGrid, Grid const& sourceGrid, Axis cutoff)
            : targetGrid(targetGrid)
            , sourceGrid(sourceGrid)
            , cutoff(cutoff)
        {
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    if(!sourceGrid.periodic[i.value])
                        return;
                    Axis const domainExtent = sourceGrid.domain.max[tag] - sourceGrid.domain.min[tag];
                    imageRadius[i.value] = static_cast<uint32_t>(cutoff / domainExtent) + 1u;
                });
        }

        [[nodiscard]] CandidateProvider auto deviceView() const
        {
            return FixedGridCandidateProviderView<CS>{targetGrid, sourceGrid, cutoff, imageRadius};
        }

        Grid targetGrid;
        Grid sourceGrid;
        Axis cutoff;
        std::array<uint32_t, CS::dimension> imageRadius{};
    };
} // namespace pmacc::spearhed
