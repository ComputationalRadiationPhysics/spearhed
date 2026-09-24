/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/RegionChart.hpp"
#include "spmacc/particles/regions/WorldAABB.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmacc::spearhed
{
    /**
     * @brief Runtime geometry of a dense, single-accelerator Cartesian grid.
     *
     * A cell is the geometric box in world space. A bucket is the particle
     * storage and ownership unit associated with one cell (typically the
     * ParticleRegion at that cell's index). A slot is the bucket's dense
     * integer ID, not the slot of a particle within a storage frame. Buckets
     * use stable row-major slots: x is the fastest varying axis. The grid
     * owns all cell geometry; callers derive charts and ownership AABBs from
     * a slot instead of retaining a separate AABB for every cell.
     *
     * Non-periodic axes accept the closed outer boundary: a point exactly at
     * @c domain.max belongs to the final cell.  All other points must be in
     * @c [min,max].  Periodic axes are canonicalised into @c [min,max), so a
     * point at @c max belongs to cell zero.
     */
    template<CoordinateSystem CS>
    struct FixedCartesianGrid
    {
        using CoordinateSystemType = CS;
        using Axis = typename CS::T_Axis;
        using Point = pmacc::spearhed::Point<CS, ValueStorage<CS>>;
        using Vec = pmacc::spearhed::Vec<CS, ValueStorage<CS>>;
        using Bounds = WorldAABB<CS>;
        using Chart = RegionChart<CS>;

        /**
         * @brief Result of locating a world-space point in this grid.
         *
         * A Location is meaningful only for the grid that produced it: slot
         * identifies the destination cell/bucket, localPosition is expressed
         * in that cell's RegionChart, and canonicalWorldPosition is wrapped
         * into the domain on periodic axes.
         */
        struct Location
        {
            uint32_t slot;
            Vec localPosition;
            Point canonicalWorldPosition;
        };

        FixedCartesianGrid() = default;

        FixedCartesianGrid(
            Bounds const& domain,
            std::array<uint32_t, CS::dimension> const& cellCounts,
            std::array<bool, CS::dimension> const& periodic = {})
            : domain(domain)
            , cellCounts(cellCounts)
            , periodic(periodic)
        {
            validate();
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                { cellExtent[tag] = (domain.max[tag] - domain.min[tag]) / static_cast<Axis>(cellCounts[i.value]); });
        }

        /** @brief Construct a grid whose count is the ceiling of domain / requested cell extent. */
        static FixedCartesianGrid fromCellExtent(
            Bounds const& domain,
            Vec const& requestedCellExtent,
            std::array<bool, CS::dimension> const& periodic = {})
        {
            std::array<uint32_t, CS::dimension> counts{};
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    Axis const length = domain.max[tag] - domain.min[tag];
                    Axis const requested = requestedCellExtent[tag];
                    if(!(requested > Axis{0}))
                        throw std::invalid_argument("fixed Cartesian cell extent must be positive");
                    uint32_t const truncated = static_cast<uint32_t>(length / requested);
                    counts[i.value] = (static_cast<Axis>(truncated) * requested < length) ? truncated + 1u : truncated;
                    if(counts[i.value] == 0u)
                        counts[i.value] = 1u;
                });
            return FixedCartesianGrid{domain, counts, periodic};
        }

        [[nodiscard]] HDINLINE constexpr uint32_t bucketCount() const
        {
            uint32_t count = 1u;
            for(uint32_t axisCount : cellCounts)
                count *= axisCount;
            return count;
        }

        [[nodiscard]] HDINLINE constexpr Vec extent() const
        {
            return domain.max - domain.min;
        }

        [[nodiscard]] HDINLINE constexpr std::array<uint32_t, CS::dimension> cellIndex(uint32_t slot) const
        {
            std::array<uint32_t, CS::dimension> index{};
            for(uint32_t axis = 0u; axis < CS::dimension; ++axis)
            {
                index[axis] = slot % cellCounts[axis];
                slot /= cellCounts[axis];
            }
            return index;
        }

        [[nodiscard]] HDINLINE constexpr uint32_t slotFor(std::array<uint32_t, CS::dimension> const& index) const
        {
            uint32_t slot = 0u;
            uint32_t stride = 1u;
            for(uint32_t axis = 0u; axis < CS::dimension; ++axis)
            {
                slot += index[axis] * stride;
                stride *= cellCounts[axis];
            }
            return slot;
        }

        [[nodiscard]] HDINLINE constexpr Chart chart(uint32_t slot) const
        {
            auto const index = cellIndex(slot);
            Point origin = domain.min;
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag) { origin[tag] += static_cast<Axis>(index[i.value]) * cellExtent[tag]; });
            return Chart{origin};
        }

        [[nodiscard]] HDINLINE constexpr Bounds ownershipBounds(uint32_t slot) const
        {
            auto const origin = chart(slot).origin;
            return Bounds{origin, origin + cellExtent};
        }

        [[nodiscard]] HDINLINE constexpr Point toWorld(uint32_t slot, Vec const& localPosition) const
        {
            return chart(slot).toWorld(localPosition);
        }

        /**
         * @brief Locate and rebase a world-space position.
         *
         * @param result Output Location. It is written only on success, so
         * callers must use it only when this function returns true.
         * @return false only for a point outside a non-periodic domain. The
         * returned local position is relative to the destination cell chart.
         *
         * The boolean represents a possibly-invalid location while Location
         * groups the three successful outputs without encoding an invalid
         * slot or position sentinel.
         */
        [[nodiscard]] HDINLINE constexpr bool locate(Point worldPosition, Location& result) const
        {
            std::array<uint32_t, CS::dimension> index{};
            Point canonical = worldPosition;
            bool valid = true;

            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    Axis coordinate = canonical[tag] - domain.min[tag];
                    Axis const length = domain.max[tag] - domain.min[tag];
                    if(periodic[i.value])
                    {
                        // Positions can cross more than one domain in a long step. This is a
                        // maintenance path, so correctness is preferred over a hand-tuned wrap.
                        while(coordinate < Axis{0})
                            coordinate += length;
                        while(coordinate >= length)
                            coordinate -= length;
                        canonical[tag] = domain.min[tag] + coordinate;
                    }
                    else if(coordinate < Axis{0} || coordinate > length)
                    {
                        valid = false;
                        return;
                    }

                    uint32_t cell = static_cast<uint32_t>(coordinate / cellExtent[tag]);
                    // Closed upper boundary on non-periodic axes belongs to the final cell.
                    if(cell == cellCounts[i.value])
                        cell = cellCounts[i.value] - 1u;
                    index[i.value] = cell;
                });

            if(!valid)
                return false;

            result.slot = slotFor(index);
            result.canonicalWorldPosition = canonical;
            result.localPosition = chart(result.slot).toLocal(canonical);
            return true;
        }

        [[nodiscard]] constexpr bool operator==(FixedCartesianGrid const&) const = default;

        Bounds domain{};
        std::array<uint32_t, CS::dimension> cellCounts{};
        std::array<bool, CS::dimension> periodic{};
        Vec cellExtent{};

    private:
        void validate() const
        {
            if(domain.empty())
                throw std::invalid_argument("fixed Cartesian domain must be non-empty");

            bool valid = true;
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                { valid = valid && cellCounts[i.value] > 0u && domain.max[tag] > domain.min[tag]; });
            if(!valid)
                throw std::invalid_argument("fixed Cartesian grid requires positive extents and cell counts");

            uint64_t count = 1u;
            for(uint32_t axisCount : cellCounts)
                count *= axisCount;
            if(count > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("fixed Cartesian grid has too many dense cell slots");
        }
    };
} // namespace pmacc::spearhed
