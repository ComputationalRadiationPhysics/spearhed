/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"

#include "spearhed/param.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/CandidateProvider.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    using CS = spearhed::CS;
    using Grid = pmacc::spearhed::FixedCartesianGrid<CS>;
    using Bounds = pmacc::spearhed::WorldAABB<CS>;

    Grid makeGrid(std::array<bool, CS::dimension> periodic = {})
    {
        Bounds::Pnt const min{0.0f, 0.0f, 0.0f};
        Bounds::Pnt const max{4.0f, 6.0f, 8.0f};
        return Grid{Bounds{min, max}, {2u, 3u, 4u}, periodic};
    }

} // namespace

TEST_CASE("fixed Cartesian grid defines dense slots and boundary location", "[spatial][fixed-grid]")
{
    auto const grid = makeGrid();
    REQUIRE(grid.bucketCount() == 24u);
    REQUIRE(grid.cellIndex(0u) == std::array<uint32_t, 3>{0u, 0u, 0u});
    REQUIRE(grid.cellIndex(23u) == std::array<uint32_t, 3>{1u, 2u, 3u});
    REQUIRE(grid.slotFor({1u, 2u, 3u}) == 23u);

    Grid::Location location{};
    REQUIRE(grid.locate(Grid::Point{0.0f, 0.0f, 0.0f}, location));
    REQUIRE(location.slot == 0u);
    REQUIRE(grid.locate(Grid::Point{4.0f, 6.0f, 8.0f}, location));
    REQUIRE(location.slot == 23u);
    REQUIRE_FALSE(grid.locate(Grid::Point{-0.001f, 0.0f, 0.0f}, location));
    REQUIRE_FALSE(grid.locate(Grid::Point{4.001f, 0.0f, 0.0f}, location));

    auto periodic = makeGrid({true, false, false});
    REQUIRE(periodic.locate(Grid::Point{4.0f, 0.0f, 0.0f}, location));
    REQUIRE(location.slot == 0u);
    REQUIRE(periodic.locate(Grid::Point{-0.25f, 0.0f, 0.0f}, location));
    REQUIRE(location.slot == 1u);
    REQUIRE(location.canonicalWorldPosition[pmacc::spearhed::tags::x] == 3.75f);
}

TEST_CASE("fixed-grid provider matches brute-force cell geometry", "[spatial][fixed-grid]")
{
    auto const grid = makeGrid({true, true, false});
    constexpr float cutoff = 1.25f;
    auto const provider = pmacc::spearhed::FixedGridCandidateProvider<CS>{grid, grid, cutoff}.deviceView();

    for(uint32_t target = 0u; target < grid.bucketCount(); ++target)
    {
        std::vector<uint32_t> actual;
        provider.forEachCandidate(target, [&](auto candidate) { actual.push_back(candidate.sourceBucketSlot); });
        std::ranges::sort(actual);

        std::vector<uint32_t> expected;
        auto const targetBounds = grid.ownershipBounds(target).expand(cutoff);

        // Independent brute-force oracle over the periodic images necessary for this cutoff.
        for(uint32_t source = 0u; source < grid.bucketCount(); ++source)
        {
            bool overlaps = false;
            for(int imageX = -1; imageX <= 1; ++imageX)
                for(int imageY = -1; imageY <= 1; ++imageY)
                {
                    auto sourceBounds = grid.ownershipBounds(source);
                    Grid::Vec shift{0.0f, 0.0f, 0.0f};
                    shift[pmacc::spearhed::tags::x] = static_cast<float>(imageX) * 4.0f;
                    shift[pmacc::spearhed::tags::y] = static_cast<float>(imageY) * 6.0f;
                    sourceBounds.min = sourceBounds.min + shift;
                    sourceBounds.max = sourceBounds.max + shift;
                    overlaps = overlaps || intersects(targetBounds, sourceBounds);
                }
            if(overlaps)
                expected.push_back(source);
        }
        std::ranges::sort(expected);
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
        actual.erase(std::unique(actual.begin(), actual.end()), actual.end());
        REQUIRE(actual == expected);
    }
}

TEST_CASE("fixed-grid provider uses distinct target and source grids", "[spatial][fixed-grid]")
{
    auto const target = makeGrid();
    Bounds::Pnt const min{2.0f, 0.0f, 0.0f};
    Bounds::Pnt const max{4.0f, 6.0f, 8.0f};
    auto const source = Grid{Bounds{min, max}, {1u, 1u, 1u}};
    auto const provider = pmacc::spearhed::FixedGridCandidateProvider<CS>{target, source, 0.0f}.deviceView();

    std::vector<uint32_t> candidates;
    provider.forEachCandidate(1u, [&](auto candidate) { candidates.push_back(candidate.sourceBucketSlot); });
    REQUIRE(candidates == std::vector<uint32_t>{0u});
}
