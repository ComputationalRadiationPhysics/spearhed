/* Copyright 2025-2026 Tapish Narwal
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

#include "CandidateTestHelpers.hpp"
#include "TestSetup.hpp"
#include "spearhed/param.hpp"
#include "spearhed/param/mallocMC.param"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/regions/RegionChart.hpp"
#include "spmacc/particles/regions/WorldAABB.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Positioning.hpp>

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "CalculateNeighbourRegions Validation", "[integration][particles][neighbours]")
{
    SECTION("multiple empty target and source regions produce the expected candidate sets")
    {
        auto setup = spearhed::EmptyNRegions<3>{};
        spearhed::InitRegions{}(*deviceHeap, setup);

        // No particles are initialised: candidate discovery depends on spatial metadata, not bucket
        // occupancy. Initialise that metadata manually for three empty regions.
        prBuf->buffer->deviceToHost();
        auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();

        hostRegions(0).spatial.chart.origin = {10.0f, 20.0f, 30.0f};
        hostRegions(1).spatial.chart.origin = {100.0f, 200.0f, 300.0f};
        hostRegions(2).spatial.chart.origin = {-10.0f, -20.0f, -30.0f};

        pmacc::spearhed::for_each_tag<spearhed::CS>(
            [&](auto tag)
            {
                // Region 0: world-space [0.0, 1.0]
                hostRegions(0).spatial.occupancy.min[tag] = 0.0f;
                hostRegions(0).spatial.occupancy.max[tag] = 1.0f;

                // Region 1: [1.5, 2.5]
                hostRegions(1).spatial.occupancy.min[tag] = 1.5f;
                hostRegions(1).spatial.occupancy.max[tag] = 2.5f;

                // Region 2: [4.0, 5.0]
                hostRegions(2).spatial.occupancy.min[tag] = 4.0f;
                hostRegions(2).spatial.occupancy.max[tag] = 5.0f;
            });

        prBuf->buffer->hostToDevice();

        // Region 0 expands to [-0.6, 1.6] and intersects source regions 0 and 1.
        // Region 1 expands to [0.9, 3.1] and intersects source regions 0 and 1.
        // Region 2 expands to [3.4, 5.6] and intersects source region 2 only.
        constexpr float smoothingLength = 0.6f;

        auto bundle = pmacc::spearhed::calculateNeighbours(*prBuf, smoothingLength, *prBuf);
        auto& entry = bundle.bySpecies(pmacc::spearhed::species::default_);
        auto const candidates = pmacc::spearhed::test::collectCandidateSourceSlots(entry, 3u);

        REQUIRE(candidates == std::vector<std::vector<uint32_t>>{{0u, 1u}, {0u, 1u}, {2u}});
        REQUIRE(pmacc::spearhed::test::countCandidateRegionPairs(candidates) == 5u);
    }

    SECTION("empty source and target buffers are valid")
    {
        using PRBuf = pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>;

        auto setup = spearhed::EmptyNRegions<2>{};
        spearhed::InitRegions{}(*deviceHeap, setup);

        PRBuf emptySource;
        emptySource.create(0u);
        auto noSources = pmacc::spearhed::calculateNeighbours(*prBuf, 1.0f, emptySource);
        auto& sourceEntry = noSources.bySpecies(pmacc::spearhed::species::default_);
        REQUIRE(
            pmacc::spearhed::test::collectCandidateSourceSlots(sourceEntry, 2u)
            == std::vector<std::vector<uint32_t>>{{}, {}});

        PRBuf emptyTarget;
        emptyTarget.create(0u);
        auto noTargets = pmacc::spearhed::calculateNeighbours(emptyTarget, 1.0f, *prBuf);
        auto& targetEntry = noTargets.bySpecies(pmacc::spearhed::species::default_);
        REQUIRE(pmacc::spearhed::test::collectCandidateSourceSlots(targetEntry, 0u).empty());
    }
}

TEST_CASE("relativePos uses the explicit region chart", "[particles][position][contract]")
{
    using Chart = pmacc::spearhed::RegionChart<spearhed::CS>;
    using LocalPosition = Chart::Vec;

    Chart const chart{{10.0f, 20.0f, 30.0f}};
    LocalPosition const relativePos{1.0f, 2.0f, 3.0f};
    auto const worldPosition = chart.toWorld(relativePos);

    using namespace pmacc::spearhed::tags;
    REQUIRE(worldPosition[x] == 11.0f);
    REQUIRE(worldPosition[y] == 22.0f);
    REQUIRE(worldPosition[z] == 33.0f);
}

TEST_CASE("WorldAABB has explicit empty and world-space operations", "[particles][bounds][contract]")
{
    using Bounds = pmacc::spearhed::WorldAABB<spearhed::CS>;
    using Point = Bounds::Pnt;

    Bounds bounds;
    Bounds const otherEmpty;
    Bounds const finiteBounds{Point{0.0f, 0.0f, 0.0f}, Point{1.0f, 1.0f, 1.0f}};
    REQUIRE(bounds.empty());
    REQUIRE(bounds.expand(1.0f).empty());
    REQUIRE_FALSE(intersects(bounds, finiteBounds));
    bounds.extend(otherEmpty);
    REQUIRE(bounds.empty());

    bounds.extend(Point{1.0f, 2.0f, 3.0f});
    bounds.extend(Point{4.0f, 5.0f, 6.0f});
    bounds.extend(Bounds{Point{-2.0f, 2.0f, 3.0f}, Point{-1.0f, 5.0f, 6.0f}});
    REQUIRE_FALSE(bounds.empty());

    auto const expanded = bounds.expand(1.0f);
    REQUIRE(intersects(expanded, Bounds{Point{0.0f, 0.0f, 0.0f}, Point{2.0f, 3.0f, 4.0f}}));
    REQUIRE_FALSE(intersects(expanded, Bounds{Point{6.0f, 0.0f, 0.0f}, Point{7.0f, 1.0f, 1.0f}}));
}
