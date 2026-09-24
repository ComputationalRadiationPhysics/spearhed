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

#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"

#include "TestSetup.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegion.hpp"
#include "spmacc/particles/regions/mapping/constant/Decomposition.hpp"
#include "spmacc/topology/CartesianStorage.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

// Define expected bounds
// We use a float value that can be exactly represented to avoid precision issues in comparison
using CS = pmacc::spearhed::Cartesian<float, TEST_DIM>;
using RelPosType = pmacc::spearhed::Vec<CS, pmacc::spearhed::ValueStorage<CS>>;
constexpr RelPosType expectedMin{0.125f, 0.125f, 0.125f};
constexpr RelPosType expectedMax{0.875f, 0.875f, 0.875f};
constexpr RelPosType defaultPos{0.5f, 0.5f, 0.5f};
constexpr RelPosType chartOrigin{10.0f, 20.0f, 30.0f};

// Functor to set specific particle positions:
// - ID 0 -> Min corner
// - ID 1 -> Max corner
// - Others -> Center
struct SetPosFunctor
{
    HDINLINE constexpr void operator()(auto& worker, auto& particle)
    {
        using namespace spearhed;

        // Reset all to center first
        constexpr auto def = defaultPos;
        particle[spearhed::relativePos].get() = def;
        // Set outliers to define the bounding box
        if(particle[particleId] == 0)
        {
            constexpr auto min = expectedMin;
            particle[spearhed::relativePos].get() = min;
        }
        else if(particle[particleId] == 1)
        {
            constexpr auto max = expectedMax;
            particle[spearhed::relativePos].get() = max;
        }
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "UpdateRegionBounds Validation", "[integration][particles][bounds]")
{
    auto setup = spearhed::EmptyNRegions<1>{};
    spearhed::InitRegions{}(*deviceHeap, setup);

    // Initialize and modify positions
    spearhed::InitParticles{}(setup);
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, SetPosFunctor{});

    // Execute
    pmacc::spearhed::UpdateVolumes<spearhed::PRType>{}();

    // Validation
    prBuf->buffer->deviceToHost();
    auto dataBox = prBuf->buffer->getHostBuffer().getDataBox();
    auto const& region = dataBox(0);

    pmacc::spearhed::for_each_tag<spearhed::CS>(
        [&](auto tag)
        {
            REQUIRE(region.spatial.occupancy.min[tag] == expectedMin[tag]);
            REQUIRE(region.spatial.occupancy.max[tag] == expectedMax[tag]);
        });
}

TEST_CASE_METHOD(
    ParticleFixture,
    "MaterialAabbDecomposition prepares a generation and owns its CSR plan",
    "[integration][spatial]")
{
    auto setup = spearhed::EmptyNRegions<1>{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    auto decomposition = pmacc::spearhed::MaterialAabbDecomposition{*prBuf};
    decomposition.prepareAfterMotion();
    auto prepared = decomposition.preparedFor(*prBuf);
    auto plan = pmacc::spearhed::makeInteractionPlan(prepared, 1.0f, prepared);
    auto multiSourcePlan = pmacc::spearhed::makeInteractionPlan(prepared, 1.0f, prepared, prepared);
    auto emptyPlan = pmacc::spearhed::makeInteractionPlan(prepared, 1.0f);

    STATIC_REQUIRE(pmacc::spearhed::PreparedRegionSet<decltype(prepared)>);
    STATIC_REQUIRE(pmacc::spearhed::SpatialDecompositionFor<decltype(decomposition), decltype(*prBuf)>);
    STATIC_REQUIRE(pmacc::spearhed::IsNeighbourBundle<decltype(plan)>);
    STATIC_REQUIRE(pmacc::spearhed::IsNeighbourBundle<decltype(multiSourcePlan)>);
    STATIC_REQUIRE(pmacc::spearhed::IsNeighbourBundle<decltype(emptyPlan)>);
    REQUIRE(prepared.generation() == 1u);
    REQUIRE(prepared.bucketCount() == 1u);
    REQUIRE(plan.size() == 1u);
    REQUIRE(multiSourcePlan.size() == 2u);
    REQUIRE(emptyPlan.size() == 0u);

    // The target may also be a source without another bounds reduction. A later
    // preparation advances the shared generation, which debug assertions on the
    // old handle and plan use to reject stale reuse.
    decomposition.prepareAfterMotion();
    auto newerPrepared = decomposition.preparedFor(*prBuf);
    REQUIRE(newerPrepared.generation() == 2u);
}

TEST_CASE_METHOD(
    ParticleFixture,
    "UpdateRegionBounds reduces world positions without rebasing the chart",
    "[integration][particles][bounds]")
{
    auto setup = spearhed::EmptyNRegions<1>{};
    spearhed::InitRegions{}(*deviceHeap, setup);

    auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();
    hostRegions(0).spatial.chart.origin = {10.0f, 20.0f, 30.0f};
    prBuf->buffer->hostToDevice();

    spearhed::InitParticles{}(setup);
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, SetPosFunctor{});
    pmacc::spearhed::UpdateVolumes<spearhed::PRType>{}();

    prBuf->buffer->deviceToHost();
    auto const dataBox = prBuf->buffer->getHostBuffer().getDataBox();
    auto const& region = dataBox(0);
    pmacc::spearhed::for_each_tag<spearhed::CS>(
        [&](auto tag)
        {
            REQUIRE(region.spatial.chart.origin[tag] == chartOrigin[tag]);
            REQUIRE(region.spatial.occupancy.min[tag] == region.spatial.chart.origin[tag] + expectedMin[tag]);
            REQUIRE(region.spatial.occupancy.max[tag] == region.spatial.chart.origin[tag] + expectedMax[tag]);
        });
}
