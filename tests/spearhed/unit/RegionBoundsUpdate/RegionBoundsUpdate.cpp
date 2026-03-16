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

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
#include "spmacc/particles/regions/ParticleRegion.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Point.hpp"
#include "spmacc/topology/PointStorage.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

// Define expected bounds
// We use a float value that can be exactly represented to avoid precision issues in comparison
using CS = pmacc::spearhed::Cartesian<float, TEST_DIM>;
using PosType = pmacc::spearhed::Point<CS, pmacc::spearhed::PointValueStorage<CS>>;
constexpr PosType expectedMin{0.125f, 0.125f, 0.125f};
constexpr PosType expectedMax{0.875f, 0.875f, 0.875f};
constexpr PosType defaultPos{0.5f, 0.5f, 0.5f};

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
        particle[spearhed::pos].get() = def;
        // Set outliers to define the bounding box
        if(*particle[particleId] == 0)
        {
            constexpr auto min = expectedMin;
            particle[spearhed::pos].get() = min;
        }
        else if(*particle[particleId] == 1)
        {
            constexpr auto max = expectedMax;
            particle[spearhed::pos].get() = max;
        }
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "UpdateRegionBounds Validation", "[integration][particles][bounds]")
{
    setupRegions(1);

    // Initialize and modify positions
    spearhed::InitParticles{}();
    pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, SetPosFunctor{});

    // Execute
    pmacc::spearhed::UpdateVolumes<spearhed::PRType>{}();

    // Validation
    prBuf->buffer->deviceToHost();
    auto const& region = prBuf->buffer->getHostBuffer().getDataBox()(0);

    for(unsigned d = 0; d < TEST_DIM; ++d)
    {
        REQUIRE(region.volume.min[d] == expectedMin[d]);
        REQUIRE(region.volume.max[d] == expectedMax[d]);
    }
}
