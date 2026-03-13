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

#include "spearhed/param.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/NeighbourRegions.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Positioning.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "CalculateNeighbourRegions Validation", "[integration][particles][neighbours]")
{
    setupRegions(3);

    // Initialize region volumes manually
    prBuf->buffer->deviceToHost();
    auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();

    for(unsigned d = 0; d < TEST_DIM; ++d)
    {
        // Region 0: [0.0, 1.0]
        hostRegions(0).volume.min[d] = 0.0f;
        hostRegions(0).volume.max[d] = 1.0f;

        // Region 1: [1.5, 2.5]
        hostRegions(1).volume.min[d] = 1.5f;
        hostRegions(1).volume.max[d] = 2.5f;

        // Region 2: [4.0, 5.0]
        hostRegions(2).volume.min[d] = 4.0f;
        hostRegions(2).volume.max[d] = 5.0f;
    }

    prBuf->buffer->hostToDevice();

    // Expected behavior with smoothingLength = 0.6f:
    // Region 0 expands to [-0.6, 1.6] -> Intersects Region 0 and 1
    // Region 1 expands to [0.9, 3.1] -> Intersects Region 0 and 1
    // Region 2 expands to [3.4, 5.6] -> Intersects Region 2 only
    constexpr float smoothingLength = 0.6f;

    auto [neighbourRegions, regionOffsets] = pmacc::spearhed::CalculateNeighbourRegions{}(*prBuf, smoothingLength);

    // Validation
    neighbourRegions.deviceToHost();
    regionOffsets.deviceToHost();

    auto const& h_neighbours = neighbourRegions.getHostBuffer().getDataBox();
    auto const& h_offsets = regionOffsets.getHostBuffer().getDataBox();

    // Validate inclusive prefix sum offsets
    REQUIRE(h_offsets(0) == 0);
    REQUIRE(h_offsets(1) == 2); // Region 0 matches: 0, 1
    REQUIRE(h_offsets(2) == 4); // Region 1 matches: 0, 1
    REQUIRE(h_offsets(3) == 5); // Region 2 matches: 2

    // Validate neighbour IDs
    REQUIRE(h_neighbours(0) == 0);
    REQUIRE(h_neighbours(1) == 1);

    REQUIRE(h_neighbours(2) == 0);
    REQUIRE(h_neighbours(3) == 1);

    REQUIRE(h_neighbours(4) == 2);
}
