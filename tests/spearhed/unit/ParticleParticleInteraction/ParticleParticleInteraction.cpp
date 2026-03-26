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

#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"

#include "TestSetup.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Positioning.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

struct InteractionCountFunc
{
    HDINLINE constexpr void operator()(auto& worker, auto& ownP, auto& neighbourP, auto count_db) const
    {
        if(*ownP[spearhed::particleId] != *neighbourP[spearhed::particleId])
        {
            alpaka::atomicAdd(worker.getAcc(), &count_db(0), static_cast<uint64_t>(1), ::alpaka::hierarchy::Blocks{});
        }
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "InteractParticles Validation", "[integration][particles][interaction]")
{
    constexpr int numRegions = 2;

    auto setup = spearhed::EmptyNRegions<numRegions>{};
    setup.setupRegions(*prBuf, *deviceHeap);
    spearhed::InitParticles{}();

    // Setup Neighbour Graph (All-to-All mapping for analytic validation)
    int const totalNeighbours = numRegions * numRegions;
    pmacc::HostDeviceBuffer<int, 1> neighbourRegions(totalNeighbours);
    auto h_neighbours = neighbourRegions.getHostBuffer().data();

    // Offsets point to the start of each region's neighbour list in h_neighbours.
    pmacc::HostDeviceBuffer<int, 1> regionOffsets(numRegions + 1);
    auto h_offsets = regionOffsets.getHostBuffer().data();

    for(int i = 0; i < numRegions; ++i)
    {
        h_offsets[i] = i * numRegions;
        for(int j = 0; j < numRegions; ++j)
        {
            h_neighbours[i * numRegions + j] = j;
        }
    }
    // Set the final boundary offset
    h_offsets[numRegions] = totalNeighbours;

    neighbourRegions.hostToDevice();
    regionOffsets.hostToDevice();

    using T_Count = uint64_t;
    pmacc::HostDeviceBuffer<T_Count, 1> countBuffer(1u);
    countBuffer.getHostBuffer().setValue(0);
    countBuffer.hostToDevice();

    auto d_count = countBuffer.getDeviceBuffer().getDataBox();

    // Use an excessively large interaction radius so the distance check always passes
    constexpr double interactionRadius = 1e9;

    pmacc::spearhed::InteractParticles{}(
        *prBuf,
        neighbourRegions,
        regionOffsets,
        interactionRadius,
        InteractionCountFunc{},
        d_count);

    countBuffer.deviceToHost();
    T_Count const h_count = countBuffer.getHostBuffer().data()[0];

    // Calculate expected interactions analytically based on InitParticles logic
    uint64_t totalParticles = 0;
    for(uint64_t i = 0; i < numRegions; ++i)
    {
        totalParticles += spearhed::init::detail::baseNumParticlesToCreate * (i + 1);
    }

    // Expected valid interactions: All particles interact with all other particles exactly once.
    // Self-interaction is excluded via our functor. P(N, 2) = N * (N - 1)
    uint64_t const expectedInteractions = totalParticles * (totalParticles - 1);

    INFO("Total Particles: " << totalParticles);
    INFO("Actual Interactions (Device): " << h_count);
    INFO("Expected Interactions: " << expectedInteractions);

    REQUIRE(h_count == expectedInteractions);
}
