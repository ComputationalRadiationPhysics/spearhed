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

#include "spmacc/particles/algorithms/ForEachParticle.hpp"

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

// Functor: Atomically add particle IDs to the sum
struct SumFunc
{
    HDINLINE constexpr void operator()(auto& worker, auto& particle, auto sum_db) const
    {
        alpaka::atomicAdd(worker.getAcc(), &sum_db(0), *particle[spearhed::particleId], ::alpaka::hierarchy::Blocks{});
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "ForEachParticleInPRBuf Validation", "[integration][particles][foreach]")
{
    constexpr uint64_t numRegions = 2;
    auto setup = spearhed::EmptyNRegions<numRegions>{};
    setup.setupRegions(*prBuf, *deviceHeap);

    spearhed::InitParticles{}(setup);

    // Execute ForEachParticleInPRBuf Test
    // Allocate memory for reduction sum

    using T_Sum = uint64_t;
    pmacc::HostDeviceBuffer<T_Sum, 1> sumBuffer(1u);

    // Initialize on host and sync to device
    sumBuffer.getHostBuffer().setValue(0);
    sumBuffer.hostToDevice();

    auto d_sum = sumBuffer.getDeviceBuffer().getDataBox();

    // Execute the utility
    pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, SumFunc{}, d_sum);

    // Sync result back to host
    sumBuffer.deviceToHost();
    T_Sum h_sum = sumBuffer.getHostBuffer().data()[0];


    //  Validation
    uint64_t totalParticles = 0;

    // Calculate expected sum analytically based on InitParticles logic
    for(uint64_t i = 0; i < numRegions; ++i)
    {
        totalParticles += setup.baseNumParticlesToCreate * (i + 1);
    }

    // Sum of arithmetic progression: n*(n-1)/2 because IDs start at 0
    uint64_t const expectedSum = (totalParticles * (totalParticles - 1)) / 2;

    INFO("Total Particles: " << totalParticles);
    INFO("Actual Sum (Device): " << h_sum);
    INFO("Expected Sum: " << expectedSum);

    REQUIRE(h_sum == expectedSum);
}
