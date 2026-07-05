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

#include "TestSetup.hpp"
#include "ValidateIdSum.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"

#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "Particle Creation and ID Sum Validation", "[integration][particles]")
{
    constexpr uint64_t numRegions = 2;

    auto setup = spearhed::EmptyNRegions<numRegions>{};
    spearhed::InitRegions{}(*deviceHeap, setup);

    spearhed::InitParticles{}(setup);

    uint64_t const actualSum = ComputeParticleIdSum{}();

    // Analytical calculation
    uint64_t totalParticles = 0;
    for(uint64_t i = 0; i < numRegions; ++i)
        totalParticles += setup.baseNumParticlesToCreate * (i + 1);

    uint64_t const expectedSum = (totalParticles * (totalParticles - 1)) / 2;
    REQUIRE(actualSum == expectedSum);
}
