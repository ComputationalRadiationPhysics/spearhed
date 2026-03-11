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

#include "ValidatePush.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/dimension.param"
#include "spearhed/param/speciesTraits.param"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
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

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "Particle Pusher Validation", "[integration][particles][pusher]")
{
    setupRegions(1);
    spearhed::InitParticles{}();

    spearhed::ParticlePush{}(1);
    ValidatePush{}();
}
