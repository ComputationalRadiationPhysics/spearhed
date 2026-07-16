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

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
#include "spmacc/particles/initialization/Random.hpp"
#include "spmacc/particles/initialization/SC.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

/**
 * Test setup that places particles in a simple cubic (SC) lattice within a unit cube.
 *
 * 8 particles in a 2x2x2 grid over [0,1]^3:
 *   cell size = 0.5 in each dimension
 *   particle positions: centers at 0.25 and 0.75 along each axis
 */
struct SCLatticeSetup
{
    // This setup fills a single species and acts as its own (only) init block.
    using Species = pmacc::spearhed::species::Default;

    uint32_t numParticles = 8u;

    pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};

    auto blocks() const
    {
        return std::tie(*this);
    }

    struct NumParticlesToCreate
    {
        DINLINE constexpr auto operator()(
            [[maybe_unused]] auto& worker,
            [[maybe_unused]] auto& particleRegion,
            uint32_t n) const
        {
            return n;
        }
    };

    auto numParticlesToCreateArgs() const
    {
        return std::make_tuple(numParticles);
    }

    using PlaceParticle = pmacc::spearhed::SC<spearhed::CS>;

    auto placeParticleArgs() const
    {
        return std::make_tuple(pmacc::spearhed::computeSCNumCells(numParticles, domain));
    }

    template<typename>
    void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
    {
        out.push_back(domain);
    }
};

/**
 * Accumulates the sum of particle positions component-wise.
 */
struct SumPositions
{
    HDINLINE constexpr void operator()(auto& worker, auto& particle, auto posSum) const
    {
        using namespace pmacc::spearhed::tags;
        alpaka::atomicAdd(worker.getAcc(), &posSum(0), particle[relativePos][x], ::alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(worker.getAcc(), &posSum(1), particle[relativePos][y], ::alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(worker.getAcc(), &posSum(2), particle[relativePos][z], ::alpaka::hierarchy::Blocks{});
    }
};

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "SC lattice places 8 particles in 2x2x2 grid", "[integration][particles][sc]")
{
    auto setup = SCLatticeSetup{};
    spearhed::InitRegions{}(*deviceHeap, setup);

    spearhed::InitParticles{}(setup);

    // Accumulate x, y, z position sums across all particles
    pmacc::HostDeviceBuffer<float, 1> posSumBuf(3u);
    posSumBuf.getHostBuffer().setValue(0.0f);
    posSumBuf.hostToDevice();

    auto posSum = posSumBuf.getDeviceBuffer().getDataBox();
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, SumPositions{}, posSum);

    posSumBuf.deviceToHost();
    auto hostData = posSumBuf.getHostBuffer().getDataBox();

    // 2x2x2 SC grid in [0,1]^3: positions at 0.25 and 0.75 on each axis, 4 particles each.
    // sum per axis = 4 * 0.25 + 4 * 0.75 = 1.0 + 3.0 = 4.0 (exact in float)
    REQUIRE(hostData(0) == 4.0f);
    REQUIRE(hostData(1) == 4.0f);
    REQUIRE(hostData(2) == 4.0f);
}

/**
 * Test setup that places particles within a unit cube.
 */
struct RandomSetup
{
    // This setup fills a single species and acts as its own (only) init block.
    using Species = pmacc::spearhed::species::Default;

    uint32_t numParticles = 64u;

    pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};

    auto blocks() const
    {
        return std::tie(*this);
    }

    struct NumParticlesToCreate
    {
        DINLINE constexpr auto operator()(
            [[maybe_unused]] auto& worker,
            [[maybe_unused]] auto& particleRegion,
            uint32_t n) const
        {
            return n;
        }
    };

    auto numParticlesToCreateArgs() const
    {
        return std::make_tuple(numParticles);
    }

    using PlaceParticle = pmacc::spearhed::Random<spearhed::CS>;

    auto placeParticleArgs() const
    {
        return std::make_tuple(42u, pmacc::spearhed::UniformDistribution{});
    }

    template<typename>
    void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
    {
        out.push_back(domain);
    }
};

/**
 * Counts particles whose position falls outside the AABB [0,1]^3.
 */
struct CountOutOfBounds
{
    HDINLINE constexpr void operator()(auto& worker, auto& particle, auto outOfBoundsCount) const
    {
        using namespace pmacc::spearhed::tags;
        auto const px = particle[relativePos][x];
        auto const py = particle[relativePos][y];
        auto const pz = particle[relativePos][z];
        if(px < 0.0f || px >= 1.0f || py < 0.0f || py >= 1.0f || pz < 0.0f || pz >= 1.0f)
            alpaka::atomicAdd(worker.getAcc(), &outOfBoundsCount(0), 1u, ::alpaka::hierarchy::Blocks{});
    }
};

TEST_CASE_METHOD(
    ParticleFixture,
    "Random placement keeps all particles within AABB",
    "[integration][particles][random]")
{
    auto setup = RandomSetup{};
    spearhed::InitRegions{}(*deviceHeap, setup);

    spearhed::InitParticles{}(setup);

    pmacc::HostDeviceBuffer<uint32_t, 1> outOfBoundsBuf(1u);
    outOfBoundsBuf.getHostBuffer().setValue(0u);
    outOfBoundsBuf.hostToDevice();

    auto outOfBoundsCount = outOfBoundsBuf.getDeviceBuffer().getDataBox();
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, CountOutOfBounds{}, outOfBoundsCount);

    outOfBoundsBuf.deviceToHost();
    REQUIRE(outOfBoundsBuf.getHostBuffer().getDataBox()(0) == 0u);
}
