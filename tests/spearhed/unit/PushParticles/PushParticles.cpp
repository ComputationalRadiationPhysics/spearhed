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
#include "ValidatePush.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/topology/CartesianStorage.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Point.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

// Define expected bounds
// We use a float value that can be exactly represented to avoid precision issues in comparison
using CS = pmacc::spearhed::Cartesian<float, TEST_DIM>;
using PosType = pmacc::spearhed::Point<CS, pmacc::spearhed::ValueStorage<CS>>;

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

struct SpeedyRegion
{
    // This setup fills a single species and acts as its own (only) init block.
    using Species = pmacc::spearhed::species::Default;
    using NumParticlesToCreate = spearhed::ScaledNumParticlesToCreate;

    static constexpr spearhed::Real velInit = 100.0f;

    // AABB constructor arguments are: {cell anchor/index}, {min corner}, {max corner}.
    pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

    uint32_t baseNumParticlesToCreate = 400u;

    auto blocks() const
    {
        return std::tie(*this);
    }

    auto numParticlesToCreateArgs() const
    {
        return std::make_tuple(baseNumParticlesToCreate);
    }

    auto placeParticleArgs() const
    {
        return std::make_tuple();
    }

    // place each particle at the center of its particle region's AABB and set velocity
    struct PlaceParticle
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            [[maybe_unused]] uint32_t globalParticleIdx) const
        {
            spearhed::CenterPlaceParticle{}(worker, particle, particleRegion, globalParticleIdx);
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { particle[spearhed::vel][tag] = velInit; });
        }
    };

    template<typename>
    void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
    {
        out.push_back(domain);
    }
};

TEST_CASE_METHOD(ParticleFixture, "Particle Pusher Validation", "[integration][particles][pusher]")
{
    auto setup = SpeedyRegion{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    constexpr spearhed::Real expectedDisplacement = SpeedyRegion::velInit * spearhed::dt;
    spearhed::ParticlePush{}(1);
    ValidatePush{expectedDisplacement}();
}
