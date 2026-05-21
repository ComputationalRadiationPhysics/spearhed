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
#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
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
    using Roles = std::tuple<pmacc::spearhed::roles::Interior>;

    // AABB constructor arguments are: {cell anchor/index}, {min corner}, {max corner}.
    pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

    uint32_t baseNumParticlesToCreate = 400u;

    template<typename Role>
    auto numParticlesToCreateArgs() const
    {
        return std::make_tuple(baseNumParticlesToCreate);
    }

    template<typename Role>
    auto placeParticleArgs() const
    {
        return std::make_tuple();
    }

    // calculate how many particles we need to make in this system
    template<typename Role>
    struct NumParticlesToCreate
    {
        constexpr auto operator()(
            [[maybe_unused]] auto& worker,
            [[maybe_unused]] auto& particleRegion,
            uint32_t baseNumParticlesToCreate) const
        {
            // Intentionally scale by (block index + 1) so each block creates a distinct
            // particle count, which makes per-block test validation deterministic.
            return baseNumParticlesToCreate * (worker.blockDomIdx() + 1);
        };
    };

    // place each particle at the center of its particle region's AABB
    template<typename Role>
    struct PlaceParticle
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            [[maybe_unused]] uint32_t globalParticleIdx) const
        {
            auto const& aabb = particleRegion.volume;
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag)
                {
                    *particle[spearhed::relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * 0.5f;
                    *particle[spearhed::vel][tag] = 100.0f;
                });
        }
    };

    template<typename Role>
    void setupRegions(
        pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType, Role>& prBuf,
        spearhed::DeviceHeap const& deviceHeap)
    {
        prBuf.create(1);
        spearhed::PRType boundedParticles{deviceHeap.getAllocatorHandle(), domain};
        prBuf.pushBack(boundedParticles);
        prBuf.buffer->hostToDevice();
    }
};

TEST_CASE_METHOD(ParticleFixture, "Particle Pusher Validation", "[integration][particles][pusher]")
{
    auto setup = SpeedyRegion{};
    setup.setupRegions(*prBuf, *deviceHeap);
    spearhed::InitParticles{}(setup);

    spearhed::ParticlePush{}(1);
    ValidatePush{}();
}
