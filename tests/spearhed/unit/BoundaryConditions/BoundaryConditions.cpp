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

/**
 * Unit test for compile-time region roles / boundary conditions.
 *
 * Verifies that:
 *   1. Interior particles advance by vel * dt after ParticlePush.
 *   2. Boundary particles are completely untouched - their positions are
 *      identical before and after the push, even though their stored
 *      velocity (999 m/s) would produce a large displacement if they
 *      were incorrectly pushed.
 *
 * This tests the central invariant of the role system: kernels constrained
 * to InteriorPRBuf cannot physically touch boundary particles.
 */

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

namespace
{
    // exposes an Interior block and a Boundary block.
    struct BoundaryWallSetup
    {
        using Roles = std::tuple<pmacc::spearhed::roles::Interior, pmacc::spearhed::roles::Boundary>;

        // Interior AABB: [-1,-1,-1] to [1,1,1], center (0,0,0).
        // PlaceParticle sets vel to 100.f.
        // After one push: relativePos = (0,0,0) + (100,100,100)*dt = (1,1,1).
        struct InteriorBlock
        {
            pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};

            struct NumParticlesToCreate
            {
                constexpr auto operator()(auto& /*worker*/, auto& /*region*/, uint32_t n) const
                {
                    return n;
                }
            };

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(4u);
            }

            struct PlaceParticle
            {
                DINLINE constexpr void operator()(
                    auto const& /*worker*/,
                    auto& particle,
                    auto const& particleRegion,
                    uint32_t /*globalParticleIdx*/) const
                {
                    using namespace pmacc::spearhed::tags;
                    using namespace spearhed::tags;
                    auto const& aabb = particleRegion.volume;
                    pmacc::spearhed::for_each_tag<spearhed::CS>(
                        [&](auto tag)
                        {
                            *particle[relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * 0.5f;
                            *particle[vel][tag] = 100.0f;
                        });
                }
            };

            auto placeParticleArgs() const
            {
                return std::make_tuple();
            }

            template<typename PRBuf, typename DeviceHeapT>
            void setupRegions(PRBuf& prBuf, DeviceHeapT const& deviceHeap) const
            {
                using PRType = typename PRBuf::ParticleRegionType;
                prBuf.create(1);
                auto deviceHeapHandle = deviceHeap.getAllocatorHandle();
                auto region = PRType{deviceHeapHandle, {{0, 0, 0}, {-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}}};
                prBuf.pushBack(region);
                prBuf.buffer->hostToDevice();
            }
        };

        // Boundary AABB: [8,8,8] to [10,10,10], center (9,9,9).
        // PlaceParticle sets vel to 999.f
        // ParticlePush never touches the Boundary PRBuf, so positions must remain (9,9,9).
        struct BoundaryBlock
        {
            pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {8.0f, 8.0f, 8.0f}, {10.0f, 10.0f, 10.0f}};

            struct NumParticlesToCreate
            {
                constexpr auto operator()(auto& /*worker*/, auto& /*region*/, uint32_t n) const
                {
                    return n;
                }
            };

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(4u);
            }

            struct PlaceParticle
            {
                DINLINE constexpr void operator()(
                    auto const& /*worker*/,
                    auto& particle,
                    auto const& particleRegion,
                    uint32_t /*globalParticleIdx*/) const
                {
                    using namespace pmacc::spearhed::tags;
                    using namespace spearhed::tags;
                    auto const& aabb = particleRegion.volume;
                    pmacc::spearhed::for_each_tag<spearhed::CS>(
                        [&](auto tag)
                        {
                            *particle[relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * 0.5f;
                            *particle[vel][tag] = 999.0f;
                        });
                }
            };

            auto placeParticleArgs() const
            {
                return std::make_tuple();
            }

            template<typename PRBuf, typename DeviceHeapT>
            void setupRegions(PRBuf& prBuf, DeviceHeapT const& deviceHeap) const
            {
                using PRType = typename PRBuf::ParticleRegionType;
                prBuf.create(1);
                auto deviceHeapHandle = deviceHeap.getAllocatorHandle();
                auto region = PRType{deviceHeapHandle, {{0, 0, 0}, {8.0f, 8.0f, 8.0f}, {10.0f, 10.0f, 10.0f}}};
                prBuf.pushBack(region);
                prBuf.buffer->hostToDevice();
            }
        };

        InteriorBlock interior;
        BoundaryBlock boundary;

        template<typename Role>
        void setupRegions(auto& prBuf, auto const& dh) const
        {
            if constexpr(std::same_as<Role, pmacc::spearhed::roles::Interior>)
                interior.setupRegions(prBuf, dh);
            else
                boundary.setupRegions(prBuf, dh);
        }

        template<typename Role>
        using NumParticlesToCreate = std::conditional_t<
            std::same_as<Role, pmacc::spearhed::roles::Interior>,
            InteriorBlock::NumParticlesToCreate,
            BoundaryBlock::NumParticlesToCreate>;

        template<typename Role>
        auto numParticlesToCreateArgs() const
        {
            if constexpr(std::same_as<Role, pmacc::spearhed::roles::Interior>)
                return interior.numParticlesToCreateArgs();
            else
                return boundary.numParticlesToCreateArgs();
        }

        template<typename Role>
        using PlaceParticle = std::conditional_t<
            std::same_as<Role, pmacc::spearhed::roles::Interior>,
            InteriorBlock::PlaceParticle,
            BoundaryBlock::PlaceParticle>;

        template<typename Role>
        auto placeParticleArgs() const
        {
            if constexpr(std::same_as<Role, pmacc::spearhed::roles::Interior>)
                return interior.placeParticleArgs();
            else
                return boundary.placeParticleArgs();
        }
    };

} // namespace

TEST_CASE_METHOD(
    ParticleFixture,
    "Boundary: wall particles are frozen while interior particles advance",
    "[boundary][roles]")
{
    namespace roles = pmacc::spearhed::roles;

    BoundaryWallSetup setup;

    // Set up region geometry for each role (creates frames on the device heap)
    setup.template setupRegions<roles::Interior>(*prBuf, *deviceHeap);
    setup.template setupRegions<roles::Boundary>(*this->template prBufFor<roles::Boundary>(), *deviceHeap);

    // Multi-role InitParticles: fills Interior and Boundary PRBufs
    spearhed::InitParticles{}(setup);

    // Run the pusher - touches only the Interior PRBuf
    spearhed::ParticlePush{}(0);

    // Verify interior particles moved by vel * dt = 100 * 0.01 = 1.0
    prBuf->buffer->deviceToHost();
    auto interiorRegions = prBuf->buffer->getHostBuffer().getDataBox();
    auto& interiorFrameList = interiorRegions(0).particleFrameList;

    uint32_t interiorChecked = 0;
    for(auto& frame : interiorFrameList)
    {
        for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
        {
            auto particle = frame[slot];
            if(*particle[pmacc::spearhed::tags::multiMask])
            {
                pmacc::spearhed::for_each_tag<spearhed::CS>(
                    [&](auto tag)
                    {
                        float const pos = *particle[spearhed::relativePos][tag];
                        INFO("Interior relativePos component: " << pos << " (expected 1.0)");
                        REQUIRE(pos == Catch::Approx(1.0f).margin(1e-4f));
                    });
                ++interiorChecked;
            }
        }
    }
    INFO("Interior particles checked: " << interiorChecked);
    REQUIRE(interiorChecked > 0);

    // Verify boundary particles are frozen at their initial center (9, 9, 9)
    this->template prBufFor<roles::Boundary>()->buffer->deviceToHost();
    auto boundaryRegions = this->template prBufFor<roles::Boundary>()->buffer->getHostBuffer().getDataBox();
    auto& boundaryFrameList = boundaryRegions(0).particleFrameList;

    constexpr float expectedBoundaryPos = 9.0f; // center of [8,10]

    uint32_t boundaryChecked = 0;
    for(auto& frame : boundaryFrameList)
    {
        for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
        {
            auto particle = frame[slot];
            if(*particle[pmacc::spearhed::tags::multiMask])
            {
                pmacc::spearhed::for_each_tag<spearhed::CS>(
                    [&](auto tag)
                    {
                        float const pos = *particle[spearhed::relativePos][tag];
                        INFO("Boundary relativePos component: " << pos << " (expected " << expectedBoundaryPos << ")");
                        REQUIRE(pos == Catch::Approx(expectedBoundaryPos).margin(1e-4f));
                    });
                ++boundaryChecked;
            }
        }
    }
    INFO("Boundary particles checked: " << boundaryChecked);
    REQUIRE(boundaryChecked > 0);
}
