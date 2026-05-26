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
 * 2D physics test for the boundary-condition role system.
 *
 * Box2DSetup places a 20x20 fluid lattice inside [-1,-1] to [1,1] surrounded
 * by four wall strips of thickness 2*h0.  Interior particles start with an
 * outward radial velocity; wall particles are at rest.
 *
 * Five SPH steps (ParticlePush -> UpdateVolumes -> NeighbourSearch ->
 * UpdateDensity -> UpdateHydroForces -> EulerIntegrate) are executed.
 *
 * Assertions:
 *   1. Every boundary particle position equals its initial position within 1e-5.
 *   2. Every interior particle position lies inside [-1-h0, 1+h0]^2.
 */

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Acceleration.hpp"
#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/DuDt.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/SmoothingLength.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/particles/density/DensitySummation.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/pusher/EulerIntegrate.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spearhed/sph/HydroForces.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

namespace
{
    // Wall thickness = 2*h0 = one full kernel support radius
    constexpr spearhed::Real wt = spearhed::Real{2} * spearhed::h0;
    // SC lattice spacing: 20 particles across the 2-unit interior
    constexpr spearhed::Real sc_spacing = spearhed::Real{2} / spearhed::Real{20};
    constexpr uint32_t gridN = 20u;
    constexpr uint32_t totalInteriorParticles = gridN * gridN;

    // Shared physical properties
    constexpr spearhed::Real rho0 = spearhed::Real{1};
    constexpr spearhed::Real P0 = spearhed::Real{1};
    constexpr spearhed::Real u0 = P0 / ((spearhed::gamma_eos - spearhed::Real{1}) * rho0);
    constexpr spearhed::Real interiorArea = spearhed::Real{4}; // 2x2 square
    constexpr spearhed::Real particleMass = rho0 * interiorArea / static_cast<spearhed::Real>(totalInteriorParticles);
    // Outward velocity scale: gives ~0.5*h0 displacement per step
    constexpr spearhed::Real outwardScale = spearhed::Real{0.1f} * spearhed::h0 / spearhed::dt;

    struct Box2DSetup
    {
        using Roles = std::tuple<pmacc::spearhed::roles::Interior, pmacc::spearhed::roles::Boundary>;

        // Required by SetupInterface concept - represents the interior region's domain
        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0}, {-1.0f, -1.0f}, {1.0f, 1.0f}};

        // InteriorBlock

        struct InteriorBlock
        {
            struct NumParticlesToCreate
            {
                DINLINE constexpr uint32_t operator()(auto&, auto&, uint32_t n) const
                {
                    return n;
                }
            };

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(totalInteriorParticles);
            }

            struct PlaceParticle
            {
                DINLINE constexpr void operator()(auto const&, auto& particle, auto const&, uint32_t globalParticleIdx)
                    const
                {
                    using namespace pmacc::spearhed::tags;
                    using namespace spearhed::tags;

                    uint32_t const ix = globalParticleIdx % gridN;
                    uint32_t const iy = globalParticleIdx / gridN;
                    spearhed::Real const px
                        = spearhed::Real{-1} + (static_cast<spearhed::Real>(ix) + spearhed::Real{0.5}) * sc_spacing;
                    spearhed::Real const py
                        = spearhed::Real{-1} + (static_cast<spearhed::Real>(iy) + spearhed::Real{0.5}) * sc_spacing;

                    *particle[relativePos][x] = px;
                    *particle[relativePos][y] = py;

                    *particle[mass] = particleMass;
                    *particle[density] = rho0;
                    *particle[smoothingLength] = spearhed::h0;
                    *particle[internalEnergy] = u0;

                    // Outward radial velocity from origin
                    *particle[vel][x] = outwardScale * px;
                    *particle[vel][y] = outwardScale * py;

                    pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag)
                                                                { *particle[dvdt][tag] = spearhed::Real{0}; });
                    *particle[dudt] = spearhed::Real{0};
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
                auto dh = deviceHeap.getAllocatorHandle();
                prBuf.pushBack(PRType{dh, {{0, 0}, {-1.0f, -1.0f}, {1.0f, 1.0f}}});
                prBuf.buffer->hostToDevice();
            }
        };

        // BoundaryBlock

        struct BoundaryBlock
        {
            // Compute particle count from AABB dimensions and the SC lattice spacing
            struct NumParticlesToCreate
            {
                DINLINE constexpr uint32_t operator()(auto&, auto& particleRegion, uint32_t) const
                {
                    using namespace pmacc::spearhed::tags;
                    auto const& aabb = particleRegion.volume;
                    auto const nx
                        = static_cast<uint32_t>((aabb.max[x] - aabb.min[x]) / sc_spacing + spearhed::Real{0.5});
                    auto const ny
                        = static_cast<uint32_t>((aabb.max[y] - aabb.min[y]) / sc_spacing + spearhed::Real{0.5});
                    return nx * ny;
                }
            };

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(0u);
            }

            struct PlaceParticle
            {
                DINLINE constexpr void operator()(
                    auto const&,
                    auto& particle,
                    auto const& particleRegion,
                    uint32_t globalParticleIdx) const
                {
                    using namespace pmacc::spearhed::tags;
                    using namespace spearhed::tags;

                    auto const& aabb = particleRegion.volume;
                    uint32_t const nx
                        = static_cast<uint32_t>((aabb.max[x] - aabb.min[x]) / sc_spacing + spearhed::Real{0.5});
                    uint32_t const ix = globalParticleIdx % nx;
                    uint32_t const iy = globalParticleIdx / nx;

                    *particle[relativePos][x]
                        = aabb.min[x] + (static_cast<spearhed::Real>(ix) + spearhed::Real{0.5}) * sc_spacing;
                    *particle[relativePos][y]
                        = aabb.min[y] + (static_cast<spearhed::Real>(iy) + spearhed::Real{0.5}) * sc_spacing;

                    *particle[mass] = particleMass;
                    *particle[density] = rho0;
                    *particle[smoothingLength] = spearhed::h0;
                    *particle[internalEnergy] = u0;

                    pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag)
                                                                { *particle[vel][tag] = spearhed::Real{0}; });
                    pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag)
                                                                { *particle[dvdt][tag] = spearhed::Real{0}; });
                    *particle[dudt] = spearhed::Real{0};
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
                prBuf.create(4);
                auto dh = deviceHeap.getAllocatorHandle();
                // Left:   [-1-wt, -1-wt] to [-1,    1+wt]  (full height, covers corners)
                prBuf.pushBack(PRType{dh, {{0, 0}, {-1.0f - wt, -1.0f - wt}, {-1.0f, 1.0f + wt}}});
                // Right:  [  1,   -1-wt] to [ 1+wt, 1+wt]  (full height, covers corners)
                prBuf.pushBack(PRType{dh, {{0, 0}, {1.0f, -1.0f - wt}, {1.0f + wt, 1.0f + wt}}});
                // Bottom: [-1,   -1-wt] to [  1,   -1   ]
                prBuf.pushBack(PRType{dh, {{0, 0}, {-1.0f, -1.0f - wt}, {1.0f, -1.0f}}});
                // Top:    [-1,    1   ] to [  1,    1+wt ]
                prBuf.pushBack(PRType{dh, {{0, 0}, {-1.0f, 1.0f}, {1.0f, 1.0f + wt}}});
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
    "Boundary: 2D box -- wall particles frozen, interior confined after 5 SPH steps",
    "[boundary][roles][sph][2d]")
{
    namespace roles = pmacc::spearhed::roles;
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    Box2DSetup setup;

    setup.template setupRegions<roles::Interior>(*prBuf, *deviceHeap);
    setup.template setupRegions<roles::Boundary>(*this->template prBufFor<roles::Boundary>(), *deviceHeap);
    spearhed::InitParticles{}(setup);

    // capture boundary initial positions

    auto& boundaryBuf = *this->template prBufFor<roles::Boundary>();
    boundaryBuf.buffer->deviceToHost();
    int64_t const initHeapOffset = spearhed::syncHeapToHost();
    auto initBox = boundaryBuf.buffer->getHostBuffer().getDataBox();

    std::vector<std::array<spearhed::Real, 2>> initBoundaryPos;
    for(int r = 0; r < boundaryBuf.size; ++r)
    {
        auto& frameList = initBox[r].particleFrameList;
        for(auto& frame : frameList.hostIterable(initHeapOffset))
        {
            for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
            {
                auto particle = frame[slot];
                if(*particle[multiMask])
                    initBoundaryPos.push_back({*particle[relativePos][x], *particle[relativePos][y]});
            }
        }
    }

    // 5-steps of SPH physics loop

    using K = spearhed::CubicSplineKernel;
    constexpr auto interactionRadius = static_cast<spearhed::CS::T_Axis>(K::supportRadius) * spearhed::h0;

    for(uint32_t step = 0; step < 5u; ++step)
    {
        spearhed::ParticlePush{}(step);
        pmacc::spearhed::UpdateVolumes<spearhed::PRType>{}();
        auto bundle = pmacc::spearhed::CalculateNeighbourRegions{}(
            *prBuf,
            interactionRadius,
            *prBuf,
            *this->template prBufFor<roles::Boundary>());
        spearhed::UpdateDensity<K>{}(bundle, spearhed::h0);
        spearhed::UpdateHydroForces<K>{spearhed::gamma_eos}(bundle, spearhed::h0);
        pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, spearhed::EulerIntegrate{}, spearhed::dt);
    }

    // CHECK 1: boundary positions frozen

    {
        boundaryBuf.buffer->deviceToHost();
        int64_t const heapOffset = spearhed::syncHeapToHost();
        auto finalBox = boundaryBuf.buffer->getHostBuffer().getDataBox();

        uint32_t checkedCount = 0u;
        std::size_t posIdx = 0u;
        for(int r = 0; r < boundaryBuf.size; ++r)
        {
            auto& frameList = finalBox[r].particleFrameList;
            for(auto& frame : frameList.hostIterable(heapOffset))
            {
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                {
                    auto particle = frame[slot];
                    if(*particle[multiMask])
                    {
                        REQUIRE(
                            static_cast<double>(*particle[relativePos][x])
                            == Catch::Approx(static_cast<double>(initBoundaryPos[posIdx][0])).margin(1e-5));
                        REQUIRE(
                            static_cast<double>(*particle[relativePos][y])
                            == Catch::Approx(static_cast<double>(initBoundaryPos[posIdx][1])).margin(1e-5));
                        ++posIdx;
                        ++checkedCount;
                    }
                }
            }
        }
        REQUIRE(checkedCount > 0u);
    }

    // CHECK 2: interior confined in [-1-h0, 1+h0]^2

    {
        prBuf->buffer->deviceToHost();
        int64_t const heapOffset = spearhed::syncHeapToHost();
        auto interiorBox = prBuf->buffer->getHostBuffer().getDataBox();

        constexpr spearhed::Real lo = -1.0f - spearhed::h0;
        constexpr spearhed::Real hi = 1.0f + spearhed::h0;

        uint32_t checkedCount = 0u;
        for(int r = 0; r < prBuf->size; ++r)
        {
            auto& frameList = interiorBox[r].particleFrameList;
            for(auto& frame : frameList.hostIterable(heapOffset))
            {
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                {
                    auto particle = frame[slot];
                    if(*particle[multiMask])
                    {
                        spearhed::Real const px = *particle[relativePos][x];
                        spearhed::Real const py = *particle[relativePos][y];
                        REQUIRE(px >= lo);
                        REQUIRE(px <= hi);
                        REQUIRE(py >= lo);
                        REQUIRE(py <= hi);
                        ++checkedCount;
                    }
                }
            }
        }
        REQUIRE(checkedCount > 0u);
    }
}

/**
 * Verifies InteractParticlesUnified (single launch, own-state persisted in SMEM)
 * produces the same interior densities as the reference InteractParticles
 * (one launch per source) in a multi-source scene (interior + boundary).
 */
TEST_CASE_METHOD(
    ParticleFixture,
    "Boundary: InteractParticlesUnified produces same density as InteractParticles",
    "[boundary][roles][unified]")
{
    namespace roles = pmacc::spearhed::roles;
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    Box2DSetup setup;
    setup.template setupRegions<roles::Interior>(*prBuf, *deviceHeap);
    setup.template setupRegions<roles::Boundary>(*this->template prBufFor<roles::Boundary>(), *deviceHeap);
    spearhed::InitParticles{}(setup);

    using K = spearhed::CubicSplineKernel;
    constexpr auto interactionRadius = static_cast<spearhed::CS::T_Axis>(K::supportRadius) * spearhed::h0;

    // Build a neighbour bundle for the multi-source scene (interior + boundary).
    // The same bundle drives both interaction strategies below.
    auto bundle = pmacc::spearhed::CalculateNeighbourRegions{}(
        *prBuf,
        interactionRadius,
        *prBuf,
        *this->template prBufFor<roles::Boundary>());

    // Reads every live interior particle's density in deterministic (region, frame, slot)
    // order. No particles move between the two passes, so the frame layout - and thus this
    // ordering - is identical, which makes the index-by-index comparison below valid.
    auto readInteriorDensities = [&]()
    {
        prBuf->buffer->deviceToHost();
        int64_t const heapOffset = spearhed::syncHeapToHost();
        auto box = prBuf->buffer->getHostBuffer().getDataBox();

        std::vector<double> out;
        for(int r = 0; r < prBuf->size; ++r)
        {
            auto& frameList = box[r].particleFrameList;
            for(auto& frame : frameList.hostIterable(heapOffset))
            {
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                {
                    auto particle = frame[slot];
                    if(*particle[multiMask])
                        out.push_back(static_cast<double>(*particle[density]));
                }
            }
        }
        return out;
    };

    // Reference: classic per-source InteractParticles (one kernel launch per source).
    pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, spearhed::DensityInitSelf<K>{});
    pmacc::spearhed::InteractParticles{}(bundle, interactionRadius, spearhed::AccumulateDensity<K>{});
    auto const referenceDensities = readInteriorDensities();

    // Unified: single-launch kernel over the same bundle (re-seed the self term first).
    pmacc::spearhed::ForEachParticleInPRBuf{}(*prBuf, spearhed::DensityInitSelf<K>{});
    pmacc::spearhed::InteractParticlesUnified{}(bundle, interactionRadius, spearhed::AccumulateDensity<K>{});
    auto const unifiedDensities = readInteriorDensities();

    REQUIRE(!referenceDensities.empty());
    REQUIRE(referenceDensities.size() == unifiedDensities.size());
    for(std::size_t i = 0; i < referenceDensities.size(); ++i)
    {
        REQUIRE(referenceDensities[i] > 0.0);
        REQUIRE(std::isfinite(unifiedDensities[i]));
        REQUIRE(unifiedDensities[i] == Catch::Approx(referenceDensities[i]).epsilon(1e-5));
    }
}
