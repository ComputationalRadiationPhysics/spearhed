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
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/particles/pusher/EulerIntegrate.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spearhed/sph/HydroForces.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
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
        // Required by SetupInterface concept - represents the interior region's domain
        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0}, {-1.0f, -1.0f}, {1.0f, 1.0f}};

        // InteriorBlock fills the Default species (interior + movable + source).

        struct InteriorBlock
        {
            using Species = pmacc::spearhed::species::Default;

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

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
            {
                out.push_back(pmacc::spearhed::AABB<spearhed::CS>{{0, 0}, {-1.0f, -1.0f}, {1.0f, 1.0f}});
            }
        };

        // BoundaryBlock fills the Boundary species (frozen + source).

        struct BoundaryBlock
        {
            using Species = pmacc::spearhed::species::Boundary;

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

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
            {
                using AABB = pmacc::spearhed::AABB<spearhed::CS>;
                // Left:   [-1-wt, -1-wt] to [-1,    1+wt]  (full height, covers corners)
                out.push_back(AABB{{0, 0}, {-1.0f - wt, -1.0f - wt}, {-1.0f, 1.0f + wt}});
                // Right:  [  1,   -1-wt] to [ 1+wt, 1+wt]  (full height, covers corners)
                out.push_back(AABB{{0, 0}, {1.0f, -1.0f - wt}, {1.0f + wt, 1.0f + wt}});
                // Bottom: [-1,   -1-wt] to [  1,   -1   ]
                out.push_back(AABB{{0, 0}, {-1.0f, -1.0f - wt}, {1.0f, -1.0f}});
                // Top:    [-1,    1   ] to [  1,    1+wt ]
                out.push_back(AABB{{0, 0}, {-1.0f, 1.0f}, {1.0f, 1.0f + wt}});
            }
        };

        InteriorBlock interior;
        BoundaryBlock boundary;

        // The setup just lists its init blocks; each block names the species it fills. Init iterates
        // this tuple. The two blocks are organised independently of any role - roles live on the
        // species (Default: Interior+Movable+Source, Boundary: Frozen+Source).
        auto blocks() const
        {
            return std::tie(interior, boundary);
        }
    };

} // namespace

TEST_CASE_METHOD(
    ParticleFixture,
    "Boundary: 2D box -- wall particles frozen, interior confined after 5 SPH steps",
    "[boundary][roles][sph][2d]")
{
    namespace species = pmacc::spearhed::species;
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    Box2DSetup setup;

    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    // capture boundary initial positions

    auto& boundaryBuf = *this->template prBufFor<species::Boundary>();
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
            *this->template prBufFor<species::Boundary>());
        spearhed::UpdateDensity<K>{}(bundle, spearhed::h0);
        spearhed::UpdateHydroForces<K>{spearhed::gamma_eos}(bundle, spearhed::h0);
        pmacc::spearhed::launchForEach(
            pmacc::spearhed::levels::particle,
            *prBuf,
            spearhed::EulerIntegrate{},
            spearhed::dt);
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
    namespace species = pmacc::spearhed::species;
    using namespace pmacc::spearhed::tags;
    using namespace spearhed::tags;

    Box2DSetup setup;
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    using K = spearhed::CubicSplineKernel;
    constexpr auto interactionRadius = static_cast<spearhed::CS::T_Axis>(K::supportRadius) * spearhed::h0;

    // Build a neighbour bundle for the multi-source scene (interior + boundary).
    // The same bundle drives both interaction strategies below.
    auto bundle = pmacc::spearhed::CalculateNeighbourRegions{}(
        *prBuf,
        interactionRadius,
        *prBuf,
        *this->template prBufFor<species::Boundary>());

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
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, spearhed::DensityInitSelf<K>{});
    pmacc::spearhed::InteractParticles{}(bundle, interactionRadius, spearhed::AccumulateDensity<K>{});
    auto const referenceDensities = readInteriorDensities();

    // Unified: single-launch kernel over the same bundle (re-seed the self term first).
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, spearhed::DensityInitSelf<K>{});
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

namespace
{
    // Counts the live particles across every region of a species buffer.
    template<typename Buf>
    int countLiveParticles(Buf& buf)
    {
        using namespace pmacc::spearhed::tags;
        using namespace spearhed::tags;
        buf.buffer->deviceToHost();
        int64_t const heapOffset = spearhed::syncHeapToHost();
        auto box = buf.buffer->getHostBuffer().getDataBox();

        int live = 0;
        for(int r = 0; r < buf.size; ++r)
        {
            auto& frameList = box[r].particleFrameList;
            for(auto& frame : frameList.hostIterable(heapOffset))
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                    if(*frame[slot][multiMask])
                        ++live;
        }
        return live;
    }

    // Minimal placement: drop each particle at its region centre. SPH attributes are left zeroed
    // by the initializer; these tests only assert on region and particle counts.
    struct PlaceAtCentre
    {
        DINLINE constexpr void operator()(auto const&, auto& particle, auto const& region, uint32_t) const
        {
            using namespace pmacc::spearhed::tags;
            auto const& aabb = region.volume;
            pmacc::spearhed::for_each_tag<spearhed::CS>(
                [&](auto tag)
                { *particle[relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * spearhed::Real{0.5}; });
        }
    };

    struct FixedCount
    {
        DINLINE constexpr uint32_t operator()(auto&, auto&, uint32_t n) const
        {
            return n;
        }
    };

    // One init block that fills the Default species with a single region of `count` particles.
    struct InteriorChunk
    {
        using Species = pmacc::spearhed::species::Default;

        spearhed::Real centreX;
        uint32_t count;

        using NumParticlesToCreate = FixedCount;

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(count);
        }

        using PlaceParticle = PlaceAtCentre;

        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        template<typename>
        void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
        {
            out.push_back(
                pmacc::spearhed::AABB<spearhed::CS>{{0, 0}, {centreX - 0.4f, -0.4f}, {centreX + 0.4f, 0.4f}});
        }
    };

    // Two independent blocks, both targeting the Default species. Their regions are concatenated
    // into Default's single buffer (overlap would be allowed; here they are merely adjacent).
    struct TwoBlocksOneSpecies
    {
        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0}, {-1.0f, -1.0f}, {1.0f, 1.0f}};
        InteriorChunk first{-0.5f, 3u};
        InteriorChunk second{0.5f, 5u};

        auto blocks() const
        {
            return std::tie(first, second);
        }
    };

    // One init block that fills two species at once: an Interior region on the left half and a
    // Boundary region on the right half. The counting/placement recipe is shared across both.
    struct SplitBlock
    {
        using Species = std::tuple<pmacc::spearhed::species::Default, pmacc::spearhed::species::Boundary>;

        using NumParticlesToCreate = FixedCount;

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(4u);
        }

        using PlaceParticle = PlaceAtCentre;

        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        template<typename S>
        void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
        {
            using AABB = pmacc::spearhed::AABB<spearhed::CS>;
            if constexpr(std::same_as<S, pmacc::spearhed::species::Default>)
                out.push_back(AABB{{0, 0}, {-1.0f, -1.0f}, {0.0f, 1.0f}});
            else
                out.push_back(AABB{{0, 0}, {0.0f, -1.0f}, {1.0f, 1.0f}});
        }
    };

    struct OneBlockTwoSpecies
    {
        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0}, {-1.0f, -1.0f}, {1.0f, 1.0f}};
        SplitBlock block;

        auto blocks() const
        {
            return std::tie(block);
        }
    };
} // namespace

TEST_CASE_METHOD(
    ParticleFixture,
    "Init: two blocks feed one species -- regions concatenated",
    "[boundary][roles][init]")
{
    TwoBlocksOneSpecies setup;
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    // Interior's buffer holds both blocks' regions back to back, each filled by its own recipe.
    REQUIRE(prBuf->size == 2);
    REQUIRE(countLiveParticles(*prBuf) == 3 + 5);
}

TEST_CASE_METHOD(ParticleFixture, "Init: one block feeds two species", "[boundary][roles][init]")
{
    namespace species = pmacc::spearhed::species;

    OneBlockTwoSpecies setup;
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    auto& boundary = *this->template prBufFor<species::Boundary>();

    REQUIRE(prBuf->size == 1);
    REQUIRE(boundary.size == 1);
    REQUIRE(countLiveParticles(*prBuf) == 4);
    REQUIRE(countLiveParticles(boundary) == 4);
}
