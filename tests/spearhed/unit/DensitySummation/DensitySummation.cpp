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
 * Unit test for SPH density summation.
 *
 * Strategy: place N particles all at the SAME point in a single region.
 * With spacing zero, every pair has r = 0, so W(r,h) = W(0,h) = sigma/h^dim.
 * Expected density for each particle = N * m * W(0, h).
 *
 * This fully exercises both DensityInitSelf (self term) and AccumulateDensity
 * (pairwise neighbours), and gives an exact analytic result independent of
 * spatial configuration.
 */

#include "spearhed/particles/density/DensitySummation.hpp"

#include "TestSetup.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/param.hpp"
#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/SmoothingLength.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spearhed/sph/KernelVariant.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/InteractParticles.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <cmath>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;

using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

namespace
{

    constexpr spearhed::Real TEST_H = spearhed::Real{0.5};
    constexpr spearhed::Real TEST_MASS = spearhed::Real{1.0};

    /**
     * Initialises each particle: place all at AABB centre, set mass and smoothingLength.
     */
    struct InitDensityTestParticle
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
            // All particles at AABB centre
            pmacc::spearhed::for_each_tag<spearhed::CS>(
                [&](auto tag) { particle[relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * spearhed::Real{0.5}; });

            particle[mass] = TEST_MASS;
            particle[smoothingLength] = TEST_H;
            particle[density] = spearhed::Real{0};
        }
    };

    struct InitDensityTestSetup
    {
        // This setup fills a single species and acts as its own (only) init block.
        using Species = pmacc::spearhed::species::Default;

        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        static constexpr uint32_t N = 4u;

        auto blocks() const
        {
            return std::tie(*this);
        }

        struct NumParticlesToCreate
        {
            constexpr auto operator()(auto& /*worker*/, auto& /*region*/, uint32_t n) const
            {
                return n;
            }
        };

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(N);
        }

        struct PlaceParticle
        {
            DINLINE constexpr void operator()(
                auto const& worker,
                auto& particle,
                auto const& particleRegion,
                uint32_t globalParticleIdx) const
            {
                InitDensityTestParticle{}(worker, particle, particleRegion, globalParticleIdx);
            }
        };

        spearhed::KernelVariant kernelVariant = makeKernel(spearhed::KernelType::CubicSpline);

        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        template<typename>
        void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
        {
            out.push_back(pmacc::spearhed::AABB<spearhed::CS>{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}});
        }
    };

    // Three particles spaced along x-axis: dx < 2h so adjacent pairs interact,
    // 2*dx > 2h so non-adjacent pair does not.
    constexpr spearhed::Real SPACED_H = spearhed::Real{1.0};
    constexpr spearhed::Real SPACED_MASS = spearhed::Real{1.5};
    constexpr spearhed::Real SPACED_DX = spearhed::Real{1.5}; // dx=1.5 < 2h=2.0 < 2*dx=3.0

    struct InitSpacedParticle
    {
        DINLINE constexpr void operator()(
            auto const& /*worker*/,
            auto& particle,
            auto const& /*particleRegion*/,
            uint32_t globalParticleIdx) const
        {
            using namespace pmacc::spearhed::tags;
            using namespace spearhed::tags;

            // particle i placed at x = i*dx, y=z=0
            pmacc::spearhed::for_each_enum_tag<spearhed::CS>(
                [&](auto axisIdx, auto tag)
                {
                    particle[relativePos][tag]
                        = (axisIdx.value == 0) ? spearhed::Real(globalParticleIdx) * SPACED_DX : spearhed::Real{0};
                });

            particle[mass] = SPACED_MASS;
            particle[smoothingLength] = SPACED_H;
            particle[density] = spearhed::Real{0};
        }
    };

    struct InitSpacedTestSetup
    {
        // This setup fills a single species and acts as its own (only) init block.
        using Species = pmacc::spearhed::species::Default;

        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0}};

        static constexpr uint32_t N = 3u;

        auto blocks() const
        {
            return std::tie(*this);
        }

        struct NumParticlesToCreate
        {
            constexpr auto operator()(auto& /*worker*/, auto& /*region*/, uint32_t n) const
            {
                return n;
            }
        };

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(N);
        }

        struct PlaceParticle
        {
            DINLINE constexpr void operator()(
                auto const& worker,
                auto& particle,
                auto const& particleRegion,
                uint32_t globalParticleIdx) const
            {
                InitSpacedParticle{}(worker, particle, particleRegion, globalParticleIdx);
            }
        };

        spearhed::KernelVariant kernelVariant = makeKernel(spearhed::KernelType::CubicSpline);

        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        template<typename>
        void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& out) const
        {
            out.push_back(pmacc::spearhed::AABB<spearhed::CS>{{0, 0, 0}, {-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0}});
        }
    };

} // namespace

TEST_CASE_METHOD(
    ParticleFixture,
    "DensitySummation: N co-located particles each have density N*m*W(0,h)",
    "[sph][density]")
{
    auto setup = InitDensityTestSetup{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    // All-to-all neighbour graph (single region)
    constexpr int numRegions = 1;
    pmacc::HostDeviceBuffer<unsigned int, 1> neighbourRegions(numRegions * numRegions);
    pmacc::HostDeviceBuffer<unsigned int, 1> regionOffsets(numRegions + 1);
    neighbourRegions.getHostBuffer().data()[0] = 0;
    regionOffsets.getHostBuffer().data()[0] = 0;
    regionOffsets.getHostBuffer().data()[1] = 1;
    neighbourRegions.hostToDevice();
    regionOffsets.hostToDevice();

    using PRBufType = pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>;
    auto bundle = pmacc::spearhed::makeNeighbourBundle(
        pmacc::spearhed::NeighbourEntry<PRBufType>{
            prBuf.get(),
            std::move(neighbourRegions),
            std::move(regionOffsets)});

    std::visit(
        [&](auto kernel)
        {
            using K = std::decay_t<decltype(kernel)>;
            // Self-contribution first, then pairwise accumulation
            pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, spearhed::DensityInitSelf<K>{});
            auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
            using PRType = spearhed::PRType;
            pmacc::spearhed::FrameIndexBuffer<PRType> index{*prBuf};
            pmacc::spearhed::interact(
                sources,
                *prBuf,
                index,
                static_cast<spearhed::CS::T_Axis>(K::supportRadius) * TEST_H,
                spearhed::AccumulateDensity<K>{})
                .waitForFinished();
        },
        setup.kernelVariant);


    // Read densities back to host
    prBuf->buffer->deviceToHost();
    int64_t const heapOffset = spearhed::syncHeapToHost();
    auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();
    auto& frameList = hostRegions(0).particleFrameList;

    spearhed::Real const expected
        = spearhed::Real(InitDensityTestSetup::N) * TEST_MASS * spearhed::CubicSplineKernel::W(0.0f, TEST_H);

    uint32_t checkedCount = 0;
    for(auto& frame : frameList.hostIterable(heapOffset))
    {
        for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
        {
            auto particle = frame[slot];
            if(particle[pmacc::spearhed::tags::multiMask])
            {
                spearhed::Real const rho = particle[spearhed::tags::density];
                REQUIRE(static_cast<double>(rho) == Catch::Approx(static_cast<double>(expected)).epsilon(1e-5));
                ++checkedCount;
            }
        }
    }
    REQUIRE(checkedCount == InitDensityTestSetup::N);
}

TEST_CASE_METHOD(
    ParticleFixture,
    "DensitySummation: 3 particles in a line have correct neighbour-dependent densities",
    "[sph][density]")
{
    auto setup = InitSpacedTestSetup{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    constexpr int numRegions = 1;
    pmacc::HostDeviceBuffer<unsigned int, 1> neighbourRegions2(numRegions * numRegions);
    pmacc::HostDeviceBuffer<unsigned int, 1> regionOffsets2(numRegions + 1);
    neighbourRegions2.getHostBuffer().data()[0] = 0;
    regionOffsets2.getHostBuffer().data()[0] = 0;
    regionOffsets2.getHostBuffer().data()[1] = 1;
    neighbourRegions2.hostToDevice();
    regionOffsets2.hostToDevice();

    using PRBufType2 = pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType>;
    auto bundle2 = pmacc::spearhed::makeNeighbourBundle(
        pmacc::spearhed::NeighbourEntry<PRBufType2>{
            prBuf.get(),
            std::move(neighbourRegions2),
            std::move(regionOffsets2)});

    std::visit(
        [&](auto kernel)
        {
            using K = std::decay_t<decltype(kernel)>;
            pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, *prBuf, spearhed::DensityInitSelf<K>{});
            auto sources = bundle2.template selectByRole<pmacc::spearhed::roles::Source>();
            using PRType = spearhed::PRType;
            pmacc::spearhed::FrameIndexBuffer<PRType> index{*prBuf};
            pmacc::spearhed::interact(
                sources,
                *prBuf,
                index,
                static_cast<spearhed::CS::T_Axis>(K::supportRadius) * SPACED_H,
                spearhed::AccumulateDensity<K>{})
                .waitForFinished();

            prBuf->buffer->deviceToHost();
            int64_t const heapOffset = spearhed::syncHeapToHost();
            auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();
            auto& frameList = hostRegions(0).particleFrameList;

            // rho_edge: self + one neighbour at distance dx
            spearhed::Real const rho_edge = SPACED_MASS
                                            * (spearhed::CubicSplineKernel::W(spearhed::Real{0}, SPACED_H)
                                               + spearhed::CubicSplineKernel::W(SPACED_DX, SPACED_H));
            // rho_mid: self + two neighbours at distance dx
            spearhed::Real const rho_mid = SPACED_MASS
                                           * (spearhed::CubicSplineKernel::W(spearhed::Real{0}, SPACED_H)
                                              + 2 * spearhed::CubicSplineKernel::W(SPACED_DX, SPACED_H));

            uint32_t countEdge = 0;
            uint32_t countMid = 0;
            for(auto& frame : frameList.hostIterable(heapOffset))
            {
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                {
                    auto particle = frame[slot];
                    if(particle[pmacc::spearhed::tags::multiMask])
                    {
                        spearhed::Real const rho = particle[spearhed::tags::density];
                        double const rho_d = static_cast<double>(rho);
                        if(Catch::Approx(rho_d).epsilon(1e-5) == static_cast<double>(rho_edge))
                            ++countEdge;
                        else if(Catch::Approx(rho_d).epsilon(1e-5) == static_cast<double>(rho_mid))
                            ++countMid;
                        else
                            FAIL("Unexpected density value: " << rho_d);
                    }
                }
            }
            REQUIRE(countEdge == 2u);
            REQUIRE(countMid == 1u);
        },
        setup.kernelVariant);
}
