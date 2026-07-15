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
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "spearhed/sph/HydroForces.hpp"

#include "TestSetup.hpp"
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
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/sph/KernelVariant.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/InteractParticles.hpp"
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

/**
 * Unit test for HydroInteraction.
 *
 * Strategy: place 2 particles along the x-axis at absolute positions -d/2 and +d/2,
 * both at rest, with identical mass, density, internal energy, and smoothing length.
 *
 * With v_i == v_j:
 *   dudt = P/rho^2 * m * dot(v_i - v_j, gradW) = 0  (exactly)
 *
 * For dvdt along x (symmetric SPH, omega=1):
 *   dv_i/dt_x = -m * 2 * P/rho^2 * dWdr(d, h) * (r_vec_x / d)
 *
 * Left particle (x = -d/2): r_vec_x = -d, so factor = +1
 *   dvdt_x = m * 2 * P/rho^2 * dWdr(d, h)   [negative since dWdr < 0 -> pushed left]
 * Right particle (x = +d/2): equal magnitude, opposite sign -> pushed right.
 * y and z components are zero by symmetry.
 */

namespace
{
    constexpr spearhed::Real TEST_H = spearhed::Real{0.5};
    constexpr spearhed::Real TEST_MASS = spearhed::Real{1.0};
    constexpr spearhed::Real TEST_RHO = spearhed::Real{1.0};
    constexpr spearhed::Real TEST_U = spearhed::Real{1.0};
    // Separation between the two particles; must be < 2*TEST_H
    constexpr spearhed::Real TEST_D = spearhed::Real{0.4f};

    struct InitMomEnergyTestSetup
    {
        // This setup fills a single species and acts as its own (only) init block.
        using Species = pmacc::spearhed::species::Default;

        pmacc::spearhed::AABB<spearhed::CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        static constexpr uint32_t N = 2u;

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
                auto const& /*worker*/,
                auto& particle,
                auto const& /*particleRegion*/,
                uint32_t globalParticleIdx) const
            {
                using namespace pmacc::spearhed::tags;
                using namespace spearhed::tags;

                // Particle 0 at x = -TEST_D/2, particle 1 at x = +TEST_D/2.
                // All y and z coords = 0. AABB origin=(0,0,0) so relativePos == absolute pos.
                constexpr spearhed::Real half_d = TEST_D * spearhed::Real{0.5};
                bool const isLeft = (globalParticleIdx % 2u == 0u);

                *particle[relativePos][x] = isLeft ? -half_d : half_d;
                *particle[relativePos][y] = spearhed::Real{0};
                *particle[relativePos][z] = spearhed::Real{0};

                *particle[mass] = TEST_MASS;
                *particle[smoothingLength] = TEST_H;
                *particle[density] = TEST_RHO;
                *particle[internalEnergy] = TEST_U;

                // Both particles at rest
                pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag)
                                                            { *particle[vel][tag] = spearhed::Real{0}; });

                // Accumulators start at zero
                pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag)
                                                            { *particle[dvdt][tag] = spearhed::Real{0}; });
                *particle[dudt] = spearhed::Real{0};
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

} // namespace

TEST_CASE_METHOD(
    ParticleFixture,
    "HydroForces: two particles at rest repel along x, dudt == 0",
    "[sph][momentum][energy]")
{
    auto setup = InitMomEnergyTestSetup{};
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

            auto sources = bundle.template selectByRole<pmacc::spearhed::roles::Source>();
            using PRType = spearhed::PRType;
            pmacc::spearhed::FrameIndexBuffer<PRType> index{*prBuf};
            pmacc::spearhed::interact(
                sources,
                *prBuf,
                index,
                static_cast<spearhed::CS::T_Axis>(K::supportRadius) * TEST_H,
                spearhed::HydroInteraction<K>{spearhed::gamma_eos})
                .waitForFinished();

            prBuf->buffer->deviceToHost();
            int64_t const heapOffset = spearhed::syncHeapToHost();
            auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();
            auto& frameList = hostRegions(0).particleFrameList;

            // Analytic expected dvdt_x for the left particle:
            //   P = (gamma - 1) * rho * u
            //   dvdt_x = m * 2 * P/rho^2 * dWdr(d, h)   (negative value since dWdr < 0)
            spearhed::Real const P = spearhed::pressure(spearhed::gamma_eos, TEST_RHO, TEST_U);
            spearhed::Real const dw = K::dWdr(TEST_D, TEST_H);
            spearhed::Real const expected_left = TEST_MASS * 2 * P / (TEST_RHO * TEST_RHO) * dw;
            uint32_t checkedCount = 0;
            for(auto& frame : frameList.hostIterable(heapOffset))
            {
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                {
                    auto particle = frame[slot];
                    if(*particle[pmacc::spearhed::tags::multiMask])
                    {
                        using namespace spearhed::tags;
                        using namespace pmacc::spearhed::tags;

                        // dudt must be zero (both particles at rest)
                        REQUIRE(static_cast<double>(*particle[dudt]) == Catch::Approx(0.0).margin(1e-6));

                        // y and z components of dvdt must be zero
                        REQUIRE(static_cast<double>(*particle[dvdt][y]) == Catch::Approx(0.0).margin(1e-6));
                        REQUIRE(static_cast<double>(*particle[dvdt][z]) == Catch::Approx(0.0).margin(1e-6));

                        // x component: sign depends on which side of x=0 the particle is on
                        spearhed::Real const abs_x = *particle[relativePos][x];
                        bool const isLeft = (abs_x < spearhed::Real{0});
                        spearhed::Real const expected_x = isLeft ? expected_left : -expected_left;
                        REQUIRE(
                            static_cast<double>(*particle[dvdt][x])
                            == Catch::Approx(static_cast<double>(expected_x)).epsilon(1e-4));

                        ++checkedCount;
                    }
                }
            }
            REQUIRE(checkedCount == InitMomEnergyTestSetup::N);
        },
        setup.kernelVariant);
}
