#include "TestMpiContext.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/control/Simulation.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/param/setup.hpp"
#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/particles/density/DensitySummation.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/particles/pusher/EulerIntegrate.hpp"
#include "spearhed/particles/pusher/ParticlePush.hpp"
#include "spearhed/sph/HydroForces.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/algorithms/LaunchForEach.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace spearhed
{
    struct SimulationTestAccess
    {
        static auto& frameIndices(Simulation& simulation)
        {
            return simulation.frameIndices;
        }
    };
} // namespace spearhed

namespace
{
    using namespace pmacc::spearhed;
    using Default = species::Default;
    using Boundary = species::Boundary;
    using Tracer = species::Tracer;
    using AABB = pmacc::spearhed::AABB<spearhed::CS>;

    template<typename T_Species, uint32_t T_Count, float T_Position, float T_Velocity>
    struct TinyBlock
    {
        using Species = T_Species;

        struct NumParticlesToCreate
        {
            constexpr uint32_t operator()(auto const&, auto const&) const
            {
                return T_Count;
            }
        };

        auto numParticlesToCreateArgs() const
        {
            return std::tuple{};
        }

        auto placeParticleArgs() const
        {
            return std::tuple{};
        }

        struct PlaceParticle
        {
            DINLINE void operator()(auto const&, auto& particle, auto const&, uint32_t globalIdx) const
            {
                using namespace pmacc::spearhed::tags;
                using namespace spearhed::tags;
                pmacc::spearhed::for_each_enum_tag<spearhed::CS>(
                    [&](auto axis, auto tag)
                    {
                        if constexpr(decltype(axis)::value == 0u)
                            particle[relativePos][tag] = T_Position;
                        else
                            particle[relativePos][tag] = 0.45f;
                    });
                pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag) { particle[vel][tag] = T_Velocity; });
                particle[mass] = spearhed::Real{1};
                if constexpr(std::same_as<T_Species, Default> || std::same_as<T_Species, Boundary>)
                {
                    particle[density] = spearhed::Real{1};
                    particle[internalEnergy] = spearhed::Real{1};
                    pmacc::spearhed::for_each_tag<spearhed::CS>([&](auto tag)
                                                                { particle[dvdt][tag] = spearhed::Real{0}; });
                    particle[dudt] = spearhed::Real{0};
                    particle[smoothingLength] = spearhed::h0;
                }
                static_cast<void>(globalIdx);
            }
        };

        template<typename>
        void addRegions(std::vector<AABB>& regions) const
        {
            constexpr float halfWidth = 0.005f;
            AABB volume;
            pmacc::spearhed::for_each_enum_tag<spearhed::CS>(
                [&](auto axis, auto tag)
                {
                    auto const center = decltype(axis)::value == 0u ? T_Position : 0.45f;
                    volume.origin[tag] = 0.0f;
                    volume.min[tag] = center - halfWidth;
                    volume.max[tag] = center + halfWidth;
                });
            regions.push_back(volume);
        }
    };

    template<uint32_t T_DefaultCount, uint32_t T_BoundaryCount, uint32_t T_TracerCount>
    struct TinySetupFor
    {
        AABB domain{{0, 0, 0}, {-1, -1, -1}, {1, 1, 1}};
        TinyBlock<Default, T_DefaultCount, 0.45f, 0.01f> fluid;
        TinyBlock<Boundary, T_BoundaryCount, 0.525f, 0.0f> wall;
        TinyBlock<Tracer, T_TracerCount, 0.35f, 0.02f> tracer;

        auto blocks() const
        {
            return std::tie(fluid, wall, tracer);
        }
    };

    using TinySetup = TinySetupFor<2u, 1u, 1u>;
    using LargerTinySetup = TinySetupFor<spearhed::numFrameSlots + 1u, 1u, 1u>;
    using EmptySetup = TinySetupFor<0u, 0u, 0u>;

    struct DefaultOnlySetup
    {
        AABB domain{{0, 0, 0}, {-1, -1, -1}, {1, 1, 1}};
        TinyBlock<Default, 0u, 0.45f, 0.01f> fluid;

        auto blocks() const
        {
            return std::tie(fluid);
        }
    };

    using Fixture = spearhed::test::SpearhedParticleFixture<spearhed::simDim>;
    using DefaultBuffer = ParticleRegionBuffer<spearhed::PRTypeFor<Default>>;
    using BoundaryBuffer = ParticleRegionBuffer<spearhed::PRTypeFor<Boundary>>;
    using TracerBuffer = ParticleRegionBuffer<spearhed::PRTypeFor<Tracer>>;

    struct ParticleState
    {
        std::array<spearhed::Real, spearhed::simDim> position{};
        std::array<spearhed::Real, spearhed::simDim> velocity{};
        spearhed::Real density = 0;
        spearhed::Real energy = 0;
    };

    template<typename T_Buffer>
    std::vector<ParticleState> snapshot(T_Buffer& buffer, int64_t heapOffset)
    {
        using namespace pmacc::spearhed::tags;
        using namespace spearhed::tags;
        auto const regions = buffer.buffer->getHostBuffer().getDataBox();
        std::vector<ParticleState> result;
        for(int region = 0; region < buffer.size; ++region)
        {
            auto& frameList = regions[region].particleFrameList;
            for(auto& frame : frameList.hostIterable(heapOffset))
            {
                for(uint32_t slot = 0; slot < spearhed::numFrameSlots; ++slot)
                {
                    auto particle = frame[slot];
                    if(!particle[multiMask])
                        continue;
                    ParticleState state;
                    size_t axis = 0u;
                    pmacc::spearhed::for_each_tag<spearhed::CS>(
                        [&](auto tag)
                        {
                            state.position[axis] = particle[relativePos][tag];
                            state.velocity[axis] = particle[vel][tag];
                            ++axis;
                        });
                    if constexpr(
                        std::same_as<typename T_Buffer::Species, Default>
                        || std::same_as<typename T_Buffer::Species, Boundary>)
                    {
                        state.density = particle[density];
                        state.energy = particle[internalEnergy];
                    }
                    result.push_back(state);
                }
            }
        }
        return result;
    }

    auto captureState()
    {
        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto defaultBuffer = dc.get<DefaultBuffer>(prBufId<Default>());
        auto boundaryBuffer = dc.get<BoundaryBuffer>(prBufId<Boundary>());
        auto tracerBuffer = dc.get<TracerBuffer>(prBufId<Tracer>());
        defaultBuffer->synchronize();
        boundaryBuffer->synchronize();
        tracerBuffer->synchronize();
        auto const heapOffset = spearhed::syncHeapToHost();
        return std::tuple{
            snapshot(*defaultBuffer, heapOffset),
            snapshot(*boundaryBuffer, heapOffset),
            snapshot(*tracerBuffer, heapOffset)};
    }

    void initializeTiny(Fixture& fixture)
    {
        auto setup = TinySetup{};
        spearhed::InitRegions{}(*fixture.deviceHeap, setup);
        spearhed::InitParticles{}(setup);
    }

    void runOldSequentialStep()
    {
        auto& dc = pmacc::Environment<>::get().DataConnector();
        auto defaultBuffer = dc.get<DefaultBuffer>(prBufId<Default>());
        auto boundaryBuffer = dc.get<BoundaryBuffer>(prBufId<Boundary>());

        spearhed::ParticlePush{}(0u);
        spearhed::test::runDevice(UpdateVolumes<spearhed::PRTypeFor<Default>>{}());
        using SmoothingKernel = typename spearhed::Setup::SmoothingKernel;
        auto const interactionRadius
            = static_cast<spearhed::CS::T_Axis>(SmoothingKernel::supportRadius) * spearhed::h0;
        auto bundle = calculateNeighbours(*defaultBuffer, interactionRadius, *defaultBuffer, *boundaryBuffer);
        FrameIndexBuffer<spearhed::PRTypeFor<Default>> index{*defaultBuffer};
        spearhed::test::runDevice(
            spearhed::UpdateDensity<typename spearhed::Setup::SmoothingKernel>{}(
                bundle.template selectByRole<roles::Source>(),
                *defaultBuffer,
                index,
                spearhed::h0));
        spearhed::test::runDevice(
            spearhed::UpdateHydroForces<typename spearhed::Setup::SmoothingKernel>{spearhed::gamma_eos}(
                bundle.template selectByRole<roles::Source>(),
                *defaultBuffer,
                index,
                spearhed::h0));
        forEachSpeciesBufWithPred(
            spearhed::allSpecies,
            pred::withAllRoles<roles::Movable, roles::Thermodynamic>,
            [&](auto& buffer) { launchForEach(levels::particle, buffer, spearhed::EulerIntegrate{}, spearhed::dt); });
    }

    void requireClose(ParticleState const& actual, ParticleState const& expected)
    {
        for(unsigned axis = 0; axis < spearhed::simDim; ++axis)
        {
            CHECK(actual.position[axis] == Catch::Approx(expected.position[axis]).margin(2.0e-5));
            CHECK(actual.velocity[axis] == Catch::Approx(expected.velocity[axis]).margin(2.0e-5));
        }
        CHECK(actual.density == Catch::Approx(expected.density).margin(2.0e-5));
        CHECK(actual.energy == Catch::Approx(expected.energy).margin(2.0e-5));
    }
} // namespace

TEST_CASE_METHOD(
    Fixture,
    "Simulation timestep composes cached indices and matches sequential execution",
    "[integration][simulation]")
{
    constexpr uint32_t steps = 3u;
    spearhed::Simulation simulation{spearhed::test::mpiContext()};
    initializeTiny(*this);

    auto initial = captureState();
    simulation.runOneStep(0u);
    auto& indices = spearhed::SimulationTestAccess::frameIndices(simulation);
    auto& defaultIndex = std::get<FrameIndexBuffer<spearhed::PRTypeFor<Default>>>(indices);
    auto const firstBuildVersion = defaultIndex.builtVersion;
    REQUIRE(defaultIndex.hasBuild);
    REQUIRE(defaultIndex.framesPerRegion.has_value());

    // A host-side marker survives an unchanged-topology step only if the cached index fast path ran.
    auto scanHost = defaultIndex.framesPerRegion->getHostBuffer().getDataBox();
    scanHost[0] = 0x5a5a'5a5au;
    simulation.runOneStep(1u);
    REQUIRE(scanHost[0] == 0x5a5a'5a5au);
    REQUIRE(defaultIndex.builtVersion == firstBuildVersion);

    auto& defaultBuffer = *pmacc::Environment<>::get().DataConnector().get<DefaultBuffer>(prBufId<Default>());
    ++defaultBuffer.topologyVersion;
    simulation.runOneStep(2u);
    REQUIRE(defaultIndex.builtVersion == defaultBuffer.topologyVersion);
    REQUIRE(defaultIndex.builtVersion != firstBuildVersion);

    auto const production = captureState();
    REQUIRE(std::get<2>(production)[0].position[0] != std::get<2>(initial)[0].position[0]);
    REQUIRE(std::get<1>(production)[0].position == std::get<1>(initial)[0].position);

    simulation.resetAll(0u);
    initializeTiny(*this);
    for(uint32_t step = 0; step < steps; ++step)
        runOldSequentialStep();
    auto const sequential = captureState();

    REQUIRE(std::get<0>(production).size() == std::get<0>(sequential).size());
    REQUIRE(std::get<1>(production).size() == std::get<1>(sequential).size());
    REQUIRE(std::get<2>(production).size() == std::get<2>(sequential).size());
    for(size_t i = 0; i < std::get<0>(production).size(); ++i)
        requireClose(std::get<0>(production)[i], std::get<0>(sequential)[i]);
    for(size_t i = 0; i < std::get<1>(production).size(); ++i)
        requireClose(std::get<1>(production)[i], std::get<1>(sequential)[i]);
    for(size_t i = 0; i < std::get<2>(production).size(); ++i)
        requireClose(std::get<2>(production)[i], std::get<2>(sequential)[i]);

    // Explicit cache reset requires the next timestep to rebuild before dispatch.
    simulation.runOneStep(3u);
    REQUIRE(defaultIndex.hasBuild);
    REQUIRE(defaultIndex.totalFrames == 1u);
    simulation.resetAll(0u);
    simulation.runOneStep(4u);
    REQUIRE(defaultIndex.hasBuild);
    REQUIRE(defaultIndex.totalFrames == 1u);
}

TEST_CASE_METHOD(
    Fixture,
    "Simulation timestep refreshes a built cache after frame topology replacement",
    "[integration][simulation]")
{
    spearhed::Simulation simulation{spearhed::test::mpiContext()};
    initializeTiny(*this);
    simulation.runOneStep(0u);

    auto& indices = spearhed::SimulationTestAccess::frameIndices(simulation);
    auto& defaultIndex = std::get<FrameIndexBuffer<spearhed::PRTypeFor<Default>>>(indices);
    auto& defaultBuffer = *pmacc::Environment<>::get().DataConnector().get<DefaultBuffer>(prBufId<Default>());
    REQUIRE(defaultIndex.hasBuild);
    REQUIRE(defaultIndex.totalFrames == 1u);
    auto const builtVersion = defaultIndex.builtVersion;

    auto largerSetup = LargerTinySetup{};
    spearhed::InitRegions{}(*deviceHeap, largerSetup);
    spearhed::InitParticles{}(largerSetup);
    REQUIRE(defaultBuffer.topologyVersion != builtVersion);
    REQUIRE(defaultIndex.builtVersion == builtVersion);

    simulation.runOneStep(1u);
    REQUIRE(defaultIndex.builtVersion == defaultBuffer.topologyVersion);
    REQUIRE(defaultIndex.totalFrames == 2u);

    auto const state = captureState();
    REQUIRE(std::get<0>(state).size() == spearhed::numFrameSlots + 1u);
    auto const expectedPosition = 0.45f + 0.01f * spearhed::dt;
    for(auto const& particle : std::get<0>(state))
        CHECK(particle.position[0] == Catch::Approx(expectedPosition).margin(2.0e-5));
}

TEST_CASE_METHOD(Fixture, "Simulation timestep accepts an empty default-only setup", "[integration][simulation]")
{
    auto setup = DefaultOnlySetup{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    auto& dc = pmacc::Environment<>::get().DataConnector();
    REQUIRE_FALSE(dc.hasId(prBufId<Boundary>()));
    REQUIRE_FALSE(dc.hasId(prBufId<Tracer>()));

    spearhed::Simulation simulation{spearhed::test::mpiContext()};
    simulation.runOneStep(0u);
    auto& indices = spearhed::SimulationTestAccess::frameIndices(simulation);
    auto& defaultIndex = std::get<FrameIndexBuffer<spearhed::PRTypeFor<Default>>>(indices);
    REQUIRE(defaultIndex.hasBuild);
    REQUIRE(defaultIndex.totalFrames == 0u);
}

TEST_CASE_METHOD(Fixture, "Simulation timestep accepts an empty all-species setup", "[integration][simulation]")
{
    auto setup = EmptySetup{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    spearhed::Simulation simulation{spearhed::test::mpiContext()};
    simulation.runOneStep(0u);
    auto& indices = spearhed::SimulationTestAccess::frameIndices(simulation);
    auto& defaultIndex = std::get<FrameIndexBuffer<spearhed::PRTypeFor<Default>>>(indices);
    auto& tracerIndex = std::get<FrameIndexBuffer<spearhed::PRTypeFor<Tracer>>>(indices);
    REQUIRE(defaultIndex.hasBuild);
    REQUIRE(tracerIndex.hasBuild);
    REQUIRE(defaultIndex.totalFrames == 0u);
    REQUIRE(tracerIndex.totalFrames == 0u);
}
