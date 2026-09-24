/* Copyright 2026 Tapish Narwal
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

#include "spearhed/param.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/algorithms/InteractParticles.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/eventSystem/waitForAllTasks.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;
    using ParticleFixture = spearhed::test::SpearhedParticleFixture<spearhed::simDim>;

    struct OneParticlePerRegionSetup
    {
        using Species = pmacc::spearhed::species::Default;

        uint32_t regionCount;
        pmacc::spearhed::AABB<spearhed::CS> domain{
            typename pmacc::spearhed::AABB<spearhed::CS>::Pnt{0.0f},
            typename pmacc::spearhed::AABB<spearhed::CS>::Vec{-1.0f},
            typename pmacc::spearhed::AABB<spearhed::CS>::Vec{1.0f}};

        auto blocks() const
        {
            return std::tie(*this);
        }

        struct NumParticlesToCreate
        {
            DINLINE constexpr uint32_t operator()(auto&, auto&, uint32_t) const
            {
                return 1u;
            }
        };

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(0u);
        }

        struct PlaceParticle
        {
            DINLINE constexpr void operator()(auto const&, auto& particle, auto const&, uint32_t) const
            {
                pmacc::spearhed::for_each_tag<spearhed::CS>(
                    [&](auto tag) { particle[pmacc::spearhed::tags::relativePos][tag] = spearhed::Real{0}; });
            }
        };

        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        template<typename>
        void addRegions(std::vector<pmacc::spearhed::AABB<spearhed::CS>>& regions) const
        {
            regions.assign(regionCount, domain);
        }
    };

    struct CountAcceptedPairs
    {
        HDINLINE void operator()(auto& worker, auto const&, auto const&, auto const& ctx, auto&, auto count) const
        {
            if(!ctx.isSelf)
                alpaka::atomicAdd(worker.getAcc(), &count(0), uint64_t{1}, ::alpaka::hierarchy::Blocks{});
        }
    };

    template<typename Fn>
    double averageMilliseconds(uint32_t repetitions, Fn&& fn)
    {
        auto const start = Clock::now();
        for(uint32_t repetition = 0; repetition < repetitions; ++repetition)
            fn();
        auto const elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return elapsed / static_cast<double>(repetitions);
    }

    struct Repetitions
    {
        uint32_t bounds;
        uint32_t plan;
        uint32_t interaction;
    };

    Repetitions repetitionsFor(uint32_t regionCount)
    {
        if(regionCount <= 16u)
            return {1000u, 200u, 100u};
        if(regionCount <= 64u)
            return {1000u, 100u, 30u};
        return {500u, 30u, 10u};
    }

    void runCase(ParticleFixture& fixture, uint32_t regionCount)
    {
        OneParticlePerRegionSetup setup{regionCount};
        spearhed::InitRegions{}(*fixture.deviceHeap, setup);
        spearhed::InitParticles{}(setup);
        auto& particleStore = *fixture.prBuf;

        auto waitForDevice = [] { pmacc::eventSystem::waitForAllTasks(); };
        Repetitions const repetitions = repetitionsFor(regionCount);
        constexpr spearhed::Real interactionRadius{1.0f};

        pmacc::spearhed::UpdateVolumes<spearhed::PRType>{}();
        waitForDevice();
        double const boundsMs = averageMilliseconds(
            repetitions.bounds,
            [&]
            {
                pmacc::spearhed::UpdateVolumes<spearhed::PRType>{}();
                waitForDevice();
            });

        {
            auto warmup = pmacc::spearhed::calculateNeighbours(particleStore, interactionRadius, particleStore);
            waitForDevice();
        }
        double const planMs = averageMilliseconds(
            repetitions.plan,
            [&]
            {
                auto plan = pmacc::spearhed::calculateNeighbours(particleStore, interactionRadius, particleStore);
                waitForDevice();
            });

        auto plan = pmacc::spearhed::calculateNeighbours(particleStore, interactionRadius, particleStore);
        waitForDevice();
        auto& entry = plan.bySpecies(pmacc::spearhed::species::default_);
        auto& provider = entry.candidateProvider;
        provider.regionOffsets.deviceToHost();
        auto const offsets = provider.regionOffsets.getHostBuffer().getDataBox();
        uint64_t const candidatePairs = offsets[regionCount];
        uint64_t const csrDeviceBytes = (candidatePairs + static_cast<uint64_t>(regionCount) + 1u) * sizeof(uint32_t);

        pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> frameIndex{particleStore};
        pmacc::HostDeviceBuffer<uint64_t, 1> acceptedPairCount(1u);
        acceptedPairCount.getHostBuffer().setValue(0u);
        acceptedPairCount.hostToDevice();
        auto countBox = acceptedPairCount.getDeviceBuffer().getDataBox();
        auto sources = plan.selectByRole(pmacc::spearhed::roles::source);

        pmacc::spearhed::interact(
            sources,
            particleStore,
            frameIndex,
            interactionRadius,
            CountAcceptedPairs{},
            countBox)
            .waitForFinished();
        double const interactionMs = averageMilliseconds(
            repetitions.interaction,
            [&]
            {
                pmacc::spearhed::interact(
                    sources,
                    particleStore,
                    frameIndex,
                    interactionRadius,
                    CountAcceptedPairs{},
                    countBox)
                    .waitForFinished();
            });

        acceptedPairCount.deviceToHost();
        uint64_t const measuredPairs = acceptedPairCount.getHostBuffer().data()[0];
        uint64_t const expectedPairsPerPass
            = static_cast<uint64_t>(regionCount) * static_cast<uint64_t>(regionCount - 1u);
        uint64_t const expectedMeasuredPairs
            = expectedPairsPerPass * static_cast<uint64_t>(repetitions.interaction + 1u);
        if(measuredPairs != expectedMeasuredPairs)
        {
            std::cerr << "interaction verification failed for " << regionCount << " regions: got " << measuredPairs
                      << ", expected " << expectedMeasuredPairs << '\n';
            std::exit(1);
        }

        std::cout << regionCount << ',' << repetitions.bounds << ',' << repetitions.plan << ','
                  << repetitions.interaction << ',' << std::fixed << std::setprecision(6) << boundsMs << ',' << planMs
                  << ',' << candidatePairs << ',' << interactionMs << ',' << csrDeviceBytes << '\n';
    }
} // namespace

int main()
{
    ParticleFixture fixture;

    std::cout << "regions,bounds_repetitions,plan_repetitions,interaction_repetitions,bounds_ms,plan_ms,"
                 "candidate_pairs,interaction_ms,csr_device_bytes\n";
    for(uint32_t const regionCount : std::array<uint32_t, 3>{16u, 64u, 256u})
        runCase(fixture, regionCount);
}
