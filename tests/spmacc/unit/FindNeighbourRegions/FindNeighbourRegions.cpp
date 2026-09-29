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
#include "spearhed/param.hpp"
#include "spearhed/param/mallocMC.param"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/regions/NeighbourRegions.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <tuple>
#include <utility>

#include <caravan/alpaka.hpp>
#include <catch2/catch_test_macros.hpp>

static constexpr unsigned TEST_DIM = spearhed::simDim;
using ParticleFixture = spearhed::test::SpearhedParticleFixture<TEST_DIM>;

TEST_CASE_METHOD(ParticleFixture, "CalculateNeighbourRegions Validation", "[integration][particles][neighbours]")
{
    auto setup = spearhed::EmptyNRegions<3>{};
    spearhed::InitRegions{}(*deviceHeap, setup);

    // Initialize region volumes manually
    spearhed::test::runDevice(prBuf->buffer->deviceToHost());
    auto hostRegions = prBuf->buffer->getHostBuffer().getDataBox();

    pmacc::spearhed::for_each_tag<spearhed::CS>(
        [&](auto tag)
        {
            // Region 0: [0.0, 1.0]
            hostRegions(0).volume.min[tag] = 0.0f;
            hostRegions(0).volume.max[tag] = 1.0f;

            // Region 1: [1.5, 2.5]
            hostRegions(1).volume.min[tag] = 1.5f;
            hostRegions(1).volume.max[tag] = 2.5f;

            // Region 2: [4.0, 5.0]
            hostRegions(2).volume.min[tag] = 4.0f;
            hostRegions(2).volume.max[tag] = 5.0f;
        });


    spearhed::test::runDevice(prBuf->buffer->hostToDevice());

    // A separate source with distinct bounds ensures each bundle entry is computed independently.
    pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType> secondSource;
    secondSource.create(3);
    for(int region = 0; region < 3; ++region)
        secondSource.pushBack(hostRegions(region));
    constexpr float sourceMins[] = {0.2f, 2.8f, 8.0f};
    constexpr float sourceMaxs[] = {0.5f, 3.0f, 9.0f};
    auto secondSourceHost = secondSource.buffer->getHostBuffer().getDataBox();
    pmacc::spearhed::for_each_tag<spearhed::CS>(
        [&](auto tag)
        {
            for(int region = 0; region < 3; ++region)
            {
                secondSourceHost(region).volume.min[tag] = sourceMins[region];
                secondSourceHost(region).volume.max[tag] = sourceMaxs[region];
            }
        });
    spearhed::test::runDevice(secondSource.buffer->hostToDevice());

    // Expected behavior with smoothingLength = 0.6f:
    // Region 0 expands to [-0.6, 1.6] -> Intersects Region 0 and 1
    // Region 1 expands to [0.9, 3.1] -> Intersects Region 0 and 1
    // Region 2 expands to [3.4, 5.6] -> Intersects Region 2 only
    constexpr float smoothingLength = 0.6f;

    using Bundle = decltype(pmacc::spearhed::calculateNeighbours(*prBuf, smoothingLength, *prBuf, secondSource));
    auto& device = pmacc::Environment<>::get().DeviceContext();
    auto preparation = pmacc::spearhed::calculateNeighboursSender(*prBuf, smoothingLength, *prBuf, secondSource);
    auto withCopies
        = std::move(preparation)
          | caravan::letValue(
              [](auto& bundle)
              {
                  auto copies = std::apply(
                      [](auto&... entries)
                      {
                          return caravan::whenAll(
                              entries.neighbourRegions.deviceToHost()...,
                              entries.regionOffsets.deviceToHost()...);
                      },
                      bundle.entries);
                  return std::move(copies) | caravan::then([bundlePtr = &bundle] { return std::move(*bundlePtr); });
              });
    auto bundle = caravan::syncWait<Bundle>(caravan::alpaka::withDevice(device, std::move(withCopies)));
    auto& entry = std::get<0>(bundle.entries);
    auto& secondEntry = std::get<1>(bundle.entries);

    // Validation
    auto const& h_neighbours = entry.neighbourRegions.getHostBuffer().getDataBox();
    auto const& h_offsets = entry.regionOffsets.getHostBuffer().getDataBox();

    // Validate inclusive prefix sum offsets
    REQUIRE(h_offsets(0) == 0);
    REQUIRE(h_offsets(1) == 2); // Region 0 matches: 0, 1
    REQUIRE(h_offsets(2) == 4); // Region 1 matches: 0, 1
    REQUIRE(h_offsets(3) == 5); // Region 2 matches: 2

    // Validate neighbour IDs
    REQUIRE(h_neighbours(0) == 0);
    REQUIRE(h_neighbours(1) == 1);

    REQUIRE(h_neighbours(2) == 0);
    REQUIRE(h_neighbours(3) == 1);

    REQUIRE(h_neighbours(4) == 2);

    auto const& secondNeighbours = secondEntry.neighbourRegions.getHostBuffer().getDataBox();
    auto const& secondOffsets = secondEntry.regionOffsets.getHostBuffer().getDataBox();
    REQUIRE(secondOffsets(0) == 0u);
    REQUIRE(secondOffsets(1) == 1u);
    REQUIRE(secondOffsets(2) == 2u);
    REQUIRE(secondOffsets(3) == 2u);
    REQUIRE(secondNeighbours(0) == 0u);
    REQUIRE(secondNeighbours(1) == 1u);

    // The sender retains references, not a snapshot of region data: mutate and upload after
    // constructing it, but before starting it. The delayed count/write kernels must see this layout.
    auto delayedPreparation
        = pmacc::spearhed::calculateNeighboursSender(*prBuf, smoothingLength, *prBuf, secondSource);
    pmacc::spearhed::for_each_tag<spearhed::CS>(
        [&](auto tag)
        {
            for(int region = 0; region < 3; ++region)
            {
                hostRegions(region).volume.min[tag] = static_cast<float>(region * 10);
                hostRegions(region).volume.max[tag] = static_cast<float>(region * 10 + 1);
            }
        });
    spearhed::test::runDevice(prBuf->buffer->hostToDevice());

    auto delayedWithCopies
        = std::move(delayedPreparation)
          | caravan::letValue(
              [](auto& delayedBundle)
              {
                  auto copies = std::apply(
                      [](auto&... entries)
                      {
                          return caravan::whenAll(
                              entries.neighbourRegions.deviceToHost()...,
                              entries.regionOffsets.deviceToHost()...);
                      },
                      delayedBundle.entries);
                  return std::move(copies)
                         | caravan::then([bundlePtr = &delayedBundle] { return std::move(*bundlePtr); });
              });
    auto delayedBundle = caravan::syncWait<Bundle>(caravan::alpaka::withDevice(device, std::move(delayedWithCopies)));
    auto const& delayedEntry = std::get<0>(delayedBundle.entries);
    auto const& delayedNeighbours = delayedEntry.neighbourRegions.getHostBuffer().getDataBox();
    auto const& delayedOffsets = delayedEntry.regionOffsets.getHostBuffer().getDataBox();
    REQUIRE(delayedOffsets(0) == 0u);
    REQUIRE(delayedOffsets(1) == 1u);
    REQUIRE(delayedOffsets(2) == 2u);
    REQUIRE(delayedOffsets(3) == 3u);
    REQUIRE(delayedNeighbours(0) == 0u);
    REQUIRE(delayedNeighbours(1) == 1u);
    REQUIRE(delayedNeighbours(2) == 2u);
}

TEST_CASE_METHOD(
    ParticleFixture,
    "CalculateNeighbourRegions handles empty targets and sources",
    "[integration][particles][neighbours]")
{
    pmacc::spearhed::ParticleRegionBuffer<spearhed::PRType> emptySource;
    constexpr float smoothingLength = 0.6f;

    auto emptyTargetBundle = pmacc::spearhed::calculateNeighbours(*prBuf, smoothingLength, emptySource);
    auto& emptyTargetEntry = std::get<0>(emptyTargetBundle.entries);
    spearhed::test::runDevice(emptyTargetEntry.regionOffsets.deviceToHost());
    REQUIRE(emptyTargetEntry.regionOffsets.getHostBuffer().data()[0] == 0u);
    REQUIRE(emptyTargetEntry.neighbourRegions.getHostBuffer().size() == 0u);

    auto setup = spearhed::EmptyNRegions<3>{};
    spearhed::InitRegions{}(*deviceHeap, setup);
    auto populatedTargetBundle = pmacc::spearhed::calculateNeighbours(*prBuf, smoothingLength, emptySource);
    auto& populatedTargetEntry = std::get<0>(populatedTargetBundle.entries);
    spearhed::test::runDevice(populatedTargetEntry.regionOffsets.deviceToHost());
    auto const offsets = populatedTargetEntry.regionOffsets.getHostBuffer().getDataBox();
    REQUIRE(offsets(0) == 0u);
    REQUIRE(offsets(1) == 0u);
    REQUIRE(offsets(2) == 0u);
    REQUIRE(offsets(3) == 0u);
    REQUIRE(populatedTargetEntry.neighbourRegions.getHostBuffer().size() == 0u);
}
