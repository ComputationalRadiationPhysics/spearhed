/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "spmacc/particles/regions/mapping/DecompositionGroup.hpp"

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param/setup.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/regions/mapping/AdaptiveSplit/Policies.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"

#include <tuple>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

namespace
{
    namespace species = pmacc::spearhed::species;

    using SeparateMaterialDecompositionGroups = std::tuple<
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Boundary>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Tracer>>;

    using SeparateStaticBoundaryDecompositionGroups = std::tuple<
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default>,
        pmacc::spearhed::ExplicitInvalidationDecompositionGroup<species::Boundary>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Tracer>>;

    struct TestGridFactory
    {
        auto operator()(auto const& setup) const
        {
            using Grid = pmacc::spearhed::FixedCartesianGrid<spearhed::CS>;
            using Bounds = typename Grid::Bounds;
            return Grid{
                Bounds{setup.domain.origin + setup.domain.min, setup.domain.origin + setup.domain.max},
                {2u, 1u, 1u}};
        }
    };

    using FixedAndMaterialDecompositionGroups = std::tuple<
        pmacc::spearhed::FixedCartesianDecompositionGroup<TestGridFactory, species::Default>,
        pmacc::spearhed::ExplicitInvalidationMaterialGroup<species::Boundary>,
        pmacc::spearhed::MaterialAabbDecompositionGroup<species::Tracer>>;

    struct FixedGridSetup : spearhed::DefaultSetup
    {
        using DecompositionGroups = FixedAndMaterialDecompositionGroups;
    };

    struct AdaptiveTriggerFactory
    {
        auto operator()(auto const&) const
        {
            return pmacc::spearhed::ParticleCountExceeds{1000u};
        }
    };

    struct AdaptivePartitionFactory
    {
        auto operator()(auto const&) const
        {
            return pmacc::spearhed::WorldAxisMidpoint<spearhed::CS, pmacc::spearhed::tags::x_t>{};
        }
    };

    using AdaptiveAndMaterialDecompositionGroups = std::tuple<
        pmacc::spearhed::
            AdaptiveSplitDecompositionGroup<AdaptiveTriggerFactory, AdaptivePartitionFactory, species::Default>,
        pmacc::spearhed::ExplicitInvalidationMaterialGroup<species::Boundary>,
        pmacc::spearhed::MaterialAabbDecompositionGroup<species::Tracer>>;

    struct AdaptiveSetup : spearhed::DefaultSetup
    {
        using DecompositionGroups = AdaptiveAndMaterialDecompositionGroups;
    };

    using DuplicateDefaultDecompositionGroups = std::tuple<
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default, species::Boundary>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default, species::Tracer>>;
} // namespace

TEST_CASE("Decomposition-group declarations assign each registered species exactly once", "[spatial][decomposition]")
{
    STATIC_REQUIRE(
        pmacc::spearhed::DecompositionGroupAssignmentFor<SeparateMaterialDecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<SeparateStaticBoundaryDecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::DecompositionGroupAssignmentFor<FixedAndMaterialDecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<AdaptiveAndMaterialDecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE_FALSE(
        pmacc::spearhed::DecompositionGroupAssignmentFor<DuplicateDefaultDecompositionGroups, spearhed::AllSpecies>);
}

TEST_CASE_METHOD(
    spearhed::test::SpearhedParticleFixture<spearhed::simDim>,
    "fixed-grid groups construct through the generic lifecycle",
    "[spatial][decomposition]")
{
    FixedGridSetup setup;
    spearhed::InitRegions{}(*deviceHeap, setup);

    using Allocator = decltype(deviceHeap->getAllocatorHandle());
    using Groups = pmacc::spearhed::
        DecompositionGroupSet<spearhed::AllSpecies, FixedAndMaterialDecompositionGroups, FixedGridSetup, Allocator>;
    Groups groups{setup, deviceHeap->getAllocatorHandle()};
    auto& store = groups.storeFor(species::Default{});
    REQUIRE(store.size == 2);

    groups.prepareAfterMotion();
    auto prepared = groups.preparedFor(store);
    REQUIRE(prepared.bucketCount() == 2u);
    REQUIRE(prepared.generation() == 2u);
}

TEST_CASE_METHOD(
    spearhed::test::SpearhedParticleFixture<spearhed::simDim>,
    "adaptive groups construct through the generic lifecycle",
    "[spatial][decomposition]")
{
    AdaptiveSetup setup;
    spearhed::InitRegions{}(*deviceHeap, setup);

    using Allocator = decltype(deviceHeap->getAllocatorHandle());
    using Groups = pmacc::spearhed::
        DecompositionGroupSet<spearhed::AllSpecies, AdaptiveAndMaterialDecompositionGroups, AdaptiveSetup, Allocator>;
    Groups groups{setup, deviceHeap->getAllocatorHandle()};
    auto& store = groups.storeFor(species::Default{});
    auto const topologyBefore = store.topologyVersion;

    groups.prepareAfterMotion();
    auto prepared = groups.preparedFor(store);
    REQUIRE(prepared.generation() == 1u);
    REQUIRE(store.topologyVersion == topologyBefore);
}
