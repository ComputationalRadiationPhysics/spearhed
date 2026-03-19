// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "spmacc/topology/Point.hpp"
#include "spmacc/topology/PointStorage.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(posi);
DEFINE_TAG(nestedPos1);
DEFINE_TAG(nestedPos2);
DEFINE_TAG(vel);
DEFINE_TAG(mass);

using Posi = ll::Record<ll::Field<pmacc::spearhed::tags::x_t, float>, ll::Field<pmacc::spearhed::tags::y_t, float>>;

using TwoLevelNest = ll::Record<ll::Field<nestedPos2_t, Posi>>;

using Particle = ll::Record<
    ll::Field<posi_t, Posi>,
    ll::Field<vel_t, float>,
    ll::Field<mass_t, double>,
    ll::Field<nestedPos1_t, TwoLevelNest>>;

using TSoA = ll::SoA<Particle, 512>;

template<auto... TagInstances>
using ParticleView = ll::SoAIndexedView<TSoA, ll::to_path_t<std::remove_cvref_t<decltype(TagInstances)>>...>;

template<>
struct ll::traits::AsType<ll::Field<posi_t, Posi>>
{
    using CS = pmacc::spearhed::Cartesian<float, 2>;
    using type = pmacc::spearhed::Point<CS, pmacc::spearhed::PointViewStorage<CS, ParticleView<posi>>>;

    constexpr type operator()(auto fieldView) const
    {
        return type{fieldView};
    }
};

TEST_CASE("LlamaLite SoA Integration with Spearhed Types", "[spearhed][llamalite]")
{
    using namespace pmacc::spearhed;
    TSoA particles_soa;

    SECTION("Scalar Field Access (Mass)")
    {
        // Write access
        *(particles_soa[mass][0]) = 10.5;

        // Read verification
        CHECK(*(particles_soa[mass][0]) == Catch::Approx(10.5));
    }

    SECTION("Structured View Access (Position)")
    {
        auto pos_view = particles_soa[posi];

        // Set value at index 2 via component view
        *pos_view[2][tags::x] = 10.5f;

        // Verify commutativity of indexing and member access
        CHECK(*pos_view[tags::x][2] == Catch::Approx(10.5f));
        CHECK(*pos_view[2][tags::x] == Catch::Approx(10.5f));

        // Create a view fixed to index 0
        auto posZero = pos_view[0];
        *posZero[tags::x] = 10.8f;

        // Verify index 2 is unaffected
        CHECK(*pos_view[2][tags::x] == Catch::Approx(10.5f));

        // Verify index 0 is updated
        CHECK(*posZero[tags::x] == Catch::Approx(10.8f));
        CHECK(*pos_view[tags::x][0] == Catch::Approx(10.8f));

        SECTION("Spearhed Point Abstraction")
        {
            // implicitly uses AsType specialization via .get()
            auto point = posZero.get();

            // Modify data via high-level Point interface
            point[tags::x] = 10.11f;

            // Verify reflected changes in Point interface
            CHECK(point[tags::x] == Catch::Approx(10.11f));
            // Verify reflected changes in underlying SoA
            CHECK(*pos_view[tags::x][0] == Catch::Approx(10.11f));
        }
    }

    SECTION("Nested Record Access")
    {
        auto nestedPos1_view = particles_soa[nestedPos1];
        auto nestedPos1_idxView = nestedPos1_view[2];
        auto nestedPos2_view = nestedPos1_idxView[nestedPos2];

        // Write to nested component
        *nestedPos2_view[tags::x] = 42.0f;

        CHECK(*nestedPos2_view[tags::x] == Catch::Approx(42.0f));
    }

    SECTION("Multi Tag View")
    {
        auto nestedPos1_pos_view = particles_soa.view(nestedPos1, posi);
        auto nestedPos1_pos_idxView = nestedPos1_pos_view[2];
        auto nestedPos1_idxView = nestedPos1_pos_idxView[nestedPos1];
        auto nestedPos2_view = nestedPos1_idxView[nestedPos2];
        *nestedPos2_view[tags::x] = 42.0f;

        CHECK(*nestedPos2_view[tags::x] == Catch::Approx(42.0f));
    }
}
