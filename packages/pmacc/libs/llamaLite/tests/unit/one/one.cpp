// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.mozilla.org/MPL/2.0/.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(posO);
DEFINE_TAG(velO);
DEFINE_TAG(massO);
DEFINE_TAG(xO);
DEFINE_TAG(yO);
DEFINE_TAG(zO);

using Pos3O = ll::Record<ll::Field<xO_t, float>, ll::Field<yO_t, float>, ll::Field<zO_t, float>>;
using ParticleOne = ll::Record<ll::Field<posO_t, Pos3O>, ll::Field<velO_t, Pos3O>, ll::Field<massO_t, double>>;

namespace
{
    template<typename Src>
    concept CanConstructParticle = requires(Src src) { ll::One<ParticleOne>{src}; };

    template<typename Src>
    concept CanAssignParticle = requires(ll::One<ParticleOne>& dest, Src src) { dest = src; };
} // namespace

TEST_CASE("One container provides correct SoA-style access and mutation", "[One][View]")
{
    ll::One<ParticleOne> one{};

    SECTION("Direct leaf spans guarantee single-element storage and const-correctness")
    {
        auto xSpan = one.getLeaf(posO / xO);
        auto mSpan = one.getLeaf(massO);

        REQUIRE(xSpan.size() == 1u);
        REQUIRE(mSpan.size() == 1u);

        xSpan[0] = 3.14f;
        mSpan[0] = 2.71828;

        auto const& cone = one;
        CHECK(cone.getLeaf(posO / xO)[0] == Catch::Approx(3.14f));
        CHECK(cone.getLeaf(massO)[0] == Catch::Approx(2.71828));
    }

    SECTION("Structural and indexed access mutate the owning storage")
    {
        auto pos_view = one[posO];
        pos_view[xO][0] = 1.0f;
        pos_view[yO][0] = 2.0f;
        pos_view[zO][0] = 3.0f;
        one[0u][massO] = 42.0;

        CHECK(one[posO][xO][0] == Catch::Approx(1.0f));
        CHECK(one[posO][yO][0] == Catch::Approx(2.0f));
        CHECK(one[posO][zO][0] == Catch::Approx(3.0f));

        auto const& cone = one;
        CHECK(cone[massO][0] == Catch::Approx(42.0));
        auto mass_span = one[massO];
        REQUIRE(mass_span.size() == 1u);
    }

    SECTION("Multi-tag views project access into the owning One")
    {
        auto pv = one.view(posO, velO);
        pv[0u][posO][xO] = 5.5f;
        CHECK(one[posO][xO][0] == Catch::Approx(5.5f));
    }

    SECTION("One assignment from a root view copies all fields")
    {
        ll::One<ParticleOne> src{};
        src[0u][posO][xO] = 1.0f;
        src[0u][posO][yO] = 2.0f;
        src[0u][posO][zO] = 3.0f;
        src[0u][massO] = 42.0;

        one = src[0u];

        CHECK(one[0u][posO][xO] == Catch::Approx(1.0f));
        CHECK(one[0u][posO][yO] == Catch::Approx(2.0f));
        CHECK(one[0u][posO][zO] == Catch::Approx(3.0f));
        CHECK(one[0u][massO] == Catch::Approx(42.0));

        one[0u][massO] = 51;
        CHECK(src[0u][massO] == Catch::Approx(42.0));
        CHECK(one[0u][massO] == Catch::Approx(51.0));
    }

    SECTION("One construction from a root view copies all fields")
    {
        ll::One<ParticleOne> src{};
        src[0u][posO][xO] = 1.0f;
        src[0u][massO] = 99.0;

        ll::One<ParticleOne> dest(src[0u]);

        CHECK(dest[0u][posO][xO] == Catch::Approx(1.0f));
        CHECK(dest[0u][massO] == Catch::Approx(99.0));
    }

    SECTION("One rejects incomplete source selections")
    {
        using MassView = decltype(one.view(massO)[0u]);
        STATIC_CHECK_FALSE(CanConstructParticle<MassView>);
        STATIC_CHECK_FALSE(CanAssignParticle<MassView>);
    }
}
