// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(posD);
DEFINE_TAG(velD);
DEFINE_TAG(massD);
DEFINE_TAG(xComp);
DEFINE_TAG(yComp);
DEFINE_TAG(zComp);

using Pos3D = ll::Record<ll::Field<xComp_t, float>, ll::Field<yComp_t, float>, ll::Field<zComp_t, float>>;

using ParticleRecord = ll::Record<ll::Field<posD_t, Pos3D>, ll::Field<velD_t, Pos3D>, ll::Field<massD_t, double>>;

TEST_CASE("DynSoA default construction", "[DynSoA]")
{
    ll::DynSoA<ParticleRecord> soa;
    CHECK(soa.size() == 0u);
}

TEST_CASE("DynSoA resize and size", "[DynSoA]")
{
    ll::DynSoA<ParticleRecord> soa;
    soa.resize(100u);
    CHECK(soa.size() == 100u);

    // Resize again
    soa.resize(200u);
    CHECK(soa.size() == 200u);
}

TEST_CASE("DynSoA leaf spans have correct size after resize", "[DynSoA]")
{
    constexpr size_t N = 50u;
    ll::DynSoA<ParticleRecord> soa(N);

    CHECK(soa.getLeaf<ll::TagPath<posD_t, xComp_t>>().size() == N);
    CHECK(soa.getLeaf<ll::TagPath<posD_t, yComp_t>>().size() == N);
    CHECK(soa.getLeaf<ll::TagPath<posD_t, zComp_t>>().size() == N);
    CHECK(soa.getLeaf<ll::TagPath<velD_t, xComp_t>>().size() == N);
    CHECK(soa.getLeaf<ll::TagPath<velD_t, yComp_t>>().size() == N);
    CHECK(soa.getLeaf<ll::TagPath<velD_t, zComp_t>>().size() == N);
    CHECK(soa.getLeaf<massD_t>().size() == N);
}

TEST_CASE("DynSoA read/write round-trip via getLeaf", "[DynSoA]")
{
    constexpr size_t N = 64u;
    ll::DynSoA<ParticleRecord> soa(N);

    // Write
    auto xSpan = soa.getLeaf<ll::TagPath<posD_t, xComp_t>>();
    auto mSpan = soa.getLeaf<massD_t>();
    for(size_t i = 0; i < N; ++i)
    {
        xSpan[i] = static_cast<float>(i) * 0.5f;
        mSpan[i] = static_cast<double>(i) * 1.5;
    }

    // Read back (via const)
    ll::DynSoA<ParticleRecord> const& csoa = soa;
    auto cxSpan = csoa.getLeaf<ll::TagPath<posD_t, xComp_t>>();
    auto cmSpan = csoa.getLeaf<massD_t>();
    for(size_t i = 0; i < N; ++i)
    {
        CHECK(cxSpan[i] == Catch::Approx(static_cast<float>(i) * 0.5f));
        CHECK(cmSpan[i] == Catch::Approx(static_cast<double>(i) * 1.5));
    }
}

TEST_CASE("DynSoA leaf spans are contiguous and independent", "[DynSoA]")
{
    constexpr size_t N = 10u;
    ll::DynSoA<ParticleRecord> soa(N);

    auto xSpan = soa.getLeaf<ll::TagPath<posD_t, xComp_t>>();
    auto ySpan = soa.getLeaf<ll::TagPath<posD_t, yComp_t>>();

    // Fill x with 1, y with 2
    std::fill(xSpan.begin(), xSpan.end(), 1.0f);
    std::fill(ySpan.begin(), ySpan.end(), 2.0f);

    // Verify independence
    for(size_t i = 0; i < N; ++i)
    {
        CHECK(xSpan[i] == Catch::Approx(1.0f));
        CHECK(ySpan[i] == Catch::Approx(2.0f));
    }

    // Spans point into different arrays (SoA layout)
    CHECK(xSpan.data() != ySpan.data());
}
