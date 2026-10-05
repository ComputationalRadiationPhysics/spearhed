// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(posD);
DEFINE_TAG(velD);
DEFINE_TAG(massD);
DEFINE_TAG(xComp);
DEFINE_TAG(yComp);
DEFINE_TAG(zComp);
DEFINE_TAG(maskD);
DEFINE_TAG(firstResizeD);
DEFINE_TAG(secondResizeD);
DEFINE_TAG(nestedD);

struct ThrowOnDefault
{
    static inline bool shouldThrow = false;
    static inline bool shouldThrowCopy = false;

    ThrowOnDefault()
    {
        if(shouldThrow)
            throw std::runtime_error("intentional resize failure");
    }

    ThrowOnDefault(ThrowOnDefault const&)
    {
        if(shouldThrowCopy)
            throw std::runtime_error("intentional copy failure");
    }
};

using Pos3D = ll::Record<ll::Field<xComp_t, float>, ll::Field<yComp_t, float>, ll::Field<zComp_t, float>>;

using ParticleRecord = ll::Record<ll::Field<posD_t, Pos3D>, ll::Field<velD_t, Pos3D>, ll::Field<massD_t, double>>;
using FlagRecord = ll::Record<ll::Field<maskD_t, std::uint8_t>>;
using NestedRecord
    = ll::Record<ll::Field<nestedD_t, ll::Record<ll::Field<maskD_t, std::uint8_t>, ll::Field<massD_t, double>>>>;
using ThrowingRecord = ll::Record<
    ll::Field<maskD_t, std::uint8_t>,
    ll::Field<firstResizeD_t, int>,
    ll::Field<secondResizeD_t, ThrowOnDefault>>;
using NestedThrowingRecord = ll::Record<ll::Field<
    nestedD_t,
    ll::Record<
        ll::Field<maskD_t, std::uint8_t>,
        ll::Field<firstResizeD_t, int>,
        ll::Field<secondResizeD_t, ThrowOnDefault>>>>;

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

    CHECK(soa.getLeaf(posD / xComp).size() == N);
    CHECK(soa.getLeaf(posD / yComp).size() == N);
    CHECK(soa.getLeaf(posD / zComp).size() == N);
    CHECK(soa.getLeaf(velD / xComp).size() == N);
    CHECK(soa.getLeaf(velD / yComp).size() == N);
    CHECK(soa.getLeaf(velD / zComp).size() == N);
    CHECK(soa.getLeaf(massD).size() == N);
}

TEST_CASE("DynSoA read/write round-trip via getLeaf", "[DynSoA]")
{
    constexpr size_t N = 64u;
    ll::DynSoA<ParticleRecord> soa(N);

    // Write
    auto xSpan = soa.getLeaf(posD / xComp);
    auto mSpan = soa.getLeaf(massD);
    for(size_t i = 0; i < N; ++i)
    {
        xSpan[i] = static_cast<float>(i) * 0.5f;
        mSpan[i] = static_cast<double>(i) * 1.5;
    }

    // Read back (via const)
    ll::DynSoA<ParticleRecord> const& csoa = soa;
    auto cxSpan = csoa.getLeaf(posD / xComp);
    auto cmSpan = csoa.getLeaf(massD);
    for(size_t i = 0; i < N; ++i)
    {
        CHECK(cxSpan[i] == Catch::Approx(static_cast<float>(i) * 0.5f));
        CHECK(cmSpan[i] == Catch::Approx(static_cast<double>(i) * 1.5));
    }
}

TEST_CASE("DynSoA supports byte-valued flag leaves", "[DynSoA]")
{
    ll::DynSoA<FlagRecord> soa(3);
    auto flags = soa.getLeaf(maskD);
    static_assert(std::is_same_v<decltype(flags), std::span<std::uint8_t>>);
    flags[0] = 1;
    CHECK(flags.data()[0] == 1);
    CHECK(soa[uint32_t{0}][maskD] == 1);
    ll::DynSoA<FlagRecord> const& constSoa = soa;
    static_assert(std::is_same_v<decltype(constSoa.getLeaf(maskD)), std::span<std::uint8_t const>>);
    CHECK(constSoa.getLeaf(maskD)[0] == 1);
}

TEST_CASE("DynSoA supports nested byte flag and numeric leaves", "[DynSoA]")
{
    ll::DynSoA<NestedRecord> soa(2);
    auto mask = soa.getLeaf(nestedD / maskD);
    auto mass = soa.getLeaf(nestedD / massD);
    mask[1] = true;
    mass[1] = 4.5;
    CHECK(mask[1]);
    CHECK(mass[1] == 4.5);
    CHECK(soa[uint32_t{1}][nestedD][maskD]);
}

TEST_CASE("DynSoA copy construction and assignment copy columns", "[DynSoA]")
{
    ll::DynSoA<ThrowingRecord> source(2);
    source.getLeaf(maskD)[0] = true;
    source.getLeaf(firstResizeD)[0] = 21;
    source.getLeaf(firstResizeD)[1] = 22;
    ll::DynSoA<ThrowingRecord> copied(source);
    ll::DynSoA<ThrowingRecord> assigned(1);
    assigned = source;
    CHECK(copied.size() == 2);
    CHECK(copied.getLeaf(maskD)[0]);
    CHECK(copied.getLeaf(firstResizeD)[1] == 22);
    CHECK(assigned.size() == 2);
    CHECK(assigned.getLeaf(maskD)[0]);
    CHECK(assigned.getLeaf(firstResizeD)[0] == 21);
}

TEST_CASE("DynSoA move resets moved-from size", "[DynSoA]")
{
    ll::DynSoA<ParticleRecord> source(2);
    ll::DynSoA<ParticleRecord> moved(std::move(source));
    CHECK(moved.size() == 2);
    CHECK(source.size() == 0);
    CHECK(source.getLeaf(massD).size() == 0);

    ll::DynSoA<ThrowingRecord> boolSource(2);
    boolSource.getLeaf(maskD)[1] = true;
    boolSource.getLeaf(firstResizeD)[0] = 123;
    ll::DynSoA<ThrowingRecord> boolDestination;
    boolDestination = std::move(boolSource);
    CHECK(boolDestination.size() == 2);
    CHECK(boolDestination.getLeaf(maskD)[1]);
    CHECK(boolDestination.getLeaf(firstResizeD)[0] == 123);
    CHECK(boolSource.size() == 0);
    CHECK(boolSource.getLeaf(maskD).size() == 0);
    boolSource.resize(1);
    CHECK(boolSource.size() == 1);
    boolDestination = std::move(boolDestination);
    CHECK(boolDestination.size() == 2);
}

TEST_CASE("DynSoA failed resize preserves column lengths", "[DynSoA]")
{
    ll::DynSoA<ThrowingRecord> soa(1);
    soa.getLeaf(maskD)[0] = true;
    soa.getLeaf(firstResizeD)[0] = 42;
    ThrowOnDefault::shouldThrow = true;
    CHECK_THROWS(soa.resize(3));
    ThrowOnDefault::shouldThrow = false;
    CHECK(soa.size() == 1);
    CHECK(soa.getLeaf(maskD).size() == 1);
    CHECK(soa.getLeaf(firstResizeD).size() == 1);
    CHECK(soa.getLeaf(secondResizeD).size() == 1);
    CHECK(soa.getLeaf(maskD)[0]);
    CHECK(soa.getLeaf(firstResizeD)[0] == 42);
    soa.resize(2);
    CHECK(soa.size() == 2);
    soa.resize(0);
    CHECK(soa.getLeaf(maskD).empty());
    CHECK(soa.getLeaf(firstResizeD).empty());
}

TEST_CASE("DynSoA nested failed resize preserves every column", "[DynSoA]")
{
    ll::DynSoA<NestedThrowingRecord> soa(1);
    soa.getLeaf(nestedD / maskD)[0] = true;
    soa.getLeaf(nestedD / firstResizeD)[0] = 84;
    ThrowOnDefault::shouldThrow = true;
    CHECK_THROWS(soa.resize(3));
    ThrowOnDefault::shouldThrow = false;
    CHECK(soa.size() == 1);
    CHECK(soa.getLeaf(nestedD / maskD).size() == 1);
    CHECK(soa.getLeaf(nestedD / firstResizeD).size() == 1);
    CHECK(soa.getLeaf(nestedD / secondResizeD).size() == 1);
    CHECK(soa.getLeaf(nestedD / maskD)[0]);
    CHECK(soa.getLeaf(nestedD / firstResizeD)[0] == 84);
}

TEST_CASE("DynSoA copy assignment preserves destination on copy failure", "[DynSoA]")
{
    ll::DynSoA<ThrowingRecord> destination(1);
    ll::DynSoA<ThrowingRecord> source(2);
    source.getLeaf(firstResizeD)[0] = 11;
    source.getLeaf(firstResizeD)[1] = 12;
    destination.getLeaf(firstResizeD)[0] = 7;
    ThrowOnDefault::shouldThrowCopy = true;
    CHECK_THROWS(destination = source);
    ThrowOnDefault::shouldThrowCopy = false;
    CHECK(destination.size() == 1);
    CHECK(destination.getLeaf(maskD).size() == 1);
    CHECK(destination.getLeaf(firstResizeD).size() == 1);
    CHECK(destination.getLeaf(secondResizeD).size() == 1);
    CHECK(destination.getLeaf(firstResizeD)[0] == 7);
}

TEST_CASE("DynSoA rejects sizes outside the row-index range", "[DynSoA]")
{
    ll::DynSoA<FlagRecord> soa;
    if constexpr(sizeof(size_t) > sizeof(uint32_t))
    {
        auto const tooLarge = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
        CHECK_THROWS_AS(soa.resize(tooLarge), std::length_error);
        CHECK(soa.size() == 0);
        CHECK_THROWS_AS(ll::DynSoA<FlagRecord>(tooLarge), std::length_error);
    }
}

TEST_CASE("DynSoA leaf spans are contiguous and independent", "[DynSoA]")
{
    constexpr size_t N = 10u;
    ll::DynSoA<ParticleRecord> soa(N);

    auto xSpan = soa.getLeaf(posD / xComp);
    auto ySpan = soa.getLeaf(posD / yComp);

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
