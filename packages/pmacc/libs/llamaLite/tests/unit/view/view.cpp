// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(posO);
DEFINE_TAG(velO);
DEFINE_TAG(massO);
DEFINE_TAG(xO);
DEFINE_TAG(yO);
DEFINE_TAG(zO);
DEFINE_TAG(throwingO);

struct ThrowingValue
{
    static inline bool throwOnAssign = false;
    int value = 0;

    ThrowingValue& operator=(ThrowingValue const& other)
    {
        if(throwOnAssign)
            throw std::runtime_error("intentional assignment failure");
        value = other.value;
        return *this;
    }
};

using Pos3O = ll::Record<ll::Field<xO_t, float>, ll::Field<yO_t, float>, ll::Field<zO_t, float>>;
using ParticleOne = ll::Record<ll::Field<posO_t, Pos3O>, ll::Field<velO_t, Pos3O>, ll::Field<massO_t, double>>;
using EmptyRecord = ll::Record<>;
using ParticleOneView = ll::ViewIndexed<ll::One<ParticleOne>, ll::Set<>>;
using ConstParticleOneView = ll::ViewIndexed<ll::One<ParticleOne> const, ll::Set<>>;
using ParticleOneRootView = ll::View<ll::One<ParticleOne>, ll::Set<>>;
using ConstParticleOneRootView = ll::View<ll::One<ParticleOne> const, ll::Set<>>;

namespace
{
    template<typename T>
    concept RvalueAssignable = requires(T&& lhs, T const& rhs) { static_cast<T&&>(lhs) = rhs; };

    template<typename T>
    concept RvalueSourceAssignable = requires(T&& lhs, T&& rhs) { static_cast<T&&>(lhs) = static_cast<T&&>(rhs); };

    template<typename Dest, typename Src>
    concept CanCopyValues = requires(Dest dest, Src src) { ll::copy_values(dest, src); };

    template<typename Dest, typename Model>
    concept CanSelectLike = requires(Dest dest, Model model) { ll::select_like(dest, model); };

    template<typename Dest, typename Src>
    concept LvalueAssignableFrom = requires(Dest& dest, Src src) { dest = src; };

    template<typename Dest, typename Src>
    concept RvalueAssignableFrom
        = requires(Dest&& dest, Src&& src) { static_cast<Dest&&>(dest) = static_cast<Src&&>(src); };

    using MassLeaves = ll::Set<ll::TagPath<massO_t>>;
    using PositionLeaves = ll::Set<ll::TagPath<posO_t, xO_t>, ll::TagPath<posO_t, yO_t>, ll::TagPath<posO_t, zO_t>>;
    using OverlappingView = decltype(std::declval<ll::One<ParticleOne>&>().view(posO, posO / xO));
    using EmptyRootView = ll::View<ll::One<EmptyRecord>, ll::Set<>>;

    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ParticleOneView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ConstParticleOneView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ParticleOneRootView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ConstParticleOneRootView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(
                      std::declval<ll::ViewIndexed<ll::One<ParticleOne>, ll::access_set_t<massO_t>> const&>())),
                  MassLeaves>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(
                      std::declval<ll::View<ll::One<ParticleOne>, ll::access_set_t<posO_t>> const&>())),
                  PositionLeaves>);
    static_assert(
        std::is_same_v<decltype(ll::getSelectedLeaves(std::declval<OverlappingView const&>())), PositionLeaves>);
    static_assert(std::is_same_v<decltype(ll::getSelectedLeaves(std::declval<EmptyRootView const&>())), ll::Set<>>);
} // namespace

TEST_CASE("getSelectedLeaves reports canonical view selections", "[View]")
{
    using RootLeaves = decltype(ll::getSelectedLeaves(std::declval<ParticleOneRootView const&>()));
    using OverlapLeaves = decltype(ll::getSelectedLeaves(std::declval<OverlappingView const&>()));
    using EmptyLeaves = decltype(ll::getSelectedLeaves(std::declval<EmptyRootView const&>()));
    STATIC_CHECK(RootLeaves::size == 7);
    STATIC_CHECK(OverlapLeaves::size == 3);
    STATIC_CHECK(EmptyLeaves::size == 0);
}

TEST_CASE("SoA getLeaf accepts tags and preserves const access", "[SoA]")
{
    ll::SoA<ParticleOne, 2> soa{};
    auto xSpan = soa.getLeaf(posO / xO);
    auto massSpan = soa.getLeaf(massO);
    xSpan[0] = 3.0f;
    massSpan[0] = 42.0;

    ll::SoA<ParticleOne, 2> const& constSoa = soa;
    auto constXSpan = constSoa.getLeaf(posO / xO);
    auto constMassSpan = constSoa.getLeaf(massO);
    STATIC_CHECK(std::is_const_v<typename decltype(constXSpan)::element_type>);
    STATIC_CHECK(std::is_const_v<typename decltype(constMassSpan)::element_type>);
    CHECK(constXSpan[0] == Catch::Approx(3.0f));
    CHECK(constMassSpan[0] == Catch::Approx(42.0));
}

TEST_CASE("View assignment rebinds named handles and rejects temporary destinations", "[View]")
{
    ll::One<ParticleOne> one{};
    ll::SoA<ParticleOne, 2> soa{};
    soa[0u][massO] = 10.0;
    soa[1u][massO] = 20.0;
    auto first = soa[0u];
    auto second = soa[1u];

    first = second;
    first[massO] = 12.0;
    auto alias = first;
    alias[massO] = 13.0;

    CHECK(soa[0u][massO] == Catch::Approx(10.0));
    CHECK(soa[1u][massO] == Catch::Approx(13.0));
    STATIC_CHECK_FALSE(RvalueAssignable<decltype(first)>);
    STATIC_CHECK_FALSE(RvalueSourceAssignable<decltype(first)>);
    STATIC_CHECK_FALSE(LvalueAssignableFrom<decltype(first), decltype(one[0u])>);
    STATIC_CHECK_FALSE(RvalueAssignableFrom<decltype(first), decltype(one[0u])>);

    ll::SoA<ParticleOne, 2> other{};
    auto root = soa.view();
    auto otherRoot = other.view();
    root = otherRoot;
    root[0u][massO] = 27.0;
    CHECK(other[0u][massO] == Catch::Approx(27.0));
    CHECK(soa[0u][massO] == Catch::Approx(10.0));
    STATIC_CHECK_FALSE(RvalueAssignable<decltype(root)>);
    STATIC_CHECK_FALSE(RvalueSourceAssignable<decltype(root)>);
    STATIC_CHECK_FALSE(RvalueAssignableFrom<decltype(root), decltype(one.view())>);
}

TEST_CASE("copy_values copies exactly the destination selection", "[View]")
{
    ll::One<ParticleOne> one{};
    ll::One<ParticleOne> src{};
    src[0u][posO][xO] = 3.0f;
    src[0u][massO] = 42.0;
    one[0u][posO][xO] = -1.0f;
    one[0u][massO] = -2.0;

    auto dstMass = one.view(massO)[0u];
    auto srcMass = src.view(massO)[0u];
    ll::copy_values(dstMass, srcMass);

    CHECK(one[0u][massO] == Catch::Approx(42.0));
    CHECK(one[0u][posO][xO] == Catch::Approx(-1.0f));

    src[0u][massO] = 99.0;
    auto const& constSrc = src;
    ll::copy_values(one.view(massO)[0u], constSrc.view(massO)[0u]);
    CHECK(one[0u][massO] == Catch::Approx(99.0));
    ll::copy_values(ll::select_like(one[0u], srcMass), srcMass);
    CHECK(one[0u][posO][xO] == Catch::Approx(-1.0f));

    using MassView = decltype(dstMass);
    using PositionView = decltype(one.view(posO)[0u]);
    using RootView = decltype(one[0u]);
    STATIC_CHECK(CanCopyValues<MassView, decltype(src.view(massO)[0u])>);
    STATIC_CHECK_FALSE(CanCopyValues<MassView, PositionView>);
    STATIC_CHECK_FALSE(CanCopyValues<RootView, MassView>);
    STATIC_CHECK(CanSelectLike<RootView, MassView>);
    STATIC_CHECK_FALSE(CanSelectLike<MassView, PositionView>);
    using ConstRootView = ll::ViewIndexed<ll::One<ParticleOne> const, ll::Set<>>;
    STATIC_CHECK(CanSelectLike<ConstRootView, MassView>);
    STATIC_CHECK_FALSE(CanCopyValues<ConstRootView, MassView>);

    using EmptyView = decltype(std::declval<ll::One<EmptyRecord>&>()[0u]);
    STATIC_CHECK_FALSE(CanSelectLike<RootView, EmptyView>);
}

TEST_CASE("copy_values copies nested multi-field selections and preserves other leaves", "[View]")
{
    ll::One<ParticleOne> dest{};
    ll::One<ParticleOne> src{};
    src[0u][posO][xO] = 8.5f;
    src[0u][velO][yO] = 6.5f;
    dest[0u][massO] = -3.0;

    ll::copy_values(dest.view(posO, velO)[0u], src.view(posO, velO)[0u]);

    CHECK(dest[0u][posO][xO] == Catch::Approx(8.5f));
    CHECK(dest[0u][velO][yO] == Catch::Approx(6.5f));
    CHECK(dest[0u][massO] == Catch::Approx(-3.0));
}

TEST_CASE("select_like copies a root subrecord into a larger record", "[View]")
{
    using Accumulator = ll::Record<ll::Field<massO_t, double>>;
    ll::One<ParticleOne> dest{};
    ll::One<Accumulator> accumulators{};
    accumulators[0u][massO] = 17.0;
    dest[0u][massO] = -1.0;
    dest[0u][posO][xO] = 5.0f;

    auto accumulator = accumulators[0u];
    ll::copy_values(ll::select_like(dest[0u], accumulator), accumulator);

    CHECK(dest[0u][massO] == Catch::Approx(17.0));
    CHECK(dest[0u][posO][xO] == Catch::Approx(5.0f));
}

TEST_CASE("copy_values propagates throwing leaf assignments", "[View]")
{
    using ThrowRecord = ll::Record<ll::Field<throwingO_t, ThrowingValue>>;
    ll::One<ThrowRecord> src{};
    ll::One<ThrowRecord> dest{};
    src[0u][throwingO].value = 9;

    ThrowingValue::throwOnAssign = true;
    bool threw = false;
    try
    {
        ll::copy_values(dest[0u], src[0u]);
    }
    catch(std::runtime_error const&)
    {
        threw = true;
    }
    ThrowingValue::throwOnAssign = false;

    CHECK(threw);
    CHECK(dest[0u][throwingO].value == 0);
}
