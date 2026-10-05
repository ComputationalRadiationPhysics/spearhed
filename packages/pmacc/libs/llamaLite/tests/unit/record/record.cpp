// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.mozilla.org/MPL/2.0/.

#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(positionRecord);
DEFINE_TAG(massRecord);
DEFINE_TAG(xRecord);

using PositionRecord = ll::Record<ll::Field<xRecord_t, float>>;
using ParticleRecord = ll::Record<ll::Field<positionRecord_t, PositionRecord>, ll::Field<massRecord_t, double>>;
using EmptyRecord = ll::Record<>;
using IndexedParticleView = ll::ViewIndexed<ll::One<ParticleRecord>, ll::Set<>>;
using ConstIndexedParticleView = ll::ViewIndexed<ll::One<ParticleRecord> const, ll::Set<>>;
using ParticleRootView = ll::View<ll::One<ParticleRecord>, ll::Set<>>;
using ConstParticleRootView = ll::View<ll::One<ParticleRecord> const, ll::Set<>>;

namespace
{
    template<typename T>
    concept HasRootRecord = requires(T const& object) { ll::getRootRecord(object); };
} // namespace

TEST_CASE("Record queries accept tag objects", "[Record]")
{
    STATIC_CHECK(ParticleRecord::hasPath(positionRecord / xRecord));
    STATIC_CHECK(ParticleRecord::hasPath(massRecord));
    STATIC_CHECK(ParticleRecord::isLeaf(massRecord));
    STATIC_CHECK(ParticleRecord::isLeaf(positionRecord / xRecord));
    STATIC_CHECK_FALSE(ParticleRecord::isLeaf(positionRecord));
    STATIC_CHECK(
        std::is_same_v<
            decltype(ParticleRecord::resolvePathToField(positionRecord / xRecord)),
            ll::Field<xRecord_t, float>>);
    STATIC_CHECK(
        std::is_same_v<decltype(ParticleRecord::resolvePathToField(massRecord)), ll::Field<massRecord_t, double>>);
}

TEST_CASE("getRootRecord returns backing record metadata", "[Record]")
{
    STATIC_CHECK(std::is_same_v<decltype(ll::getRootRecord(std::declval<ParticleRecord const&>())), ParticleRecord>);
    STATIC_CHECK(std::is_same_v<decltype(ll::getRootRecord(std::declval<EmptyRecord const&>())), EmptyRecord>);
    STATIC_CHECK(
        std::is_same_v<decltype(ll::getRootRecord(std::declval<ll::One<ParticleRecord> const&>())), ParticleRecord>);
    STATIC_CHECK(
        std::
            is_same_v<decltype(ll::getRootRecord(std::declval<ll::SoA<ParticleRecord, 2> const&>())), ParticleRecord>);
    STATIC_CHECK(
        std::is_same_v<decltype(ll::getRootRecord(std::declval<IndexedParticleView const&>())), ParticleRecord>);
    STATIC_CHECK(
        std::is_same_v<decltype(ll::getRootRecord(std::declval<ConstIndexedParticleView const&>())), ParticleRecord>);
    STATIC_CHECK(std::is_same_v<decltype(ll::getRootRecord(std::declval<ParticleRootView const&>())), ParticleRecord>);
    STATIC_CHECK(
        std::is_same_v<decltype(ll::getRootRecord(std::declval<ConstParticleRootView const&>())), ParticleRecord>);
    STATIC_CHECK(
        std::
            is_same_v<decltype(ll::getRootRecord(std::declval<ll::DynSoA<ParticleRecord> const&>())), ParticleRecord>);

    STATIC_CHECK(ll::getRootRecord(ParticleRecord{}).leaf_count == 2);
    STATIC_CHECK(ll::getRootRecord(EmptyRecord{}).leaf_count == 0);
    STATIC_CHECK_FALSE(HasRootRecord<int>);
}
