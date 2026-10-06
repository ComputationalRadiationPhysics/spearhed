// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.mozilla.org/MPL/2.0/.

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
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
using IndexedParticleView = ll::ViewIndexed<ll::One<ParticleRecord>>;
using ConstIndexedParticleView = ll::ViewIndexed<ll::One<ParticleRecord> const>;
using ParticleRootView = ll::View<ll::One<ParticleRecord>>;
using ConstParticleRootView = ll::View<ll::One<ParticleRecord> const>;

void tupleFunction()
{
}

struct TupleCatCounter
{
    static inline int copyCount = 0;
    static inline int moveCount = 0;

    explicit TupleCatCounter(int input) : value(input)
    {
    }

    TupleCatCounter(TupleCatCounter const& other) : value(other.value)
    {
        ++copyCount;
    }

    TupleCatCounter(TupleCatCounter&& other) noexcept : value(other.value)
    {
        ++moveCount;
    }

    int value;
};

struct TupleCatUnconstructible
{
    TupleCatUnconstructible() = delete;
    TupleCatUnconstructible(TupleCatUnconstructible const&) = delete;
};

namespace
{
    template<typename T>
    concept HasRootRecord = requires(T const& object) { ll::getRootRecord(object); };

    template<typename T>
    concept CanTieRvalue = requires(T&& value) { ll::tie(std::forward<T>(value)); };

    template<typename T, typename U>
    concept CanAssignFrom = requires(T& target, U&& source) { target = std::forward<U>(source); };
} // namespace

TEST_CASE("Record queries accept tag objects", "[Record]")
{
    STATIC_CHECK(ParticleRecord::hasPath(ll::TagPath<>{}));
    STATIC_CHECK(EmptyRecord::hasPath(ll::TagPath<>{}));
    STATIC_CHECK(ParticleRecord::hasPath(positionRecord / xRecord));
    STATIC_CHECK_FALSE(ParticleRecord::hasPath(xRecord));
    STATIC_CHECK_FALSE(ParticleRecord::hasPath(ll::TagPath<massRecord_t, xRecord_t>{}));
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

TEST_CASE("Tuple get and apply preserve value categories", "[Tuple]")
{
    using ValueTuple = ll::Tuple<int>;
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple const&>())), int const&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple&&>())), int&&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple const&&>())), int const&&>);

    using ReferenceTuple = ll::Tuple<int&>;
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ReferenceTuple&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ReferenceTuple const&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ReferenceTuple&&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ReferenceTuple const&&>())), int&>);

    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple const&>())), int const&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple&&>())), int&&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ValueTuple const&&>())), int const&&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ReferenceTuple&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<ReferenceTuple const&&>())), int&>);

    using RvalueReferenceTuple = ll::Tuple<int&&>;
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<RvalueReferenceTuple&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<RvalueReferenceTuple const&>())), int&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<RvalueReferenceTuple&&>())), int&&>);
    STATIC_CHECK(std::is_same_v<decltype(ll::get<0>(std::declval<RvalueReferenceTuple const&&>())), int&&>);

    auto tuple = ll::makeTuple(std::make_unique<int>(17));
    auto value = ll::tuple::apply([](std::unique_ptr<int> ptr) { return *ptr; }, std::move(tuple));
    CHECK(value == 17);

    struct RvalueCallable
    {
        int operator()(int value) &&
        {
            return value + 1;
        }
    };

    CHECK(ll::tuple::apply(RvalueCallable{}, ll::Tuple<int>{3}) == 4);
}

TEST_CASE("Tuple factory decays values and unwraps reference wrappers", "[Tuple]")
{
    int value = 3;
    auto values = ll::makeTuple(value);
    STATIC_CHECK(std::is_same_v<decltype(values), ll::Tuple<int>>);
    ll::tuple::get<0>(values) = 4;
    CHECK(value == 3);

    auto reference = ll::makeTuple(std::ref(value));
    STATIC_CHECK(std::is_same_v<decltype(reference), ll::Tuple<int&>>);
    ll::tuple::get<0>(reference) = 5;
    CHECK(value == 5);

    int const constValue = 6;
    auto constValues = ll::makeTuple(constValue);
    STATIC_CHECK(std::is_same_v<decltype(constValues), ll::Tuple<int>>);
    CHECK(ll::tuple::get<0>(constValues) == 6);

    auto constReference = ll::makeTuple(std::cref(value));
    STATIC_CHECK(std::is_same_v<decltype(constReference), ll::Tuple<int const&>>);
    STATIC_CHECK(std::is_same_v<decltype(ll::tuple::get<0>(constReference)), int const&>);
    CHECK(ll::tuple::get<0>(constReference) == value);

    auto text = ll::makeTuple("hello");
    STATIC_CHECK(std::is_same_v<decltype(text), ll::Tuple<char const*>>);
    CHECK(std::string(ll::tuple::get<0>(text)) == "hello");

    auto function = ll::makeTuple(tupleFunction);
    STATIC_CHECK(std::is_same_v<decltype(function), ll::Tuple<void (*)()>>);

    ll::Tuple directDeduction{value};
    STATIC_CHECK(std::is_same_v<decltype(directDeduction), ll::Tuple<int>>);
    ll::tuple::get<0>(directDeduction) = 8;
    CHECK(value == 5);

    ll::Tuple constDeduction{constValue};
    STATIC_CHECK(std::is_same_v<decltype(constDeduction), ll::Tuple<int>>);

    int array[2] = {1, 2};
    ll::Tuple arrayDeduction{array};
    STATIC_CHECK(std::is_same_v<decltype(arrayDeduction), ll::Tuple<int*>>);

    ll::Tuple functionDeduction{tupleFunction};
    STATIC_CHECK(std::is_same_v<decltype(functionDeduction), ll::Tuple<void (*)()>>);

    ll::Tuple wrapperDeduction{std::ref(value)};
    STATIC_CHECK(std::is_same_v<decltype(wrapperDeduction), ll::Tuple<std::reference_wrapper<int>>>);
    ll::tuple::get<0>(wrapperDeduction).get() = 9;
    CHECK(value == 9);

    auto directMoveOnly = ll::Tuple{std::make_unique<int>(10)};
    STATIC_CHECK(std::is_same_v<decltype(directMoveOnly), ll::Tuple<std::unique_ptr<int>>>);
    CHECK(*ll::tuple::get<0>(std::move(directMoveOnly)) == 10);

    auto empty = ll::makeTuple();
    STATIC_CHECK(std::is_same_v<decltype(empty), ll::Tuple<>>);
    STATIC_CHECK(std::is_trivially_copyable_v<decltype(empty)>);

    auto moveOnly = ll::makeTuple(std::make_unique<int>(7));
    STATIC_CHECK(std::is_same_v<decltype(moveOnly), ll::Tuple<std::unique_ptr<int>>>);
    CHECK(*ll::tuple::get<0>(std::move(moveOnly)) == 7);
}

TEST_CASE("tupleCat concatenates values, references, and move-only elements", "[Tuple]")
{
    auto empty = ll::tupleCat();
    STATIC_CHECK(std::is_same_v<decltype(empty), ll::Tuple<>>);

    auto only = ll::makeTuple(1, 2.0);
    auto onlyCopy = ll::tupleCat(only);
    STATIC_CHECK(std::is_same_v<decltype(onlyCopy), decltype(only)>);
    CHECK(ll::get<0>(onlyCopy) == 1);
    CHECK(ll::get<1>(onlyCopy) == 2.0);

    int first = 3;
    double second = 4.0;
    auto firstTuple = ll::makeTuple(first);
    auto const secondTuple = ll::makeTuple(second);
    auto joined = ll::tupleCat(firstTuple, secondTuple, ll::Tuple<char>{'x'});
    STATIC_CHECK(std::is_same_v<decltype(joined), ll::Tuple<int, double, char>>);
    CHECK(ll::get<0>(joined) == 3);
    CHECK(ll::get<1>(joined) == 4.0);
    CHECK(ll::get<2>(joined) == 'x');

    auto references = ll::tupleCat(ll::tie(first), ll::Tuple<double&>{second});
    STATIC_CHECK(std::is_same_v<decltype(references), ll::Tuple<int&, double&>>);
    ll::get<0>(references) = 7;
    ll::get<1>(references) = 8.0;
    CHECK(first == 7);
    CHECK(second == 8.0);

    auto moveOnly = ll::tupleCat(ll::makeTuple(std::make_unique<int>(9)), ll::makeTuple(std::make_unique<int>(10)));
    STATIC_CHECK(std::is_same_v<decltype(moveOnly), ll::Tuple<std::unique_ptr<int>, std::unique_ptr<int>>>);
    CHECK(*ll::get<0>(moveOnly) == 9);
    CHECK(*ll::get<1>(moveOnly) == 10);

    ll::Tuple<TupleCatCounter> counterFirst{TupleCatCounter{11}};
    ll::Tuple<TupleCatCounter> counterSecond{TupleCatCounter{12}};
    TupleCatCounter::copyCount = 0;
    TupleCatCounter::moveCount = 0;
    auto counters = ll::tupleCat(std::move(counterFirst), std::move(counterSecond));
    CHECK(TupleCatCounter::copyCount == 0);
    CHECK(TupleCatCounter::moveCount == 2);
    CHECK(ll::get<0>(counters).value == 11);
    CHECK(ll::get<1>(counters).value == 12);

    using TypeOnly
        = decltype(ll::tupleCat(std::declval<ll::Tuple<TupleCatUnconstructible>&>(), std::declval<ll::Tuple<int>&>()));
    using Expected = typename ll::ConcatTuples<ll::Tuple<TupleCatUnconstructible>, ll::Tuple<int>>::type;
    STATIC_CHECK(std::is_same_v<TypeOnly, Expected>);
}

TEST_CASE("Tuple tie and forwarding-reference factories preserve references", "[Tuple]")
{
    int value = 3;
    auto tied = ll::tie(value);
    STATIC_CHECK(std::is_same_v<decltype(tied), ll::Tuple<int&>>);
    ll::tuple::get<0>(tied) = 4;
    CHECK(value == 4);
    STATIC_CHECK(noexcept(ll::tie(value)));
    STATIC_CHECK_FALSE(CanTieRvalue<int>);
    STATIC_CHECK(CanTieRvalue<int&>);

    int const constValue = 5;
    auto tiedConst = ll::tie(constValue);
    STATIC_CHECK(std::is_same_v<decltype(tiedConst), ll::Tuple<int const&>>);
    STATIC_CHECK(std::is_same_v<decltype(ll::tuple::get<0>(tiedConst)), int const&>);

    auto emptyTie = ll::tie();
    STATIC_CHECK(std::is_same_v<decltype(emptyTie), ll::Tuple<>>);

    auto forwardedLvalue = ll::forwardAsTuple(value);
    STATIC_CHECK(std::is_same_v<decltype(forwardedLvalue), ll::Tuple<int&>>);
    ll::tuple::get<0>(forwardedLvalue) = 6;
    CHECK(value == 6);
    STATIC_CHECK(noexcept(ll::forwardAsTuple(value)));

    STATIC_CHECK(std::is_same_v<decltype(ll::forwardAsTuple(std::declval<int>())), ll::Tuple<int&&>>);
    CHECK(ll::tuple::apply([](int&& item) { return item; }, ll::forwardAsTuple(7)) == 7);

    auto emptyForwardingTuple = ll::forwardAsTuple();
    STATIC_CHECK(std::is_same_v<decltype(emptyForwardingTuple), ll::Tuple<>>);
}

TEST_CASE("Tuple reference assignment writes through and preserves bindings", "[Tuple]")
{
    int target = 0;
    ll::tie(target) = ll::makeTuple(42);
    CHECK(target == 42);

    int sameTypeSource = 17;
    auto destination = ll::tie(target);
    auto source = ll::tie(sameTypeSource);
    int* const originalReference = std::addressof(ll::tuple::get<0>(destination));
    destination = source;
    CHECK(target == 17);
    CHECK(std::addressof(ll::tuple::get<0>(destination)) == originalReference);

    target = 0;
    destination = std::move(source);
    CHECK(target == 17);
    CHECK(std::addressof(ll::tuple::get<0>(destination)) == originalReference);

    short narrowSource = 23;
    destination = ll::makeTuple(narrowSource);
    CHECK(target == 23);

    ll::Tuple<int&, long> mixedDestination{target, 0};
    ll::Tuple<short, int> mixedSource{short{31}, 32};
    mixedDestination = mixedSource;
    CHECK(target == 31);
    CHECK(ll::get<1>(mixedDestination) == 32);

    struct NoComma
    {
        int value;

        void operator,(NoComma&) = delete;
    };

    NoComma first{0};
    NoComma second{0};
    ll::Tuple<NoComma&, NoComma&> noCommaDestination{first, second};
    ll::Tuple<NoComma, NoComma> noCommaSource{NoComma{11}, NoComma{12}};
    noCommaDestination = noCommaSource;
    CHECK(first.value == 11);
    CHECK(second.value == 12);

    auto uniqueTarget = std::make_unique<int>(0);
    ll::tie(uniqueTarget) = ll::makeTuple(std::make_unique<int>(8));
    REQUIRE(uniqueTarget != nullptr);
    CHECK(*uniqueTarget == 8);
    ll::tie(uniqueTarget) = ll::forwardAsTuple(std::make_unique<int>(9));
    REQUIRE(uniqueTarget != nullptr);
    CHECK(*uniqueTarget == 9);

    STATIC_CHECK(!CanAssignFrom<ll::Tuple<int const&>, ll::Tuple<int>>);
    STATIC_CHECK(!CanAssignFrom<ll::Tuple<int&>, ll::Tuple<std::string>>);
    STATIC_CHECK(!CanAssignFrom<ll::Tuple<int&, long>, ll::Tuple<int>>);

    struct ThrowingAssign
    {
        ThrowingAssign& operator=(ThrowingAssign const&)
        {
            throw std::runtime_error("copy assignment failed");
        }

        ThrowingAssign& operator=(ThrowingAssign&&)
        {
            throw std::runtime_error("move assignment failed");
        }
    };

    STATIC_CHECK(std::is_nothrow_copy_assignable_v<ll::Tuple<int&>>);
    STATIC_CHECK(std::is_nothrow_move_assignable_v<ll::Tuple<int&>>);
    STATIC_CHECK(!std::is_nothrow_copy_assignable_v<ll::Tuple<ThrowingAssign&>>);
    STATIC_CHECK(!std::is_nothrow_move_assignable_v<ll::Tuple<ThrowingAssign&>>);

    ThrowingAssign throwingTarget;
    ThrowingAssign throwingSource;
    auto throwingDestination = ll::tie(throwingTarget);
    auto throwingTuple = ll::tie(throwingSource);
    CHECK_THROWS(throwingDestination = throwingTuple);
    CHECK_THROWS(throwingDestination = std::move(throwingTuple));
}

TEST_CASE("Tuple forwards explicit element construction", "[Tuple]")
{
    struct NoThrowValue
    {
        explicit NoThrowValue(int) noexcept
        {
        }
    };

    struct ThrowingValue
    {
        explicit ThrowingValue(int) noexcept(false)
        {
        }
    };

    STATIC_CHECK(std::is_nothrow_constructible_v<ll::Tuple<NoThrowValue>, int>);
    STATIC_CHECK(!std::is_nothrow_constructible_v<ll::Tuple<ThrowingValue>, int>);

    struct ExplicitValue
    {
        explicit ExplicitValue(int input) : value(input)
        {
        }

        int value;
    };

    STATIC_CHECK(std::is_constructible_v<ll::Tuple<ExplicitValue>, int>);
    STATIC_CHECK(!std::is_constructible_v<ll::Tuple<int>, std::string>);
    STATIC_CHECK(std::is_nothrow_constructible_v<ll::Tuple<int>, int>);

    auto tuple = ll::Tuple<ExplicitValue>{42};
    CHECK(ll::get<0>(tuple).value == 42);

    auto initialized = ll::Tuple<int, double>{};
    STATIC_CHECK(std::is_trivially_copyable_v<decltype(initialized)>);
    STATIC_CHECK(std::is_trivially_copy_assignable_v<decltype(initialized)>);
    STATIC_CHECK(std::is_trivially_move_assignable_v<decltype(initialized)>);
    CHECK(ll::get<0>(initialized) == 0);
    CHECK(ll::get<1>(initialized) == 0.0);

    auto [first, second] = ll::Tuple<int, double>{1, 2.0};
    CHECK(first == 1);
    CHECK(second == 2.0);

    auto strings = ll::Tuple<std::string>{std::string("copied")};
    auto copied = strings;
    auto moved = std::move(copied);
    CHECK(ll::get<0>(strings) == "copied");
    CHECK(ll::get<0>(moved) == "copied");
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
