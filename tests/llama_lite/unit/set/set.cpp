// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

namespace
{
    // Define dummy tags for testing
    struct TagA : llama_lite::TagBase
    {
    };

    struct TagB : llama_lite::TagBase
    {
    };

    struct TagC : llama_lite::TagBase
    {
    };

    struct TagD : llama_lite::TagBase
    {
    };

    inline constexpr TagA a{};
    inline constexpr TagB b{};
    inline constexpr TagC c{};
    inline constexpr TagD d{};

    // Helper alias to make tests more readable
    template<typename... Ts>
    using S = llama_lite::Set<Ts...>;

    // Not compiled - documents the uniqueness guard:
    // llama_lite::Set<TagA, TagA> dup; // static_assert: Duplicate elements detected in Set.
    // llama_lite::add(llama_lite::makeSet(a), a); // static_assert: Element already present in Set.
} // namespace

TEST_CASE("Set Properties and Concepts", "[Set][Meta]")
{
    using namespace llama_lite;

    SECTION("Concept Validation")
    {
        STATIC_CHECK(IsSet<S<TagA>>);
        STATIC_CHECK(IsSet<S<>>);
        STATIC_CHECK(!IsSet<int>);
        STATIC_CHECK(!IsSet<TagA>);
    }

    SECTION("Size and Empty")
    {
        STATIC_CHECK(S<>::size == 0);
        STATIC_CHECK(S<>::empty);
        STATIC_CHECK(S<TagA, TagB>::size == 2);
        STATIC_CHECK(!S<TagA, TagB>::empty);
    }
}

TEST_CASE("Set makeSet", "[Set][Construction]")
{
    using namespace llama_lite;

    SECTION("Deduces element types")
    {
        constexpr auto s = makeSet(a, b, c);
        STATIC_CHECK(s.size == 3);
        STATIC_CHECK(s.contains(a));
        STATIC_CHECK(s.contains(b));
        STATIC_CHECK(s.contains(c));
        STATIC_CHECK_FALSE(s.contains(d));
    }

    SECTION("Deduplicates repeated elements")
    {
        constexpr auto s = makeSet(a, a, b, a);
        STATIC_CHECK(s.size == 2);
        STATIC_CHECK(s.contains(a));
        STATIC_CHECK(s.contains(b));
    }

    SECTION("Empty makeSet")
    {
        constexpr auto s = makeSet();
        STATIC_CHECK(s.empty);
    }

    SECTION("Accepts TagPaths as elements too")
    {
        constexpr auto p = TagA{} / TagB{};
        constexpr auto s = makeSet(a, p);
        STATIC_CHECK(s.size == 2);
        STATIC_CHECK(s.contains(a));
        STATIC_CHECK(s.contains(p));
        STATIC_CHECK_FALSE(s.contains(b));
    }
}

TEST_CASE("Set Membership", "[Set][Operator]")
{
    using namespace llama_lite;
    constexpr auto s = makeSet(a, b);

    SECTION("contains")
    {
        STATIC_CHECK(s.contains(a));
        STATIC_CHECK(s.contains(b));
        STATIC_CHECK_FALSE(s.contains(c));
    }

    SECTION("operator() shorthand")
    {
        STATIC_CHECK(s(a));
        STATIC_CHECK(s(b));
        STATIC_CHECK_FALSE(s(c));
    }
}

TEST_CASE("Set Algebra", "[Set][Operator]")
{
    using namespace llama_lite;

    SECTION("Union (|) deduplicates shared elements")
    {
        constexpr auto u = makeSet(a, b) | makeSet(b, c);
        STATIC_CHECK(u.size == 3);
        STATIC_CHECK(u.contains(a));
        STATIC_CHECK(u.contains(b));
        STATIC_CHECK(u.contains(c));
        STATIC_CHECK(u == makeSet(a, b, c));
    }

    SECTION("Intersection (&)")
    {
        constexpr auto i = makeSet(a, b, c) & makeSet(b, c, d);
        STATIC_CHECK(i.size == 2);
        STATIC_CHECK(i.contains(b));
        STATIC_CHECK(i.contains(c));
        STATIC_CHECK_FALSE(i.contains(a));
        STATIC_CHECK_FALSE(i.contains(d));
    }

    SECTION("Intersection with disjoint set is empty")
    {
        constexpr auto i = makeSet(a, b) & makeSet(c, d);
        STATIC_CHECK(i.empty);
    }

    SECTION("Difference (-)")
    {
        constexpr auto diff = makeSet(a, b, c) - makeSet(b, c);
        STATIC_CHECK(diff.size == 1);
        STATIC_CHECK(diff.contains(a));
        STATIC_CHECK_FALSE(diff.contains(b));
    }

    SECTION("Symmetric difference (^)")
    {
        constexpr auto sym = makeSet(a, b) ^ makeSet(b, c);
        STATIC_CHECK(sym.size == 2);
        STATIC_CHECK(sym.contains(a));
        STATIC_CHECK(sym.contains(c));
        STATIC_CHECK_FALSE(sym.contains(b));
    }
}

TEST_CASE("Set Relations", "[Set][Operator]")
{
    using namespace llama_lite;

    SECTION("Equality is order independent")
    {
        STATIC_CHECK(makeSet(a, b, c) == makeSet(c, b, a));
        STATIC_CHECK_FALSE(makeSet(a, b) == makeSet(a, b, c));
        STATIC_CHECK(makeSet() == makeSet());
    }

    SECTION("Inequality")
    {
        STATIC_CHECK(makeSet(a, b) != makeSet(a, c));
    }

    SECTION("Subset (<=)")
    {
        STATIC_CHECK(makeSet(a, b) <= makeSet(a, b, c));
        STATIC_CHECK(makeSet(a, b) <= makeSet(a, b));
        STATIC_CHECK_FALSE(makeSet(a, d) <= makeSet(a, b, c));
        STATIC_CHECK(makeSet() <= makeSet(a, b));
    }

    SECTION("Superset (>=)")
    {
        STATIC_CHECK(makeSet(a, b, c) >= makeSet(a, b));
        STATIC_CHECK_FALSE(makeSet(a, b, c) >= makeSet(a, d));
    }

    SECTION("is_subset free function")
    {
        STATIC_CHECK(is_subset(makeSet(a, b), makeSet(a, b, c)));
        STATIC_CHECK_FALSE(is_subset(makeSet(a, d), makeSet(a, b, c)));
    }
}

TEST_CASE("Set add/remove", "[Set][Operator]")
{
    using namespace llama_lite;

    SECTION("add appends a new element")
    {
        constexpr auto s = add(makeSet(a, b), c);
        STATIC_CHECK(s == makeSet(a, b, c));
    }

    SECTION("remove drops an element")
    {
        constexpr auto s = remove(makeSet(a, b, c), b);
        STATIC_CHECK(s == makeSet(a, c));
    }

    SECTION("remove is a no-op if the element is absent")
    {
        constexpr auto s = remove(makeSet(a, b), d);
        STATIC_CHECK(s == makeSet(a, b));
    }
}

TEST_CASE("Set forEach visitation", "[Set][Algorithm]")
{
    using namespace llama_lite;

    SECTION("Visits every element exactly once")
    {
        unsigned mask = 0;
        forEach(
            makeSet(a, b, c),
            [&](auto tag)
            {
                using T = decltype(tag);
                if constexpr(std::is_same_v<T, TagA>)
                    mask |= 1U;
                else if constexpr(std::is_same_v<T, TagB>)
                    mask |= 2U;
                else if constexpr(std::is_same_v<T, TagC>)
                    mask |= 4U;
            });
        CHECK(mask == 0b111U);
    }

    SECTION("Empty set visits nothing")
    {
        int count = 0;
        forEach(makeSet(), [&](auto /*tag*/) { ++count; });
        CHECK(count == 0);
    }
}
