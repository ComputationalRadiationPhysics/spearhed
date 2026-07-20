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

    // Helper alias to make tests more readable
    template<typename... Ts>
    using Path = llama_lite::TagPath<Ts...>;
} // namespace

TEST_CASE("TagPath Properties and Concepts", "[TagPath][Meta]")
{
    using namespace llama_lite;

    SECTION("Concept Validation")
    {
        STATIC_CHECK(IsTag<TagA>);
        STATIC_CHECK(!IsTag<int>);

        STATIC_CHECK(IsTagPath<Path<TagA>>);
        STATIC_CHECK(IsTagPath<Path<>>);

        STATIC_CHECK(IsRecordAccess<TagA>);
        STATIC_CHECK(IsRecordAccess<Path<TagA>>);
    }

    SECTION("Depth and Tuples")
    {
        STATIC_CHECK(Path<>::depth == 0);
        STATIC_CHECK(Path<TagA>::depth == 1);
        STATIC_CHECK(Path<TagA, TagB>::depth == 2);

        STATIC_CHECK(std::is_same_v<Path<TagA, TagB>::TagsTuple, Tuple<TagA, TagB>>);
    }

    SECTION("Element Access (tag_at)")
    {
        using P = Path<TagA, TagB, TagC>;
        STATIC_CHECK(std::is_same_v<P::tag_at<0>, TagA>);
        STATIC_CHECK(std::is_same_v<P::tag_at<1>, TagB>);
        STATIC_CHECK(std::is_same_v<P::tag_at<2>, TagC>);
    }
}

TEST_CASE("TagPath Manipulation", "[TagPath][Meta]")
{
    using namespace llama_lite;
    using P = Path<TagA, TagB, TagC>;

    SECTION("take_first")
    {
        STATIC_CHECK(P::take_first<0>::isSame<Path<>>());
        STATIC_CHECK(P::take_first<1>::isSame<Path<TagA>>());
        STATIC_CHECK(P::take_first<2>::isSame<Path<TagA, TagB>>());
        STATIC_CHECK(P::take_first<3>::isSame<P>());
    }

    SECTION("drop_first")
    {
        STATIC_CHECK(P::drop_first<0>::isSame<P>());
        STATIC_CHECK(P::drop_first<1>::isSame<Path<TagB, TagC>>());
        STATIC_CHECK(P::drop_first<2>::isSame<Path<TagC>>());
        STATIC_CHECK(P::drop_first<3>::isSame<Path<>>());
    }
}

TEST_CASE("TagPath Relationships", "[TagPath][Meta]")
{
    using namespace llama_lite;
    using Empty = Path<>;
    using P_A = Path<TagA>;
    using P_B = Path<TagB>;
    using P_AB = Path<TagA, TagB>;
    using P_AC = Path<TagA, TagC>;
    using P_ABC = Path<TagA, TagB, TagC>;

    SECTION("Equality (isSame)")
    {
        STATIC_CHECK(P_A::isSame<P_A>());
        STATIC_CHECK(P_A::isSame<TagA>()); // Tag normalizes to TagPath
        STATIC_CHECK_FALSE(P_A::isSame<P_B>());
        STATIC_CHECK(Empty::isSame<Path<>>());
    }

    SECTION("Ancestry (isAncestorOf)")
    {
        // Empty path is ancestor of everything
        STATIC_CHECK(Empty::isAncestorOf<P_A>());
        STATIC_CHECK(Empty::isAncestorOf<Empty>());

        // Identity
        STATIC_CHECK(P_AB::isAncestorOf<P_AB>());

        // True ancestry
        STATIC_CHECK(P_A::isAncestorOf<P_AB>());
        STATIC_CHECK(P_A::isAncestorOf<P_ABC>());
        STATIC_CHECK(P_AB::isAncestorOf<P_ABC>());

        // False ancestry
        STATIC_CHECK_FALSE(P_B::isAncestorOf<P_A>());
        STATIC_CHECK_FALSE(P_AB::isAncestorOf<P_A>()); // Parent is not ancestor of child
        STATIC_CHECK_FALSE(P_AB::isAncestorOf<P_AC>());
    }

    SECTION("Strict Ancestry")
    {
        STATIC_CHECK(P_A::isStrictAncestorOf<P_AB>());
        STATIC_CHECK_FALSE(P_A::isStrictAncestorOf<P_A>()); // Strict implies not equal
        STATIC_CHECK(Empty::isStrictAncestorOf<P_A>());
    }

    SECTION("Descendancy")
    {
        STATIC_CHECK(P_AB::isDescendantOf<P_A>());
        STATIC_CHECK(P_AB::isStrictDescendantOf<P_A>());
        STATIC_CHECK_FALSE(P_A::isDescendantOf<P_B>());
    }

    SECTION("Common Prefix Length")
    {
        STATIC_CHECK(P_AB::commonPrefixLength<P_AC>() == 1); // TagA
        STATIC_CHECK(P_AB::commonPrefixLength<P_ABC>() == 2); // TagA, TagB
        STATIC_CHECK(P_A::commonPrefixLength<P_B>() == 0);
        STATIC_CHECK(Empty::commonPrefixLength<P_ABC>() == 0);
        STATIC_CHECK(P_ABC::commonPrefixLength<Empty>() == 0);
    }
}

TEST_CASE("TagPath Concatenation", "[TagPath][Operator]")
{
    using namespace llama_lite;

    SECTION("Operator /")
    {
        auto path = TagA{} / TagB{};
        STATIC_CHECK(std::is_same_v<decltype(path), Path<TagA, TagB>>);

        auto path2 = Path<TagA>{} / TagB{};
        STATIC_CHECK(std::is_same_v<decltype(path2), Path<TagA, TagB>>);

        auto path3 = Path<TagA>{} / Path<TagB, TagC>{};
        STATIC_CHECK(std::is_same_v<decltype(path3), Path<TagA, TagB, TagC>>);
    }
}

TEST_CASE("TagPath Redundancy Checks", "[TagPath][Meta]")
{
    using namespace llama_lite;

    // hasRedundantPaths returns true if any path is a strict ancestor of another

    SECTION("No Redundancy")
    {
        // Distinct paths
        STATIC_CHECK_FALSE(hasRedundantPaths<Path<TagA>, Path<TagB>>());
        // Disjoint branches
        STATIC_CHECK_FALSE(hasRedundantPaths<Path<TagA, TagB>, Path<TagA, TagC>>());
        // Single path
        STATIC_CHECK_FALSE(hasRedundantPaths<Path<TagA>>());
    }

    SECTION("Redundancy Detected")
    {
        // TagA is ancestor of TagA/TagB
        STATIC_CHECK(hasRedundantPaths<Path<TagA>, Path<TagA, TagB>>());

        // Order independent
        STATIC_CHECK(hasRedundantPaths<Path<TagA, TagB>, Path<TagA>>());

        // Mixed Types (Tag vs Path)
        STATIC_CHECK(hasRedundantPaths<TagA, Path<TagA, TagB>>());
    }
}
