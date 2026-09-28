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
    using namespace llama_lite;

    // Tags for a small physics-flavoured record.
    struct pos_t : TagBase
    {
    };

    struct vel_t : TagBase
    {
    };

    struct mass_t : TagBase
    {
    };

    struct x_t : TagBase
    {
    };

    struct y_t : TagBase
    {
    };

    struct z_t : TagBase
    {
    };

    // Depth-2 record: leaves (declaration order) pos/x pos/y pos/z vel/x vel/y vel/z mass.
    using Vec3 = Record<Field<x_t, float>, Field<y_t, float>, Field<z_t, float>>;
    using R = Record<Field<pos_t, Vec3>, Field<vel_t, Vec3>, Field<mass_t, float>>;

    // Convenience path aliases.
    using PosX = TagPath<pos_t, x_t>;
    using PosY = TagPath<pos_t, y_t>;
    using PosZ = TagPath<pos_t, z_t>;
    using VelX = TagPath<vel_t, x_t>;

    // Tags for a depth-3 record to exercise recursion through GetLeafPaths.
    struct a_t : TagBase
    {
    };

    struct b_t : TagBase
    {
    };

    using Inner = Record<Field<a_t, float>, Field<b_t, float>>;
    using Mid = Record<Field<x_t, Inner>, Field<y_t, float>>;
    // Leaves (declaration order): pos/x/a pos/x/b pos/y mass.
    using Deep = Record<Field<pos_t, Mid>, Field<mass_t, float>>;

    using DeepPosXA = TagPath<pos_t, x_t, a_t>;
    using DeepPosXB = TagPath<pos_t, x_t, b_t>;
    using DeepPosY = TagPath<pos_t, y_t>;
} // namespace

TEST_CASE("AccessSet concept and record leaf set", "[AccessSet][Meta]")
{
    STATIC_CHECK(IsAccessSet<Set<pos_t>>);
    STATIC_CHECK(IsAccessSet<Set<PosX, vel_t>>);
    STATIC_CHECK(IsAccessSet<Set<>>);
    STATIC_CHECK_FALSE(IsAccessSet<Set<int>>);
    STATIC_CHECK_FALSE(IsAccessSet<int>);

    // record_leaf_set_t enumerates every leaf in declaration order.
    STATIC_CHECK(record_leaf_set_t<R>::size == 7);
    STATIC_CHECK(
        std::is_same_v<
            record_leaf_set_t<R>,
            Set<PosX, PosY, PosZ, VelX, TagPath<vel_t, y_t>, TagPath<vel_t, z_t>, TagPath<mass_t>>>);
}

TEST_CASE("AccessSet canonicalization", "[AccessSet][Meta]")
{
    // {pos} and {pos/x, pos/y, pos/z} select the same leaves, so they normalize identically.
    STATIC_CHECK(std::is_same_v<leaf_set_t<R, Set<pos_t>>, leaf_set_t<R, Set<PosX, PosY, PosZ>>>);

    // Canonical form is exactly the selected leaf paths.
    STATIC_CHECK(std::is_same_v<leaf_set_t<R, Set<pos_t>>, Set<PosX, PosY, PosZ>>);

    // A redundant input {pos, pos/x} canonicalizes the same as {pos}.
    STATIC_CHECK(std::is_same_v<leaf_set_t<R, Set<pos_t, PosX>>, leaf_set_t<R, Set<pos_t>>>);

    // A leaf access normalizes to itself.
    STATIC_CHECK(std::is_same_v<leaf_set_t<R, Set<PosX>>, Set<PosX>>);
}

TEST_CASE("AccessSet semantic algebra", "[AccessSet][Operator]")
{
    // Semantic intersection via canonicalize-then-&: {pos} & {pos/x} == {pos/x}.
    constexpr auto inter = leaf_set_t<R, Set<pos_t>>{} & leaf_set_t<R, Set<PosX>>{};
    STATIC_CHECK(inter == Set<PosX>{});

    // Disjoint subtrees intersect to nothing.
    constexpr auto disjoint = leaf_set_t<R, Set<pos_t>>{} & leaf_set_t<R, Set<vel_t>>{};
    STATIC_CHECK(disjoint.empty);

    // Union of the two halves plus mass is the whole record.
    constexpr auto whole = leaf_set_t<R, Set<pos_t>>{} | leaf_set_t<R, Set<vel_t, mass_t>>{};
    STATIC_CHECK(whole == record_leaf_set_t<R>{});
}

TEST_CASE("AccessSet complement", "[AccessSet][Operator]")
{
    // Complement of {pos} is the rest of the record.
    STATIC_CHECK(std::is_same_v<leaf_set_complement_t<R, Set<pos_t>>, leaf_set_t<R, Set<vel_t, mass_t>>>);

    // Complement round-trip: complement of complement is the identity selection.
    STATIC_CHECK(
        std::is_same_v<leaf_set_complement_t<R, leaf_set_complement_t<R, Set<pos_t>>>, leaf_set_t<R, Set<pos_t>>>);

    // Edge cases: empty set selects nothing, whole-record set selects everything.
    STATIC_CHECK(leaf_set_t<R, Set<>>::empty);
    STATIC_CHECK(std::is_same_v<leaf_set_complement_t<R, Set<>>, record_leaf_set_t<R>>);
    STATIC_CHECK(leaf_set_complement_t<R, record_leaf_set_t<R>>::empty);
}

TEST_CASE("AccessSet predicates", "[AccessSet][Operator]")
{
    // selects is closure-aware: {pos} selects the query {pos/x} ...
    STATIC_CHECK(selects(R{}, Set<pos_t>{}, Set<PosX>{}));
    // ... but the finer {pos/x} does not select the coarser {pos}.
    STATIC_CHECK_FALSE(selects(R{}, Set<PosX>{}, Set<pos_t>{}));

    // A plain Set stays exact: {pos} does not literally contain the path pos/x.
    STATIC_CHECK_FALSE(Set<pos_t>::contains(PosX{}));

    // intersects overlaps closure-aware.
    STATIC_CHECK(intersects(R{}, Set<pos_t>{}, Set<PosX>{}));
    STATIC_CHECK_FALSE(intersects(R{}, Set<pos_t>{}, Set<vel_t>{}));

    // sameSelection across different representatives of the same leaf set.
    STATIC_CHECK(sameSelection(R{}, Set<pos_t>{}, Set<PosX, PosY, PosZ>{}));
    STATIC_CHECK(sameSelection(R{}, Set<pos_t, PosX>{}, Set<pos_t>{}));
    STATIC_CHECK_FALSE(sameSelection(R{}, Set<pos_t>{}, Set<PosX>{}));
}

TEST_CASE("AccessSet nested record recursion", "[AccessSet][Meta]")
{
    // GetLeafPaths recurses to depth 3.
    STATIC_CHECK(record_leaf_set_t<Deep>::size == 4);
    STATIC_CHECK(std::is_same_v<record_leaf_set_t<Deep>, Set<DeepPosXA, DeepPosXB, DeepPosY, TagPath<mass_t>>>);

    // Selecting pos reaches every leaf beneath it, including the depth-3 ones.
    STATIC_CHECK(std::is_same_v<leaf_set_t<Deep, Set<pos_t>>, Set<DeepPosXA, DeepPosXB, DeepPosY>>);
    STATIC_CHECK(leaf_set_t<Deep, Set<pos_t>>::size == 3);

    // Selecting the intermediate pos/x reaches only its two leaves.
    STATIC_CHECK(std::is_same_v<leaf_set_t<Deep, Set<TagPath<pos_t, x_t>>>, Set<DeepPosXA, DeepPosXB>>);

    STATIC_CHECK(selects(Deep{}, Set<pos_t>{}, Set<DeepPosXA>{}));
    STATIC_CHECK(selects(Deep{}, Set<TagPath<pos_t, x_t>>{}, Set<DeepPosXB>{}));
    STATIC_CHECK_FALSE(selects(Deep{}, Set<TagPath<pos_t, x_t>>{}, Set<DeepPosY>{}));
}
