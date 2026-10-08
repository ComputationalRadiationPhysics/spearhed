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

    // A tag that exists in no record, for rejection tests.
    struct missing_t : TagBase
    {
    };

    using EmptyRecord = Record<>;
    using VelY = TagPath<vel_t, y_t>;
    using VelZ = TagPath<vel_t, z_t>;
    using Mass = TagPath<mass_t>;

    // Record-bound selection aliases: canonical leaves plus the owning record.
    using SetAll = AccessSet<R, detail::record_leaf_set_t<R>>;
    using SetNone = AccessSet<R, Set<>>;
    using SetPos = access_set_t<R, Set<pos_t>>;
    using SetPosX = access_set_t<R, Set<PosX>>;
    using SetVelX = access_set_t<R, Set<VelX>>;

    // A second record sharing the same tags but a different field set; ops must not mix records.
    using Other = Record<Field<pos_t, Vec3>, Field<mass_t, float>>;
    using OtherAccessPos = access_set_t<Other, Set<pos_t>>;

    // Concepts used to probe construction and operator availability without hard errors.
    template<typename... Args>
    concept CanMakeAccessSet = requires(Args... args) { makeAccessSet<R>(args...); };

    template<typename Rr, typename Leaves>
    concept CanNameAccessSet = requires { typename AccessSet<Rr, Leaves>; };

    template<typename A, typename B>
    concept CanCombine = requires(A a, B b) { a & b; };

    template<typename A, typename B>
    concept CanCompare = requires(A a, B b) { a == b; };
} // namespace

TEST_CASE("AccessSet concept and record leaf set", "[AccessSet][Meta]")
{
    STATIC_CHECK(IsRawAccessSet<Set<pos_t>>);
    STATIC_CHECK(IsRawAccessSet<Set<PosX, vel_t>>);
    STATIC_CHECK(IsRawAccessSet<Set<>>);
    STATIC_CHECK_FALSE(IsRawAccessSet<Set<int>>);
    STATIC_CHECK_FALSE(IsRawAccessSet<int>);

    // IsAccessSet names the record-bound wrapper, not the raw Set: the two are disjoint.
    STATIC_CHECK(IsAccessSet<AccessSet<R, detail::record_leaf_set_t<R>>>);
    STATIC_CHECK(IsAccessSet<access_set_t<R, Set<pos_t>>>);
    STATIC_CHECK_FALSE(IsAccessSet<Set<pos_t>>);
    STATIC_CHECK_FALSE(IsAccessSet<Set<>>);
    STATIC_CHECK_FALSE(IsAccessSet<int>);

    // raw_access_set_t preserves the pack verbatim; deduped_access_set_t inserts each access once.
    STATIC_CHECK(std::is_same_v<raw_access_set_t<pos_t>, Set<TagPath<pos_t>>>);
    STATIC_CHECK(std::is_same_v<raw_access_set_t<pos_t, PosX>, Set<TagPath<pos_t>, PosX>>);
    STATIC_CHECK(std::is_same_v<detail::deduped_access_set_t<pos_t>, Set<TagPath<pos_t>>>);
    STATIC_CHECK(std::is_same_v<detail::deduped_access_set_t<pos_t, PosX>, Set<TagPath<pos_t>, PosX>>);
    // Deduplication is the distinguishing behavior: a repeated access collapses to one element,
    // whereas the same pack through raw_access_set_t would trip Set's uniqueness static_assert.
    STATIC_CHECK(std::is_same_v<detail::deduped_access_set_t<pos_t, pos_t>, Set<TagPath<pos_t>>>);
    STATIC_CHECK(detail::deduped_access_set_t<pos_t, pos_t>::size == 1);
    // Order is preserved either way; neither does subtree/closure canonicalization.
    STATIC_CHECK(std::is_same_v<detail::deduped_access_set_t<PosX, PosY>, Set<PosX, PosY>>);
    // The bound wrapper exposes its raw leaves (but is not itself a raw access set).
    STATIC_CHECK(IsRawAccessSet<AccessSet<R, detail::record_leaf_set_t<R>>::leaf_set>);
    STATIC_CHECK_FALSE(IsRawAccessSet<AccessSet<R, detail::record_leaf_set_t<R>>>);

    // record_leaf_set_t enumerates every leaf in declaration order.
    STATIC_CHECK(detail::record_leaf_set_t<R>::size == 7);
    STATIC_CHECK(
        std::is_same_v<
            detail::record_leaf_set_t<R>,
            Set<PosX, PosY, PosZ, VelX, TagPath<vel_t, y_t>, TagPath<vel_t, z_t>, TagPath<mass_t>>>);
}

TEST_CASE("AccessSet canonicalization", "[AccessSet][Meta]")
{
    // {pos} and {pos/x, pos/y, pos/z} select the same leaves, so they normalize identically.
    STATIC_CHECK(std::is_same_v<detail::leaf_set_t<R, Set<pos_t>>, detail::leaf_set_t<R, Set<PosX, PosY, PosZ>>>);

    // Canonical form is exactly the selected leaf paths.
    STATIC_CHECK(std::is_same_v<detail::leaf_set_t<R, Set<pos_t>>, Set<PosX, PosY, PosZ>>);

    // A redundant input {pos, pos/x} canonicalizes the same as {pos}.
    STATIC_CHECK(std::is_same_v<detail::leaf_set_t<R, Set<pos_t, PosX>>, detail::leaf_set_t<R, Set<pos_t>>>);

    // A leaf access normalizes to itself.
    STATIC_CHECK(std::is_same_v<detail::leaf_set_t<R, Set<PosX>>, Set<PosX>>);
}

TEST_CASE("AccessSet semantic algebra", "[AccessSet][Operator]")
{
    // Semantic intersection via canonicalize-then-&: {pos} & {pos/x} == {pos/x}.
    constexpr auto inter = detail::leaf_set_t<R, Set<pos_t>>{} & detail::leaf_set_t<R, Set<PosX>>{};
    STATIC_CHECK(inter == Set<PosX>{});

    // Disjoint subtrees intersect to nothing.
    constexpr auto disjoint = detail::leaf_set_t<R, Set<pos_t>>{} & detail::leaf_set_t<R, Set<vel_t>>{};
    STATIC_CHECK(disjoint.empty);

    // Union of the two halves plus mass is the whole record.
    constexpr auto whole = detail::leaf_set_t<R, Set<pos_t>>{} | detail::leaf_set_t<R, Set<vel_t, mass_t>>{};
    STATIC_CHECK(whole == detail::record_leaf_set_t<R>{});
}

TEST_CASE("AccessSet complement", "[AccessSet][Operator]")
{
    // Complement of {pos} is the rest of the record.
    STATIC_CHECK(
        std::is_same_v<detail::leaf_set_complement_t<R, Set<pos_t>>, detail::leaf_set_t<R, Set<vel_t, mass_t>>>);

    // Complement round-trip: complement of complement is the identity selection.
    STATIC_CHECK(
        std::is_same_v<
            detail::leaf_set_complement_t<R, detail::leaf_set_complement_t<R, Set<pos_t>>>,
            detail::leaf_set_t<R, Set<pos_t>>>);

    // Edge cases: empty set selects nothing, whole-record set selects everything.
    STATIC_CHECK(detail::leaf_set_t<R, Set<>>::empty);
    STATIC_CHECK(std::is_same_v<detail::leaf_set_complement_t<R, Set<>>, detail::record_leaf_set_t<R>>);
    STATIC_CHECK(detail::leaf_set_complement_t<R, detail::record_leaf_set_t<R>>::empty);
}

TEST_CASE("AccessSet canonical construction", "[AccessSet][AccessSet]")
{
    // Ordinary accesses, redundant ancestors, aliases and duplicate tags all canonicalize.
    constexpr auto fromTag = makeAccessSet<R>(pos_t{});
    constexpr auto fromLeaves = makeAccessSet<R>(PosX{}, PosY{}, PosZ{});
    constexpr auto fromRedundant = makeAccessSet<R>(pos_t{}, PosX{});
    constexpr auto fromDuplicates = makeAccessSet<R>(pos_t{}, pos_t{});

    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(fromTag)>, SetPos>);
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(fromLeaves)>, SetPos>);
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(fromRedundant)>, SetPos>);
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(fromDuplicates)>, SetPos>);

    // A tag and its single-element path are the same access.
    constexpr auto fromPath = makeAccessSet<R>(TagPath<pos_t>{});
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(fromTag)>, std::remove_cvref_t<decltype(fromPath)>>);
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(makeAccessSet<R>(pos_t{}, TagPath<pos_t>{}))>, SetPos>);

    // A Set-based factory canonicalizes the same way.
    constexpr auto fromSet = makeAccessSet<R>(Set<PosZ, PosX, PosY>{});
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(fromSet)>, SetPos>);

    // Size and emptiness report the canonical leaf count.
    STATIC_CHECK(SetPos::size == 3);
    STATIC_CHECK_FALSE(SetPos::empty);
    STATIC_CHECK(SetNone::size == 0);
    STATIC_CHECK(SetNone::empty);

    // Empty pack selects nothing; the root path selects the whole record.
    constexpr auto none = makeAccessSet<R>();
    constexpr auto all = makeAccessSet<R>(TagPath<>{});
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(none)>, SetNone>);
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(all)>, SetAll>);
    STATIC_CHECK(all == makeAccessSet<R>(pos_t{}, vel_t{}, mass_t{}));

    // makeAccessSet is idempotent over a selection's own leaves.
    STATIC_CHECK(makeAccessSet<R>(PosX{}, PosY{}, PosZ{}) == makeAccessSet<R>(pos_t{}));

    // getSelectedLeaves is the explicit bridge back to the canonical Set.
    STATIC_CHECK(std::is_same_v<decltype(getSelectedLeaves(fromTag)), Set<PosX, PosY, PosZ>>);
    STATIC_CHECK(getSelectedLeaves(all) == detail::record_leaf_set_t<R>{});
    STATIC_CHECK(getSelectedLeaves(none).empty);

    // An empty record admits only the empty selection.
    STATIC_CHECK(std::is_same_v<decltype(makeAccessSet<EmptyRecord>()), AccessSet<EmptyRecord, Set<>>>);
    STATIC_CHECK(makeAccessSet<EmptyRecord>(TagPath<>{}) == makeAccessSet<EmptyRecord>());
}

TEST_CASE("AccessSet record-aware algebra", "[AccessSet][AccessSet]")
{
    constexpr auto pos = makeAccessSet<R>(pos_t{});
    constexpr auto posX = makeAccessSet<R>(PosX{});
    constexpr auto posY = makeAccessSet<R>(PosY{});
    constexpr auto vel = makeAccessSet<R>(vel_t{});

    // Closure-aware intersection: {pos} & {pos/x} == {pos/x}, unlike raw Set.
    constexpr auto inter = pos & posX;
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(inter)>, SetPosX>);
    STATIC_CHECK(inter == posX);

    // Disjoint subtrees intersect to nothing, independent of operand order.
    STATIC_CHECK((pos & vel).empty);
    STATIC_CHECK((vel & pos).empty);
    STATIC_CHECK(std::is_same_v<decltype(vel & pos), SetNone>);

    // Difference removes exactly the shared leaves.
    constexpr auto leftOver = pos - posX;
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(leftOver)>, access_set_t<R, Set<PosY, PosZ>>>);
    STATIC_CHECK((leftOver | posX) == pos);
    STATIC_CHECK((pos - pos).empty);

    // Union is order-insensitive yet still canonical (declaration order, not operand order).
    constexpr auto unionSel = posX | posY;
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(unionSel)>, access_set_t<R, Set<PosX, PosY>>>);
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(posY | posX)>, std::remove_cvref_t<decltype(unionSel)>>);
    STATIC_CHECK((posX | pos) == pos);

    // Symmetric difference keeps the non-shared leaves in declaration order.
    constexpr auto sym = posX ^ posY;
    STATIC_CHECK(std::is_same_v<std::remove_cvref_t<decltype(sym)>, access_set_t<R, Set<PosX, PosY>>>);
    STATIC_CHECK(std::is_same_v<decltype(posY ^ posX), std::remove_cvref_t<decltype(sym)>>);
    STATIC_CHECK((posX ^ posX).empty);

    // Complement is the rest of the record; it round-trips.
    constexpr auto complement = ~pos;
    STATIC_CHECK(
        std::is_same_v<std::remove_cvref_t<decltype(complement)>, access_set_t<R, Set<VelX, VelY, VelZ, Mass>>>);
    STATIC_CHECK((~complement) == pos);
    STATIC_CHECK((~SetNone{}) == SetAll{});
    STATIC_CHECK((~SetAll{}).empty);

    // De Morgan: complement exchanges union and intersection.
    STATIC_CHECK((~(pos | vel) == (~pos & ~vel)));
    STATIC_CHECK((~(pos & vel) == (~pos | ~vel)));
    // Complementing one operand complements the symmetric difference.
    STATIC_CHECK((~(pos ^ vel) == (~pos ^ vel)));

    // Subset/superset and equality are meaning-based; != is synthesized from ==.
    STATIC_CHECK(posX <= pos);
    STATIC_CHECK_FALSE(pos <= posX);
    STATIC_CHECK(pos >= posX);
    STATIC_CHECK(posX != pos);
    STATIC_CHECK(pos == makeAccessSet<R>(pos_t{}));
    STATIC_CHECK((posY <= (posX | posY)));

    // A leaf selection reaches depth-3 leaves too.
    STATIC_CHECK(
        (makeAccessSet<Deep>(pos_t{}) & makeAccessSet<Deep>(DeepPosXA{})) == makeAccessSet<Deep>(DeepPosXA{}));
}

TEST_CASE("AccessSet rejects invalid construction and cross-record use", "[AccessSet][AccessSet]")
{
    // Constructing from an unknown path or a non-access argument is ill-formed.
    STATIC_CHECK_FALSE(CanMakeAccessSet<missing_t>);
    STATIC_CHECK_FALSE(CanMakeAccessSet<int>);
    STATIC_CHECK_FALSE(CanMakeAccessSet<PosX, int>);
    STATIC_CHECK(CanMakeAccessSet<pos_t>);
    STATIC_CHECK(CanMakeAccessSet<PosX, PosY>);

    // The token itself admits only canonical leaf sets: declaration order, no ancestors.
    STATIC_CHECK(CanNameAccessSet<R, Set<PosX>>);
    STATIC_CHECK(CanNameAccessSet<R, Set<PosX, PosY>>);
    STATIC_CHECK_FALSE(CanNameAccessSet<R, Set<pos_t>>);
    STATIC_CHECK_FALSE(CanNameAccessSet<R, Set<PosY, PosX>>);
    STATIC_CHECK_FALSE(CanNameAccessSet<R, Set<missing_t>>);

    // Operators require the same record on both sides.
    STATIC_CHECK(CanCombine<SetPos, SetVelX>);
    STATIC_CHECK_FALSE(CanCombine<SetPos, OtherAccessPos>);
    STATIC_CHECK(CanCompare<SetPos, SetPos>);
    STATIC_CHECK_FALSE(CanCompare<SetPos, OtherAccessPos>);

    // Raw Set semantics stay type-identity based; AccessSet does not change them.
    STATIC_CHECK((Set<pos_t>{} & Set<PosX>{}).empty);
    STATIC_CHECK_FALSE(Set<pos_t>::contains(PosX{}));
}

TEST_CASE("AccessSet nested record recursion", "[AccessSet][Meta]")
{
    // GetLeafPaths recurses to depth 3.
    STATIC_CHECK(detail::record_leaf_set_t<Deep>::size == 4);
    STATIC_CHECK(
        std::is_same_v<detail::record_leaf_set_t<Deep>, Set<DeepPosXA, DeepPosXB, DeepPosY, TagPath<mass_t>>>);

    // Selecting pos reaches every leaf beneath it, including the depth-3 ones.
    STATIC_CHECK(std::is_same_v<detail::leaf_set_t<Deep, Set<pos_t>>, Set<DeepPosXA, DeepPosXB, DeepPosY>>);
    STATIC_CHECK(detail::leaf_set_t<Deep, Set<pos_t>>::size == 3);

    // Selecting the intermediate pos/x reaches only its two leaves.
    STATIC_CHECK(std::is_same_v<detail::leaf_set_t<Deep, Set<TagPath<pos_t, x_t>>>, Set<DeepPosXA, DeepPosXB>>);

    STATIC_CHECK(
        (makeAccessSet<Deep>(pos_t{}) & makeAccessSet<Deep>(DeepPosXA{})) == makeAccessSet<Deep>(DeepPosXA{}));
    STATIC_CHECK(
        (makeAccessSet<Deep>(TagPath<pos_t, x_t>{}) & makeAccessSet<Deep>(DeepPosXB{}))
        == makeAccessSet<Deep>(DeepPosXB{}));
    STATIC_CHECK((makeAccessSet<Deep>(TagPath<pos_t, x_t>{}) & makeAccessSet<Deep>(DeepPosY{})).empty);
}
