// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Record.hpp"
#include "llamaLite/Set.hpp"
#include "llamaLite/tag/TagPath.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

// A record-relative leaf-set algebra.
//
// A Record is a finite prefix-closed set of paths with typed leaves. Interpreted against a
// Record R, a set of accesses S (tags or paths) denotes the set of *leaves* it selects:
//
//     [[S]] = { leaf L of R : some p in S is an ancestor-or-equal of L }
//
// e.g. against a record with pos/x, pos/y, pos/z the access {pos} and {pos/x, pos/y, pos/z}
// denote the same leaves. The semantic domain is the powerset boolean algebra over leaves(R);
// leaves are the atoms, so union/intersection/difference/complement are all exact there.
//
// Raw Set operators compute the boolean algebra of *type identity*, which is the wrong quotient
// for paths (Set<pos> & Set<pos/x> is empty type-wise, but semantically {pos/x}). The fix is
// normalization: detail::leaf_set_t maps any access set to its canonical form -- the selected
// leaf paths in record declaration order. Canonical forms are both unique and deterministically
// ordered, so equal selections are the *same type* and Set's existing | & - ^ == <= are then
// exactly the right operations. Exclusion-style access is just the complement.
//
// AccessSet<R, Leaves> packages that canonical form together with its owning Record, so the record-relative meaning
// travels with the value. Its | & - ^ ~ == <= >= therefore mean the semantic operations directly. An AccessSet is a
// distinct type from Set and is not implicitly convertible to one. getSelectedLeaves(AccessSet) returns the raw Set
// explicitly. IsRawAccessSet names the structural Set form, IsAccessSet the record-bound wrapper.

namespace llama_lite
{
    namespace detail
    {
        template<typename S>
        inline constexpr bool is_access_set_v = false;

        template<IsRecordAccess... Ts>
        inline constexpr bool is_access_set_v<Set<Ts...>> = true;
    } // namespace detail

    /// A raw Set whose elements are all record accesses (tags or TagPaths). This recognizes
    /// *syntax* only: it names no record, validates no path and carries no record-aware operators.
    /// Use IsAccessSet for the record-bound wrapper that does all three.
    template<typename S>
    concept IsRawAccessSet = detail::is_access_set_v<S>;

    /// Structural raw access Set for a pack of record accesses: each tag is lifted to a
    /// single-tag path, so a pack that mixes tags and TagPaths forms one Set type. This preserves
    /// the pack verbatim, including duplicates -- and duplicate accesses are a hard error, because
    /// Set requires unique elements. Use deduped_access_set_t when the pack may repeat an access.
    template<IsRecordAccess... RAs>
    using raw_access_set_t = Set<to_path_t<RAs>...>;

    namespace detail
    {
        template<typename LeafTuple>
        struct TupleToSet;

        template<typename... Ls>
        struct TupleToSet<Tuple<Ls...>>
        {
            using type = Set<Ls...>;
        };

        // True if some access in S is an ancestor-or-equal of leaf path L, i.e. S selects L.
        template<IsTagPath L, typename S>
        struct SelectsLeaf;

        template<IsTagPath L, IsRecordAccess... Ss>
        struct SelectsLeaf<L, Set<Ss...>>
        {
            static constexpr bool value = (isAncestorOf(Ss{}, L{}) || ... || false);
        };

        // Fold the leaf paths Ls... into Acc, keeping those whose selected-by-S status matches
        // KeepIfSelected. Declaration order is preserved, so the result is canonical.
        template<typename S, bool KeepIfSelected, typename Acc, typename... Ls>
        struct FilterBySelection
        {
            using type = Acc;
        };

        template<typename S, bool KeepIfSelected, typename... AccTs, typename L0, typename... Rest>
        struct FilterBySelection<S, KeepIfSelected, Set<AccTs...>, L0, Rest...>
        {
            using NextAcc
                = std::conditional_t<SelectsLeaf<L0, S>::value == KeepIfSelected, Set<AccTs..., L0>, Set<AccTs...>>;
            using type = typename FilterBySelection<S, KeepIfSelected, NextAcc, Rest...>::type;
        };

        template<IsRecord R, typename S, bool KeepIfSelected>
        struct MakeLeafSet
        {
            template<typename LeafTuple>
            struct Build;

            template<typename... Ls>
            struct Build<Tuple<Ls...>>
            {
                using type = typename FilterBySelection<S, KeepIfSelected, Set<>, Ls...>::type;
            };

            using type = typename Build<typename GetLeafPaths<R>::type>::type;
        };

        // All accesses in S are paths that exist in R.
        template<IsRecord R, typename S>
        struct AllPathsInRecord;

        template<IsRecord R, IsRecordAccess... Ss>
        struct AllPathsInRecord<R, Set<Ss...>>
        {
            static constexpr bool value = (R::hasPath(Ss{}) && ... && true);
        };

        template<typename R, typename S>
        concept ValidAccessSetFor = IsRecord<R> && AllPathsInRecord<R, S>::value;

        /// All leaf paths of R as a Set, in declaration order.
        template<IsRecord R>
        using record_leaf_set_t = typename TupleToSet<typename GetLeafPaths<R>::type>::type;

        /// Canonical form of the access set S against R: the leaves of R that S selects, in
        /// declaration order. Equal selections normalize to the same type.
        template<IsRecord R, IsRawAccessSet S>
        requires ValidAccessSetFor<R, S>
        using leaf_set_t = typename MakeLeafSet<R, S, true>::type;

        /// Exclusion semantics = complement: the leaves of R that S does NOT select.
        template<IsRecord R, IsRawAccessSet S>
        requires ValidAccessSetFor<R, S>
        using leaf_set_complement_t = typename MakeLeafSet<R, S, false>::type;

        /// Leaves is a valid access set for R and already equals its own canonical leaf set.
        /// Canonical form is a fixed point of leaf_set_t, so this is exactly "declaration order,
        /// no redundant ancestors, no duplicates" and makes equal meaning the same type.
        template<typename R, typename Leaves>
        concept CanonicalLeafSetFor = IsRecord<R> && IsRawAccessSet<Leaves> && ValidAccessSetFor<R, Leaves>
                                      && std::same_as<Leaves, leaf_set_t<R, Leaves>>;

        /// Like raw_access_set_t, but each access is inserted at most once, so a pack that repeats
        /// an access (e.g. {pos, pos}) collapses instead of tripping Set's uniqueness static_assert.
        /// Order is preserved; this deduplicates only -- subtree/closure canonicalization is
        /// leaf_set_t's job. Use for caller-supplied packs whose accesses are not known unique.
        template<IsRecordAccess... RAs>
        using deduped_access_set_t = std::remove_cvref_t<decltype(makeSet(to_path_t<RAs>{}...))>;
    } // namespace detail

    /**
     * A canonical, record-bound access set: a zero-overhead token denoting exactly @p Leaves
     * within @p R. @p Leaves must be the canonical leaf set of R, so two sets denote the same
     * leaves exactly when they are the same type; construct via makeAccessSet()/access_set_t
     * rather than spelling it out. Unlike raw Set, | & - ^ ~ == <= >= here mean the record-aware
     * (closure-aware) operations, because both operands share R and are already canonical.
     */
    template<IsRecord R, IsRawAccessSet Leaves = Set<>>
    requires detail::CanonicalLeafSetFor<R, Leaves>
    struct AccessSet
    {
        using record_type = R;
        /// The raw canonical leaf Set this access set denotes.
        using leaf_set = Leaves;

        static constexpr std::size_t size = Leaves::size;
        static constexpr bool empty = Leaves::empty;
    };

    namespace detail
    {
        template<typename S>
        inline constexpr bool is_record_bound_access_set_v = false;

        template<IsRecord R, IsRawAccessSet L>
        inline constexpr bool is_record_bound_access_set_v<AccessSet<R, L>> = true;
    } // namespace detail

    /// A record-bound, canonical access set: the semantic object users manipulate. Recognizes
    /// only AccessSet<R, Leaves> specializations, so IsAccessSet<AccessSet<R, L>> is true while
    /// IsAccessSet<raw Set<...>> is false. Use IsRawAccessSet for the structural Set form.
    template<typename S>
    concept IsAccessSet = detail::is_record_bound_access_set_v<S>;

    /// Canonical access set of @p S against @p R. Leading R binds it to its record.
    template<IsRecord R, IsRawAccessSet S>
    requires detail::ValidAccessSetFor<R, S>
    using access_set_t = AccessSet<R, detail::leaf_set_t<R, S>>;

    /**
     * Build a canonical AccessSet against @p R from ordinary accesses (tags or paths).
     * Redundant ancestors, overlapping accesses and repeated tags are all normalized away, so
     * makeAccessSet<R>(posO) and makeAccessSet<R>(posO, posO / xO) are the same type.
     * An empty pack selects nothing; makeAccessSet<R>(TagPath<>{}) selects the whole record.
     */
    template<IsRecord R, IsRecordAccess... RAs>
    requires detail::ValidAccessSetFor<R, detail::deduped_access_set_t<RAs...>>
    [[nodiscard]] consteval auto makeAccessSet(RAs... /*accesses*/)
    {
        return access_set_t<R, detail::deduped_access_set_t<RAs...>>{};
    }

    /// Build a canonical AccessSet against @p R from an already-assembled access Set.
    template<IsRecord R, IsRawAccessSet S>
    requires detail::ValidAccessSetFor<R, S>
    [[nodiscard]] consteval auto makeAccessSet(S /*accesses*/)
    {
        return access_set_t<R, S>{};
    }

    /// Union: leaves selected by either selection.
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval auto operator|(AccessSet<R, A>, AccessSet<R, B>)
    {
        return access_set_t<R, decltype(A{} | B{})>{};
    }

    /// Intersection: leaves selected by both. Closure-aware, so {pos} & {pos/x} is {pos/x}.
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval auto operator&(AccessSet<R, A>, AccessSet<R, B>)
    {
        return access_set_t<R, decltype(A{} & B{})>{};
    }

    /// Difference: leaves selected by the left selection but not the right.
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval auto operator-(AccessSet<R, A>, AccessSet<R, B>)
    {
        return access_set_t<R, decltype(A{} - B{})>{};
    }

    /// Symmetric difference: leaves selected by exactly one of the two selections.
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval auto operator^(AccessSet<R, A>, AccessSet<R, B>)
    {
        return access_set_t<R, decltype(A{} ^ B{})>{};
    }

    /// Complement: the leaves of R the selection does not select.
    template<IsRecord R, IsRawAccessSet A>
    requires detail::CanonicalLeafSetFor<R, A>
    [[nodiscard]] consteval auto operator~(AccessSet<R, A>)
    {
        return access_set_t<R, detail::leaf_set_complement_t<R, A>>{};
    }

    /// Equal selections are the same type, so comparing canonical sets compares meaning.
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval bool operator==(AccessSet<R, A>, AccessSet<R, B>)
    {
        return A{} == B{};
    }

    /// [[left]] is a subset of [[right]] -- the closure-aware "contains".
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval bool operator<=(AccessSet<R, A>, AccessSet<R, B>)
    {
        return A{} <= B{};
    }

    /// [[left]] is a superset of [[right]].
    template<IsRecord R, IsRawAccessSet A, IsRawAccessSet B>
    requires detail::CanonicalLeafSetFor<R, A> && detail::CanonicalLeafSetFor<R, B>
    [[nodiscard]] consteval bool operator>=(AccessSet<R, A>, AccessSet<R, B>)
    {
        return A{} >= B{};
    }

    /// Explicit bridge: the canonical leaf Set a AccessSet denotes, for existing Set-based APIs.
    template<IsRecord R, IsRawAccessSet A>
    requires detail::CanonicalLeafSetFor<R, A>
    [[nodiscard]] constexpr auto getSelectedLeaves(AccessSet<R, A>) noexcept
    {
        return A{};
    }

} // namespace llama_lite
