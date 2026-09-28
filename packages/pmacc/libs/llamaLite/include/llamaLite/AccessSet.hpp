// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Record.hpp"
#include "llamaLite/Set.hpp"
#include "llamaLite/tag/TagPath.hpp"

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
// normalization: leaf_set_t maps any access set to its canonical form -- the selected leaf
// paths in record declaration order. Canonical forms are both unique and deterministically
// ordered, so equal selections are the *same type* and Set's existing | & - ^ == <= are then
// exactly the right operations. Exclusion-style access is just the complement.

namespace llama_lite
{
    namespace detail
    {
        template<typename S>
        inline constexpr bool is_access_set_v = false;

        template<IsRecordAccess... Ts>
        inline constexpr bool is_access_set_v<Set<Ts...>> = true;
    } // namespace detail

    /// A Set whose elements are all record accesses (tags or TagPaths).
    template<typename S>
    concept IsAccessSet = detail::is_access_set_v<S>;

    /// The normalized access Set for a pack of record accesses: each tag is lifted to a
    /// single-tag path, so a pack that mixes tags and TagPaths canonicalizes to one Set type.
    template<IsRecordAccess... RAs>
    using access_set_t = Set<to_path_t<RAs>...>;

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
            static constexpr bool value = (to_path_t<Ss>::template isAncestorOf<L>() || ... || false);
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
    } // namespace detail

    /// All leaf paths of R as a Set, in declaration order.
    template<IsRecord R>
    using record_leaf_set_t = typename detail::TupleToSet<typename GetLeafPaths<R>::type>::type;

    /// Canonical form of the access set S against R: the leaves of R that S selects, in
    /// declaration order. Equal selections normalize to the same type.
    template<IsRecord R, IsAccessSet S>
    requires detail::ValidAccessSetFor<R, S>
    using leaf_set_t = typename detail::MakeLeafSet<R, S, true>::type;

    /// Exclusion semantics = complement: the leaves of R that S does NOT select.
    template<IsRecord R, IsAccessSet S>
    requires detail::ValidAccessSetFor<R, S>
    using leaf_set_complement_t = typename detail::MakeLeafSet<R, S, false>::type;

    /// [[q]] is a subset of [[s]] against R -- the closure-aware "contains".
    template<IsRecord R, IsAccessSet S, IsAccessSet Q>
    requires detail::ValidAccessSetFor<R, S> && detail::ValidAccessSetFor<R, Q>
    [[nodiscard]] consteval bool selects(R /*record*/, S /*s*/, Q /*q*/)
    {
        return leaf_set_t<R, Q>{} <= leaf_set_t<R, S>{};
    }

    /// Concept form of selects(): [[Q]] is a subset of [[S]] against R. Use this in
    /// requires-clauses where an element-wise IsInSet check would be reached for -- unlike
    /// IsInSet it is closure-aware, so a set selecting {pos} contains the set selecting {pos/x}.
    /// Short-circuits on validity before comparing leaves, so it is SFINAE-friendly for accesses
    /// that do not name paths in R.
    template<typename R, typename S, typename Q>
    concept Selects = IsRecord<R> && IsAccessSet<S> && IsAccessSet<Q> && detail::ValidAccessSetFor<R, S>
                      && detail::ValidAccessSetFor<R, Q> && (leaf_set_t<R, Q>{} <= leaf_set_t<R, S>{});

    /// [[q]] and [[s]] share at least one leaf against R.
    template<IsRecord R, IsAccessSet S, IsAccessSet Q>
    requires detail::ValidAccessSetFor<R, S> && detail::ValidAccessSetFor<R, Q>
    [[nodiscard]] consteval bool intersects(R /*record*/, S /*s*/, Q /*q*/)
    {
        return !(leaf_set_t<R, S>{} & leaf_set_t<R, Q>{}).empty;
    }

    /// [[a]] == [[b]] against R -- the two access sets select the same leaves.
    template<IsRecord R, IsAccessSet A, IsAccessSet B>
    requires detail::ValidAccessSetFor<R, A> && detail::ValidAccessSetFor<R, B>
    [[nodiscard]] consteval bool sameSelection(R /*record*/, A /*a*/, B /*b*/)
    {
        return leaf_set_t<R, A>{} == leaf_set_t<R, B>{};
    }

} // namespace llama_lite
