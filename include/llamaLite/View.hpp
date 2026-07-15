// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Record.hpp"
#include "llamaLite/AccessSet.hpp"
#include "llamaLite/Set.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"
#include "traits.hpp"

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace llama_lite
{
    // A View is parameterized on a storage and an access Set S -- the set of record accesses
    // (tags or TagPaths) it exposes over that storage. S is kept at the granularity the caller
    // named it (a node stays a node) so that composite-node handling via traits::AsType and
    // node-level indexing keep working; the closure-aware leaf semantics live in AccessSet and
    // are only reached for in the constraints (Selects) and in deep copies.
    //
    // The empty set S = Set<> is the "root" cursor: it denotes the whole record and can be
    // drilled into from the top. A size-1 set is a cursor at a single node/leaf. A size>1 set
    // is a selection of sibling accesses.
    //
    // TODO decouple view from storage.
    // TODO merge deepCopyTo and deepCopyFrom and clarify copy semantics. Do we copy intersections? should the user
    // check if they are subsets if they want a full copy? What if they want the sets to be equal? Should they check
    // this or should we?

    namespace detail
    {
        // The single element of a size-1 access Set.
        template<IsAccessSet S>
        struct SingleAccess;

        template<typename A>
        struct SingleAccess<Set<A>>
        {
            using type = A;
        };

        template<IsAccessSet S>
        using single_access_t = typename SingleAccess<S>::type;

        // All accesses named by S resolve to a field of the storage record.
        template<typename TStorage, typename S>
        concept ViewStorageFor = IsAccessSet<S> && requires { typename TStorage::record_type; }
                                 && ValidAccessSetFor<typename TStorage::record_type, S>;
    } // namespace detail

    /**
     * Resolves a single-access view (View or ViewIndexed) to its value.
     *
     * This is the one place the leaf/node distinction and the traits::AsType
     * mapping live; the views delegate their terminal-access resolution (drilling
     * to a leaf, get() and operator*) here so they carry no resolution logic of
     * their own. Given a view over exactly one record access it returns:
     *   - a composite node carrying a traits::AsType specialization:
     *       the AsType-constructed object, built from the view;
     *   - a leaf reached through an indexed view:
     *       a reference to the scalar element at the view's index;
     *   - a leaf reached through a non-indexed view:
     *       the std::span over the whole leaf column.
     *
     * A composite node without an AsType specialization has no value to resolve
     * to and is rejected at compile time.
     *
     * @tparam V a View or ViewIndexed whose access set names a single access
     * @param  view the view to resolve (cheap to copy: a storage pointer, plus
     *              an index for indexed views)
     */
    template<typename V>
    requires(V::access_set::size == 1)
    [[nodiscard]] static constexpr decltype(auto) resolve(V view)
    {
        using Record = typename V::record_type;
        using Access = decltype(view.getRecordAccess());
        using Field = typename Record::template field_for<Access>;

        if constexpr(traits::IsTraitSpecialized<traits::AsType, Field>::value)
        {
            return traits::AsType<Field>{}(view);
        }
        else
        {
            static_assert(
                Record::template isLeaf<Access>(),
                "resolve: access names a composite node without an AsType specialization; nothing to resolve to.");

            if constexpr(requires { view.idx; })
                return view.storage->template getLeaf<Access>()[view.idx];
            else
                return view.storage->template getLeaf<Access>();
        }
    }

    namespace detail
    {
        // Result of drilling a cursor into a single child access: if the child names a leaf
        // (a terminal access) it is resolved to its value -- an element reference for an indexed
        // view, the std::span over the column otherwise. A composite node (including one carrying
        // an AsType specialization) stays a cursor, so it can be drilled further or .get()'d.
        template<typename ChildView>
        [[nodiscard]] static constexpr decltype(auto) resolveIfLeaf(ChildView child)
        {
            using Record = typename ChildView::record_type;
            using Access = single_access_t<typename ChildView::access_set>;
            if constexpr(Record::template isLeaf<Access>())
                return resolve(child);
            else
                return child;
        }
    } // namespace detail


    template<typename TStorage, IsAccessSet S>
    requires detail::ViewStorageFor<TStorage, S>
    struct ViewIndexed;

    template<typename TStorage, IsAccessSet S>
    requires detail::ViewStorageFor<TStorage, S>
    struct View
    {
        using record_type = TStorage::record_type;
        using access_set = S;

        TStorage* storage;

        // Construct over storage with a given (or empty/root) access set.
        constexpr View(TStorage& storage_, S /*accessSet*/ = {}) noexcept : storage{&storage_}
        {
        }

        // Narrow from a parent view whose selection contains ours.
        template<IsAccessSet ParentS>
        constexpr View(View<TStorage, ParentS> view, S /*accessSet*/ = {}) noexcept
            requires Selects<record_type, ParentS, S>
            : storage{view.storage}
        {
        }

        [[nodiscard]] constexpr decltype(auto) operator[](uint32_t idx)
        {
            return ViewIndexed<TStorage, S>(*this, idx);
        }

        [[nodiscard]] constexpr decltype(auto) operator[](uint32_t idx) const
        {
            return ViewIndexed<TStorage, S>(*this, idx);
        }

        // Drill into the single current node. A terminal (leaf) child resolves to its
        // std::span over the column; a composite node stays a cursor.
        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) requires(S::size == 1)
        {
            using Path = append_t<detail::single_access_t<S>, RA>;
            return detail::resolveIfLeaf(View<TStorage, access_set_t<Path>>{*(this->storage)});
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            if constexpr(S::size == 1)
                return detail::single_access_t<S>{};
            else
                return S{};
        }
    };

    // Deduce the whole-record (root) view from storage alone.
    template<typename TStorage>
    View(TStorage&) -> View<TStorage, Set<>>;

    template<typename TStorage, IsAccessSet S>
    requires detail::ViewStorageFor<TStorage, S>
    struct ViewIndexed
    {
        using record_type = TStorage::record_type;
        using access_set = S;

        TStorage* storage;
        uint32_t idx;

        // consteval default constructor, to help get the type of a view more easily
        consteval ViewIndexed() = default;

        // Construct over storage at an index with a given (or empty/root) access set.
        constexpr ViewIndexed(TStorage& storage_, uint32_t index, S /*accessSet*/ = {}) noexcept
            : storage{&storage_}
            , idx{index}
        {
        }

        constexpr ViewIndexed(View<TStorage, S> view, uint32_t index) noexcept : storage{view.storage}, idx{index} {};

        // Convert to this view's access set from another indexed view over the same storage.
        // A root source (whole record) may narrow to anything; otherwise its selection must
        // contain ours. This subsumes narrowing from a parent (indexed) view.
        template<IsAccessSet OtherS>
        constexpr ViewIndexed(ViewIndexed<TStorage, OtherS> const& other) noexcept
            requires(OtherS::size == 0 || Selects<record_type, OtherS, S>)
            : storage{other.storage}
            , idx{other.idx}
        {
        }

        // Drill into the single current node (or from the root for an empty set). A terminal
        // (leaf) child resolves to the element reference at this index; a composite node stays
        // a cursor.
        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) const requires(S::size <= 1)
        {
            if constexpr(S::size == 1)
            {
                using Path = append_t<detail::single_access_t<S>, RA>;
                return detail::resolveIfLeaf(ViewIndexed<TStorage, access_set_t<Path>>{*(this->storage), idx});
            }
            else
            {
                using Path = to_path_t<RA>;
                return detail::resolveIfLeaf(ViewIndexed<TStorage, access_set_t<Path>>{*(this->storage), idx});
            }
        }

        // Select one access out of a multi-access selection. A selected leaf resolves to its
        // element reference; a selected composite node stays a cursor.
        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) const
            requires((S::size > 1) && Selects<record_type, S, access_set_t<RA>>)
        {
            return detail::resolveIfLeaf(ViewIndexed<TStorage, access_set_t<RA>>(*this));
        }

        // needs a leaf access RA in an indexed view. Should only happen when casting to such a type
        // for example implicitly when the user requests it
        [[nodiscard]] constexpr decltype(auto) operator*()
            requires((S::size == 1) && (TStorage::record_type::template isLeaf<detail::single_access_t<S>>()))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr decltype(auto) operator*() const
            requires((S::size == 1) && (TStorage::record_type::template isLeaf<detail::single_access_t<S>>()))
        {
            return resolve(*this);
        }

        // requires we are a leaf node or AsType is
        [[nodiscard]] constexpr decltype(auto) get() requires(
            (S::size == 1)
            && (TStorage::record_type::template isLeaf<detail::single_access_t<S>>()
                || traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<detail::single_access_t<S>>>::value))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr decltype(auto) get() const requires(
            (S::size == 1)
            && (TStorage::record_type::template isLeaf<detail::single_access_t<S>>()
                || traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<detail::single_access_t<S>>>::value))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            if constexpr(S::size == 1)
                return detail::single_access_t<S>{};
            else
                return S{};
        }

        template<typename OtherTStorage, typename OtherS>
        constexpr void deepCopyFrom(ViewIndexed<OtherTStorage, OtherS> other) noexcept
        {
            using SrcR = typename OtherTStorage::record_type;
            using DestR = record_type;

            using DestLeafPaths = GetLeafPaths<DestR>::type;
            [&]<typename... Paths>(Tuple<Paths...>)
            {
                static_assert(
                    (SrcR::hasPath(Paths{}) && ...),
                    "Source storage does not contain all required paths to fulfill this SubRecord.");

                static_assert(
                    (std::is_same_v<
                         typename DestR::template value_type_for<Paths>,
                         typename SrcR::template value_type_for<Paths>>
                     && ...),
                    "Type mismatch between source and destination fields.");

                (((*this)[Paths{}] = other[Paths{}]), ...);
            }(DestLeafPaths{});
        }

        // Flush this sub-record's fields into a (potentially larger) destination record.
        // Walks this record's leaf paths and asserts the destination contains all of them.
        template<typename OtherTStorage, typename OtherS>
        constexpr void deepCopyTo(ViewIndexed<OtherTStorage, OtherS> other) const noexcept
        {
            using SrcR = record_type;
            using DestR = typename OtherTStorage::record_type;

            using SrcLeafPaths = typename GetLeafPaths<SrcR>::type;
            [&]<typename... Paths>(Tuple<Paths...>)
            {
                static_assert(
                    (DestR::hasPath(Paths{}) && ...),
                    "Destination storage does not contain all paths from this record.");

                static_assert(
                    (std::is_same_v<
                         typename SrcR::template value_type_for<Paths>,
                         typename DestR::template value_type_for<Paths>>
                     && ...),
                    "Type mismatch between source and destination fields.");

                ((other[Paths{}] = (*this)[Paths{}]), ...);
            }(SrcLeafPaths{});
        }

        // deep copy
        template<typename OtherTStorage, typename OtherS>
        requires(!std::same_as<TStorage, OtherTStorage>)
        constexpr ViewIndexed& operator=(ViewIndexed<OtherTStorage, OtherS> other) noexcept
        {
            deepCopyFrom(other);
            return *this;
        }
    };

    // Deduce the whole-record (root) indexed view from storage and an index.
    template<typename TStorage>
    ViewIndexed(TStorage&, uint32_t) -> ViewIndexed<TStorage, Set<>>;

} // namespace llama_lite
