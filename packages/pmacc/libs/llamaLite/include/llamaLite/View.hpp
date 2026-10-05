// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/AccessSet.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/Set.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/traits.hpp"
#include "llamaLite/utility.hpp"

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

        template<IsRecord Record, IsAccessSet S>
        using ViewLeafSet = std::conditional_t<S::size == 0, record_leaf_set_t<Record>, leaf_set_t<Record, S>>;
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
                Record::isLeaf(Access{}),
                "resolve: access names a composite node without an AsType specialization; nothing to resolve to.");

            if constexpr(requires { view.idx; })
                return view.storage->getLeaf(Access{})[view.idx];
            else
                return view.storage->getLeaf(Access{});
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
            if constexpr(Record::isLeaf(Access{}))
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

        constexpr View(View const&) = default;
        constexpr View(View&&) = default;
        constexpr View& operator=(View const&) & = default;
        constexpr View& operator=(View&&) & = default;

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

        constexpr ViewIndexed(ViewIndexed const&) = default;
        constexpr ViewIndexed(ViewIndexed&&) = default;
        constexpr ViewIndexed& operator=(ViewIndexed const&) & = default;
        constexpr ViewIndexed& operator=(ViewIndexed&&) & = default;

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
            requires((S::size == 1) && (TStorage::record_type::isLeaf(detail::single_access_t<S>{})))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr decltype(auto) operator*() const
            requires((S::size == 1) && (TStorage::record_type::isLeaf(detail::single_access_t<S>{})))
        {
            return resolve(*this);
        }

        // requires we are a leaf node or AsType is
        [[nodiscard]] constexpr decltype(auto) get() requires(
            (S::size == 1)
            && (TStorage::record_type::isLeaf(detail::single_access_t<S>{})
                || traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<detail::single_access_t<S>>>::value))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr decltype(auto) get() const requires(
            (S::size == 1)
            && (TStorage::record_type::isLeaf(detail::single_access_t<S>{})
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
    };

    // Deduce the whole-record (root) indexed view from storage and an index.
    template<typename TStorage>
    ViewIndexed(TStorage&, uint32_t) -> ViewIndexed<TStorage, Set<>>;

    template<typename TStorage, IsAccessSet S>
    requires detail::ViewStorageFor<TStorage, S>
    [[nodiscard]] constexpr auto getSelectedLeaves(View<TStorage, S> const&) noexcept
    {
        return detail::ViewLeafSet<typename TStorage::record_type, S>{};
    }

    template<typename TStorage, IsAccessSet S>
    requires detail::ViewStorageFor<TStorage, S>
    [[nodiscard]] constexpr auto getSelectedLeaves(ViewIndexed<TStorage, S> const&) noexcept
    {
        return detail::ViewLeafSet<typename TStorage::record_type, S>{};
    }

    namespace detail
    {
        template<typename Record, typename Paths>
        struct PathsExist : std::false_type
        {
        };

        template<typename Record, IsTagPath... Paths>
        struct PathsExist<Record, Set<Paths...>> : std::bool_constant<(Record::hasPath(Paths{}) && ... && true)>
        {
        };

        template<typename DestStorage, typename SrcStorage, typename DestLeaves, typename SrcLeaves>
        struct CopyValuesCompatible : std::false_type
        {
        };

        template<typename DestStorage, typename SrcStorage, typename SrcLeaves, IsTagPath... DestPaths>
        struct CopyValuesCompatible<DestStorage, SrcStorage, Set<DestPaths...>, SrcLeaves>
        {
            template<typename Path>
            static consteval bool canCopyPath()
            {
                using DestRecord = typename DestStorage::record_type;
                using SrcRecord = typename SrcStorage::record_type;
                if constexpr(!DestRecord::hasPath(Path{}))
                    return false;
                else if constexpr(!SrcRecord::hasPath(Path{}))
                    return false;
                else if constexpr(!SrcLeaves::contains(Path{}))
                    return false;
                else
                    return requires(DestStorage& dest, SrcStorage const& src, uint32_t index) {
                        dest.getLeaf(Path{})[index] = src.getLeaf(Path{})[index];
                    };
            }

            static constexpr bool value = (canCopyPath<DestPaths>() && ... && true);
        };

        template<typename DestStorage, typename SrcStorage, typename DestLeaves, typename SrcLeaves>
        inline constexpr bool copy_values_compatible_v
            = CopyValuesCompatible<DestStorage, SrcStorage, DestLeaves, SrcLeaves>::value;
    } // namespace detail

    /**
     * Copy the destination view's selected leaf values from the source view.
     * A root view selects every leaf in its record. Every destination-selected leaf must
     * also be selected by the source; extra source leaves are ignored. If a leaf assignment
     * throws, earlier leaves may already have been copied. Handle construction and assignment
     * remain shallow.
     */
    template<typename DestStorage, IsAccessSet DestS, typename SrcStorage, IsAccessSet SrcS>
    requires(
        !std::is_const_v<DestStorage>
        && detail::copy_values_compatible_v<
            DestStorage,
            SrcStorage,
            detail::ViewLeafSet<typename DestStorage::record_type, DestS>,
            detail::ViewLeafSet<typename SrcStorage::record_type, SrcS>>)
    constexpr void copy_values(ViewIndexed<DestStorage, DestS> dest, ViewIndexed<SrcStorage, SrcS> src)
    {
        using DestLeaves = detail::ViewLeafSet<typename DestStorage::record_type, DestS>;
        auto const& sourceStorage = *src.storage;
        [&]<IsTagPath... Paths>(Set<Paths...>)
        { ((dest.storage->getLeaf(Paths{})[dest.idx] = sourceStorage.getLeaf(Paths{})[src.idx]), ...); }(DestLeaves{});
    }

    /**
     * Return a shallow destination handle narrowed to the model view's selected leaves.
     * This is useful when copying a selected subrecord into a broader record: the selection
     * is derived from the model's record (including all leaves for a root view), then checked
     * against the destination selection. Missing paths are rejected; no intersection is taken.
     * Empty models are rejected because an empty access set denotes a root view, not an empty
     * selection; callers should skip empty-record transfers at compile time.
     */
    template<typename DestStorage, IsAccessSet DestS, typename ModelStorage, IsAccessSet ModelS>
    requires(
        (detail::ViewLeafSet<typename ModelStorage::record_type, ModelS>::size > 0)
        && detail::PathsExist<
            typename DestStorage::record_type,
            detail::ViewLeafSet<typename ModelStorage::record_type, ModelS>>::value
        && (detail::ViewLeafSet<typename ModelStorage::record_type, ModelS>{}
            <= detail::ViewLeafSet<typename DestStorage::record_type, DestS>{}))
    [[nodiscard]] constexpr auto select_like(
        ViewIndexed<DestStorage, DestS> dest,
        ViewIndexed<ModelStorage, ModelS> /*model*/)
    {
        using ModelLeaves = detail::ViewLeafSet<typename ModelStorage::record_type, ModelS>;
        return ViewIndexed<DestStorage, ModelLeaves>{dest};
    }

} // namespace llama_lite
