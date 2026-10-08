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

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace llama_lite
{
    namespace detail
    {
        /// The raw canonical leaf Set a bound AccessSet denotes.
        template<typename Access>
        using access_set_leaves_t = typename Access::leaf_set;

        /// The record a bound access set is expressed against.
        template<IsAccessSet Access>
        using access_record_t = typename Access::record_type;

        /// The record's whole leaf set, wrapped as a bound access set.
        template<IsRecord Record>
        using whole_record_access_t = AccessSet<Record, record_leaf_set_t<Record>>;

        /// The leaves under a navigation path, wrapped as a bound access set.
        template<IsRecord Record, IsTagPath Path>
        using subtree_access_t = access_set_t<Record, Set<Path>>;

        /// The part of @p Access under @p Path. Both operands share Access's record, so the
        /// record-aware intersection applies directly; the result stays canonical and bound.
        template<IsAccessSet Access, IsTagPath Path>
        using child_access_t = decltype(Access{} & subtree_access_t<access_record_t<Access>, Path>{});

        /// The canonical access set for a pack of accesses relative to @p Root.
        template<IsRecord Record, IsTagPath Root, IsRecordAccess... RAs>
        using requested_access_t = access_set_t<Record, raw_access_set_t<append_t<Root, RAs>...>>;

        // Handles store a record-bound, canonical AccessSet whose record is the storage record and
        // whose leaves lie within the navigation root. TagPath<> is the record root; an empty leaf
        // set is an empty selection.
        template<typename TStorage, typename Access, typename Root>
        concept ViewStorageFor = requires { typename TStorage::record_type; } && IsAccessSet<Access> && IsTagPath<Root>
                                 && std::same_as<access_record_t<Access>, typename TStorage::record_type>
                                 && TStorage::record_type::hasPath(Root{})
                                 && (Access{} <= subtree_access_t<typename TStorage::record_type, Root>{});

        // Constraint order matters: whole-kind checks and path existence are validated before any
        // constrained alias (child_access_t/requested_access_t/subtree_access_t) is instantiated.
        template<typename Access, typename Root, typename RA>
        concept CanDrill = IsAccessSet<Access> && IsTagPath<Root> && IsRecordAccess<RA>
                           && access_record_t<Access>::hasPath(append_t<Root, RA>{})
                           && !child_access_t<Access, append_t<Root, RA>>::empty;

        template<typename Access, typename Root, typename... RAs>
        concept CanSelect = IsAccessSet<Access> && IsTagPath<Root> && (IsRecordAccess<RAs> && ...)
                            && ValidAccessSetFor<access_record_t<Access>, raw_access_set_t<append_t<Root, RAs>...>>
                            && (requested_access_t<access_record_t<Access>, Root, RAs...>{} <= Access{});

        template<typename Access, typename Root>
        concept CanResolve = IsAccessSet<Access> && IsTagPath<Root> && (Root::depth > 0)
                             && (subtree_access_t<access_record_t<Access>, Root>{} <= Access{})
                             && (access_record_t<Access>::isLeaf(Root{})
                                 || traits::IsTraitSpecialized<
                                     traits::AsType,
                                     typename access_record_t<Access>::template field_for<Root>>::value);
    } // namespace detail

    /**
     * Resolve a fully selected leaf or adapted subtree cursor.
     * Indexed leaves return scalar references; column leaves return spans.
     * Composite nodes use traits::AsType and require all subtree leaves to be selected.
     * @param view a shallow handle (storage pointer, plus an index for indexed views)
     */
    template<typename V>
    requires detail::CanResolve<typename V::access_set, typename V::root_access>
    [[nodiscard]] constexpr decltype(auto) resolve(V view)
    {
        using Access = typename V::root_access;
        using Field = typename V::record_type::template field_for<Access>;
        if constexpr(traits::IsTraitSpecialized<traits::AsType, Field>::value)
            return traits::AsType<Field>{}(view);
        else if constexpr(requires { view.idx; })
            return view.storage->getLeaf(Access{})[view.idx];
        else
            return view.storage->getLeaf(Access{});
    }

    namespace detail
    {
        // Bracket navigation resolves terminal leaves, but keeps composite nodes as cursors.
        template<typename ChildView>
        [[nodiscard]] constexpr decltype(auto) resolveIfLeaf(ChildView child)
        {
            using Access = typename ChildView::root_access;
            if constexpr(Access::depth == 0)
                return child;
            else if constexpr(ChildView::record_type::isLeaf(Access{}))
                return resolve(child);
            else
                return child;
        }
    } // namespace detail

    template<
        typename TStorage,
        typename Access = detail::whole_record_access_t<typename TStorage::record_type>,
        IsTagPath Root = TagPath<>>
    requires detail::ViewStorageFor<TStorage, Access, Root>
    struct ViewIndexed;

    /// Column cursor with a record-bound access set and a relative navigation root.
    template<
        typename TStorage,
        typename Access = detail::whole_record_access_t<typename TStorage::record_type>,
        IsTagPath Root = TagPath<>>
    requires detail::ViewStorageFor<TStorage, Access, Root>
    struct View
    {
        using record_type = TStorage::record_type;
        using access_set = Access;
        using root_access = Root;

        TStorage* storage;

        constexpr View(View const&) = default;
        constexpr View(View&&) = default;
        constexpr View& operator=(View const&) & = default;
        constexpr View& operator=(View&&) & = default;

        constexpr View(TStorage& storage_) noexcept : storage{&storage_}
        {
        }

        // Shallow narrowing is allowed only within the same navigation root and record.
        template<typename ParentAccess>
        constexpr View(View<TStorage, ParentAccess, Root> view) noexcept requires(Access{} <= ParentAccess{})
            : storage{view.storage}
        {
        }

        [[nodiscard]] constexpr auto operator[](uint32_t idx) const
        {
            return ViewIndexed<TStorage, Access, Root>(*this, idx);
        }

        [[nodiscard]] constexpr auto view() const
        {
            return *this;
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA tag) const requires detail::CanDrill<Access, Root, RA>
        {
            return detail::resolveIfLeaf(this->view(tag));
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto view(RA) const requires detail::CanDrill<Access, Root, RA>
        {
            using Path = append_t<Root, RA>;
            return View<TStorage, detail::child_access_t<Access, Path>, Path>{*storage};
        }

        // Every requested leaf must already be selected; an empty pack selects no leaves.
        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto select(RAs... /*accesses*/) const requires detail::CanSelect<Access, Root, RAs...>
        {
            using Selected = detail::requested_access_t<record_type, Root, RAs...>;
            return View<TStorage, Selected, Root>{*storage};
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            return Root{};
        }
    };

    template<typename TStorage>
    View(TStorage&) -> View<TStorage>;

    /// Row cursor; construction and assignment rebind the handle, not the selected values.
    template<typename TStorage, typename Access, IsTagPath Root>
    requires detail::ViewStorageFor<TStorage, Access, Root>
    struct ViewIndexed
    {
        using record_type = TStorage::record_type;
        using access_set = Access;
        using root_access = Root;

        TStorage* storage;
        uint32_t idx;

        constexpr ViewIndexed(ViewIndexed const&) = default;
        constexpr ViewIndexed(ViewIndexed&&) = default;
        constexpr ViewIndexed& operator=(ViewIndexed const&) & = default;
        constexpr ViewIndexed& operator=(ViewIndexed&&) & = default;

        constexpr ViewIndexed(TStorage& storage_, uint32_t index) noexcept : storage{&storage_}, idx{index}
        {
        }

        constexpr ViewIndexed(View<TStorage, Access, Root> view, uint32_t index) noexcept
            : storage{view.storage}
            , idx{index}
        {
        }

        template<typename OtherAccess>
        constexpr ViewIndexed(ViewIndexed<TStorage, OtherAccess, Root> const& other) noexcept
            requires(Access{} <= OtherAccess{})
            : storage{other.storage}
            , idx{other.idx}
        {
        }

        [[nodiscard]] constexpr auto view() const
        {
            return *this;
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA tag) const requires detail::CanDrill<Access, Root, RA>
        {
            return detail::resolveIfLeaf(this->view(tag));
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto view(RA) const requires detail::CanDrill<Access, Root, RA>
        {
            using Path = append_t<Root, RA>;
            return ViewIndexed<TStorage, detail::child_access_t<Access, Path>, Path>{*storage, idx};
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto select(RAs... /*accesses*/) const requires detail::CanSelect<Access, Root, RAs...>
        {
            using Selected = detail::requested_access_t<record_type, Root, RAs...>;
            return ViewIndexed<TStorage, Selected, Root>{*storage, idx};
        }

        [[nodiscard]] constexpr decltype(auto) operator*() const
            requires detail::CanResolve<Access, Root> && (record_type::isLeaf(Root{}))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr decltype(auto) get() const requires detail::CanResolve<Access, Root>
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            return Root{};
        }
    };

    template<typename TStorage>
    ViewIndexed(TStorage&, uint32_t) -> ViewIndexed<TStorage>;

    template<typename TStorage, typename Access, IsTagPath Root>
    [[nodiscard]] constexpr auto getSelectedLeaves(View<TStorage, Access, Root> const&) noexcept
    {
        return detail::access_set_leaves_t<Access>{};
    }

    template<typename TStorage, typename Access, IsTagPath Root>
    [[nodiscard]] constexpr auto getSelectedLeaves(ViewIndexed<TStorage, Access, Root> const&) noexcept
    {
        return detail::access_set_leaves_t<Access>{};
    }

    namespace detail
    {
        // Cross-record copies compare raw leaf paths, never differently bound access sets.
        template<typename DestStorage, typename SrcStorage, typename DestLeaves, typename SrcLeaves>
        struct CopyValuesCompatible;

        template<typename DestStorage, typename SrcStorage, typename SrcLeaves, IsTagPath... DestPaths>
        struct CopyValuesCompatible<DestStorage, SrcStorage, Set<DestPaths...>, SrcLeaves>
        {
            template<typename Path>
            static consteval bool canCopyPath()
            {
                using SrcRecord = typename SrcStorage::record_type;
                if constexpr(!SrcRecord::hasPath(Path{}))
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
    } // namespace detail

    /**
     * Copy exactly the destination-selected leaves; extra source leaves are ignored.
     * If assignment throws, earlier leaves may already have been copied.
     * Handle construction and assignment remain shallow.
     */
    template<
        typename DestStorage,
        typename DestAccess,
        IsTagPath DestRoot,
        typename SrcStorage,
        typename SrcAccess,
        IsTagPath SrcRoot>
    requires(
        !std::is_const_v<DestStorage> && detail::ViewStorageFor<DestStorage, DestAccess, DestRoot>
        && detail::ViewStorageFor<SrcStorage, SrcAccess, SrcRoot>
        && detail::CopyValuesCompatible<
            DestStorage,
            SrcStorage,
            detail::access_set_leaves_t<DestAccess>,
            detail::access_set_leaves_t<SrcAccess>>::value)
    constexpr void copy_values(
        ViewIndexed<DestStorage, DestAccess, DestRoot> dest,
        ViewIndexed<SrcStorage, SrcAccess, SrcRoot> src)
    {
        auto const& sourceStorage = *src.storage;
        using DestLeaves = detail::access_set_leaves_t<DestAccess>;
        [&]<IsTagPath... Paths>(Set<Paths...>)
        { ((dest.storage->getLeaf(Paths{})[dest.idx] = sourceStorage.getLeaf(Paths{})[src.idx]), ...); }(DestLeaves{});
    }

    /**
     * Narrow a destination handle to the model's absolute selected leaf paths.
     * Missing or unselected paths are rejected; no intersection is taken.
     * Empty models are rejected; skip empty-record transfers at compile time.
     */
    template<
        typename DestStorage,
        typename DestAccess,
        IsTagPath DestRoot,
        typename ModelStorage,
        typename ModelAccess,
        IsTagPath ModelRoot>
    requires(
        detail::ViewStorageFor<DestStorage, DestAccess, DestRoot>
        && detail::ViewStorageFor<ModelStorage, ModelAccess, ModelRoot>
        && detail::access_set_leaves_t<ModelAccess>::size > 0
        && detail::ValidAccessSetFor<typename DestStorage::record_type, detail::access_set_leaves_t<ModelAccess>>
        && (detail::access_set_leaves_t<ModelAccess>{} <= detail::access_set_leaves_t<DestAccess>{}))
    [[nodiscard]] constexpr auto select_like(
        ViewIndexed<DestStorage, DestAccess, DestRoot> dest,
        ViewIndexed<ModelStorage, ModelAccess, ModelRoot> /*model*/)
    {
        // Different records may declare the same leaves in a different order, so rebind the
        // model's raw leaf paths into the destination record and its declaration order.
        using Selected = access_set_t<typename DestStorage::record_type, detail::access_set_leaves_t<ModelAccess>>;
        return ViewIndexed<DestStorage, Selected, DestRoot>{dest};
    }
} // namespace llama_lite
