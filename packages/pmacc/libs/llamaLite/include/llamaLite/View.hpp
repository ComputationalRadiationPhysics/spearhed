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
        // Handles store canonical absolute leaf permissions within one navigation path.
        // TagPath<> is the record root; Set<> is an empty selection, not whole-record access.
        template<typename TStorage, typename Leaves, typename Root>
        concept ViewStorageFor
            = requires { typename TStorage::record_type; } && IsAccessSet<Leaves> && IsTagPath<Root>
              && ValidAccessSetFor<typename TStorage::record_type, Leaves> && TStorage::record_type::hasPath(Root{})
              && std::same_as<Leaves, leaf_set_t<typename TStorage::record_type, Leaves>>
              && (Leaves{} <= leaf_set_t<typename TStorage::record_type, Set<Root>>{});

        template<IsRecord Record, IsTagPath Path>
        using subtree_leaves_t = leaf_set_t<Record, Set<Path>>;

        template<IsRecord Record, IsAccessSet Leaves, IsTagPath Path>
        using child_leaves_t = decltype(Leaves{} & subtree_leaves_t<Record, Path>{});

        template<IsRecord Record, IsTagPath Root, IsRecordAccess... RAs>
        using requested_leaves_t = leaf_set_t<Record, access_set_t<append_t<Root, RAs>...>>;

        template<typename Record, typename Leaves, typename Root, typename RA>
        concept CanDrill = IsRecordAccess<RA> && Record::hasPath(append_t<Root, RA>{})
                           && !child_leaves_t<Record, Leaves, append_t<Root, RA>>::empty;

        template<typename Record, typename Leaves, typename Root, typename... RAs>
        concept CanSelect
            = (IsRecordAccess<RAs> && ...) && ValidAccessSetFor<Record, access_set_t<append_t<Root, RAs>...>>
              && (requested_leaves_t<Record, Root, RAs...>{} <= Leaves{});

        template<typename Record, typename Leaves, typename Root>
        concept CanResolve
            = (Root::depth > 0) && (subtree_leaves_t<Record, Root>{} <= Leaves{})
              && (Record::isLeaf(Root{})
                  || traits::IsTraitSpecialized<traits::AsType, typename Record::template field_for<Root>>::value);
    } // namespace detail

    /**
     * Resolve a fully selected leaf or adapted subtree cursor.
     * Indexed leaves return scalar references; column leaves return spans.
     * Composite nodes use traits::AsType and require all subtree leaves to be selected.
     * @param view a shallow handle (storage pointer, plus an index for indexed views)
     */
    template<typename V>
    requires detail::CanResolve<typename V::record_type, typename V::access_set, typename V::root_access>
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
        IsAccessSet Leaves = record_leaf_set_t<typename TStorage::record_type>,
        IsTagPath Root = TagPath<>>
    requires detail::ViewStorageFor<TStorage, Leaves, Root>
    struct ViewIndexed;

    /// Column cursor with canonical absolute leaf permissions and a relative navigation root.
    template<
        typename TStorage,
        IsAccessSet Leaves = record_leaf_set_t<typename TStorage::record_type>,
        IsTagPath Root = TagPath<>>
    requires detail::ViewStorageFor<TStorage, Leaves, Root>
    struct View
    {
        using record_type = TStorage::record_type;
        using access_set = Leaves;
        using root_access = Root;

        TStorage* storage;

        constexpr View(View const&) = default;
        constexpr View(View&&) = default;
        constexpr View& operator=(View const&) & = default;
        constexpr View& operator=(View&&) & = default;

        constexpr View(TStorage& storage_) noexcept : storage{&storage_}
        {
        }

        // Shallow narrowing is allowed only within the same navigation root.
        template<IsAccessSet ParentLeaves>
        constexpr View(View<TStorage, ParentLeaves, Root> view) noexcept requires(Leaves{} <= ParentLeaves{})
            : storage{view.storage}
        {
        }

        [[nodiscard]] constexpr auto operator[](uint32_t idx) const
        {
            return ViewIndexed<TStorage, Leaves, Root>(*this, idx);
        }

        [[nodiscard]] constexpr auto view() const
        {
            return *this;
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA tag) const
            requires detail::CanDrill<record_type, Leaves, Root, RA>
        {
            return detail::resolveIfLeaf(this->view(tag));
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto view(RA) const requires detail::CanDrill<record_type, Leaves, Root, RA>
        {
            using Path = append_t<Root, RA>;
            return View<TStorage, detail::child_leaves_t<record_type, Leaves, Path>, Path>{*storage};
        }

        // Every requested leaf must already be selected; an empty pack selects no leaves.
        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto select(RAs... /*accesses*/) const
            requires detail::CanSelect<record_type, Leaves, Root, RAs...>
        {
            return View<TStorage, detail::requested_leaves_t<record_type, Root, RAs...>, Root>{*storage};
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            return Root{};
        }
    };

    template<typename TStorage>
    View(TStorage&) -> View<TStorage>;

    /// Row cursor; construction and assignment rebind the handle, not the selected values.
    template<typename TStorage, IsAccessSet Leaves, IsTagPath Root>
    requires detail::ViewStorageFor<TStorage, Leaves, Root>
    struct ViewIndexed
    {
        using record_type = TStorage::record_type;
        using access_set = Leaves;
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

        constexpr ViewIndexed(View<TStorage, Leaves, Root> view, uint32_t index) noexcept
            : storage{view.storage}
            , idx{index}
        {
        }

        template<IsAccessSet OtherLeaves>
        constexpr ViewIndexed(ViewIndexed<TStorage, OtherLeaves, Root> const& other) noexcept
            requires(Leaves{} <= OtherLeaves{})
            : storage{other.storage}
            , idx{other.idx}
        {
        }

        [[nodiscard]] constexpr auto view() const
        {
            return *this;
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA tag) const
            requires detail::CanDrill<record_type, Leaves, Root, RA>
        {
            return detail::resolveIfLeaf(this->view(tag));
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto view(RA) const requires detail::CanDrill<record_type, Leaves, Root, RA>
        {
            using Path = append_t<Root, RA>;
            return ViewIndexed<TStorage, detail::child_leaves_t<record_type, Leaves, Path>, Path>{*storage, idx};
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] constexpr auto select(RAs... /*accesses*/) const
            requires detail::CanSelect<record_type, Leaves, Root, RAs...>
        {
            return ViewIndexed<TStorage, detail::requested_leaves_t<record_type, Root, RAs...>, Root>{*storage, idx};
        }

        [[nodiscard]] constexpr decltype(auto) operator*() const
            requires detail::CanResolve<record_type, Leaves, Root> && (record_type::isLeaf(Root{}))
        {
            return resolve(*this);
        }

        [[nodiscard]] constexpr decltype(auto) get() const requires detail::CanResolve<record_type, Leaves, Root>
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

    template<typename TStorage, IsAccessSet Leaves, IsTagPath Root>
    [[nodiscard]] constexpr auto getSelectedLeaves(View<TStorage, Leaves, Root> const&) noexcept
    {
        return Leaves{};
    }

    template<typename TStorage, IsAccessSet Leaves, IsTagPath Root>
    [[nodiscard]] constexpr auto getSelectedLeaves(ViewIndexed<TStorage, Leaves, Root> const&) noexcept
    {
        return Leaves{};
    }

    namespace detail
    {
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
        IsAccessSet DestLeaves,
        IsTagPath DestRoot,
        typename SrcStorage,
        IsAccessSet SrcLeaves,
        IsTagPath SrcRoot>
    requires(
        !std::is_const_v<DestStorage>
        && detail::CopyValuesCompatible<DestStorage, SrcStorage, DestLeaves, SrcLeaves>::value)
    constexpr void copy_values(
        ViewIndexed<DestStorage, DestLeaves, DestRoot> dest,
        ViewIndexed<SrcStorage, SrcLeaves, SrcRoot> src)
    {
        auto const& sourceStorage = *src.storage;
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
        IsAccessSet DestLeaves,
        IsTagPath DestRoot,
        typename ModelStorage,
        IsAccessSet ModelLeaves,
        IsTagPath ModelRoot>
    requires(
        ModelLeaves::size > 0 && detail::ValidAccessSetFor<typename DestStorage::record_type, ModelLeaves>
        && (ModelLeaves{} <= DestLeaves{}))
    [[nodiscard]] constexpr auto select_like(
        ViewIndexed<DestStorage, DestLeaves, DestRoot> dest,
        ViewIndexed<ModelStorage, ModelLeaves, ModelRoot> /*model*/)
    {
        // Different records may declare the same leaves in a different order.
        using Leaves = leaf_set_t<typename DestStorage::record_type, ModelLeaves>;
        return ViewIndexed<DestStorage, Leaves, DestRoot>{dest};
    }
} // namespace llama_lite
