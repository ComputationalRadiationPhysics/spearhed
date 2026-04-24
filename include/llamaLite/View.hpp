// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// #include "AccessSet.hpp"
#include "Record.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"
#include "traits.hpp"

#include <concepts>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace llama_lite
{
    template<typename TStorage, IsRecordAccess... RAs>
    requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
    struct ViewIndexed;

    template<typename TStorage, IsRecordAccess... RAs>
    requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
    struct View
    {
        using record_type = TStorage::record_type;

        TStorage* storage;

        // constructor only available if RAs exist in the TStorage record
        constexpr View(TStorage& storage_, RAs...) noexcept
            requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
            : storage{&storage_}
        {
        }

        template<typename... ParentRAs>
        constexpr View(View<TStorage, ParentRAs...> view, RAs...) noexcept
            requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
                    && (IsInSet<to_path_t<RAs>, to_path_t<ParentRAs>...> && ...)
            : storage{view.storage}
        {
        }

        [[nodiscard]] constexpr decltype(auto) operator[](uint32_t idx)
        {
            return ViewIndexed(*this, idx);
        }

        [[nodiscard]] constexpr decltype(auto) operator[](uint32_t idx) const
        {
            return ViewIndexed(*this, idx);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) requires(sizeof...(RAs) == 1)
        {
            using ViewRA = typename SingleElementPack<RAs...>::type;
            using Path = append_t<ViewRA, RA>;
            return View<TStorage, Path>(*(this->storage), Path{});
        }

        [[nodiscard]] constexpr decltype(auto) getSpan()
            requires((sizeof...(RAs) == 1) && (TStorage::record_type::template isLeaf<RAs...>()))
        {
            return storage->template getLeaf<RAs...>();
        }

        [[nodiscard]] constexpr decltype(auto) getSpan() const
            requires((sizeof...(RAs) == 1) && (TStorage::record_type::template isLeaf<RAs...>()))
        {
            return storage->template getLeaf<RAs...>();
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            if constexpr(sizeof...(RAs) == 1)
                return typename SingleElementPack<RAs...>::type{};
            else
                return std::tuple<RAs...>{};
        }
    };

    template<typename TStorage, IsRecordAccess... RAs>
    requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
    struct ViewIndexed
    {
        using record_type = TStorage::record_type;

        TStorage* storage;
        uint32_t idx;

        // consteval default constructor, to help get the type of a view more easily
        consteval ViewIndexed() = default;

        // constructor only available if RAs exist in the TStorage record
        constexpr ViewIndexed(TStorage& storage_, uint32_t index, RAs...) noexcept
            requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
            : storage{&storage_}
            , idx{index}
        {
        }

        constexpr ViewIndexed(View<TStorage, RAs...> view, uint32_t index) noexcept
            : storage{view.storage}
            , idx{index} {};

        template<typename... ParentRAs>
        constexpr ViewIndexed(View<TStorage, ParentRAs...> view, uint32_t index, RAs...) noexcept
            requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
                        && (IsInSet<to_path_t<RAs>, to_path_t<ParentRAs>...> && ...)
            : storage{view.storage}
            , idx{index}
        {
        }

        template<typename... ParentRAs>
        constexpr ViewIndexed(ViewIndexed<TStorage, ParentRAs...> idxView, RAs...) noexcept
            requires(requires { typename TStorage::record_type::template field_for<RAs>; } && ...)
                        && (IsInSet<to_path_t<RAs>, ParentRAs...> && ...)
            : storage{idxView.storage}
            , idx{idxView.idx}
        {
        }

        // conversion constructor to defined RAs from another view.
        template<typename... OtherRAs>
        constexpr ViewIndexed(ViewIndexed<TStorage, OtherRAs...> const& other) noexcept
            requires(
                        // Allow conversion from Root view
                        sizeof...(OtherRAs) == 0 ||
                        // OR Ensure all RAs in this view are present in the OtherRAs
                        (IsInSet<to_path_t<RAs>, to_path_t<OtherRAs>...> && ...))
            : storage{other.storage}
            , idx{other.idx}
        {
        }

        // TODO add checks on RA being valid for the storage record
        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) const requires(sizeof...(RAs) <= 1)
        {
            if constexpr(sizeof...(RAs) == 1)
            {
                using ViewRA = typename SingleElementPack<RAs...>::type;
                using Path = append_t<ViewRA, RA>;
                return ViewIndexed<TStorage, Path>(*(this->storage), idx, Path{});
            }
            else
            {
                using Path = to_path_t<RA>;
                return ViewIndexed<TStorage, Path>(*(this->storage), idx, Path{});
            }
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto operator[](RA) const
            requires((sizeof...(RAs) > 1) && IsInSet<to_path_t<RA>, to_path_t<RAs>...>)
        {
            return ViewIndexed<TStorage, to_path_t<RA>>(*this);
        }

        // needs a leaf access RA in an indexed view. Should only happen when casting to such a type
        // for example implicitly when the user requests it
        [[nodiscard]] constexpr decltype(auto) operator*()
            requires((sizeof...(RAs) == 1) && (TStorage::record_type::template isLeaf<RAs...>()))
        {
            return storage->template getLeaf<RAs...>()[idx];
        }

        [[nodiscard]] constexpr decltype(auto) operator*() const
            requires((sizeof...(RAs) == 1) && (TStorage::record_type::template isLeaf<RAs...>()))
        {
            return storage->template getLeaf<RAs...>()[idx];
        }

        // requires we are a leaf node or AsType is
        [[nodiscard]] constexpr decltype(auto) get() requires(
            (sizeof...(RAs) == 1)
            && (TStorage::record_type::template isLeaf<RAs...>()
                || traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<RAs...>>::value))
        {
            if constexpr(
                traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<RAs...>>::value)
            {
                return traits::AsType<typename TStorage::record_type::template field_for<RAs...>>{}(*this);
            }
            else // is a leaf
            {
                return *(*this);
            }
        }

        [[nodiscard]] constexpr decltype(auto) get() const requires(
            (sizeof...(RAs) == 1)
            && (TStorage::record_type::template isLeaf<RAs...>()
                || traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<RAs...>>::value))
        {
            if constexpr(
                traits::IsTraitSpecialized<
                    traits::AsType,
                    typename TStorage::record_type::template field_for<RAs...>>::value)
            {
                return traits::AsType<typename TStorage::record_type::template field_for<RAs...>>{}(*this);
            }
            else // is a leaf
            {
                return *(*this);
            }
        }

        [[nodiscard]] constexpr auto getRecordAccess() const
        {
            if constexpr(sizeof...(RAs) == 1)
                return typename SingleElementPack<RAs...>::type{};
            else
                return std::tuple<RAs...>{};
        }

        template<typename OtherTStorage, typename... OtherRAs>
        constexpr void deepCopyFrom(ViewIndexed<OtherTStorage, OtherRAs...> other) noexcept
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

                ((*(*this)[Paths{}] = *other[Paths{}]), ...);
            }(DestLeafPaths{});
        }

        // deep copy
        template<typename OtherTStorage, typename... OtherRAs>
        requires(!std::same_as<TStorage, OtherTStorage>)
        constexpr ViewIndexed& operator=(ViewIndexed<OtherTStorage, OtherRAs...> other) noexcept
        {
            deepCopyFrom(other);
            return *this;
        }
    };

    // template<template<typename> typename Func, typename T_Record, IsRecordAccess... RAs>
    // constexpr void for_each(T_Record Record, RAs...)
    //     requires(requires { typename T_Record::template field_for<RAs>; } && ...)
    // {
    //     // if view has 0 RAs, go over all elements in a record
    //     // else go over all RAs only.

    //     // if Func<RA> is defined call it
    //     // else (if RA points to a record, call Func<> on its fields, and so on recursively. If RA is a field )

    //     using CurrentRecordType = std::conditional_t<
    //         sizeof...(RAs) == 0,
    //         typename TStorage::record_type,
    //         typename TStorage::record_type::template value_type_for<
    //             to_path_t<std::tuple_element_t<0, Tuple<RAs...>>>>>;

    //     // We inspect the structure of the record currently pointed to by this View
    //     using Fields = typename CurrentRecordType::fields_tuple_type;

    //     // Fold expression to apply function to all children
    //     [&]<typename... Fs>(Tuple<Fs...>) { (func((*this)[typename Fs::tag_type{}]), ...); }(Fields{});
    // }

} // namespace llama_lite
