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

#include <cstdint>
#include <type_traits>

namespace llama_lite
{
    template<typename TSoA, IsRecordAccess... RAs>
    requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
    struct SoAIndexedView;

    template<typename TSoA, IsRecordAccess... RAs>
    requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
    struct SoAView
    {
        using record_type = TSoA::record_type;

        TSoA* soa;

        // constructor only available if RAs exist in the TSoA record
        constexpr SoAView(TSoA& soa_, RAs...) noexcept
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
            : soa{&soa_}
        {
        }

        template<typename... ParentRAs>
        constexpr SoAView(SoAView<TSoA, ParentRAs...> view, RAs...) noexcept
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
                    && (IsInSet<to_path_t<RAs>, to_path_t<ParentRAs>...> && ...)
            : soa{view.soa}
        {
        }

        [[nodiscard]] constexpr decltype(auto) operator[](uint32_t idx)
        {
            return SoAIndexedView(*this, idx);
        }

        [[nodiscard]] constexpr decltype(auto) operator[](uint32_t idx) const
        {
            return SoAIndexedView(*this, idx);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) requires(sizeof...(RAs) == 1)
        {
            using ViewRA = typename SingleElementPack<RAs...>::type;
            using Path = append_t<ViewRA, RA>;
            return SoAView<TSoA, Path>(*(this->soa), Path{});
        }
    };

    template<typename TSoA, IsRecordAccess... RAs>
    requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
    struct SoAIndexedView
    {
        using record_type = TSoA::record_type;

        TSoA* soa;
        uint32_t idx;

        // consteval default constructor, to help get the type of a view more easily
        consteval SoAIndexedView() = default;

        // constructor only available if RAs exist in the TSoA record
        constexpr SoAIndexedView(TSoA& soa_, uint32_t index, RAs...) noexcept
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
            : soa{&soa_}
            , idx{index}
        {
        }

        constexpr SoAIndexedView(SoAView<TSoA, RAs...> view, uint32_t index) noexcept : soa{view.soa}, idx{index} {};

        template<typename... ParentRAs>
        constexpr SoAIndexedView(SoAView<TSoA, ParentRAs...> view, uint32_t index, RAs...) noexcept
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
                        && (IsInSet<to_path_t<RAs>, to_path_t<ParentRAs>...> && ...)
            : soa{view.soa}
            , idx{index}
        {
        }

        template<typename... ParentRAs>
        constexpr SoAIndexedView(SoAIndexedView<TSoA, ParentRAs...> idxView, RAs...) noexcept
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
                        && (IsInSet<to_path_t<RAs>, ParentRAs...> && ...)
            : soa{idxView.soa}
            , idx{idxView.idx}
        {
        }

        // conversion constructor to defined RAs from another view.
        template<typename... OtherRAs>
        constexpr SoAIndexedView(SoAIndexedView<TSoA, OtherRAs...> const& other) noexcept
            requires(
                        // Allow conversion from Root view
                        sizeof...(OtherRAs) == 0 ||
                        // OR Ensure all RAs in this view are present in the OtherRAs
                        (IsInSet<to_path_t<RAs>, to_path_t<OtherRAs>...> && ...))
            : soa{other.soa}
            , idx{other.idx}
        {
        }

        // TODO add checks on RA being valid for the soa record
        template<IsRecordAccess RA>
        [[nodiscard]] constexpr decltype(auto) operator[](RA) const requires(sizeof...(RAs) <= 1)
        {
            if constexpr(sizeof...(RAs) == 1)
            {
                using ViewRA = typename SingleElementPack<RAs...>::type;
                using Path = append_t<ViewRA, RA>;
                return SoAIndexedView<TSoA, Path>(*(this->soa), idx, Path{});
            }
            else
            {
                using Path = to_path_t<RA>;
                return SoAIndexedView<TSoA, Path>(*(this->soa), idx, Path{});
            }
        }

        template<IsRecordAccess RA>
        [[nodiscard]] constexpr auto operator[](RA) const
            requires((sizeof...(RAs) > 1) && IsInSet<to_path_t<RA>, to_path_t<RAs>...>)
        {
            return SoAIndexedView<TSoA, to_path_t<RA>>(*this);
        }

        // needs a leaf access RA in an indexed view. Should only happen when casting to such a type
        // for example implicitly when the user requests it
        [[nodiscard]] constexpr decltype(auto) operator*()
            requires((sizeof...(RAs) == 1) && (TSoA::record_type::template isLeaf<RAs...>()))
        {
            return soa->template getLeaf<RAs...>()[idx];
        }

        [[nodiscard]] constexpr decltype(auto) operator*() const
            requires((sizeof...(RAs) == 1) && (TSoA::record_type::template isLeaf<RAs...>()))
        {
            return soa->template getLeaf<RAs...>()[idx];
        }

        // requires we are a leaf node or AsType is
        [[nodiscard]] constexpr decltype(auto) get() requires(
            (sizeof...(RAs) == 1)
            && (TSoA::record_type::template isLeaf<RAs...>()
                || traits::IsTraitSpecialized<traits::AsType, typename TSoA::record_type::template field_for<RAs...>>::
                    value))
        {
            if constexpr(traits::IsTraitSpecialized<
                             traits::AsType,
                             typename TSoA::record_type::template field_for<RAs...>>::value)
            {
                return traits::AsType<typename TSoA::record_type::template field_for<RAs...>>{}(*this);
            }
            else // is a leaf
            {
                return *(*this);
            }
        }

        [[nodiscard]] constexpr decltype(auto) get() const requires(
            (sizeof...(RAs) == 1)
            && (TSoA::record_type::template isLeaf<RAs...>()
                || traits::IsTraitSpecialized<traits::AsType, typename TSoA::record_type::template field_for<RAs...>>::
                    value))
        {
            if constexpr(traits::IsTraitSpecialized<
                             traits::AsType,
                             typename TSoA::record_type::template field_for<RAs...>>::value)
            {
                return traits::AsType<typename TSoA::record_type::template field_for<RAs...>>{}(*this);
            }
            else // is a leaf
            {
                return *(*this);
            }
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
    //         typename TSoA::record_type,
    //         typename TSoA::record_type::template value_type_for<
    //             typename ToPath<std::tuple_element_t<0, Tuple<RAs...>>>::type>>;

    //     // We inspect the structure of the record currently pointed to by this View
    //     using Fields = typename CurrentRecordType::fields_tuple_type;

    //     // Fold expression to apply function to all children
    //     [&]<typename... Fs>(Tuple<Fs...>) { (func((*this)[typename Fs::tag_type{}]), ...); }(Fields{});
    // }

} // namespace llama_lite
