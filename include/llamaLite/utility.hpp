// Copyright 2025 Tapish Narwal, René Widera
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.


#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace llama_lite
{
    namespace ll = llama_lite;

/** Perfectly forward an instance as argument. */
#define LL_FORWARD(instance) std::forward<decltype(instance)>(instance)

/** Get the type of instance
 *
 * References will be removed which is often required because traits are mostly defined for the type only.
 */
#define LL_TYPEOF(...) std::remove_cvref_t<decltype(__VA_ARGS__)>

    namespace traits
    {

        /**
         * Type trait to check if a type is a specialization of a template
         * Similar to P2078 - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2098r0.pdf
         * Note that this cant be used with template types which have NTTPs
         * To fix this limitation we need PR1985 Universal Template Parameters
         * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1985r3.pdf
         */

        template<typename, template<typename...> typename>
        struct IsSpecializationOf : std::false_type
        {
        };

        template<template<typename...> typename Template, typename... Args>
        struct IsSpecializationOf<Template<Args...>, Template> : std::true_type
        {
        };

    } // namespace traits

    template<typename T, template<typename...> typename Template>
    inline constexpr bool isSpecializationOf_v = traits::IsSpecializationOf<T, Template>::value;

    namespace concepts
    {
        template<typename T, template<typename...> typename Template>
        concept SpecializationOf = isSpecializationOf_v<T, Template>;

    } // namespace concepts

    template<typename Tag, typename... Set>
    concept IsInSet = (... || std::is_same_v<Tag, Set>);

    template<typename T>
    struct SingleElementPack
    {
        using type = T;
    };

    template<typename QueryT, typename... SearchTs>
    static consteval uint32_t countTypeOccurrences()
    {
        return ((std::is_same_v<QueryT, SearchTs> ? 1 : 0) + ... + 0);
    }


} // namespace llama_lite
