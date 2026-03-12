// Copyright 2025 Tapish Narwal, René Widera
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/utility.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace llama_lite
{
    template<typename... T_Args>
    struct Tuple;

    namespace detail
    {
        template<std::size_t I, typename T>
        struct TupleLeaf
        {
            using type = T;
            [[no_unique_address]] T value;
        };

        template<typename IndexSequence, typename... T_Args>
        struct TupleImpl;

        template<std::size_t... Is, typename... T_Args>
        struct TupleImpl<std::index_sequence<Is...>, T_Args...> : TupleLeaf<Is, T_Args>...
        {
            template<typename... T_CArgs>
            constexpr TupleImpl(T_CArgs&&... us) noexcept((std::is_nothrow_constructible_v<T_Args, T_CArgs&&> && ...))
                : TupleLeaf<Is, T_Args>{std::forward<T_CArgs>(us)}...
            {
            }

            constexpr TupleImpl() requires(std::is_default_constructible_v<T_Args> && ...)
            = default;
        };
    } // namespace detail

    /** basic tuple implementation
     *
     * Unlike a standard tuple, default construction doesnt value initialize all members.
     * So Tuple t; would leave all primitive types uninitialized
     * Tuple t{}; would do value initialization
     *
     * This class is trivially copyable if all members are trivially copable too and can therefore used for a
     * collection to pass arguments into kernels. You should use @see alpaka::apply to apply operation to the tuple.
     */
    template<typename... T_Args>
    struct Tuple : detail::TupleImpl<std::make_index_sequence<sizeof...(T_Args)>, T_Args...>
    {
        using StdTuple = std::tuple<T_Args...>;
        using Base = detail::TupleImpl<std::make_index_sequence<sizeof...(T_Args)>, T_Args...>;

        template<typename... T_CArgs>
        requires(
            sizeof...(T_Args) == sizeof...(T_CArgs) && sizeof...(T_Args) > 0
            && (!std::is_same_v<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<T_CArgs...>>>, Tuple>)
            && (std::is_constructible_v<T_Args, T_CArgs &&> && ...))
        constexpr Tuple(T_CArgs&&... us) noexcept((std::is_nothrow_constructible_v<T_Args, T_CArgs&&> && ...))
            : Base(std::forward<T_CArgs>(us)...)
        {
        }

        constexpr Tuple() requires(std::is_default_constructible_v<T_Args> && ...)
        = default;

        /** get element by index
         *
         * @tparam I index which should not be larger than the number of elements -1
         * @{
         */
        template<size_t I>
        constexpr auto const& get() const
        {
            static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
            return static_cast<detail::TupleLeaf<I, std::tuple_element_t<I, StdTuple>> const&>(*this).value;
        }

        template<size_t I>
        constexpr auto& get()
        {
            static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
            return static_cast<detail::TupleLeaf<I, std::tuple_element_t<I, StdTuple>>&>(*this).value;
        }

        /** @} */
    };

    template<typename... T_Args>
    Tuple(T_Args&&...) -> Tuple<T_Args...>;

    constexpr auto makeTuple(auto&&... args)
    {
        return Tuple{LL_FORWARD(args)...};
    }

    // Flatten multiple Tuples into one for metaprogramming with Tuple types which hold tags
    template<typename... Tuples>
    struct ConcatTuples;

    template<>
    struct ConcatTuples<>
    {
        using type = Tuple<>;
    };

    template<typename... Ts>
    struct ConcatTuples<Tuple<Ts...>>
    {
        using type = Tuple<Ts...>;
    };

    template<typename... T1, typename... T2, typename... Rest>
    struct ConcatTuples<Tuple<T1...>, Tuple<T2...>, Rest...>
    {
        using type = typename ConcatTuples<Tuple<T1..., T2...>, Rest...>::type;
    };

    namespace tuple
    {
        template<size_t T_idx>
        constexpr decltype(auto) get(auto&& t) noexcept requires(ll::isSpecializationOf_v<LL_TYPEOF(t), Tuple>)
        {
            return LL_FORWARD(t).template get<T_idx>();
        }

        namespace detail
        {
            template<typename T_Func, typename T_TupleLike, std::size_t... T_idx>
            constexpr decltype(auto) applyImpl(T_Func&& func, T_TupleLike&& tuple, std::index_sequence<T_idx...>)
            {
                return func(get<T_idx>(std::forward<T_TupleLike>(tuple))...);
            }
        } // namespace detail

        /** Applies a function to the elements of a tuple-like object.
         *
         * This function forwards the function and the tuple-like object, and uses an index sequence to unpack the
         * tuple.
         *
         * @param func The function to apply.
         * @param tuple The tuple-like object containing the arguments for the function.
         * @return The result of applying the function to the elements of the tuple-like object.
         */
        template<typename T_Func, typename T_TupleLike>
        constexpr decltype(auto) apply(T_Func&& func, T_TupleLike&& tuple)
        {
            /** @attention Do not use std::tuple_size_v here because it results in compile issues with gcc11.4 */
            return detail::applyImpl(
                std::forward<T_Func>(func),
                std::forward<T_TupleLike>(tuple),
                std::make_index_sequence<std::tuple_size<std::decay_t<T_TupleLike>>::value>{});
        }

    } // namespace tuple

} // namespace llama_lite

namespace std
{
    // Specialization of tuple_size for our custom Tuple
    template<typename... T_Args>
    struct tuple_size<llama_lite::Tuple<T_Args...>> : std::integral_constant<std::size_t, sizeof...(T_Args)>
    {
    };

    template<std::size_t I, typename... T_Args>
    struct tuple_element<I, llama_lite::Tuple<T_Args...>>
    {
        using type = typename std::tuple_element_t<I, typename llama_lite::Tuple<T_Args...>::StdTuple>;
    };
} // namespace std
