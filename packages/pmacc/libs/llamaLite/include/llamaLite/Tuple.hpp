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

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...>& tuple) noexcept;

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...> const& tuple) noexcept;

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...>&& tuple) noexcept;

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...> const&& tuple) noexcept;

    namespace detail
    {
        template<std::size_t I, typename T>
        struct TupleLeaf
        {
            using type = T;
            [[no_unique_address]] T value;

            constexpr TupleLeaf() requires(std::is_default_constructible_v<T>)
            = default;

            template<typename U>
            requires(std::is_constructible_v<T, U&&>)
            constexpr explicit(!std::is_convertible_v<U&&, T>)
                TupleLeaf(U&& arg) noexcept(std::is_nothrow_constructible_v<T, U&&>)
                : value(std::forward<U>(arg))
            {
            }
        };

        template<typename IndexSequence, typename... T_Args>
        struct TupleStorage;

        template<std::size_t... Is, typename... T_Args>
        struct TupleStorage<std::index_sequence<Is...>, T_Args...> : TupleLeaf<Is, T_Args>...
        {
            template<typename... T_CArgs>
            requires(
                sizeof...(T_CArgs) == sizeof...(T_Args) && sizeof...(T_Args) > 0
                && (std::is_constructible_v<T_Args, T_CArgs&&> && ...))
            constexpr TupleStorage(T_CArgs&&... us) noexcept(
                (std::is_nothrow_constructible_v<T_Args, T_CArgs&&> && ...))
                : TupleLeaf<Is, T_Args>{std::forward<T_CArgs>(us)}...
            {
            }

            constexpr TupleStorage() requires(std::is_default_constructible_v<T_Args> && ...)
            = default;
        };

        template<typename IndexSequence, bool HasReferences, typename... T_Args>
        struct TupleImpl;

        template<typename IndexSequence, typename... T_Args>
        struct TupleImpl<IndexSequence, false, T_Args...> : TupleStorage<IndexSequence, T_Args...>
        {
            using Base = TupleStorage<IndexSequence, T_Args...>;
            using Base::Base;

            constexpr TupleImpl() requires(std::is_default_constructible_v<T_Args> && ...)
            = default;
            constexpr TupleImpl(TupleImpl const&) = default;
            constexpr TupleImpl(TupleImpl&&) = default;
            constexpr TupleImpl& operator=(TupleImpl const&) = default;
            constexpr TupleImpl& operator=(TupleImpl&&) = default;
        };

        template<std::size_t... Is, typename... T_Args>
        struct TupleImpl<std::index_sequence<Is...>, true, T_Args...>
            : TupleStorage<std::index_sequence<Is...>, T_Args...>
        {
            using Base = TupleStorage<std::index_sequence<Is...>, T_Args...>;
            using Base::Base;

            constexpr TupleImpl() requires(std::is_default_constructible_v<T_Args> && ...)
            = default;
            constexpr TupleImpl(TupleImpl const&) = default;
            constexpr TupleImpl(TupleImpl&&) = default;

            constexpr TupleImpl& operator=(TupleImpl const& other) noexcept(
                (std::is_nothrow_assignable_v<T_Args&, T_Args const&> && ...))
                requires((std::is_assignable_v<T_Args&, T_Args const&> && ...))
            {
                (static_cast<void>(
                     static_cast<TupleLeaf<Is, T_Args>&>(*this).value
                     = static_cast<T_Args const&>(static_cast<TupleLeaf<Is, T_Args> const&>(other).value)),
                 ...);
                return *this;
            }

            constexpr TupleImpl& operator=(TupleImpl&& other) noexcept(
                (std::is_nothrow_assignable_v<T_Args&, T_Args&&> && ...))
                requires((std::is_assignable_v<T_Args&, T_Args&&> && ...))
            {
                (static_cast<void>(
                     static_cast<TupleLeaf<Is, T_Args>&>(*this).value
                     = static_cast<T_Args&&>(static_cast<TupleLeaf<Is, T_Args>&&>(other).value)),
                 ...);
                return *this;
            }
        };
    } // namespace detail

    /** basic tuple implementation
     *
     * Unlike a standard tuple, default construction doesnt value initialize all members.
     * So Tuple t; would leave all primitive types uninitialized
     * Tuple t{}; would do value initialization
     *
     * Value tuples remain trivially copyable when all elements are trivially copyable. Tuples containing references
     * assign through those references and are not generally trivially copyable. Use @see alpaka::apply or
     * @see llama_lite::tuple::apply to apply an operation to the tuple.
     */
    template<typename... T_Args>
    struct Tuple
        : detail::
              TupleImpl<std::make_index_sequence<sizeof...(T_Args)>, (std::is_reference_v<T_Args> || ...), T_Args...>
    {
        using StdTuple = std::tuple<T_Args...>;
        using Base = detail::
            TupleImpl<std::make_index_sequence<sizeof...(T_Args)>, (std::is_reference_v<T_Args> || ...), T_Args...>;

        template<typename... T_CArgs>
        requires(
            sizeof...(T_Args) == sizeof...(T_CArgs) && sizeof...(T_Args) > 0
            && (!std::is_same_v<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<T_CArgs...>>>, Tuple>)
            && (std::is_constructible_v<T_Args, T_CArgs&&> && ...))
        constexpr Tuple(T_CArgs&&... us) noexcept((std::is_nothrow_constructible_v<T_Args, T_CArgs&&> && ...))
            : Base(std::forward<T_CArgs>(us)...)
        {
        }

        constexpr Tuple() requires(std::is_default_constructible_v<T_Args> && ...)
        = default;

        template<typename... T_OtherArgs>
        requires(
            sizeof...(T_Args) == sizeof...(T_OtherArgs) && !std::is_same_v<Tuple, Tuple<T_OtherArgs...>>
            && (std::is_assignable_v<T_Args&, T_OtherArgs const&> && ...))
        constexpr Tuple& operator=(Tuple<T_OtherArgs...> const& other) noexcept(
            (std::is_nothrow_assignable_v<T_Args&, T_OtherArgs const&> && ...))
        {
            assignFrom(other, std::index_sequence_for<T_Args...>{});
            return *this;
        }

        template<typename... T_OtherArgs>
        requires(
            sizeof...(T_Args) == sizeof...(T_OtherArgs)
            && !std::is_same_v<Tuple, Tuple<T_OtherArgs...>> && (std::is_assignable_v<T_Args&, T_OtherArgs&&> && ...))
        constexpr Tuple& operator=(Tuple<T_OtherArgs...>&& other) noexcept(
            (std::is_nothrow_assignable_v<T_Args&, T_OtherArgs&&> && ...))
        {
            assignFrom(std::move(other), std::index_sequence_for<T_Args...>{});
            return *this;
        }

    private:
        template<typename T_Other, std::size_t... T_Is>
        constexpr void assignFrom(T_Other&& other, std::index_sequence<T_Is...>) noexcept(
            (std::is_nothrow_assignable_v<T_Args&, decltype(llama_lite::get<T_Is>(std::forward<T_Other>(other)))>
             && ...))
        {
            (static_cast<void>(llama_lite::get<T_Is>(*this) = llama_lite::get<T_Is>(std::forward<T_Other>(other))),
             ...);
        }
    };

    /** Deduction decays argument types, as with std::tuple CTAD. Use tie or forwardAsTuple for references. */
    template<typename... T_Args>
    Tuple(T_Args&&...) -> Tuple<std::decay_t<T_Args>...>;

    /** Create a value tuple, decaying arguments and unwrapping std::reference_wrapper.
     *
     * Lvalues are copied into the tuple. Use `makeTuple(std::ref(x))` to store an explicit reference.
     */
    constexpr auto makeTuple(auto&&... args)
    {
        return Tuple<std::unwrap_ref_decay_t<decltype(args)>...>{LL_FORWARD(args)...};
    }

    /** Create a tuple of lvalue references, similar to std::tie. Assignment writes through the references. */
    template<typename... T_Args>
    constexpr auto tie(T_Args&... args) noexcept
    {
        return Tuple<T_Args&...>{args...};
    }

    /** Create a tuple of forwarding references, similar to std::forward_as_tuple.
     *
     * This does not extend the lifetime of referenced objects. Do not retain the result when any argument is a
     * temporary.
     */
    template<typename... T_Args>
    constexpr auto forwardAsTuple(T_Args&&... args) noexcept
    {
        return Tuple<T_Args&&...>{std::forward<T_Args>(args)...};
    }

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...>& tuple) noexcept
    {
        static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
        using Element = std::tuple_element_t<I, typename Tuple<T_Args...>::StdTuple>;
        using Leaf = detail::TupleLeaf<I, Element>;
        return static_cast<Element&>(static_cast<Leaf&>(tuple).value);
    }

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...> const& tuple) noexcept
    {
        static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
        using Element = std::tuple_element_t<I, typename Tuple<T_Args...>::StdTuple>;
        using Leaf = detail::TupleLeaf<I, Element>;
        return static_cast<std::add_const_t<Element>&>(static_cast<Leaf const&>(tuple).value);
    }

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...>&& tuple) noexcept
    {
        static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
        using Element = std::tuple_element_t<I, typename Tuple<T_Args...>::StdTuple>;
        using Leaf = detail::TupleLeaf<I, Element>;
        return static_cast<Element&&>(static_cast<Leaf&&>(tuple).value);
    }

    template<std::size_t I, typename... T_Args>
    constexpr decltype(auto) get(Tuple<T_Args...> const&& tuple) noexcept
    {
        static_assert(I < sizeof...(T_Args), "Index is outside of the allowed range.");
        using Element = std::tuple_element_t<I, typename Tuple<T_Args...>::StdTuple>;
        using ConstElement = std::conditional_t<std::is_reference_v<Element>, Element, Element const>;
        using Leaf = detail::TupleLeaf<I, Element>;
        return static_cast<ConstElement&&>(static_cast<Leaf const&&>(tuple).value);
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
        using llama_lite::get;

        namespace detail
        {
            template<typename T_Func, typename T_TupleLike, std::size_t... T_idx>
            constexpr decltype(auto) applyImpl(T_Func&& func, T_TupleLike&& tuple, std::index_sequence<T_idx...>)
            {
                return std::forward<T_Func>(func)(get<T_idx>(std::forward<T_TupleLike>(tuple))...);
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

    namespace detail
    {
        template<std::size_t T_Index, typename T_Inputs, typename T_Func, typename... T_Elements>
        constexpr decltype(auto) tupleCatApply(T_Inputs&& inputs, T_Func&& func, T_Elements&&... elements)
        {
            if constexpr(T_Index == std::tuple_size_v<std::remove_cvref_t<T_Inputs>>)
            {
                return std::forward<T_Func>(func)(std::forward<T_Elements>(elements)...);
            }
            else
            {
                return tuple::apply(
                    [&]<typename... T_Current>(T_Current&&... current) -> decltype(auto)
                    {
                        return tupleCatApply<T_Index + 1>(
                            std::forward<T_Inputs>(inputs),
                            std::forward<T_Func>(func),
                            std::forward<T_Elements>(elements)...,
                            std::forward<T_Current>(current)...);
                    },
                    std::get<T_Index>(std::forward<T_Inputs>(inputs)));
            }
        }
    } // namespace detail

    /** Concatenate llamaLite tuples into one tuple, forwarding each element once into the result.
     *
     * The explicit return type allows type-only use in unevaluated contexts without constructing the input elements.
     * This does not guarantee the same compile-time cost as using @ref ConcatTuples directly. References already
     * present in tuple element types remain references and retain their original lifetime requirements.
     */
    template<typename... T_Tuples>
    requires((llama_lite::isSpecializationOf_v<std::remove_cvref_t<T_Tuples>, Tuple> && ...))
    constexpr typename ConcatTuples<std::remove_cvref_t<T_Tuples>...>::type tupleCat(T_Tuples&&... tuples)
    {
        using Result = typename ConcatTuples<std::remove_cvref_t<T_Tuples>...>::type;
        auto inputs = std::forward_as_tuple(std::forward<T_Tuples>(tuples)...);
        return detail::tupleCatApply<0>(
            std::move(inputs),
            []<typename... T_Elements>(T_Elements&&... elements) -> Result
            { return Result{std::forward<T_Elements>(elements)...}; });
    }

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
