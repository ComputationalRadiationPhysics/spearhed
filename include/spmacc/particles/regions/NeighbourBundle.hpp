/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spmacc/particles/regions/RegionRole.hpp"

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>

#include <functional>
#include <tuple>
#include <type_traits>

#include <llamaLite/utility.hpp>

namespace pmacc::spearhed
{
    // Forward declaration and concept

    template<bool Owning, typename... Entries>
    struct NeighbourBundle;

    namespace detail
    {
        template<typename T>
        struct IsNeighbourBundleImpl : std::false_type
        {
        };

        template<bool Owning, typename... Entries>
        struct IsNeighbourBundleImpl<NeighbourBundle<Owning, Entries...>> : std::true_type
        {
        };
    } // namespace detail

    /** @brief Concept satisfied by every NeighbourBundle<Owning, Entries...>. */
    template<typename T>
    concept IsNeighbourBundle = detail::IsNeighbourBundleImpl<std::remove_cvref_t<T>>::value;

    namespace detail
    {
        // deref(): unwrap reference_wrapper for viewing, identity for owning.
        template<typename T>
        constexpr decltype(auto) deref(T& entry)
        {
            return (entry);
        }

        template<typename T>
        constexpr decltype(auto) deref(std::reference_wrapper<T>& ref)
        {
            return (ref.get());
        }

        template<typename T>
        constexpr decltype(auto) deref(std::reference_wrapper<T> const& ref)
        {
            return (ref.get());
        }

        // Compile-time index filtering for a predicate.

        template<typename Pred, std::size_t Offset, typename... Entries>
        struct FilteredIndicesImpl;

        template<typename Pred, std::size_t Offset>
        struct FilteredIndicesImpl<Pred, Offset>
        {
            using type = std::index_sequence<>;
        };

        template<typename Pred, std::size_t Offset, typename Entry, typename... Rest>
        struct FilteredIndicesImpl<Pred, Offset, Entry, Rest...>
        {
            using Tail = typename FilteredIndicesImpl<Pred, Offset + 1, Rest...>::type;

            template<std::size_t... TailIs>
            static auto prependIfAccepted(std::index_sequence<TailIs...>) -> std::conditional_t<
                Pred::template value<Entry>,
                std::index_sequence<Offset, TailIs...>,
                std::index_sequence<TailIs...>>;

            using type = decltype(prependIfAccepted(Tail{}));
        };

        template<typename Pred, typename... Entries>
        using FilteredIndices = typename FilteredIndicesImpl<Pred, 0, Entries...>::type;
    } // namespace detail

    template<bool Owning, typename... Entries>
    struct NeighbourBundle
    {
        using Storage
            = std::conditional_t<Owning, std::tuple<Entries...>, std::tuple<std::reference_wrapper<Entries>...>>;

        Storage entries;

        // Iteration

        template<typename Fn>
        constexpr void forEachEntry(Fn&& fn)
        {
            std::apply([&](auto&... stored) { (fn(detail::deref(stored)), ...); }, entries);
        }

        template<typename Fn>
        constexpr void forEachEntry(Fn&& fn) const
        {
            std::apply([&](auto const&... stored) { (fn(detail::deref(stored)), ...); }, entries);
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn)
        {
            forEachEntry([&](auto& entry) { fn(entry.deviceView()); });
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn) const
        {
            forEachEntry([&](auto const& entry) { fn(entry.deviceView()); });
        }

        auto makeDeviceViewTuple()
        {
            return std::apply(
                [](auto&... stored)
                { return pmacc::memory::tuple::make_tuple(detail::deref(stored).deviceView()...); },
                entries);
        }

        auto makeDeviceViewTuple() const
        {
            return std::apply(
                [](auto const&... stored)
                { return pmacc::memory::tuple::make_tuple(detail::deref(stored).deviceView()...); },
                entries);
        }

        static constexpr std::size_t size()
        {
            return sizeof...(Entries);
        }

        // Specific entry access

        template<SpeciesTag S>
        auto& bySpecies(S = {})
        {
            constexpr std::size_t idx = llama_lite::indexOfType<S, typename Entries::Species...>();
            static_assert(idx < sizeof...(Entries), "Species not present in NeighbourBundle");
            return detail::deref(std::get<idx>(entries));
        }

        template<SpeciesTag S>
        auto const& bySpecies(S = {}) const
        {
            constexpr std::size_t idx = llama_lite::indexOfType<S, typename Entries::Species...>();
            static_assert(idx < sizeof...(Entries), "Species not present in NeighbourBundle");
            return detail::deref(std::get<idx>(entries));
        }

        template<typename Pred>
        auto select(Pred = {})
        {
            return [this]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return NeighbourBundle<false, std::tuple_element_t<Is, std::tuple<Entries...>>...>{
                    std::tuple{std::ref(detail::deref(std::get<Is>(entries)))...}};
            }(detail::FilteredIndices<Pred, Entries...>{});
        }

        template<typename Pred>
        auto select(Pred = {}) const
        {
            return [this]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                return NeighbourBundle<false, std::tuple_element_t<Is, std::tuple<Entries const...>>...>{
                    std::tuple{std::cref(detail::deref(std::get<Is>(entries)))...}};
            }(detail::FilteredIndices<Pred, Entries...>{});
        }

        // Convenience wrappers

        template<RoleTag R>
        auto selectByRole()
        {
            return select(pred::HasRole<R>{});
        }

        template<RoleTag R>
        auto selectByRole() const
        {
            return select(pred::HasRole<R>{});
        }

        template<SpeciesTag S>
        auto selectBySpecies()
        {
            return select(pred::IsSpecies<S>{});
        }

        template<SpeciesTag S>
        auto selectBySpecies() const
        {
            return select(pred::IsSpecies<S>{});
        }

        template<SpeciesTag... S>
        auto selectSpecies()
        {
            return select(pred::IsAnySpecies<S...>{});
        }

        template<SpeciesTag... S>
        auto selectSpecies() const
        {
            return select(pred::IsAnySpecies<S...>{});
        }
    };

    /** @brief Create an owning NeighbourBundle, deducing entry types from the arguments. */
    template<typename... Entries>
    [[nodiscard]] auto makeNeighbourBundle(Entries&&... entries)
    {
        static_assert(
            (!llama_lite::isSpecializationOf_v<std::decay_t<Entries>, std::reference_wrapper> && ...),
            "Pass entries by value/move; use select() or a viewing bundle for references");
        return NeighbourBundle<true, std::decay_t<Entries>...>{
            std::forward_as_tuple(std::forward<Entries>(entries)...)};
    }
} // namespace pmacc::spearhed
