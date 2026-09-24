/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/NeighbourBundle.hpp"
#include "spmacc/particles/spatial/InteractionEntry.hpp"
#include "spmacc/particles/spatial/InteractionQuery.hpp"
#include "spmacc/particles/spatial/PairCandidateProvider.hpp"
#include "spmacc/particles/spatial/PreparedRegionSet.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    namespace detail
    {
        void makeCandidateProviderImpl() = delete;
    } // namespace detail

    /** @brief Select a candidate provider from the target/source mapping pair. */
    template<PreparedRegionSet T_Target, typename T_Radius, PreparedRegionSet T_Source>
    [[nodiscard]] constexpr decltype(auto) makeCandidateProvider(
        T_Target const& target,
        InteractionQuery<T_Radius> const& query,
        T_Source const& source)
    {
        using detail::makeCandidateProviderImpl;
        return makeCandidateProviderImpl(
            typename std::remove_cvref_t<T_Target>::MappingTag{},
            typename std::remove_cvref_t<T_Source>::MappingTag{},
            target,
            query,
            source);
    }

    /** @brief An owning interaction bundle tied to prepared decomposition states. */
    template<IsNeighbourBundle T_Bundle, PreparedRegionSet... T_Prepared>
    struct InteractionPlan
    {
        T_Bundle bundle;
        std::tuple<T_Prepared...> prepared;

        void assertCurrent() const
        {
            std::apply([](auto const&... state) { (state.assertCurrent(), ...); }, prepared);
        }

        template<typename Fn>
        void forEachEntry(Fn&& fn)
        {
            assertCurrent();
            bundle.forEachEntry(std::forward<Fn>(fn));
        }

        template<typename Fn>
        void forEachEntry(Fn&& fn) const
        {
            assertCurrent();
            bundle.forEachEntry(std::forward<Fn>(fn));
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn)
        {
            assertCurrent();
            bundle.forEachDeviceView(std::forward<Fn>(fn));
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn) const
        {
            assertCurrent();
            bundle.forEachDeviceView(std::forward<Fn>(fn));
        }

        auto makeDeviceViewTuple()
        {
            assertCurrent();
            return bundle.makeDeviceViewTuple();
        }

        auto makeDeviceViewTuple() const
        {
            assertCurrent();
            return bundle.makeDeviceViewTuple();
        }

        static constexpr std::size_t size()
        {
            return T_Bundle::size();
        }

        template<typename Pred>
        auto select(Pred predicate = {})
        {
            assertCurrent();
            return bundle.select(predicate);
        }

        template<typename Pred>
        auto select(Pred predicate = {}) const
        {
            assertCurrent();
            return bundle.select(predicate);
        }

        auto selectByRole(RoleTag auto role)
        {
            assertCurrent();
            return bundle.selectByRole(role);
        }

        auto selectByRole(RoleTag auto role) const
        {
            assertCurrent();
            return bundle.selectByRole(role);
        }

        auto& bySpecies(SpeciesTag auto species)
        {
            assertCurrent();
            return bundle.bySpecies(species);
        }

        auto const& bySpecies(SpeciesTag auto species) const
        {
            assertCurrent();
            return bundle.bySpecies(species);
        }
    };

    /** @brief Assemble one heterogeneous candidate provider per prepared source. */
    template<PreparedRegionSet T_Target, typename T_Radius, PreparedRegionSet... T_Sources>
    [[nodiscard]] auto makeInteractionPlan(
        T_Target const& target,
        InteractionQuery<T_Radius> const& query,
        T_Sources const&... sources)
    {
        target.assertCurrent();
        if constexpr(sizeof...(sources) > 0u)
            (sources.assertCurrent(), ...);

        // Construct each owning provider before moving entries into the bundle. Function-argument
        // pack evaluation order is unspecified, while CSR construction launches maintenance work.
        auto entries
            = std::tuple{InteractionEntry{&sources.store(), makeCandidateProvider(target, query, sources)}...};
        auto bundle
            = std::apply([](auto&&... entry) { return makeNeighbourBundle(std::move(entry)...); }, std::move(entries));
        using Plan = InteractionPlan<
            std::remove_cvref_t<decltype(bundle)>,
            std::remove_cvref_t<T_Target>,
            std::remove_cvref_t<T_Sources>...>;
        return Plan{std::move(bundle), std::tuple{target, sources...}};
    }

    /** @brief Compatibility overload for callers that pass only an interaction radius. */
    template<PreparedRegionSet T_Target, typename T_Radius, PreparedRegionSet... T_Sources>
    requires(!requires(T_Radius const& radius) { radius.radius; })
    [[nodiscard]] auto makeInteractionPlan(T_Target const& target, T_Radius radius, T_Sources const&... sources)
    {
        return makeInteractionPlan(target, InteractionQuery{radius}, sources...);
    }

    namespace detail
    {
        template<IsNeighbourBundle Bundle, PreparedRegionSet... Prepared>
        struct IsNeighbourBundleImpl<InteractionPlan<Bundle, Prepared...>> : std::true_type
        {
        };
    } // namespace detail
} // namespace pmacc::spearhed
