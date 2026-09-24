/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"
#include "spmacc/particles/spatial/InteractionQuery.hpp"

#include <pmacc/eventSystem/events/EventTask.hpp>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace spearhed::detail
{
    template<typename T, typename T_Tuple>
    struct TupleContains;

    template<typename T, typename... T_Elements>
    struct TupleContains<T, std::tuple<T_Elements...>> : std::bool_constant<(std::same_as<T, T_Elements> || ...)>
    {
    };

    template<typename T, typename T_Tuple>
    inline constexpr bool tupleContains = TupleContains<T, T_Tuple>::value;

    template<typename T_Tuple, typename T>
    struct TupleAppendUnique;

    template<typename... T_Elements, typename T>
    struct TupleAppendUnique<std::tuple<T_Elements...>, T>
    {
        using type = std::conditional_t<
            tupleContains<T, std::tuple<T_Elements...>>,
            std::tuple<T_Elements...>,
            std::tuple<T_Elements..., T>>;
    };

    template<typename T_Accumulated, typename T_Tuple>
    struct TupleAppendAll;

    template<typename T_Accumulated>
    struct TupleAppendAll<T_Accumulated, std::tuple<>>
    {
        using type = T_Accumulated;
    };

    template<typename T_Accumulated, typename T_Head, typename... T_Tail>
    struct TupleAppendAll<T_Accumulated, std::tuple<T_Head, T_Tail...>>
    {
        using type =
            typename TupleAppendAll<typename TupleAppendUnique<T_Accumulated, T_Head>::type, std::tuple<T_Tail...>>::
                type;
    };

    template<typename T_Left, typename T_Right>
    using TupleUnion = typename TupleAppendAll<T_Left, T_Right>::type;

    template<typename T_Tuple>
    inline constexpr bool tupleIsUnique = std::same_as<typename TupleAppendAll<std::tuple<>, T_Tuple>::type, T_Tuple>;

    template<typename T_Tuple, typename T_RegistrySpecies>
    struct TupleIsSpeciesSubset;

    template<typename... T_Targets, typename... T_RegistrySpecies>
    struct TupleIsSpeciesSubset<std::tuple<T_Targets...>, std::tuple<T_RegistrySpecies...>>
        : std::bool_constant<
              (pmacc::spearhed::SpeciesTag<T_Targets> && ...)
              && (tupleContains<T_Targets, std::tuple<T_RegistrySpecies...>> && ...)>
    {
    };

    template<typename T_Tuple, typename T_RegistrySpecies>
    inline constexpr bool tupleIsSpeciesSubset = TupleIsSpeciesSubset<T_Tuple, T_RegistrySpecies>::value;

    template<typename T_Setup, pmacc::spearhed::SpeciesRegistryTag T_Registry, typename = void>
    struct DensityTargetsFor
    {
        using type = pmacc::spearhed::SpeciesWithPred<
            decltype(pmacc::spearhed::pred::
                         withAllRoles<pmacc::spearhed::roles::Movable, pmacc::spearhed::roles::Thermodynamic>),
            typename T_Registry::List>;
    };

    template<typename T_Setup, pmacc::spearhed::SpeciesRegistryTag T_Registry>
    struct DensityTargetsFor<T_Setup, T_Registry, std::void_t<typename T_Setup::DensityTargets>>
    {
        using type = typename T_Setup::DensityTargets;
    };

    template<typename T_Setup, pmacc::spearhed::SpeciesRegistryTag T_Registry, typename = void>
    struct HydroTargetsFor
    {
        using type = pmacc::spearhed::SpeciesWithPred<
            decltype(pmacc::spearhed::pred::
                         withAllRoles<pmacc::spearhed::roles::Movable, pmacc::spearhed::roles::Thermodynamic>),
            typename T_Registry::List>;
    };

    template<typename T_Setup, pmacc::spearhed::SpeciesRegistryTag T_Registry>
    struct HydroTargetsFor<T_Setup, T_Registry, std::void_t<typename T_Setup::HydroTargets>>
    {
        using type = typename T_Setup::HydroTargets;
    };
} // namespace spearhed::detail

namespace spearhed
{
    /**
     * @brief Configured target packs for the density and hydro phases.
     *
     * Setups can override either pack with a @c std::tuple of species tags. The
     * default remains the old behaviour: every movable thermodynamic species is
     * both a density and hydro target. A source not in DensityTargets therefore
     * has the explicit contract that its density is fixed or prepared elsewhere.
     */
    template<typename T_Setup, pmacc::spearhed::SpeciesRegistryTag T_Registry>
    struct InteractionTargetSets
    {
        using DensityTargets = typename detail::DensityTargetsFor<T_Setup, T_Registry>::type;
        using HydroTargets = typename detail::HydroTargetsFor<T_Setup, T_Registry>::type;
        using Targets = detail::TupleUnion<DensityTargets, HydroTargets>;

        static_assert(
            detail::tupleIsSpeciesSubset<DensityTargets, typename T_Registry::List>
                && detail::tupleIsUnique<DensityTargets>,
            "Setup::DensityTargets must be a duplicate-free tuple of registered species");
        static_assert(
            detail::tupleIsSpeciesSubset<HydroTargets, typename T_Registry::List>
                && detail::tupleIsUnique<HydroTargets>,
            "Setup::HydroTargets must be a duplicate-free tuple of registered species");
    };

    /**
     * @brief Durable frame-index cache for one configured target pack.
     *
     * Frame indices depend only on frame-list topology, not on an interaction
     * plan or spatial mapping generation. This cache is deliberately kept
     * outside a timestep's TargetWorkSet, so refreshIfStale() is the only path
     * that rebuilds an index after the initial construction.
     */
    template<pmacc::spearhed::SpeciesRegistryTag T_Registry, typename T_Targets>
    class TargetFrameIndexCache;

    template<pmacc::spearhed::SpeciesRegistryTag T_Registry, pmacc::spearhed::SpeciesTag... T_Targets>
    class TargetFrameIndexCache<T_Registry, std::tuple<T_Targets...>>
    {
    public:
        [[nodiscard]] auto& forTarget(auto& target)
        {
            using T_Species = typename std::remove_cvref_t<decltype(target)>::Species;
            constexpr std::size_t index = indexFor<T_Species>();
            auto& cached = std::get<index>(m_indices);
            if(!cached)
                cached.emplace(target);
            else
                cached->refreshIfStale(target);
            return *cached;
        }

    private:
        template<pmacc::spearhed::SpeciesTag T_Species, std::size_t T_Index = 0u>
        static consteval std::size_t indexFor()
        {
            static_assert(T_Index < sizeof...(T_Targets), "target species is absent from this frame-index cache");
            if constexpr(T_Index == sizeof...(T_Targets))
                return 0u;
            else if constexpr(std::same_as<T_Species, std::tuple_element_t<T_Index, std::tuple<T_Targets...>>>)
                return T_Index;
            else
                return indexFor<T_Species, T_Index + 1u>();
        }

        std::tuple<
            std::optional<pmacc::spearhed::FrameIndexBuffer<typename T_Registry::template PRType<T_Targets>>>...>
            m_indices;
    };

    /** @brief One target's plan, prepared handle, reusable frame index, and phase completion events. */
    template<pmacc::spearhed::SpeciesTag T_Species, typename T_Store, typename T_Prepared, typename T_Plan>
    struct TargetWork
    {
        using Species = T_Species;

        T_Store* target;
        T_Prepared targetPrepared;
        T_Plan plan;
        pmacc::spearhed::FrameIndexBuffer<typename T_Store::ParticleRegionType>* frameIndex;
        pmacc::EventTask densityDone;
        pmacc::EventTask hydroDone;
    };

    /**
     * @brief Heterogeneous target work that remains alive across density and hydro phases.
     *
     * Candidate plans are target-specific but source mappings have already been
     * prepared by the owning decomposition groups. The set owns the plans and
     * prepared target handles; frame indices are borrowed from the durable cache.
     */
    template<typename T_DensityTargets, typename T_HydroTargets, typename... T_Work>
    class TargetWorkSet
    {
    public:
        explicit TargetWorkSet(std::tuple<T_Work...> work) : m_work(std::move(work))
        {
        }

        template<typename Fn>
        [[nodiscard]] pmacc::EventTask launchDensity(Fn&& launch)
        {
            return launchFor<T_DensityTargets>(
                [&](auto& work)
                {
                    work.densityDone = launch(work);
                    return work.densityDone;
                });
        }

        template<typename Fn>
        [[nodiscard]] pmacc::EventTask launchHydro(Fn&& launch)
        {
            return launchFor<T_HydroTargets>(
                [&](auto& work)
                {
                    work.hydroDone = launch(work);
                    return work.hydroDone;
                });
        }

    private:
        template<typename T_Targets, typename Fn>
        [[nodiscard]] pmacc::EventTask launchFor(Fn&& launch)
        {
            pmacc::EventTask done;
            std::apply(
                [&](auto&... work)
                {
                    (
                        [&]
                        {
                            using Work = std::remove_cvref_t<decltype(work)>;
                            if constexpr(detail::tupleContains<typename Work::Species, T_Targets>)
                                done += launch(work);
                        }(),
                        ...);
                },
                m_work);
            return done;
        }

        std::tuple<T_Work...> m_work;
    };

    namespace detail
    {
        template<
            pmacc::spearhed::SpeciesTag T_Target,
            typename T_Groups,
            typename T_Cache,
            typename T_Query,
            pmacc::spearhed::SpeciesTag... T_Sources>
        [[nodiscard]] auto makeTargetWork(
            T_Groups& groups,
            T_Cache& frameIndices,
            T_Query const& query,
            std::tuple<T_Sources...>)
        {
            auto& target = groups.storeFor(T_Target{});
            auto targetPrepared = groups.preparedFor(target);
            auto plan = pmacc::spearhed::makeInteractionPlan(
                targetPrepared,
                query,
                groups.preparedFor(groups.storeFor(T_Sources{}))...);
            auto& frameIndex = frameIndices.forTarget(target);

            using TargetStore = std::remove_cvref_t<decltype(target)>;
            using TargetPrepared = std::remove_cvref_t<decltype(targetPrepared)>;
            using Plan = std::remove_cvref_t<decltype(plan)>;
            return TargetWork<T_Target, TargetStore, TargetPrepared, Plan>{
                &target,
                std::move(targetPrepared),
                std::move(plan),
                &frameIndex,
                {},
                {}};
        }

        template<
            typename T_DensityTargets,
            typename T_HydroTargets,
            typename T_Groups,
            typename T_Cache,
            typename T_Query,
            pmacc::spearhed::SpeciesTag... T_Targets,
            pmacc::spearhed::SpeciesTag... T_Sources>
        [[nodiscard]] auto makeTargetWorkSet(
            T_Groups& groups,
            T_Cache& frameIndices,
            T_Query const& query,
            std::tuple<T_Targets...>,
            std::tuple<T_Sources...> sources)
        {
            return TargetWorkSet<
                T_DensityTargets,
                T_HydroTargets,
                decltype(makeTargetWork<T_Targets>(groups, frameIndices, query, sources))...>{
                std::tuple{makeTargetWork<T_Targets>(groups, frameIndices, query, sources)...}};
        }

        template<typename T_Groups, typename Fn, pmacc::spearhed::SpeciesTag... T_Targets>
        void forEachTargetStore(T_Groups& groups, Fn&& fn, std::tuple<T_Targets...>)
        {
            (fn(groups.storeFor(T_Targets{})), ...);
        }
    } // namespace detail

    /**
     * @brief Build one work item per density or hydro target from already-prepared groups.
     *
     * No preparation happens here. Every source role in the configured registry is
     * represented, including zero-sized stores, so the work-set shape depends on
     * configuration rather than the runtime subset of present particle buffers.
     */
    template<typename T_DensityTargets, typename T_HydroTargets, typename T_Groups, typename T_Cache, typename T_Query>
    [[nodiscard]] auto makeTargetWorkSet(T_Groups& groups, T_Cache& frameIndices, T_Query const& query)
    {
        using Sources = pmacc::spearhed::SpeciesWithPred<
            decltype(pmacc::spearhed::pred::withRole<pmacc::spearhed::roles::Source>),
            typename T_Groups::Registry::List>;
        using Targets = detail::TupleUnion<T_DensityTargets, T_HydroTargets>;
        return detail::makeTargetWorkSet<T_DensityTargets, T_HydroTargets>(
            groups,
            frameIndices,
            query,
            Targets{},
            Sources{});
    }

    /** @brief Visit configured target stores without querying runtime buffer presence. */
    template<typename T_Targets, typename T_Groups, typename Fn>
    void forEachConfiguredTargetStore(T_Groups& groups, Fn&& fn)
    {
        detail::forEachTargetStore(groups, std::forward<Fn>(fn), T_Targets{});
    }
} // namespace spearhed
