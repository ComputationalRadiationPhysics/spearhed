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

#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"
#include "spmacc/particles/regions/mapping/AdaptiveSplit/Decomposition.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/Decomposition.hpp"
#include "spmacc/particles/regions/mapping/constant/Decomposition.hpp"

#include <pmacc/Environment.hpp>

#include <cassert>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    /** @brief Controls when a decomposition group's mapping is prepared. */
    enum class DecompositionGroupPreparation
    {
        EveryMotion,
        OnInvalidation
    };

    /** @brief Structural setup declaration for one concrete mapping factory. */
    template<typename T_Factory, DecompositionGroupPreparation T_Preparation, SpeciesTag... T_Species>
    struct DecompositionGroupDeclaration
    {
        using Factory = T_Factory;
        using SpeciesList = std::tuple<T_Species...>;
        static constexpr auto preparation = T_Preparation;
    };

    /** @brief Factory retaining the existing material-AABB decomposition. */
    struct MaterialAabbDecompositionFactory
    {
        template<typename T_Setup, typename T_Allocator, typename... T_Stores>
        [[nodiscard]] auto operator()(T_Setup const&, T_Allocator const&, T_Stores&... stores) const
        {
            return MaterialAabbDecomposition<T_Stores...>{stores...};
        }
    };

    /** @brief Factory that classifies setup buckets into one dense fixed Cartesian grid. */
    template<typename T_GridFactory>
    struct FixedCartesianDecompositionFactory
    {
        template<typename T_Setup, typename T_Allocator, typename... T_Stores>
        [[nodiscard]] auto operator()(T_Setup const& setup, T_Allocator const& allocator, T_Stores&... stores) const
        {
            auto const grid = T_GridFactory{}(setup);
            using Grid = std::remove_cvref_t<decltype(grid)>;
            FixedCartesianDecomposition<typename Grid::CoordinateSystemType, T_Stores...> decomposition{
                grid,
                stores...};
            (decomposition.initializeFromMaterial(stores, allocator), ...);
            return decomposition;
        }
    };

    /** @brief Factory for adaptive mappings with setup-owned trigger and partition values. */
    template<typename T_TriggerFactory, typename T_PartitionFactory>
    struct AdaptiveSplitDecompositionFactory
    {
        template<typename T_Setup, typename T_Allocator, typename... T_Stores>
        [[nodiscard]] auto operator()(T_Setup const& setup, T_Allocator const&, T_Stores&... stores) const
        {
            return AdaptiveSplitDecomposition{T_TriggerFactory{}(setup), T_PartitionFactory{}(setup), stores...};
        }
    };

    template<SpeciesTag... T_Species>
    using MaterialAabbDecompositionGroup = DecompositionGroupDeclaration<
        MaterialAabbDecompositionFactory,
        DecompositionGroupPreparation::EveryMotion,
        T_Species...>;

    template<SpeciesTag... T_Species>
    using ExplicitInvalidationMaterialGroup = DecompositionGroupDeclaration<
        MaterialAabbDecompositionFactory,
        DecompositionGroupPreparation::OnInvalidation,
        T_Species...>;

    template<typename T_GridFactory, SpeciesTag... T_Species>
    using FixedCartesianDecompositionGroup = DecompositionGroupDeclaration<
        FixedCartesianDecompositionFactory<T_GridFactory>,
        DecompositionGroupPreparation::EveryMotion,
        T_Species...>;

    template<typename T_TriggerFactory, typename T_PartitionFactory, SpeciesTag... T_Species>
    using AdaptiveSplitDecompositionGroup = DecompositionGroupDeclaration<
        AdaptiveSplitDecompositionFactory<T_TriggerFactory, T_PartitionFactory>,
        DecompositionGroupPreparation::EveryMotion,
        T_Species...>;

    /** @brief Placeholder context used only by legacy material-only group sets. */
    struct NoDecompositionContext
    {
    };

    // Temporary names retained for setup compatibility.
    template<SpeciesTag... T_Species>
    using StaticMappingDecompositionGroup = MaterialAabbDecompositionGroup<T_Species...>;

    template<SpeciesTag... T_Species>
    using ExplicitInvalidationDecompositionGroup = ExplicitInvalidationMaterialGroup<T_Species...>;

    namespace detail
    {
        template<typename T>
        struct IsDecompositionGroup : std::false_type
        {
        };

        template<typename T_Factory, DecompositionGroupPreparation T_Preparation, SpeciesTag... T_Species>
        struct IsDecompositionGroup<DecompositionGroupDeclaration<T_Factory, T_Preparation, T_Species...>>
            : std::true_type
        {
        };

        template<typename T>
        inline constexpr bool isDecompositionGroup = IsDecompositionGroup<T>::value;

        template<typename T_Species, typename T_Group>
        struct GroupSpeciesCount : std::integral_constant<std::size_t, 0u>
        {
        };

        template<
            typename T_Species,
            typename T_Factory,
            DecompositionGroupPreparation T_Preparation,
            SpeciesTag... T_GroupSpecies>
        struct GroupSpeciesCount<T_Species, DecompositionGroupDeclaration<T_Factory, T_Preparation, T_GroupSpecies...>>
            : std::integral_constant<std::size_t, (std::size_t{0u} + ... + std::same_as<T_Species, T_GroupSpecies>)>
        {
        };

        template<typename T_Species, typename... T_Groups>
        inline constexpr std::size_t groupSpeciesCount
            = (std::size_t{0u} + ... + GroupSpeciesCount<T_Species, T_Groups>::value);

        template<typename T_Group, typename T_SpeciesList>
        struct GroupOnlyUses : std::false_type
        {
        };

        template<
            typename T_Factory,
            DecompositionGroupPreparation T_Preparation,
            SpeciesTag... T_GroupSpecies,
            SpeciesTag... T_Species>
        struct GroupOnlyUses<
            DecompositionGroupDeclaration<T_Factory, T_Preparation, T_GroupSpecies...>,
            std::tuple<T_Species...>>
            : std::bool_constant<
                  (sizeof...(T_GroupSpecies) > 0u)
                  && ((GroupSpeciesCount<
                           T_GroupSpecies,
                           DecompositionGroupDeclaration<T_Factory, T_Preparation, T_Species...>>::value
                       == 1u)
                      && ...)>
        {
        };

        template<typename T_Assignment, typename T_SpeciesList>
        struct IsDecompositionGroupAssignmentFor : std::false_type
        {
        };

        template<typename... T_Groups, SpeciesTag... T_Species>
        struct IsDecompositionGroupAssignmentFor<std::tuple<T_Groups...>, std::tuple<T_Species...>>
            : std::bool_constant<
                  (sizeof...(T_Groups) > 0u) && (isDecompositionGroup<T_Groups> && ...)
                  && (GroupOnlyUses<T_Groups, std::tuple<T_Species...>>::value && ...)
                  && ((groupSpeciesCount<T_Species, T_Groups...> == 1u) && ...)>
        {
        };

        template<typename T_SpeciesList>
        struct DefaultDecompositionGroups;

        template<SpeciesTag... T_Species>
        struct DefaultDecompositionGroups<std::tuple<T_Species...>>
        {
            using type = std::tuple<MaterialAabbDecompositionGroup<T_Species...>>;
        };

        template<typename T_Setup, typename T_Registry, typename = void>
        struct DecompositionGroupsFor
        {
            using type = typename DefaultDecompositionGroups<typename T_Registry::List>::type;
        };

        template<typename T_Setup, typename T_Registry>
        struct DecompositionGroupsFor<T_Setup, T_Registry, std::void_t<typename T_Setup::DecompositionGroups>>
        {
            using type = typename T_Setup::DecompositionGroups;
        };
    } // namespace detail

    template<typename T_Setup, SpeciesRegistryTag T_Registry>
    using DecompositionGroupsFor = typename detail::DecompositionGroupsFor<T_Setup, T_Registry>::type;

    template<typename T_Assignment, typename T_Registry>
    concept DecompositionGroupAssignmentFor
        = SpeciesRegistryTag<T_Registry>
          && detail::IsDecompositionGroupAssignmentFor<T_Assignment, typename T_Registry::List>::value;

    /** @brief Owns one decomposition lifecycle for a species decomposition group. */
    template<typename T_Decomposition>
    class DecompositionGroup
    {
    public:
        explicit DecompositionGroup(T_Decomposition decomposition, DecompositionGroupPreparation preparation)
            : m_decomposition(std::move(decomposition))
            , m_preparation(preparation)
        {
        }

        void prepareAfterMotion()
        {
            if(m_preparation == DecompositionGroupPreparation::OnInvalidation && m_isPrepared)
                return;
            m_decomposition.prepareAfterMotion();
            m_isPrepared = true;
        }

        void invalidate()
        {
            m_isPrepared = false;
        }

        [[nodiscard]] bool isPrepared() const
        {
            return m_isPrepared;
        }

        template<typename T_Store>
        [[nodiscard]] auto preparedFor(T_Store& store) const
        {
            return m_decomposition.preparedFor(store);
        }

        [[nodiscard]] T_Decomposition& decomposition()
        {
            return m_decomposition;
        }

        [[nodiscard]] T_Decomposition const& decomposition() const
        {
            return m_decomposition;
        }

    private:
        T_Decomposition m_decomposition;
        DecompositionGroupPreparation m_preparation;
        bool m_isPrepared = false;
    };

    namespace detail
    {
        template<SpeciesRegistryTag T_Registry, typename T_Group, typename T_Setup, typename T_Allocator>
        struct DecompositionGroupRuntime;

        template<
            SpeciesRegistryTag T_Registry,
            typename T_Factory,
            DecompositionGroupPreparation T_Preparation,
            SpeciesTag... T_Species,
            typename T_Setup,
            typename T_Allocator>
        struct DecompositionGroupRuntime<
            T_Registry,
            DecompositionGroupDeclaration<T_Factory, T_Preparation, T_Species...>,
            T_Setup,
            T_Allocator>
        {
            using Decomposition = decltype(T_Factory{}(
                std::declval<T_Setup const&>(),
                std::declval<T_Allocator const&>(),
                std::declval<ParticleRegionBuffer<typename T_Registry::template PRType<T_Species>>&>()...));
            using type = DecompositionGroup<Decomposition>;

            [[nodiscard]] static type create(T_Setup const& setup, T_Allocator const& allocator)
            {
                auto& dc = pmacc::Environment<>::get().DataConnector();
                [[maybe_unused]] bool const allStoresPresent = (dc.hasId(prBufId(T_Species{})) && ...);
                assert(allStoresPresent && "configured species must have a particle buffer");
                return type{
                    T_Factory{}(
                        setup,
                        allocator,
                        *dc.get<ParticleRegionBuffer<typename T_Registry::template PRType<T_Species>>>(
                            prBufId(T_Species{}))...),
                    T_Preparation};
            }
        };

        template<typename T_Species, typename T_Group>
        inline constexpr bool groupContainsSpecies = GroupSpeciesCount<T_Species, T_Group>::value == 1u;

        template<SpeciesRegistryTag T_Registry, typename T_Assignment, typename T_Setup, typename T_Allocator>
        struct DecompositionGroupSet;

        template<SpeciesRegistryTag T_Registry, typename T_Setup, typename T_Allocator, typename... T_Groups>
        struct DecompositionGroupSet<T_Registry, std::tuple<T_Groups...>, T_Setup, T_Allocator>
        {
            static_assert(
                IsDecompositionGroupAssignmentFor<std::tuple<T_Groups...>, typename T_Registry::List>::value,
                "decomposition groups must assign every registered species exactly once");

            using Registry = T_Registry;
            using Assignment = std::tuple<T_Groups...>;
            using RuntimeGroups
                = std::tuple<typename DecompositionGroupRuntime<T_Registry, T_Groups, T_Setup, T_Allocator>::type...>;

            DecompositionGroupSet(T_Setup const& setup, T_Allocator const& allocator)
                : m_groups(
                      DecompositionGroupRuntime<T_Registry, T_Groups, T_Setup, T_Allocator>::create(
                          setup,
                          allocator)...)
            {
            }

            DecompositionGroupSet()
                requires(std::default_initializable<T_Setup> && std::default_initializable<T_Allocator>)
                : DecompositionGroupSet(T_Setup{}, T_Allocator{})
            {
            }

            void prepareAfterMotion()
            {
                std::apply([](auto&... group) { (group.prepareAfterMotion(), ...); }, m_groups);
            }

            [[nodiscard]] decltype(auto) groupFor(SpeciesTag auto species)
            {
                using T_Species = decltype(species);
                return (std::get<groupIndex<T_Species>()>(m_groups));
            }

            [[nodiscard]] decltype(auto) groupFor(SpeciesTag auto species) const
            {
                using T_Species = decltype(species);
                return (std::get<groupIndex<T_Species>()>(m_groups));
            }

            [[nodiscard]] auto& storeFor(SpeciesTag auto species)
            {
                using T_Species = decltype(species);
                using Store = ParticleRegionBuffer<typename T_Registry::template PRType<T_Species>>;
                auto& dc = pmacc::Environment<>::get().DataConnector();
                assert(dc.hasId(prBufId(species)) && "configured species must have a particle buffer");
                return *dc.get<Store>(prBufId(species));
            }

            template<typename T_Store>
            [[nodiscard]] auto preparedFor(T_Store& store)
            {
                using Species = typename T_Store::Species;
                return groupFor(Species{}).preparedFor(store);
            }

            template<typename T_Store>
            [[nodiscard]] auto preparedFor(T_Store& store) const
            {
                using Species = typename T_Store::Species;
                return groupFor(Species{}).preparedFor(store);
            }

        private:
            template<SpeciesTag T_Species, std::size_t T_Index = 0u>
            static consteval std::size_t groupIndex()
            {
                static_assert(T_Index < sizeof...(T_Groups), "species is absent from this decomposition-group set");
                if constexpr(T_Index == sizeof...(T_Groups))
                    return 0u;
                else if constexpr(
                    groupContainsSpecies<T_Species, std::tuple_element_t<T_Index, std::tuple<T_Groups...>>>)
                    return T_Index;
                else
                    return groupIndex<T_Species, T_Index + 1u>();
            }

            RuntimeGroups m_groups;
        };
    } // namespace detail

    template<
        SpeciesRegistryTag T_Registry,
        typename T_Assignment,
        typename T_Setup = NoDecompositionContext,
        typename T_Allocator = NoDecompositionContext>
    using DecompositionGroupSet = detail::DecompositionGroupSet<T_Registry, T_Assignment, T_Setup, T_Allocator>;
} // namespace pmacc::spearhed
