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

#include "spmacc/meta/String.hpp"
#include "spmacc/particles/Predicate.hpp"
#include "utility.hpp"

#include <concepts>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

/**
 * Roles and species.
 *
 * Two distinct, orthogonal concepts live here:
 *
 *  - A *role* is a pure compile-time tag that states a *requirement an algorithm
 *    cares about* (e.g. "is integrated in time", "contributes to a neighbour sum").
 *    Roles never key storage; they are queried by algorithms to decide what to do.
 *
 *  - A *species* is a particle population. It owns one ParticleRegionBuffer and is
 *    *assigned a set of roles* via `using Roles = role_set<...>`. Algorithms select
 *    the species they act on by role (see NeighbourBundle::byRole).
 *
 * Init is a third, separate concern: setup "blocks" describe how a species' regions
 * and particles are created. Blocks are organised freely by the setup author and only
 * reference a species through `using Species = ...`; they never mention roles.
 */


namespace pmacc::spearhed
{
    // Roles

    namespace roles
    {
        /** Marker base so a role can be recognised generically (see RoleTag). */
        struct RoleBase
        {
        };

        /** Base for a role tag. Define a new role by inheriting with its name:
         *
         *      struct Movable : Role<"Movable"> {};
         */
        template<meta::FixedString Name>
        struct Role : RoleBase
        {
            static constexpr std::string_view name = Name.view();
        };

        // Built-in roles. Add new ones here (or in a setup header) the same way.
        struct Movable : Role<"Movable">
        {
        }; ///< particles are advanced in time by the integrator

        struct Frozen : Role<"Frozen">
        {
        }; ///< particles never move (e.g. solid walls)

        struct Source : Role<"Source">
        {
        }; ///< particles contribute to neighbour interactions (density, forces)

        struct Interior : Role<"Interior">
        {
        }; ///< interior fluid species (not a boundary/wall)

        struct Thermodynamic : Role<"Thermodynamic">
        {
        }; ///< carries energy/internal-energy fields (dvdt, dudt, internalEnergy)

        struct OpenPMDOutput : Role<"OpenPMDOutput">
        {
        }; ///< particles are serialised to openPMD output by the OpenPMDPlugin

        // constexpr instances for value-based use
        inline constexpr Movable movable{};
        inline constexpr Frozen frozen{};
        inline constexpr Source source{};
        inline constexpr Interior interior{};
        inline constexpr Thermodynamic thermodynamic{};
        inline constexpr OpenPMDOutput openPMDOutput{};
    } // namespace roles

    /** A role is any tag deriving from roles::Role (i.e. carrying a compile-time name). */
    template<typename T>
    concept RoleTag = std::derived_from<T, roles::RoleBase>;

    /** A compile-time set of roles, assigned to a species via `using Roles = role_set<...>`. */
    template<typename... R>
    using role_set = std::tuple<R...>;

    namespace detail
    {
        template<typename R, typename Set>
        struct InRoleSet : std::false_type
        {
        };

        template<typename R, typename... Rs>
        struct InRoleSet<R, role_set<Rs...>> : std::bool_constant<(std::same_as<R, Rs> || ...)>
        {
        };
    } // namespace detail

    /** The role set assigned to a species. */
    template<typename Species>
    using RolesOf = typename Species::Roles;

    // Species

    /** A species is any tag exposing a (possibly empty) role set and a name. */
    template<typename T>
    concept SpeciesTag = requires {
        typename RolesOf<T>;
        { T::name.view() } -> std::convertible_to<std::string_view>;
    };

    /** True iff @p species carries @p role. Both are passed as concrete objects:
     *
     *      if constexpr(hasRole(species::default_, roles::source)) { ... }
     *
     * consteval, so the result is usable in `if constexpr`, as a template argument, etc.
     * The arguments are read for their *type* only, so pass prvalues (e.g. `Species{}`,
     * `roles::source`) -- a non-constexpr variable cannot be forwarded into the immediate call. */
    template<SpeciesTag S, RoleTag R>
    [[nodiscard]] consteval bool hasRole(S /*species*/, R /*role*/)
    {
        return detail::InRoleSet<R, RolesOf<S>>::value;
    }

    namespace species
    {
        /** A list of species tags. Constrained so a non-species sneaking in is an error at the
         *  definition site rather than deep inside a forEachSpecies instantiation. */
        template<SpeciesTag... S>
        using species_list = std::tuple<S...>;
    } // namespace species

    /**
     * @brief The simulation's species vector: pairs a per-species PRType template with a species list.
     *
     * @tparam T_PRTypeFor  Unary template alias: T_PRTypeFor<Species> -> ParticleRegion<...>
     * @tparam T_SpeciesList species_list<...> of every registered species.
     *
     * Role-driven buffer helpers (forEachSpeciesBufWithRole, withSpeciesBufsWithRole) take a SpeciesRegistry
     * and compute the correct ParticleRegionBuffer type for each species via T_PRTypeFor<Species>.
     */
    template<template<typename> typename T_PRTypeFor, typename T_SpeciesList>
    struct SpeciesRegistry
    {
        template<typename S>
        using PRType = T_PRTypeFor<S>;
        using List = T_SpeciesList;
    };

    /** A species registry is any tag exposing a species List; the role-driven buffer helpers take one
     *  as a concrete object (e.g. an `inline constexpr` SpeciesRegistry instance). */
    template<typename T>
    concept SpeciesRegistryTag = requires { typename T::List; };

    /** DataConnector id for a species' particle-region buffer -- type-based (template). */
    template<SpeciesTag T>
    inline std::string prBufId()
    {
        return std::string("PRBuf_") + T::name.c_str();
    }

    /** DataConnector id for a species' particle-region buffer -- value-based.
     *  Regular function (not consteval) because the return type std::string is not
     *  a literal type usable in constant expressions. */
    [[nodiscard]] inline std::string prBufId(SpeciesTag auto species)
    {
        return std::string("PRBuf_") + std::remove_cvref_t<decltype(species)>::name.c_str();
    }

    // Species predicates & iteration
    //
    // A *species predicate* is a composable predicate value (see spmacc/particles/Predicate.hpp)
    // callable on a species tag: `pred::withRole<Source>(SomeSpecies{})`. It shares the one predicate
    // vocabulary with the runtime particle predicates -- compose with && || !, negate with !, lift an
    // ad-hoc check with pred::fn(...). forEachSpeciesIf walks a species list and invokes a functor for
    // each species the predicate accepts, letting algorithms state requirements like "every Source".

    namespace pred
    {
        /** Predicate value: species carries role @p R. A compile-time leaf; the body reads the role
         *  membership trait directly (not the consteval hasRole) so the shared HDINLINE combinator
         *  nodes stay device-safe even when instantiated for a species. */
        template<RoleTag R>
        struct HasRole : PredicateBase
        {
            HDINLINE constexpr bool operator()(SpeciesTag auto species) const
            {
                return detail::InRoleSet<R, RolesOf<decltype(species)>>::value;
            }

            /** Type-level query: true iff @p Entry's species carries role R.  Used by
             *  NeighbourBundle::select<P>() for compile-time entry filtering. */
            template<typename Entry>
            static constexpr bool value = detail::InRoleSet<R, RolesOf<typename Entry::Species>>::value;
        };

        /** Value form: `pred::withRole<roles::Source>`. */
        template<RoleTag R>
        inline constexpr HasRole<R> withRole{};

        /** Predicate: entry's species is exactly S.  For use with NeighbourBundle::select<>(). */
        template<SpeciesTag S>
        struct IsSpecies
        {
            template<typename Entry>
            static constexpr bool value = std::same_as<typename Entry::Species, S>;
        };

        /** Predicate: entry's species is one of S...  For use with NeighbourBundle::select<>(). */
        template<SpeciesTag... S>
        struct IsAnySpecies
        {
            template<typename Entry>
            static constexpr bool value = (std::same_as<typename Entry::Species, S> || ...);
        };
    } // namespace pred

    /** Walk @p SpeciesList and invoke @p fn(species, args...) for every species accepted by predicate
     *  @p p. @p p is a composable predicate *value* (e.g. pred::withRole<R>); it must be stateless, as
     *  it is re-formed from its type to be evaluated in a constant-expression context. */
    template<typename SpeciesList, typename Pred, typename Fn>
    constexpr void forEachSpeciesIf(Pred /*p*/, Fn&& fn, auto&&... args)
    {
        [&]<typename... S>(std::tuple<S...>*)
        {
            (
                [&]
                {
                    if constexpr(pred::eval<S>(Pred{}))
                        fn(S{}, std::forward<LL_TYPEOF(args)>(args)...);
                }(),
                ...);
        }(static_cast<SpeciesList*>(nullptr));
    }

    // value-based iteration
    /** Value-based forEachSpecies: iterates every species in @p speciesTuple. */
    template<typename Fn>
    constexpr void forEachSpecies(auto speciesTuple, Fn&& fn, auto&&... args)
    {
        std::apply([&](auto... s) { (fn(s, std::forward<LL_TYPEOF(args)>(args)...), ...); }, speciesTuple);
    }

    /** Value-based: invoke @p fn(species) for each species in @p speciesTuple carrying @p role.
     *
     *      forEachSpeciesWithRole(species::all, roles::source, [&](auto s) { ... });
     */
    template<RoleTag R, typename Fn>
    constexpr void forEachSpeciesWithRole(auto speciesTuple, R /*role*/, Fn&& fn, auto&&... args)
    {
        forEachSpecies(
            speciesTuple,
            [&](auto species)
            {
                // species is a runtime parameter here, so recover its type for the consteval query.
                if constexpr(hasRole(decltype(species){}, R{}))
                    fn(species, std::forward<LL_TYPEOF(args)>(args)...);
            });
    }

    namespace detail
    {
        template<typename R, typename List>
        struct FilterByRole;

        template<typename R, typename... S>
        struct FilterByRole<R, std::tuple<S...>>
        {
            using type = decltype(std::tuple_cat(
                std::declval<std::conditional_t<hasRole(S{}, R{}), std::tuple<S>, std::tuple<>>>()...));
        };
    } // namespace detail

    /** The subset of @p List (default: every registered species type) whose species carry role @p R,
     *  as a std::tuple of species tags. Empty tuple if none qualify. The type-level counterpart of
     *  forEachSpeciesWithRole, for code that must materialise the matching species as a pack. */
    template<typename R, typename List>
    using SpeciesWithRole = typename detail::FilterByRole<R, List>::type;

} // namespace pmacc::spearhed
