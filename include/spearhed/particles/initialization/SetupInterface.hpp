/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace spearhed
{
    // Init is organised as a free collection of "blocks". A block is decoupled from roles: it only
    // names the *species* it fills (one, or several via std::tuple) and describes, per target species,
    // which region volumes to create and a single (per-block) recipe for counting and placing particles.
    //
    // Region creation is append-only: a block emits AABB volumes into a vector via
    // addRegions<Species>; the init driver owns buffer allocation. This lets several blocks feed the
    // same species (their regions are simply concatenated, overlaps allowed) and lets one block feed
    // several species. The counting/placement recipe is uniform across a block's regions - if it must
    // differ per region it branches on the region it is handed (as SodShockTube does for left/right).
    //
    // @TODO investigate a way to define setups and their parameters in a text file, read at runtime.

    namespace detail
    {
        // A block's Species may be a single tag or a std::tuple of tags; normalise to a tuple.
        template<typename T>
        struct AsSpeciesList
        {
            using type = std::tuple<T>;
        };

        template<typename... T>
        struct AsSpeciesList<std::tuple<T...>>
        {
            using type = std::tuple<T...>;
        };

        template<typename T, typename Tuple>
        struct InList : std::false_type
        {
        };

        template<typename T, typename... Us>
        struct InList<T, std::tuple<Us...>> : std::bool_constant<(std::same_as<T, Us> || ...)>
        {
        };
    } // namespace detail

    /** The (always-tuple) list of species a block fills. */
    template<typename B>
    using BlockSpeciesList = typename detail::AsSpeciesList<typename B::Species>::type;

    /** True iff block @p B creates regions for species @p S. */
    template<typename B, typename S>
    inline constexpr bool blockTargets = detail::InList<S, BlockSpeciesList<B>>::value;

    template<typename B>
    concept SetupBlock = requires(B const b, std::vector<pmacc::spearhed::AABB<CS>>& out) {
        // The species (one tag, or a std::tuple of tags) this block fills.
        typename B::Species;
        // Append the region volumes this block contributes to species S (host-side, no device alloc).
        { b.template addRegions<std::tuple_element_t<0, BlockSpeciesList<B>>>(out) } -> std::same_as<void>;
        // Functor type deciding how many particles a region creates.
        typename B::NumParticlesToCreate;
        // Host-side args forwarded into NumParticlesToCreate.
        // TODO switch to a device friendly compile time dictionary
        { b.numParticlesToCreateArgs() };
        // Functor type placing each particle's attributes.
        typename B::PlaceParticle;
        // Host-side args forwarded into PlaceParticle.
        { b.placeParticleArgs() };
    };

    namespace detail
    {
        template<typename Tuple>
        inline constexpr bool allSetupBlocks = false;

        template<typename... B>
        inline constexpr bool allSetupBlocks<std::tuple<B...>> = (SetupBlock<std::remove_cvref_t<B>> && ...);
    } // namespace detail

    // A setup exposes its blocks as a tuple (typically of references via std::tie). Init iterates that
    // tuple per species. Roles never appear in a setup definition.
    template<typename T>
    concept SetupInterface = requires(T const a) {
        { a.blocks() };
        { a.domain } -> std::convertible_to<pmacc::spearhed::AABB<CS>>;
    } && detail::allSetupBlocks<std::remove_cvref_t<decltype(std::declval<T const&>().blocks())>>;

} // namespace spearhed
