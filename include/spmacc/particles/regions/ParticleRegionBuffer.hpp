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

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace pmacc::spearhed
{
    template<typename T_ParticleRegion>
    struct ParticleRegionBuffer : ISimulationData
    {
        using ParticleRegionType = T_ParticleRegion;
        using Species = typename T_ParticleRegion::Species;

        // replaces the old buffer with a new one with the given size
        // Does not communicate this to the GPU yet
        // Actually I want to have the ParticleRegion to be an SoA, and i want to resize the SoA and then hold a
        // HDBuffer to the current SoA
        auto create(size_t capacity)
        {
            buffer = pmacc::HostDeviceBuffer<ParticleRegionType, DIM1>(pmacc::DataSpace<DIM1>(capacity), false);
            ++topologyVersion;
        }

        // Copies the particle region to the host buffer
        // Remember to send buf to device before use
        auto pushBack(ParticleRegionType const& pr)
        {
            buffer->getHostBuffer().getDataBox()[size++] = pr;
            ++topologyVersion;
        }

        auto getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        void synchronize() override
        {
            buffer->deviceToHost();
        };

        SimulationDataId getUniqueId() override
        {
            return prBufId<Species>();
        }

        std::optional<pmacc::HostDeviceBuffer<ParticleRegionType, DIM1>> buffer;
        int size = 0;

        /**
         * @brief Monotonically increasing counter of frame-list topology mutations.
         *
         * Bumped whenever the set or ordering of frames owned by this buffer can change: frame
         * allocation/free, region add/remove. FrameIndexBuffer (see FrameIndex.hpp) compares its
         * builtVersion against this counter to detect staleness without relying on callers to track
         * it by hand. Host code that launches a device kernel which mutates frame-list topology
         * (e.g. allocates or frees frames) must increment this after the launch.
         */
        uint64_t topologyVersion = 0;
    };

    /**
     * @brief Invokes @p fn(prBuf) for every species carrying role @p R that the DataConnector holds a buffer for.
     *
     * Wraps the recurring "walk species by role, fetch each ParticleRegionBuffer from the
     * DataConnector" idiom. The region type is read straight off @p registry (the simulation's species
     * vector, which pairs the species list with the ParticleRegion their buffers use), so callers no
     * longer pass it. Registered species the active setup never created have no entry in the
     * DataConnector and are silently skipped, so callers need not guard with hasId themselves.
     *
     * @param  registry The simulation's species vector (see SpeciesRegistry), passed as a concrete object;
     *                  supplies both the species list walked here and the per-species PRType template.
     * @param  role     Role the species must carry, passed as a concrete object (e.g. roles::movable).
     * @param  fn       Functor invoked as fn(ParticleRegionBuffer<Registry::PRType<Species>>&).
     */
    template<SpeciesRegistryTag Registry, RoleTag R, typename Fn>
    void forEachSpeciesBufWithRole(Registry /*registry*/, R /*role*/, Fn&& fn, auto&&... args)
    {
        auto& dc = pmacc::Environment<>::get().DataConnector();

        forEachSpeciesIf<typename Registry::List>(
            pred::withRole<R>,
            [&](auto species)
            {
                using Species = decltype(species);
                using PRType = typename Registry::template PRType<Species>;
                auto const id = prBufId(species);
                if(!dc.hasId(id))
                    return;
                fn(*dc.get<ParticleRegionBuffer<PRType>>(id), std::forward<LL_TYPEOF(args)>(args)...);
            });
    }

    namespace detail
    {
        // Base case: no species left to consider; call fn with the buffers gathered so far.
        template<SpeciesRegistryTag Registry, typename Fn, typename... Present>
        void withPresentPRBufs(Registry /*registry*/, std::tuple<> /*remaining*/, Fn&& fn, Present&... present)
        {
            fn(present...);
        }

        // Recursive case: prepend head's buffer to the pack iff the DataConnector holds it, then recurse.
        template<SpeciesRegistryTag Registry, typename Head, typename... Tail, typename Fn, typename... Present>
        void withPresentPRBufs(
            Registry registry,
            std::tuple<Head, Tail...> /*remaining*/,
            Fn&& fn,
            Present&... present)
        {
            auto const id = prBufId(Head{});
            auto& dc = pmacc::Environment<>::get().DataConnector();
            if(dc.hasId(id))
            {
                using PRType = typename Registry::template PRType<Head>;
                using PRBuf = ParticleRegionBuffer<PRType>;
                auto& buf = *dc.get<PRBuf>(id);
                withPresentPRBufs(registry, std::tuple<Tail...>{}, fn, present..., buf);
            }
            else
                withPresentPRBufs(registry, std::tuple<Tail...>{}, fn, present...);
        }
    } // namespace detail

    /**
     * @brief Invokes @p fn(bufs&...) once with every present ParticleRegionBuffer whose species carries role @p R.
     *
     * Unlike forEachSpeciesBufWithRole (one call per buffer), this gathers all present buffers and makes a
     * single call with them as a pack -- suited to variadic consumers such as CalculateNeighbourRegions,
     * which must see every source together to build a single neighbour bundle. The region type is read
     * straight off @p registry, so callers no longer pass it.
     *
     * Species the active setup never created have no entry in the DataConnector and are dropped from the
     * pack. Because presence is a runtime property while the pack is fixed at compile time, the present
     * subset is materialised by recursing over the role-filtered species list, branching once per species
     * on hasId.
     *
     * @param  registry The simulation's species vector (see SpeciesRegistry), passed as a concrete object;
     *                  supplies both the species list filtered here and the per-species PRType template.
     * @param  role     Role the species must carry, passed as a concrete object (e.g. roles::source).
     * @param  fn       Functor invoked as fn(ParticleRegionBuffer<Registry::PRType<Species>>&...).
     */
    template<SpeciesRegistryTag Registry, RoleTag R, typename Fn>
    void withSpeciesBufsWithRole(Registry registry, R /*role*/, Fn&& fn)
    {
        using Sources = SpeciesWithRole<R, typename Registry::List>;
        detail::withPresentPRBufs(registry, Sources{}, std::forward<Fn>(fn));
    }

} // namespace pmacc::spearhed
