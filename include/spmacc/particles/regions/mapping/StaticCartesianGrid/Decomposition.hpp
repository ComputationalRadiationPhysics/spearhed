/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/NeighbourBundle.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/CandidateProvider.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/Relocation.hpp"
#include "spmacc/particles/spatial/InteractionEntry.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"
#include "spmacc/particles/spatial/InteractionQuery.hpp"
#include "spmacc/particles/spatial/PreparedRegionSet.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    /** @brief Tag selecting implicit fixed-grid pairwise candidate construction. */
    struct FixedCartesianMappingTag
    {
    };

    /** @brief Device-copyable broad-phase geometry derived from fixed-grid ownership. */
    template<CoordinateSystem CS>
    struct FixedCartesianBroadPhaseView
    {
        FixedCartesianGrid<CS> grid;

        [[nodiscard]] HDINLINE uint32_t bucketCount() const
        {
            return grid.bucketCount();
        }

        [[nodiscard]] HDINLINE auto chart(uint32_t slot) const
        {
            return grid.chart(slot);
        }

        [[nodiscard]] HDINLINE auto bounds(uint32_t slot) const
        {
            return grid.ownershipBounds(slot);
        }

        [[nodiscard]] bool hasPeriodicImages() const
        {
            for(bool const periodicAxis : grid.periodic)
                if(periodicAxis)
                    return true;
            return false;
        }
    };

    namespace detail
    {
        struct FixedCartesianMappingState
        {
            uint64_t generation = 0u;
            bool isPrepared = false;
        };
    } // namespace detail

    /** @brief Generation-checked prepared fixed-grid mapping for one particle store. */
    template<CoordinateSystem CS, typename T_Store>
    class FixedCartesianPreparedRegionSet
    {
    public:
        using CoordinateSystemType = CS;
        using MappingTag = FixedCartesianMappingTag;
        using Store = T_Store;
        using Grid = FixedCartesianGrid<CS>;

        FixedCartesianPreparedRegionSet(
            T_Store& store,
            Grid const& grid,
            std::shared_ptr<detail::FixedCartesianMappingState> state)
            : m_store(&store)
            , m_grid(grid)
            , m_state(std::move(state))
            , m_generation(m_state->generation)
        {
        }

        [[nodiscard]] uint64_t generation() const
        {
            return m_generation;
        }

        [[nodiscard]] uint32_t bucketCount() const
        {
            return m_grid.bucketCount();
        }

        [[nodiscard]] auto chart(uint32_t localBucketSlot) const
        {
            return m_grid.chart(localBucketSlot);
        }

        [[nodiscard]] auto deviceView() const
        {
            return m_store->getDeviceDataBox();
        }

        [[nodiscard]] auto broadPhaseView() const
        {
            return FixedCartesianBroadPhaseView<CS>{m_grid};
        }

        [[nodiscard]] T_Store& store() const
        {
            return *m_store;
        }

        [[nodiscard]] Grid const& grid() const
        {
            return m_grid;
        }

        void assertCurrent() const
        {
#ifndef NDEBUG
            assert(m_state->isPrepared && "fixed grid was used before preparation");
            assert(m_state->generation == m_generation && "interaction plan uses a stale fixed grid mapping");
#endif
        }

    private:
        T_Store* m_store;
        Grid m_grid;
        std::shared_ptr<detail::FixedCartesianMappingState> m_state;
        uint64_t m_generation;
    };

    /**
     * @brief Dense fixed Cartesian decomposition for one accelerator.
     *
     * Attached stores must already have one region per grid slot, with particle
     * positions relative to that slot's chart. prepareAfterMotion() performs a
     * bulk relocation for each attached store and deliberately never reduces or
     * reads material occupancy AABBs.
     */
    template<CoordinateSystem CS, typename... T_Stores>
    class FixedCartesianDecomposition
    {
    public:
        using Grid = FixedCartesianGrid<CS>;

        FixedCartesianDecomposition(Grid grid, T_Stores&... stores)
            : m_grid(std::move(grid))
            , m_stores(&stores...)
            , m_state(std::make_shared<detail::FixedCartesianMappingState>())
        {
        }

        void prepareAfterMotion()
        {
            relocateAttachedStores();
            ++m_state->generation;
            m_state->isPrepared = true;
        }

        /**
         * @brief Classify setup-created material buckets into this decomposition's grid.
         *
         * The setup's source buckets may use arbitrary chart origins and need not
         * match the grid count. Once this returns, @p store has one dense grid
         * region per slot and this decomposition has established generation one.
         */
        template<typename T_Store, typename T_AllocatorHandle>
        void initializeFromMaterial(T_Store& store, T_AllocatorHandle const& allocatorHandle)
        {
            static_assert(
                (std::is_same_v<std::remove_cvref_t<T_Store>, T_Stores> || ...),
                "initializeFromMaterial() requires a store attached to this fixed decomposition");
#ifndef NDEBUG
            assert(
                isAttached(store) && "initializeFromMaterial() requires this decomposition's attached store instance");
#endif
            FixedCartesianInitialClassifier<CS>{m_grid}.classify(store, allocatorHandle);
            ++m_state->generation;
            m_state->isPrepared = true;
        }

        template<typename T_Store>
        [[nodiscard]] auto preparedFor(T_Store& store) const
        {
            static_assert(
                (std::is_same_v<std::remove_cvref_t<T_Store>, T_Stores> || ...),
                "preparedFor() requires a store attached to this fixed decomposition");
#ifndef NDEBUG
            assert(m_state->isPrepared && "preparedFor() requires prepareAfterMotion() first");
            assert(isAttached(store) && "preparedFor() requires this decomposition's attached store instance");
#endif
            return FixedCartesianPreparedRegionSet<CS, T_Store>{store, m_grid, m_state};
        }

        [[nodiscard]] Grid const& grid() const
        {
            return m_grid;
        }

    private:
        template<std::size_t I = 0u>
        void relocateAttachedStores()
        {
            if constexpr(I < sizeof...(T_Stores))
            {
                auto* store = std::get<I>(m_stores);
                if(!isAttachedBefore<I>(store))
                    FixedCartesianRelocator<CS>{m_grid}.relocate(*store);
                relocateAttachedStores<I + 1u>();
            }
        }

        template<std::size_t I, typename T_Store>
        [[nodiscard]] bool isAttachedBefore(T_Store const* store) const
        {
            if constexpr(I == 0u)
            {
                static_cast<void>(store);
                return false;
            }
            else
                return isAttachedBefore(store, std::make_index_sequence<I>{});
        }

        template<typename T_Store, std::size_t... I>
        [[nodiscard]] bool isAttachedBefore(T_Store const* store, std::index_sequence<I...>) const
        {
            return (
                false || ... || (static_cast<void const*>(std::get<I>(m_stores)) == static_cast<void const*>(store)));
        }

        template<typename T_Store>
        [[nodiscard]] bool isAttached(T_Store const& store) const
        {
            return std::apply(
                [&](auto const*... attached)
                { return (false || ... || (static_cast<void const*>(attached) == static_cast<void const*>(&store))); },
                m_stores);
        }

        Grid m_grid;
        std::tuple<T_Stores*...> m_stores;
        std::shared_ptr<detail::FixedCartesianMappingState> m_state;
    };

    /** @brief CSR-free provider for any pair of fixed Cartesian grids. */
    template<CoordinateSystem CS, typename T_TargetStore, typename T_Radius, typename T_SourceStore>
    [[nodiscard]] auto makeCandidateProviderImpl(
        FixedCartesianMappingTag,
        FixedCartesianMappingTag,
        FixedCartesianPreparedRegionSet<CS, T_TargetStore> const& target,
        InteractionQuery<T_Radius> const& query,
        FixedCartesianPreparedRegionSet<CS, T_SourceStore> const& source)
    {
        return FixedGridCandidateProvider<CS>{
            target.grid(),
            source.grid(),
            static_cast<typename CS::T_Axis>(query.radius)};
    }
} // namespace pmacc::spearhed
