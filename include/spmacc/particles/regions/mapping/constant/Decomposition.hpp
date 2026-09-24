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

#include "spmacc/particles/regions/RegionBoundsUpdate.hpp"
#include "spmacc/particles/spatial/BroadPhaseView.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"
#include "spmacc/particles/spatial/PreparedRegionSet.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    namespace detail
    {
        struct MaterialAabbMappingState
        {
            uint64_t generation = 0;
            bool isPrepared = false;
        };
    } // namespace detail

    /** @brief Tag selecting material-like pairwise candidate construction. */
    struct MaterialAabbMappingTag
    {
    };

    /** @brief Prepared material-bucket mapping for one particle store. */
    template<typename T_Store>
    class MaterialAabbPreparedRegionSet
    {
    public:
        using MappingTag = MaterialAabbMappingTag;
        using Store = T_Store;

        MaterialAabbPreparedRegionSet(T_Store& store, std::shared_ptr<detail::MaterialAabbMappingState> state)
            : m_store(&store)
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
            return static_cast<uint32_t>(m_store->size);
        }

        [[nodiscard]] decltype(auto) chart(uint32_t localBucketSlot) const
        {
            return (m_store->buffer->getHostBuffer().getDataBox()[static_cast<int>(localBucketSlot)].spatial.chart);
        }

        [[nodiscard]] auto deviceView() const
        {
            return m_store->getDeviceDataBox();
        }

        [[nodiscard]] auto broadPhaseView() const
        {
            return MaterialAabbBroadPhaseView{m_store->getDeviceDataBox(), static_cast<uint32_t>(m_store->size)};
        }

        [[nodiscard]] T_Store& store() const
        {
            return *m_store;
        }

        /** @brief Debug-only check that no later preparation replaced this mapping. */
        void assertCurrent() const
        {
#ifndef NDEBUG
            assert(m_state->isPrepared && "spatial mapping was used before preparation");
            assert(m_state->generation == m_generation && "interaction plan uses a stale spatial mapping");
#endif
        }

    private:
        T_Store* m_store;
        std::shared_ptr<detail::MaterialAabbMappingState> m_state;
        uint64_t m_generation;
    };

    /**
     * @brief Current persistent-cohort decomposition behind the spatial lifecycle.
     *
     * Attached stores retain material membership: after motion only each bucket's
     * world-space occupancy is reduced. Candidate construction deliberately uses
     * the existing all-pairs AABB CSR builder as the material baseline.
     */
    template<typename... T_Stores>
    class MaterialAabbDecomposition
    {
    public:
        explicit MaterialAabbDecomposition(T_Stores&... stores)
            : m_stores(&stores...)
            , m_state(std::make_shared<detail::MaterialAabbMappingState>())
        {
        }

        /**
         * @brief Establish a new material mapping generation after particle motion.
         *
         * Every attached store is reduced in place exactly once, even if it was
         * supplied more than once during construction. PMacc transaction ordering
         * keeps the reductions before plans constructed from the resulting handles.
         */
        void prepareAfterMotion()
        {
            ++m_state->generation;
            m_state->isPrepared = true;
            updateAttachedStores();
        }

        /**
         * @brief Return the current lightweight mapping handle for an attached store.
         *
         * This never schedules a reduction or any other decomposition maintenance.
         * The returned handle is valid until the next prepareAfterMotion() on this
         * decomposition, subject to the usual asynchronous plan lifetime contract.
         */
        template<typename T_Store>
        [[nodiscard]] auto preparedFor(T_Store& store) const
        {
            static_assert(
                (std::is_same_v<std::remove_cvref_t<T_Store>, T_Stores> || ...),
                "preparedFor() requires a store attached to this decomposition type");
#ifndef NDEBUG
            assert(m_state->isPrepared && "preparedFor() requires prepareAfterMotion() first");
            assert(isAttached(store) && "preparedFor() requires this decomposition's attached store instance");
#endif
            return MaterialAabbPreparedRegionSet<T_Store>{store, m_state};
        }

    private:
        template<std::size_t I = 0u>
        void updateAttachedStores()
        {
            if constexpr(I < sizeof...(T_Stores))
            {
                auto* store = std::get<I>(m_stores);
                if(!isAttachedBefore<I>(store))
                    updateMaterialAabbBounds(*store);
                updateAttachedStores<I + 1u>();
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

        std::tuple<T_Stores*...> m_stores;
        std::shared_ptr<detail::MaterialAabbMappingState> m_state;
    };

} // namespace pmacc::spearhed
