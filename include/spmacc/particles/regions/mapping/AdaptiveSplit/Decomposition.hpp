/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/mapping/AdaptiveSplit/Splitting.hpp"
#include "spmacc/particles/spatial/BroadPhaseView.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    /** @brief Tag selecting material-style pairwise candidates for adaptive buckets. */
    struct AdaptiveSplitMappingTag
    {
    };

    namespace detail
    {
        struct AdaptiveSplitMappingState
        {
            uint64_t generation = 0u;
            bool isPrepared = false;
        };
    } // namespace detail

    template<typename T_Store>
    class AdaptiveSplitPreparedRegionSet
    {
    public:
        using MappingTag = AdaptiveSplitMappingTag;
        using Store = T_Store;

        AdaptiveSplitPreparedRegionSet(T_Store& store, std::shared_ptr<detail::AdaptiveSplitMappingState> state)
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

        [[nodiscard]] decltype(auto) chart(uint32_t slot) const
        {
            return (m_store->buffer->getHostBuffer().getDataBox()[static_cast<int>(slot)].spatial.chart);
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

        void assertCurrent() const
        {
#ifndef NDEBUG
            assert(m_state->isPrepared && "adaptive split mapping was used before preparation");
            assert(m_state->generation == m_generation && "interaction plan uses a stale adaptive split mapping");
#endif
        }

    private:
        T_Store* m_store;
        std::shared_ptr<detail::AdaptiveSplitMappingState> m_state;
        uint64_t m_generation;
    };

    /** @brief Adaptive binary decomposition with independent trigger and partition values. */
    template<typename T_Trigger, typename T_Partition, typename... T_Stores>
    class AdaptiveSplitDecomposition
    {
    public:
        AdaptiveSplitDecomposition(T_Trigger trigger, T_Partition partition, T_Stores&... stores)
            : m_splitter(std::move(trigger), std::move(partition))
            , m_stores(&stores...)
            , m_state(std::make_shared<detail::AdaptiveSplitMappingState>())
        {
        }

        void prepareAfterMotion()
        {
            splitAttachedStores();
            ++m_state->generation;
            m_state->isPrepared = true;
        }

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
            return AdaptiveSplitPreparedRegionSet<T_Store>{store, m_state};
        }

    private:
        template<std::size_t I = 0u>
        void splitAttachedStores()
        {
            if constexpr(I < sizeof...(T_Stores))
            {
                auto* store = std::get<I>(m_stores);
                if(!isAttachedBefore<I>(store))
                    m_splitter.split(*store);
                splitAttachedStores<I + 1u>();
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

        adaptive_split::BinarySplitter<T_Trigger, T_Partition> m_splitter;
        std::tuple<T_Stores*...> m_stores;
        std::shared_ptr<detail::AdaptiveSplitMappingState> m_state;
    };
} // namespace pmacc::spearhed
