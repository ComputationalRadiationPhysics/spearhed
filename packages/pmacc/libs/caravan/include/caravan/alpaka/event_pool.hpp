/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <caravan/alpaka/queue/traits.hpp>

namespace caravan::alpaka::detail
{
    /** Device-local event storage. Leases must outlive native use and the pool must outlive its leases. */
    template<typename T_Queue>
    class EventPool
    {
        using Event = ::alpaka::onHost::Event<QueueDevice<T_Queue>>;

        struct Entry
        {
            explicit Entry(QueueDevice<T_Queue>& device) : event(device.makeEvent())
            {
            }

            Event event;
            Entry* next = nullptr;
        };

    public:
        class Lease
        {
        public:
            Lease(Lease const&) = delete;
            Lease& operator=(Lease const&) = delete;

            Lease(Lease&& other) noexcept : m_pool(std::exchange(other.m_pool, nullptr)), m_entry(other.m_entry)
            {
            }

            ~Lease()
            {
                if(m_pool)
                {
                    std::lock_guard lock(m_pool->m_mutex);
                    m_entry->next = m_pool->m_free;
                    m_pool->m_free = m_entry;
                }
            }

            Event const& event() const noexcept
            {
                return m_entry->event;
            }

        private:
            friend class EventPool;

            Lease(EventPool& pool, Entry& entry) : m_pool(&pool), m_entry(&entry)
            {
            }

            EventPool* m_pool;
            Entry* m_entry;
        };

        explicit EventPool(QueueDevice<T_Queue> device) : m_device(std::move(device))
        {
        }

        EventPool(EventPool const&) = delete;
        EventPool& operator=(EventPool const&) = delete;

        Lease acquire()
        {
            std::lock_guard lock(m_mutex);
            if(m_free)
            {
                auto* entry = m_free;
                m_free = entry->next;
                return Lease{*this, *entry};
            }
            // ponytail: retain the high-water mark; trim idle events only if measured resource use warrants it.
            m_entries.push_back(std::make_unique<Entry>(m_device));
            return Lease{*this, *m_entries.back()};
        }

    private:
        QueueDevice<T_Queue> m_device;
        std::mutex m_mutex;
        std::vector<std::unique_ptr<Entry>> m_entries;
        Entry* m_free = nullptr;
    };
} // namespace caravan::alpaka::detail
