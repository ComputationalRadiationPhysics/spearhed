/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include <caravan/alpaka/queue/queue_pool.hpp>

namespace caravan::alpaka
{
    /** Device-local pool shared by all connected graphs.
     *
     * The pool may grow before its first connection and is fixed-size afterwards.
     * At connection, logical lane i maps to (base + i) % queueCount; base advances by the graph's lane count.
     * sequence retains logical affinity, while whenAll branches may serialize on the same physical queue.
     * There are no exclusive leases or completion-time admission waits. The cap applies to this pool instance.
     *
     * Entire graph submissions are serialized. Nonblocking queues can execute different graphs concurrently
     * after submission. Backend, submission, and CPU task failures are fatal. Callables
     * must only enqueue work on the supplied queue: no recursive graph starts, external queue submissions, or
     * blocking on other branches. Queue references must not escape for later use. The pool must outlive all its
     * senders and connected operations. Cross-pool native composition is rejected, as for QueuePool.
     * Native completion events are pooled independently of the queue cap and reused after graph quiescence.
     */
    template<typename T_Queue>
    class SharedQueuePool
    {
        static_assert(detail::QueueHandle<T_Queue>);

    public:
        using Queue = T_Queue;

        class Binding
        {
        public:
            T_Queue& operator[](std::size_t lane) const noexcept
            {
                return m_pool->m_queues[(m_base + lane) % m_pool->m_queues.size()];
            }

            /** Lock only the physical queues this graph uses, in a deterministic order.
             *
             * Independent graphs on disjoint queues may submit concurrently; graphs sharing any queue still
             * serialize, preserving CPU queue error attribution per queue. Submission happens under the locks;
             * publication happens after they are released because it can synchronously deliver a receiver and
             * destroy the operation.
             */
            template<typename T_Operation, std::size_t N>
            void start(T_Operation& operation, std::array<std::size_t, N> const& lanes) const noexcept
            {
                std::array<std::size_t, N> distinct{};
                std::size_t count = 0u;
                for(auto lane : lanes)
                {
                    auto const index = (m_base + lane) % m_pool->m_queues.size();
                    bool seen = false;
                    for(std::size_t i = 0u; i < count; ++i)
                        seen = seen || distinct[i] == index;
                    if(!seen)
                        distinct[count++] = index;
                }
                std::sort(distinct.begin(), distinct.begin() + count);
                for(std::size_t i = 0u; i < count; ++i)
                    m_pool->m_queueMutexes[distinct[i]]->lock();
                operation.submit();
                for(std::size_t i = count; i-- > 0u;)
                    m_pool->m_queueMutexes[distinct[i]]->unlock();

                operation.publish();
            }

        private:
            friend class SharedQueuePool;

            Binding(SharedQueuePool& pool, std::size_t base) : m_pool(&pool), m_base(base)
            {
            }

            SharedQueuePool* m_pool;
            std::size_t m_base;
        };

        class SubmissionFactory
        {
        public:
            template<typename T_Submit>
            auto submit(T_Submit submit) const
            {
                return PoolSubmitSender<SharedQueuePool, std::decay_t<T_Submit>>{*m_pool, {std::move(submit)}};
            }

        private:
            friend class SharedQueuePool;

            explicit SubmissionFactory(SharedQueuePool& pool) : m_pool(&pool)
            {
            }

            SharedQueuePool* m_pool;
        };

        SharedQueuePool(detail::QueueDevice<T_Queue> device, std::size_t queueCount)
            : m_device(std::move(device))
            , m_events(m_device)
        {
            if(queueCount == 0u)
                throw std::invalid_argument("SharedQueuePool requires at least one queue");
            m_queues.reserve(queueCount);
            m_queueMutexes.reserve(queueCount);
            for(std::size_t i = 0u; i < queueCount; ++i)
            {
                m_queues.emplace_back(detail::makeQueue<T_Queue>(m_device));
                m_queueMutexes.push_back(std::make_unique<std::mutex>());
            }
        }

        SharedQueuePool(SharedQueuePool const&) = delete;
        SharedQueuePool& operator=(SharedQueuePool const&) = delete;

        SubmissionFactory submissions() noexcept
        {
            return SubmissionFactory{*this};
        }

        /** Add queues before the pool is first used. */
        void addQueues(std::size_t count)
        {
            std::lock_guard lock(m_mutex);
            if(m_started)
                throw std::logic_error("SharedQueuePool cannot grow after its first submission");
            while(count-- != 0u)
            {
                m_queues.emplace_back(detail::makeQueue<T_Queue>(m_device));
                m_queueMutexes.push_back(std::make_unique<std::mutex>());
            }
        }

        std::size_t size() const
        {
            std::lock_guard lock(m_mutex);
            return m_queues.size();
        }

    private:
        template<typename, typename, typename...>
        friend class detail::PooledSubmitOperation;

        Binding acquire(std::size_t laneCount)
        {
            std::lock_guard lock(m_mutex);
            m_started = true;
            auto base = m_next;
            m_next = (m_next + laneCount % m_queues.size()) % m_queues.size();
            return Binding{*this, base};
        }

        detail::QueueDevice<T_Queue> m_device;
        detail::EventPool<T_Queue> m_events;
        std::vector<T_Queue> m_queues;
        // One submission lock per physical queue instead of one pool-wide lock. See Binding::start.
        std::vector<std::unique_ptr<std::mutex>> m_queueMutexes;
        // ponytail: serialize pool growth and lease assignment, not graph submission.
        mutable std::mutex m_mutex;
        std::size_t m_next = 0u;
        bool m_started = false;
    };
} // namespace caravan::alpaka
