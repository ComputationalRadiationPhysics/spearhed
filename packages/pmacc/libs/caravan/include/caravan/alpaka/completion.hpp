/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>

#include <caravan/alpaka/event_pool.hpp>
#include <caravan/alpaka/queue/traits.hpp>
#include <caravan/core/eager.hpp>

namespace caravan::alpaka::detail
{
    /** A submission-time fence. Query failure does not establish quiescence. */
    template<typename T_Queue>
    class CompletionFence
    {
    public:
        explicit CompletionFence(T_Queue const& queue, EventPool<T_Queue>* pool = nullptr)
            : m_lease(pool ? std::make_optional(pool->acquire()) : std::nullopt)
            , m_event(m_lease ? m_lease->event() : queue.getDevice().makeEvent())
        {
        }

        void record(T_Queue& queue)
        {
            if(!m_recorded)
            {
                queue.enqueue(m_event);
                m_recorded = true;
            }
        }

        void waitOn(T_Queue& queue)
        {
            queue.waitFor(m_event);
        }

        bool poll() noexcept
        {
            if(!m_recorded || m_complete)
                return true;
            m_complete = m_event.isComplete();
            return m_complete;
        }

    private:
        std::optional<typename EventPool<T_Queue>::Lease> m_lease;
        ::alpaka::onHost::Event<QueueDevice<T_Queue>> m_event;
        bool m_recorded = false;
        bool m_complete = false;
    };

    class CompletionTask
    {
    public:
        // A true result may destroy this task; false retains it for the next scan.
        virtual bool poll() noexcept = 0;

        CompletionTask* next = nullptr;

    protected:
        ~CompletionTask() = default;
    };

    enum class CompletionPollingPolicy
    {
        timed,
        continuous
    };

    /** Bounded active-polling window (microseconds) before the completion thread blocks idle. */
    inline std::uint64_t completionLingerMicroseconds() noexcept;

    /** Observes terminal fences and delivers receivers without blocking on pending backend work. */
    class CompletionThread
    {
    public:
        explicit CompletionThread(CompletionPollingPolicy pollingPolicy = CompletionPollingPolicy::timed)
            : m_pollingPolicy(pollingPolicy)
            , m_lingerUs(completionLingerMicroseconds())
            , m_thread([this] { run(); })
        {
        }

        CompletionThread(CompletionThread const&) = delete;
        CompletionThread& operator=(CompletionThread const&) = delete;

        ~CompletionThread()
        {
            {
                std::lock_guard lock(m_mutex);
                m_stopped = true;
            }
            m_ready.notify_one();
            m_thread.join();
        }

        void post(CompletionTask& task) noexcept
        {
            {
                std::lock_guard lock(m_mutex);
                if(m_tail)
                    m_tail->next = &task;
                else
                    m_head = &task;
                m_tail = &task;
            }
            m_ready.notify_one();
        }

    private:
        void run() noexcept
        {
            ExecutorThreadGuard guard;
            CompletionTask* pending = nullptr;
            while(true)
            {
                {
                    std::unique_lock lock(m_mutex);
                    if(!pending && m_lingerUs != 0u)
                    {
                        // Bounded active polling across an idle gap: catch a post without a condition-variable
                        // wakeup. The lock is not held while spinning so a concurrent post() can proceed.
                        lock.unlock();
                        auto const deadline = std::chrono::steady_clock::now() + std::chrono::microseconds{m_lingerUs};
                        while(std::chrono::steady_clock::now() < deadline)
                        {
                            {
                                std::lock_guard probe(m_mutex);
                                if(m_head || m_stopped)
                                    break;
                            }
                            std::this_thread::yield();
                        }
                        lock.lock();
                    }
                    if(pending && m_pollingPolicy == CompletionPollingPolicy::timed)
                        m_ready.wait_for(lock, std::chrono::microseconds{100}, [this] { return m_head; });
                    else if(!pending && !m_head)
                        m_ready.wait(lock, [this] { return m_stopped || m_head; });
                    if(m_head)
                    {
                        m_tail->next = pending;
                        pending = std::exchange(m_head, nullptr);
                        m_tail = nullptr;
                    }
                    if(!pending && m_stopped)
                        return;
                }

                if(pending == nullptr)
                    continue;
                {
                    CompletionTask* deferred = nullptr;
                    auto** tail = &deferred;
                    while(pending)
                    {
                        auto* task = pending;
                        pending = std::exchange(task->next, nullptr);
                        if(!task->poll())
                        {
                            *tail = task;
                            tail = &task->next;
                        }
                    }
                    pending = deferred;
                }
            }
        }

        std::mutex m_mutex;
        std::condition_variable m_ready;
        CompletionTask* m_head = nullptr;
        CompletionTask* m_tail = nullptr;
        bool m_stopped = false;
        CompletionPollingPolicy m_pollingPolicy;
        std::uint64_t m_lingerUs;
        std::thread m_thread;
    };

    /** Select the completion policy before the process first submits alpaka work.
     *
     * Continuous polling is the default because it keeps short-step latency low. Set the environment
     * variable to "timed" to restore the lower-CPU 100 us polling policy.
     */
    inline CompletionPollingPolicy completionPollingPolicy() noexcept
    {
        auto const* value = std::getenv("CARAVAN_ALPAKA_COMPLETION_POLLING");
        return value && std::string_view{value} == "timed" ? CompletionPollingPolicy::timed
                                                           : CompletionPollingPolicy::continuous;
    }

    /** Bounded active-polling window before the completion thread blocks. 0 disables the spin. */
    inline std::uint64_t completionLingerMicroseconds() noexcept
    {
        auto const* value = std::getenv("CARAVAN_COMPLETION_LINGER_US");
        return value == nullptr ? 50u : std::strtoull(value, nullptr, 10);
    }

    inline CompletionThread& completionThread()
    {
        static CompletionThread thread{completionPollingPolicy()};
        return thread;
    }
} // namespace caravan::alpaka::detail
