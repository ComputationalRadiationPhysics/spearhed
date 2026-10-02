/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan
{
    class RunLoopScheduler;
    class RunLoopScheduleSender;

    /** Bounded active-polling window (microseconds) before the run loop blocks on its task condition variable.
     *
     * Zero disables the spin. Set CARAVAN_RUNLOOP_LINGER_US to override the default.
     */
    inline std::uint64_t runLoopLingerMicroseconds() noexcept
    {
        static std::uint64_t const value = []
        {
            auto const* env = std::getenv("CARAVAN_RUNLOOP_LINGER_US");
            return env == nullptr ? 50u : std::strtoull(env, nullptr, 10);
        }();
        return value;
    }

    /** Manually driven single-thread queue for host/control work. */
    class RunLoop
    {
    public:
        RunLoopScheduler scheduler() noexcept;

        void run()
        {
            while(runOne())
            {
            }
        }

        /** Execute one task, blocking until work is ready or the loop is finished. */
        bool runOne()
        {
            ExecutorThreadGuard guard;
            std::function<void()> task;
            {
                std::unique_lock lock(m_mutex);
                if(m_tasks.empty() && !m_finished)
                {
                    auto const lingerUs = runLoopLingerMicroseconds();
                    if(lingerUs != 0u)
                    {
                        // Bounded active wait: catch a task posted by another thread without a
                        // condition-variable wakeup. The lock is not held while spinning.
                        lock.unlock();
                        auto const deadline = std::chrono::steady_clock::now() + std::chrono::microseconds{lingerUs};
                        while(std::chrono::steady_clock::now() < deadline)
                        {
                            {
                                std::lock_guard probe(m_mutex);
                                if(m_finished || !m_tasks.empty())
                                    break;
                            }
                            std::this_thread::yield();
                        }
                        lock.lock();
                    }
                }
                m_ready.wait(lock, [this] { return m_finished || !m_tasks.empty(); });
                if(m_tasks.empty())
                    return false;
                task = std::move(m_tasks.front());
                m_tasks.pop_front();
            }
            task();
            return true;
        }

        /** Execute a snapshot of ready work without blocking.
         *
         * Work posted while this batch runs is deferred to the next call, so a
         * self-reposting task cannot monopolize the caller.
         */
        void runReady()
        {
            ExecutorThreadGuard guard;
            std::size_t ready;
            {
                std::lock_guard lock(m_mutex);
                ready = m_tasks.size();
            }
            while(ready-- > 0u)
            {
                std::function<void()> task;
                {
                    std::lock_guard lock(m_mutex);
                    if(m_tasks.empty())
                        return;
                    task = std::move(m_tasks.front());
                    m_tasks.pop_front();
                }
                task();
            }
        }

        void finish()
        {
            {
                std::lock_guard lock(m_mutex);
                m_finished = true;
            }
            m_ready.notify_all();
        }

    private:
        template<typename T_Function>
        void post(T_Function&& function)
        {
            {
                std::lock_guard lock(m_mutex);
                if(m_finished)
                    throw std::logic_error("Cannot post to a finished Caravan run loop");
                m_tasks.emplace_back(std::forward<T_Function>(function));
            }
            m_ready.notify_one();
        }

        std::mutex m_mutex;
        std::condition_variable m_ready;
        std::deque<std::function<void()>> m_tasks;
        bool m_finished = false;

        friend class RunLoopScheduler;
    };

    /** Cheap scheduling handle; its RunLoop must outlive it. */
    class RunLoopScheduler
    {
    public:
        RunLoopScheduleSender schedule() const noexcept;

    private:
        // Callable submission is only for the scheduling sender and eager wait observers.
        template<typename T_Function>
        void post(T_Function&& function) const
        {
            m_loop->post(std::forward<T_Function>(function));
        }

        explicit RunLoopScheduler(RunLoop& loop) : m_loop(&loop)
        {
        }

        RunLoop* m_loop;

        friend class RunLoop;
        friend class RunLoopScheduleSender;
        friend class Event;
    };

    /** Lazy scheduling operation for the manually driven run loop. */
    class RunLoopScheduleSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit RunLoopScheduleSender(RunLoopScheduler scheduler) : m_scheduler(scheduler)
        {
        }

        template<typename T_Receiver>
        class Operation
        {
        public:
            Operation(RunLoopScheduler scheduler, T_Receiver receiver)
                : m_scheduler(scheduler)
                , m_receiver(std::move(receiver))
            {
            }

            Operation(Operation const&) = delete;
            Operation& operator=(Operation const&) = delete;
            Operation(Operation&&) = delete;
            Operation& operator=(Operation&&) = delete;

            void start() & noexcept
            {
                m_scheduler.post([this] { m_receiver.set_value(); });
            }

        private:
            RunLoopScheduler m_scheduler;
            T_Receiver m_receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{m_scheduler, std::forward<T_Receiver>(receiver)};
        }

    private:
        RunLoopScheduler m_scheduler;
    };

    inline RunLoopScheduleSender RunLoopScheduler::schedule() const noexcept
    {
        return RunLoopScheduleSender{*this};
    }

    inline RunLoopScheduler RunLoop::scheduler() noexcept
    {
        return RunLoopScheduler{*this};
    }

} // namespace caravan
