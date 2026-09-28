/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/core/sender.hpp>

namespace caravan
{
    enum class CompletionState : std::uint8_t
    {
        pending,
        ready
    };

    class ExecutorThreadGuard
    {
    public:
        ExecutorThreadGuard();
        ~ExecutorThreadGuard();

        ExecutorThreadGuard(ExecutorThreadGuard const&) = delete;
        ExecutorThreadGuard& operator=(ExecutorThreadGuard const&) = delete;
    };

    namespace detail
    {
        inline thread_local std::size_t executorDepth = 0u;

        struct DispatchTask
        {
            explicit DispatchTask(std::function<void()> function) : function(std::move(function))
            {
            }

            std::function<void()> function;
            std::unique_ptr<DispatchTask> next;
        };

        struct DispatchQueue
        {
            bool running = false;
            std::unique_ptr<DispatchTask> head;
            DispatchTask* tail = nullptr;
        };

        inline thread_local DispatchQueue dispatchQueue;

        inline void dispatch(std::unique_ptr<DispatchTask> task)
        {
            auto* const tail = task.get();
            if(dispatchQueue.tail)
                dispatchQueue.tail->next = std::move(task);
            else
                dispatchQueue.head = std::move(task);
            dispatchQueue.tail = tail;
            if(dispatchQueue.running)
                return;

            dispatchQueue.running = true;

            struct ResetRunning
            {
                ~ResetRunning()
                {
                    dispatchQueue.running = false;
                }
            } resetRunning;

            while(dispatchQueue.head)
            {
                auto next = std::move(dispatchQueue.head);
                dispatchQueue.head = std::move(next->next);
                if(!dispatchQueue.head)
                    dispatchQueue.tail = nullptr;
                next->function();
            }
        }

        class State
        {
        public:
            CompletionState get() const noexcept
            {
                return m_completion.load(std::memory_order_acquire);
            }

            void wait() const
            {
                std::unique_lock lock(m_mutex);
                if(m_completion.load(std::memory_order_acquire) == CompletionState::pending && executorDepth != 0u)
                    throw std::logic_error("An executor thread cannot block on pending Caravan work");
                m_completed.wait(
                    lock,
                    [this] { return m_completion.load(std::memory_order_acquire) != CompletionState::pending; });
            }

            void subscribe(std::function<void()> continuation)
            {
                auto task = std::make_unique<DispatchTask>(std::move(continuation));
                {
                    std::lock_guard lock(m_mutex);
                    if(m_completion.load(std::memory_order_relaxed) == CompletionState::pending)
                    {
                        m_continuations.emplace_back(std::move(task));
                        return;
                    }
                }
                dispatch(std::move(task));
            }

            bool complete(CompletionState completion)
            {
                return complete(completion, [] {});
            }

        protected:
            template<typename T_Prepare>
            bool complete(CompletionState completion, T_Prepare&& prepare)
            {
                std::vector<std::unique_ptr<DispatchTask>> continuations;
                {
                    std::lock_guard lock(m_mutex);
                    if(m_completion.load(std::memory_order_relaxed) != CompletionState::pending)
                        return false;
                    std::forward<T_Prepare>(prepare)();
                    m_completion.store(completion, std::memory_order_release);
                    continuations.swap(m_continuations);
                }
                m_completed.notify_all();
                for(auto& continuation : continuations)
                    dispatch(std::move(continuation));
                return true;
            }

        private:
            std::atomic<CompletionState> m_completion{CompletionState::pending};
            mutable std::mutex m_mutex;
            mutable std::condition_variable m_completed;
            std::vector<std::unique_ptr<DispatchTask>> m_continuations;
        };

        template<typename T>
        class FutureState final : public State
        {
        public:
            template<typename U>
            bool setValue(U&& value)
            {
                return State::complete(
                    CompletionState::ready,
                    [this, &value] { m_value.emplace(std::forward<U>(value)); });
            }

            T const& value() const
            {
                if(!m_value)
                    throw std::logic_error("Caravan future value was already consumed");
                return *m_value;
            }

            T takeValue()
            {
                std::lock_guard lock(m_consumeMutex);
                if(!m_value)
                    throw std::logic_error("Caravan future value was already consumed");
                T value = std::move(*m_value);
                m_value.reset();
                return value;
            }

        private:
            std::optional<T> m_value;
            std::mutex m_consumeMutex;
        };
    } // namespace detail

    inline ExecutorThreadGuard::ExecutorThreadGuard()
    {
        ++detail::executorDepth;
    }

    inline ExecutorThreadGuard::~ExecutorThreadGuard()
    {
        --detail::executorDepth;
    }

    inline bool isExecutorThread() noexcept
    {
        return detail::executorDepth != 0u;
    }

    class EventSource;
    template<typename T_Receiver>
    class EventOperation;
    template<typename T>
    class Promise;
    template<typename T>
    class Future;

    class Event
    {
    public:
        Event() = default;

        CompletionState state() const noexcept
        {
            return m_state ? m_state->get() : CompletionState::ready;
        }

        bool isReady() const noexcept
        {
            return state() == CompletionState::ready;
        }

        void wait() const
        {
            if(m_state)
                m_state->wait();
        }

        /** Eager observation for runtime wait/wakeup boundaries. */
        template<typename T_Executor, typename T_Continuation>
        Event continueWith(T_Executor executor, T_Continuation&& continuation) const;

    private:
        explicit Event(std::shared_ptr<detail::State> state) : m_state(std::move(state))
        {
        }

        void subscribe(std::function<void()> continuation) const
        {
            if(m_state)
                m_state->subscribe(std::move(continuation));
            else
                detail::dispatch(std::make_unique<detail::DispatchTask>(std::move(continuation)));
        }

        std::shared_ptr<detail::State> m_state;

        friend class EventSource;
        template<typename T_Receiver>
        friend class EventOperation;
        friend Event whenAll(std::span<Event const>);
        template<typename T>
        friend class Future;
    };

    class EventSource
    {
    public:
        EventSource() : m_state(std::make_shared<detail::State>())
        {
        }

        Event event() const
        {
            return Event{m_state};
        }

        bool setReady() const
        {
            return m_state->complete(CompletionState::ready);
        }

    private:
        std::shared_ptr<detail::State> m_state;
    };

    inline Event readyEvent()
    {
        return {};
    }

    template<typename T_Executor, typename T_Continuation>
    Event Event::continueWith(T_Executor executor, T_Continuation&& continuation) const
    {
        EventSource successor;
        auto result = successor.event();
        auto predecessor = *this;
        auto work = std::make_shared<std::decay_t<T_Continuation>>(std::forward<T_Continuation>(continuation));
        subscribe(
            [predecessor, successor, work, executor = std::move(executor)]() mutable noexcept
            {
                auto task = [predecessor, successor, work]() noexcept
                {
                    ExecutorThreadGuard guard;
                    std::invoke(*work, predecessor);
                    successor.setReady();
                };
                executor.post(std::move(task));
            });
        return result;
    }

    namespace detail
    {
        class WhenAllState final : public State
        {
        public:
            explicit WhenAllState(std::size_t count) : m_remaining(count)
            {
            }

            void arrive(Event const&)
            {
                if(m_remaining.fetch_sub(1u, std::memory_order_acq_rel) == 1u)
                    complete(CompletionState::ready);
            }

        private:
            std::atomic<std::size_t> m_remaining;
        };
    } // namespace detail

    inline Event whenAll(std::span<Event const> events)
    {
        if(events.empty())
            return readyEvent();
        if(events.size() == 1u)
            return events.front();

        auto state = std::make_shared<detail::WhenAllState>(events.size());
        Event result{state};
        for(auto const& event : events)
            event.subscribe([state, event] { state->arrive(event); });
        return result;
    }

    template<typename T>
    class Future
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);

    public:
        Future() = default;

        bool valid() const noexcept
        {
            return static_cast<bool>(m_state);
        }

        Event event() const
        {
            if(!m_state)
                throw std::logic_error("Invalid Caravan future");
            return Event{m_state};
        }

        CompletionState state() const
        {
            return event().state();
        }

        T const& result() const
        {
            event().wait();
            return m_state->value();
        }

        /** Consume the shared result.
         *
         * The caller must have exclusive logical ownership: aliases remain valid
         * completion handles but can no longer read the consumed value.
         */
        T takeResult() &&
        {
            event().wait();
            auto state = std::exchange(m_state, {});
            return state->takeValue();
        }

    private:
        explicit Future(std::shared_ptr<detail::FutureState<T>> state) : m_state(std::move(state))
        {
        }

        std::shared_ptr<detail::FutureState<T>> m_state;

        friend class Promise<T>;
    };

    template<typename T>
    class Promise
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);

    public:
        Promise() : m_state(std::make_shared<detail::FutureState<T>>())
        {
        }

        Future<T> future() const
        {
            return Future<T>{m_state};
        }

        template<typename U = T>
        bool setValue(U&& value) const
        {
            return m_state->setValue(std::forward<U>(value));
        }

    private:
        std::shared_ptr<detail::FutureState<T>> m_state;
    };

    /** A lazy sender bridge for an already-started Event.
     *
     * The connected operation must remain at a stable address from start until
     * receiver completion, matching the P2300 operation-state lifetime rule.
     */
    template<typename T_Receiver>
    class EventOperation
    {
    public:
        EventOperation(Event event, T_Receiver receiver) : m_event(std::move(event)), m_receiver(std::move(receiver))
        {
        }

        EventOperation(EventOperation const&) = delete;
        EventOperation& operator=(EventOperation const&) = delete;
        EventOperation(EventOperation&&) = delete;
        EventOperation& operator=(EventOperation&&) = delete;

        void start() & noexcept
        {
            m_event.subscribe([this] { complete(); });
        }

    private:
        void complete() noexcept
        {
            if(m_event.state() != CompletionState::ready)
                std::terminate();
            m_receiver.set_value();
        }

        Event m_event;
        T_Receiver m_receiver;
    };

    class EventSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit EventSender(Event event) : m_event(std::move(event))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) const
        {
            return EventOperation<std::decay_t<T_Receiver>>{m_event, std::forward<T_Receiver>(receiver)};
        }

    private:
        Event m_event;
    };

    inline EventSender asSender(Event event)
    {
        return EventSender{std::move(event)};
    }

    /** Lazy, typed, consuming bridge for an already-started Future.
     *
     * Starting the sender consumes the Future's shared value. No Future alias may
     * read the result afterward. This explicit consuming contract supports
     * move-only values without adding another eager transformation model.
     */
    template<typename T>
    auto consumeAsSender(Future<T>&& future)
    {
        auto ready = future.event();
        return then(
            asSender(std::move(ready)),
            [future = std::move(future)]() mutable { return std::move(future).takeResult(); });
    }

    namespace detail
    {
        template<typename T>
        struct SyncWaitReceiver
        {
            template<typename U>
            void set_value(U&& value) noexcept
            {
                output.setValue(std::forward<U>(value));
            }

            Promise<T> output;
        };

        struct SyncWaitVoidReceiver
        {
            void set_value() noexcept
            {
                output.setReady();
            }

            EventSource output;
        };
    } // namespace detail

    /** Start a sender and block at an imperative boundary.
     *
     * Executor/progress threads must use a consumer that drives their progress mechanism instead.
     */
    template<typename T, typename T_Sender>
    T syncWait(T_Sender sender)
    {
        if(isExecutorThread())
            throw std::logic_error("An executor thread cannot call Caravan syncWait");
        Promise<T> output;
        auto result = output.future();
        auto operation = std::move(sender).connect(detail::SyncWaitReceiver<T>{output});
        operation.start();
        return std::move(result).takeResult();
    }

    template<typename T_Sender>
    void syncWait(T_Sender sender)
    {
        if(isExecutorThread())
            throw std::logic_error("An executor thread cannot call Caravan syncWait");
        EventSource output;
        auto result = output.event();
        auto operation = std::move(sender).connect(detail::SyncWaitVoidReceiver{output});
        operation.start();
        result.wait();
    }

} // namespace caravan
