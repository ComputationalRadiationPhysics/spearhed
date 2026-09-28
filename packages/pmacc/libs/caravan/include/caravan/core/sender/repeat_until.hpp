/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<typename T_Factory, typename T_Receiver>
        class RepeatUntilOperation
        {
            struct Receiver
            {
                void set_value(bool done) noexcept
                {
                    owner->complete(done);
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return std::as_const(owner->m_receiver).get_env();
                }

                RepeatUntilOperation* owner;
            };

            enum class State
            {
                starting,
                waiting,
                completed
            };

            using Iteration = ConnectedOperation<std::invoke_result_t<T_Factory&>, Receiver>;

        public:
            RepeatUntilOperation(T_Factory factory, T_Receiver receiver)
                : m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
            {
            }

            RepeatUntilOperation(RepeatUntilOperation const&) = delete;
            RepeatUntilOperation& operator=(RepeatUntilOperation const&) = delete;
            RepeatUntilOperation(RepeatUntilOperation&&) = delete;
            RepeatUntilOperation& operator=(RepeatUntilOperation&&) = delete;

            void start() & noexcept
            {
                run();
            }

        private:
            void complete(bool done) noexcept
            {
                m_done = done;
                // If start() is still on the stack, its caller drives the next iteration.
                // Otherwise this completion owns the driver. Do not access members after handing it off.
                if(m_state.exchange(State::completed, std::memory_order_acq_rel) == State::waiting)
                    run();
            }

            void run() noexcept
            {
                // done = falso;
                // while (!done)
                for(;;)
                {
                    // Completion permits destruction of the child, but never before its start() returns.
                    m_iteration.reset();
                    if(m_done)
                    {
                        m_receiver.set_value();
                        return;
                    }
                    m_state.store(State::starting, std::memory_order_relaxed);
                    m_iteration.emplace(std::invoke(m_factory), m_iterationReceiver);
                    m_iteration->start();
                    // Inline (or racing) completion is drained iteratively, not recursively.
                    // Once waiting is published, completion may also destroy this operation.
                    if(m_state.exchange(State::waiting, std::memory_order_acq_rel) != State::completed)
                        return;
                }
            }

            T_Factory m_factory;
            T_Receiver m_receiver;
            Receiver m_iterationReceiver{this};
            std::atomic<State> m_state{State::starting};
            bool m_done = false;
            std::optional<Iteration> m_iteration;
        };
    } // namespace detail

    template<typename T_Factory>
    class RepeatUntilSender
    {
        static_assert(
            std::is_same_v<
                CompletionSignaturesOf<std::invoke_result_t<T_Factory&>>,
                detail::DefaultCompletionSignatures<ValueSignature<bool>>>,
            "repeatUntil requires a factory returning a sender of bool");

    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit RepeatUntilSender(T_Factory factory) : m_factory(std::move(factory))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::RepeatUntilOperation<T_Factory, std::decay_t<T_Receiver>>{
                std::move(m_factory),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Factory m_factory;
    };

    /** Lazily invoke a factory at least once, repeating until its sender completes with true.
     *
     * Each iteration gets a fresh sender and forwards the receiver environment. The loop retains
     * only one child operation and uses constant
     * stack space for inline completion. The factory is owned by the operation, so its captures may hold
     * state borrowed by iteration senders. Like then/letValue, no scheduler hop is introduced.
     */
    template<typename T_Factory>
    auto repeatUntil(T_Factory factory)
    {
        return RepeatUntilSender<T_Factory>{std::move(factory)};
    }
} // namespace caravan
