/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <type_traits>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan
{
    /** Scheduler that completes work immediately on the calling thread. */
    class InlineScheduler
    {
        class ScheduleSender
        {
        public:
            using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

            template<typename T_Receiver>
            class Operation
            {
            public:
                explicit Operation(T_Receiver receiver) : m_receiver(std::move(receiver))
                {
                }

                void start() & noexcept
                {
                    ExecutorThreadGuard guard;
                    m_receiver.set_value();
                }

            private:
                T_Receiver m_receiver;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{std::forward<T_Receiver>(receiver)};
            }
        };

    public:
        auto schedule() const noexcept
        {
            return ScheduleSender{};
        }

    private:
        template<typename T_Function>
        void post(T_Function&& function) const
        {
            std::forward<T_Function>(function)();
        }

        friend class Event;
    };

} // namespace caravan
