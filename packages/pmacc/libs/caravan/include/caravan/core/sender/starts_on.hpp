/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <caravan/core/sender/common.hpp>
#include <caravan/core/sender/environment.hpp>

namespace caravan
{
    namespace detail
    {
        template<typename T_Scheduler, typename T_Sender, typename T_Receiver>
        class StartsOnOperation
        {
            struct ChildReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    receiver->set_value(std::forward<T>(values)...);
                }

                auto get_env() const noexcept
                {
                    return SchedulerEnvironment<T_Scheduler, T_Receiver>{scheduler, receiver};
                }

                T_Scheduler const* scheduler;
                T_Receiver* receiver;
            };

            struct ScheduleReceiver
            {
                void set_value() noexcept
                {
                    owner->m_child.start();
                }

                decltype(auto) get_env() const noexcept(noexcept(getEnvironment(std::declval<T_Receiver const&>())))
                {
                    return getEnvironment(*receiver);
                }

                StartsOnOperation* owner;
                T_Receiver* receiver;
            };

        public:
            StartsOnOperation(T_Scheduler scheduler, T_Sender sender, T_Receiver receiver)
                : m_scheduler(std::move(scheduler))
                , m_receiver(std::move(receiver))
                , m_child(std::move(sender).connect(ChildReceiver{&m_scheduler, &m_receiver}))
                , m_scheduled(m_scheduler.schedule().connect(ScheduleReceiver{this, &m_receiver}))
            {
            }

            StartsOnOperation(StartsOnOperation const&) = delete;
            StartsOnOperation& operator=(StartsOnOperation const&) = delete;
            StartsOnOperation(StartsOnOperation&&) = delete;
            StartsOnOperation& operator=(StartsOnOperation&&) = delete;

            void start() & noexcept
            {
                m_scheduled.start();
            }

        private:
            T_Scheduler m_scheduler;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<ChildReceiver>())) m_child;
            decltype(std::declval<T_Scheduler&>().schedule().connect(std::declval<ScheduleReceiver>())) m_scheduled;
        };
    } // namespace detail

    template<typename T_Scheduler, typename T_Sender>
    class StartsOnSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<T_Sender>;

        StartsOnSender(T_Scheduler scheduler, T_Sender sender)
            : m_scheduler(std::move(scheduler))
            , m_sender(std::move(sender))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::StartsOnOperation<T_Scheduler, T_Sender, std::decay_t<T_Receiver>>{
                std::move(m_scheduler),
                std::move(m_sender),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Scheduler m_scheduler;
        T_Sender m_sender;
    };

    /** Lazily start a sender through a scheduler, without restoring completion placement.
     *
     * The child sees this scheduler through getScheduler(receiver.get_env()), even
     * if it later completes on another thread. Unrelated environment queries are
     * forwarded. Both operations connect before start. Connection and scheduling
     * failures are fatal, and failed scheduling does not start the child.
     * Scheduler resources must outlive the operation; no worker or queue is added.
     */
    template<typename T_Scheduler, Sender T_Sender>
    auto startsOn(T_Scheduler scheduler, T_Sender sender)
    {
        return StartsOnSender<T_Scheduler, T_Sender>{std::move(scheduler), std::move(sender)};
    }

    template<typename T_Scheduler>
    auto startsOn(T_Scheduler scheduler)
    {
        return detail::SenderAdaptorClosure{[scheduler = std::move(scheduler)](auto sender) mutable
                                            { return startsOn(std::move(scheduler), std::move(sender)); }};
    }
} // namespace caravan
