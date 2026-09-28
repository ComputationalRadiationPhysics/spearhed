/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<typename T_Sender, typename T_Scheduler, typename T_Receiver>
        class ContinuesOnOperation
        {
            struct TransferReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->transferValue(std::forward<T>(values)...);
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                ContinuesOnOperation* owner;
            };

            struct ScheduleReceiver
            {
                void set_value() noexcept
                {
                    owner->complete();
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                ContinuesOnOperation* owner;
            };

        public:
            ContinuesOnOperation(T_Sender sender, T_Scheduler scheduler, T_Receiver receiver)
                : m_receiver(std::move(receiver))
                , m_scheduled(scheduler.schedule().connect(ScheduleReceiver{this}))
                , m_upstream(std::move(sender).connect(TransferReceiver{this}))
            {
            }

            ContinuesOnOperation(ContinuesOnOperation const&) = delete;
            ContinuesOnOperation& operator=(ContinuesOnOperation const&) = delete;
            ContinuesOnOperation(ContinuesOnOperation&&) = delete;
            ContinuesOnOperation& operator=(ContinuesOnOperation&&) = delete;

            void start() & noexcept
            {
                m_upstream.start();
            }

        private:
            template<typename... T>
            void transferValue(T&&... values) noexcept
            {
                m_values.emplace(std::forward<T>(values)...);
                m_scheduled.start();
            }

            void complete() noexcept
            {
                std::apply(
                    [this](auto&&... values) { m_receiver.set_value(std::forward<decltype(values)>(values)...); },
                    std::move(*m_values));
            }

            T_Receiver m_receiver;
            std::optional<StoredValueTuple<T_Sender>> m_values;
            decltype(std::declval<T_Scheduler&>().schedule().connect(std::declval<ScheduleReceiver>())) m_scheduled;
            decltype(std::declval<T_Sender&&>().connect(std::declval<TransferReceiver>())) m_upstream;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Scheduler>
    class ContinuesOnSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<T_Sender>;

        ContinuesOnSender(T_Sender sender, T_Scheduler scheduler)
            : m_sender(std::move(sender))
            , m_scheduler(std::move(scheduler))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ContinuesOnOperation<T_Sender, T_Scheduler, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_scheduler),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Scheduler m_scheduler;
    };

    /** Transfer upstream values onto an explicit scheduler. */
    template<Sender T_Sender, typename T_Scheduler>
    auto continuesOn(T_Sender sender, T_Scheduler scheduler)
    {
        return ContinuesOnSender<T_Sender, T_Scheduler>{std::move(sender), std::move(scheduler)};
    }

    template<typename T_Scheduler>
    auto continuesOn(T_Scheduler scheduler)
    {
        return detail::SenderAdaptorClosure{[scheduler = std::move(scheduler)](auto sender) mutable
                                            { return continuesOn(std::move(sender), std::move(scheduler)); }};
    }
} // namespace caravan
