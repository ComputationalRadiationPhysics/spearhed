/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <caravan/core/sender/continues_on.hpp>
#include <caravan/core/sender/starts_on.hpp>

namespace caravan
{
    template<typename T_Scheduler, typename T_Sender>
    class OnSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<T_Sender>;

        OnSender(T_Scheduler scheduler, T_Sender sender)
            : m_scheduler(std::move(scheduler))
            , m_sender(std::move(sender))
        {
        }

        template<typename T_Receiver>
        requires requires(T_Receiver const& receiver) { getScheduler(detail::getEnvironment(receiver)); }
        auto connect(T_Receiver&& receiver) &&
        {
            auto restoration = getScheduler(detail::getEnvironment(receiver));
            return continuesOn(startsOn(std::move(m_scheduler), std::move(m_sender)), std::move(restoration))
                .connect(std::forward<T_Receiver>(receiver));
        }

    private:
        T_Scheduler m_scheduler;
        T_Sender m_sender;
    };

    /** Scoped placement: start on scheduler, restore completion to the ambient scheduler.
     *
     * The receiver environment must answer getScheduler(), usually supplied by an
     * enclosing startsOn(). Capture it at connect, not from the initiating thread.
     * Successful completion is restored via continuesOn. Scheduling and
     * restoration failures are fatal.
     * An inline ambient scheduler restores inline on the completing thread, with
     * no return hop. An ambient run loop must be driven to deliver restoration.
     */
    template<typename T_Scheduler, Sender T_Sender>
    auto on(T_Scheduler scheduler, T_Sender sender)
    {
        return OnSender<T_Scheduler, T_Sender>{std::move(scheduler), std::move(sender)};
    }

    template<typename T_Scheduler>
    auto on(T_Scheduler scheduler)
    {
        return detail::SenderAdaptorClosure{[scheduler = std::move(scheduler)](auto sender) mutable
                                            { return on(std::move(scheduler), std::move(sender)); }};
    }
} // namespace caravan
