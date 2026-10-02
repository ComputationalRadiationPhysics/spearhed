/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <caravan/core/sender/common.hpp>
#include <caravan/core/sender/let_value.hpp>

namespace caravan
{
    /** Sequence two senders, discarding predecessor values and forwarding successor completion.
     *
     * A common domain may fuse native dependencies (e.g. alpaka FIFO/events). Domain-specific resource
     * constraints, such as using one queue pool, still apply. Otherwise, the successor is connected and
     * started on the thread delivering successful predecessor completion, without blocking or a scheduler
     * transfer. Use letValue when the successor needs predecessor values or an explicit host boundary.
     * Callback, submission, and asynchronous execution failures are fatal.
     */
    template<Sender T_Previous, Sender T_Next>
    auto sequence(T_Previous previous, T_Next next)
    {
        auto domain = detail::commonDomain(previous, next);
        if constexpr(requires { domain.transform(SequenceTag{}, std::move(previous), std::move(next)); })
            return domain.transform(SequenceTag{}, std::move(previous), std::move(next));
        else
            return caravan::letValue(
                std::move(previous),
                [next = std::move(next)](auto&&...) mutable { return std::move(next); });
    }

    /** Pipe adaptor: previous | sequence(next). */
    template<Sender T_Next>
    auto sequence(T_Next next)
    {
        return detail::SenderAdaptorClosure{[next = std::move(next)](auto previous) mutable
                                            { return caravan::sequence(std::move(previous), std::move(next)); }};
    }
} // namespace caravan
