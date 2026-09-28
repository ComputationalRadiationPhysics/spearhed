/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

namespace caravan::alpaka::detail
{
    /** Detect an alpaka 3 queue handle.
     *
     * alpaka 3 does not provide a public queue concept, therefore the public handle interface is checked
     * directly.
     */
    template<typename T>
    concept QueueHandle = requires(T const& queue) {
        { queue.getQueueKind() } -> ::alpaka::concepts::QueueKind;
        { queue.getDevice() };
    };

    /** Device handle type of an alpaka 3 queue handle.
     *
     * alpaka 3 queues are bound to a device and carry the queue kind in their type.
     */
    template<typename T_Queue>
    using QueueDevice = std::remove_cvref_t<decltype(std::declval<T_Queue const&>().getDevice())>;

    /** Queue-kind tag type of an alpaka 3 queue handle. */
    template<typename T_Queue>
    using QueueKind = std::remove_cvref_t<decltype(std::declval<T_Queue const&>().getQueueKind())>;

    /** Create a queue of the same type as T_Queue on the given device.
     *
     * alpaka 3 creates queues through the device handle instead of using a queue type that is
     * constructible from a device.
     */
    template<typename T_Queue>
    auto makeQueue(QueueDevice<T_Queue> device)
    {
        return device.makeQueue(QueueKind<T_Queue>{});
    }
} // namespace caravan::alpaka::detail
