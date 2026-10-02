/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <tuple>
#include <utility>

#include <caravan/alpaka/queue/managed.hpp>
#include <caravan/alpaka/submission.hpp>
#include <caravan/core/retained.hpp>

namespace caravan::alpaka
{
    /** Queue-free lazy enqueue of a native alpaka host task. The task must not let exceptions escape. */
    template<typename T_Task>
    auto enqueue(T_Task task)
    {
        return detail::managedSubmit([task = std::move(task)](auto& nativeQueue) mutable
                                     { nativeQueue.enqueueHostFn(task); });
    }

    /** Lazy byte fill. The buffer/view and any explicit owner are retained by value. */
    template<typename T_Queue, typename T_Buffer>
    auto fill(T_Queue& queue, T_Buffer buffer, std::uint8_t byte)
    {
        return submit(
            queue,
            [buffer = std::move(buffer), byte](T_Queue& nativeQueue) mutable
            { ::alpaka::onHost::memset(nativeQueue, caravan::unwrap(buffer), byte); });
    }

    /** Queue-free lazy byte fill; withDevice supplies the managed queue context. */
    template<typename T_Buffer>
    auto fill(T_Buffer buffer, std::uint8_t byte)
    {
        return detail::managedSubmit([buffer = std::move(buffer), byte](auto& nativeQueue) mutable
                                     { ::alpaka::onHost::memset(nativeQueue, caravan::unwrap(buffer), byte); });
    }

    /** Lazy copy. Buffer/views, explicit owners, and the extent are retained by value. */
    template<typename T_Queue, typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Queue& queue, T_Destination destination, T_Source source, T_Extent extent)
    {
        return submit(
            queue,
            [destination = std::move(destination), source = std::move(source), extent](T_Queue& nativeQueue) mutable
            { ::alpaka::onHost::memcpy(nativeQueue, caravan::unwrap(destination), caravan::unwrap(source), extent); });
    }

    /** Queue-free lazy copy; withDevice supplies the managed queue context. */
    template<typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Destination destination, T_Source source, T_Extent extent)
    {
        return detail::managedSubmit(
            [destination = std::move(destination), source = std::move(source), extent](auto& nativeQueue) mutable
            { ::alpaka::onHost::memcpy(nativeQueue, caravan::unwrap(destination), caravan::unwrap(source), extent); });
    }

    /** Lazy kernel launch retaining the launch configuration, kernel, arguments, and explicit owners. */
    template<typename T_Queue, typename T_LaunchConfig, typename T_Kernel, typename... T_Args>
    requires(detail::QueueHandle<T_Queue>)
    auto kernel(T_Queue& queue, T_LaunchConfig launchConfig, T_Kernel kernel, T_Args... args)
    {
        return submit(
            queue,
            [launchConfig = std::move(launchConfig),
             kernel = std::move(kernel),
             args = std::tuple<T_Args...>{std::move(args)...}](T_Queue& nativeQueue) mutable
            {
                std::apply(
                    [&](auto&... values)
                    { nativeQueue.enqueue(launchConfig, ::alpaka::KernelBundle{kernel, caravan::unwrap(values)...}); },
                    args);
            });
    }

    /** Queue-free lazy kernel launch; withDevice supplies the managed queue context. */
    template<typename T_LaunchConfig, typename T_Kernel, typename... T_Args>
    auto kernel(T_LaunchConfig launchConfig, T_Kernel kernel, T_Args... args)
    {
        return detail::managedSubmit(
            [launchConfig = std::move(launchConfig),
             kernel = std::move(kernel),
             args = std::tuple<T_Args...>{std::move(args)...}](auto& nativeQueue) mutable
            {
                std::apply(
                    [&](auto&... values)
                    { nativeQueue.enqueue(launchConfig, ::alpaka::KernelBundle{kernel, caravan::unwrap(values)...}); },
                    args);
            });
    }
} // namespace caravan::alpaka
