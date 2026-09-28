/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <array>
#include <barrier>
#include <cassert>
#include <chrono>
#include <future>
#include <optional>
#include <thread>

#include <caravan/alpaka.hpp>

namespace
{
#if ALPAKA_LANG_CUDA
    inline constexpr auto computeApi = alpaka::api::cuda;
    inline constexpr auto computeDeviceKind = alpaka::deviceKind::nvidiaGpu;
#elif ALPAKA_LANG_HIP
    inline constexpr auto computeApi = alpaka::api::hip;
    inline constexpr auto computeDeviceKind = alpaka::deviceKind::amdGpu;
#else
    inline constexpr auto computeApi = alpaka::api::host;
    inline constexpr auto computeDeviceKind = alpaka::deviceKind::cpu;
#endif
    using Device = alpaka::onHost::Device<ALPAKA_TYPEOF(computeApi), ALPAKA_TYPEOF(computeDeviceKind)>;
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
    inline constexpr auto computeQueueKind = alpaka::queueKind::nonBlocking;
#else
    // Exercise actual alpaka events on the CPU backend.
    inline constexpr auto computeQueueKind = alpaka::queueKind::blocking;
#endif
    using Queue = alpaka::onHost::Queue<Device, ALPAKA_TYPEOF(computeQueueKind)>;
    using Pool = caravan::alpaka::detail::EventPool<Queue>;
    using Event = alpaka::onHost::Event<Device>;

    struct Receiver
    {
        void set_value() noexcept
        {
            finish();
        }

        void finish() noexcept
        {
            // The connected operation is still alive: recycling must precede receiver delivery.
            {
                auto lease = pool->acquire();
                assert(lease.event() == expected);
            }
            result->set_value();
        }

        Pool* pool;
        Event expected;
        std::promise<void>* result;
    };
} // namespace

int main()
{
    auto const device = alpaka::onHost::makeDeviceSelector(computeApi, computeDeviceKind).makeDevice(0u);
    Pool pool{device}, otherPool{device};
    auto original = [&] { return Event{pool.acquire().event()}; }();
    {
        auto first = pool.acquire();
        assert(first.event() == original);
        auto moved = std::move(first);
        auto second = pool.acquire();
        assert(moved.event() != second.event());
        auto other = otherPool.acquire();
        assert(other.event() != moved.event());
        assert(other.event().getDevice() == device);
    }
    assert(pool.acquire().event() == original);

    // Simultaneous leases are exclusive even across submitting threads.
    {
        std::array<std::optional<Event>, 4u> events;
        std::array<std::thread, 4u> threads;
        std::barrier held{4};
        for(std::size_t i = 0u; i < threads.size(); ++i)
            threads[i] = std::thread(
                [&, i]
                {
                    auto lease = pool.acquire();
                    events[i] = lease.event();
                    held.arrive_and_wait();
                });
        for(auto& thread : threads)
            thread.join();
        for(std::size_t i = 0u; i < events.size(); ++i)
            for(std::size_t j = 0u; j < i; ++j)
                assert(*events[i] != *events[j]);
    }

    auto queue = caravan::alpaka::detail::makeQueue<Queue>(device);
    // Fresh bookkeeping on every reuse, including destruction without start.
    for(unsigned iteration = 0u; iteration < 8u; ++iteration)
    {
        auto expected = Event{pool.acquire().event()};
        std::promise<void> result;
        auto completed = result.get_future();
        auto submit = [](Queue& q) { q.enqueueHostFn([] {}); };
        auto connect = [&]
        {
            return caravan::alpaka::detail::SubmitOperation{
                std::array{&queue},
                std::tuple{submit},
                caravan::alpaka::detail::SubmissionDependencies<1u>::linear(),
                Receiver{&pool, expected, &result},
                &pool};
        };
        {
            auto unstarted = connect();
            assert(pool.acquire().event() != expected);
        }
        assert(pool.acquire().event() == expected);
        auto operation = connect();
        operation.start();
        completed.get();
    }

#if ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED
    // A completed branch must not recycle its event while another branch still borrows captures.
    {
        Pool pendingPool{device};
        auto events = [&]
        {
            auto first = pendingPool.acquire();
            auto second = pendingPool.acquire();
            return std::array{Event{first.event()}, Event{second.event()}};
        }();
        // Fences are acquired in reverse stage order and returned in stage order.
        auto expected = events[0];
        auto otherQueue = caravan::alpaka::detail::makeQueue<Queue>(device);
        std::promise<void> release, entered;
        auto gate = release.get_future().share();
        auto running = entered.get_future();
        std::promise<void> result;
        auto completed = result.get_future();
        auto readyBranch = [](Queue& q) { q.enqueueHostFn([] {}); };
        auto pendingBranch = [&](Queue& q)
        {
            q.enqueueHostFn(
                [&]
                {
                    entered.set_value();
                    gate.wait();
                });
        };
        auto operation = caravan::alpaka::detail::SubmitOperation{
            std::array{&queue, &otherQueue},
            std::tuple{readyBranch, pendingBranch},
            caravan::alpaka::detail::SubmissionDependencies<2u>{},
            Receiver{&pendingPool, expected, &result},
            &pendingPool};
        operation.start();
        running.get();
        alpaka::onHost::wait(queue);
        assert(completed.wait_for(std::chrono::milliseconds{10}) == std::future_status::timeout);
        {
            // Both original events are leased: this must allocate a third distinct event.
            auto spare = pendingPool.acquire();
            assert(spare.event() != events[0] && spare.event() != events[1]);
        }
        release.set_value();
        completed.get();
    }
#endif

    // Both context variants use pooled native fences for repeated cross-queue fork/join graphs.
    auto exercise = [](auto& context)
    {
        for(unsigned iteration = 0u; iteration < 8u; ++iteration)
        {
            int left = 0, right = 0;
            caravan::syncWait(
                caravan::alpaka::withDevice(
                    context,
                    caravan::whenAll(
                        caravan::alpaka::enqueue([&] { left = 1; }),
                        caravan::alpaka::enqueue([&] { right = 2; }))
                        | caravan::sequence(caravan::alpaka::enqueue([&] { assert(left + right == 3); }))));
        }
    };
    caravan::alpaka::QueuePool<Queue> exclusive{device};
    caravan::alpaka::SharedQueuePool<Queue> shared{device, 2u};
    exercise(exclusive);
    exercise(shared);
}
