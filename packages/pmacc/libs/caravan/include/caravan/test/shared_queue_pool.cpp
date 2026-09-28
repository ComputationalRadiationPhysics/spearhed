/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

int main()
{
    constexpr auto computeApi = alpaka::api::host;
    constexpr auto computeDeviceKind = alpaka::deviceKind::cpu;
    constexpr auto computeQueueKind = alpaka::queueKind::nonBlocking;
    using Device = alpaka::onHost::Device<ALPAKA_TYPEOF(computeApi), ALPAKA_TYPEOF(computeDeviceKind)>;
    using Queue = alpaka::onHost::Queue<Device, ALPAKA_TYPEOF(computeQueueKind)>;
    auto const device = alpaka::onHost::makeDeviceSelector(computeApi, computeDeviceKind).makeDevice(0u);
    try
    {
        caravan::alpaka::SharedQueuePool<Queue> invalid{device, 0u};
        assert(false);
    }
    catch(std::invalid_argument const&)
    {
    }

    // The pool may grow during startup, but not once queue references have been handed out.
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 1u};
        pool.addQueues(6u);
        assert(pool.size() == 7u);
        caravan::syncWait(caravan::alpaka::withDevice(pool, caravan::alpaka::enqueue([] {})));
        try
        {
            pool.addQueues(1u);
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }
    }

    // Nested logical forks keep their dependencies when fixed physical lanes alias.
    for(std::size_t count : {1u, 2u, 3u})
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, count};
        std::array<int, 4u> values{};
        int seed = 0;
        auto branch = [&](std::size_t index)
        {
            return caravan::alpaka::enqueue(
                [&, index]
                {
                    assert(seed == 42);
                    if(index >= 2u)
                        assert(values[1u] == seed);
                    values[index] = seed;
                });
        };
        auto graph = caravan::alpaka::enqueue([&] { seed = 42; })
                     | caravan::alpaka::sequence(
                         caravan::whenAll(
                             branch(0u),
                             branch(1u) | caravan::alpaka::sequence(caravan::whenAll(branch(2u), branch(3u)))))
                     | caravan::alpaka::sequence(
                         caravan::alpaka::enqueue([&] { assert((values == std::array{42, 42, 42, 42})); }));
        caravan::syncWait(caravan::alpaka::withDevice(pool, std::move(graph)));
    }

    // Arbitrary named graphs retain native dependencies when logical lanes alias physical queues.
    for(std::size_t count : {1u, 2u, 3u})
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, count};
        std::array<std::atomic<unsigned>, 4u> ran{};
        auto a = caravan::node<"a">(caravan::alpaka::enqueue([&] { ++ran[0]; }));
        auto b = caravan::node<"b">(caravan::alpaka::enqueue([&] { ++ran[1]; }));
        auto c = caravan::node<"c">(
            caravan::alpaka::enqueue(
                [&]
                {
                    assert(ran[0] == 1u && ran[1] == 1u);
                    ++ran[2];
                }),
            caravan::after(a, b));
        auto d = caravan::node<"d">(
            caravan::alpaka::enqueue(
                [&]
                {
                    assert(ran[1] == 1u);
                    ++ran[3];
                }),
            caravan::after(b));
        caravan::syncWait(
            caravan::alpaka::withDevice(pool, caravan::graph(std::move(a), std::move(b), std::move(c), std::move(d))));
        for(auto const& calls : ran)
            assert(calls == 1u);
    }

    // D depends only on B and can finish while the independent A branch is still blocked.
    {
        using namespace std::chrono_literals;
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 2u};
        caravan::AsyncScope scope;
        std::promise<void> releaseA, dFinished;
        auto aGate = releaseA.get_future().share();
        auto dReady = dFinished.get_future();
        std::atomic<bool> bRan = false;
        auto a = caravan::node<"a">(caravan::alpaka::enqueue([aGate] { aGate.wait(); }));
        auto b = caravan::node<"b">(caravan::alpaka::enqueue([&] { bRan = true; }));
        auto c = caravan::node<"c">(caravan::alpaka::enqueue([&] { assert(bRan); }), caravan::after(a, b));
        auto d = caravan::node<"d">(
            caravan::alpaka::enqueue(
                [&]
                {
                    assert(bRan);
                    dFinished.set_value();
                }),
            caravan::after(b));
        auto completion = scope.spawn(
            caravan::alpaka::withDevice(pool, caravan::graph(std::move(a), std::move(b), std::move(c), std::move(d))));
        auto const dDidNotWaitForA = dReady.wait_for(2s) == std::future_status::ready;
        assert(completion.state() == caravan::CompletionState::pending);
        releaseA.set_value();
        completion.wait();
        assert(dDidNotWaitForA);
        scope.join().wait();
    }

    // The pool-bound factory uses the same graph lowering without a receiver environment.
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 2u};
        auto submissions = pool.submissions();
        std::atomic<unsigned> step = 0u;
        auto first
            = caravan::node<"first">(submissions.submit([&](Queue& queue) { queue.enqueueHostFn([&] { ++step; }); }));
        auto second = caravan::node<"second">(
            submissions.submit(
                [&](Queue& queue)
                {
                    queue.enqueueHostFn(
                        [&]
                        {
                            assert(step == 1u);
                            ++step;
                        });
                }),
            caravan::after(first));
        caravan::syncWait(caravan::graph(std::move(first), std::move(second)));
        assert(step == 2u);
    }

    // The fixed cap allows two queues to run while further work aliases a busy queue.
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 2u};
        caravan::AsyncScope scope;
        std::promise<void> release, firstTwoStarted;
        auto gate = release.get_future().share();
        auto started = firstTwoStarted.get_future();
        std::atomic<unsigned> entered = 0u;
        auto blocked = [&]
        {
            return caravan::alpaka::withDevice(
                pool,
                caravan::alpaka::enqueue(
                    [&, gate]
                    {
                        if(++entered == 2u)
                            firstTwoStarted.set_value();
                        gate.wait();
                    }));
        };
        auto first = scope.spawn(blocked());
        auto second = scope.spawn(blocked());
        auto third = scope.spawn(blocked());
        started.get();
        assert(entered == 2u);
        assert(first.state() == caravan::CompletionState::pending);
        assert(second.state() == caravan::CompletionState::pending);
        assert(third.state() == caravan::CompletionState::pending);
        release.set_value();
        first.wait();
        second.wait();
        third.wait();
        assert(entered == 3u);
        scope.join().wait();
    }

    // Shared-queue operations retain their own storage until queued work is quiescent.
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 1u};
        caravan::AsyncScope scope;
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        auto started = entered.get_future();
        auto storage = std::make_shared<int>(42);
        std::weak_ptr<int> retained = storage;
        auto first = scope.spawn(
            caravan::alpaka::withDevice(
                pool,
                caravan::alpaka::enqueue(
                    [storage = std::move(storage), gate, &entered]
                    {
                        entered.set_value();
                        gate.wait();
                        assert(*storage == 42);
                    })));
        started.get();
        std::atomic<bool> secondRan = false;
        auto second
            = scope.spawn(caravan::alpaka::withDevice(pool, caravan::alpaka::enqueue([&] { secondRan = true; })));
        assert(!retained.expired() && !secondRan);
        assert(first.state() == caravan::CompletionState::pending);
        assert(second.state() == caravan::CompletionState::pending);
        release.set_value();
        first.wait();
        second.wait();
        scope.join().wait();
        assert(secondRan && retained.expired());
    }

    // Independently bound pool strategies compose without exposing their queues.
    {
        caravan::alpaka::SharedQueuePool<Queue> first{device, 1u}, second{device, 1u};
        caravan::alpaka::QueuePool<Queue> leased{device};
        std::atomic<unsigned> ran = 0u;
        caravan::syncWait(
            caravan::whenAll(
                caravan::alpaka::withDevice(first, caravan::alpaka::enqueue([&] { ++ran; })),
                caravan::alpaka::withDevice(second, caravan::alpaka::enqueue([&] { ++ran; })),
                caravan::alpaka::withDevice(leased, caravan::alpaka::enqueue([&] { ++ran; }))));
        assert(ran == 3u);
    }
}
