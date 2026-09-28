/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

namespace
{

    /** alpaka 1.x submitted host tasks through alpaka::enqueue(queue, task); alpaka 3 spells this
     * queue.enqueueHostFn(task). */
    template<typename T_Queue, typename T_Task>
    void enqueueHostTask(T_Queue& queue, T_Task const& task)
    {
        queue.enqueueHostFn(task);
    }

    struct Increment
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value) const
        {
            ++*value;
        }
    };

    struct Add
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value, int const* other) const
        {
            *value += *other;
        }
    };
} // namespace

int main()
{
    assert(!caravan::alpaka::isCompletionCallback());
    using Idx = std::size_t;
#if ALPAKA_LANG_CUDA
    constexpr auto computeApi = alpaka::api::cuda;
    constexpr auto computeDeviceKind = alpaka::deviceKind::nvidiaGpu;
#elif ALPAKA_LANG_HIP
    constexpr auto computeApi = alpaka::api::hip;
    constexpr auto computeDeviceKind = alpaka::deviceKind::amdGpu;
#else
    constexpr auto computeApi = alpaka::api::host;
    constexpr auto computeDeviceKind = alpaka::deviceKind::cpu;
#endif
    constexpr auto computeQueueKind = alpaka::queueKind::nonBlocking;
    using Device = alpaka::onHost::Device<ALPAKA_TYPEOF(computeApi), ALPAKA_TYPEOF(computeDeviceKind)>;
    using Queue = alpaka::onHost::Queue<Device, ALPAKA_TYPEOF(computeQueueKind)>;

    auto const device = alpaka::onHost::makeDeviceSelector(computeApi, computeDeviceKind).makeDevice(0u);
    auto const host = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu).makeDevice(0u);
    auto queue = caravan::alpaka::detail::makeQueue<Queue>(device);
    auto secondQueue = caravan::alpaka::detail::makeQueue<Queue>(device);
    auto const one = alpaka::Vec{Idx{1u}};
    auto const threadSpec = alpaka::onHost::ThreadSpec{one, one};
    auto deviceValue = alpaka::onHost::alloc<int>(device, one);
    auto hostValue = alpaka::onHost::alloc<int>(host, one);
    hostValue[0] = 41;

    bool submitted = false;
    auto retained = std::make_shared<int>(7);
    std::weak_ptr<int> retainedObserver = retained;
    auto sender
        = caravan::alpaka::fill(queue, caravan::Retained{deviceValue, retained}, 0u)
          | caravan::alpaka::sequence(
              caravan::alpaka::copy(
                  queue,
                  caravan::Retained{deviceValue, retained},
                  caravan::Retained{hostValue, retained},
                  one))
          | caravan::alpaka::sequence(
              caravan::alpaka::kernel(
                  queue,
                  threadSpec,
                  Increment{},
                  caravan::retain(alpaka::onHost::data(deviceValue), caravan::Retained{deviceValue, retained})))
          | caravan::alpaka::sequence(
              caravan::alpaka::copy(
                  queue,
                  caravan::Retained{hostValue, retained},
                  caravan::Retained{deviceValue, retained},
                  one))
          | caravan::alpaka::sequence(
              caravan::alpaka::submit(
                  queue,
                  [&, value = std::make_unique<int>(7)](Queue&)
                  {
                      assert(*value == 7);
                      submitted = true;
                  }));

    static_assert(caravan::Sender<decltype(sender)>);
    retained.reset();
    assert(!submitted);
    assert(!retainedObserver.expired());

    caravan::RunLoop loop;
    caravan::AsyncScope scope;

    // Managed operations remain bound to their device context across a host scheduler transfer.
    caravan::alpaka::Context<Device> context{device};
    auto managedInput = alpaka::onHost::alloc<int>(host, one);
    auto managedOutput = alpaka::onHost::alloc<int>(host, one);
    auto managedOtherOutput = alpaka::onHost::alloc<int>(host, one);
    auto managedValue = alpaka::onHost::alloc<int>(device, one);
    auto managedOther = alpaka::onHost::alloc<int>(device, one);
    managedInput[0] = 8;
    managedOutput[0] = 0;
    managedOtherOutput[0] = 0;
    bool changedScheduler = false;
    auto managedCompletion = scope.spawn(
        caravan::alpaka::withDevice(
            context,
            caravan::startsOn(
                loop.scheduler(),
                caravan::whenAll(
                    caravan::alpaka::copy(managedValue, managedInput, one),
                    caravan::alpaka::fill(managedOther, 0u))
                    | caravan::alpaka::sequence(
                        caravan::whenAll(
                            caravan::alpaka::kernel(threadSpec, Increment{}, alpaka::onHost::data(managedValue)),
                            caravan::alpaka::kernel(threadSpec, Increment{}, alpaka::onHost::data(managedOther))))
                    | caravan::alpaka::sequence(caravan::alpaka::copy(managedOutput, managedValue, one))
                    | caravan::continuesOn(loop.scheduler())
                    | caravan::letValue(
                        [&]
                        {
                            changedScheduler = true;
                            return caravan::alpaka::copy(managedOtherOutput, managedOther, one);
                        }))));
    while(managedCompletion.state() == caravan::CompletionState::pending)
    {
        loop.runReady();
        std::this_thread::yield();
    }
    managedCompletion.wait();
    assert(changedScheduler && managedOutput[0] == 9 && managedOtherOutput[0] == 1);

    std::thread::id continuationThread;
    auto completion = scope.spawn(
        std::move(sender) | caravan::continuesOn(loop.scheduler())
        | caravan::then(
            [&]
            {
                assert(!caravan::alpaka::isCompletionCallback());
                continuationThread = std::this_thread::get_id();
            }));
    assert(submitted);

    while(completion.state() == caravan::CompletionState::pending)
    {
        loop.runReady();
        std::this_thread::yield();
    }
    completion.wait();
    assert(hostValue[0] == 42);
    assert(continuationThread == std::this_thread::get_id());
    assert(retainedObserver.expired());

    // A non-owning view and a raw kernel argument survive their original allocation handle's scope.
    auto localStep = [&]
    {
        auto allocation = alpaka::onHost::alloc<int>(device, one);
        auto view = caravan::retain(alpaka::makeView(device, alpaka::onHost::data(allocation), one), allocation);
        return caravan::alpaka::fill(queue, view, 0u)
               | caravan::alpaka::sequence(
                   caravan::alpaka::kernel(
                       queue,
                       threadSpec,
                       Increment{},
                       caravan::retain(alpaka::onHost::data(allocation), view)))
               | caravan::alpaka::sequence(caravan::alpaka::copy(queue, hostValue, view, one));
    }();
    caravan::syncWait(std::move(localStep));
    assert(hostValue[0] == 1);

    // A queue change is lowered to an alpaka event/native wait, not host completion between these copies.
    auto crossInput = alpaka::onHost::alloc<int>(host, one);
    auto crossOutput = alpaka::onHost::alloc<int>(host, one);
    auto crossDevice = alpaka::onHost::alloc<int>(device, one);
    crossInput[0] = 73;
    crossOutput[0] = 0;
    scope
        .spawn(
            caravan::alpaka::copy(queue, crossDevice, crossInput, one)
            | caravan::alpaka::sequence(caravan::alpaka::copy(secondQueue, crossOutput, crossDevice, one)))
        .wait();
    assert(crossOutput[0] == 73);

    // Explicit graph edges lower to native queue ordering without adding the false A -> D dependency.
    std::atomic<unsigned> graphA = 0u, graphB = 0u, graphC = 0u, graphD = 0u;
    auto graphNodeA = caravan::node<"a">(
        caravan::alpaka::submit(queue, [&](Queue& nativeQueue) { enqueueHostTask(nativeQueue, [&] { ++graphA; }); })
        | caravan::alpaka::sequence(
            caravan::alpaka::submit(
                queue,
                [&](Queue& nativeQueue)
                {
                    enqueueHostTask(
                        nativeQueue,
                        [&]
                        {
                            assert(graphA == 1u);
                            ++graphA;
                        });
                })));
    auto graphNodeB = caravan::node<"b">(caravan::alpaka::submit(
        secondQueue,
        [&](Queue& nativeQueue) { enqueueHostTask(nativeQueue, [&] { ++graphB; }); }));
    // Exercise graph-node detection in CUDA/HIP translation units as well as host builds.
    static_assert(caravan::detail::isGraphNode<decltype(graphNodeA)>);
    static_assert(caravan::detail::GraphNodeType<decltype(graphNodeA) const&>);
    static_assert(!caravan::detail::GraphNodeType<int>);
    auto graphNodeC = caravan::node<"c">(
        caravan::alpaka::submit(
            queue,
            [&](Queue& nativeQueue)
            {
                enqueueHostTask(
                    nativeQueue,
                    [&]
                    {
                        assert(graphA == 2u && graphB == 1u);
                        ++graphC;
                    });
            })
            | caravan::alpaka::sequence(
                caravan::alpaka::submit(
                    queue,
                    [&](Queue& nativeQueue)
                    {
                        enqueueHostTask(
                            nativeQueue,
                            [&]
                            {
                                assert(graphC == 1u);
                                ++graphC;
                            });
                    })),
        caravan::after(graphNodeA, graphNodeB));
    auto graphNodeD = caravan::node<"d">(
        caravan::alpaka::submit(
            secondQueue,
            [&](Queue& nativeQueue)
            {
                enqueueHostTask(
                    nativeQueue,
                    [&]
                    {
                        assert(graphB == 1u);
                        ++graphD;
                    });
            }),
        caravan::after(graphNodeB));
    caravan::syncWait(
        caravan::graph(std::move(graphNodeA), std::move(graphNodeB), std::move(graphNodeC), std::move(graphNodeD)));
    assert(graphA == 2u && graphB == 1u && graphC == 2u && graphD == 1u);

    bool mixedNativeRan = false;
    bool mixedHostRan = false;
    auto mixedNative = caravan::node<"native">(caravan::alpaka::submit(
        queue,
        [&](Queue& nativeQueue) { enqueueHostTask(nativeQueue, [&] { mixedNativeRan = true; }); }));
    auto mixedHost = caravan::node<"host">(
        caravan::InlineScheduler{}.schedule()
            | caravan::then(
                [&]
                {
                    assert(mixedNativeRan);
                    mixedHostRan = true;
                }),
        caravan::after(mixedNative));
    caravan::syncWait(caravan::graph(std::move(mixedNative), std::move(mixedHost)));
    assert(mixedNativeRan && mixedHostRan);

    // whenAll and sequence remain native across independent queues and join before the final copy.
    caravan::syncWait(
        caravan::whenAll(
            caravan::alpaka::kernel(queue, threadSpec, Increment{}, alpaka::onHost::data(deviceValue)),
            caravan::alpaka::kernel(secondQueue, threadSpec, Increment{}, alpaka::onHost::data(crossDevice)))
        | caravan::alpaka::sequence(
            caravan::alpaka::kernel(
                queue,
                threadSpec,
                Add{},
                alpaka::onHost::data(deviceValue),
                alpaka::onHost::data(crossDevice)))
        | caravan::alpaka::sequence(caravan::alpaka::copy(queue, hostValue, deviceValue, one)));
    assert(hostValue[0] == 117); // (42 + 1) + (73 + 1)

    bool callbackContextObserved = false;
    scope
        .spawn(
            caravan::alpaka::submit(queue, [](Queue&) {})
            | caravan::then([&] { callbackContextObserved = caravan::alpaka::isCompletionCallback(); }))
        .wait();
    assert(!callbackContextObserved);

    // Supported alpaka queues accept concurrent starts; Caravan adds no submission thread or serialization layer.
    std::atomic<unsigned> callbacks = 0u;
    std::vector<std::thread> submitters;
    for(unsigned i = 0u; i < 8u; ++i)
        submitters.emplace_back(
            [&]
            {
                scope.spawn(
                    caravan::alpaka::submit(
                        queue,
                        [&](Queue& nativeQueue) { enqueueHostTask(nativeQueue, [&] { ++callbacks; }); }));
            });
    for(auto& thread : submitters)
        thread.join();

    // No external owner: retained allocations are reclaimed only after terminal synchronization, off backend
    // callbacks.
    scope.spawn(caravan::alpaka::fill(queue, alpaka::onHost::alloc<int>(device, one), 0u)).wait();

    scope.join().wait();
    assert(callbacks == 8u);
}
