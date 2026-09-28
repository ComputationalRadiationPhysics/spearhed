/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

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

    thread_local unsigned allocationFailures = 0u;
    std::weak_ptr<int> fatalOwner;
    std::atomic<bool> fatalTaskFinished = false;
} // namespace

// Inject only on the submitting thread, after a native task has borrowed retained storage.
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
void* operator new(std::size_t bytes)
{
    if(allocationFailures != 0u)
    {
        --allocationFailures;
        throw std::bad_alloc{};
    }
    if(auto* memory = std::malloc(bytes == 0u ? 1u : bytes))
        return memory;
    throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}
#endif

int main(int argc, char** argv)
{
    // These regressions exercise CPU queues even in CUDA/HIP builds; no accelerator device is required.
    constexpr auto computeApi = alpaka::api::host;
    constexpr auto computeDeviceKind = alpaka::deviceKind::cpu;
    using Device = alpaka::onHost::Device<ALPAKA_TYPEOF(computeApi), ALPAKA_TYPEOF(computeDeviceKind)>;
    using Queue = alpaka::onHost::Queue<Device, alpaka::queueKind::NonBlocking>;
    using BlockingQueue = alpaka::onHost::Queue<Device, alpaka::queueKind::Blocking>;
    auto const device = alpaka::onHost::makeDeviceSelector(computeApi, computeDeviceKind).makeDevice(0u);
    auto queue = caravan::alpaka::detail::makeQueue<Queue>(device);
    auto secondQueue = caravan::alpaka::detail::makeQueue<Queue>(device);
    auto blockerQueue = caravan::alpaka::detail::makeQueue<Queue>(device);
    caravan::AsyncScope scope;

    if(argc > 1 && std::string_view(argv[1]) == "--fatal-fence")
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto storage = std::make_shared<int>(42);
        fatalOwner = storage;
        std::set_terminate(
            [] { std::_Exit(!fatalOwner.expired() && !fatalTaskFinished.load() ? EXIT_SUCCESS : EXIT_FAILURE); });
        scope.spawn(
            caravan::alpaka::submit(
                queue,
                [storage = std::move(storage), gate](Queue& nativeQueue)
                {
                    enqueueHostTask(
                        nativeQueue,
                        [raw = storage.get(), gate]
                        {
                            gate.wait();
                            assert(*raw == 42);
                            fatalTaskFinished = true;
                        });
                    allocationFailures = std::numeric_limits<unsigned>::max();
                }));
        // Expected termination must occur in cleanup, before receiver delivery or scope destruction.
        std::_Exit(EXIT_FAILURE);
    }

    // Queue-free graphs preserve fork/join dependencies and lease another queue for concurrent work.
    {
        caravan::alpaka::QueuePool<Queue> pool{device};
        std::array<int, 4u> values{};
        int seed = 0;
        auto branch = [&](std::size_t index)
        {
            return caravan::alpaka::enqueue(
                [&, index]
                {
                    assert(seed == 42);
                    if(index >= 2u)
                        assert(values[1u] == seed + 1);
                    values[index] = seed + static_cast<int>(index);
                });
        };
        auto work
            = caravan::alpaka::enqueue([&] { seed = 42; })
              | caravan::sequence(
                  caravan::whenAll(
                      branch(0u),
                      branch(1u) | caravan::alpaka::sequence(caravan::whenAll(branch(2u), branch(3u)))))
              | caravan::sequence(caravan::alpaka::enqueue([&] { assert((values == std::array{42, 43, 44, 45})); }));
        static_assert(caravan::Sender<decltype(work)>);
        static_assert(std::is_same_v<decltype(caravan::getDomain(work)), caravan::alpaka::ManagedSubmissionDomain>);
        caravan::syncWait(caravan::alpaka::withDevice(pool, std::move(work)));

        int stage = 0;
        caravan::syncWait(
            caravan::alpaka::withDevice(
                pool,
                caravan::alpaka::enqueue([&] { stage = 1; }) | caravan::then([&] { assert(stage == 1); })
                    | caravan::sequence(caravan::alpaka::enqueue([&] { stage = 2; }))
                    | caravan::then([&] { assert(stage == 2); })
                    | caravan::sequence(caravan::alpaka::enqueue([&] { stage = 3; }))));
        assert(stage == 3);

        std::promise<void> release, started;
        auto gate = release.get_future().share();
        auto branchStarted = started.get_future();
        std::thread busy(
            [&]
            {
                caravan::syncWait(
                    caravan::alpaka::withDevice(
                        pool,
                        caravan::alpaka::enqueue(
                            [&, gate]
                            {
                                started.set_value();
                                gate.wait();
                            })));
            });
        branchStarted.get();
        bool concurrentRan = false;
        caravan::syncWait(caravan::alpaka::withDevice(pool, caravan::alpaka::enqueue([&] { concurrentRan = true; })));
        assert(concurrentRan);
        release.set_value();
        busy.join();
    }

    // Pool-bound senders use the same native dispatch and retain queue affinity.
    {
        caravan::alpaka::QueuePool<Queue> pool{device};
        auto submissions = pool.submissions();
        int value = 0;
        auto work = caravan::sequence(
            submissions.submit([&](Queue& q) { enqueueHostTask(q, [&] { value = 42; }); }),
            submissions.submit([&](Queue& q) { enqueueHostTask(q, [&] { assert(value == 42); }); }));
        static_assert(
            std::is_same_v<decltype(caravan::getDomain(work)), caravan::alpaka::PooledSubmissionDomain<Queue>>);
        caravan::syncWait(std::move(work));
    }

    // Independently bound contexts join through ordinary sender composition.
    {
        caravan::alpaka::QueuePool<Queue> firstPool{device}, secondPool{device};
        bool firstRan = false, secondRan = false;
        caravan::syncWait(
            caravan::whenAll(
                caravan::alpaka::withDevice(firstPool, caravan::alpaka::enqueue([&] { firstRan = true; })),
                caravan::alpaka::withDevice(secondPool, caravan::alpaka::enqueue([&] { secondRan = true; }))));
        assert(firstRan && secondRan);
    }

    // Native fork/join submits the join before host completion, without serializing branches.
    // The fork follows a seed on queue A; branch B must start even while branch A is blocked.
    {
        std::promise<void> release, branchStarted;
        auto gate = release.get_future().share();
        auto started = branchStarted.get_future();
        int seed = 0, left = 0, right = 0;
        bool joinSubmitted = false;
        std::atomic<bool> hostCompleted = false;
        auto seedWork = caravan::alpaka::submit(queue, [&](Queue& q) { enqueueHostTask(q, [&] { seed = 42; }); });
        static_assert(
            std::is_same_v<decltype(caravan::getDomain(seedWork)), caravan::alpaka::SubmissionDomain<Queue>>);
        auto work = std::move(seedWork)
                    | caravan::sequence(
                        caravan::whenAll(
                            caravan::alpaka::submit(
                                queue,
                                [&](Queue& q)
                                {
                                    enqueueHostTask(
                                        q,
                                        [&]
                                        {
                                            gate.wait();
                                            left = seed;
                                        });
                                }),
                            caravan::whenAll(
                                caravan::alpaka::submit(
                                    secondQueue,
                                    [&](Queue& q)
                                    {
                                        enqueueHostTask(
                                            q,
                                            [&]
                                            {
                                                branchStarted.set_value();
                                                gate.wait();
                                                right = seed;
                                            });
                                    }),
                                caravan::alpaka::submit(blockerQueue, [](Queue&) {}))))
                    | caravan::sequence(
                        caravan::alpaka::submit(
                            queue,
                            [&, owner = std::make_unique<int>(42)](Queue& q)
                            {
                                joinSubmitted = true;
                                enqueueHostTask(q, [&, raw = owner.get()] { assert(left == *raw && right == *raw); });
                            }))
                    | caravan::then(
                        [&]
                        {
                            assert(left == 42 && right == 42);
                            hostCompleted = true;
                        });
        auto done = scope.spawn(std::move(work));
        assert(joinSubmitted);
        assert(!hostCompleted);
        assert(done.state() == caravan::CompletionState::pending);
        started.get();
        release.set_value();
        done.wait();
        assert(hostCompleted);
    }

    // A mixed join falls back to host completion, waiting for both native work and the event.
    {
        caravan::EventSource release;
        std::promise<void> nativeRelease;
        auto gate = nativeRelease.get_future().share();
        std::atomic<bool> submitted = false;
        auto done = scope.spawn(
            caravan::whenAll(
                caravan::alpaka::submit(queue, [gate](Queue& q) { enqueueHostTask(q, [gate] { gate.wait(); }); }),
                caravan::asSender(release.event()))
            | caravan::sequence(caravan::alpaka::submit(secondQueue, [&](Queue&) { submitted = true; })));
        assert(!submitted);
        release.setReady();
        assert(!submitted && done.state() == caravan::CompletionState::pending);
        nativeRelease.set_value();
        done.wait();
        assert(submitted);
    }

    // Neither an ordinary host callback nor explicit placement can be fused away.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        std::atomic<bool> submitted = false;
        auto done = scope.spawn(
            caravan::alpaka::submit(queue, [gate](Queue& q) { enqueueHostTask(q, [gate] { gate.wait(); }); })
            | caravan::then([] { assert(caravan::isExecutorThread()); })
            | caravan::sequence(caravan::alpaka::submit(secondQueue, [&](Queue&) { submitted = true; })));
        assert(!submitted);
        release.set_value();
        done.wait();
        assert(submitted);

        submitted = false;
        caravan::RunLoop loop;
        auto placed = scope.spawn(
            caravan::alpaka::submit(queue, [](Queue&) {}) | caravan::continuesOn(loop.scheduler())
            | caravan::sequence(caravan::alpaka::submit(secondQueue, [&](Queue&) { submitted = true; })));
        assert(!submitted);
        while(placed.state() == caravan::CompletionState::pending)
        {
            loop.runReady();
            std::this_thread::yield();
        }
        placed.wait();
        assert(submitted);
    }

    // A pending queue must not hold up a ready queue whose continuation releases it.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto pending = scope.spawn(
            caravan::alpaka::submit(
                queue,
                [gate](Queue& nativeQueue) { enqueueHostTask(nativeQueue, [gate] { gate.wait(); }); }));
        scope.spawn(caravan::alpaka::submit(secondQueue, [](Queue&) {}) | caravan::then([&] { release.set_value(); }))
            .wait();
        pending.wait();
    }

    // A late observer does not wait for a later operation on the same queue.
    {
        std::promise<void> release, entered;
        auto gate = release.get_future().share();
        auto started = entered.get_future();
        auto earlier = scope.spawn(caravan::alpaka::submit(queue, [](Queue&) {}));
        auto later = scope.spawn(
            caravan::alpaka::submit(
                queue,
                [&](Queue& nativeQueue)
                {
                    enqueueHostTask(
                        nativeQueue,
                        [&, gate]
                        {
                            entered.set_value();
                            gate.wait();
                        });
                }));
        started.get();
        earlier.wait();
        assert(later.state() == caravan::CompletionState::pending);
        release.set_value();
        later.wait();
    }

    // Work appended after a sender may depend on that sender's completion without moving its fence.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto blocker = scope.spawn(
            caravan::alpaka::submit(
                blockerQueue,
                [gate](Queue& nativeQueue) { enqueueHostTask(nativeQueue, [gate] { gate.wait(); }); }));
        auto predecessor = scope.spawn(caravan::alpaka::submit(queue, [](Queue&) {}));
        enqueueHostTask(queue, [predecessor] { predecessor.wait(); });
        release.set_value();
        predecessor.wait();
        blocker.wait();
        alpaka::onHost::wait(queue);
    }

    // Blocking CPU queues exercise the native-event fence implementation too.
    auto blockingQueue = caravan::alpaka::detail::makeQueue<BlockingQueue>(device);
    auto secondBlockingQueue = caravan::alpaka::detail::makeQueue<BlockingQueue>(device);
    scope
        .spawn(
            caravan::alpaka::sequence(
                caravan::alpaka::submit(blockingQueue, [](auto&) {}),
                caravan::alpaka::submit(secondBlockingQueue, [](auto&) {})))
        .wait();

    // Receiver delivery uses the same blocking guards as the other Caravan progress authorities.
    bool nestedWaitRejected = false;
    scope
        .spawn(
            caravan::alpaka::submit(queue, [](Queue&) {})
            | caravan::then(
                [&]
                {
                    assert(caravan::isExecutorThread());
                    try
                    {
                        caravan::syncWait(caravan::alpaka::submit(secondQueue, [](Queue&) {}));
                    }
                    catch(std::logic_error const&)
                    {
                        nestedWaitRejected = true;
                    }
                }))
        .wait();
    assert(nestedWaitRejected);

    scope.join().wait();
}
