/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <thread>
#include <utility>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

namespace
{
    thread_local bool countAllocations = false;
    thread_local std::size_t allocationCount = 0u;

    struct Increment
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value) const
        {
            ++*value;
        }
    };

    struct VoidReceiver
    {
        void set_value() noexcept
        {
            output.setReady();
        }

        caravan::EventSource output;
    };
} // namespace

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
void* operator new(std::size_t bytes)
{
    if(countAllocations)
        ++allocationCount;
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
    return caravan::MpiRuntime::run(
        argc,
        argv,
        [](caravan::MpiContext& mpi)
        {
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
            auto const host
                = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu).makeDevice(0u);
            auto queue = caravan::alpaka::detail::makeQueue<Queue>(device);
            auto const one = alpaka::Vec{Idx{1u}};
            auto const threadSpec = alpaka::onHost::ThreadSpec{one, one};
            auto producerInput = alpaka::onHost::alloc<int>(host, one);
            auto producerDevice = alpaka::onHost::alloc<int>(device, one);
            auto mpiInput = alpaka::onHost::alloc<int>(host, one);
            auto received = alpaka::onHost::alloc<int>(host, one);
            auto consumerDevice = alpaka::onHost::alloc<int>(device, one);
            auto consumerOutput = alpaka::onHost::alloc<int>(host, one);
            producerInput[0] = mpi.topology().rank;
            mpiInput[0] = received[0] = consumerOutput[0] = -1;

            auto const applicationThread = std::this_thread::get_id();

            // Ordinary sender construction and connection remain allocation-free.
            caravan::EventSource allocationOutput;
            auto allocationResult = allocationOutput.event();
            allocationCount = 0u;
            countAllocations = true;
            auto allocationSender = caravan::mpi::barrier(mpi);
            auto allocationOperation = std::move(allocationSender).connect(VoidReceiver{allocationOutput});
            countAllocations = false;
            assert(allocationCount == 0u);
            allocationOperation.start();
            allocationResult.wait();

            // Real A -> M -> B: A produces MPI's input; B consumes and transforms its receive.
            std::thread::id aCompletionThread;
            std::thread::id mpiCompletionThread;
            std::thread::id bSubmissionThread;
            std::thread::id bCompletionThread;
            caravan::syncWait(
                caravan::alpaka::copy(queue, producerDevice, producerInput, one)
                | caravan::alpaka::sequence(
                    caravan::alpaka::kernel(queue, threadSpec, Increment{}, alpaka::onHost::data(producerDevice)))
                | caravan::alpaka::sequence(caravan::alpaka::copy(queue, mpiInput, producerDevice, one))
                | caravan::letValue(
                    [&]
                    {
                        aCompletionThread = std::this_thread::get_id();
                        return caravan::whenAll(
                            caravan::mpi::send(
                                mpi,
                                std::as_bytes(std::span{&mpiInput[0], 1}),
                                caravan::Peer{mpi.topology().rank},
                                caravan::MessageTag{951}),
                            caravan::mpi::receive(
                                mpi,
                                std::as_writable_bytes(std::span{&received[0], 1}),
                                caravan::Peer{mpi.topology().rank},
                                caravan::MessageTag{951}));
                    })
                | caravan::letValue(
                    [&](caravan::SendResult const& sentResult, caravan::ReceiveResult const& receivedResult)
                    {
                        assert(sentResult.bytes == sizeof(int) && receivedResult.bytes == sizeof(int));
                        mpiCompletionThread = std::this_thread::get_id();
                        return caravan::alpaka::submit(
                                   queue,
                                   [&](Queue&) { bSubmissionThread = std::this_thread::get_id(); })
                               | caravan::alpaka::sequence(caravan::alpaka::copy(queue, consumerDevice, received, one))
                               | caravan::alpaka::sequence(
                                   caravan::alpaka::kernel(
                                       queue,
                                       threadSpec,
                                       Increment{},
                                       alpaka::onHost::data(consumerDevice)))
                               | caravan::alpaka::sequence(
                                   caravan::alpaka::copy(queue, consumerOutput, consumerDevice, one));
                    })
                | caravan::then([&] { bCompletionThread = std::this_thread::get_id(); }));
            assert(mpiInput[0] == mpi.topology().rank + 1);
            assert(received[0] == mpiInput[0]);
            assert(consumerOutput[0] == received[0] + 1);
            assert(aCompletionThread != applicationThread);
            assert(mpiCompletionThread == bSubmissionThread);
            assert(bCompletionThread == aCompletionThread);
            assert(bCompletionThread != mpiCompletionThread);

            // Without an explicit transfer, MPI completion starts the next submission on the owner.
            std::thread::id directMpiCompletion;
            std::thread::id directBSubmission;
            caravan::syncWait(
                caravan::alpaka::submit(queue, [](Queue&) {})
                | caravan::letValue(
                    [&]
                    {
                        return caravan::mpi::barrier(mpi)
                               | caravan::then([&] { directMpiCompletion = std::this_thread::get_id(); });
                    })
                | caravan::letValue(
                    [&]
                    {
                        return caravan::alpaka::submit(
                            queue,
                            [&](Queue&) { directBSubmission = std::this_thread::get_id(); });
                    }));
            assert(directMpiCompletion == directBSubmission);

            // Application restoration keeps the requested hop: B waits for the application loop.
            caravan::RunLoop loop;
            caravan::AsyncScope scope;
            std::atomic<bool> applicationMpiCompleted = false;
            std::atomic<bool> applicationBSubmitted = false;
            std::thread::id applicationBSubmissionThread;
            auto completion = scope.spawn(
                caravan::alpaka::submit(queue, [](Queue&) {})
                | caravan::letValue(
                    [&]
                    { return caravan::mpi::barrier(mpi) | caravan::then([&] { applicationMpiCompleted = true; }); })
                | caravan::continuesOn(loop.scheduler())
                | caravan::letValue(
                    [&]
                    {
                        return caravan::alpaka::submit(
                            queue,
                            [&](Queue&)
                            {
                                applicationBSubmissionThread = std::this_thread::get_id();
                                applicationBSubmitted = true;
                            });
                    }));

            while(!applicationMpiCompleted.load())
                std::this_thread::yield();
            assert(!applicationBSubmitted.load());
            while(completion.state() == caravan::CompletionState::pending)
            {
                loop.runReady();
                std::this_thread::yield();
            }
            completion.wait();
            scope.join().wait();
            assert(applicationBSubmitted.load());
            assert(applicationBSubmissionThread == applicationThread);
            return 0;
        });
}
