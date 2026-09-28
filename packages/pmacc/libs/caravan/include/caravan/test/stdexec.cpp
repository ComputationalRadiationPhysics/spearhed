/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstddef>
#include <latch>
#include <memory>
#include <span>
#include <thread>
#include <utility>

#include <caravan/alpaka.hpp>
#include <caravan/mpi.hpp>
#include <caravan/stdexec.hpp>
#include <exec/async_scope.hpp>
#include <stdexec/execution.hpp>

namespace
{
    struct Preserve
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value) const
        {
            *value += 0;
        }
    };
} // namespace

int main(int argc, char** argv)
{
    return caravan::MpiRuntime::run(
        argc,
        argv,
        [](caravan::MpiContext& mpi)
        {
            namespace ex = stdexec;
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
            auto queue = caravan::alpaka::detail::makeQueue<Queue>(device);
            auto const one = alpaka::Vec{Idx{1u}};
            auto const threadSpec = alpaka::onHost::ThreadSpec{one, one};
            auto value = alpaka::onHost::alloc<int>(device, one);
            int sent = mpi.topology().rank;

            stdexec::run_loop controlLoop;
            std::thread::id controlThread;
            auto chain = caravan::stdexecInterop::adapt(
                             caravan::alpaka::kernel(queue, threadSpec, Preserve{}, alpaka::onHost::data(value)))
                         | ex::let_value(
                             [&]
                             {
                                 return caravan::stdexecInterop::adapt(
                                     caravan::mpi::send(
                                         mpi,
                                         std::as_bytes(std::span{&sent, 1}),
                                         caravan::Peer{mpi.topology().rank},
                                         caravan::MessageTag{952}));
                             })
                         | ex::continues_on(controlLoop.get_scheduler())
                         | ex::then(
                             [&](caravan::SendResult result)
                             {
                                 assert(std::this_thread::get_id() == controlThread);
                                 return result.bytes;
                             });

            static_assert(ex::sender<decltype(chain)>);
#if ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED
            // Translation-only gate: NVIDIA/stdexec async_scope does not instantiate with nvcc 13.3.
            return 0;
#else
            int received = -1;
            std::latch running{1};
            std::jthread control(
                [&]
                {
                    controlThread = std::this_thread::get_id();
                    running.count_down();
                    controlLoop.run();
                });
            running.wait();

            exec::async_scope scope;
            auto incoming = scope.spawn_future(
                caravan::stdexecInterop::adapt(
                    caravan::mpi::receive(
                        mpi,
                        std::as_writable_bytes(std::span{&received, 1}),
                        caravan::Peer{mpi.topology().rank},
                        caravan::MessageTag{952})));
            auto result = scope.spawn_future(std::move(chain));
            auto sentBytes = ex::sync_wait(std::move(result));
            auto receiveResult = ex::sync_wait(std::move(incoming));
            ex::sync_wait(scope.on_empty());
            controlLoop.finish();

            assert(std::get<0>(*sentBytes) == sizeof(sent));
            assert(std::get<0>(*receiveResult).bytes == sizeof(received));
            assert(received == sent);
            return 0;
#endif
        });
}
