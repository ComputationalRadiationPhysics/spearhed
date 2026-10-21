/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/alpakaHelper/acc.hpp>
#include <pmacc/device/Reduce.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/buffers/size.hpp>
#include <pmacc/particles/memory/buffers/StackExchangeBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <memory>
#include <thread>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

namespace operations_ut
{
    struct LargeValue
    {
        int values[40];
    };
} // namespace operations_ut

namespace
{
    struct Increment
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC void operator()(T_Acc const&, T_View values) const
        {
            ++values[0];
        }
    };
} // namespace

namespace
{
    [[maybe_unused]] auto compileParticleStackSizes(pmacc::StackExchangeBuffer<int, int, DIM1>& stack)
    {
        return stack.reset() | caravan::sequence(stack.publishDeviceSizes());
    }
} // namespace

TEST_CASE("PMacc explicitly composes and owns a local accelerator step", "[async][memory]")
{
    auto const deviceManager = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
    pmacc::ComputeDeviceQueue queue = caravan::alpaka::detail::makeQueue<pmacc::ComputeDeviceQueue>(deviceManager);
    auto const one = pmacc::MemSpace<DIM1>::create(1);
    auto const extent = one.toAlpakaMemVec();
    auto const workExtent = pmacc::DataSpace<DIM1>::create(1).toAlpakaKernelVec();
    auto const threadSpec = ::alpaka::onHost::ThreadSpec{workExtent, workExtent};

    auto input = std::make_unique<pmacc::HostBuffer<int, DIM1>>(one);
    auto device = std::make_unique<pmacc::DeviceBuffer<int, DIM1>>(one, true);
    pmacc::HostBuffer<int, DIM1> output(one);
    input->data()[0] = 41;

    auto step = caravan::alpaka::fill(queue, device->getOwnedAlpakaView(), 0u)
                | caravan::sequence(
                    caravan::alpaka::copy(queue, device->getOwnedAlpakaView(), input->getOwnedAlpakaView(), extent))
                | caravan::sequence(
                    caravan::alpaka::kernel(
                        queue,
                        threadSpec,
                        Increment{},
                        caravan::retain(device->data(), device->getOwnedAlpakaView())))
                | caravan::sequence(pmacc::size(queue, device->sizeOnDeviceBuffer(), device->sizeHostSideBuffer()))
                | caravan::sequence(
                    caravan::alpaka::copy(queue, output.getOwnedAlpakaView(), device->getOwnedAlpakaView(), extent));

    caravan::ControlContext context;
    auto const applicationThread = std::this_thread::get_id();
    bool continued = false;
    auto completion = context.spawn(
        context.onControl(std::move(step))
        | caravan::then(
            [&]
            {
                CHECK(std::this_thread::get_id() == applicationThread);
                continued = true;
            }));
    input.reset();
    device.reset();
    context.wait(completion);

    CHECK(output.data()[0] == 42);
    CHECK(continued);
}

TEST_CASE("PMacc size copies synchronize buffer size storage", "[async][memory]")
{
    auto& device = pmacc::Environment<>::get().DeviceContext();
    pmacc::ComputeDeviceQueue queue = caravan::alpaka::detail::makeQueue<pmacc::ComputeDeviceQueue>(
        pmacc::manager::Device<pmacc::ComputeDevice>::get().current());
    pmacc::DeviceBuffer<int, DIM1> buffer(pmacc::MemSpace<DIM1>{128u}, true);
    pmacc::HostBuffer<int, DIM1> output(pmacc::MemSpace<DIM1>{128u});
    buffer.setSizeHostSide(123u);
    output.setSizeHostSide(0u);

    SECTION("Explicit queue")
    {
        caravan::syncWait(
            pmacc::size(queue, buffer.sizeOnDeviceBuffer(), buffer.sizeHostSideBuffer())
            | caravan::alpaka::sequence(
                pmacc::size(queue, output.getOwnedSizeHostBuffer(), buffer.sizeOnDeviceBuffer())));
    }
    SECTION("Managed queue")
    {
        caravan::syncWait(
            caravan::alpaka::withDevice(
                device,
                pmacc::size(buffer.sizeOnDeviceBuffer(), buffer.sizeHostSideBuffer())
                    | caravan::alpaka::sequence(
                        pmacc::size(output.getOwnedSizeHostBuffer(), buffer.sizeOnDeviceBuffer()))));
    }
    CHECK(output.size() == 123u);
}

TEST_CASE("DeviceBuffer value fill is a lazy sender", "[async][memory]")
{
    auto& device = pmacc::Environment<>::get().DeviceContext();
    auto const extent = pmacc::MemSpace<DIM1>{3u};

    pmacc::HostDeviceBuffer<int, DIM1> small(extent);
    auto smallFill = small.getDeviceBuffer().setValue(42);
    static_assert(caravan::Sender<decltype(smallFill)>);
    caravan::syncWait(caravan::alpaka::withDevice(device, std::move(smallFill)));
    caravan::syncWait(caravan::alpaka::withDevice(device, small.deviceToHost()));
    for(size_t i = 0u; i < 3u; ++i)
        CHECK(small.getHostBuffer().data()[i] == 42);

    pmacc::HostDeviceBuffer<operations_ut::LargeValue, DIM1> large(extent);
    operations_ut::LargeValue value{};
    value.values[0] = 17;
    value.values[39] = 23;
    caravan::syncWait(caravan::alpaka::withDevice(device, large.getDeviceBuffer().setValue(value)));
    caravan::syncWait(caravan::alpaka::withDevice(device, large.deviceToHost()));
    for(size_t i = 0u; i < 3u; ++i)
    {
        CHECK(large.getHostBuffer().data()[i].values[0] == 17);
        CHECK(large.getHostBuffer().data()[i].values[39] == 23);
    }
}

TEST_CASE("Device reduction returns a lazy sender", "[async][reduce]")
{
    auto& device = pmacc::Environment<>::get().DeviceContext();
    pmacc::HostDeviceBuffer<int, DIM1> input(pmacc::MemSpace<DIM1>{4u});
    input.getHostBuffer().data()[0] = 1;
    input.getHostBuffer().data()[1] = 2;
    input.getHostBuffer().data()[2] = 3;
    input.getHostBuffer().data()[3] = 4;
    caravan::syncWait(caravan::alpaka::withDevice(device, input.hostToDevice()));

    pmacc::device::Reduce reduce(1024u);
    auto reduction = reduce.reduce(pmacc::math::operation::Add{}, input.getDeviceBuffer().getDataBox(), 4u);
    static_assert(caravan::Sender<decltype(reduction)>);
    auto result = caravan::syncWait<int>(caravan::alpaka::withDevice(device, std::move(reduction)));

    CHECK(result == 10);
}

TEST_CASE("Host-device buffer queue overloads return lazy copies", "[async][memory]")
{
    auto& device = pmacc::Environment<>::get().DeviceContext();
    pmacc::HostDeviceBuffer<int, DIM1> buffer(pmacc::MemSpace<DIM1>{2u}, true);
    buffer.getHostBuffer().data()[0] = 41;
    buffer.getHostBuffer().data()[1] = 99;
    buffer.getHostBuffer().setSizeHostSide(1u);

    caravan::ControlContext context;
    context.wait(context.spawn(caravan::alpaka::withDevice(device, buffer.hostToDevice())));

    buffer.getDeviceBuffer().setSizeHostSide(0u);
    context.wait(context.spawn(
        caravan::alpaka::withDevice(
            device,
            pmacc::size(
                buffer.getDeviceBuffer().sizeHostSideBuffer(),
                buffer.getDeviceBuffer().sizeOnDeviceBuffer()))));
    CHECK(buffer.getDeviceBuffer().size() == 1u);

    buffer.getHostBuffer().data()[0] = 0;
    buffer.getHostBuffer().data()[1] = 77;
    context.wait(context.spawn(caravan::alpaka::withDevice(device, buffer.deviceToHost())));
    CHECK(buffer.getHostBuffer().size() == 1u);
    CHECK(buffer.getHostBuffer().data()[0] == 41);
    CHECK(buffer.getHostBuffer().data()[1] == 77);
}
