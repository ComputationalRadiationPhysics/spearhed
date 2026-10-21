/* Copyright 2024-2024 Rene Widera
 *
 * This file is part of PMacc.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <cstdint>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

struct VectorConstructorGeneratorKernel
{
    template<typename T_Acc>
    DINLINE void operator()(T_Acc const&, pmacc::math::Vector<uint32_t, TEST_DIM>* data) const
    {
        using pmacc::math::Vector;

        // constexpr lambda generator
        constexpr auto vec = Vector<uint32_t, TEST_DIM>([](uint32_t const i) constexpr { return i; });
        data[0] = vec;
        // non constexpr lambda generator
        data[1] = Vector<uint32_t, TEST_DIM>([](uint32_t const i) { return i * 2u; });
    }
};

TEST_CASE("vector constructor generator", "[vector]")
{
    using namespace pmacc;
    using namespace pmacc::math;
    uint32_t const numElements = 2u;
    using VecType = Vector<uint32_t, TEST_DIM>;

    auto hostDeviceBuffer = HostDeviceBuffer<VecType, DIM1>(DataSpace<DIM1>{numElements});

    auto& device = Environment<>::get().DeviceContext();
    caravan::ControlContext context;
    auto kernel = PMACC_KERNEL(VectorConstructorGeneratorKernel{})(1, 1)(caravan::retain(
        hostDeviceBuffer.getDeviceBuffer().data(),
        hostDeviceBuffer.getDeviceBuffer().getOwnedAlpakaView()));
    auto copy = hostDeviceBuffer.deviceToHost();
    context.wait(
        context.spawn(caravan::alpaka::withDevice(device, std::move(kernel) | caravan::sequence(std::move(copy)))));

    REQUIRE(hostDeviceBuffer.getHostBuffer().data()[0] == Vector<uint32_t, 3u>(0u, 1u, 2u).shrink<TEST_DIM>());
    REQUIRE(hostDeviceBuffer.getHostBuffer().data()[1] == Vector<uint32_t, 3u>(0u, 2u, 4u).shrink<TEST_DIM>());
}
