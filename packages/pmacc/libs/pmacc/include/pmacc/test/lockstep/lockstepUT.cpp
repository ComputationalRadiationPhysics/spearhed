/* Copyright 2015-2024 Erik Zenker, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/verify.hpp>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

/** @file
 *
 *  This file is testing common lockstep pattern.
 *  There are many code duplications, those are necessary because the code snippets are include into the documentation.
 */

constexpr uint32_t numElements = 4096u;

// doc-include-start: lockstep generic kernel
struct IotaGenericKernel
{
    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        constexpr uint32_t blockDomSize = T_Worker::blockDomSize();
        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        // grid-strided loop over the chunked data
        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = idx;
                    }
                });
        }
    }
};

template<uint32_t T_chunkSize, typename T_DeviceBuffer>
inline auto iotaGerneric(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.size();
    // use only half of the blocks needed to process the full data
    uint32_t const numBlocks = bufferSize / T_chunkSize / 2u;
    return PMACC_LOCKSTEP_KERNEL(IotaGenericKernel{})
        .config<T_chunkSize>(numBlocks)(devBuffer.getOwnedDataBox(), bufferSize);
}

// doc-include-end: lockstep generic kernel

// doc-include-start: lockstep generic kernel buffer selected domain size
namespace pmacc::lockstep::traits
{
    //! Specialization to create a lockstep block configuration out of a device buffer.
    template<>
    struct MakeBlockCfg<pmacc::DeviceBuffer<uint32_t, DIM1>> : std::true_type
    {
        using type = BlockCfg<math::CT::UInt32<53>>;
    };
} // namespace pmacc::lockstep::traits

template<typename T_DeviceBuffer>
inline auto iotaGernericBufferDerivedChunksize(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.size();
    constexpr uint32_t numBlocks = 9;
    return PMACC_LOCKSTEP_KERNEL(IotaGenericKernel{})
        .config(numBlocks, devBuffer)(devBuffer.getOwnedDataBox(), bufferSize);
}

// doc-include-end: lockstep generic kernel buffer selected domain size

// doc-include-start: lockstep generic kernel hard coded domain size
struct IotaFixedChunkSizeKernel
{
    static constexpr uint32_t blockDomSize = 42;

    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        static_assert(blockDomSize == T_Worker::blockDomSize());

        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        // grid-strided loop over the chunked data
        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = idx;
                    }
                });
        }
    }
};

template<typename T_DeviceBuffer>
inline auto iotaFixedChunkSize(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.size();
    constexpr uint32_t numBlocks = 10;
    return PMACC_LOCKSTEP_KERNEL(IotaFixedChunkSizeKernel{})
        .config(numBlocks)(devBuffer.getOwnedDataBox(), bufferSize);
}

// doc-include-end: lockstep generic kernel hard coded domain size

// doc-include-start: lockstep generic kernel hard coded N dimensional domain size
struct IotaFixedChunkSizeKernelND
{
    using BlockDomSizeND = pmacc::math::CT::UInt32<42>;

    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        static constexpr uint32_t blockDomSize = BlockDomSizeND::x::value;

        static_assert(blockDomSize == T_Worker::blockDomSize());

        // grid-strided loop over the chunked data
        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = idx;
                    }
                });
        }
    }
};

template<typename T_DeviceBuffer>
inline auto iotaFixedChunkSizeND(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.size();
    constexpr uint32_t numBlocks = 11;
    return PMACC_LOCKSTEP_KERNEL(IotaFixedChunkSizeKernelND{})
        .config(numBlocks)(devBuffer.getOwnedDataBox(), bufferSize);
}

// doc-include-end: lockstep generic kernel hard coded N dimensional domain size

// doc-include-start: lockstep generic kernel with dynamic shared memory
struct IotaGenericKernelWithDynSharedMem
{
    template<typename T_Worker, typename T_DataBox>
    HDINLINE void operator()(T_Worker const& worker, T_DataBox data, uint32_t size) const
    {
        constexpr uint32_t blockDomSize = T_Worker::blockDomSize();
        auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

        uint32_t* s_mem = ::alpaka::onAcc::getDynSharedMem<uint32_t>(worker.getAcc());

        // grid-strided loop over the chunked data
        for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
        {
            auto dataBlockOffset = dataBlock * blockDomSize;
            auto forEach = pmacc::lockstep::makeForEach(worker);
            forEach(
                [&](uint32_t const inBlockIdx)
                {
                    auto idx = dataBlockOffset + inBlockIdx;
                    s_mem[inBlockIdx] = idx;
                    if(idx < size)
                    {
                        // ensure that each block is not overwriting data from other blocks
                        PMACC_DEVICE_VERIFY_MSG(data[idx] == 0u, "%s\n", "Result buffer not valid initialized!");
                        data[idx] = s_mem[inBlockIdx];
                    }
                });
        }
    }
};

template<uint32_t T_chunkSize, typename T_DeviceBuffer>
inline auto iotaGernericWithDynSharedMem(T_DeviceBuffer& devBuffer)
{
    auto bufferSize = devBuffer.size();
    // use only half of the blocks needed to process the full data
    uint32_t const numBlocks = bufferSize / T_chunkSize / 2u;
    constexpr size_t requiredSharedMemBytes = T_chunkSize * sizeof(uint32_t);
    return PMACC_LOCKSTEP_KERNEL(IotaGenericKernelWithDynSharedMem{})
        .configSMem<T_chunkSize>(numBlocks, requiredSharedMemBytes)(devBuffer.getOwnedDataBox(), bufferSize);
}

// doc-include-end: lockstep generic kernel with dynamic shared memory

template<typename T_HostBuffer>
void validate(T_HostBuffer& results, T_HostBuffer& reference)
{
    auto refBufferSize = reference.size();
    auto* refPtr = reference.data();

    auto resultBufferSize = results.size();
    auto* resultPtr = results.data();

    PMACC_VERIFY(resultBufferSize == refBufferSize);
    for(uint32_t i = 0u; i < refBufferSize; ++i)
    {
        REQUIRE(refPtr[i] == resultPtr[i]);
    }
}

#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1 && defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
struct ThrowingKernel
{
    template<typename T_Worker>
    HDINLINE void operator()(T_Worker const&) const
    {
        ALPAKA_THROW_ACC("blocking kernel diagnostic test");
    }
};

TEST_CASE("blocking kernel diagnostics", "[lockstep]")
{
    using namespace pmacc;

    auto const device = manager::Device<ComputeDevice>::get().current();
    ComputeDeviceQueue queue = caravan::alpaka::detail::makeQueue<ComputeDeviceQueue>(device);
    std::ostringstream diagnostics;
    auto* const previousBuffer = std::cerr.rdbuf(diagnostics.rdbuf());
    auto const sourceLine = __LINE__ + 1u;
    auto kernel = PMACC_LOCKSTEP_KERNEL(ThrowingKernel{}).config<1>(1u)(queue);
    bool continuationRan = false;
    bool failed = false;
    try
    {
        caravan::syncWait(std::move(kernel) | caravan::then([&] { continuationRan = true; }));
    }
    catch(std::exception const&)
    {
        failed = true;
    }
    std::cerr.rdbuf(previousBuffer);

    CHECK(failed);
    CHECK_FALSE(continuationRan);
    CHECK(diagnostics.str().find("Crash after kernel call") != std::string::npos);
    CHECK(diagnostics.str().find(std::string(__FILE__) + ":" + std::to_string(sourceLine)) != std::string::npos);
}
#endif

TEST_CASE("lockstep kernel", "[iota]")
{
    using namespace pmacc;

    auto referenceBuffer = HostBuffer<uint32_t, DIM1>(DataSpace<DIM1>{numElements});

    auto* refHostPtr = referenceBuffer.data();
    for(uint32_t i = 0u; i < numElements; ++i)
    {
        refHostPtr[i] = i;
    }

    auto hostDeviceBuffer = HostDeviceBuffer<uint32_t, DIM1>(DataSpace<DIM1>{numElements});
    using DeviceBuf = DeviceBuffer<uint32_t, DIM1>;
    auto& device = Environment<>::get().DeviceContext();
    caravan::ControlContext context;

    // register all required test functions
    auto testsFunctions = std::make_tuple(
        // generic host size chunk size selection
        iotaGerneric<128, DeviceBuf>,
        iotaGerneric<16, DeviceBuf>,
        // generic host size chunk size selection and dynamic shared memory
        iotaGernericWithDynSharedMem<23, DeviceBuf>,
        // derive the chunk size from the result buffer
        iotaGernericBufferDerivedChunksize<DeviceBuf>,
        // kernel defined fixed chunk size (kernel defines value blockDomSize)
        iotaFixedChunkSize<DeviceBuf>,
        // kernel defined fixed chunk size (kernel defines type BlockDomSizeND)
        iotaFixedChunkSizeND<DeviceBuf>);

    auto runTest = [&](auto&& function)
    {
        auto initialize = caravan::alpaka::fill(hostDeviceBuffer.getDeviceBuffer().getOwnedAlpakaView(), 0u);
        auto kernel = function(hostDeviceBuffer.getDeviceBuffer());
        auto copy = hostDeviceBuffer.deviceToHost();
        context.wait(context.spawn(
            caravan::alpaka::withDevice(
                device,
                std::move(initialize) | caravan::sequence(std::move(kernel)) | caravan::sequence(std::move(copy)))));
        validate(hostDeviceBuffer.getHostBuffer(), referenceBuffer);
    };

    // execute all tests
    std::apply([&](auto&... x) { (..., runTest(x)); }, testsFunctions);
}
