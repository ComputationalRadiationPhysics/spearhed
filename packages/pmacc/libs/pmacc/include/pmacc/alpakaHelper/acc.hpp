/* Copyright 2024-2024 Rene Widera
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

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc
{
    using IdxType = uint32_t;
    using MemIdxType = size_t;

    /* Compute backend
     *
     * A backend bundles the alpaka API and device kind (which device the code runs on) with the executor
     * (how the parallelism on that device is organised). Exactly one backend is active per build; it is selected
     * via the CMake option PMACC_BACKEND, which defines one of the PMACC_BACKEND_* macros below.
     *
     * - ComputeDevice: alpaka device type the code is built for
     * - computeExec / ComputeExec: the executor (value and type) describing the parallelism on ComputeDevice
     */
#if defined(PMACC_BACKEND_GpuCuda)
    inline constexpr auto computeApi = ::alpaka::api::cuda;
    inline constexpr auto computeDeviceKind = ::alpaka::deviceKind::nvidiaGpu;
    inline constexpr auto computeExec = ::alpaka::exec::gpuCuda;
#elif defined(PMACC_BACKEND_GpuHip)
    inline constexpr auto computeApi = ::alpaka::api::hip;
    inline constexpr auto computeDeviceKind = ::alpaka::deviceKind::amdGpu;
    inline constexpr auto computeExec = ::alpaka::exec::gpuHip;
#elif defined(PMACC_BACKEND_OneApi)
    inline constexpr auto computeApi = ::alpaka::api::oneApi;
    inline constexpr auto computeDeviceKind = ::alpaka::deviceKind::intelGpu;
    inline constexpr auto computeExec = ::alpaka::exec::oneApi;
#elif defined(PMACC_BACKEND_CpuOmpBlocks)
    inline constexpr auto computeApi = ::alpaka::api::host;
    inline constexpr auto computeDeviceKind = ::alpaka::deviceKind::cpu;
    inline constexpr auto computeExec = ::alpaka::exec::cpuOmpBlocks;
#elif defined(PMACC_BACKEND_CpuTbbBlocks)
    inline constexpr auto computeApi = ::alpaka::api::host;
    inline constexpr auto computeDeviceKind = ::alpaka::deviceKind::cpu;
    inline constexpr auto computeExec = ::alpaka::exec::cpuTbbBlocks;
#elif defined(PMACC_BACKEND_CpuSerial)
    inline constexpr auto computeApi = ::alpaka::api::host;
    inline constexpr auto computeDeviceKind = ::alpaka::deviceKind::cpu;
    inline constexpr auto computeExec = ::alpaka::exec::cpuSerial;
#else
#    error                                                                                                            \
        "No PMacc compute backend selected. Set the CMake option PMACC_BACKEND (CpuSerial, CpuOmpBlocks, CpuTbbBlocks, GpuCuda, GpuHip or OneApi)."
#endif

    using ComputeDevice = ::alpaka::onHost::Device<ALPAKA_TYPEOF(computeApi), ALPAKA_TYPEOF(computeDeviceKind)>;

    //! type of the selected backend's executor
    using ComputeExec = ALPAKA_TYPEOF(computeExec);

    using HostDevice = ::alpaka::onHost::Device<::alpaka::api::Host, ::alpaka::deviceKind::Cpu>;

#if (PMACC_USE_ASYNC_QUEUES == 1)
    using ComputeDeviceQueue = ::alpaka::onHost::Queue<ComputeDevice, ::alpaka::queueKind::NonBlocking>;
#else
    using ComputeDeviceQueue = ::alpaka::onHost::Queue<ComputeDevice, ::alpaka::queueKind::Blocking>;
#endif

    using ComputeDeviceEvent = ::alpaka::onHost::Event<ComputeDevice>;

    /*! device compile flag
     *
     * Enabled if the compiler processes currently a separate compile path for the device code
     *
     * @attention value is always 0 for alpaka CPU accelerators
     *
     * Value is 1 if device path is compiled else 0
     */
#if ALPAKA_LANG_CUDA && (ALPAKA_COMP_CLANG_CUDA || ALPAKA_COMP_NVCC) && __CUDA_ARCH__
#    define PMACC_DEVICE_COMPILE 1
#elif ALPAKA_LANG_HIP && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
#    define PMACC_DEVICE_COMPILE 1
#else
#    define PMACC_DEVICE_COMPILE 0
#endif

} // namespace pmacc
