/* Copyright 2017-2024 Rene Widera
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

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/types.hpp"

#include <algorithm>

namespace pmacc
{
    namespace traits
    {
        namespace detail
        {
            /** Compile-time safe upper bound on workers for the build's executor.
             *
             * Delegates to alpaka::onHost::getMaxThreadsPerBlock, which returns the minimum guaranteed
             * threads-per-block for the build's API/device-kind/executor combination. The runtime device may
             * support more, but this is just a compile-time safety net for the ThreadSpec-based launch path.
             */
            inline constexpr uint32_t currentMaxWorkers
                = ::alpaka::onHost::getMaxThreadsPerBlock(computeApi, computeDeviceKind, computeExec);
        } // namespace detail

        /** Get number of workers
         *
         * the number of workers for a kernel depending on the used accelerator
         *
         * @tparam T_maxWorkers the maximum number of workers
         * @return @p ::value number of workers, clamped to the backend's compile-time maximum
         *
         * @warning This keys on the build's global pmacc::ComputeExec, so it is only correct while
         *          every launch uses that executor. This trait would ignore per-launch executors if/when they are
         *          possible in the future and keep deciding based on ComputeExec, so a launch with a different
         *          executor would be wrong. The fix is then to key the worker count on the *actual* executor used
         *          for the launch.
         */
        template<uint32_t T_maxWorkers>
        struct GetNumWorkers
        {
            static constexpr uint32_t value = std::min(T_maxWorkers, detail::currentMaxWorkers);
        };
    } // namespace traits
} // namespace pmacc
