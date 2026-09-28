/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Benjamin Worpitz
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


#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/types.hpp"

#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
#    include "pmacc/alpakaHelper/ValidateCall.hpp"

#    include <string>
#    include <typeinfo>
#endif

#include <caravan/alpaka.hpp>

namespace pmacc::exec::detail
{
    /** Detect an alpaka 3 queue handle.
     *
     * alpaka 3 does not provide a public queue concept, therefore the public queue interface is checked directly.
     * This is used to distinguish the borrowed-queue launcher from the queue-free launcher.
     */
    template<typename T>
    concept QueueHandle = requires(T const& queue) {
        { queue.getQueueKind() } -> ::alpaka::concepts::QueueKind;
    };

    template<typename T_Kernel, uint32_t T_dim>
    struct KernelLauncher
    {
        //! kernel functor
        T_Kernel const m_kernel;
#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
        char const* const m_file;
        size_t const m_line;
#endif
        //! grid extents for the kernel
        math::Vector<IdxType, T_dim> const m_gridExtent;
        //! block extents for the kernel
        math::Vector<IdxType, T_dim> const m_blockExtent;

        /** kernel starter object
         *
         * @param kernel pmacc Kernel
         */
        template<typename T_VectorGrid, typename T_VectorBlock>
        HINLINE KernelLauncher(
            T_Kernel const& kernel,
            char const* const file,
            size_t const line,
            T_VectorGrid const& gridExtent,
            T_VectorBlock const& blockExtent)
            : m_kernel(kernel)
#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
            , m_file(file)
            , m_line(line)
#endif
            , m_gridExtent(gridExtent)
            , m_blockExtent(blockExtent)
        {
#if !defined(PMACC_SYNC_KERNEL) || PMACC_SYNC_KERNEL != 1
            static_cast<void>(file);
            static_cast<void>(line);
#endif
        }

        /** Enqueue this kernel from a submission state that retains all arguments through completion. */
        template<typename T_Queue, typename... T_Args>
        HINLINE void enqueueNative(T_Queue& queue, T_Args&&... args) const
        {
#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
            std::string const kernelInfo
                = std::string(typeid(m_kernel).name()) + " [" + m_file + ":" + std::to_string(m_line) + " ]";
            PMACC_CHECK_ALPAKA_CALL_MSG(::alpaka::onHost::wait(queue), "Crash before kernel call " + kernelInfo);
#endif
            auto const gridExtent = m_gridExtent.toAlpakaKernelVec();
            auto const blockExtent = m_blockExtent.toAlpakaKernelVec();
            auto const threadSpec = ::alpaka::onHost::ThreadSpec{gridExtent, blockExtent};
            auto const kernelBundle = ::alpaka::KernelBundle{m_kernel, caravan::unwrap(args)...};
#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
            PMACC_CHECK_ALPAKA_CALL_MSG(
                queue.enqueue(threadSpec, kernelBundle),
                "Crash during kernel launch " + kernelInfo);
            PMACC_CHECK_ALPAKA_CALL_MSG(::alpaka::onHost::wait(queue), "Crash after kernel call " + kernelInfo);
#else
            queue.enqueue(threadSpec, kernelBundle);
#endif
        }

        /** Lazily describe this kernel on an explicitly borrowed queue. */
        template<typename T_Queue, typename... T_Args>
        requires(QueueHandle<T_Queue>)
        [[nodiscard]] HINLINE auto operator()(T_Queue& queue, T_Args... args) const
        {
            return caravan::alpaka::submit(
                queue,
                [launcher = *this, args = std::tuple<T_Args...>{std::move(args)...}](T_Queue& nativeQueue) mutable
                { std::apply([&](auto&... values) { launcher.enqueueNative(nativeQueue, values...); }, args); });
        }

        /** Lazily describe this kernel; the receiver environment supplies its queue. */
        template<typename... T_Args>
        [[nodiscard]] HINLINE auto operator()(T_Args... args) const
        {
            return caravan::alpaka::submit(
                [launcher = *this, args = std::tuple<T_Args...>{std::move(args)...}](auto& nativeQueue) mutable
                { std::apply([&](auto&... values) { launcher.enqueueNative(nativeQueue, values...); }, args); });
        }
    };

} // namespace pmacc::exec::detail
