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
#include "pmacc/exec/KernelWithDynSharedMem.hpp"
#include "pmacc/types.hpp"

namespace pmacc::exec
{
    namespace detail
    {
        template<typename T, typename T_Sfinae = void>
        struct GetDim
        {
            static constexpr uint32_t dim = T::dim;
        };

        template<typename T>
        struct GetDim<T, std::enable_if_t<std::is_integral_v<T>>>
        {
            static constexpr uint32_t dim = 1;
        };

        /** Wrap a user kernel functor and apply launch extents and dynamic shared memory. */
        template<typename T_KernelFunctor>
        struct KernelPreperationWrapper
        {
            T_KernelFunctor const m_kernelFunctor;
#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
            char const* const m_file;
            size_t const m_line;
#else
            static constexpr char const* m_file = "";
            static constexpr size_t m_line = 0u;
#endif

            HINLINE KernelPreperationWrapper(
                T_KernelFunctor const& kernelFunctor,
                char const* const file = "",
                size_t const line = 0u)
                : m_kernelFunctor(kernelFunctor)
#if defined(PMACC_SYNC_KERNEL) && PMACC_SYNC_KERNEL == 1
                , m_file(file)
                , m_line(line)
#endif
            {
#if !defined(PMACC_SYNC_KERNEL) || PMACC_SYNC_KERNEL != 1
                static_cast<void>(file);
                static_cast<void>(line);
#endif
            }

            /** Apply grid and block extents and optionally dynamic shared memory to the wrapped functor.
             *
             * @tparam T_VectorGrid type which defines the grid extents
             * @tparam T_VectorBlock type which defines the block extents
             *
             * @param gridExtent grid extent configuration for the kernel
             * @param blockExtent block extent configuration for the kernel
             *
             * @return object with user kernel functor and launch parameters
             *
             * @{
             */
            template<typename T_VectorGrid, typename T_VectorBlock>
            HINLINE auto operator()(T_VectorGrid const& gridExtent, T_VectorBlock const& blockExtent) const
                -> KernelLauncher<T_KernelFunctor, GetDim<T_VectorGrid>::dim>;

            /**
             * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
             */
            template<typename T_VectorGrid, typename T_VectorBlock>
            HINLINE auto operator()(
                T_VectorGrid const& gridExtent,
                T_VectorBlock const& blockExtent,
                size_t const sharedMemByte) const
                -> KernelLauncher<KernelWithDynSharedMem<T_KernelFunctor>, GetDim<T_VectorGrid>::dim>;
            /**@}*/
        };
    } // namespace detail

    /** Creates a kernel object.
     *
     * example for lambda usage:
     *
     * @code{.cpp}
     *   PMACC_KERNEL([]ALPAKA_FN_ACC(auto const& acc) -> void{
     *       printf("Hello World.\n");
     *   })(1,1)(queue)
     * @endcode
     *
     * @tparam T_KernelFunctor type of the kernel functor
     * @param kernelFunctor instance of the functor, lambda are supported
     * @param file kernel call-site file used by blocking-kernel diagnostics
     * @param line kernel call-site line used by blocking-kernel diagnostics
     */
    template<typename T_KernelFunctor>
    auto kernel(T_KernelFunctor const& kernelFunctor, char const* const file = "", size_t const line = 0u)
        -> detail::KernelPreperationWrapper<T_KernelFunctor>
    {
        return detail::KernelPreperationWrapper<T_KernelFunctor>(kernelFunctor, file, line);
    }
} // namespace pmacc::exec

/** Create a kernel object out of a functor instance and retain its call site for blocking-kernel diagnostics.
 *
 * This macro add the current filename and line number to the kernel object.
 * @see ::pmacc::exec::kernel
 *
 * @param ... instance of kernel functor
 */
#define PMACC_KERNEL(...) ::pmacc::exec::kernel(__VA_ARGS__, __FILE__, static_cast<size_t>(__LINE__))


#include "pmacc/exec/Kernel.tpp"
#include "pmacc/exec/KernelLauncher.tpp"
