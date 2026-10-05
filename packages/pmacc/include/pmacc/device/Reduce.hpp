/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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


#include "pmacc/device/reduce/Kernel.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/math/operation.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/traits/GetValueType.hpp"
#include "pmacc/types.hpp"

#include <memory>
#include <type_traits>

namespace pmacc
{
    namespace device
    {
        class Reduce
        {
        public:
            static constexpr uint32_t defaultSharedMmemSize = 4 * 1024;

            /** Constructor
             *
             * The memory required to hold the reduced result on the host and device will be allocated on the first
             * reduction call.
             *
             * @param byte how many bytes in global gpu memory can reserved for the reduce algorithm
             * @param sharedMemByte limit the usage of shared memory per block on gpu
             */
            HINLINE Reduce(uint32_t const byte, uint32_t const sharedMemByte = defaultSharedMmemSize)
                : byte(byte)
                , sharedMemByte(sharedMemByte)
            {
            }

            void tmpBufferAlloc()
            {
                // lazy allocation of the result buffer
                if(!reduceBuffer)
                    reduceBuffer = std::make_unique<GridBuffer<char, DIM1>>(DataSpace<DIM1>(byte));
            }

            /** Lazily reduce device values and copy the result to the host.
             *
             * The queue and source storage must remain alive through completion. Operations using the same reducer
             * must not overlap because they share its scratch buffer.
             */
            template<class Functor, typename Src>
            [[nodiscard]] HINLINE auto reduce(Functor func, Src src, uint32_t n)
            {
                /* - the result of a functor can be a reference or a const value
                 * - it is not allowed to create const or reference memory
                 *   thus we remove `references` and `const` qualifiers */
                using Type
                    = std::remove_const_t<std::remove_reference_t<typename traits::GetValueType<Src>::ValueType>>;

                tmpBufferAlloc();
                auto* destination = reinterpret_cast<Type*>(reduceBuffer->getDeviceBuffer().data());
                auto kernels = caravan::alpaka::submit(
                    [func = std::move(func),
                     src = std::move(src),
                     n,
                     destination,
                     scratchBytes = byte,
                     sharedBytes = sharedMemByte](auto& nativeQueue) mutable
                    { enqueueReduction(nativeQueue, func, src, n, destination, scratchBytes, sharedBytes); });
                auto copy = reduceBuffer->deviceToHost();
                auto host = reduceBuffer->getHostBuffer().getOwnedAlpakaView();
                return std::move(kernels) | caravan::sequence(std::move(copy))
                       | caravan::then([host = std::move(host)]
                                       { return *reinterpret_cast<Type const*>(::alpaka::onHost::data(host.value)); });
            }

        private:
            template<typename T_Queue, class Functor, typename Src, typename Type>
            HINLINE static void enqueueReduction(
                T_Queue& queue,
                Functor func,
                Src src,
                uint32_t n,
                Type* destination,
                uint32_t scratchBytes,
                uint32_t sharedBytes)
            {
                uint32_t blockcount = optimalThreadsPerBlock(n, sizeof(Type), sharedBytes);
                uint32_t const nBuffer = scratchBytes / sizeof(Type);
                uint32_t threads = nBuffer * blockcount * 2u;
                if(threads > n)
                    threads = n;

                uint32_t blocks = threads / 2u / blockcount;
                if(blocks == 0u)
                    blocks = 1u;
                enqueueReduceKernel<Type>(
                    queue,
                    blocks,
                    blockcount,
                    blockcount * sizeof(Type),
                    src,
                    n,
                    destination,
                    func,
                    pmacc::math::operation::Assign{});
                n = blocks;
                blockcount = optimalThreadsPerBlock(n, sizeof(Type), sharedBytes);
                blocks = n / 2u / blockcount;
                if(blocks == 0u && n > 1u)
                    blocks = 1u;

                while(blocks != 0u)
                {
                    if(blocks > 1u)
                    {
                        uint32_t const blockOffset = ceil(static_cast<double>(blocks) / blockcount);
                        uint32_t const useBlocks = blocks - blockOffset;
                        uint32_t const problemSize = n - blockOffset * blockcount;
                        Type* source = destination + blockOffset * blockcount;
                        enqueueReduceKernel<Type>(
                            queue,
                            useBlocks,
                            blockcount,
                            blockcount * sizeof(Type),
                            source,
                            problemSize,
                            destination,
                            func,
                            func);
                        blocks = blockOffset * blockcount;
                    }
                    else
                    {
                        enqueueReduceKernel<Type>(
                            queue,
                            blocks,
                            blockcount,
                            blockcount * sizeof(Type),
                            destination,
                            n,
                            destination,
                            func,
                            pmacc::math::operation::Assign{});
                    }

                    n = blocks;
                    blockcount = optimalThreadsPerBlock(n, sizeof(Type), sharedBytes);
                    blocks = n / 2u / blockcount;
                    if(blocks == 0u && n > 1u)
                        blocks = 1u;
                }
            }

            /** calculate number of threads per block
             *
             * @param threads maximal number of threads per block
             * @return number of threads per block
             */
            HINLINE static uint32_t getThreadsPerBlock(uint32_t threads)
            {
                /// \todo this list is not complete
                ///        extend it and maybe check for sm_version
                ///        and add possible threads accordingly.
                ///        maybe this function should be exported
                ///        to a more general nvidia class, too.
                if(threads >= 512)
                    return 512;
                if(threads >= 256)
                    return 256;
                if(threads >= 128)
                    return 128;
                if(threads >= 64)
                    return 64;
                if(threads >= 32)
                    return 32;
                if(threads >= 16)
                    return 16;
                if(threads >= 8)
                    return 8;
                if(threads >= 4)
                    return 4;
                if(threads >= 2)
                    return 2;

                return 1;
            }

            template<typename Type, typename T_Queue, typename... T_Args>
            HINLINE static void enqueueReduceKernel(
                T_Queue& queue,
                uint32_t blocks,
                uint32_t threads,
                uint32_t sharedMemSize,
                T_Args&&... args)
            {
                if(threads >= 512u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<512u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 256u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<256u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 128u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<128u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 64u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<64u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 32u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<32u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 16u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<16u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 8u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<8u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 4u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<4u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else if(threads >= 2u)
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<2u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
                else
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type>{})
                        .template configSMem<1u>(blocks, sharedMemSize)
                        .enqueueNative(queue, std::forward<T_Args>(args)...);
            }

            /** calculate optimal number of threads per block with respect to shared memory limitations
             *
             * @param n number of elements to reduce
             * @param sizePerElement size in bytes per elements
             * @return optimal count of threads per block to solve the problem
             */
            HINLINE static uint32_t optimalThreadsPerBlock(uint32_t n, uint32_t sizePerElement, uint32_t sharedMemByte)
            {
                uint32_t const sharedBorder = sharedMemByte / sizePerElement;
                return getThreadsPerBlock(std::min(sharedBorder, n));
            }

            /*buffer size limit in bytes on gpu*/
            uint32_t byte;
            /*shared memory limit in byte for one block*/
            uint32_t sharedMemByte = defaultSharedMmemSize;

            /*global gpu buffer for reduce steps*/
            std::unique_ptr<GridBuffer<char, DIM1>> reduceBuffer;
        };

    } // namespace device
} // namespace pmacc
