/* Copyright 2025-2026 Tapish Narwal
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

#include <cstddef>
#include <cstdint>
#include <new>

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif

namespace pmacc::spearhed::memory
{
    static constexpr int allocationMaxRetries = 13;

    // Raw allocation and retry logic using mallocMC
    [[nodiscard]] constexpr void* allocateRawMemory(
        auto const& worker,
        auto deviceHeapHandle,
        size_t size,
        std::align_val_t alignment)
    {
        for(int i = 0; i < allocationMaxRetries; ++i)
        {
            void* rawPtr = nullptr;
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
            rawPtr = deviceHeapHandle.malloc(worker.getAcc(), size);
#else
            // Use nothrow to ensure nullptr is returned on failure,
            // preventing exceptions from breaking the retry loop.
            rawPtr = operator new(size, alignment, std::nothrow);
#endif
            if(rawPtr != nullptr)
            {
                return rawPtr;
            }
        }
        return nullptr;
    }

    // Allocates memory unintialized
    template<typename T>
    [[nodiscard]] constexpr T* allocateMemory(auto const& worker, auto deviceHeapHandle)
    {
        void* mem = allocateRawMemory(worker, deviceHeapHandle, sizeof(T), std::align_val_t{alignof(T)});
        if(mem)
        {
            return static_cast<T*>(mem);
        }
        return nullptr;
    }

    /** Translate a device heap pointer to its host mirror address.
     *
     * After MallocMCBuffer::synchronize() copies the device heap to a pinned
     * host buffer, device pointers stored in frame metadata can be translated
     * to valid host pointers using the heap offset returned by
     * MallocMCBuffer::getOffset().
     *
     * On CPU serial backends heapOffset is 0 and this is an identity function.
     *
     * @param devPtr  Pointer value from device heap (may be nullptr).
     * @param heapOffset  deviceHeapBase - hostHeapBase in bytes.
     * @return Valid host pointer into the pinned heap copy, or nullptr.
     */
    template<typename T>
    T* mapToHost(T* devPtr, int64_t heapOffset) noexcept
    {
        if(!devPtr)
            return nullptr;
        return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(devPtr) - heapOffset);
    }

} // namespace pmacc::spearhed::memory
