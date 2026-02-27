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
#include <new>

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif

namespace pmacc::spearhed::memory
{
    static constexpr int allocationMaxRetries = 13;

    // Raw allocation and retry logic using mallocMC
    [[nodiscard]] constexpr void* allocateRawMemory(auto const& worker, auto const& deviceHeapHandle, size_t size)
    {
        for(int i = 0; i < allocationMaxRetries; ++i)
        {
            void* rawPtr = nullptr;
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
            // Explicit cast required for C++
            rawPtr = deviceHeapHandle.malloc(worker.getAcc(), size);
#else
            // Use nothrow to ensure nullptr is returned on failure,
            // preventing exceptions from breaking the retry loop.
            rawPtr = operator new(size, std::nothrow);
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
    [[nodiscard]] constexpr T* allocateMemory(auto const& worker, auto const& deviceHeapHandle)
    {
        void* mem = allocateRawMemory(worker, deviceHeapHandle, sizeof(T));
        if(mem)
        {
            return static_cast<T*>(mem);
        }
        return nullptr;
    }

} // namespace pmacc::spearhed::memory
