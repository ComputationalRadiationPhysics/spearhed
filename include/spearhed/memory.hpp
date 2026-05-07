/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED, derived from PIConGPU.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>

#include <cstdint>

namespace spearhed
{
    /** Sync the device heap to its host-mapped buffer and return the pointer offset.
     *
     * After all kernel work is complete, call this to make device-heap-allocated
     * frame data readable on the host. Pass the returned offset to
     * FrameList::hostIterable() or CopyParticlesToDynSoA.
     *
     * On CPU serial backends MallocMCBuffer::synchronize() is a no-op and the
     * returned offset is 0.
     */
    inline int64_t syncHeapToHost()
    {
        auto& dc = pmacc::Environment<simDim>::get().DataConnector();
        auto buf = dc.get<pmacc::MallocMCBuffer<DeviceHeap>>(pmacc::MallocMCBuffer<DeviceHeap>::getName());
        buf->synchronize();
        return buf->getOffset();
    }

} // namespace spearhed
