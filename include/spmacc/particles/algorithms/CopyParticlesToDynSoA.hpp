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

#include "llamaLite/llamaLite.hpp"
#include "spmacc/memory/utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<typename TagPath, typename SrcSoA, typename DstDynSoA>
        void copyOneField(SrcSoA const& src, DstDynSoA& dst, uint32_t count, uint32_t dstOffset)
        {
            auto srcSpan = src.template getLeaf<TagPath>();
            auto dstSpan = dst.template getLeaf<TagPath>();
            std::copy_n(srcSpan.data(), count, dstSpan.data() + dstOffset);
        }

        template<typename Record, typename SrcSoA, typename DstDynSoA>
        void copyAllFields(SrcSoA const& src, DstDynSoA& dst, uint32_t count, uint32_t dstOffset)
        {
            using LeafPaths = typename ll::GetLeafPaths<Record>::type;
            [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                (copyOneField<std::tuple_element_t<Is, LeafPaths>>(src, dst, count, dstOffset), ...);
            }(std::make_index_sequence<std::tuple_size_v<LeafPaths>>{});
        }
    } // namespace detail

    /**
     * Copy all particle data from a ParticleRegionBuffer into a DynSoA.
     *
     * Prerequisite: the caller must have already called
     *   prBuf.synchronize()           -- copies ParticleRegion metadata to host
     *   mallocMCBuffer->synchronize() -- bulk-copies device heap to host
     *
     * All frames are assumed to be fully packed except the last frame of each
     * region, which may be partially filled (no holes within frames).
     *
     * @param prBuf       Particle region buffer (already synchronized to host).
     * @param dynSoa      Output DynSoA; will be resized to totalParticles.
     * @param heapOffset  MallocMCBuffer::getOffset() on GPU, 0 on CPU serial.
     */
    struct CopyParticlesToDynSoA
    {
        template<typename T_PRBuf, typename T_DynSoA>
        void operator()(T_PRBuf& prBuf, T_DynSoA& dynSoa, int64_t heapOffset) const
        {
            using ParticleRecord = typename T_DynSoA::record_type;

            auto hostBox = prBuf.buffer->getHostBuffer().getDataBox();

            uint32_t total = 0;
            for(int r = 0; r < prBuf.size; ++r)
                total += hostBox[r].particleFrameList.getNumParticles();
            dynSoa.resize(total);

            uint32_t writeOffset = 0;
            for(int r = 0; r < prBuf.size; ++r)
            {
                auto& fl = hostBox[r].particleFrameList;
                uint32_t const nFrames = fl.numFrames();
                if(nFrames == 0)
                    continue;

                using FrameType = typename std::remove_reference_t<decltype(fl)>::FrameType;
                constexpr uint32_t frameSize = FrameType::frameSize;
                uint32_t const lastCount = fl.getSizeLastFrame();

                // fl.begin().operator->() returns the raw FrameType* device pointer
                // stored in the FrameList without dereferencing device memory.
                auto* devFramePtr = fl.begin().operator->();
                for(uint32_t f = 0; f < nFrames; ++f)
                {
                    auto* hostFrame = memory::mapToHost(devFramePtr, heapOffset);
                    uint32_t const count = (f == nFrames - 1u) ? lastCount : frameSize;

                    detail::copyAllFields<ParticleRecord>(hostFrame->particlesSoa, dynSoa, count, writeOffset);

                    writeOffset += count;
                    // hostFrame->next is a device pointer stored in host-accessible
                    // memory; translate it in the next iteration.
                    devFramePtr = hostFrame->next;
                }
            }
        }
    };

} // namespace pmacc::spearhed
