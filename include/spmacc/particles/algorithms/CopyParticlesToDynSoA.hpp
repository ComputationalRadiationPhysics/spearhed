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

#include "spmacc/memory/utils.hpp"
#include "utility.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    /**
     * Default per-leaf copy trait for use with iterate_path.
     *
     * Receives the full source SoA (frame SoA), the destination DynSoA, particle count,
     * write offset, and any extra args (e.g. the ParticleRegion for transformations).
     * The @p Path template parameter is the full TagPath to the leaf in the output record.
     *
     * By default, bulk-copies the leaf at @p Path from srcSoa to dst.
     *
     * Specialise on a full TagPath to redirect the source leaf or apply a transformation.
     * For example, specialising on @c TagPath<position_t, x_t> allows reading from
     * @c relativePos.x and adding the region origin instead of copying from @c position.x.
     *
     * Signature: @c void(srcSoa, dst, count, offset, extraArgs...)
     */
    template<typename Path>
    struct FieldCopyTrait
    {
        using _default_sentinel = void;

        void operator()(auto const& srcSoa, auto& dst, uint32_t count, uint32_t offset, auto&&...) const
        {
            std::ranges::copy_n(
                srcSoa.template getLeaf<Path>().data(),
                count,
                dst.template getLeaf<Path>().data() + offset);
        }
    };

    /**
     * Copy all particle data from a ParticleRegionBuffer into a DynSoA.
     *
     * Calls prBuf.synchronize() internally. For GPU builds the caller must also
     * call mallocMCBuffer->synchronize() and pass the resulting heap offset;
     * on CPU serial backends heapOffset = 0 is correct.
     *
     * All frames are assumed fully packed except for one frame of each region, which may be partially filled (it isnt
     * necessarily the last frame since adding frames is done in parallel).
     * TODO Think if I should serialize adding frames so always the last frame is the incomplete one.
     *
     * Iterates over the leaf paths of the output record (T_DynSoA::record_type)
     * via ll::iterate_path, dispatching each leaf through FieldCopyTrait<LeafPath>.
     * The default FieldCopyTrait bulk-copies the same-path leaf from the source SoA.
     * Specialise FieldCopyTrait on a TagPath to redirect the source or transform the data.
     *
     * The per-region ParticleRegion is forwarded to every FieldCopyTrait invocation as
     * an extra arg after (srcSoa, dst, count, offset).
     *
     * @tparam Selector   ll::selectors policy controlling which leaf paths are visited.
     *                    Defaults to SelectAll.
     * @param prBuf       Particle region buffer (synchronized internally).
     * @param dynSoa      Output DynSoA; will be resized to totalParticles.
     * @param heapOffset  MallocMCBuffer::getOffset() on GPU, 0 on CPU serial.
     */
    struct CopyParticlesToDynSoA
    {
        template<typename T_DynSoA, typename Selector = ll::selectors::SelectAll>
        void operator()(auto& prBuf, T_DynSoA& dynSoa, int64_t heapOffset) const
        {
            prBuf.synchronize();

            using OutputRecord = typename T_DynSoA::record_type;

            auto hostBox = prBuf.buffer->getHostBuffer().getDataBox();

            uint32_t total = 0;
            for(int r = 0; r < prBuf.size; ++r)
                total += hostBox[r].particleFrameList.getNumParticles();
            dynSoa.resize(total);

            using FrameListType = decltype(hostBox[0].particleFrameList);
            using FrameType = typename std::remove_reference_t<FrameListType>::FrameType;
            using RegionType = std::remove_reference_t<decltype(hostBox[0])>;
            constexpr uint32_t frameSize = FrameType::frameSize;

            struct CopyTask
            {
                FrameType* hostFrame;
                uint32_t count;
                uint32_t writeOffset;
                RegionType const* region;
            };

            std::vector<CopyTask> tasks;
            tasks.reserve(total / frameSize + prBuf.size);

            uint32_t writeOffset = 0;
            for(int r = 0; r < prBuf.size; ++r)
            {
                auto& fl = hostBox[r].particleFrameList;
                uint32_t const nFrames = fl.numFrames();
                if(nFrames == 0)
                    continue;

                auto* devFramePtr = &(*fl.begin());

                for(uint32_t f = 0; f < nFrames; ++f)
                {
                    auto* hostFrame = memory::mapToHost(devFramePtr, heapOffset);
                    tasks.push_back({hostFrame, hostFrame->liveParticles, writeOffset, &hostBox[r]});

                    writeOffset += hostFrame->liveParticles;
                    devFramePtr = hostFrame->next;
                }
            }

            // if we require TBB can be done with par_unseq
            std::for_each(
                tasks.begin(),
                tasks.end(),
                [&](CopyTask const& task)
                {
                    ll::iterate_path<OutputRecord, Selector, FieldCopyTrait>(
                        task.hostFrame->particlesSoa,
                        dynSoa,
                        task.count,
                        task.writeOffset,
                        *task.region);
                });
        }
    };

} // namespace pmacc::spearhed
