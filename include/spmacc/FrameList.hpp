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

#include "spmacc/Frame.hpp"
#include "spmacc/SinglyLinkedListDevice.hpp"
#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/memory/utils.hpp"

#include <pmacc/traits/IsSpecializationOf.hpp>

#include <cstdint>

namespace pmacc::spearhed
{
    /**
     * Copy of a frame list still points to the same frame list
     */
    template<concepts::SpecializationOf<Frame> T_Frame, typename T_DeviceHeapHandle>
    struct FrameList
    {
        using FrameType = T_Frame;

    private:
        struct DeviceAdvance
        {
            constexpr FrameType* init(FrameType* ptr) const
            {
                return ptr;
            }

            constexpr FrameType* advance(FrameType* ptr) const
            {
                return ptr->next;
            }

            constexpr bool operator==(DeviceAdvance const&) const = default;
        };

        //! Advance policy that translates device heap pointers to host addresses via heapOffset.
        struct HostAdvance
        {
            int64_t heapOffset;

            constexpr FrameType* init(FrameType* ptr) const
            {
                return memory::mapToHost(ptr, heapOffset);
            }

            constexpr FrameType* advance(FrameType* ptr) const
            {
                return memory::mapToHost(ptr->next, heapOffset);
            }

            constexpr bool operator==(HostAdvance const&) const = default;
        };

        template<typename AdvancePolicy>
        struct IteratorImpl
        {
            using value_type = FrameType;
            using pointer = FrameType*;
            using reference = FrameType&;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;

            constexpr IteratorImpl(pointer node, AdvancePolicy policy = {})
                : m_current(policy.init(node))
                , m_policy(policy)
            {
            }

            constexpr reference operator*() const
            {
                return *m_current;
            }

            constexpr pointer operator->() const
            {
                return m_current;
            }

            constexpr IteratorImpl& operator++()
            {
                // caller needs to ensure validity of this call
                m_current = m_policy.advance(m_current);
                return *this;
            }

            constexpr IteratorImpl operator++(int)
            {
                IteratorImpl tmp = *this;
                ++(*this);
                return tmp;
            }

            constexpr bool operator==(IteratorImpl const& other) const = default;

        private:
            pointer m_current;
            [[no_unique_address]] AdvancePolicy m_policy;
        };

        using Iterator = IteratorImpl<DeviceAdvance>;
        using HostIterator = IteratorImpl<HostAdvance>;

        struct HostRange
        {
            HostIterator m_begin;
            HostIterator m_end;

            constexpr HostIterator begin() const
            {
                return m_begin;
            }

            constexpr HostIterator end() const
            {
                return m_end;
            }
        };

    public:
        HDINLINE constexpr FrameList(T_DeviceHeapHandle const& deviceHeapHandle) : list{deviceHeapHandle}
        {
        }

        //! get number of particle in the last frame
        HDINLINE constexpr uint32_t getSizeLastFrame() const
        {
            constexpr uint32_t frameSize = T_Frame::frameSize;

            /* NOTE on result expression understanding:
             * (numParticles % frameSize) =^= how many particle did not fit in a full frame?
             *
             * but we need how many are in the last frame,
             * => (numParticles - 1u) % frameSize + 1u
             *   only shift by one which is reversed by + 1u
             * => will return the same result for numParticles =/= i * frameSize ;i \in N
             * and for numParticles == i * frameSize, i \in N it will return
             *  ((frameSize * i) - 1u) % frameSize + 1u = (frameSize - 1u) + 1u = frameSize
             */
            // avoids underflow for uint32_t numParticles = 0u
            return numParticles ? ((numParticles - 1u) % frameSize + 1u) : 0u;
        }

        HDINLINE constexpr pmacc::spearhed::memory::FramePointer<FrameType> getEmptyFrame(auto const& worker)
        {
            auto framePtr = list.getEmptyNode(worker);
            list.pushBack(worker, framePtr);
            return framePtr;
        }

        /**
         * @brief Detach the complete frame chain before a bulk rebuild.
         *
         * The caller must retain frame addresses independently and release each
         * detached frame after all reads complete.  This is the relocation
         * primitive: the active list can immediately receive a freshly packed
         * chain while the old frames remain valid scatter sources.
         */
        HDINLINE constexpr FrameType* detachAllFrames()
        {
            numParticles = 0u;
            return list.detachAllNodes();
        }

        /** Release one frame previously detached with detachAllFrames(). */
        HDINLINE constexpr void destroyDetachedFrame(auto const& worker, FrameType* frame)
        {
            list.removeNode(worker, frame);
        }

        /** Free every active frame and restore an empty, valid list. */
        HDINLINE constexpr void destroyAllFrames(auto const& worker)
        {
            FrameType* frame = detachAllFrames();
            while(frame != nullptr)
            {
                FrameType* const next = frame->next;
                destroyDetachedFrame(worker, frame);
                frame = next;
            }
        }

        HDINLINE constexpr Iterator begin() const
        {
            return Iterator{list.begin()};
        }

        HDINLINE constexpr Iterator end() const
        {
            return Iterator{list.end()};
        }

        //! Returns a host-side iterable range.
        //! @param heapOffset MallocMCBuffer::getOffset() on GPU, 0 on CPU serial backends.
        constexpr HostRange hostIterable(int64_t heapOffset) const
        {
            HostAdvance policy{heapOffset};
            return {HostIterator{list.begin(), policy}, HostIterator{nullptr, policy}};
        }

        HDINLINE constexpr uint32_t getNumParticles() const
        {
            return numParticles;
        }

        // Num particles must be set before we can call this
        HDINLINE constexpr uint32_t numFrames() const
        {
            return alpaka::core::divCeil(numParticles, T_Frame::frameSize);
        }

        HDINLINE constexpr void setNumParticles(uint32_t n)
        {
            numParticles = n;
        }

        HDINLINE constexpr auto size() const
        {
            return list.size();
        }

        /**
         * @brief Verify the packed-frame invariant with a caller-supplied liveness predicate.
         *
         * The predicate receives a slot view and returns whether the slot is live.
         * This is useful in tests and debug maintenance checks without making the
         * generic frame list depend on a particular particle-record tag.
         */
        template<typename IsLive>
        HDINLINE constexpr bool isPacked(IsLive isLive) const
        {
            uint32_t countedParticles = 0u;
            uint32_t countedFrames = 0u;
            for(auto const& frame : *this)
            {
                bool seenGap = false;
                uint32_t liveInFrame = 0u;
                for(uint32_t slot = 0u; slot < FrameType::frameSize; ++slot)
                {
                    bool const live = isLive(frame[slot]);
                    if(!live)
                        seenGap = true;
                    else
                    {
                        if(seenGap)
                            return false;
                        ++liveInFrame;
                    }
                }
                if(frame.liveParticles != liveInFrame)
                    return false;
                if(frame.next != nullptr && liveInFrame != FrameType::frameSize)
                    return false;
                countedParticles += liveInFrame;
                ++countedFrames;
            }
            return countedParticles == numParticles && countedFrames == numFrames();
        }

    private:
        pmacc::spearhed::SingleLinkedListDevice<T_Frame, T_DeviceHeapHandle> list;
        PMACC_ALIGN(numParticles, uint32_t) { 0 };
    };

    template<concepts::SpecializationOf<FrameList> T_FrameList, typename F>
    HDINLINE constexpr void forEachFrame(T_FrameList list, F&& func)
    {
        for(auto& frame : list)
        {
            func(&frame);
        }
    }

    template<concepts::SpecializationOf<FrameList> T_FrameList, typename F>
    void forEachHostFrame(T_FrameList const& frameList, int64_t heapOffset, F&& func)
    {
        for(auto& frame : frameList.hostIterable(heapOffset))
        {
            func(&frame);
        }
    }


} // namespace pmacc::spearhed
