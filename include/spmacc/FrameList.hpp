#pragma once

#include "spmacc/Frame.hpp"
#include "spmacc/SinglyLinkedListDevice.hpp"
#include "spmacc/memory/FramePointer.hpp"

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
        struct Iterator
        {
            using value_type = FrameType;
            using pointer = FrameType*;
            using reference = FrameType&;

            constexpr Iterator(pointer node = nullptr) : m_current(node)
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

            constexpr Iterator& operator++()
            {
                if(m_current)
                    m_current = m_current->next;
                return *this;
            }

            constexpr bool operator==(Iterator const& other) const = default;

        private:
            pointer m_current;
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

        HDINLINE constexpr Iterator begin() const
        {
            return Iterator{list.begin()};
        }

        HDINLINE constexpr Iterator end() const
        {
            return Iterator{list.end()};
        }

        HDINLINE constexpr uint32_t getNumParticles() const
        {
            return numParticles;
        }

        HDINLINE constexpr void setNumParticles(uint32_t n)
        {
            numParticles = n;
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


} // namespace pmacc::spearhed
