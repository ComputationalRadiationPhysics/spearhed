#pragma once

#include "spmacc/FrameList.hpp"

#include <pmacc/traits/IsSpecializationOf.hpp>

namespace pmacc
{
    namespace spearhed
    {
        /**
         * Holds a defined volume and the paricles in that volume
         * Maybe this should be held in an SoA, to do quick ops on the Volume/Frame
         * is copied in particleRegionBuffer push back, should be fast to copy.
         */
        template<typename TVolume, concepts::SpecializationOf<Frame> T_Frame, typename T_DeviceHeapHandle>
        struct ParticleRegion
        {
            constexpr ParticleRegion(T_DeviceHeapHandle const& deviceHeapHandle) : particleFrameList{deviceHeapHandle}
            {
            }

            // return a reference to the Frame list
            auto& getParticleFrameList()
            {
                return particleFrameList;
            }

            TVolume volume;
            FrameList<T_Frame, T_DeviceHeapHandle> particleFrameList;
        };
    } // namespace spearhed
} // namespace pmacc
