#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <optional>

namespace pmacc::spearhed
{
    template<typename T_ParticleRegion>
    struct ParticleRegionBuffer
    {
        using ParticleRegionType = T_ParticleRegion;

        // replaces the old buffer with a new one with the given size
        // Does not communicate this to the GPU yet
        // Actually I want to have the ParticleRegion to be an SoA, and i want to resize the SoA and then hold a
        // HDBuffer to the current SoA
        auto create(int capacity)
        {
            buffer = pmacc::HostDeviceBuffer<ParticleRegionType, DIM1>(pmacc::DataSpace<DIM1>(capacity), false);
        }

        // Copies the particle region to the host buffer
        // Remember to send buf to device before use
        auto pushBack(ParticleRegionType const& pr)
        {
            buffer->getHostBuffer().getDataBox()[size++] = pr;
        }

        auto getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        std::optional<pmacc::HostDeviceBuffer<ParticleRegionType, DIM1>> buffer;
        int size = 0;
    };

} // namespace pmacc::spearhed
