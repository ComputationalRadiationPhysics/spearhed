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

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <optional>

namespace pmacc::spearhed
{
    template<typename T_ParticleRegion>
    struct ParticleRegionBuffer : ISimulationData
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

        void synchronize() override
        {
            buffer->deviceToHost();
        };

        SimulationDataId getUniqueId() override
        {
            return "PRBuf";
        }

        std::optional<pmacc::HostDeviceBuffer<ParticleRegionType, DIM1>> buffer;
        int size = 0;
    };

} // namespace pmacc::spearhed
