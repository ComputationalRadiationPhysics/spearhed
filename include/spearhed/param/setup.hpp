/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
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

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <cstdint>
#include <tuple>

namespace spearhed
{
    // Default setup. Override at CMake configure time with -DSPEARHED_SETUP_FILE=/path/to/MySetup.hpp
    struct DefaultSetup
    {
        pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        uint32_t totalParticles = 1000u;

        struct NumParticlesToCreate
        {
            constexpr auto operator()(auto& /*worker*/, auto& /*particleRegion*/, uint32_t totalParticles) const
            {
                return totalParticles;
            }
        };

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(totalParticles);
        }

        void setupRegions(pmacc::spearhed::ParticleRegionBuffer<PRType>& prBuf, DeviceHeap const& deviceHeap) const
        {
            prBuf.create(1);

            auto deviceHeapHandle = deviceHeap.getAllocatorHandle();

            auto region = PRType{deviceHeapHandle, {{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}}};

            prBuf.pushBack(region);
            prBuf.buffer->hostToDevice();
        }
    };

    // Our simulation creates the setup instance as Setup{}. So custom setups need to be provide Setup, which must be
    // default constructible.
    using Setup = DefaultSetup;
} // namespace spearhed
