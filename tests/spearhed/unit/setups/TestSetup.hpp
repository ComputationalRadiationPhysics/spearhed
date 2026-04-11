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

namespace spearhed
{

    template<uint32_t N>
    struct EmptyNRegions
    {
        // AABB constructor arguments are: {cell anchor/index}, {min corner}, {max corner}.
        pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        uint32_t baseNumParticlesToCreate = 400u;

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(baseNumParticlesToCreate);
        }

        // calculate how many particles we need to make in this system
        struct NumParticlesToCreate
        {
            constexpr auto operator()(
                [[maybe_unused]] auto& worker,
                [[maybe_unused]] auto& particleRegion,
                uint32_t baseNumParticlesToCreate) const
            {
                // Intentionally scale by (block index + 1) so each block creates a distinct
                // particle count, which makes per-block test validation deterministic.
                return baseNumParticlesToCreate * (worker.blockDomIdx() + 1);
            };
        };

        void setupRegions(pmacc::spearhed::ParticleRegionBuffer<PRType>& prBuf, DeviceHeap const& deviceHeap)
        {
            prBuf.create(N);
            PRType boundedParticles{deviceHeap.getAllocatorHandle()};
            for(size_t i = 0; i < N; ++i)
            {
                prBuf.pushBack(boundedParticles);
            }
            prBuf.buffer->hostToDevice();
        }
    };
} // namespace spearhed
