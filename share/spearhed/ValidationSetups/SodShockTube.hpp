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
    struct SodShockTube
    {
        pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        // Standard Sod shock tube initial conditions:
        // contact discontinuity at x=0, zero velocity everywhere
        struct InitialConditions
        {
            // Left state (x < 0)
            float densityLeft = 1.0f;
            // Right state (x > 0)
            float densityRight = 0.125f;
        };

        InitialConditions initialConditions;

        // Region volumes shared between setupRegions and NumParticlesToCreate
        pmacc::spearhed::AABB<CS> leftVolume{{0, 0, 0}, {-1.0, -1.0, -1.0}, {0.0, 1.0, 1.0}};
        pmacc::spearhed::AABB<CS> rightVolume{{0, 0, 0}, {0.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        float computeAABBVolume(pmacc::spearhed::AABB<CS> const& aabb)
        {
            float v = 1.0f;
            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { v *= aabb.max[tag] - aabb.min[tag]; });
            return v;
        }

        // Sum of rho_i * V_i across all regions which is used to distribute totalParticles
        float totalWeightedVolume = computeAABBVolume(leftVolume) * initialConditions.densityLeft
                                    + computeAABBVolume(rightVolume) * initialConditions.densityRight;

        // Total number of particles across all regions.
        // Each region receives a share proportional to rho * V, giving equal particle mass.
        uint32_t totalParticles = 32000u;

        struct NumParticlesToCreate
        {
            // Number of particles to create per particle region.
            // N_i = totalParticles * (rho_i * V_i) / sum_j(rho_j * V_j)
            constexpr auto operator()(
                auto& worker,
                auto& particleRegion,
                float densityLeft,
                float densityRight,
                uint32_t totalParticles,
                float totalWeightedVolume) const
            {
                // Region 0 = left, region 1 = right (insertion order in setupRegions)
                float const density = (worker.blockDomIdx() == 0) ? densityLeft : densityRight;

                auto const& vol = particleRegion.volume;
                float regionVolume = 1.0f;
                pmacc::spearhed::for_each_tag<CS>([&](auto tag) { regionVolume *= vol.max[tag] - vol.min[tag]; });

                return static_cast<uint32_t>(
                    static_cast<float>(totalParticles) * density * regionVolume / totalWeightedVolume);
            }
        };

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(
                initialConditions.densityLeft,
                initialConditions.densityRight,
                totalParticles,
                totalWeightedVolume);
        }

        HINLINE void setupRegions(pmacc::spearhed::ParticleRegionBuffer<PRType>& prBuf, DeviceHeap const& deviceHeap)
            const
        {
            prBuf.create(2);

            auto deviceHeapHandle = deviceHeap.getAllocatorHandle();

            auto leftRegion = PRType{deviceHeapHandle, leftVolume};
            auto rightRegion = PRType{deviceHeapHandle, rightVolume};

            // Add regions to the particleRegions buffer
            prBuf.pushBack(leftRegion);
            prBuf.pushBack(rightRegion);
            prBuf.buffer->hostToDevice();
        }
    };

    using Setup = SodShockTube;
} // namespace spearhed
