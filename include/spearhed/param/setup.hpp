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
#include "spearhed/sph/KernelVariant.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/topology/Cartesian.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cstdint>
#include <tuple>

namespace spearhed
{
    // Default setup. Override at CMake configure time with -DSPEARHED_SETUP_FILE=/path/to/MySetup.hpp
    struct DefaultSetup
    {
        using Roles = std::tuple<pmacc::spearhed::roles::Interior>;

        pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        uint32_t totalParticles = 1000u;

        template<typename Role>
        struct NumParticlesToCreate
        {
            constexpr auto operator()(auto& /*worker*/, auto& /*particleRegion*/, uint32_t totalParticles) const
            {
                return totalParticles;
            }
        };

        template<typename Role>
        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(totalParticles);
        }

        template<typename Role>
        struct PlaceParticle
        {
            DINLINE constexpr void operator()(
                [[maybe_unused]] auto const& worker,
                auto& particle,
                auto const& particleRegion,
                [[maybe_unused]] uint32_t globalParticleIdx) const
            {
                auto const& aabb = particleRegion.volume;
                pmacc::spearhed::for_each_tag<CS>(
                    [&](auto tag)
                    { *particle[relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * CS::T_Axis{0.5}; });
            }
        };

        template<typename Role>
        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        KernelVariant kernelVariant = makeKernel(KernelType::CubicSpline);

        template<typename Role>
        void setupRegions(pmacc::spearhed::ParticleRegionBuffer<PRType, Role>& prBuf, DeviceHeap const& deviceHeap)
            const
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
