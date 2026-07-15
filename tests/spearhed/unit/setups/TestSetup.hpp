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
#include <vector>

namespace spearhed
{

    // calculate how many particles we need to make in this system
    struct ScaledNumParticlesToCreate
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

    // place each particle at the center of its particle region's AABB
    struct CenterPlaceParticle
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            [[maybe_unused]] uint32_t globalParticleIdx) const
        {
            auto const& aabb = particleRegion.volume;
            pmacc::spearhed::for_each_tag<CS>(
                [&](auto tag) { *particle[relativePos][tag] = (aabb.min[tag] + aabb.max[tag]) * 0.5f; });
        }
    };

    template<uint32_t N>
    struct EmptyNRegions
    {
        // This setup fills a single species and acts as its own (only) init block.
        using Species = pmacc::spearhed::species::Default;
        using NumParticlesToCreate = ScaledNumParticlesToCreate;
        using PlaceParticle = CenterPlaceParticle;

        // AABB constructor arguments are: {cell anchor/index}, {min corner}, {max corner}.
        pmacc::spearhed::AABB<CS> domain{{0, 0, 0}, {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};

        uint32_t baseNumParticlesToCreate = 400u;

        auto blocks() const
        {
            return std::tie(*this);
        }

        auto numParticlesToCreateArgs() const
        {
            return std::make_tuple(baseNumParticlesToCreate);
        }

        auto placeParticleArgs() const
        {
            return std::make_tuple();
        }

        template<typename>
        void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
        {
            for(size_t i = 0; i < N; ++i)
                out.push_back(pmacc::spearhed::AABB<CS>{});
        }
    };
} // namespace spearhed
