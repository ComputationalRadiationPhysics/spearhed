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

#include <pmacc/Environment.hpp>

namespace spearhed
{
    namespace init::detail
    {
    }

    struct InitRegions
    {
        void operator()(DeviceHeap const& deviceHeap)
        {
            PRType boundedParticles{deviceHeap.getAllocatorHandle()};

            auto& dc = pmacc::Environment<>::get().DataConnector();
            auto prBuf = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<PRType>>();
            dc.share(prBuf);

            prBuf->create(2);

            prBuf->pushBack(boundedParticles);
            // push back creates a copy
            prBuf->pushBack({deviceHeap.getAllocatorHandle(), pmacc::spearhed::AABB<CS>{}});

            prBuf->buffer->hostToDevice();
        }
    };
} // namespace spearhed
