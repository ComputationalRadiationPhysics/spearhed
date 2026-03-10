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
#include "spearhed/particles/pusher/PushVelocity.hpp"
#include "spmacc/ParticleRegionBuffer.hpp"
#include "spmacc/particles/algorithms/ForEachParticle.hpp"

namespace spearhed
{
    struct ParticlePush
    {
        void operator()(uint32_t currentStep) const
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();
            auto& prBuf = *dc.get<pmacc::spearhed::ParticleRegionBuffer<PRType>>("PRBuf");

            pmacc::spearhed::ForEachParticleInPRBuf{}(prBuf, PushVelocity{}, dt);
            // forEachParticleInPR();
            // Push particles
            // PushDistance{}();
            // Update bounding boxes
        }
    };

} // namespace spearhed
