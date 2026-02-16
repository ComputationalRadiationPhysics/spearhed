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
