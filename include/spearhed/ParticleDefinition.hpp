/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED, derived from PIConGPU.
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

#include "spearhed/param.hpp"
#include "spmacc/Frame.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegion.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

#include <pmacc/meta/Pair.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/particles/memory/dataTypes/StaticArray.hpp>

namespace spearhed
{
    /* extend particle description with pointer to a frame*/
    using FrameDescription = std::remove_const_t<decltype(particleDesc)>;

    /** frame definition
     *
     * a group of particles is stored as frame
     */
    using FrameType = pmacc::spearhed::Frame<FrameDescription>;

    /** Per-species frame type.  The species is embedded in the ParticleDescription carried by
     *  the frame, so FrameTypeFor<Species> produces a distinct type per species. */
    template<typename S>
    using FrameDescFor = std::remove_const_t<decltype(makeParticleDesc(S{}))>;

    template<typename S>
    using FrameTypeFor = pmacc::spearhed::Frame<FrameDescFor<S>>;

    /** Per-species ParticleRegion type.  Species propagates from ParticleDescription -> Frame ->
     *  ParticleRegion, so callers of ParticleRegionBuffer never name the species explicitly. */
    template<typename S>
    using PRTypeFor = pmacc::spearhed::
        ParticleRegion<pmacc::spearhed::AABB<CS>, FrameTypeFor<S>, typename DeviceHeap::AllocatorHandle>;

    /** The simulation's species vector: every species (pmacc::spearhed::species::AllTypes) paired with
     *  the per-species PRType template.  Predicate-driven buffer helpers (forEachSpeciesBufWithPred,
     *  withSpeciesBufsWithPred) compute the correct ParticleRegionBuffer type for each species. */
    using AllSpecies = pmacc::spearhed::SpeciesRegistry<PRTypeFor, pmacc::spearhed::species::AllTypes>;

    /** Concrete instance of the species registry, for the value-based role helpers. */
    inline constexpr AllSpecies allSpecies{};

    // Convenience alias for the most common PRType
    using PRType = PRTypeFor<pmacc::spearhed::species::Default>;

} // namespace spearhed
