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
#include "spmacc/particles/View.hpp"

#include <llamaLite/llamaLite.hpp>

namespace spearhed
{
    /** Per-species particle view: constrains a particle operand to the SoA layout derived
     *  from @p S (a species tag) and a selected set of attribute tags.
     *
     *  The species flows through FrameDescFor<S> to pick up the correct ParticleRecord
     *  and frame size, so different species with different attribute sets produce distinct
     *  View types.
     *
     *  Usage in a kernel functor that is specialised per species:
     *  @code
     *      template<typename Species>
     *      struct MyFunctor {
     *          void operator()(auto worker, ParticleView<Species, relativePos, vel> view, T_dt dt) { ... }
     *      };
     *
     *      initSpecies<Species>(setup);
     *      // inside: launchForEach(levels::particle, prBuf, MyFunctor<Species>{}, ...);
     *  @endcode
     *
     *  For species-agnostic functors that only access fields present in every species'
     *  record (e.g. multiMask, relativePos), prefer an unconstrained auto& particle
     *  parameter, the compiler checks the fields at instantiation time anyway.
     */
    template<pmacc::spearhed::SpeciesTag S, auto... TagInstances>
    using ParticleView = pmacc::spearhed::
        ParticleView<ll::SoA<typename FrameDescFor<S>::ParticleRecord, FrameDescFor<S>::numSlots>, TagInstances...>;

} // namespace spearhed
