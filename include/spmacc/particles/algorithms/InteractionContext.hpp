/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Vec.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace pmacc::spearhed
{
    /**
     * @brief Geometric context passed to particle-particle interaction functors.
     *
     * All geometry is resolved by the interaction framework before the functor runs, so
     * functors never recompute distances:
     *   - @ref rVec is the displacement r_i - r_j expressed in the own particle's region
     *     coordinates (the neighbour position was pre-shifted by the origin offset during
     *     staging, so no per-pair coordinate reconstruction is needed).
     *   - @ref r2 is the squared distance; the framework's radius cull runs on it, avoiding a
     *     square root on the reject path.
     *   - @ref invR is the reciprocal distance 1/|rVec|, computed once per accepted pair with a
     *     single reciprocal square root (pmacc::math::rsqrt(r2)).
     *   - @ref r is the Euclidean distance |rVec|, derived as r2 * invR (no second root).
     *   - @ref isSelf is true only for the pairing of a particle with its own frame slot.
     *
     * Degenerate invariant: for coincident non-self particles (r2 == 0) rsqrt yields +inf, so
     * @ref invR is +inf and @ref r is NaN. Functors must therefore guard the r2 == 0 case; the
     * SPH kernels do this in gradWScalar (which returns zero when the distance is not strictly
     * positive), matching the old vector gradW that returned zero for r == 0. The self pair
     * (@ref isSelf) is likewise degenerate and is expected to be short-circuited by the functor.
     */
    template<CoordinateSystem CS>
    struct PairContext
    {
        //! Displacement r_i - r_j in the own particle's region coordinates.
        Vec<CS, ValueStorage<CS>> rVec;
        //! Squared distance dot(rVec, rVec); the radius cull ran on this value.
        typename CS::T_Axis r2;
        //! Euclidean distance |rVec| = r2 * invR; NaN in the degenerate r2 == 0 case.
        typename CS::T_Axis r;
        //! Reciprocal distance 1/|rVec| = rsqrt(r2); +inf in the degenerate r2 == 0 case.
        typename CS::T_Axis invR;
        //! True only when this pair is the particle interacting with its own slot.
        bool isSelf;
    };

} // namespace pmacc::spearhed
