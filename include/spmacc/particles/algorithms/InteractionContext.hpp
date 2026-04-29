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
     * Geometric context passed to particle-particle interaction functors.
     *
     * Carries the displacement vector r_vec = r_i - r_j and a self-interaction flag.
     * Additional fields (relative velocity, etc.) can be added here in later phases.
     * Scalar distance r() is computed lazily to avoid paying for a sqrt in functors
     * that only need the direction.
     */
    template<CoordinateSystem CS>
    struct InteractionContext
    {
        Vec<CS, ValueStorage<CS>> r_vec;
        bool is_self;

        /** Euclidean distance |r_vec|. Recomputed on each call; no caching. */
        [[nodiscard]] HDINLINE constexpr typename CS::T_Axis r() const
        {
            return norm2(r_vec);
        }
    };

} // namespace pmacc::spearhed
