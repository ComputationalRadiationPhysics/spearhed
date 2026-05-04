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
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <llamaLite/llamaLite.hpp>

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(dvdt);

        // Per-axis acceleration accumulator; component count and scalar type follow CS.
        template<pmacc::spearhed::CoordinateSystem CS>
        using dvdtField = ll::Field<dvdt_t, pmacc::spearhed::CartesianRecord<CS>>;
    } // namespace tags

} // namespace spearhed
