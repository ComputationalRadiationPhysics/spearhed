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

#include "spearhed/param/speciesDefinition.param"
#include "spmacc/Frame.hpp"
#include "spmacc/ListPointer.hpp"
#include "spmacc/meta/ComponentList.hpp"

#include <pmacc/meta/Pair.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/particles/memory/dataTypes/StaticArray.hpp>

namespace spearhed
{
    /** linked list pointer */
    using LinkedListPointer = pmacc::spearhed::meta::ComponentList<pmacc::spearhed::NextPtr>;

    /* extent particle description with pointer to a frame*/
    using FrameDescription = decltype(particleDesc.replaceFrameExtensionSeq<LinkedListPointer>());

    /** frame definition
     *
     * a group of particles is stored as frame
     */
    using FrameType = pmacc::spearhed::Frame<FrameDescription>;

} // namespace spearhed
