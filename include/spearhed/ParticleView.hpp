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

#include "llamaLite/llamaLite.hpp"
#include "spearhed/param/speciesDefinition.param"
#include "spmacc/particles/View.hpp"

namespace spearhed
{

    namespace particleView
    {
        using PartDescT = decltype(particleDesc);

        using SoaT = ll::SoA<PartDescT::ParticleRecord, PartDescT::numSlots>;

    } // namespace particleView

    template<auto... TagInstances>
    using ParticleView = pmacc::spearhed::ParticleView<particleView::SoaT, TagInstances...>;

} // namespace spearhed
