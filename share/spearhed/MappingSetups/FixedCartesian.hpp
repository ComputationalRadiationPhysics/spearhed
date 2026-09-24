/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spearhed/param/setup.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"

namespace spearhed::mapping_setups
{
    struct FluidGrid
    {
        auto operator()(auto const& setup) const
        {
            using Grid = pmacc::spearhed::FixedCartesianGrid<CS>;
            return Grid{
                {setup.domain.origin + setup.domain.min, setup.domain.origin + setup.domain.max},
                {8u, 8u, 8u}};
        }
    };

    /** @brief Fluid grid with material AABBs retained for boundary and tracer species. */
    struct FixedCartesianSetup : DefaultSetup
    {
        using DecompositionGroups = std::tuple<
            pmacc::spearhed::FixedCartesianDecompositionGroup<FluidGrid, pmacc::spearhed::species::Default>,
            pmacc::spearhed::ExplicitInvalidationMaterialGroup<pmacc::spearhed::species::Boundary>,
            pmacc::spearhed::MaterialAabbDecompositionGroup<pmacc::spearhed::species::Tracer>>;
    };
} // namespace spearhed::mapping_setups
