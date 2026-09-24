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

namespace spearhed::mapping_setups
{
    /** @brief Default persistent material-cohort mapping. */
    struct MaterialAabbSetup : DefaultSetup
    {
        using DecompositionGroups = std::tuple<pmacc::spearhed::MaterialAabbDecompositionGroup<
            pmacc::spearhed::species::Default,
            pmacc::spearhed::species::Boundary,
            pmacc::spearhed::species::Tracer>>;
    };
} // namespace spearhed::mapping_setups
