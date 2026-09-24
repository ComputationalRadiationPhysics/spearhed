/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <cstdint>

namespace pmacc::spearhed
{
    /**
     * @brief One directed source-bucket candidate for a target bucket.
     *
     * The translation maps a source particle's chart-local position into the
     * target chart. Distinct periodic images may therefore name the same source
     * bucket with different translations.
     */
    template<typename T_Translation>
    struct RegionCandidate
    {
        using Translation = T_Translation;

        uint32_t sourceBucketSlot;
        Translation sourceChartOffsetInTarget;
        bool isUnshiftedImage;
    };
} // namespace pmacc::spearhed
