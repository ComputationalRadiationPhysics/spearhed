/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/spatial/BroadPhaseView.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"
#include "spmacc/particles/spatial/InteractionQuery.hpp"

namespace pmacc::spearhed
{
    /**
     * @brief Compatibility wrapper for callers that explicitly request the CSR fallback.
     *
     * The generic pairwise path now selects the same fallback whenever neither
     * mapping pair supplies a more specific provider.
     */
    template<BroadPhasePreparedRegionSet T_Target, typename T_Radius, BroadPhasePreparedRegionSet... T_Sources>
    [[nodiscard]] auto makeMaterializedCsrInteractionPlan(
        T_Target const& target,
        InteractionQuery<T_Radius> const& query,
        T_Sources const&... sources)
    {
        return makeInteractionPlan(target, query, sources...);
    }

    template<BroadPhasePreparedRegionSet T_Target, typename T_Radius, BroadPhasePreparedRegionSet... T_Sources>
    [[nodiscard]] auto makeMaterializedCsrInteractionPlan(
        T_Target const& target,
        T_Radius radius,
        T_Sources const&... sources)
    {
        return makeInteractionPlan(target, InteractionQuery{radius}, sources...);
    }
} // namespace pmacc::spearhed
