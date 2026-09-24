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

#include "spmacc/particles/spatial/CandidateProvider.hpp"

#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    /** @brief Compact device view of one interaction source. */
    template<typename T_SourceStoreView, CandidateProvider T_CandidateProviderView>
    struct SourceView
    {
        T_SourceStoreView sourceStore;
        T_CandidateProviderView candidates;
    };

    /**
     * @brief Host-side source store and its candidate-enumeration strategy.
     *
     * Species remains an associated type of the source store so NeighbourBundle's
     * compile-time role and species selection is independent of provider type.
     */
    template<typename T_SourceStore, typename T_CandidateProvider>
    struct InteractionEntry
    {
        using Species = typename T_SourceStore::Species;
        using CandidateProviderType = T_CandidateProvider;

        T_SourceStore* sourceStore;
        T_CandidateProvider candidateProvider;

        auto deviceView() const
        {
            return SourceView{sourceStore->getDeviceDataBox(), candidateProvider.deviceView()};
        }
    };
} // namespace pmacc::spearhed
