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

#include "spmacc/particles/traits.hpp"

#include <type_traits>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    /**
     * Per-particle view with a tag-selected access set.
     *
     * With no tags the whole record is selected, inferred from ViewIndexed's default; with tags
     * given the canonical access set is spelled out so the view is narrowed to exactly those
     * subtree(s).
     */
    template<typename TSoA, auto... TagInstances>
    using ParticleView = std::conditional_t<
        sizeof...(TagInstances) == 0,
        ll::ViewIndexed<TSoA>,
        ll::ViewIndexed<
            TSoA,
            ll::access_set_t<
                typename TSoA::record_type,
                ll::raw_access_set_t<std::remove_cvref_t<decltype(TagInstances)>...>>>>;

} // namespace pmacc::spearhed
