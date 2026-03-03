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
#include "spmacc/particles/traits.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace spearhed
{
    namespace tags
    {
        DEFINE_TAG(particleId);

        using idField = ll::Field<particleId_t, uint64_t>;
    } // namespace tags


} // namespace spearhed

namespace pmacc::spearhed
{
    template<>
    struct Init<::spearhed::tags::idField>
    {
        HDINLINE constexpr void operator()(auto idView, auto worker, auto idGen) const
        {
            *idView = idGen.fetchInc(worker);
        }
    };

} // namespace pmacc::spearhed
