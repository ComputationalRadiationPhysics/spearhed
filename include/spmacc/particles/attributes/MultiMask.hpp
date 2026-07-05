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

#include "spmacc/particles/Predicate.hpp"
#include "spmacc/particles/traits.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace tags
    {
        DEFINE_TAG(multiMask);
        using MultiMaskField = ll::Field<multiMask_t, uint8_t>;
    } // namespace tags

    template<>
    struct InitValue<tags::MultiMaskField>
    {
        HDINLINE constexpr void operator()(auto multiMaskView, uint8_t val) const
        {
            *multiMaskView = val;
        }
    };

    namespace pred
    {
        /** Predicate: the slot holds a live particle (multiMask set). The ubiquitous liveness check --
         *  algorithms skip every slot this rejects. Compose with && || ! like any other predicate. */
        struct Occupied : PredicateBase
        {
            HDINLINE constexpr bool operator()(auto const& particle) const
            {
                return static_cast<bool>(*particle[tags::multiMask]);
            }
        };

        inline constexpr Occupied occupied{};
    } // namespace pred

} // namespace pmacc::spearhed
