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

#include "spearhed/particles/attributes/Id.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spmacc/particles/algorithms/CopyParticlesToDynSoA.hpp"
#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <cstdint>

#include <llamaLite/llamaLite.hpp>

namespace spearhed::output
{
    // Tag for the absolute (world-space) position field in the output record.
    // Distinct from relativePos (which is AABB-relative) to make the output self-contained.
    DEFINE_TAG(position);

    using positionField = ll::Field<
        position_t,
        ll::Record<
            ll::Field<pmacc::spearhed::tags::x_t, float>,
            ll::Field<pmacc::spearhed::tags::y_t, float>,
            ll::Field<pmacc::spearhed::tags::z_t, float>>>;

    /**
     * Output record for particle serialisation.
     */
    using OutputParticleRecord
        = ll::Record<spearhed::tags::idField, positionField, spearhed::tags::massField, spearhed::tags::velField>;

} // namespace spearhed::output

namespace pmacc::spearhed
{
    /**
     * Partial specialisation for position leaf paths.
     *
     * Reads each relativePos axis leaf from the source frame SoA, adds the
     * region origin component for that axis, and writes the absolute coordinate
     * to the matching position leaf in the output DynSoA.
     */
    template<>
    struct FieldCopyTrait<ll::TagPath<::spearhed::output::position_t>>
    {
        void operator()(auto const& srcSoa, auto& dst, uint32_t count, uint32_t offset, auto const& region, auto&&...)
            const
        {
            using namespace pmacc::spearhed::tags;
            using ::spearhed::output::position_t;
            using CS = std::remove_cvref_t<decltype(region.volume.origin)>::CS;

            pmacc::spearhed::for_each_tag<CS>(
                [&](auto axisTag)
                {
                    auto const o = region.volume.origin[axisTag];
                    auto const srcSpan = srcSoa.template getLeaf<ll::TagPath<relativePos_t, decltype(axisTag)>>();
                    auto dstSpan = dst.template getLeaf<ll::TagPath<position_t, decltype(axisTag)>>();

                    for(uint32_t i = 0; i < count; ++i)
                        dstSpan[offset + i] = o + srcSpan[i];
                });
        }
    };

} // namespace pmacc::spearhed
