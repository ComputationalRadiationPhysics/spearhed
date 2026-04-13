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

#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cmath>
#include <cstdint>

namespace pmacc::spearhed
{
    // TODO think about what a reasonable way to set up number of particles in a simulation is. Forcing a strict number
    // of particles will make problems with particle placement. Currently, most likely some of the simple cubic cells
    // are simply left if there are not enough particles, or if there are too many it overflows.


    /**
     * Compute the number of unit cells along the x-axis for a 3D simple cubic (SC)
     * layout that best matches the domain aspect ratio.
     *
     * SC has 1 atom per unit cell. The cell counts per axis are derived by requiring:
     *   - nx * ny * nz = numParticles
     *   - nx / ny ~ Lx / Ly  and  nx / nz ~ Lx / Lz  (to match domain aspect ratio)
     *
     * @param numParticles total number of particles to place
     * @param aabb bounding box; all three CS axes define the SC volume
     * @return optimal number of unit cells along the x-axis, at least 1
     */
    template<CoordinateSystem CS>
    requires(CS::dimension == 3)
    constexpr uint32_t computeSCNumCells(uint32_t numParticles, AABB<CS> const& aabb)
    {
        using Scalar = typename CS::T_Axis;

        auto const extents = aabb.max - aabb.min;

        // nx^3 = numParticles * Lx^2 / (Ly * Lz)
        auto const nx = static_cast<uint32_t>(std::cbrt(
            static_cast<Scalar>(numParticles) * extents[tags::x] * extents[tags::x]
            / (extents[tags::y] * extents[tags::z])));

        return (nx < 1u) ? 1u : nx;
    }

    /**
     * Place particles in a 3D simple cubic (SC) lattice.
     *
     * Particles are arranged in a simple cubic structure with 1 atom per unit cell,
     * placed at the center of each cell:
     *   (0.5, 0.5, 0.5) in fractional cell coordinates
     *
     * Needs the number of cells in x as an additional argument to placeParticle
     *
     * The number of cells along y and z are derived from the provided nx to preserve the
     * domain aspect ratio. The globalParticleIdx maps directly to a cell index:
     *   iz = globalParticleIdx / (nx * ny)
     *   iy = (globalParticleIdx % (nx * ny)) / nx
     *   ix = globalParticleIdx % nx
     *
     */
    template<CoordinateSystem CS>
    requires(CS::dimension == 3)
    struct SC
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            uint32_t globalParticleIdx,
            uint32_t numCells) const
        {
            // TODO this only works for Cartesian systems
            // For others either figure out how to do it in those systems, or convert to and from cartesian
            using Scalar = typename CS::T_Axis;
            auto const& aabb = particleRegion.volume;

            auto const extents = aabb.max - aabb.min;

            uint32_t const nx = numCells;
            // Round to nearest integer while preserving aspect ratio
            auto const ny
                = static_cast<uint32_t>(static_cast<Scalar>(nx) * extents[tags::y] / extents[tags::x] + Scalar{0.5});
            auto const nz
                = static_cast<uint32_t>(static_cast<Scalar>(nx) * extents[tags::z] / extents[tags::x] + Scalar{0.5});

            uint32_t const iz = globalParticleIdx / (nx * ny);
            uint32_t const iy = (globalParticleIdx % (nx * ny)) / nx;
            uint32_t const ix = globalParticleIdx % nx;

            Scalar const dx = extents[tags::x] / static_cast<Scalar>(nx);
            Scalar const dy = extents[tags::y] / static_cast<Scalar>(ny > 0u ? ny : 1u);
            Scalar const dz = extents[tags::z] / static_cast<Scalar>(nz > 0u ? nz : 1u);

            // Place particle at cell center with + 0.5
            *particle[tags::relativePos][tags::x] = aabb.min[tags::x] + (static_cast<Scalar>(ix) + Scalar{0.5}) * dx;
            *particle[tags::relativePos][tags::y] = aabb.min[tags::y] + (static_cast<Scalar>(iy) + Scalar{0.5}) * dy;
            *particle[tags::relativePos][tags::z] = aabb.min[tags::z] + (static_cast<Scalar>(iz) + Scalar{0.5}) * dz;
        }
    };

} // namespace pmacc::spearhed
