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

#include <array>
#include <cmath>
#include <cstdint>

namespace pmacc::spearhed
{
    // TODO consider a particles-per-cell formulation (a la PIConGPU)


    /**
     * Compute the number of unit cells along the x-axis for a simple cubic (SC)
     * layout that best matches the domain aspect ratio.
     *
     * SC has 1 atom per unit cell. Per-axis cell counts are derived by requiring:
     *   - prod_i(n_i) = numParticles
     *   - n_i / n_x ~ L_i / L_x  (to match domain aspect ratio)
     *
     * Combining these gives n_x^Dim = numParticles * prod_{i>0}(L_x / L_i),
     * so n_x = pow(numParticles * prod_{i>0}(L_x / L_i), 1/Dim).
     * In 1D this collapses to n_x = numParticles.
     *
     * @param numParticles total number of particles to place
     * @param aabb bounding box; all CS axes define the SC volume
     * @return optimal number of unit cells along the x-axis, at least 1
     */
    template<CoordinateSystem CS>
    constexpr uint32_t computeSCNumCells(uint32_t numParticles, AABB<CS> const& aabb)
    {
        using Scalar = typename CS::T_Axis;
        constexpr std::size_t Dim = CS::dimension;

        auto const extents = aabb.max - aabb.min;
        using x_t = tag_of<CS, 0>;
        Scalar const Lx = extents[x_t{}];

        Scalar ratioProduct{1};
        for_each_index<CS>(
            [&](auto i)
            {
                if constexpr(i.value > 0)
                {
                    using tag_i = tag_of<CS, i.value>;
                    ratioProduct *= Lx / extents[tag_i{}];
                }
            });

        auto const nxFloat
            = std::pow(static_cast<Scalar>(numParticles) * ratioProduct, Scalar{1} / static_cast<Scalar>(Dim));
        auto const nx = static_cast<uint32_t>(nxFloat);
        return (nx < 1u) ? 1u : nx;
    }

    /**
     * Place particles in a simple cubic (SC) lattice.
     *
     * Particles are arranged in an SC structure with 1 atom per unit cell, placed
     * at the center of each cell (0.5 in fractional cell coordinates per axis).
     *
     * Cell counts per non-x axis are derived from `numCells` (= n_x) preserving
     * the domain aspect ratio. The globalParticleIdx maps to a per-axis cell
     * index via mixed-radix decomposition over the per-axis counts:
     *   i_k = (globalParticleIdx / prod_{j<k} n_j) mod n_k
     */
    template<CoordinateSystem CS>
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
            constexpr std::size_t Dim = CS::dimension;
            auto const& aabb = particleRegion.volume;
            auto const extents = aabb.max - aabb.min;

            using x_t = tag_of<CS, 0>;
            Scalar const Lx = extents[x_t{}];
            uint32_t const nx = numCells;

            // Per-axis cell counts: n_0 = nx; n_i = round(nx * L_i / L_x) (>= 1)
            std::array<uint32_t, Dim> n{};
            for_each_index<CS>(
                [&](auto i)
                {
                    if constexpr(i.value == 0)
                    {
                        n[0] = nx;
                    }
                    else
                    {
                        using tag_i = tag_of<CS, i.value>;
                        auto const val
                            = static_cast<uint32_t>(static_cast<Scalar>(nx) * extents[tag_i{}] / Lx + Scalar{0.5});
                        n[i.value] = (val == 0u) ? 1u : val;
                    }
                });

            // Mixed-radix index decomposition + position write at cell centre.
            uint32_t accum = 1u;
            for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    uint32_t const ik = (globalParticleIdx / accum) % n[i.value];
                    accum *= n[i.value];
                    Scalar const di = extents[tag] / static_cast<Scalar>(n[i.value]);
                    particle[tags::relativePos][tag] = aabb.min[tag] + (static_cast<Scalar>(ik) + Scalar{0.5}) * di;
                });
        }
    };

} // namespace pmacc::spearhed
