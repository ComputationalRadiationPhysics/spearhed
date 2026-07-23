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
#include <limits>

namespace pmacc::spearhed
{
    // TODO consider a particles-per-cell formulation (a la PIConGPU)

    namespace detail
    {
        /**
         * Truncate a positive floating-point value, correcting only roundoff
         * immediately below an integer boundary.
         *
         * The tolerance scales with the quotient because its absolute rounding
         * error does too. Four epsilon covers rounding of both operands and the
         * division while remaining far below a genuinely fractional cell.
         */
        template<typename Scalar>
        HDINLINE constexpr uint32_t truncateNearInteger(Scalar value)
        {
            auto const truncated = static_cast<uint32_t>(value);
            auto const scale = (value > Scalar{1}) ? value : Scalar{1};
            auto const tolerance = Scalar{4} * std::numeric_limits<Scalar>::epsilon() * scale;
            auto const distanceToNext = static_cast<Scalar>(truncated + 1u) - value;

            return (distanceToNext >= Scalar{0} && distanceToNext <= tolerance) ? truncated + 1u : truncated;
        }
    } // namespace detail

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
     * Compute per-axis simple cubic (SC) cell counts from an AABB and a uniform spacing.
     *
     * Axis 0 (x) truncates L_0 / spacing (consistent with the existing 1D scalar
     * computeSCNumCells), except that values within floating-point roundoff of
     * the next integer snap to that integer. Transverse axes round to nearest.
     * Each count is clamped to at least 1 so every axis has at least one cell.
     *
     * @param aabb    Axis-aligned bounding box of the region
     * @param spacing Uniform inter-particle spacing
     * @return Per-axis cell count array; the product equals the number of lattice sites
     */
    template<CoordinateSystem CS>
    HDINLINE constexpr std::array<uint32_t, CS::dimension> computeSCCellCounts(
        AABB<CS> const& aabb,
        typename CS::T_Axis spacing)
    {
        using Scalar = typename CS::T_Axis;
        constexpr std::size_t Dim = CS::dimension;
        auto const extents = aabb.max - aabb.min;
        std::array<uint32_t, Dim> counts{};

        // Axis 0: truncate, while tolerating roundoff just below an integer.
        counts[0] = detail::truncateNearInteger(extents[tag_of<CS, 0>{}] / spacing);
        counts[0] = (counts[0] == 0u) ? 1u : counts[0];

        // Transverse axes: round to nearest integer
        for_each_index<CS>(
            [&](auto i)
            {
                if constexpr(i.value > 0)
                {
                    using tag_i = tag_of<CS, i.value>;
                    auto const val = static_cast<uint32_t>(extents[tag_i{}] / spacing + Scalar{0.5});
                    counts[i.value] = (val == 0u) ? 1u : val;
                }
            });

        return counts;
    }

    /**
     * Compute the product of per-axis cell counts, i.e. the total number of
     * lattice sites in the SC grid.
     *
     * @param n Per-axis cell count array
     * @return Total number of SC lattice sites = prod_i(n_i)
     */
    template<std::size_t Dim>
    HDINLINE constexpr uint32_t numSCLatticeSites(std::array<uint32_t, Dim> const& n)
    {
        uint32_t prod = 1u;
        for(std::size_t i = 0; i < Dim; ++i)
            prod *= n[i];
        return prod;
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
        /**
         * Place particles using a per-axis cell-count array supplied by the caller.
         *
         * This overload decouples the count computation from the placement:
         * the caller guarantees count == prod(n), which prevents SC lattice
         * aliasing (particles stacking at identical positions) in multi-D.
         * The mixed-radix decomposition, cell-centre position write, and
         * aspect-ratio derivation of transverse counts are inherited from the
         * scalar overload below.
         */
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            uint32_t globalParticleIdx,
            std::array<uint32_t, CS::dimension> const& n) const
        {
            using Scalar = typename CS::T_Axis;
            auto const& aabb = particleRegion.volume;
            auto const extents = aabb.max - aabb.min;

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

        /**
         * Place particles using a scalar numCells (legacy 1D / aspect-ratio interface).
         *
         * numCells is the count along axis 0; transverse counts are derived from
         * the AABB aspect ratio.  In 2+D this derivation can fail to satisfy
         count == prod(n), which causes lattice aliasing -- prefer the array
         * overload when exact particle counts matter.
         */
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

            // Delegate to the array overload
            (*this)(worker, particle, particleRegion, globalParticleIdx, n);
        }
    };

} // namespace pmacc::spearhed
