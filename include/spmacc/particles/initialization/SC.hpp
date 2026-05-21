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

#include <pmacc/assert.hpp>
#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/math/functions/Pow.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pmacc::spearhed
{
    // An exact, arbitrary particle count cannot in general be realized as an isotropic SC lattice that tiles an
    // arbitrary AABB (the count must factor as prod_i(n_i) with the right aspect ratio). What actually prevents
    // boundary artifacts is a *fully filled* lattice, so here `numParticles` is treated as a lower bound: the lattice
    // shape is rounded to the nearest aspect-ratio-preserving fit and grown to prod_i(n_i) >= numParticles. The
    // resulting count (computeSCTotalParticles) must then be used for weight/charge normalization.
    // TODO consider a particles-per-cell formulation (a la PIConGPU)


    /**
     * Compute the per-axis number of unit cells for a simple cubic (SC) lattice.
     *
     * This is the single source of truth for the lattice shape: both the particle
     * count to allocate (computeSCTotalParticles) and the placement in
     * SC::operator() are derived from it, so allocation and placement can never
     * diverge.
     *
     * SC has 1 atom per unit cell. The per-axis counts n_i are chosen to
     *   - match the domain aspect ratio: n_i / n_x ~ L_i / L_x, and
     *   - yield prod_i(n_i) >= numParticles (a fully filled lattice, so there is
     *     never a partially populated boundary plane).
     *
     * Starting from the isotropic lattice constant (equal spacing on every axis)
     *   a = pow(prod_i(L_i) / numParticles, 1/Dim),   n_i = L_i / a,
     * every axis is rounded to nearest. Any remaining shortfall is removed by
     * repeatedly refining the axis with the coarsest spacing (largest L_i / n_i),
     * which keeps the per-axis spacings balanced instead of distorting a single
     * axis. In 1D the result is exactly numParticles.
     *
     * @param numParticles requested particle count (treated as a lower bound)
     * @param aabb bounding box; all CS axes define the SC volume
     * @return per-axis cell counts; product is >= numParticles (all zero if numParticles == 0)
     */
    template<CoordinateSystem CS>
    constexpr std::array<uint32_t, CS::dimension> computeSCCellCounts(uint32_t numParticles, AABB<CS> const& aabb)
    {
        using Scalar = typename CS::T_Axis;
        constexpr std::size_t Dim = CS::dimension;

        std::array<uint32_t, Dim> n{};
        if(numParticles == 0u)
            return n; // all-zero -> product 0, no particles

        auto const extents = aabb.max - aabb.min;

        // Per-axis extents in index order, for the spacing computations below.
        std::array<Scalar, Dim> L{};
        for_each_index<CS>(
            [&](auto i)
            {
                using tag_i = tag_of<CS, i.value>;
                L[i.value] = extents[tag_i{}];
            });

        if constexpr(Dim == 1)
        {
            n[0] = numParticles; // exact in 1D
            return n;
        }
        else
        {
            // Isotropic lattice constant from volume per particle: a = (V / N)^(1/Dim),
            // with V = prod_i(L_i). Equal spacing a on every axis then gives n_i = L_i / a.
            Scalar volume{1};
            for(std::size_t i = 0; i < Dim; ++i)
                volume *= L[i];

            Scalar const a
                = pmacc::math::pow(volume / static_cast<Scalar>(numParticles), Scalar{1} / static_cast<Scalar>(Dim));

            // Initial guess: round every axis to nearest (>= 1).
            for(std::size_t i = 0; i < Dim; ++i)
            {
                auto const val = static_cast<uint32_t>(L[i] / a + Scalar{0.5});
                n[i] = (val == 0u) ? 1u : val;
            }

            // Spread any shortfall across axes: grow the coarsest axis (largest
            // L_i / n_i) until the lattice holds at least numParticles cells.
            auto product = [&]()
            {
                uint64_t p = 1u;
                for(auto v : n)
                    p *= v;
                return p;
            };
            while(product() < static_cast<uint64_t>(numParticles))
            {
                std::size_t coarsest = 0;
                Scalar maxSpacing = L[0] / static_cast<Scalar>(n[0]);
                for(std::size_t i = 1; i < Dim; ++i)
                {
                    Scalar const s = L[i] / static_cast<Scalar>(n[i]);
                    if(s > maxSpacing)
                    {
                        maxSpacing = s;
                        coarsest = i;
                    }
                }
                ++n[coarsest];
            }
            return n;
        }
    }

    /**
     * Total number of particles created for an SC lattice = prod_i(n_i).
     *
     * Use this as the particle count to create so that every particle maps to a
     * unique, fully populated lattice cell. This (not the requested count) is the
     * value to use for weight/charge normalization.
     *
     * @param numParticles requested particle count (treated as a lower bound)
     * @param aabb bounding box
     * @return actual particle count >= numParticles (0 if numParticles == 0)
     */
    template<CoordinateSystem CS>
    constexpr uint32_t computeSCTotalParticles(uint32_t numParticles, AABB<CS> const& aabb)
    {
        auto const n = computeSCCellCounts(numParticles, aabb);
        uint64_t total = 1u;
        for(auto v : n)
            total *= v;
        PMACC_ASSERT(total <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        return static_cast<uint32_t>(total);
    }

    /**
     * Place particles in a simple cubic (SC) lattice.
     *
     * Particles are arranged in an SC structure with 1 atom per unit cell, placed
     * at the center of each cell (0.5 in fractional cell coordinates per axis).
     *
     * The per-axis cell counts are precomputed on the host via computeSCCellCounts
     * and passed in as a std::array, so this functor performs no shape computation
     * of its own and can never disagree with the allocated particle count. The globalParticleIdx
     * maps to a per-axis cell index via mixed-radix decomposition over the per-axis
     * counts:
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
            std::array<uint32_t, CS::dimension> n) const
        {
            // TODO this only works for Cartesian systems
            // For others either figure out how to do it in those systems, or convert to and from cartesian
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
                    *particle[tags::relativePos][tag] = aabb.min[tag] + (static_cast<Scalar>(ik) + Scalar{0.5}) * di;
                });
        }
    };

} // namespace pmacc::spearhed
