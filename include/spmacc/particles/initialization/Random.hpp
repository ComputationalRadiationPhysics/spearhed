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
#include <pmacc/random/Random.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/methods/XorMin.hpp>

#include <cstdint>
#include <type_traits>

namespace pmacc::spearhed
{
    /**
     * Distribution functor for uniform placement within the AABB along each axis.
     *
     * Maps a uniform sample in [0, 1) to a position in [axisMin, axisMax).
     */
    struct UniformDistribution
    {
        template<typename T_Scalar>
        HDINLINE constexpr T_Scalar operator()(T_Scalar uniform01, T_Scalar axisMin, T_Scalar axisMax) const
        {
            return axisMin + uniform01 * (axisMax - axisMin);
        }
    };

    /**
     * Place particles at positions drawn from an arbitrary probability distribution.
     *
     * Each particle is assigned a position by:
     *   1. Initialising a per-particle RNG state from seed and globalParticleIdx as the
     *      subsequence, so every particle gets an independent random stream.
     *   2. Drawing one uniform sample in [0, 1) per spatial axis via the PMacc RNG.
     *   3. Mapping each sample through the distribution functor to obtain the final position.
     *
     * Needs the seed and distribution function as additional arguments to placeParticle
     *
     * The distribution functor must satisfy:
     * @code
     *   T_Scalar dist(T_Scalar uniform01, T_Scalar axisMin, T_Scalar axisMax);
     * @endcode
     * It receives a uniform sample in [0, 1) and the axis bounds, and returns the sampled
     * position along that axis.  For a simple uniform layout use UniformDistribution{}.
     * For a custom PDF, implement the inverse CDF and wrap it in a functor.
     *
     */
    template<CoordinateSystem CS, template<typename> class T_RNGMethod = random::methods::XorMin>
    struct Random
    {
        DINLINE void operator()(
            auto const& worker,
            auto& particle,
            auto const& particleRegion,
            uint32_t globalParticleIdx,
            uint32_t seed,
            auto const& dist) const
        {
            using Scalar = typename CS::T_Axis;
            using T_Worker = std::remove_cvref_t<decltype(worker)>;
            using RngMethod = T_RNGMethod<typename T_Worker::Acc>;
            using State = typename RngMethod::StateType;
            using Distribution = random::distributions::Uniform<Scalar, RngMethod>;

            // Construct the RNG adapter targeting a local state pointer
            using Rng = random::Random<Distribution, RngMethod, State*>;

            auto const localMin = particleRegion.spatial.localMin();
            auto const localMax = particleRegion.spatial.localMax();
            State state;
            // globalParticleIdx as subsequence gives each particle an independent RNG stream
            RngMethod{}.init(worker, state, seed, globalParticleIdx);

            Rng rng(&state);

            for_each_tag<CS>([&](auto tag)
                             { particle[tags::relativePos][tag] = dist(rng(worker), localMin[tag], localMax[tag]); });
        }
    };

} // namespace pmacc::spearhed
