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

#include "spearhed/param.hpp"
#include "spmacc/topology/CartesianStorage.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"
#include "spmacc/topology/Vec.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <numbers>

namespace spearhed
{
    /**
     * Quintic-spline (M5 / "super spline") SPH kernel.
     *
     * Compact support of 3h. Smoother than the cubic spline; provides a
     * larger neighbour count for the same h but increases interaction cost.
     */
    struct QuinticSplineKernel
    {
        static constexpr int supportRadius = 3;

        /**
         * Dimension-dependent normalisation constant sigma.
         *   1D: 1/120    2D: 7/(478*pi)    3D: 1/(120*pi)
         */
        HDINLINE static constexpr typename CS::T_Axis norm() noexcept
        {
            using T = typename CS::T_Axis;
            constexpr auto dim = CS::dimension;
            if constexpr(dim == 1)
                return T{1} / T{120};
            else if constexpr(dim == 2)
                return T{7} / (T{478} * std::numbers::pi_v<T>);
            else
                return T{1} / (T{120} * std::numbers::pi_v<T>);
        }

        /**
         * Quintic spline kernel W(r, h).
         *
         * @param r  Distance |r_i - r_j|  (must be >= 0).
         * @param h  Smoothing length.
         * @return   W value; zero for r >= 3h.
         */
        HDINLINE static constexpr typename CS::T_Axis W(typename CS::T_Axis r, typename CS::T_Axis h) noexcept
        {
            using T = typename CS::T_Axis;
            constexpr auto dim = CS::dimension;

            T const x = r / h;
            constexpr T sigma = norm();

            T kernelVal{0};
            if(x < T{1})
            {
                T const a = T{3} - x;
                T const b = T{2} - x;
                T const c = T{1} - x;
                kernelVal = a * a * a * a * a - T{6} * b * b * b * b * b + T{15} * c * c * c * c * c;
            }
            else if(x < T{2})
            {
                T const a = T{3} - x;
                T const b = T{2} - x;
                kernelVal = a * a * a * a * a - T{6} * b * b * b * b * b;
            }
            else if(x < T{3})
            {
                T const a = T{3} - x;
                kernelVal = a * a * a * a * a;
            }

            T hPowDim{1};
            for(pmacc::spearhed::T_Dim i = 0; i < dim; ++i)
                hPowDim *= h;

            return sigma / hPowDim * kernelVal;
        }

        /**
         * Scalar radial derivative dW/dr for the quintic spline.
         *
         * Returns zero for r >= 3h.
         */
        HDINLINE static constexpr typename CS::T_Axis dWdr(typename CS::T_Axis r, typename CS::T_Axis h) noexcept
        {
            using T = typename CS::T_Axis;
            constexpr auto dim = CS::dimension;

            T const x = r / h;
            constexpr T sigma = norm();

            T kernelVal{0};
            if(x < T{1})
            {
                T const a = T{3} - x;
                T const b = T{2} - x;
                T const c = T{1} - x;
                kernelVal = -T{5} * a * a * a * a + T{30} * b * b * b * b - T{75} * c * c * c * c;
            }
            else if(x < T{2})
            {
                T const a = T{3} - x;
                T const b = T{2} - x;
                kernelVal = -T{5} * a * a * a * a + T{30} * b * b * b * b;
            }
            else if(x < T{3})
            {
                T const a = T{3} - x;
                kernelVal = -T{5} * a * a * a * a;
            }

            T hPowDimP1{1};
            for(pmacc::spearhed::T_Dim i = 0; i <= dim; ++i)
                hPowDimP1 *= h;

            return sigma / hPowDimP1 * kernelVal;
        }

        /**
         * Scalar kernel gradient factor: gradWScalar(r, invR, h) = (dW/dr) * (1/r).
         *
         * The symmetric SPH gradient is always a scalar multiple of r_vec:
         *   gradW(r_vec, h) = dWdr(r, h) * r_vec / r = gradWScalar(r, invR, h) * r_vec.
         * Passing the pre-computed reciprocal distance @p invR (= 1/r) avoids a per-pair
         * division. Returns zero for the degenerate/out-of-support cases, exactly matching the
         * vector gradW guard (r == 0 || r >= supportRadius * h -> 0). The !(r > 0) test also
         * rejects the r == NaN produced by the framework's r = r2 * invR when r2 == 0.
         *
         * @param r     Distance |r_i - r_j| (NaN in the coincident r2 == 0 case).
         * @param invR  Reciprocal distance 1/r (+inf in the coincident r2 == 0 case).
         * @param h     Smoothing length.
         */
        HDINLINE static constexpr typename CS::T_Axis gradWScalar(
            typename CS::T_Axis r,
            typename CS::T_Axis invR,
            typename CS::T_Axis h) noexcept
        {
            using T = typename CS::T_Axis;
            if(!(r > T{0}) || r >= static_cast<T>(supportRadius) * h)
                return T{0};
            return dWdr(r, h) * invR;
        }

        /**
         * Vector kernel gradient: gradW(r_vec, h) = (dW/dr) * r_hat.
         */
        HDINLINE static constexpr pmacc::spearhed::Vec<CS, pmacc::spearhed::ValueStorage<CS>> gradW(
            pmacc::spearhed::Vec<CS, pmacc::spearhed::ValueStorage<CS>> const& rVec,
            typename CS::T_Axis r,
            typename CS::T_Axis h) noexcept
        {
            using T = typename CS::T_Axis;
            pmacc::spearhed::Vec<CS, pmacc::spearhed::ValueStorage<CS>> result{T{0}};

            if(r == T{0} || r >= static_cast<T>(supportRadius) * h)
                return result;

            T const dwdr = dWdr(r, h);
            T const inv_r = T{1} / r;

            pmacc::spearhed::for_each_tag<CS>([&](auto tag) { result[tag] = dwdr * rVec[tag] * inv_r; });
            return result;
        }
    };
} // namespace spearhed
