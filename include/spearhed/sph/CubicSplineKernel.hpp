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
     * Cubic-spline (M4) SPH kernel.
     *
     * Compact support of 2h. Stateless tag struct; all members are static so a
     * value of CubicSplineKernel{} is just a type carrier for compile-time
     * dispatch (e.g. via std::variant of kernel tag types).
     */
    struct CubicSplineKernel
    {
        // TODO units?
        static constexpr int supportRadius = 2;

        /**
         * Dimension-dependent normalisation constant sigma.
         *   1D: 2/3    2D: 10/(7*pi)    3D: 1/pi
         */
        HDINLINE static constexpr typename CS::T_Axis norm() noexcept
        {
            using T = typename CS::T_Axis;
            constexpr auto dim = CS::dimension;
            if constexpr(dim == 1)
                return T{2} / T{3};
            else if constexpr(dim == 2)
                return T{10} / (T{7} * std::numbers::pi_v<T>);
            else
                return T{1} / std::numbers::pi_v<T>;
        }

        /**
         * Cubic spline kernel W(r, h).
         *
         * @param r  Distance |r_i - r_j|  (must be >= 0).
         * @param h  Smoothing length.
         * @return   W value; zero for r >= 2h.
         */
        HDINLINE static constexpr typename CS::T_Axis W(typename CS::T_Axis r, typename CS::T_Axis h) noexcept
        {
            using T = typename CS::T_Axis;
            constexpr auto dim = CS::dimension;

            T const x = r / h;
            constexpr T sigma = norm();

            T kernelVal{0};
            if(x < T{1})
                kernelVal = T{1} - T{1.5} * x * x + T{0.75} * x * x * x;
            else if(x < T{2})
                kernelVal = T{0.25} * (T{2} - x) * (T{2} - x) * (T{2} - x);

            // Compute h^dim (host/device compatible, small exponent)
            T hPowDim{1};
            for(pmacc::spearhed::T_Dim i = 0; i < dim; ++i)
                hPowDim *= h;

            return sigma / hPowDim * kernelVal;
        }

        /**
         * Scalar radial derivative dW/dr for the cubic spline.
         *
         * gradW = (dW/dr) * r_hat = dWdr * (r_vec / r).
         * Returns zero for r >= 2h or r == 0.
         */
        HDINLINE static constexpr typename CS::T_Axis dWdr(typename CS::T_Axis r, typename CS::T_Axis h) noexcept
        {
            using T = typename CS::T_Axis;
            constexpr auto dim = CS::dimension;

            T const x = r / h;
            constexpr T sigma = norm();

            T kernelVal{0};
            if(x >= T{0} && x < T{1})
                kernelVal = T{2.25} * x * x - T{3} * x;
            else if(x < T{2})
                kernelVal = -T{0.75} * (T{2} - x) * (T{2} - x);

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
