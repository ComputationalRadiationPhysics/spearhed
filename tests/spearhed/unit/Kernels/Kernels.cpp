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

/**
 * Host-only unit tests for the SPH kernel family.
 *
 * Validates each SphKernel implementation against three properties that are
 * cheap to check and catch the typical sign / coefficient bugs introduced
 * when porting kernels:
 *
 *   1. Zero outside compact support: W(supportRadius * h, h) == 0.
 *   2. Self-symmetry of gradW: gradW(-r, |r|, h) == -gradW(r, |r|, h).
 *   3. Approximate normalisation: integral of W over the support is ~1.
 */

#include "spearhed/param.hpp"
#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spearhed/sph/QuinticSplineKernel.hpp"
#include "spearhed/sph/SphKernel.hpp"
#include "spmacc/topology/Vec.hpp"

#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using spearhed::CS;
using spearhed::Real;
using Vec = pmacc::spearhed::Vec<CS, pmacc::spearhed::ValueStorage<CS>>;

static_assert(spearhed::SphKernel<spearhed::CubicSplineKernel>);
static_assert(spearhed::SphKernel<spearhed::QuinticSplineKernel>);

namespace
{
    template<typename KernelT>
    double radialIntegral(double h)
    {
        // Spherical-shell midpoint quadrature: int_0^{R*h} W(r,h) * 4*pi*r^2 dr (3D)
        constexpr int nSteps = 10000;
        double const upper = static_cast<double>(KernelT::supportRadius) * h;
        double const dr = upper / nSteps;
        double acc = 0.0;
        for(int i = 0; i < nSteps; ++i)
        {
            double const r = (static_cast<double>(i) + 0.5) * dr;
            double const w = static_cast<double>(KernelT::W(static_cast<Real>(r), static_cast<Real>(h)));
            acc += w * 4.0 * std::numbers::pi * r * r * dr;
        }
        return acc;
    }
} // namespace

TEMPLATE_TEST_CASE(
    "SPH kernels: zero outside compact support",
    "[sph][kernel]",
    spearhed::CubicSplineKernel,
    spearhed::QuinticSplineKernel)
{
    using K = TestType;
    constexpr Real h = Real{0.5};
    Real const r_outside = static_cast<Real>(K::supportRadius) * h;

    REQUIRE(static_cast<double>(K::W(r_outside, h)) == Catch::Approx(0.0).margin(1e-7));
    REQUIRE(static_cast<double>(K::W(r_outside + Real{0.1f}, h)) == Catch::Approx(0.0).margin(1e-7));

    // Inside support: positive at the centre.
    REQUIRE(static_cast<double>(K::W(Real{0}, h)) > 0.0);
}

TEMPLATE_TEST_CASE(
    "SPH kernels: gradW is anti-symmetric in r_vec",
    "[sph][kernel]",
    spearhed::CubicSplineKernel,
    spearhed::QuinticSplineKernel)
{
    using K = TestType;
    constexpr Real h = Real{0.5};

    Vec rVec{Real{0}};
    rVec[pmacc::spearhed::tags::x_t{}] = Real{0.4f};
    rVec[pmacc::spearhed::tags::y_t{}] = Real{-0.3f};
    rVec[pmacc::spearhed::tags::z_t{}] = Real{0.2f};
    auto const r = Real(std::sqrt(0.4f * 0.4f + 0.3f * 0.3f + 0.2f * 0.2f));

    Vec gPos = K::gradW(rVec, r, h);

    Vec rVecNeg{Real{0}};
    pmacc::spearhed::for_each_tag<CS>([&](auto tag) { rVecNeg[tag] = -rVec[tag]; });
    Vec gNeg = K::gradW(rVecNeg, r, h);

    pmacc::spearhed::for_each_tag<CS>(
        [&](auto tag)
        {
            double const a = static_cast<double>(gPos[tag]);
            double const b = static_cast<double>(gNeg[tag]);
            REQUIRE(a == Catch::Approx(-b).margin(1e-6));
        });
}

TEMPLATE_TEST_CASE(
    "SPH kernels: 3D radial integral approximates 1",
    "[sph][kernel]",
    spearhed::CubicSplineKernel,
    spearhed::QuinticSplineKernel)
{
    using K = TestType;
    // Real is float in the default config; the per-step roundoff of a 10k-step
    // quadrature accumulates, so a 1% tolerance is the right ballpark here.
    REQUIRE(radialIntegral<K>(0.5) == Catch::Approx(1.0).epsilon(1e-2));
}
