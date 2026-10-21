/* Copyright 2026 PMacc contributors
 * SPDX-License-Identifier: LGPL-3.0-or-later OR GPL-3.0-or-later
 */

#include <pmacc/Environment.hpp>
#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <alpaka/math/Complex.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <utility>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    constexpr uint32_t numResults = 10u;

    template<typename T>
    HDINLINE void evaluateComplex(alpaka::math::Complex<T>* output)
    {
        using Complex = alpaka::math::Complex<T>;
        using pmacc::algorithms::precisionCast::precisionCast;
        output[0] = pmacc::math::euler(T{2}, T{0.5});
        output[1] = pmacc::math::euler(T{2}, T{0.6}, T{0.8});
        output[2] = Complex{pmacc::math::norm(Complex{T{3}, T{4}})};
        output[3] = precisionCast<T>(Complex{T{1.25}, T{-2.5}});
        output[4] = precisionCast<T>(alpaka::math::Complex<float>{1.25f, -2.5f});
        output[5] = precisionCast<T>(alpaka::math::Complex<double>{1.25, -2.5});
        output[6] = pmacc::math::bessel::j0(Complex{T{0}, T{1}});
        output[7] = pmacc::math::bessel::j1(Complex{T{0}, T{1}});
        // Exercise the asymptotic branch, including complex pow, sqrt, sin and cos.
        output[8] = pmacc::math::bessel::j0(Complex{T{20}});
        output[9] = pmacc::math::bessel::j1(Complex{T{20}});
    }

    template<typename T>
    struct ComplexKernel
    {
        template<typename T_Acc>
        DINLINE void operator()(T_Acc const&, alpaka::math::Complex<T>* output) const
        {
            evaluateComplex(output);
        }
    };

    template<typename T>
    void checkResults(alpaka::math::Complex<T> const* output)
    {
        auto const tolerance = 100.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
        auto check = [&](uint32_t index, std::complex<double> expected)
        {
            CAPTURE(index);
            REQUIRE(std::abs(static_cast<double>(output[index].real()) - expected.real()) < tolerance);
            REQUIRE(std::abs(static_cast<double>(output[index].imag()) - expected.imag()) < tolerance);
        };
        check(0, std::polar(2.0, 0.5));
        check(1, {1.6, 1.2});
        check(2, {25.0, 0.0});
        check(3, {1.25, -2.5});
        check(4, {1.25, -2.5});
        check(5, {1.25, -2.5});
        // J0(i) = I0(1), J1(i) = i I1(1).
        check(6, {std::cyl_bessel_i(0.0, 1.0), 0.0});
        check(7, {0.0, std::cyl_bessel_i(1.0, 1.0)});
        check(8, {std::cyl_bessel_j(0.0, 20.0), 0.0});
        check(9, {std::cyl_bessel_j(1.0, 20.0), 0.0});
    }

    template<typename T>
    void checkComplex()
    {
        using Complex = alpaka::math::Complex<T>;
        Complex hostResults[numResults];
        evaluateComplex(hostResults);
        checkResults(hostResults);

        pmacc::HostDeviceBuffer<Complex, DIM1> buffer{pmacc::DataSpace<DIM1>{numResults}};
        auto& device = pmacc::Environment<>::get().DeviceContext();
        caravan::ControlContext context;
        auto kernel = PMACC_KERNEL(ComplexKernel<T>{})(1, 1)(
            caravan::retain(buffer.getDeviceBuffer().data(), buffer.getDeviceBuffer().getOwnedAlpakaView()));
        auto copy = buffer.deviceToHost();
        context.wait(context.spawn(
            caravan::alpaka::withDevice(device, std::move(kernel) | caravan::sequence(std::move(copy)))));
        checkResults(buffer.getHostBuffer().data());

        auto const mpiType = pmacc::mpi::getMPI_StructAsArray<Complex>();
        auto const scalarType = pmacc::mpi::getMPI_StructAsArray<T>();
        REQUIRE(mpiType.dataType == scalarType.dataType);
        REQUIRE(mpiType.sizeMultiplier == 2u * scalarType.sizeMultiplier);
    }
} // namespace

TEST_CASE("Complex math supports alpaka3 types on host and device", "[math][complex]")
{
    SECTION("float")
    {
        checkComplex<float>();
    }
    SECTION("double")
    {
        checkComplex<double>();
    }
}
