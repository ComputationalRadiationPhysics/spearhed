#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>

namespace spearhed
{
    // type which holds the number of dimensions
    using T_Dim = uint32_t;

    // Currently assumes constant metrics
    enum class MetricKind
    {
        // Dense metric tensor
        General,
        // Diagonal metric tensor
        Orthogonal,
        // Identity metric tensor
        Orthonormal
    };

    template<typename CS>
    concept CoordinateSystem = requires {
        // type in which the coordinates are stored
        typename CS::Scalar;
        { CS::dimension } -> std::convertible_to<std::size_t>;
        { CS::metricKind } -> std::convertible_to<MetricKind>;
    };

    // Cartesian coordinate system chart
    template<typename T, T_Dim Dim>
    struct Cartesian
    {
        static constexpr char const* name = "Cartesian";
        using Scalar = T;
        static constexpr std::size_t dimension = Dim;
        static constexpr MetricKind metricKind = MetricKind::Orthonormal;

        // Math conversions
        static constexpr void from_spherical(double r, double th, double ph, double& x, double& y, double& z)
        {
            x = r * std::sin(th) * std::cos(ph);
            y = r * std::sin(th) * std::sin(ph);
            z = r * std::cos(th);
        }

        // No-op for self conversion
        static constexpr void from_cartesian(double x, double y, double z, double& out_x, double& out_y, double& out_z)
        {
            out_x = x;
            out_y = y;
            out_z = z;
        }
    };

    // Polar coordinate system chart
    template<typename T, T_Dim Dim>
    struct Polar
    {
        static constexpr char const* name = "Polar";
        using Scalar = T;
        static constexpr std::size_t dimension = Dim;
        static constexpr MetricKind metricKind = MetricKind::Orthonormal;
    };

    //     template<typename T, T_Dim Dim>
    // struct Spherical
    // {
    //     static constexpr char const* name = "Spherical";
    //     using Scalar = T;
    //     static constexpr std::size_t dimension = Dim;
    //     static constexpr MetricKind metricKind = MetricKind::Orthonormal;
    //     static constexpr void from_cartesian(double x, double y, double z, double& r, double& th, double& ph)
    //     {
    //         r = std::sqrt(x * x + y * y + z * z);
    //         th = (r > 1e-12) ? std::acos(z / r) : 0.0;
    //         ph = std::atan2(y, x);
    //     }

    //     static constexpr void from_spherical(
    //         double r,
    //         double th,
    //         double ph,
    //         double& out_r,
    //         double& out_th,
    //         double& out_ph)
    //     {
    //         out_r = r;
    //         out_th = th;
    //         out_ph = ph;
    //     }
    // };


} // namespace spearhed
