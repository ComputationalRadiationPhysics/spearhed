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
#include "spmacc/topology/CoordinateSystem.hpp"

#include <cmath>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<T_Dim D>
        struct CartesianTags;

        template<>
        struct CartesianTags<1>
        {
            using type = std::tuple<tags::x_t>;
        };

        template<>
        struct CartesianTags<2>
        {
            using type = std::tuple<tags::x_t, tags::y_t>;
        };

        template<>
        struct CartesianTags<3>
        {
            using type = std::tuple<tags::x_t, tags::y_t, tags::z_t>;
        };
    } // namespace detail

    // Cartesian coordinate system chart
    template<typename T, T_Dim Dim>
    struct Cartesian
    {
        static constexpr char const* name = "Cartesian";
        using T_Axis = T;
        static constexpr std::size_t dimension = Dim;
        static constexpr MetricKind metricKind = MetricKind::Orthonormal;

        // Math conversions
        // TODO make this dimension independent
        static constexpr void from_spherical(double r, double th, double ph, double& x, double& y, double& z)
        {
            x = r * std::sin(th) * std::cos(ph);
            y = r * std::sin(th) * std::sin(ph);
            z = r * std::cos(th);
        }

        static_assert(Dim >= 1 && Dim <= 3, "Cartesian dimension must be 1, 2, or 3.");
        using tags = typename detail::CartesianTags<Dim>::type;

        // No-op for self conversion
        // TODO make this dimension independent
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
        using T_Axis = T;
        static constexpr std::size_t dimension = Dim;
        static constexpr MetricKind metricKind = MetricKind::Orthonormal;
    };

    //     template<typename T, T_Dim Dim>
    // struct Spherical
    // {
    //     static constexpr char const* name = "Spherical";
    //     using T_Axis = T;
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


} // namespace pmacc::spearhed
