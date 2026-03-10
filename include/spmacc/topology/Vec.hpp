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

#include "spmacc/topology/CoordinateSystem.hpp"

#include <array>

namespace pmacc::spearhed
{

    // Think about alignment and memory laybout
    template<CoordinateSystem CS>
    struct Vec
    {
        using Scalar = typename CS::Scalar;

        static constexpr T_Dim dim = CS::dimension;

        std::array<Scalar, dim> data_;

        [[nodiscard]] constexpr Scalar* data() noexcept
        {
            return data_.data();
        }

        [[nodiscard]] constexpr Scalar const* data() const noexcept
        {
            return data_.data();
        }

        [[nodiscard]] constexpr Scalar& operator[](std::size_t i) noexcept
        {
            return data_[i];
        }

        [[nodiscard]] constexpr Scalar operator[](std::size_t i) const noexcept
        {
            return data_[i];
        }

        // friend constexpr VectorType operator-(Point const& lhs, Point const& rhs) noexcept
        // {
        //     VectorType v;
        //     for(T_Dim i = 0; i < dim; ++i)
        //         v.data_[i] = lhs.data_[i] - rhs.data_[i];
        //     return v;
        // }

        // friend constexpr Point operator+(Point p, VectorType const& v) noexcept
        // {
        //     for(T_Dim i = 0; i < dim; ++i)
        //         p.data_[i] += v.data_[i];
        //     return p;
        // }
    };
} // namespace pmacc::spearhed
