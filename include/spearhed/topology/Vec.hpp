#pragma once

#include "spearhed/topology/CoordinateSystem.hpp"

#include <array>

namespace spearhed
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
} // namespace spearhed
