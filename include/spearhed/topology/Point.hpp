#pragma once

#include "spearhed/topology/CoordinateSystem.hpp"

#include <concepts>

namespace spearhed
{

    // Think about alignment and memory laybout
    template<CoordinateSystem CS, typename Storage>
    struct Point;

    // {
    //     using Scalar = typename CS::Scalar;

    //     static constexpr T_Dim dim = CS::dimension;

    //     using Storage::Storage;

    //     // friend constexpr VectorType operator-(Point const& lhs, Point const& rhs) noexcept
    //     // {
    //     //     VectorType v;
    //     //     for(T_Dim i = 0; i < dim; ++i)
    //     //         v.data_[i] = lhs.data_[i] - rhs.data_[i];
    //     //     return v;
    //     // }

    //     // friend constexpr Point operator+(Point p, VectorType const& v) noexcept
    //     // {
    //     //     for(T_Dim i = 0; i < dim; ++i)
    //     //         p.data_[i] += v.data_[i];
    //     //     return p;
    //     // }
    // };

    template<typename T, T_Dim Dim, typename Storage>
    struct Point<Cartesian<T, Dim>, Storage> : public Storage
    {
        using CS = Cartesian<T, Dim>;
        using Scalar = typename CS::Scalar;
        static constexpr T_Dim dim = CS::dimension;

        using Storage::Storage;

        template<CoordinateSystem OtherCS, typename OtherStorage>
        constexpr explicit(false) Point(Point<OtherCS, OtherStorage> const& other) noexcept
        {
            transformPoint(other, this);
        }

        // Assignment operator for View = Value (Scatter) or View = View
        template<CoordinateSystem OtherCS, typename OtherStorage>
        constexpr Point& operator=(Point<OtherCS, OtherStorage> const& other) noexcept
        {
        }
    };

    template<typename T, T_Dim Dim, typename Storage>
    struct Point<Polar<T, Dim>, Storage> : public Storage
    {
        using CS = Polar<T, Dim>;
        using Scalar = typename CS::Scalar;
        static constexpr T_Dim dim = CS::dimension;

        using Storage::Storage;
    };

    template<CoordinateSystem From, CoordinateSystem To, typename StorageFrom, typename StorageTo>
    void transformPoint(Point<From, StorageFrom> const& x, Point<To, StorageTo>& y)
    {
        // Careful. Do read copy write.
        // x and y might be the same particle, so we dont want to overwrite componenets prematurely
        if constexpr(!std::same_as<From, To>)
        {
            // if constexpr(std::same_as<To, Cartesian>)
            // {
            // OtherCS::from_spherical(a, b, c, this->get_x(), this->template get<1>(), this->template get<2>());
            // }
            // elif so on
        }
    }

} // namespace spearhed
