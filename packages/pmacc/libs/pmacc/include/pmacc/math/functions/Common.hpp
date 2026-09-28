/* Copyright 2024-2024 Rene Widera
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

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/dimensions/Definition.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/math/internal/math.hpp>
#include <alpaka/math/internal/stlMath.hpp>

#include <type_traits>

namespace pmacc::math
{
    namespace detail
    {
#if PMACC_DEVICE_COMPILE == 0
        using MathImplType = ::alpaka::math::internal::StlMath;
        inline constexpr auto mathImplInstance = ::alpaka::math::internal::stlMath;
#else
        using MathImplType = ::alpaka::math::internal::CudaHipMath;
        inline constexpr auto mathImplInstance = ::alpaka::math::internal::cudaHipMath;
#endif
    } // namespace detail

#define ALPAKA_UNARY_MATH_FN(functionName, alpakaMathTrait)                                                           \
    ALPAKA_NO_HOST_ACC_WARNING                                                                                        \
    template<typename T_Type>                                                                                         \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(T_Type const& arg)                                          \
    {                                                                                                                 \
        return ::alpaka::math::internal::alpakaMathTrait::Op<::pmacc::math::detail::MathImplType, T_Type>{}(          \
            ::pmacc::math::detail::mathImplInstance,                                                                  \
            arg);                                                                                                     \
    }

#define ALPAKA_BINARY_MATH_FN(functionName, alpakaMathTrait)                                                          \
    ALPAKA_NO_HOST_ACC_WARNING                                                                                        \
    template<typename T_Type1, typename T_Type2>                                                                      \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(T_Type1 const& arg1, T_Type2 const& arg2)                   \
    {                                                                                                                 \
        return ::alpaka::math::internal::alpakaMathTrait::                                                            \
            Op<::pmacc::math::detail::MathImplType, T_Type1, T_Type2>{}(                                              \
                ::pmacc::math::detail::mathImplInstance,                                                              \
                arg1,                                                                                                 \
                arg2);                                                                                                \
    }

} // namespace pmacc::math
