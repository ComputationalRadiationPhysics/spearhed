/* Copyright 2018-2026 Rene Widera, Tapish Narwal
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

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace pmacc::spearhed::meta
{
    namespace internal
    {
        template<size_t N>
        struct FixedString
        {
            char data[N];

            constexpr FixedString(char const (&str)[N])
            {
                std::copy_n(str, N, data);
            }

            [[nodiscard]] constexpr std::string_view view() const noexcept
            {
                // Exclude null terminator
                return {data, N - 1};
            }

            [[nodiscard]] constexpr auto size() const noexcept -> size_t
            {
                return N - 1;
            }

            [[nodiscard]] constexpr auto c_str() const noexcept -> char const*
            {
                return data;
            }
        };
    } // namespace internal

    /** compile time string
     *
     * The size of the instance is 1 byte.
     */
    template<internal::FixedString Str>
    struct String
    {
        static consteval auto view() noexcept -> std::string_view
        {
            return Str.view();
        }

        static consteval auto c_str() noexcept -> char const*
        {
            return Str.c_str();
        }

        static consteval auto size() noexcept -> size_t
        {
            return Str.size();
        }
    };
} // namespace pmacc::spearhed::meta

/** create a compile time string type
 *
 * usage example:
 * @code{.cpp}
 * // create an instance of the compile time string
 * auto particleName = PMACC_CSTRING( "electrons" ){};
 * // create a C++ type (can be used as template parameter)
 * using Electrons = PMACC_CSTRING( "electrons" );
 * @endcode
 */
#define SPMACC_CSTRING(str) pmacc::spearhed::meta::String<str>
