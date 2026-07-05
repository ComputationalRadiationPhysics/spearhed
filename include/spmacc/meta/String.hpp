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
    /** A string literal usable as a non-type template parameter.
     *
     * Wraps a fixed-size character buffer (a C++20 structural type) so a string literal can be
     * passed *by value* as a template argument, e.g. `template<FixedString Name> struct Role`.
     *
     * This is the single compile-time-string building block for spearhed/spmacc. Use it directly
     * as the NTTP type when a name should be carried *into* a class (roles, species, coordinate
     * systems); use the @ref String wrapper below when a *distinct type per string* is required.
     */
    template<std::size_t N>
    struct FixedString
    {
        char data[N]{};

        constexpr FixedString(char const (&str)[N])
        {
            std::copy_n(str, N, data);
        }

        /** The string without its trailing null terminator. */
        [[nodiscard]] constexpr auto view() const noexcept -> std::string_view
        {
            return {data, N - 1};
        }

        /** Null-terminated pointer, for C APIs and std::string concatenation. */
        [[nodiscard]] constexpr auto c_str() const noexcept -> char const*
        {
            return data;
        }

        [[nodiscard]] constexpr auto size() const noexcept -> std::size_t
        {
            return N - 1;
        }

        constexpr operator std::string_view() const noexcept
        {
            return view();
        }

        /** Compile-time comparison against any other FixedString (enables name dispatch). */
        template<std::size_t M>
        [[nodiscard]] constexpr bool operator==(FixedString<M> const& rhs) const noexcept
        {
            return view() == rhs.view();
        }
    };

    /** A distinct empty type for each compile-time string.
     *
     * Use when a name must serve as a *type* -- e.g. a template argument that participates in a
     * type's identity such as a particle/frame name. `sizeof(String<...>) == 1`.
     *
     * @code{.cpp}
     * using Electrons = String<"electrons">;   // a C++ type usable as a template parameter
     * auto particleName = String<"electrons">{};
     * @endcode
     */
    template<FixedString Str>
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

        static consteval auto size() noexcept -> std::size_t
        {
            return Str.size();
        }
    };
} // namespace pmacc::spearhed::meta
