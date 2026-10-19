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

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    template<ll::IsField F>
    struct Init
    {
        using _default_sentinel = void;
    };

    template<ll::IsField F>
    struct InitZero
    {
        using _default_sentinel = void;

        // A leaf field drills to its element reference, so assign through it directly.
        constexpr void operator()(auto&& fieldView) const
        {
            fieldView = typename F::value_type{0};
        }
    };

    template<ll::IsField F>
    struct InitDefault
    {
        using _default_sentinel = void;

        constexpr void operator()(auto&& fieldView) const
        {
            fieldView = {};
        }
    };

    template<ll::IsField F>
    struct InitValue
    {
        using _default_sentinel = void;

        constexpr void operator()(auto&& fieldView, F::value_type val) const
        {
            fieldView = val;
        }
    };

    template<ll::IsField F>
    struct InitGenerator
    {
        using _default_sentinel = void;

        // Accepts any callable (lambda, function object) that returns a compatible type
        constexpr void operator()(auto&& fieldView, std::invocable auto&& gen) const
            requires std::convertible_to<std::invoke_result_t<decltype(gen)>, typename F::value_type>
        {
            fieldView = gen();
        }
    };

    template<ll::IsField F>
    struct InitRandom
    {
        using _default_sentinel = void;

        // Accepts any callable (lambda, function object) that returns a compatible type
        constexpr void operator()(auto&& fieldView, std::invocable auto&& gen) const
            requires std::convertible_to<std::invoke_result_t<decltype(gen)>, typename F::value_type>
        {
            fieldView = gen();
        }
    };


} // namespace pmacc::spearhed
