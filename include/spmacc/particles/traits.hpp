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

        constexpr void operator()(auto fieldView) const
        {
            *fieldView = typename F::value_type{0};
        }
    };

    template<ll::IsField F>
    struct InitDefault
    {
        using _default_sentinel = void;

        constexpr void operator()(auto fieldView) const
        {
            *fieldView = {};
        }
    };

    template<ll::IsField F>
    struct InitValue
    {
        using _default_sentinel = void;

        constexpr void operator()(auto fieldView, F::value_type val) const
        {
            *fieldView = val;
        }
    };

    template<ll::IsField F>
    struct InitGenerator
    {
        using _default_sentinel = void;

        // Accepts any callable (lambda, function object) that returns a compatible type
        constexpr void operator()(auto fieldView, std::invocable auto&& gen) const
            requires std::convertible_to<std::invoke_result_t<decltype(gen)>, typename F::value_type>
        {
            *fieldView = gen();
        }
    };

    template<ll::IsField F>
    struct InitRandom
    {
        using _default_sentinel = void;

        // Accepts any callable (lambda, function object) that returns a compatible type
        constexpr void operator()(auto fieldView, std::invocable auto&& gen) const
            requires std::convertible_to<std::invoke_result_t<decltype(gen)>, typename F::value_type>
        {
            *fieldView = gen();
        }
    };


} // namespace pmacc::spearhed
