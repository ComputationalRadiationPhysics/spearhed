#pragma once

#include "llamaLite/Field.hpp"

namespace llama_lite::traits
{

    template<IsField F>
    struct llama_traits;

    template<typename Tag, typename T>
    struct llama_traits<Field<Tag, T>>
    {
        using value_type = T;
        using tag_type = Tag;
        // Default: identity mapping
        using accessor_type = T;
        // Default: lvalue reference
        using reference_type = T&;
    };

    // Mapping from F to type, to represent the field type as type
    template<IsField F>
    struct AsType
    {
        using type = void;
        using _default_sentinel = void;

        constexpr type operator()(auto fieldView) const
        {
        }
    };

    // template<typename T>
    // using accessor_t = typename llama_traits<T>::accessor_type;

    template<typename T>
    using reference_t = typename llama_traits<T>::reference_type;

    template<template<typename> typename Trait, typename... T>
    struct IsTraitSpecialized
    {
        static constexpr bool value = !requires { typename Trait<T...>::_default_sentinel; };
    };

} // namespace llama_lite::traits
