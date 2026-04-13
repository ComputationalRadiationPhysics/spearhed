
// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/Tuple.hpp"

namespace llama_lite::transform
{

    template<IsField F, template<typename> class LeafPolicy>
    struct GenericTransform;

    template<template<typename> class LeafPolicy, IsField... Fs>
    auto transform_types(Tuple<Fs...>) -> Tuple<typename GenericTransform<Fs, LeafPolicy>::type...>;

    template<typename Record, template<typename> class LeafPolicy>
    using transform_record_t
        = decltype(transform_types<LeafPolicy>(std::declval<typename Record::fields_tuple_type>()));

    // Nested record case
    template<IsField F, template<typename> class LeafPolicy>
    requires IsRecord<typename F::value_type>
    struct GenericTransform<F, LeafPolicy>
    {
        using type = transform_record_t<typename F::value_type, LeafPolicy>;
    };

    // Leaf field case
    template<IsField F, template<typename> class LeafPolicy>
    requires(!IsRecord<typename F::value_type>)
    struct GenericTransform<F, LeafPolicy>
    {
        using type = typename LeafPolicy<typename F::value_type>::type;
    };

} // namespace llama_lite::transform
