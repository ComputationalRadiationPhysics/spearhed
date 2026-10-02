/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>

#include <caravan/mpi/context.hpp>

namespace caravan
{
    struct SendResult
    {
        std::size_t bytes;
    };

    struct ReceiveResult
    {
        Peer source;
        MessageTag tag;
        std::size_t bytes;
    };

    struct AllReduceResult
    {
        std::size_t elements;
    };

    struct ReduceResult
    {
        std::size_t elements;
    };

    struct GatherResult
    {
        std::size_t bytes;
    };
} // namespace caravan
