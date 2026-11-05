/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc::particles::algorithm
{
    /** Distribute a frame's slot indices across physical threads, without storage or occupancy semantics.
     *
     * Calls body(slot) as an lvalue. No callback copies, allocations, or implicit barriers are introduced.
     * The callback must not synchronize the block: threads can receive different numbers of slots.
     * Slot ownership must not be used to index legacy lockstep::Variable storage.
     */
    template<uint32_t T_FrameSize>
    DINLINE void forEachFrameSlot(auto const& worker, auto&& body)
    {
        for(auto [slot] : alpaka::onAcc::makeIdxMap(
                worker.getAcc(),
                alpaka::onAcc::worker::linearThreadsInBlock,
                alpaka::IdxRange{alpaka::CVec<uint32_t, T_FrameSize>{}},
                alpaka::onAcc::traverse::Flat{},
                alpaka::onAcc::layout::Strided{}))
            body(slot);
    }
} // namespace pmacc::particles::algorithm
