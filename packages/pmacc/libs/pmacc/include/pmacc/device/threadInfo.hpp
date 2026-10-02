/* Copyright 2014-2024 Rene Widera
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

#include "pmacc/dimensions/DataSpace.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc::device
{
    /** Get the number of threads within a block
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getBlockSize(T_Acc const& acc)
    {
        auto alpakaBlockExtent = acc.getExtentsOf(::alpaka::onAcc::origin::block, ::alpaka::onAcc::unit::threads);
        constexpr uint32_t dim = std::decay_t<decltype(alpakaBlockExtent)>::dim();
        return DataSpace<dim>(alpakaBlockExtent);
    }

    /** Get the number of blocks within a grid
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getGridSize(T_Acc const& acc)
    {
        auto alpakaGridExtent = acc.getExtentsOf(::alpaka::onAcc::origin::grid, ::alpaka::onAcc::unit::blocks);
        constexpr uint32_t dim = std::decay_t<decltype(alpakaGridExtent)>::dim();
        return DataSpace<dim>(alpakaGridExtent);
    }

    /** Get the thread index within a block
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getThreadIdx(T_Acc const& acc)
    {
        auto alpakaThreadIdx = acc.getIdxWithin(::alpaka::onAcc::origin::block, ::alpaka::onAcc::unit::threads);
        constexpr uint32_t dim = std::decay_t<decltype(alpakaThreadIdx)>::dim();
        return DataSpace<dim>(alpakaThreadIdx);
    }

    /** Get the block index within a grid
     *
     * @param acc alpaka accelerator
     */
    template<typename T_Acc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getBlockIdx(T_Acc const& acc)
    {
        auto alpakaBlockdIdx = acc.getIdxWithin(::alpaka::onAcc::origin::grid, ::alpaka::onAcc::unit::blocks);
        constexpr uint32_t dim = std::decay_t<decltype(alpakaBlockdIdx)>::dim();
        return DataSpace<dim>(alpakaBlockdIdx);
    }
} // namespace pmacc::device
