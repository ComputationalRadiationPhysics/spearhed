/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/assert.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"
#include "pmacc/memory/buffers/size.hpp"
#include "pmacc/particles/memory/boxes/ExchangePopDataBox.hpp"
#include "pmacc/particles/memory/boxes/ExchangePushDataBox.hpp"

#include <utility>

#include <caravan/alpaka.hpp>

namespace pmacc
{
    /**
     * Can be used for creating several DataBox types from an Exchange.
     *
     * @tparam FRAME frame datatype
     */
    template<class FRAME, class FRAMEINDEX, unsigned DIM, unsigned T_CommDim = DIM1>
    class StackExchangeBuffer
    {
    public:
        /**
         * Create a stack from any ExchangeBuffer<FRAME,DIM>.
         *
         * If the stack's internal GridBuffer has no sizeOnDevice, no device querys are allowed.
         *
         * @param stack Exchange
         * @param stackIndexer Exchange for the index data
         */
        StackExchangeBuffer(
            Exchange<FRAME, DIM1, T_CommDim>& stack,
            Exchange<FRAMEINDEX, DIM1, T_CommDim>& stackIndexer)
            : stack(stack)
            , stackIndexer(stackIndexer)
        {
        }

        /**
         * Returns a PopDataBox for the internal HostBuffer.
         *
         * @return PopDataBox for host buffer
         */
        ExchangePopDataBox<vint_t, FRAME, DIM> getHostExchangePopDataBox()
        {
            return ExchangePopDataBox<vint_t, FRAME, DIM>(
                stack.getHostBuffer().getDataBox(),
                stackIndexer.getHostBuffer().getDataBox());
        }

        /**
         * Returns a PushDataBox for the internal DeviceBuffer.
         *
         * @return PushDataBox for device buffer
         */
        ExchangePushDataBox<vint_t, FRAME, DIM> getDeviceExchangePushDataBox()
        {
            PMACC_ASSERT(stack.getDeviceBuffer().hasCurrentSizeOnDevice() == true);
            PMACC_ASSERT(stackIndexer.getDeviceBuffer().hasCurrentSizeOnDevice() == true);
            return ExchangePushDataBox<vint_t, FRAME, DIM>(
                stack.getDeviceBuffer().data(),
                (vint_t*) alpaka::onHost::data(stack.getDeviceBuffer().sizeDeviceSideBuffer()),
                stack.getDeviceBuffer().capacityND().productOfComponents(),
                PushDataBox<vint_t, FRAMEINDEX>(
                    stackIndexer.getDeviceBuffer().data(),
                    (vint_t*) alpaka::onHost::data(stackIndexer.getDeviceBuffer().sizeDeviceSideBuffer())));
        }

        /**
         * Returns a PopDataBox for the internal DeviceBuffer.
         *
         * @return PopDataBox for device buffer
         */
        ExchangePopDataBox<vint_t, FRAME, DIM> getDeviceExchangePopDataBox()
        {
            return ExchangePopDataBox<vint_t, FRAME, DIM>(
                stack.getDeviceBuffer().getDataBox(),
                stackIndexer.getDeviceBuffer().getDataBox());
        }

        /** Reset host-side exchange metadata now; return a sender publishing the zero sizes. */
        [[nodiscard]] auto reset()
        {
            stack.getDeviceBuffer().setSizeHostSide(0u);
            stackIndexer.getDeviceBuffer().setSizeHostSide(0u);
            auto stackSize = pmacc::size(
                stack.getDeviceBuffer().sizeOnDeviceBuffer(),
                stack.getDeviceBuffer().sizeHostSideBuffer());
            auto indexSize = pmacc::size(
                stackIndexer.getDeviceBuffer().sizeOnDeviceBuffer(),
                stackIndexer.getDeviceBuffer().sizeHostSideBuffer());
            return std::move(stackSize) | caravan::sequence(std::move(indexSize));
        }

        /** Return a sender copying the device-side exchange sizes to the host. */
        [[nodiscard]] auto publishDeviceSizes()
        {
            auto stackSize = pmacc::size(
                stack.getDeviceBuffer().sizeHostSideBuffer(),
                stack.getDeviceBuffer().sizeOnDeviceBuffer());
            auto indexSize = pmacc::size(
                stackIndexer.getDeviceBuffer().sizeHostSideBuffer(),
                stackIndexer.getDeviceBuffer().sizeOnDeviceBuffer());
            return std::move(stackSize) | caravan::sequence(std::move(indexSize));
        }

        size_t getHostCurrentSize()
        {
            size_t result = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                result = stackIndexer.getDeviceBuffer().size();
            else
                result = stackIndexer.getHostBuffer().size();

            return result;
        }

        size_t getDeviceCurrentSize()
        {
            return stackIndexer.getDeviceBuffer().size();
        }

        size_t getDeviceParticlesCurrentSize()
        {
            return stack.getDeviceBuffer().size();
        }

        size_t getHostParticlesCurrentSize()
        {
            if(Environment<>::get().isMpiDirectEnabled())
                return stack.getDeviceBuffer().size();

            return stack.getHostBuffer().size();
        }

        size_t getMaxParticlesCount()
        {
            size_t result = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                result = stack.getDeviceBuffer().capacityND().productOfComponents();
            else
                result = stack.getHostBuffer().capacityND().productOfComponents();

            return result;
        }

    private:
        Exchange<FRAME, DIM1, T_CommDim>& getExchangeBuffer()
        {
            return stack;
        }

        Exchange<FRAME, DIM1, T_CommDim>& stack;
        Exchange<FRAMEINDEX, DIM1, T_CommDim>& stackIndexer;
    };
} // namespace pmacc
