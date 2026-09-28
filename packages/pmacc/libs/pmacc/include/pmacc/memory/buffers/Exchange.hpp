/* Copyright 2013-2024 Rene Widera, Benjamin Worpitz
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
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <memory>
#include <optional>
#include <stdexcept>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

namespace pmacc
{
    /**
     * DIM-dimensional buffer used for data exchange.
     *
     * Exchange defines an interface for exchanging data between hosts.
     * Equally sized buffers are created on the device as well as on the host.
     * Exchange buffers may use parts of existing GridBuffer or
     * be used as dedicated memory.
     * @attention There will be no host double buffer available if MPI direct for PMacc is enabled.
     *
     * @tparam TYPE the datatype for internal buffers
     * @tparam DIM the dimension of the internal buffers
     * @tparam T_CommDim dimension of the communicator used for the exchange. Defaults to
     *         DIM; one-dimensional exchange buffers embedded in a higher-dimensional
     *         simulation must pass the simulation dimension here.
     */
    template<class TYPE, unsigned DIM, unsigned T_CommDim = DIM>
    class Exchange
    {
    public:
        Exchange(
            DeviceBuffer<TYPE, DIM>& source,
            GridLayout<DIM> memoryLayout,
            DataSpace<DIM> guardingCells,
            uint32_t extype,
            uint32_t commTag,
            uint32_t area = BORDER,
            bool sizeOnDevice = false)
            : hostBuffer(nullptr)
            , deviceDoubleBuffer(nullptr)
            , exchange(extype)
            , communicationTag(commTag)
        {
            DataSpace<DIM> tmp_size = memoryLayout.sizeWithoutGuardND();

            DataSpace<DIM> exchangeDimensions = exchangeTypeToDim(exchange);

            for(uint32_t dim = 0; dim < DIM; dim++)
            {
                if(DIM > dim && exchangeDimensions[dim] == 1)
                    tmp_size[dim] = guardingCells[dim];
            }

            /*This is only a pointer to other device data
             */
            using DeviceBuffer = DeviceBuffer<TYPE, DIM>;
            deviceBuffer = std::make_unique<DeviceBuffer>(
                source,
                tmp_size,
                exchangeTypeToOffset(exchange, memoryLayout, guardingCells, area),
                sizeOnDevice);
            if constexpr(DIM > DIM1)
            {
                /*create double buffer on gpu for faster memory transfers*/
                deviceDoubleBuffer = std::make_unique<DeviceBuffer>(tmp_size, false);
            }

            if(!Environment<>::get().isMpiDirectEnabled())
            {
                using HostBuffer = HostBuffer<TYPE, DIM>;
                hostBuffer = std::make_unique<HostBuffer>(tmp_size);
            }
        }

        Exchange(DataSpace<DIM> exchangeDataSpace, uint32_t extype, uint32_t commTag, bool sizeOnDevice = false)
            : hostBuffer(nullptr)
            , deviceDoubleBuffer(nullptr)
            , exchange(extype)
            , communicationTag(commTag)
        {
            using DeviceBuffer = DeviceBuffer<TYPE, DIM>;
            deviceBuffer = std::make_unique<DeviceBuffer>(exchangeDataSpace, sizeOnDevice);

            if constexpr(DIM > DIM1)
            {
                /*create double buffer on gpu for faster memory transfers*/
                deviceDoubleBuffer = std::make_unique<DeviceBuffer>(exchangeDataSpace, false);
            }

            if(!Environment<>::get().isMpiDirectEnabled())
            {
                using HostBuffer = HostBuffer<TYPE, DIM>;
                hostBuffer = std::make_unique<HostBuffer>(exchangeDataSpace);
            }
        }

        /**
         * specifies in returned DataSpace which dimensions exchange data
         * @param exchange the exchange mask
         * @return DIM1 DataSpace of size 3 where 1 means exchange, 0 means no exchange
         */
        DataSpace<DIM> exchangeTypeToDim(uint32_t exchange) const
        {
            DataSpace<DIM> result;

            Mask exchangeMask(exchange);

            if(exchangeMask.containsExchangeType(LEFT) || exchangeMask.containsExchangeType(RIGHT))
                result[0] = 1;

            if constexpr(DIM > DIM1)
                if(exchangeMask.containsExchangeType(TOP) || exchangeMask.containsExchangeType(BOTTOM))
                    result[1] = 1;

            if constexpr(DIM > DIM2)
                if(exchangeMask.containsExchangeType(FRONT) || exchangeMask.containsExchangeType(BACK))
                    result[2] = 1;

            return result;
        }

        virtual ~Exchange() = default;

        DataSpace<DIM> exchangeTypeToOffset(
            uint32_t exchange,
            GridLayout<DIM>& memoryLayout,
            DataSpace<DIM> guardingCells,
            uint32_t area) const
        {
            DataSpace<DIM> size = memoryLayout.sizeND();
            DataSpace<DIM> border = memoryLayout.guardSizeND();
            Mask mask(exchange);
            DataSpace<DIM> tmp_offset;
            if constexpr(DIM >= DIM1)
            {
                if(mask.containsExchangeType(RIGHT))
                {
                    tmp_offset[0] = size[0] - border[0] - guardingCells[0];
                    if(area == GUARD)
                    {
                        tmp_offset[0] += guardingCells[0];
                    }
                }
                else
                {
                    tmp_offset[0] = border[0];
                    if(area == GUARD && mask.containsExchangeType(LEFT))
                    {
                        tmp_offset[0] -= guardingCells[0];
                    }
                }
            }
            if constexpr(DIM >= DIM2)
            {
                if(mask.containsExchangeType(BOTTOM))
                {
                    tmp_offset[1] = size[1] - border[1] - guardingCells[1];
                    if(area == GUARD)
                    {
                        tmp_offset[1] += guardingCells[1];
                    }
                }
                else
                {
                    tmp_offset[1] = border[1];
                    if(area == GUARD && mask.containsExchangeType(TOP))
                    {
                        tmp_offset[1] -= guardingCells[1];
                    }
                }
            }
            if constexpr(DIM == DIM3)
            {
                if(mask.containsExchangeType(BACK))
                {
                    tmp_offset[2] = size[2] - border[2] - guardingCells[2];
                    if(area == GUARD)
                    {
                        tmp_offset[2] += guardingCells[2];
                    }
                }
                else /*all other begin from front*/
                {
                    tmp_offset[2] = border[2];
                    if(area == GUARD && mask.containsExchangeType(FRONT))
                    {
                        tmp_offset[2] -= guardingCells[2];
                    }
                }
            }

            return tmp_offset;
        }

        /**
         * Returns the exchange buffer on the host.
         *
         * @return Exchange buffer on host
         */
        HostBuffer<TYPE, DIM>& getHostBuffer()
        {
            PMACC_ASSERT(hostBuffer != nullptr);
            return *hostBuffer;
        }

        /**
         * Returns the exchange buffer on the device.
         *
         * @return Exchange buffer on device
         */
        DeviceBuffer<TYPE, DIM>& getDeviceBuffer()
        {
            PMACC_ASSERT(deviceBuffer != nullptr);
            return *deviceBuffer;
        }

        bool hasDeviceDoubleBuffer()
        {
            return deviceDoubleBuffer != nullptr;
        }

        DeviceBuffer<TYPE, DIM>& getDeviceDoubleBuffer()
        {
            PMACC_ASSERT(deviceDoubleBuffer != nullptr);
            return *deviceDoubleBuffer;
        }

        struct ReceiveMetadata
        {
            size_t const elements;
            caravan::ReceiveResult const mpi;
        };

        /** Describe one lazy send. The exchange and borrowed buffers must outlive it. */
        [[nodiscard]] auto send()
        {
            auto& communicator = Environment<T_CommDim>::get().GridController().getCommunicator();
            auto source = getDeviceBuffer().getOwnedAlpakaView();
            std::optional<decltype(source)> deviceStaging;
            if(hasDeviceDoubleBuffer())
                deviceStaging.emplace(getDeviceDoubleBuffer().getOwnedAlpakaView());
            using HostView = decltype(getHostBuffer().getOwnedAlpakaView());
            std::optional<HostView> hostStaging;
            if(!Environment<>::get().isMpiDirectEnabled())
                hostStaging.emplace(getHostBuffer().getOwnedAlpakaView());

            auto queueTail = caravan::alpaka::submit(
                [this,
                 source = std::move(source),
                 deviceStaging = std::move(deviceStaging),
                 hostStaging = std::move(hostStaging)](auto& nativeQueue) mutable
                {
                    auto const elements = getDeviceBuffer().size();
                    auto const extent = getDeviceBuffer().sizeND(elements).toAlpakaMemVec();
                    if(deviceStaging)
                        getDeviceDoubleBuffer().setSizeHostSide(elements);
                    if(hostStaging)
                        getHostBuffer().setSizeHostSide(elements);
                    if(deviceStaging)
                        ::alpaka::onHost::memcpy(nativeQueue, deviceStaging->value, source.value, extent);
                    if(hostStaging)
                    {
                        if(deviceStaging)
                            copyStaging(nativeQueue, hostStaging->value, deviceStaging->value, elements);
                        else
                            ::alpaka::onHost::memcpy(nativeQueue, hostStaging->value, source.value, extent);
                    }
                });
            return std::move(queueTail)
                   | caravan::letValue(
                       [this, &communicator]
                       {
                           auto const buffer = getCPtrCurrentSize();
                           return communicator
                               .send(exchange, buffer.asCharPtr(), buffer.sizeInBytes(), communicationTag);
                       });
        }

        /** Describe one lazy receive followed by size publication and device copies. */
        [[nodiscard]] auto receive()
        {
            auto& communicator = Environment<T_CommDim>::get().GridController().getCommunicator();
            auto destination = getDeviceBuffer().getOwnedAlpakaView();
            std::optional<decltype(destination)> deviceStaging;
            if(hasDeviceDoubleBuffer())
                deviceStaging.emplace(getDeviceDoubleBuffer().getOwnedAlpakaView());
            using HostView = decltype(getHostBuffer().getOwnedAlpakaView());
            std::optional<HostView> hostStaging;
            if(!Environment<>::get().isMpiDirectEnabled())
                hostStaging.emplace(getHostBuffer().getOwnedAlpakaView());

            auto const buffer = getCPtrCapacity();
            auto receive = communicator.receive(exchange, buffer.asCharPtr(), buffer.sizeInBytes(), communicationTag);
            return std::move(receive)
                   | caravan::letValue(
                       [this,
                        destination = std::move(destination),
                        deviceStaging = std::move(deviceStaging),
                        hostStaging = std::move(hostStaging)](caravan::ReceiveResult result) mutable
                       {
                           auto const metadata = receiveMetadata(result);
                           getDeviceBuffer().setSizeHostSide(metadata.elements);
                           if(deviceStaging)
                               getDeviceDoubleBuffer().setSizeHostSide(metadata.elements);
                           if(hostStaging)
                               getHostBuffer().setSizeHostSide(metadata.elements);

                           auto deviceSize = getDeviceBuffer().currentSizeBufferDevice;
                           auto hostSize = getDeviceBuffer().sizeHostSideBuffer();
                           auto const sizeExtent = MemSpace<DIM1>(1).toAlpakaMemVec();
                           auto const dataExtent = getDeviceBuffer().sizeND(metadata.elements).toAlpakaMemVec();
                           auto copy = caravan::alpaka::submit(
                               [destination = std::move(destination),
                                deviceStaging = std::move(deviceStaging),
                                hostStaging = std::move(hostStaging),
                                deviceSize = std::move(deviceSize),
                                hostSize = std::move(hostSize),
                                sizeExtent,
                                dataExtent,
                                elements = metadata.elements](auto& nativeQueue) mutable
                               {
                                   if(deviceSize)
                                       ::alpaka::onHost::memcpy(nativeQueue, *deviceSize, hostSize, sizeExtent);
                                   if(hostStaging)
                                   {
                                       if(deviceStaging)
                                       {
                                           copyStaging(
                                               nativeQueue,
                                               deviceStaging->value,
                                               hostStaging->value,
                                               elements);
                                           ::alpaka::onHost::memcpy(
                                               nativeQueue,
                                               destination.value,
                                               deviceStaging->value,
                                               dataExtent);
                                       }
                                       else
                                           ::alpaka::onHost::memcpy(
                                               nativeQueue,
                                               destination.value,
                                               hostStaging->value,
                                               dataExtent);
                                   }
                                   else if(deviceStaging)
                                       ::alpaka::onHost::memcpy(
                                           nativeQueue,
                                           destination.value,
                                           deviceStaging->value,
                                           dataExtent);
                               });
                           return std::move(copy) | caravan::then([metadata] { return metadata; });
                       });
        }

        /**
         * Returns the type describing exchange directions
         *
         * @return a value describing exchange directions
         */
        uint32_t getExchangeType() const
        {
            return exchange;
        }

        /**
         * Returns the value used for tagging ('naming') communicated messages
         *
         * @return the communication tag
         */
        uint32_t getCommunicationTag() const
        {
            return communicationTag;
        }

        /**
         * Return the buffer which can be used for data exchange with MPI
         *
         * The buffer can point to device or host memory.
         */
        typename Buffer<TYPE, DIM>::CPtr getCPtrCapacity()
        {
            if(Environment<>::get().isMpiDirectEnabled())
            {
                if(hasDeviceDoubleBuffer())
                    return getDeviceDoubleBuffer().getCPtrCapacity();
                else
                    return getDeviceBuffer().getCPtrCapacity();
            }

            return getHostBuffer().getCPtrCapacity();
        }

        typename Buffer<TYPE, DIM>::CPtr getCPtrCurrentSize()
        {
            if(Environment<>::get().isMpiDirectEnabled())
            {
                if(hasDeviceDoubleBuffer())
                    return getDeviceDoubleBuffer().getCPtrCurrentSize();
                else
                    return getDeviceBuffer().getCPtrCurrentSize();
            }

            return getHostBuffer().getCPtrCurrentSize();
        }

    private:
        /** Copy between the contiguous staging allocations, not the potentially strided grid views.
         * Flattening avoids a multidimensional host/device transfer with many tiny rows for thin halos.
         * The caller retains the allocations until the submitted queue work completes.
         */
        template<typename T_Queue, typename T_Destination, typename T_Source>
        static void copyStaging(T_Queue& queue, T_Destination& destination, T_Source const& source, size_t elements)
        {
            auto const extent = MemSpace<DIM1>(elements).toAlpakaMemVec();
            auto destinationView = ::alpaka::makeView(destination, ::alpaka::onHost::data(destination), extent);
            auto sourceView = ::alpaka::makeView(source, ::alpaka::onHost::data(source), extent);
            ::alpaka::onHost::memcpy(queue, destinationView, sourceView, extent);
        }

        ReceiveMetadata receiveMetadata(caravan::ReceiveResult const& result) const
        {
            if(result.bytes % sizeof(TYPE) != 0u)
                throw std::runtime_error("Received exchange byte count is not an element count");
            auto const elements = result.bytes / sizeof(TYPE);
            if(elements > static_cast<size_t>(deviceBuffer->capacityND().productOfComponents()))
                throw std::runtime_error("Received exchange exceeds its device buffer");
            return {elements, result};
        }

    protected:
        /** host double buffer of the exchange data
         *
         * Is always a nullptr if MPI direct is used
         */
        std::unique_ptr<HostBuffer<TYPE, DIM>> hostBuffer;

        //! This buffer is a vector which is used as message buffer for faster memcopy
        std::unique_ptr<DeviceBuffer<TYPE, DIM>> deviceDoubleBuffer;
        std::unique_ptr<DeviceBuffer<TYPE, DIM>> deviceBuffer;

        uint32_t exchange = std::numeric_limits<uint32_t>::max();
        uint32_t communicationTag = std::numeric_limits<uint32_t>::max();
    };

} // namespace pmacc
