/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/types.hpp"

#include <memory>

namespace pmacc
{
    class EventTask;

    template<class T_Type, unsigned T_dim>
    class HostBuffer;

    template<class T_Type, unsigned T_dim>
    class Buffer;

    /** N-dimensional device buffer
     *
     * @tparam T_Type datatype of the buffer
     * @tparam T_dim dimension of the buffer
     */
    template<class T_Type, unsigned T_dim>
    class DeviceBuffer : public Buffer<T_Type, T_dim>
    {
        using BufferType = decltype(alpaka::onHost::alloc<T_Type>(
            std::declval<ComputeDevice>(),
            std::declval<alpaka::Vec<MemIdxType, 1u>>()));
        using ViewType = decltype(alpaka::makeView(
            std::declval<ComputeDevice>(),
            std::declval<T_Type*>(),
            std::declval<alpaka::Vec<MemIdxType, T_dim>>(),
            std::declval<alpaka::Vec<MemIdxType, T_dim>>()));
        using CurrentSizeBufferDevice = decltype(alpaka::onHost::alloc<size_t>(
            std::declval<ComputeDevice>(),
            std::declval<alpaka::Vec<MemIdxType, 1u>>()));

        void createSizeOnDeviceBuffers()
        {
            currentSizeBufferDevice.emplace(
                alpaka::onHost::alloc<size_t>(
                    manager::Device<ComputeDevice>::get().current(),
                    MemSpace<DIM1>(1).toAlpakaMemVec()));
        }

    public:
        using DataBoxType = typename Buffer<T_Type, T_dim>::DataBoxType;
        std::optional<BufferType> devBuffer;
        std::optional<ViewType> view;
        std::optional<CurrentSizeBufferDevice> currentSizeBufferDevice;

        auto as1DBuffer()
        {
            auto numElements = this->size();
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return alpaka::makeView(
                *devBuffer,
                alpaka::onHost::data(*view),
                MemSpace<DIM1>(numElements).toAlpakaMemVec());
        }

        auto as1DBufferNElem(size_t const numElements)
        {
            PMACC_ASSERT(numElements < this->size());
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return alpaka::makeView(
                *devBuffer,
                alpaka::onHost::data(*view),
                MemSpace<DIM1>(numElements).toAlpakaMemVec());
        }

        ViewType getAlpakaView() const
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return *view;
        }

        /** allocate data accessible from the device
         *
         * @param size extent for each dimension (in elements)
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         *
         * @attention offset + size must be less or equal to the size of the source buffer
         */
        DeviceBuffer(MemSpace<T_dim> const& size, bool sizeOnDevice = false)
            : Buffer<T_Type, T_dim>(size)
            , devBuffer(
                  alpaka::onHost::alloc<T_Type>(
                      manager::Device<ComputeDevice>::get().current(),
                      MemSpace<DIM1>(size.productOfComponents()).toAlpakaMemVec()))
        {
            MemSpace<T_dim> pitchInBytes;
            pitchInBytes.x() = sizeof(T_Type);
            for(uint32_t d = 1u; d < T_dim; ++d)
                pitchInBytes[d] = pitchInBytes[d - 1u] * size[d - 1u];
            view.emplace(
                alpaka::makeView(
                    *devBuffer,
                    alpaka::onHost::data(*devBuffer),
                    size.toAlpakaMemVec(),
                    pitchInBytes.toAlpakaMemVec()));

            if(sizeOnDevice)
            {
                createSizeOnDeviceBuffers();
            }
            this->setSize(size.productOfComponents());
            this->isMemoryContiguous = true;
            reset(false);
        }

        /** create a shallow view into an existing buffer
         *
         * @param source buffer to create the view on
         * @param size extent for each dimension (in elements)
         * @param offset offset within the source (in elements)
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         *
         * @attention offset + size must be less or equal to the size of the source buffer
         */
        DeviceBuffer(
            DeviceBuffer<T_Type, T_dim>& source,
            MemSpace<T_dim> size,
            MemSpace<T_dim> offset,
            bool sizeOnDevice = false)
            : Buffer<T_Type, T_dim>(size)
            , devBuffer(source.devBuffer)
        {
            view.emplace(source.view->getSubView(offset.toAlpakaMemVec(), size.toAlpakaMemVec()));
            if(sizeOnDevice)
            {
                createSizeOnDeviceBuffers();
            }
            this->setSize(size.productOfComponents());
            this->isMemoryContiguous = T_dim == DIM1;
            reset(true);
        }

        ~DeviceBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            eventSystem::startOperation(ITask::TASK_HOST);
        }

        void reset(bool preserveData = true) override
        {
            this->setSize(Buffer<T_Type, T_dim>::capacityND().productOfComponents());

            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(!preserveData)
            {
                // Using Array is a workaround for types without default constructor
                memory::Array<uint8_t, sizeof(T_Type)> tmp(uint8_t{0});
                // use first element to avoid issue because Array is aligned (sizeof can be larger than component type)
                setValue(*reinterpret_cast<T_Type*>(tmp.data()));
            }
        }

        T_Type* data() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            return alpaka::onHost::data(*view);
        }

        DataBoxType getDataBox() override
        {
            auto pitchBytes = MemSpace<T_dim>(view->getPitches());
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return DataBoxType(PitchedBox<T_Type, T_dim>(alpaka::onHost::data(*view), pitchBytes));
        }

        /** Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        bool hasCurrentSizeOnDevice() const
        {
            return currentSizeBufferDevice.has_value();
        }

        /** get the device alpaka buffer with the current size
         *
         * @return device side current size buffer
         */
        CurrentSizeBufferDevice sizeOnDeviceBuffer()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(!hasCurrentSizeOnDevice())
            {
                throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
            }
            return currentSizeBufferDevice.value();
        }

        /** get the host alpaka buffer with the current size
         *
         * @return host side current size buffer
         */
        typename Buffer<T_Type, T_dim>::CurrentSizeBufferHost sizeHostSideBuffer()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return this->currentSizeBufferHost;
        }

        size_t size() override
        {
            if(hasCurrentSizeOnDevice())
            {
                eventSystem::startTransaction(eventSystem::getTransactionEvent());
                Environment<>::get().Factory().createTaskGetCurrentSizeFromDevice(*this);
                eventSystem::endTransaction().waitForFinished();
            }

            return Buffer<T_Type, T_dim>::size();
        }

        void setSize(size_t const newSize) override
        {
            Buffer<T_Type, T_dim>::setSize(newSize);

            if(hasCurrentSizeOnDevice())
            {
                Environment<>::get().Factory().createTaskSetCurrentSizeOnDevice(*this, newSize);
            }
        }

        /** Copies data from the given HostBuffer to this DeviceBuffer.
         *
         * @param other the HostBuffer to copy from
         */
        void copyFrom(HostBuffer<T_Type, T_dim>& other)
        {
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        /** Copies data from the given DeviceBuffer to this DeviceBuffer.
         *
         * @param other the DeviceBuffer to copy from
         */
        void copyFrom(DeviceBuffer<T_Type, T_dim>& other)
        {
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        void setValue(T_Type const& value) override
        {
            Environment<>::get().Factory().createTaskSetValue(*this, value);
        };

        auto sizeDeviceSideBuffer()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return currentSizeBufferDevice.value();
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCurrentSize() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            size_t const size = this->size();
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return {alpaka::onHost::data(*view), size};
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCapacity() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            eventSystem::startOperation(ITask::TASK_DEVICE);
            size_t const size = this->capacityND().productOfComponents();
            return {alpaka::onHost::data(*view), size};
        }
    };

    /** Factory for a new heap-allocated DeviceBuffer buffer object that is a deep copy of the given device
     * buffer
     *
     * @tparam T_Type value type
     * @tparam T_dim index dimensionality
     *
     * @param source source device buffer
     */
    template<class T_Type, unsigned T_dim>
    HINLINE std::unique_ptr<DeviceBuffer<T_Type, T_dim>> makeDeepCopy(DeviceBuffer<T_Type, T_dim>& source)
    {
        // We have to call this constructor to allocate a new data storage and not shallow-copy the source
        auto result = std::make_unique<DeviceBuffer<T_Type, T_dim>>(source.capacityND());
        result->copyFrom(source);
        // Wait for copy to finish, so that the resulting object is safe to use after return
        eventSystem::getTransactionEvent().waitForFinished();
        return result;
    }

} // namespace pmacc
