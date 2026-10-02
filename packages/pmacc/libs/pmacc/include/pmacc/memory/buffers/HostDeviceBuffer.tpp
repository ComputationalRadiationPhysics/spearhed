/* Copyright 2016-2024 Alexander Grund
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

#include "HostDeviceBuffer.hpp"

#include <type_traits>
#include <utility>

#include <caravan/alpaka.hpp>

namespace pmacc
{
    namespace detail
    {
        template<uint32_t T_dim>
        MemSpace<T_dim> copyExtent(size_t size, MemSpace<T_dim> const& capacity)
        {
            MemSpace<T_dim> extent;
            if constexpr(T_dim == DIM1)
                extent.x() = size;
            else if constexpr(T_dim == DIM2)
            {
                extent.x() = size <= capacity.x() ? size : capacity.x();
                extent.y() = size <= capacity.x() ? 1u : (size + capacity.x() - 1u) / capacity.x();
            }
            else
            {
                extent.x() = size <= capacity.x() ? size : capacity.x();
                extent.y() = size <= capacity.x() ? 1u : capacity.y();
                extent.z() = size <= capacity.x() * capacity.y()
                                 ? 1u
                                 : (size + capacity.x() * capacity.y() - 1u) / (capacity.x() * capacity.y());
                if(size > capacity.x() && size <= capacity.x() * capacity.y())
                    extent.y() = (size + capacity.x() - 1u) / capacity.x();
            }
            return extent;
        }

        template<uint32_t T_dim, typename T_Destination, typename T_Source, typename T_UpdateSize>
        auto copyBuffer(T_Destination& destinationBuffer, T_Source& sourceBuffer, T_UpdateSize updateSize)
        {
            auto destination = destinationBuffer.getOwnedAlpakaView();
            auto source = sourceBuffer.getOwnedAlpakaView();
            auto destinationSize = destinationBuffer.getOwnedSizeHostBuffer();
            auto sourceSize = sourceBuffer.getOwnedSizeHostBuffer();
            auto const capacity = sourceBuffer.capacityND();
            bool const contiguous = destinationBuffer.isContiguous() && sourceBuffer.isContiguous();
            return caravan::alpaka::submit(
                [destination = std::move(destination),
                 source = std::move(source),
                 destinationSize = std::move(destinationSize),
                 sourceSize = std::move(sourceSize),
                 capacity,
                 contiguous,
                 updateSize = std::move(updateSize)](auto& nativeQueue) mutable
                {
                    auto const size = *::alpaka::onHost::data(sourceSize);
                    *::alpaka::onHost::data(destinationSize) = size;
                    updateSize(nativeQueue, destinationSize);
                    if(contiguous)
                    {
                        auto const extent = MemSpace<DIM1>{size}.toAlpakaMemVec();
                        auto destinationView
                            = ::alpaka::makeView(destination.value, ::alpaka::onHost::data(destination.value), extent);
                        auto sourceView
                            = ::alpaka::makeView(source.value, ::alpaka::onHost::data(source.value), extent);
                        ::alpaka::onHost::memcpy(nativeQueue, destinationView, sourceView, extent);
                    }
                    else
                    {
                        auto const extent = copyExtent<T_dim>(size, capacity).toAlpakaMemVec();
                        ::alpaka::onHost::memcpy(nativeQueue, destination.value, source.value, extent);
                    }
                });
        }
    } // namespace detail

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(DataSpace<T_dim> const& size, bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBuffer<T_Type, T_dim>>(size);
        deviceBuffer = std::make_unique<DeviceBuffer<T_Type, T_dim>>(size, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        DBuffer& otherDeviceBuffer,
        DataSpace<T_dim> const& size,
        bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBuffer<T_Type, T_dim>>(size);
        deviceBuffer = std::make_unique<DeviceBufferType>(otherDeviceBuffer, size, DataSpace<T_dim>(), sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        HBuffer& otherHostBuffer,
        DataSpace<T_dim> const& offsetHost,
        DBuffer& otherDeviceBuffer,
        DataSpace<T_dim> const& offsetDevice,
        GridLayout<T_dim> const size,
        bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBufferType>(otherHostBuffer, size, offsetHost);
        deviceBuffer = std::make_unique<DeviceBufferType>(otherDeviceBuffer, size, offsetDevice, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostBuffer<T_Type, T_dim>& HostDeviceBuffer<T_Type, T_dim>::getHostBuffer() const
    {
        return *hostBuffer;
    }

    template<typename T_Type, unsigned T_dim>
    DeviceBuffer<T_Type, T_dim>& HostDeviceBuffer<T_Type, T_dim>::getDeviceBuffer() const
    {
        return *deviceBuffer;
    }

    template<typename T_Type, unsigned T_dim>
    auto HostDeviceBuffer<T_Type, T_dim>::hostToDevice()
    {
        auto deviceSize = deviceBuffer->currentSizeBufferDevice;
        return detail::copyBuffer<T_dim>(
            *deviceBuffer,
            *hostBuffer,
            [deviceSize = std::move(deviceSize)](auto& nativeQueue, auto const& hostSize) mutable
            {
                if(deviceSize)
                    ::alpaka::onHost::memcpy(nativeQueue, *deviceSize, hostSize, MemSpace<DIM1>(1).toAlpakaMemVec());
            });
    }

    template<typename T_Type, unsigned T_dim>
    auto HostDeviceBuffer<T_Type, T_dim>::deviceToHost()
    {
        return detail::copyBuffer<T_dim>(*hostBuffer, *deviceBuffer, [](auto&, auto const&) {});
    }


} // namespace pmacc
