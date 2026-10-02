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

#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/types.hpp"

#include <memory>
#include <type_traits>

namespace pmacc
{
    /** Buffer that contains a host and device buffer and allows synchronizing those 2 */
    template<typename T_Type, unsigned T_dim>
    class HostDeviceBuffer
    {
        using HostBufferType = HostBuffer<T_Type, T_dim>;
        using DeviceBufferType = DeviceBuffer<T_Type, T_dim>;

    public:
        using ValueType = T_Type;
        using HBuffer = HostBuffer<T_Type, T_dim>;
        using DBuffer = DeviceBuffer<T_Type, T_dim>;
        using DataBoxType = typename HostBufferType::DataBoxType;
        PMACC_CASSERT_MSG(
            DataBoxTypes_must_match,
            std::is_same_v<DataBoxType, typename DeviceBufferType::DataBoxType>);

        /**
         * Constructor that creates the buffers with the given size
         *
         * @param size DataSpace representing buffer size
         * @param sizeOnDevice if true, allocate device-side size storage. Initialize it explicitly, for example by
         *        starting hostToDevice(), before device use.
         */
        HostDeviceBuffer(DataSpace<T_dim> const& size, bool sizeOnDevice = false);

        /**
         * Constructor that reuses the given device buffer instead of creating an own one.
         * Sizes should match. If size is smaller than the buffer size, then only the part near the origin is used.
         * Passing a size bigger than the buffer is undefined.
         */
        HostDeviceBuffer(DBuffer& otherDeviceBuffer, DataSpace<T_dim> const& size, bool sizeOnDevice = false);

        /**
         * Constructor that reuses the given buffers instead of creating own ones.
         * The data from [offset, offset+size) is used
         * Passing a size bigger than the buffer (minus the offset) is undefined.
         */
        HostDeviceBuffer(
            HBuffer& otherHostBuffer,
            DataSpace<T_dim> const& offsetHost,
            DBuffer& otherDeviceBuffer,
            DataSpace<T_dim> const& offsetDevice,
            GridLayout<T_dim> const size,
            bool sizeOnDevice = false);

        /**
         * Returns the internal data buffer on host side
         *
         * @return internal HBuffer
         */
        HINLINE HBuffer& getHostBuffer() const;

        /**
         * Returns the internal data buffer on device side
         *
         * @return internal DBuffer
         */
        HINLINE DBuffer& getDeviceBuffer() const;

        /** Return a lazy copy from the internal host buffer to the device buffer. */
        [[nodiscard]] HINLINE auto hostToDevice();

        /** Return a lazy copy from the internal device buffer to the host buffer. */
        [[nodiscard]] HINLINE auto deviceToHost();

    private:
        std::unique_ptr<HBuffer> hostBuffer;
        std::unique_ptr<DBuffer> deviceBuffer;
    };

} // namespace pmacc

#include "pmacc/memory/buffers/HostDeviceBuffer.tpp"
