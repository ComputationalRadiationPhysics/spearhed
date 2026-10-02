/* Copyright 2015-2024 Erik Zenker
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

/* #includes in "test/memoryUT.cu" */

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

namespace pmacc
{
    namespace test
    {
        namespace memory
        {
            namespace HostBuffer
            {
                /**
                 * Checks if data is copied correctly from device to
                 * host.
                 */
                struct CopyFromTest
                {
                    template<typename T_Dim>
                    void exec(T_Dim)
                    {
                        using Data = uint8_t;
                        using Extents = size_t;

                        using ::pmacc::test::memory::getElementsPerDim;

                        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_Dim>();
                        auto const device = manager::Device<ComputeDevice>::get().current();
                        ComputeDeviceQueue queue = caravan::alpaka::detail::makeQueue<ComputeDeviceQueue>(device);
                        caravan::ControlContext context;

                        for(unsigned i = 0; i < nElementsPerDim.size(); ++i)
                        {
                            auto const dataSpace = ::pmacc::DataSpace<T_Dim::value>::create(nElementsPerDim[i]);
                            ::pmacc::HostBuffer<Data, T_Dim::value> hostBuffer(dataSpace);
                            ::pmacc::DeviceBuffer<Data, T_Dim::value> deviceBuffer(dataSpace);

                            hostBuffer.reset();

                            auto fillBox = hostBuffer.getDataBox();
                            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i)
                            {
                                fillBox.getPointer()[i] = static_cast<Data>(i);
                            }

                            context.wait(context.spawn(
                                caravan::alpaka::copy(
                                    queue,
                                    deviceBuffer.getOwnedAlpakaView(),
                                    hostBuffer.getOwnedAlpakaView(),
                                    dataSpace.toAlpakaMemVec())));
                            hostBuffer.reset();
                            context.wait(context.spawn(
                                caravan::alpaka::copy(
                                    queue,
                                    hostBuffer.getOwnedAlpakaView(),
                                    deviceBuffer.getOwnedAlpakaView(),
                                    dataSpace.toAlpakaMemVec())));

                            auto compareBox = hostBuffer.getDataBox();
                            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i)
                            {
                                REQUIRE(compareBox.getPointer()[i] == static_cast<Data>(i));
                            }
                        }
                    }

                    PMACC_NO_NVCC_HDWARNING
                    template<typename T_Dim>
                    HDINLINE void operator()(T_Dim dim)
                    {
                        exec(dim);
                    }
                };

            } // namespace HostBuffer
        } // namespace memory
    } // namespace test
} // namespace pmacc

TEST_CASE("HostBuffer::copyFrom", "[copyFrom]")
{
    using namespace pmacc::test::memory::HostBuffer;
    ::pmacc::mp_for_each<Dims>(CopyFromTest());
}
