/* Copyright 2024-2024 Rene Widera
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

#include "pmacc/alpakaHelper/acc.hpp"

#include <alpaka/alpaka.hpp>

#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace pmacc
{
    namespace manager
    {
        template<typename T_DeviceType>
        struct Device
        {
            using DeviceType = T_DeviceType;
            std::optional<DeviceType> m_device;

            int m_devIdx = -1;

            static Device& get()
            {
                static Device instance;
                return instance;
            }

            auto getDeviceSelector() const
            {
                return []<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>(
                           std::type_identity<::alpaka::onHost::Device<T_Api, T_DeviceKind>>)
                {
                    return ::alpaka::onHost::makeDeviceSelector(T_Api{}, T_DeviceKind{});
                }(std::type_identity<DeviceType>{});
            }

            auto device(int idx = 0) -> DeviceType&
            {
                if(m_devIdx != -1)
                {
                    if(m_devIdx == idx)
                        return *m_device;
                    else
                        throw std::runtime_error(
                            std::string("Device with id '") + std::to_string(m_devIdx)
                            + "' is already selected, changing the device is not allowed.");
                }
                else
                {
                    int const numDevices = count();
                    if(idx >= numDevices)
                    {
                        std::stringstream err;
                        err << "Unable to return device " << idx << ". There are only " << numDevices << " devices!";
                        throw std::runtime_error(err.str());
                    }

                    std::optional<DeviceType> dev;

                    try
                    {
                        dev = std::make_optional<DeviceType>(getDeviceSelector().makeDevice(idx));
                    }
                    catch(std::runtime_error const& e)
                    {
                        throw std::runtime_error(e.what());
                    }
                    m_device = std::move(dev);
                    m_devIdx = idx;
                    return *m_device;
                }
            }

            /**! reset the current device
             *
             * streams, memory and events on the current device must be
             * deleted at first by the user else the call will throw an error
             */
            void reset()
            {
                if(!m_device)
                {
                    std::stringstream err;
                    err << "device " << m_devIdx << " can not destroyed (was never created) ";
                    throw std::runtime_error(err.str());
                }
                else
                {
                    m_devIdx = -1;
                    m_device.reset();
                }
            }

            auto id() -> int
            {
                return m_devIdx;
            }

            auto current() -> DeviceType&
            {
                assert(m_device.has_value());
                return *m_device;
            }

            auto count() -> int
            {
                return static_cast<int>(getDeviceSelector().getDeviceCount());
            }

        protected:
            Device()
            {
            }
        };

    } // namespace manager
} // namespace pmacc
