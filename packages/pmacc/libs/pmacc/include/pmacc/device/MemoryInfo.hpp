/* Copyright 2013-2024 Felix Schmitt, Rene Widera
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


#include "pmacc/Environment.def"
#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace device
    {
        /**
         * Provides convenience methods for querying memory information.
         * Singleton class.
         */
        class MemoryInfo
        {
        public:
            /**
             * Returns information about device memory.
             *
             * @param free amount of free memory in bytes. can be nullptr
             * @param total total amount of memory in bytes. can be nullptr. (nullptr by default)
             */
            void getMemoryInfo(size_t* free, size_t* total = nullptr) const
            {
                auto& device = manager::Device<ComputeDevice>::get().current();

                if(free != nullptr)
                {
                    size_t freeInternal = device.getFreeGlobalMemBytes();
                    if(reservedMem > freeInternal)
                        freeInternal = 0;
                    else
                        freeInternal -= reservedMem;

                    *free = freeInternal;
                }
                if(total != nullptr)
                {
                    size_t totalInternal = device.getDeviceProperties().globalMemCapacityBytes;
                    if(reservedMem > totalInternal)
                        totalInternal = 0;
                    else
                        totalInternal -= reservedMem;

                    *total = totalInternal;
                }
            }

            void setReservedMemory(size_t reservedMem)
            {
                this->reservedMem = reservedMem;
            }

        protected:
            size_t reservedMem{0};

        private:
            friend struct detail::Environment;

            static MemoryInfo& getInstance()
            {
                static MemoryInfo instance;
                return instance;
            }

            MemoryInfo() = default;
        };

    } // namespace device
} // namespace pmacc
