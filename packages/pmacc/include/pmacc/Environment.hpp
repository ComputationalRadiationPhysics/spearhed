/* Copyright 2014-2024 Felix Schmitt, Conrad Schumann,
 *                     Alexander Grund, Axel Huebl
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
#include "pmacc/assert.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/device/MemoryInfo.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/mappings/simulation/SubGrid.hpp"
#include "pmacc/pluginSystem/PluginConnector.hpp"
#include "pmacc/simulationControl/SimulationDescription.hpp"

namespace pmacc
{
    namespace detail
    {
        /** PMacc environment
         *
         * Get access to all PMacc singleton classes those not depend on a dimension.
         */
        struct Environment
        {
            Environment() = default;

            /** cleanup the environment */
            void finalize()
            {
                EnvironmentContext::getInstance().finalize();
            }

            /** get the singleton DataConnector
             *
             * @return instance of DataConnector
             */
            HINLINE pmacc::DataConnector& DataConnector();

            /** get the singleton PluginConnector
             *
             * @return instance of PluginConnector
             */
            HINLINE pmacc::PluginConnector& PluginConnector();

            /** get the attached Caravan MPI context */
            HINLINE caravan::MpiContext& getMpiContext();

            /** Get the process-global accelerator queue pool. */
            HINLINE caravan::alpaka::SharedQueuePool<ComputeDeviceQueue>& DeviceContext();

            /** get the singleton MemoryInfo
             *
             * @return instance of MemoryInfo
             */
            HINLINE device::MemoryInfo& MemoryInfo();

            /** get the singleton SimulationDescription
             *
             * @return instance of SimulationDescription
             */
            HINLINE simulationControl::SimulationDescription& SimulationDescription();
        };
    } // namespace detail

    /** Global Environment singleton for PMacc
     */
    template<uint32_t T_dim>
    class Environment : public detail::Environment
    {
    public:
        HINLINE void enableMpiDirect();

        HINLINE bool isMpiDirectEnabled() const;

        /** get the singleton GridController
         *
         * @return instance of GridController
         */
        HINLINE pmacc::GridController<T_dim>& GridController();

        /** get the singleton SubGrid
         *
         * @return instance of SubGrid
         */
        HINLINE pmacc::SubGrid<T_dim>& SubGrid();

        /** get the singleton Environment< DIM >
         *
         * @return instance of Environment<DIM >
         */
        static Environment<T_dim>& get()
        {
            static Environment<T_dim> instance;
            return instance;
        }

        /** initialize PMacc under the dedicated Caravan MPI runtime */
        HINLINE void initDevices(caravan::MpiContext& mpiContext, DataSpace<T_dim> devices, DataSpace<T_dim> periodic);

        /** initialize the computing domain information of PMacc
         *
         * @param globalDomainSize size of the global simulation domain [cells]
         * @param localDomainSize size of the local simulation domain [cells]
         * @param localDomainOffset local domain offset [cells]
         */
        HINLINE void initGrids(
            DataSpace<T_dim> globalDomainSize,
            DataSpace<T_dim> localDomainSize,
            DataSpace<T_dim> localDomainOffset);

        Environment(Environment const&) = delete;

        Environment& operator=(Environment const&) = delete;

    private:
        Environment() = default;

        ~Environment() = default;
    };

} // namespace pmacc

#include "pmacc/Environment.tpp"
