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

#include "pmacc/Environment.hpp"
#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/types.hpp"

#include <stdexcept>

#if !defined(ALPAKA_API_PREFIX)
/* ALPAKA_API_PREFIX was removed in alpaka 1.0.0 but is required to get access cuda/hip functions directly.
 * @todo find a better way to access native cuda/Hip functions or try to avoid accessing these at all.
 */
#    include "pmacc/ppFunctions.hpp"
#    if (ALPAKA_LANG_CUDA)
#        define ALPAKA_API_PREFIX(name) PMACC_JOIN(cuda, name)
#    elif (ALPAKA_LANG_HIP)
#        define ALPAKA_API_PREFIX(name) PMACC_JOIN(hip, name)
#    endif
#endif

namespace pmacc
{
    namespace detail
    {
        pmacc::DataConnector& Environment::DataConnector()
        {
            return DataConnector::getInstance();
        }

        pmacc::PluginConnector& Environment::PluginConnector()
        {
            return PluginConnector::getInstance();
        }

        caravan::MpiContext& Environment::getMpiContext()
        {
            return *EnvironmentContext::getInstance().m_mpiContext;
        }

        caravan::alpaka::SharedQueuePool<ComputeDeviceQueue>& Environment::DeviceContext()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().m_deviceContext,
                "Environment< DIM >::initDevices() must be called before this method!");
            return *EnvironmentContext::getInstance().m_deviceContext;
        }

        device::MemoryInfo& Environment::MemoryInfo()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return device::MemoryInfo::getInstance();
        }

        simulationControl::SimulationDescription& Environment::SimulationDescription()
        {
            return simulationControl::SimulationDescription::getInstance();
        }

    } // namespace detail

    template<uint32_t T_dim>
    void Environment<T_dim>::enableMpiDirect()
    {
        detail::EnvironmentContext::getInstance().enableMpiDirect();
    }

    template<uint32_t T_dim>
    bool Environment<T_dim>::isMpiDirectEnabled() const
    {
        return detail::EnvironmentContext::getInstance().isMpiDirectEnabled();
    }

    template<uint32_t T_dim>
    pmacc::GridController<T_dim>& Environment<T_dim>::GridController()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isMpiInitialized(),
            "Environment< DIM >::initDevices() must be called before this method!");
        return pmacc::GridController<T_dim>::getInstance();
    }

    template<uint32_t T_dim>
    pmacc::SubGrid<T_dim>& Environment<T_dim>::SubGrid()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isSubGridDefined(),
            "Environment< DIM >::initGrids() must be called before this method!");
        return pmacc::SubGrid<T_dim>::getInstance();
    }

    template<uint32_t T_dim>
    void Environment<T_dim>::initDevices(
        caravan::MpiContext& mpiContext,
        DataSpace<T_dim> devices,
        DataSpace<T_dim> periodic)
    {
        detail::EnvironmentContext::getInstance().init(mpiContext);
        GridController().init(mpiContext, devices, periodic);
        detail::EnvironmentContext::getInstance().setDevice(static_cast<int>(GridController().getHostRank()));
        MemoryInfo();
        SimulationDescription();
    }

    template<uint32_t T_dim>
    void Environment<T_dim>::initGrids(
        DataSpace<T_dim> globalDomainSize,
        DataSpace<T_dim> localDomainSize,
        DataSpace<T_dim> localDomainOffset)
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isMpiInitialized(),
            "Environment< DIM >::initDevices() must be called before this method!");

        detail::EnvironmentContext::getInstance().m_isSubGridDefined = true;

        // create singleton instances
        SubGrid().init(localDomainSize, globalDomainSize, localDomainOffset);

        DataConnector();

        PluginConnector();
    }

    namespace detail
    {
        void EnvironmentContext::init(caravan::MpiContext& mpiContext)
        {
            m_mpiContext = &mpiContext;
            m_isMpiInitialized = true;
        }

        void EnvironmentContext::finalize()
        {
            if(m_isMpiInitialized)
            {
                // Required by scorep for flushing the buffers. Application async contexts must already be joined.
                alpaka::onHost::wait(manager::Device<ComputeDevice>::get().current());
                m_deviceContext.reset();
                m_isDeviceSelected = false;
                m_isMpiInitialized = false;
                m_mpiContext = nullptr;
            }
        }

        void EnvironmentContext::setDevice(int deviceNumber)
        {
            int numAvailableDevices = manager::Device<ComputeDevice>::get().count();

#if (ALPAKA_LANG_CUDA || ALPAKA_COMP_HIP)
            // check if device is found
            if(numAvailableDevices < 1)
            {
                throw std::runtime_error("no capable alpaka compute devices detected");
            }
#endif

            int maxTries = numAvailableDevices;
            bool deviceSelectionSuccessful = false;

            // search the first selectable device in the compute node
            for(int deviceOffset = 0; deviceOffset < maxTries; ++deviceOffset)
            {
                // true if an error happened, else false
                bool errorOccured = false;

                /* Modulo 'numAvailableDevices' avoids invalid device indices for systems where the environment
                 * variable `CUDA_VISIBLE_DEVICES` is used to pre-select a device.
                 */
                int const tryDeviceId = (deviceOffset + deviceNumber) % numAvailableDevices;

                log<ggLog::CUDA_RT>("Trying to allocate device %1%.") % tryDeviceId;

#if (ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP)
#    if (ALPAKA_LANG_CUDA)
                int computeMode = 0;
                cudaError_t err = cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, tryDeviceId);
#    elif (ALPAKA_LANG_HIP)
                hipDeviceProp_t devProp;
                hipError_t err = hipGetDeviceProperties(&devProp, tryDeviceId);
                auto computeMode = devProp.computeMode;

#    endif

                if(err != ALPAKA_API_PREFIX(Success))
                    throw std::runtime_error("Error reading device properties.");

                /* If the cuda gpu compute mode is 'default'
                 * (https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-modes)
                 * then we try to get a device only once.
                 * The index used to select a device is based on the local MPI rank so
                 * that each rank tries a different device.
                 */
                if(computeMode == ALPAKA_API_PREFIX(ComputeModeDefault))
                {
                    maxTries = 1;
                    log<ggLog::CUDA_RT>("Device %1% is running in default mode.") % tryDeviceId;
                }
#endif

                try
                {
                    manager::Device<ComputeDevice>::get().device(tryDeviceId);
                }
                catch(std::system_error const& e)
                {
                    errorOccured = true;
                }

                if(!errorOccured)
                {
                    /* Create a dummy stream to check if the device is already used by another process. This could
                     * happen on NVIDIA devices. alpaka is performing the same check during the device selection but
                     * not for all device types. This is a safety check if alpaka is not performing this check.
                     */
                    try
                    {
                        auto testStream = manager::Device<ComputeDevice>::get().current().makeQueue();
                    }
                    catch(std::system_error const& e)
                    {
                        errorOccured = true;
                    }
                }

                if(!errorOccured)
                {
                    deviceSelectionSuccessful = true;

                    break;
                }
                else
                {
                    log<ggLog::CUDA_RT>("Device %1% already in use, try next.") % tryDeviceId;
                    continue;
                }
            }
            if(!deviceSelectionSuccessful)
            {
                std::cerr << "Failed to select one of the " << numAvailableDevices << " devices." << std::endl;
                throw std::runtime_error("Compute device selection failed.");
            }

            // initialize the default host device
            manager::Device<HostDevice>::get().device();
            m_isDeviceSelected = true;
            m_deviceContext = std::make_unique<caravan::alpaka::SharedQueuePool<ComputeDeviceQueue>>(
                manager::Device<ComputeDevice>::get().current(),
                1u);
        }

    } // namespace detail
} // namespace pmacc
