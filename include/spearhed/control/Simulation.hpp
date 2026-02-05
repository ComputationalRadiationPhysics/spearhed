/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED, derived from PIConGPU.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/param/dimension.param"
#include "spearhed/param/mallocMC.param"

#include <pmacc/simulationControl/Checkpointing.hpp>
#include <pmacc/simulationControl/SimulationHelper.hpp>

#include <boost/program_options/options_description.hpp>

#include <memory>
#include <vector>

namespace spearhed
{

    class Simulation
        : public pmacc::SimulationHelper<
              simDim,
              pmacc::simulationControl::Checkpointing<pmacc::simulationControl::CheckpointingAvailability::DISABLED>>
    {
        using BaseType = pmacc::SimulationHelper<
            simDim,
            pmacc::simulationControl::Checkpointing<pmacc::simulationControl::CheckpointingAvailability::DISABLED>>;

    public:
        Simulation();
        ~Simulation() override;

        void pluginRegisterHelp(pmacc::po::options_description& desc) override;
        std::string pluginGetName() const override;
        void pluginLoad() override;
        void pluginUnload() override;
        void notify(uint32_t) override;
        void startSimulation() override;
        void runOneStep(uint32_t currentStep) override;
        void init() override;
        uint32_t fillSimulation() override;
        void resetAll(uint32_t currentStep) override;
        void movingWindowCheck(uint32_t currentStep) override;

    private:
        /** Get available memory on device
         *
         * @attention This method is using MPI collectives and must be called from all MPI processes collectively.
         *
         * The function is performing test memory allocations on the device therefore do not call this function within
         * a loop! This could slowdown the application.
         *
         * @return Available memory on device in bytes.
         */
        size_t freeDeviceMemory() const;

    private:
        std::shared_ptr<DeviceHeap> deviceHeap;

        // layout parameter
        std::vector<uint32_t> devices;
        std::vector<uint32_t> gridSize;
        /** Without guards */
        pmacc::DataSpace<simDim> gridSizeLocal;
        std::vector<uint32_t> periodic;

        bool showVersionOnce{false};
        bool autoAdjustGrid = true;
        uint32_t numRanksPerDevice = 1u;
        bool skipSimulation{false};
    };

} // namespace spearhed
