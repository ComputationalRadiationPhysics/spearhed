/* Copyright 2013-2026 Axel Huebl, Felix Schmitt, Rene Widera, Alexander Debus,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov, Pawel Ordyna
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

#include "pmacc/simulationControl/SimulationHelper.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/particles/IdProvider.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/containsStep.hpp"
#include "pmacc/pluginSystem/toSlice.hpp"
#include "pmacc/simulationControl/Checkpointing.hpp"
#include "pmacc/simulationControl/signal.hpp"
#include "pmacc/types.hpp"

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace pmacc
{
    template<unsigned DIM, typename CheckpointingClass>
    SimulationHelper<DIM, CheckpointingClass>::SimulationHelper() : author("")
    {
        tSimulation.toggleStart();
        tInit.toggleStart();
    }

    template<unsigned DIM, typename CheckpointingClass>
    SimulationHelper<DIM, CheckpointingClass>::~SimulationHelper()
    {
        asyncContext.wait(signalCompletion);
        checkpointing.finishTimeBasedCheckpointing();
        tSimulation.toggleEnd();
        if(output)
        {
            std::cout << "full simulation time: " << tSimulation.printInterval() << " = " << std::fixed
                      << std::setprecision(3) << (tSimulation.getInterval() / 1000.)
                      << std::resetiosflags(std::ios::showbase) << " sec" << std::endl;
        }
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::notifyPlugins(uint32_t currentStep)
    {
        checkSignals(currentStep);
        Environment<DIM>::get().PluginConnector().notifyPlugins(currentStep);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::dumpOneStep(uint32_t currentStep)
    {
        checkSignals(currentStep);
        checkpointing.template dump<DIM>(currentStep, asyncContext);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::dumpTimes(
        TimeInterval& tSimCalculation,
        TimeInterval&,
        double& roundAvg,
        uint32_t currentStep)
    {
        /*dump 100% after simulation*/
        bool const writeAtPercent = progress && (currentStep % showProgressAnyStep) == 0;
        bool const writeAtStepPeriod
            = (progressStepPeriodEnabled && pluginSystem::containsStep(seqProgressPeriod, currentStep));
        if(output && (writeAtPercent || writeAtStepPeriod))
        {
            tSimCalculation.toggleEnd();
            uint32_t progressInterval{currentStep - lastProgressStep};
            // avoid division by 0 at 0% output
            if(progressInterval == 0u)
            {
                progressInterval = 1u;
            }
            std::cout << std::setw(3)
                      << uint16_t(
                             static_cast<double>(currentStep)
                             / static_cast<double>(Environment<>::get().SimulationDescription().getRunSteps()) * 100.)
                      << " % = " << std::setw(8) << currentStep << " | time elapsed:" << std::setw(25)
                      << tSimCalculation.printInterval() << " | avg time per step: "
                      << TimeInterval::printTime(roundAvg / static_cast<double>(progressInterval)) << std::endl;
            std::cout.flush();

            lastProgressStep = currentStep;
            roundAvg = 0.0; // clear round avg timer
        }
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::startSimulation()
    {
        if(useMpiDirect)
            Environment<>::get().enableMpiDirect();

        // Install a signal handler
        signal::activateSignalHandling();

        checkpointing.startTimeBasedCheckpointing();

        uint64_t maxRanks = Environment<DIM>::get().GridController().getGpuNodes().productOfComponents();
        uint64_t rank = Environment<DIM>::get().GridController().getScalarPosition();

        DataConnector& dc = Environment<>::get().DataConnector();
        auto idProvider = std::make_shared<IdProvider>("globalId", rank, maxRanks);
        dc.share(idProvider);
        asyncContext.wait(asyncContext.spawn(
            caravan::alpaka::withDevice(Environment<>::get().DeviceContext(), idProvider->initialize())));

        init();

        while(checkpointing.hasSoftRestartAttemptsLeft())
        {
            /* Global offset is updated during the simulation. In case we perform a soft restart we need to reset
             * the offset here to be valid for the next simulation run.
             */
            Environment<DIM>::get().SubGrid().setGlobalDomainOffset(DataSpace<DIM>::create(0));
            resetAll(0);
            uint32_t currentStep = fillSimulation();
            Environment<>::get().SimulationDescription().setCurrentStep(currentStep);

            /* Ensure all ranks finished the initialization.
             * This synchronization costs a little bit time but possible errors during the initialization will be
             * easier to hunt because the rank that outputs timings will only show the timing for the initialization if
             * all ranks reached this point.
             */
            auto& communicator = Environment<DIM>::get().GridController().getCommunicator();
            asyncContext.wait(asyncContext.spawn(communicator.barrier()));

            tInit.toggleEnd();
            if(output)
            {
                std::cout << "initialization time: " << tInit.printInterval() << " = " << std::fixed
                          << std::setprecision(3) << (tInit.getInterval() / 1000.)
                          << std::resetiosflags(std::ios::showbase) << " sec" << std::endl;
            }

            TimeInterval tSimCalculation;
            TimeInterval tRound;
            double roundAvg = 0.0;

            /* Since in the main loop movingWindow is called always before the dump, we also call it here for
             * consistency. This becomes only important, if movingWindowCheck does more than merely checking for a
             * slide. TO DO in a new feature: Turn this into a general hook for pre-checks (window slides are just
             * one possible action).
             */
            movingWindowCheck(currentStep);

            /* call plugins and dump initial step if simulation starts without restart */
            if(checkpointing.getRestartState() != simulationControl::RestartState::SUCCESS)
            {
                notifyPlugins(currentStep);
                dumpOneStep(currentStep);
            }

            /* dump 0% output */
            dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);


            /** \todo currently we assume this is the only point in the simulation
             *        that is allowed to manipulate `currentStep`. Else, one needs to
             *        add and act on changed values via
             *        `SimulationDescription().getCurrentStep()` in this loop
             */
            while(currentStep < Environment<>::get().SimulationDescription().getRunSteps())
            {
                tRound.toggleStart();
                runOneStep(currentStep);
                tRound.toggleEnd();
                roundAvg += tRound.getInterval();

                /* Next timestep starts here.
                 * Thus, for each timestep the plugins and checkpoint are called first.
                 * And the computational stages later on (on the next iteration of this loop).
                 */
                currentStep++;
                Environment<>::get().SimulationDescription().setCurrentStep(currentStep);
                /* output times after a round */
                dumpTimes(tSimCalculation, tRound, roundAvg, currentStep);

                movingWindowCheck(currentStep);
                /* call all plugins */
                notifyPlugins(currentStep);
                /* dump at the beginning of the simulated step */
                dumpOneStep(currentStep);
            }

            // The simulation is finished, wait until all MPI ranks finished the time step loop.
            auto barrier = asyncContext.spawn(communicator.barrier());
            asyncContext.wait(barrier, [&] { checkSignals(currentStep); });

            tSimCalculation.toggleEnd();

            if(output)
            {
                std::cout << "calculation  simulation time: " << tSimCalculation.printInterval() << " = " << std::fixed
                          << std::setprecision(3) << (tSimCalculation.getInterval() / 1000.)
                          << std::resetiosflags(std::ios::showbase) << " sec" << std::endl;
            }

        } // softRestarts loop
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::pluginRegisterHelp(po::options_description& desc)
    {
        // clang-format off
            desc.add_options()
                ("steps,s", po::value<uint32_t>(&runSteps), "Simulation steps")
                ("percent,p", po::value<uint16_t>(&progress)->default_value(5),
                 "Print time statistics after p percent to stdout")
                ("progressPeriod",po::value<std::string>(&progressPeriod),
                "write progress [for each n-th step], plugin period syntax can be used here.")
                ("author", po::value<std::string>(&author)->default_value(std::string("")),
                 "The author that runs the simulation and is responsible for created output files")
                ("mpiDirect", po::value<bool>(&useMpiDirect)->zero_tokens(),
                 "use device direct for MPI communication e.g. GPU direct");
        // clang-format on
        checkpointing.registerHelp(desc);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::pluginLoad()
    {
        Environment<>::get().SimulationDescription().setRunSteps(runSteps);
        Environment<>::get().SimulationDescription().setAuthor(author);

        calcProgress();
        progressStepPeriodEnabled = !progressPeriod.empty();
        if(progressStepPeriodEnabled)
            seqProgressPeriod = pluginSystem::toTimeSlice(progressPeriod);

        output = (getGridController().getGlobalRank() == 0);
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::checkSignals(uint32_t const currentStep)
    {
        asyncContext.runReady();
        if(signalCompletion.state() == caravan::CompletionState::pending)
            return;
        signalCompletion.wait();
        if(!signal::received())
            return;

        struct State
        {
            uint32_t processAtStep;
            uint32_t commonStep = 0u;
            std::array<uint32_t, 2u> send{signal::stopSimulation(), signal::createCheckpoint()};
            std::array<uint32_t, 2u> global{};
        };

        auto state = std::make_shared<State>(State{currentStep + 1u});
        if(output)
            std::cout << "SIGNAL: received." << std::endl;

        auto& communicator = Environment<DIM>::get().GridController().getCommunicator();
        auto reductions = caravan::whenAll(
            communicator.signalAllReduce(
                &state->processAtStep,
                &state->commonStep,
                sizeof(state->processAtStep),
                caravan::ScalarType::uint32,
                caravan::ReduceOperation::maximum),
            communicator.signalAllReduce(
                state->send.data(),
                state->global.data(),
                sizeof(state->send),
                caravan::ScalarType::uint32,
                caravan::ReduceOperation::sum));
        auto handle = asyncContext.onControl(std::move(reductions))
                      | caravan::then(
                          [this, state](caravan::AllReduceResult, caravan::AllReduceResult)
                          {
                              auto const ranks = Environment<DIM>::get().GridController().getCommunicator().getSize();
                              bool const shouldStop = state->global[0] == static_cast<uint32_t>(ranks);
                              bool const shouldCheckpoint = state->global[1] == static_cast<uint32_t>(ranks);
                              if(shouldCheckpoint)
                              {
                                  if(output)
                                      std::cout << "SIGNAL: Activate checkpointing for step " << state->commonStep
                                                << std::endl;
                                  checkpointing.addCheckpoint(state->commonStep);
                              }
                              if(shouldStop)
                              {
                                  if(output)
                                      std::cout << "SIGNAL: Shutdown simulation at step " << state->commonStep
                                                << std::endl;
                                  Environment<>::get().SimulationDescription().setRunSteps(state->commonStep);
                              }
                              signal::release(shouldCheckpoint, shouldStop);
                          });
        signalCompletion = asyncContext.spawn(std::move(handle));
    }

    template<unsigned DIM, typename CheckpointingClass>
    void SimulationHelper<DIM, CheckpointingClass>::calcProgress()
    {
        if(progress == 0 || progress > 100)
            progress = 100;

        showProgressAnyStep
            = uint32_t(double(Environment<>::get().SimulationDescription().getRunSteps()) / 100. * double(progress));
        if(showProgressAnyStep == 0)
            showProgressAnyStep = 1;
    }


    // Explicit template instantiation to provide symbols for usage together with PMacc
    template class SimulationHelper<
        DIM1,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::ENABLED>>;
    template class SimulationHelper<
        DIM2,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::ENABLED>>;
    template class SimulationHelper<
        DIM3,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::ENABLED>>;
    template class SimulationHelper<
        DIM1,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::DISABLED>>;
    template class SimulationHelper<
        DIM2,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::DISABLED>>;
    template class SimulationHelper<
        DIM3,
        simulationControl::Checkpointing<simulationControl::CheckpointingAvailability::DISABLED>>;
} // namespace pmacc
