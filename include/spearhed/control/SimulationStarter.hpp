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

#include "spearhed/ArgsParser.hpp"
#include "spearhed/control/Simulation.hpp"
#include "spearhed/plugins/PluginController.hpp"

#include <pmacc/debug/PMaccVerbose.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>

#include <boost/program_options/options_description.hpp>

#include <iostream>

namespace spearhed
{

    class SimulationStarter : public pmacc::IPlugin
    {
    private:
        using BoostOptionsList = std::list<boost::program_options::options_description>;
        Simulation simulationClass{};
        PluginController pluginClass{};

    public:
        SimulationStarter() = default;

        std::string pluginGetName() const override
        {
            return "SPH simulation starter";
        }

        void start()
        {
            pmacc::PluginConnector& pluginConnector = pmacc::Environment<>::get().PluginConnector();
            pluginConnector.loadPlugins();
            // pmacc::log<pmacc::PMaccVerbose::SIMULATION_STATE>("Startup");
            simulationClass.startSimulation();
        }

        void pluginRegisterHelp(boost::program_options::options_description&) override
        {
        }

        void notify(uint32_t) override
        {
        }

        ArgsParser::Status parseConfigs(int argc, char** argv)
        {
            namespace po = boost::program_options;

            ArgsParser& ap = ArgsParser::getInstance();
            pmacc::PluginConnector& pluginConnector = pmacc::Environment<>::get().PluginConnector();

            po::options_description simDesc(simulationClass.pluginGetName());
            simulationClass.pluginRegisterHelp(simDesc);
            ap.addOptions(simDesc);

            po::options_description pluginDesc(pluginClass.pluginGetName());
            pluginClass.pluginRegisterHelp(pluginDesc);
            ap.addOptions(pluginDesc);

            BoostOptionsList options = pluginConnector.registerHelp();
            for(auto iter = options.cbegin(); iter != options.cend(); ++iter)
            {
                ap.addOptions(*iter);
            }

            // parse environment variables, config files and command line
            return ap.parse(argc, argv);
        }

        void restart(uint32_t, std::string const) override
        {
            // nothing to do here
        }

        void checkpoint(uint32_t, std::string const) override
        {
            // nothing to do here
        }


    protected:
        void pluginLoad() override
        {
            simulationClass.load();
            pluginClass.load();
        }

        void pluginUnload() override
        {
            auto& pluginConnector = pmacc::Environment<>::get().PluginConnector();
            pluginConnector.unloadPlugins();
            pluginClass.unload();
            simulationClass.unload();
        }

    private:
        void printStartParameters(int argc, char** argv)
        {
            std::cout << "Start Parameters: ";
            for(int i = 0; i < argc; ++i)
            {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;
        }
    };
} // namespace spearhed
