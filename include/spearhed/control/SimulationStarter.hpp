#pragma once

#include "spearhed/ArgsParser.hpp"
#include "spearhed/control/Simulation.hpp"

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

            po::options_description simDesc(simulationClass.pluginGetName());
            simulationClass.pluginRegisterHelp(simDesc);
            ap.addOptions(simDesc);

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
        }

        void pluginUnload() override
        {
            auto& pluginConnector = pmacc::Environment<>::get().PluginConnector();
            pluginConnector.unloadPlugins();
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
