
#include <pmacc/boost_workaround.hpp>

#include "spearhed/ArgsParser.hpp"
#include "spearhed/control/SimulationStarter.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/types.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <typeinfo>

/** Run a spearHED simulation
 *
 * @param argc count of arguments in argv (same as for main() )
 * @param argv arguments of program start (same as for main() )
 */
int runSimulation(int argc, char** argv)
{
    int errorCode = EXIT_FAILURE;

    // control the simulation lifetime
    {
        auto sim = spearhed::SimulationStarter{};
        auto const parserStatus = sim.parseConfigs(argc, argv);

        switch(parserStatus)
        {
        case spearhed::ArgsParser::Status::error:
            errorCode = EXIT_FAILURE;
            break;
        case spearhed::ArgsParser::Status::success:
            sim.load();
            sim.start();
            sim.unload();
            [[fallthrough]];
        case spearhed::ArgsParser::Status::successExit:
            errorCode = 0;
            break;
        };
    }

    // finalize the pmacc context */
    pmacc::Environment<>::get().finalize();

    return errorCode;
}

/** Start of spearHED
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char** argv)
{
    try
    {
        return runSimulation(argc, argv);
    }
    // A last-ditch effort to report exceptions to a user
    catch(std::exception const& ex)
    {
        auto const typeName = std::string(typeid(ex).name());
        std::cerr << "Unhandled exception of type '" + typeName + "' with message '" + ex.what() + "', terminating\n";
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unhandled exception of unknown type, terminating\n";
        return EXIT_FAILURE;
    }
}
