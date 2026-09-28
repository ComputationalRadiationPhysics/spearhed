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

#include <caravan/mpi.hpp>

/** Run a spearHED simulation
 *
 * @param argc count of arguments in argv (same as for main() )
 * @param argv arguments of program start (same as for main() )
 */
int runSimulation(int argc, char** argv, caravan::MpiContext& mpiContext)
{
    struct EnvironmentFinalizer
    {
        ~EnvironmentFinalizer()
        {
            pmacc::Environment<>::get().finalize();
        }
    } finalizer;

    int errorCode = EXIT_FAILURE;

    // control the simulation lifetime
    {
        auto sim = spearhed::SimulationStarter{mpiContext};
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
        return caravan::MpiRuntime::run(
            argc,
            argv,
            [&](caravan::MpiContext& mpiContext) { return runSimulation(argc, argv, mpiContext); });
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
