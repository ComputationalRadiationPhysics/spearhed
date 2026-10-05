/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <caravan/mpi.hpp>
#include <catch2/catch_session.hpp>

int main(int argc, char** argv)
{
    return caravan::MpiRuntime::run(
        argc,
        argv,
        [&](caravan::MpiContext& mpi)
        {
            auto const topology = mpi.topology();
            auto processes = pmacc::DataSpace<TEST_DIM>::create(1);
            processes.x() = topology.size;
            pmacc::Environment<TEST_DIM>::get().initDevices(mpi, processes, pmacc::DataSpace<TEST_DIM>::create(1));
            auto const result = Catch::Session().run(argc, argv);
            pmacc::Environment<>::get().finalize();
            return result;
        });
}
