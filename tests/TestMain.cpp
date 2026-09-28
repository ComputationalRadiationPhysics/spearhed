/* Copyright 2025-2026 Tapish Narwal
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "spearhed/param.hpp"

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
            struct EnvironmentFinalizer
            {
                ~EnvironmentFinalizer()
                {
                    pmacc::Environment<>::get().finalize();
                }
            } finalizer;

            auto const topology = mpi.topology();
            auto processes = pmacc::DataSpace<spearhed::simDim>::create(1);
            processes.x() = topology.size;
            pmacc::Environment<spearhed::simDim>::get().initDevices(
                mpi,
                processes,
                pmacc::DataSpace<spearhed::simDim>::create(0));
            return Catch::Session().run(argc, argv);
        });
}
