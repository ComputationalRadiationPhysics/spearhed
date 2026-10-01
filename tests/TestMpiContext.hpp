/* Test-only access to the MPI runtime owned by TestMain.cpp. */
#pragma once

#include <cstdlib>

#include <caravan/mpi.hpp>

namespace spearhed::test
{
    inline caravan::MpiContext* activeMpiContext = nullptr;

    inline caravan::MpiContext& mpiContext()
    {
        if(activeMpiContext == nullptr)
            std::abort();
        return *activeMpiContext;
    }
} // namespace spearhed::test
