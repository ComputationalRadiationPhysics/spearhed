/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <exception>
#include <future>
#include <thread>

#include <caravan/mpi/runtime.hpp>
#include <mpi.h>

namespace caravan
{
    int MpiRuntime::runImpl(int& argc, char**& argv, std::function<int(MpiContext&)> application)
    {
        std::promise<MpiContext*> startup;
        auto ready = startup.get_future();
        std::jthread mpiWorker(
            [&]() noexcept
            {
                int provided = MPI_THREAD_SINGLE;
                if(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
                    std::terminate();
                if(provided < MPI_THREAD_FUNNELED)
                    std::terminate();
                if(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) != MPI_SUCCESS)
                    std::terminate();

                MpiContext context;
                startup.set_value(&context);
                context.run();

                if(MPI_Finalize() != MPI_SUCCESS)
                    std::terminate();
            });

        MpiContext* context = ready.get();

        // The application is a synchronous top-level boundary, not asynchronous work: keep its exception,
        // shut the worker down cleanly, then rethrow to the process entry point.
        int applicationResult = 0;
        std::exception_ptr applicationError;
        try
        {
            applicationResult = application(*context);
        }
        catch(...)
        {
            applicationError = std::current_exception();
        }
        context->requestShutdown();
        mpiWorker.join();

        if(applicationError)
            std::rethrow_exception(applicationError);
        return applicationResult;
    }
} // namespace caravan
