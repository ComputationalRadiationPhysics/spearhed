/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include <caravan/mpi/context.hpp>

namespace caravan
{
    /** Convenience MPI lifecycle/driver using a dedicated FUNNELED thread. */
    class MpiRuntime
    {
    public:
        template<typename T_Application>
        static int run(int& argc, char**& argv, T_Application&& application)
        {
            auto invoke = [&application](MpiContext& context)
            {
                if constexpr(std::is_invocable_v<T_Application&, MpiContext&>)
                {
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&, MpiContext&>>)
                    {
                        std::invoke(application, context);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application, context));
                }
                else
                {
                    static_assert(std::is_invocable_v<T_Application&>);
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&>>)
                    {
                        std::invoke(application);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application));
                }
            };
            return runImpl(argc, argv, invoke);
        }

    private:
        static int runImpl(int& argc, char**& argv, std::function<int(MpiContext&)> application);
    };
} // namespace caravan
