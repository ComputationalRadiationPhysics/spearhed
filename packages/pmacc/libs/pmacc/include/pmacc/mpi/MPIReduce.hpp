/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/mpi/GetMPI_StructAsArray.hpp"
#include "pmacc/mpi/reduceMethods/AllReduce.hpp"
#include "pmacc/types.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <caravan/core.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace pmacc
{
    namespace mpi
    {
        /** reduce data over selected mpi ranks */
        struct MPIReduce
        {
            MPIReduce() = default;

            virtual ~MPIReduce()
            {
                if(!caravanCommunicator)
                    return;
                try
                {
                    caravan::syncWait(
                        caravan::mpi::destroyCommunicator(*mpiContext, caravanCommunicator->communicator));
                }
                catch(std::exception const& error)
                {
                    std::cerr << "Failed to destroy Caravan reduction communicator: " << error.what() << '\n';
                }
                catch(...)
                {
                    std::cerr << "Failed to destroy Caravan reduction communicator\n";
                }
            }

            /* defines if the result of the MPI operation is valid
             *
             * @tparam MPIMethod type of the reduction method
             * @param method used reduction method e.g.,
             *                reduceMethods::AllReduce, reduceMethods::Reduce
             * @return if resut of operator() is valid*/
            template<class MPIMethod>
            bool hasResult(MPIMethod const& method)
            {
                if(!isMPICommInitialized)
                    participate(true);
                return method.hasResult(mpiRank);
            }

            /** defines if the result of the MPI operation is valid
             *
             * The reduction method reduceMethods::AllReduce is used.
             *
             * @return if result of operator() is valid
             */
            bool hasResult()
            {
                if(!isMPICommInitialized)
                    participate(true);
                return this->hasResult(::pmacc::mpi::reduceMethods::AllReduce());
            }

            /* Activate participation for reduce algorithm.
             * Must called from any mpi process. This function use global blocking mpi calls.
             * @param isActive true if mpi rank should be part of reduce operation, else false
             */
            void participate(bool isActive)
            {
                if(caravanCommunicator)
                    caravan::syncWait(
                        caravan::mpi::destroyCommunicator(*mpiContext, caravanCommunicator->communicator));

                mpiRank = -1;
                numRanks = 0;
                caravanCommunicator.reset();
                mpiContext = &Environment<>::get().getMpiContext();
                auto const world = mpiContext->topology();
                caravanCommunicator
                    = caravan::syncWait<std::optional<caravan::CommunicatorInfo>>(caravan::mpi::splitCommunicator(
                        *mpiContext,
                        isActive ? std::optional<int>{0} : std::nullopt,
                        world.rank));
                if(caravanCommunicator)
                {
                    mpiRank = caravanCommunicator->rank;
                    numRanks = caravanCommunicator->size;
                }
                isMPICommInitialized = true;
            }

            /* Reduce elements on cpu memory
             * call hasResult to see if returned value is valid
             *
             * @param func binary functor for reduce which takes two arguments, first argument is the source and get
             * the new reduced value. Functor must specialize the function getMPI_Op.
             * @param dest buffer for result data
             * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one
             * dimension access)
             * @param n number of elements to reduce
             * @param method mpi method for reduce
             *
             */
            template<class Functor, typename Type, class ReduceMethod>
            [[nodiscard]] HINLINE auto reduce(Functor, Type* dest, Type const* src, size_t const n, ReduceMethod const)
            {
                if(!isMPICommInitialized)
                    participate(true);
                using ValueType = Type;

                if(!caravanCommunicator)
                    throw std::logic_error("Inactive rank cannot submit an MPI reduction");
                return caravan::mpi::request<void>(
                    *mpiContext,
                    [=, communicator = caravanCommunicator->communicator](caravan::NativeMpiContext& context)
                    {
                        auto const descriptor = ::pmacc::mpi::getMPI_StructAsArray<ValueType>();
                        auto const elements = n * descriptor.sizeMultiplier;
                        caravan::NativeRequestBatch batch({MPI_REQUEST_NULL});
                        int error;
                        if constexpr(std::is_same_v<std::remove_cvref_t<ReduceMethod>, reduceMethods::AllReduce>)
                            error = MPI_Iallreduce(
                                src,
                                dest,
                                static_cast<int>(elements),
                                descriptor.dataType,
                                ::pmacc::mpi::getMPI_Op<Functor>(),
                                context.communicator(communicator),
                                &batch.requests[0]);
                        else
                            error = MPI_Ireduce(
                                src,
                                dest,
                                static_cast<int>(elements),
                                descriptor.dataType,
                                ::pmacc::mpi::getMPI_Op<Functor>(),
                                0,
                                context.communicator(communicator),
                                &batch.requests[0]);
                        if(error != MPI_SUCCESS)
                            throw std::runtime_error("PMacc native MPI reduction start failed");
                        return batch;
                    },
                    [](std::span<MPI_Status const>) {});
            }

            /** Lazily describe an all-reduce on caller-owned host storage. */
            template<class Functor, typename Type>
            [[nodiscard]] HINLINE auto reduce(Functor func, Type* dest, Type const* src, size_t const n)
            {
                return reduce(func, dest, src, n, ::pmacc::mpi::reduceMethods::AllReduce{});
            }

        private:
            std::optional<caravan::CommunicatorInfo> caravanCommunicator;
            caravan::MpiContext* mpiContext{nullptr};
            int mpiRank{-1};
            int numRanks{0};
            bool isMPICommInitialized{false};
        };
    } // namespace mpi
} // namespace pmacc
