/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <cstddef>
#include <utility>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

namespace pmacc
{
    /*! communication via MPI
     */
    template<unsigned DIM>
    class CommunicatorMPI
    {
    public:
        CommunicatorMPI() = default;

        int getRank()
        {
            return mpiRank;
        }

        int getSize()
        {
            return mpiSize;
        }

        DataSpace<DIM3> getPeriodic() const
        {
            return this->periodic;
        }

        /*! initializes all processes to build a 3D-grid
         *
         * @param nodes number of GPU nodes in each dimension
         * @param periodic specifying whether the grid is periodic (1) or not (0) in each dimension
         *
         * \warning throws invalid argument if cx*cy*cz != totalnodes
         */
        /** initialize from MPI-thread-owned immutable topology data */
        void init(caravan::MpiContext& mpiContext, DataSpace<DIM3> numberProcesses, DataSpace<DIM3> periodic);

        caravan::CommunicatorId getCommunicatorId() const
        {
            return communicatorId;
        }

        caravan::CommunicatorId getSignalCommunicatorId() const
        {
            return signalCommunicatorId;
        }

        /*! returns a rank number (0-n) for each host
         *
         * E.g. if 8 GPUs are on 2 Hosts (4 GPUs each), the GPUs on each host will get hostrank 0 to 3
         *
         */
        uint32_t getHostRank()
        {
            return hostRank;
        }

        Mask const& getCommunicationMask() const
        {
            return communicationMask;
        }

        /*! returns coordinate of this process in (via init) created grid
         *
         * Coordinates are between [0-cx, 0-cy, 0-cz]
         *
         */
        DataSpace<DIM> const getCoordinates() const
        {
            return this->coordinates;
        }

        [[nodiscard]] auto send(uint32_t ex, char const* sendData, size_t sendBytes, uint32_t tag)
        {
            return caravan::mpi::send(
                *mpiContext,
                std::span<std::byte const>{reinterpret_cast<std::byte const*>(sendData), sendBytes},
                caravan::Peer{ExchangeTypeToRank(ex)},
                caravan::MessageTag{static_cast<int>(gridExchangeTag + tag)},
                communicatorId);
        }

        [[nodiscard]] auto receive(uint32_t ex, char* receiveData, size_t receiveBytes, uint32_t tag)
        {
            return caravan::mpi::receive(
                *mpiContext,
                std::span<std::byte>{reinterpret_cast<std::byte*>(receiveData), receiveBytes},
                caravan::Peer{ExchangeTypeToRank(ex)},
                caravan::MessageTag{static_cast<int>(gridExchangeTag + tag)},
                communicatorId);
        }

        auto signalAllReduce(
            void const* input,
            void* output,
            size_t bytes,
            caravan::ScalarType type,
            caravan::ReduceOperation operation)
        {
            return caravan::mpi::allReduce(
                *mpiContext,
                std::span<std::byte const>{static_cast<std::byte const*>(input), bytes},
                std::span<std::byte>{static_cast<std::byte*>(output), bytes},
                type,
                operation,
                signalCommunicatorId);
        }

        auto barrier()
        {
            return caravan::mpi::barrier(*mpiContext, communicatorId);
        }

        bool slide();

        bool setStateAfterSlides(size_t numSlides);

        /*! converts an exchangeType (e.g. RIGHT) to an MPI-rank
         */
        int ExchangeTypeToRank(uint32_t type)
        {
            return ranks[type];
        }


    protected:
        /*! update coordinates @see getCoordinates
         */
        void updateCoordinates();

    private:
        //! coordinates in GPU-Grid [0:cx-1,0:cy-1,0:cz-1]
        DataSpace<DIM> coordinates;
        DataSpace<DIM> baseCoordinates;

        DataSpace<DIM3> periodic;
        //! Opaque communicators owned by the Caravan MPI thread.
        caravan::CommunicatorId communicatorId{caravan::worldCommunicator};
        caravan::CommunicatorId signalCommunicatorId{caravan::worldCommunicator};
        caravan::MpiContext* mpiContext{nullptr};
        //! array for exchangetype-to-rank conversion @see ExchangeTypeToRank
        int ranks[27];
        //! size of pmacc [cx,cy,cz]
        int dims[3];
        //! @see getCommunicationMask
        Mask communicationMask;
        //! rank of this process local to its host (node)
        int hostRank{0};
        //! offset for sliding window
        int yoffset;

        int mpiRank;
        int mpiSize;
        static constexpr uint32_t gridExchangeTag = 5u;
    };

} // namespace pmacc
