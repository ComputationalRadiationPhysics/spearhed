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

#include "pmacc/communication/CommunicatorMPI.hpp"

#include "pmacc/dimensions/Definition.hpp"

#include <cstddef>
#include <memory>
#include <span>

#include <caravan/core.hpp>
#include <caravan/mpi/native.hpp>

namespace pmacc
{
    namespace detail
    {
        template<unsigned T_DIM>
        struct LogRankCoords;

        template<>
        struct LogRankCoords<DIM1>
        {
            void operator()(int rank, int const (&coords)[DIM1]) const
            {
                log<ggLog::MPI>("Rank: %1% ; coords %2%") % rank % coords[0];
            }
        };

        template<>
        struct LogRankCoords<DIM2>
        {
            void operator()(int rank, int const (&coords)[DIM2]) const
            {
                log<ggLog::MPI>("Rank: %1% ; coords %2% %3%") % rank % coords[0] % coords[1];
            }
        };

        template<>
        struct LogRankCoords<DIM3>
        {
            void operator()(int rank, int const (&coords)[DIM3]) const
            {
                log<ggLog::MPI>("Rank: %1% ; coords %2% %3% %4%") % rank % coords[0] % coords[1] % coords[2];
            }
        };

    } // namespace detail

    template<unsigned DIM>
    void CommunicatorMPI<DIM>::init(
        caravan::MpiContext& context,
        DataSpace<DIM3> numberProcesses,
        DataSpace<DIM3> periodic)
    {
        this->periodic = periodic;
        mpiContext = &context;
        yoffset = 0;
        for(unsigned dimension = 0; dimension < DIM3; ++dimension)
            dims[dimension] = numberProcesses[dimension];

        std::vector<int> dimensions(DIM);
        std::vector<bool> periods(DIM);
        for(unsigned dimension = 0; dimension < DIM; ++dimension)
        {
            dimensions[dimension] = numberProcesses[dimension];
            periods[dimension] = periodic[dimension] != 0;
        }

        auto const snapshot = caravan::syncWait<caravan::TopologySnapshot>(
            caravan::mpi::createCartesian(context, std::move(dimensions), std::move(periods)));
        communicatorId = snapshot.communicator;
        signalCommunicatorId
            = caravan::syncWait<caravan::CommunicatorId>(caravan::mpi::duplicateCommunicator(context, communicatorId));
        mpiRank = snapshot.rank;
        mpiSize = snapshot.size;
        hostRank = snapshot.hostLocalRank;
        for(unsigned dimension = 0; dimension < DIM; ++dimension)
            baseCoordinates[dimension] = snapshot.coordinates[dimension];
        updateCoordinates();
    }

    template<unsigned DIM>
    bool CommunicatorMPI<DIM>::slide()
    {
        // we can only slide in y direction right now
        if constexpr(DIM < DIM2)
            return false;

        yoffset--;
        if(yoffset == -dims[1])
            yoffset = 0;

        updateCoordinates();

        return coordinates[1] == dims[1] - 1;
    }

    template<unsigned DIM>
    bool CommunicatorMPI<DIM>::setStateAfterSlides(size_t numSlides)
    {
        // nothing happens
        if(numSlides == 0)
            return false;

        // we can only slide in y direction right now
        if constexpr(DIM < DIM2)
            return false;

        bool result = false;

        // only need to apply (numSlides % num-gpus-y) slides
        for(size_t i = 0; i < (numSlides % dims[1]); ++i)
            result = slide();

        return result;
    }

    template<unsigned DIM>
    void CommunicatorMPI<DIM>::updateCoordinates()
    {
        // get own coordinates
        int coords[DIM];
        int rank = mpiRank;
        for(unsigned dimension = 0; dimension < DIM; ++dimension)
            coords[dimension] = baseCoordinates[dimension];

        if(DIM >= DIM2)
        {
            if(dims[1] > 1)
                coords[1] = (coords[1] + yoffset) % dims[1];

            while(coords[1] < 0)
                coords[1] += dims[1];
        }

        detail::LogRankCoords<DIM>()(rank, coords);

        for(uint32_t i = 0; i < DIM; ++i)
            this->coordinates[i] = coords[i];

        // init ranks of other hosts
        int mcoords[3];

        communicationMask = Mask();

        for(int i = 1; i < -12 * (int) DIM + 6 * (int) DIM * (int) DIM + 9; i++)
        {
            for(uint32_t j = 0; j < DIM; j++)
                mcoords[j] = coords[j];

            Mask m(i);
            if(m.containsExchangeType(LEFT))
                mcoords[0]--;
            if(m.containsExchangeType(RIGHT))
                mcoords[0]++;

            if constexpr(DIM >= DIM2)
            {
                if(m.containsExchangeType(TOP))
                    mcoords[1]--;
                if(m.containsExchangeType(BOTTOM))
                    mcoords[1]++;
            }

            if constexpr(DIM == DIM3)
            {
                if(m.containsExchangeType(BACK))
                    mcoords[2]++;
                if(m.containsExchangeType(FRONT))
                    mcoords[2]--;
            }

            bool ok = true;
            for(uint32_t j = 0; j < DIM; j++)
                if(periodic[j] == 0
                   && (mcoords[j] < 0 || mcoords[j] >= dims[j])) /*only check if no perodic for j dimension is set*/
                    ok = false;

            if(ok)
            {
                if(dims[1] > 1)
                    mcoords[1] = (mcoords[1] - yoffset) % dims[1];

                ranks[i] = 0;
                for(unsigned dimension = 0; dimension < DIM; ++dimension)
                {
                    int coordinate = mcoords[dimension] % dims[dimension];
                    if(coordinate < 0)
                        coordinate += dims[dimension];
                    ranks[i] = ranks[i] * dims[dimension] + coordinate;
                }
                communicationMask = communicationMask + Mask(i);
            }
            else
            {
                ranks[i] = -1;
            }

            // std::cout << "rank: " << rank << " " << i << " : " << ranks[i] << std::endl;
        }
    }

    // Explicit template instantiation to provide symbols for usage together with PMacc
    template class CommunicatorMPI<DIM1>;
    template class CommunicatorMPI<DIM2>;
    template class CommunicatorMPI<DIM3>;
} // namespace pmacc
