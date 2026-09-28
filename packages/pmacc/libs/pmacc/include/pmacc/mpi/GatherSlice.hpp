/* Copyright 2023-2024 Rene Widera
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
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

namespace pmacc
{
    namespace mpi
    {
        //! Gather data of a 2D Cartesian host buffer into a single MPI rank's host memory.
        class GatherSlice
        {
        private:
            std::optional<caravan::CommunicatorInfo> caravanGatherComm;
            caravan::MpiContext* mpiContext = nullptr;
            // gather rank zero will hold final data
            int gatherRank = -1;
            // number of ranks participating in the gather operation
            int numRanksInPlane = 0;

        public:
            GatherSlice() = default;

            virtual ~GatherSlice()
            {
                if(!caravanGatherComm)
                    return;
                try
                {
                    caravan::syncWait(caravan::mpi::destroyCommunicator(*mpiContext, caravanGatherComm->communicator));
                }
                catch(std::exception const& error)
                {
                    std::cerr << "Failed to destroy Caravan gather communicator: " << error.what() << '\n';
                }
            }

            /** Check if MPI rank is the gather master rank.
             *
             * @return True if this MPI rank receives the gathered data, else false.
             */
            bool isMaster() const
            {
                return gatherRank == 0;
            }

            /** Check if this MPI rank gathers the data.
             *
             * @return True if this MPI rank receives the gathered data, else false.
             */
            bool hasResult() const
            {
                return isMaster();
            }

            /** Query if MPI rank is part of the gather group.
             *
             * @return True if MPI rank is taking part on the gather operation, else false.
             */
            bool isParticipating() const
            {
                return gatherRank != -1;
            }

            /** Announce participation of the MPI rank in the gather operation
             *
             * @attention Must be called from all MPI ranks even if they do not participate.
             *
             * @param isActive True if MPI rank has data to gather, else false.
             * @return If the caller will contain the gathered data. @see isMaster()
             */
            bool participate(bool isActive)
            {
                mpiContext = &Environment<>::get().getMpiContext();
                auto const world = mpiContext->topology();
                caravanGatherComm
                    = caravan::syncWait<std::optional<caravan::CommunicatorInfo>>(caravan::mpi::splitCommunicator(
                        *mpiContext,
                        isActive ? std::optional<int>{0} : std::nullopt,
                        world.rank));
                gatherRank = caravanGatherComm ? caravanGatherComm->rank : -1;
                numRanksInPlane = caravanGatherComm ? caravanGatherComm->size : 0;
                return isMaster();
            }

            /** Gather after the caller has explicitly completed writes to localInputSlice. */
            template<typename T_DataType>
            auto gatherSliceExplicit(
                HostBuffer<T_DataType, DIM2>& localInputSlice,
                DataSpace<DIM2> globalSliceExtent,
                DataSpace<DIM2> localSliceOffset) const
            {
                using ValueType = T_DataType;
                // Guard against wrong usage, only MPI ranks which are participating into the gather are allowed to
                // call corresponding MPI functions.
                if(!isParticipating())
                    return std::shared_ptr<HostBuffer<ValueType, DIM2>>{};

                // get number of elements per participating mpi rank
                auto extentPerDevice = std::vector<DataSpace<DIM2>>(numRanksInPlane);
                auto offsetPerDevice = std::vector<DataSpace<DIM2>>(numRanksInPlane);
                auto localSliceSize = localInputSlice.capacityND();

                std::array<int, 2> localExtent{localSliceSize.x(), localSliceSize.y()};
                std::array<int, 2> localOffset{localSliceOffset.x(), localSliceOffset.y()};
                std::vector<std::array<int, 2>> extents(numRanksInPlane);
                std::vector<std::array<int, 2>> offsets(numRanksInPlane);
                caravan::syncWait<caravan::GatherResult>(caravan::mpi::gather(
                    *mpiContext,
                    std::as_bytes(std::span{localExtent}),
                    std::as_writable_bytes(std::span{extents}),
                    caravan::Peer{0},
                    caravanGatherComm->communicator));
                caravan::syncWait<caravan::GatherResult>(caravan::mpi::gather(
                    *mpiContext,
                    std::as_bytes(std::span{localOffset}),
                    std::as_writable_bytes(std::span{offsets}),
                    caravan::Peer{0},
                    caravanGatherComm->communicator));
                if(isMaster())
                    for(int rank = 0; rank < numRanksInPlane; ++rank)
                    {
                        extentPerDevice[rank] = DataSpace<DIM2>(extents[rank][0], extents[rank][1]);
                        offsetPerDevice[rank] = DataSpace<DIM2>(offsets[rank][0], offsets[rank][1]);
                    }

                std::vector<int> displs(numRanksInPlane);
                std::vector<int> count(numRanksInPlane);

                int offset = 0;
                int globalNumElements = 0u;

                if(isMaster())
                {
                    //! @todo replace by std::scan
                    for(int i = 0; i < numRanksInPlane; ++i)
                    {
                        displs[i] = offset * sizeof(ValueType);
                        count[i] = extentPerDevice[i].productOfComponents() * sizeof(ValueType);
                        offset += extentPerDevice[i].productOfComponents();
                        globalNumElements += extentPerDevice[i].productOfComponents();
                    }
                }

                // gather all data from other ranks
                auto allData = std::vector<ValueType>(globalNumElements);
                int localNumElements = localSliceSize.productOfComponents();

                std::vector<std::size_t> receiveBytes(count.begin(), count.end());
                std::vector<std::size_t> displacements(displs.begin(), displs.end());
                caravan::syncWait<caravan::GatherResult>(caravan::mpi::gatherV(
                    *mpiContext,
                    std::as_bytes(std::span{localInputSlice.data(), static_cast<std::size_t>(localNumElements)}),
                    std::as_writable_bytes(std::span{allData}),
                    std::move(receiveBytes),
                    std::move(displacements),
                    caravan::Peer{0},
                    caravanGatherComm->communicator));

                std::shared_ptr<HostBuffer<ValueType, DIM2>> globalField;
                if(isMaster())
                {
                    // globalNumElements is only on the master rank valid
                    PMACC_VERIFY_MSG(
                        globalSliceExtent.productOfComponents() == globalNumElements,
                        "Expected and gathered number of elements differ.");

                    globalField = std::make_shared<HostBuffer<ValueType, DIM2>>(globalSliceExtent);
                    auto globalFieldBox = globalField->getDataBox();

                    // aggregate data of all MPI ranks into a single 2D buffer
                    for(int dataSetNumber = 0; dataSetNumber < numRanksInPlane; ++dataSetNumber)
                    {
                        for(int y = 0; y < extentPerDevice[dataSetNumber].y(); ++y)
                            for(int x = 0; x < extentPerDevice[dataSetNumber].x(); ++x)
                            {
                                globalFieldBox(DataSpace<DIM2>(x, y) + offsetPerDevice[dataSetNumber]) = allData
                                    [displs[dataSetNumber] / sizeof(ValueType) + y * extentPerDevice[dataSetNumber].x()
                                     + x];
                            }
                    }
                }
                return globalField;
            }
        };
    } // namespace mpi
} // namespace pmacc
