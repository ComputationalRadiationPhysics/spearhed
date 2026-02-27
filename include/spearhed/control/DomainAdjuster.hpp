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


#pragma once

#include "spearhed/param/dimension.param"

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/mpi/GetMPI_StructAsArray.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <array>
#include <stdexcept>

namespace spearhed
{
    /** adjust domain sizes
     *
     * Extend the local offset, the local and global domain size to fulfill all
     * PMacc conditions.
     */
    class DomainAdjuster
    {
    public:
        /** constructor
         *
         * @param numDevices number of devices [per dimension]
         * @param mpiPosition the position of the local device [per dimension]
         * @param isPeriodic if the outer simulation boundaries are periodic [per dimension]
         *                   1 is meaning periodic else 0
         * @param movingWindowEnabled if moving window is enabled
         */
        DomainAdjuster(
            pmacc::DataSpace<simDim> const& numDevices,
            pmacc::DataSpace<simDim> const& mpiPosition,
            pmacc::DataSpace<simDim> const& isPeriodic)
            : m_numDevices(numDevices)
            , m_mpiPosition(mpiPosition)
            , m_isPeriodic(isPeriodic)
            , m_isMaster(mpiPosition == pmacc::DataSpace<simDim>::create(0))
        {
        }

        /** adjust the domain size
         *
         * This method is a MPI collective operation and must be called from all MPI ranks.
         *
         * @param[in,out] globalDomainSize size of the global volume [in cells]
         * @param[in,out] localDomainSize size of the local volume [in cells]
         * @param[out] localDomainOffset local offset [in cells] relative to the origin of the global domain
         */
        void operator()(
            pmacc::DataSpace<simDim>& globalDomainSize,
            pmacc::DataSpace<simDim>& localDomainSize,
            pmacc::DataSpace<simDim>& localDomainOffset)
        {
            m_globalDomainSize = globalDomainSize;
            m_localDomainSize = localDomainSize;

            for(T_Dim d = 0; d < simDim; ++d)
            {
                deriveGlobalDomainSize(d);
                updateLocalDomainOffset(d);
            }

            if(globalDomainSize != m_globalDomainSize || localDomainSize != m_localDomainSize)
            {
                std::cout << " new grid size (global|local|offset): " << m_globalDomainSize.toString() << "|"
                          << m_localDomainSize.toString() << "|" << m_localDomainOffset.toString() << std::endl;
            }

            // write results back
            globalDomainSize = m_globalDomainSize;
            localDomainSize = m_localDomainSize;
            localDomainOffset = m_localDomainOffset;
        }

        /** only validate conditions
         *
         * Disable domain sizes auto adjustment.
         * The domain condition will be still checked.
         */
        void validateOnly()
        {
            m_validateOnly = true;
        }

    private:
        /** update local domain offset
         *
         * Share the local domain size with all MPI ranks and calculate the offset of the
         * local domain [in cells] relative to the origin of the global domain.
         *
         * @param dim dimension to update
         */
        void updateLocalDomainOffset(T_Dim const dim)
        {
            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();

            int mpiPos(gc.getPosition()[dim]);
            auto numMpiRanks = gc.getGlobalSize();

            // gather mpi position in the direction we are checking
            std::vector<int> mpiPositions(numMpiRanks);
            MPI_CHECK(MPI_Allgather(
                &mpiPos,
                1,
                MPI_INT,
                mpiPositions.data(),
                1,
                MPI_INT,
                gc.getCommunicator().getMPIComm()));

            // gather local sizes in the direction we are checking
            std::vector<uint64_t> allLocalSizes(numMpiRanks);
            auto lSize = static_cast<uint64_t>(m_localDomainSize[dim]);
            MPI_CHECK(MPI_Allgather(
                &lSize,
                1,
                MPI_UINT64_T,
                allLocalSizes.data(),
                1,
                MPI_UINT64_T,
                gc.getCommunicator().getMPIComm()));

            uint64_t offset = 0u;
            for(size_t i = 0u; i < mpiPositions.size(); ++i)
            {
                if(mpiPositions[i] < mpiPos)
                    offset += allLocalSizes[i];
            }

            /* since we are not doing independent reduces per slice we need
             * to adjust the offset result by dividing with the number of
             * MPI ranks in all other dimensions.
             */
            offset /= static_cast<uint64_t>(m_numDevices.productOfComponents() / m_numDevices[dim]);
            m_localDomainOffset[dim] = static_cast<int>(offset);
        }

        /** derive the global domain size
         *
         * Calculate the global domain size.
         *
         * @param dim dimension to update
         */
        void deriveGlobalDomainSize(T_Dim const dim)
        {
            int validGlobalGridSize = 0u;

            auto localDomainSize = m_localDomainSize[dim];
            pmacc::mpi::MPIReduce mpiReduce;
            mpiReduce(
                pmacc::math::operation::Add(),
                &validGlobalGridSize,
                &localDomainSize,
                1,
                pmacc::mpi::reduceMethods::AllReduce());
            /* since we are not doing independent reduces per slice we need
             * to adjust the reduce result by dividing the sizes of all other dimensions
             * we are not check within the method call
             */
            validGlobalGridSize /= static_cast<int>(m_numDevices.productOfComponents() / m_numDevices[dim]);


            if(m_isMaster && validGlobalGridSize != m_globalDomainSize[dim])
            {
                showMessage(dim, "Invalid global grid size.", m_globalDomainSize[dim], validGlobalGridSize);
            }

            m_globalDomainSize[dim] = static_cast<int>(validGlobalGridSize);
        }

        /** print a message to the user
         *
         * Throw an error with the message if is validateOnly was called.
         *
         * @param dim dimension index which was checked
         * @param msg problem description
         * @param currentSize current domain size in the given direction
         * @param updatedSize updated/corrected domain size for the given dimension
         * @param postMsg optional postfix message
         */
        void showMessage(
            size_t const dim,
            std::string const& msg,
            int const currentSize,
            int const updatedSize,
            std::string postMsg = "") const
        {
            /**! lookup table to translate a dimension index into a name
             */
            std::array<char, 3> const dimNames = {'x', 'y', 'z'};

            if(m_validateOnly)
                throw std::runtime_error(
                    std::string("Dimension ") + dimNames[dim] + ": " + msg + " Suggestion: set "
                    + std::to_string(currentSize) + " to " + std::to_string(updatedSize) + postMsg);
            else
                std::cout << "Dimension " << dimNames[dim] << ": " << msg << " Auto adjust from " << currentSize
                          << " to " << updatedSize << postMsg << std::endl;
        }

        pmacc::DataSpace<simDim> m_globalDomainSize;
        pmacc::DataSpace<simDim> m_localDomainSize;
        pmacc::DataSpace<simDim> m_localDomainOffset;
        pmacc::DataSpace<simDim> const m_numDevices;
        pmacc::DataSpace<simDim> const m_mpiPosition;
        pmacc::DataSpace<simDim> const m_isPeriodic;
        bool const m_isMaster;

        //! if true it will only validate the conditions
        bool m_validateOnly = false;
    };

} // namespace spearhed
