/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/meta/Pair.hpp"
#include "pmacc/meta/conversion/MakeSeq.hpp"
#include "pmacc/particles/Identifier.hpp"
#include "pmacc/particles/ParticleDescription.hpp"
#include "pmacc/particles/boostExtension/InheritGenerators.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/StackExchangeBuffer.hpp"
#include "pmacc/particles/memory/dataTypes/ListPointer.hpp"
#include "pmacc/particles/memory/dataTypes/StaticArray.hpp"
#include "pmacc/particles/memory/dataTypes/SuperCell.hpp"
#include "pmacc/particles/memory/frames/Frame.hpp"
#include "pmacc/traits/GetUniqueTypeId.hpp"

#include <algorithm>
#include <array>
#include <memory>

namespace pmacc
{
    namespace detail
    {
        /** create static array
         */
        template<uint32_t T_size>
        struct OperatorCreatePairStaticArray
        {
            template<typename X>
            struct apply
            {
                using type = meta::Pair<
                    X,
                    StaticArray<typename traits::Resolve<X>::type::type, std::integral_constant<uint32_t, T_size>>>;
            };
        };
    } // namespace detail

    /**
     * Describes DIM-dimensional buffer for particles data on the host.
     *
     * @tParam T_ParticleDescription Object which describe a frame @see ParticleDescription.hpp
     * @tparam T_SuperCellSize TVec which descripe size of a superce
     * @tparam DIM dimension of the buffer (1-3)
     */
    template<typename T_ParticleDescription, typename T_SuperCellSize, typename T_DeviceHeap, unsigned DIM>
    class ParticlesBuffer
    {
    public:
        /** type of the border frame management object
         *
         * contains:
         *   - superCell position of the border frames inside a given range
         *   - start position inside the exchange stack for frames
         *   - number of frames corresponding to the superCell position
         */
        using BorderFrameIndex = ExchangeMemoryIndex<vint_t, DIM - 1>;

        using SuperCellSize = T_SuperCellSize;

        using ParticleAttributeList = MakeSeq_t<typename T_ParticleDescription::ValueTypeSeq, localCellIdx, multiMask>;

        using ParticleAttributeListBorder = MakeSeq_t<typename T_ParticleDescription::ValueTypeSeq, localCellIdx>;

        using FrameDescriptionWithManagementAttributes =
            typename ReplaceValueTypeSeq<T_ParticleDescription, ParticleAttributeList>::type;

        /** double linked list pointer */
        using LinkedListPointer = MakeSeq_t<PreviousFramePtr<>, NextFramePtr<>>;

        /* extent particle description with pointer to a frame*/
        using FrameDescription =
            typename ReplaceFrameExtensionSeq<FrameDescriptionWithManagementAttributes, LinkedListPointer>::type;

        /** frame definition
         *
         * a group of particles is stored as frame
         */
        using FrameType
            = Frame<detail::OperatorCreatePairStaticArray<T_ParticleDescription::NumSlots::value>, FrameDescription>;

        using FrameDescriptionBorder =
            typename ReplaceValueTypeSeq<T_ParticleDescription, ParticleAttributeListBorder>::type;

        /** frame which is used to communicate particles to neighbors
         *
         * - each frame contains only one particle
         * - local administration attributes of a particle are removed
         */
        using FrameTypeBorder = Frame<detail::OperatorCreatePairStaticArray<1U>, FrameDescriptionBorder>;

        using SuperCellType = SuperCell<FrameType, SuperCellSize>;

        using DeviceHeap = T_DeviceHeap;
        /* Type of the particle box which particle buffer create */
        using ParticlesBoxType = ParticlesBox<FrameType, typename DeviceHeap::AllocatorHandle, SuperCellSize, DIM>;

    private:
        /* this enum is used only for internal calculations */
        enum
        {
            SizeOfOneBorderElement = (sizeof(FrameTypeBorder) + sizeof(BorderFrameIndex))
        };

    public:
        /**
         * Constructor.
         *
         * @param deviceHeap device heap memory allocator
         * @param layout number of cell per dimension
         * @param superCellSize size of one super cell
         * @param gpuMemory how many memory on device is used for this instance (in byte)
         */
        ParticlesBuffer(
            std::shared_ptr<DeviceHeap> const& deviceHeap,
            DataSpace<DIM> layout,
            DataSpace<DIM> superCellSize)
            : superCellSize(superCellSize)
            , gridSize(layout)
            , m_deviceHeap(deviceHeap)
            , exchangeMemoryIndexerTag(traits::getUniqueId<uint32_t>())
        {
            exchangeMemoryIndexer
                = std::make_unique<GridBuffer<BorderFrameIndex, DIM1, BorderFrameIndex, DIM>>(DataSpace<DIM1>(0));
            framesExchanges = std::make_unique<GridBuffer<FrameType, DIM1, FrameTypeBorder, DIM>>(DataSpace<DIM1>(0));

            DataSpace<DIM> superCellsCount = gridSize / superCellSize;

            superCells = std::make_unique<GridBuffer<SuperCellType, DIM>>(superCellsCount);
        }

        /** Lazily initialize host and device supercell storage, returning a sender.
         *
         * This only zeroes storage; it does not traverse or delete live particle frames.
         * Use ParticlesBase::reset() to delete live particles.
         */
        [[nodiscard]] auto reset()
        {
            auto host = superCells->getHostBuffer().getOwnedAlpakaView();
            auto device = superCells->getDeviceBuffer().getOwnedAlpakaView();
            auto const elements = superCells->getHostBuffer().capacityND().productOfComponents();
            return caravan::alpaka::submit(
                [host = std::move(host), device = std::move(device), elements](auto& nativeQueue) mutable
                {
                    std::fill_n(::alpaka::onHost::data(host.value), elements, SuperCellType{});
                    ::alpaka::onHost::memcpy(
                        nativeQueue,
                        device.value,
                        host.value,
                        ::alpaka::onHost::getExtents(host.value));
                });
        }

        /**
         * Adds an exchange buffer to frames.
         *
         * @param receive Mask describing receive directions
         * @param usedMemory memory to be used for this exchange
         */
        void addExchange(Mask receive, size_t usedMemory, uint32_t communicationTag)
        {
            size_t numFrameTypeBorders = usedMemory / SizeOfOneBorderElement;

            framesExchanges
                ->addExchangeBuffer(receive, DataSpace<DIM1>(numFrameTypeBorders), communicationTag, true, false);

            exchangeMemoryIndexer->addExchangeBuffer(
                receive,
                DataSpace<DIM1>(numFrameTypeBorders),
                exchangeMemoryIndexerTag,
                true,
                false);
        }

        /**
         * Returns a ParticlesBox for device frame data.
         *
         * @return device frames ParticlesBox
         */
        ParticlesBoxType getDeviceParticleBox()
        {
            return ParticlesBoxType(superCells->getDeviceBuffer().getDataBox(), m_deviceHeap->getAllocatorHandle());
        }

        /**
         * Returns a ParticlesBox for host frame data.
         *
         * @return host frames ParticlesBox
         */
        ParticlesBoxType getHostParticleBox(int64_t memoryOffset)
        {
            return ParticlesBoxType(
                superCells->getHostBuffer().getDataBox(),
                m_deviceHeap->getAllocatorHandle(),
                memoryOffset);
        }

        /**
         * Returns if the buffer has a send exchange in ex direction.
         *
         * @param ex direction to query
         * @return true if buffer has send exchange for ex
         */
        bool hasSendExchange(uint32_t ex)
        {
            return framesExchanges->hasSendExchange(ex);
        }

        /**
         * Returns if the buffer has a receive exchange in ex direction.
         *
         * @param ex direction to query
         * @return true if buffer has receive exchange for ex
         */
        bool hasReceiveExchange(uint32_t ex)
        {
            return framesExchanges->hasReceiveExchange(ex);
        }

        StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1, DIM> getSendExchangeStack(uint32_t ex)
        {
            return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1, DIM>(
                framesExchanges->getSendExchange(ex),
                exchangeMemoryIndexer->getSendExchange(ex));
        }

        StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1, DIM> getReceiveExchangeStack(uint32_t ex)
        {
            return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1, DIM>(
                framesExchanges->getReceiveExchange(ex),
                exchangeMemoryIndexer->getReceiveExchange(ex));
        }

        /**
         * Starts sync data from own device buffer to neighbor device buffer.
         *
         * GridBuffer
         *
         */
        caravan::Event sendCompletion(uint32_t exchange) const
        {
            std::array completions{
                framesExchanges->sendCompletion(exchange),
                exchangeMemoryIndexer->sendCompletion(exchange)};
            return caravan::whenAll(completions);
        }

        caravan::Event receiveCompletion(uint32_t exchange) const
        {
            std::array completions{
                framesExchanges->receiveCompletion(exchange),
                exchangeMemoryIndexer->receiveCompletion(exchange)};
            return caravan::whenAll(completions);
        }

        void setSendCompletion(uint32_t exchange, caravan::Event completion)
        {
            framesExchanges->setSendCompletion(exchange, completion);
            exchangeMemoryIndexer->setSendCompletion(exchange, std::move(completion));
        }

        void setReceiveCompletion(uint32_t exchange, caravan::Event completion)
        {
            framesExchanges->setReceiveCompletion(exchange, completion);
            exchangeMemoryIndexer->setReceiveCompletion(exchange, std::move(completion));
        }

        [[nodiscard]] auto sendParticles(uint32_t exchange)
        {
            return caravan::whenAll(framesExchanges->send(exchange), exchangeMemoryIndexer->send(exchange));
        }

        [[nodiscard]] auto receiveParticles(uint32_t exchange)
        {
            return caravan::whenAll(framesExchanges->receive(exchange), exchangeMemoryIndexer->receive(exchange))
                   | caravan::then([](auto&&...) {});
        }

        /**
         * Returns number of supercells in each dimension.
         *
         * @return number of supercells
         */
        DataSpace<DIM> getSuperCellsCount()
        {
            PMACC_ASSERT(superCells != nullptr);
            return superCells->getGridLayout().sizeND();
        }

        /**
         * Returns number of supercells in each dimension.
         *
         * @return number of supercells
         */
        GridLayout<DIM> getSuperCellsLayout()
        {
            PMACC_ASSERT(superCells != nullptr);
            return superCells->getGridLayout();
        }

        /**
         * Returns size of supercells in each dimension.
         *
         * @return size of supercells
         */
        DataSpace<DIM> getSuperCellSize()
        {
            return superCellSize;
        }

    private:
        std::unique_ptr<GridBuffer<BorderFrameIndex, DIM1, BorderFrameIndex, DIM>> exchangeMemoryIndexer;

        std::unique_ptr<GridBuffer<SuperCellType, DIM>> superCells;
        /*GridBuffer for hold borderFrames, we need a own buffer to create first exchanges without core memory*/
        std::unique_ptr<GridBuffer<FrameType, DIM1, FrameTypeBorder, DIM>> framesExchanges;

        DataSpace<DIM> superCellSize;
        DataSpace<DIM> gridSize;
        std::shared_ptr<DeviceHeap> m_deviceHeap;
        // tag used to communicate the exchange memory indexer data via MPI
        uint32_t exchangeMemoryIndexerTag;
    };

    namespace lockstep::traits
    {
        //! Specialization to create a lockstep block configuration out of a particle buffer.
        template<typename T_ParticleDescription, typename T_SuperCellSize, typename T_DeviceHeap, unsigned T_dim>
        struct MakeBlockCfg<ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>>
            : std::true_type
        {
            static constexpr uint32_t frameSize
                = ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>::FrameType::frameSize;
            using type = BlockCfg<math::CT::UInt32<frameSize>>;
        };
    } // namespace lockstep::traits

} // namespace pmacc
