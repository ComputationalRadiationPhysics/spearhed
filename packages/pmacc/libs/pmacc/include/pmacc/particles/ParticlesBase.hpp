/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/assert.hpp"
#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/mappings/kernel/StrideMapperFactory.hpp"
#include "pmacc/particles/ParticlesBase.kernel"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"
#include "pmacc/static_assert.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

#include <memory>

namespace pmacc
{
    /* Tag used for marking particle types */
    struct ParticlesTag;

    template<typename T_ParticleDescription, class T_MappingDesc, typename T_DeviceHeap>
    class ParticlesBase : public SimulationFieldHelper<T_MappingDesc>
    {
        using ParticleDescription = T_ParticleDescription;

    public:
        using MappingDesc = T_MappingDesc;

        /* Type of used particles buffer
         */
        using BufferType = ParticlesBuffer<
            ParticleDescription,
            typename MappingDesc::SuperCellSize,
            T_DeviceHeap,
            MappingDesc::Dim>;

        /* Type of frame in particles buffer
         */
        using FrameType = typename BufferType::FrameType;
        /* Type of border frame in a particle buffer
         */
        using FrameTypeBorder = typename BufferType::FrameTypeBorder;

        /* Type of the particle box which particle buffer create
         */
        using ParticlesBoxType = typename BufferType::ParticlesBoxType;

        /* Policies for handling particles in guard cells */
        using HandleGuardRegion = typename ParticleDescription::HandleGuardRegion;

        static constexpr uint32_t dim = MappingDesc::Dim;

        /* Mark this simulation data as a particle type */
        using SimulationDataTag = ParticlesTag;

    protected:
        BufferType* particlesBuffer;

        ParticlesBase(std::shared_ptr<T_DeviceHeap> const& deviceHeap, MappingDesc description)
            : SimulationFieldHelper<MappingDesc>(description)
            , particlesBuffer(nullptr)
        {
            particlesBuffer
                = new BufferType(deviceHeap, description.getGridLayout().sizeND(), MappingDesc::SuperCellSize::toRT());
        }

        ~ParticlesBase() override
        {
            delete this->particlesBuffer;
        }

        /** Lazily shift particles in an area defined by a mapper factory. */
        template<uint32_t T_area>
        [[nodiscard]] auto shiftParticles(bool onlyProcessMustShiftSupercells)
        {
            return shiftParticles(StrideAreaMapperFactory<T_area, 3>{}, onlyProcessMustShiftSupercells);
        }

        template<typename T_MapperFactory>
        [[nodiscard]] auto shiftParticles(T_MapperFactory const& mapperFactory, bool onlyProcessMustShiftSupercells)
        {
            return shiftParticlesImpl(
                StrideMapperFactory<T_MapperFactory, 3>{mapperFactory},
                onlyProcessMustShiftSupercells);
        }

    public:
        /** Lazily fill gaps in the area defined by a mapper factory, returning a sender. */
        template<typename T_MapperFactory>
        [[nodiscard]] auto fillGaps(T_MapperFactory const& mapperFactory)
        {
            auto const mapper = mapperFactory(this->cellDescription);
            return PMACC_LOCKSTEP_KERNEL(KernelFillGaps{})
                .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
        }

        /** Lazily fill all gaps (core, border and guard), returning a sender. */
        [[nodiscard]] auto fillAllGaps()
        {
            return fillGaps(AreaMapperFactory<CORE + BORDER + GUARD>{});
        }

        /** Lazily fill border gaps, returning a sender. */
        [[nodiscard]] auto fillBorderGaps()
        {
            return fillGaps(AreaMapperFactory<BORDER>{});
        }

        [[nodiscard]] auto deleteGuardParticles(uint32_t exchangeType);

        template<uint32_t T_area>
        [[nodiscard]] auto deleteParticlesInArea();

        /** copy guard particles to intermediate exchange buffer
         *
         * Copy all particles from the guard of a direction to the device exchange buffer.
         * @warning This method resets the number of particles in the processed supercells even
         * if there are particles left in the supercell and does not guarantee that the last frame is
         * contiguous filled.
         * Call fillAllGaps afterwards if you need a valid number of particles
         * and a contiguously filled last frame.
         */
        [[nodiscard]] auto copyGuardToExchange(uint32_t exchangeType);

        [[nodiscard]] auto insertParticles(uint32_t exchangeType, size_t numIndexEntries);

        ParticlesBoxType getDeviceParticlesBox()
        {
            return particlesBuffer->getDeviceParticleBox();
        }

        ParticlesBoxType getHostParticlesBox(int64_t const memoryOffset)
        {
            return particlesBuffer->getHostParticleBox(memoryOffset);
        }

        /* Get the particles buffer which is used for the particles.
         */
        BufferType& getParticlesBuffer()
        {
            PMACC_ASSERT(particlesBuffer != nullptr);
            return *particlesBuffer;
        }

        /** Lazily delete all live particles and reset supercell metadata, returning a sender.
         *
         * Unlike ParticlesBuffer::reset(), this traverses and deletes live particles.
         */
        [[nodiscard]] auto reset()
        {
            return deleteParticlesInArea<CORE + BORDER + GUARD>() | caravan::sequence(particlesBuffer->reset());
        }

    private:
        /** Shift all particles in the area defined by the given strided factory
         *
         * Note that the area itself is not strided, but the factory must produce stride mappers for the area.
         *
         * @tparam T_strideMapperFactory factory type to construct a stride mapper,
         *                               resulting mapper must have stride of at least 3,
         *                               adheres to the MapperFactory concept
         *
         * @param strideMapperFactory factory to construct a strided mapper,
         *                            the area is defined by the constructed mapper object
         * @param onlyProcessMustShiftSupercells whether to process only supercells with mustShift set to true
         * (optimization to be used with particle pusher) or process all supercells
         */
        template<typename T_StrideMapperFactory>
        [[nodiscard]] auto shiftParticlesImpl(
            T_StrideMapperFactory const& strideMapperFactory,
            bool onlyProcessMustShiftSupercells)
        {
            auto mapper = strideMapperFactory(this->cellDescription);
            PMACC_CASSERT_MSG(
                shiftParticles_stride_mapper_condition_failure____stride_must_be_at_least_3,
                decltype(mapper)::stride >= 3);
            auto const pBox = particlesBuffer->getDeviceParticleBox();
            auto const numSupercellsWithGuards = particlesBuffer->getSuperCellsCount();
            return caravan::alpaka::submit(
                [mapper, pBox, numSupercellsWithGuards, onlyProcessMustShiftSupercells](auto& nativeQueue) mutable
                {
                    do
                    {
                        PMACC_LOCKSTEP_KERNEL(KernelShiftParticles{})
                            .config(mapper.getGridDim(), pBox)
                            .enqueueNative(
                                nativeQueue,
                                pBox,
                                mapper,
                                numSupercellsWithGuards,
                                onlyProcessMustShiftSupercells);
                    } while(mapper.next());
                });
        }
    };

} // namespace pmacc

#include "pmacc/particles/ParticlesBase.tpp"
