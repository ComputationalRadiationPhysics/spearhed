/* Copyright 2013-2024 Heiko Burau, Rene Widera
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
#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/mappings/kernel/ExchangeMapping.hpp"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <utility>

namespace pmacc
{
    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteGuardParticles(uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        return PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<uint32_t T_area>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteParticlesInArea()
    {
        auto const mapper = makeAreaMapper<T_area>(this->cellDescription);
        return PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::copyGuardToExchange(uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        auto stack = particlesBuffer->getSendExchangeStack(exchangeType);
        auto reset = stack.reset();
        auto copy = PMACC_LOCKSTEP_KERNEL(KernelCopyGuardToExchange{})
                        .config(mapper.getGridDim(), *particlesBuffer)(
                            particlesBuffer->getDeviceParticleBox(),
                            stack.getDeviceExchangePushDataBox(),
                            mapper);
        return std::move(reset) | caravan::sequence(std::move(copy)) | caravan::sequence(stack.publishDeviceSizes());
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::insertParticles(
        uint32_t exchangeType,
        size_t numIndexEntries)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        return PMACC_LOCKSTEP_KERNEL(KernelInsertParticles{})
            .config(numIndexEntries, *particlesBuffer)(
                particlesBuffer->getDeviceParticleBox(),
                particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                mapper);
    }

} // namespace pmacc
