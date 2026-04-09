/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
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

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/alpakaHelper/acc.hpp>
#include <pmacc/particles/IdProvider.hpp>
#include <pmacc/particles/memory/buffers/MallocMCBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Positioning.hpp>

#include <cstdint>
#include <memory>
#include <optional>

namespace spearhed::test
{
    template<unsigned DIM>
    struct SpearhedParticleFixture
    {
        pmacc::test::PMaccFixture<DIM> pmaccBase{};
        std::optional<DeviceHeap> deviceHeap{std::nullopt};
        std::shared_ptr<pmacc::spearhed::ParticleRegionBuffer<PRType>> prBuf;
        pmacc::DataConnector& dc;

        SpearhedParticleFixture() : dc(pmacc::Environment<DIM>::get().DataConnector())
        {
            auto& env = pmacc::Environment<DIM>::get();

            // ID Provider Setup
            uint64_t maxRanks = env.GridController().getGpuNodes().productOfComponents();
            uint64_t rank = env.GridController().getScalarPosition();
            dc.share(std::make_shared<pmacc::IdProvider>("globalId", rank, maxRanks));

            // Device Heap Setup
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
            constexpr auto testHeapSize = 256ull * 1024 * 1024;
            auto& deviceManager = pmacc::manager::Device<pmacc::ComputeDevice>::get();
            auto alpakaDevice = deviceManager.current();
            auto alpakaQueue = pmacc::eventSystem::getComputeDeviceQueue(pmacc::ITask::TASK_DEVICE)->getAlpakaQueue();

            deviceHeap.emplace(alpakaDevice, alpakaQueue, testHeapSize);
            alpaka::wait(alpakaQueue);

            dc.consume(std::make_unique<pmacc::MallocMCBuffer<DeviceHeap>>(*deviceHeap));
#endif
            dc.template get<pmacc::IdProvider>("globalId")->reset();

            // Particle Region Buffer Setup
            prBuf = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<PRType>>();
            dc.share(prBuf);
        }

        ~SpearhedParticleFixture()
        {
            dc.clean();
        }
    };
} // namespace spearhed::test
