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
#include "spearhed/particles/initialization/SetupInterface.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

#include <pmacc/Environment.hpp>

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

namespace spearhed
{
    /**
     * Creates and shares one particle-region buffer per species used by the setup.
     *
     * For each registered species, every block that targets it appends its region volumes (in block
     * order, overlaps allowed); the concatenated list is allocated once into the species' buffer.
     * Buffer ownership therefore lives here, not in the blocks, which is what lets several blocks feed
     * one species. A buffer already shared under the species id (e.g. a test fixture's) is reused and
     * filled in place instead of duplicated.
     */
    struct InitRegions
    {
        void operator()(DeviceHeap const& deviceHeap, SetupInterface auto& setup)
        {
            pmacc::spearhed::forEachSpecies(
                pmacc::spearhed::species::all,
                [&](auto species) { createSpecies<std::remove_cvref_t<decltype(species)>>(deviceHeap, setup); });
        }

    private:
        template<typename Species>
        static void createSpecies(DeviceHeap const& deviceHeap, SetupInterface auto& setup)
        {
            using PRBuf = pmacc::spearhed::ParticleRegionBuffer<PRTypeFor<Species>>;
            auto& dc = pmacc::Environment<>::get().DataConnector();

            // Gather, in block order, every region volume contributed to this species.
            std::vector<pmacc::spearhed::AABB<CS>> volumes;
            bool targeted = false;
            std::apply(
                [&](auto const&... block)
                {
                    (
                        [&]
                        {
                            using Block = std::remove_cvref_t<decltype(block)>;
                            if constexpr(blockTargets<Block, Species>)
                            {
                                targeted = true;
                                block.template addRegions<Species>(volumes);
                            }
                        }(),
                        ...);
                },
                setup.blocks());

            // Species not referenced by any block of this setup: leave it without a buffer.
            if(!targeted)
                return;

            auto const id = pmacc::spearhed::prBufId<Species>();
            std::shared_ptr<PRBuf> prBuf;
            if(dc.hasId(id))
                prBuf = dc.get<PRBuf>(id); // reuse a pre-shared buffer (e.g. test fixture)
            else
            {
                prBuf = std::make_shared<PRBuf>();
                dc.share(prBuf);
            }

            prBuf->create(volumes.size());
            auto const deviceHeapHandle = deviceHeap.getAllocatorHandle();
            for(auto const& volume : volumes)
                prBuf->pushBack(typename PRBuf::ParticleRegionType{deviceHeapHandle, volume});
            prBuf->buffer->hostToDevice();
        }
    };
} // namespace spearhed
