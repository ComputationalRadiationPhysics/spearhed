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

#include <pmacc/Environment.hpp>

namespace spearhed
{
    namespace init::detail
    {
    }

    struct InitRegions
    {
        // creates and shares one PRBuf per role.
        void operator()(DeviceHeap const& deviceHeap, SetupInterface auto& setup)
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();
            using Roles = typename std::remove_cvref_t<decltype(setup)>::Roles;

            auto createOne = [&]<typename Role>()
            {
                auto prBuf = std::make_shared<pmacc::spearhed::ParticleRegionBuffer<PRType, Role>>();
                dc.share(prBuf);
                setup.template block<Role>().setupRegions(*prBuf, deviceHeap);
            };

            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                (createOne.template operator()<std::tuple_element_t<I, Roles>>(), ...);
            }(std::make_index_sequence<std::tuple_size_v<Roles>>{});
        }
    };
} // namespace spearhed
