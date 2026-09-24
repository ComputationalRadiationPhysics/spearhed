/* Copyright 2025-2026 Tapish Narwal
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

#include "spmacc/FrameList.hpp"

#include <pmacc/traits/IsSpecializationOf.hpp>

namespace pmacc
{
    namespace spearhed
    {
        /**
         * Holds bucket-local spatial metadata and the particles in that bucket.
         * Metadata is copied when the particle-region buffer is populated, so it must stay compact.
         */
        template<typename TVolume, concepts::SpecializationOf<Frame> T_Frame, typename T_DeviceHeapHandle>
        struct ParticleRegion
        {
            using VolumeType = TVolume;
            using FrameType = T_Frame;
            using Species = typename T_Frame::ParticleDescription::Species;

            constexpr ParticleRegion(T_DeviceHeapHandle const& deviceHeapHandle) : particleFrameList{deviceHeapHandle}
            {
            }

            constexpr ParticleRegion(T_DeviceHeapHandle const& deviceHeapHandle, TVolume const& spatial)
                : spatial(spatial)
                , particleFrameList{deviceHeapHandle}
            {
            }

            // return a reference to the Frame list
            auto& getParticleFrameList()
            {
                return particleFrameList;
            }

            TVolume spatial{};
            FrameList<T_Frame, T_DeviceHeapHandle> particleFrameList;
        };
    } // namespace spearhed
} // namespace pmacc
