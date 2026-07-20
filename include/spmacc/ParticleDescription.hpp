/* Copyright 2014-2026 Rene Widera, Tapish Narwal
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

#include "spmacc/meta/ComponentList.hpp"
#include "spmacc/meta/String.hpp"
#include "spmacc/meta/TypeList.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

#include <pmacc/meta/conversion/ToSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>
#include <pmacc/static_assert.hpp>

#include <cstdint>
#include <type_traits>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    /** ParticleDescription defines attributes etc of a particle
     *
     * This class holds no runtime data.
     * The class holds information about the name, attributes, etc of a
     * particle. The particle name is taken from the species (@p T_Species::name).
     *
     * @tparam T_NumSlots compile time size of number of particles
     * @tparam T_ParticleRecord ll::Record with description of particle attribues
     * @tparam T_Species species tag the particle belongs to; supplies the particle name
     * @tparam T_FrameExtensionList sequence or single class with frame extensions
     *                    - a pmacc::spearhed::meta::ComponentList
     *                    - extension must be an unary template class that supports boost::mpl::apply1<>
     *                    - type of the final frame is applied to each extension class
     *                      (this allows pointers and references to a frame itself)
     *                    - the final frame that uses ParticleDescription inherits from all
     *                      extension classes
     */
    template<
        SpeciesTag T_Species,
        typename T_NumSlots,
        typename T_ParticleRecord,
        typename T_FrameExtensionList = pmacc::spearhed::meta::ComponentList<>>
    struct ParticleDescription
    {
        using Species = T_Species;
        using Name = pmacc::spearhed::meta::String<T_Species::name>;
        using ParticleRecord = T_ParticleRecord;
        using FrameExtensionList = T_FrameExtensionList;
        static constexpr uint32_t numSlots = T_NumSlots::value;
    };

    template<
        SpeciesTag T_Species,
        typename T_NumSlots,
        typename T_ParticleRecord,
        typename T_FrameExtensionList = pmacc::spearhed::meta::ComponentList<>>
    consteval auto createParticleDescription(
        T_Species,
        T_NumSlots,
        T_ParticleRecord,
        std::type_identity<T_FrameExtensionList> = {})
    {
        return ParticleDescription<T_Species, T_NumSlots, T_ParticleRecord, T_FrameExtensionList>{};
    }

} // namespace pmacc::spearhed
