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

#include "llamaLite/llamaLite.hpp"
#include "spmacc/meta/ComponentList.hpp"
#include "spmacc/meta/TypeList.hpp"

#include <pmacc/meta/conversion/ToSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>
#include <pmacc/static_assert.hpp>

#include <cstdint>
#include <type_traits>

namespace pmacc::spearhed
{
    /** ParticleDescription defines attributes, methods and flags of a particle
     *
     * This class holds no runtime data.
     * The class holds information about the name, attributes, flags and methods of a
     * particle.
     *
     * @tparam T_Name name of described particle (e.g. electron, ion)
     *                type must be a SPMACC_CSTRING
     * @tparam T_NumSlots compile time size of number of particles
     * @tparam T_ParticleRecord ll::Record with description of particle attribues
     * @tparam T_Flags sequence or single type with identifier to add flags on a frame, must not have duplicates
     * @tparam T_FrameExtensionList sequence or single class with frame extensions
     *                    - a pmacc::spearhed::meta::ComponentList
     *                    - extension must be an unary template class that supports boost::mpl::apply1<>
     *                    - type of the final frame is applied to each extension class
     *                      (this allows pointers and references to a frame itself)
     *                    - the final frame that uses ParticleDescription inherits from all
     *                      extension classes
     */
    template<
        typename T_Name,
        typename T_NumSlots,
        typename T_ParticleRecord,
        typename T_Flags = pmacc::spearhed::meta::TypeList<>,
        typename T_FrameExtensionList = pmacc::spearhed::meta::ComponentList<>>
    struct ParticleDescription
    {
        using Name = T_Name;
        using ParticleRecord = T_ParticleRecord;
        using FlagsList = pmacc::spearhed::meta::ToTypeList_t<T_Flags>;
        using FrameExtensionList = T_FrameExtensionList;
        static constexpr uint32_t numSlots = T_NumSlots::value;

        // Compile-time check uniqueness of attributes and flags
        // PMACC_CASSERT_MSG(
        //     _error_particles_must_not_have_duplicate_attributes____check_your_speciesDefinition_param_file,
        //     isUnique<T_ParticleRecord>);
        PMACC_CASSERT_MSG(
            _error_particles_must_not_have_duplicate_flags____check_your_speciesDefinition_param_file,
            pmacc::spearhed::meta::isUnique_v<FlagsList>);

        template<typename NewFrameExtensionSeq>
        consteval auto replaceFrameExtensionSeq()
        {
            return ParticleDescription<T_Name, T_NumSlots, T_ParticleRecord, T_Flags, NewFrameExtensionSeq>{};
        }
    };

    template<typename T_Name, typename T_NumSlots, typename T_ParticleRecord>
    consteval auto createParticleDescription(T_Name, T_NumSlots, T_ParticleRecord)
    {
        return ParticleDescription<T_Name, T_NumSlots, T_ParticleRecord>{};
    }

} // namespace pmacc::spearhed
