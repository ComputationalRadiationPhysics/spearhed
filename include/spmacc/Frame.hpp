/* Copyright 2013-2026 Rene Widera, Alexander Grund, Tapish Narwal
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

#include "spearhed/param.hpp"
#include "spmacc/ParticleDescription.hpp"
#include "spmacc/meta/TypeList.hpp"

#include <pmacc/particles/Identifier.hpp>
#include <pmacc/traits/IsSpecializationOf.hpp>

#include <boost/mpl/apply.hpp>

#include <llamaLite/llamaLite.hpp>

namespace pmacc
{
    namespace spearhed
    {
        namespace pmath = ::pmacc::math;

        template<concepts::SpecializationOf<ParticleDescription> T_ParticleDescription, typename T_ValueTypeSeq>
        struct Particle;

        /** Frame is a storage for arbitrary number >0 of Particles with attributes
         * move only type
         *
         * @tparam T_CreatePairOperator unary template operator to create a boost pair
         *                              from single type ( pair<name,dataType> )
         *                              @see MapTupel
         * @tparam T_ValueTypeSeq sequence with value_identifier
         */
        template<concepts::SpecializationOf<ParticleDescription> T_ParticleDescription>
        struct Frame
            : pmacc::spearhed::meta::InheritComponentsFrom<
                  Frame<T_ParticleDescription>,
                  typename T_ParticleDescription::FrameExtensionList>
        {
            using ParticleDescription = T_ParticleDescription;
            using Name = typename ParticleDescription::Name;
            //! Number of particle slots within the frame
            static constexpr uint32_t frameSize = ParticleDescription::numSlots;
            using ParticleRecord = typename ParticleDescription::ParticleRecord;

            /* type of a single particle*/
            // using ParticleType = Particle<ParticleDescription, ParticleRecord>;

            using SoAType = ll::SoA<ParticleRecord, frameSize>;

            SoAType particlesSoa;
            PMACC_ALIGN(liveParticles, uint32_t) { 0 };


        public:
            constexpr Frame()
            {
                // can this call a kernel? but what if i dont want to call a kernel.... what if i want a kernel for all
                // frames in a list or all frameLists in the sim action both predicate and action are passed in the idx
                // and the view at the idx forEachSlotInFrame(predicate, action);

                // lambda cannot work, since nvcc is so primitive
                // forEachSlotInFrame(true, [](auto view, auto idx){view[::spearhed::multiMask] = 0}]);

                // forEachSlotInFrame(true, [](someMultiMaskViewType multimask, auto idx){*multimask = 0}]);

                // view[tag] -> where this is a leaf access. then we get a raw ref
                // if we have a view to a leaf, then we can call derefernce to get raw ref.

                // think about doing this in parallel, since it is called inside a kernel. Maybe will need to be moved
                // out of the constructor
                /* disable all particles since we can not assume that newly allocated memory contains zeros */
                for(int i = 0; i < static_cast<int>(frameSize); ++i)
                    *particlesSoa[::spearhed::multiMask][i] = 0;
            }

            constexpr Frame(Frame const&) = delete;
            constexpr Frame& operator=(Frame const&) = delete;
            constexpr Frame(Frame&&) noexcept = default;
            constexpr Frame& operator=(Frame&&) noexcept = default;

            /** access attribute with a tag
             *
             * @param T_Key instance of tag type
             * @return View
             */
            template<ll::IsRecordAccess RA>
            [[nodiscard]] constexpr auto operator[](RA tag)

            {
                return particlesSoa[tag];
            }

            template<ll::IsRecordAccess RA>
            [[nodiscard]] constexpr auto operator[](RA tag) const
            {
                return particlesSoa[tag];
            }

            // Particle Access via Index
            // @returns ViewIndexed looking at a specific particle index
            [[nodiscard]] constexpr auto operator[](uint32_t idx)
            {
                return particlesSoa[idx];
            }

            /** access the Nth particle*/
            [[nodiscard]] constexpr auto operator[](uint32_t idx) const
            {
                return particlesSoa[idx];
            }

            static constexpr std::string getName()
            {
                return std::string(Name::view());
            }
        };
    } // namespace spearhed


} // namespace pmacc
