#pragma once

#include "spmacc/Particle.hpp"
#include "spmacc/ParticleDescription.hpp"

#include <pmacc/math/MapTuple.hpp>
#include <pmacc/meta/GetKeyFromAlias.hpp>
#include <pmacc/meta/conversion/OperateOnSeq.hpp>
#include <pmacc/meta/conversion/SeqToMap.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/boostExtension/InheritLinearly.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/traits/IsSpecializationOf.hpp>

#include <boost/mpl/apply.hpp>

namespace pmacc
{
    namespace spearhed
    {
        namespace pmath = ::pmacc::math;

        /** Frame is a storage for arbitrary number >0 of Particles with attributes
         *
         * @tparam T_CreatePairOperator unary template operator to create a boost pair
         *                              from single type ( pair<name,dataType> )
         *                              @see MapTupel
         * @tparam T_ValueTypeSeq sequence with value_identifier
         * @tparam T_MethodsList sequence of classes with particle methods
         *                       (e.g. calculate mass, gamma, ...)
         * @tparam T_Flags sequence with identifiers to add flags on a frame
         *                 (e.g. useSolverXY, calcRadiation, ...)
         */
        template<typename T_CreatePairOperator, concepts::SpecializationOf<ParticleDescription> T_ParticleDescription>
        struct Frame;

        template<typename T_CreatePairOperator, concepts::SpecializationOf<ParticleDescription> T_ParticleDescription>
        struct Frame
            : protected pmath::MapTuple<
                  typename SeqToMap<typename T_ParticleDescription::ValueTypeSeq, T_CreatePairOperator>::type>
            , public InheritLinearly<mp_append<
                  typename T_ParticleDescription::MethodsList,
                  typename OperateOnSeq<
                      typename T_ParticleDescription::FrameExtensionList,
                      boost::mpl::apply1<boost::mpl::_1, Frame<T_CreatePairOperator, T_ParticleDescription>>>::type>>
        {
            using ParticleDescription = T_ParticleDescription;
            using Name = typename ParticleDescription::Name;
            //! Number of particle slots within the frame
            using NumSlots = typename ParticleDescription::NumSlots;
            static constexpr uint32_t frameSize = NumSlots::value;
            using ValueTypeSeq = typename ParticleDescription::ValueTypeSeq;
            using MethodsList = typename ParticleDescription::MethodsList;
            using FlagList = typename ParticleDescription::FlagsList;
            using FrameExtensionList = typename ParticleDescription::FrameExtensionList;
            /* definition of the MapTupel where we inherit from*/
            using BaseType = pmath::MapTuple<typename SeqToMap<ValueTypeSeq, T_CreatePairOperator>::type>;

            /* type of a single particle*/
            using ParticleType = Particle<Frame>;

        public:
            constexpr Frame()
            {
                /* disable all particles since we can not assume that newly allocated memory contains zeros */
                for(int i = 0; i < static_cast<int>(frameSize); ++i)
                    (*this)[i][multiMask_] = 0;
            }

            /** access the Nth particle*/
            constexpr ParticleType operator[](uint32_t const idx)
            {
                return ParticleType(*this, idx);
            }

            /** access the Nth particle*/
            constexpr ParticleType const operator[](uint32_t const idx) const
            {
                return ParticleType(*this, idx);
            }

            /** access attribute with a identifier
             *
             * @param T_Key instance of identifier type
             *              (can be an alias, value_identifier or any other class)
             * @return result of operator[] of MapTuple
             */
            template<typename T_Key>
            constexpr auto& getIdentifier(T_Key const)
            {
                using Key =
                    typename GetKeyFromAlias<ValueTypeSeq, T_Key, errorHandlerPolicies::ThrowValueNotFound>::type;
                return BaseType::operator[](Key());
            }

            /** const version of method getIdentifier(const T_Key) */
            template<typename T_Key>
            constexpr auto const& getIdentifier(T_Key const) const
            {
                using Key =
                    typename GetKeyFromAlias<ValueTypeSeq, T_Key, errorHandlerPolicies::ThrowValueNotFound>::type;
                return BaseType::operator[](Key());
            }

            static constexpr std::string getName()
            {
                return Name::str();
            }
        };
    } // namespace spearhed

    namespace traits
    {
        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct HasIdentifier<pmacc::spearhed::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::spearhed::Frame<T_CreatePairOperator, T_ParticleDescription>;

        public:
            using ValueTypeSeq = typename FrameType::ValueTypeSeq;
            /* if T_IdentifierName is void_ than we have no T_IdentifierName in our Sequence.
             * check is also valid if T_Key is a alias
             */
            using SolvedAliasName = typename GetKeyFromAlias<ValueTypeSeq, T_IdentifierName>::type;

            using type = boost::mp11::mp_contains<ValueTypeSeq, SolvedAliasName>; // FIXME(bgruber): boost::mp11::
                                                                                  // needed because of nvcc 11.0 bug
        };

        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct HasFlag<pmacc::spearhed::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::spearhed::Frame<T_CreatePairOperator, T_ParticleDescription>;
            using SolvedAliasName = typename pmacc::traits::GetFlagType<FrameType, T_IdentifierName>::type;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = mp_contains<FlagList, SolvedAliasName>;
        };

        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct GetFlagType<pmacc::spearhed::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::spearhed::Frame<T_CreatePairOperator, T_ParticleDescription>;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = typename GetKeyFromAlias<FlagList, T_IdentifierName>::type;
        };

    } // namespace traits

} // namespace pmacc
