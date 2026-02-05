#pragma once

#include "spmacc/Frame.hpp"
#include "spmacc/ParticleDescription.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/GetKeyFromAlias.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/meta/conversion/ResolveAliases.hpp>
#include <pmacc/meta/errorHandlerPolicies/ReturnValue.hpp>
#include <pmacc/particles/boostExtension/InheritLinearly.hpp>
#include <pmacc/particles/operations/Assign.hpp>
#include <pmacc/particles/operations/CopyValueIdentifier.hpp>
#include <pmacc/particles/operations/DeriveValueIdentifier.hpp>
#include <pmacc/particles/operations/Deselect.hpp>
#include <pmacc/particles/operations/InitValueIdentifier.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/IsSpecializationOf.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/types.hpp>

#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/plus.hpp>

namespace pmacc
{
    namespace spearhed
    {
        /** A single particle of a @see Frame
         *
         * A instance of this Particle is a representation ("pointer") to the memory
         * where the frame is stored.
         *
         * @tparam T_FrameType type of the parent frame
         * @tparam T_ValueTypeSeq sequence with all attribute identifiers
         *                        (can be a subset of T_FrameType::ValueTypeSeq)
         */
        template<
            concepts::SpecializationOf<ParticleDescription> T_ParticleDescription,
            typename T_ValueTypeSeq = typename T_ParticleDescription::ParticleRecord>
        struct Particle
        {
        private:
            /** Get the size in bytes for a value identifier */
            template<typename T_ValueIdentifier>
            struct GetSizeOfValueIdentifier
            {
                using ResolvedValueIdentifier = typename pmacc::traits::Resolve<T_ValueIdentifier>::type;
                using type = boost::mpl::integral_c<size_t, sizeof(typename ResolvedValueIdentifier::type)>;
            };

        public:
            using FrameType = Frame<T_ParticleDescription>;
            using ValueTypeSeq = T_ValueTypeSeq;
            using Name = typename FrameType::Name;

            using MethodsList = typename FrameType::MethodsList;

            /** The sum of the bytes required to store all value identifiers (attributes) of a particle.
             *
             * @return size in bytes
             */
            static constexpr size_t sizeInByte()
            {
                namespace bmpl = boost::mpl;
                return bmpl::accumulate<
                    ValueTypeSeq,
                    bmpl::integral_c<size_t, 0u>,
                    bmpl::plus<bmpl::_1, GetSizeOfValueIdentifier<bmpl::_2>>>::type::value;
            }

            /** pointer to parent frame where this particle is from
             *
             * ATTENTION: The pointer must be the last member to avoid local memory usage
             *            https://github.com/ComputationalRadiationPhysics/picongpu/pull/762
             */
            PMACC_ALIGN(frame, FrameType*) = nullptr;

            /** index of particle inside the Frame*/
            PMACC_ALIGN(idx, uint32_t) = std::numeric_limits<uint32_t>::max();

            /** set particle handle to invalid
             *
             * This method sets the particle handle to invalid. It is possible to test with
             * the method isHandleValid if the particle is valid.
             * If the particle is set to invalid it is not allowed to call any method other
             * than isHandleValid or setHandleInvalid, but it does not mean the particle is
             * deactivated outside of this instance.
             */
            HDINLINE void setHandleInvalid()
            {
                frame = nullptr;
            }

            /** check if particle handle is valid
             *
             * A valid particle handle means that the memory behind the handle can be used
             * savely. A valid handle does not mean that the particle's multiMask is valid (>=1).
             *
             * @return true if the particle handle is valid, else false
             */
            HDINLINE bool isHandleValid() const
            {
                return frame != nullptr;
            }

            HDINLINE Particle() = default;

            /** create particle
             *
             * @param frame reference to parent frame
             * @param idx index of particle inside the frame
             */
            HDINLINE Particle(FrameType& frame, uint32_t idx) : frame(&frame), idx(idx)
            {
            }

            template<typename T_OtherParticle>
            HDINLINE Particle(T_OtherParticle const& other) : frame(other.frame)
                                                            , idx(other.idx)
            {
            }

            /** access attribute with a identifier
             *
             * @param T_Key instance of identifier type
             *              (can be an alias, value_identifier or any other class)
             * @return result of operator[] of the Frame
             */
            template<typename T_Key>
            HDINLINE auto& operator[](T_Key const key)
            {
                PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);

                return frame->getIdentifier(key)[idx];
            }

            /** const version of method operator(const T_Key) */
            template<typename T_Key>
            HDINLINE const auto& operator[](T_Key const key) const
            {
                PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);

                return frame->getIdentifier(key)[idx];
            }

            HDINLINE
            Particle& operator=(Particle const& other) = default;

            /** Derive attributes
             *
             * The common subset of the attribute lists from both particles is
             * used to set the attributes in this particle with the corresponding ones from source particle.
             * The remaining attributes that only exist in this particle is simply set to their default values.
             */
            template<typename T_Worker, typename T_OtherFrameType, typename T_OtherValueTypeSeq>
            HDINLINE void derive(
                T_Worker const& worker,
                IdGenerator& idGen,
                pmacc::spearhed::Particle<T_OtherFrameType, T_OtherValueTypeSeq> const& srcParticle)
            {
                using DestTypeSeq = typename Particle::ValueTypeSeq;

                using pmacc::meta::ForEach;
                /* derive attributes */
                ForEach<DestTypeSeq, DeriveValueIdentifier<boost::mpl::_1>> derive;
                derive(worker, idGen, *this, srcParticle);
            }

            /** Assign common attributes of one particle to another
             *
             *  The source particle must have at least the attributes this particle has.
             */
            template<typename T_OtherFrameType, typename T_OtherValueTypeSeq>
            HDINLINE Particle& operator=(pmacc::spearhed::Particle<T_OtherFrameType, T_OtherValueTypeSeq> const& other)
            {
                /* create sequences with disjunctive attributes */
                using UniqueInDestTypeSeq = mp_set_difference<ValueTypeSeq, T_OtherValueTypeSeq>;

                static_assert(
                    pmacc::mp_size<UniqueInDestTypeSeq>::value == 0u,
                    "Source particle is not providing all attributes required to update the destination particle.");

                using pmacc::meta::ForEach;
                /* derive attributes */
                ForEach<ValueTypeSeq, CopyValueIdentifier<boost::mpl::_1>> copy;
                copy(*this, other);
                return *this;
            }
        };
    } // namespace spearhed

    namespace particles
    {
        namespace operations
        {
            namespace detail
            {
                template<typename T_MPLSeqWithObjectsToRemove, typename T_FrameType, typename T_ValueTypeSeq>
                struct Deselect<T_MPLSeqWithObjectsToRemove, pmacc::spearhed::Particle<T_FrameType, T_ValueTypeSeq>>
                {
                    using FrameType = T_FrameType;
                    using ValueTypeSeq = T_ValueTypeSeq;
                    using ParticleType = pmacc::spearhed::Particle<FrameType, ValueTypeSeq>;
                    using MPLSeqWithObjectsToRemove = T_MPLSeqWithObjectsToRemove;

                    /* translate aliases to full specialized identifier*/
                    using ResolvedSeqWithObjectsToRemove = typename ResolveAliases<
                        MPLSeqWithObjectsToRemove,
                        ValueTypeSeq,
                        errorHandlerPolicies::ReturnValue>::type;
                    /* remove types from original particle attribute list*/
                    using NewValueTypeSeq = typename RemoveFromSeq<ValueTypeSeq, ResolvedSeqWithObjectsToRemove>::type;
                    /* new particle type*/
                    using ResultType = pmacc::spearhed::Particle<FrameType, NewValueTypeSeq>;

                    HDINLINE
                    ResultType operator()(ParticleType const& particle)
                    {
                        return ResultType(particle);
                    };
                };

            } // namespace detail
        } // namespace operations
    } // namespace particles

} // namespace pmacc
