/* Copyright 2017-2024 Axel Huebl, Rene Widera, Sergei Bastrakov
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
#include "pmacc/attribute/Constexpr.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/lockstep/Config.hpp"
#include "pmacc/lockstep/ForEach.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/particles/algorithm/ForEachFrameSlot.hpp"
#include "pmacc/particles/algorithm/detail/ForEach.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace pmacc::particles::algorithm
{
    namespace acc
    {
        //! Policy to forward iterate over particles/frames of a supercell.
        struct Forward
        {
            template<typename T_ParBox>
            DINLINE static auto getFirst(T_ParBox& pb, DataSpace<T_ParBox::Dim> const& superCellIdx)
            {
                return pb.getFirstFrame(superCellIdx);
            }

            template<typename T_ParBox>
            DINLINE static auto next(T_ParBox& pb, typename T_ParBox::FramePtr const& framePtr)
            {
                return pb.getNextFrame(framePtr);
            }

            template<typename T_ForEach>
            DINLINE static auto createCtx(T_ForEach& forEachParticleInFrame)
            {
                return lockstep::makeVar<uint32_t>(forEachParticleInFrame, 0u);
            }
        };

        //! Policy to reverse iterate over particles/frames of a supercell.
        struct Reverse
        {
            template<typename T_ParBox>
            DINLINE static auto getFirst(T_ParBox& pb, DataSpace<T_ParBox::Dim> const& superCellIdx)
            {
                return pb.getLastFrame(superCellIdx);
            }

            template<typename T_ParBox>
            DINLINE static auto next(T_ParBox& pb, typename T_ParBox::FramePtr const& framePtr)
            {
                return pb.getPreviousFrame(framePtr);
            }

            template<typename T_ForEach>
            DINLINE static auto createCtx(T_ForEach& forEachParticleInFrame)
            {
                return lockstep::makeVar<lcellId_t>(forEachParticleInFrame, lcellId_t(0u));
            }
        };

        /** Particle traversal selected explicitly by callback interface.
         *
         * The primary template is intentionally undefined; only the particle and frame interfaces
         * below are supported.
         *
         * @tparam T_AccessTag detail::CallParticleFunctor or detail::CallFrameFunctor
         * @tparam T_Order Iteration order. valid options: Forward or Reverse
         * @tparam T_ParBox Type of the particle box to traverse.
         * @tparam T_Worker lockstep worker
         */
        template<typename T_AccessTag, typename T_Order, typename T_ParBox, typename T_Worker>
        class ForEachParticle;

        /** Legacy collective frame-batch traversal. */
        template<typename T_Order, typename T_ParBox, typename T_Worker>
        class ForEachParticle<detail::CallFrameFunctor, T_Order, T_ParBox, T_Worker>
        {
        private:
            static constexpr uint32_t dim = T_ParBox::Dim;

            static constexpr uint32_t frameSize = T_ParBox::frameSize;
            static constexpr uint32_t numWorkers = T_Worker::numWorkers();

            /** Number of frames to skip.
             *
             * If we have more workers than particles in a frame we skip frames depending on the worker id.
             */
            static constexpr uint32_t frameDistance = (numWorkers + frameSize - 1u) / frameSize;

            using ForEachParticleInFrame = lockstep::traits::MakeForEach_t<T_Worker, frameSize * frameDistance, 1>;

            DataSpace<dim> const m_superCellIdx;
            ForEachParticleInFrame const forEachParticleInFrame;
            T_ParBox m_particlesBox;

        public:
            /** Provide the lockstep foreach executor to execute a functor for each particle within a set of frames.
             *
             * Depending of the number of workers used it could be that more than one frame is executed by all workers
             * in parallel. The frame functor interface is therefore a context with one frame per virtual worker.
             * It is guaranteed that the executor fits to the frameCtx provided by the frame functor interface. @see
             * detail::FrameFunctorInterface
             *
             * @return Lockstep foreach executor to operate on frames.
             */
            DINLINE auto lockstepForEach() const
            {
                return forEachParticleInFrame;
            }

            /** Construct algoritm to operate on particles in a superCell
             *
             * @param workerIdx workerIdx index of the worker: range [0;workerSize)
             * @param particlesBox particles memory
             *                     It is not allowed concurrently to add or remove particles during the
             *                     execution of this algorithm.
             * @param superCellIdx index of the superCell where particles should be processed
             */
            DINLINE ForEachParticle(
                T_Worker const& worker,
                T_ParBox const& particlesBox,
                DataSpace<dim> const& superCellIdx)
                : m_superCellIdx(superCellIdx)
                , forEachParticleInFrame(worker)
                , m_particlesBox(particlesBox)
            {
            }

            /** Execute a frame callback with the legacy mapping-bound batch context.
             *
             * @tparam T_Functor Frame functor following detail::FrameFunctorInterface.
             * @param unaryFunctor Callable as void(worker, FrameCtx). The context can hold several
             *                     frames and invalid handles; use lockstepForEach() to distribute it.
             *                     The caller must ensure data-race freedom and uniform collective
             *                     entry/termination when synchronizing within a callback.
             */
            template<typename T_Functor>
            DINLINE void operator()(T_Functor&& unaryFunctor) const
            {
                auto const worker = forEachParticleInFrame.getWorker();
                if constexpr(numWorkers > frameSize)
                {
                    PMACC_CASSERT_MSG(
                        __number_of_workers_must_be_either_less_or_a_multiple_of_framesize,
                        (numWorkers % frameSize) == 0);
                }

                constexpr bool isSupportedOrder = std::is_same_v<Reverse, T_Order> || std::is_same_v<Forward, T_Order>;
                static_assert(isSupportedOrder, "Unsupported order policy");

                auto& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                uint32_t const numParticlesInSupercell = superCell.getNumParticles();

                // end kernel if we have no particles
                if(numParticlesInSupercell == 0)
                    return;

                // Information used to know if a particle within a frame is active without checking the multiMask.
                auto frameConditionDataCtx = T_Order::createCtx(forEachParticleInFrame);

                // get starting frame
                auto framePtrCtx = forEachParticleInFrame(
                    [&](uint32_t const idx, auto& frameConditionData)
                    {
                        auto frame = T_Order::getFirst(m_particlesBox, m_superCellIdx);

                        if constexpr(std::is_same_v<Reverse, T_Order>)
                        {
                            if(frame.isValid())
                                frameConditionData = superCell.getSizeLastFrame();
                        }

                        /* Same as frameDistance, variable is required to workaround an nvcc compiler bug.
                         * Without redefinition within the lambda the variable will be undefined.
                         */
                        constexpr uint32_t frameDist = (numWorkers + frameSize - 1u) / frameSize;
                        if constexpr(frameDist > 1)
                        {
                            uint32_t const skipFrames = idx / frameSize;
                            /* select N-th (N=frameIdx) frame from the list */
                            for(uint32_t i = 1; i <= skipFrames && frame.isValid(); ++i)
                            {
                                frame = T_Order::next(m_particlesBox, frame);
                            }
                            if constexpr(std::is_same_v<Reverse, T_Order>)
                                frameConditionData = skipFrames != 0 ? frameSize : frameConditionData;
                        }
                        return frame;
                    },
                    frameConditionDataCtx);

                bool hasMoreWork = true;
                // iterate over all frames in the supercell
                do
                {
                    auto frameFunctor = detail::makeFrameFunctorInterface(std::forward<T_Functor>(unaryFunctor));
                    frameFunctor(worker, framePtrCtx);

                    hasMoreWork = false;

                    // get next frame
                    forEachParticleInFrame(
                        [&](uint32_t const idx, auto& framePtr, auto& frameConditionData)
                        {
                            if constexpr(std::is_same_v<Reverse, T_Order>)
                                frameConditionData = frameSize;
                            else if constexpr(std::is_same_v<Forward, T_Order>)
                                frameConditionData += frameSize * frameDistance;

                            for(uint32_t i = 0; i < frameDistance; ++i)
                                if(framePtr.isValid())
                                    framePtr = T_Order::next(m_particlesBox, framePtr);

                            hasMoreWork = hasMoreWork || framePtr.isValid();
                        },
                        framePtrCtx,
                        frameConditionDataCtx);

                } while(hasMoreWork);
            }

            DINLINE bool hasParticles() const
            {
                return numParticles() != 0u;
            }

            DINLINE uint32_t numParticles() const
            {
                auto const& superCell = m_particlesBox.getSuperCell(m_superCellIdx);
                return superCell.getNumParticles();
            }
        };

        /** Unary traversal: block-uniform linked-frame walking and independent alpaka slot loops.
         *
         * All frames except the last must be full; the last has a packed live prefix. PIC multiMask
         * values (including exchange encodings) are deliberately not interpreted here. Frame order
         * is preserved, but no serial order of particle execution is promised. Callbacks must not
         * synchronize or mutate the frame topology. Collective frame batches use the frame specialization.
         */
        template<typename T_Order, typename T_ParBox, typename T_Worker>
        class ForEachParticle<detail::CallParticleFunctor, T_Order, T_ParBox, T_Worker>
        {
            static_assert(std::is_same_v<T_Order, Forward> || std::is_same_v<T_Order, Reverse>);
            static constexpr uint32_t frameSize = T_ParBox::frameSize;
            T_Worker m_worker;
            T_ParBox m_particlesBox;
            DataSpace<T_ParBox::Dim> m_superCellIdx;

        public:
            DINLINE ForEachParticle(
                T_Worker const& worker,
                T_ParBox const& particlesBox,
                DataSpace<T_ParBox::Dim> const& superCellIdx)
                : m_worker(worker)
                , m_particlesBox(particlesBox)
                , m_superCellIdx(superCellIdx)
            {
            }

            DINLINE uint32_t numParticles() const
            {
                return m_particlesBox.getSuperCell(m_superCellIdx).getNumParticles();
            }

            DINLINE bool hasParticles() const
            {
                return numParticles() != 0u;
            }

            template<typename T_Functor>
            DINLINE void operator()(T_Functor&& unaryFunctor) const
            {
                uint32_t remaining = numParticles();
                if(remaining == 0u)
                    return;

                // Bind once, including rvalues; every invocation below uses the same lvalue.
                auto particleFunctor = detail::makeParticleFunctorInterface(std::forward<T_Functor>(unaryFunctor));
                auto frame = T_Order::getFirst(m_particlesBox, m_superCellIdx);
                uint32_t activeSlots = frameSize;
                if constexpr(std::is_same_v<T_Order, Reverse>)
                    activeSlots = m_particlesBox.getSuperCell(m_superCellIdx).getSizeLastFrame();

                while(frame.isValid() && remaining != 0u)
                {
                    if constexpr(std::is_same_v<T_Order, Forward>)
                        activeSlots = remaining < frameSize ? remaining : frameSize;

                    forEachFrameSlot<frameSize>(
                        m_worker,
                        [&](uint32_t slot)
                        {
                            // Test the packed-prefix bound before constructing/accessing a particle handle.
                            if(slot < activeSlots)
                            {
                                auto particle = frame[slot];
                                static_assert(
                                    std::is_void_v<decltype(unaryFunctor(m_worker, particle))>,
                                    "Unary particle functors must return void");
                                if(particle.isHandleValid())
                                    particleFunctor(m_worker, particle);
                            }
                        });
                    remaining -= activeSlots;
                    frame = T_Order::next(m_particlesBox, frame);
                    activeSlots = frameSize;
                }
            }
        };

        /** Creates an executor to iterate over all particles in a supercell.
         *
         * @return ForEach executor which can be invoked with a particle functor as argument. @see
         * detail::ParticleFunctorInterface
         *
         * @{
         * @tparam T_Worker lockstep worker
         * @tparam T_ParBox Type of the particle box.
         * @param particlesBox particle box
         * @param superCellIdx supercell index
         */
        template<typename T_Worker, typename T_ParBox>
        DINLINE auto makeForEach(
            T_Worker const& worker,
            T_ParBox const& particlesBox,
            DataSpace<T_ParBox::Dim> const& superCellIdx)
        {
            return ForEachParticle<detail::CallParticleFunctor, Reverse, T_ParBox, T_Worker>(
                worker,
                particlesBox,
                superCellIdx);
        }

        /** Creates an executor to iterate over all frames in a supercell.
         *
         * @tparam T_Order Iteration order. valid options: Forward or Reverse
         * @return ForEach executor which can be invoked with a frame functor as argument. @see
         * detail::FrameFunctorInterface
         */
        template<typename T_Order, typename T_ParBox, typename T_Worker>
        DINLINE auto makeForEachFrame(
            T_Worker const& worker,
            T_ParBox const& particlesBox,
            DataSpace<T_ParBox::Dim> const& superCellIdx)
        {
            return ForEachParticle<detail::CallFrameFunctor, T_Order, T_ParBox, T_Worker>(
                worker,
                particlesBox,
                superCellIdx);
        }

        /** Creates an executor to iterate over all particles in a supercell.
         *
         * @tparam T_Order Iteration order. valid options: Forward or Reverse
         * @return ForEach executor which can be invoked with a particle functor as argument. @see
         * detail::ParticleFunctorInterface
         */
        template<typename T_Order, typename T_ParBox, typename T_Worker>
        DINLINE auto makeForEach(
            T_Worker const& worker,
            T_ParBox const& particlesBox,
            DataSpace<T_ParBox::Dim> const& superCellIdx)
        {
            return ForEachParticle<detail::CallParticleFunctor, T_Order, T_ParBox, T_Worker>(
                worker,
                particlesBox,
                superCellIdx);
        }

        /** @} */

        namespace detail
        {
            //! operate on particles of a species
            struct KernelForEachParticle
            {
                /** operate on particles
                 *
                 * @tparam T_Worker lockstep worker type
                 * @tparam T_Functor type of the functor to operate on a particle
                 * @tparam T_Mapping mapping functor type
                 * @tparam T_ParBox pmacc::ParticlesBox, type of the species box
                 *
                 * @param acc alpaka accelerator
                 * @param functor functor to operate on a particle
                 *                must fulfill the interface pmacc::functor::Interface<F, 1u, void>
                 * @param mapper functor to map a block to a supercell
                 * @param pb particles species box
                 */
                template<typename T_Worker, typename T_Functor, typename T_Mapping, typename T_ParBox>
                DINLINE void operator()(T_Worker const& worker, T_Functor functor, T_Mapping const mapper, T_ParBox pb)
                    const
                {
                    constexpr uint32_t dim = T_ParBox::Dim;

                    DataSpace<dim> const superCellIdx(
                        mapper.getSuperCellIndex(DataSpace<dim>(worker.blockDomIdxND())));

                    auto forEachParticle = makeForEach(worker, pb, superCellIdx);

                    // end kernel if we have no particles
                    if(!forEachParticle.hasParticles())
                        return;

                    // offset of the superCell (in cells, without any guards) to the origin of the local
                    // domain
                    DataSpace<dim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();

                    auto accFunctor = functor(worker, localSuperCellOffset);

                    forEachParticle(accFunctor);
                }
            };

        } // namespace detail
    } // namespace acc

    /** Run a unary functor for each particle of a species in the given area
     *
     * Has a version for a fixed area, and for a user-provided mapper factory.
     * They differ only in how the area is defined.
     *
     * @warning Does NOT fill gaps automatically! If the
     *          operation deactivates particles or creates "gaps" in any
     *          other way, CallFillAllGaps needs to be called for the
     *          species manually afterwards!
     *
     * @tparam T_Species type of the species
     * @tparam T_Functor unary particle functor type which follows the interface of
     *                   pmacc::functor::Interface<F, 1u, void>
     *
     * @param species species to operate on
     * @param functor operation which is applied to each particle of the species
     *
     * @{
     */

    /** Version for a custom area mapper factory
     *
     * @tparam T_AreaMapperFactory factory type to construct an area mapper that defines the area to
     * process, adheres to the AreaMapperFactory concept
     *
     * @param areaMapperFactory factory to construct an area mapper,
     *                          the area is defined by the constructed mapper object
     */
    template<typename T_Species, typename T_Functor, typename T_AreaMapperFactory>
    [[nodiscard]] HINLINE auto forEach(
        T_Species&& species,
        T_Functor functor,
        T_AreaMapperFactory const& areaMapperFactory)
    {
        auto const mapper = areaMapperFactory(species.getCellDescription());
        return PMACC_LOCKSTEP_KERNEL(acc::detail::KernelForEachParticle{})
            .config(mapper.getGridDim(), species)(std::move(functor), mapper, species.getDeviceParticlesBox());
    }

    /** Version for a fixed area
     *
     * @tparam T_area area to process particles in
     */
    template<uint32_t T_area, typename T_Species, typename T_Functor>
    [[nodiscard]] HINLINE auto forEach(T_Species&& species, T_Functor functor)
    {
        return forEach(std::forward<T_Species>(species), std::move(functor), AreaMapperFactory<T_area>{});
    }

    /** @} */

} // namespace pmacc::particles::algorithm
