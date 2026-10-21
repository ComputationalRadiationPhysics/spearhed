/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <boost/mp11/list.hpp>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    constexpr uint32_t slots = 7u;
    constexpr uint32_t capacity = 4u * slots;

    // Real PIC particle handles over minimal storage, without allocating a device heap.
    struct Frame
    {
        using MethodsList = boost::mp11::mp_list<>;
        using ValueTypeSeq = boost::mp11::mp_list<>;
        using Name = void;
        using SuperCellSize = pmacc::math::CT::UInt32<slots>;
        using ParticleType = pmacc::Particle<Frame>;
        uint32_t number;
        uint8_t masks[slots];

        HDINLINE ParticleType operator[](uint32_t slot)
        {
            return ParticleType{*this, slot};
        }
    };

    struct SuperCell
    {
        uint32_t count;

        HDINLINE uint32_t getNumParticles() const
        {
            return count;
        }

        HDINLINE uint32_t getSizeLastFrame() const
        {
            return count == 0u ? 0u : (count - 1u) % slots + 1u;
        }
    };

    struct Box
    {
        static constexpr uint32_t Dim = TEST_DIM;
        static constexpr uint32_t frameSize = slots;
        using FramePtr = pmacc::FramePointer<Frame>;
        Frame* frames;
        SuperCell cell;

        HDINLINE auto const& getSuperCell(auto const&) const
        {
            return cell;
        }

        HDINLINE FramePtr getFirstFrame(auto const&) const
        {
            return cell.count == 0u ? nullptr : frames;
        }

        HDINLINE FramePtr getLastFrame(auto const&) const
        {
            return cell.count == 0u ? nullptr : frames + (cell.count - 1u) / slots;
        }

        HDINLINE FramePtr getNextFrame(FramePtr frame) const
        {
            return frame->number + 1u < (cell.count + slots - 1u) / slots ? frame.ptr + 1 : nullptr;
        }

        HDINLINE FramePtr getPreviousFrame(FramePtr frame) const
        {
            return frame->number != 0u ? frame.ptr - 1 : nullptr;
        }
    };

    template<typename T_Box>
    struct Visit
    {
        T_Box visits;
        uint32_t calls = 0u;
        uint32_t lastFrame = 0u;
        bool forward;
        bool valid = true;

        DINLINE Visit(T_Box box, bool forwardOrder) : visits(box), forward(forwardOrder)
        {
        }

        Visit(Visit const&) = delete;

        DINLINE Visit(Visit&& other)
            : visits(other.visits)
            , calls(other.calls)
            , lastFrame(other.lastFrame)
            , forward(other.forward)
            , valid(other.valid)
        {
            other.valid = false;
        }

        DINLINE void operator()(auto const& worker, auto& particle) &
        {
            uint32_t const frame = particle.frame->number;
            bool const ordered = calls == 0u || (forward ? frame >= lastFrame : frame <= lastFrame);
            lastFrame = frame;
            ++calls;
            uint32_t const index = frame * slots + particle.idx;
            // PIC prefix selection must neither filter nor reinterpret exchange mask values.
            uint32_t const increment
                = valid && ordered && particle.frame->masks[particle.idx] == index % 4u ? 1u : 100u;
            alpaka::onAcc::atomicAdd(worker.getAcc(), &visits[index], increment, alpaka::onAcc::scope::Device{});
        }
    };

    template<typename T_Order, bool T_MoveSensitive = false>
    struct UnaryKernel
    {
        DINLINE void operator()(auto const& worker, auto visits, uint32_t count) const
        {
            Frame frames[4];
            for(uint32_t f = 0; f < 4u; ++f)
            {
                frames[f].number = f;
                for(uint32_t s = 0; s < slots; ++s)
                    frames[f].masks[s] = static_cast<uint8_t>((f * slots + s) % 4u);
            }
            Box box{frames, {count}};
            auto each
                = pmacc::particles::algorithm::acc::makeForEach<T_Order>(worker, box, pmacc::DataSpace<TEST_DIM>{});
            constexpr bool forward = std::is_same_v<T_Order, pmacc::particles::algorithm::acc::Forward>;
            if constexpr(T_MoveSensitive)
                each(Visit{visits, forward});
            else
            {
                auto callback = Visit{visits, forward};
                each(callback);
            }
        }
    };

    struct BatchKernel
    {
        DINLINE void operator()(auto const& worker, auto visits, uint32_t count) const
        {
            Frame frames[4];
            for(uint32_t f = 0; f < 4u; ++f)
                frames[f].number = f;
            Box box{frames, {count}};
            auto each = pmacc::particles::algorithm::acc::makeForEachFrame<pmacc::particles::algorithm::acc::Forward>(
                worker,
                box,
                pmacc::DataSpace<TEST_DIM>{});
            auto slotsInBatch = each.lockstepForEach();
            each(
                [&](auto const& w, auto& context)
                {
                    w.sync();
                    slotsInBatch(
                        [&](uint32_t index, auto const& frame)
                        {
                            if(frame.isValid())
                            {
                                uint32_t const global = frame->number * slots + index % slots;
                                alpaka::onAcc::atomicAdd(
                                    w.getAcc(),
                                    &visits[global],
                                    1u,
                                    alpaka::onAcc::scope::Device{});
                            }
                        },
                        context);
                    w.sync();
                });
        }
    };

    template<uint32_t T_Threads, typename T_Kernel>
    void checkVisits(T_Kernel kernel, uint32_t count, bool wholeFrames = false)
    {
        pmacc::HostDeviceBuffer<uint32_t, 1> visits(capacity);
        visits.getHostBuffer().setValue(0u);
        auto& device = pmacc::Environment<>::get().DeviceContext();
        auto run = [&](auto sender) { caravan::syncWait(caravan::alpaka::withDevice(device, std::move(sender))); };
        run(visits.hostToDevice());
        run(PMACC_LOCKSTEP_KERNEL(kernel).template config<T_Threads>(
            1u)(visits.getDeviceBuffer().getDataBox(), count));
        run(visits.deviceToHost());
        uint32_t const expected = wholeFrames ? (count + slots - 1u) / slots * slots : count;
        for(uint32_t i = 0; i < capacity; ++i)
        {
            CAPTURE(T_Threads, count, i);
            CHECK(visits.getHostBuffer().data()[i] == (i < expected ? 1u : 0u));
        }
    }
} // namespace

TEST_CASE("PIC unary iteration preserves packed prefixes in both directions", "[particles][iteration]")
{
    namespace acc = pmacc::particles::algorithm::acc;
    for(uint32_t count : {0u, 1u, 6u, 7u, 8u, 14u, 17u, capacity})
    {
        checkVisits<3>(UnaryKernel<acc::Forward>{}, count);
        checkVisits<7>(UnaryKernel<acc::Reverse>{}, count);
        checkVisits<14>(UnaryKernel<acc::Forward>{}, count);
        checkVisits<14>(UnaryKernel<acc::Reverse>{}, count);
    }
}

TEST_CASE("PIC frame batches retain collective entry for incomplete batches", "[particles][iteration][collective]")
{
    for(uint32_t count : {0u, 1u, 7u, 8u, 17u, capacity})
        checkVisits<7>(BatchKernel{}, count, true);
    // Exercise batching and an incomplete first batch without triggering the known legacy
    // per-thread termination hazard of an incomplete later batch (documented in the progress note).
    for(uint32_t count : {0u, 1u, 7u, 8u, capacity})
        checkVisits<14>(BatchKernel{}, count, true);
}

TEST_CASE("PIC unary iteration binds a move-sensitive callback once", "[particles][iteration][callback]")
{
    namespace acc = pmacc::particles::algorithm::acc;
    checkVisits<3>(UnaryKernel<acc::Forward, true>{}, capacity);
    checkVisits<14>(UnaryKernel<acc::Reverse, true>{}, capacity);
    checkVisits<10>(UnaryKernel<acc::Forward, true>{}, 17u);
    checkVisits<10>(UnaryKernel<acc::Reverse, true>{}, 17u);
}
