/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "spearhed/test/SpearhedParticleFixture.hpp"

#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <spmacc/particles/algorithms/FrameSchedule.hpp>

#include <algorithm>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

namespace sp = pmacc::spearhed;

namespace
{
    constexpr uint32_t slots = 17u;

    struct SparseFrame
    {
        static constexpr uint32_t frameSize = slots;
        uint8_t masks[slots];

        struct Particle
        {
            uint32_t slot;
            uint8_t mask;

            DINLINE uint8_t operator[](sp::tags::multiMask_t) const
            {
                return mask;
            }
        };

        DINLINE Particle operator[](uint32_t slot) const
        {
            return {slot, masks[slot]};
        }
    };

    DINLINE bool occupied(uint32_t pattern, uint32_t slot)
    {
        return pattern == 1u ? slot == 0u : pattern == 2u || (pattern == 3u && slot % 3u == 1u);
    }

    template<typename T_Box, typename T_Worker>
    struct StatefulVisit
    {
        T_Box output;
        T_Worker const& worker;
        uint32_t calls = 0u;

        DINLINE StatefulVisit(T_Box box, T_Worker const& w) : output(box), worker(w)
        {
        }

        StatefulVisit(StatefulVisit const&) = delete;

        StatefulVisit(StatefulVisit&&) = delete;

        DINLINE void operator()(SparseFrame::Particle particle) &
        {
            ++calls;
            alpaka::onAcc::atomicAdd(worker.getAcc(), &output[particle.slot], 1u, alpaka::onAcc::scope::Device{});
        }
    };

    struct SparseKernel
    {
        DINLINE void operator()(auto const& worker, auto output, uint32_t pattern) const
        {
            SparseFrame frame;
            for(uint32_t slot = 0; slot < slots; ++slot)
                frame.masks[slot] = occupied(pattern, slot) ? uint8_t{1} : uint8_t{0};
            auto body = StatefulVisit{output, worker};
            sp::lockstepForEachParticle(worker, frame, body);
            output[slots + worker.workerIdx()] = body.calls;
        }
    };

    template<uint32_t T_Threads>
    void checkSparse()
    {
        constexpr uint32_t workers = pmacc::traits::GetNumWorkers<T_Threads>::value;
        pmacc::HostDeviceBuffer<uint32_t, 1> output(slots + workers);
        for(uint32_t pattern : {0u, 1u, 2u, 3u})
        {
            output.getHostBuffer().setValue(0u);
            spearhed::test::runDevice(output.hostToDevice());
            spearhed::test::runDevice(PMACC_LOCKSTEP_KERNEL(SparseKernel{})
                                          .config<T_Threads>(1u)(output.getDeviceBuffer().getDataBox(), pattern));
            spearhed::test::runDevice(output.deviceToHost());
            auto data = output.getHostBuffer().data();
            for(uint32_t slot = 0; slot < slots; ++slot)
            {
                bool const live = pattern == 1u ? slot == 0u : pattern == 2u || (pattern == 3u && slot % 3u == 1u);
                CAPTURE(T_Threads, pattern, slot);
                CHECK(data[slot] == (live ? 1u : 0u));
            }
            for(uint32_t w = 0; w < workers; ++w)
            {
                uint32_t expected = 0u;
                for(uint32_t s = w; s < slots; s += workers)
                    expected += pattern == 1u ? s == 0u : pattern == 2u || (pattern == 3u && s % 3u == 1u);
                CHECK(data[slots + w] == expected);
            }
        }
    }

    struct ScheduleKernel
    {
        DINLINE void operator()(auto const& worker, auto output, int frames, auto schedule) const
        {
            schedule(
                worker,
                frames,
                [](int f) { return f; },
                [&](int f)
                {
                    // A frame callback is collective, unlike a selected-particle callback.
                    worker.sync();
                    uint32_t const index = 2u * (static_cast<uint32_t>(f) * worker.numWorkers() + worker.workerIdx());
                    alpaka::onAcc::atomicAdd(worker.getAcc(), &output[index], 1u, alpaka::onAcc::scope::Device{});
                    alpaka::onAcc::atomicAdd(
                        worker.getAcc(),
                        &output[index + 1u],
                        worker.blockDomIdx() + 1u,
                        alpaka::onAcc::scope::Device{});
                    worker.sync();
                });
        }
    };

    template<uint32_t T_Threads, typename T_Schedule>
    void checkSchedule(T_Schedule schedule)
    {
        constexpr uint32_t workers = pmacc::traits::GetNumWorkers<T_Threads>::value;
        pmacc::HostDeviceBuffer<uint32_t, 1> output(2u * slots * workers);
        for(int frames : {0, 1, 2, 7, 17})
            for(uint32_t blocks : {1u, 2u, 3u, 7u, 8u, 17u, 24u})
            {
                if constexpr(std::is_same_v<T_Schedule, sp::OneToOne>)
                {
                    if(frames == 0 || blocks != static_cast<uint32_t>(frames))
                        continue;
                }
                output.getHostBuffer().setValue(0u);
                spearhed::test::runDevice(output.hostToDevice());
                spearhed::test::runDevice(
                    PMACC_LOCKSTEP_KERNEL(ScheduleKernel{})
                        .config<T_Threads>(blocks)(output.getDeviceBuffer().getDataBox(), frames, schedule));
                spearhed::test::runDevice(output.deviceToHost());
                for(uint32_t f = 0; f < slots; ++f)
                {
                    uint32_t owner = f % blocks;
                    if constexpr(std::is_same_v<T_Schedule, sp::Contiguous>)
                    {
                        uint32_t const total = static_cast<uint32_t>(frames);
                        for(uint32_t b = 0; b < blocks; ++b)
                        {
                            uint32_t const first = b * (total / blocks) + std::min(b, total % blocks);
                            uint32_t const last = first + total / blocks + (b < total % blocks ? 1u : 0u);
                            if(f >= first && f < last)
                                owner = b;
                        }
                    }
                    for(uint32_t w = 0; w < workers; ++w)
                    {
                        CAPTURE(T_Threads, frames, blocks, f, w);
                        uint32_t const index = 2u * (f * workers + w);
                        bool const active = f < static_cast<uint32_t>(frames);
                        CHECK(output.getHostBuffer().data()[index] == (active ? 1u : 0u));
                        CHECK(output.getHostBuffer().data()[index + 1u] == (active ? owner + 1u : 0u));
                    }
                }
            }
    }
} // namespace

using IterationFixture = spearhed::test::SpearhedParticleFixture<spearhed::simDim>;

TEST_CASE_METHOD(
    IterationFixture,
    "Alpaka slot iteration filters sparse masks and preserves callback state",
    "[particles][iteration]")
{
    checkSparse<3>();
    checkSparse<17>();
    checkSparse<32>();
}

TEST_CASE_METHOD(
    IterationFixture,
    "Alpaka frame schedules assign every frame uniformly across block threads",
    "[particles][iteration]")
{
    checkSchedule<3>(sp::GridStride{});
    checkSchedule<17>(sp::Contiguous{});
    checkSchedule<32>(sp::OneToOne{});
}
