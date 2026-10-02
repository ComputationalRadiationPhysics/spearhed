/* Copyright 2016-2024 Alexander Grund
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

#include <pmacc/boost_workaround.hpp>

#include "IdProvider.hpp"
#include "memory/SuperCell.hpp"

#include <pmacc/HandleGuardRegion.hpp>
#include <pmacc/fields/Communication.hpp>
#include <pmacc/particles/Communication.hpp>
#include <pmacc/particles/policies/DeleteParticles.hpp>
#include <pmacc/particles/policies/DoNothing.hpp>

#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    /** Release one pending test gate before its owning ControlContext unwinds. */
    struct ReleaseGate
    {
        caravan::EventSource& source;

        ~ReleaseGate()
        {
            source.setReady();
        }
    };

    /** Release two pending test gates before their owning ControlContext unwinds. */
    struct ReleaseGates2
    {
        caravan::EventSource& first;
        caravan::EventSource& second;

        ~ReleaseGates2()
        {
            first.setReady();
            second.setReady();
        }
    };

    struct MockStack
    {
        size_t getMaxParticlesCount() const
        {
            return 2u;
        }

        size_t getDeviceParticlesCurrentSize() const
        {
            return *size;
        }

        size_t getHostParticlesCurrentSize() const
        {
            return *size;
        }

        size_t getHostCurrentSize() const
        {
            return *size;
        }

        size_t const* size;
    };

    struct MockParticlesBuffer
    {
        bool hasSendExchange(uint32_t exchange) const
        {
            return exchange == 1u;
        }

        bool hasReceiveExchange(uint32_t exchange) const
        {
            return exchange == 1u;
        }

        MockStack getSendExchangeStack(uint32_t) const
        {
            return {&sendSize};
        }

        MockStack getReceiveExchangeStack(uint32_t) const
        {
            return {&receiveSize};
        }

        caravan::Event sendCompletion(uint32_t) const
        {
            return sendTail;
        }

        caravan::Event receiveCompletion(uint32_t) const
        {
            return receiveTail;
        }

        void setSendCompletion(uint32_t, caravan::Event completion)
        {
            sendTail = std::move(completion);
        }

        void setReceiveCompletion(uint32_t, caravan::Event completion)
        {
            receiveTail = std::move(completion);
        }

        auto sendParticles(uint32_t)
        {
            auto sent = caravan::alpaka::submit([this](auto&) { sentChunks.push_back(sendSize); });
            // Particle data and frame indices each produce MPI send metadata.
            return caravan::whenAll(
                std::move(sent) | caravan::then([] { return caravan::SendResult{0u}; }),
                caravan::InlineScheduler{}.schedule() | caravan::then([] { return caravan::SendResult{0u}; }));
        }

        auto receiveParticles(uint32_t)
        {
            return caravan::alpaka::submit([this](auto&) { receiveSize = receiveChunks.at(receiveChunk++); });
        }

        std::vector<size_t> sendChunks{2u, 1u};
        std::vector<size_t> receiveChunks{2u, 1u};
        std::vector<size_t> sentChunks;
        size_t sendChunk = 0u;
        size_t receiveChunk = 0u;
        size_t sendSize = 0u;
        size_t receiveSize = 0u;
        caravan::Event sendTail;
        caravan::Event receiveTail;
    };

    struct MockParticles
    {
        using HandleGuardRegion
            = pmacc::HandleGuardRegion<pmacc::particles::policies::DoNothing, pmacc::particles::policies::DoNothing>;

        struct FrameType
        {
            static char const* getName()
            {
                return "mock";
            }
        };

        static constexpr uint32_t dim = TEST_DIM;

        MockParticlesBuffer& getParticlesBuffer()
        {
            return buffer;
        }

        auto copyGuardToExchange(uint32_t)
        {
            buffer.sendSize = buffer.sendChunks.at(buffer.sendChunk++);
            return caravan::asSender(caravan::readyEvent());
        }

        auto insertParticles(uint32_t, size_t count)
        {
            inserted += count;
            insertedChunks.push_back(count);
            return caravan::asSender(caravan::readyEvent());
        }

        auto fillBorderGaps()
        {
            return caravan::alpaka::submit([this](auto&) { gapsFilled = true; });
        }

        MockParticlesBuffer buffer;
        size_t inserted = 0u;
        std::vector<size_t> insertedChunks;
        bool gapsFilled = false;
    };

    /** Mock particles recording guard deletions initiated by a policy. */
    struct PolicyMockParticles
    {
        int deletions = 0;

        auto deleteGuardParticles(int32_t)
        {
            return caravan::alpaka::submit([this](auto&) { ++deletions; });
        }
    };
} // namespace

TEST_CASE("Particle chunk senders are lazy", "[particles][async]")
{
    auto& device = pmacc::Environment<>::get().DeviceContext();
    MockParticles particles;
    caravan::ControlContext context;
    auto sender = pmacc::particles::sendChunks(particles, 1u);
    static_assert(caravan::Sender<decltype(sender)>);
    static_assert(caravan::Sender<decltype(pmacc::particles::receiveChunks(particles, 1u))>);
    CHECK(particles.buffer.sendChunk == 0u);
    context.wait(context.spawn(caravan::alpaka::withDevice(device, std::move(sender))));
    CHECK(particles.buffer.sendChunk == 2u);
}

TEST_CASE("Particle communication handles exact and partial chunks", "[particles][async]")
{
    MockParticles particles;
    caravan::ControlContext context;
    context.wait(pmacc::particles::spawnCommunication(context, particles));
    CHECK(particles.buffer.sendChunk == 2u);
    CHECK(particles.buffer.receiveChunk == 2u);
    CHECK(particles.inserted == 3u);
    CHECK(particles.buffer.sentChunks == std::vector<size_t>{2u, 1u});
    CHECK(particles.insertedChunks == std::vector<size_t>{2u, 1u});
    CHECK(particles.gapsFilled);
}

TEST_CASE("Particle communication handles empty chunks", "[particles][async]")
{
    MockParticles particles;
    caravan::ControlContext context;
    particles.buffer.sendChunks = {0u, 0u};
    particles.buffer.receiveChunks = {0u, 0u};
    context.wait(pmacc::particles::spawnCommunication(context, particles));
    CHECK(particles.buffer.sendChunk == 1u);
    CHECK(particles.buffer.receiveChunk == 1u);
    CHECK(particles.inserted == 0u);
    CHECK(particles.buffer.sentChunks == std::vector<size_t>{0u});
    CHECK(particles.insertedChunks.empty());
    CHECK(particles.gapsFilled);
}

TEST_CASE("Full particle chunks require an empty terminator", "[particles][async]")
{
    for(size_t fullChunks : {1u, 32u})
    {
        MockParticles particles;
        caravan::ControlContext context;
        particles.buffer.sendChunks.assign(fullChunks, 2u);
        particles.buffer.sendChunks.push_back(0u);
        particles.buffer.receiveChunks = particles.buffer.sendChunks;
        context.wait(pmacc::particles::spawnCommunication(context, particles));
        CHECK(particles.buffer.sendChunk == fullChunks + 1u);
        CHECK(particles.buffer.receiveChunk == fullChunks + 1u);
        CHECK(particles.buffer.sentChunks == particles.buffer.sendChunks);
        CHECK(particles.insertedChunks == std::vector<size_t>(fullChunks, 2u));
        CHECK(particles.inserted == fullChunks * 2u);
        CHECK(particles.gapsFilled);
    }
}

TEST_CASE("Particle communication waits for its predecessor", "[particles][async]")
{
    MockParticles particles;
    caravan::ControlContext context;
    caravan::EventSource push;
    ReleaseGate release{push};
    auto communication = pmacc::particles::spawnCommunication(context, particles, push.event());

    // Nothing may start while the push predecessor is pending.
    context.runReady();
    CHECK(particles.buffer.sendChunk == 0u);
    CHECK(particles.buffer.sentChunks.empty());
    CHECK(particles.inserted == 0u);

    push.setReady();
    context.wait(communication);
    CHECK(particles.buffer.sendChunk == 2u);
    CHECK(particles.inserted == 3u);
    CHECK(particles.gapsFilled);
}

TEST_CASE("Independent species communication can advance separately", "[particles][async]")
{
    MockParticles first;
    MockParticles second;
    caravan::ControlContext context;
    caravan::EventSource firstPush;
    caravan::EventSource secondPush;
    ReleaseGates2 release{firstPush, secondPush};
    auto firstCommunication = pmacc::particles::spawnCommunication(context, first, firstPush.event());
    auto secondCommunication = pmacc::particles::spawnCommunication(context, second, secondPush.event());

    firstPush.setReady();
    context.wait(firstCommunication);
    CHECK(first.buffer.sendChunk == 2u);
    // The second species must not advance while its own push is still pending.
    CHECK(second.buffer.sendChunk == 0u);

    secondPush.setReady();
    context.wait(secondCommunication);
    CHECK(second.buffer.sendChunk == 2u);
    CHECK(second.inserted == 3u);
}

TEST_CASE("Particle exchange storage is reused only after the previous exchange", "[particles][async]")
{
    MockParticles particles;
    caravan::ControlContext context;
    particles.buffer.sendChunks = {2u, 0u, 2u, 0u};
    particles.buffer.receiveChunks = {2u, 0u, 2u, 0u};
    caravan::EventSource firstPush;
    caravan::EventSource secondPush;
    ReleaseGates2 release{firstPush, secondPush};
    auto first = pmacc::particles::spawnCommunication(context, particles, firstPush.event());
    /* Start the reuse with an already-ready predecessor: it must still wait for the first
     * exchange's buffer-reuse tail before touching the shared exchange storage.
     */
    auto second = pmacc::particles::spawnCommunication(context, particles, secondPush.event());
    secondPush.setReady();

    context.runReady();
    CHECK(particles.buffer.sendChunk == 0u);

    firstPush.setReady();
    context.wait(first);
    context.wait(second);
    CHECK(particles.buffer.sendChunk == 4u);
    CHECK(particles.inserted == 4u);
}

TEST_CASE("Guard region policies return lazy senders", "[particles][policies]")
{
    using pmacc::particles::policies::DeleteParticles;
    using pmacc::particles::policies::DoNothing;

    auto& device = pmacc::Environment<>::get().DeviceContext();
    PolicyMockParticles particles;

    auto doNothingOutgoing = DoNothing{}.handleOutgoing(particles, 1);
    auto doNothingIncoming = DoNothing{}.handleIncoming(particles, 1);
    static_assert(caravan::Sender<decltype(doNothingOutgoing)>);
    static_assert(caravan::Sender<decltype(doNothingIncoming)>);
    CHECK(particles.deletions == 0);

    auto deletion = DeleteParticles{}.handleOutgoing(particles, 1);
    static_assert(caravan::Sender<decltype(deletion)>);
    static_assert(caravan::Sender<decltype(DeleteParticles{}.handleIncoming(particles, 1))>);
    // Deletion is deferred: constructing the sender must not perform it.
    CHECK(particles.deletions == 0);

    caravan::syncWait(caravan::alpaka::withDevice(device, std::move(deletion)));
    CHECK(particles.deletions == 1);
}
