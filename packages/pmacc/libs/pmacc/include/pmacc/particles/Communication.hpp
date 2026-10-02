/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/type/Exchange.hpp"

#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/core.hpp>

namespace pmacc::particles
{
    namespace detail
    {
        /** Skip empty chunks without constructing or launching a zero-sized insertion kernel. */
        template<typename T_Particles>
        struct InsertNonEmptySender
        {
            using completion_signatures = caravan::CompletionSignatures<caravan::ValueSignature<>>;

            template<typename T_Receiver>
            class Operation
            {
                using InsertSender = decltype(std::declval<T_Particles&>().insertParticles(
                    std::declval<uint32_t>(),
                    std::declval<size_t>()));

            public:
                Operation(InsertNonEmptySender sender, T_Receiver receiver)
                    : m_sender(sender)
                    , m_receiver(std::move(receiver))
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    if(m_sender.numIndexEntries == 0u)
                    {
                        m_receiver.set_value();
                        return;
                    }
                    m_insert.emplace(
                        m_sender.particles.insertParticles(m_sender.exchange, m_sender.numIndexEntries),
                        m_receiver);
                    m_insert->start();
                }

            private:
                InsertNonEmptySender m_sender;
                T_Receiver m_receiver;
                std::optional<caravan::detail::ConnectedOperation<InsertSender, T_Receiver>> m_insert;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{*this, std::forward<T_Receiver>(receiver)};
            }

            T_Particles& particles;
            uint32_t exchange;
            //! number of per-supercell exchange-index entries, not the particle payload count
            size_t numIndexEntries;
        };
    } // namespace detail

    template<typename T_Particles>
    [[nodiscard]] auto sendChunks(T_Particles& particles, uint32_t exchange)
    {
        return caravan::repeatUntil(
            [&particles, exchange, retries = size_t{0u}, lastSize = size_t{0u}]() mutable
            {
                auto const maxSize
                    = particles.getParticlesBuffer().getSendExchangeStack(exchange).getMaxParticlesCount();
                return particles.copyGuardToExchange(exchange)
                       | caravan::letValue(
                           [&particles, exchange, maxSize, &lastSize]
                           {
                               lastSize = particles.getParticlesBuffer()
                                              .getSendExchangeStack(exchange)
                                              .getDeviceParticlesCurrentSize();
                               PMACC_ASSERT(lastSize <= maxSize);
                               return particles.getParticlesBuffer().sendParticles(exchange);
                           })
                       | caravan::then(
                           [exchange, maxSize, &lastSize, &retries](auto&&...)
                           {
                               if(lastSize == maxSize)
                               {
                                   ++retries;
                                   return false;
                               }
                               if(retries != 0u)
                                   std::cerr << "Performance warning: send/receive buffer for species "
                                             << T_Particles::FrameType::getName() << " is too small (max: " << maxSize
                                             << ", direction: " << exchange << " '" << ExchangeTypeNames{}[exchange]
                                             << "', retries: " << retries
                                             << "). To remove this warning consider increasing BYTES_EXCHANGE_{X,Y,Z} "
                                                "in memory.param"
                                             << std::endl;
                               return true;
                           });
            });
    }

    template<typename T_Particles>
    [[nodiscard]] auto receiveChunks(T_Particles& particles, uint32_t exchange)
    {
        return caravan::repeatUntil(
            [&particles, exchange]
            {
                auto const maxSize
                    = particles.getParticlesBuffer().getReceiveExchangeStack(exchange).getMaxParticlesCount();
                return particles.getParticlesBuffer().receiveParticles(exchange)
                       | caravan::letValue(
                           [&particles, exchange, maxSize]
                           {
                               auto stack = particles.getParticlesBuffer().getReceiveExchangeStack(exchange);
                               /* The insertion kernel is launched with one block per
                                * exchange-index entry (one per source supercell), so it needs
                                * the stack-indexer size, not the received particle payload
                                * size. The payload size still decides whether another chunk
                                * must be received.
                                */
                               auto const payloadCount = stack.getHostParticlesCurrentSize();
                               auto const indexCount = stack.getHostCurrentSize();
                               PMACC_ASSERT(payloadCount <= maxSize);
                               return detail::InsertNonEmptySender<T_Particles>{particles, exchange, indexCount}
                                      | caravan::then([payloadCount, maxSize]
                                                      { return payloadCount == 0u || payloadCount < maxSize; });
                           });
            });
    }

    /** Eager runtime-sized adapter for all particle exchange directions. */
    template<typename T_Particles>
    [[nodiscard]] caravan::Event spawnCommunication(
        caravan::ControlContext& context,
        T_Particles& particles,
        caravan::Event previous = {})
    {
        using HandleGuardRegion = typename T_Particles::HandleGuardRegion;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;
        auto& buffer = particles.getParticlesBuffer();
        auto& device = Environment<>::get().DeviceContext();
        std::vector<caravan::Event> sends;
        std::vector<caravan::Event> receives;
        constexpr auto numExchanges = pmacc::traits::NumberOfExchanges<T_Particles::dim>::value;
        sends.reserve(numExchanges);
        receives.reserve(numExchanges);

        for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
        {
            if(buffer.hasSendExchange(exchange))
            {
                std::array dependencies{previous, buffer.sendCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(caravan::whenAll(dependencies))
                            | caravan::sequence(sendChunks(particles, exchange))));
                buffer.setSendCompletion(exchange, completion);
                sends.push_back(std::move(completion));
            }
            else
                sends.push_back(context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(previous)
                            | caravan::sequence(HandleNotExchanged{}.handleOutgoing(particles, exchange)))));

            if(buffer.hasReceiveExchange(exchange))
            {
                std::array dependencies{previous, buffer.receiveCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(caravan::whenAll(dependencies))
                            | caravan::sequence(receiveChunks(particles, exchange))));
                buffer.setReceiveCompletion(exchange, completion);
                receives.push_back(std::move(completion));
            }
            else
                receives.push_back(context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(previous)
                            | caravan::sequence(HandleNotExchanged{}.handleIncoming(particles, exchange)))));
        }

        auto received = caravan::whenAll(receives);
        auto filled = context.spawn(
            caravan::alpaka::withDevice(
                device,
                caravan::asSender(std::move(received)) | caravan::sequence(particles.fillBorderGaps())));
        sends.push_back(std::move(filled));
        return caravan::whenAll(sends);
    }
} // namespace pmacc::particles
