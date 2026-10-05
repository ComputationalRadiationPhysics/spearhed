/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/fields/operations/AddExchangeToBorder.hpp"
#include "pmacc/fields/operations/CopyGuardToExchange.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

#include <array>

#include <caravan/core.hpp>

namespace pmacc::fields
{
    /** Describe one lazy field receive/insert branch. */
    template<typename T_Field>
    [[nodiscard]] auto receiveExchange(T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return buffer.receive(exchange)
               | caravan::sequence(operations::AddExchangeToBorder{}.sender(buffer, SuperCellSize{}, exchange));
    }

    /** Describe one lazy field pack/send branch. */
    template<typename T_Field>
    [[nodiscard]] auto sendExchange(T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return operations::CopyGuardToExchange{}.sender(buffer, SuperCellSize{}, exchange)
               | caravan::sequence(buffer.send(exchange));
    }

    /** Eager runtime-sized adapter for additive guard-to-border field communication.
     *
     * The returned event includes previous, even when this rank has no active exchanges.
     */
    template<typename T_Field>
    [[nodiscard]] caravan::Event spawnCommunication(
        caravan::ControlContext& context,
        T_Field& field,
        caravan::Event previous = {})
    {
        auto& buffer = field.getGridBuffer();
        auto& device = Environment<>::get().DeviceContext();
        // Keep only the tail of the ordered receives, plus the independent sends.
        std::array<caravan::Event, traits::NumberOfExchanges<T_Field::dim>::value> branches{};
        branches.front() = previous;
        auto receivePrevious = previous;

        for(uint32_t exchange = 1u; exchange < traits::NumberOfExchanges<T_Field::dim>::value; ++exchange)
        {
            if(buffer.hasReceiveExchange(exchange))
            {
                // Inserts use += and neighboring directions overlap at edges; retain their former FIFO ordering.
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::whenAll(
                            caravan::asSender(receivePrevious),
                            caravan::asSender(buffer.receiveCompletion(exchange)))
                            | caravan::sequence(receiveExchange(field, exchange))));
                buffer.setReceiveCompletion(exchange, completion);
                receivePrevious = completion;
                branches.front() = std::move(completion);
            }

            if(buffer.hasSendExchange(exchange))
            {
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::whenAll(
                            caravan::asSender(previous),
                            caravan::asSender(buffer.sendCompletion(exchange)))
                            | caravan::sequence(sendExchange(field, exchange))));
                buffer.setSendCompletion(exchange, completion);
                branches[exchange] = std::move(completion);
            }
        }
        return caravan::whenAll(branches);
    }
} // namespace pmacc::fields
