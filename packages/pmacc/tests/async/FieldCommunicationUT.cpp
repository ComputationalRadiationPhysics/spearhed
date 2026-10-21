/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>
#include <pmacc/fields/Communication.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    struct MockField
    {
        using SuperCellSize = typename pmacc::math::CT::shrinkTo<pmacc::math::CT::Int<1, 1, 1>, TEST_DIM>::type;
        using MappingDesc = pmacc::MappingDescription<TEST_DIM, SuperCellSize>;
        static constexpr uint32_t dim = TEST_DIM;

        pmacc::GridBuffer<int, TEST_DIM>& getGridBuffer()
        {
            return buffer;
        }

        pmacc::GridBuffer<int, TEST_DIM> buffer{
            pmacc::GridLayout<TEST_DIM>{pmacc::DataSpace<TEST_DIM>::create(2), pmacc::DataSpace<TEST_DIM>::create(1)}};
    };

    [[maybe_unused]] auto compileFieldSend(MockField& field)
    {
        auto sender = pmacc::fields::sendExchange(field, 1u);
        static_assert(caravan::Sender<decltype(sender)>);
        return sender;
    }

    [[maybe_unused]] auto compileFieldReceive(MockField& field)
    {
        auto sender = pmacc::fields::receiveExchange(field, 1u);
        static_assert(caravan::Sender<decltype(sender)>);
        return sender;
    }
} // namespace

TEST_CASE("Field communication preserves overwrite and additive semantics", "[async][fields]")
{
    bool additive = false;
    SECTION("Border-to-guard halo overwrites")
    {
    }
    SECTION("Guard-to-border scatter adds")
    {
        additive = true;
    }

    MockField field;
    auto& buffer = field.getGridBuffer();
    auto const tag = pmacc::traits::getUniqueId<uint32_t>();
    for(uint32_t exchange = 1u; exchange < pmacc::traits::NumberOfExchanges<TEST_DIM>::value; ++exchange)
    {
        auto const direction = pmacc::Mask::getRelativeDirections<TEST_DIM>(exchange);
        auto extent = pmacc::DataSpace<TEST_DIM>::create(1);
        if(additive)
        {
            for(uint32_t d = 0u; d < TEST_DIM; ++d)
                if(direction[d] == 0)
                    extent[d] = 2;
            buffer.addExchangeBuffer(exchange, extent, tag);
        }
        else
            buffer.addExchange(pmacc::GUARD, exchange, extent, tag);
    }

    auto& device = pmacc::Environment<>::get().DeviceContext();
    auto const topology = pmacc::Environment<>::get().getMpiContext().topology();
    auto const extent = buffer.getGridLayout().sizeND();
    auto box = buffer.getHostBuffer().getDataBox();
    caravan::ControlContext context;
    for(int step = 0; step < 2; ++step)
    {
        for(int i = 0; i < extent.productOfComponents(); ++i)
        {
            auto const cell = pmacc::math::mapToND(extent, i);
            bool guard = false;
            for(uint32_t d = 0u; d < TEST_DIM; ++d)
                guard = guard || cell[d] == 0 || cell[d] == 3;
            box(cell) = guard ? (additive ? 1 : -100) : (additive ? 10 : topology.rank + 1 + step * 10);
        }
        context.wait(context.spawn(caravan::alpaka::withDevice(device, buffer.hostToDevice())));
        context.wait(
            additive ? pmacc::fields::spawnCommunication(context, field) : buffer.spawnCommunication(context));
        context.wait(context.spawn(caravan::alpaka::withDevice(device, buffer.deviceToHost())));

        for(int i = 0; i < extent.productOfComponents(); ++i)
        {
            auto const cell = pmacc::math::mapToND(extent, i);
            bool guard = false;
            for(uint32_t d = 0u; d < TEST_DIM; ++d)
                guard = guard || cell[d] == 0 || cell[d] == 3;
            auto const neighbor
                = (topology.rank + (cell.x() == 0 ? -1 : (cell.x() == 3 ? 1 : 0)) + topology.size) % topology.size;
            CHECK(box(cell) == (additive ? (guard ? 1 : 10 + (1 << TEST_DIM) - 1) : neighbor + 1 + step * 10));
        }
    }
}

TEST_CASE("Field communication without exchanges preserves previous work", "[async][fields]")
{
    bool additive = false;
    SECTION("overwrite")
    {
    }
    SECTION("additive")
    {
        additive = true;
    }

    MockField field;
    caravan::ControlContext context;
    caravan::EventSource previous;
    auto communication = additive ? pmacc::fields::spawnCommunication(context, field, previous.event())
                                  : field.buffer.spawnCommunication(context, previous.event());
    CHECK_FALSE(communication.isReady());
    previous.setReady();
    context.wait(communication);
}
