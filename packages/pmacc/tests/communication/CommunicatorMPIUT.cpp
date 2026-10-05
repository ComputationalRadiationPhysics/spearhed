/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/operation/Add.hpp>
#include <pmacc/mpi/GatherSlice.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <array>
#include <stdexcept>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("CommunicatorMPI consumes Caravan topology snapshots")
{
    auto& mpi = pmacc::Environment<>::get().getMpiContext();
    auto const topology = mpi.topology();
    auto& communicator = pmacc::Environment<TEST_DIM>::get().GridController().getCommunicator();
    if(communicator.getRank() != topology.rank || communicator.getSize() != topology.size
       || communicator.getHostRank() != static_cast<uint32_t>(topology.hostLocalRank))
        throw std::runtime_error("Unexpected topology snapshot");
    if(communicator.getCommunicatorId() == caravan::worldCommunicator)
        throw std::runtime_error("Cartesian communicator was not retained");
    if(communicator.getSignalCommunicatorId() == caravan::worldCommunicator
       || communicator.getSignalCommunicatorId() == communicator.getCommunicatorId())
        throw std::runtime_error("Signal communicator was not duplicated");

    std::array<std::uint32_t, 2> signalInput{static_cast<std::uint32_t>(topology.rank + 1), 1u};
    std::array<std::uint32_t, 2> signalOutput{};
    caravan::ControlContext context;
    auto signalReduction = context.spawnFuture<caravan::AllReduceResult>(communicator.signalAllReduce(
        signalInput.data(),
        signalOutput.data(),
        sizeof(signalInput),
        caravan::ScalarType::uint32,
        caravan::ReduceOperation::sum));
    context.wait(signalReduction.event());
    if(signalReduction.result().elements != signalInput.size()
       || signalOutput[0] != static_cast<std::uint32_t>(topology.size * (topology.size + 1) / 2)
       || signalOutput[1] != static_cast<std::uint32_t>(topology.size))
        throw std::runtime_error("Signal all-reduce sender failed");
    context.wait(context.spawn(communicator.barrier()));

    {
        pmacc::mpi::MPIReduce reduce;
        std::uint32_t local = static_cast<std::uint32_t>(topology.rank + 1);
        std::uint32_t global = 0u;
        context.wait(context.spawn(reduce.reduce(pmacc::math::operation::Add{}, &global, &local, 1u)));
        if(global != static_cast<std::uint32_t>(topology.size * (topology.size + 1) / 2))
            throw std::runtime_error("Caravan-backed PMacc all-reduce failed");

        global = 0u;
        context.wait(context.spawn(
            reduce.reduce(pmacc::math::operation::Add{}, &global, &local, 1u, pmacc::mpi::reduceMethods::Reduce{})));
        if(reduce.hasResult(pmacc::mpi::reduceMethods::Reduce{})
           && global != static_cast<std::uint32_t>(topology.size * (topology.size + 1) / 2))
            throw std::runtime_error("Caravan-backed PMacc root reduction failed");
    }

    {
        pmacc::mpi::GatherSlice gather;
        static_cast<void>(gather.participate(true));
        pmacc::HostBuffer<int, DIM2> local(pmacc::DataSpace<DIM2>{1, 1});
        local.data()[0] = topology.rank + 1;
        auto gathered = gather.gatherSliceExplicit(
            local,
            pmacc::DataSpace<DIM2>{topology.size, 1},
            pmacc::DataSpace<DIM2>{topology.rank, 0});
        if(gather.isMaster())
            for(int rank = 0; rank < topology.size; ++rank)
                if(gathered->data()[rank] != rank + 1)
                    throw std::runtime_error("Caravan-backed PMacc slice gather failed");
        if(!gather.isMaster() && gathered)
            throw std::runtime_error("Non-root PMacc slice gather returned data");
    }

    for(int exchange = 1; exchange < -12 * TEST_DIM + 6 * TEST_DIM * TEST_DIM + 9; ++exchange)
    {
        auto const direction = pmacc::Mask::getRelativeDirections<TEST_DIM>(exchange);
        int expected = (topology.rank + direction.x() + topology.size) % topology.size;
        if(communicator.ExchangeTypeToRank(exchange) != expected)
            throw std::runtime_error("Invalid periodic neighbor rank");
    }

    int sent = topology.rank;
    int received = -1;
    auto receive = context.spawnFuture<caravan::ReceiveResult>(
        communicator.receive(pmacc::LEFT, reinterpret_cast<char*>(&received), sizeof(int), 7u));
    auto send = context.spawnFuture<caravan::SendResult>(
        communicator.send(pmacc::RIGHT, reinterpret_cast<char*>(&sent), sizeof(int), 7u));
    std::array transfers{receive.event(), send.event()};
    context.wait(caravan::whenAll(transfers));
    if(received != (topology.rank + topology.size - 1) % topology.size || receive.result().bytes != sizeof(int)
       || send.result().bytes != sizeof(int))
        throw std::runtime_error("Caravan point-to-point adapter failed");
}
