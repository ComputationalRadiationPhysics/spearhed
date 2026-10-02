/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <caravan/mpi/native.hpp>

namespace caravan::mpi
{
    inline auto send(
        MpiContext& context,
        ConstMpiBuffer buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator = worldCommunicator)
    {
        auto const bytes = buffer.value.size_bytes();
        return request<SendResult>(
            context,
            [buffer = std::move(buffer), destination, tag, communicator](NativeMpiContext& native)
            { return detail::startSend(native, buffer, destination, tag, communicator); },
            [bytes](std::span<MPI_Status const>) { return SendResult{bytes}; });
    }

    inline auto receive(
        MpiContext& context,
        MpiBuffer buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator = worldCommunicator)
    {
        return request<ReceiveResult>(
            context,
            [buffer = std::move(buffer), source, tag, communicator](NativeMpiContext& native)
            { return detail::startReceive(native, buffer, source, tag, communicator); },
            [](std::span<MPI_Status const> statuses) { return detail::completeReceive(statuses); });
    }

    inline auto allReduce(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator = worldCommunicator)
    {
        auto const bytes = input.value.size_bytes();
        return request<AllReduceResult>(
            context,
            [input = std::move(input), output = std::move(output), type, operation, communicator](
                NativeMpiContext& native)
            { return detail::startAllReduce(native, input, output, type, operation, communicator); },
            [bytes, type](std::span<MPI_Status const>)
            { return AllReduceResult{detail::scalarElements(bytes, type)}; },
            communicator);
    }

    inline auto reduce(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator = worldCommunicator)
    {
        auto const bytes = input.value.size_bytes();
        return request<ReduceResult>(
            context,
            [input = std::move(input), output = std::move(output), type, operation, root, communicator](
                NativeMpiContext& native)
            { return detail::startReduce(native, input, output, type, operation, root, communicator); },
            [bytes, type](std::span<MPI_Status const>) { return ReduceResult{detail::scalarElements(bytes, type)}; },
            communicator);
    }

    inline auto gather(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        Peer root,
        CommunicatorId communicator = worldCommunicator)
    {
        return request<GatherResult>(
            context,
            std::size_t{0u},
            [input = std::move(input), output = std::move(output), root, communicator](
                std::size_t& resultBytes,
                NativeMpiContext& native)
            { return detail::startGather(native, input, output, root, communicator, resultBytes); },
            [](std::size_t& resultBytes, std::span<MPI_Status const>) { return GatherResult{resultBytes}; },
            communicator);
    }

    inline auto allGather(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        CommunicatorId communicator = worldCommunicator)
    {
        return request<GatherResult>(
            context,
            std::size_t{0u},
            [input = std::move(input),
             output = std::move(output),
             communicator](std::size_t& resultBytes, NativeMpiContext& native)
            { return detail::startAllGather(native, input, output, communicator, resultBytes); },
            [](std::size_t& resultBytes, std::span<MPI_Status const>) { return GatherResult{resultBytes}; },
            communicator);
    }

    inline auto gatherV(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        std::vector<std::size_t> receiveBytes,
        std::vector<std::size_t> displacements,
        Peer root,
        CommunicatorId communicator = worldCommunicator)
    {
        return request<GatherResult>(
            context,
            std::size_t{0u},
            [input = std::move(input),
             output = std::move(output),
             receiveBytes = std::move(receiveBytes),
             displacements = std::move(displacements),
             root,
             communicator](std::size_t& resultBytes, NativeMpiContext& native)
            {
                return detail::startGatherV(
                    native,
                    input,
                    output,
                    receiveBytes,
                    displacements,
                    root,
                    communicator,
                    resultBytes);
            },
            [](std::size_t& resultBytes, std::span<MPI_Status const>) { return GatherResult{resultBytes}; },
            communicator);
    }

    inline auto barrier(MpiContext& context, CommunicatorId communicator = worldCommunicator)
    {
        return request<void>(
            context,
            [communicator](NativeMpiContext& native) { return detail::startBarrier(native, communicator); },
            [](std::span<MPI_Status const>) {},
            communicator);
    }

    inline auto createCartesian(MpiContext& context, std::vector<int> dimensions, std::vector<bool> periodic)
    {
        auto const topology = context.topology();
        return invoke(
            context,
            [dimensions = std::move(dimensions),
             periodic = std::move(periodic),
             worldSize = topology.size,
             hostLocalRank = topology.hostLocalRank](NativeMpiContext& native) mutable
            {
                return detail::createCartesian(
                    native,
                    std::move(dimensions),
                    std::move(periodic),
                    worldSize,
                    hostLocalRank);
            },
            worldCommunicator);
    }

    inline auto duplicateCommunicator(MpiContext& context, CommunicatorId communicator = worldCommunicator)
    {
        return invoke(
            context,
            [communicator](NativeMpiContext& native) { return detail::duplicateCommunicator(native, communicator); },
            communicator);
    }

    inline auto splitCommunicator(
        MpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator = worldCommunicator)
    {
        return invoke(
            context,
            [color, key, communicator](NativeMpiContext& native)
            { return detail::splitCommunicator(native, color, key, communicator); },
            communicator);
    }

    inline auto destroyCommunicator(MpiContext& context, CommunicatorId communicator)
    {
        return invoke(
            context,
            [communicator](NativeMpiContext& native) { detail::destroyCommunicator(native, communicator); },
            communicator);
    }
} // namespace caravan::mpi
