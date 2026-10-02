/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <caravan/mpi/error.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace caravan
{
    using detail::mpiError;

    namespace
    {
        template<typename T_Buffer>
        bool validBuffer(T_Buffer const& buffer)
        {
            return buffer.value.size_bytes() <= static_cast<std::size_t>(INT_MAX);
        }

        bool buffersOverlap(ConstMpiBuffer const& input, MpiBuffer const& output) noexcept
        {
            if(input.value.size_bytes() == 0u || output.value.size_bytes() == 0u)
                return false;
            auto const inputAddress = reinterpret_cast<std::uintptr_t>(input.value.data());
            auto const outputAddress = reinterpret_cast<std::uintptr_t>(output.value.data());
            return inputAddress <= outputAddress ? outputAddress - inputAddress < input.value.size_bytes()
                                                 : inputAddress - outputAddress < output.value.size_bytes();
        }

        std::size_t scalarSize(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::int32:
            case ScalarType::uint32:
            case ScalarType::float32:
                return 4u;
            case ScalarType::int64:
            case ScalarType::uint64:
            case ScalarType::float64:
                return 8u;
            }
            return 0u;
        }

        MPI_Datatype nativeType(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::int32:
                return MPI_INT32_T;
            case ScalarType::uint32:
                return MPI_UINT32_T;
            case ScalarType::int64:
                return MPI_INT64_T;
            case ScalarType::uint64:
                return MPI_UINT64_T;
            case ScalarType::float32:
                return MPI_FLOAT;
            case ScalarType::float64:
                return MPI_DOUBLE;
            }
            throw std::invalid_argument("Unknown Caravan scalar type");
        }

        MPI_Op nativeOperation(ReduceOperation operation)
        {
            switch(operation)
            {
            case ReduceOperation::sum:
                return MPI_SUM;
            case ReduceOperation::minimum:
                return MPI_MIN;
            case ReduceOperation::maximum:
                return MPI_MAX;
            case ReduceOperation::product:
                return MPI_PROD;
            }
            throw std::invalid_argument("Unknown Caravan reduce operation");
        }

    } // namespace

    NativeRequestBatch detail::startSend(
        NativeMpiContext& context,
        ConstMpiBuffer const& buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || destination.any || destination.value < 0 || tag.any || tag.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI send");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.owner});
        int const error = MPI_Isend(
            buffer.value.data(),
            static_cast<int>(buffer.value.size_bytes()),
            MPI_BYTE,
            destination.value,
            tag.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Isend", error);
        return batch;
    }

    NativeRequestBatch detail::startReceive(
        NativeMpiContext& context,
        MpiBuffer const& buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || (!source.any && source.value < 0) || (!tag.any && tag.value < 0))
            throw std::invalid_argument("Invalid Caravan MPI receive");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.owner});
        int const error = MPI_Irecv(
            buffer.value.data(),
            static_cast<int>(buffer.value.size_bytes()),
            MPI_BYTE,
            source.any ? MPI_ANY_SOURCE : source.value,
            tag.any ? MPI_ANY_TAG : tag.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Irecv", error);
        return batch;
    }

    ReceiveResult detail::completeReceive(std::span<MPI_Status const> statuses)
    {
        int bytes = 0;
        int const error = MPI_Get_count(&statuses.front(), MPI_BYTE, &bytes);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Get_count", error);
        if(bytes == MPI_UNDEFINED)
            throw std::runtime_error("MPI_Get_count returned MPI_UNDEFINED");
        return ReceiveResult{
            Peer{statuses.front().MPI_SOURCE},
            MessageTag{statuses.front().MPI_TAG},
            static_cast<std::size_t>(bytes)};
    }

    std::size_t detail::scalarElements(std::size_t bytes, ScalarType type)
    {
        return bytes / scalarSize(type);
    }

    NativeRequestBatch detail::startAllReduce(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output)
           || input.value.size_bytes() % elementBytes != 0u || output.value.size_bytes() < input.value.size_bytes()
           || (input.value.data() != output.value.data() && buffersOverlap(input, output)))
            throw std::invalid_argument("Invalid Caravan MPI all-reduce");
        auto const elements = input.value.size_bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner});
        void const* const sendBuffer = input.value.data() == output.value.data() ? MPI_IN_PLACE : input.value.data();
        int const error = MPI_Iallreduce(
            sendBuffer,
            output.value.data(),
            static_cast<int>(elements),
            nativeType(type),
            nativeOperation(operation),
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Iallreduce", error);
        return batch;
    }

    NativeRequestBatch detail::startReduce(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || input.value.size_bytes() % elementBytes != 0u || root.any
           || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI reduce");
        auto const elements = input.value.size_bytes() / elementBytes;
        auto const native = context.communicator(communicator);
        int rank = -1;
        int error = MPI_Comm_rank(native, &rank);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI reduce communicator query", error);
        if(rank == root.value
           && (!validBuffer(output) || output.value.size_bytes() < input.value.size_bytes()
               || buffersOverlap(input, output)))
            throw std::invalid_argument("Invalid Caravan MPI reduce output");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner});
        error = MPI_Ireduce(
            input.value.data(),
            output.value.data(),
            static_cast<int>(elements),
            nativeType(type),
            nativeOperation(operation),
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ireduce", error);
        return batch;
    }

    NativeRequestBatch detail::startGather(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        Peer root,
        CommunicatorId communicator,
        std::size_t& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || buffersOverlap(input, output) || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI gather");

        auto const native = context.communicator(communicator);
        int rank = -1;
        int size = 0;
        int error = MPI_Comm_rank(native, &rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI gather communicator query", error);
        if(rank == root.value)
        {
            if(size <= 0 || input.value.size_bytes() > output.value.size_bytes() / static_cast<std::size_t>(size))
                throw std::invalid_argument("Caravan MPI gather output is too small");
            resultBytes = input.value.size_bytes() * static_cast<std::size_t>(size);
        }

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner});
        error = MPI_Igather(
            input.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            output.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igather", error);
        return batch;
    }

    NativeRequestBatch detail::startAllGather(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        CommunicatorId communicator,
        std::size_t& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || buffersOverlap(input, output))
            throw std::invalid_argument("Invalid Caravan MPI all-gather");

        auto const native = context.communicator(communicator);
        int size = 0;
        int error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI all-gather communicator query", error);
        if(size <= 0 || input.value.size_bytes() > output.value.size_bytes() / static_cast<std::size_t>(size))
            throw std::invalid_argument("Caravan MPI all-gather output is too small");
        resultBytes = input.value.size_bytes() * static_cast<std::size_t>(size);

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner});
        error = MPI_Iallgather(
            input.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            output.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Iallgather", error);
        return batch;
    }

    NativeRequestBatch detail::startGatherV(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        std::vector<std::size_t> const& receiveBytes,
        std::vector<std::size_t> const& displacements,
        Peer root,
        CommunicatorId communicator,
        std::size_t& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || buffersOverlap(input, output)
           || receiveBytes.size() != displacements.size() || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI variable gather");

        auto layout = std::make_shared<std::array<std::vector<int>, 2>>();
        auto& [counts, offsets] = *layout;
        counts.reserve(receiveBytes.size());
        offsets.reserve(displacements.size());
        for(std::size_t i = 0u; i < receiveBytes.size(); ++i)
        {
            if(receiveBytes[i] > static_cast<std::size_t>(INT_MAX)
               || displacements[i] > static_cast<std::size_t>(INT_MAX))
                throw std::invalid_argument("Invalid Caravan MPI variable gather layout");
            counts.emplace_back(static_cast<int>(receiveBytes[i]));
            offsets.emplace_back(static_cast<int>(displacements[i]));
        }

        auto const native = context.communicator(communicator);
        int rank = -1;
        int size = 0;
        int error = MPI_Comm_rank(native, &rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI variable gather communicator query", error);
        if(rank == root.value)
        {
            if(size <= 0 || counts.size() != static_cast<std::size_t>(size))
                throw std::invalid_argument("Invalid Caravan MPI variable gather rank count");
            for(std::size_t i = 0u; i < counts.size(); ++i)
            {
                auto const end = static_cast<std::size_t>(offsets[i]) + static_cast<std::size_t>(counts[i]);
                if(end > output.value.size_bytes())
                    throw std::invalid_argument("Caravan MPI variable gather output is too small");
                resultBytes += static_cast<std::size_t>(counts[i]);
            }
        }

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner, layout});
        error = MPI_Igatherv(
            input.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            output.value.data(),
            counts.data(),
            offsets.data(),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igatherv", error);
        return batch;
    }

    NativeRequestBatch detail::startBarrier(NativeMpiContext& context, CommunicatorId communicator)
    {
        NativeRequestBatch batch({MPI_REQUEST_NULL});
        int const error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ibarrier", error);
        return batch;
    }

} // namespace caravan
