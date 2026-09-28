/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <caravan/mpi/error.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace caravan
{
    using detail::mpiError;

    TopologySnapshot detail::createCartesian(
        NativeMpiContext& context,
        std::vector<int> dimensions,
        std::vector<bool> periodic,
        int worldSize,
        int hostLocalRank)
    {
        std::size_t ranks = 1u;
        for(int dimension : dimensions)
        {
            if(dimension <= 0 || ranks > static_cast<std::size_t>(worldSize) / static_cast<std::size_t>(dimension))
                throw std::invalid_argument("Invalid Cartesian topology dimensions");
            ranks *= static_cast<std::size_t>(dimension);
        }
        if(dimensions.empty() || dimensions.size() != periodic.size() || ranks != static_cast<std::size_t>(worldSize))
            throw std::invalid_argument("Invalid Cartesian topology dimensions");

        std::vector<int> periods;
        periods.reserve(periodic.size());
        for(bool value : periodic)
            periods.emplace_back(value ? 1 : 0);

        TopologySnapshot snapshot;
        snapshot.hostLocalRank = hostLocalRank;
        snapshot.dimensions = std::move(dimensions);
        snapshot.periodic = std::move(periodic);
        snapshot.coordinates.resize(snapshot.dimensions.size());
        snapshot.neighbors.reserve(snapshot.dimensions.size() * 2u);

        MPI_Comm cartesian = MPI_COMM_NULL;
        int error = MPI_Cart_create(
            context.communicator(worldCommunicator),
            static_cast<int>(snapshot.dimensions.size()),
            snapshot.dimensions.data(),
            periods.data(),
            0,
            &cartesian);
        if(error != MPI_SUCCESS || cartesian == MPI_COMM_NULL)
            throw error == MPI_SUCCESS ? std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL")
                                       : mpiError("MPI_Cart_create", error);

        error = MPI_Comm_rank(cartesian, &snapshot.rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(cartesian, &snapshot.size);
        if(error == MPI_SUCCESS)
            error = MPI_Cart_coords(
                cartesian,
                snapshot.rank,
                static_cast<int>(snapshot.coordinates.size()),
                snapshot.coordinates.data());
        for(int dimension = 0; error == MPI_SUCCESS && dimension < static_cast<int>(snapshot.dimensions.size());
            ++dimension)
        {
            int negative = MPI_PROC_NULL;
            int positive = MPI_PROC_NULL;
            error = MPI_Cart_shift(cartesian, dimension, 1, &negative, &positive);
            snapshot.neighbors.emplace_back(negative == MPI_PROC_NULL ? -1 : negative);
            snapshot.neighbors.emplace_back(positive == MPI_PROC_NULL ? -1 : positive);
        }
        if(error != MPI_SUCCESS)
        {
            MPI_Comm_free(&cartesian);
            throw mpiError("MPI Cartesian topology query", error);
        }

        snapshot.communicator = context.adoptCommunicator(cartesian);
        return snapshot;
    }

    CommunicatorId detail::duplicateCommunicator(NativeMpiContext& context, CommunicatorId communicator)
    {
        MPI_Comm duplicate = MPI_COMM_NULL;
        int const error = MPI_Comm_dup(context.communicator(communicator), &duplicate);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Comm_dup", error);
        return context.adoptCommunicator(duplicate);
    }

    std::optional<CommunicatorInfo> detail::splitCommunicator(
        NativeMpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator)
    {
        MPI_Comm split = MPI_COMM_NULL;
        int error = MPI_Comm_split(context.communicator(communicator), color.value_or(MPI_UNDEFINED), key, &split);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Comm_split", error);
        if(split == MPI_COMM_NULL)
            return std::nullopt;

        CommunicatorInfo info;
        error = MPI_Comm_rank(split, &info.rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(split, &info.size);
        if(error != MPI_SUCCESS)
        {
            MPI_Comm_free(&split);
            throw mpiError("MPI split communicator query", error);
        }
        info.communicator = context.adoptCommunicator(split);
        return info;
    }

    void detail::destroyCommunicator(NativeMpiContext& context, CommunicatorId communicator)
    {
        if(communicator == worldCommunicator)
            throw std::invalid_argument("The world communicator cannot be destroyed");
        context.destroyCommunicator(communicator);
    }

} // namespace caravan
