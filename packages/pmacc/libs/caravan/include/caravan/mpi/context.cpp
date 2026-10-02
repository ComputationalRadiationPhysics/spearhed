/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <caravan/core/eager.hpp>
#include <caravan/mpi/error.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace caravan
{
    using detail::mpiError;

    MpiContext::MpiContext() : m_owner(std::this_thread::get_id()), m_communicators{MPI_COMM_WORLD}
    {
        m_topology.communicator = worldCommunicator;
        int const rankError = MPI_Comm_rank(MPI_COMM_WORLD, &m_topology.rank);
        if(rankError != MPI_SUCCESS)
            throw mpiError("MPI_Comm_rank", rankError);
        int const sizeError = MPI_Comm_size(MPI_COMM_WORLD, &m_topology.size);
        if(sizeError != MPI_SUCCESS)
            throw mpiError("MPI_Comm_size", sizeError);

        MPI_Comm host = MPI_COMM_NULL;
        int const splitError
            = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, m_topology.rank, MPI_INFO_NULL, &host);
        if(splitError != MPI_SUCCESS)
            throw mpiError("MPI_Comm_split_type", splitError);
        int const hostRankError = MPI_Comm_rank(host, &m_topology.hostLocalRank);
        int const freeError = MPI_Comm_free(&host);
        if(hostRankError != MPI_SUCCESS)
            throw mpiError("MPI_Comm_rank(host)", hostRankError);
        if(freeError != MPI_SUCCESS)
            throw mpiError("MPI_Comm_free(host)", freeError);
    }

    TopologySnapshot MpiContext::topology() const
    {
        return m_topology;
    }

    bool MpiContext::accepting() const noexcept
    {
        std::lock_guard lock(m_queueMutex);
        return m_accepting;
    }

    void MpiContext::submitNative(detail::NativeSubmission submission)
    {
        submit(std::move(submission), [this](detail::NativeSubmission output) { startNative(std::move(output)); });
    }

    void MpiContext::invokeNative(detail::NativeInvocation submission)
    {
        submit(std::move(submission), [this](detail::NativeInvocation output) { invoke(std::move(output)); });
    }

    detail::ManagedCollectiveTicket MpiContext::reserveManagedCollective(CommunicatorId communicator)
    {
        std::lock_guard lock(m_queueMutex);
        if(!m_accepting)
            throw std::runtime_error("MPI context is shutting down");
        auto& lane = m_managedCollectives[communicator.value];
        auto const ticket = lane.reserve(communicator);
        ++m_outstanding;
        return ticket;
    }

    void MpiContext::releaseManagedCollective(detail::ManagedCollectiveTicket ticket, std::function<void()> start)
    {
        {
            std::lock_guard lock(m_queueMutex);
            auto lane = m_managedCollectives.find(ticket.communicator.value);
            if(lane == m_managedCollectives.end())
                throw std::logic_error("Unknown managed collective ticket");
            lane->second.commit(ticket.sequence, std::move(start));
        }
        m_queueReady.notify_one();
    }

    void MpiContext::abandonManagedCollective(detail::ManagedCollectiveTicket ticket) noexcept
    {
        {
            std::lock_guard lock(m_queueMutex);
            auto lane = m_managedCollectives.find(ticket.communicator.value);
            if(lane == m_managedCollectives.end())
                return;
            lane->second.skip(ticket.sequence);
        }
        m_queueReady.notify_one();
    }

    void MpiContext::run()
    try
    {
        assertOwner();
        ExecutorThreadGuard guard;
        while(progress())
        {
            std::unique_lock lock(m_queueMutex);
            if(m_requests.empty() && m_queue.empty() && !hasReadyManagedCollective())
                m_queueReady.wait(
                    lock,
                    [this]
                    {
                        return !m_queue.empty() || hasReadyManagedCollective() || (m_stopping && m_outstanding == 0u);
                    });
        }
    }
    catch(...)
    {
        abortMpi();
    }

    bool MpiContext::progress() noexcept
    try
    {
        assertOwner();
        {
            std::lock_guard lock(m_queueMutex);
            if(m_finished)
                return false;
        }

        drainQueue(submissionBatchSize);
        drainManagedCollectives(submissionBatchSize);
        progressRequests();

        {
            std::lock_guard lock(m_queueMutex);
            if(!m_stopping || m_outstanding != 0u)
                return true;
        }
        releaseCommunicators();
        {
            std::lock_guard lock(m_queueMutex);
            m_finished = true;
        }
        return false;
    }
    catch(...)
    {
        // Never unwind the context and release owners while MPI may still use them.
        abortMpi();
    }

    void MpiContext::requestShutdown()
    {
        {
            std::lock_guard lock(m_queueMutex);
            m_accepting = false;
            m_stopping = true;
            for(auto& [communicator, lane] : m_managedCollectives)
                lane.skipReserved();
        }
        m_queueReady.notify_one();
    }

    bool MpiContext::NativeGroup::retire(NativeMpiContext& context, std::size_t index, MPI_Status const& status)
    {
        statuses[index] = status;
        if(--remaining != 0u)
            return false;
        completed(context, statuses);
        return true;
    }

    detail::ManagedCollectiveTicket MpiContext::ManagedCollectiveLane::reserve(CommunicatorId communicator)
    {
        auto const sequence = firstSequence + entries.size();
        entries.emplace_back();
        return {communicator, sequence};
    }

    void MpiContext::ManagedCollectiveLane::commit(std::size_t sequence, std::function<void()> start)
    {
        auto* entry = find(sequence);
        if(entry == nullptr || entry->state != State::reserved)
            throw std::logic_error("Inactive managed collective ticket");
        entry->start = std::move(start);
        entry->state = State::committed;
    }

    void MpiContext::ManagedCollectiveLane::skip(std::size_t sequence) noexcept
    {
        auto* entry = find(sequence);
        if(entry != nullptr && entry->state == State::reserved)
            entry->state = State::skipped;
    }

    void MpiContext::ManagedCollectiveLane::skipReserved() noexcept
    {
        for(auto& entry : entries)
            if(entry.state == State::reserved)
                entry.state = State::skipped;
    }

    bool MpiContext::ManagedCollectiveLane::ready() const noexcept
    {
        return !entries.empty() && entries.front().state != State::reserved;
    }

    std::optional<std::function<void()>> MpiContext::ManagedCollectiveLane::popReady()
    {
        if(!ready())
            return std::nullopt;
        std::optional<std::function<void()>> start{std::move(entries.front().start)};
        entries.pop_front();
        ++firstSequence;
        return start;
    }

    MpiContext::ManagedCollectiveLane::Entry* MpiContext::ManagedCollectiveLane::find(std::size_t sequence) noexcept
    {
        if(sequence < firstSequence || sequence - firstSequence >= entries.size())
            return nullptr;
        return &entries[sequence - firstSequence];
    }

    template<typename T_Output, typename T_Start>
    void MpiContext::submit(T_Output output, T_Start&& start)
    {
        if(detail::nativeCallbackDepth != 0u)
            std::terminate();
        std::function<void()> command
            = [this, output = std::move(output), start = std::forward<T_Start>(start)]() mutable
        { std::invoke(start, output); };

        {
            std::lock_guard lock(m_queueMutex);
            if(!m_accepting)
                std::terminate();
            m_queue.emplace_back(std::move(command));
            ++m_outstanding;
        }
        m_queueReady.notify_one();
    }

    bool MpiContext::hasReadyManagedCollective() const
    {
        for(auto const& [communicator, lane] : m_managedCollectives)
            if(lane.ready())
                return true;
        return false;
    }

    void MpiContext::drainManagedCollectives(std::size_t remaining)
    {
        assertOwner();
        while(remaining-- > 0u)
        {
            std::optional<std::function<void()>> start;
            {
                std::lock_guard lock(m_queueMutex);
                for(auto& [communicator, lane] : m_managedCollectives)
                {
                    start = lane.popReady();
                    if(start)
                    {
                        --m_outstanding;
                        break;
                    }
                }
            }
            if(!start)
                return;
            if(*start)
                (*start)();
        }
    }

    void MpiContext::assertOwner() const
    {
        assert(std::this_thread::get_id() == m_owner && "MPI operation executed outside the MPI owner thread");
    }

    void MpiContext::drainQueue(std::size_t remaining)
    {
        assertOwner();
        while(remaining-- > 0u)
        {
            std::function<void()> command;
            {
                std::lock_guard lock(m_queueMutex);
                if(m_queue.empty())
                    return;
                command = std::move(m_queue.front());
                m_queue.pop_front();
            }
            command();
        }
    }

    MPI_Comm MpiContext::communicator(CommunicatorId id) const
    {
        if(id.value >= m_communicators.size() || m_communicators[id.value] == MPI_COMM_NULL)
            throw std::invalid_argument("Unknown Caravan communicator");
        return m_communicators[id.value];
    }

    CommunicatorId MpiContext::adoptCommunicator(MPI_Comm native)
    {
        assertOwner();
        if(native == MPI_COMM_NULL)
            throw std::invalid_argument("Cannot adopt MPI_COMM_NULL");
        try
        {
            if(m_communicators.size() >= std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Too many Caravan communicators");
            int const error = MPI_Comm_set_errhandler(native, MPI_ERRORS_RETURN);
            if(error != MPI_SUCCESS)
                throw mpiError("MPI_Comm_set_errhandler", error);
            auto const id = CommunicatorId{static_cast<std::uint32_t>(m_communicators.size())};
            m_communicators.emplace_back(native);
            return id;
        }
        catch(...)
        {
            MPI_Comm_free(&native);
            throw;
        }
    }

    void MpiContext::destroyCommunicator(CommunicatorId id)
    {
        assertOwner();
        if(id == worldCommunicator || id.value >= m_communicators.size() || m_communicators[id.value] == MPI_COMM_NULL)
            throw std::invalid_argument("Unknown or immutable Caravan communicator");
        int const error = MPI_Comm_free(&m_communicators[id.value]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Comm_free", error);
    }

    NativeMpiContext MpiContext::nativeContext()
    {
        return detail::NativeContextFactory::create(
            this,
            [](void* implementation, CommunicatorId id)
            { return static_cast<MpiContext*>(implementation)->communicator(id); },
            [](void* implementation, MPI_Comm native)
            { return static_cast<MpiContext*>(implementation)->adoptCommunicator(native); },
            [](void* implementation, CommunicatorId id)
            { static_cast<MpiContext*>(implementation)->destroyCommunicator(id); });
    }

    template<typename T>
    void MpiContext::reserveForAppend(std::vector<T>& values, std::size_t additional)
    {
        if(additional > values.max_size() - values.size())
            throw std::length_error("Too many native MPI requests");
        auto const required = values.size() + additional;
        if(required > values.capacity())
            values.reserve(
                std::max(
                    required,
                    values.capacity() + std::min(values.capacity(), values.max_size() - values.capacity())));
    }

    void MpiContext::trackNative(detail::NativeSubmission const& output, NativeRequestBatch& batch)
    {
        auto context = nativeContext();
        auto const activeRequests = static_cast<std::size_t>(std::count_if(
            batch.requests.begin(),
            batch.requests.end(),
            [](MPI_Request request) { return request != MPI_REQUEST_NULL; }));

        if(batch.requests.empty())
        {
            detail::NativeAccess::release(batch);
            output.completed(context, {});
            finishOperation();
            return;
        }

        // Both arrays must have capacity before registering any request from the batch.
        reserveForAppend(m_requests, activeRequests);
        reserveForAppend(m_active, activeRequests);
        std::vector<MPI_Status> statuses(batch.requests.size());
        auto group = std::make_shared<NativeGroup>(
            output.completed,
            std::move(statuses),
            std::vector<std::shared_ptr<void>>{},
            batch.requests.size());
        group->lifetimes.swap(batch.lifetimes);
        for(std::size_t index = 0u; index < batch.requests.size(); ++index)
        {
            if(batch.requests[index] == MPI_REQUEST_NULL)
            {
                if(group->retire(context, index, MPI_Status{}))
                    finishOperation();
                continue;
            }
            m_requests.emplace_back(batch.requests[index]);
            m_active.emplace_back(group, index);
        }
        detail::NativeAccess::release(batch);
    }

    [[noreturn]] void MpiContext::abortMpi(int error) const noexcept
    {
        MPI_Abort(MPI_COMM_WORLD, error);
        std::terminate();
    }

    void MpiContext::startNative(detail::NativeSubmission output)
    {
        assertOwner();
        auto context = nativeContext();
        auto batch = output.start(context);
        output.start = {};
        trackNative(output, batch);
    }

    void MpiContext::invoke(detail::NativeInvocation output)
    {
        assertOwner();
        auto context = nativeContext();
        output.invoke(context);
        finishOperation();
    }

    void MpiContext::releaseCommunicators()
    {
        assertOwner();
        for(std::size_t i = 1u; i < m_communicators.size(); ++i)
        {
            if(m_communicators[i] == MPI_COMM_NULL)
                continue;
            int const error = MPI_Comm_free(&m_communicators[i]);
            if(error != MPI_SUCCESS)
                abortMpi(error);
        }
    }

    void MpiContext::retireActive(NativeCompletion& active, MPI_Status const& status)
    {
        auto context = nativeContext();
        if(active.group->retire(context, active.index, status))
            finishOperation();
    }

    void MpiContext::progressRequests()
    {
        assertOwner();
        if(m_requests.empty())
            return;

        m_completedIndices.resize(m_requests.size());
        m_statuses.resize(m_requests.size());
        int completed = 0;
        int const error = MPI_Testsome(
            static_cast<int>(m_requests.size()),
            m_requests.data(),
            &completed,
            m_completedIndices.data(),
            m_statuses.data());
        if(error != MPI_SUCCESS && error != MPI_ERR_IN_STATUS)
            abortMpi(error);
        if(completed == MPI_UNDEFINED || completed == 0)
            return;

        for(int i = 0; i < completed; ++i)
        {
            auto const position = static_cast<std::size_t>(i);
            auto const index = static_cast<std::size_t>(m_completedIndices[position]);
            auto const requestError = error == MPI_ERR_IN_STATUS ? m_statuses[position].MPI_ERROR : MPI_SUCCESS;
            if(requestError == MPI_ERR_PENDING || m_requests[index] != MPI_REQUEST_NULL)
                continue;
            if(requestError != MPI_SUCCESS)
                abortMpi(requestError);
            retireActive(m_active[index], m_statuses[position]);
        }

        std::size_t output = 0u;
        for(std::size_t input = 0u; input < m_requests.size(); ++input)
        {
            if(m_requests[input] == MPI_REQUEST_NULL)
                continue;
            if(output != input)
            {
                m_requests[output] = m_requests[input];
                m_active[output] = std::move(m_active[input]);
            }
            ++output;
        }
        m_requests.resize(output);
        m_active.resize(output);
    }

    void MpiContext::finishOperation()
    {
        {
            std::lock_guard lock(m_queueMutex);
            --m_outstanding;
        }
        m_queueReady.notify_one();
    }

    detail::ManagedCollectiveTicket detail::CollectiveAccess::reserve(MpiContext& context, CommunicatorId communicator)
    {
        return context.reserveManagedCollective(communicator);
    }

    void detail::CollectiveAccess::release(
        MpiContext& context,
        detail::ManagedCollectiveTicket ticket,
        std::function<void()> start)
    {
        context.releaseManagedCollective(ticket, std::move(start));
    }

    void detail::CollectiveAccess::abandon(MpiContext& context, detail::ManagedCollectiveTicket ticket) noexcept
    {
        context.abandonManagedCollective(ticket);
    }

    void detail::NativeAccess::submit(MpiContext& context, detail::NativeSubmission submission)
    {
        context.submitNative(std::move(submission));
    }

    void detail::NativeAccess::invoke(MpiContext& context, detail::NativeInvocation submission)
    {
        context.invokeNative(std::move(submission));
    }

} // namespace caravan
