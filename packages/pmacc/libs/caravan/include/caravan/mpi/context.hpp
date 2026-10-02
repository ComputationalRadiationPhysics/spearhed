/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <caravan/core/retained.hpp>
#include <mpi.h>

namespace caravan
{
    class NativeMpiContext;
    class NativeRequestBatch;

    namespace detail
    {
        struct NativeAccess;
        struct NativeInvocation;
        struct NativeSubmission;
    } // namespace detail

    struct CommunicatorId
    {
        std::uint32_t value;

        friend bool operator==(CommunicatorId const&, CommunicatorId const&) = default;
    };

    inline constexpr CommunicatorId worldCommunicator{0u};

    class MpiContext;
    class MpiRuntime;

    namespace detail
    {
        struct ManagedCollectiveTicket
        {
            CommunicatorId communicator;
            std::size_t sequence;
        };

        struct CollectiveAccess
        {
            static ManagedCollectiveTicket reserve(MpiContext& context, CommunicatorId communicator);
            static void release(MpiContext& context, ManagedCollectiveTicket ticket, std::function<void()> start);
            static void abandon(MpiContext& context, ManagedCollectiveTicket ticket) noexcept;
        };
    } // namespace detail

    struct Peer
    {
        int value;
        bool any = false;
    };

    inline constexpr Peer anyPeer{0, true};

    struct MessageTag
    {
        int value;
        bool any = false;
    };

    inline constexpr MessageTag anyMessageTag{0, true};

    namespace detail
    {
        template<typename T_Byte>
        struct MpiBufferView : Retained<std::span<T_Byte>, std::shared_ptr<void>>
        {
            using Base = Retained<std::span<T_Byte>, std::shared_ptr<void>>;
            using Base::Base;

            MpiBufferView(Base buffer) : Base(std::move(buffer))
            {
            }

            template<typename T, std::size_t T_Extent>
            requires std::convertible_to<std::span<T, T_Extent>, std::span<T_Byte>>
            MpiBufferView(std::span<T, T_Extent> bytes) : Base(bytes, {})
            {
            }
        };
    } // namespace detail

    /** MPI byte spans with optional ownership retained until native work completes.
     *
     * Plain spans borrow storage: the caller must keep it valid from sender construction
     * through completion, including failure cleanup. Do not modify send storage or access
     * receive storage while MPI may use it. Use retain(span, owner) to keep an owner alive;
     * retention does not prevent conflicting accesses.
     */
    using ConstMpiBuffer = detail::MpiBufferView<std::byte const>;
    using MpiBuffer = detail::MpiBufferView<std::byte>;

    enum class ScalarType : std::uint8_t
    {
        int32,
        uint32,
        int64,
        uint64,
        float32,
        float64
    };

    enum class ReduceOperation : std::uint8_t
    {
        sum,
        minimum,
        maximum,
        product
    };

    struct CommunicatorInfo
    {
        CommunicatorId communicator;
        int rank;
        int size;
    };

    struct TopologySnapshot
    {
        int rank;
        int size;
        int hostLocalRank;
        CommunicatorId communicator;
        std::vector<int> dimensions;
        std::vector<int> coordinates;
        std::vector<bool> periodic;
        // Negative then positive neighbor for each dimension; -1 means no neighbor.
        std::vector<int> neighbors;
    };

    /** MPI backend authority owning lifecycle, progress, and native resources.
     *
     * This context is not a scheduler for application continuations. Unexpected
     * progress-engine failures abort MPI before releasing outstanding storage.
     */
    class MpiContext
    {
    public:
        MpiContext(MpiContext const&) = delete;
        MpiContext& operator=(MpiContext const&) = delete;

        TopologySnapshot topology() const;

        /** True while the context accepts new native submissions; false once shutdown has begun. */
        bool accepting() const noexcept;

    private:
        struct NativeGroup
        {
            std::function<void(NativeMpiContext&, std::span<MPI_Status const>)> completed;
            std::vector<MPI_Status> statuses;
            std::vector<std::shared_ptr<void>> lifetimes;
            std::size_t remaining;

            bool retire(NativeMpiContext& context, std::size_t index, MPI_Status const& status);
        };

        struct NativeCompletion
        {
            std::shared_ptr<NativeGroup> group;
            std::size_t index;
        };

        struct ManagedCollectiveLane
        {
            enum class State : std::uint8_t
            {
                reserved,
                committed,
                skipped
            };

            struct Entry
            {
                std::function<void()> start;
                State state = State::reserved;
            };

            /* The deque contains every non-retired ticket in contiguous sequence
             * order. Only the front may retire, and only after commit or skip.
             * Each entry contributes one to m_outstanding until popReady. These
             * transitions and all accesses happen under m_queueMutex; callbacks
             * run only after the lock is released.
             */
            detail::ManagedCollectiveTicket reserve(CommunicatorId communicator);
            void commit(std::size_t sequence, std::function<void()> start);
            void skip(std::size_t sequence) noexcept;
            void skipReserved() noexcept;
            bool ready() const noexcept;
            std::optional<std::function<void()>> popReady();

        private:
            Entry* find(std::size_t sequence) noexcept;

            std::size_t firstSequence = 0u;
            std::deque<Entry> entries;
        };

        MpiContext();

        void run();
        bool progress() noexcept;
        void requestShutdown();
        void submitNative(detail::NativeSubmission submission);
        void invokeNative(detail::NativeInvocation submission);
        detail::ManagedCollectiveTicket reserveManagedCollective(CommunicatorId communicator);
        void releaseManagedCollective(detail::ManagedCollectiveTicket ticket, std::function<void()> start);
        void abandonManagedCollective(detail::ManagedCollectiveTicket ticket) noexcept;

        template<typename T_Output, typename T_Start>
        void submit(T_Output output, T_Start&& start);
        bool hasReadyManagedCollective() const;
        void drainManagedCollectives(std::size_t remaining);
        void assertOwner() const;
        void drainQueue(std::size_t remaining);
        MPI_Comm communicator(CommunicatorId id) const;
        CommunicatorId adoptCommunicator(MPI_Comm native);
        void destroyCommunicator(CommunicatorId id);
        NativeMpiContext nativeContext();
        template<typename T>
        static void reserveForAppend(std::vector<T>& values, std::size_t additional);
        void trackNative(detail::NativeSubmission const& output, NativeRequestBatch& batch);
        [[noreturn]] void abortMpi(int error = MPI_ERR_OTHER) const noexcept;
        void startNative(detail::NativeSubmission output);
        void invoke(detail::NativeInvocation output);
        void releaseCommunicators();
        void retireActive(NativeCompletion& active, MPI_Status const& status);
        void progressRequests();
        void finishOperation();

        static constexpr std::size_t submissionBatchSize = 64u;

        std::thread::id m_owner;
        TopologySnapshot m_topology{};
        mutable std::mutex m_queueMutex;
        std::condition_variable m_queueReady;
        std::deque<std::function<void()>> m_queue;
        std::size_t m_outstanding = 0u;
        bool m_accepting = true;
        bool m_stopping = false;
        bool m_finished = false;
        std::unordered_map<std::uint32_t, ManagedCollectiveLane> m_managedCollectives;
        std::vector<MPI_Comm> m_communicators;
        std::vector<MPI_Request> m_requests;
        std::vector<NativeCompletion> m_active;
        std::vector<int> m_completedIndices;
        std::vector<MPI_Status> m_statuses;

        friend class MpiRuntime;
        friend struct detail::CollectiveAccess;
        friend struct detail::NativeAccess;
    };

} // namespace caravan
