/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <caravan/core/sender.hpp>
#include <caravan/mpi/context.hpp>
#include <caravan/mpi/result.hpp>
#include <mpi.h>

namespace caravan
{
    class NativeMpiContext;

    namespace detail
    {
        struct NativeAccess;
    } // namespace detail

    /** Native requests and lifetime tokens transferred to the MPI context.
     *
     * Build the batch before starting requests. Destroying a live batch is a fatal
     * contract violation; callbacks must not fail after starting requests.
     */
    class NativeRequestBatch
    {
    public:
        NativeRequestBatch() = default;

        NativeRequestBatch(
            std::vector<MPI_Request> nativeRequests,
            std::vector<std::shared_ptr<void>> lifetimeTokens = {})
            : requests(std::move(nativeRequests))
            , lifetimes(std::move(lifetimeTokens))
        {
        }

        NativeRequestBatch(NativeRequestBatch const&) = delete;
        NativeRequestBatch& operator=(NativeRequestBatch const&) = delete;

        NativeRequestBatch(NativeRequestBatch&& other) noexcept
            : requests(std::move(other.requests))
            , lifetimes(std::move(other.lifetimes))
            , m_ownsRequests(std::exchange(other.m_ownsRequests, false))
        {
        }

        NativeRequestBatch& operator=(NativeRequestBatch&&) = delete;

        ~NativeRequestBatch();

        std::vector<MPI_Request> requests;
        std::vector<std::shared_ptr<void>> lifetimes;

    private:
        void release() noexcept
        {
            m_ownsRequests = false;
        }

        bool m_ownsRequests = true;

        friend struct detail::NativeAccess;
    };

    inline NativeRequestBatch::~NativeRequestBatch()
    {
        if(!m_ownsRequests)
            return;
        bool active = false;
        for(auto const request : requests)
            active |= request != MPI_REQUEST_NULL;
        if(!active)
            return;

        std::terminate();
    }

    namespace detail
    {
        struct NativeSubmission
        {
            std::function<NativeRequestBatch(NativeMpiContext&)> start;
            std::function<void(NativeMpiContext&, std::span<MPI_Status const>)> completed;
        };

        struct NativeInvocation
        {
            std::function<void(NativeMpiContext&)> invoke;
        };

        inline thread_local std::size_t nativeCallbackDepth = 0u;

        class NativeCallbackGuard
        {
        public:
            NativeCallbackGuard()
            {
                ++nativeCallbackDepth;
            }

            ~NativeCallbackGuard()
            {
                --nativeCallbackDepth;
            }
        };

        template<typename T_Callable, typename... T_Args>
        decltype(auto) invokeNative(T_Callable&& callable, T_Args&&... args)
        {
            NativeCallbackGuard guard;
            return std::invoke(std::forward<T_Callable>(callable), std::forward<T_Args>(args)...);
        }

        struct NativeAccess
        {
            static void release(NativeRequestBatch& batch)
            {
                batch.release();
            }

            static void submit(MpiContext& context, NativeSubmission submission);
            static void invoke(MpiContext& context, NativeInvocation submission);
        };

        struct NativeContextFactory;
    } // namespace detail

    /** MPI-native access valid only for the duration of an MPI-context hook. */
    class NativeMpiContext
    {
    public:
        NativeMpiContext(NativeMpiContext const&) = delete;
        NativeMpiContext& operator=(NativeMpiContext const&) = delete;

        MPI_Comm communicator(CommunicatorId id) const
        {
            return m_resolve(m_implementation, id);
        }

        /** Transfer ownership of a newly created communicator to Caravan. */
        CommunicatorId adoptCommunicator(MPI_Comm communicator) const
        {
            return m_adopt(m_implementation, communicator);
        }

        /** Destroy a communicator previously adopted by Caravan. */
        void destroyCommunicator(CommunicatorId communicator) const
        {
            m_destroy(m_implementation, communicator);
        }

    private:
        using Resolve = MPI_Comm (*)(void*, CommunicatorId);
        using Adopt = CommunicatorId (*)(void*, MPI_Comm);
        using Destroy = void (*)(void*, CommunicatorId);

        NativeMpiContext(void* implementation, Resolve resolve, Adopt adopt, Destroy destroy)
            : m_implementation(implementation)
            , m_resolve(resolve)
            , m_adopt(adopt)
            , m_destroy(destroy)
        {
        }

        void* m_implementation;
        Resolve m_resolve;
        Adopt m_adopt;
        Destroy m_destroy;

        friend struct detail::NativeContextFactory;
    };

    namespace detail
    {
        struct NativeContextFactory
        {
            static NativeMpiContext create(
                void* implementation,
                NativeMpiContext::Resolve resolve,
                NativeMpiContext::Adopt adopt,
                NativeMpiContext::Destroy destroy)
            {
                return NativeMpiContext{implementation, resolve, adopt, destroy};
            }
        };
    } // namespace detail

    namespace detail
    {
        NativeRequestBatch startSend(
            NativeMpiContext& context,
            ConstMpiBuffer const& buffer,
            Peer destination,
            MessageTag tag,
            CommunicatorId communicator);

        NativeRequestBatch startReceive(
            NativeMpiContext& context,
            MpiBuffer const& buffer,
            Peer source,
            MessageTag tag,
            CommunicatorId communicator);

        ReceiveResult completeReceive(std::span<MPI_Status const> statuses);

        NativeRequestBatch startAllReduce(
            NativeMpiContext& context,
            ConstMpiBuffer const& input,
            MpiBuffer const& output,
            ScalarType type,
            ReduceOperation operation,
            CommunicatorId communicator);

        NativeRequestBatch startReduce(
            NativeMpiContext& context,
            ConstMpiBuffer const& input,
            MpiBuffer const& output,
            ScalarType type,
            ReduceOperation operation,
            Peer root,
            CommunicatorId communicator);

        NativeRequestBatch startGather(
            NativeMpiContext& context,
            ConstMpiBuffer const& input,
            MpiBuffer const& output,
            Peer root,
            CommunicatorId communicator,
            std::size_t& resultBytes);

        NativeRequestBatch startAllGather(
            NativeMpiContext& context,
            ConstMpiBuffer const& input,
            MpiBuffer const& output,
            CommunicatorId communicator,
            std::size_t& resultBytes);

        NativeRequestBatch startGatherV(
            NativeMpiContext& context,
            ConstMpiBuffer const& input,
            MpiBuffer const& output,
            std::vector<std::size_t> const& receiveBytes,
            std::vector<std::size_t> const& displacements,
            Peer root,
            CommunicatorId communicator,
            std::size_t& resultBytes);

        std::size_t scalarElements(std::size_t bytes, ScalarType type);

        NativeRequestBatch startBarrier(NativeMpiContext& context, CommunicatorId communicator);

        TopologySnapshot createCartesian(
            NativeMpiContext& context,
            std::vector<int> dimensions,
            std::vector<bool> periodic,
            int worldSize,
            int hostLocalRank);

        CommunicatorId duplicateCommunicator(NativeMpiContext& context, CommunicatorId communicator);

        std::optional<CommunicatorInfo> splitCommunicator(
            NativeMpiContext& context,
            std::optional<int> color,
            int key,
            CommunicatorId communicator);

        void destroyCommunicator(NativeMpiContext& context, CommunicatorId communicator);
    } // namespace detail

    namespace mpi
    {
        /** Lazy sender for one or more native nonblocking MPI requests. */
        template<typename T, typename T_Start, typename T_Complete, typename T_State = std::monostate>
        class RequestSender
        {
            static_assert(std::is_void_v<T> || (!std::is_reference_v<T> && !std::is_const_v<T>) );

        public:
            using completion_signatures
                = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<T>>;

            RequestSender(
                MpiContext& context,
                T_Start start,
                T_Complete complete,
                T_State state = {},
                std::optional<CommunicatorId> collective = std::nullopt)
                : m_context(&context)
                , m_start(std::move(start))
                , m_complete(std::move(complete))
                , m_state(std::move(state))
                , m_collective(collective)
            {
            }

            template<typename T_Receiver>
            class Operation
            {
            public:
                Operation(MpiContext& context, T_Start start, T_Complete complete, T_State state, T_Receiver receiver)
                    : m_context(&context)
                    , m_start(std::move(start))
                    , m_complete(std::move(complete))
                    , m_state(std::move(state))
                    , m_receiver(std::move(receiver))
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    if(std::exchange(m_started, true))
                        std::terminate();

                    detail::NativeAccess::submit(
                        *m_context,
                        detail::NativeSubmission{
                            [this](NativeMpiContext& context)
                            {
                                if constexpr(std::is_invocable_v<T_Start&, T_State&, NativeMpiContext&>)
                                    return detail::invokeNative(m_start, m_state, context);
                                else
                                    return detail::invokeNative(m_start, context);
                            },
                            [this](NativeMpiContext& context, std::span<MPI_Status const> statuses)
                            {
                                auto complete = [&]() -> decltype(auto)
                                {
                                    if constexpr(std::is_invocable_v<
                                                     T_Complete&,
                                                     T_State&,
                                                     NativeMpiContext&,
                                                     std::span<MPI_Status const>>)
                                        return detail::invokeNative(m_complete, m_state, context, statuses);
                                    else if constexpr(std::is_invocable_v<
                                                          T_Complete&,
                                                          T_State&,
                                                          std::span<MPI_Status const>>)
                                        return detail::invokeNative(m_complete, m_state, statuses);
                                    else if constexpr(std::is_invocable_v<
                                                          T_Complete&,
                                                          NativeMpiContext&,
                                                          std::span<MPI_Status const>>)
                                        return detail::invokeNative(m_complete, context, statuses);
                                    else
                                        return detail::invokeNative(m_complete, statuses);
                                };
                                if constexpr(std::is_void_v<T>)
                                {
                                    complete();
                                    m_receiver.set_value();
                                }
                                else
                                    m_receiver.set_value(complete());
                            }});
                }

            private:
                MpiContext* m_context;
                T_Start m_start;
                T_Complete m_complete;
                [[no_unique_address]] T_State m_state;
                T_Receiver m_receiver;
                bool m_started = false;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{
                    *m_context,
                    std::move(m_start),
                    std::move(m_complete),
                    std::move(m_state),
                    std::forward<T_Receiver>(receiver)};
            }

            bool managedCollectiveOn(MpiContext const& context, CommunicatorId communicator) const noexcept
            {
                return m_context == &context && m_collective == communicator;
            }

        private:
            MpiContext* m_context;
            T_Start m_start;
            T_Complete m_complete;
            [[no_unique_address]] T_State m_state;
            std::optional<CommunicatorId> m_collective;
        };

        /** Describe native MPI work without initiating it until operation start.
         *
         * The queue mutex linearizes submissions and the worker consumes that FIFO.
         * Callers are responsible for making collective queue-commit order identical
         * across ranks; use CollectiveLane when dependency readiness or concurrent
         * submission can invert that order.
         */
        template<typename T, typename T_Start, typename T_Complete>
        auto request(
            MpiContext& context,
            T_Start&& start,
            T_Complete&& complete,
            std::optional<CommunicatorId> collective = std::nullopt)
        {
            return RequestSender<T, std::decay_t<T_Start>, std::decay_t<T_Complete>>{
                context,
                std::forward<T_Start>(start),
                std::forward<T_Complete>(complete),
                {},
                collective};
        }

        template<typename T, typename T_State, typename T_Start, typename T_Complete>
        auto request(
            MpiContext& context,
            T_State state,
            T_Start&& start,
            T_Complete&& complete,
            std::optional<CommunicatorId> collective)
        {
            return RequestSender<T, std::decay_t<T_Start>, std::decay_t<T_Complete>, T_State>{
                context,
                std::forward<T_Start>(start),
                std::forward<T_Complete>(complete),
                std::move(state),
                collective};
        }

        template<typename T, typename T_Operation>
        class ContextSender
        {
            static_assert(std::is_void_v<T> || (!std::is_reference_v<T> && !std::is_const_v<T>) );

        public:
            using completion_signatures
                = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<T>>;

            ContextSender(
                MpiContext& context,
                T_Operation operation,
                std::optional<CommunicatorId> collective = std::nullopt)
                : m_context(&context)
                , m_operation(std::move(operation))
                , m_collective(collective)
            {
            }

            template<typename T_Receiver>
            class Operation
            {
            public:
                Operation(MpiContext& context, T_Operation operation, T_Receiver receiver)
                    : m_context(&context)
                    , m_operation(std::move(operation))
                    , m_receiver(std::move(receiver))
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    if(std::exchange(m_started, true))
                        std::terminate();

                    detail::NativeAccess::invoke(
                        *m_context,
                        detail::NativeInvocation{[this](NativeMpiContext& context) { complete(context); }});
                }

            private:
                void complete(NativeMpiContext& context)
                {
                    if constexpr(std::is_void_v<T>)
                    {
                        detail::invokeNative(m_operation, context);
                        m_receiver.set_value();
                    }
                    else
                        m_receiver.set_value(detail::invokeNative(m_operation, context));
                }

                MpiContext* m_context;
                T_Operation m_operation;
                T_Receiver m_receiver;
                bool m_started = false;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{
                    *m_context,
                    std::move(m_operation),
                    std::forward<T_Receiver>(receiver)};
            }

            bool managedCollectiveOn(MpiContext const& context, CommunicatorId communicator) const noexcept
            {
                return m_context == &context && m_collective == communicator;
            }

        private:
            MpiContext* m_context;
            T_Operation m_operation;
            std::optional<CommunicatorId> m_collective;
        };

        /** Lazily invoke an operation on the MPI owner.
         *
         * The callback runs on the owner and blocks request progress while it runs.
         * Collective calls use the same caller-managed ordering contract as
         * request(). Pass their communicator to make the sender compatible with a
         * matching CollectiveLane.
         */
        template<typename T_Operation>
        auto invoke(
            MpiContext& context,
            T_Operation&& operation,
            std::optional<CommunicatorId> collective = std::nullopt)
        {
            using Operation = std::decay_t<T_Operation>;
            using Result
                = std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<Operation&, NativeMpiContext&>>>;
            return ContextSender<Result, Operation>{context, std::forward<T_Operation>(operation), collective};
        }

    } // namespace mpi

} // namespace caravan
