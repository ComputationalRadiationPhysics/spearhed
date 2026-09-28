/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <caravan/core/sender.hpp>
#include <caravan/mpi/context.hpp>

namespace caravan::mpi
{
    namespace collective_detail
    {
        /** Move-only ownership of one reserved collective slot.
         *
         * Destroying an unreleased token skips its slot without running user code.
         * The associated MpiContext must outlive the token.
         */
        class ManagedCollectiveToken
        {
        public:
            ManagedCollectiveToken(MpiContext& context, caravan::detail::ManagedCollectiveTicket ticket)
                : m_context(&context)
                , m_ticket(ticket)
            {
            }

            ManagedCollectiveToken(ManagedCollectiveToken const&) = delete;
            ManagedCollectiveToken& operator=(ManagedCollectiveToken const&) = delete;

            ManagedCollectiveToken(ManagedCollectiveToken&& other) noexcept
                : m_context(std::exchange(other.m_context, nullptr))
                , m_ticket(other.m_ticket)
            {
            }

            ManagedCollectiveToken& operator=(ManagedCollectiveToken&&) = delete;

            ~ManagedCollectiveToken()
            {
                if(m_context)
                    caravan::detail::CollectiveAccess::abandon(*m_context, m_ticket);
            }

            void release(std::function<void()> start)
            {
                auto* context = std::exchange(m_context, nullptr);
                caravan::detail::CollectiveAccess::release(*context, m_ticket, std::move(start));
            }

            template<typename T_Sender>
            bool accepts(T_Sender const& sender) const noexcept
            {
                return sender.managedCollectiveOn(*m_context, m_ticket.communicator);
            }

        private:
            MpiContext* m_context;
            caravan::detail::ManagedCollectiveTicket m_ticket;
        };

        template<typename T_Sender, typename T_Factory, typename T_Receiver>
        class ManagedCollectiveOperation
        {
            struct PredecessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->prepareSuccessor(std::forward<T>(values)...);
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                ManagedCollectiveOperation* owner;
            };

            using SuccessorOperation = caravan::detail::
                ConnectedOperation<caravan::detail::SuccessorSender<T_Sender, T_Factory>, T_Receiver>;

        public:
            ManagedCollectiveOperation(
                ManagedCollectiveToken token,
                T_Sender sender,
                T_Factory factory,
                T_Receiver receiver)
                : m_token(std::move(token))
                , m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
                , m_predecessor(std::move(sender).connect(PredecessorReceiver{this}))
            {
            }

            ManagedCollectiveOperation(ManagedCollectiveOperation const&) = delete;
            ManagedCollectiveOperation& operator=(ManagedCollectiveOperation const&) = delete;
            ManagedCollectiveOperation(ManagedCollectiveOperation&&) = delete;
            ManagedCollectiveOperation& operator=(ManagedCollectiveOperation&&) = delete;

            void start() & noexcept
            {
                m_predecessor.start();
            }

        private:
            template<typename... T>
            void prepareSuccessor(T&&... values) noexcept
            {
                m_values.emplace(std::forward<T>(values)...);
                auto successor
                    = std::apply([this](auto&... stored) { return std::invoke(m_factory, stored...); }, *m_values);
                if(!m_token.accepts(successor))
                    std::terminate();
                m_successor.emplace(std::move(successor), m_receiver);
                release([this]() noexcept { m_successor->start(); });
            }

            template<typename T_Start>
            void release(T_Start start) noexcept
            {
                m_token.release(std::function<void()>{std::move(start)});
            }

            ManagedCollectiveToken m_token;
            T_Factory m_factory;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<PredecessorReceiver>())) m_predecessor;
            std::optional<caravan::detail::StoredValueTuple<T_Sender>> m_values;
            std::optional<SuccessorOperation> m_successor;
        };

        template<typename T_Sender, typename T_Factory>
        class ManagedCollectiveSender
        {
        public:
            using completion_signatures
                = CompletionSignaturesOf<caravan::detail::SuccessorSender<T_Sender, T_Factory>>;

            ManagedCollectiveSender(ManagedCollectiveToken token, T_Sender sender, T_Factory factory)
                : m_token(std::move(token))
                , m_sender(std::move(sender))
                , m_factory(std::move(factory))
            {
            }

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return ManagedCollectiveOperation<T_Sender, T_Factory, std::decay_t<T_Receiver>>{
                    std::move(m_token),
                    std::move(m_sender),
                    std::move(m_factory),
                    std::forward<T_Receiver>(receiver)};
            }

        private:
            ManagedCollectiveToken m_token;
            T_Sender m_sender;
            T_Factory m_factory;
        };
    } // namespace collective_detail

    /** Plan local collective initiation order independently of predecessor readiness.
     *
     * Every rank must reserve and start the same sequence on this communicator.
     * Abandonment must also match across ranks. Violating this distributed
     * contract can mismatch collectives or hang MPI.
     *
     * A value successor must be an immediate Caravan MPI collective on this lane's
     * context and communicator. MpiContext must outlive all entries.
     */
    class CollectiveLane
    {
    public:
        CollectiveLane(MpiContext& context, CommunicatorId communicator = worldCommunicator)
            : m_context(&context)
            , m_communicator(communicator)
        {
        }

        template<Sender T_Sender, typename T_Factory>
        auto submit(T_Sender sender, T_Factory factory)
        {
            collective_detail::ManagedCollectiveToken token{
                *m_context,
                caravan::detail::CollectiveAccess::reserve(*m_context, m_communicator)};
            return collective_detail::ManagedCollectiveSender<T_Sender, std::decay_t<T_Factory>>{
                std::move(token),
                std::move(sender),
                std::move(factory)};
        }

    private:
        MpiContext* m_context;
        CommunicatorId m_communicator;
    };
} // namespace caravan::mpi
