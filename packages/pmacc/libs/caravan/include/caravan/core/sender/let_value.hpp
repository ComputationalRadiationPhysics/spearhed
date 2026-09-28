/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<typename T_Sender, typename T_Factory, typename T_Receiver>
        class LetValueOperation
        {
            struct PredecessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->startSuccessor(std::forward<T>(values)...);
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                LetValueOperation* owner;
            };

            using SuccessorOperation = ConnectedOperation<SuccessorSender<T_Sender, T_Factory>, T_Receiver>;

        public:
            LetValueOperation(T_Sender sender, T_Factory factory, T_Receiver receiver)
                : m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
                , m_predecessor(std::move(sender).connect(PredecessorReceiver{this}))
            {
            }

            LetValueOperation(LetValueOperation const&) = delete;
            LetValueOperation& operator=(LetValueOperation const&) = delete;
            LetValueOperation(LetValueOperation&&) = delete;
            LetValueOperation& operator=(LetValueOperation&&) = delete;

            void start() & noexcept
            {
                m_predecessor.start();
            }

        private:
            template<typename... T>
            void startSuccessor(T&&... values) noexcept
            {
                m_values.emplace(std::forward<T>(values)...);
                auto successor
                    = std::apply([this](auto&... stored) { return std::invoke(m_factory, stored...); }, *m_values);
                m_successor.emplace(std::move(successor), m_receiver);
                m_successor->start();
            }

            T_Factory m_factory;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<PredecessorReceiver>())) m_predecessor;
            std::optional<StoredValueTuple<T_Sender>> m_values;
            std::optional<SuccessorOperation> m_successor;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Factory>
    class LetValueSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<detail::SuccessorSender<T_Sender, T_Factory>>;

        LetValueSender(T_Sender sender, T_Factory factory) : m_sender(std::move(sender)), m_factory(std::move(factory))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::LetValueOperation<T_Sender, T_Factory, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_factory),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Factory m_factory;
    };

    template<Sender T_Sender, typename T_Factory>
    auto letValue(T_Sender sender, T_Factory factory)
    {
        return LetValueSender<T_Sender, T_Factory>{std::move(sender), std::move(factory)};
    }

    template<typename T_Factory>
    auto letValue(T_Factory factory)
    {
        return detail::SenderAdaptorClosure{[factory = std::move(factory)](auto sender) mutable
                                            { return letValue(std::move(sender), std::move(factory)); }};
    }
} // namespace caravan
