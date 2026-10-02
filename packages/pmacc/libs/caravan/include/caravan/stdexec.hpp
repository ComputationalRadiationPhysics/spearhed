/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <type_traits>
#include <utility>

#include <caravan/core/sender.hpp>
#include <stdexec/execution.hpp>

namespace caravan::stdexecInterop
{
    namespace detail
    {
        template<typename T_Signature>
        struct CompletionSignature;

        template<typename... T>
        struct CompletionSignature<ValueSignature<T...>>
        {
            using type = stdexec::set_value_t(T...);
        };

        template<typename T_Signatures>
        struct CompletionSignatures;

        template<typename... T_Signatures>
        struct CompletionSignatures<caravan::CompletionSignatures<T_Signatures...>>
        {
            using type = stdexec::completion_signatures<typename CompletionSignature<T_Signatures>::type...>;
        };

        template<typename T_Receiver>
        class Receiver
        {
        public:
            explicit Receiver(T_Receiver receiver) : m_receiver(std::move(receiver))
            {
            }

            template<typename... T>
            void set_value(T&&... values) noexcept
            {
                stdexec::set_value(std::move(m_receiver), std::forward<T>(values)...);
            }

            auto get_env() const noexcept
            {
                return stdexec::get_env(m_receiver);
            }

        private:
            T_Receiver m_receiver;
        };
    } // namespace detail

    /** Adapt a Caravan migration sender to the stdexec sender contract.
     *
     * The adapter translates completion signatures and receiver CPO calls. It
     * deliberately adds no scheduler/domain attributes and Caravan backends do
     * not yet consume the receiver environment or its stop token.
     */
    template<caravan::Sender T_Sender>
    class Sender
    {
    public:
        using sender_concept = stdexec::sender_t;
        using StandardCompletionSignatures =
            typename detail::CompletionSignatures<caravan::CompletionSignaturesOf<T_Sender>>::type;

        explicit Sender(T_Sender sender) : m_sender(std::move(sender))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver receiver) &&
        {
            return std::move(m_sender).connect(detail::Receiver<T_Receiver>{std::move(receiver)});
        }

        template<typename... T_Env>
        auto get_completion_signatures(T_Env&&...) const noexcept -> StandardCompletionSignatures
        {
            return {};
        }

        auto get_env() const noexcept
        {
            return stdexec::env{};
        }

    private:
        T_Sender m_sender;
    };

    template<caravan::Sender T_Sender>
    auto adapt(T_Sender sender)
    {
        return Sender<T_Sender>{std::move(sender)};
    }
} // namespace caravan::stdexecInterop
