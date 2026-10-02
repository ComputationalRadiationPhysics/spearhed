/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<std::size_t T_Index, typename T_Owner, typename T_Environment>
        struct WhenAllReceiver
        {
            template<typename... T>
            void set_value(T&&... values) noexcept
            {
                owner->template setValue<T_Index>(std::forward<T>(values)...);
            }

            decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Environment const&>().get_env()))
                requires requires(T_Environment const& value) { value.get_env(); }
            {
                return environment->get_env();
            }

            T_Owner* owner;
            T_Environment const* environment;
        };

        template<typename T_Receiver>
        class WhenAllReceiverHolder
        {
        protected:
            explicit WhenAllReceiverHolder(T_Receiver receiver) : m_receiver(std::move(receiver))
            {
            }

            T_Receiver m_receiver;
        };

        template<std::size_t T_Index, typename T_Owner, typename T_Environment, typename T_Sender>
        class WhenAllOperationHolder
        {
            using Receiver = WhenAllReceiver<T_Index, T_Owner, T_Environment>;

        public:
            WhenAllOperationHolder(T_Sender sender, T_Owner* owner, T_Environment const* environment)
                : m_operation(std::move(sender).connect(Receiver{owner, environment}))
            {
            }

            void start() noexcept
            {
                m_operation.start();
            }

        private:
            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };

        template<typename T_Receiver, typename T_Indices, typename... T_Senders>
        class WhenAllOperation;

        template<typename T_Receiver, std::size_t... T_Index, typename... T_Senders>
        class WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>
            : private WhenAllReceiverHolder<T_Receiver>
            , private WhenAllOperationHolder<
                  T_Index,
                  WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>,
                  T_Receiver,
                  T_Senders>...
        {
            using Self = WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>;
            using ReceiverHolder = WhenAllReceiverHolder<T_Receiver>;

            template<std::size_t T_I>
            using Holder
                = WhenAllOperationHolder<T_I, Self, T_Receiver, std::tuple_element_t<T_I, std::tuple<T_Senders...>>>;

        public:
            WhenAllOperation(std::tuple<T_Senders...> senders, T_Receiver receiver)
                : ReceiverHolder(std::move(receiver))
                , Holder<T_Index>(std::move(std::get<T_Index>(senders)), this, &this->m_receiver)...
            {
            }

            WhenAllOperation(WhenAllOperation const&) = delete;
            WhenAllOperation& operator=(WhenAllOperation const&) = delete;
            WhenAllOperation(WhenAllOperation&&) = delete;
            WhenAllOperation& operator=(WhenAllOperation&&) = delete;

            void start() & noexcept
            {
                if constexpr(sizeof...(T_Senders) == 0u)
                    this->m_receiver.set_value();
                else
                    (Holder<T_Index>::start(), ...);
            }

            template<std::size_t T_I, typename... T>
            void setValue(T&&... values) noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    std::get<T_I>(m_values).emplace(std::forward<T>(values)...);
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

        private:
            void finish() noexcept
            {
                auto values
                    = std::apply([](auto&... value) { return std::tuple_cat(std::move(*value)...); }, m_values);
                std::apply(
                    [this](auto&&... value) { this->m_receiver.set_value(std::forward<decltype(value)>(value)...); },
                    std::move(values));
            }

            std::mutex m_mutex;
            std::size_t m_remaining = sizeof...(T_Senders);
            std::tuple<std::optional<ValueTupleOf<T_Senders>>...> m_values;
        };
    } // namespace detail

    template<typename... T_Senders>
    class WhenAllSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<
            typename detail::ValueSignatureFromTuple<detail::CombinedValueTuple<T_Senders...>>::type>;

        explicit WhenAllSender(T_Senders... senders) : m_senders(std::move(senders)...)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::
                WhenAllOperation<std::decay_t<T_Receiver>, std::index_sequence_for<T_Senders...>, T_Senders...>{
                    std::move(m_senders),
                    std::forward<T_Receiver>(receiver)};
        }

    private:
        std::tuple<T_Senders...> m_senders;
    };

    /** Join senders without domain-specific fusion. */
    template<Sender... T_Senders>
    auto genericWhenAll(T_Senders... senders)
    {
        return WhenAllSender<T_Senders...>{std::move(senders)...};
    }

    template<typename... T_Senders>
    requires(Sender<T_Senders> && ...)
    auto whenAll(T_Senders... senders)
    {
        auto domain = detail::commonDomain(senders...);
        if constexpr(requires { domain.transform(WhenAllTag{}, std::move(senders)...); })
            return domain.transform(WhenAllTag{}, std::move(senders)...);
        else
            return genericWhenAll(std::move(senders)...);
    }
} // namespace caravan
