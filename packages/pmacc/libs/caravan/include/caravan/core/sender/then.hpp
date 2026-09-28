/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<typename T_Sender, typename T_Function, typename T_Receiver>
        class ThenOperation
        {
            struct PredecessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->completeValue(std::forward<T>(values)...);
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                ThenOperation* owner;
            };

        public:
            ThenOperation(T_Sender sender, T_Function function, T_Receiver receiver)
                : m_function(std::move(function))
                , m_receiver(std::move(receiver))
                , m_predecessor(std::move(sender).connect(PredecessorReceiver{this}))
            {
            }

            ThenOperation(ThenOperation const&) = delete;
            ThenOperation& operator=(ThenOperation const&) = delete;
            ThenOperation(ThenOperation&&) = delete;
            ThenOperation& operator=(ThenOperation&&) = delete;

            void start() & noexcept
            {
                m_predecessor.start();
            }

        private:
            template<typename... T>
            void completeValue(T&&... values) noexcept
            {
                if constexpr(std::is_void_v<std::invoke_result_t<T_Function&, T...>>)
                {
                    std::invoke(m_function, std::forward<T>(values)...);
                    m_receiver.set_value();
                }
                else
                    m_receiver.set_value(std::invoke(m_function, std::forward<T>(values)...));
            }

            T_Function m_function;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<PredecessorReceiver>())) m_predecessor;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Function>
    class ThenSender
    {
    public:
        using completion_signatures = detail::ThenCompletionSignatures<T_Sender, T_Function>;

        ThenSender(T_Sender sender, T_Function function) : m_sender(std::move(sender)), m_function(std::move(function))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ThenOperation<T_Sender, T_Function, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_function),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Function m_function;
    };

    template<Sender T_Sender, typename T_Function>
    auto then(T_Sender sender, T_Function function)
    {
        return ThenSender<T_Sender, T_Function>{std::move(sender), std::move(function)};
    }

    template<typename T_Function>
    auto then(T_Function function)
    {
        return detail::SenderAdaptorClosure{[function = std::move(function)](auto sender) mutable
                                            { return caravan::then(std::move(sender), std::move(function)); }};
    }
} // namespace caravan
