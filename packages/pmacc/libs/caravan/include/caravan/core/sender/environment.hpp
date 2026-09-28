/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <type_traits>
#include <utility>

namespace caravan
{
    /** Query the logical current scheduler, not the thread delivering completion.
     *
     * Environments opt in with query(GetScheduler). There is no default scheduler.
     */
    struct GetScheduler
    {
        template<typename T_Environment>
        auto operator()(T_Environment const& environment) const noexcept(noexcept(environment.query(*this)))
            -> decltype(environment.query(*this))
        {
            return environment.query(*this);
        }
    };

    inline constexpr GetScheduler getScheduler{};

    struct DefaultDomain
    {
    };

    /** Eager customization for explicitly typed backend descriptions, not completion placement. */
    struct GetDomain
    {
        template<typename T>
        auto operator()(T const& value) const noexcept(noexcept(value.query(*this)))
            requires requires { value.query(*this); }
        {
            return value.query(*this);
        }

        template<typename T>
        DefaultDomain operator()(T const&) const noexcept requires(!requires(T const& value) { value.query(*this); })
        {
            return {};
        }
    };

    inline constexpr GetDomain getDomain{};

    namespace detail
    {
        struct EmptyEnvironment
        {
        };

        template<typename T_Receiver>
        decltype(auto) getEnvironment(T_Receiver const& receiver) noexcept(noexcept(receiver.get_env()))
            requires requires { receiver.get_env(); }
        {
            return receiver.get_env();
        }

        template<typename T_Receiver>
        EmptyEnvironment getEnvironment(T_Receiver const&) noexcept
            requires(!requires(T_Receiver const& receiver) { receiver.get_env(); })
        {
            return {};
        }

        /** Borrowed overlay; unrelated query CPOs are forwarded to the receiver's environment. */
        template<typename T_Scheduler, typename T_Receiver>
        struct SchedulerEnvironment
        {
            T_Scheduler query(GetScheduler) const noexcept(std::is_nothrow_copy_constructible_v<T_Scheduler>)
            {
                return *scheduler;
            }

            template<typename T_Query>
            requires(!std::is_same_v<T_Query, GetScheduler>)
            auto query(T_Query query) const
                noexcept(noexcept(query(getEnvironment(std::declval<T_Receiver const&>()))))
                    -> decltype(query(getEnvironment(std::declval<T_Receiver const&>())))
            {
                return query(getEnvironment(*receiver));
            }

            T_Scheduler const* scheduler;
            T_Receiver const* receiver;
        };
    } // namespace detail
} // namespace caravan
