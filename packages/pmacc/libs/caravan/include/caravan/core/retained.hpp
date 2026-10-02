/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <concepts>
#include <utility>

namespace caravan
{
    /** A value plus an owner kept solely for lifetime, independent of any async backend.
     *
     * The caller must supply an owner that keeps the value's referenced storage valid.
     * An async adapter stores this whole object until native work is quiescent, including
     * completion, and passes only unwrap(value) to the native API. This wrapper does
     * not synchronize, prevent conflicting accesses, or make a borrowed owner owning.
     */
    template<typename T_Value, typename T_Owner>
    struct Retained
    {
        T_Value value;
        T_Owner owner;

        Retained(T_Value value, T_Owner owner) : value(std::move(value)), owner(std::move(owner))
        {
        }

        template<typename T_OtherValue, typename T_OtherOwner>
        requires std::convertible_to<T_OtherValue, T_Value> && std::convertible_to<T_OtherOwner, T_Owner>
        Retained(Retained<T_OtherValue, T_OtherOwner> other)
            : value(std::move(other.value))
            , owner(std::move(other.owner))
        {
        }
    };

    /** Attach an owner by value; move-only owners can be transferred with std::move. */
    template<typename T_Value, typename T_Owner>
    auto retain(T_Value value, T_Owner owner)
    {
        return Retained{std::move(value), std::move(owner)};
    }

    /** Attach the owner of an existing retained value to another view/argument. */
    template<typename T_Value, typename T_Previous, typename T_Owner>
    auto retain(T_Value value, Retained<T_Previous, T_Owner> previous)
    {
        return Retained{std::move(value), std::move(previous.owner)};
    }

    /** Borrow a native operand without transferring its lifetime owner. */
    template<typename T>
    constexpr T& unwrap(T& value) noexcept
    {
        return value;
    }

    template<typename T_Value, typename T_Owner>
    constexpr T_Value& unwrap(Retained<T_Value, T_Owner>& value) noexcept
    {
        return value.value;
    }

    template<typename T_Value, typename T_Owner>
    constexpr T_Value const& unwrap(Retained<T_Value, T_Owner> const& value) noexcept
    {
        return value.value;
    }
} // namespace caravan
