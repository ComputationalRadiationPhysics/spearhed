/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <cassert>
#include <memory>
#include <type_traits>

#include <caravan/core/retained.hpp>

int main()
{
    auto allocation = std::make_shared<int>(42);
    std::weak_ptr<int> lifetime = allocation;
    {
        auto value = caravan::retain(allocation.get(), allocation);
        allocation.reset();
        auto argument = caravan::retain(value.value, std::as_const(value));
        static_assert(std::is_same_v<decltype(argument.owner), std::shared_ptr<int>>);
        assert(argument.owner == value.owner);
        value.owner.reset();
        assert(!lifetime.expired());
        assert(*caravan::unwrap(argument) == 42);
        assert(&caravan::unwrap(argument) == &argument.value);
        static_assert(std::is_same_v<decltype(caravan::unwrap(std::as_const(argument))), int* const&>);
        caravan::Retained<int const*, std::shared_ptr<void>> converted = argument;
        assert(*converted.value == 42 && converted.owner == argument.owner);
    }
    assert(lifetime.expired());

    auto unique = std::make_unique<int>(7);
    auto* pointer = unique.get();
    auto value = caravan::retain(pointer, std::move(unique));
    auto argument = caravan::retain(pointer, std::move(value));
    static_assert(std::is_same_v<decltype(argument.owner), std::unique_ptr<int>>);
    assert(!unique && !value.owner);
    assert(*caravan::unwrap(argument) == 7);
    assert(&caravan::unwrap(pointer) == &pointer); // Unwrapped arguments remain borrowed.
}
