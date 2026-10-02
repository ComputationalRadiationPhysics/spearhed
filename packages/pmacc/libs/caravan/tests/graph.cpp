/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <cassert>
#include <memory>
#include <thread>

#include <caravan/core.hpp>

namespace
{
    struct EnvironmentReceiver
    {
        void set_value() noexcept
        {
            *completed = true;
        }

        int get_env() const noexcept
        {
            return 42;
        }

        bool* completed;
    };

    struct EnvironmentSender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

        template<typename T_Receiver>
        struct Operation
        {
            void start() & noexcept
            {
                assert(receiver.get_env() == 42);
                receiver.set_value();
            }

            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            assert(receiver.get_env() == 42);
            return Operation<std::decay_t<T_Receiver>>{std::forward<T_Receiver>(receiver)};
        }
    };

    struct ImmediateSender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

        template<typename T_Receiver>
        struct Operation
        {
            void start() & noexcept
            {
                ++*starts;
                receiver.set_value();
            }

            unsigned* starts;
            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{starts, std::forward<T_Receiver>(receiver)};
        }

        unsigned* starts;
    };

    void testArbitraryDag()
    {
        caravan::EventSource aReady, bReady;
        unsigned cStarts = 0u, dStarts = 0u;
        auto a = caravan::node<"a">(caravan::asSender(aReady.event()));
        auto b = caravan::node<"b">(caravan::asSender(bReady.event()));
        auto c = caravan::node<"c">(ImmediateSender{&cStarts}, caravan::after(a, b));
        auto d = caravan::node<"d">(ImmediateSender{&dStarts}, caravan::after(b));

        auto work = caravan::graph(std::move(a), std::move(b), std::move(c), std::move(d));
        static_assert(caravan::Sender<decltype(work)>);
        assert(cStarts == 0u && dStarts == 0u);

        caravan::AsyncScope scope;
        auto completion = scope.spawn(std::move(work));
        assert(cStarts == 0u && dStarts == 0u);
        bReady.setReady();
        assert(cStarts == 0u && dStarts == 1u);
        assert(completion.state() == caravan::CompletionState::pending);
        aReady.setReady();
        completion.wait();
        assert(cStarts == 1u && dStarts == 1u);
        scope.join().wait();
    }

    void testConcurrentConvergence()
    {
        for(unsigned iteration = 0u; iteration < 200u; ++iteration)
        {
            caravan::EventSource aReady, bReady;
            unsigned successorStarts = 0u;
            auto a = caravan::node<"a">(caravan::asSender(aReady.event()));
            auto b = caravan::node<"b">(caravan::asSender(bReady.event()));
            auto successor = caravan::node<"successor">(ImmediateSender{&successorStarts}, caravan::after(a, b));

            caravan::AsyncScope scope;
            auto completion = scope.spawn(caravan::graph(std::move(a), std::move(b), std::move(successor)));
            std::thread first([&] { aReady.setReady(); });
            std::thread second([&] { bReady.setReady(); });
            first.join();
            second.join();
            completion.wait();
            assert(successorStarts == 1u);
            scope.join().wait();
        }
    }

    void testEnvironmentForwarding()
    {
        auto first = caravan::node<"first">(EnvironmentSender{});
        auto second = caravan::node<"second">(EnvironmentSender{}, caravan::after(first));
        bool completed = false;
        auto operation = caravan::graph(std::move(first), std::move(second)).connect(EnvironmentReceiver{&completed});
        operation.start();
        assert(completed);
    }

    void testMoveOnlyNode()
    {
        unsigned starts = 0u;
        auto owned = caravan::node<"owned">(
            caravan::InlineScheduler{}.schedule()
            | caravan::then(
                [value = std::make_unique<int>(42), &starts]
                {
                    assert(*value == 42);
                    ++starts;
                }));
        caravan::syncWait(caravan::graph(std::move(owned)));
        assert(starts == 1u);
    }

    using A = decltype(caravan::node<"a">(ImmediateSender{nullptr}));
    using DuplicateA = decltype(caravan::node<"a">(ImmediateSender{nullptr}));
    using BAfterA = decltype(caravan::node<"b">(ImmediateSender{nullptr}, caravan::after(std::declval<A const&>())));
    static_assert(caravan::detail::GraphTopology<A, BAfterA>::hasUniqueNames);
    static_assert(caravan::detail::GraphTopology<A, BAfterA>::isTopologicallyOrdered);
    static_assert(!caravan::detail::GraphTopology<A, DuplicateA>::hasUniqueNames);
    static_assert(!caravan::detail::GraphTopology<BAfterA, A>::isTopologicallyOrdered);
} // namespace

int main()
{
    testArbitraryDag();
    testConcurrentConvergence();
    testEnvironmentForwarding();
    testMoveOnlyNode();
}
