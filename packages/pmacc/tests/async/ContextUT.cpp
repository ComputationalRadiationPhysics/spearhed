/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>

#include <chrono>
#include <stdexcept>
#include <thread>

#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PMacc environment owns one accelerator queue", "[async]")
{
    CHECK(pmacc::Environment<>::get().DeviceContext().size() == 1u);
}

TEST_CASE("PMacc async context owns work and drives host continuations", "[async]")
{
    caravan::ControlContext context;
    auto const applicationThread = std::this_thread::get_id();
    std::thread::id completionThread;
    bool ran = false;
    caravan::EventSource backendCompletion;

    auto operation = context.spawn(
        context.onControl(caravan::asSender(backendCompletion.event()))
        | caravan::then(
            [&]
            {
                ran = true;
                CHECK(std::this_thread::get_id() == applicationThread);
                CHECK(std::this_thread::get_id() != completionThread);
            }));

    std::thread backend(
        [&]
        {
            completionThread = std::this_thread::get_id();
            backendCompletion.setReady();
        });
    CHECK(operation.state() == caravan::CompletionState::pending);
    context.wait(operation);
    backend.join();
    CHECK(ran);

    caravan::EventSource externalCompletion;
    std::thread external(
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            externalCompletion.setReady();
        });
    context.wait(externalCompletion.event());
    external.join();

    caravan::EventSource progressed;
    size_t progressCalls = 0u;
    context.wait(
        progressed.event(),
        [&]
        {
            ++progressCalls;
            progressed.setReady();
        });
    CHECK(progressCalls == 1u);

    caravan::EventSource pending;
    auto checked = context.spawn(
        context.scheduler().schedule()
        | caravan::then([&] { CHECK_THROWS_AS(context.wait(pending.event()), std::logic_error); }));
    context.wait(checked);
}

TEST_CASE("PMacc wait survives progress errors", "[async]")
{
    caravan::ControlContext context;
    caravan::EventSource pending;
    CHECK_THROWS_AS(
        context.wait(pending.event(), [] { throw std::runtime_error("progress failure"); }),
        std::runtime_error);
    pending.setReady(); // The earlier wait's wakeup must not reference its destroyed stack.
    context.runReady();
    context.wait(pending.event());
    // Context destruction also waits on the already-joining scope.
}
