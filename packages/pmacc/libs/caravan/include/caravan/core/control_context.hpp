/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <stdexcept>
#include <utility>

#include <caravan/core/async_scope.hpp>
#include <caravan/core/run_loop.hpp>
#include <caravan/core/sender.hpp>

namespace caravan
{
    /** Manually driven host event loop combined with a scope owning asynchronous work.
     *
     * This convenience policy has the following semantics:
     * - onControl() returns a lazy sender transferring completion to this loop;
     * - spawn() starts and owns a sender, transfers terminal completion to this
     *   loop, and returns an eager Event;
     * - spawnFuture() does the same while retaining one result value;
     * - runReady() executes a snapshot of queued control work without waiting;
     * - wait() pumps control work until its Event completes, then reports the
     *   ready outcome; and
     * - destruction closes and joins the scope while pumping the loop, then
     *   finishes the loop, so destruction may block.
     *
     * No worker thread is created. Control work runs on whichever thread calls
     * wait() or runReady(); "control" expresses intended ownership, not an
     * enforced thread identity. Attach application continuations after
     * onControl(): wrapping an existing chain in spawn() transfers only that
     * chain's terminal completion.
     *
     * This is one policy, not a required Caravan architecture. Applications can
     * instead drive a RunLoop on a dedicated thread, integrate scheduling with
     * an existing GUI/event loop, or pair RunLoop and AsyncScope with explicit
     * shutdown rather than blocking destruction.
     */
    class ControlContext
    {
    public:
        ControlContext() = default;
        ControlContext(ControlContext const&) = delete;
        ControlContext& operator=(ControlContext const&) = delete;

        ~ControlContext()
        {
            wait(m_scope.join());
            m_loop.finish();
        }

        /** Transfer completion to the control loop before attaching host continuations. */
        template<typename T_Sender>
        auto onControl(T_Sender sender)
        {
            return continuesOn(std::move(sender), m_loop.scheduler());
        }

        /** Own and start work; only terminal completion is transferred to the control loop. */
        template<typename T_Sender>
        Event spawn(T_Sender sender)
        {
            return m_scope.spawn(onControl(std::move(sender)));
        }

        template<typename T, typename T_Sender>
        Future<T> spawnFuture(T_Sender sender)
        {
            return m_scope.spawnFuture<T>(onControl(std::move(sender)));
        }

        /** Drive control work until this operation is terminal. */
        void wait(Event const& event)
        {
            wait(event, [] {});
        }

        /** Drive control work and an application progress hook until completion. */
        template<typename T_Progress>
        void wait(Event const& event, T_Progress progress)
        {
            if(isExecutorThread() && event.state() == CompletionState::pending)
                throw std::logic_error("A Caravan control continuation cannot wait on pending work");
            auto scheduler = m_loop.scheduler();
            // This wakeup also works while m_scope is joining; a stack operation
            // or spawn into that scope cannot provide that guarantee.
            auto wake = event.continueWith(scheduler, [](Event) {});
            static_cast<void>(wake);
            while(event.state() == CompletionState::pending)
            {
                progress();
                m_loop.runOne();
            }
            event.wait();
        }

        /** Execute a snapshot of queued control work without waiting for more. */
        void runReady()
        {
            m_loop.runReady();
        }

        /** Close the scope to new work and pump the control loop until every owned
         * operation is terminal.
         *
         * Idempotent. Call this before destroying or resetting resources borrowed by
         * outstanding operations (for example fields, particle buffers, or queues).
         * After the first call, `spawn()` rejects new work.
         */
        void drain()
        {
            wait(m_scope.join());
        }

        RunLoopScheduler scheduler() noexcept
        {
            return m_loop.scheduler();
        }

    private:
        RunLoop m_loop;
        AsyncScope m_scope;
    };
} // namespace caravan
