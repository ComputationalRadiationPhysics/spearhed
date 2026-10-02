/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan
{
    enum class AsyncScopeStatus : std::uint8_t
    {
        open,
        joining,
        joined
    };

    namespace detail
    {
        class AsyncScopeState
        {
        public:
            void reserve()
            {
                std::lock_guard lock(m_mutex);
                if(m_status != AsyncScopeStatus::open)
                    throw std::logic_error("Cannot spawn into a joined Caravan async scope");
                ++m_operations;
            }

            void complete() noexcept
            {
                bool joined = false;
                {
                    std::lock_guard lock(m_mutex);
                    --m_operations;
                    joined = m_status == AsyncScopeStatus::joining && m_operations == 0u;
                    if(joined)
                        m_status = AsyncScopeStatus::joined;
                }
                if(joined)
                    m_joined.setReady();
            }

            Event join()
            {
                bool joined = false;
                {
                    std::lock_guard lock(m_mutex);
                    if(m_status == AsyncScopeStatus::open)
                        m_status = AsyncScopeStatus::joining;
                    joined = m_operations == 0u;
                    if(joined)
                        m_status = AsyncScopeStatus::joined;
                }
                if(joined)
                    m_joined.setReady();
                return m_joined.event();
            }

            AsyncScopeStatus status() const noexcept
            {
                std::lock_guard lock(m_mutex);
                return m_status;
            }

        private:
            mutable std::mutex m_mutex;
            std::size_t m_operations = 0u;
            AsyncScopeStatus m_status = AsyncScopeStatus::open;
            EventSource m_joined;
        };

        struct SpawnOwner
        {
            void* operation;
            void (*destroy)(void*) noexcept;
        };

        inline void finishSpawn(SpawnOwner owner, std::shared_ptr<AsyncScopeState> scope) noexcept
        {
            owner.destroy(owner.operation);
            scope->complete();
        }

        template<typename T_Output>
        struct ScopeReceiver
        {
            template<typename... T>
            void set_value(T&&...) noexcept
            {
                output.setReady();
                finishSpawn(owner, std::move(scope));
            }

            std::shared_ptr<AsyncScopeState> scope;
            T_Output output;
            SpawnOwner owner;
        };

        template<typename T_Output>
        struct FutureScopeReceiver
        {
            template<typename U>
            void set_value(U&& value) noexcept
            {
                output.setValue(std::forward<U>(value));
                finishSpawn(owner, std::move(scope));
            }

            std::shared_ptr<AsyncScopeState> scope;
            T_Output output;
            SpawnOwner owner;
        };

        template<typename T_Sender, template<typename> typename T_Receiver, typename T_Output>
        class SpawnOperation
        {
        public:
            using Receiver = T_Receiver<T_Output>;

            SpawnOperation(T_Sender sender, std::shared_ptr<AsyncScopeState> scope, T_Output output)
                : m_operation(std::move(sender).connect(Receiver{std::move(scope), std::move(output), owner()}))
            {
            }

            void start() noexcept
            {
                m_operation.start();
            }

        private:
            static void destroy(void* operation) noexcept
            {
                delete static_cast<SpawnOperation*>(operation);
            }

            SpawnOwner owner() noexcept
            {
                return {this, destroy};
            }

            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };
    } // namespace detail

    /** Owns eagerly spawned sender operations until receiver completion.
     *
     * join() closes the scope to new work and completes when every operation is
     * quiescent. The owner must provide progress and wait for that Event before
     * destruction; destroying an unjoined or non-quiescent scope terminates
     * instead of attempting hidden, potentially unbounded progress. The returned
     * Event reports completion; values are deliberately type-erased at this boundary.
     * After reserving a scope entry, construction and execution failures are fatal.
     */
    class AsyncScope
    {
    public:
        AsyncScope() : m_state(std::make_shared<detail::AsyncScopeState>())
        {
        }

        AsyncScope(AsyncScope const&) = delete;
        AsyncScope& operator=(AsyncScope const&) = delete;

        ~AsyncScope()
        {
            if(status() != AsyncScopeStatus::joined)
                std::terminate();
        }

        template<typename T_Sender>
        Event spawn(T_Sender sender)
        {
            EventSource output;
            auto result = output.event();
            m_state->reserve();
            [&]() noexcept
            {
                using Operation = detail::SpawnOperation<T_Sender, detail::ScopeReceiver, EventSource>;
                auto* operation = new Operation(std::move(sender), m_state, output);
                operation->start();
            }();
            return result;
        }

        /** Eagerly spawn a single-value sender and retain its operation state. */
        template<typename T, typename T_Sender>
        Future<T> spawnFuture(T_Sender sender)
        {
            Promise<T> output;
            auto result = output.future();
            m_state->reserve();
            [&]() noexcept
            {
                using Operation = detail::SpawnOperation<T_Sender, detail::FutureScopeReceiver, Promise<T>>;
                auto* operation = new Operation(std::move(sender), m_state, output);
                operation->start();
            }();
            return result;
        }

        Event join()
        {
            return m_state->join();
        }

        AsyncScopeStatus status() const noexcept
        {
            return m_state->status();
        }

    private:
        std::shared_ptr<detail::AsyncScopeState> m_state;
    };
} // namespace caravan
