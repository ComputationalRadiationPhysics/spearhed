/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/alpaka/completion.hpp>
#include <caravan/core/eager.hpp>

namespace caravan::alpaka
{
    template<typename T_Queue>
    struct SubmissionDomain;

    template<typename T_Queue, typename... T_Submits>
    class SubmitSender;

    namespace detail
    {
        template<std::size_t T_Count>
        struct SubmissionDependencies
        {
            /** Direct dependency edges between stages stored in topological order.
             *
             * If predecessors[stage][predecessor] is true, predecessor must be smaller than stage. Sender
             * composition and graph lowering must preserve this invariant so submission can process stages in
             * tuple order and inspect only earlier stages. The matrix uses O(N^2) storage and scans for fixed
             * submission expressions; use sparse edges if large graphs matter.
             */
            std::array<std::array<bool, T_Count>, T_Count> predecessors{};

            static auto linear()
            {
                SubmissionDependencies result;
                for(std::size_t i = 1u; i < T_Count; ++i)
                    result.predecessors[i][i - 1u] = true;
                return result;
            }
        };

        template<typename T, typename T_Queue>
        inline constexpr bool isSubmitSenderFor = false;

        template<typename T_Queue, typename... T_Submits>
        inline constexpr bool isSubmitSenderFor<SubmitSender<T_Queue, T_Submits...>, T_Queue> = true;

        template<bool T_Ordered, std::size_t T_LeftCount, std::size_t T_RightCount>
        auto composeDependencies(
            SubmissionDependencies<T_LeftCount> const& left,
            SubmissionDependencies<T_RightCount> const& right)
        {
            SubmissionDependencies<T_LeftCount + T_RightCount> dependencies;
            std::array<bool, T_LeftCount> tails;
            tails.fill(true);
            for(std::size_t i = 0u; i < T_LeftCount; ++i)
                for(std::size_t j = 0u; j < T_LeftCount; ++j)
                {
                    auto edge = left.predecessors[i][j];
                    dependencies.predecessors[i][j] = edge;
                    if(edge)
                        tails[j] = false;
                }
            for(std::size_t i = 0u; i < T_RightCount; ++i)
            {
                auto const& predecessors = right.predecessors[i];
                std::copy(
                    predecessors.begin(),
                    predecessors.end(),
                    dependencies.predecessors[T_LeftCount + i].begin() + T_LeftCount);
                if constexpr(T_Ordered)
                    if(std::none_of(predecessors.begin(), predecessors.end(), [](bool edge) { return edge; }))
                        std::copy(tails.begin(), tails.end(), dependencies.predecessors[T_LeftCount + i].begin());
            }
            return dependencies;
        }

        template<typename T_Topology, std::size_t T_NodeCount, std::size_t T_StageCount>
        void addGraphDependencies(
            SubmissionDependencies<T_StageCount>& dependencies,
            std::array<std::size_t, T_NodeCount> const& counts)
        {
            std::array<std::size_t, T_NodeCount + 1u> offsets{};
            for(std::size_t node = 0u; node < T_NodeCount; ++node)
                offsets[node + 1u] = offsets[node] + counts[node];

            for(std::size_t node = 0u; node < T_NodeCount; ++node)
                for(std::size_t predecessor = 0u; predecessor < node; ++predecessor)
                    if(T_Topology::predecessors[node][predecessor])
                        for(std::size_t root = offsets[node]; root < offsets[node + 1u]; ++root)
                        {
                            bool isRoot = true;
                            for(std::size_t candidate = offsets[node]; candidate < offsets[node + 1u]; ++candidate)
                                isRoot = isRoot && !dependencies.predecessors[root][candidate];
                            if(!isRoot)
                                continue;

                            for(std::size_t tail = offsets[predecessor]; tail < offsets[predecessor + 1u]; ++tail)
                            {
                                bool isTail = true;
                                for(std::size_t candidate = offsets[predecessor];
                                    candidate < offsets[predecessor + 1u];
                                    ++candidate)
                                    isTail = isTail && !dependencies.predecessors[candidate][tail];
                                if(isTail)
                                    dependencies.predecessors[root][tail] = true;
                            }
                        }
        }

        template<typename T_Queue, typename T_Receiver, typename... T_Submits>
        class SubmitOperation : private CompletionTask
        {
            static constexpr auto stageCount = sizeof...(T_Submits);
            using Fence = CompletionFence<T_Queue>;

        public:
            SubmitOperation(
                std::array<T_Queue*, stageCount> queues,
                std::tuple<T_Submits...> submits,
                SubmissionDependencies<stageCount> dependencies,
                T_Receiver receiver,
                EventPool<T_Queue>* eventPool = nullptr)
                : m_completionThread(completionThread())
                , m_queues(queues)
                , m_submits(std::move(submits))
                , m_dependencies(dependencies)
                , m_receiver(std::move(receiver))
            {
                std::array<std::size_t, stageCount> incoming{}, outgoing{};
                for(std::size_t i = 0u; i < stageCount; ++i)
                    for(std::size_t j = 0u; j < i; ++j)
                        if(m_dependencies.predecessors[i][j])
                        {
                            ++incoming[i];
                            ++outgoing[j];
                        }

                // Preallocate fences before borrowing captures. Only unbranched, adjacent same-queue stages
                // share a fence. whenAll(A, B) -> C on A's queue retains A's branch fence despite FIFO ordering.
                for(std::size_t i = stageCount; i-- > 0u;)
                {
                    if(i + 1u < stageCount && *m_queues[i] == *m_queues[i + 1u]
                       && m_dependencies.predecessors[i + 1u][i] && outgoing[i] == 1u && incoming[i + 1u] == 1u)
                        m_fenceIndices[i] = m_fenceIndices[i + 1u];
                    else
                    {
                        m_fenceIndices[i] = i;
                        m_fences[i].emplace(*m_queues[i], eventPool);
                    }
                }
            }

            SubmitOperation(SubmitOperation const&) = delete;
            SubmitOperation& operator=(SubmitOperation const&) = delete;
            SubmitOperation(SubmitOperation&&) = delete;
            SubmitOperation& operator=(SubmitOperation&&) = delete;

            /** Submit every stage without making completion observable yet.
             *
             * Queue-pool bindings use this so their submission locks can be released before publish() can run a
             * receiver and destroy the connected operation.
             */
            void submit() & noexcept
            {
                submitStage<0u>();
            }

            /** Publish a submitted operation to completion progress.
             *
             * Receiver delivery may destroy this operation, so callers must not access it after this call.
             */
            void publish() & noexcept
            {
                m_completionThread.post(*this);
            }

            void start(std::mutex* submissionMutex = nullptr) & noexcept
            {
                if(submissionMutex)
                {
                    std::lock_guard lock(*submissionMutex);
                    submit();
                }
                else
                    submit();
                // Publish only after unlocking: completion can destroy this operation and start another graph.
                publish();
            }

        private:
            template<std::size_t T_Index>
            void submitStage() noexcept
            {
                auto& queue = *m_queues[T_Index];
                auto& fence = m_fences[m_fenceIndices[T_Index]];
                for(std::size_t i = 0u; i != T_Index; ++i)
                    if(m_dependencies.predecessors[T_Index][i] && queue != *m_queues[i])
                        m_fences[m_fenceIndices[i]]->waitOn(queue);
                std::invoke(std::get<T_Index>(m_submits), queue);
                if(m_fenceIndices[T_Index] == T_Index)
                    fence->record(queue);

                if constexpr(T_Index + 1u < stageCount)
                    submitStage<T_Index + 1u>();
            }

            bool poll() noexcept override
            {
                bool ready = true;
                for(auto& fence : m_fences)
                    if(fence && !fence->poll())
                        ready = false;
                if(!ready)
                    return false;
                // Return event leases only after the whole graph is quiescent, before a receiver starts more work.
                for(auto& fence : m_fences)
                    fence.reset();
                m_receiver.set_value();
                // The receiver may destroy this operation. Do not access members below this point.
                return true;
            }

            CompletionThread& m_completionThread;
            std::array<T_Queue*, stageCount> m_queues;
            std::tuple<T_Submits...> m_submits;
            SubmissionDependencies<stageCount> m_dependencies;
            std::array<std::optional<Fence>, stageCount> m_fences;
            std::array<std::size_t, stageCount> m_fenceIndices{};
            T_Receiver m_receiver;
        };

        /** Explicit deduction guide for host compilers that cannot form the implicit guide.
         *
         * The pack appears in the `sizeof...(T_Submits)` array/dependency parameters before the tuple
         * parameter that deduces it. nvcc rejects the implicit guide for this ordering, so spell it out.
         */
        template<typename T_Queue, typename T_Receiver, typename... T_Submits>
        SubmitOperation(
            std::array<T_Queue*, sizeof...(T_Submits)>,
            std::tuple<T_Submits...>,
            SubmissionDependencies<sizeof...(T_Submits)>,
            T_Receiver,
            EventPool<T_Queue>* = nullptr) -> SubmitOperation<T_Queue, T_Receiver, T_Submits...>;
    } // namespace detail

    /** Compatibility query: Caravan terminal completion no longer runs in alpaka callbacks. */
    inline bool isCompletionCallback() noexcept
    {
        return false;
    }

    /** Lazy alpaka-native submissions over borrowed caller-supplied queues.
     *
     * Every queue must outlive the connected operation. Captures are retained until all submitted work is quiescent;
     * storage referenced by unowned views remains borrowed. Callables run on the submitting host thread and must
     * only enqueue tracked work on their supplied queue, not read unfinished results or block for completion.
     * sequence uses FIFO/events; whenAll preserves independent branches. Ordinary then/letValue
     * callbacks and explicit placement wrappers remain host-completion boundaries. Unbranched same-queue runs
     * share a fence. A shared progress thread polls all recorded fences before terminal completion and reclamation.
     * Backend and submission failures crossing Caravan's noexcept execution boundary terminate the process.
     * Ordinary queue events establish completion, not task success. Asynchronous host callbacks must not let
     * exceptions escape; native submissions are a trusted escape hatch and callers must enforce this contract.
     * Receivers run on an executor thread: use continuesOn before blocking callbacks.
     */
    template<typename T_Queue, typename... T_Submits>
    class SubmitSender
    {
        static constexpr auto stageCount = sizeof...(T_Submits);
        static_assert(stageCount > 0u, "An alpaka submission chain must contain at least one stage");

    public:
        static constexpr auto stage_count = stageCount;
        using completion_signatures = CompletionSignatures<ValueSignature<>>;

        SubmitSender(
            std::array<T_Queue*, stageCount> queues,
            std::tuple<T_Submits...> submits,
            detail::SubmissionDependencies<stageCount> dependencies
            = detail::SubmissionDependencies<stageCount>::linear())
            : m_queues(queues)
            , m_submits(std::move(submits))
            , m_dependencies(dependencies)
        {
        }

        auto query(GetDomain) const noexcept -> SubmissionDomain<T_Queue>
        {
            return {};
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::SubmitOperation<T_Queue, std::decay_t<T_Receiver>, T_Submits...>{
                m_queues,
                std::move(m_submits),
                m_dependencies,
                std::forward<T_Receiver>(receiver)};
        }

        template<typename, typename...>
        friend class SubmitSender;

        friend struct SubmissionDomain<T_Queue>;

        template<typename T_OtherQueue, typename... T_Left, typename... T_Right>
        friend auto sequence(SubmitSender<T_OtherQueue, T_Left...>, SubmitSender<T_OtherQueue, T_Right...>);

    private:
        template<bool T_Ordered, typename... T_Right>
        auto compose(SubmitSender<T_Queue, T_Right...> right) &&
        {
            constexpr auto rightCount = sizeof...(T_Right);
            std::array<T_Queue*, stageCount + rightCount> queues;
            auto output = std::copy(m_queues.begin(), m_queues.end(), queues.begin());
            std::copy(right.m_queues.begin(), right.m_queues.end(), output);
            return SubmitSender<T_Queue, T_Submits..., T_Right...>{
                queues,
                std::tuple_cat(std::move(m_submits), std::move(right.m_submits)),
                detail::composeDependencies<T_Ordered>(m_dependencies, right.m_dependencies)};
        }

        std::array<T_Queue*, stageCount> m_queues;
        std::tuple<T_Submits...> m_submits;
        detail::SubmissionDependencies<stageCount> m_dependencies;
    };

    /** Native lowering only for explicit submissions. Queue type compatibility is not queue identity. */
    template<typename T_Queue>
    struct SubmissionDomain
    {
        template<typename... T_Left, typename... T_Right>
        auto transform(SequenceTag, SubmitSender<T_Queue, T_Left...> left, SubmitSender<T_Queue, T_Right...> right)
            const
        {
            return std::move(left).template compose<true>(std::move(right));
        }

        template<typename... T_Submits>
        auto transform(WhenAllTag, SubmitSender<T_Queue, T_Submits...> sender) const
        {
            return sender;
        }

        template<typename... T_Left, typename... T_Right, typename... T_Rest>
        auto transform(
            WhenAllTag tag,
            SubmitSender<T_Queue, T_Left...> left,
            SubmitSender<T_Queue, T_Right...> right,
            T_Rest... rest) const
        {
            return transform(tag, std::move(left).template compose<false>(std::move(right)), std::move(rest)...);
        }

        template<caravan::detail::GraphNodeType... T_Nodes>
        requires(detail::isSubmitSenderFor<typename T_Nodes::sender_type, T_Queue> && ...)
        auto transform(GraphTag, T_Nodes... nodes) const
        {
            auto flattened = merge(std::move(nodes).releaseSender()...);
            detail::addGraphDependencies<caravan::detail::GraphTopology<T_Nodes...>>(
                flattened.m_dependencies,
                std::array<std::size_t, sizeof...(T_Nodes)>{T_Nodes::sender_type::stage_count...});
            return flattened;
        }

    private:
        template<typename T_First>
        static auto merge(T_First first)
        {
            return first;
        }

        template<typename T_First, typename T_Second, typename... T_Rest>
        static auto merge(T_First first, T_Second second, T_Rest... rest)
        {
            return merge(std::move(first).template compose<false>(std::move(second)), std::move(rest)...);
        }
    };

    /** Lazily describe one native submission stage. The queue is borrowed. */
    template<typename T_Queue, typename T_Submit>
    auto submit(T_Queue& queue, T_Submit submit)
    {
        static_assert(detail::QueueHandle<T_Queue>);
        using Submit = std::decay_t<T_Submit>;
        return SubmitSender<T_Queue, Submit>{{&queue}, {std::move(submit)}};
    }

    /** Alpaka-native sequencing preserving FIFO/events instead of crossing host-visible completion. */
    template<typename T_Queue, typename... T_Left, typename... T_Right>
    auto sequence(SubmitSender<T_Queue, T_Left...> left, SubmitSender<T_Queue, T_Right...> right)
    {
        return std::move(left).template compose<true>(std::move(right));
    }

    /** Pipe adaptor preserving alpaka-native sequencing: previous | sequence(next). */
    template<typename T_Queue, typename... T_Submits>
    auto sequence(SubmitSender<T_Queue, T_Submits...> next)
    {
        return caravan::detail::SenderAdaptorClosure{
            [next = std::move(next)](auto previous) mutable
            { return caravan::alpaka::sequence(std::move(previous), std::move(next)); }};
    }
} // namespace caravan::alpaka
