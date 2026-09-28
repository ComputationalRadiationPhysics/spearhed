/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/alpaka/queue/queue_pool.hpp>
#include <caravan/core/sender/common.hpp>

namespace caravan::alpaka
{
    /** Query the queue-management resource bound to the receiver environment. */
    struct GetDeviceContext
    {
        template<typename T_Environment>
        auto operator()(T_Environment const& environment) const noexcept(noexcept(environment.query(*this)))
            -> decltype(environment.query(*this))
        {
            return environment.query(*this);
        }
    };

    inline constexpr GetDeviceContext getDeviceContext{};

    /** Default automatically growing device context for nonblocking alpaka queues. */
    template<typename T_Device>
    using Context = QueuePool<::alpaka::onHost::Queue<T_Device, ::alpaka::queueKind::NonBlocking>>;

    template<typename... T_Submits>
    class ManagedSubmitSender;

    struct ManagedSubmissionDomain;

    namespace detail
    {
        template<typename T>
        inline constexpr bool isManagedSubmitSender = false;

        template<typename... T_Submits>
        inline constexpr bool isManagedSubmitSender<ManagedSubmitSender<T_Submits...>> = true;

        template<typename T_Context, typename T_Receiver>
        struct DeviceContextEnvironment
        {
            T_Context& query(GetDeviceContext) const noexcept
            {
                return *context;
            }

            template<typename T_Query>
            requires(!std::is_same_v<T_Query, GetDeviceContext>)
            auto query(T_Query query) const
                noexcept(noexcept(query(caravan::detail::getEnvironment(std::declval<T_Receiver const&>()))))
                    -> decltype(query(caravan::detail::getEnvironment(std::declval<T_Receiver const&>())))
            {
                return query(caravan::detail::getEnvironment(*receiver));
            }

            T_Context* context;
            T_Receiver const* receiver;
        };

        template<typename T_Context, typename T_Receiver>
        struct DeviceContextReceiver
        {
            template<typename... T>
            void set_value(T&&... values) noexcept
            {
                receiver.set_value(std::forward<T>(values)...);
            }

            auto get_env() const noexcept
            {
                return DeviceContextEnvironment<T_Context, T_Receiver>{context, &receiver};
            }

            T_Context* context;
            T_Receiver receiver;
        };

        template<typename T_Receiver>
        concept ReceiverWithDeviceContext
            = requires(T_Receiver const& receiver) { getDeviceContext(caravan::detail::getEnvironment(receiver)); };

        struct MissingDeviceContextOperation
        {
            void start() & noexcept
            {
            }
        };
    } // namespace detail

    /** Bind a queue-free alpaka graph to a device context through its receiver environment.
     *
     * Scheduler changes forward this independent property. A nested withDevice explicitly overrides it.
     * The context is borrowed and must outlive the connected operation.
     */
    template<typename T_Context, typename T_Sender>
    class WithDeviceSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<T_Sender>;

        WithDeviceSender(T_Context& context, T_Sender sender) : m_context(&context), m_sender(std::move(sender))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            using Receiver = std::decay_t<T_Receiver>;
            return std::move(m_sender).connect(
                detail::DeviceContextReceiver<T_Context, Receiver>{m_context, std::forward<T_Receiver>(receiver)});
        }

    private:
        T_Context* m_context;
        T_Sender m_sender;
    };

    template<typename T_Context, Sender T_Sender>
    auto withDevice(T_Context& context, T_Sender sender)
    {
        return WithDeviceSender<T_Context, T_Sender>{context, std::move(sender)};
    }

    template<typename T_Context>
    auto withDevice(T_Context& context)
    {
        return caravan::detail::SenderAdaptorClosure{[context = &context](auto sender)
                                                     { return withDevice(*context, std::move(sender)); }};
    }

    /** Queue-free alpaka submissions bound from the receiver environment at connection. */
    template<typename... T_Submits>
    class ManagedSubmitSender
    {
        static constexpr auto stageCount = sizeof...(T_Submits);
        static_assert(stageCount > 0u, "An alpaka submission chain must contain at least one stage");

    public:
        static constexpr auto stage_count = stageCount;
        using completion_signatures = CompletionSignatures<ValueSignature<>>;

        ManagedSubmitSender(
            std::tuple<T_Submits...> submits,
            std::array<std::size_t, stageCount> lanes = {},
            std::size_t laneCount = 1u,
            detail::SubmissionDependencies<stageCount> dependencies
            = detail::SubmissionDependencies<stageCount>::linear())
            : m_submits(std::move(submits))
            , m_lanes(lanes)
            , m_laneCount(laneCount)
            , m_dependencies(dependencies)
        {
        }

        auto query(GetDomain) const noexcept -> ManagedSubmissionDomain;

        template<detail::ReceiverWithDeviceContext T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            auto& context = getDeviceContext(caravan::detail::getEnvironment(receiver));
            using Context = std::remove_reference_t<decltype(context)>;
            return detail::PooledSubmitOperation<Context, std::decay_t<T_Receiver>, T_Submits...>{
                context,
                m_laneCount,
                m_lanes,
                std::move(m_submits),
                m_dependencies,
                std::forward<T_Receiver>(receiver)};
        }

        template<typename T_Receiver>
        requires(!detail::ReceiverWithDeviceContext<T_Receiver>)
        auto connect(T_Receiver&&) && -> detail::MissingDeviceContextOperation
        {
            static_assert(
                detail::ReceiverWithDeviceContext<T_Receiver>,
                "Caravan managed alpaka sender has no device context; wrap it with "
                "caravan::alpaka::withDevice(context, sender)");
            return {};
        }

        template<typename...>
        friend class ManagedSubmitSender;

        friend struct ManagedSubmissionDomain;

        template<typename... T_Left, typename... T_Right>
        friend auto sequence(ManagedSubmitSender<T_Left...>, ManagedSubmitSender<T_Right...>);

    private:
        template<bool T_Ordered, typename... T_Right>
        auto compose(ManagedSubmitSender<T_Right...> right) &&
        {
            constexpr auto rightCount = sizeof...(T_Right);
            std::array<std::size_t, stageCount + rightCount> lanes;
            auto output = std::copy(m_lanes.begin(), m_lanes.end(), lanes.begin());
            if constexpr(T_Ordered)
                std::copy(right.m_lanes.begin(), right.m_lanes.end(), output);
            else
                std::transform(
                    right.m_lanes.begin(),
                    right.m_lanes.end(),
                    output,
                    [offset = m_laneCount](std::size_t lane) { return offset + lane; });

            auto laneCount = T_Ordered ? std::max(m_laneCount, right.m_laneCount) : m_laneCount + right.m_laneCount;
            return ManagedSubmitSender<T_Submits..., T_Right...>{
                std::tuple_cat(std::move(m_submits), std::move(right.m_submits)),
                lanes,
                laneCount,
                detail::composeDependencies<T_Ordered>(m_dependencies, right.m_dependencies)};
        }

        std::tuple<T_Submits...> m_submits;
        std::array<std::size_t, stageCount> m_lanes;
        std::size_t m_laneCount;
        detail::SubmissionDependencies<stageCount> m_dependencies;
    };

    /** Native lowering for queue-free alpaka graphs before device binding. */
    struct ManagedSubmissionDomain
    {
        template<typename... T_Left, typename... T_Right>
        auto transform(SequenceTag, ManagedSubmitSender<T_Left...> left, ManagedSubmitSender<T_Right...> right) const
        {
            return std::move(left).template compose<true>(std::move(right));
        }

        template<typename... T_Submits>
        auto transform(WhenAllTag, ManagedSubmitSender<T_Submits...> sender) const
        {
            return sender;
        }

        template<typename... T_Left, typename... T_Right, typename... T_Rest>
        auto transform(
            WhenAllTag tag,
            ManagedSubmitSender<T_Left...> left,
            ManagedSubmitSender<T_Right...> right,
            T_Rest... rest) const
        {
            return transform(tag, std::move(left).template compose<false>(std::move(right)), std::move(rest)...);
        }

        template<caravan::detail::GraphNodeType... T_Nodes>
        requires(detail::isManagedSubmitSender<typename T_Nodes::sender_type> && ...)
        auto transform(GraphTag, T_Nodes... nodes) const
        {
            // Each node currently receives a disjoint logical lane range. A future graph-coloring pass could
            // reuse lanes for ordered nodes whose lifetimes do not overlap, reducing the number of leased queues.
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

    template<typename... T_Submits>
    auto ManagedSubmitSender<T_Submits...>::query(GetDomain) const noexcept -> ManagedSubmissionDomain
    {
        return {};
    }

    template<typename... T_Left, typename... T_Right>
    auto sequence(ManagedSubmitSender<T_Left...> left, ManagedSubmitSender<T_Right...> right)
    {
        return std::move(left).template compose<true>(std::move(right));
    }

    template<typename... T_Submits>
    auto sequence(ManagedSubmitSender<T_Submits...> next)
    {
        return caravan::detail::SenderAdaptorClosure{
            [next = std::move(next)](auto previous) mutable
            { return caravan::alpaka::sequence(std::move(previous), std::move(next)); }};
    }

    namespace detail
    {
        template<typename T_Submit>
        auto managedSubmit(T_Submit submit)
        {
            using Submit = std::decay_t<T_Submit>;
            return ManagedSubmitSender<Submit>{{std::move(submit)}};
        }
    } // namespace detail

    /** Lazily describe one native submission whose queue comes from the receiver environment. */
    template<typename T_Submit>
    auto submit(T_Submit submit)
    {
        return detail::managedSubmit(std::move(submit));
    }
} // namespace caravan::alpaka
