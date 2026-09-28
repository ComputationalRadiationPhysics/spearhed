/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    template<std::size_t T_Size>
    struct FixedString
    {
        constexpr FixedString(char const (&value)[T_Size])
        {
            for(std::size_t i = 0u; i < T_Size; ++i)
                data[i] = value[i];
        }

        char data[T_Size];
    };

    template<FixedString T_Name>
    struct NodeIdentity
    {
        static constexpr auto name = T_Name;
    };

    template<typename... T_Identities>
    struct After
    {
    };

    template<FixedString T_Name, Sender T_Sender, typename T_After = After<>>
    class GraphNode
    {
    public:
        using identity = NodeIdentity<T_Name>;
        using sender_type = T_Sender;
        using predecessors = T_After;

        explicit GraphNode(T_Sender sender) : m_sender(std::move(sender))
        {
        }

        auto query(GetDomain query) const noexcept(noexcept(query(m_sender)))
        {
            return query(m_sender);
        }

        T_Sender releaseSender() &&
        {
            return std::move(m_sender);
        }

    private:
        T_Sender m_sender;
    };

    namespace detail
    {
        // Avoid partial specialization on FixedString: NVCC fails to match it.
        template<typename T>
        consteval bool checkGraphNode()
        {
            if constexpr(requires {
                             typename GraphNode<T::identity::name, typename T::sender_type, typename T::predecessors>;
                         })
                return std::
                    is_same_v<T, GraphNode<T::identity::name, typename T::sender_type, typename T::predecessors>>;
            else
                return false;
        }

        template<typename T>
        inline constexpr bool isGraphNode = checkGraphNode<T>();

        template<typename T>
        concept GraphNodeType = isGraphNode<std::remove_cvref_t<T>>;

        template<typename T_List, typename T_Identity>
        inline constexpr bool containsIdentity = false;

        template<typename... T_Identities, typename T_Identity>
        inline constexpr bool containsIdentity<After<T_Identities...>, T_Identity>
            = (std::is_same_v<T_Identities, T_Identity> || ...);

        template<typename T_Identity, typename T_Tuple, std::size_t... T_Index>
        consteval std::size_t identityCount(std::index_sequence<T_Index...>)
        {
            return (
                std::size_t{0u} + ...
                + std::size_t{std::is_same_v<T_Identity, typename std::tuple_element_t<T_Index, T_Tuple>::identity>});
        }

        template<typename T_Identity, typename T_Tuple, std::size_t... T_Index>
        consteval bool identityAppearsBefore(std::index_sequence<T_Index...>)
        {
            return (
                false || ... || std::is_same_v<T_Identity, typename std::tuple_element_t<T_Index, T_Tuple>::identity>);
        }

        template<typename T_List, typename T_Tuple, std::size_t T_Limit>
        inline constexpr bool predecessorsAppearBefore = false;

        template<typename... T_Identities, typename T_Tuple, std::size_t T_Limit>
        inline constexpr bool predecessorsAppearBefore<After<T_Identities...>, T_Tuple, T_Limit>
            = (identityAppearsBefore<T_Identities, T_Tuple>(std::make_index_sequence<T_Limit>{}) && ...);

        template<typename... T_Nodes>
        struct GraphTopology
        {
            using Nodes = std::tuple<T_Nodes...>;
            static constexpr auto size = sizeof...(T_Nodes);

            template<std::size_t T_Index>
            using Node = std::tuple_element_t<T_Index, Nodes>;

            template<std::size_t T_Index, std::size_t... T_Other>
            static consteval bool uniqueAt(std::index_sequence<T_Other...>)
            {
                return identityCount<typename Node<T_Index>::identity, Nodes>(std::index_sequence<T_Other...>{}) == 1u;
            }

            template<std::size_t... T_Index>
            static consteval bool unique(std::index_sequence<T_Index...>)
            {
                return (uniqueAt<T_Index>(std::make_index_sequence<size>{}) && ...);
            }

            template<std::size_t... T_Index>
            static consteval bool ordered(std::index_sequence<T_Index...>)
            {
                return (predecessorsAppearBefore<typename Node<T_Index>::predecessors, Nodes, T_Index> && ...);
            }

            static constexpr bool hasUniqueNames = unique(std::make_index_sequence<size>{});
            static constexpr bool isTopologicallyOrdered = ordered(std::make_index_sequence<size>{});

            template<std::size_t T_Row, std::size_t... T_Column>
            static consteval auto makeRow(std::index_sequence<T_Column...>)
            {
                return std::array<bool, size>{
                    containsIdentity<typename Node<T_Row>::predecessors, typename Node<T_Column>::identity>...};
            }

            template<std::size_t... T_Row>
            static consteval auto makePredecessors(std::index_sequence<T_Row...>)
            {
                return std::array<std::array<bool, size>, size>{makeRow<T_Row>(std::make_index_sequence<size>{})...};
            }

            static constexpr auto predecessors = makePredecessors(std::make_index_sequence<size>{});
        };

        template<typename T_Receiver>
        class GraphReceiverHolder
        {
        protected:
            explicit GraphReceiverHolder(T_Receiver receiver) : m_receiver(std::move(receiver))
            {
            }

            T_Receiver m_receiver;
        };

        template<std::size_t T_Index, typename T_Owner, typename T_Environment>
        struct GraphNodeReceiver
        {
            template<typename... T>
            void set_value(T&&...) noexcept
            {
                owner->template complete<T_Index>();
            }

            decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Environment const&>().get_env()))
                requires requires(T_Environment const& environment) { environment.get_env(); }
            {
                return environment->get_env();
            }

            T_Owner* owner;
            T_Environment const* environment;
        };

        template<std::size_t T_Index, typename T_Owner, typename T_Environment, typename T_Sender>
        class GraphOperationHolder
        {
            using Receiver = GraphNodeReceiver<T_Index, T_Owner, T_Environment>;

        public:
            GraphOperationHolder(T_Sender sender, T_Owner* owner, T_Environment const* environment)
                : m_operation(std::move(sender).connect(Receiver{owner, environment}))
            {
            }

            void start() noexcept
            {
                m_operation.start();
            }

        private:
            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };

        template<typename T_Receiver, typename T_Indices, typename... T_Nodes>
        class GraphOperation;

        template<typename T_Receiver, std::size_t... T_Index, typename... T_Nodes>
        class GraphOperation<T_Receiver, std::index_sequence<T_Index...>, T_Nodes...>
            : private GraphReceiverHolder<T_Receiver>
            , private GraphOperationHolder<
                  T_Index,
                  GraphOperation<T_Receiver, std::index_sequence<T_Index...>, T_Nodes...>,
                  T_Receiver,
                  typename T_Nodes::sender_type>...
        {
            using Self = GraphOperation<T_Receiver, std::index_sequence<T_Index...>, T_Nodes...>;
            using ReceiverHolder = GraphReceiverHolder<T_Receiver>;
            using Topology = GraphTopology<T_Nodes...>;
            static constexpr auto nodeCount = sizeof...(T_Nodes);

            template<std::size_t T_I>
            using Holder = GraphOperationHolder<
                T_I,
                Self,
                T_Receiver,
                typename std::tuple_element_t<T_I, std::tuple<T_Nodes...>>::sender_type>;

        public:
            GraphOperation(std::tuple<typename T_Nodes::sender_type...> senders, T_Receiver receiver)
                : ReceiverHolder(std::move(receiver))
                , Holder<T_Index>(std::move(std::get<T_Index>(senders)), this, &this->m_receiver)...
            {
                for(std::size_t node = 0u; node < nodeCount; ++node)
                    for(bool predecessor : Topology::predecessors[node])
                        m_remainingPredecessors[node] += predecessor;
            }

            GraphOperation(GraphOperation const&) = delete;
            GraphOperation& operator=(GraphOperation const&) = delete;
            GraphOperation(GraphOperation&&) = delete;
            GraphOperation& operator=(GraphOperation&&) = delete;

            void start() & noexcept
            {
                std::array<std::size_t, nodeCount> ready{};
                std::size_t count = 0u;
                for(std::size_t node = 0u; node < nodeCount; ++node)
                    if(m_remainingPredecessors[node] == 0u)
                        ready[count++] = node;
                startNodes(this, ready, count);
            }

            template<std::size_t T_I>
            void complete() noexcept
            {
                std::array<std::size_t, nodeCount> ready{};
                std::size_t readyCount = 0u;
                bool finished = false;
                {
                    std::lock_guard lock(m_mutex);
                    resolve(T_I);
                    for(std::size_t successor = T_I + 1u; successor < nodeCount; ++successor)
                        if(!m_resolved[successor] && Topology::predecessors[successor][T_I]
                           && --m_remainingPredecessors[successor] == 0u)
                            ready[readyCount++] = successor;
                    finished = m_unresolved == 0u;
                }

                if(finished)
                {
                    this->m_receiver.set_value();
                    return;
                }
                if(readyCount != 0u)
                    startNodes(this, ready, readyCount);
            }

        private:
            void resolve(std::size_t node) noexcept
            {
                m_resolved[node] = true;
                --m_unresolved;
            }

            template<std::size_t T_I>
            static void startNode(Self* owner) noexcept
            {
                static_cast<Holder<T_I>&>(*owner).start();
            }

            static constexpr std::array<void (*)(Self*) noexcept, nodeCount> startFunctions{&startNode<T_Index>...};

            static void startNodes(
                Self* owner,
                std::array<std::size_t, nodeCount> const& nodes,
                std::size_t count) noexcept
            {
                for(std::size_t i = 0u; i < count; ++i)
                    startFunctions[nodes[i]](owner);
            }

            std::mutex m_mutex;
            std::array<std::size_t, nodeCount> m_remainingPredecessors{};
            std::array<bool, nodeCount> m_resolved{};
            std::size_t m_unresolved = nodeCount;
        };
    } // namespace detail

    template<detail::GraphNodeType... T_Nodes>
    auto after(T_Nodes const&...)
    {
        return After<typename std::remove_cvref_t<T_Nodes>::identity...>{};
    }

    template<FixedString T_Name, Sender T_Sender>
    auto node(T_Sender sender)
    {
        return GraphNode<T_Name, T_Sender>{std::move(sender)};
    }

    template<FixedString T_Name, Sender T_Sender, typename... T_Identities>
    auto node(T_Sender sender, After<T_Identities...>)
    {
        return GraphNode<T_Name, T_Sender, After<T_Identities...>>{std::move(sender)};
    }

    template<detail::GraphNodeType... T_Nodes>
    class GraphSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit GraphSender(T_Nodes... nodes) : m_senders(std::move(nodes).releaseSender()...)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::GraphOperation<std::decay_t<T_Receiver>, std::index_sequence_for<T_Nodes...>, T_Nodes...>{
                std::move(m_senders),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        std::tuple<typename T_Nodes::sender_type...> m_senders;
    };

    /** Build one lazy, fixed-topology sender graph.
     *
     * @code
     * auto load = node<"load">(loadSender);
     * auto run = node<"run">(runSender, after(load));
     * auto work = graph(std::move(load), std::move(run));
     * @endcode
     *
     * Arguments must be in topological order. after() expresses ordering only: predecessor values are discarded.
     * Every node is owned and started at most once. Callback, submission, and asynchronous execution failures
     * are fatal.
     */
    template<detail::GraphNodeType... T_Nodes>
    auto graph(T_Nodes... nodes)
    {
        using Topology = detail::GraphTopology<T_Nodes...>;
        static_assert(sizeof...(T_Nodes) > 0u, "A Caravan graph must contain at least one node");
        static_assert(Topology::hasUniqueNames, "Caravan graph node names must be unique");
        static_assert(
            Topology::isTopologicallyOrdered,
            "Every Caravan graph predecessor must exist and precede its dependent node in graph(...) order");

        auto domain = detail::commonDomain(nodes...);
        if constexpr(requires { domain.transform(GraphTag{}, std::move(nodes)...); })
            return domain.transform(GraphTag{}, std::move(nodes)...);
        else
            return GraphSender<T_Nodes...>{std::move(nodes)...};
    }
} // namespace caravan
