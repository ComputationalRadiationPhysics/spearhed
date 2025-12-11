#pragma once

#include "spmacc/memory/utils.hpp"

#include <pmacc/memory/Align.hpp>
#include <pmacc/particles/Identifier.hpp>

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/mem/fence/Traits.hpp>

#include <concepts>
#include <new>
#include <type_traits>

namespace pmacc::spearhed
{
    namespace detail
    {
        template<typename T>
        struct Node
        {
            using NodePtr = Node<T>*;
            PMACC_ALIGN(data, T);
            PMACC_ALIGN(next, NodePtr);
        };
    } // namespace detail

    /**
     * A singly-linked list of frames on the acc
     * uses new and delete on CPU and mallocMC on GPU
     *
     * @tparam Type held in the list. Requires that the type includes the next pointer
     * @tparam T_DeviceHeapHandle device heap handle type
     */
    template<typename T, typename T_DeviceHeapHandle>
    requires requires(T t) { t.next; }
    struct SingleLinkedListDevice
    {
        using PtrType = T*;

        constexpr SingleLinkedListDevice(T_DeviceHeapHandle const& deviceHeapHandle)
            : m_deviceHeapHandle(deviceHeapHandle)
        {
        }

        /**
         * Returns a pointer to a free node from data heap.
         * If T is default initializable, the type is constructed after allocation, else it is not constructed
         *
         * @param worker
         */
        [[nodiscard]] constexpr PtrType getEmptyNode(auto const& worker)
        {
            PtrType tmp = memory::allocateMemory<T>(worker);

            PMACC_DEVICE_VERIFY_MSG(tmp != nullptr, "Error: Out of device heap memory in %s:%u\n", __FILE__, __LINE__);

            if constexpr(std::default_initializable<T>)
            {
                if(tmp)
                {
                    new(tmp) T{};
                }
            }
            // TODO check if this is necessary for iteration end or if it is already set
            tmp->next = nullptr;
            return tmp;
        }

        /**
         * Removes frame from heap data heap.
         * Takes ownership and sets the user provided ptr to nullptr
         *
         * @param worker
         * @param node pointer to node to remove
         */
        constexpr void removeNode(auto const& worker, PtrType& node)
        {
            if(!node)
                return;

            if constexpr(!std::is_trivially_destructible_v<T>)
            {
                node->data.~T();
            }

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
            m_deviceHeapHandle.free(worker.getAcc(), (void*) node);
#else
            operator delete(node, std::nothrow);
#endif
            node = nullptr;
        }

        /**
         * Thread-safe insertion of a node at the back of the list.
         *
         * @param worker
         * @param node pointer to node to insert
         */
        constexpr void pushBack(auto const& worker, PtrType node)
        {
            if(!node)
                return;

            node->next = nullptr;

            PtrType oldLast = alpaka::atomicExch(worker.getAcc(), &m_lastNode, node);
            if(oldLast != nullptr)
            {
                // List was non-empty, link old last to new node
                oldLast->next = node;
            }
            else
            {
                // List was empty, update first node
                m_firstNode = node;
            }
            // fence to publish changes to the list to everyone
            // TODO use a release fence
            alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});
        }

        constexpr auto begin() const
        {
            return m_firstNode;
        }

        constexpr auto end() const
        {
            return nullptr;
        }


    private:

    private:
        // TODO try to move this out. Not every list on my device needs to hold a copy of the heap handle
        PMACC_ALIGN(m_deviceHeapHandle, T_DeviceHeapHandle);
        PMACC_ALIGN(m_firstNode, PtrType) { nullptr };
        PMACC_ALIGN(m_lastNode, PtrType) { nullptr };
    };
} // namespace pmacc::spearhed
