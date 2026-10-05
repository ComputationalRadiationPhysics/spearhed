// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/ResolveLeaf.hpp"
#include "llamaLite/Set.hpp"
#include "llamaLite/Transform.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/View.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace llama_lite
{
    namespace util
    {
        /**
         * An allocator that aligns memory to a specified boundary (default 64 bytes) and bypasses zero-initialization
         * for trivial types.
         */
        template<typename T, std::size_t Alignment = 64>
        struct UninitializedAlignedAllocator
        {
            using value_type = T;
            using is_always_equal = std::true_type;

            static constexpr std::size_t actual_alignment = Alignment > alignof(T) ? Alignment : alignof(T);

            template<typename U>
            struct rebind
            {
                using other = UninitializedAlignedAllocator<U, Alignment>;
            };

            UninitializedAlignedAllocator() noexcept = default;

            [[nodiscard]] T* allocate(std::size_t n)
            {
                if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
                {
                    throw std::bad_array_new_length();
                }
                return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{actual_alignment}));
            }

            void deallocate(T* p, std::size_t n) noexcept
            {
                ::operator delete(p, n * sizeof(T), std::align_val_t{actual_alignment});
            }

            template<typename U, typename... Args>
            void construct(U* p, Args&&... args)
            {
                // If calling the default constructor of a trivial type (like int, float), do nothing.
                // This bypasses the O(N) zeroing in std::vector::resize.
                if constexpr(sizeof...(args) == 0 && std::is_trivially_default_constructible_v<U>)
                {
                    // Intentionally empty to keep memory uninitialized
                }
                else
                {
                    // Fallback for non-trivial types or specific values
                    std::construct_at(p, std::forward<Args>(args)...);
                }
            }

            constexpr bool operator==(UninitializedAlignedAllocator const&) const noexcept = default;
        };
    } // namespace util

    namespace transform
    {
        template<typename T>
        struct PolicyDynSoA
        {
            static_assert(
                !std::is_same_v<std::remove_cv_t<T>, bool>,
                "DynSoA does not support bool fields; use uint8_t for flags");
            static_assert(
                std::is_nothrow_destructible_v<T>
                    && (std::is_copy_constructible_v<T> || std::is_nothrow_move_constructible_v<T>),
                "DynSoA fields must be nothrow destructible and copyable or nothrow movable");
            using type = std::vector<T, util::UninitializedAlignedAllocator<T>>;
        };

        template<typename Record>
        using transform_record_dynsoa_t = transform_record_t<Record, PolicyDynSoA>;

    } // namespace transform

    namespace detail
    {
        /**
         * Recursively resize all leaf columns within a DynSoA storage tuple.
         *
         * The storage is either a Tuple (recurse into each element) or a
         * resizable contiguous leaf column.
         */
        template<typename Storage>
        void resizeDynStorage(Storage& storage, size_t n)
        {
            if constexpr(isSpecializationOf_v<Storage, Tuple>)
            {
                [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    (resizeDynStorage(tuple::get<Is>(storage), n), ...);
                }(std::make_index_sequence<std::tuple_size_v<Storage>>{});
            }
            else
            {
                storage.resize(n);
            }
        }

        template<typename Storage>
        void truncateDynStorage(Storage& storage, size_t n) noexcept
        {
            if constexpr(isSpecializationOf_v<Storage, Tuple>)
            {
                [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    (truncateDynStorage(tuple::get<Is>(storage), n), ...);
                }(std::make_index_sequence<std::tuple_size_v<Storage>>{});
            }
            else
            {
                while(storage.size() > n)
                    storage.pop_back();
            }
        }

        template<typename Storage>
        void swapDynStorage(Storage& lhs, Storage& rhs) noexcept
        {
            if constexpr(isSpecializationOf_v<Storage, Tuple>)
            {
                [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    (swapDynStorage(tuple::get<Is>(lhs), tuple::get<Is>(rhs)), ...);
                }(std::make_index_sequence<std::tuple_size_v<Storage>>{});
            }
            else
            {
                using std::swap;
                swap(lhs, rhs);
            }
        }

    } // namespace detail

    /**
     * Dynamic host-side Structure-of-Arrays container for a Record.
     *
     * Mirrors SoA<R, N> but uses dynamically sized contiguous storage per leaf
     * field, allowing runtime resizing. Boolean leaves are unsupported because
     * std::vector<bool> cannot expose contiguous bool storage; use uint8_t for
     * flags. Row counts are limited to UINT32_MAX because indexed views use
     * 32-bit row indices. Intended for host-side serialization buffers (e.g.
     * device-to-host particle copies for I/O).
     *
     * The getLeaf(RA{}) interface is identical to SoA - both return
     * std::span<T>, so code that reads from either container is the same.
     *
     * @tparam R  Record type describing the field hierarchy.
     */
    template<IsRecord R>
    class DynSoA
    {
    public:
        using record_type = R;

        template<IsRecordAccess... Tags>
        using view_type = View<DynSoA, access_set_t<Tags...>>;

        template<IsRecordAccess... Tags>
        using indexed_view_type = ViewIndexed<DynSoA, access_set_t<Tags...>>;

        DynSoA() = default;
        DynSoA(DynSoA const&) = default;

        DynSoA& operator=(DynSoA const& other)
        {
            if(this != &other)
            {
                DynSoA copy(other);
                detail::swapDynStorage(channels_, copy.channels_);
                std::swap(size_, copy.size_);
            }
            return *this;
        }

        DynSoA(DynSoA&& other) noexcept(std::is_nothrow_move_constructible_v<decltype(channels_)>)
            : channels_(std::move(other.channels_))
            , size_(std::exchange(other.size_, 0))
        {
        }

        DynSoA& operator=(DynSoA&& other) noexcept(std::is_nothrow_move_assignable_v<decltype(channels_)>)
        {
            if(this != &other)
            {
                channels_ = std::move(other.channels_);
                size_ = std::exchange(other.size_, 0);
            }
            return *this;
        }

        explicit DynSoA(size_t n)
        {
            resize(n);
        }

        /**
         * Resize all leaf arrays to n elements. Failed growth preserves sizes and existing values,
         * though successful earlier reallocations may invalidate previously returned spans.
         */
        void resize(size_t n)
        {
            if(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
                throw std::length_error("DynSoA size exceeds the 32-bit row-index limit");

            auto const oldSize = size_;
            try
            {
                detail::resizeDynStorage(channels_, n);
            }
            catch(...)
            {
                detail::truncateDynStorage(channels_, oldSize);
                throw;
            }
            size_ = n;
        }

        [[nodiscard]] size_t size() const
        {
            return size_;
        }

        /**
         * Return a std::span over the contiguous data for the leaf field
         * identified by the TagPath RA.
         *
         * Example:
         * @code
         * auto xSpan = dynSoa.getLeaf(vel / x);
         * @endcode
         */
        template<IsRecordAccess RA>
        [[nodiscard]] auto getLeaf(RA /*tag*/)
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf.data(), leaf.size());
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto getLeaf(RA /*tag*/) const
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf.data(), leaf.size());
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] auto view(RAs... /*tags*/)
        {
            return View<DynSoA, access_set_t<RAs...>>(*this);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] auto view(RAs... /*tags*/) const
        {
            return View<DynSoA const, access_set_t<RAs...>>(*this);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA tag)
        {
            return detail::resolveIfLeaf(view(tag));
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA tag) const
        {
            return detail::resolveIfLeaf(view(tag));
        }

        [[nodiscard]] auto operator[](uint32_t idx)
        {
            return ViewIndexed<DynSoA, Set<>>(*this, idx);
        }

        [[nodiscard]] auto operator[](uint32_t idx) const
        {
            return ViewIndexed<DynSoA const, Set<>>(*this, idx);
        }

    private:
        transform::transform_record_dynsoa_t<R> channels_;
        size_t size_ = 0;
    };

} // namespace llama_lite
