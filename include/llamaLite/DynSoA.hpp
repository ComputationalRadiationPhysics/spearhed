// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "llamaLite/Field.hpp"
#include "llamaLite/Record.hpp"
#include "llamaLite/ResolveLeaf.hpp"
#include "llamaLite/Transform.hpp"
#include "llamaLite/Tuple.hpp"
#include "llamaLite/View.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <tuple>
#include <type_traits>
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
            using type = std::vector<T, util::UninitializedAlignedAllocator<T>>;
        };

        template<typename Record>
        using transform_record_dynsoa_t = transform_record_t<Record, PolicyDynSoA>;

    } // namespace transform

    namespace detail
    {
        /**
         * Recursively resize all leaf std::vector elements within a DynSoA storage tuple.
         *
         * The storage is either a Tuple (recurse into each element) or a
         * std::vector<T> (resize directly).
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

    } // namespace detail

    /**
     * Dynamic host-side Structure-of-Arrays container for a Record.
     *
     * Mirrors SoA<R, N> but uses std::vector<T> per leaf field instead of
     * fixed-size arrays, allowing runtime resizing. Intended for host-side
     * serialization buffers (e.g. device-to-host particle copies for I/O).
     *
     * The getLeaf<RA>() interface is identical to SoA - both return
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
        using view_type = View<DynSoA, Tags...>;

        template<IsRecordAccess... Tags>
        using indexed_view_type = ViewIndexed<DynSoA, Tags...>;

        DynSoA() = default;

        explicit DynSoA(size_t n)
        {
            resize(n);
        }

        /** Resize all leaf arrays to n elements. */
        void resize(size_t n)
        {
            detail::resizeDynStorage(channels_, n);
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
         * auto xSpan = dynSoa.getLeaf<TagPath<vel_t, x_t>>();
         * @endcode
         */
        template<IsRecordAccess RA>
        [[nodiscard]] auto getLeaf()
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto getLeaf() const
        {
            auto& leaf = resolveLeaf<RA, R>(channels_);
            using ElementType = std::remove_pointer_t<decltype(leaf.data())>;
            return std::span<ElementType>(leaf);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] auto view(RAs... tags)
        {
            return View<DynSoA, to_path_t<RAs>...>(*this, to_path_t<RAs>{}...);
        }

        template<IsRecordAccess... RAs>
        [[nodiscard]] auto view(RAs... tags) const
        {
            return View<DynSoA const, to_path_t<RAs>...>(*this, to_path_t<RAs>{}...);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA tag)
        {
            return view(tag);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA tag) const
        {
            return view(tag);
        }

        [[nodiscard]] auto operator[](uint32_t idx)
        {
            return ViewIndexed(*this, idx);
        }

        [[nodiscard]] auto operator[](uint32_t idx) const
        {
            return ViewIndexed(*this, idx);
        }

    private:
        transform::transform_record_dynsoa_t<R> channels_;
        size_t size_ = 0;
    };

} // namespace llama_lite
