/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/dimensions/Definition.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>

#include <tuple>
#include <type_traits>

#include <llamaLite/utility.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        // Named callable used in the IsNeighbourBundle concept instead of a generic lambda.
        // NVCC has a bug where anonymous generic lambdas in requires-expressions fail concept
        // checking for multi-entry bundles due to void fold expression expansion.
        struct AnyDevViewConsumer
        {
            template<typename T>
            void operator()(T&&) const
            {
            }
        };
    } // namespace detail

    /**
     * @brief Compact device-side view of one source's neighbour data.
     *
     * Passed directly as a single kernel argument, replacing the three separate
     * device boxes (sourcePRDeviceBox, neighbourRegionsBox, regionOffsetsBox).
     */
    template<typename T_SourceDeviceBox, typename T_IdxBox>
    struct SourceView
    {
        T_SourceDeviceBox sourcePRDeviceBox;
        T_IdxBox neighbourRegionsBox;
        T_IdxBox regionOffsetsBox;
        int numSourceRegions;
    };

    /**
     * @brief Host-side record for one source buffer and its pre-computed neighbour lists.
     *
     * A *source* buffer contributes particle data to a *target* region. For a given target
     * region index i, regionOffsets[i]..regionOffsets[i+1] indexes into neighbourRegions,
     * which holds the source-region indices that neighbour target region i.
     *
     * Owns the two neighbour-index buffers; holds a non-owning pointer to the source
     * ParticleRegionBuffer. Role is inherited from the source buffer's Role typedef.
     *
     * @tparam T_SourcePRBuf ParticleRegionBuffer<PRType, T_Role> type.
     */
    template<typename T_SourcePRBuf>
    struct NeighbourEntry
    {
        using Role = typename T_SourcePRBuf::Role;

        T_SourcePRBuf* sourceBufPtr;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> neighbourRegions;
        pmacc::HostDeviceBuffer<unsigned int, DIM1> regionOffsets;

        auto deviceView()
        {
            using SrcBox = decltype(sourceBufPtr->getDeviceDataBox());
            using IdxBox = decltype(neighbourRegions.getDeviceBuffer().getDataBox());
            return SourceView<SrcBox, IdxBox>{
                sourceBufPtr->getDeviceDataBox(),
                neighbourRegions.getDeviceBuffer().getDataBox(),
                regionOffsets.getDeviceBuffer().getDataBox(),
                sourceBufPtr->size};
        }
    };

    /**
     * @brief Non-owning view over a subset of entries from an owning NeighbourBundle.
     *
     * Returned by NeighbourBundle::subset<Roles...>(). Satisfies IsNeighbourBundle.
     *
     * @tparam T_TargetPRBuf Target buffer type.
     * @tparam T_Entries     The selected NeighbourEntry types (raw, not pointers).
     */
    template<typename T_TargetPRBuf, typename... T_Entries>
    struct NeighbourBundleView
    {
        T_TargetPRBuf* targetPtr;
        std::tuple<T_Entries*...> entryPtrs;

        auto& target()
        {
            return *targetPtr;
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn)
        {
            std::apply([&](auto*... ptrs) { (fn(ptrs->deviceView()), ...); }, entryPtrs);
        }

        /**
         * @brief Return a std::tuple of device views, one per entry.
         *
         * Used by InteractParticlesUnified to pass compile-time-iterable
         * source views into the single-launch kernel.
         */
        auto makeDeviceViewTuple()
        {
            return std::apply(
                [](auto*... ptrs) { return pmacc::memory::tuple::make_tuple(ptrs->deviceView()...); },
                entryPtrs);
        }

        template<typename T_Role>
        auto& get()
        {
            constexpr std::size_t idx = llama_lite::indexOfType<T_Role, typename T_Entries::Role...>();
            static_assert(idx < sizeof...(T_Entries), "Role not present in NeighbourBundleView");
            return *std::get<idx>(entryPtrs);
        }
    };

    /**
     * @brief Owning bundle: target buffer reference + one NeighbourEntry per source buffer.
     *
     * Represents the neighbourhood of a single target region: for each source buffer,
     * the pre-computed set of source regions that neighbour that target region is stored
     * in its NeighbourEntry (keyed by target-region index via regionOffsets).
     *
     * Produced by CalculateNeighbourRegions. Algorithms call forEachDeviceView to iterate
     * over all source contributions, or subset<Roles...>() to select a narrower set.
     *
     * @tparam T_TargetPRBuf Target buffer type.
     * @tparam T_Entries     NeighbourEntry<SrcBuf> types, one per source buffer.
     */
    template<typename T_TargetPRBuf, typename... T_Entries>
    struct NeighbourBundle
    {
        T_TargetPRBuf* targetPtr;
        std::tuple<T_Entries...> entries;

        auto& target()
        {
            return *targetPtr;
        }

        /**
         * @brief Returns the NeighbourEntry for the given Role tag.
         * Compile error if the role is not present in this bundle.
         */
        template<typename T_Role>
        auto& get()
        {
            constexpr std::size_t idx = llama_lite::indexOfType<T_Role, typename T_Entries::Role...>();
            static_assert(idx < sizeof...(T_Entries), "Role not present in NeighbourBundle");
            return std::get<idx>(entries);
        }

        /**
         * @brief Returns a non-owning NeighbourBundleView over the requested roles.
         *
         * Zero-cost: no buffer copies. Compile error if any requested role is absent.
         * Usage: bundle.subset<roles::Interior>() or bundle.subset<roles::Interior, roles::Boundary>()
         */
        template<typename... T_Roles>
        auto subset()
        {
            return NeighbourBundleView<T_TargetPRBuf, std::remove_reference_t<decltype(get<T_Roles>())>...>{
                targetPtr,
                std::tuple<std::remove_reference_t<decltype(get<T_Roles>())>*...>{&get<T_Roles>()...}};
        }

        /**
         * @brief Calls fn(sourceView) for every entry in this bundle.
         * Use subset<Roles...>() first to restrict which entries are iterated.
         */
        template<typename Fn>
        void forEachDeviceView(Fn&& fn)
        {
            std::apply([&](auto&... entry) { (fn(entry.deviceView()), ...); }, entries);
        }

        /**
         * @brief Return a std::tuple of device views, one per entry.
         *
         * Used by InteractParticlesUnified to pass compile-time-iterable
         * source views into the single-launch kernel.
         */
        auto makeDeviceViewTuple()
        {
            return std::apply(
                [](auto&... entry) { return pmacc::memory::tuple::make_tuple(entry.deviceView()...); },
                entries);
        }
    };

    /**
     * @brief Concept satisfied by both NeighbourBundle and NeighbourBundleView.
     * Algorithms accept any IsNeighbourBundle. They don't care if it owns or references entries.
     */
    template<typename T>
    concept IsNeighbourBundle = requires(T b) {
        b.target();
        b.forEachDeviceView(detail::AnyDevViewConsumer{});
    };

} // namespace pmacc::spearhed
