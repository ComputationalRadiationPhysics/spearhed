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

#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <pmacc/eventSystem/events/EventTask.hpp>

namespace pmacc::spearhed
{
    // Policy tags for explicit dispatch control (optional override).
    struct PerSourcePolicy
    {
    };

    struct UnifiedPolicy
    {
    };

    inline constexpr PerSourcePolicy perSource{};
    inline constexpr UnifiedPolicy unified{};

    /**
     * @brief Perform pairwise particle interactions using candidate-provider entries.
     *
     * Dispatches automatically based on the number of source entries in the bundle:
     *   - 0 sources: no-op.
     *   - 1 source:  FrameInteractionKernel (per-source launch - lower register pressure).
     *   - >=2 sources: UnifiedFrameInteractionKernel (single launch - own-particle state
     *                  loaded/flushed once, amortising target-side work across all sources).
     *
     * Pass an explicit policy tag as the first argument to override the automatic choice:
     *   interact(perSource, bundle, target, index, radius, fn, args...);
     *   interact(unified,   bundle, target, index, radius, fn, args...);
     *
     * All launches are asynchronous: interact() returns as soon as the kernels are enqueued.
     *
     * Lifetime contract: @p bundle (including the source ParticleRegionBuffers its entries point to),
     * @p target and @p index own device memory the kernels read, so ALL of them must outlive kernel
     * COMPLETION, not merely this call. PMacc buffer destructors do not wait for in-flight kernels.
     * Wait on the returned event before any of them is destroyed. Ordering against later device work
     * needs no wait: subsequent kernels and copies are transaction-ordered behind these launches.
     *
     * @param bundle         NeighbourBundle compatibility plan or selected view.
     * @param target         The target ParticleRegionBuffer.
     * @param index          FrameIndexBuffer for @p target.
     * @param radius         Maximum pairwise interaction distance.
     * @param fn             Interaction functor (see FrameInteractionKernel contract).
     * @param args           Additional forwarded arguments to @p fn.
     * @return EventTask covering every kernel enqueued by this call (already finished when nothing
     *         was launched). Call waitForFinished() on it at the caller's natural sync point.
     */
    template<IsNeighbourBundle Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
    [[nodiscard]] pmacc::EventTask interact(
        Bundle&& bundle,
        Target& target,
        Index& index,
        Radius radius,
        Fn&& fn,
        Args&&... args)
    {
        if(target.size == 0)
            return {};

        constexpr std::size_t numSources = std::remove_cvref_t<Bundle>::size();

        if constexpr(numSources == 0)
        {
            return {};
        }
        else if constexpr(numSources == 1)
        {
            pmacc::EventTask event;
            bundle.forEachDeviceView(
                [&](auto sourceView)
                {
                    event = launchForEachFrameInBlockIndexed(
                        launchConfig<64>(oneBlockPerFrame),
                        target,
                        index,
                        detail::FrameInteractionKernel<pred::Occupied>{},
                        sourceView,
                        radius,
                        fn,
                        std::forward<decltype(args)>(args)...);
                });
            return event;
        }
        else
        {
            auto sourceViews = bundle.makeDeviceViewTuple();
            return launchForEachFrameInBlockIndexed(
                launchConfig<64>(oneBlockPerFrame),
                target,
                index,
                detail::UnifiedFrameInteractionKernel<pred::Occupied>{},
                std::move(sourceViews),
                radius,
                fn,
                std::forward<decltype(args)>(args)...);
        }
    }

    // Explicit-policy overloads

    /** @brief Force per-source kernel launches regardless of bundle arity.
     *         Same asynchrony and lifetime contract as the automatic overload. */
    template<IsNeighbourBundle Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
    [[nodiscard]] pmacc::EventTask interact(
        PerSourcePolicy,
        Bundle&& bundle,
        Target& target,
        Index& index,
        Radius radius,
        Fn&& fn,
        Args&&... args)
    {
        if(target.size == 0)
            return {};

        pmacc::EventTask combined;
        bundle.forEachDeviceView(
            [&](auto sourceView)
            {
                combined += launchForEachFrameInBlockIndexed(
                    launchConfig<64>(oneBlockPerFrame),
                    target,
                    index,
                    detail::FrameInteractionKernel<pred::Occupied>{},
                    sourceView,
                    radius,
                    fn,
                    std::forward<decltype(args)>(args)...);
            });
        return combined;
    }

    /** @brief Force a single unified kernel launch regardless of bundle arity.
     *         Same asynchrony and lifetime contract as the automatic overload. */
    template<IsNeighbourBundle Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
    [[nodiscard]] pmacc::EventTask interact(
        UnifiedPolicy,
        Bundle&& bundle,
        Target& target,
        Index& index,
        Radius radius,
        Fn&& fn,
        Args&&... args)
    {
        if(target.size == 0)
            return {};

        if constexpr(std::remove_cvref_t<Bundle>::size() == 0)
        {
            return {};
        }
        else
        {
            auto sourceViews = bundle.makeDeviceViewTuple();
            return launchForEachFrameInBlockIndexed(
                launchConfig<64>(oneBlockPerFrame),
                target,
                index,
                detail::UnifiedFrameInteractionKernel<pred::Occupied>{},
                std::move(sourceViews),
                radius,
                fn,
                std::forward<decltype(args)>(args)...);
        }
    }

} // namespace pmacc::spearhed
