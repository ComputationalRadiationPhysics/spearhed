/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or the GNU
 * Lesser General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License and
 * the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License and the
 * GNU Lesser General Public License along with PMacc. If not, see
 * <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spmacc/particles/algorithms/ParticleParticleInteraction.hpp"
#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

#include <caravan/alpaka.hpp>

namespace pmacc::spearhed
{
    struct PerSourcePolicy
    {
    };

    struct UnifiedPolicy
    {
    };

    inline constexpr PerSourcePolicy perSource{};
    inline constexpr UnifiedPolicy unified{};

    namespace detail
    {
        template<
            std::size_t I,
            typename Views,
            typename Target,
            typename Index,
            typename Radius,
            typename Fn,
            typename... Args>
        [[nodiscard]] auto perSourceSender(
            Views views,
            Target& target,
            Index& index,
            Radius radius,
            Fn fn,
            Args... args)
        {
            constexpr std::size_t count = pmacc::memory::tuple::tuple_size_v<Views>;
            auto current = launchForEachFrameInBlockIndexed(
                launchConfig<64>(oneBlockPerFrame),
                target,
                index,
                FrameInteractionKernel<pred::Occupied>{},
                pmacc::memory::tuple::get<I>(views),
                radius,
                fn,
                args...);
            if constexpr(I + 1u == count)
                return current;
            else
                return std::move(current)
                       | caravan::alpaka::sequence(perSourceSender<I + 1u>(views, target, index, radius, fn, args...));
        }

        template<typename Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
        [[nodiscard]] auto perSource(
            Bundle&& bundle,
            Target& target,
            Index& index,
            Radius radius,
            Fn&& fn,
            Args&&... args)
        {
            constexpr std::size_t count = std::remove_cvref_t<Bundle>::size();
            if constexpr(count == 0u)
                return caravan::alpaka::submit([](auto&) {});
            else
            {
                auto views = bundle.makeDeviceViewTuple();
                return perSourceSender<0>(
                    std::move(views),
                    target,
                    index,
                    radius,
                    std::forward<Fn>(fn),
                    std::forward<Args>(args)...);
            }
        }

        template<typename Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
        [[nodiscard]] auto unifiedSource(
            Bundle&& bundle,
            Target& target,
            Index& index,
            Radius radius,
            Fn&& fn,
            Args&&... args)
        {
            constexpr std::size_t count = std::remove_cvref_t<Bundle>::size();
            if constexpr(count == 0u)
                return caravan::alpaka::submit([](auto&) {});
            else
            {
                auto views = bundle.makeDeviceViewTuple();
                return launchForEachFrameInBlockIndexed(
                    launchConfig<64>(oneBlockPerFrame),
                    target,
                    index,
                    detail::UnifiedFrameInteractionKernel<pred::Occupied>{},
                    std::move(views),
                    radius,
                    std::forward<Fn>(fn),
                    std::forward<Args>(args)...);
            }
        }
    } // namespace detail

    /**
     * @brief Lazily describe pairwise interactions with pre-computed neighbour lists.
     *
     * Zero sources produce a no-op sender. One source uses a lower-register per-source kernel;
     * multiple sources use a unified kernel that loads and stores target state once. The returned
     * sender borrows the bundle's source buffers, target, and index until completion. Compose it
     * with dependent work using Caravan sequence, or execute it with syncWait and withDevice.
     */
    template<IsNeighbourBundle Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
    [[nodiscard]] auto interact(Bundle&& bundle, Target& target, Index& index, Radius radius, Fn&& fn, Args&&... args)
    {
        if constexpr(std::remove_cvref_t<Bundle>::size() == 1u)
            return detail::perSource(
                std::forward<Bundle>(bundle),
                target,
                index,
                radius,
                std::forward<Fn>(fn),
                std::forward<Args>(args)...);
        else
            return detail::unifiedSource(
                std::forward<Bundle>(bundle),
                target,
                index,
                radius,
                std::forward<Fn>(fn),
                std::forward<Args>(args)...);
    }

    /** Force per-source launches, retaining source order on the native queue. */
    template<IsNeighbourBundle Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
    [[nodiscard]] auto interact(
        PerSourcePolicy,
        Bundle&& bundle,
        Target& target,
        Index& index,
        Radius radius,
        Fn&& fn,
        Args&&... args)
    {
        return detail::perSource(
            std::forward<Bundle>(bundle),
            target,
            index,
            radius,
            std::forward<Fn>(fn),
            std::forward<Args>(args)...);
    }

    /** Force one unified kernel launch. */
    template<IsNeighbourBundle Bundle, typename Target, typename Index, typename Radius, typename Fn, typename... Args>
    [[nodiscard]] auto interact(
        UnifiedPolicy,
        Bundle&& bundle,
        Target& target,
        Index& index,
        Radius radius,
        Fn&& fn,
        Args&&... args)
    {
        return detail::unifiedSource(
            std::forward<Bundle>(bundle),
            target,
            index,
            radius,
            std::forward<Fn>(fn),
            std::forward<Args>(args)...);
    }
} // namespace pmacc::spearhed
