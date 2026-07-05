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

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

/*
 * Pure multi-level iterator over the particle hierarchy:
 *
 *     MultiSpecies  >  Species  >  Region  >  Frame  >  Particle
 *
 * Iteration model: forEach visits elements *for the calling thread*. Every level -- including the
 * particle leaf -- is a plain sequential loop; there is no worker, no lockstep, and no implicit
 * work distribution anywhere in the traversal. Liveness is iteration semantics: dead slots are not
 * particles, so the leaf filters with pred::occupied.
 *
 * Consequence on device: calling forEach from all threads of a kernel visits every element once
 * PER THREAD. Distribute work first -- via a launch schedule (see LaunchForEach.hpp), the
 * lockstepForEachParticle slot combinator (see FrameSchedule.hpp), or a master-only section -- and
 * iterate inside whatever sub-tree a thread/block owns.
 *
 * The only parameter besides the view is *heap access*: how frame-list next-pointers are chased.
 *   - deviceHeap:            native device pointers (inside kernels).
 *   - hostHeap(heapOffset):  host-translated pointers via hostIterable (host-side walks; obtain the
 *                            offset from syncHeapToHost() after prBuf.synchronize()).
 * That is genuinely one concern -- pointer dereference -- not thread distribution.
 *
 * Composable nesting threads one access value through all levels:
 *
 *   // Host:
 *   auto const access = sp::hostHeap(heapOffset);
 *   sp::forEach(sp::levels::region, access, hostSpeciesView,
 *       [&](auto region) { sp::forEach(sp::levels::particle, access, region, [&](auto p) {...}); });
 *
 *   // Device, one block owning a region; slots distributed explicitly:
 *   sp::forEach(sp::levels::frame, sp::deviceHeap, sp::makeRegionView(myRegion),
 *       [&](auto frame) { sp::lockstepForEachParticle(worker, frame, [&](auto p) {...}); });
 *
 * CUDA qualifier rules (invariants; the CPU-serial backend cannot exercise them):
 *   - All shared traversal code is HDINLINE, never DINLINE (DINLINE is device-only under
 *     ALPAKA_ACC_GPU_CUDA_ONLY_MODE).
 *   - Host-only code lives exclusively inside HostHeapAccess members (unqualified => __host__) and
 *     is reached only via dependent access.frames(...) calls, so nvcc defers the address-space
 *     check to instantiation. Never branch on if constexpr(isDevice) around space-specific bodies.
 *   - Escape hatch if an nvcc version still warns: PMACC_NO_NVCC_HDWARNING on the Region->frames
 *     overload.
 */

namespace pmacc::spearhed
{
    // Hierarchy levels as compile-time value objects

    //! Hierarchy levels, ranked finest (Particle = 0) to coarsest. The rank orders containment and
    //! drives the generic descent in forEach. Passed as constexpr value objects, not name suffixes.
    namespace levels
    {
        template<int T_rank>
        struct Level
        {
            static constexpr int rank = T_rank;
        };

        using Particle = Level<0>;
        using Frame = Level<1>;
        using Region = Level<2>;
        using Species = Level<3>;

        inline constexpr Particle particle{};
        inline constexpr Frame frame{};
        inline constexpr Region region{};
        inline constexpr Species species{};
    } // namespace levels

    //! A hierarchy level (levels::particle / frame / region / species) vs. a policy/config value.
    //! Used to disambiguate level-first and cfg-first overload sets by their first argument.
    template<typename T>
    concept IsHierarchyLevel = requires { T::rank; };

    // Heap access (the single point of host/device variation: pointer translation only)

    //! Chase frame-list pointers natively (device pointers valid in the calling address space).
    struct DeviceHeapAccess
    {
        static constexpr bool isDevice = true;

        HDINLINE auto& frames(auto& frameList) const
        {
            return frameList; // native begin()/end()
        }
    };

    //! Chase frame-list pointers from the host by translating device addresses with the mallocMC
    //! heap offset (see syncHeapToHost()). Host-only: frames() is deliberately unqualified.
    struct HostHeapAccess
    {
        static constexpr bool isDevice = false;

        int64_t heapOffset;

        auto frames(auto& frameList) const
        {
            return frameList.hostIterable(heapOffset); // host-only
        }
    };

    template<typename T>
    concept HeapAccess = requires {
        { std::remove_cvref_t<T>::isDevice } -> std::convertible_to<bool>;
    };

    inline constexpr DeviceHeapAccess deviceHeap{};

    inline auto hostHeap(int64_t heapOffset)
    {
        return HostHeapAccess{heapOffset};
    }

    // Views (cheap handles over the existing storage)

    //! A single frame: indexing yields the particle proxy for a slot.
    template<typename T_FramePtr>
    struct FrameView
    {
        using FrameType = typename T_FramePtr::type;
        static constexpr uint32_t frameSize = FrameType::frameSize;

        T_FramePtr frame;

        HDINLINE decltype(auto) operator[](uint32_t slot) const
        {
            return frame[slot];
        }
    };

    //! A single ParticleRegion: its frame list and volume.
    template<typename T_Region>
    struct RegionView
    {
        T_Region& region;

        HDINLINE auto& frameList() const
        {
            return region.particleFrameList;
        }

        HDINLINE auto& volume() const
        {
            return region.volume;
        }
    };

    HDINLINE auto makeRegionView(auto& region)
    {
        return RegionView<std::remove_reference_t<decltype(region)>>{region};
    }

    //! One species: a data box of regions (device or host flavour) plus the region count.
    template<typename T_RegionBox>
    struct SpeciesView
    {
        T_RegionBox regionBox;
        int numRegions;

        HDINLINE auto region(int idx)
        {
            return makeRegionView(regionBox[idx]);
        }
    };

    //! A compile-time bundle of species (heterogeneous: each may have a different record type).
    //! Backed by the device-capable pmacc tuple, so the view is copyable by value into kernels
    //! like the others.
    template<typename... T_Species>
    struct MultiSpeciesView
    {
        pmacc::memory::tuple::Tuple<T_Species...> species;
    };

    // View constructors from a ParticleRegionBuffer
    //! Device-side species view (regions from the device data box).
    constexpr auto deviceSpecies(auto& prBuf)
    {
        return SpeciesView{prBuf.getDeviceDataBox(), prBuf.size};
    }

    //! Host-side species view (regions from the host data box; requires prBuf.synchronize() first).
    constexpr auto hostSpecies(auto& prBuf)
    {
        return SpeciesView{prBuf.buffer->getHostBuffer().getDataBox(), prBuf.size};
    }

    template<typename... T_PRBuf>
    constexpr auto deviceMultiSpecies(T_PRBuf&... prBufs)
    {
        return MultiSpeciesView<decltype(deviceSpecies(prBufs))...>{
            pmacc::memory::tuple::make_tuple(deviceSpecies(prBufs)...)};
    }

    template<typename... T_PRBuf>
    constexpr auto hostMultiSpecies(T_PRBuf&... prBufs)
    {
        return MultiSpeciesView<decltype(hostSpecies(prBufs))...>{
            pmacc::memory::tuple::make_tuple(hostSpecies(prBufs)...)};
    }

    // Per-view child iteration + level query
    // One primitive per view: iterate the immediate children (one level down) sequentially for the
    // calling thread. The heap access parameter decides how frame-list pointers are dereferenced;
    // everything else is identical on host and device. childLevel(view) reports the child level.

    // Frame -> particles (leaf; live slots only)
    template<typename T_FramePtr>
    HDINLINE void forEachChild(HeapAccess auto const& /*access*/, FrameView<T_FramePtr> frame, auto body)
    {
        for(uint32_t slot = 0; slot < FrameView<T_FramePtr>::frameSize; ++slot)
        {
            auto particle = frame[slot];
            if(pred::occupied(particle))
                body(particle);
        }
    }

    template<typename T_FramePtr>
    HDINLINE constexpr levels::Particle childLevel(FrameView<T_FramePtr>)
    {
        return {};
    }

    // Region -> frames (pointer chasing via the heap access)
    template<typename T_Region>
    HDINLINE void forEachChild(HeapAccess auto const& access, RegionView<T_Region> region, auto body)
    {
        for(auto& frame : access.frames(region.frameList()))
            body(FrameView{memory::FramePointer{&frame}});
    }

    template<typename T_Region>
    HDINLINE constexpr levels::Frame childLevel(RegionView<T_Region>)
    {
        return {};
    }

    // Species -> regions (plain index loop)
    template<typename T_RegionBox>
    HDINLINE void forEachChild(HeapAccess auto const& /*access*/, SpeciesView<T_RegionBox> species, auto body)
    {
        for(int r = 0; r < species.numRegions; ++r)
            body(species.region(r));
    }

    template<typename T_RegionBox>
    HDINLINE constexpr levels::Region childLevel(SpeciesView<T_RegionBox>)
    {
        return {};
    }

    // MultiSpecies -> species (compile-time fold over the pmacc tuple)
    template<typename... T_Species>
    HDINLINE void forEachChild(HeapAccess auto const& /*access*/, MultiSpeciesView<T_Species...> multi, auto body)
    {
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        { (body(pmacc::memory::tuple::get<Is>(multi.species)), ...); }(std::index_sequence_for<T_Species...>{});
    }

    template<typename... T_Species>
    HDINLINE constexpr levels::Species childLevel(MultiSpeciesView<T_Species...>)
    {
        return {};
    }

    // Generic level-driven forEach

    /**
     * @brief Iterate, for the calling thread, every element at hierarchy level @p target contained
     *        in @p view. Every level is a sequential loop; nothing is distributed.
     *
     * Examples:
     *   forEach(levels::particle, deviceHeap, regionView, body);          // particles in a region
     *   forEach(levels::particle, hostHeap(heapOffset), speciesView, body); // host-side sweep
     *   forEach(levels::frame,    deviceHeap, speciesView, body);         // frames in a species
     *
     * @param target A levels::* value; must be at or below the view's child level.
     * @param access Heap access policy: deviceHeap inside kernels, hostHeap(heapOffset) on the host
     *               (after prBuf.synchronize() and syncHeapToHost()).
     * @param view   The container view to traverse.
     * @param body   Invoked as body(element) for each element at @p target.
     */
    template<typename T_Target>
    HDINLINE void forEach(T_Target target, HeapAccess auto const& access, auto view, auto body)
    {
        using ChildLevel = decltype(childLevel(view));
        static_assert(T_Target::rank <= ChildLevel::rank, "forEach target level is not contained in the view");
        if constexpr(ChildLevel::rank == T_Target::rank)
            forEachChild(access, view, body);
        else
            forEachChild(access, view, [&](auto child) { forEach(target, access, child, body); });
    }

} // namespace pmacc::spearhed
