/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/plugins/openPMD/Position.hpp"
#include "spearhed/sph/KernelVariant.hpp"
#include "spmacc/particles/initialization/SC.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

#include <llamaLite/Record.hpp>

namespace spearhed
{
    namespace sod
    {
        // AABB helper

        namespace detail
        {
            /// Build a Vec from a per-axis array via index_sequence.
            template<pmacc::spearhed::CoordinateSystem T_CS, std::size_t N, std::size_t... Is>
            constexpr auto makeVec(std::array<typename T_CS::T_Axis, N> const& arr, std::index_sequence<Is...>)
            {
                return pmacc::spearhed::Vec<T_CS, pmacc::spearhed::ValueStorage<T_CS>>{arr[Is]...};
            }

            /// Build an AABB from full per-axis lo/hi arrays.
            template<pmacc::spearhed::CoordinateSystem T_CS, std::size_t N>
            constexpr auto aabbFromArrays(
                std::array<typename T_CS::T_Axis, N> const& lo,
                std::array<typename T_CS::T_Axis, N> const& hi)
            {
                static_assert(N == T_CS::dimension, "array size must match CS dimension");
                constexpr auto Dim = T_CS::dimension;
                return pmacc::spearhed::AABB<T_CS>{
                    pmacc::spearhed::Point<T_CS, pmacc::spearhed::ValueStorage<T_CS>>{},
                    makeVec<T_CS>(lo, std::make_index_sequence<Dim>{}),
                    makeVec<T_CS>(hi, std::make_index_sequence<Dim>{}),
                };
            }
        } // namespace detail

        //
        // Per-dimension constants (compile-time, no dimension-dependent types)
        //

        constexpr Real wallThickness()
        {
            return Real{2} * h0;
        }

        constexpr Real spacingLeft()
        {
#if SIM_DIM == 1
            return Real{1.125} / Real{32000};
#elif SIM_DIM == 2
            return Real{0.005};
#else
            return Real{1} / Real{30};
#endif
        }

        constexpr Real spacingRatio()
        {
#if SIM_DIM == 1
            return Real{8};
#elif SIM_DIM == 2
            return Real{2.8284271f};
#else
            return Real{2};
#endif
        }

        constexpr Real spacingRight()
        {
            return spacingLeft() * spacingRatio();
        }

        constexpr Real transverseExtent()
        {
#if SIM_DIM == 1
            return Real{0};
#elif SIM_DIM == 2
            return Real{0.07};
#else
            return Real{1} / Real{3};
#endif
        }

        struct InitialConditions
        {
            Real densityLeft = Real{1.0};
            Real pressureLeft = Real{1.0};
            Real densityRight = Real{0.125};
            Real pressureRight = Real{0.1};
        };

        constexpr uint32_t defaultTotalParticles = 32000u;

        // Shared functors

        struct SpacingNumParticles
        {
            DINLINE constexpr auto operator()(
                auto const& worker,
                auto const& particleRegion,
                Real dxLeft,
                Real dxRight) const
            {
                using x_t = std::tuple_element_t<0, typename CS::tags>;
                Real const s = (particleRegion.volume.max[x_t{}] <= Real{0}) ? dxLeft : dxRight;
                auto const counts = pmacc::spearhed::computeSCCellCounts(particleRegion.volume, s);
                return pmacc::spearhed::numSCLatticeSites(counts);
            }
        };

        struct SpacingPlaceParticle
        {
            DINLINE constexpr void operator()(
                auto const& worker,
                auto& particle,
                auto const& particleRegion,
                uint32_t globalParticleIdx,
                Real dxLeft,
                Real dxRight,
                Real pmass,
                Real densityLeft,
                Real densityRight,
                Real pressureLeft,
                Real pressureRight) const
            {
                using x_t = std::tuple_element_t<0, typename CS::tags>;
                auto const& aabb = particleRegion.volume;
                bool const isLeft = (aabb.max[x_t{}] <= Real{0});

                Real const rho = isLeft ? densityLeft : densityRight;
                Real const P = isLeft ? pressureLeft : pressureRight;
                Real const s = isLeft ? dxLeft : dxRight;

                auto const counts = pmacc::spearhed::computeSCCellCounts(aabb, s);
                pmacc::spearhed::SC<CS>{}(worker, particle, particleRegion, globalParticleIdx, counts);

                pmacc::spearhed::for_each_tag<CS>([&](auto axisTag) { particle[vel][axisTag] = Real{0}; });

                particle[mass] = pmass;
                particle[smoothingLength] = h0;
                particle[density] = rho;
                particle[internalEnergy] = P / ((gamma_eos - Real{1}) * rho);
            }
        };

        // Dimension-specific AABB constructors

#if SIM_DIM == 1

        inline auto leftFluidVol()
        {
            std::array<Real, 1> lo{Real{-1}}, hi{Real{0}};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto rightFluidVol()
        {
            std::array<Real, 1> lo{Real{0}}, hi{Real{1}};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto fluidDomain()
        {
            std::array<Real, 1> lo{Real{-1}}, hi{Real{1}};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto fullDomain()
        {
            Real const W = wallThickness();
            std::array<Real, 1> lo{Real{-1} - W}, hi{Real{1} + W};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline void addWalls(std::vector<pmacc::spearhed::AABB<CS>>& out)
        {
            Real const W = wallThickness();
            out.push_back(
                detail::aabbFromArrays<CS>(std::array<Real, 1>{Real{-1} - W}, std::array<Real, 1>{Real{-1}}));
            out.push_back(detail::aabbFromArrays<CS>(std::array<Real, 1>{Real{1}}, std::array<Real, 1>{Real{1} + W}));
        }

#elif SIM_DIM == 2

        inline auto leftFluidVol()
        {
            Real const Ly = transverseExtent();
            std::array<Real, 2> lo{Real{-1}, Real{0}}, hi{Real{0}, Ly};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto rightFluidVol()
        {
            Real const Ly = transverseExtent();
            std::array<Real, 2> lo{Real{0}, Real{0}}, hi{Real{1}, Ly};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto fluidDomain()
        {
            Real const Ly = transverseExtent();
            std::array<Real, 2> lo{Real{-1}, Real{0}}, hi{Real{1}, Ly};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto fullDomain()
        {
            Real const W = wallThickness();
            Real const Ly = transverseExtent();
            std::array<Real, 2> lo{Real{-1} - W, Real{-W}}, hi{Real{1} + W, Ly + W};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline void addWalls(std::vector<pmacc::spearhed::AABB<CS>>& out)
        {
            Real const W = wallThickness();
            Real const Ly = transverseExtent();
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 2>{Real{-1} - W, Real{-W}},
                    std::array<Real, 2>{Real{-1}, Ly + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 2>{Real{1}, Real{-W}},
                    std::array<Real, 2>{Real{1} + W, Ly + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 2>{Real{-1}, Real{-W}},
                    std::array<Real, 2>{Real{0}, Real{0}}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 2>{Real{0}, Real{-W}},
                    std::array<Real, 2>{Real{1}, Real{0}}));
            out.push_back(
                detail::aabbFromArrays<CS>(std::array<Real, 2>{Real{-1}, Ly}, std::array<Real, 2>{Real{0}, Ly + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(std::array<Real, 2>{Real{0}, Ly}, std::array<Real, 2>{Real{1}, Ly + W}));
        }

#else // SIM_DIM == 3

        inline auto leftFluidVol()
        {
            Real const L = transverseExtent();
            std::array<Real, 3> lo{Real{-1}, Real{0}, Real{0}}, hi{Real{0}, L, L};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto rightFluidVol()
        {
            Real const L = transverseExtent();
            std::array<Real, 3> lo{Real{0}, Real{0}, Real{0}}, hi{Real{1}, L, L};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto fluidDomain()
        {
            Real const L = transverseExtent();
            std::array<Real, 3> lo{Real{-1}, Real{0}, Real{0}}, hi{Real{1}, L, L};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline auto fullDomain()
        {
            Real const W = wallThickness();
            Real const L = transverseExtent();
            std::array<Real, 3> lo{Real{-1} - W, Real{-W}, Real{-W}}, hi{Real{1} + W, L + W, L + W};
            return detail::aabbFromArrays<CS>(lo, hi);
        }

        inline void addWalls(std::vector<pmacc::spearhed::AABB<CS>>& out)
        {
            Real const W = wallThickness();
            Real const L = transverseExtent();
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{-1} - W, Real{-W}, Real{-W}},
                    std::array<Real, 3>{Real{-1}, L + W, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{1}, Real{-W}, Real{-W}},
                    std::array<Real, 3>{Real{1} + W, L + W, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{-1}, Real{-W}, Real{-W}},
                    std::array<Real, 3>{Real{0}, Real{0}, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{0}, Real{-W}, Real{-W}},
                    std::array<Real, 3>{Real{1}, Real{0}, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{-1}, L, Real{-W}},
                    std::array<Real, 3>{Real{0}, L + W, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{0}, L, Real{-W}},
                    std::array<Real, 3>{Real{1}, L + W, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{-1}, Real{0}, Real{-W}},
                    std::array<Real, 3>{Real{0}, L, Real{0}}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{0}, Real{0}, Real{-W}},
                    std::array<Real, 3>{Real{1}, L, Real{0}}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{-1}, Real{0}, L},
                    std::array<Real, 3>{Real{0}, L, L + W}));
            out.push_back(
                detail::aabbFromArrays<CS>(
                    std::array<Real, 3>{Real{0}, Real{0}, L},
                    std::array<Real, 3>{Real{1}, L, L + W}));
        }

#endif

        // Blocks

        struct InteriorBlock
        {
            using Species = pmacc::spearhed::species::Default;
            using NumParticlesToCreate = SpacingNumParticles;
            using PlaceParticle = SpacingPlaceParticle;

            InitialConditions initialConditions{};
            pmacc::spearhed::AABB<CS> domain = fluidDomain();

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(spacingLeft(), spacingRight());
            }

            Real particleMass() const
            {
                Real const dx = spacingLeft();
#if SIM_DIM == 1
                return initialConditions.densityLeft * dx;
#elif SIM_DIM == 2
                return initialConditions.densityLeft * dx * dx;
#else
                return initialConditions.densityLeft * dx * dx * dx;
#endif
            }

            auto placeParticleArgs() const
            {
                return std::make_tuple(
                    spacingLeft(),
                    spacingRight(),
                    particleMass(),
                    initialConditions.densityLeft,
                    initialConditions.densityRight,
                    initialConditions.pressureLeft,
                    initialConditions.pressureRight);
            }

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
            {
                out.push_back(leftFluidVol());
                out.push_back(rightFluidVol());
            }
        };

        struct BoundaryBlock
        {
            using Species = pmacc::spearhed::species::Boundary;
            using NumParticlesToCreate = SpacingNumParticles;
            using PlaceParticle = SpacingPlaceParticle;

            InitialConditions initialConditions{};
            pmacc::spearhed::AABB<CS> domain = fullDomain();

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(spacingLeft(), spacingRight());
            }

            auto placeParticleArgs() const
            {
                Real const dx = spacingLeft();
#if SIM_DIM == 1
                Real const m = initialConditions.densityLeft * dx;
#elif SIM_DIM == 2
                Real const m = initialConditions.densityLeft * dx * dx;
#else
                Real const m = initialConditions.densityLeft * dx * dx * dx;
#endif
                return std::make_tuple(
                    spacingLeft(),
                    spacingRight(),
                    m,
                    initialConditions.densityLeft,
                    initialConditions.densityRight,
                    initialConditions.pressureLeft,
                    initialConditions.pressureRight);
            }

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
            {
                addWalls(out);
            }
        };

    } // namespace sod

    struct SodShockTube
    {
        pmacc::spearhed::AABB<CS> domain = sod::fullDomain();

        sod::InitialConditions initialConditions{};

        sod::InteriorBlock interior{.initialConditions = initialConditions};
        sod::BoundaryBlock boundary{.initialConditions = initialConditions};

        auto blocks() const
        {
            return std::tie(interior, boundary);
        }

        KernelVariant kernelVariant = makeKernel(KernelType::CubicSpline);

        using OutputParticleRecord = ll::Record<
            spearhed::tags::idField,
            spearhed::output::positionField<CS>,
            spearhed::tags::massField<Real>,
            spearhed::tags::velField<CS>,
            spearhed::tags::densityField<Real>,
            spearhed::tags::internalEnergyField<Real>>;
    };

    using Setup = SodShockTube;
} // namespace spearhed
