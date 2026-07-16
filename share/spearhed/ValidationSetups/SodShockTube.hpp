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
        // Wall thickness must cover one kernel support radius so the kernel sums for
        // fluid particles right at the end caps see a full neighbour disk.
        // CubicSplineKernel::supportRadius = 2 => W = 2 * h0 = 0.1.
        constexpr Real wallThickness = Real{2} * h0;

        struct InitialConditions
        {
            // Left state (x < 0)
            Real densityLeft = Real{1.0};
            Real pressureLeft = Real{1.0};
            // Right state (x >= 0)
            Real densityRight = Real{0.125};
            Real pressureRight = Real{0.1};
        };

        // Fluid: x is in [-1, 1], split at 0
        constexpr pmacc::spearhed::AABB<CS> leftFluidVol{{Real{0}}, {Real{-1.0}}, {Real{0.0}}};
        constexpr pmacc::spearhed::AABB<CS> rightFluidVol{{Real{0}}, {Real{0.0}}, {Real{1.0}}};

        // Walls flank the fluid by one support radius
        constexpr pmacc::spearhed::AABB<CS> leftWallVol{{Real{0}}, {Real{-1.0} - wallThickness}, {Real{-1.0}}};
        constexpr pmacc::spearhed::AABB<CS> rightWallVol{{Real{0}}, {Real{1.0}}, {Real{1.0} + wallThickness}};

        constexpr uint32_t defaultTotalParticles = 32000u;

        // Helpers shared by both blocks

        // Equal-mass distribution: mass per particle is fixed so each region's particle
        // count is proportional to rho * V. Using float32 internally for the volume sum.
        constexpr Real totalWeightedFluidVolume(InitialConditions const& ic)
        {
            return pmacc::spearhed::computeVolume(leftFluidVol) * ic.densityLeft
                   + pmacc::spearhed::computeVolume(rightFluidVol) * ic.densityRight;
        }

        constexpr Real particleMass(InitialConditions const& ic, uint32_t totalParticles)
        {
            return totalWeightedFluidVolume(ic) / static_cast<Real>(totalParticles);
        }

        constexpr uint32_t numFluidLeft(InitialConditions const& ic, uint32_t totalParticles)
        {
            return static_cast<uint32_t>(
                static_cast<Real>(totalParticles) * ic.densityLeft * pmacc::spearhed::computeVolume(leftFluidVol)
                / totalWeightedFluidVolume(ic));
        }

        constexpr uint32_t numFluidRight(InitialConditions const& ic, uint32_t totalParticles)
        {
            return static_cast<uint32_t>(
                static_cast<Real>(totalParticles) * ic.densityRight * pmacc::spearhed::computeVolume(rightFluidVol)
                / totalWeightedFluidVolume(ic));
        }

        // Wall counts match adjacent fluid density => same particle spacing as the fluid,
        // which keeps neighbour sums consistent across the fluid/wall interface.
        constexpr uint32_t numWallLeft(InitialConditions const& ic, uint32_t totalParticles)
        {
            return static_cast<uint32_t>(
                static_cast<Real>(numFluidLeft(ic, totalParticles)) * pmacc::spearhed::computeVolume(leftWallVol)
                / pmacc::spearhed::computeVolume(leftFluidVol));
        }

        constexpr uint32_t numWallRight(InitialConditions const& ic, uint32_t totalParticles)
        {
            return static_cast<uint32_t>(
                static_cast<Real>(numFluidRight(ic, totalParticles)) * pmacc::spearhed::computeVolume(rightWallVol)
                / pmacc::spearhed::computeVolume(rightFluidVol));
        }

        // Interior

        struct InteriorBlock
        {
            pmacc::spearhed::AABB<CS> domain{{Real{0}}, {Real{-1.0}}, {Real{1.0}}};

            using Species = pmacc::spearhed::species::Default;

            InitialConditions initialConditions{};
            uint32_t totalParticles = defaultTotalParticles;

            struct NumParticlesToCreate
            {
                DINLINE constexpr auto operator()(
                    auto& /*worker*/,
                    auto& particleRegion,
                    Real densityLeft,
                    Real densityRight,
                    uint32_t totalParticles,
                    Real totalWeightedVolume) const
                {
                    using x_t = std::tuple_element_t<0, typename CS::tags>;
                    Real const rho = (particleRegion.volume.max[x_t{}] <= Real{0}) ? densityLeft : densityRight;
                    Real const regionVolume = pmacc::spearhed::computeVolume(particleRegion.volume);
                    return static_cast<uint32_t>(
                        static_cast<Real>(totalParticles) * rho * regionVolume / totalWeightedVolume);
                }
            };

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(
                    initialConditions.densityLeft,
                    initialConditions.densityRight,
                    totalParticles,
                    totalWeightedFluidVolume(initialConditions));
            }

            struct PlaceParticle
            {
                DINLINE constexpr void operator()(
                    auto const& worker,
                    auto& particle,
                    auto const& particleRegion,
                    uint32_t globalParticleIdx,
                    uint32_t numCellsLeft,
                    uint32_t numCellsRight,
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
                    uint32_t const numCells = isLeft ? numCellsLeft : numCellsRight;

                    pmacc::spearhed::SC<CS>{}(worker, particle, particleRegion, globalParticleIdx, numCells);

                    pmacc::spearhed::for_each_tag<CS>([&](auto axisTag) { particle[vel][axisTag] = Real{0}; });

                    particle[mass] = pmass;
                    particle[smoothingLength] = h0;
                    particle[density] = rho;
                    particle[internalEnergy] = P / ((gamma_eos - Real{1}) * rho);
                }
            };

            auto placeParticleArgs() const
            {
                return std::make_tuple(
                    pmacc::spearhed::computeSCNumCells(numFluidLeft(initialConditions, totalParticles), leftFluidVol),
                    pmacc::spearhed::computeSCNumCells(
                        numFluidRight(initialConditions, totalParticles),
                        rightFluidVol),
                    particleMass(initialConditions, totalParticles),
                    initialConditions.densityLeft,
                    initialConditions.densityRight,
                    initialConditions.pressureLeft,
                    initialConditions.pressureRight);
            }

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
            {
                out.push_back(leftFluidVol);
                out.push_back(rightFluidVol);
            }
        };

        // Boundary (frozen walls)

        struct BoundaryBlock
        {
            pmacc::spearhed::AABB<CS> domain{{Real{0}}, {Real{-1.0} - wallThickness}, {Real{1.0} + wallThickness}};

            using Species = pmacc::spearhed::species::Boundary;

            InitialConditions initialConditions{};
            uint32_t totalParticles = defaultTotalParticles;

            struct NumParticlesToCreate
            {
                DINLINE constexpr auto operator()(
                    auto& /*worker*/,
                    auto& particleRegion,
                    uint32_t numLeft,
                    uint32_t numRight) const
                {
                    using x_t = std::tuple_element_t<0, typename CS::tags>;
                    return (particleRegion.volume.max[x_t{}] <= Real{0}) ? numLeft : numRight;
                }
            };

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(
                    numWallLeft(initialConditions, totalParticles),
                    numWallRight(initialConditions, totalParticles));
            }

            struct PlaceParticle
            {
                DINLINE constexpr void operator()(
                    auto const& worker,
                    auto& particle,
                    auto const& particleRegion,
                    uint32_t globalParticleIdx,
                    uint32_t numCellsLeft,
                    uint32_t numCellsRight,
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
                    uint32_t const numCells = isLeft ? numCellsLeft : numCellsRight;

                    pmacc::spearhed::SC<CS>{}(worker, particle, particleRegion, globalParticleIdx, numCells);

                    pmacc::spearhed::for_each_tag<CS>([&](auto axisTag) { particle[vel][axisTag] = Real{0}; });

                    particle[mass] = pmass;
                    particle[smoothingLength] = h0;
                    particle[density] = rho;
                    particle[internalEnergy] = P / ((gamma_eos - Real{1}) * rho);
                }
            };

            auto placeParticleArgs() const
            {
                return std::make_tuple(
                    pmacc::spearhed::computeSCNumCells(numWallLeft(initialConditions, totalParticles), leftWallVol),
                    pmacc::spearhed::computeSCNumCells(numWallRight(initialConditions, totalParticles), rightWallVol),
                    particleMass(initialConditions, totalParticles),
                    initialConditions.densityLeft,
                    initialConditions.densityRight,
                    initialConditions.pressureLeft,
                    initialConditions.pressureRight);
            }

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
            {
                out.push_back(leftWallVol);
                out.push_back(rightWallVol);
            }
        };
    } // namespace sod

    struct SodShockTube
    {
        // Full domain spans fluid + walls, used for diagnostic printing only.
        pmacc::spearhed::AABB<CS> domain{
            {Real{0}},
            {Real{-1.0} - sod::wallThickness},
            {Real{1.0} + sod::wallThickness}};

        sod::InitialConditions initialConditions{};
        uint32_t totalParticles = sod::defaultTotalParticles;

        sod::InteriorBlock interior{.initialConditions = initialConditions, .totalParticles = totalParticles};
        sod::BoundaryBlock boundary{.initialConditions = initialConditions, .totalParticles = totalParticles};

        auto blocks() const
        {
            return std::tie(interior, boundary);
        }

        KernelVariant kernelVariant = makeKernel(KernelType::CubicSpline);

        // Sod validation needs density and internal energy in addition to id/position/mass/velocity
        // so a Python post-processor can reconstruct pressure via P = (gamma - 1) * rho * u.
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
