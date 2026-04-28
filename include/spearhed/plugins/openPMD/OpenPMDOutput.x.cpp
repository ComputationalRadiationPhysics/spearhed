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

#include "spearhed/plugins/openPMD/OpenPMDOutput.hpp"

#include <llamaLite/tag/TagPath.hpp>

#ifdef SPEARHED_ENABLE_OPENPMD

namespace spearhed::output
{
    OpenPMDOutput::OpenPMDOutput(std::string const& filepath) : series_(filepath, openPMD::Access::CREATE)
    {
        series_.setAuthor("spearhed");
        series_.setSoftware("spearhed");
    }

    OpenPMDOutput::~OpenPMDOutput() = default;

    void OpenPMDOutput::writeStep(uint32_t const step, llama_lite::DynSoA<OutputParticleRecord> const& particles)
    {
        if(particles.size() == 0u)
            return;

        auto& iter = series_.iterations[step];
        auto& species = iter.particles["fluid"];

        writeScalarField<spearhed::tags::particleId_t>(species, particles);
        writeVectorField<position_t>(species, particles);
        writeScalarField<spearhed::tags::mass_t>(species, particles);
        writeVectorField<spearhed::tags::vel_t>(species, particles);

        series_.flush();
    }

    template<typename Tag>
    void OpenPMDOutput::writeScalarField(
        openPMD::ParticleSpecies& species,
        llama_lite::DynSoA<OutputParticleRecord> const& particles)
    {
        using Trait = OpenPMDTrait<Tag>;
        using ValueType = typename OutputParticleRecord::template value_type_for<Tag>;

        auto& record = species[std::string(Trait::record)];
        record.setUnitDimension(
            {{openPMD::UnitDimension::L, Trait::unitDimension[0]},
             {openPMD::UnitDimension::M, Trait::unitDimension[1]},
             {openPMD::UnitDimension::T, Trait::unitDimension[2]},
             {openPMD::UnitDimension::I, Trait::unitDimension[3]},
             {openPMD::UnitDimension::theta, Trait::unitDimension[4]},
             {openPMD::UnitDimension::N, Trait::unitDimension[5]},
             {openPMD::UnitDimension::J, Trait::unitDimension[6]}});

        auto& comp = record[openPMD::RecordComponent::SCALAR];
        auto const n = static_cast<uint64_t>(particles.size());
        comp.resetDataset(openPMD::Dataset(openPMD::determineDatatype<ValueType>(), {n}));
        comp.setUnitSI(Trait::unitSI);

        auto span = particles.getLeaf<Tag>();
        comp.storeChunkRaw(span.data(), {0}, {n});
    }

    template<typename Tag>
    void OpenPMDOutput::writeVectorField(
        openPMD::ParticleSpecies& species,
        llama_lite::DynSoA<OutputParticleRecord> const& particles)
    {
        using Trait = OpenPMDTrait<Tag>;
        using RecordType = typename OutputParticleRecord::template value_type_for<Tag>;
        static_assert(ll::IsRecord<RecordType>, "writeVectorField requires Tag to resolve to a nested Record");

        auto& record = species[std::string(Trait::record)];
        record.setUnitDimension(
            {{openPMD::UnitDimension::L, Trait::unitDimension[0]},
             {openPMD::UnitDimension::M, Trait::unitDimension[1]},
             {openPMD::UnitDimension::T, Trait::unitDimension[2]},
             {openPMD::UnitDimension::I, Trait::unitDimension[3]},
             {openPMD::UnitDimension::theta, Trait::unitDimension[4]},
             {openPMD::UnitDimension::N, Trait::unitDimension[5]},
             {openPMD::UnitDimension::J, Trait::unitDimension[6]}});

        auto const n = static_cast<uint64_t>(particles.size());

        using LeafPaths = typename ll::GetLeafPaths<RecordType, ll::TagPath<Tag>>::type;

        [&]<size_t... I>(std::index_sequence<I...>)
        {
            (
                [&]<typename LeafPath>(LeafPath)
                {
                    using ValueType = typename OutputParticleRecord::template value_type_for<LeafPath>;
                    auto& comp = record[ll::tagPathToString<ll::relative_path_t<LeafPath, Tag>>()];
                    comp.resetDataset(openPMD::Dataset(openPMD::determineDatatype<ValueType>(), {n}));
                    comp.setUnitSI(Trait::unitSI);
                    auto span = particles.getLeaf<LeafPath>();
                    comp.storeChunkRaw(span.data(), {0}, {n});
                }(std::tuple_element_t<I, LeafPaths>{}),
                ...);
        }(std::make_index_sequence<std::tuple_size_v<LeafPaths>>{});
    }

} // namespace spearhed::output

#endif // SPEARHED_ENABLE_OPENPMD
