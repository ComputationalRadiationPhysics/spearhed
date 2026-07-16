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

#include "spearhed/plugins/openPMD/Position.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include <llamaLite/DynSoA.hpp>
#include <llamaLite/Record.hpp>
#include <llamaLite/tag/TagPath.hpp>

#ifdef SPEARHED_ENABLE_OPENPMD

#    include "spearhed/plugins/openPMD/ParticleTraits.hpp"

#    include <openPMD/openPMD.hpp>

namespace spearhed::output
{
    inline constexpr bool openPMDEnabled = true;

    /**
     * Writes particle data to an openPMD Series (HDF5 or ADIOS2 depending on file extension).
     *
     * The destructor flushes and closes the Series. Ensure the object outlives all writeStep calls.
     *
     * The set of fields written is determined by the Record type passed to writeStep:
     * each top-level field is written as a scalar (leaf value type) or vector (nested Record)
     * record component, with openPMD metadata sourced from OpenPMDTrait<Tag>.
     */
    class OpenPMDOutput
    {
    public:
        /**
         * @param filepath  openPMD filename pattern, e.g. "output_%06T.h5" or "output_%06T.bp"
         *                  The pattern token %T is replaced by the iteration/step number.
         */
        explicit OpenPMDOutput(std::string const& filepath);

        ~OpenPMDOutput();

        /** Write all particles from @p particles at simulation step @p step. */
        template<typename Record>
        void writeStep(uint32_t step, llama_lite::DynSoA<Record> const& particles)
        {
            if(particles.size() == 0u)
                return;

            auto& iter = series_.iterations[step];
            auto& species = iter.particles["fluid"];

            using Fields = typename Record::fields_tuple_type;
            constexpr std::size_t N = std::tuple_size_v<Fields>;

            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                (writeOneField<std::tuple_element_t<I, Fields>>(species, particles), ...);
            }(std::make_index_sequence<N>{});

            series_.flush();
        }

    private:
        openPMD::Series series_;

        template<typename Field, typename Record>
        static void writeOneField(openPMD::ParticleSpecies& species, llama_lite::DynSoA<Record> const& particles)
        {
            using Tag = typename Field::tag_type;
            using Val = typename Field::value_type;
            if constexpr(llama_lite::IsRecord<Val>)
                writeVectorField<Tag>(species, particles);
            else
                writeScalarField<Tag>(species, particles);
        }

        template<typename Tag, typename Record>
        static void writeScalarField(openPMD::ParticleSpecies& species, llama_lite::DynSoA<Record> const& particles)
        {
            using Trait = OpenPMDTrait<Tag>;
            using ValueType = typename Record::template value_type_for<Tag>;

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

            auto span = particles.template getLeaf<Tag>();
            comp.storeChunkRaw(span.data(), {0}, {n});
        }

        template<typename Tag, typename Record>
        static void writeVectorField(openPMD::ParticleSpecies& species, llama_lite::DynSoA<Record> const& particles)
        {
            namespace ll = llama_lite;

            using Trait = OpenPMDTrait<Tag>;
            using RecordType = typename Record::template value_type_for<Tag>;
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

            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                (
                    [&]<typename LeafPath>(LeafPath)
                    {
                        using ValueType = typename Record::template value_type_for<LeafPath>;
                        auto& comp = record[ll::tagPathToString<ll::relative_path_t<LeafPath, Tag>>()];
                        comp.resetDataset(openPMD::Dataset(openPMD::determineDatatype<ValueType>(), {n}));
                        comp.setUnitSI(Trait::unitSI);
                        auto span = particles.template getLeaf<LeafPath>();
                        comp.storeChunkRaw(span.data(), {0}, {n});
                    }(std::tuple_element_t<I, LeafPaths>{}),
                    ...);
            }(std::make_index_sequence<std::tuple_size_v<LeafPaths>>{});
        }
    };

} // namespace spearhed::output

#else // SPEARHED_ENABLE_OPENPMD

namespace spearhed::output
{
    inline constexpr bool openPMDEnabled = false;

    class OpenPMDOutput
    {
    public:
        explicit OpenPMDOutput(std::string const&)
        {
        }

        template<typename Record>
        void writeStep(uint32_t, llama_lite::DynSoA<Record> const&)
        {
        }

        ~OpenPMDOutput() = default;
    };

} // namespace spearhed::output

#endif // SPEARHED_ENABLE_OPENPMD
