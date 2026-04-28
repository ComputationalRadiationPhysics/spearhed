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

#include "spearhed/plugins/openPMD/OutputParticleRecord.hpp"

#include <cstdint>
#include <string>

#include <llamaLite/DynSoA.hpp>

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
     */
    class OpenPMDOutput
    {
    public:
        /**
         * @param filepath  openPMD filename pattern, e.g. "output_%06T.h5" or "output_%06T.bp"
         *                  The pattern token %T is replaced by the iteration/step number.
         */
        explicit OpenPMDOutput(std::string const& filepath);

        /** Write all particles from @p particles at simulation step @p step. */
        void writeStep(uint32_t step, llama_lite::DynSoA<OutputParticleRecord> const& particles);

        ~OpenPMDOutput();

    private:
        openPMD::Series series_;

        /** Write a scalar (single-component) record component. */
        template<typename Tag>
        void writeScalarField(
            openPMD::ParticleSpecies& species,
            llama_lite::DynSoA<OutputParticleRecord> const& particles);

        /** Write a multi-component vector record; components are derived from the leaf fields of Tag's nested Record.
         */
        template<typename Tag>
        void writeVectorField(
            openPMD::ParticleSpecies& species,
            llama_lite::DynSoA<OutputParticleRecord> const& particles);
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

        void writeStep(uint32_t, llama_lite::DynSoA<OutputParticleRecord> const&)
        {
        }

        ~OpenPMDOutput() = default;
    };

} // namespace spearhed::output

#endif // SPEARHED_ENABLE_OPENPMD
