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

#include "spearhed/plugins/openPMD/OpenPMDPlugin.hpp"

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/memory.hpp"
#include "spearhed/plugins/openPMD/OutputParticleRecord.hpp"
#include "spmacc/particles/algorithms/CopyParticlesToDynSoA.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"

#include <pmacc/Environment.hpp>

#include <string>

namespace spearhed
{
    OpenPMDPlugin::OpenPMDPlugin()
    {
        pmacc::Environment<>::get().PluginConnector().registerPlugin(this);
    }

    void OpenPMDPlugin::pluginRegisterHelp(boost::program_options::options_description& desc)
    {
        if constexpr(output::openPMDEnabled)
        {
            // clang-format off
            desc.add_options()
                ("openPMD.period", pmacc::po::value<uint32_t>(&period)->default_value(0u),
                 "write openPMD output every N steps (0 = disabled)")
                ("openPMD.file", pmacc::po::value<std::string>(&filepath)->default_value("output_%06T.h5"),
                 "openPMD filename pattern, e.g. output_%06T.h5 or output_%06T.bp");
            // clang-format on
        }
    }

    std::string OpenPMDPlugin::pluginGetName() const
    {
        return "OpenPMDOutput";
    }

    void OpenPMDPlugin::pluginLoad()
    {
        if constexpr(output::openPMDEnabled)
        {
            if(period > 0u)
            {
                writer.emplace(filepath);
                pmacc::Environment<>::get().PluginConnector().setNotificationPeriod(this, std::to_string(period));
            }
        }
    }

    void OpenPMDPlugin::pluginUnload()
    {
        writer.reset();
    }

    void OpenPMDPlugin::notify(uint32_t currentStep)
    {
        if constexpr(output::openPMDEnabled)
        {
            auto& dc = pmacc::Environment<>::get().DataConnector();
            auto& mallocMCBuf
                = *dc.get<pmacc::MallocMCBuffer<DeviceHeap>>(pmacc::MallocMCBuffer<DeviceHeap>::getName());
            mallocMCBuf.synchronize();

            // Serialise every present species carrying the OpenPMDOutput role, rather than a hardcoded
            // species. Species the active setup never created are skipped automatically.
            pmacc::spearhed::forEachSpeciesBufWithPred(
                allSpecies,
                pmacc::spearhed::pred::withRole<pmacc::spearhed::roles::OpenPMDOutput>,
                [&](auto& prBuf)
                {
                    llama_lite::DynSoA<output::OutputParticleRecord> hostParticles;
                    pmacc::spearhed::CopyParticlesToDynSoA{}(prBuf, hostParticles, syncHeapToHost());
                    writer->writeStep(currentStep, hostParticles);
                });
        }
    }
} // namespace spearhed
