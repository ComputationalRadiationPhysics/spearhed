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

#include "spearhed/plugins/IStatelessPlugin.hpp"
#include "spearhed/plugins/openPMD/OpenPMDOutput.hpp"

#include <optional>
#include <string>

namespace spearhed
{
    /**
     * Plugin that writes particle data to an openPMD Series at a configurable period.
     *
     * Self-registers with PluginConnector on construction. When loaded, calls
     * setNotificationPeriod so PluginConnector drives notify() at the right steps.
     */
    class OpenPMDPlugin : public IStatelessPlugin
    {
    public:
        OpenPMDPlugin();
        ~OpenPMDPlugin() override = default;

        void pluginRegisterHelp(boost::program_options::options_description& desc) override;
        std::string pluginGetName() const override;
        void notify(uint32_t currentStep) override;

    protected:
        void pluginLoad() override;
        void pluginUnload() override;

    private:
        uint32_t period{0u};
        std::string filepath{"output_%06T.h5"};
        std::optional<output::OpenPMDOutput> writer;
    };
} // namespace spearhed
