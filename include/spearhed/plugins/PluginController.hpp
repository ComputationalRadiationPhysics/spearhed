/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of SPEARHED, derived from PIConGPU.
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

#include <pmacc/pluginSystem/IPlugin.hpp>

#include <memory>
#include <vector>

namespace spearhed
{
    /**
     * Plugin management controller for user-level plugins.
     */
    class PluginController : public IStatelessPlugin
    {
    private:
        std::vector<std::shared_ptr<pmacc::IPlugin>> plugins;

        /**
         * Initializes the controller by adding all user plugins to its internal list.
         */
        virtual void init()
        {
        }

    public:
        PluginController()
        {
            init();
        }

        ~PluginController() override = default;

        void pluginRegisterHelp(boost::program_options::options_description&) override
        {
            // no help required at the moment
        }

        std::string pluginGetName() const override
        {
            return "PluginController";
        }

        void notify(uint32_t) override
        {
        }

        void pluginUnload() override
        {
            plugins.clear();
        }
    };

} // namespace spearhed
