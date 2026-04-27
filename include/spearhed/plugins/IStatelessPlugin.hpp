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

#include <pmacc/pluginSystem/IPlugin.hpp>

#include <cstdint>
#include <string>

namespace spearhed
{
    /**
     * Interface for a lightweight simulation plugin
     * without checkpoint/restart capabilities.
     */
    class IStatelessPlugin : public pmacc::IPlugin
    {
    public:
        void restart(uint32_t, std::string const) override
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        void checkpoint(uint32_t, std::string const) override
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        ~IStatelessPlugin() override = default;
    };
} // namespace spearhed
