/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <type_traits>

namespace pmacc::spearhed
{
    /** @brief Query parameters that determine one directed interaction plan. */
    template<typename T_Radius>
    struct InteractionQuery
    {
        T_Radius radius;
    };
} // namespace pmacc::spearhed
