/* Copyright 2021-2024 Sergei Bastrakov
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstdint>

#include <caravan/alpaka.hpp>

namespace pmacc
{
    namespace particles
    {
        namespace policies
        {
            //! Policy for HandleGuardParticles that does nothing
            struct DoNothing
            {
                /** Empty sender for the outgoing direction. */
                template<typename T_Particles>
                [[nodiscard]] auto handleOutgoing(T_Particles&, int32_t) const
                {
                    return caravan::alpaka::submit([](auto&) {});
                }

                /** Empty sender for the incoming direction. */
                template<typename T_Particles>
                [[nodiscard]] auto handleIncoming(T_Particles&, int32_t) const
                {
                    return caravan::alpaka::submit([](auto&) {});
                }
            };

        } // namespace policies
    } // namespace particles
} // namespace pmacc
