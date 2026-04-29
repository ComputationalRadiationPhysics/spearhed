/* Copyright 2025-2026 Tapish Narwal
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

#include "spmacc/topology/CoordinateSystem.hpp"

#include <utility>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace tags
    {
        DEFINE_TAG(x);
        DEFINE_TAG(y);
        DEFINE_TAG(z);

    } // namespace tags

    // // Restrict access to valid cartesian tags
    template<typename T>
    concept CartesianTag = std::same_as<T, tags::x_t> || std::same_as<T, tags::y_t> || std::same_as<T, tags::z_t>;

    namespace detail
    {
        template<CoordinateSystem CS, typename T, typename Idx>
        struct CartesianRecordImpl;

        template<CoordinateSystem CS, typename T, std::size_t... Is>
        struct CartesianRecordImpl<CS, T, std::index_sequence<Is...>>
        {
            using type = ll::Record<ll::Field<tag_of<CS, Is>, T>...>;
        };
    } // namespace detail

    template<CoordinateSystem CS>
    using CartesianRecord = typename detail::CartesianRecordImpl<
        CS,
        typename CS::T_Axis,
        std::make_index_sequence<std::tuple_size_v<typename CS::tags>>>::type;

} // namespace pmacc::spearhed
