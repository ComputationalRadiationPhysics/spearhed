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

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cstddef>
#include <span>
#include <tuple>

#include <llamaLite/llamaLite.hpp>

namespace pmacc::spearhed
{
    namespace detail
    {
        /**
         * @brief Compile-time index lookup of a TagPath in a Tuple<TagPath...>.
         */
        template<ll::IsTagPath Query, ll::IsTagPath... Paths>
        constexpr size_t leafPathIndex(ll::Tuple<Paths...>)
        {
            return ll::indexOfType<Query, Paths...>();
        }
    } // namespace detail

    /**
     * @brief Zero-overhead POD view of common attributes across species.
     *
     * Stores one std::span<T> per leaf field in CommonRec. The type of
     * CommonFrameView<CommonRec> is identical regardless of which species'
     * frame produced it; only CommonRec matters. This gives uniform
     * attribute access without frame-list type erasure.
     *
     * @tparam CommonRec The subset Record, typically derived from an
     *                   interaction functor's neighbourReads tags plus
     *                   mandatory geometry fields (multiMask, relativePos).
     */
    template<ll::IsRecord CommonRec>
    struct CommonFrameView
    {
    private:
        using LeafPaths = typename ll::GetLeafPaths<CommonRec>::type;

        template<ll::IsTagPath... Paths>
        static constexpr auto makeSpansType(ll::Tuple<Paths...>)
            -> std::tuple<std::span<typename CommonRec::template value_type_for<Paths>>...>;

    public:
        using SpansTuple = decltype(makeSpansType(LeafPaths{}));
        SpansTuple spans;

        /**
         * @brief Access the frame-wide span for leaf field identified by RA.
         */
        template<ll::IsRecordAccess RA>
        [[nodiscard]] HDINLINE constexpr auto getSpan()
        {
            constexpr size_t idx = detail::leafPathIndex<ll::to_path_t<RA>>(LeafPaths{});
            return std::get<idx>(spans);
        }

        /** @brief Const-qualified span access. */
        template<ll::IsRecordAccess RA>
        [[nodiscard]] HDINLINE constexpr auto getSpan() const
        {
            constexpr size_t idx = detail::leafPathIndex<ll::to_path_t<RA>>(LeafPaths{});
            return std::get<idx>(spans);
        }
    };

    /**
     * @brief Build a CommonFrameView<CommonRec> from any frame whose
     *        record contains all paths in CommonRec.
     *
     * Compile error if the frame's SoA is missing any leaf field.
     *
     * @tparam CommonRec The common record subset.
     * @param frame      A Frame (containing an ll::SoA) or raw ll::SoA.
     */
    template<ll::IsRecord CommonRec, typename FrameStorage>
    HDINLINE constexpr CommonFrameView<CommonRec> makeCommonFrameView(FrameStorage& frame)
    {
        using LeafPaths = typename ll::GetLeafPaths<CommonRec>::type;
        return {
            .spans = [&]<ll::IsTagPath... Paths>(ll::Tuple<Paths...>)
            { return std::make_tuple(frame.template getLeaf<Paths>()...); }(LeafPaths{})};
    }

} // namespace pmacc::spearhed
