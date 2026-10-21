/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include <concepts>
#include <cstdint>

namespace pmacc::spearhed
{
    /** Non-owning view of one valid frame in the calling address space.
     *
     * The frame must outlive the view and any asynchronous work using it. Construction from a
     * reference excludes an empty state; nullable storage/allocation handles remain raw pointers.
     * Like std::span, constness of the view does not propagate: FrameView<Frame> permits writes,
     * while FrameView<Frame const> yields read-only proxies.
     *
     * Indexing exposes all slots (including dead slots) or frame-wide attributes. Liveness filtering
     * and work distribution belong to the traversal/lockstep algorithms, not this view.
     *
     * @tparam T_Frame Frame storage type, optionally const-qualified.
     */
    template<typename T_Frame>
    class FrameView
    {
    public:
        using FrameType = T_Frame;
        static constexpr uint32_t frameSize = FrameType::frameSize;

        HDINLINE explicit constexpr FrameView(T_Frame& frame) noexcept : m_frame(&frame)
        {
        }

        //! Allow mutable-to-read-only conversion without copying frame storage.
        template<typename T_OtherFrame>
        requires std::convertible_to<T_OtherFrame*, T_Frame*>
        HDINLINE constexpr FrameView(FrameView<T_OtherFrame> other) noexcept : m_frame(other.get())
        {
        }

        //! Access a particle slot or attribute, preserving the underlying proxy/reference type.
        [[nodiscard]] HDINLINE constexpr decltype(auto) operator[](auto access) const
        {
            return (*m_frame)[access];
        }

        //! Frame identity and explicit access to storage/metadata, without changing pointee constness.
        [[nodiscard]] HDINLINE constexpr T_Frame* get() const noexcept
        {
            return m_frame;
        }

    private:
        T_Frame* m_frame;
    };
} // namespace pmacc::spearhed
