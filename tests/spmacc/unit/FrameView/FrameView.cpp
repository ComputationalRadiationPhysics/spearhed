/* Copyright 2026 Tapish Narwal
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

#include <spmacc/Frame.hpp>
#include <spmacc/particles/algorithms/HierarchyForEach.hpp>
#include <spmacc/particles/views/FrameView.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>

namespace sp = pmacc::spearhed;

namespace
{
    DEFINE_TAG(value);

    struct Species
    {
        using Roles = sp::role_set<>;
        static constexpr auto name = pmacc::meta::FixedString{"FrameViewTest"};
    };

    using Record = ll::Record<sp::tags::MultiMaskField, ll::Field<value_t, float>>;
    using Description = sp::ParticleDescription<Species, std::integral_constant<uint32_t, 4u>, Record>;
    using Frame = sp::Frame<Description>;
    using MutableView = sp::FrameView<Frame>;
    using ReadOnlyView = sp::FrameView<Frame const>;

    static_assert(std::is_trivially_copyable_v<MutableView>);
    static_assert(std::is_trivially_copyable_v<ReadOnlyView>);
    static_assert(std::is_standard_layout_v<MutableView>);
    static_assert(sizeof(MutableView) == sizeof(Frame*));
    static_assert(sizeof(ReadOnlyView) == sizeof(Frame const*));
    static_assert(!std::is_default_constructible_v<MutableView>);
    static_assert(!std::is_constructible_v<MutableView, Frame*>);
    static_assert(!std::is_constructible_v<MutableView, std::nullptr_t>);
    static_assert(std::convertible_to<MutableView, ReadOnlyView>);
    static_assert(!std::is_constructible_v<MutableView, ReadOnlyView>);
    static_assert(MutableView::frameSize == Frame::frameSize);

    // View-object constness does not propagate; storage constness does, for slots and columns.
    static_assert(std::same_as<decltype(std::declval<MutableView const&>()[0u][value]), float&>);
    static_assert(std::same_as<decltype(std::declval<ReadOnlyView const&>()[0u][value]), float const&>);
    static_assert(std::same_as<decltype(std::declval<MutableView const&>()[value][0u]), float&>);
    static_assert(std::same_as<decltype(std::declval<ReadOnlyView const&>()[value][0u]), float const&>);
    static_assert(!std::is_assignable_v<decltype(std::declval<ReadOnlyView>()[0u][value]), float>);
    static_assert(std::same_as<decltype(std::declval<MutableView const&>().get()), Frame*>);
    static_assert(std::same_as<decltype(std::declval<ReadOnlyView const&>().get()), Frame const*>);
} // namespace

TEST_CASE("FrameView aliases slots and attributes without owning storage", "[frame_view]")
{
    Frame frame;
    sp::FrameView const view{frame};
    STATIC_REQUIRE(std::same_as<std::remove_cv_t<decltype(view)>, MutableView>);
    REQUIRE(view.get() == &frame);

    // Dead slots remain accessible for initialization; the view does not perform live filtering.
    REQUIRE_FALSE(sp::pred::Occupied{}(view[1u]));
    view[1u][value] = 7.0f;
    view[value][2u] = 9.0f;
    REQUIRE(frame[1u][value] == 7.0f);
    REQUIRE(frame[2u][value] == 9.0f);

    auto copy = view;
    copy[1u][value] = 11.0f;
    REQUIRE(view[1u][value] == 11.0f);

    Frame other;
    copy = MutableView{other};
    REQUIRE(copy.get() == &other);
    REQUIRE(view.get() == &frame);
}

TEST_CASE("FrameView const storage yields read-only proxies", "[frame_view]")
{
    Frame frame;
    sp::FrameView const mutableView{frame};
    mutableView[0u][value] = 5.0f;

    sp::FrameView const readOnlyView{std::as_const(frame)};
    STATIC_REQUIRE(std::same_as<std::remove_cv_t<decltype(readOnlyView)>, ReadOnlyView>);
    REQUIRE(readOnlyView[0u][value] == 5.0f);
    REQUIRE(readOnlyView[value][0u] == 5.0f);

    ReadOnlyView converted = mutableView;
    REQUIRE(converted.get() == readOnlyView.get());
    mutableView[0u][value] = 13.0f;
    REQUIRE(converted[0u][value] == 13.0f);
}

TEST_CASE("FrameView traversal filters live slots separately from access", "[frame_view][hierarchy]")
{
    Frame frame;
    sp::FrameView const view{frame};
    for(uint32_t slot = 0u; slot < view.frameSize; ++slot)
        view[slot][value] = 1.0f;
    view[1u][sp::tags::multiMask] = 1u;
    view[3u][sp::tags::multiMask] = 1u;

    sp::forEach(sp::levels::particle, sp::deviceHeap, view, [](auto particle) { particle[value] += 2.0f; });
    REQUIRE(view[0u][value] == 1.0f);
    REQUIRE(view[1u][value] == 3.0f);
    REQUIRE(view[2u][value] == 1.0f);
    REQUIRE(view[3u][value] == 3.0f);

    float sum = 0.0f;
    sp::forEach(
        sp::levels::particle,
        sp::deviceHeap,
        ReadOnlyView{view},
        [&](auto particle) { sum += particle[value]; });
    REQUIRE(sum == 6.0f);
}
