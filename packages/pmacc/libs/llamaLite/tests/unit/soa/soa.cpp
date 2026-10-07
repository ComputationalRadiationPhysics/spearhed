// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the license was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(firstColumn);
DEFINE_TAG(secondColumn);
DEFINE_TAG(nestedColumn);
DEFINE_TAG(byteColumn);
DEFINE_TAG(overAlignedColumn);

namespace
{
    struct alignas(64) OverAligned
    {
        int value;
    };

    struct alignas(256) OverAligned256
    {
        int value;
    };

    using ByteRecord = ll::Record<ll::Field<byteColumn_t, unsigned char>>;
    using PairRecord = ll::Record<ll::Field<firstColumn_t, int>, ll::Field<secondColumn_t, int>>;
    using NestedRecord = ll::Record<
        ll::Field<nestedColumn_t, ll::Record<ll::Field<firstColumn_t, int>>>,
        ll::Field<byteColumn_t, unsigned char>,
        ll::Field<overAlignedColumn_t, OverAligned>,
        ll::Field<secondColumn_t, OverAligned256>>;
} // namespace

TEST_CASE("SoA column alignment policies control storage and preserve access", "[SoA]")
{
    using DefaultSoA = ll::SoA<PairRecord, 2>;
    using CompactSoA = ll::SoA<PairRecord, 2, ll::CompactAlignment>;
    using AlignedSoA = ll::SoA<PairRecord, 2, ll::ColumnAlignment<32>>;

    STATIC_CHECK(alignof(DefaultSoA) == 128);
    STATIC_CHECK(sizeof(DefaultSoA) == 256);
    STATIC_CHECK(alignof(CompactSoA) == alignof(int));
    STATIC_CHECK(sizeof(CompactSoA) == 2 * 2 * sizeof(int));
    STATIC_CHECK(alignof(AlignedSoA) == 32);
    STATIC_CHECK(sizeof(AlignedSoA) == 64);

    AlignedSoA aligned{};
    auto first = aligned.getLeaf(firstColumn);
    auto second = aligned.getLeaf(secondColumn);
    CHECK(reinterpret_cast<std::uintptr_t>(first.data()) % 32 == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(second.data()) % 32 == 0);

    first[0] = 17;
    auto mutableView = aligned.view(firstColumn);
    *mutableView[0] = 29;
    CHECK(first[0] == 29);
    AlignedSoA const& constAligned = aligned;
    auto constView = constAligned.view(firstColumn);
    CHECK(*constView[0] == 29);
    STATIC_CHECK(std::is_same_v<decltype(*constView[0]), int const&>);
}

TEST_CASE("SoA policies preserve nested and naturally over-aligned columns", "[SoA]")
{
    using CompactSoA = ll::SoA<NestedRecord, 2, ll::CompactAlignment>;
    using CompactByteSoA = ll::SoA<ByteRecord, 2, ll::CompactAlignment>;
    using ExplicitSoA = ll::SoA<NestedRecord, 2, ll::ColumnAlignment<32>>;

    STATIC_CHECK(sizeof(CompactByteSoA) == 2 * sizeof(unsigned char));

    CompactSoA compact{};
    auto nested = compact.getLeaf(nestedColumn / firstColumn);
    auto bytes = compact.getLeaf(byteColumn);
    auto overAligned = compact.getLeaf(overAlignedColumn);
    auto overAligned256 = compact.getLeaf(secondColumn);
    CHECK(reinterpret_cast<std::uintptr_t>(nested.data()) % alignof(int) == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(overAligned.data()) % alignof(OverAligned) == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(overAligned256.data()) % alignof(OverAligned256) == 0);
    nested[0] = 23;
    bytes[0] = 1;
    overAligned[0].value = 42;
    overAligned256[0].value = 64;
    CHECK(nested[0] == 23);
    CHECK(bytes[0] == 1);
    CHECK(overAligned[0].value == 42);
    CHECK(overAligned256[0].value == 64);

    STATIC_CHECK(alignof(CompactSoA) >= alignof(OverAligned256));
    STATIC_CHECK(alignof(ExplicitSoA) >= 32);
    STATIC_CHECK(alignof(ExplicitSoA) >= alignof(OverAligned));
    STATIC_CHECK(alignof(ExplicitSoA) >= alignof(OverAligned256));
    ExplicitSoA explicitAligned{};
    CHECK(reinterpret_cast<std::uintptr_t>(explicitAligned.getLeaf(byteColumn).data()) % 32 == 0);
    CHECK(
        reinterpret_cast<std::uintptr_t>(explicitAligned.getLeaf(overAlignedColumn).data()) % alignof(OverAligned)
        == 0);
}
