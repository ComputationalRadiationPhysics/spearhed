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

#include "spmacc/particles/regions/NeighbourBundle.hpp"

#include <type_traits>

#include <catch2/catch_test_macros.hpp>

// Minimal mock species and entries for compile-time tests (no GPU needed).

namespace pmacc::spearhed::test
{
    struct SpeciesA
    {
        using Roles = role_set<roles::Interior, roles::Source>;
        static constexpr auto name = meta::FixedString{"A"};
    };

    struct SpeciesB
    {
        using Roles = role_set<roles::Frozen>;
        static constexpr auto name = meta::FixedString{"B"};
    };

    struct SpeciesC
    {
        using Roles = role_set<roles::Interior, roles::Thermodynamic>;
        static constexpr auto name = meta::FixedString{"C"};
    };

    template<typename S>
    struct MockEntry
    {
        using Species = S;
        int dummy = 0;

        auto deviceView()
        {
            return 0;
        }
    };

    using EntryA = MockEntry<SpeciesA>;
    using EntryB = MockEntry<SpeciesB>;
    using EntryC = MockEntry<SpeciesC>;

    // Custom predicate for use with select<>().
    struct StartsWithAB
    {
        template<typename Entry>
        static constexpr bool value
            = (std::same_as<typename Entry::Species, SpeciesA> || std::same_as<typename Entry::Species, SpeciesB>);
    };
} // namespace pmacc::spearhed::test

// Tests

TEST_CASE("species predicate requires all requested roles", "[region_role]")
{
    using namespace pmacc::spearhed;
    using namespace pmacc::spearhed::test;

    STATIC_REQUIRE(pred::eval<SpeciesA>(pred::withAllRoles<roles::Interior, roles::Source>));
    STATIC_REQUIRE_FALSE(pred::eval<SpeciesA>(pred::withAllRoles<roles::Interior, roles::Thermodynamic>));
    STATIC_REQUIRE(pred::eval<SpeciesB>(pred::withAllRoles<roles::Frozen>));
}

TEST_CASE("NeighbourBundle: concept and size", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    using Bundle = NeighbourBundle<true, test::EntryA, test::EntryB, test::EntryC>;
    STATIC_REQUIRE(Bundle::size() == 3);
    STATIC_REQUIRE(IsNeighbourBundle<Bundle>);

    using View = NeighbourBundle<false, test::EntryA, test::EntryB>;
    STATIC_REQUIRE(View::size() == 2);
    STATIC_REQUIRE(IsNeighbourBundle<View>);

    STATIC_REQUIRE(!IsNeighbourBundle<int>);
}

TEST_CASE("NeighbourBundle: selectByRole", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    test::EntryA a;
    test::EntryB b;
    test::EntryC c;

    NeighbourBundle<true, test::EntryA, test::EntryB, test::EntryC> bundle{std::tuple{a, b, c}};

    // A and C have Interior; B does not.
    auto interiorView = bundle.selectByRole(roles::interior);
    STATIC_REQUIRE(decltype(interiorView)::size() == 2);
    STATIC_REQUIRE(IsNeighbourBundle<decltype(interiorView)>);

    std::size_t count = 0;
    interiorView.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 2);

    // B has Frozen; A and C do not.
    auto frozenView = bundle.selectByRole(roles::frozen);
    STATIC_REQUIRE(decltype(frozenView)::size() == 1);
    count = 0;
    frozenView.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 1);

    // No species has Movable, so the result should be empty.
    // (empty is allowed: NeighbourBundle with 0 entries is still valid for views)
    auto movableView = bundle.selectByRole(roles::movable);
    STATIC_REQUIRE(decltype(movableView)::size() == 0);
}

TEST_CASE("NeighbourBundle: selectBySpecies", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    test::EntryA a;
    test::EntryB b;
    test::EntryC c;

    NeighbourBundle<true, test::EntryA, test::EntryB, test::EntryC> bundle{std::tuple{a, b, c}};

    auto aView = bundle.selectBySpecies(test::SpeciesA{});
    STATIC_REQUIRE(decltype(aView)::size() == 1);
    std::size_t count = 0;
    aView.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 1);
}

TEST_CASE("NeighbourBundle: selectSpecies (multi)", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    test::EntryA a;
    test::EntryB b;
    test::EntryC c;

    NeighbourBundle<true, test::EntryA, test::EntryB, test::EntryC> bundle{std::tuple{a, b, c}};

    auto abView = bundle.selectSpecies(test::SpeciesA{}, test::SpeciesB{});
    STATIC_REQUIRE(decltype(abView)::size() == 2);
    std::size_t count = 0;
    abView.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 2);
}

TEST_CASE("NeighbourBundle: chained select", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    test::EntryA a;
    test::EntryB b;
    test::EntryC c;

    NeighbourBundle<true, test::EntryA, test::EntryB, test::EntryC> bundle{std::tuple{a, b, c}};

    // Chain: role then species
    auto interiorView = bundle.selectByRole(roles::interior); // A + C
    auto aOnly = interiorView.selectBySpecies(test::SpeciesA{}); // A only
    STATIC_REQUIRE(decltype(aOnly)::size() == 1);
    STATIC_REQUIRE(IsNeighbourBundle<decltype(aOnly)>);
    std::size_t count = 0;
    aOnly.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 1);
}

TEST_CASE("NeighbourBundle: select<P>() with custom predicate", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    test::EntryA a;
    test::EntryB b;
    test::EntryC c;

    NeighbourBundle<true, test::EntryA, test::EntryB, test::EntryC> bundle{std::tuple{a, b, c}};

    auto abView = bundle.select(test::StartsWithAB{});
    STATIC_REQUIRE(decltype(abView)::size() == 2);
    std::size_t count = 0;
    abView.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 2);
}

TEST_CASE("NeighbourBundle: deref handles both storage variants", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    test::EntryA a;
    a.dummy = 42;

    // Owning: makes a copy, so addresses differ.  The value is preserved.
    NeighbourBundle<true, test::EntryA> owning{std::tuple{a}};
    owning.forEachEntry(
        [&](auto& entry)
        {
            STATIC_REQUIRE(std::same_as<std::remove_cvref_t<decltype(entry)>, test::EntryA>);
            REQUIRE(&entry != &a); // owning storage is a copy
            REQUIRE(entry.dummy == 42); // value matches
        });

    // Viewing: holds a reference_wrapper, so deref yields the SAME object.
    NeighbourBundle<false, test::EntryA> viewing{std::tuple{std::ref(a)}};
    viewing.forEachEntry(
        [&](auto& entry)
        {
            STATIC_REQUIRE(std::same_as<std::remove_cvref_t<decltype(entry)>, test::EntryA>);
            REQUIRE(&entry == &a); // viewing storage is a reference
        });
}

TEST_CASE("NeighbourBundle: makeNeighbourBundle deduces an owning bundle", "[neighbour_bundle_v2]")
{
    using namespace pmacc::spearhed;

    auto bundle = makeNeighbourBundle(test::EntryA{}, test::EntryB{});
    STATIC_REQUIRE(std::is_same_v<decltype(bundle), NeighbourBundle<true, test::EntryA, test::EntryB>>);
    STATIC_REQUIRE(decltype(bundle)::size() == 2);
    STATIC_REQUIRE(IsNeighbourBundle<decltype(bundle)>);

    std::size_t count = 0;
    bundle.forEachEntry([&](auto&) { ++count; });
    REQUIRE(count == 2);
}
