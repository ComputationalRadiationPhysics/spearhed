// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

namespace
{
    DEFINE_TAG(TagA);
    DEFINE_TAG(TagB);
    DEFINE_TAG(TagC);
    DEFINE_TAG(TagD);

    // Inner Record: Contains C and D
    using InnerRecord = ll::Record<ll::Field<TagC_t, int>, ll::Field<TagD_t, float>>;

    // Outer Record: Contains A and B (Inner)
    using OuterRecord = ll::Record<ll::Field<TagA_t, int>, ll::Field<TagB_t, InnerRecord>>;

    // Use SoA to generate a valid view type
    using TestSoA = ll::SoA<OuterRecord, 1>;

    // A generic visitor that records visited field tags
    template<ll::IsField F>
    struct TrackingVisitor
    {
        using type = void;
        using _default_sentinel = void;

        constexpr type operator()(auto&& fieldView, std::vector<std::string>& visited) const
        {
            using Tag = typename F::tag_type;
            if constexpr(std::is_same_v<Tag, TagA_t>)
                visited.push_back("TagA");
            else if constexpr(std::is_same_v<Tag, TagB_t>)
                visited.push_back("TagB");
            else if constexpr(std::is_same_v<Tag, TagC_t>)
                visited.push_back("TagC");
            else if constexpr(std::is_same_v<Tag, TagD_t>)
                visited.push_back("TagD");
        }
    };

} // namespace

TEST_CASE("LlamaLite Record Iteration Policies", "[llamaLite][iteration]")
{
    TestSoA soa;
    auto root_view = soa[0];
    std::vector<std::string> visited;

    SECTION("SelectAll Policy")
    {
        // Should visit leaf A, recurse into B, visit leaves C and D.
        // B itself is not visited because the visitor is not specialized for it,
        // so the default behavior is to recurse.
        llama_lite::iterate<OuterRecord, llama_lite::selectors::SelectAll, TrackingVisitor>(root_view, visited);

        CHECK(visited == std::vector<std::string>{"TagA", "TagC", "TagD"});
    }

    SECTION("Include Policy: Subtree (TagB)")
    {
        // Path logic:
        // - TagA: Not ancestor or descendant of TagB -> Skip
        // - TagB: Match -> Recurse
        //   - TagC: Descendant of TagB -> Visit
        //   - TagD: Descendant of TagB -> Visit
        llama_lite::iterate_only<OuterRecord, TrackingVisitor, TagB>(root_view, visited);

        CHECK(visited == std::vector<std::string>{"TagC", "TagD"});
    }

    SECTION("Include Policy: Deep Leaf (TagC)")
    {
        // Path logic:
        // - TagB: Ancestor of target (TagB/TagC) -> Enter
        //   - TagC: Match -> Visit
        //   - TagD: Not matched -> Skip
        llama_lite::iterate_only<OuterRecord, TrackingVisitor, TagB / TagC>(root_view, visited);

        CHECK(visited == std::vector<std::string>{"TagC"});
    }

    SECTION("Exclude Policy: Subtree (TagB)")
    {
        // Path logic:
        // - TagA: Safe -> Visit
        // - TagB: Is target -> Skip (prune subtree)
        llama_lite::iterate_except<OuterRecord, TrackingVisitor, TagB>(root_view, visited);

        CHECK(visited == std::vector<std::string>{"TagA"});
    }

    SECTION("Exclude Policy: Deep Leaf (TagC)")
    {
        // Path logic:
        // - TagA: Safe -> Visit
        // - TagB: Not target, check children -> Enter
        //   - TagC: Is target -> Skip
        //   - TagD: Safe -> Visit
        llama_lite::iterate_except<OuterRecord, TrackingVisitor, TagB / TagC>(root_view, visited);

        CHECK(visited == std::vector<std::string>{"TagA", "TagD"});
    }
}
