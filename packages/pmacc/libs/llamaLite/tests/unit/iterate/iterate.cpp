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
    DEFINE_TAG(TagPosition);
    DEFINE_TAG(TagX);
    DEFINE_TAG(TagY);
    DEFINE_TAG(TagOther);

    // Inner Record: Contains C and D
    using InnerRecord = ll::Record<ll::Field<TagC_t, int>, ll::Field<TagD_t, float>>;

    // Outer Record: Contains A and B (Inner)
    using OuterRecord = ll::Record<ll::Field<TagA_t, int>, ll::Field<TagB_t, InnerRecord>>;

    // Use SoA to generate a valid view type
    using TestSoA = ll::SoA<OuterRecord, 1>;

    using PositionRecord = ll::Record<ll::Field<TagX_t, int>, ll::Field<TagY_t, int>>;
    using CompositeRecord = ll::Record<ll::Field<TagPosition_t, PositionRecord>, ll::Field<TagOther_t, int>>;
    using CompositeSoA = ll::SoA<CompositeRecord, 1>;

    template<ll::IsField F>
    struct CompositeVisitor
    {
        using type = void;
        using _default_sentinel = void;

        constexpr type operator()(auto&&, std::vector<std::string>&) const
        {
        }
    };

    template<>
    struct CompositeVisitor<ll::Field<TagPosition_t, PositionRecord>>
    {
        using type = void;

        constexpr type operator()(auto&& fieldView, std::vector<std::string>& visited) const
        {
            fieldView[TagX] = 10;
            fieldView[TagY] = 20;
            visited.push_back("position");
        }
    };

    template<>
    struct CompositeVisitor<ll::Field<TagX_t, int>>
    {
        using type = void;

        constexpr type operator()(auto&& fieldView, std::vector<std::string>& visited) const
        {
            fieldView = 1;
            visited.push_back("x");
        }
    };

    template<>
    struct CompositeVisitor<ll::Field<TagY_t, int>>
    {
        using type = void;

        constexpr type operator()(auto&& fieldView, std::vector<std::string>& visited) const
        {
            fieldView = 2;
            visited.push_back("y");
        }
    };

    template<typename Path>
    struct CompositePathVisitor
    {
        using type = void;
        using _default_sentinel = void;

        constexpr type operator()(auto&&, std::vector<std::string>&) const
        {
        }
    };

    template<>
    struct CompositePathVisitor<ll::TagPath<TagPosition_t>>
    {
        using type = void;

        constexpr type operator()(auto&& rootView, std::vector<std::string>& visited) const
        {
            rootView[TagPosition][TagX] = 10;
            rootView[TagPosition][TagY] = 20;
            visited.push_back("position");
        }
    };

    template<>
    struct CompositePathVisitor<ll::TagPath<TagPosition_t, TagX_t>>
    {
        using type = void;

        constexpr type operator()(auto&& rootView, std::vector<std::string>& visited) const
        {
            rootView[TagPosition][TagX] = 1;
            visited.push_back("x");
        }
    };

    template<>
    struct CompositePathVisitor<ll::TagPath<TagPosition_t, TagY_t>>
    {
        using type = void;

        constexpr type operator()(auto&& rootView, std::vector<std::string>& visited) const
        {
            rootView[TagPosition][TagY] = 2;
            visited.push_back("y");
        }
    };

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

TEST_CASE("Composite visitors respect selector subtree coverage", "[llamaLite][iteration]")
{
    CompositeSoA soa;
    auto root_view = soa[0];
    std::vector<std::string> visited;
    root_view[TagPosition][TagX] = 0;
    root_view[TagPosition][TagY] = 0;

    SECTION("Field visitor recurses for partial selection")
    {
        llama_lite::iterate_except<CompositeRecord, CompositeVisitor, TagPosition / TagY>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"x"});
        CHECK((root_view[TagPosition][TagX] == 1));
        CHECK((root_view[TagPosition][TagY] == 0));

        visited.clear();
        root_view[TagPosition][TagX] = 0;
        root_view[TagPosition][TagY] = 0;
        llama_lite::iterate_only<CompositeRecord, CompositeVisitor, TagPosition / TagY>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"y"});
        CHECK((root_view[TagPosition][TagX] == 0));
        CHECK((root_view[TagPosition][TagY] == 2));
    }

    SECTION("Field visitor handles fully selected composite")
    {
        llama_lite::iterate_only<CompositeRecord, CompositeVisitor, TagPosition>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"position"});

        visited.clear();
        llama_lite::iterate_except<CompositeRecord, CompositeVisitor, TagOther>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"position"});

        visited.clear();
        llama_lite::iterate<CompositeRecord, llama_lite::selectors::SelectAll, CompositeVisitor>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"position"});

        visited.clear();
        llama_lite::iterate_except<CompositeRecord, CompositeVisitor, TagPosition>(root_view, visited);
        CHECK(visited.empty());
    }

    SECTION("Path visitor respects partial and full selection")
    {
        llama_lite::iterate_path_except<CompositeRecord, CompositePathVisitor, TagPosition / TagY>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"x"});
        CHECK((root_view[TagPosition][TagX] == 1));
        CHECK((root_view[TagPosition][TagY] == 0));

        visited.clear();
        root_view[TagPosition][TagX] = 0;
        root_view[TagPosition][TagY] = 0;
        llama_lite::iterate_path_only<CompositeRecord, CompositePathVisitor, TagPosition / TagY>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"y"});
        CHECK((root_view[TagPosition][TagX] == 0));
        CHECK((root_view[TagPosition][TagY] == 2));

        visited.clear();
        llama_lite::iterate_path_only<CompositeRecord, CompositePathVisitor, TagPosition>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"position"});

        visited.clear();
        llama_lite::iterate_path_except<CompositeRecord, CompositePathVisitor, TagOther>(root_view, visited);
        CHECK(visited == std::vector<std::string>{"position"});

        visited.clear();
        llama_lite::iterate_path<CompositeRecord, llama_lite::selectors::SelectAll, CompositePathVisitor>(
            root_view,
            visited);
        CHECK(visited == std::vector<std::string>{"position"});

        visited.clear();
        llama_lite::iterate_path_except<CompositeRecord, CompositePathVisitor, TagPosition>(root_view, visited);
        CHECK(visited.empty());
    }
}

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
        llama_lite::iterate_except<OuterRecord, TrackingVisitor, ll::append_t<TagB_t, TagC_t>{}>(root_view, visited);

        CHECK(visited == std::vector<std::string>{"TagA", "TagD"});
    }
}
