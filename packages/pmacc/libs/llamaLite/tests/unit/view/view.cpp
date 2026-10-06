// Copyright 2025 Tapish Narwal
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.mozilla.org/MPL/2.0/.

#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/llamaLite.hpp>

DEFINE_TAG(posO);
DEFINE_TAG(velO);
DEFINE_TAG(massO);
DEFINE_TAG(xO);
DEFINE_TAG(yO);
DEFINE_TAG(zO);
DEFINE_TAG(throwingO);

struct ThrowingValue
{
    static inline bool throwOnAssign = false;
    int value = 0;

    ThrowingValue& operator=(ThrowingValue const& other)
    {
        if(throwOnAssign)
            throw std::runtime_error("intentional assignment failure");
        value = other.value;
        return *this;
    }
};

using Pos3O = ll::Record<ll::Field<xO_t, float>, ll::Field<yO_t, float>, ll::Field<zO_t, float>>;
using ParticleOne = ll::Record<ll::Field<posO_t, Pos3O>, ll::Field<velO_t, Pos3O>, ll::Field<massO_t, double>>;
using EmptyRecord = ll::Record<>;
using ParticleOneView = ll::ViewIndexed<ll::One<ParticleOne>>;
using ConstParticleOneView = ll::ViewIndexed<ll::One<ParticleOne> const>;
using ParticleOneRootView = ll::View<ll::One<ParticleOne>>;
using ConstParticleOneRootView = ll::View<ll::One<ParticleOne> const>;

namespace
{
    template<typename T>
    concept RvalueAssignable = requires(T&& lhs, T const& rhs) { static_cast<T&&>(lhs) = rhs; };

    template<typename T>
    concept RvalueSourceAssignable = requires(T&& lhs, T&& rhs) { static_cast<T&&>(lhs) = static_cast<T&&>(rhs); };

    template<typename Dest, typename Src>
    concept CanCopyValues = requires(Dest dest, Src src) { ll::copy_values(dest, src); };

    template<typename Dest, typename Model>
    concept CanSelectLike = requires(Dest dest, Model model) { ll::select_like(dest, model); };

    template<typename Dest, typename Src>
    concept LvalueAssignableFrom = requires(Dest& dest, Src src) { dest = src; };

    template<typename Dest, typename Src>
    concept RvalueAssignableFrom
        = requires(Dest&& dest, Src&& src) { static_cast<Dest&&>(dest) = static_cast<Src&&>(src); };

    using MassLeaves = ll::Set<ll::TagPath<massO_t>>;
    using PositionLeaves = ll::Set<ll::TagPath<posO_t, xO_t>, ll::TagPath<posO_t, yO_t>, ll::TagPath<posO_t, zO_t>>;
    using OverlappingView = decltype(std::declval<ll::One<ParticleOne>&>().select(posO, posO / xO));
    using EmptyRootView = ll::View<ll::One<EmptyRecord>>;
    using EmptySelectionView = ll::View<ll::One<ParticleOne>, ll::Set<>>;
    using EmptySelectionIndexedView = ll::ViewIndexed<ll::One<ParticleOne>, ll::Set<>>;
    using RootNonIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().view());
    using MassNonIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(massO));
    using RootIndexedView = decltype(std::declval<ll::One<ParticleOne>&>()[uint32_t{0}]);
    using MassIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(massO)[uint32_t{0}]);
    using PositionNonIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(posO));
    using PositionCursorNonIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().view(posO));
    using PositionXNonIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(posO / xO));
    using PositionIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(posO)[uint32_t{0}]);
    using PositionCursorIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().view(posO)[uint32_t{0}]);
    using PositionXSelectedCursor = decltype(std::declval<ll::One<ParticleOne>&>().view(posO)[uint32_t{0}].select(xO));
    using PositionXIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(posO / xO)[uint32_t{0}]);
    using VelIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().select(velO)[uint32_t{0}]);
    using ConstMassIndexedView = decltype(std::declval<ll::One<ParticleOne> const&>().view(massO)[uint32_t{0}]);
    using MassCursorIndexedView = decltype(std::declval<ll::One<ParticleOne>&>().view(massO)[uint32_t{0}]);
    using SingleFieldRecord = ll::Record<ll::Field<massO_t, double>>;
    using SingleFieldRootView = decltype(std::declval<ll::One<SingleFieldRecord>&>()[uint32_t{0}]);
    using OneRootView = ll::View<ll::One<ParticleOne>>;
    using OneConstRootView = ll::View<ll::One<ParticleOne> const>;
    using SoaRootView = ll::View<ll::SoA<ParticleOne, 2>>;
    using SoaConstRootView = ll::View<ll::SoA<ParticleOne, 2> const>;
    using DynSoaRootView = ll::View<ll::DynSoA<ParticleOne>>;
    using DynSoaConstRootView = ll::View<ll::DynSoA<ParticleOne> const>;
    using OneRootIndexedView = ll::ViewIndexed<ll::One<ParticleOne>>;
    using OneConstRootIndexedView = ll::ViewIndexed<ll::One<ParticleOne> const>;
    using SoaRootIndexedView = ll::ViewIndexed<ll::SoA<ParticleOne, 2>>;
    using SoaConstRootIndexedView = ll::ViewIndexed<ll::SoA<ParticleOne, 2> const>;
    using DynSoaRootIndexedView = ll::ViewIndexed<ll::DynSoA<ParticleOne>>;
    using DynSoaConstRootIndexedView = ll::ViewIndexed<ll::DynSoA<ParticleOne> const>;

    template<typename T>
    concept HasMassAccess = requires(T view) { view[massO]; };

    template<typename T>
    concept HasVelocityAccess = requires(T view) { view[velO]; };

    template<typename T>
    concept HasPositionXAccess = requires(T view) { view[posO / xO]; };

    template<typename T>
    concept HasUnknownAccess = requires(T view) { view[throwingO]; };

    template<typename T>
    concept HasTwoTagView = requires(T& owner) { owner.view(posO, velO); };

    template<typename T>
    concept OwnerCanSelectUnknownAccess = requires(T& owner) { owner.select(throwingO); };

    template<typename T>
    concept OwnerCanViewUnknownAccess = requires(T& owner) { owner.view(throwingO); };

    template<typename T>
    concept OwnerCanSelectMixedAccesses = requires(T& owner) { owner.select(posO, throwingO); };

    template<typename T>
    concept OwnerCanViewUnknownNestedAccess = requires(T& owner) { owner.view(posO / throwingO); };

    template<typename T>
    concept CanNavigateSelectedX = requires(T view) { view[posO][xO]; };

    template<typename T>
    concept CanNavigateUnselectedY = requires(T view) { view[posO][yO]; };

    template<typename T>
    concept HasGet = requires(T view) { view.get(); };

    template<typename T>
    concept HasDereference = requires(T view) { *view; };

    template<typename T, typename... RAs>
    concept CanSelect = requires(T view, RAs... accesses) { view.select(accesses...); };

    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ParticleOneView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ConstParticleOneView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ParticleOneRootView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(std::declval<ConstParticleOneRootView const&>())),
                  ll::record_leaf_set_t<ParticleOne>>);
    static_assert(std::is_same_v<
                  decltype(ll::getSelectedLeaves(
                      std::declval<ll::ViewIndexed<ll::One<ParticleOne>, ll::access_set_t<massO_t>> const&>())),
                  MassLeaves>);
    static_assert(
        std::is_same_v<
            decltype(ll::getSelectedLeaves(std::declval<ll::View<ll::One<ParticleOne>, PositionLeaves> const&>())),
            PositionLeaves>);
    static_assert(
        std::is_same_v<decltype(ll::getSelectedLeaves(std::declval<OverlappingView const&>())), PositionLeaves>);
    static_assert(std::is_same_v<decltype(ll::getSelectedLeaves(std::declval<EmptyRootView const&>())), ll::Set<>>);
    static_assert(
        std::is_same_v<decltype(ll::getSelectedLeaves(std::declval<EmptySelectionView const&>())), ll::Set<>>);
    static_assert(
        std::is_same_v<decltype(ll::getSelectedLeaves(std::declval<EmptySelectionIndexedView const&>())), ll::Set<>>);
    static_assert(std::is_convertible_v<RootNonIndexedView, MassNonIndexedView>);
    static_assert(!std::is_convertible_v<MassNonIndexedView, RootNonIndexedView>);
    static_assert(std::is_convertible_v<MassNonIndexedView, EmptySelectionView>);
    static_assert(!std::is_convertible_v<EmptySelectionView, MassNonIndexedView>);
    static_assert(std::is_convertible_v<RootIndexedView, MassIndexedView>);
    static_assert(!std::is_convertible_v<MassIndexedView, RootIndexedView>);
    static_assert(std::is_convertible_v<MassIndexedView, EmptySelectionIndexedView>);
    static_assert(!std::is_convertible_v<EmptySelectionIndexedView, MassIndexedView>);
    static_assert(!std::is_convertible_v<EmptySelectionView, RootNonIndexedView>);
    static_assert(!std::is_convertible_v<EmptySelectionIndexedView, RootIndexedView>);
    static_assert(!std::is_convertible_v<PositionIndexedView, VelIndexedView>);
    static_assert(std::is_convertible_v<PositionIndexedView, PositionXIndexedView>);
    static_assert(!std::is_convertible_v<PositionNonIndexedView, MassNonIndexedView>);
    static_assert(std::is_convertible_v<PositionNonIndexedView, PositionXNonIndexedView>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(*std::declval<ConstMassIndexedView>())>>);
    static_assert(std::is_same_v<decltype(std::declval<MassIndexedView const&>()[massO]), double&>);
    static_assert(std::is_same_v<decltype(*std::declval<MassCursorIndexedView const&>()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<MassCursorIndexedView const&>().get()), double&>);
    static_assert(std::is_same_v<decltype(*std::declval<ConstMassIndexedView const&>()), double const&>);
    static_assert(std::is_same_v<decltype(std::declval<ConstMassIndexedView const&>().get()), double const&>);
    static_assert(std::is_same_v<decltype(std::declval<SingleFieldRootView>()[massO]), double&>);
    using MultiFieldView = decltype(std::declval<ll::One<ParticleOne>&>().select(massO, velO));
    static_assert(HasMassAccess<OneRootView>);
    static_assert(HasMassAccess<MultiFieldView>);
    static_assert(!HasPositionXAccess<MultiFieldView>);
    static_assert(HasPositionXAccess<OneConstRootView>);
    static_assert(requires(PositionNonIndexedView const& view) { view[posO / xO]; });
    static_assert(
        std::is_const_v<typename decltype(std::declval<OneConstRootView const&>()[posO / xO])::element_type>);
    static_assert(!HasMassAccess<EmptySelectionView>);
    static_assert(!HasMassAccess<EmptySelectionIndexedView>);
    static_assert(!HasVelocityAccess<MassNonIndexedView>);
    static_assert(!HasVelocityAccess<MassIndexedView>);
    static_assert(!HasMassAccess<EmptyRootView>);
    static_assert(!HasUnknownAccess<RootNonIndexedView>);
    static_assert(!HasUnknownAccess<RootIndexedView>);
    static_assert(!CanSelect<RootNonIndexedView, decltype(throwingO)>);
    static_assert(!CanSelect<RootIndexedView, decltype(throwingO)>);
    static_assert(!HasTwoTagView<ll::One<ParticleOne>>);
    static_assert(!HasTwoTagView<ll::One<ParticleOne> const>);
    static_assert(!HasTwoTagView<ll::SoA<ParticleOne, 2>>);
    static_assert(!HasTwoTagView<ll::SoA<ParticleOne, 2> const>);
    static_assert(!HasTwoTagView<ll::DynSoA<ParticleOne>>);
    static_assert(!HasTwoTagView<ll::DynSoA<ParticleOne> const>);
    static_assert(!OwnerCanSelectUnknownAccess<ll::One<ParticleOne>>);
    static_assert(!OwnerCanSelectUnknownAccess<ll::One<ParticleOne> const>);
    static_assert(!OwnerCanSelectUnknownAccess<ll::SoA<ParticleOne, 2>>);
    static_assert(!OwnerCanSelectUnknownAccess<ll::SoA<ParticleOne, 2> const>);
    static_assert(!OwnerCanSelectUnknownAccess<ll::DynSoA<ParticleOne>>);
    static_assert(!OwnerCanSelectUnknownAccess<ll::DynSoA<ParticleOne> const>);
    static_assert(!OwnerCanViewUnknownAccess<ll::One<ParticleOne>>);
    static_assert(!OwnerCanViewUnknownAccess<ll::One<ParticleOne> const>);
    static_assert(!OwnerCanViewUnknownAccess<ll::SoA<ParticleOne, 2>>);
    static_assert(!OwnerCanViewUnknownAccess<ll::SoA<ParticleOne, 2> const>);
    static_assert(!OwnerCanViewUnknownAccess<ll::DynSoA<ParticleOne>>);
    static_assert(!OwnerCanViewUnknownAccess<ll::DynSoA<ParticleOne> const>);
    static_assert(!OwnerCanSelectMixedAccesses<ll::One<ParticleOne>>);
    static_assert(!OwnerCanSelectMixedAccesses<ll::One<ParticleOne> const>);
    static_assert(!OwnerCanSelectMixedAccesses<ll::SoA<ParticleOne, 2>>);
    static_assert(!OwnerCanSelectMixedAccesses<ll::SoA<ParticleOne, 2> const>);
    static_assert(!OwnerCanSelectMixedAccesses<ll::DynSoA<ParticleOne>>);
    static_assert(!OwnerCanSelectMixedAccesses<ll::DynSoA<ParticleOne> const>);
    static_assert(!OwnerCanViewUnknownNestedAccess<ll::One<ParticleOne>>);
    static_assert(!OwnerCanViewUnknownNestedAccess<ll::One<ParticleOne> const>);
    static_assert(!OwnerCanViewUnknownNestedAccess<ll::SoA<ParticleOne, 2>>);
    static_assert(!OwnerCanViewUnknownNestedAccess<ll::SoA<ParticleOne, 2> const>);
    static_assert(!OwnerCanViewUnknownNestedAccess<ll::DynSoA<ParticleOne>>);
    static_assert(!OwnerCanViewUnknownNestedAccess<ll::DynSoA<ParticleOne> const>);
    using PartialPositionProjection = decltype(std::declval<ll::One<ParticleOne>&>().select(posO / xO)[0u]);
    using FullPositionProjection = decltype(std::declval<ll::One<ParticleOne>&>().select(posO)[0u]);
    static_assert(!std::is_convertible_v<PartialPositionProjection, FullPositionProjection>);
    static_assert(!std::is_convertible_v<PositionCursorIndexedView, PositionIndexedView>);
    static_assert(!std::is_convertible_v<PositionIndexedView, PositionCursorIndexedView>);
    static_assert(!std::is_convertible_v<PositionCursorNonIndexedView, PositionNonIndexedView>);
    static_assert(!std::is_convertible_v<PositionNonIndexedView, PositionCursorNonIndexedView>);
    static_assert(std::is_same_v<typename PositionXSelectedCursor::root_access, ll::TagPath<posO_t>>);
    static_assert(!HasDereference<MassIndexedView>);
    static_assert(HasDereference<ConstMassIndexedView>);
    static_assert(!CanSelect<MassNonIndexedView, decltype(posO)>);
    static_assert(!CanSelect<MassIndexedView, decltype(posO)>);
    static_assert(!CanSelect<PositionXNonIndexedView, decltype(posO)>);
    static_assert(!CanSelect<PositionXIndexedView, decltype(posO)>);
    using PositionXCursorNonIndexed = decltype(std::declval<ll::One<ParticleOne>&>().view(posO).select(xO));
    using PositionXCursorIndexed = decltype(std::declval<ll::One<ParticleOne>&>().view(posO)[0u].select(xO));
    static_assert(CanSelect<PositionXCursorNonIndexed, decltype(xO)>);
    static_assert(!CanSelect<PositionXCursorNonIndexed, decltype(xO), decltype(yO)>);
    static_assert(!CanSelect<PositionXCursorIndexed, decltype(xO), decltype(yO)>);
    static_assert(CanSelect<RootNonIndexedView>);
    static_assert(CanSelect<RootIndexedView>);
    static_assert(std::is_same_v<decltype(std::declval<OneRootView const&>().getRecordAccess()), ll::TagPath<>>);
    static_assert(std::is_same_v<decltype(std::declval<RootIndexedView const&>().getRecordAccess()), ll::TagPath<>>);
    static_assert(std::is_same_v<decltype(std::declval<ll::One<ParticleOne>&>().view()), OneRootView>);
    static_assert(std::is_same_v<
                  ll::One<ParticleOne>::view_type<posO_t>,
                  decltype(std::declval<ll::One<ParticleOne>&>().view(posO))>);
    static_assert(std::is_same_v<
                  ll::One<ParticleOne>::indexed_view_type<posO_t>,
                  decltype(std::declval<ll::One<ParticleOne>&>().view(posO)[uint32_t{0}])>);
    static_assert(std::is_same_v<
                  ll::One<ParticleOne>::selection_view_type<posO_t, massO_t>,
                  decltype(std::declval<ll::One<ParticleOne>&>().select(posO, massO))>);
    static_assert(std::is_same_v<
                  ll::One<ParticleOne>::indexed_selection_view_type<posO_t, massO_t>,
                  decltype(std::declval<ll::One<ParticleOne>&>().select(posO, massO)[uint32_t{0}])>);
    static_assert(std::is_same_v<decltype(std::declval<ll::One<ParticleOne> const&>().view()), OneConstRootView>);
    static_assert(std::is_same_v<decltype(std::declval<ll::SoA<ParticleOne, 2>&>().view()), SoaRootView>);
    static_assert(std::is_same_v<decltype(std::declval<ll::SoA<ParticleOne, 2> const&>().view()), SoaConstRootView>);
    static_assert(std::is_same_v<decltype(std::declval<ll::DynSoA<ParticleOne>&>().view()), DynSoaRootView>);
    static_assert(
        std::is_same_v<decltype(std::declval<ll::DynSoA<ParticleOne> const&>().view()), DynSoaConstRootView>);
    static_assert(std::is_same_v<decltype(std::declval<ll::One<ParticleOne>&>()[uint32_t{0}]), OneRootIndexedView>);
    static_assert(
        std::is_same_v<decltype(std::declval<ll::One<ParticleOne> const&>()[uint32_t{0}]), OneConstRootIndexedView>);
    static_assert(std::is_same_v<decltype(std::declval<ll::SoA<ParticleOne, 2>&>()[uint32_t{0}]), SoaRootIndexedView>);
    static_assert(std::is_same_v<
                  decltype(std::declval<ll::SoA<ParticleOne, 2> const&>()[uint32_t{0}]),
                  SoaConstRootIndexedView>);
    static_assert(
        std::is_same_v<decltype(std::declval<ll::DynSoA<ParticleOne>&>()[uint32_t{0}]), DynSoaRootIndexedView>);
    static_assert(std::is_same_v<
                  decltype(std::declval<ll::DynSoA<ParticleOne> const&>()[uint32_t{0}]),
                  DynSoaConstRootIndexedView>);

    template<typename Owner>
    consteval bool canonicalFactories()
    {
        using Root = decltype(std::declval<Owner&>().view());
        using Position = decltype(std::declval<Owner&>().view(posO));
        using Projection = decltype(std::declval<Owner&>().select(posO));
        static_assert(std::is_same_v<Position, decltype(std::declval<Root>().view(posO))>);
        static_assert(std::is_same_v<Projection, decltype(std::declval<Root>().select(posO))>);
        static_assert(std::is_same_v<Projection, decltype(std::declval<Owner&>().select(posO, posO / xO))>);
        static_assert(
            std::is_same_v<Projection, decltype(std::declval<Owner&>().select(posO / zO, posO / xO, posO / yO))>);
        static_assert(std::is_same_v<Position, decltype(std::declval<Projection>().view(posO))>);
        static_assert(std::is_same_v<Root, decltype(std::declval<Root>().view(ll::TagPath<>{}))>);
        static_assert(
            std::is_same_v<decltype(std::declval<Position>()[0u]), decltype(std::declval<Root>()[0u].view(posO))>);
        static_assert(
            std::is_same_v<decltype(std::declval<Projection>()[0u]), decltype(std::declval<Root>()[0u].select(posO))>);
        static_assert(std::is_same_v<
                      decltype(std::declval<Position>().view(xO)),
                      decltype(std::declval<Root>().view(posO / xO))>);
        static_assert(std::is_same_v<typename Position::access_set, PositionLeaves>);
        static_assert(std::is_same_v<typename Position::root_access, ll::TagPath<posO_t>>);
        return true;
    }

    static_assert(canonicalFactories<ll::One<ParticleOne>>());
    static_assert(canonicalFactories<ll::One<ParticleOne> const>());
    static_assert(canonicalFactories<ll::SoA<ParticleOne, 2>>());
    static_assert(canonicalFactories<ll::SoA<ParticleOne, 2> const>());
    static_assert(canonicalFactories<ll::DynSoA<ParticleOne>>());
    static_assert(canonicalFactories<ll::DynSoA<ParticleOne> const>());

    template<typename Owner>
    consteval bool canonicalAliases()
    {
        static_assert(std::is_same_v<typename Owner::template view_type<>, decltype(std::declval<Owner&>().view())>);
        static_assert(
            std::is_same_v<typename Owner::template indexed_view_type<>, decltype(std::declval<Owner&>()[0u])>);
        static_assert(
            std::is_same_v<typename Owner::template selection_view_type<>, decltype(std::declval<Owner&>().select())>);
        static_assert(std::is_same_v<
                      typename Owner::template indexed_selection_view_type<>,
                      decltype(std::declval<Owner&>().select()[0u])>);
        return true;
    }

    static_assert(canonicalAliases<ll::One<ParticleOne>>());
    static_assert(canonicalAliases<ll::SoA<ParticleOne, 2>>());
    static_assert(canonicalAliases<ll::DynSoA<ParticleOne>>());
    static_assert(canonicalAliases<ll::One<EmptyRecord>>());
    static_assert(canonicalAliases<ll::SoA<EmptyRecord, 2>>());
    static_assert(canonicalAliases<ll::DynSoA<EmptyRecord>>());
    static_assert(
        std::is_same_v<decltype(std::declval<PositionCursorNonIndexedView>().view()), PositionCursorNonIndexedView>);
    static_assert(
        std::is_same_v<decltype(std::declval<PositionCursorIndexedView>().view()), PositionCursorIndexedView>);

    // Reject noncanonical permissions and selections outside the cursor's subtree.
    template<typename Leaves, typename Root = ll::TagPath<>>
    concept ValidView = requires { typename ll::View<ll::One<ParticleOne>, Leaves, Root>; };
    static_assert(ValidView<PositionLeaves, ll::TagPath<posO_t>>);
    static_assert(!ValidView<ll::access_set_t<posO_t>>);
    static_assert(!ValidView<MassLeaves, ll::TagPath<posO_t>>);
    static_assert(!ValidView<MassLeaves, ll::TagPath<throwingO_t>>);
    static_assert(ValidView<ll::Set<>, ll::TagPath<posO_t>>);
    static_assert(!HasGet<RootIndexedView>);
    static_assert(!HasDereference<RootIndexedView>);
    static_assert(!HasGet<EmptySelectionIndexedView>);
    static_assert(!HasGet<PositionXSelectedCursor>);
    static_assert(!HasGet<PositionCursorIndexedView>); // No AsType for this record.
    static_assert(!CanSelect<PositionXSelectedCursor, decltype(yO)>);

    struct RowHandleLayout
    {
        ll::One<ParticleOne>* storage;
        uint32_t idx;
    };

    static_assert(sizeof(OneRootView) == sizeof(ll::One<ParticleOne>*));
    static_assert(sizeof(PositionCursorNonIndexedView) == sizeof(OneRootView));
    static_assert(sizeof(OneRootIndexedView) == sizeof(RowHandleLayout));
    static_assert(alignof(OneRootIndexedView) == alignof(RowHandleLayout));
    static_assert(sizeof(PositionXSelectedCursor) == sizeof(OneRootIndexedView));
    static_assert(std::is_trivially_copyable_v<OneRootView>);
    static_assert(std::is_trivially_copyable_v<OneRootIndexedView>);
    static_assert(std::is_standard_layout_v<OneRootView>);
    static_assert(std::is_standard_layout_v<OneRootIndexedView>);

    static_assert(std::is_same_v<decltype(ll::View{std::declval<ll::One<ParticleOne>&>()}), OneRootView>);
    static_assert(std::is_same_v<decltype(ll::View{std::declval<ll::SoA<ParticleOne, 2>&>()}), SoaRootView>);
    static_assert(std::is_same_v<decltype(ll::View{std::declval<ll::DynSoA<ParticleOne>&>()}), DynSoaRootView>);
    static_assert(std::is_same_v<
                  decltype(ll::ViewIndexed{std::declval<ll::One<ParticleOne>&>(), uint32_t{0}}),
                  OneRootIndexedView>);
    static_assert(std::is_same_v<
                  decltype(ll::ViewIndexed{std::declval<ll::SoA<ParticleOne, 2>&>(), uint32_t{0}}),
                  SoaRootIndexedView>);
    static_assert(std::is_same_v<
                  decltype(ll::ViewIndexed{std::declval<ll::DynSoA<ParticleOne>&>(), uint32_t{0}}),
                  DynSoaRootIndexedView>);
} // namespace

TEST_CASE("getSelectedLeaves reports canonical view selections", "[View]")
{
    using RootLeaves = decltype(ll::getSelectedLeaves(std::declval<ParticleOneRootView const&>()));
    using OverlapLeaves = decltype(ll::getSelectedLeaves(std::declval<OverlappingView const&>()));
    using EmptyLeaves = decltype(ll::getSelectedLeaves(std::declval<EmptyRootView const&>()));
    STATIC_CHECK(RootLeaves::size == 7);
    STATIC_CHECK(OverlapLeaves::size == 3);
    STATIC_CHECK(EmptyLeaves::size == 0);
}

TEST_CASE("Projection preserves the particle root for singleton and multi-field selections", "[View]")
{
    ll::One<ParticleOne> particles{};
    particles[0u][posO][xO] = 2.0f;
    particles[0u][massO] = 3.0;

    auto positionOnly = particles.select(posO);
    auto positionAndMass = particles.select(posO, massO);
    STATIC_CHECK(std::is_same_v<typename decltype(positionOnly)::root_access, ll::TagPath<>>);
    CHECK(positionOnly[0u][posO][xO] == Catch::Approx(2.0f));
    CHECK(positionAndMass[0u][posO][xO] == Catch::Approx(2.0f));
    CHECK(positionAndMass[0u][massO] == Catch::Approx(3.0));

    auto relativePosition = particles.select(posO, massO).view(posO);
    CHECK(relativePosition[0u][xO] == Catch::Approx(2.0f));
    auto relativeProjection = particles.view(posO).select(xO);
    STATIC_CHECK(std::is_same_v<typename decltype(relativeProjection)::root_access, ll::TagPath<posO_t>>);
    CHECK(relativeProjection[0u][xO] == Catch::Approx(2.0f));
    auto emptySelection = particles.view().select();
    STATIC_CHECK(decltype(ll::getSelectedLeaves(emptySelection))::size == 0);
    auto emptyIndexedSelection = particles.view()[0u].select();
    STATIC_CHECK(decltype(ll::getSelectedLeaves(emptyIndexedSelection))::size == 0);
    auto relativeX = relativePosition[0u].view(xO);
    CHECK(relativeX.get() == Catch::Approx(2.0f));
}

TEST_CASE("Subtree cursor access stays scoped for selection queries and copies", "[View]")
{
    ll::One<ParticleOne> source{};
    ll::One<ParticleOne> destination{};
    source[0u][posO][xO] = 11.0f;
    source[0u][massO] = 101.0;
    destination[0u][posO][xO] = -1.0f;
    destination[0u][massO] = -2.0;

    auto sourcePosition = source[0u][posO];
    auto destinationPosition = destination[0u][posO];
    STATIC_CHECK(std::is_same_v<decltype(ll::getSelectedLeaves(sourcePosition)), PositionLeaves>);

    ll::copy_values(destinationPosition, sourcePosition);
    CHECK(destination[0u][posO][xO] == Catch::Approx(11.0f));
    CHECK(destination[0u][massO] == Catch::Approx(-2.0));
}

TEST_CASE("Projection permissions are retained through relative cursor navigation", "[View]")
{
    ll::One<ParticleOne> particles{};
    auto partial = particles.select(posO / xO)[0u];
    auto narrowedAgain = particles.view().select(posO / xO).select(posO / xO);
    STATIC_CHECK(std::is_same_v<decltype(ll::getSelectedLeaves(narrowedAgain)), ll::Set<ll::TagPath<posO_t, xO_t>>>);
    auto indexedAgain = particles.view()[0u].select(posO / xO).select(posO / xO);
    STATIC_CHECK(std::is_same_v<decltype(ll::getSelectedLeaves(indexedAgain)), ll::Set<ll::TagPath<posO_t, xO_t>>>);
    using PositionXCursor = decltype(particles.view(posO).select(xO));
    static_assert(!CanSelect<PositionXCursor, decltype(yO)>);
    STATIC_CHECK(CanNavigateSelectedX<decltype(partial)>);
    STATIC_CHECK_FALSE(CanNavigateUnselectedY<decltype(partial)>);
    partial[posO][xO] = 9.0f;
    CHECK(particles[0u][posO][xO] == Catch::Approx(9.0f));
}

TEST_CASE("Root and multi-field views support tag indexing", "[View]")
{
    ll::One<ParticleOne> particles{};
    auto root = particles.view();
    root[massO][0] = 2.5;
    CHECK(particles[0u][massO] == Catch::Approx(2.5));

    auto multi = particles.select(massO, velO);
    multi[massO][0] = 4.5;
    CHECK(particles[0u][massO] == Catch::Approx(4.5));

    auto const position = particles.view(posO);
    position[xO][0] = 6.5f;
    CHECK(particles[0u][posO][xO] == Catch::Approx(6.5f));

    ll::One<ParticleOne> const& constParticles = particles;
    auto constView = constParticles.view();
    auto constPosition = constView[posO / xO];
    STATIC_CHECK(std::is_const_v<typename decltype(constPosition)::element_type>);
    CHECK(constPosition[0] == particles[0u][posO][xO]);
}

TEST_CASE("Root cursors support singleton and empty records", "[View]")
{
    ll::One<SingleFieldRecord> single{};
    auto root = single[0u];
    root[massO] = 12.5;
    CHECK(single[0u][massO] == Catch::Approx(12.5));

    ll::One<EmptyRecord> empty{};
    auto emptyRoot = empty[0u];
    STATIC_CHECK(std::is_same_v<decltype(ll::getSelectedLeaves(emptyRoot)), ll::Set<>>);
}

TEST_CASE("View narrowing preserves storage and empty copies select no fields", "[View]")
{
    ll::One<ParticleOne> source{};
    ll::One<ParticleOne> destination{};
    source[0u][massO] = 42.0;
    destination[0u][massO] = -1.0;
    destination[0u][posO][xO] = -3.0f;

    MassIndexedView narrowedIndexed = source[0u];
    CHECK(narrowedIndexed.storage == &source);
    CHECK(narrowedIndexed.idx == 0u);
    CHECK(narrowedIndexed[massO] == Catch::Approx(42.0));

    MassNonIndexedView narrowed = source.view();
    CHECK(narrowed.storage == &source);

    EmptySelectionIndexedView emptyDestination = destination[0u];
    ll::copy_values(emptyDestination, source[0u]);
    CHECK(destination[0u][massO] == Catch::Approx(-1.0));
    CHECK(destination[0u][posO][xO] == Catch::Approx(-3.0f));
}

TEST_CASE("SoA getLeaf accepts tags and preserves const access", "[SoA]")
{
    ll::SoA<ParticleOne, 2> soa{};
    auto xSpan = soa.getLeaf(posO / xO);
    auto massSpan = soa.getLeaf(massO);
    auto massTagSpan = soa[massO];
    STATIC_CHECK(std::is_same_v<decltype(massTagSpan), std::span<double, 2>>);
    xSpan[0] = 3.0f;
    massSpan[0] = 42.0;

    ll::SoA<ParticleOne, 2> const& constSoa = soa;
    auto constXSpan = constSoa.getLeaf(posO / xO);
    auto constMassSpan = constSoa.getLeaf(massO);
    auto constMassTagSpan = constSoa[massO];
    STATIC_CHECK(std::is_same_v<decltype(constMassTagSpan), std::span<double const, 2>>);
    STATIC_CHECK(std::is_same_v<decltype(xSpan), std::span<float, 2>>);
    STATIC_CHECK(std::is_same_v<decltype(massSpan), std::span<double, 2>>);
    STATIC_CHECK(std::is_same_v<decltype(constXSpan), std::span<float const, 2>>);
    STATIC_CHECK(std::is_same_v<decltype(constMassSpan), std::span<double const, 2>>);
    STATIC_CHECK(std::is_const_v<typename decltype(constXSpan)::element_type>);
    STATIC_CHECK(std::is_const_v<typename decltype(constMassSpan)::element_type>);
    CHECK(constXSpan[0] == Catch::Approx(3.0f));
    CHECK(constMassSpan[0] == Catch::Approx(42.0));

    auto constMassProjection = constSoa.select(massO);
    auto constSelectedMass = constMassProjection[massO];
    STATIC_CHECK(std::is_same_v<decltype(constSelectedMass), std::span<double const, 2>>);
}

TEST_CASE("View assignment rebinds named handles and rejects temporary destinations", "[View]")
{
    ll::One<ParticleOne> one{};
    ll::SoA<ParticleOne, 2> soa{};
    soa[0u][massO] = 10.0;
    soa[1u][massO] = 20.0;
    auto first = soa[0u];
    auto second = soa[1u];

    first = second;
    first[massO] = 12.0;
    auto alias = first;
    alias[massO] = 13.0;

    CHECK(soa[0u][massO] == Catch::Approx(10.0));
    CHECK(soa[1u][massO] == Catch::Approx(13.0));
    STATIC_CHECK_FALSE(RvalueAssignable<decltype(first)>);
    STATIC_CHECK_FALSE(RvalueSourceAssignable<decltype(first)>);
    STATIC_CHECK_FALSE(LvalueAssignableFrom<decltype(first), decltype(one[0u])>);
    STATIC_CHECK_FALSE(RvalueAssignableFrom<decltype(first), decltype(one[0u])>);

    ll::SoA<ParticleOne, 2> other{};
    auto root = soa.view();
    auto otherRoot = other.view();
    root = otherRoot;
    root[0u][massO] = 27.0;
    CHECK(other[0u][massO] == Catch::Approx(27.0));
    CHECK(soa[0u][massO] == Catch::Approx(10.0));
    STATIC_CHECK_FALSE(RvalueAssignable<decltype(root)>);
    STATIC_CHECK_FALSE(RvalueSourceAssignable<decltype(root)>);
    STATIC_CHECK_FALSE(RvalueAssignableFrom<decltype(root), decltype(one.view())>);
}

TEST_CASE("copy_values copies exactly the destination selection", "[View]")
{
    ll::One<ParticleOne> one{};
    ll::One<ParticleOne> src{};
    src[0u][posO][xO] = 3.0f;
    src[0u][massO] = 42.0;
    one[0u][posO][xO] = -1.0f;
    one[0u][massO] = -2.0;

    auto dstMass = one.view(massO)[0u];
    auto srcMass = src.view(massO)[0u];
    ll::copy_values(dstMass, srcMass);

    CHECK(one[0u][massO] == Catch::Approx(42.0));
    CHECK(one[0u][posO][xO] == Catch::Approx(-1.0f));

    src[0u][massO] = 99.0;
    auto const& constSrc = src;
    ll::copy_values(one.view(massO)[0u], constSrc.view(massO)[0u]);
    CHECK(one[0u][massO] == Catch::Approx(99.0));
    ll::copy_values(ll::select_like(one[0u], srcMass), srcMass);
    CHECK(one[0u][posO][xO] == Catch::Approx(-1.0f));

    using MassView = decltype(dstMass);
    using PositionView = decltype(one.view(posO)[0u]);
    using RootView = decltype(one[0u]);
    STATIC_CHECK(CanCopyValues<MassView, decltype(src.view(massO)[0u])>);
    STATIC_CHECK_FALSE(CanCopyValues<MassView, PositionView>);
    STATIC_CHECK_FALSE(CanCopyValues<RootView, MassView>);
    STATIC_CHECK(CanSelectLike<RootView, MassView>);
    STATIC_CHECK_FALSE(CanSelectLike<MassView, PositionView>);
    using ConstRootView = ll::ViewIndexed<ll::One<ParticleOne> const>;
    STATIC_CHECK(CanSelectLike<ConstRootView, MassView>);
    STATIC_CHECK_FALSE(CanCopyValues<ConstRootView, MassView>);

    using EmptyView = decltype(std::declval<ll::One<EmptyRecord>&>()[0u]);
    STATIC_CHECK_FALSE(CanSelectLike<RootView, EmptyView>);
}

TEST_CASE("copy_values copies nested multi-field selections and preserves other leaves", "[View]")
{
    ll::One<ParticleOne> dest{};
    ll::One<ParticleOne> src{};
    src[0u][posO][xO] = 8.5f;
    src[0u][velO][yO] = 6.5f;
    dest[0u][massO] = -3.0;

    ll::copy_values(dest.select(posO, velO)[0u], src.select(posO, velO)[0u]);

    CHECK(dest[0u][posO][xO] == Catch::Approx(8.5f));
    CHECK(dest[0u][velO][yO] == Catch::Approx(6.5f));
    CHECK(dest[0u][massO] == Catch::Approx(-3.0));
}

TEST_CASE("select_like copies a root subrecord into a larger record", "[View]")
{
    using Accumulator = ll::Record<ll::Field<massO_t, double>>;
    ll::One<ParticleOne> dest{};
    ll::One<Accumulator> accumulators{};
    accumulators[0u][massO] = 17.0;
    dest[0u][massO] = -1.0;
    dest[0u][posO][xO] = 5.0f;

    auto accumulator = accumulators[0u];
    ll::copy_values(ll::select_like(dest[0u], accumulator), accumulator);

    CHECK(dest[0u][massO] == Catch::Approx(17.0));
    CHECK(dest[0u][posO][xO] == Catch::Approx(5.0f));
}

TEST_CASE("select_like normalizes model leaves into destination declaration order", "[View]")
{
    using Reordered = ll::Record<ll::Field<massO_t, double>, ll::Field<posO_t, Pos3O>>;
    ll::One<Reordered> model{};
    ll::One<ParticleOne> destination{};
    model[0u][massO] = 19.0;
    model[0u][posO][xO] = 7.0f;
    destination[0u][velO][xO] = -4.0f;

    auto selected = ll::select_like(destination[0u], model[0u]);
    using Expected = decltype(destination.select(posO, massO)[0u]);
    STATIC_CHECK(std::is_same_v<decltype(selected), Expected>);
    ll::copy_values(selected, model[0u]);
    CHECK(destination[0u][massO] == Catch::Approx(19.0));
    CHECK(destination[0u][posO][xO] == Catch::Approx(7.0f));
    CHECK(destination[0u][velO][xO] == Catch::Approx(-4.0f));

    auto modelPosition = model[0u].view(posO);
    auto destinationPosition = destination[0u].view(posO);
    auto selectedPosition = ll::select_like(destinationPosition, modelPosition);
    STATIC_CHECK(std::is_same_v<decltype(selectedPosition), decltype(destinationPosition)>);
    using ModelMass = decltype(model[0u].view(massO));
    STATIC_CHECK_FALSE(CanSelectLike<decltype(destinationPosition), ModelMass>);
}

TEST_CASE("copy_values propagates throwing leaf assignments", "[View]")
{
    using ThrowRecord = ll::Record<ll::Field<throwingO_t, ThrowingValue>>;
    ll::One<ThrowRecord> src{};
    ll::One<ThrowRecord> dest{};
    src[0u][throwingO].value = 9;

    ThrowingValue::throwOnAssign = true;
    bool threw = false;
    try
    {
        ll::copy_values(dest[0u], src[0u]);
    }
    catch(std::runtime_error const&)
    {
        threw = true;
    }
    ThrowingValue::throwOnAssign = false;

    CHECK(threw);
    CHECK(dest[0u][throwingO].value == 0);
}
