#pragma once

// #include "AccessSet.hpp"
#include "Record.hpp"
#include "llamaLite/tag/TagPath.hpp"
#include "llamaLite/utility.hpp"
#include "traits.hpp"

#include <cstdint>
#include <type_traits>

namespace llama_lite
{
    template<typename TSoA, IsRecordAccess... RAs>
    requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
    struct SoAIndexedView;

    template<typename TSoA, IsRecordAccess... RAs>
    requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
    struct SoAView
    {
        using record_type = TSoA::record_type;

        TSoA* soa;

        // constructor only available if RAs exist in the TSoA record
        SoAView(TSoA& soa_, RAs...) requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
            : soa{&soa_} {};

        template<typename... ParentRAs>
        SoAView(SoAView<TSoA, ParentRAs...> view, RAs...)
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
                    && (IsInSet<RAs, ParentRAs...> && ...)
            : soa{view.soa} {};

        [[nodiscard]] decltype(auto) operator[](uint32_t idx)
        {
            return SoAIndexedView(*this, idx);
        }

        [[nodiscard]] decltype(auto) operator[](uint32_t idx) const
        {
            return SoAIndexedView(*this, idx);
        }

        template<IsRecordAccess RA>
        [[nodiscard]] decltype(auto) operator[](RA tag) requires(sizeof...(RAs) == 1) //&& (RAs::isRootOf(tag) || ...)
        {
            using ViewRA = typename SingleElementPack<RAs...>::type;
            using Path = append_t<ViewRA, RA>;
            return SoAView<TSoA, Path>(*(this->soa), Path{});
        }
    };

    template<typename TSoA, IsRecordAccess... RAs>
    requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
    struct SoAIndexedView
    {
        using record_type = TSoA::record_type;

        TSoA* soa;
        uint32_t idx;

        // consteval default constructor, to help get the type of a view more easily
        consteval SoAIndexedView() = default;

        // constructor only available if RAs exist in the TSoA record
        SoAIndexedView(TSoA& soa_, uint32_t index, RAs...)
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
            : soa{&soa_}
            , idx{index} {};

        SoAIndexedView(SoAView<TSoA, RAs...> view, uint32_t index) : soa{view.soa}, idx{index} {};

        template<typename... ParentRAs>
        SoAIndexedView(SoAView<TSoA, ParentRAs...> view, uint32_t index, RAs...)
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
                        && (IsInSet<RAs, ParentRAs...> && ...)
            : soa{view.soa}
            , idx{index} {};

        template<typename... ParentRAs>
        SoAIndexedView(SoAIndexedView<TSoA, ParentRAs...> idxView, RAs...)
            requires(requires { typename TSoA::record_type::template field_for<RAs>; } && ...)
                        && (IsInSet<RAs, ParentRAs...> && ...)
            : soa{idxView.soa}
            , idx{idxView.idx} {};

        // conversion constructor to defined RAs from another view.
        template<typename... OtherRAs>
        SoAIndexedView(SoAIndexedView<TSoA, OtherRAs...> const& other)
            requires(
                        // Allow conversion from Root view
                        sizeof...(OtherRAs) == 0 ||
                        // OR Ensure all RAs in this view are present in the OtherRAs
                        (IsInSet<RAs, OtherRAs...> && ...))
            : soa{other.soa}
            , idx{other.idx} {};

        // TODO add checks on RA being valid for the soa record
        template<IsRecordAccess RA>
        [[nodiscard]] decltype(auto) operator[](RA) const
            requires(sizeof...(RAs) <= 1) //&& (RAs::isRootOf(tag) || ...)
        {
            if constexpr(sizeof...(RAs) == 1)
            {
                using ViewRA = typename SingleElementPack<RAs...>::type;
                using Path = append_t<ViewRA, RA>;
                return SoAIndexedView<TSoA, Path>(*(this->soa), idx, Path{});
            }
            else
            {
                using Path = RA;
                return SoAIndexedView<TSoA, Path>(*(this->soa), idx, Path{});
            }
        }

        template<IsRecordAccess RA>
        [[nodiscard]] auto operator[](RA query) const
            requires((sizeof...(RAs) > 1) && (RAs::isRootOf(query) || ...) && IsInSet<RA, RAs...>)
        {
            return SoAIndexedView(*this, query);
        }

        // needs a leaf access RA in an indexed view. Should only happen when casting to such a type
        // for example implicitly when the user requests it
        [[nodiscard]] decltype(auto) operator*()
            requires((sizeof...(RAs) == 1) && (TSoA::record_type::template isLeaf<RAs...>()))
        {
            return soa->template getLeaf<RAs...>()[idx];
        }

        [[nodiscard]] decltype(auto) operator*() const
            requires((sizeof...(RAs) == 1) && (TSoA::record_type::template isLeaf<RAs...>()))
        {
            return soa->template getLeaf<RAs...>()[idx];
        }

        // requires we are a leaf node or AsType is
        [[nodiscard]] decltype(auto) get() requires(
            (sizeof...(RAs) == 1)
            && (TSoA::record_type::template isLeaf<RAs...>()
                || traits::IsTraitSpecialized<traits::AsType, typename TSoA::record_type::template field_for<RAs...>>::
                    value))
        {
            if constexpr(traits::IsTraitSpecialized<
                             traits::AsType,
                             typename TSoA::record_type::template field_for<RAs...>>::value)
            {
                return traits::AsType<typename TSoA::record_type::template field_for<RAs...>>{}(*this);
            }
            else // is a leaf
            {
                return *(*this);
            }
        }

        [[nodiscard]] decltype(auto) get() const requires(
            (sizeof...(RAs) == 1)
            && (TSoA::record_type::template isLeaf<RAs...>()
                || traits::IsTraitSpecialized<traits::AsType, typename TSoA::record_type::template field_for<RAs...>>::
                    value))
        {
            if constexpr(traits::IsTraitSpecialized<
                             traits::AsType,
                             typename TSoA::record_type::template field_for<RAs...>>::value)
            {
                return traits::AsType<typename TSoA::record_type::template field_for<RAs...>>{}(*this);
            }
            else // is a leaf
            {
                return *(*this);
            }
        }
    };

    // template<template<typename> typename Func, typename T_Record, IsRecordAccess... RAs>
    // constexpr void for_each(T_Record Record, RAs...)
    //     requires(requires { typename T_Record::template field_for<RAs>; } && ...)
    // {
    //     // if view has 0 RAs, go over all elements in a record
    //     // else go over all RAs only.

    //     // if Func<RA> is defined call it
    //     // else (if RA points to a record, call Func<> on its fields, and so on recursively. If RA is a field )

    //     using CurrentRecordType = std::conditional_t<
    //         sizeof...(RAs) == 0,
    //         typename TSoA::record_type,
    //         typename TSoA::record_type::template value_type_for<
    //             typename ToPath<std::tuple_element_t<0, Tuple<RAs...>>>::type>>;

    //     // We inspect the structure of the record currently pointed to by this View
    //     using Fields = typename CurrentRecordType::fields_tuple_type;

    //     // Fold expression to apply function to all children
    //     [&]<typename... Fs>(Tuple<Fs...>) { (func((*this)[typename Fs::tag_type{}]), ...); }(Fields{});
    // }

} // namespace llama_lite
