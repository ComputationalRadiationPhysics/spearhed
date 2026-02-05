#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace pmacc::spearhed::meta
{
    template<typename... Ts>
    requires(std::is_class_v<Ts> && ...)
    struct TypeList
    {
        static constexpr std::size_t size = sizeof...(Ts);
    };

    namespace detail
    {
        template<typename T_Type>
        struct ToTypeList
        {
            using type = TypeList<T_Type>;
        };

        template<typename... Ts>
        struct ToTypeList<TypeList<Ts...>>
        {
            using type = TypeList<Ts...>;
        };
    } // namespace detail

    /** If T_Type is an TypeList, return it. Otherwise wrap it in an TypeList.
     */
    template<typename T_Type>
    using ToTypeList_t = typename detail::ToTypeList<T_Type>::type;


    template<typename T, typename... Ts>
    constexpr bool contains_type = (std::is_same_v<T, Ts> || ...);

    template<typename T>
    struct IsUnique : std::true_type
    {
    };

    template<typename... Ts>
    struct IsUnique<TypeList<Ts...>>
    {
        static constexpr bool check()
        {
            if constexpr(sizeof...(Ts) == 0)
                return true;
            else
                return check_impl<Ts...>();
        }

        template<typename Head, typename... Tail>
        static constexpr bool check_impl()
        {
            if constexpr(contains_type<Head, Tail...>)
                return false;
            else if constexpr(sizeof...(Tail) == 0)
                return true;
            else
                return check_impl<Tail...>();
        }

        static constexpr bool value = check();
    };

    template<typename T>
    constexpr bool isUnique_v = IsUnique<T>::value;


    template<typename T_List>
    struct InheritFrom;

    template<typename... Ts>
    struct InheritFrom<TypeList<Ts...>> : public Ts...
    {
    };

    namespace detail
    {
        template<typename T_List>
        struct AsTuple;

        template<typename... Ts>
        struct AsTuple<TypeList<Ts...>>
        {
            using type = std::tuple<Ts...>;
        };
    } // namespace detail

    template<typename T_Type>
    using AsTuple_t = typename detail::AsTuple<T_Type>::type;

} // namespace pmacc::spearhed::meta
