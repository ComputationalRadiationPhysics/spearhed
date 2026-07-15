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

#include <concepts>
#include <type_traits>
#include <utility>

/**
 * A single, composable predicate vocabulary shared by every level of the code.
 *
 * A *predicate* is a stateless-or-small callable value returning bool. The same combinators serve
 * both runtime, device-side particle checks (`occupied(particle)`) and compile-time species checks
 * (`withRole<Source>(SomeSpecies{})`) -- only the leaf differs, never the composition. Everything is
 * `HDINLINE constexpr`, so one definition is valid inside a `__device__` kernel *and* in an
 * `if constexpr`.
 *
 * Two composition axes:
 *  - boolean: `a && b`, `a || b`, `!a`  (operator overloads, gated on the Predicate concept)
 *  - functional: `on(valuePred, projection)` -- retarget a value-predicate onto a projected input,
 *    with point-free sugar `get<Tag> > bound`.
 *
 * Note: overloaded && / || do not short-circuit. Harmless for side-effect-free predicates.
 */
namespace pmacc::spearhed::pred
{
    /** Marker base so the composition operators fire only for our predicates (never hijack && / ||
     *  for unrelated types). Every predicate leaf and node derives from it. */
    struct PredicateBase
    {
    };

    template<typename T>
    concept Predicate = std::derived_from<std::remove_cvref_t<T>, PredicateBase>;

    // boolean combinators

    template<Predicate A, Predicate B>
    struct And : PredicateBase
    {
        [[no_unique_address]] A a;
        [[no_unique_address]] B b;

        HDINLINE constexpr bool operator()(auto const&... x) const
        {
            return a(x...) && b(x...);
        }
    };

    template<Predicate A, Predicate B>
    struct Or : PredicateBase
    {
        [[no_unique_address]] A a;
        [[no_unique_address]] B b;

        HDINLINE constexpr bool operator()(auto const&... x) const
        {
            return a(x...) || b(x...);
        }
    };

    template<Predicate A>
    struct Neg : PredicateBase
    {
        [[no_unique_address]] A a;

        HDINLINE constexpr bool operator()(auto const&... x) const
        {
            return !a(x...);
        }
    };

    template<Predicate A, Predicate B>
    HDINLINE constexpr auto operator&&(A a, B b)
    {
        return And<A, B>{{}, a, b};
    }

    template<Predicate A, Predicate B>
    HDINLINE constexpr auto operator||(A a, B b)
    {
        return Or<A, B>{{}, a, b};
    }

    template<Predicate A>
    HDINLINE constexpr auto operator!(A a)
    {
        return Neg<A>{{}, a};
    }

    // constant predicates

    struct Always : PredicateBase
    {
        HDINLINE constexpr bool operator()(auto const&...) const
        {
            return true;
        }
    };

    struct Never : PredicateBase
    {
        HDINLINE constexpr bool operator()(auto const&...) const
        {
            return false;
        }
    };

    inline constexpr Always always{};
    inline constexpr Never never{};

    // adapt an arbitrary callable (lambda, std:: predicate) into the algebra
    // Device use requires F itself be device-callable; prefer named leaves for kernels.

    template<typename F>
    struct Fn : PredicateBase
    {
        [[no_unique_address]] F f;

        HDINLINE constexpr bool operator()(auto const&... a) const
        {
            return static_cast<bool>(f(a...));
        }
    };

    template<typename F>
    HDINLINE constexpr Fn<std::decay_t<F>> fn(F&& f)
    {
        return {{}, std::forward<F>(f)};
    }

    // functional composition: predicate on a projected value
    // `on(p, proj)` == p . proj : project the input(s), then test.

    template<Predicate P, typename Proj>
    struct On : PredicateBase
    {
        [[no_unique_address]] P p;
        [[no_unique_address]] Proj proj;

        HDINLINE constexpr bool operator()(auto const&... x) const
        {
            return p(proj(x...));
        }
    };

    template<Predicate P, typename Proj>
    HDINLINE constexpr auto on(P p, Proj proj)
    {
        return On<P, std::decay_t<Proj>>{{}, p, proj};
    }

    // value predicates: curried comparisons against a bound
    // These carry the bound, so they are not empty; keep bounds trivially copyable for device use.

    template<typename T>
    struct Gt : PredicateBase
    {
        T bound;

        HDINLINE constexpr bool operator()(auto const& v) const
        {
            return v > bound;
        }
    };

    template<typename T>
    struct Lt : PredicateBase
    {
        T bound;

        HDINLINE constexpr bool operator()(auto const& v) const
        {
            return v < bound;
        }
    };

    template<typename T>
    struct Ge : PredicateBase
    {
        T bound;

        HDINLINE constexpr bool operator()(auto const& v) const
        {
            return v >= bound;
        }
    };

    template<typename T>
    struct Le : PredicateBase
    {
        T bound;

        HDINLINE constexpr bool operator()(auto const& v) const
        {
            return v <= bound;
        }
    };

    template<typename T>
    struct Eq : PredicateBase
    {
        T bound;

        HDINLINE constexpr bool operator()(auto const& v) const
        {
            return v == bound;
        }
    };

    template<typename T>
    struct Ne : PredicateBase
    {
        T bound;

        HDINLINE constexpr bool operator()(auto const& v) const
        {
            return v != bound;
        }
    };

    template<typename T>
    HDINLINE constexpr Gt<T> gt(T b)
    {
        return {{}, b};
    }

    template<typename T>
    HDINLINE constexpr Lt<T> lt(T b)
    {
        return {{}, b};
    }

    template<typename T>
    HDINLINE constexpr Ge<T> ge(T b)
    {
        return {{}, b};
    }

    template<typename T>
    HDINLINE constexpr Le<T> le(T b)
    {
        return {{}, b};
    }

    template<typename T>
    HDINLINE constexpr Eq<T> eq(T b)
    {
        return {{}, b};
    }

    template<typename T>
    HDINLINE constexpr Ne<T> ne(T b)
    {
        return {{}, b};
    }

    // projections

    /** Projection: particle -> value of attribute @p Tag (by value, so no dangling into a frame). */
    template<typename Tag>
    struct Get
    {
        HDINLINE constexpr auto operator()(auto const& particle) const
        {
            return particle[Tag{}];
        }
    };

    template<typename Tag>
    inline constexpr Get<Tag> get{};

    /** Predicate: attribute @p Tag is truthy (e.g. a bool/mask field). */
    template<typename Tag>
    struct FieldSet : PredicateBase
    {
        HDINLINE constexpr bool operator()(auto const& particle) const
        {
            return static_cast<bool>(particle[Tag{}]);
        }
    };

    template<typename Tag>
    inline constexpr FieldSet<Tag> fieldSet{};

    // point-free sugar:  get<Tag> <op> bound  -> a particle predicate
    // The projection sits on the left; the bound is any comparable value.

    template<typename Tag, typename T>
    HDINLINE constexpr auto operator>(Get<Tag>, T b)
    {
        return on(gt(b), get<Tag>);
    }

    template<typename Tag, typename T>
    HDINLINE constexpr auto operator<(Get<Tag>, T b)
    {
        return on(lt(b), get<Tag>);
    }

    template<typename Tag, typename T>
    HDINLINE constexpr auto operator>=(Get<Tag>, T b)
    {
        return on(ge(b), get<Tag>);
    }

    template<typename Tag, typename T>
    HDINLINE constexpr auto operator<=(Get<Tag>, T b)
    {
        return on(le(b), get<Tag>);
    }

    template<typename Tag, typename T>
    HDINLINE constexpr auto operator==(Get<Tag>, T b)
    {
        return on(eq(b), get<Tag>);
    }

    template<typename Tag, typename T>
    HDINLINE constexpr auto operator!=(Get<Tag>, T b)
    {
        return on(ne(b), get<Tag>);
    }

    // compile-time evaluation front door

    /** Force compile-time evaluation of a stateless predicate on tag @p S (e.g. a species tag):
     *
     *      if constexpr(pred::eval<S>(withRole<Source> && !withRole<Frozen>)) { ... }
     *
     * consteval, so misuse on a runtime value is a hard error. @p p must be stateless. */
    template<typename S>
    consteval bool eval(Predicate auto p)
    {
        return p(S{});
    }
} // namespace pmacc::spearhed::pred
