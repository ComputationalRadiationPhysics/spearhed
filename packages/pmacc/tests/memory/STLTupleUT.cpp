#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/tuple/STLTuple.hpp>
#include <pmacc/memory/tuple/utility.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llamaLite/Tuple.hpp>

namespace tuple = pmacc::memory::tuple;

template<typename T>
concept CanTieRvalue = requires(T&& value) { tuple::tie(std::forward<T>(value)); };

template<typename T, typename U>
concept CanAssignFrom = requires(T& target, U&& source) { target = std::forward<U>(source); };

struct Empty
{
};

struct alignas(32) AlignedEmpty
{
};

void sampleFunction()
{
}

struct ThrowingCopy
{
    ThrowingCopy() = default;

    ThrowingCopy(ThrowingCopy const&) noexcept(false)
    {
        throw std::runtime_error("copy failed");
    }
};

struct ThrowingMove
{
    ThrowingMove() = default;

    ThrowingMove(ThrowingMove&&) noexcept(false)
    {
        throw std::runtime_error("move failed");
    }
};

struct ThrowingAssign
{
    ThrowingAssign& operator=(ThrowingAssign const&) noexcept(false)
    {
        throw std::runtime_error("assignment failed");
    }

    ThrowingAssign& operator=(ThrowingAssign&&) noexcept(false)
    {
        throw std::runtime_error("assignment failed");
    }
};

struct TupleDeviceKernel
{
    template<typename T_Acc>
    DINLINE void operator()(T_Acc const&, int* result) const
    {
        auto values = tuple::append(tuple::make_tuple(2), 3);
        auto valueCopy = values;
        valueCopy = values;
        int target = 0;
        int source = 7;
        auto targetReference = tuple::tie(target);
        auto sourceReference = tuple::tie(source);
        targetReference = sourceReference;
        int llamaTarget = 0;
        int llamaSource = 9;
        llama_lite::tie(llamaTarget) = llama_lite::tie(llamaSource);
        auto llamaValues = llama_lite::tupleCat(llama_lite::makeTuple(2), llama_lite::makeTuple(3));
        result[0] = tuple::get<0>(valueCopy) + tuple::get<1>(valueCopy) + target + llamaTarget
                    + llama_lite::get<0>(llamaValues) + llama_lite::get<1>(llamaValues);
    }
};

TEST_CASE("llamaLite tuple works with alpaka::apply")
{
    using llama_lite::Tuple;
    using ValueTuple = Tuple<int, double>;
    static_assert(std::is_trivially_copyable_v<ValueTuple>);
    static_assert(std::is_trivially_copy_assignable_v<ValueTuple>);
    static_assert(std::is_trivially_move_assignable_v<ValueTuple>);

    Tuple<int, int> values{1, 2};
    auto sum = alpaka::apply(
        [](int& first, int& second)
        {
            first += 10;
            return first + second;
        },
        values);
    CHECK(sum == 13);
    CHECK(llama_lite::get<0>(values) == 11);

    auto moveOnly = Tuple<std::unique_ptr<int>>{std::make_unique<int>(42)};
    auto result = alpaka::apply([](std::unique_ptr<int> value) { return *value; }, std::move(moveOnly));
    CHECK(result == 42);
}

TEST_CASE("Tuple get preserves cv and value categories")
{
    using Values = tuple::Tuple<int>;
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<Values&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<Values const&>())), int const&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<Values&&>())), int&&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<Values const&&>())), int const&&>);

    using TwoValues = tuple::Tuple<int, long>;
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<TwoValues&>())), long&>);
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<TwoValues const&>())), long const&>);
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<TwoValues&&>())), long&&>);
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<TwoValues const&&>())), long const&&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<tuple::Tuple<int&>&&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<tuple::Tuple<int&> const&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<tuple::Tuple<int&&>&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<tuple::Tuple<int&&> const&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<tuple::Tuple<int&&>&&>())), int&&>);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::declval<tuple::Tuple<int&&> const&&>())), int&&>);

    using ReferenceTail = tuple::Tuple<int, int&&>;
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<ReferenceTail&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<ReferenceTail const&>())), int&>);
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<ReferenceTail&&>())), int&&>);
    static_assert(std::is_same_v<decltype(tuple::get<1>(std::declval<ReferenceTail const&&>())), int&&>);

    int value = 1;
    auto values = tuple::make_tuple(value);
    tuple::get<0>(values) = 2;
    CHECK(value == 1);
    CHECK(tuple::get<0>(values) == 2);

    auto references = tuple::tie(value);
    static_assert(std::is_same_v<decltype(tuple::get<0>(std::move(references))), int&>);
    tuple::get<0>(std::move(references)) = 3;
    CHECK(value == 3);
}

TEST_CASE("Tuple factories decay arrays and unwrap reference wrappers")
{
    auto text = tuple::make_tuple("hello");
    static_assert(std::is_same_v<decltype(text), tuple::Tuple<char const*>>);
    CHECK(std::string(tuple::get<0>(text)) == "hello");

    int value = 3;
    auto reference = tuple::make_tuple(std::ref(value));
    static_assert(std::is_same_v<decltype(reference), tuple::Tuple<int&>>);
    tuple::get<0>(reference) = 4;
    CHECK(value == 4);

    tuple::Tuple deduced("world");
    static_assert(std::is_same_v<decltype(deduced), tuple::Tuple<char const*>>);

    auto function = tuple::make_tuple(sampleFunction);
    static_assert(std::is_same_v<decltype(function), tuple::Tuple<void (*)()>>);
    auto constReference = tuple::make_tuple(std::cref(value));
    static_assert(std::is_same_v<decltype(constReference), tuple::Tuple<int const&>>);
}

TEST_CASE("Tuple append forwards elements and concatenates tuples")
{
    auto values = tuple::make_tuple(1);
    auto withString = tuple::append(values, std::string("value"));
    CHECK(tuple::get<0>(withString) == 1);
    CHECK(tuple::get<1>(withString) == "value");

    auto moved = tuple::append(tuple::make_tuple(std::make_unique<int>(9)), 2);
    CHECK(*tuple::get<0>(moved) == 9);
    CHECK(tuple::get<1>(moved) == 2);

    auto first = tuple::make_tuple(3);
    auto second = tuple::make_tuple(4);
    auto joined = tuple::append(first, second);
    CHECK(tuple::get<0>(joined) == 3);
    CHECK(tuple::get<1>(joined) == 4);

    auto joinedRvalues
        = tuple::append(tuple::make_tuple(std::make_unique<int>(11)), tuple::make_tuple(std::make_unique<int>(12)));
    CHECK(*tuple::get<0>(joinedRvalues) == 11);
    CHECK(*tuple::get<1>(joinedRvalues) == 12);

    auto constFirst = tuple::make_tuple(14);
    auto constSecond = tuple::make_tuple(15);
    auto joinedConst = tuple::append(std::as_const(constFirst), std::as_const(constSecond));
    CHECK(tuple::get<0>(joinedConst) == 14);
    CHECK(tuple::get<1>(joinedConst) == 15);

    auto appendedToEmpty = tuple::append(tuple::Tuple<>{}, 13);
    CHECK(tuple::get<0>(appendedToEmpty) == 13);
    auto emptyThenValues = tuple::append(tuple::Tuple<>{}, values);
    auto valuesThenEmpty = tuple::append(values, tuple::Tuple<>{});
    CHECK(tuple::get<0>(emptyThenValues) == 1);
    CHECK(tuple::get<0>(valuesThenEmpty) == 1);
}

TEST_CASE("Tuple assignment supports reference tuples and stays trivial for values")
{
    static_assert(std::is_trivially_copy_assignable_v<tuple::Tuple<int>>);
    static_assert(!CanTieRvalue<int>);

    int target = 0;
    tuple::tie(target) = tuple::make_tuple(42);
    CHECK(target == 42);

    short source = 17;
    tuple::tie(target) = tuple::tie(source);
    CHECK(target == 17);

    int const constant = 1;
    auto constReference = tuple::tie(constant);
    static_assert(!std::is_copy_assignable_v<decltype(constReference)>);
    static_assert(!CanAssignFrom<tuple::Tuple<int&>, tuple::Tuple<int, int>>);
    static_assert(!CanAssignFrom<decltype(constReference), tuple::Tuple<int>>);
    static_assert(std::is_nothrow_copy_assignable_v<tuple::Tuple<int&>>);
    static_assert(!std::is_nothrow_copy_assignable_v<tuple::Tuple<ThrowingAssign&>>);

    int copyTarget = 0;
    auto copyDestination = tuple::tie(copyTarget);
    auto copySource = tuple::tie(target);
    copyDestination = copySource;
    CHECK(copyTarget == 17);
    copyTarget = 0;
    copyDestination = std::move(copySource);
    CHECK(copyTarget == 17);

    tuple::Tuple<int&, long> mixedDestination{copyTarget, 0};
    tuple::Tuple<short, int> mixedSource{short{21}, 22};
    mixedDestination = mixedSource;
    CHECK(copyTarget == 21);
    CHECK(tuple::get<1>(mixedDestination) == 22);

#if !defined(__CUDACC__)
    ThrowingAssign throwingTarget;
    ThrowingAssign throwingSource;
    auto throwingDestination = tuple::tie(throwingTarget);
    auto throwingTuple = tuple::tie(throwingSource);
    CHECK_THROWS(throwingDestination = throwingTuple);
#endif

    auto uniqueTarget = std::make_unique<int>(0);
    tuple::tie(uniqueTarget) = tuple::make_tuple(std::make_unique<int>(8));
    REQUIRE(uniqueTarget != nullptr);
    CHECK(*uniqueTarget == 8);
}

TEST_CASE("Tuple empty subobjects do not add avoidable storage overhead")
{
    static_assert(sizeof(tuple::Tuple<double>) == sizeof(double));
    static_assert(sizeof(tuple::Tuple<Empty>) == sizeof(Empty));
    static_assert(sizeof(tuple::Tuple<Empty, Empty>) >= 2);
    static_assert(alignof(tuple::Tuple<AlignedEmpty>) == alignof(AlignedEmpty));
    static_assert(sizeof(tuple::Tuple<Empty, int, Empty>) >= sizeof(int));
    static_assert(std::is_standard_layout_v<tuple::Tuple<int, double>>);
    static_assert(std::is_trivially_copy_constructible_v<tuple::Tuple<int, double>>);
    static_assert(std::is_trivially_copy_assignable_v<tuple::Tuple<int, double>>);
    static_assert(std::is_trivially_destructible_v<tuple::Tuple<int, double>>);
    static_assert(std::is_trivially_copyable_v<tuple::Tuple<int, double>>);

    tuple::Tuple<Empty, Empty> emptyValues{Empty{}, Empty{}};
    CHECK(std::addressof(tuple::get<0>(emptyValues)) != std::addressof(tuple::get<1>(emptyValues)));
}

TEST_CASE("Tuple construction constraints and noexcept reflect elements")
{
    static_assert(!std::is_constructible_v<tuple::Tuple<int>, std::string>);
    static_assert(std::is_nothrow_constructible_v<tuple::Tuple<int>, int>);
    static_assert(!std::is_nothrow_copy_constructible_v<tuple::Tuple<ThrowingCopy>>);
    static_assert(!std::is_nothrow_move_constructible_v<tuple::Tuple<ThrowingMove>>);

#if !defined(__CUDACC__)
    ThrowingCopy copySource;
    CHECK_THROWS(tuple::Tuple<ThrowingCopy>{copySource});
    ThrowingMove moveSource;
    CHECK_THROWS(tuple::Tuple<ThrowingMove>{std::move(moveSource)});
#endif
}

TEST_CASE("Tuple construction, access, assignment, and append compile and run on device")
{
    using namespace pmacc;
    auto buffer = HostDeviceBuffer<int, DIM1>(DataSpace<DIM1>{1u});
    auto& device = Environment<>::get().DeviceContext();
    caravan::ControlContext context;
    auto kernel = PMACC_KERNEL(TupleDeviceKernel{})(1u, 1u)(buffer.getDeviceBuffer().data());
    auto copy = buffer.deviceToHost();
    context.wait(
        context.spawn(caravan::alpaka::withDevice(device, std::move(kernel) | caravan::sequence(std::move(copy)))));
    CHECK(buffer.getHostBuffer().data()[0] == 26);
}

TEST_CASE("Tuple get and apply support move-only elements")
{
    auto values = tuple::make_tuple(std::make_unique<int>(42));
    auto moved = tuple::get<0>(std::move(values));
    REQUIRE(moved != nullptr);
    CHECK(*moved == 42);

    auto mutableValues = tuple::make_tuple(1);
    tuple::apply([](int& value) { value = 5; }, mutableValues);
    CHECK(tuple::get<0>(mutableValues) == 5);

#if !defined(__CUDACC__)
    auto moveValues = tuple::make_tuple(std::make_unique<int>(7));
    int result = tuple::apply([](std::unique_ptr<int> value) { return *value; }, std::move(moveValues));
    CHECK(result == 7);
#endif
}
