/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/mpi.hpp>

using Context = caravan::MpiContext;
using ConstBuffer = caravan::ConstMpiBuffer;
using Buffer = caravan::MpiBuffer;

static_assert(std::is_same_v<decltype(std::declval<ConstBuffer>().value.data()), std::byte const*>);
static_assert(std::is_same_v<decltype(std::declval<Buffer>().value.data()), std::byte*>);
static_assert(caravan::Sender<decltype(caravan::mpi::send(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  caravan::Peer{},
                  caravan::MessageTag{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::receive(
                  std::declval<Context&>(),
                  std::declval<Buffer>(),
                  caravan::Peer{},
                  caravan::MessageTag{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::allReduce(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  caravan::ScalarType::int32,
                  caravan::ReduceOperation::sum))>);
static_assert(caravan::Sender<decltype(caravan::mpi::reduce(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  caravan::ScalarType::int32,
                  caravan::ReduceOperation::sum,
                  caravan::Peer{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::gather(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  caravan::Peer{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::gatherV(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  std::vector<std::size_t>{},
                  std::vector<std::size_t>{},
                  caravan::Peer{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::barrier(std::declval<Context&>()))>);
static_assert(
    caravan::Sender<
        decltype(caravan::mpi::createCartesian(std::declval<Context&>(), std::vector<int>{}, std::vector<bool>{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::duplicateCommunicator(std::declval<Context&>()))>);
static_assert(
    caravan::Sender<decltype(caravan::mpi::splitCommunicator(std::declval<Context&>(), std::optional<int>{}, 0))>);
static_assert(caravan::Sender<
              decltype(caravan::mpi::destroyCommunicator(std::declval<Context&>(), caravan::worldCommunicator))>);

using RetainedBytes = caravan::Retained<std::span<std::byte>, std::shared_ptr<int>>;
using RetainedConstBytes = caravan::Retained<std::span<std::byte const>, std::shared_ptr<int>>;
static_assert(std::is_convertible_v<RetainedBytes, Buffer>);
static_assert(std::is_convertible_v<RetainedBytes, ConstBuffer>);
static_assert(std::is_convertible_v<RetainedConstBytes, ConstBuffer>);
static_assert(!std::is_constructible_v<Buffer, RetainedConstBytes>);
static_assert(!std::is_constructible_v<Buffer, caravan::Retained<std::span<std::byte>, int*>>);

using Bytes = std::span<std::byte>;
using ConstBytes = std::span<std::byte const>;
static_assert(std::is_convertible_v<Bytes, Buffer>);
static_assert(std::is_convertible_v<Bytes, ConstBuffer>);
static_assert(std::is_convertible_v<ConstBytes, ConstBuffer>);
static_assert(std::is_convertible_v<std::span<std::byte, 4>, Buffer>);
static_assert(std::is_convertible_v<std::span<std::byte, 4>, ConstBuffer>);
static_assert(std::is_convertible_v<std::span<std::byte const, 4>, ConstBuffer>);
static_assert(!std::is_constructible_v<Buffer, ConstBytes>);
static_assert(!std::is_constructible_v<Buffer, std::span<std::byte const, 4>>);
static_assert(!std::is_constructible_v<Buffer, std::span<int>>);
static_assert(std::is_convertible_v<caravan::Retained<Bytes, std::shared_ptr<void>>, Buffer>);
static_assert(std::is_convertible_v<caravan::Retained<ConstBytes, std::shared_ptr<void>>, ConstBuffer>);
static_assert(std::is_convertible_v<Buffer, ConstBuffer>);
static_assert(!std::is_constructible_v<Buffer, ConstBuffer>);
static_assert(
    requires(Context& mpi, ConstBytes input, Bytes output, ConstBuffer retainedInput, Buffer retainedOutput) {
        caravan::mpi::send(mpi, input, caravan::Peer{}, caravan::MessageTag{});
        caravan::mpi::receive(mpi, output, caravan::Peer{}, caravan::MessageTag{});
        caravan::mpi::allReduce(mpi, input, output, caravan::ScalarType::int32, caravan::ReduceOperation::sum);
        caravan::mpi::reduce(
            mpi,
            input,
            output,
            caravan::ScalarType::int32,
            caravan::ReduceOperation::sum,
            caravan::Peer{});
        caravan::mpi::gather(mpi, input, output, caravan::Peer{});
        caravan::mpi::allGather(mpi, input, output);
        caravan::mpi::gatherV(mpi, input, output, {}, {}, caravan::Peer{});
        caravan::mpi::allGather(mpi, retainedInput, output);
        caravan::mpi::allGather(mpi, input, retainedOutput);
    });

int main()
{
    int borrowed = 7;
    Buffer output = std::as_writable_bytes(std::span<int, 1>{&borrowed, 1});
    ConstBuffer input = std::as_bytes(std::span{&borrowed, 1});
    assert(output.value.data() == reinterpret_cast<std::byte*>(&borrowed));
    assert(input.value.data() == output.value.data());
    assert(input.value.size_bytes() == sizeof(int) && output.value.size_bytes() == sizeof(int));
    assert(!input.owner && !output.owner);

    auto allocation = std::make_shared<int>(42);
    auto* pointer = allocation.get();
    std::weak_ptr<int> lifetime = allocation;
    {
        Buffer buffer = caravan::retain(std::as_writable_bytes(std::span<int, 1>{pointer, 1}), allocation);
        ConstBuffer input = caravan::retain(std::as_bytes(std::span{pointer, 1}), allocation);
        allocation.reset();
        assert(!lifetime.expired());
        assert(buffer.value.data() == reinterpret_cast<std::byte*>(pointer));
        assert(buffer.value.size_bytes() == sizeof(int));
        assert(input.value.data() == reinterpret_cast<std::byte const*>(pointer));
        assert(input.value.size_bytes() == sizeof(int));
        assert(buffer.owner == input.owner);
    }
    assert(lifetime.expired());
    Buffer empty{std::span<std::byte>{}, {}};
    assert(empty.value.empty() && !empty.owner);
    Buffer borrowedEmpty = std::span<std::byte>{};
    assert(borrowedEmpty.value.empty() && !borrowedEmpty.owner);
}
