/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>
#include <caravan/mpi/native.hpp>

namespace
{
    template<typename T>
    struct ValueReceiver
    {
        void set_value(T value) noexcept
        {
            output.setValue(std::move(value));
        }

        caravan::Promise<T> output;
    };

    struct VoidReceiver
    {
        void set_value() noexcept
        {
            output.setReady();
        }

        caravan::EventSource output;
    };

    struct SelfDeletingReceiver
    {
        void finish() noexcept
        {
            auto* const operation = std::exchange(*operationAddress, nullptr);
            auto* const completedFlag = completed;
            destroy(operation);
            completedFlag->store(true);
        }

        void set_value() noexcept
        {
            finish();
        }

        void** operationAddress;
        void (*destroy)(void*);
        std::atomic<bool>* completed;
    };
} // namespace

int main(int argc, char** argv)
{
    auto const processMain = std::this_thread::get_id();
    std::atomic<bool> shutdownGateStarted = false;
    std::atomic<bool> releaseShutdownGate = false;
    std::jthread shutdownReleaser;
    caravan::AsyncScope shutdownScope;
    int shutdownSent = 42;
    int shutdownReceived = -1;
    auto const result = caravan::MpiRuntime::run(
        argc,
        argv,
        [&](caravan::MpiContext& mpi)
        {
            assert(std::this_thread::get_id() == processMain);
            auto const topology = mpi.topology();
            assert(topology.size > 0);
            assert(topology.rank >= 0 && topology.rank < topology.size);
            assert(topology.hostLocalRank >= 0);

            std::vector<int> initiationOrder;
            caravan::EventSource firstQueuedOutput;
            caravan::EventSource secondQueuedOutput;
            auto firstQueued
                = caravan::mpi::invoke(mpi, [&](caravan::NativeMpiContext&) { initiationOrder.push_back(1); });
            auto secondQueued
                = caravan::mpi::invoke(mpi, [&](caravan::NativeMpiContext&) { initiationOrder.push_back(2); });
            auto firstQueuedOperation = std::move(firstQueued).connect(VoidReceiver{firstQueuedOutput});
            auto secondQueuedOperation = std::move(secondQueued).connect(VoidReceiver{secondQueuedOutput});
            firstQueuedOperation.start();
            secondQueuedOperation.start();
            std::array queued{firstQueuedOutput.event(), secondQueuedOutput.event()};
            caravan::whenAll(queued).wait();
            assert((initiationOrder == std::vector<int>{1, 2}));

            auto const cartesian = caravan::syncWait<caravan::TopologySnapshot>(
                caravan::mpi::createCartesian(mpi, std::vector<int>{topology.size}, std::vector<bool>{true}));
            assert(cartesian.communicator != caravan::worldCommunicator);
            assert(cartesian.dimensions == std::vector<int>{topology.size});
            assert(cartesian.coordinates == std::vector<int>{topology.rank});
            assert(cartesian.periodic == std::vector<bool>{true});
            assert(cartesian.neighbors[0] == (topology.rank + topology.size - 1) % topology.size);
            assert(cartesian.neighbors[1] == (topology.rank + 1) % topology.size);

            auto lazyCartesian
                = caravan::mpi::createCartesian(mpi, std::vector<int>{topology.size}, std::vector<bool>{true});
            caravan::Promise<caravan::TopologySnapshot> lazyCartesianOutput;
            auto lazyCartesianResult = lazyCartesianOutput.future();
            auto lazyCartesianOperation
                = std::move(lazyCartesian).connect(ValueReceiver<caravan::TopologySnapshot>{lazyCartesianOutput});
            assert(lazyCartesianResult.state() == caravan::CompletionState::pending);
            lazyCartesianOperation.start();
            auto const lazyTopology = lazyCartesianResult.result();

            auto lazyDuplicate = caravan::mpi::duplicateCommunicator(mpi, lazyTopology.communicator);
            caravan::Promise<caravan::CommunicatorId> lazyDuplicateOutput;
            auto lazyDuplicateResult = lazyDuplicateOutput.future();
            auto lazyDuplicateOperation
                = std::move(lazyDuplicate).connect(ValueReceiver<caravan::CommunicatorId>{lazyDuplicateOutput});
            assert(lazyDuplicateResult.state() == caravan::CompletionState::pending);
            lazyDuplicateOperation.start();
            auto const lazyDuplicateId = lazyDuplicateResult.result();

            auto lazySplit = caravan::mpi::splitCommunicator(mpi, 0, topology.rank, lazyDuplicateId);
            caravan::Promise<std::optional<caravan::CommunicatorInfo>> lazySplitOutput;
            auto lazySplitResult = lazySplitOutput.future();
            auto lazySplitOperation = std::move(lazySplit).connect(
                ValueReceiver<std::optional<caravan::CommunicatorInfo>>{lazySplitOutput});
            assert(lazySplitResult.state() == caravan::CompletionState::pending);
            lazySplitOperation.start();
            auto const lazySplitInfo = lazySplitResult.result();
            assert(lazySplitInfo && lazySplitInfo->rank == topology.rank && lazySplitInfo->size == topology.size);

            auto lazyDestroy = caravan::mpi::destroyCommunicator(mpi, lazySplitInfo->communicator);
            caravan::EventSource lazyDestroyOutput;
            auto lazyDestroyResult = lazyDestroyOutput.event();
            auto lazyDestroyOperation = std::move(lazyDestroy).connect(VoidReceiver{lazyDestroyOutput});
            assert(lazyDestroyResult.state() == caravan::CompletionState::pending);
            lazyDestroyOperation.start();
            lazyDestroyResult.wait();

            auto destroyLazyDuplicate = caravan::mpi::destroyCommunicator(mpi, lazyDuplicateId);
            caravan::EventSource destroyLazyDuplicateOutput;
            auto destroyLazyDuplicateResult = destroyLazyDuplicateOutput.event();
            auto destroyLazyDuplicateOperation
                = std::move(destroyLazyDuplicate).connect(VoidReceiver{destroyLazyDuplicateOutput});
            destroyLazyDuplicateOperation.start();
            destroyLazyDuplicateResult.wait();

            auto destroyLazyCartesian = caravan::mpi::destroyCommunicator(mpi, lazyTopology.communicator);
            caravan::EventSource destroyLazyCartesianOutput;
            auto destroyLazyCartesianResult = destroyLazyCartesianOutput.event();
            auto destroyLazyCartesianOperation
                = std::move(destroyLazyCartesian).connect(VoidReceiver{destroyLazyCartesianOutput});
            destroyLazyCartesianOperation.start();
            destroyLazyCartesianResult.wait();

            auto firstLifecycle = caravan::mpi::splitCommunicator(mpi, 0, topology.rank, cartesian.communicator);
            auto secondLifecycle
                = caravan::mpi::splitCommunicator(mpi, topology.rank % 2, topology.rank, cartesian.communicator);
            caravan::Promise<std::optional<caravan::CommunicatorInfo>> firstLifecycleOutput;
            caravan::Promise<std::optional<caravan::CommunicatorInfo>> secondLifecycleOutput;
            auto firstLifecycleResult = firstLifecycleOutput.future();
            auto secondLifecycleResult = secondLifecycleOutput.future();
            auto firstLifecycleOperation
                = std::move(firstLifecycle)
                      .connect(ValueReceiver<std::optional<caravan::CommunicatorInfo>>{firstLifecycleOutput});
            auto secondLifecycleOperation
                = std::move(secondLifecycle)
                      .connect(ValueReceiver<std::optional<caravan::CommunicatorInfo>>{secondLifecycleOutput});
            firstLifecycleOperation.start();
            secondLifecycleOperation.start();
            auto const firstLifecycleInfo = firstLifecycleResult.result();
            auto const secondLifecycleInfo = secondLifecycleResult.result();
            assert(firstLifecycleInfo && firstLifecycleInfo->size == topology.size);
            assert(secondLifecycleInfo);
            assert(
                secondLifecycleInfo->size == (topology.rank % 2 == 0 ? (topology.size + 1) / 2 : topology.size / 2));
            caravan::syncWait(caravan::mpi::destroyCommunicator(mpi, firstLifecycleInfo->communicator));
            caravan::syncWait(caravan::mpi::destroyCommunicator(mpi, secondLifecycleInfo->communicator));

            auto const portableDuplicate = caravan::syncWait<caravan::CommunicatorId>(
                caravan::mpi::duplicateCommunicator(mpi, cartesian.communicator));
            assert(portableDuplicate != cartesian.communicator);
            caravan::syncWait(caravan::mpi::barrier(mpi, portableDuplicate));
            caravan::mpi::CollectiveLane duplicateLane{mpi, portableDuplicate};
            caravan::syncWait(duplicateLane.submit(
                caravan::asSender(caravan::readyEvent()),
                [&] { return caravan::mpi::destroyCommunicator(mpi, portableDuplicate); }));

            auto const split
                = caravan::syncWait<std::optional<caravan::CommunicatorInfo>>(caravan::mpi::splitCommunicator(
                    mpi,
                    topology.rank % 2 == 0 ? std::optional<int>{0} : std::nullopt,
                    topology.rank));
            if(topology.rank % 2 == 0)
            {
                assert(split.has_value());
                assert(split->rank == topology.rank / 2);
                assert(split->size == (topology.size + 1) / 2);
                caravan::syncWait(caravan::mpi::destroyCommunicator(mpi, split->communicator));
            }
            else
                assert(!split.has_value());

            caravan::syncWait(caravan::mpi::barrier(mpi, cartesian.communicator));

            caravan::EventSource dependency;
            caravan::AsyncScope dependencyScope;
            auto second = dependencyScope.spawn(
                caravan::asSender(dependency.event())
                | caravan::letValue([&mpi] { return caravan::mpi::barrier(mpi); }));
            assert(second.state() == caravan::CompletionState::pending);
            dependency.setReady();
            second.wait();
            dependencyScope.join().wait();

            caravan::mpi::CollectiveLane collectiveLane{mpi, cartesian.communicator};
            bool abandonedFactoryCalled = false;
            {
                auto abandoned = collectiveLane.submit(
                    caravan::asSender(caravan::readyEvent()),
                    [&]
                    {
                        abandonedFactoryCalled = true;
                        return caravan::mpi::barrier(mpi, cartesian.communicator);
                    });
                static_cast<void>(abandoned);
            }
            caravan::syncWait(collectiveLane.submit(
                caravan::asSender(caravan::readyEvent()),
                [&] { return caravan::mpi::barrier(mpi, cartesian.communicator); }));
            assert(!abandonedFactoryCalled);

            {
                auto unstarted = collectiveLane.submit(
                    caravan::asSender(caravan::readyEvent()),
                    [&]
                    {
                        abandonedFactoryCalled = true;
                        return caravan::mpi::barrier(mpi, cartesian.communicator);
                    });
                caravan::EventSource unstartedOutput;
                auto operation = std::move(unstarted).connect(VoidReceiver{unstartedOutput});
                static_cast<void>(operation);
            }
            caravan::syncWait(collectiveLane.submit(
                caravan::asSender(caravan::readyEvent()),
                [&] { return caravan::mpi::barrier(mpi, cartesian.communicator); }));
            assert(!abandonedFactoryCalled);

            caravan::syncWait(collectiveLane.submit(
                caravan::asSender(caravan::readyEvent()),
                [&] { return caravan::mpi::barrier(mpi, cartesian.communicator); }));

            std::atomic<bool> selfDeleted = false;
            void* operationAddress = nullptr;
            auto selfDeletingCollective = collectiveLane.submit(
                caravan::asSender(caravan::readyEvent()),
                [&] { return caravan::mpi::barrier(mpi, cartesian.communicator); });
            using SelfDeletingOperation = decltype(std::move(selfDeletingCollective).connect(SelfDeletingReceiver{}));
            auto destroySelfDeletingOperation
                = [](void* operation) { delete static_cast<SelfDeletingOperation*>(operation); };
            auto* selfDeletingOperation = new SelfDeletingOperation(
                std::move(selfDeletingCollective)
                    .connect(SelfDeletingReceiver{&operationAddress, destroySelfDeletingOperation, &selfDeleted}));
            operationAddress = selfDeletingOperation;
            selfDeletingOperation->start();
            while(!selfDeleted.load())
                std::this_thread::yield();

            caravan::EventSource firstCollectiveReady;
            caravan::EventSource secondCollectiveReady;
            auto firstCollectiveInput = std::make_shared<std::int32_t>(topology.rank + 1);
            auto secondCollectiveInput = std::make_shared<std::int32_t>(100 + topology.rank);
            auto firstCollectiveOutput = std::make_shared<std::int32_t>(-1);
            auto secondCollectiveOutput = std::make_shared<std::int32_t>(-1);
            caravan::AsyncScope collectiveScope;
            auto firstCollective = collectiveScope.spawnFuture<caravan::AllReduceResult>(collectiveLane.submit(
                caravan::asSender(firstCollectiveReady.event()),
                [&mpi, firstCollectiveInput, firstCollectiveOutput, communicator = cartesian.communicator]
                {
                    return caravan::mpi::allReduce(
                        mpi,
                        caravan::retain(std::as_bytes(std::span{firstCollectiveInput.get(), 1}), firstCollectiveInput),
                        caravan::retain(
                            std::as_writable_bytes(std::span{firstCollectiveOutput.get(), 1}),
                            firstCollectiveOutput),
                        caravan::ScalarType::int32,
                        caravan::ReduceOperation::sum,
                        communicator);
                }));
            auto secondCollective = collectiveScope.spawnFuture<caravan::AllReduceResult>(collectiveLane.submit(
                caravan::asSender(secondCollectiveReady.event()),
                [&mpi, secondCollectiveInput, secondCollectiveOutput, communicator = cartesian.communicator]
                {
                    return caravan::mpi::allReduce(
                        mpi,
                        caravan::retain(
                            std::as_bytes(std::span{secondCollectiveInput.get(), 1}),
                            secondCollectiveInput),
                        caravan::retain(
                            std::as_writable_bytes(std::span{secondCollectiveOutput.get(), 1}),
                            secondCollectiveOutput),
                        caravan::ScalarType::int32,
                        caravan::ReduceOperation::sum,
                        communicator);
                }));

            if(topology.rank % 2 == 0)
                secondCollectiveReady.setReady();
            else
                firstCollectiveReady.setReady();

            auto laneSendBuffer = std::make_shared<int>(topology.rank);
            auto laneReceiveBuffer = std::make_shared<int>(-1);
            caravan::AsyncScope laneScope;
            auto laneReceive = laneScope.spawnFuture<caravan::ReceiveResult>(caravan::mpi::receive(
                mpi,
                caravan::retain(std::as_writable_bytes(std::span{laneReceiveBuffer.get(), 1}), laneReceiveBuffer),
                caravan::Peer{topology.rank},
                caravan::MessageTag{916},
                cartesian.communicator));
            auto laneSend = laneScope.spawnFuture<caravan::SendResult>(caravan::mpi::send(
                mpi,
                caravan::retain(std::as_bytes(std::span{laneSendBuffer.get(), 1}), laneSendBuffer),
                caravan::Peer{topology.rank},
                caravan::MessageTag{916},
                cartesian.communicator));
            std::array laneTransfers{laneReceive.event(), laneSend.event()};
            caravan::whenAll(laneTransfers).wait();
            laneScope.join().wait();
            assert(*laneReceiveBuffer == topology.rank);

            if(topology.rank % 2 == 0)
                firstCollectiveReady.setReady();
            else
                secondCollectiveReady.setReady();
            firstCollective.result();
            secondCollective.result();
            assert(*firstCollectiveOutput == topology.size * (topology.size + 1) / 2);
            assert(*secondCollectiveOutput == topology.size * 100 + topology.size * (topology.size - 1) / 2);

            collectiveScope.join().wait();

            int const destination = (topology.rank + 1) % topology.size;
            int const source = (topology.rank + topology.size - 1) % topology.size;
            auto sentValue = std::make_shared<int>(topology.rank);
            auto receivedValues = std::make_shared<std::array<int, 2>>(std::array{-1, -1});
            caravan::AsyncScope transferScope;
            auto received = transferScope.spawnFuture<caravan::ReceiveResult>(caravan::mpi::receive(
                mpi,
                caravan::retain(std::as_writable_bytes(std::span{*receivedValues}), receivedValues),
                caravan::anyPeer,
                caravan::anyMessageTag,
                cartesian.communicator));
            auto sent = transferScope.spawnFuture<caravan::SendResult>(caravan::mpi::send(
                mpi,
                caravan::retain(std::as_bytes(std::span{sentValue.get(), 1}), sentValue),
                caravan::Peer{destination},
                caravan::MessageTag{17},
                cartesian.communicator));
            std::array transfers{received.event(), sent.event()};
            caravan::whenAll(transfers).wait();
            transferScope.join().wait();
            assert(receivedValues->front() == source);
            assert(receivedValues->back() == -1);
            assert(sent.result().bytes == sizeof(int));
            assert(received.result().source.value == source);
            assert(received.result().tag.value == 17);
            assert(received.result().bytes == sizeof(int));

            constexpr std::size_t largeMessageBytes = 1024u * 1024u;
            auto largeSendBuffer = std::make_shared<std::vector<std::byte>>(largeMessageBytes, std::byte{42});
            auto largeReceiveBuffer = std::make_shared<std::vector<std::byte>>(largeMessageBytes);
            caravan::AsyncScope largeTransferScope;
            auto largeReceive = largeTransferScope.spawnFuture<caravan::ReceiveResult>(caravan::mpi::receive(
                mpi,
                caravan::retain(std::span{*largeReceiveBuffer}, largeReceiveBuffer),
                caravan::Peer{source},
                caravan::MessageTag{18},
                cartesian.communicator));
            auto largeSend = largeTransferScope.spawnFuture<caravan::SendResult>(caravan::mpi::send(
                mpi,
                caravan::retain(std::span{*largeSendBuffer}, largeSendBuffer),
                caravan::Peer{destination},
                caravan::MessageTag{18},
                cartesian.communicator));
            std::array largeTransfers{largeReceive.event(), largeSend.event()};
            caravan::whenAll(largeTransfers).wait();
            largeTransferScope.join().wait();
            assert(largeReceive.result().bytes == largeMessageBytes);
            assert(largeReceiveBuffer->front() == std::byte{42});
            assert(largeReceiveBuffer->back() == std::byte{42});

            auto reductionInput
                = std::make_shared<std::array<std::int32_t, 3>>(std::array{topology.rank + 1, 1, topology.rank});
            auto reductionOutput
                = std::make_shared<std::array<std::int32_t, 3>>(std::array<std::int32_t, 3>{-1, -1, -1});
            auto reductionSender = caravan::mpi::allReduce(
                mpi,
                caravan::retain(std::as_bytes(std::span{*reductionInput}), reductionInput),
                caravan::retain(std::as_writable_bytes(std::span{*reductionOutput}), reductionOutput),
                caravan::ScalarType::int32,
                caravan::ReduceOperation::sum,
                cartesian.communicator);
            caravan::Promise<caravan::AllReduceResult> reductionOutputPromise;
            auto reductionResult = reductionOutputPromise.future();
            auto reductionOperation
                = std::move(reductionSender).connect(ValueReceiver<caravan::AllReduceResult>{reductionOutputPromise});
            assert((*reductionOutput)[0] == -1);
            reductionOperation.start();
            assert(reductionResult.result().elements == reductionInput->size());
            assert((*reductionOutput)[0] == topology.size * (topology.size + 1) / 2);
            assert((*reductionOutput)[1] == topology.size);
            assert((*reductionOutput)[2] == topology.size * (topology.size - 1) / 2);

            // Exact in-place all-reduce remains supported.
            std::int32_t inPlace = topology.rank + 1;
            auto const inPlaceResult = caravan::syncWait<caravan::AllReduceResult>(caravan::mpi::allReduce(
                mpi,
                std::as_bytes(std::span{&inPlace, 1}),
                std::as_writable_bytes(std::span{&inPlace, 1}),
                caravan::ScalarType::int32,
                caravan::ReduceOperation::sum,
                cartesian.communicator));
            assert(inPlaceResult.elements == 1u && inPlace == topology.size * (topology.size + 1) / 2);

            auto const emptyReduction = caravan::syncWait<caravan::ReduceResult>(caravan::mpi::reduce(
                mpi,
                std::span<std::byte const>{},
                std::span<std::byte>{},
                caravan::ScalarType::int32,
                caravan::ReduceOperation::sum,
                caravan::Peer{0},
                cartesian.communicator));
            assert(emptyReduction.elements == 0u);

            auto reduceInput = std::make_shared<std::int32_t>(topology.rank + 1);
            auto reduceOutput = std::make_shared<std::int32_t>(-1);
            auto reduceSender = caravan::mpi::reduce(
                mpi,
                caravan::retain(std::as_bytes(std::span{reduceInput.get(), 1}), reduceInput),
                caravan::retain(
                    topology.rank == 0 ? std::as_writable_bytes(std::span{reduceOutput.get(), 1})
                                       : std::span<std::byte>{},
                    reduceOutput),
                caravan::ScalarType::int32,
                caravan::ReduceOperation::sum,
                caravan::Peer{0},
                cartesian.communicator);
            caravan::Promise<caravan::ReduceResult> reduceOutputPromise;
            auto reduceResult = reduceOutputPromise.future();
            auto reduceOperation
                = std::move(reduceSender).connect(ValueReceiver<caravan::ReduceResult>{reduceOutputPromise});
            reduceOperation.start();
            assert(reduceResult.result().elements == 1u);
            if(topology.rank == 0)
                assert(*reduceOutput == topology.size * (topology.size + 1) / 2);

            // Non-root output is ignored even if it aliases the input; use a nonzero root too.
            auto const ignoredOutput = caravan::syncWait<caravan::ReduceResult>(caravan::mpi::reduce(
                mpi,
                std::as_bytes(std::span{reduceInput.get(), 1}),
                std::as_writable_bytes(
                    std::span{topology.rank == topology.size - 1 ? reduceOutput.get() : reduceInput.get(), 1}),
                caravan::ScalarType::int32,
                caravan::ReduceOperation::sum,
                caravan::Peer{topology.size - 1},
                cartesian.communicator));
            assert(ignoredOutput.elements == 1u);
            if(topology.rank == topology.size - 1)
                assert(*reduceOutput == topology.size * (topology.size + 1) / 2);

            auto gatherInput = std::make_shared<int>(topology.rank);
            auto gatherOutput = std::make_shared<std::vector<int>>(topology.size, -1);
            auto gatherSender = caravan::mpi::gather(
                mpi,
                caravan::retain(std::as_bytes(std::span{gatherInput.get(), 1}), gatherInput),
                caravan::retain(std::as_writable_bytes(std::span{*gatherOutput}), gatherOutput),
                caravan::Peer{0},
                cartesian.communicator);
            caravan::Promise<caravan::GatherResult> gatherOutputPromise;
            auto gatherResult = gatherOutputPromise.future();
            auto gatherOperation
                = std::move(gatherSender).connect(ValueReceiver<caravan::GatherResult>{gatherOutputPromise});
            gatherOperation.start();
            assert(gatherResult.result().bytes == (topology.rank == 0 ? gatherOutput->size() * sizeof(int) : 0u));
            if(topology.rank == 0)
                for(int rank = 0; rank < topology.size; ++rank)
                    assert((*gatherOutput)[rank] == rank);

            std::fill(gatherOutput->begin(), gatherOutput->end(), -1);
            auto const allGatherResult = caravan::syncWait<caravan::GatherResult>(caravan::mpi::allGather(
                mpi,
                caravan::retain(std::as_bytes(std::span{gatherInput.get(), 1}), gatherInput),
                caravan::retain(std::as_writable_bytes(std::span{*gatherOutput}), gatherOutput),
                cartesian.communicator));
            assert(allGatherResult.bytes == gatherOutput->size() * sizeof(int));
            for(int rank = 0; rank < topology.size; ++rank)
                assert((*gatherOutput)[rank] == rank);

            // Borrowed spans remain valid until syncWait completes; no shared owner is needed.
            int const borrowedRank = topology.rank;
            std::vector<int> borrowedRanks(topology.size, -1);
            auto const borrowedGatherResult = caravan::syncWait<caravan::GatherResult>(caravan::mpi::allGather(
                mpi,
                std::as_bytes(std::span<int const, 1>{&borrowedRank, 1}),
                std::as_writable_bytes(std::span{borrowedRanks}),
                cartesian.communicator));
            assert(borrowedGatherResult.bytes == borrowedRanks.size() * sizeof(int));
            for(int rank = 0; rank < topology.size; ++rank)
                assert(borrowedRanks[rank] == rank);

            auto gatherVInput = std::make_shared<std::vector<int>>(topology.rank + 1, topology.rank);
            std::vector<std::size_t> gatherVCounts(topology.size);
            std::vector<std::size_t> gatherVOffsets(topology.size);
            std::size_t gatherVElements = 0u;
            for(int rank = 0; rank < topology.size; ++rank)
            {
                gatherVOffsets[rank] = gatherVElements * sizeof(int);
                gatherVCounts[rank] = static_cast<std::size_t>(rank + 1) * sizeof(int);
                gatherVElements += static_cast<std::size_t>(rank + 1);
            }
            auto gatherVOutput = std::make_shared<std::vector<int>>(gatherVElements, -1);
            auto gatherVSender = caravan::mpi::gatherV(
                mpi,
                caravan::retain(std::as_bytes(std::span{*gatherVInput}), gatherVInput),
                caravan::retain(std::as_writable_bytes(std::span{*gatherVOutput}), gatherVOutput),
                gatherVCounts,
                gatherVOffsets,
                caravan::Peer{0},
                cartesian.communicator);
            caravan::Promise<caravan::GatherResult> gatherVOutputPromise;
            auto gatherVResult = gatherVOutputPromise.future();
            auto gatherVOperation
                = std::move(gatherVSender).connect(ValueReceiver<caravan::GatherResult>{gatherVOutputPromise});
            gatherVOperation.start();
            assert(gatherVResult.result().bytes == (topology.rank == 0 ? gatherVOutput->size() * sizeof(int) : 0u));
            if(topology.rank == 0)
                for(int rank = 0; rank < topology.size; ++rank)
                    for(std::size_t i = 0u; i < static_cast<std::size_t>(rank + 1); ++i)
                        assert((*gatherVOutput)[gatherVOffsets[rank] / sizeof(int) + i] == rank);

            std::atomic<bool> senderStarted = false;
            auto requestSender = caravan::mpi::request<int>(
                mpi,
                [&senderStarted, communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    senderStarted.store(true);
                    caravan::NativeRequestBatch batch(std::vector<MPI_Request>(1, MPI_REQUEST_NULL));
                    int const error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[0]);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("sender MPI_Ibarrier failed");
                    return batch;
                },
                [](std::span<MPI_Status const> statuses)
                {
                    assert(statuses.size() == 1u);
                    return 42;
                });
            caravan::Promise<int> senderOutput;
            auto senderResult = senderOutput.future();
            auto requestOperation = std::move(requestSender).connect(ValueReceiver<int>{senderOutput});
            assert(!senderStarted.load());
            requestOperation.start();
            assert(senderResult.result() == 42);
            assert(senderStarted.load());

            bool scopedSenderStarted = false;
            auto scopedSender = caravan::mpi::request<int>(
                mpi,
                [&scopedSenderStarted](caravan::NativeMpiContext&)
                {
                    scopedSenderStarted = true;
                    return caravan::NativeRequestBatch{};
                },
                [](std::span<MPI_Status const> statuses)
                {
                    assert(statuses.empty());
                    return 42;
                });
            assert(!scopedSenderStarted);
            caravan::RunLoop controlLoop;
            caravan::AsyncScope scope;
            std::thread::id scopedContinuationThread;
            auto scopedEvent = scope.spawn(
                std::move(scopedSender) | caravan::continuesOn(controlLoop.scheduler())
                | caravan::then(
                    [&](int value)
                    {
                        assert(value == 42);
                        scopedContinuationThread = std::this_thread::get_id();
                    }));
            std::thread controlThread([&controlLoop] { controlLoop.run(); });
            auto const controlThreadId = controlThread.get_id();
            scopedEvent.wait();
            controlLoop.finish();
            controlThread.join();
            scope.join().wait();
            assert(scopedSenderStarted);
            assert(scopedContinuationThread == controlThreadId);

            bool invokeStarted = false;
            auto invokeSender = caravan::mpi::invoke(
                mpi,
                [&invokeStarted, communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    invokeStarted = true;
                    int rank = -1;
                    int const error = MPI_Comm_rank(context.communicator(communicator), &rank);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("sender MPI_Comm_rank failed");
                    return rank;
                });
            caravan::Promise<int> invokeOutput;
            auto invokeResult = invokeOutput.future();
            auto invokeOperation = std::move(invokeSender).connect(ValueReceiver<int>{invokeOutput});
            assert(!invokeStarted);
            invokeOperation.start();
            assert(invokeResult.result() == cartesian.rank);
            assert(invokeStarted);

            auto receivedAfterBlocking = std::make_shared<int>(-1);
            auto sentAfterBlocking = std::make_shared<int>(topology.rank);
            constexpr int blockingTestTag = 919;
            auto pendingReceive = caravan::mpi::receive(
                mpi,
                caravan::retain(
                    std::as_writable_bytes(std::span{receivedAfterBlocking.get(), 1}),
                    receivedAfterBlocking),
                caravan::Peer{cartesian.rank},
                caravan::MessageTag{blockingTestTag},
                cartesian.communicator);
            caravan::AsyncScope activeScope;
            auto receiveAfterBlocking = activeScope.spawn(std::move(pendingReceive));

            auto blockingSender = caravan::mpi::invoke(
                mpi,
                [communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    int rank = -1;
                    int const error = MPI_Comm_rank(context.communicator(communicator), &rank);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("blocking sender MPI_Comm_rank failed");
                    return rank;
                });
            caravan::Promise<int> blockingOutput;
            auto blockingResult = blockingOutput.future();
            auto blockingOperation = std::move(blockingSender).connect(ValueReceiver<int>{blockingOutput});
            blockingOperation.start();
            assert(blockingResult.result() == cartesian.rank);
            assert(receiveAfterBlocking.state() == caravan::CompletionState::pending);

            auto matchingSend = caravan::mpi::send(
                mpi,
                caravan::retain(std::as_bytes(std::span{sentAfterBlocking.get(), 1}), sentAfterBlocking),
                caravan::Peer{cartesian.rank},
                caravan::MessageTag{blockingTestTag},
                cartesian.communicator);
            auto sendAfterBlocking = activeScope.spawn(std::move(matchingSend));
            std::array blockingEvents{receiveAfterBlocking, sendAfterBlocking};
            caravan::whenAll(blockingEvents).wait();
            activeScope.join().wait();
            assert(*receivedAfterBlocking == topology.rank);

            caravan::EventSource senderBarrierReady;
            caravan::AsyncScope barrierScope;
            auto senderBarrier = barrierScope.spawn(
                caravan::asSender(senderBarrierReady.event())
                | caravan::letValue([&mpi, communicator = cartesian.communicator]
                                    { return caravan::mpi::barrier(mpi, communicator); }));
            assert(senderBarrier.state() == caravan::CompletionState::pending);
            senderBarrierReady.setReady();
            senderBarrier.wait();
            barrierScope.join().wait();

            int borrowedReceive = -1;
            auto retainedSend = std::make_shared<int>(topology.rank);
            std::weak_ptr<int> retainedSendLifetime = retainedSend;
            auto borrowedReceiveSender = caravan::mpi::receive(
                mpi,
                std::as_writable_bytes(std::span{&borrowedReceive, 1}),
                caravan::Peer{cartesian.rank},
                caravan::MessageTag{920},
                cartesian.communicator);
            auto retainedSendSender = caravan::mpi::send(
                mpi,
                caravan::retain(std::as_bytes(std::span{retainedSend.get(), 1}), retainedSend),
                caravan::Peer{cartesian.rank},
                caravan::MessageTag{920},
                cartesian.communicator);
            retainedSend.reset();
            assert(!retainedSendLifetime.expired());
            caravan::AsyncScope lifetimeScope;
            auto borrowedReceiveEvent = lifetimeScope.spawn(std::move(borrowedReceiveSender));
            auto retainedSendEvent = lifetimeScope.spawn(std::move(retainedSendSender));
            std::array lifetimeEvents{borrowedReceiveEvent, retainedSendEvent};
            caravan::whenAll(lifetimeEvents).wait();
            lifetimeScope.join().wait();
            assert(borrowedReceive == topology.rank);

            auto nativeInput = std::make_shared<int>(topology.rank + 1);
            auto nativeOutput = std::make_shared<int>(0);
            auto const nativeReduction = caravan::syncWait<int>(caravan::mpi::request<int>(
                mpi,
                [nativeInput, nativeOutput, communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    caravan::NativeRequestBatch batch(
                        std::vector<MPI_Request>(2, MPI_REQUEST_NULL),
                        {nativeInput, nativeOutput});
                    int error = MPI_Iallreduce(
                        nativeInput.get(),
                        nativeOutput.get(),
                        1,
                        MPI_INT,
                        MPI_SUM,
                        context.communicator(communicator),
                        &batch.requests[0]);
                    if(error == MPI_SUCCESS)
                        error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[1]);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("native MPI request start failed");
                    return batch;
                },
                [nativeOutput,
                 communicator
                 = cartesian.communicator](caravan::NativeMpiContext& context, std::span<MPI_Status const> statuses)
                {
                    int rank = -1;
                    assert(MPI_Comm_rank(context.communicator(communicator), &rank) == MPI_SUCCESS);
                    assert(rank >= 0 && statuses.size() == 2u);
                    return *nativeOutput;
                }));
            assert(nativeReduction == topology.size * (topology.size + 1) / 2);

            caravan::syncWait(
                caravan::mpi::request<void>(
                    mpi,
                    [processMain](caravan::NativeMpiContext&)
                    {
                        assert(std::this_thread::get_id() != processMain);
                        return caravan::NativeRequestBatch{};
                    },
                    [](std::span<MPI_Status const> statuses) { assert(statuses.empty()); }));

            caravan::AsyncScope nativeScope;
            auto queuedBarrier = nativeScope.spawn(caravan::mpi::barrier(mpi, cartesian.communicator));
            auto const duplicatedCommunicator = caravan::syncWait<caravan::CommunicatorId>(caravan::mpi::invoke(
                mpi,
                [communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    MPI_Comm native = MPI_COMM_NULL;
                    int const error = MPI_Comm_dup(context.communicator(communicator), &native);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("native MPI_Comm_dup failed");
                    return context.adoptCommunicator(native);
                }));
            queuedBarrier.wait();
            nativeScope.join().wait();
            caravan::syncWait(caravan::mpi::barrier(mpi, duplicatedCommunicator));
            caravan::syncWait(caravan::mpi::destroyCommunicator(mpi, duplicatedCommunicator));
            caravan::syncWait(
                caravan::mpi::barrier(mpi, cartesian.communicator)
                | caravan::letValue([&mpi, communicator = cartesian.communicator]
                                    { return caravan::mpi::barrier(mpi, communicator); }));

            caravan::syncWait(caravan::mpi::destroyCommunicator(mpi, cartesian.communicator));

            // Hold the worker after starting a receive and queue its matching send. A rejected probe confirms
            // shutdown has begun before releasing the worker, so MpiRuntime must drain both requests on shutdown.
            constexpr int shutdownTag = 921;
            static_cast<void>(shutdownScope.spawn(
                caravan::mpi::receive(
                    mpi,
                    std::as_writable_bytes(std::span{&shutdownReceived, 1}),
                    caravan::Peer{topology.rank},
                    caravan::MessageTag{shutdownTag})));
            static_cast<void>(shutdownScope.spawn(
                caravan::mpi::invoke(
                    mpi,
                    [&](caravan::NativeMpiContext&)
                    {
                        shutdownGateStarted.store(true);
                        while(!releaseShutdownGate.load())
                            std::this_thread::yield();
                    })));
            while(!shutdownGateStarted.load())
                std::this_thread::yield();

            static_cast<void>(shutdownScope.spawn(
                caravan::mpi::send(
                    mpi,
                    std::as_bytes(std::span{&shutdownSent, 1}),
                    caravan::Peer{topology.rank},
                    caravan::MessageTag{shutdownTag})));
            shutdownReleaser = std::jthread(
                [&]
                {
                    for(;;)
                    {
                        if(!mpi.accepting())
                        {
                            releaseShutdownGate.store(true);
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                });
            return 0;
        });
    shutdownReleaser.join();
    shutdownScope.join().wait();
    assert(shutdownReceived == shutdownSent);
    return result;
}
