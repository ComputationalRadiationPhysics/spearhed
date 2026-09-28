# This file is part of Caravan.
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.28)
project(Caravan LANGUAGES CXX)

add_library(caravan-core INTERFACE)
add_library(caravan::core ALIAS caravan-core)
target_compile_features(caravan-core INTERFACE cxx_std_20)
target_include_directories(
    caravan-core
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
option(CARAVAN_BUILD_MPI "Build the Caravan MPI backend" ON)
if(CARAVAN_BUILD_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(caravan-mpi STATIC mpi/communicator.cpp mpi/context.cpp mpi/operations.cpp mpi/runtime.cpp)
    add_library(caravan::mpi ALIAS caravan-mpi)
    target_link_libraries(caravan-mpi PUBLIC caravan::core MPI::MPI_CXX PRIVATE Threads::Threads)
endif()

if(TARGET alpaka::alpaka)
    set(_caravan_alpaka_default ON)
else()
    set(_caravan_alpaka_default OFF)
endif()
option(CARAVAN_BUILD_ALPAKA "Build the Caravan alpaka adapter" ${_caravan_alpaka_default})
unset(_caravan_alpaka_default)
if(CARAVAN_BUILD_ALPAKA)
    if(NOT TARGET alpaka::alpaka)
        find_package(alpaka REQUIRED)
    endif()
    add_library(caravan-alpaka INTERFACE)
    add_library(caravan::alpaka ALIAS caravan-alpaka)
    target_link_libraries(caravan-alpaka INTERFACE caravan::core alpaka::alpaka Threads::Threads)
endif()

option(CARAVAN_BUILD_TESTING "Build Caravan tests" ${PROJECT_IS_TOP_LEVEL})
option(CARAVAN_BUILD_STDEXEC_SPIKE "Build the optional NVIDIA/stdexec interoperability spike" OFF)
if(CARAVAN_BUILD_TESTING)
    enable_testing()
    add_executable(caravan-core-test test/core.cpp)
    target_link_libraries(caravan-core-test PRIVATE caravan::core Threads::Threads)
    add_test(NAME caravan-core COMMAND caravan-core-test)

    add_executable(caravan-retained-test test/retained.cpp)
    target_link_libraries(caravan-retained-test PRIVATE caravan::core)
    add_test(NAME caravan-retained COMMAND caravan-retained-test)

    add_executable(caravan-graph-test test/graph.cpp)
    target_link_libraries(caravan-graph-test PRIVATE caravan::core Threads::Threads)
    add_test(NAME caravan-graph COMMAND caravan-graph-test)

    if(TARGET caravan::alpaka)
        add_executable(caravan-alpaka-test test/alpaka.cpp)
        target_link_libraries(caravan-alpaka-test PRIVATE caravan::alpaka)
        alpaka_finalize(caravan-alpaka-test)
        add_test(NAME caravan-alpaka COMMAND caravan-alpaka-test)
        set_tests_properties(caravan-alpaka PROPERTIES TIMEOUT 30)

        add_executable(caravan-alpaka-completion-test test/alpaka_completion.cpp)
        target_link_libraries(caravan-alpaka-completion-test PRIVATE caravan::alpaka)
        alpaka_finalize(caravan-alpaka-completion-test)
        add_test(NAME caravan-alpaka-completion COMMAND caravan-alpaka-completion-test)
        add_test(NAME caravan-alpaka-fence-failure COMMAND caravan-alpaka-completion-test --fatal-fence)
        set_tests_properties(caravan-alpaka-completion caravan-alpaka-fence-failure PROPERTIES TIMEOUT 30)

        add_executable(caravan-event-pool-test test/event_pool.cpp)
        target_link_libraries(caravan-event-pool-test PRIVATE caravan::alpaka)
        alpaka_finalize(caravan-event-pool-test)
        add_test(NAME caravan-event-pool COMMAND caravan-event-pool-test)
        set_tests_properties(caravan-event-pool PROPERTIES TIMEOUT 30)

        add_executable(caravan-shared-queue-pool-test test/shared_queue_pool.cpp)
        target_link_libraries(caravan-shared-queue-pool-test PRIVATE caravan::alpaka)
        alpaka_finalize(caravan-shared-queue-pool-test)
        add_test(NAME caravan-shared-queue-pool COMMAND caravan-shared-queue-pool-test)
        set_tests_properties(caravan-shared-queue-pool PROPERTIES TIMEOUT 30)

        if(TARGET caravan::mpi)
            add_executable(caravan-alpaka-mpi-test test/alpaka_mpi.cpp)
            target_link_libraries(caravan-alpaka-mpi-test PRIVATE caravan::alpaka caravan::mpi)
            alpaka_finalize(caravan-alpaka-mpi-test)
            add_test(
                NAME caravan-alpaka-mpi
                COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:caravan-alpaka-mpi-test> ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(caravan-alpaka-mpi PROPERTIES PROCESSORS 1 TIMEOUT 30)
        endif()
    endif()

    if(CARAVAN_BUILD_STDEXEC_SPIKE)
        if(NOT TARGET caravan::alpaka OR NOT TARGET caravan::mpi)
            message(FATAL_ERROR "CARAVAN_BUILD_STDEXEC_SPIKE requires Caravan alpaka and MPI")
        endif()
        find_path(CARAVAN_STDEXEC_INCLUDE_DIR NAMES stdexec/execution.hpp REQUIRED)
        add_executable(caravan-stdexec-spike test/stdexec.cpp)
        target_include_directories(caravan-stdexec-spike PRIVATE ${CARAVAN_STDEXEC_INCLUDE_DIR})
        target_link_libraries(caravan-stdexec-spike PRIVATE caravan::alpaka caravan::mpi)
        alpaka_finalize(caravan-stdexec-spike)
        add_test(
            NAME caravan-stdexec-spike
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:caravan-stdexec-spike>
                ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(caravan-stdexec-spike PROPERTIES PROCESSORS 1 TIMEOUT 30)
    endif()

    if(TARGET caravan::mpi)
        add_executable(caravan-mpi-header-test test/mpi_header.cpp)
        target_link_libraries(caravan-mpi-header-test PRIVATE caravan::mpi)
        add_test(NAME caravan-mpi-header COMMAND caravan-mpi-header-test)

        add_executable(caravan-mpi-test test/mpi.cpp)
        target_link_libraries(caravan-mpi-test PRIVATE caravan::mpi)
        foreach(ranks IN ITEMS 1 2 4)
            add_test(
                NAME caravan-mpi-${ranks}-ranks
                COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:caravan-mpi-test> ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(caravan-mpi-${ranks}-ranks PROPERTIES PROCESSORS ${ranks} TIMEOUT 30)
        endforeach()
    endif()
endif()
