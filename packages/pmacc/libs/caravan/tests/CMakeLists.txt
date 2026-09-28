# This file is part of Caravan.
# SPDX-License-Identifier: MPL-2.0

foreach(name IN ITEMS core retained graph)
    add_executable(caravan-${name}-test ${name}.cpp)
    target_link_libraries(caravan-${name}-test PRIVATE caravan::core)
    add_test(NAME caravan-${name} COMMAND caravan-${name}-test)
endforeach()

if(TARGET caravan::alpaka)
    foreach(name IN ITEMS alpaka alpaka_completion event_pool shared_queue_pool)
        add_executable(caravan-${name}-test ${name}.cpp)
        target_link_libraries(caravan-${name}-test PRIVATE caravan::alpaka)
        alpaka_finalize(caravan-${name}-test)
        add_test(NAME caravan-${name} COMMAND caravan-${name}-test)
        set_tests_properties(caravan-${name} PROPERTIES TIMEOUT 30)
    endforeach()
    add_test(NAME caravan-alpaka-fence-failure COMMAND caravan-alpaka_completion-test --fatal-fence)
    set_tests_properties(caravan-alpaka-fence-failure PROPERTIES TIMEOUT 30)
    if(TARGET caravan::mpi)
        add_executable(caravan-alpaka-mpi-test alpaka_mpi.cpp)
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

option(CARAVAN_BUILD_STDEXEC_SPIKE "Build the optional NVIDIA/stdexec interoperability spike" OFF)
if(CARAVAN_BUILD_STDEXEC_SPIKE)
    if(NOT TARGET caravan::alpaka OR NOT TARGET caravan::mpi)
        message(FATAL_ERROR "CARAVAN_BUILD_STDEXEC_SPIKE requires Caravan alpaka and MPI")
    endif()
    find_path(CARAVAN_STDEXEC_INCLUDE_DIR NAMES stdexec/execution.hpp REQUIRED)
    add_executable(caravan-stdexec-spike stdexec.cpp)
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
    add_executable(caravan-mpi-header-test mpi_header.cpp)
    target_link_libraries(caravan-mpi-header-test PRIVATE caravan::mpi)
    add_test(NAME caravan-mpi-header COMMAND caravan-mpi-header-test)

    add_executable(caravan-mpi-test mpi.cpp)
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
