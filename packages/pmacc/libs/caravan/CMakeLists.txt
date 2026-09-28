# This file is part of Caravan.
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.28)
project(Caravan LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

add_library(caravan-core INTERFACE)
add_library(caravan::core ALIAS caravan-core)
set_target_properties(caravan-core PROPERTIES EXPORT_NAME core)
target_compile_features(caravan-core INTERFACE cxx_std_20)
target_include_directories(
    caravan-core
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(caravan-core INTERFACE Threads::Threads)

option(CARAVAN_BUILD_MPI "Build the Caravan MPI adapter" ${PROJECT_IS_TOP_LEVEL})
if(CARAVAN_BUILD_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(
        caravan-mpi
        STATIC
        include/caravan/mpi/communicator.cpp
        include/caravan/mpi/context.cpp
        include/caravan/mpi/operations.cpp
        include/caravan/mpi/runtime.cpp
    )
    add_library(caravan::mpi ALIAS caravan-mpi)
    set_target_properties(caravan-mpi PROPERTIES EXPORT_NAME mpi)
    target_link_libraries(caravan-mpi PUBLIC caravan::core MPI::MPI_CXX)
    target_compile_features(caravan-mpi PUBLIC cxx_std_20)
    target_include_directories(
        caravan-mpi
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

option(CARAVAN_BUILD_ALPAKA "Build the Caravan alpaka adapter" OFF)
if(CARAVAN_BUILD_ALPAKA)
    if(NOT TARGET alpaka::alpaka)
        find_package(alpaka 3 CONFIG REQUIRED)
    endif()
    add_library(caravan-alpaka INTERFACE)
    add_library(caravan::alpaka ALIAS caravan-alpaka)
    set_target_properties(caravan-alpaka PROPERTIES EXPORT_NAME alpaka)
    target_link_libraries(caravan-alpaka INTERFACE caravan::core alpaka::alpaka)
    target_include_directories(
        caravan-alpaka
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

option(CARAVAN_BUILD_TESTING "Build Caravan tests" ${PROJECT_IS_TOP_LEVEL})
if(CARAVAN_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

# The alpaka adapter is source-tree only until alpaka's installed config and
# target-export behaviour allow a reliable installed consumer.
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} PATTERN "*.cpp" EXCLUDE)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/Caravan)
set(_caravan_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/Caravan)
foreach(component IN ITEMS core mpi)
    if(TARGET caravan-${component})
        install(TARGETS caravan-${component} EXPORT Caravan${component}Targets)
        install(EXPORT Caravan${component}Targets NAMESPACE caravan:: DESTINATION ${_caravan_cmake_dir})
    endif()
endforeach()
configure_package_config_file(
    cmake/CaravanConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CaravanConfig.cmake"
    INSTALL_DESTINATION ${_caravan_cmake_dir}
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CaravanConfig.cmake" DESTINATION ${_caravan_cmake_dir})
