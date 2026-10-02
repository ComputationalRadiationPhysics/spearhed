# Copyright 2015-2024 Erik Zenker, Alexander Grund
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.28)
project(PMacc LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Backend.cmake")

# The parent must prepare a compatible accelerator, Caravan's MPI/alpaka
# adapters, and llamaLite before adding this component. It can be a package root
# or a consumer; installed llamaLite can also be found below.
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka 3 CONFIG REQUIRED)
endif()
if(NOT TARGET caravan::mpi OR NOT TARGET caravan::alpaka)
    find_package(Caravan CONFIG REQUIRED COMPONENTS mpi alpaka)
endif()
if(NOT TARGET llamaLite::llamaLite)
    find_package(llamaLite CONFIG REQUIRED)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Boost 1.74 REQUIRED CONFIG COMPONENTS program_options)
find_package(Threads REQUIRED)
if(PMACC_BACKEND STREQUAL "GpuCuda" OR PMACC_BACKEND STREQUAL "GpuHip")
    if(NOT TARGET mallocMC::mallocMC)
        find_package(mallocMC 3 CONFIG REQUIRED)
    endif()
endif()

add_library(
    pmacc
    STATIC
    include/pmacc/communication/CommunicatorMPI.cpp
    include/pmacc/dataManagement/DataConnector.cpp
    include/pmacc/mappings/simulation/Filesystem.cpp
    include/pmacc/misc/splitString.cpp
    include/pmacc/pluginSystem/PluginConnector.cpp
    include/pmacc/simulationControl/signal.cpp
    include/pmacc/simulationControl/SimulationHelper.cpp
)
add_library(pmacc::pmacc ALIAS pmacc)
target_compile_features(pmacc PUBLIC cxx_std_20)
target_include_directories(pmacc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(
    pmacc
    PUBLIC alpaka::alpaka caravan::mpi caravan::alpaka llamaLite::llamaLite MPI::MPI_CXX Boost::program_options
)
if(PMACC_BACKEND STREQUAL "GpuCuda" OR PMACC_BACKEND STREQUAL "GpuHip")
    target_link_libraries(pmacc PUBLIC mallocMC::mallocMC)
endif()
target_compile_definitions(pmacc PUBLIC PMACC_BACKEND_${PMACC_BACKEND}=1 BOOST_NO_AUTO_PTR)
option(PMACC_ASYNC_QUEUES "Use asynchronous queues" ON)
if(PMACC_ASYNC_QUEUES)
    target_compile_definitions(pmacc PUBLIC PMACC_USE_ASYNC_QUEUES=1)
endif()
option(PMACC_BLOCKING_KERNEL "Synchronize after each kernel" OFF)
if(PMACC_BLOCKING_KERNEL)
    target_compile_definitions(pmacc PUBLIC PMACC_SYNC_KERNEL=1)
endif()
set(PMACC_VERBOSE 0 CACHE STRING "PMacc verbosity level")
target_compile_definitions(pmacc PUBLIC PMACC_VERBOSE_LVL=${PMACC_VERBOSE})
if(UNIX AND NOT APPLE)
    find_library(PMACC_RT_LIBRARY rt)
    if(PMACC_RT_LIBRARY)
        target_link_libraries(pmacc PRIVATE ${PMACC_RT_LIBRARY})
    endif()
endif()

# A target compiling alpaka headers must be finalized, including downstream consumers.
alpaka_finalize(pmacc)
option(PMACC_BUILD_TESTING "Build PMacc tests" ${PROJECT_IS_TOP_LEVEL})
if(PMACC_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installed PMacc is deferred: the pinned alpaka installed config cannot yet
# supply valid relocatable, exported accelerator targets. Do not export a
# partially working package.
