# Copyright 2013-2024 Rene Widera, Axel Huebl, Tapish Narwal
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
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.25)

################################################################################
# Project
################################################################################

project(SPEARHED)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif()

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")

# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

###############################################################################
# Language Flags
###############################################################################

# enforce C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

################################################################################
# Find PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/picongpu/include/pmacc")

################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
list(PREPEND HOST_LIBS MPI::MPI_CXX)

################################################################################
# Find Boost
################################################################################

find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options)
if(TARGET Boost::boost)
    list(PREPEND HOST_LIBS Boost::boost Boost::program_options)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    list(PREPEND HOST_LIBS ${Boost_LIBRARIES})
endif()

# ################################################################################
# # Build type (debug, release)
# ################################################################################

option(SPEARHED_RELEASE "Build release version, disables all runtime asserts" OFF)
if(SPEARHED_RELEASE)
    message(STATUS "Release version")
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
else(SPEARHED_RELEASE)
    message(STATUS "Debug version")
    set(CMAKE_BUILD_TYPE Debug)
endif(SPEARHED_RELEASE)

################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    # new warning in gcc 4.8 (flag ignored in previous version)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
    # ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    # PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()

################################################################################
# Compile & Link SPEARHED
################################################################################

include_directories(..)

# files with accelerator code (kernel and API calls)
file(GLOB_RECURSE ACCSRCFILES "*.x.cpp")

# host-only sources
file(GLOB_RECURSE SRCFILES "*.cpp")
list(REMOVE_ITEM SRCFILES ${ACCSRCFILES})

add_library(spearhed-hostonly STATIC ${SRCFILES})
target_link_libraries(spearhed-hostonly PUBLIC ${HOST_LIBS})
# Some host compiled libraries depend on std::filesystem,
# compatibility to std::experimental::filesystem is handled in pmacc.
target_link_libraries(spearhed-hostonly PRIVATE pmacc::filesystem)

alpaka_add_executable(spearhed
     ${ACCSRCFILES}
)

target_link_libraries(spearhed PUBLIC ${HOST_LIBS} spearhed-hostonly)
target_link_libraries(spearhed PRIVATE pmacc::pmacc)

################################################################################
# Math
################################################################################

if(NOT WIN32)
    # automatically added on windows
    target_link_libraries(spearhed PRIVATE m)
endif()

################################################################################
# Install SPEARHED
################################################################################

install(TARGETS spearhed RUNTIME DESTINATION bin)
