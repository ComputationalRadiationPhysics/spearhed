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

project(spearhed)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif()

# Set helper variables
set(_spearhed_ROOT_DIR ${spearhed_SOURCE_DIR})
set(_spearhed_TESTING_DIR "${_spearhed_ROOT_DIR}/tests")
set(_spearhed_INC_DIR "${_spearhed_ROOT_DIR}/include/spearhed")

################################################################################
# Build Configuration & C++ Standard
################################################################################

# Default to Release for performance if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Enable Interprocedural Optimization (LTO)
option(SPEARHED_ENABLE_IPO "Enable IPO/LTO")

if(SPEARHED_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_output)
    if(_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "IPO is not supported: ${_ipo_output}")
    endif()
endif()

# interface library for common compile options
add_library(project_options INTERFACE)

# alpaka/PMacc scripts require these global variables to be set
# BEFORE find_package is called.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_target_properties(
    project_options
    PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
)

target_compile_options(
    project_options
    INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wno-c2y-extensions
        -Wno-sign-conversion
        -Wno-unknown-pragmas
        -Wno-unused-parameter
        -Wno-unused-local-typedefs>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,IntelLLVM>>:-march=native
        -mtune=native>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4
        /permissive->
        $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-Wall
        -Wextra>
)

################################################################################
# Dependencies
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/picongpu/include/pmacc")
find_package(MPI REQUIRED)
find_package(Boost 1.74.0 REQUIRED CONFIG COMPONENTS program_options)

add_subdirectory(include/llamaLite)
################################################################################
# Compile & Link SPEARHED
################################################################################

add_subdirectory(include/spearhed)

################################################################################
# Testing and Subdirectories
################################################################################

option(SPEARHED_TESTING "Enable/Disable testing" OFF)

if(SPEARHED_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory("${_spearhed_TESTING_DIR}")
endif()
