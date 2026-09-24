#
# Copyright 2025 René Widera, Simeon Ehrig
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################
cmake_minimum_required(VERSION 3.25)

################################################################################
# Install Catch2
################################################################################

include(FetchContent)
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.3)
FetchContent_MakeAvailable(Catch2)
target_compile_features(Catch2 PUBLIC cxx_std_20)
include(Catch)

# hide Catch2 cmake variables by default in cmake gui
get_cmake_property(variables VARIABLES)
foreach(var ${variables})
    if(var MATCHES "^CATCH_")
        mark_as_advanced(${var})
    endif()
endforeach()

################################################################################
# Add subdirectories.
################################################################################

#------------------------------------------------------------------------------
# Search for sub folders in the given folder.
# Creates a Catch2 test executable for each .cpp file in each sub folder.
# If a include folder is located in a sub folder, add it as include path to the
# target.
# Args:
#  - target_prefix: Prefix for the target name. The name of the executable test
#    file is formed with target_prefix followed by the source file name without
#    extension.
#  - layer_label: Short label prepended to every CTest test name so tests can be
#    filtered by layer (e.g. ctest -R llamaLite).
#  - folder: Takes all .cpp files in the specified folder and creates an test
#    executable for each.
function(add_test_target target_prefix layer_label folder)
    message(DEBUG "test target prefix: ${target_prefix}")

    file(GLOB test_folders CONFIGURE_DEPENDS ${folder})
    foreach(test_folder ${test_folders})
        if(IS_DIRECTORY ${test_folder})
            message(DEBUG "test folder: ${test_folder}")

            file(GLOB test_sources "${test_folder}/**")

            # iterate over all sub folders
            foreach(source_file ${test_sources})
                # the include folder is a special case, therefore ignore it
                if(source_file STREQUAL "${test_folder}/include")
                    continue()
                endif()

                get_filename_component(source_file_extension ${source_file} EXT)
                get_filename_component(source_file_name ${source_file} NAME_WE)
                # get absolute path of the folder of the source file
                get_filename_component(test_folder_name ${source_file} DIRECTORY)
                # get name of the folder
                get_filename_component(test_folder_name ${test_folder_name} NAME)

                if("${source_file_extension}" STREQUAL ".cpp")
                    set(_test_target_name "${target_prefix}${test_folder_name}_${source_file_name}")
                    message(DEBUG "  add test target ${_test_target_name}: ${source_file}")

                    alpaka_add_executable(${_test_target_name} ${source_file})
                    get_filename_component(src_dir ${source_file} DIRECTORY)
                    target_include_directories(${_test_target_name} PRIVATE ${src_dir})
                    if(EXISTS "${test_folder}/include")
                        message(STATUS "  add include folder: ${test_folder}/include")
                        target_include_directories(${_test_target_name} PRIVATE ${test_folder}/include)
                    endif()

                    target_link_libraries(
                        ${_test_target_name}
                        PRIVATE Catch2::Catch2WithMain alpaka::alpaka spearhed-hostonly pmacc::pmacc
                    )
                    catch_discover_tests(${_test_target_name} TEST_PREFIX "${layer_label}::")
                endif()
            endforeach()
        endif()
    endforeach()
endfunction()

add_test_target(unit_test_llama_ "llamaLite" ${CMAKE_CURRENT_SOURCE_DIR}/llama_lite/unit/**)
add_test_target(unit_test_spmacc_ "spmacc" ${CMAKE_CURRENT_SOURCE_DIR}/spmacc/unit/**)
add_test_target(unit_test_spearhed_ "spearhed" ${CMAKE_CURRENT_SOURCE_DIR}/spearhed/unit/**)
target_include_directories(unit_test_spearhed_MappingSetups_MappingSetups PRIVATE ${PROJECT_SOURCE_DIR}/share)

# Repeatable Phase-0 spatial baseline. This is built with the tests but is not registered with
# CTest: timings are hardware-dependent and are recorded manually in the benchmark notes.
alpaka_add_executable(
    benchmark_spmacc_spatial_baseline
    ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/spatial/SpatialBaseline.cpp
)
target_link_libraries(benchmark_spmacc_spatial_baseline PRIVATE alpaka::alpaka spearhed-hostonly pmacc::pmacc)

# Per-test DIM2 override: shadow the global dimension.param with a 2D version
# for the BoundaryConditions test only. BEFORE ensures the local file wins.
target_include_directories(
    unit_test_spearhed_BoundaryConditions_BoundaryConditions
    BEFORE
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/spearhed/unit/BoundaryConditions/param
)

## Share a setup among selected tests via the spearhed test-setup directory.
set(_setup_dir ${CMAKE_CURRENT_SOURCE_DIR}/spearhed/unit/setups)
foreach(
    t
    unit_test_spmacc_FindNeighbourRegions_FindNeighbourRegions
    unit_test_spmacc_ParticleParticleInteraction_ParticleParticleInteraction
    unit_test_spmacc_RegionBoundsUpdate_RegionBoundsUpdate
    unit_test_spmacc_CopyParticlesToHost_CopyParticlesToHost
    unit_test_spmacc_HierarchyForEach_HierarchyForEach
    unit_test_spearhed_PushParticles_PushParticles
    unit_test_spearhed_DensitySummation_DensitySummation
    unit_test_spearhed_HydroForces_HydroForces
    unit_test_spearhed_BoundaryConditions_BoundaryConditions
)
    target_include_directories(${t} PRIVATE ${_setup_dir})
endforeach()
