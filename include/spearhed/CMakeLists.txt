# Setup file: include/spearhed/param/setup.hpp is used by default.
# To override, pass -DSPEARHED_SETUP_FILE=/path/to/MySetup.hpp at configure time.
# The provided file is copied to the build directory and shadows the default via include order.
set(SPEARHED_SETUP_FILE
    ""
    CACHE FILEPATH
    "Path to a setup header to use instead of the default include/spearhed/setup.hpp"
)

if(SPEARHED_SETUP_FILE)
    if(NOT IS_ABSOLUTE "${SPEARHED_SETUP_FILE}")
        get_filename_component(SPEARHED_SETUP_FILE "${SPEARHED_SETUP_FILE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    endif()
    configure_file("${SPEARHED_SETUP_FILE}" "${CMAKE_BINARY_DIR}/include/spearhed/param/setup.hpp" COPYONLY)
    message(STATUS "Using custom setup file: ${SPEARHED_SETUP_FILE}")
else()
    message(STATUS "Using default setup file: include/spearhed/param/setup.hpp")
endif()

# Param file overrides: pass -DSPEARHED_PARAM_DIR=/path/to/params at configure time.
# Any .param file in that folder shadows the matching default in include/spearhed/param/.
set(SPEARHED_PARAM_DIR "" CACHE PATH "Folder of .param files that override defaults in include/spearhed/param/")

if(SPEARHED_PARAM_DIR)
    if(NOT IS_ABSOLUTE "${SPEARHED_PARAM_DIR}")
        get_filename_component(SPEARHED_PARAM_DIR "${SPEARHED_PARAM_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    endif()
    file(GLOB _param_overrides "${SPEARHED_PARAM_DIR}/*.param")
    foreach(_param_file ${_param_overrides})
        get_filename_component(_param_name "${_param_file}" NAME)
        configure_file("${_param_file}" "${CMAKE_BINARY_DIR}/include/spearhed/param/${_param_name}" COPYONLY)
        message(STATUS "Overriding param: ${_param_name} from ${_param_file}")
    endforeach()
endif()

# Use CONFIGURE_DEPENDS so cmake re-runs if files are added/removed
file(GLOB_RECURSE ACCSRCFILES CONFIGURE_DEPENDS "*.x.cpp")
file(GLOB_RECURSE SRCFILES CONFIGURE_DEPENDS "*.cpp")
list(REMOVE_ITEM SRCFILES ${ACCSRCFILES})

add_library(spearhed-hostonly STATIC ${SRCFILES})

target_link_libraries(spearhed-hostonly PUBLIC project_options MPI::MPI_CXX Boost::program_options llamaLite)

if(NOT WIN32)
    target_link_libraries(spearhed-hostonly PRIVATE m)
endif()

target_include_directories(
    spearhed-hostonly
    PUBLIC $<BUILD_INTERFACE:${_spearhed_ROOT_DIR}/include> $<INSTALL_INTERFACE:include>
)

alpaka_add_executable(spearhed ${ACCSRCFILES})

target_link_libraries(spearhed PUBLIC spearhed-hostonly)
target_link_libraries(spearhed PRIVATE pmacc::pmacc)

# Prepend the build include dir if either setup or any param files were overridden,
# so the copied files shadow the source-tree defaults
if(SPEARHED_SETUP_FILE OR SPEARHED_PARAM_DIR)
    target_include_directories(spearhed-hostonly BEFORE PRIVATE "${CMAKE_BINARY_DIR}/include")
    target_include_directories(spearhed BEFORE PRIVATE "${CMAKE_BINARY_DIR}/include")
endif()

include(GNUInstallDirs)
install(TARGETS spearhed RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
