# Use CONFIGURE_DEPENDS so cmake re-runs if files are added/removed
file(GLOB_RECURSE ACCSRCFILES CONFIGURE_DEPENDS "*.x.cpp")
file(GLOB_RECURSE SRCFILES CONFIGURE_DEPENDS "*.cpp")
list(REMOVE_ITEM SRCFILES ${ACCSRCFILES})

add_library(spearhed-hostonly STATIC ${SRCFILES})

target_link_libraries(spearhed-hostonly PUBLIC project_options MPI::MPI_CXX Boost::program_options llamaLite)

# Some host compiled libraries depend on std::filesystem,
# compatibility to std::experimental::filesystem is handled in pmacc.
target_link_libraries(spearhed-hostonly PRIVATE pmacc::filesystem)

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

include(GNUInstallDirs)
install(TARGETS spearhed RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
