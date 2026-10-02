cmake_minimum_required(VERSION 3.25)
project(llamaLite LANGUAGES CXX)

include(GNUInstallDirs)
add_library(llamaLite-headers INTERFACE)
add_library(llamaLite::llamaLite ALIAS llamaLite-headers)
set_target_properties(llamaLite-headers PROPERTIES EXPORT_NAME llamaLite)

target_compile_options(
    llamaLite-headers
    INTERFACE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fsized-deallocation>
)

target_include_directories(
    llamaLite-headers
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(llamaLite-headers INTERFACE cxx_std_20)

include(CMakePackageConfigHelpers)
install(TARGETS llamaLite-headers EXPORT llamaLiteTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/llamaLite)
install(EXPORT llamaLiteTargets NAMESPACE llamaLite:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llamaLite)
configure_package_config_file(
    cmake/llamaLiteConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/llamaLiteConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llamaLite
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llamaLiteConfig.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llamaLite)

option(LLAMALITE_BUILD_TESTING "Build llamaLite's own tests" ${PROJECT_IS_TOP_LEVEL})
if(LLAMALITE_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
