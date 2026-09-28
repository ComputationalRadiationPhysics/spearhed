cmake_minimum_required(VERSION 3.25)
project(llamaLite LANGUAGES CXX)

add_library(llamaLite INTERFACE)

target_compile_options(
    llamaLite
    INTERFACE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fsized-deallocation>
)

target_include_directories(
    llamaLite
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_compile_features(llamaLite INTERFACE cxx_std_20)

option(LLAMALITE_BUILD_TESTING "Build llamaLite's own tests" ${PROJECT_IS_TOP_LEVEL})
if(LLAMALITE_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
