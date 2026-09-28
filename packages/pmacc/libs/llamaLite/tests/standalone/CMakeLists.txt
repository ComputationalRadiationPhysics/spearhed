cmake_minimum_required(VERSION 3.25)
project(llamaLiteStandalone LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
endif()

if(NOT TARGET llamaLite)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." llamaLite)
endif()
add_executable(llamaLiteStandalone main.cpp)
target_link_libraries(llamaLiteStandalone PRIVATE llamaLite)
add_test(NAME llamaLite_standalone COMMAND llamaLiteStandalone)
