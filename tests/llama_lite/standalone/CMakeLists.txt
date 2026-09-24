cmake_minimum_required(VERSION 3.25)
project(llamaLiteStandalone LANGUAGES CXX)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../include/llamaLite" llamaLite)
add_executable(llamaLiteStandalone main.cpp)
target_link_libraries(llamaLiteStandalone PRIVATE llamaLite)
