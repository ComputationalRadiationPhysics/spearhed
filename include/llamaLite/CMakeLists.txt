project(llamaLite)

add_library(llamaLite INTERFACE)

target_include_directories(llamaLite INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(llamaLite INTERFACE pmacc::pmacc)
