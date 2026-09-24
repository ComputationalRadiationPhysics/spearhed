project(llamaLite)

add_library(llamaLite INTERFACE)

target_compile_options(
    llamaLite
    INTERFACE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-fsized-deallocation>
)

target_include_directories(
    llamaLite
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_compile_features(llamaLite INTERFACE cxx_std_20)
