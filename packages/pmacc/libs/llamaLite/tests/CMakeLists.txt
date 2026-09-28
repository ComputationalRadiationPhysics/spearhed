if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.3)
    FetchContent_MakeAvailable(Catch2)
endif()

if(NOT Catch2_DIR)
    include(FetchContent)
    FetchContent_GetProperties(Catch2)
endif()
if(catch2_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
elseif(Catch2_DIR AND EXISTS "${Catch2_DIR}/Catch.cmake")
    list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()
include(Catch)

foreach(
    name
    IN
    ITEMS accessSet dynSoa iterate one set tagPath
)
    add_executable(llamaLite_test_${name} unit/${name}/${name}.cpp)
    target_link_libraries(llamaLite_test_${name} PRIVATE llamaLite Catch2::Catch2WithMain)
    catch_discover_tests(llamaLite_test_${name} TEST_PREFIX "llamaLite::")
endforeach()

add_subdirectory(standalone)
