include(FetchContent)
if(NOT TARGET Catch2::Catch2)
    FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.3)
    FetchContent_MakeAvailable(Catch2)
endif()

file(GLOB_RECURSE _pmacc_test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*UT.cpp")
foreach(dim IN ITEMS 2 3)
    foreach(source IN LISTS _pmacc_test_sources)
        get_filename_component(name "${source}" NAME_WE)
        set(target "pmacc-${name}-${dim}D")
        add_executable(${target} "${source}" TestMain.cpp)
        target_compile_definitions(${target} PRIVATE TEST_DIM=${dim})
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(${target} PRIVATE Catch2::Catch2 pmacc::pmacc)
        alpaka_finalize(${target})
        add_test(
            NAME ${target}
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>
                ${MPIEXEC_POSTFLAGS}
        )
        if(name STREQUAL "CommunicatorMPIUT")
            add_test(
                NAME ${target}-4-ranks
                COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>
                    ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(${target}-4-ranks PROPERTIES PROCESSORS 4 TIMEOUT 30)
        endif()
    endforeach()
endforeach()
add_subdirectory(random)
