include(FetchContent)
if(NOT TARGET Catch2::Catch2)
    FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.3)
    FetchContent_MakeAvailable(Catch2)
endif()

# These sources do not depend on the environment dimension. Some exercise multiple
# dimensions internally (e.g. memoryUT), so compiling them for each TEST_DIM adds no coverage.
set(_pmacc_dimension_independent_sources
    async/ContextUT.cpp
    async/OperationsUT.cpp
    lockstep/lockstepUT.cpp
    math/ComplexUT.cpp
    memory/memoryUT.cpp
    memory/STLTupleUT.cpp
    particles/memory/SuperCellUT.cpp
    vector/vectorUT.cpp
)

# Keep discovery for new tests; sources not listed above retain 2D and 3D coverage.
file(
    GLOB_RECURSE _pmacc_dimension_dependent_sources
    CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*UT.cpp"
)
list(REMOVE_ITEM _pmacc_dimension_dependent_sources ${_pmacc_dimension_independent_sources})

function(pmacc_add_unit_test target source label)
    add_executable(${target} "${source}" TestMain.cpp)
    get_filename_component(source_dir "${source}" DIRECTORY)
    target_include_directories(
        ${target}
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}"
    )
    target_link_libraries(${target} PRIVATE Catch2::Catch2 pmacc::pmacc)
    alpaka_finalize(${target})
    add_test(
        NAME ${target}
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(${target} PROPERTIES LABELS "${label}")
endfunction()

foreach(source IN LISTS _pmacc_dimension_independent_sources)
    get_filename_component(name "${source}" NAME_WE)
    pmacc_add_unit_test("pmacc-${name}" "${source}" dimension-independent)
endforeach()

foreach(dim IN ITEMS 2 3)
    foreach(source IN LISTS _pmacc_dimension_dependent_sources)
        get_filename_component(name "${source}" NAME_WE)
        set(target "pmacc-${name}-${dim}D")
        pmacc_add_unit_test(${target} "${source}" dimension-dependent)
        target_compile_definitions(${target} PRIVATE TEST_DIM=${dim})
        if(name STREQUAL "CommunicatorMPIUT")
            add_test(
                NAME ${target}-4-ranks
                COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>
                    ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(
                ${target}-4-ranks
                PROPERTIES
                    PROCESSORS 4
                    TIMEOUT 30
                    ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1"
                    LABELS dimension-dependent
            )
        endif()
    endforeach()
endforeach()
add_subdirectory(random)
