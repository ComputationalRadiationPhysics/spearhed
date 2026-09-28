# llamaLite

Independent header-only C++20 library. Public headers live under
`include/llamaLite/`. Configure this directory directly, or add it with
`add_subdirectory`. Link the `llamaLite::llamaLite` target. Standalone tests
default to on only when this is the top-level CMake project; control them with
`LLAMALITE_BUILD_TESTING`.

To consume an installation, use `find_package(llamaLite CONFIG REQUIRED)`
and link `llamaLite::llamaLite`. Its standalone smoke test is in
`tests/standalone/`. No PMacc or accelerator is required.
