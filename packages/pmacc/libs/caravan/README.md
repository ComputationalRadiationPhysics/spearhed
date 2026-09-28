# Caravan

Independent C++20 async library. Configure this component directly or use
`add_subdirectory`. Its `caravan::core` target requires Threads only.
`CARAVAN_BUILD_MPI` enables `caravan::mpi` and MPI; `CARAVAN_BUILD_ALPAKA`
enables the source-tree `caravan::alpaka` adapter and needs alpaka3. Tests are
controlled by `CARAVAN_BUILD_TESTING` (on for direct top-level builds).

An installation exports `caravan::core`, and `caravan::mpi` if built. A core
consumer uses `find_package(Caravan CONFIG REQUIRED)` without MPI discovery;
an MPI consumer requests `find_package(Caravan CONFIG REQUIRED COMPONENTS mpi)`.
The installed alpaka adapter is intentionally unavailable until alpaka's
installed-target and include-path issues are resolved.
