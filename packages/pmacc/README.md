# PMacc source package

This directory is intended to become a standalone repository root.

## Build from source

Initialize `packages/pmacc/thirdParty/alpaka` and `mallocMC`, then configure:

```sh
cmake -S packages/pmacc -B "$HOME/scratch/pmacc-package" -DPMACC_BACKEND=CpuSerial
cmake --build "$HOME/scratch/pmacc-package" -j 4
```

The package root orchestrates PMacc, Caravan and llamaLite. To configure only
Caravan/llamaLite without alpaka, set `PMACC_PACKAGE_BUILD_PMACC=OFF` and
`CARAVAN_BUILD_ALPAKA=OFF`. Set `CARAVAN_BUILD_MPI=OFF` as well only if MPI is
also unwanted. Each component has its own CMake entry point under `libs/`.
PMacc requires `alpaka::alpaka`, `caravan::mpi`, `caravan::alpaka`, and
`llamaLite::llamaLite`. The package root configures llamaLite before PMacc and
builds Caravan's adapters when needed. A standalone PMacc component requires its
parent to provide these targets or make them discoverable with `find_package`;
llamaLite is linked publicly as a declared dependency for planned PMacc integration.
If `PMACC_PACKAGE_BUILD_LLAMALITE=OFF`, the package root searches for an installed
llamaLite package.

### Embedding from a parent project

The ergonomic source-tree workflow is to select the backend once at the
common CMake ancestor, add the package, and link the public target. Configure
with the source package path and desired backend:

```sh
cmake -S . -B build -DPMACC_SOURCE_DIR=/path/to/packages/pmacc -DPMACC_BACKEND=CpuSerial
```

Then use ordinary CMake target commands in the parent project:

```cmake
cmake_minimum_required(VERSION 3.28)
project(MySimulation LANGUAGES CXX)

include("${PMACC_SOURCE_DIR}/cmake/PMaccBackend.cmake")
add_subdirectory("${PMACC_SOURCE_DIR}" pmacc)

add_executable(simulation main.cpp)
target_link_libraries(simulation PRIVATE pmacc::pmacc)
pmacc_finalize(simulation)
```

Here `PMACC_SOURCE_DIR` is the path to this package. Set `PMACC_BACKEND`
(`CpuSerial`, `CpuOmpBlocks`, `CpuTbbBlocks`, `GpuCuda`, `GpuHip`, or `OneApi`)
before including the helper. The package root builds Caravan's MPI and alpaka
adapters along with PMacc. Targets that compile alpaka/PMacc headers must link
their dependencies and call `pmacc_finalize(target)` after linking. This macro
wraps alpaka's finalizer and retains its caller-scope behavior. Backend and
CUDA/HIP language selection must happen before adding targets that compile
those device sources. For custom component-level builds, the parent must make
`alpaka::alpaka`, `caravan::mpi` and `caravan::alpaka` available before adding
PMacc; the component checks/discovers these required dependencies rather than
silently building without them.

This uses ordinary CMake conventions: a cache-selectable backend, namespaced
targets (`pmacc::pmacc`, `caravan::core`, `caravan::mpi`,
`caravan::alpaka`, and `llamaLite::llamaLite`), and explicit linking and
finalization. Keep standard CMake `add_executable` and `target_link_libraries`;
CUDA/HIP language setup occurs in the parent directory scope when
`PMaccBackend.cmake` is included. The tests under each `libs/<component>/tests/`
exercise the component from its own build.

For tests, set `PMACC_BUILD_TESTING=ON` and/or `CARAVAN_BUILD_TESTING=ON` and/or
`LLAMALITE_BUILD_TESTING=ON`. The random-distribution executable
`pmacc-TestRng` is built when PMacc tests are on; the sampling diagnostic is
run only via the opt-in `pmacc-run-rng` target.

## Installation status

llamaLite installs `llamaLite::llamaLite` with `find_package(llamaLite CONFIG)`.
Caravan installs `caravan::core`, plus `caravan::mpi` if built;
`find_package(Caravan CONFIG REQUIRED COMPONENTS mpi)` loads the MPI adapter.
A core-only `find_package(Caravan CONFIG REQUIRED)` does not discover MPI or
alpaka. Both components can be installed into a prefix and consumed after
relocating that prefix.

**PMacc and Caravan's alpaka adapter are not installable yet.** The pinned
alpaka reconstructs non-exported targets and its installed config has an
incorrect host include path and hardcoded installation prefix. Installing
alpaka separately does not repair these issues. The source-tree accelerator
build works; do not assume a relocatable installed accelerator consumer.

The root `.gitmodules` is owned by the spearhed Git repository. On future
extraction into a standalone repository, create a package-root `.gitmodules`
with paths `thirdParty/alpaka` and `thirdParty/mallocMC`, preserving gitlinks.
