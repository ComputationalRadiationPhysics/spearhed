# PMacc

C++20 accelerator library. This component can be added to a CMake build when
its parent has already selected a backend via `cmake/Backend.cmake` and made
`alpaka::alpaka`, `caravan::mpi`, `caravan::alpaka`, and `llamaLite::llamaLite`
available. PMacc links llamaLite publicly as a declared dependency for planned
integration. The package root (`packages/pmacc/CMakeLists.txt`) supplies these
from the bundled sources and configures llamaLite before PMacc.
A standalone component can also discover installed packages; it does not
implicitly traverse sibling source directories, so provide llamaLite with
`add_subdirectory` first or install it for `find_package` discovery. CUDA/HIP
additionally needs `mallocMC::mallocMC`. Link `pmacc::pmacc`, then call
`alpaka_finalize(target)` for any consumer compiling alpaka/PMacc headers.

Tests are controlled by `PMACC_BUILD_TESTING`. The distribution sampling
program has a separate, opt-in `pmacc-run-rng` target.

No installed PMacc config or exported target is provided yet: pinned alpaka's
installed config and target-export behaviour block reliable installed
accelerator consumers. Source-tree use is supported; installation is deferred.
