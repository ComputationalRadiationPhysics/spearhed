SPEARHED - Licenses
===================

**Copyright 2025-2026** (in alphabetical order)

Tapish Narwal

SPEARHED is a program collection containing the main simulation, independent
scripts and auxiliary libraries. If not stated otherwise explicitly, the
following licenses apply:


### SPEARHED

The **main simulation** is licensed under the **GPLv3+**. See
[COPYING](COPYING). If not stated otherwise explicitly, that affects:
 - `include/spearhed`

### SPMacc
SPMacc is a part of the PMacc project, developed as an official 
extension. It may be merged into the main PMacc project in the future.
SPMacc is thus licensed identically to PMacc.
SPMacc is licensed under the **LGPLv3+**. See
[COPYING.LESSER](COPYING.LESSER).
If not stated otherwise explicitly, that affects:
 - `include/spmacc`

 ### llamaLite
 
LlamaLite is licensed under the **MPLv2+**. You can obtain a copy of the 
license at https://mozilla.org/MPL/2.0/.
 - `include/llamaLite`
 - `tests/llamaLite`


### Documentation

Documentation is licensed under CC-BY 4.0.
See https://creativecommons.org/licenses/by/4.0/ for the license.

If not stated otherwise explicitly, that affects files in:

- `docs`


### Third party software and other licenses

We include a list of (GPL-) compatible third party software for the sake
of an easier install of `SPEARHED`. Contributions to these parts of the
repository should *not* be made in the `thirdParty/` directory but in
*their according repositories* (that we import).

 - `- thirdParty/picongpu`:
   The PIConGPU project is included as is, and is licensed as mentioned
   in `thirdParty/picongpu/LICENSE.md`. PIConGPU code is not coupled with
   SPEARHED and it is merely included for easy aggregation. Where ever
   code is derived from PIConGPU, it is clearly mentioned and follows the
   **GPLv3+** license of PIConGPU.

 - `thirdParty/picongpu/include/pmacc`:
   PMacc is the particle mesh acceleration framework used in PIConGPU.
   PMacc is licensed under the **LGPLv3+**.
   This project is continued as SPMacc in SPEARHED. SPMacc is strictly a 
   part of the PMacc project, and may be merged into PMacc in the future.

 - `thirdParty/picongpu/thirdParty/mallocMC`:
   mallocMC is a fast memory allocator for many core accelerators and was
   originally forked from the `ScatterAlloc` project.
   It is licensed under the *MIT License*.
   Please visit
     https://github.com/ComputationalRadiationPhysics/mallocMC
   for further details and contributions.

 - `thirdParty/picongpu/thirdParty/cuda_memtest`:
   CUDA MemTest is an *independent program* developed by the University
   Illinois, published under the *Illinois Open Source License*.
   Please refer to the file `thirdParty/cuda_memtest/README` for license information.
   We redistribute this modified version of CUDA MemTest under the same license
   [thirdParty/cuda_memtest/README](thirdParty/cuda_memtest/README).
   The original release was published at
     http://sourceforge.net/projects/cudagpumemtest
   and our modified version is hosted at
     https://github.com/ComputationalRadiationPhysics/cuda_memtest
   for further reference.

- `thirdParty/picongpu/thirdParty/alpaka`:
   The alpaka library is a header-only C++20 abstraction library for accelerator development.  
   It aims to provide performance portability across accelerators through the abstraction (not hiding!)
   of the underlying levels of parallelism.
   Please visit
     https://github.com/alpaka-group/alpaka
   for further details and contributions.

- `thirdParty/picongpu/thirdParty/nlohmann_json`:
   nlohmann_json is a modern C++ library for working with JSON data, developed
   by Niels Lohmann, published under the MIT License.
   Please refer to the file `thirdParty/nlohmann_json/LICENSE.MIT` for license
   information.
   Please visit https://github.com/nlohmann/json for further details
   and contributions.
