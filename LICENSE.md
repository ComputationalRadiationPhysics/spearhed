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
 - `tests/spearhed`

### SPMacc
SPMacc is a part of the PMacc project, developed as an official 
extension. It may be merged into the main PMacc project in the future.
SPMacc is thus licensed identically to PMacc.
SPMacc is licensed under the **LGPLv3+**. See
[COPYING.LESSER](COPYING.LESSER).
If not stated otherwise explicitly, that affects:
 - `include/spmacc`

### llamaLite

llamaLite is licensed under MPL-2.0; see
`packages/pmacc/libs/llamaLite/LICENSE` and the notices in its source files.
Its library and library-owned tests live in `packages/pmacc/libs/llamaLite`.
The spmacc integration test remains in `tests/llama_lite/integration`.


### Documentation

Documentation is licensed under CC-BY 4.0.
See https://creativecommons.org/licenses/by/4.0/ for the license.

If not stated otherwise explicitly, that affects files in:

- `docs`


### PMacc package components

The PMacc, Caravan, and llamaLite source trees are maintained as sibling
libraries in `packages/pmacc/libs/`; these are not PIConGPU submodules.

- **PMacc** (`packages/pmacc/libs/pmacc`) is licensed under LGPL-3.0-or-later
  OR GPL-3.0-or-later. See [COPYING](packages/pmacc/libs/pmacc/COPYING) and
  [COPYING.LESSER](packages/pmacc/libs/pmacc/COPYING.LESSER).
- **Caravan** (`packages/pmacc/libs/caravan`) is licensed under MPL-2.0; see
  its [LICENSE](packages/pmacc/libs/caravan/LICENSE).
- **llamaLite** (`packages/pmacc/libs/llamaLite`) is licensed under MPL-2.0;
  see its [LICENSE](packages/pmacc/libs/llamaLite/LICENSE) and source notices.

### Third-party dependencies

The alpaka3 and mallocMC dependencies are Git submodules under
`packages/pmacc/thirdParty/`. Their license terms, copyright notices, and
upstream contribution guidance are provided by the respective upstream
repositories and are present in each initialized submodule checkout:

- alpaka3: MPL-2.0; see its [LICENSE](packages/pmacc/thirdParty/alpaka/LICENSE)
  and <https://github.com/alpaka-group/alpaka3>.
- mallocMC: MIT; see its [LICENSE](packages/pmacc/thirdParty/mallocMC/LICENSE)
  and <https://github.com/ikbuibui/mallocMC>.

The spearhed repository owns the top-level `.gitmodules`; submodules are
tracked at package-relative paths. No PIConGPU source tree, CUDA MemTest, or
nlohmann_json submodule is included at those former paths in this checkout.
