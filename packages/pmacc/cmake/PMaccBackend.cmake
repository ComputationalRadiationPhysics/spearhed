# Copyright 2015-2025 PMacc contributors
# SPDX-License-Identifier: LGPL-3.0-or-later OR GPL-3.0-or-later
# Include from the common ancestor before adding PMacc or accelerator targets.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../libs/pmacc/cmake/Backend.cmake")

# Keep macro scope to preserve alpaka_finalize()'s caller-scope effects.
macro(pmacc_finalize target)
    alpaka_finalize(${target})
endmacro()
