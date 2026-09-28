# Copyright 2016-2024 Alexander Grund
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.

add_executable(pmacc-TestRng 2DDistribution.cpp)
target_link_libraries(pmacc-TestRng PRIVATE pmacc::pmacc)
alpaka_finalize(pmacc-TestRng)
# This diagnostic writes a sampled distribution; keep execution opt-in.
add_custom_target(
    pmacc-run-rng
    COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:pmacc-TestRng>
        ${MPIEXEC_POSTFLAGS}
    DEPENDS pmacc-TestRng
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
