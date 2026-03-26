#!/bin/bash

set -e
set -o pipefail


#############################################################################
# Conformance with Alpaka: Do not write __global__ CUDA kernels directly    #
#############################################################################
# test/hasCudaGlobalKeyword include/pmacc
# test/hasCudaGlobalKeyword share/pmacc/examples
# test/hasCudaGlobalKeyword include/picongpu
# test/hasCudaGlobalKeyword share/picongpu/examples

#############################################################################
# Enforce angle brackets <...> for includes of external library files       #
#############################################################################
test/hasExtLibIncludeBrackets include boost
test/hasExtLibIncludeBrackets include alpaka
test/hasExtLibIncludeBrackets include mallocMC
test/hasExtLibIncludeBrackets include pmacc

#############################################################################
# Disallow doxygen with \                                                   #
#############################################################################
test/hasWrongDoxygenStyle include param
test/hasWrongDoxygenStyle include tparam
test/hasWrongDoxygenStyle include see
test/hasWrongDoxygenStyle include return
test/hasWrongDoxygenStyle include treturn
test/hasWrongDoxygenStyle share param
test/hasWrongDoxygenStyle share tparam
test/hasWrongDoxygenStyle share see
test/hasWrongDoxygenStyle share return
test/hasWrongDoxygenStyle share treturn
