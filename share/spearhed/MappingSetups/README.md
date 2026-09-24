# Mapping setup examples

Each header exposes a setup under `spearhed::mapping_setups`; choose one in a custom setup file:

```cpp
#include "spearhed/MappingSetups/AdaptiveSpatialSplit.hpp"
namespace spearhed
{
    using Setup = mapping_setups::AdaptiveSpatialSplitSetup;
}
```

- `MaterialAabb.hpp`: persistent material cohorts with refreshed occupancy bounds.
- `FixedCartesian.hpp`: dense fixed grid for fluid, material mappings for boundary and tracers.
- `AdaptiveSpatialSplit.hpp`: particle-count trigger and world-x midpoint partition.
- `AdaptiveVelocitySplit.hpp`: maximum-speed trigger and fast/slow partition using `spearhed::tags::vel`.

All examples use the normal decomposition-group lifecycle; no simulation-control changes are required.
