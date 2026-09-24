# Spatial baseline

This benchmark records the baseline for the material-region implementation before the spatial decomposition interfaces are changed.

## Workload

`SpatialBaseline.cpp` creates 16, 64, and 256 overlapping regions with one live particle and one frame per region. All region pairs are candidates. It measures:

- `UpdateVolumes` completion time;
- `calculateNeighbours` construction and completion time;
- directed candidate region-pair count;
- one `interact` completion time using an atomic accepted-pair counter; and
- device bytes in the two CSR arrays (`neighbourRegions` and `regionOffsets`).

Frame-index construction and particle initialisation are outside the timed regions. Times include waiting for device completion. The CSR byte count excludes the equal-sized host mirrors owned by `HostDeviceBuffer` and allocator overhead.

Build and run with:

```bash
cmake -S . -B $HOME/scratch/spearhed-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPEARHED_TESTING=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON
cmake --build $HOME/scratch/spearhed-build \
  --target benchmark_spmacc_spatial_baseline -j 4
TMPDIR=$HOME/scratch \
  $HOME/scratch/spearhed-build/tests/benchmark_spmacc_spatial_baseline
```

The executable emits CSV so results can be archived and compared after later phases.
