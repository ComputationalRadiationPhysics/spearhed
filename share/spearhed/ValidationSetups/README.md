# Sod Shock Tube Validation Setups

1D, 2D, and 3D Sod shock tube validation for SPEARHED. The two fluid states split
along x; transverse directions are symmetric slabs with frozen-wall boundaries.

All three dimensions share a single setup header (`SodShockTube.hpp`) and
dimension-specific param directories.

## Manual run

The following example builds and runs the 2D setup with the CPU OMP2 backend.
Run the first block from the repository root:

```bash
SPEARHED_ROOT="$PWD"
SOD_BUILD=path-to-spearhed-sod-2d-build
SOD_OUTPUT=path-to-spearhed-sod-2d-output

cmake -S "$SPEARHED_ROOT" -B "$SOD_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dalpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON \
    -DSPEARHED_ENABLE_OPENPMD=ON \
    -DSPEARHED_SETUP_FILE="$SPEARHED_ROOT/share/spearhed/ValidationSetups/SodShockTube.hpp" \
    -DSPEARHED_PARAM_DIR="$SPEARHED_ROOT/share/spearhed/ValidationSetups/SodShockTube2D"

cmake --build "$SOD_BUILD" -j 4
mkdir -p "$SOD_OUTPUT"
```

Run from the output directory so the openPMD file and validation plot are kept
together:

```bash
cd "$SOD_OUTPUT"
"$SOD_BUILD/include/spearhed/spearhed" \
    -s 200 -d 1 --openPMD.period 200 --openPMD.file 'sod_%06T.h5'

python "$SPEARHED_ROOT/share/spearhed/ValidationSetups/SodShockTube/validate_sod.py" \
    . --step 200 --dt 0.001 --bins 200
```

This writes `sod_000200.h5` and `sod_validation.png` which can be viewed to analyse the validation.

To select another dimension, use the corresponding parameter directory and a
fresh build directory:

| Dimension | Parameter directory |
|---|---|
| 1D | `SodShockTube` |
| 2D | `SodShockTube2D` |
| 3D | `SodShockTube3D` |

The serial CPU backend can be selected with
`-Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON`, but OMP2 is substantially faster for
larger runs.

## Parameter table

| | 1D | 2D | 3D |
|---|---|---|---|
| **simDim** | DIM1 | DIM2 | DIM3 |
| **h0** | 0.05 | 0.02 | 0.08 |
| **dt** | 0.001 | 0.001 | 0.001 |
| **dxLeft** | 1.125/32000 ~ 3.5e-5 | 0.005 | 1/30 ~ 0.033 |
| **dxRight** | 8 x dxLeft | sqrt(8) x dxLeft ~ 0.0141 | 2 x dxLeft ~ 0.067 |
| **Ly (=Lz)** | -- | 0.07 | 1/3 |
| **Wall thickness W** | 2×h0 = 0.1 | 0.04 | 0.16 |
| **Fluid particles** | 28444+3555 | 2800+350 | 3000+375 |
| **Total incl. walls** | ~35k | ~7.0k | ~15k |
| **Wall pieces** | 2 (x-caps) | 6 | 10 |

## Tuning knobs

- `dxLeft` -- controls particle resolution. Edit `spacingLeft()` in `SodShockTube.hpp`.
- `Ly`, `Lz` -- transverse extent. Edit `transverseExtent()`.
- `dt` is compile-time. Change `dt` in the selected `dimension.param` before
  configuring, or copy that parameter directory to a build-local location,
  edit the copy, and pass the copy as `SPEARHED_PARAM_DIR`. Pass exactly the
  same value to `validate_sod.py`.
- The simulation step count is the value passed to `spearhed -s`. Use the same
  value for `--openPMD.period` and the validator's `--step` so the requested
  snapshot exists. `--bins` only controls the validator's x binning.

## Validation reference

The 1D exact Riemann solution is the correct reference in 2D/3D -- the Sod
problem is a 1D Riemann problem; transverse directions are symmetric.
`validate_sod.py` projects onto x and bins there, averaging over transverse
directions.

Transverse-velocity symmetry diagnostics (`max|v_y|`, `L1(v_y)`, etc.) are
printed for 2D/3D snapshots to catch multi-D contamination.

## Expected L1 magnitudes (indicative, may drift with code changes)

| Field | Typical 1D L1 |
|---|---|
| density | ~0.02 |
| velocity x | ~0.03 |
| pressure | ~0.03 |
| internal energy | ~0.05 |

For 2D/3D, expect roughly comparable L1 values. Transverse velocities should
have `max|v_y,z|` < 0.01 for a converged run (small non-zero values near walls
are expected due to frozen-wall shear).

## Notes

- The 2D transverse extent is chosen so the left and right particle counts have
  the exact 8:1 ratio required by the equal-mass density jump.
- Right wall has ~2.4 particle layers in 3D (W = 2×h0 covers one support radius);
  if artifacts appear, increase h0 slightly.
- Frozen transverse walls keep their initial state; post-shock fluid sliding
  along them sees stale wall pressure -> small non-zero v_y/v_z near walls.
