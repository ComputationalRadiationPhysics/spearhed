#!/usr/bin/env python3
"""
Validate and visualize a SPEARHED Sod shock tube simulation.

Usage:
    python3 validate_sod.py [OUTPUT_DIR] [--step N] [--bins N] [--out FILE]

OUTPUT_DIR defaults to the current directory. The script reads the latest
(or --step) snapshot from sod_*.h5, projects the particle cloud onto the
x-axis, reconstructs pressure via P = (gamma - 1) * rho * u, and overlays
the exact Riemann solution. A 2x2 PNG and per-field L1 errors are produced.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import openpmd_api as io

    HAS_OPENPMD = True
except ImportError:
    HAS_OPENPMD = False


# Exact Sod Riemann solver


def _f(p_star, rho, p, gamma):
    """Hugoniot/rarefaction function used in the pressure iteration."""
    A = 2.0 / ((gamma + 1) * rho)
    B = (gamma - 1) / (gamma + 1) * p
    if p_star > p:
        return (p_star - p) * np.sqrt(A / (p_star + B))
    else:
        return (2 * np.sqrt(gamma * p / rho) / (gamma - 1)) * ((p_star / p) ** ((gamma - 1) / (2 * gamma)) - 1)


def _df(p_star, rho, p, gamma):
    A = 2.0 / ((gamma + 1) * rho)
    B = (gamma - 1) / (gamma + 1) * p
    c = np.sqrt(gamma * p / rho)
    if p_star > p:
        return np.sqrt(A / (p_star + B)) * (1 - (p_star - p) / (2 * (p_star + B)))
    else:
        return (1 / (rho * c)) * (p_star / p) ** (-(gamma + 1) / (2 * gamma))


def exact_sod(x, t, gamma=1.4, rho_l=1.0, p_l=1.0, u_l=0.0, rho_r=0.125, p_r=0.1, u_r=0.0):
    """Return rho, u, p, e (specific internal energy) arrays on grid x at time t.

    Uses the iterative exact Riemann solver (Newton-Raphson on the
    star-region pressure equation).
    """
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Initial guess: two-shock approximation
    p_star = (
        (c_l + c_r - 0.5 * (gamma - 1) * (u_r - u_l))
        / (c_l / p_l ** ((gamma - 1) / (2 * gamma)) + c_r / p_r ** ((gamma - 1) / (2 * gamma)))
    ) ** (2 * gamma / (gamma - 1))

    for _ in range(100):
        f_l = _f(p_star, rho_l, p_l, gamma)
        f_r = _f(p_star, rho_r, p_r, gamma)
        df_l = _df(p_star, rho_l, p_l, gamma)
        df_r = _df(p_star, rho_r, p_r, gamma)
        dp = -(f_l + f_r + (u_r - u_l)) / (df_l + df_r)
        p_star += dp
        if abs(dp) / p_star < 1e-10:
            break

    u_star = 0.5 * (u_l + u_r) + 0.5 * (_f(p_star, rho_r, p_r, gamma) - _f(p_star, rho_l, p_l, gamma))

    # Density in the star regions
    rho_star_l = rho_l * (
        (p_star / p_l + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * p_star / p_l + 1)
        if p_star > p_l
        else (p_star / p_l) ** (1.0 / gamma)
    )
    rho_star_r = rho_r * (
        (p_star / p_r + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * p_star / p_r + 1)
        if p_star > p_r
        else (p_star / p_r) ** (1.0 / gamma)
    )

    # Wave speeds
    if p_star > p_l:
        # Left shock
        s_l = u_l - c_l * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p_l + (gamma - 1) / (2 * gamma))
    else:
        # Left rarefaction: head and tail
        s_hl = u_l - c_l
        c_star_l = c_l * (p_star / p_l) ** ((gamma - 1) / (2 * gamma))
        s_tl = u_star - c_star_l

    if p_star > p_r:
        # Right shock
        s_r = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p_r + (gamma - 1) / (2 * gamma))
    else:
        # Right rarefaction: head and tail
        s_hr = u_r + c_r
        c_star_r = c_r * (p_star / p_r) ** ((gamma - 1) / (2 * gamma))
        s_tr = u_star + c_star_r

    # Contact discontinuity
    s_contact = u_star

    rho_arr = np.empty_like(x)
    u_arr = np.empty_like(x)
    p_arr = np.empty_like(x)

    xi = x / t  # self-similar variable

    for i, s in enumerate(xi):
        if p_star > p_l:
            # Left shock
            if s < s_l:
                rho_arr[i], u_arr[i], p_arr[i] = rho_l, u_l, p_l
            elif s < s_contact:
                rho_arr[i], u_arr[i], p_arr[i] = rho_star_l, u_star, p_star
            elif s < s_r if p_star > p_r else s < s_tr:
                rho_arr[i], u_arr[i], p_arr[i] = rho_star_r, u_star, p_star
            elif p_star <= p_r and s < s_hr:
                u_fan = 2 / (gamma + 1) * (s - c_r + (gamma - 1) / 2 * u_r)
                c_fan = s - u_fan
                p_fan = p_r * (c_fan / c_r) ** (2 * gamma / (gamma - 1))
                rho_fan = gamma * p_fan / c_fan**2
                rho_arr[i], u_arr[i], p_arr[i] = rho_fan, u_fan, p_fan
            else:
                rho_arr[i], u_arr[i], p_arr[i] = rho_r, u_r, p_r
        else:
            # Left rarefaction
            if s < s_hl:
                rho_arr[i], u_arr[i], p_arr[i] = rho_l, u_l, p_l
            elif s < s_tl:
                # Fan interior
                u_fan = 2 / (gamma + 1) * (c_l + s + (gamma - 1) / 2 * u_l)
                c_fan = u_fan - s
                p_fan = p_l * (c_fan / c_l) ** (2 * gamma / (gamma - 1))
                rho_fan = gamma * p_fan / c_fan**2
                rho_arr[i], u_arr[i], p_arr[i] = rho_fan, u_fan, p_fan
            elif s < s_contact:
                rho_arr[i], u_arr[i], p_arr[i] = rho_star_l, u_star, p_star
            elif s < (s_r if p_star > p_r else s_tr):
                rho_arr[i], u_arr[i], p_arr[i] = rho_star_r, u_star, p_star
            elif p_star <= p_r and s < s_hr:
                u_fan = 2 / (gamma + 1) * (s - c_r + (gamma - 1) / 2 * u_r)
                c_fan = s - u_fan
                p_fan = p_r * (c_fan / c_r) ** (2 * gamma / (gamma - 1))
                rho_fan = gamma * p_fan / c_fan**2
                rho_arr[i], u_arr[i], p_arr[i] = rho_fan, u_fan, p_fan
            else:
                rho_arr[i], u_arr[i], p_arr[i] = rho_r, u_r, p_r

    e_arr = p_arr / ((gamma - 1) * rho_arr)
    return rho_arr, u_arr, p_arr, e_arr


# Data loading


def load_snapshot_openpmd(path_pattern, step):
    s = io.Series(path_pattern, io.Access.read_only)
    it = s.iterations[step]
    fluid = it.particles["fluid"]

    SCALAR = io.Record_Component.SCALAR

    def load_scalar(record):
        rc = fluid[record][SCALAR]
        arr = rc.load_chunk()
        s.flush()
        return arr

    def load_component(record, comp):
        rc = fluid[record][comp]
        arr = rc.load_chunk()
        s.flush()
        return arr

    def try_load_component(record, comp):
        """Return component array or None if the record/component is absent (1D runs)."""
        try:
            rc = fluid[record][comp]
            arr = rc.load_chunk()
            s.flush()
            return arr
        except Exception:
            return None

    x = load_component("position", "x")
    vx = load_component("velocity", "x")
    vy = try_load_component("velocity", "y")
    vz = try_load_component("velocity", "z")
    rho = load_scalar("mass_density")
    u = load_scalar("specific_internal_energy")
    return x, vx, vy, vz, rho, u


# Main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default=".")
    parser.add_argument("--step", type=int, default=200, help="Snapshot step (default 200 = t=0.2 with dt=0.001)")
    parser.add_argument("--bins", type=int, default=200, help="Number of x-bins")
    parser.add_argument("--out", default="sod_validation.png", help="Output plot file")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()

    pattern = os.path.join(args.output_dir, "sod_%06T.h5")
    t_end = args.step * args.dt

    if not HAS_OPENPMD:
        sys.exit("openpmd_api not available. Activate the pixi environment first.")

    print(f"Loading step {args.step} (t = {t_end:.3f}) from {pattern}")
    x, vx, vy, vz, rho, u = load_snapshot_openpmd(pattern, args.step)
    print(f"  Loaded {len(x)} particles")

    # Pressure
    gamma = args.gamma
    pressure = (gamma - 1.0) * rho * u

    # Bin along x
    x_min, x_max = -1.0, 1.0
    bin_edges = np.linspace(x_min, x_max, args.bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    idx = np.digitize(x, bin_edges) - 1
    idx = np.clip(idx, 0, args.bins - 1)

    def bin_mean(field):
        sums = np.bincount(idx, weights=field, minlength=args.bins).astype(float)
        counts = np.bincount(idx, minlength=args.bins).astype(float)
        mask = counts > 0
        out = np.full(args.bins, np.nan)
        out[mask] = sums[mask] / counts[mask]
        return out

    rho_sph = bin_mean(rho)
    vx_sph = bin_mean(vx)
    p_sph = bin_mean(pressure)
    u_sph = bin_mean(u)

    # Exact solution
    x_exact = np.linspace(x_min + 1e-6, x_max - 1e-6, 1000)
    rho_ex, v_ex, p_ex, u_ex = exact_sod(
        x_exact,
        t_end,
        gamma=gamma,
        rho_l=1.0,
        p_l=1.0,
        u_l=0.0,
        rho_r=0.125,
        p_r=0.1,
        u_r=0.0,
    )

    # L1 errors (only over bins with particles)
    x_ex_bins = np.interp(bin_centers, x_exact, rho_ex)
    v_ex_bins = np.interp(bin_centers, x_exact, v_ex)
    p_ex_bins = np.interp(bin_centers, x_exact, p_ex)
    u_ex_bins = np.interp(bin_centers, x_exact, u_ex)

    valid = np.isfinite(rho_sph)

    def l1(sph, exact):
        m = valid & np.isfinite(sph)
        return np.mean(np.abs(sph[m] - exact[m]))

    l1_rho = l1(rho_sph, x_ex_bins)
    l1_vx = l1(vx_sph, v_ex_bins)
    l1_p = l1(p_sph, p_ex_bins)
    l1_u = l1(u_sph, u_ex_bins)

    print(f"\nL1 errors at t = {t_end:.3f}  (gamma = {gamma}):")
    print(f"  density          L1 = {l1_rho:.4f}")
    print(f"  velocity x       L1 = {l1_vx:.4f}")
    print(f"  pressure         L1 = {l1_p:.4f}")
    print(f"  internal energy  L1 = {l1_u:.4f}")

    # Transverse symmetry diagnostics (2D / 3D only)
    if vy is not None:
        print("\nTransverse symmetry (vs exact 0):")
        print(f"  max|v_y| = {np.max(np.abs(vy)):.6f}")
        print(f"  L1(v_y)  = {np.mean(np.abs(vy)):.6f}")
    if vz is not None:
        if vy is None:
            print("\nTransverse symmetry (vs exact 0):")
        print(f"  max|v_z| = {np.max(np.abs(vz)):.6f}")
        print(f"  L1(v_z)  = {np.mean(np.abs(vz)):.6f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Sod shock tube  t = {t_end:.2f}  (gamma = {gamma},  N = {len(x)})", fontsize=13)

    panel_data = [
        (axes[0, 0], rho_sph, rho_ex, r"Density $\rho$", f"L1 = {l1_rho:.4f}"),
        (axes[0, 1], vx_sph, v_ex, r"Velocity $v_x$", f"L1 = {l1_vx:.4f}"),
        (axes[1, 0], p_sph, p_ex, r"Pressure $P$", f"L1 = {l1_p:.4f}"),
        (axes[1, 1], u_sph, u_ex, r"Internal energy $u$", f"L1 = {l1_u:.4f}"),
    ]

    for ax, sph_vals, ex_vals, ylabel, label in panel_data:
        ax.plot(x_exact, ex_vals, "k-", linewidth=1.5, label="Exact")
        ax.plot(bin_centers[valid], sph_vals[valid], ".", ms=3, alpha=0.7, color="#1f77b4", label=f"SPH  {label}")
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, args.out)
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
