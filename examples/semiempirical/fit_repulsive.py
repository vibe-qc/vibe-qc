#!/usr/bin/env python3
"""Repulsive potential fitting tool for production DFTB.

Given DFT reference energies along a bond-stretch coordinate,
fits a cubic spline repulsive potential V_rep(R) such that:
    E_DFTB0(R) + V_rep(R) ≈ E_DFT(R)

The repulsive is defined as the difference between DFT and DFTB0
energies minus a smooth baseline.

Usage:
    # Fit O-H repulsive from DFT data
    python fit_repulsive.py --Z1 8 --Z2 1 --data oh_scan.csv --output oh_spline.toml

Input CSV format: R_bohr, E_DFT_Ha, E_DFTB0_Ha
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fit_repulsive_spline(
    R: np.ndarray,
    e_dft: np.ndarray,
    e_dftb0: np.ndarray,
    n_knots: int = 8,
    r_min: float = 0.5,
    r_max: float = 5.0,
    smooth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a repulsive spline from DFT vs DFTB0 energies.

    V_rep(R) = E_DFT(R) - E_DFTB0(R) - baseline

    Parameters
    ----------
    R : ndarray
        Bond distances (bohr), sorted ascending.
    e_dft : ndarray
        DFT total energies (Ha).
    e_dftb0 : ndarray
        DFTB0 total energies (Ha) at same geometries.
    n_knots : int
        Number of spline knots.
    r_min, r_max : float
        Range for knot placement (bohr).
    smooth : float
        Smoothing factor (0 = interpolating).

    Returns
    -------
    R_knots, V_knots : ndarray
        Knot positions and repulsive values.
    """
    diff = e_dft - e_dftb0

    # The repulsive should be non-negative and go to zero at large R
    # Shift so min value is zero at large R
    baseline = np.min(diff[R > 3.0]) if np.any(R > 3.0) else diff[-1]
    v_rep = diff - baseline
    v_rep = np.maximum(v_rep, 0.0)  # repulsive is non-negative

    # Place knots uniformly in r_min..r_max
    knots_r = np.linspace(r_min, r_max, n_knots)
    knots_v = np.interp(knots_r, R, v_rep)

    # Ensure V=0 at last knot (cutoff)
    knots_v[-1] = 0.0

    # Smooth by averaging nearby points
    if smooth > 0:
        from scipy.ndimage import gaussian_filter1d

        knots_v = gaussian_filter1d(knots_v, sigma=smooth)
        knots_v[-1] = 0.0

    return knots_r, knots_v


def spline_to_toml(
    Z1: int, Z2: int, R_knots: np.ndarray, V_knots: np.ndarray, path: Path
) -> None:
    """Write repulsive spline to TOML file."""
    with open(path, "w") as f:
        f.write(f"# DFT-fitted repulsive spline for Z1={Z1}, Z2={Z2}\n")
        f.write(f"Z1 = {Z1}\n")
        f.write(f"Z2 = {Z2}\n")
        f.write(f"n_knots = {len(R_knots)}\n")
        f.write("R_bohr = [\n")
        for r in R_knots:
            f.write(f"  {r:.6f},\n")
        f.write("]\n")
        f.write("V_Ha = [\n")
        for v in V_knots:
            f.write(f"  {v:.8f},\n")
        f.write("]\n")
    print(f"Wrote {path}")


def load_spline_from_toml(path: Path) -> tuple[int, int, list[float], list[float]]:
    """Load repulsive spline from TOML file. Returns (Z1, Z2, R, V)."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data["Z1"], data["Z2"], data["R_bohr"], data["V_Ha"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fit DFTB repulsive spline")
    parser.add_argument("--Z1", type=int, required=True)
    parser.add_argument("--Z2", type=int, required=True)
    parser.add_argument("--data", type=str, help="CSV: R,E_DFT,E_DFTB0")
    parser.add_argument("--output", type=str, default="repulsive.toml")
    parser.add_argument("--n-knots", type=int, default=8)
    parser.add_argument("--r-min", type=float, default=0.5)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--smooth", type=float, default=0.0)
    args = parser.parse_args()

    if args.data:
        data = np.loadtxt(args.data, delimiter=",", skiprows=1)
        R, e_dft, e_dftb0 = data[:, 0], data[:, 1], data[:, 2]
    else:
        # Demo: generate synthetic data
        print("No data file provided. Generating synthetic demo data...")
        R = np.linspace(0.8, 5.0, 50)
        # Synthetic DFT energy: Morse-like
        e_dft = 0.2 * (1 - np.exp(-1.5 * (R - 1.8))) ** 2 - 0.2
        # Synthetic DFTB0: monotonic decreasing
        e_dftb0 = -0.5 / R - 0.1
        print(f"  R range: {R[0]:.2f} - {R[-1]:.2f} bohr")
        print(f"  DFT min: {np.min(e_dft):.4f} Ha")
        print(f"  DFTB0 min: {np.min(e_dftb0):.4f} Ha")

    R_knots, V_knots = fit_repulsive_spline(
        R, e_dft, e_dftb0, args.n_knots, args.r_min, args.r_max, args.smooth
    )

    print(f"\nFitted spline ({len(R_knots)} knots):")
    for r, v in zip(R_knots, V_knots):
        print(f"  R={r:.3f}  V={v:.6f} Ha")

    spline_to_toml(args.Z1, args.Z2, R_knots, V_knots, Path(args.output))


if __name__ == "__main__":
    main()
