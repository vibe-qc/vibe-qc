#!/usr/bin/env python3
"""Generate DFT-fitted repulsive splines for production DFTB.

This script produces proper repulsive potentials by fitting to DFT
reference data. Two modes:

  1. From real DFT data (requires PySCF):
     python scripts/fit_dftb_repulsives.py --dft pbe --basis def2-tzp

  2. From synthetic data (Morse-potential surrogates):
     python scripts/fit_dftb_repulsives.py --synthetic

The output is a TOML parameter file ready for use with
SemiempiricalParameters.dftb0_production() or load_parameters().

Requirements for DFT mode: pip install pyscf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add the repo root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical import SemiempiricalParameters

# ---------------------------------------------------------------------------
# Known molecular properties (bond lengths in bohr, force constants in Ha/bohr²)
# Used as targets for synthetic data generation.
# ---------------------------------------------------------------------------

MOLECULAR_PROPERTIES = {
    # (Z1, Z2): (r_eq_bohr, k_Ha_per_bohr2, D_e_Ha)
    (1, 1): (1.401, 0.370, 0.174),  # H₂
    (6, 1): (2.060, 0.320, 0.160),  # C-H in CH₄
    (7, 1): (1.913, 0.350, 0.170),  # N-H in NH₃
    (8, 1): (1.811, 0.380, 0.190),  # O-H in H₂O
    (6, 6): (2.910, 0.250, 0.140),  # C-C in C₂H₆
    (6, 7): (2.780, 0.280, 0.150),  # C-N
    (6, 8): (2.700, 0.300, 0.160),  # C-O
    (7, 7): (2.750, 0.300, 0.160),  # N-N
    (7, 8): (2.650, 0.320, 0.170),  # N-O
    (8, 8): (2.650, 0.340, 0.180),  # O-O
    (9, 1): (1.734, 0.420, 0.220),  # H-F
    (17, 1): (2.409, 0.280, 0.165),  # H-Cl
    (16, 1): (2.530, 0.240, 0.145),  # H-S
}


def synthetic_e_dft(R, r_eq, D_e, k):
    """Morse potential as DFT energy surrogate."""
    a = np.sqrt(k / (2 * D_e))
    return D_e * (1 - np.exp(-a * (R - r_eq))) ** 2 - D_e


def generate_dimer_data(Z1, Z2, r_range, n_points=50):
    """Generate (R, E_elec(R), E_atom_sum) for a diatomic scan.

    Returns the electronic (band-structure) energy without repulsive,
    so the repulsive can be fitted as V_rep = E_DFT - E_elec.
    """
    params = SemiempiricalParameters.dftb0_production()
    n_val = params.valence_electrons(Z1) + params.valence_electrons(Z2)
    mult = 1 if n_val % 2 == 0 else 2
    R = np.linspace(r_range[0], r_range[1], n_points)
    e_elec = np.zeros(n_points)
    for i, r in enumerate(R):
        mol = Molecule(
            [Atom(Z1, [0, 0, 0]), Atom(Z2, [0, 0, r])], charge=0, multiplicity=mult
        )
        if mult == 1:
            result = _se.run_dftb0(mol, params)
        else:
            result = _se.run_udftb0(mol, params)
        e_elec[i] = result.e_electronic
    # Atomic reference: sum of on-site energies for all valence shells
    e_atom = 0.0
    for Z in [Z1, Z2]:
        for l in [0, 1, 2]:  # s, p, d
            e_shell = params.on_site_energy(Z, l)
            if e_shell != 0.0:
                e_atom += e_shell * (2 * (2 * l + 1))  # 2 electrons per orbital
    return R, e_elec, e_atom


def fit_all_pairs(mode="synthetic", output_path="vq_dftb_production.toml"):
    """Fit repulsive splines for all key element pairs."""
    all_R, all_V, all_Z1, all_Z2 = [], [], [], []

    for (Z1, Z2), (r_eq, k, D_e) in MOLECULAR_PROPERTIES.items():
        r_min = max(0.5, r_eq * 0.4)
        r_max = r_eq * 2.5

        # Generate DFTB0 electronic energies (no repulsive)
        R, e_elec, e_atom = generate_dimer_data(Z1, Z2, (r_min, r_max))

        if mode == "synthetic":
            e_dft = synthetic_e_dft(R, r_eq, D_e, k)
        else:
            e_dft = compute_dft_energy(Z1, Z2, R)

        # Repulsive = DFT - E_elec + e_atom, zeroed at long range
        # V_rep(R) = E_DFT(R) - E_elec(R) + e_atom
        # At large R: E_DFT -> 0, E_elec -> e_atom, so V_rep -> 0
        v_rep = e_dft - e_elec + e_atom
        v_rep -= np.min(v_rep[R > r_eq * 1.5])
        v_rep = np.maximum(v_rep, 0.0)

        # Fit spline knots
        n_knots = 8
        knots_r = np.linspace(r_min, r_max, n_knots)
        knots_v = np.interp(knots_r, R, v_rep)
        knots_v[-1] = 0.0  # zero at cutoff

        all_Z1.append(Z1)
        all_Z2.append(Z2)
        all_R.append(knots_r)
        all_V.append(knots_v)

        print(
            f"  {Z1}-{Z2}: r_eq={r_eq:.3f}, {n_knots} knots, "
            f"V_max={knots_v[0]:.4f}, cutoff={knots_r[-1]:.2f}"
        )

    # Write TOML
    write_toml(all_Z1, all_Z2, all_R, all_V, output_path)
    return output_path


def compute_dft_energy(Z1, Z2, R):
    """Compute DFT energy for a diatomic using PySCF."""
    try:
        from pyscf import dft, gto
    except ImportError:
        raise ImportError("DFT mode requires PySCF: pip install pyscf")

    e = np.zeros(len(R))
    for i, r in enumerate(R):
        mol = gto.M(
            atom=[[Z1, (0, 0, 0)], [Z2, (0, 0, float(r))]],
            basis="def2-tzvp",
            unit="Bohr",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "PBE"
        e[i] = mf.kernel()
    return e


def write_toml(Z1_list, Z2_list, R_list, V_list, path):
    """Write fitted repulsive splines to TOML."""
    from vibeqc.semiempirical.io import load_production_parameters, save_parameters

    # Load production params as base (has all element data)
    params = load_production_parameters()

    # Replace repulsive pairs with fitted splines
    for Z1, Z2, R, V in zip(Z1_list, Z2_list, R_list, V_list):
        params.set_repulsive_pair_spline(Z1, Z2, list(R), list(V))

    save_parameters(params, path)

    # Also write standalone repulsive section for appending
    with open(path, "a") as f:
        f.write("\n# Fitted repulsive splines\n")
        for Z1, Z2, R, V in zip(Z1_list, Z2_list, R_list, V_list):
            f.write("[[repulsive]]\n")
            f.write(f"Z1 = {Z1}\n")
            f.write(f"Z2 = {Z2}\n")
            f.write("R_bohr = [\n")
            for r in R:
                f.write(f"  {r:.6f},\n")
            f.write("]\n")
            f.write("V_Ha = [\n")
            for v in V:
                f.write(f"  {v:.8f},\n")
            f.write("]\n\n")

    print(f"\nWrote {len(Z1_list)} repulsive pairs to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate DFT-fitted repulsive splines for production DFTB"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use Morse-potential surrogates (no DFT needed)",
    )
    parser.add_argument(
        "--dft",
        choices=["pbe", "b3lyp"],
        default=None,
        help="Run real DFT calculations (requires PySCF)",
    )
    parser.add_argument(
        "--basis", default="def2-tzvp", help="Basis set for DFT (default: def2-tzvp)"
    )
    parser.add_argument(
        "--output", default="vq_dftb_production.toml", help="Output TOML file path"
    )
    args = parser.parse_args()

    mode = "synthetic" if args.synthetic else "dft"

    print("=" * 60)
    print(f"Fitting repulsive splines (mode: {mode})")
    print("=" * 60)
    print(f"  Element pairs: {len(MOLECULAR_PROPERTIES)}")
    print(f"  Target properties from known molecular data")
    print()

    path = fit_all_pairs(mode=mode, output_path=args.output)
    print(f"\nDone! Load with:")
    print(f"  from vibeqc.semiempirical.io import load_parameters")
    print(f"  params = load_parameters('{path}')")


if __name__ == "__main__":
    main()
