#!/usr/bin/env python3
"""MSINDO implicit solvation (COSMO) — solvation energies in water.

MSINDO (Ahlswede & Jug, *J. Comput. Chem.* **20**, 563 & 572 (1999); © Mulliken
Center for Theoretical Chemistry, University of Bonn) gains a COSMO implicit-
solvation driver in vibe-qc, ``vibeqc.semiempirical.methods.msindo_cosmo``.  It is
the MSINDO arm of vibe-qc's reference-pluggable CPCM/COSMO engine: the cavity,
the segment matrix, the dielectric screening and the apparent-surface-charge
solve are shared with the HF/DFT path; only the **solute↔cavity coupling** is
MSINDO-specific — a distributed-multipole B-matrix of two-centre Slater
penetration integrals (Klamt-Schüürmann conductor model,
``f(ε)=(ε−1)/(ε+0.5)``).

This script runs a few small molecules in water (ε = 78.39) and reports the
solvation energy ``E_solv = E(solvent) − E(gas)`` and the in-solvent total.
With the default GEPOL cavity (MSINDO's own pentakisdodecahedron/SAS
tessellation + COSMOR radii) vibe-qc reproduces the reference MSINDO oracle to
~1e-9 Ha; the lighter ``cavity="lebedev"`` (shared with HF/DFT CPCM) instead
carries that cavity convention (see ``docs/user_guide/msindo.md`` § Implicit
solvation (COSMO)).

Run:

    .venv/bin/python examples/semiempirical/23_msindo_cosmo.py
"""

from __future__ import annotations

from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

EPSILON_WATER = 78.39

# (name, Z, geometry in Å, reference MSINDO oracle E_solv in mHa)
#   oracle_total = E(CARTES RHF COSMO DIELEC 78.39 NOSYM), MSINDO 2025e — the
#   no-symmetry run, which vibe-qc's independent per-atom cavity build matches.
_CASES = [
    ("H2O", [8, 1, 1],
     [(0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)], -17.0264718414),
    ("HF", [9, 1],
     [(0.0, 0.0, 0.0), (0.0, 0.0, 0.917)], -24.3141073696),
    ("H2S", [16, 1, 1],
     [(0.0, 0.0, 0.0), (0.9627, 0.0, 0.9264), (-0.9627, 0.0, 0.9264)], -11.2393992045),
    ("CH4", [6, 1, 1, 1, 1],
     [(0.0, 0.0, 0.0), (0.629, 0.629, 0.629), (-0.629, -0.629, 0.629),
      (0.629, -0.629, -0.629), (-0.629, 0.629, -0.629)], -8.2961423173),
]


def main() -> None:
    print(f"MSINDO COSMO in water (ε = {EPSILON_WATER}), default GEPOL cavity\n")
    print(f"{'mol':4s} {'E_solv (mHa)':>13} {'COSMO total':>16} "
          f"{'oracle total':>16} {'Δ (Ha)':>11}")
    print("-" * 64)
    for name, Z, xyz, oracle_total in _CASES:
        r = msindo_cosmo(Z, xyz, epsilon=EPSILON_WATER)        # default cavity="gepol"
        print(f"{name:4s} {r.e_solv * 1e3:13.3f} {r.total_energy:16.10f} "
              f"{oracle_total:16.10f} {r.total_energy - oracle_total:11.2e}")
    print("\nThe GEPOL cavity reproduces reference MSINDO to ~1e-9 Ha. The lighter")
    print("Lebedev cavity (cavity='lebedev', shared with HF/DFT CPCM) instead")
    print("carries that cavity convention — e.g. H2O E_solv -6.83 vs -8.26 mHa:")
    leb = msindo_cosmo(_CASES[0][1], _CASES[0][2], epsilon=EPSILON_WATER,
                       cavity="lebedev")
    print(f"  H2O Lebedev: E_solv = {leb.e_solv * 1e3:.3f} mHa")


if __name__ == "__main__":
    main()
