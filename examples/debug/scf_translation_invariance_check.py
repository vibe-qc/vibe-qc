"""Translation-invariance check — periodic SCF should be invariant under
shifting the molecule by a constant vector inside the simulation cell.

This is the cleanest sanity-test for the gauge consistency between the
Hartree J build (which currently pins G=0 to zero, FFT-Poisson) and the
electron-nuclear V_ne (which currently uses the bare libint
``compute_nuclear_lattice`` direct-truncated lattice sum without any
matching G=0 omission).

PySCF.pbc passes this test by construction because it builds V_ne with
``coulG[G=0] = 0`` — the same gauge as J. v0.6.x vibe-qc does NOT.

The script:

  - Runs H2 at the box origin and at the box centre, L = 30 bohr,
    EWALD_3D, fixed FFT spacing for reproducibility.
  - Reports the energy diff. Should be < 1e-5 Ha when the gauge bug
    is fixed; v0.6.2 reports ~ 0.6 Ha (catastrophic).

Wall: ~30 s on a laptop.

Run:
    .venv/bin/python examples/debug/scf_translation_invariance_check.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "translation-invariance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TranslationCheck:
    label: str
    L_bohr: float
    e_origin: float
    e_centered: float
    diff: float
    verdict: str


def h2_atoms_at(offset_bohr) -> list:
    """Return [Atom(1, R0), Atom(1, R0 + (0, 0, 1.4))] with R0 = offset."""
    R0 = np.asarray(offset_bohr, dtype=float)
    return [
        vq.Atom(1, R0.tolist()),
        vq.Atom(1, (R0 + np.array([0.0, 0.0, 1.4])).tolist()),
    ]


def run_one(label: str, L_bohr: float, offset_bohr) -> float:
    atoms = h2_atoms_at(offset_bohr)
    sys_periodic = vq.PeriodicSystem(
        dim=3,
        lattice=L_bohr * np.eye(3),
        unit_cell=atoms,
    )
    basis = vq.BasisSet(sys_periodic.unit_cell_molecule(), "sto-3g")

    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.conv_tol_energy = 1e-7

    out_stem = OUT_DIR / label
    t0 = time.perf_counter()
    res = vq.run_rhf_periodic_scf(
        sys_periodic, basis,
        vq.KPoints.gamma(sys_periodic),
        opts, progress=False,
        spacing_bohr=0.4,
    )
    t = time.perf_counter() - t0
    print(f"  {label:18s}  E = {float(res.energy):+.6f} Ha   "
          f"({res.n_iter} iters, {t:.2f} s)")
    return float(res.energy)


def main() -> None:
    print("=" * 72)
    print(" Translation-invariance check — H2 at L = 30 bohr cubic")
    print("=" * 72)
    print("  Both runs use identical lattice/basis/integration grid.")
    print("  Diff > 1e-5 Ha means the V_ne gauge does not match J.")
    print()

    # H2 sitting at the origin: nuclei at (0,0,0) and (0,0,1.4)
    e_origin = run_one("h2_at_origin", 30.0, [0.0, 0.0, 0.0])

    # H2 sitting at box centre: nuclei centered around L/2.
    # Pick offset so the molecule's centre-of-mass is at (L/2, L/2, L/2).
    L = 30.0
    centre = np.array([L / 2.0, L / 2.0, L / 2.0])
    bond_offset = np.array([0.0, 0.0, 0.7])  # half of 1.4 bohr bond
    e_centered = run_one("h2_centered",  L, (centre - bond_offset).tolist())

    diff = e_centered - e_origin
    print()
    print(f"  E(centered) - E(origin) = {diff:+.6f} Ha")
    if abs(diff) < 1e-5:
        verdict = "✓ translation invariant within 1e-5 Ha — gauge consistent"
    elif abs(diff) < 1e-2:
        verdict = (f"~ small offset ({diff:+.2e} Ha) — gauge approximately "
                   "consistent")
    else:
        verdict = (f"✗ TRANSLATION INVARIANCE BROKEN — V_ne / J gauge "
                   "mismatch (PySCF gives diff ~1e-13)")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
