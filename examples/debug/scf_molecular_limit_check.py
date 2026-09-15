"""Molecular-limit check — periodic SCF should match molecular at large L.

A periodic SCF on a single H₂ molecule sitting in a 30-bohr cubic
box of vacuum should give the SAME energy as a plain molecular
RHF — periodic images at 30 bohr have negligible overlap, the
calculation is "molecular" by every physical measure.

If the periodic energy is **over-bound** vs the molecular reference
by a 1/L-scaling shift, you're hitting the **Madelung self-image
leak** that v0.6.0 ships with: ``build_j_ewald_3d`` pins the G=0
Fourier mode of the electronic potential to zero (a Makov-Payne
α_e·S shift), and ``nuclear_repulsion_per_cell`` with EWALD_3D
carries the matching Madelung self-image term. For a neutral cell
these should cancel — but cancellation is "the caller's
responsibility" and the SCF drivers don't apply it.

Magnitude predictor — ``α_M · (Q_n² + Q_e²) / (2L)``, with α_M =
2.837 (simple-cubic Madelung constant). For He at L=30, Z=2:

    2.837 × (2² + 2²) / (2 × 30) = 0.378 Ha

— matches the observed v0.6.0 diff to 4 sig figs (per
``tests/test_periodic_atomic_limit_bug.py``).

This script:

  - Runs He atom + H₂ in 30 bohr cubic boxes (RHF, sto-3g, EWALD_3D)
  - Runs each as a plain molecular RHF
  - Reports the diff and the Madelung-leak prediction
  - Prints a verdict: ✓ matches molecular  /  ✗ Madelung leak

Use as:
  - Regression detector — once engineering's Madelung-cancellation
    fix lands, ``E_periodic - E_molecular`` should drop to <1e-5 Ha
  - Pedagogy — concrete view of why the molecular-limit test is the
    cleanest correctness check for any periodic SCF implementation

Wall: ~1 min on a laptop.

Run:
    .venv/bin/python examples/debug/scf_molecular_limit_check.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "molecular-limit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple-cubic Madelung constant (dimensionless, charge-units of Z²).
ALPHA_M_CUBIC = 2.837


@dataclass
class LimitCheck:
    label: str
    L_bohr: float
    n_electrons: int
    e_molecular: float          # plain molecular RHF reference (Ha)
    e_periodic: float           # periodic RHF in vacuum-padded box (Ha)
    diff_obs: float             # E_periodic - E_molecular (Ha)
    diff_predicted: float       # α_M · (Q_n² + Q_e²) / (2L) — over-bind by this
    verdict: str


def he_atom() -> tuple[vq.Molecule, str, int]:
    mol = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    return mol, "sto-3g", 2


def h2_molecule() -> tuple[vq.Molecule, str, int]:
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    return mol, "sto-3g", 2


def run_check(label: str, builder: Callable, L_bohr: float = 30.0,
              ) -> LimitCheck:
    print(f"\n[{label}]  L = {L_bohr} bohr", flush=True)
    mol, basis_name, n_e = builder()

    # 1. Molecular reference
    mol_basis = vq.BasisSet(mol, basis_name)
    t0 = time.perf_counter()
    mol_result = vq.run_rhf(mol, mol_basis)
    t_mol = time.perf_counter() - t0
    e_molecular = float(mol_result.energy)
    print(f"  molecular RHF: E = {e_molecular:.6f} Ha   ({t_mol:.2f} s)")

    # 2. Periodic in vacuum-padded cubic box at the same geometry
    sys_periodic = vq.PeriodicSystem(
        dim=3,
        lattice=L_bohr * np.eye(3),
        unit_cell=list(mol.atoms),
    )
    per_basis = vq.BasisSet(sys_periodic.unit_cell_molecule(), basis_name)

    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.conv_tol_energy = 1e-7

    out_stem = OUT_DIR / label
    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                per_result = vq.run_rhf_periodic_scf(
                    sys_periodic, per_basis,
                    vq.KPoints.gamma(sys_periodic),
                    opts, progress=False,
                    spacing_bohr=0.4,
                )
        e_periodic = float(per_result.energy)
        t_per = time.perf_counter() - t0
        print(f"  periodic RHF:  E = {e_periodic:.6f} Ha   "
              f"({per_result.n_iter} iters, {t_per:.2f} s)")
    except Exception as exc:
        print(f"  periodic RHF FAILED: {type(exc).__name__}: {str(exc)[:60]}")
        return LimitCheck(label, L_bohr, n_e, e_molecular,
                          float("nan"), float("nan"), float("nan"),
                          f"FAIL: {type(exc).__name__}")

    diff_obs = e_periodic - e_molecular
    # Predicted over-binding: α_M · (Q_n² + Q_e²) / (2L)
    # Charge-neutral cell so Q_n = Q_e = n_e (in units of e).
    diff_pred = -ALPHA_M_CUBIC * (n_e ** 2 + n_e ** 2) / (2.0 * L_bohr)

    abs_match = abs(diff_obs - diff_pred) < 0.02
    is_close = abs(diff_obs) < 1e-3
    if is_close:
        verdict = "✓ matches molecular within 1 mHa — Madelung leak fixed"
    elif abs_match:
        verdict = (f"✗ Madelung leak: observed {diff_obs:+.4f} Ha matches "
                   f"prediction {diff_pred:+.4f} Ha within 20 mHa")
    else:
        verdict = (f"? diff {diff_obs:+.4f} Ha doesn't match Madelung "
                   f"prediction {diff_pred:+.4f} Ha — investigate")

    print(f"  diff:        {diff_obs:+.6f} Ha "
          f"(predicted Madelung leak: {diff_pred:+.4f} Ha)")
    print(f"  verdict:     {verdict}")
    return LimitCheck(label, L_bohr, n_e, e_molecular, e_periodic,
                      diff_obs, diff_pred, verdict)


def main() -> None:
    print("=" * 72)
    print(" Molecular-limit check — periodic vs molecular at large L")
    print("=" * 72)
    print(f"  Madelung constant (simple cubic): α_M = {ALPHA_M_CUBIC}")
    print(f"  Predicted over-bind: α_M · (Q_n² + Q_e²) / (2L)")

    cases = [
        ("01_He_atom_L30",  he_atom,        30.0),
        ("02_H2_L30",       h2_molecule,    30.0),
        ("03_H2_L50",       h2_molecule,    50.0),
        ("04_H2_L100",      h2_molecule,   100.0),
    ]

    results = [run_check(lbl, fn, L) for lbl, fn, L in cases]

    print()
    print("=" * 72)
    print(" Summary — 1/L scaling check")
    print("=" * 72)
    print(f"  {'label':18s}  {'L (bohr)':>9s}  "
          f"{'diff (Ha)':>11s}  {'predicted':>11s}  ratio")
    for r in results:
        ratio = (r.diff_obs / r.diff_predicted
                 if abs(r.diff_predicted) > 1e-9 else float("nan"))
        print(f"  {r.label:18s}  {r.L_bohr:9.0f}  "
              f"{r.diff_obs:+11.4f}  {r.diff_predicted:+11.4f}  "
              f"{ratio:.3f}")
    print()
    print("  If `ratio` is consistently near 1.0 across L, you have the")
    print("  textbook Madelung self-image leak — the fix is the")
    print("  Madelung-cancellation pass that v0.6.x will land. Once")
    print("  fixed, all `diff (Ha)` values should drop to <1e-3.")


if __name__ == "__main__":
    main()
