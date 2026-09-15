"""Initial-guess A/B test: SAD vs HCORE on the hard cases.

v0.6.1 wired the **SAD (Superposition of Atomic Densities)** initial
guess into all four periodic SCF drivers (RHF / UHF / RKS / UKS).
SAD is the standard answer to the "Hcore guess is too far from
physical for ionic insulators" pathology that v0.5.6's divergence
detection flagged. This script answers the obvious question:

    "Does SAD fix the NaCl-LDA / MgO-LDA divergence that
     Hcore couldn't handle?"

Runs each of three "hard" closed-shell ionic systems with both
initial guesses, default convergence aids elsewhere, reports for
each (system × guess):

    converged?  |  n_iter  |  E (Ha)  |  wall (s)

Hard cases:
  - LiH conventional cubic (8 atoms, sto-3g, ~13 bf)
  - NaCl conventional cubic (8 atoms, sto-3g, ~72 bf)
  - MgO conventional cubic (8 atoms, sto-3g, ~52 bf)

Each is a closed-shell ionic insulator — exactly the regime
where Hcore guess = "all electrons in one big well" produces a
density that's nothing like the actual atomic-localised picture
the SCF needs to relax to.

Wall: ~5-15 min total on a laptop. Each cell × guess wraps in
crash_dump_context + perf_log so failures leave artifacts.

Run:
    .venv/bin/python examples/debug/scf_sad_vs_hcore.py
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "sad-vs-hcore"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "summary.csv"


# --- Systems ------------------------------------------------------

def _rocksalt(a_ang: float, z_a: int, z_b: int) -> vq.PeriodicSystem:
    a = a_ang / 0.529177210903
    A_FRAC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    B_FRAC = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = (
        [vq.Atom(z_a, [f * a for f in p]) for p in A_FRAC] +
        [vq.Atom(z_b, [f * a for f in p]) for p in B_FRAC]
    )
    return vq.PeriodicSystem(3, a * np.eye(3), cell)


def lih_conventional() -> vq.PeriodicSystem:
    return _rocksalt(4.084, 3, 1)   # Li, H


def nacl_conventional() -> vq.PeriodicSystem:
    return _rocksalt(5.640, 11, 17)  # Na, Cl


def mgo_conventional() -> vq.PeriodicSystem:
    return _rocksalt(4.213, 12, 8)   # Mg, O


@dataclass
class GuessResult:
    system: str
    guess: str
    n_atoms: int
    n_bf: int
    converged: Optional[bool]
    n_iter: Optional[int]
    energy_ha: Optional[float]
    wall_s: float
    note: str


def make_opts(guess: vq.InitialGuess) -> vq.PeriodicKSOptions:
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.lattice_opts.cutoff_bohr = 10.0
    o.lattice_opts.nuclear_cutoff_bohr = 18.0
    o.conv_tol_energy = 1e-6
    o.max_iter = 30
    o.initial_guess = guess
    return o


def run_one(system_name: str, builder, guess: vq.InitialGuess) -> GuessResult:
    label = f"{system_name}_{guess.name}"
    print(f"\n[{label}]", flush=True)
    system = builder()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    print(f"  {len(system.unit_cell)} atoms, {basis.nbasis} bf, "
          f"guess={guess.name}")

    out_stem = OUT_DIR / label
    rec = GuessResult(
        system=system_name, guess=guess.name,
        n_atoms=len(system.unit_cell), n_bf=basis.nbasis,
        converged=None, n_iter=None, energy_ha=None,
        wall_s=0.0, note="",
    )

    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                result = vq.run_rks_periodic_scf(
                    system, basis, vq.KPoints.gamma(system),
                    make_opts(guess), progress=False,
                    spacing_bohr=0.5,
                )
        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(result.converged)
        rec.n_iter = int(result.n_iter)
        rec.energy_ha = float(result.energy)
        rec.note = "ok" if rec.converged else "max_iter, no convergence"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:60]}"

    flag = "✓" if rec.converged else "✗"
    print(f"  {flag}  iter={rec.n_iter}  E={rec.energy_ha}  "
          f"wall={rec.wall_s:.1f}s  {rec.note}")
    return rec


def main() -> None:
    print("=" * 72)
    print(" SAD vs HCORE initial-guess A/B on hard ionic systems")
    print(" (v0.6.1 wired SAD into all 4 periodic SCF drivers)")
    print("=" * 72)

    systems = [
        ("LiH",  lih_conventional),
        ("NaCl", nacl_conventional),
        ("MgO",  mgo_conventional),
    ]
    guesses = [vq.InitialGuess.HCORE, vq.InitialGuess.SAD]

    results: list[GuessResult] = []
    for sys_name, builder in systems:
        for guess in guesses:
            results.append(run_one(sys_name, builder, guess))

    # CSV
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "guess", "n_atoms", "n_bf",
                    "converged", "n_iter", "energy_ha", "wall_s", "note"])
        for r in results:
            w.writerow([r.system, r.guess, r.n_atoms, r.n_bf,
                        r.converged, r.n_iter, r.energy_ha,
                        r.wall_s, r.note])

    # Pretty pivoted summary
    print()
    print("=" * 72)
    print(" Summary (pivoted by system × guess)")
    print("=" * 72)
    print(f"  {'system':6s}  {'HCORE iter':>10s}  {'HCORE conv':>10s}  "
          f"{'SAD iter':>10s}  {'SAD conv':>10s}")
    for sys_name, _ in systems:
        h = next(r for r in results if r.system == sys_name and r.guess == "HCORE")
        s = next(r for r in results if r.system == sys_name and r.guess == "SAD")
        print(f"  {sys_name:6s}  "
              f"{h.n_iter or '-':>10}  {('OK' if h.converged else 'FAIL'):>10s}  "
              f"{s.n_iter or '-':>10}  {('OK' if s.converged else 'FAIL'):>10s}")

    sad_wins = sum(
        1 for sys_name, _ in systems
        if (next(r for r in results if r.system == sys_name and r.guess == "SAD").converged
            and not next(r for r in results if r.system == sys_name and r.guess == "HCORE").converged)
    )
    if sad_wins:
        print()
        print(f"  ✓ SAD rescues {sad_wins} system(s) that HCORE could not converge.")
    else:
        print()
        print("  Both guesses behave identically on these systems — either")
        print("  both converged (great) or both failed (the bug is deeper")
        print("  than the initial guess). Inspect the .dump files.")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
