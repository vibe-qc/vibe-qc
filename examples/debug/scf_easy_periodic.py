"""Easy-to-converge periodic reference systems — no convergence aids needed.

Three "boring" closed-shell, wide-gap, well-localised periodic
systems that should converge from a Hcore guess in <10 iterations
with default settings — no damping ramp, no level shift, no SAD
guess, no smearing. If any of these fails, you have a bug
upstream of any system-specific instability.

Systems in order of simplicity:

  1. **Solid Ne** (4-atom FCC conventional cubic, sto-3g)
     Noble-gas van-der-Waals crystal. ~14 eV gap experimental.
     Atomic-localised; no bonding, no charge transfer. Already
     a fixture in vibe-qc's bulk-benchmark suite.

  2. **Solid Ar** (4-atom FCC conventional cubic, sto-3g)
     Same idea as Ne with a heavier noble gas. ~12 eV gap.
     Slightly more basis functions but the same trivially-easy
     SCF picture.

  3. **Diamond C** (8-atom conventional cubic, sto-3g)
     The textbook covalent semiconductor. ~5 eV gap. sto-3g
     keeps it small (no linear dependencies, unlike Si/pob-TZVP).
     Tightly bound sp³ network.

All three:
  - Closed shell (no spin contamination, no UHF)
  - Wide gap (no smearing, no metallic instability)
  - Small basis (sto-3g; ~5-40 bf per cell)
  - Orthorhombic cubic cell (EWALD_3D legal)
  - Γ-only k-mesh (smallest)
  - Default convergence aids only (damping=0.5, DIIS on, no shift)

Wall: ~30 s total on a laptop. If any of these systems takes
more than 10 iters or diverges, vibe-qc has a regression — they
are the canonical "if this doesn't work nothing will" set.

Use as:
  - Smoke test before / after invasive changes
  - Reference baseline against which to compare hard cases
    (NaCl-LDA, MgO-LDA, transition metals)
  - Demonstration that EWALD_3D + RKS-LDA isn't itself broken

Run:
    .venv/bin/python examples/debug/scf_easy_periodic.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "easy-periodic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_EV = 27.211386245988

# --- 1. Solid Ne -------------------------------------------------
def neon_conventional() -> vq.PeriodicSystem:
    """Ne FCC, conventional cubic cell (a ≈ 4.43 Å @ 4 K)."""
    a = 4.43 / 0.529177210903
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    cell = [vq.Atom(10, [f * a for f in p]) for p in fcc]
    return vq.PeriodicSystem(3, a * np.eye(3), cell)


# --- 2. Solid Ar -------------------------------------------------
def argon_conventional() -> vq.PeriodicSystem:
    """Ar FCC, conventional cubic cell (a ≈ 5.26 Å @ 4 K)."""
    a = 5.26 / 0.529177210903
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    cell = [vq.Atom(18, [f * a for f in p]) for p in fcc]
    return vq.PeriodicSystem(3, a * np.eye(3), cell)


# --- 3. Diamond C ------------------------------------------------
def diamond_conventional() -> vq.PeriodicSystem:
    """Diamond C, conventional cubic cell (a ≈ 3.567 Å)."""
    a = 3.567 / 0.529177210903
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    displaced = [(fx + 0.25, fy + 0.25, fz + 0.25) for fx, fy, fz in fcc]
    cell = [vq.Atom(6, [f * a for f in p]) for p in fcc + displaced]
    return vq.PeriodicSystem(3, a * np.eye(3), cell)


@dataclass
class EasyResult:
    label: str
    n_atoms: int
    n_bf: int
    n_iter: Optional[int]
    converged: Optional[bool]
    energy_ha: Optional[float]
    gap_ev: Optional[float]
    wall_s: float
    note: str


def make_opts() -> vq.PeriodicKSOptions:
    """Default-ish RKS/LDA — no convergence aids beyond DIIS + damping."""
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.lattice_opts.cutoff_bohr = 10.0
    o.lattice_opts.nuclear_cutoff_bohr = 18.0
    o.conv_tol_energy = 1e-7
    o.max_iter = 30
    # Defaults: damping=0.5, use_diis=True, level_shift=0.0,
    #           smearing_temperature=0.0, initial_guess=HCORE
    return o


def run_one(label: str, builder, basis_name: str = "sto-3g") -> EasyResult:
    print(f"\n[{label}]", flush=True)
    system = builder()
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    print(f"  {len(system.unit_cell)} atoms, {basis.nbasis} bf "
          f"(basis: {basis_name})")

    out_stem = OUT_DIR / label
    plog = ProgressLogger(log_path=str(out_stem) + ".out", verbose=4)

    rec = EasyResult(
        label=label, n_atoms=len(system.unit_cell), n_bf=basis.nbasis,
        n_iter=None, converged=None, energy_ha=None, gap_ev=None,
        wall_s=0.0, note="",
    )

    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                result = vq.run_rks_periodic_scf(
                    system, basis, vq.KPoints.gamma(system),
                    make_opts(), progress=plog,
                    spacing_bohr=0.4,
                )
        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(result.converged)
        rec.n_iter = int(result.n_iter)
        rec.energy_ha = float(result.energy)
        # Read gap from per-k mo_energies (Γ-only, so just one entry)
        n_occ = sum(int(at.Z) for at in system.unit_cell) // 2
        eps = result.mo_energies[0]
        rec.gap_ev = (
            float(eps[n_occ]) - float(eps[n_occ - 1])
        ) * HARTREE_TO_EV
        rec.note = "ok" if rec.converged else "max_iter, no convergence"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:60]}"

    flag = "✓" if rec.converged else "✗"
    g_str = f"gap={rec.gap_ev:.2f} eV" if rec.gap_ev is not None else ""
    print(f"  {flag}  iter={rec.n_iter}  E={rec.energy_ha}  "
          f"{g_str}  wall={rec.wall_s:.1f}s")
    return rec


def main() -> None:
    print("=" * 72)
    print(" Easy-to-converge periodic reference systems")
    print(" Closed-shell, wide-gap, well-localised — no convergence aids")
    print("=" * 72)

    cases = [
        ("01_solid_neon",   neon_conventional),
        ("02_solid_argon",  argon_conventional),
        ("03_diamond_C",    diamond_conventional),
    ]

    results = [run_one(lbl, fn) for lbl, fn in cases]

    print()
    print("=" * 72)
    print(" Summary")
    print("=" * 72)
    n_ok = sum(1 for r in results if r.converged)
    print(f"  {n_ok} / {len(results)} systems converged with default aids")
    print()
    print(f"  {'system':18s}  {'atom':4s}  {'bf':4s}  {'iter':4s}  "
          f"{'gap (eV)':>8s}  {'wall':>6s}  flag")
    for r in results:
        flag = "OK" if r.converged else "FAIL"
        i_str = f"{r.n_iter:4d}" if r.n_iter is not None else "  - "
        g_str = f"{r.gap_ev:8.2f}" if r.gap_ev is not None else " " * 8
        print(f"  {r.label:18s}  {r.n_atoms:4d}  {r.n_bf:4d}  "
              f"{i_str}  {g_str}  {r.wall_s:5.1f}s  {flag}")

    if n_ok == len(results):
        print()
        print("  ✓ All easy systems converged — vibe-qc periodic SCF + "
              "EWALD_3D + RKS-LDA is healthy.")
        print("    Any harder failure (NaCl-LDA, MgO-LDA, transition")
        print("    metals) is a system-specific instability, not a")
        print("    fundamental SCF bug.")
    else:
        print()
        print("  ✗ One of the easy systems failed — that's a regression.")
        print("    Inspect the .dump files under output/easy-periodic/")
        print("    or run scf_iteration_recorder.py for a trajectory plot.")


if __name__ == "__main__":
    main()
