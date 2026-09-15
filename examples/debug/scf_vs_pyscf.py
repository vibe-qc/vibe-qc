"""vibe-qc vs PySCF — head-to-head periodic-SCF cross-check.

Runs the same closed-shell, wide-gap, periodic RKS-LDA / sto-3g /
Γ-only SCF on three "easy" reference systems through both:

    vibe-qc    via run_rks_periodic_scf  + EWALD_3D
    PySCF      via pyscf.pbc.dft.RKS

Reports for each system × code:
    converged?  |  n_iter  |  E (Ha)  |  wall (s)

If both converge to the same energy (within 1e-3 Ha): ✓ codes agree.
If vibe-qc diverges and PySCF converges: definitive vibe-qc bug.
If both diverge: hard system, not a vibe-qc-specific issue.

Three reference systems (matched to ``scf_easy_periodic.py``):
  1. Solid Ne  (4-atom FCC conventional cubic)
  2. Solid Ar  (4-atom FCC conventional cubic)
  3. Diamond C (8-atom conventional cubic)

Skip PySCF if it isn't installed in the venv — the vibe-qc side
runs either way and the comparison columns just show "—".

Wall: ~1-2 min total on a laptop. PySCF's pbc.dft is comparable
in cost to vibe-qc on these small systems; both are dominated by
the FFT grid + lattice ERI build.

Run:
    .venv/bin/python examples/debug/scf_vs_pyscf.py

Install PySCF for the cross-check:
    pip install pyscf
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "vs-pyscf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "summary.csv"

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# --- PySCF availability ----------------------------------------------
try:
    from pyscf.pbc import gto as pbc_gto, dft as pbc_dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


# --- System definitions: (label, builder_vibeqc, atoms_pyscf, lat_ang) ---

def neon_conventional() -> Tuple[vq.PeriodicSystem, str, list, float]:
    """Ne FCC conventional cubic, a = 4.43 Å."""
    a_ang = 4.43
    a = a_ang * ANGSTROM_TO_BOHR
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    sys_ = vq.PeriodicSystem(
        3, a * np.eye(3),
        [vq.Atom(10, [f * a for f in p]) for p in fcc],
    )
    pyscf_atoms = "; ".join(
        f"Ne {fx*a_ang:.4f} {fy*a_ang:.4f} {fz*a_ang:.4f}"
        for (fx, fy, fz) in fcc
    )
    return sys_, "sto-3g", pyscf_atoms, a_ang


def argon_conventional() -> Tuple[vq.PeriodicSystem, str, list, float]:
    a_ang = 5.26
    a = a_ang * ANGSTROM_TO_BOHR
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    sys_ = vq.PeriodicSystem(
        3, a * np.eye(3),
        [vq.Atom(18, [f * a for f in p]) for p in fcc],
    )
    pyscf_atoms = "; ".join(
        f"Ar {fx*a_ang:.4f} {fy*a_ang:.4f} {fz*a_ang:.4f}"
        for (fx, fy, fz) in fcc
    )
    return sys_, "sto-3g", pyscf_atoms, a_ang


def diamond_conventional() -> Tuple[vq.PeriodicSystem, str, list, float]:
    a_ang = 3.567
    a = a_ang * ANGSTROM_TO_BOHR
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    displaced = [(fx + 0.25, fy + 0.25, fz + 0.25) for fx, fy, fz in fcc]
    all_pos = fcc + displaced
    sys_ = vq.PeriodicSystem(
        3, a * np.eye(3),
        [vq.Atom(6, [f * a for f in p]) for p in all_pos],
    )
    pyscf_atoms = "; ".join(
        f"C {fx*a_ang:.4f} {fy*a_ang:.4f} {fz*a_ang:.4f}"
        for (fx, fy, fz) in all_pos
    )
    return sys_, "sto-3g", pyscf_atoms, a_ang


@dataclass
class CodeResult:
    code: str           # "vibeqc" or "pyscf"
    label: str
    converged: Optional[bool]
    n_iter: Optional[int]
    energy_ha: Optional[float]
    wall_s: float
    note: str


def run_vibeqc(label: str, system: vq.PeriodicSystem,
               basis_name: str) -> CodeResult:
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts = vq.PeriodicKSOptions()
    opts.functional = "LDA"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 18.0
    opts.conv_tol_energy = 1e-7
    opts.max_iter = 30

    out_stem = OUT_DIR / f"{label}_vibeqc"
    rec = CodeResult(code="vibeqc", label=label,
                     converged=None, n_iter=None, energy_ha=None,
                     wall_s=0.0, note="")
    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                result = vq.run_rks_periodic_scf(
                    system, basis, vq.KPoints.gamma(system),
                    opts, progress=False, spacing_bohr=0.4,
                )
        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(result.converged)
        rec.n_iter = int(result.n_iter)
        rec.energy_ha = float(result.energy)
        rec.note = "ok" if rec.converged else "max_iter"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def run_pyscf(label: str, pyscf_atoms: str, lat_ang: float,
              basis_name: str) -> CodeResult:
    rec = CodeResult(code="pyscf", label=label,
                     converged=None, n_iter=None, energy_ha=None,
                     wall_s=0.0, note="")
    if not PYSCF_AVAILABLE:
        rec.note = "pyscf not installed"
        return rec

    try:
        cell = pbc_gto.M(
            atom=pyscf_atoms,
            a=[[lat_ang, 0, 0], [0, lat_ang, 0], [0, 0, lat_ang]],
            basis=basis_name,
            unit="A",
            verbose=0,
        )
        mf = pbc_dft.RKS(cell)
        mf.xc = "lda"
        mf.max_cycle = 30
        mf.conv_tol = 1e-7

        t0 = time.perf_counter()
        e = mf.kernel()
        rec.wall_s = time.perf_counter() - t0
        rec.energy_ha = float(e)
        rec.converged = bool(mf.converged)
        # pyscf's mf.cycles isn't always populated; fall back to
        # mf.scf_summary or just report the max if needed.
        rec.n_iter = int(getattr(mf, "cycles", 0)) or None
        rec.note = "ok" if rec.converged else "max_iter / not converged"
    except Exception as exc:
        rec.wall_s = 0.0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def main() -> None:
    print("=" * 72)
    print(" vibe-qc vs PySCF — periodic RKS-LDA / sto-3g cross-check")
    print("=" * 72)
    print(f"  PySCF available: {PYSCF_AVAILABLE}")

    cases = [
        ("01_solid_ne",      neon_conventional),
        ("02_solid_ar",      argon_conventional),
        ("03_diamond_c",     diamond_conventional),
    ]

    rows: list[tuple[CodeResult, CodeResult]] = []
    for label, builder in cases:
        print(f"\n[{label}]")
        sys_, basis_name, pyscf_atoms, lat_ang = builder()
        print(f"  {len(sys_.unit_cell)} atoms")

        v = run_vibeqc(label, sys_, basis_name)
        flag = "✓" if v.converged else "✗"
        print(f"  vibeqc {flag}  iter={v.n_iter}  E={v.energy_ha}  "
              f"wall={v.wall_s:.1f}s  {v.note}")

        p = run_pyscf(label, pyscf_atoms, lat_ang, basis_name)
        if PYSCF_AVAILABLE:
            flag = "✓" if p.converged else "✗"
            print(f"  pyscf  {flag}  iter={p.n_iter}  E={p.energy_ha}  "
                  f"wall={p.wall_s:.1f}s  {p.note}")
        else:
            print(f"  pyscf  -  (not installed; pip install pyscf)")
        rows.append((v, p))

    # CSV
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "code", "converged", "n_iter",
                    "energy_ha", "wall_s", "note"])
        for v, p in rows:
            for r in (v, p):
                w.writerow([r.label, r.code, r.converged, r.n_iter,
                            r.energy_ha, r.wall_s, r.note])

    print()
    print("=" * 72)
    print(" Summary — energies side-by-side")
    print("=" * 72)
    print(f"  {'system':14s}  {'vibeqc E':>14s}  {'pyscf E':>14s}  "
          f"{'ΔE (Ha)':>10s}  verdict")
    for v, p in rows:
        v_e = f"{v.energy_ha:14.6f}" if v.energy_ha is not None else " " * 14
        p_e = (f"{p.energy_ha:14.6f}" if p.energy_ha is not None
               else " " * 14)
        if v.energy_ha is not None and p.energy_ha is not None:
            de = v.energy_ha - p.energy_ha
            de_str = f"{de:+10.4f}"
            if abs(de) < 1e-3:
                verdict = "✓ match"
            elif abs(de) < 0.5:
                verdict = "small disagreement"
            else:
                verdict = "✗ large disagreement"
        else:
            de_str = "         -"
            verdict = ("vibeqc only" if v.converged else
                       "neither converged" if not (v.converged or
                                                    (p and p.converged))
                       else "see notes")
        print(f"  {v.label:14s}  {v_e}  {p_e}  {de_str}  {verdict}")

    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent)}")
    print()
    print("  Reading the verdict column:")
    print("    ✓ match              — vibe-qc and PySCF agree to <1 mHa")
    print("    small disagreement   — same physics, different XC grid /")
    print("                            FFT spacing / Ewald split")
    print("    ✗ large disagreement — vibe-qc bug; cross-check with the")
    print("                            molecular-limit script for Madelung leak")


if __name__ == "__main__":
    main()
