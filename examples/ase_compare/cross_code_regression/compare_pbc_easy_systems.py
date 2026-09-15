"""Periodic vibe-qc vs PySCF.pbc — easy reference systems.

Same closed-shell wide-gap RKS-LDA / sto-3g / Γ-only setup
through both codes, on three "boring" reference systems:

    Ne FCC conventional      (4 atoms, 20 bf,  ~14 eV gap)
    Ar FCC conventional      (4 atoms, 36 bf,  ~12 eV gap)
    Diamond C conventional   (8 atoms, 40 bf,  ~5 eV gap)

These should both converge to the same energy across both codes.
If vibe-qc and PySCF disagree by more than ~10 mHa per cell on
any of these (after correcting for the known v0.6.x Madelung
shift, see ``scf_molecular_limit_check.py``), it's a real
bug, not a hard-system pathology.

Run:
    .venv/bin/python examples/ase_compare/cross_code_regression/compare_pbc_easy_systems.py

Outputs:
    output-pbc-easy-systems.csv
    Per-config .out / .perf / .system / .dump under
    output/<label>/ for post-mortem (vibe-qc side only)
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
OUT_DIR = HERE / "output" / "pbc-easy"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = HERE / "output-pbc-easy-systems.csv"

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

try:
    from pyscf.pbc import gto as pbc_gto, dft as pbc_dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


# --- Systems: builders return (vq.PeriodicSystem, basis_name,
#                                pyscf_atom_string, lattice_ang) -------

def neon_conv():
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


def argon_conv():
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


def diamond_conv():
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
class CodeRow:
    code: str
    label: str
    converged: Optional[bool]
    n_iter: Optional[int]
    energy_ha: Optional[float]
    wall_s: float
    note: str


def run_vibeqc(label: str, system, basis_name: str) -> CodeRow:
    rec = CodeRow("vibe-qc", label, None, None, None, 0.0, "")
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts = vq.PeriodicKSOptions()
    opts.functional = "LDA"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 18.0
    opts.conv_tol_energy = 1e-7
    opts.max_iter = 30
    out_stem = OUT_DIR / label
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
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def run_pyscf(label: str, pyscf_atoms: str, lat_ang: float,
              basis_name: str) -> CodeRow:
    rec = CodeRow("pyscf", label, None, None, None, 0.0, "")
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
        rec.n_iter = int(getattr(mf, "cycles", 0)) or None
        rec.note = "ok" if rec.converged else "not converged"
    except Exception as exc:
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def main() -> None:
    print("=" * 72)
    print(" Periodic SCF — vibe-qc vs PySCF.pbc  (RKS-LDA / sto-3g / Γ)")
    print("=" * 72)
    print(f"  PySCF.pbc available: {PYSCF_AVAILABLE}")

    cases = [
        ("01_solid_ne",  neon_conv),
        ("02_solid_ar",  argon_conv),
        ("03_diamond",   diamond_conv),
    ]

    rows = []
    for label, builder in cases:
        print(f"\n[{label}]")
        sys_, basis_name, pyscf_atoms, lat_ang = builder()
        print(f"  {len(sys_.unit_cell)} atoms, {basis_name}")
        v = run_vibeqc(label, sys_, basis_name)
        flag = "✓" if v.converged else "✗"
        print(f"  vibe-qc {flag}  iter={v.n_iter}  E={v.energy_ha}  wall={v.wall_s:.1f}s  {v.note}")
        p = run_pyscf(label, pyscf_atoms, lat_ang, basis_name)
        if PYSCF_AVAILABLE:
            flag = "✓" if p.converged else "✗"
            print(f"  pyscf   {flag}  iter={p.n_iter}  E={p.energy_ha}  wall={p.wall_s:.1f}s  {p.note}")
        rows.extend([v, p])

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "code", "converged", "n_iter",
                    "energy_ha", "wall_s", "note"])
        for r in rows:
            w.writerow([r.label, r.code, r.converged, r.n_iter,
                        r.energy_ha, r.wall_s, r.note])

    print()
    print("=" * 72)
    print(" Pivoted summary")
    print("=" * 72)
    print(f"  {'system':14s}  {'vibe-qc E':>14s}  {'pyscf E':>14s}  "
          f"{'ΔE (Ha)':>10s}  verdict")
    for label, _ in cases:
        v = next((r for r in rows if r.label == label and r.code == "vibe-qc"), None)
        p = next((r for r in rows if r.label == label and r.code == "pyscf"), None)
        v_e = f"{v.energy_ha:14.6f}" if (v and v.energy_ha is not None) else " " * 14
        p_e = f"{p.energy_ha:14.6f}" if (p and p.energy_ha is not None) else " " * 14
        if v and p and v.energy_ha is not None and p.energy_ha is not None:
            de = v.energy_ha - p.energy_ha
            de_str = f"{de:+10.4f}"
            if abs(de) < 1e-3:
                verdict = "✓ match"
            elif abs(de) < 0.5:
                verdict = "small disagreement (XC-grid / FFT)"
            else:
                verdict = "✗ large disagreement — investigate"
        else:
            de_str = " " * 10
            verdict = "incomplete"
        print(f"  {label:14s}  {v_e}  {p_e}  {de_str}  {verdict}")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
