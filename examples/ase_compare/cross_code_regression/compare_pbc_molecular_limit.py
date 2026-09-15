"""Periodic molecular-limit cross-check — vibe-qc vs PySCF.pbc.

Direct probe of the v0.6.x Madelung-self-image leak. The same
H₂ molecule sitting in a 30-bohr cubic vacuum-padded box should
give the SAME energy as molecular RHF — periodic images don't
overlap, the calculation is "molecular" by every physical
measure.

Three reference channels:
  - molecular RHF (vibe-qc)               — definitive reference
  - periodic RHF (vibe-qc)                — current bug suspect
  - periodic RHF (PySCF.pbc.scf.RHF)      — independent reference

If vibe-qc-periodic differs from PySCF-periodic AND from
molecular by the same Madelung-leak shift α_M·(Q²+Q²)/(2L), the
bug is reproduced. If PySCF-periodic and molecular agree but
vibe-qc-periodic doesn't, the bug is squarely in vibe-qc's
periodic Coulomb pipeline.

Scans L ∈ {30, 50, 100} bohr to verify 1/L scaling of the
discrepancy.

Run:
    .venv/bin/python examples/ase_compare/cross_code_regression/compare_pbc_molecular_limit.py

CSV: output-pbc-molecular-limit.csv
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
OUT_DIR = HERE / "output" / "pbc-molecular-limit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = HERE / "output-pbc-molecular-limit.csv"

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
ALPHA_M_CUBIC = 2.837   # simple-cubic Madelung constant

try:
    from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
    from pyscf import gto, scf
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


@dataclass
class LimitRow:
    label: str
    L_bohr: float
    n_e: int
    e_mol_vq: Optional[float]      # molecular RHF (vibe-qc)
    e_pbc_vq: Optional[float]      # periodic RHF (vibe-qc)
    e_pbc_pyscf: Optional[float]   # periodic RHF (PySCF.pbc)
    diff_pred: float               # Madelung leak prediction


def _h2(L_bohr: float) -> tuple[vq.Molecule, vq.PeriodicSystem, str]:
    """H2 molecule + same atoms in an L_bohr cubic cell."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    sys_ = vq.PeriodicSystem(
        dim=3, lattice=L_bohr * np.eye(3), unit_cell=list(mol.atoms),
    )
    L_ang = L_bohr / ANGSTROM_TO_BOHR
    pyscf_atom = f"H 0.0 0.0 0.0; H 0.0 0.0 {1.4 / ANGSTROM_TO_BOHR:.6f}"
    return mol, sys_, pyscf_atom


def run_molecular_vibeqc(mol: vq.Molecule) -> Optional[float]:
    try:
        basis = vq.BasisSet(mol, "sto-3g")
        return float(vq.run_rhf(mol, basis).energy)
    except Exception:
        return None


def run_periodic_vibeqc(label: str, sys_: vq.PeriodicSystem,
                        ) -> Optional[float]:
    try:
        basis = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicSCFOptions()
        opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
        opts.lattice_opts.cutoff_bohr = 12.0
        opts.lattice_opts.nuclear_cutoff_bohr = 25.0
        opts.conv_tol_energy = 1e-7
        out_stem = OUT_DIR / label
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                r = vq.run_rhf_periodic_scf(
                    sys_, basis, vq.KPoints.gamma(sys_), opts,
                    progress=False, spacing_bohr=0.4,
                )
        return float(r.energy)
    except Exception:
        return None


def run_periodic_pyscf(pyscf_atom: str, L_bohr: float) -> Optional[float]:
    if not PYSCF_AVAILABLE:
        return None
    try:
        L_ang = L_bohr / ANGSTROM_TO_BOHR
        cell = pbc_gto.M(
            atom=pyscf_atom,
            a=[[L_ang, 0, 0], [0, L_ang, 0], [0, 0, L_ang]],
            basis="sto-3g",
            unit="A",
            verbose=0,
        )
        mf = pbc_scf.RHF(cell)
        mf.max_cycle = 30
        mf.conv_tol = 1e-7
        return float(mf.kernel())
    except Exception:
        return None


def main() -> None:
    print("=" * 72)
    print(" PBC molecular-limit cross-check  —  H2 / RHF / sto-3g")
    print("=" * 72)
    print(f"  PySCF.pbc available: {PYSCF_AVAILABLE}")
    print(f"  Madelung constant (simple cubic): α_M = {ALPHA_M_CUBIC}")

    rows = []
    for L in [30.0, 50.0, 100.0]:
        label = f"H2_L{int(L)}"
        print(f"\n[{label}]  L = {L} bohr")
        mol, sys_, pyscf_atom = _h2(L)
        n_e = sum(int(at.Z) for at in mol.atoms)

        e_mol = run_molecular_vibeqc(mol)
        e_pbc_vq = run_periodic_vibeqc(label, sys_)
        e_pbc_pyscf = run_periodic_pyscf(pyscf_atom, L)
        diff_pred = -ALPHA_M_CUBIC * (n_e ** 2 + n_e ** 2) / (2.0 * L)

        rows.append(LimitRow(label, L, n_e, e_mol, e_pbc_vq,
                             e_pbc_pyscf, diff_pred))

        def fmt(e):
            return f"{e:+.6f} Ha" if e is not None else "—"
        print(f"  molecular RHF (vibe-qc):       {fmt(e_mol)}")
        print(f"  periodic RHF (vibe-qc):        {fmt(e_pbc_vq)}")
        print(f"  periodic RHF (PySCF.pbc):      {fmt(e_pbc_pyscf)}")
        print(f"  predicted Madelung leak:       {diff_pred:+.4f} Ha")

        if e_mol is not None and e_pbc_vq is not None:
            print(f"  vq pbc - vq mol:               "
                  f"{e_pbc_vq - e_mol:+.4f} Ha   (= leak prediction?)")
        if e_pbc_pyscf is not None and e_mol is not None:
            print(f"  pyscf pbc - vq mol:            "
                  f"{e_pbc_pyscf - e_mol:+.4f} Ha")

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "L_bohr", "n_e",
                    "e_mol_vq_ha", "e_pbc_vq_ha", "e_pbc_pyscf_ha",
                    "diff_pred_ha"])
        for r in rows:
            w.writerow([r.label, r.L_bohr, r.n_e,
                        r.e_mol_vq, r.e_pbc_vq, r.e_pbc_pyscf,
                        r.diff_pred])

    print()
    print("=" * 72)
    print(" Summary — 1/L scaling of (vq pbc - vq mol) and (pyscf pbc - vq mol)")
    print("=" * 72)
    print(f"  {'L':>5s}  {'(vq pbc - vq mol)':>22s}  "
          f"{'(pyscf pbc - vq mol)':>22s}  {'predicted':>12s}")
    for r in rows:
        d_vq = (r.e_pbc_vq - r.e_mol_vq
                if (r.e_pbc_vq is not None and r.e_mol_vq is not None)
                else None)
        d_pyscf = (r.e_pbc_pyscf - r.e_mol_vq
                   if (r.e_pbc_pyscf is not None
                       and r.e_mol_vq is not None) else None)
        print(f"  {int(r.L_bohr):5d}  "
              f"{d_vq:+22.4f}  " if d_vq is not None else f"  {int(r.L_bohr):5d}  {' ' * 22}  ",
              f"{d_pyscf:+22.4f}  " if d_pyscf is not None else " " * 24,
              f"{r.diff_pred:+12.4f}")
    print()
    print("  Diagnostics:")
    print("    If both diffs match the prediction → both codes have the leak")
    print("    If only vq-pbc matches the prediction → vq-only bug (this is what we expect)")
    print("    If neither matches → some other bug; investigate")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
