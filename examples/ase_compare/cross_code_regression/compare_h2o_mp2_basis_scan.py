"""H2O / MP2 / 4 bases — vibe-qc vs PySCF vs ORCA.

MP2 was never wired through the (now retired) PySCFCalculator ASE
shim, so this dev script builds the comparison directly via each
code's native MP2 entry point. It is a dev-handover helper under
``examples/``, not part of the vibe-qc runtime: the in-process
``import pyscf`` below is outside the CLAUDE.md sec 10 boundary,
which only forbids in-process imports of external QC programs
under ``python/vibeqc/`` and ``cpp/``. Reports both the SCF
reference and the MP2 correlation energy:

    label                E(SCF)         E(MP2 corr)    E(total)

Bases (small → bigger):
    sto-3g          — minimal
    6-31g*          — small Pople
    cc-pVDZ         — Dunning DZ
    cc-pVTZ         — Dunning TZ (largest exercise of integrals)

Expected: SCF energies match to ~µHa across all three codes (same
basis = same SCF). All three MP2 calls are explicitly all-electron:
vibe-qc uses ``n_frozen_core=0``, PySCF uses ``frozen=0``, and ORCA
uses ``NoFrozenCore``. The MP2 correlation energy should therefore
also match to ~µHa — there's no XC grid or DIIS subtlety. Any
divergence here points to vibe-qc's MP2 integral transform / amplitude
solve rather than to the post-#140 default convention.

Run:
    .venv/bin/python examples/ase_compare/cross_code_regression/compare_h2o_mp2_basis_scan.py

CSV: output-h2o-mp2-basis-scan.csv
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ase.build import molecule

import vibeqc as vq

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "output-h2o-mp2-basis-scan.csv"

BASES = ["sto-3g", "6-31g*", "cc-pvdz", "cc-pvtz"]
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Optional codes
try:
    from pyscf import gto, scf, mp
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


@dataclass
class Row:
    code: str
    basis: str
    e_scf: Optional[float]
    e_mp2_corr: Optional[float]
    e_total: Optional[float]
    wall_s: float
    note: str


def _ase_to_pyscf_atoms(atoms) -> str:
    return "; ".join(
        f"{atoms[i].symbol} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
        for i, p in enumerate(atoms.positions)
    )


def _ase_to_vq_mol(atoms) -> vq.Molecule:
    """ASE Atoms (Å) → vq.Molecule (bohr)."""
    return vq.Molecule([
        vq.Atom(int(z), [p * ANGSTROM_TO_BOHR for p in pos])
        for z, pos in zip(atoms.numbers, atoms.positions)
    ])


def run_vibeqc(atoms, basis_name: str) -> Row:
    rec = Row("vibe-qc", basis_name, None, None, None, 0.0, "")
    try:
        mol = _ase_to_vq_mol(atoms)
        basis = vq.BasisSet(mol, basis_name)
        t0 = time.perf_counter()
        scf_res = vq.run_rhf(mol, basis)
        mp2_options = vq.MP2Options()
        mp2_options.n_frozen_core = 0
        mp2_res = vq.run_mp2(mol, basis, scf_res, mp2_options)
        rec.wall_s = time.perf_counter() - t0
        rec.e_scf = float(scf_res.energy)
        rec.e_mp2_corr = float(mp2_res.energy_correlation)
        rec.e_total = rec.e_scf + rec.e_mp2_corr
        rec.note = "ok"
    except Exception as exc:
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def run_pyscf(atoms, basis_name: str) -> Row:
    rec = Row("pyscf", basis_name, None, None, None, 0.0, "")
    if not PYSCF_AVAILABLE:
        rec.note = "pyscf not installed"
        return rec
    try:
        mol = gto.M(
            atom=_ase_to_pyscf_atoms(atoms),
            basis=basis_name,
            unit="Angstrom",
            verbose=0,
        )
        t0 = time.perf_counter()
        mf = scf.RHF(mol).run()
        mp2 = mp.MP2(mf, frozen=0).run()
        rec.wall_s = time.perf_counter() - t0
        rec.e_scf = float(mf.e_tot)
        rec.e_mp2_corr = float(mp2.e_corr)
        rec.e_total = float(mp2.e_tot)
        rec.note = "ok"
    except Exception as exc:
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def run_orca(atoms, basis_name: str) -> Row:
    """ORCA-via-ASE driver. Skips if orca isn't on PATH."""
    rec = Row("orca", basis_name, None, None, None, 0.0, "")
    import shutil
    if not (shutil.which("orca")
            or os.environ.get("ORCA_COMMAND")
            or os.environ.get("ASE_ORCA_COMMAND")):
        rec.note = "orca not on PATH"
        return rec
    try:
        from ase.calculators.orca import ORCA, OrcaProfile
        atoms2 = atoms.copy()
        out_dir = HERE / f"_orca_h2o_mp2_{basis_name.replace('*', 's')}"
        out_dir.mkdir(exist_ok=True)
        atoms2.calc = ORCA(
            profile=OrcaProfile(command=(
                os.environ.get("ORCA_COMMAND")
                or os.environ.get("ASE_ORCA_COMMAND")
                or shutil.which("orca")
            )),
            directory=str(out_dir),
            orcasimpleinput=f"MP2 {basis_name} NoFrozenCore",
            label="orca-h2o-mp2",
        )
        t0 = time.perf_counter()
        rec.e_total = atoms2.get_potential_energy() / 27.211386245988  # eV → Ha
        rec.wall_s = time.perf_counter() - t0
        # ORCA's ASE wrapper doesn't expose the SCF / corr split — read
        # from the .out file post-hoc:
        out_file = out_dir / "orca-h2o-mp2.out"
        if out_file.exists():
            text = out_file.read_text()
            for line in text.splitlines():
                if "Total Energy" in line and "Eh" in line:
                    pass  # header line
                if line.startswith("E(SCF)") or "E(0)" in line:
                    try:
                        rec.e_scf = float(line.split()[-2])
                    except Exception:
                        pass
                if "MP2 CORRELATION ENERGY" in line:
                    try:
                        rec.e_mp2_corr = float(line.split()[-1])
                    except Exception:
                        pass
        rec.note = "ok"
    except Exception as exc:
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"
    return rec


def main() -> None:
    print("=" * 72)
    print(" H2O / MP2 / 4-basis scan  —  vibe-qc vs PySCF vs ORCA")
    print("=" * 72)
    print(f"  PySCF available: {PYSCF_AVAILABLE}")

    atoms = molecule("H2O")

    rows: list[Row] = []
    for basis in BASES:
        print(f"\n--- basis = {basis} ---")
        for runner, name in [
            (run_vibeqc, "vibeqc"),
            (run_pyscf, "pyscf"),
            (run_orca, "orca"),
        ]:
            r = runner(atoms, basis)
            rows.append(r)
            flag = "✓" if r.note == "ok" else "✗"
            scf_s = f"E_SCF={r.e_scf:.6f}" if r.e_scf is not None else "—"
            mp2_s = f"E_corr={r.e_mp2_corr:+.6f}" if r.e_mp2_corr is not None else "—"
            tot_s = f"E_tot={r.e_total:.6f}" if r.e_total is not None else "—"
            print(f"  {flag} {name:6s}  {scf_s:>18s}  {mp2_s:>20s}  "
                  f"{tot_s:>18s}  wall={r.wall_s:.1f}s  {r.note}")

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basis", "code", "e_scf_ha", "e_mp2_corr_ha",
                    "e_total_ha", "wall_s", "note"])
        for r in rows:
            w.writerow([r.basis, r.code, r.e_scf, r.e_mp2_corr,
                        r.e_total, r.wall_s, r.note])

    print()
    print("=" * 72)
    print(" Pivoted summary — total energies (Ha) by basis × code")
    print("=" * 72)
    print(f"  {'basis':10s}  {'vibe-qc':>14s}  {'PySCF':>14s}  {'ORCA':>14s}  "
          f"{'Δ(vq-pyscf)':>12s}")
    for basis in BASES:
        cells = [basis]
        es = {}
        for code in ("vibe-qc", "pyscf", "orca"):
            r = next((rr for rr in rows if rr.basis == basis and rr.code == code), None)
            v = r.e_total if (r and r.e_total is not None) else None
            es[code] = v
            cells.append(f"{v:14.6f}" if v is not None else " " * 14)
        if es["vibe-qc"] is not None and es["pyscf"] is not None:
            cells.append(f"{(es['vibe-qc'] - es['pyscf']) * 1e6:+12.2f} µHa")
        else:
            cells.append(" " * 12)
        print(f"  {cells[0]:10s}  {cells[1]}  {cells[2]}  {cells[3]}  {cells[4]}")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    import os
    main()
