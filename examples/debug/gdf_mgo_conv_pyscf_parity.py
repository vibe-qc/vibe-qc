"""v0.9.0 prep datapoint: MgO 8-atom conv cell PBCGDF parity to PySCF.

The MgO headline number from
``examples/regression/crystal_parity/baseline_sto3g/PARITY_TABLE.md`` was
the v0.8.0 prep doc's flagship periodic GDF datapoint. After the
release chat's audit it was moved out of v0.8.0 (deferred to v0.9.0).
This script runs the apples-to-apples comparison so the periodic-GDF
chat has a number to track going into v0.9.0:

  - vibe-qc PBCGDF compcell (Γ-only, exxdiv='ewald', def2-svp-jk),
    swept across compcell_eta with apply_aft_correction=True and
    rcut_strategy='pyscf_auto' (the H2-sub-mHa configuration).
  - PySCF KRHF.density_fit() at Γ-only with exxdiv='ewald', same cell
    + same basis + same aux, invoked out-of-process per CLAUDE.md §10.

Output: side-by-side table of E_total per η, with the SCF-converged
flag, iteration count, and the Δ vs PySCF in Ha. The CRYSTAL14 baseline
(SHRINK 8 8) is a different BZ sampling and is reported separately for
context but is NOT the parity target here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import vibeqc as vq


ANG2BOHR = 1.0 / 0.529177210903
A_MGO_ANG = 4.21


def build_mgo_conv():
    a = A_MGO_ANG * ANG2BOHR
    cation_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    anion_frac  = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = [vq.Atom(12, [fx * a, fy * a, fz * a]) for fx, fy, fz in cation_frac]
    atoms += [vq.Atom(8,  [fx * a, fy * a, fz * a]) for fx, fy, fz in anion_frac]
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis, a


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf

A_MGO_ANG = {a_ang}

cation_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
anion_frac  = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
atom = []
for fx, fy, fz in cation_frac:
    atom.append(("Mg", (fx * A_MGO_ANG, fy * A_MGO_ANG, fz * A_MGO_ANG)))
for fx, fy, fz in anion_frac:
    atom.append(("O",  (fx * A_MGO_ANG, fy * A_MGO_ANG, fz * A_MGO_ANG)))

cell = pbc_gto.M(
    atom=atom,
    a=np.eye(3) * A_MGO_ANG,
    unit="A",
    basis="sto-3g",
    precision=1e-10,
    verbose=0,
)
mf = pbc_scf.RHF(cell).density_fit(auxbasis="def2-svp-jkfit")
mf.exxdiv = "ewald"
mf.conv_tol = 1e-8
mf.max_cycle = 100
E = mf.kernel()
print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "E_total": float(E),
    "converged": bool(mf.converged),
    "n_iter": int(getattr(mf, "cycles", -1)),
    "n_elec": int(cell.nelectron),
    "n_ao": int(cell.nao_nr()),
}}, sort_keys=True))
'''


def run_pyscf(a_ang: float) -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(a_ang=a_ang)
    proc = subprocess.run([py, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    print("PySCF subprocess produced no result marker", file=sys.stderr)
    print("stdout (tail):", proc.stdout[-2000:], file=sys.stderr)
    print("stderr (tail):", proc.stderr[-2000:], file=sys.stderr)
    sys.exit(2)


def main() -> int:
    system, basis, a_bohr = build_mgo_conv()
    n_elec = system.n_electrons()
    print(f"vibeqc {vq.__version__}  —  MgO 8-atom conv cell, a={A_MGO_ANG} Å")
    print(f"  n_ao = {basis.nbasis}, n_electrons = {n_elec}")
    print()

    print("PySCF KRHF.density_fit (Γ-only, exxdiv='ewald', def2-svp-jkfit)...")
    pyscf = run_pyscf(A_MGO_ANG)
    print(f"  E_total      = {pyscf['E_total']:.6f} Ha")
    print(f"  converged    = {pyscf['converged']}, n_iter = {pyscf['n_iter']}")
    print(f"  n_ao={pyscf['n_ao']}, n_elec={pyscf['n_elec']}")
    print()

    target = pyscf["E_total"]
    print(f"vibe-qc PBCGDF compcell, η sweep "
          f"(apply_aft_correction=True, aft_ft_convention=..., "
          f"rcut_strategy='pyscf_auto'):")
    print(f"  {'conv':>8s}  {'eta':>5s}  {'E_total':>16s}  {'Δ vs PySCF':>14s}  cnv  iter")
    for ft_conv in ("libint", "libcint"):
        for eta in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
            opts = vq.PeriodicRHFOptions()
            opts.use_diis = True
            opts.damping = 0.0
            opts.max_iter = 60
            opts.conv_tol_energy = 1e-7
            opts.lattice_opts.cutoff_bohr = 30.0
            opts.lattice_opts.nuclear_cutoff_bohr = 30.0
            try:
                r = vq.run_pbc_gdf_rhf(
                    system, basis, opts,
                    aux_basis="def2-svp-jk",
                    exxdiv="ewald",
                    compcell_eta=eta,
                    apply_aft_correction=True,
                    aft_ft_convention=ft_conv,
                    rcut_strategy="pyscf_auto",
                    progress=False,
                )
                delta = r.energy - target
                print(f"  {ft_conv:>8s}  {eta:>4.2f}  {r.energy:>16.6f}  "
                      f"{delta:>+14.4e}  {str(r.converged):>3s}  {r.n_iter}")
            except Exception as exc:
                print(f"  {ft_conv:>8s}  {eta:>4.2f}  ERR: "
                      f"{type(exc).__name__}: {str(exc)[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
