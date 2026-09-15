"""Multi-k compcell GDF SCF parity vs PySCF KRHF.density_fit().

After commits 4c7949a + e379005 the multi-k path in
`run_krhf_periodic_gdf(use_compcell=True)` ships — this script
validates it against PySCF on H2 at a few k-meshes.

H2 is the cleanest test system (all L=0 AOs, no pair-FT convention
issue) so any residual is due to the multi-k SCF wiring itself
(per-q Lpq cache, per-k J/K build, exxdiv='ewald' shift at multi-k)
rather than the L>0 pair-FT issue that blocks tight ionic systems.

PySCF subprocess per CLAUDE.md §10. Both runs use exxdiv='ewald'
and def2-svp-jk(fit) aux.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import vibeqc as vq


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf

# H2 in 12-bohr cubic box, separation 1.4 bohr
cell = pbc_gto.M(
    atom=[("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    a=np.eye(3) * 12.0,
    unit="B",
    basis="sto-3g",
    precision=1e-12,
    verbose=0,
)

KMESH = {kmesh}
kpts = cell.make_kpts(KMESH)

mf = pbc_scf.KRHF(cell, kpts).density_fit(auxbasis="def2-svp-jkfit")
mf.exxdiv = "ewald"
mf.conv_tol = 1e-9
mf.max_cycle = 50
E = mf.kernel()

print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "E_total": float(E),
    "converged": bool(mf.converged),
    "n_iter": int(getattr(mf, "cycles", -1)),
    "n_kpts": int(len(kpts)),
    "kmesh": KMESH,
}}, sort_keys=True))
'''


def run_pyscf(kmesh) -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(kmesh=list(kmesh))
    proc = subprocess.run([py, "-c", script], capture_output=True,
                          text=True, check=False)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    sys.exit(2)


def _h2_system():
    atoms = [vq.Atom(1, [0, 0, -0.7]), vq.Atom(1, [0, 0, 0.7])]
    system = vq.PeriodicSystem(3, np.diag([12.0, 12.0, 12.0]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def vibeqc_compcell(system, basis, kmesh):
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    t0 = time.perf_counter()
    r = vq.run_krhf_periodic_gdf(
        system, basis, kmesh=tuple(kmesh), options=opts,
        aux_basis="def2-svp-jk",
        use_compcell=True,
        compcell_eta=0.25,
        progress=False,
    )
    return r, time.perf_counter() - t0


def main() -> int:
    system, basis = _h2_system()
    print(f"vibeqc {vq.__version__}: multi-k compcell GDF parity vs PySCF")
    print(f"  H2 / 12-bohr cubic / STO-3G / def2-svp-jk / exxdiv='ewald'")
    print()
    print(f"  {'kmesh':>10s}  {'PySCF E':>16s}  {'vibe-qc E':>16s}  "
          f"{'Δ (Ha)':>14s}  {'cnv':>5s}  iter")
    for kmesh in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)]:
        py = run_pyscf(kmesh)
        vqc, dt = vibeqc_compcell(system, basis, kmesh)
        delta = vqc.energy - py["E_total"]
        print(f"  {str(kmesh):>10s}  {py['E_total']:>16.8f}  "
              f"{vqc.energy:>16.8f}  {delta:>+14.4e}  "
              f"{str(vqc.converged):>5s}  {vqc.n_iter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
