#!/usr/bin/env python
"""Multi-k Ewald UHF against PySCF KUHF on open-shell H cells (GitLab #399).

Until #399 ``run_uhf_periodic_multi_k_ewald3d`` summed its full-range exchange
as a bare ``1/r`` kernel over a radial image ball. On a finite k mesh the SCF
density is Born-von-Karman-torus periodic, so D(g) does not decay and that sum
does not converge: the energy fell by Hartrees on compact cells and kept falling
as the lattice cutoff grew. The driver now takes the corrected Ewald split the
RHF and UKS siblings use (PySCF's ``exxdiv='ewald'``) on a real mesh.

This script compares both open-shell states against PySCF KUHF/GDF with
``exxdiv='ewald'`` on the dilute cubic H cell of
``parity_ccm_scm_vs_pyscf.py`` (a = 8 bohr), and prints the literals that
``tests/test_periodic_uhf_multi_k_ewald.py`` pins. The two codes use different
Coulomb methods (Ewald-3D FFT vs Gaussian density fitting), so they agree to
the Coulomb-method gap, measured at 19-58 uHa for the closed-shell cell.

PySCF runs in a child interpreter and is parsed back; vibe-qc never imports it.

Usage:
    python examples/regression/parity_uhf_multik_ewald_vs_pyscf.py
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

A_BOHR = 8.0
D_BOHR = 1.4
KMESHES = ((2, 2, 2),)

# name -> (atoms in bohr, 2S = n_alpha - n_beta per cell)
STATES = {
    "h2_triplet": ([["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, D_BOHR]]], 2),
    "h_doublet": ([["H", [0.0, 0.0, 0.0]]], 1),
}

_PYSCF_CHILD = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto, scf
p = json.loads(sys.argv[1])
out = {}
for name, (atoms, two_s) in p["states"].items():
    cell = gto.Cell()
    cell.a = np.eye(3) * p["a"]
    cell.atom = atoms
    cell.unit = "Bohr"; cell.basis = "sto3g"; cell.verbose = 0
    cell.spin = int(two_s)
    cell.precision = 1e-10
    cell.build()
    ne = sum(1 for _ in atoms)  # hydrogen: one electron per atom
    na, nb = (ne + two_s) // 2, (ne - two_s) // 2
    for km in p["kmeshes"]:
        nk = int(np.prod(km))
        mf = scf.KUHF(cell, cell.make_kpts(km)).density_fit()
        # KUHF counts electrons over the whole k supercell.
        mf.nelec = (na * nk, nb * nk)
        mf.exxdiv = "ewald"
        mf.conv_tol = 1e-10
        # e_tot is already per unit cell.
        out[f"{name}|{tuple(km)}"] = float(mf.kernel())
print("PYSCF-RESULT:" + json.dumps(out), flush=True)
"""


def pyscf_kuhf():
    proc = subprocess.run(
        [sys.executable, "-c", _PYSCF_CHILD, json.dumps({
            "a": A_BOHR,
            "states": STATES,
            "kmeshes": [list(k) for k in KMESHES],
        })],
        capture_output=True, text=True, check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PYSCF-RESULT:"):
            return json.loads(line[len("PYSCF-RESULT:"):])
    raise RuntimeError("PySCF child printed no result:\n" + proc.stderr)


def vibeqc_kuhf(name, kmesh):
    import vibeqc as vq

    atoms, two_s = STATES[name]
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * A_BOHR,
        [vq.Atom(1, xyz) for _, xyz in atoms],
        0,
        int(two_s) + 1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    res = vq.run_uhf_periodic_multi_k_ewald3d(
        system, basis, vq.monkhorst_pack(system, list(kmesh))
    )
    return float(res.energy), bool(res.converged)


def main() -> int:
    ref = pyscf_kuhf()
    print(f"{'state':12s} {'mesh':>10s} {'vibe-qc UHF':>18s} {'PySCF KUHF':>18s}"
          f" {'diff (Ha)':>11s}")
    for name in STATES:
        for km in KMESHES:
            e_vq, conv = vibeqc_kuhf(name, km)
            e_py = ref[f"{name}|{tuple(km)}"]
            print(f"{name:12s} {str(km):>10s} {e_vq:+18.10f} {e_py:+18.10f}"
                  f" {e_vq - e_py:+11.2e}{'' if conv else '  (not converged)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
