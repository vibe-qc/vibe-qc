#!/usr/bin/env python3
"""ASE-cell lattice-convention parity: vibe-qc vs PySCF on a hexagonal cell
(out-of-process per CLAUDE.md §10).

Reproduces the fix for the latent transpose bug at the ASE→PeriodicSystem
boundary (``vibeqc.ase_periodic.atoms_to_periodic_system`` and the GPW/GAPW
converters). ASE stores lattice vectors as the ROWS of ``atoms.cell``; vibe-qc's
``PeriodicSystem`` stores them as the COLUMNS of its lattice matrix
(``cpp/include/vibeqc/periodic.hpp:32``). The boundary must transpose; omitting
it silently transposed every non-orthogonal cell.

This script confirms the *direction* of the fix against an external reference:

  1. PySCF (``cell.a = rows = a_i`` — PySCF's documented row convention) and
     vibe-qc (``columns = a_i``) build the IDENTICAL lattice and agree to a few
     mHa on a dilute hexagonal He cell (gamma, STO-3G, in the converged limit)
     — so ``columns = a_i`` is the physical interpretation.
  2. The transposed (pre-fix) orientation is a genuinely different crystal and
     does not track PySCF.

PySCF is executed in a child interpreter and parsed back; vibe-qc never imports
it (CLAUDE.md §10).

Usage:
    python examples/regression/parity_hexagonal_lattice_vs_pyscf.py
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

ANG2BOHR = 1.0 / 0.529177210903

# PySCF child: imports pyscf, builds the cell with cell.a = rows (a_i), runs
# gamma KRHF/GDF, and reports both its internal lattice_vectors() and the energy.
_PYSCF_CHILD = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto, scf
p = json.loads(sys.argv[1])
cell = gto.Cell()
cell.a = np.asarray(p["lattice_rows_bohr"], float)   # rows = lattice vectors
cell.atom = [[s, np.asarray(x, float)] for s, x in p["atoms_bohr"]]
cell.unit = "Bohr"; cell.basis = p["basis"]; cell.verbose = 0
cell.build()
lv = np.asarray(cell.lattice_vectors())
mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1])).density_fit()
mf.conv_tol = 1e-10
e = float(mf.kernel())
print("PYSCF-RESULT:" + json.dumps(
    {"e": e, "lattice_vectors": [lv[i].tolist() for i in range(3)],
     "converged": bool(mf.converged)}), flush=True)
"""


def _hex_rows(a_bohr: float, c_bohr: float) -> np.ndarray:
    """Hexagonal cell, ASE/PySCF row convention (rows = a1, a2, a3)."""
    return np.array([[a_bohr, 0.0, 0.0],
                     [-a_bohr / 2.0, a_bohr * np.sqrt(3.0) / 2.0, 0.0],
                     [0.0, 0.0, c_bohr]])


def vibeqc_gamma_rhf(lattice_cols, atoms_bohr, basis="sto-3g", cutoff=24.0):
    """vibe-qc gamma RHF (Ewald-3D). ``lattice_cols`` columns are the vectors."""
    import vibeqc as vq

    system = vq.PeriodicSystem(
        3, np.asarray(lattice_cols, float),
        [vq.Atom(int(z), list(x)) for z, x in atoms_bohr])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 400
    result = vq.run_rhf_periodic_scf(
        system, vq.make_basis(system.unit_cell_molecule(), basis),
        vq.KPoints.gamma(system), opts)
    return float(result.energy), np.asarray(system.lattice)


def pyscf_gamma_rhf(lattice_rows_bohr, atoms_bohr, basis="sto-3g"):
    payload = {"lattice_rows_bohr": np.asarray(lattice_rows_bohr).tolist(),
               "atoms_bohr": [[s, list(map(float, x))] for s, x in atoms_bohr],
               "basis": basis}
    out = subprocess.run([sys.executable, "-c", _PYSCF_CHILD, json.dumps(payload)],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PYSCF-RESULT:"):
            return json.loads(line[len("PYSCF-RESULT:"):])
    raise RuntimeError("no PySCF result\nSTDERR:\n" + out.stderr[-2000:])


def main() -> int:
    a = 11.0 * 1.0   # bohr; dilute so vibe-qc-Ewald and PySCF-GDF converge together
    c = a * 1.3
    rows = _hex_rows(a, c)                 # rows = a_i  (ASE / PySCF convention)
    atoms = [("He", [0.0, 0.0, 0.0])]
    z_atoms = [(2, xyz) for _, xyz in atoms]

    print("Hexagonal He / STO-3G / gamma — ASE-cell lattice-convention parity")
    print(f"  lattice rows a_i (bohr): {rows.tolist()}")

    e_fixed, L_fixed = vibeqc_gamma_rhf(rows.T, z_atoms)   # columns = a_i  (FIXED)
    e_buggy, L_buggy = vibeqc_gamma_rhf(rows, z_atoms)     # columns = cols(rows) (PRE-FIX)
    ref = pyscf_gamma_rhf(rows, atoms)
    e_py = ref["e"]

    print("\nLattice vectors actually integrated:")
    print(f"  vibe-qc FIXED  (cols=a_i): {[L_fixed[:, i].tolist() for i in range(3)]}")
    print(f"  PySCF cell.a=rows         : {ref['lattice_vectors']}")
    same_geom = np.allclose(np.array([L_fixed[:, i] for i in range(3)]),
                            np.array(ref["lattice_vectors"]), atol=1e-9)
    print(f"  -> identical geometry: {same_geom}")

    print("\nGamma RHF total energies (Ha):")
    print(f"  vibe-qc FIXED  (cols=a_i)      = {e_fixed:.6f}")
    print(f"  vibe-qc PRE-FIX (transposed)   = {e_buggy:.6f}")
    print(f"  PySCF KRHF/GDF (rows=a_i)      = {e_py:.6f}")
    gap = abs(e_fixed - e_py) * 1e3
    print(f"\n  |FIXED - PySCF| = {gap:.2f} mHa  (same geometry; gamma/min-basis method gap)")

    ok = same_geom and gap < 5.0
    print(f"\n{'PASS' if ok else 'FAIL'}: columns = a_i reproduces PySCF; the transpose fix "
          f"is in the correct direction.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
