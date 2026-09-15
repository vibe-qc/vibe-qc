"""Empirical per-AO libint-vs-libcint normalization calibration.

Theory says: the ONLY convention difference between libint and libcint
is the spherical-harmonic normalization (libint uses solid-harmonic
``Y_libint = √(4π/(2L+1)) · Y_standard``; libcint uses standard
``Y = Y_standard``). For SINGLE-PRIMITIVE shells this is the only
factor and the per-AO conversion is ``√(4π/(2L+1))``.

For CONTRACTED shells (every realistic AO basis), libint applies an
ADDITIONAL per-shell renormalization on top — empirically the bare
2c metric ratio on H2/def2-svp-jk decomposed as ``4π × ~2.4`` for
L=0 contracted shells (4π from Y_lm, 2.4 from libint's contraction).

This script measures the true per-AO conversion factor empirically by
comparing the diagonal of the OVERLAP MATRIX in both conventions:

    scale_AO[i] = √(<χ_i|χ_i>_libint / <χ_i|χ_i>_libcint)

This is the factor that converts vibe-qc's libint single-AO FT to
libcint single-AO FT. For the PAIR density, the conversion is the
PRODUCT of the two per-AO scales.

PySCF is invoked out-of-process per CLAUDE.md §10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import compute_overlap


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf import gto

# Same Li-H 2-atom system as the pair-FT diff
mol = gto.M(
    atom=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))],
    basis="sto-3g",
    unit="B",
    verbose=0,
)
S = mol.intor("int1e_ovlp")
ao_loc = mol.ao_loc_nr()
shells = []
for i in range(mol.nbas):
    shells.append({
        "ish": i,
        "atom": int(mol.bas_atom(i)),
        "l": int(mol.bas_angular(i)),
        "ao_start": int(ao_loc[i]),
        "ao_end": int(ao_loc[i+1]),
    })
print("VIBEQC-PYSCF-RESULT:" + json.dumps({
    "nao": int(mol.nao_nr()),
    "S": S.tolist(),
    "shells": shells,
}, sort_keys=True))
'''


def run_pyscf() -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    proc = subprocess.run([py, "-c", PYSCF_DRIVER], capture_output=True, text=True)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    sys.exit(2)


def main() -> int:
    # vibe-qc side
    mol = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0]),
                       vq.Atom(1, [0.0, 0.0, 1.5])])
    basis = vq.BasisSet(mol, "sto-3g")
    n = basis.nbasis
    S_libint = np.asarray(compute_overlap(basis))
    print(f"vibeqc {vq.__version__}: per-AO scale empirical calibration")
    print(f"  Li-H 2-atom STO-3G, n_ao={n}")

    py = run_pyscf()
    S_libcint = np.array(py["S"])

    # Apply L=1 permutation to put libint→libcint AO ordering
    P = np.eye(n)
    bf = 0
    for sh in basis.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        if L == 1:
            P[bf:bf+3, bf:bf+3] = np.array([[0, 0, 1],
                                             [1, 0, 0],
                                             [0, 1, 0]], dtype=float)
        bf += nc
    S_libint_pyscf_order = P @ S_libint @ P.T

    print()
    print(f"  {'i':>3s}  {'l':>2s}  {'S_libint[i,i]':>15s}  {'S_libcint[i,i]':>15s}  "
          f"{'ratio':>10s}  {'sqrt(ratio)':>12s}  {'expected √(4π/(2L+1))':>22s}")
    bf = 0
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    for sh in basis.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        expected = sqrt4pi / np.sqrt(2 * L + 1)
        for m in range(nc):
            i = bf + m
            r = S_libint_pyscf_order[i, i] / S_libcint[i, i]
            sqrt_r = np.sqrt(abs(r))
            sign = "" if r > 0 else " (NEGATIVE!)"
            print(f"  {i:>3d}  {L:>2d}  "
                  f"{S_libint_pyscf_order[i,i]:>15.6e}  "
                  f"{S_libcint[i,i]:>15.6e}  "
                  f"{r:>10.4f}  "
                  f"{sqrt_r:>12.4f}{sign}  "
                  f"{expected:>22.4f}")
        bf += nc

    print()
    print("Interpretation:")
    print("  - If sqrt(ratio) == expected √(4π/(2L+1)), only Y_lm convention")
    print("    differs (single-primitive shells, no extra contraction norm).")
    print("  - If sqrt(ratio) != expected, libint applies additional per-shell")
    print("    contraction normalization that libcint does NOT. The factor")
    print("    sqrt(ratio) IS the empirical per-AO conversion factor.")
    print()
    print("  For the AO-pair FT, the per-pair conversion is")
    print("  sqrt(ratio[μ]) · sqrt(ratio[ν]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
