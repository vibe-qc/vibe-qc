"""Quantify the libint↔libcint aux convention mismatch on a tiny system.

Molecular limit on H2 / def2-svp-jk:

  - vibe-qc (libint):  M_vqc = compute_2c_eri(aux)
  - PySCF  (libcint): M_pyscf = auxmol.intor("int2c2e")

If the two libraries agreed bit-exactly, the matrices would match
element-wise. If only the primitive-level convention differs (the
prompt's Stage-2 case for SINGLE-PRIMITIVE shells), a diagonal
rescaling would relate them. The prompt's claim for CONTRACTED shells
is that NO diagonal rescaling can reconcile them — there's an
additional libint shell-contraction normalization that breaks the
per-AO rescaling story.

This script:
  1. Builds aux for H2 / def2-svp-jk in both libraries.
  2. Computes M in both, with matched AO ordering.
  3. Reports ||M_vqc − M_pyscf||, the per-shell diagonal ratios
     (M_vqc[i,i] / M_pyscf[i,i]) — if these are uniform per shell,
     a diagonal rescaling exists; if they vary within a shell or
     across atoms, the contraction-normalization bug is real.

PySCF is invoked via subprocess per CLAUDE.md §10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import compute_2c_eri


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf import gto

# H2 geometry passed in Bohr; pyscf wants Ang by default
mol = gto.M(
    atom="H 0 0 -0.37041; H 0 0 0.37041",
    basis="sto-3g",
    unit="A",
)
# Build the aux basis on the same atoms.
auxmol = gto.M(
    atom="H 0 0 -0.37041; H 0 0 0.37041",
    basis="def2-universal-jkfit",   # PySCF name for what vibe-qc calls "def2-svp-jk" approximately
    unit="A",
)
# Use the explicit aux name PySCF knows about as def2 JKfit:
try:
    auxmol = gto.M(atom="H 0 0 -0.37041; H 0 0 0.37041",
                   basis="def2-svp-jkfit", unit="A")
except Exception:
    pass

M = auxmol.intor("int2c2e")
# Capture per-shell metadata so the consumer can align AOs.
# PySCF: bas_atom(i), bas_angular(i), bas_nctr(i), ao_loc.
ao_loc = auxmol.ao_loc_nr()
shells = []
for i in range(auxmol.nbas):
    shells.append({
        "ish": i,
        "atom": int(auxmol.bas_atom(i)),
        "l": int(auxmol.bas_angular(i)),
        "nctr": int(auxmol.bas_nctr(i)),
        "ao_start": int(ao_loc[i]),
        "ao_end": int(ao_loc[i+1]),
    })

print("VIBEQC-PYSCF-RESULT:" + json.dumps({
    "M": M.tolist(),
    "shells": shells,
    "nao": int(auxmol.nao_nr()),
    "basis": "def2-svp-jkfit (pyscf name)",
}, sort_keys=True))
'''


def run_pyscf() -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    proc = subprocess.run([py, "-c", PYSCF_DRIVER],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    print("PySCF subprocess produced no result marker", file=sys.stderr)
    print(proc.stdout, file=sys.stderr)
    sys.exit(2)


def vibeqc_aux_M():
    h_sep_bohr = 1.4
    half = 0.5 * h_sep_bohr
    atoms = [
        vq.Atom(1, [0.0, 0.0, -half]),
        vq.Atom(1, [0.0, 0.0, +half]),
    ]
    # Vacuum box just so we can build a PeriodicSystem if needed; for
    # this diagnostic we only need a Molecule for the aux BasisSet.
    mol = vq.Molecule(atoms)
    aux = vq.BasisSet(mol, "def2-svp-jk")
    M = compute_2c_eri(aux)
    # Per-shell metadata from vibe-qc.
    shells = []
    bf_offset = 0
    for ish, shell in enumerate(aux.shells()):
        L = int(shell.l)
        n_comp = 2*L + 1 if shell.pure else (L+1)*(L+2)//2
        shells.append({
            "ish": ish,
            "l": L,
            "nctr": 1,  # vibe-qc reports per-segment-contracted shells
            "n_comp": n_comp,
            "ao_start": bf_offset,
            "ao_end": bf_offset + n_comp,
            "n_prims": len(shell.exponents),
            "exps": list(map(float, shell.exponents)),
            "coefs": list(map(float, shell.coefficients)),
            "origin": list(map(float, shell.origin)),
        })
        bf_offset += n_comp
    return M, shells, int(aux.nbasis)


def main() -> int:
    print(f"vibeqc {vq.__version__}: H2 / def2-svp-jk molecular 2c metric")
    M_vqc, shells_vqc, nao_vqc = vibeqc_aux_M()
    print(f"  vibe-qc: n_aux={nao_vqc}, n_shells={len(shells_vqc)}")
    print(f"  ||M_vqc||_F = {np.linalg.norm(M_vqc):.4e}")
    print(f"  diag(M_vqc)[:8] = {np.diag(M_vqc)[:8]}")
    print()

    pyscf = run_pyscf()
    M_py = np.array(pyscf["M"])
    nao_py = pyscf["nao"]
    shells_py = pyscf["shells"]
    print(f"  PySCF:   n_aux={nao_py}, n_shells={len(shells_py)}, "
          f"basis={pyscf['basis']}")
    print(f"  ||M_py||_F = {np.linalg.norm(M_py):.4e}")
    print(f"  diag(M_py)[:8] = {np.diag(M_py)[:8]}")
    print()

    if nao_vqc != nao_py:
        print(f"  ✗ AO count mismatch: vibe-qc={nao_vqc} vs PySCF={nao_py}")
        print("    (different aux basis name resolution — re-check naming)")
        # Continue with shell breakdown anyway to surface the structure.
    else:
        print(f"  ✓ AO count matches ({nao_vqc})")

    # Print shell-by-shell breakdown so the user can see whether each
    # SHELL has matching (l, nprim, atom) and what the diagonal-block
    # ratio looks like — that's the strongest signal for the contraction-
    # normalization story.
    print()
    print("Shell-by-shell layout:")
    print(f"  {'ish':>4s}  {'atom':>4s}  {'l':>3s}  {'n_prims':>8s}  "
          f"{'ao_start':>10s}  {'first_exp':>12s}")
    for s in shells_vqc[:20]:
        print(f"  {s['ish']:>4d}  {'?':>4s}  {s['l']:>3d}  "
              f"{s.get('n_prims', '-'):>8}  {s['ao_start']:>10d}  "
              f"{s['exps'][0]:>12.4f}")
    print()
    if nao_vqc == nao_py:
        # Diagonal-ratio sanity per AO.
        ratios = np.diag(M_vqc) / np.diag(M_py)
        print(f"  diag(M_vqc) / diag(M_py): min={ratios.min():.6f}, "
              f"max={ratios.max():.6f}, std={ratios.std():.4e}")
        # Raw error (no permutation applied).
        err_raw = np.linalg.norm(M_vqc - M_py)
        rel_raw = err_raw / max(np.linalg.norm(M_py), 1e-30)
        print(f"  ||M_vqc − M_py||_F (raw) = {err_raw:.4e}  (rel = {rel_raw:.4e})")

        # Try applying the libint→libcint AO permutation.
        # libint:  L=1 (py, pz, px) i.e. m=-1, 0, +1
        # libcint: L=1 (px, py, pz)                 — pyscf convention
        # For L>=2 both use m=-l..+l real-spherical so identity is OK
        # for those (verify by orbital-by-orbital comparison below).
        # Build a per-shell within-shell permutation P_shell such that
        #   libcint_AO[i] = sum_j P_shell[i,j] · libint_AO[j]
        P = np.eye(nao_vqc)
        for s in shells_vqc:
            L = s["l"]
            i0, i1 = s["ao_start"], s["ao_end"]
            if L == 1:
                # libint (py, pz, px) → libcint (px, py, pz)
                # libcint[0]=px=libint[2]; libcint[1]=py=libint[0]; libcint[2]=pz=libint[1]
                p = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]], dtype=float)
                P[i0:i1, i0:i1] = p
            # L=0 and L>=2 left as identity for now.
        M_vqc_pyscf_order = P @ M_vqc @ P.T
        err_perm = np.linalg.norm(M_vqc_pyscf_order - M_py)
        rel_perm = err_perm / max(np.linalg.norm(M_py), 1e-30)
        print(f"  ||M_vqc − M_py||_F (L=1 perm) = {err_perm:.4e}  "
              f"(rel = {rel_perm:.4e})")

        # If still off, list the largest residual elements so we can
        # see whether they're L=2 cross-terms (then we need L>=2 perm).
        if rel_perm > 1e-8:
            diff = M_vqc_pyscf_order - M_py
            idx = np.unravel_index(np.argsort(np.abs(diff), axis=None)[::-1][:10],
                                   diff.shape)
            print("  Top 10 residual elements (i, j, vqc, py, diff):")
            for i, j in zip(*idx):
                # Identify which shell each AO belongs to.
                shi = next((s for s in shells_vqc
                            if s["ao_start"] <= i < s["ao_end"]), None)
                shj = next((s for s in shells_vqc
                            if s["ao_start"] <= j < s["ao_end"]), None)
                Li = shi["l"] if shi else "?"
                Lj = shj["l"] if shj else "?"
                print(f"    ({i:>2d}, {j:>2d}) L={Li},{Lj}  "
                      f"vqc={M_vqc_pyscf_order[i,j]:>12.6f}  "
                      f"py={M_py[i,j]:>12.6f}  "
                      f"diff={diff[i,j]:>12.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
