"""Element-wise pair-FT comparison: vibe-qc vs PySCF.

The H2 sub-mHa SCF win + LiH divergence pattern in the AFT pipeline
points at the AO-PAIR FT convention as the remaining unsolved
convention question (single-AO FT and the j2c_p subtraction both match
PySCF bit-perfectly per earlier diagnostics).

This script computes the pair-density FT ρ̂_μν(G) = ∫ χ_μ(r) χ_ν(r)
e^{-iG·r} d³r on a SHRUNK Li+H 2-atom system (no periodicity needed —
we only want to compare the molecular pair FT, since at large
intermolecular distance the periodic Bloch-summed pair density
reduces to the molecular one for tight atomic shells):

  vibe-qc:  _aopair_ft.ao_pair_fourier_transform(ao_basis, G)
             (libint convention)
  PySCF:    ft_ao.ft_aopair_kpts(cell, G)  via subprocess
             (libcint convention)

Then applies the L=1 (py,pz,px)↔(px,py,pz) AO permutation AND the
per-AO convention factor (√(4π/(2L+1))) and compares element-wise.

If the per-AO factor is sufficient → all blocks should agree to
quadrature precision (~1e-5).
If a higher-order Gaunt-expansion correction is needed for L>0 pair
components → the L=0/L=0 block agrees but L=0/L=1, L=1/L=1, etc.
show residuals — pointing at exactly which (L_μ, L_ν) pair conventions
are wrong.

PySCF is invoked out-of-process per CLAUDE.md §10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import vibeqc as vq
from vibeqc._aopair_ft import ao_pair_fourier_transform


# Small Li-H test (Li brings L=0, L=1 contracted shells; H brings L=0)
LI_POS_BOHR = (0.0, 0.0, 0.0)
H_POS_BOHR  = (0.0, 0.0, 1.5)


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.df import ft_ao

LI = ({li_x}, {li_y}, {li_z})
H  = ({h_x}, {h_y}, {h_z})

# Large vacuum box so the periodic Bloch sum collapses to molecular.
A = 60.0
cell = pbc_gto.M(
    atom=[("Li", LI), ("H", H)],
    a=np.eye(3) * A,
    unit="B",
    basis="sto-3g",
    precision=1e-12,
    verbose=0,
)
nao = cell.nao_nr()

# Small G-mesh: enough to expose convention factors, small enough to
# diff quickly. Place a few G vectors at various |G|.
Gv = np.array([
    [0.1, 0.0, 0.0],
    [0.0, 0.0, 0.5],
    [0.3, 0.4, 0.5],
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
])

# ft_aopair returns ρ̂_μν(G) of shape (n_G, nao, nao), complex
Gpq = ft_ao.ft_aopair(cell, Gv)

# Per-shell metadata so the caller can align AOs
ao_loc = cell.ao_loc_nr()
shells = []
for i in range(cell.nbas):
    shells.append({{
        "ish": i,
        "atom": int(cell.bas_atom(i)),
        "l": int(cell.bas_angular(i)),
        "n_prims": int(cell.bas_nprim(i)),
        "ao_start": int(ao_loc[i]),
        "ao_end": int(ao_loc[i+1]),
        "exps": [float(x) for x in cell.bas_exp(i)],
    }})

print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "nao": int(nao),
    "Gv_real": Gv.real.tolist(),
    "Gpq_real": Gpq.real.tolist(),
    "Gpq_imag": Gpq.imag.tolist(),
    "shells": shells,
}}, sort_keys=True))
'''


def run_pyscf() -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(
        li_x=LI_POS_BOHR[0], li_y=LI_POS_BOHR[1], li_z=LI_POS_BOHR[2],
        h_x=H_POS_BOHR[0],  h_y=H_POS_BOHR[1],  h_z=H_POS_BOHR[2],
    )
    proc = subprocess.run([py, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    print("PySCF subprocess produced no result marker", file=sys.stderr)
    print(proc.stdout[-2000:], file=sys.stderr)
    sys.exit(2)


def build_libint_l1_perm(basis: vq.BasisSet) -> np.ndarray:
    """Return (n_ao, n_ao) permutation P such that
    M_libcint_order = P · M_libint_order · P.T

    libint p-shell order: (m=-1, m=0, m=+1) = (py, pz, px)
    libcint p-shell order: (px, py, pz)
    """
    n = basis.nbasis
    P = np.eye(n)
    bf = 0
    for shell in basis.shells():
        L = int(shell.l)
        nc = 2 * L + 1
        if L == 1:
            # libcint[0]=px=libint[2]; libcint[1]=py=libint[0]; libcint[2]=pz=libint[1]
            P[bf:bf+3, bf:bf+3] = np.array([[0, 0, 1],
                                             [1, 0, 0],
                                             [0, 1, 0]], dtype=float)
        bf += nc
    return P


def build_per_ao_scale(basis: vq.BasisSet) -> np.ndarray:
    """libint single-AO FT / libcint single-AO FT = √(4π/(2L+1))."""
    n = basis.nbasis
    s = np.empty(n)
    bf = 0
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    for shell in basis.shells():
        L = int(shell.l)
        nc = 2 * L + 1
        scale = sqrt4pi / np.sqrt(2 * L + 1)
        s[bf:bf+nc] = scale
        bf += nc
    return s


def main() -> int:
    print(f"vibeqc {vq.__version__}: pair-FT vs PySCF on Li-H 2-atom")
    print(f"  Li at {LI_POS_BOHR}, H at {H_POS_BOHR} (bohr)")

    # vibe-qc side
    mol = vq.Molecule([vq.Atom(3, list(LI_POS_BOHR)),
                       vq.Atom(1, list(H_POS_BOHR))])
    basis = vq.BasisSet(mol, "sto-3g")
    n_orb = basis.nbasis

    # Same G-mesh as PySCF driver
    Gv = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.3, 0.4, 0.5],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])

    rho_libint = ao_pair_fourier_transform(basis, Gv)  # (n_orb, n_orb, n_G)

    # PySCF side
    py = run_pyscf()
    if py["nao"] != n_orb:
        print(f"  AO count mismatch: vibe-qc={n_orb} vs PySCF={py['nao']}")
        return 1
    # PySCF returns (n_G, nao, nao); transpose to (nao, nao, n_G) for diff
    rho_py = np.array(py["Gpq_real"]) + 1j * np.array(py["Gpq_imag"])
    rho_py = rho_py.transpose(1, 2, 0)  # (nao, nao, n_G)

    # 2026-05-17 FINDING (refined 2026-05-18 after AFT-q residue diag):
    # the L=0 components are convention-identical between libint and
    # libcint (S_ii=1 in both → χ_μ·χ_ν is the same pair density).
    # For L>0 single-prim shells, an empirical per-AO SIGN+SCALE
    # factor was found (production: _per_ao_pair_libint_to_libcint_scale)
    # — magnitude 1/√(4π/(2L+1)), sign (-1)^L. Apply the same here so
    # the diff mirrors what production computes; residual blocks point
    # to L_μ,L_ν pairs where the per-AO calibration is INCOMPLETE
    # (e.g., contracted vs single-prim, or higher-L Gaunt coupling).
    P = build_libint_l1_perm(basis)
    from vibeqc.aux_basis import _per_ao_pair_libint_to_libcint_scale
    s_pair = _per_ao_pair_libint_to_libcint_scale(basis)
    inv_s = 1.0 / s_pair
    # vibe-qc → libcint convention via the per-AO sign+scale factor
    # (libint pair FT / s[μ]·s[ν] = libcint pair FT).
    rho_libcint_via_scale = (rho_libint
                             * inv_s[:, None, None] * inv_s[None, :, None])
    # Then permute L=1 m-ordering (libint py,pz,px → libcint px,py,pz)
    # to align AO axes with PySCF's row/col layout.
    rho_libcint_via_scale = np.einsum("ij,jkG,kl->ilG", P,
                                      rho_libcint_via_scale, P.T,
                                      optimize=True)

    print(f"\n  Shell layout:")
    print(f"  {'ish':>3s}  {'atom':>4s}  {'l':>2s}  {'ao_start':>8s}  {'first_exp':>10s}")
    for s_dict in py["shells"]:
        print(f"  {s_dict['ish']:>3d}  {s_dict['atom']:>4d}  {s_dict['l']:>2d}  "
              f"{s_dict['ao_start']:>8d}  {s_dict['exps'][0]:>10.4f}")

    print(f"\n  ||ρ_libint||_F per G: {[round(float(np.linalg.norm(rho_libint[:,:,k])), 4) for k in range(len(Gv))]}")
    print(f"  ||ρ_libint->libcint||_F per G: {[round(float(np.linalg.norm(rho_libcint_via_scale[:,:,k])), 4) for k in range(len(Gv))]}")
    print(f"  ||ρ_pyscf||_F per G:  {[round(float(np.linalg.norm(rho_py[:,:,k])), 4) for k in range(len(Gv))]}")

    print(f"\n  ||diff||_F per G (vibe-qc converted vs PySCF):")
    print(f"  {'k':>3s}  {'|G|':>6s}  {'||diff||':>12s}  {'||PySCF||':>12s}  {'rel':>10s}")
    for k in range(len(Gv)):
        d = np.linalg.norm(rho_libcint_via_scale[:,:,k] - rho_py[:,:,k])
        n = np.linalg.norm(rho_py[:,:,k])
        Gnorm = np.linalg.norm(Gv[k])
        rel = d / max(n, 1e-30)
        print(f"  {k:>3d}  {Gnorm:>6.3f}  {d:>12.4e}  {n:>12.4e}  {rel:>10.4e}")

    # Detailed inspection of the off-center Li L=1 × H L=0 block:
    # for each G, print the 3x1 vibe-qc vs PySCF complex values to
    # see if it's a sign flip, a constant scale, or a G-dependent
    # phase factor.
    print(f"\n  Off-center block (Li 2p × H 1s) per G:")
    print(f"  shell layout: Li 2p (px,py,pz at AO 2,3,4) × H 1s (AO 5)")
    print(f"  Convention: rows are libint AO ordering AFTER L=1 perm,")
    print(f"  which gives libcint ordering (px=0, py=1, pz=2 within shell)")
    for k in range(len(Gv)):
        G = Gv[k]
        Gn = np.linalg.norm(G)
        ours = rho_libcint_via_scale[2:5, 5:6, k].flatten()   # (3,)
        ref  = rho_py[2:5, 5:6, k].flatten()                  # (3,)
        # complex ratio elementwise
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ours / ref
        print(f"  G={G}, |G|={Gn:.3f}:")
        print(f"    vibe-qc Li2p×H1s = {ours}")
        print(f"    PySCF   Li2p×H1s = {ref}")
        print(f"    ratio (ours/PySCF) = {ratio}")
        print(f"    log10|ratio|       = {np.log10(np.abs(ratio))}")

    # Per-shell-pair diagnosis: which (L_μ, L_ν) blocks differ?
    print(f"\n  Per-shell-pair block diff at G_k=2 (mid-G):")
    print(f"  {'ish_μ':>5s}  {'ish_ν':>5s}  {'L_μ,L_ν':>8s}  {'||diff||':>12s}  {'||PySCF||':>12s}  {'rel':>10s}")
    shells = py["shells"]
    for sm in shells:
        for sn in shells:
            i0, i1 = sm["ao_start"], sm["ao_end"]
            j0, j1 = sn["ao_start"], sn["ao_end"]
            d = np.linalg.norm(rho_libcint_via_scale[i0:i1, j0:j1, 2]
                               - rho_py[i0:i1, j0:j1, 2])
            n = np.linalg.norm(rho_py[i0:i1, j0:j1, 2])
            rel = d / max(n, 1e-30)
            if rel > 1e-3:
                flag = " ← significant"
            else:
                flag = ""
            print(f"  {sm['ish']:>5d}  {sn['ish']:>5d}  "
                  f"({sm['l']},{sn['l']:>1d})    {d:>12.4e}  {n:>12.4e}  "
                  f"{rel:>10.4e}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
