"""Element-wise PySCF↔vibe-qc compcell-AFT comparison — LiH primitive.

LiH variant of ``gdf_aft_pyscf_diff.py`` (which targets H2). The H2
system has L=0 AOs only and L≥1 only on the aux side; LiH primitive
brings a Li 2p (L=1) ORBITAL shell and exposes the convention layer
that the +22 Ha LiH multi-k residue lives in (see
``handovers/HANDOVER_GDF_V0_11_2026_05_29.md`` § "LiH residue").

Compares, on the compcell fused (modrho_aux ∪ chg) basis:

  (1) Bare 2c metric on fused:
      PySCF:   fused_cell.pbc_intor('int2c2e', kpts=[0])
      vibeqc:  compute_2c_eri_lattice(fused, system, lat_opts)

  (2) AFT j2c_p correction (chg-anything LR Coulomb in G-space):
      PySCF:   internal `j2c_p` from _CCGDFBuilder.get_2c2e
      vibeqc:  _compcell_aft_correction(fused, n_aux, system, eta)

  (3) Compensated 2c metric after AFT subtraction + fuse transform.

Per-block ||·||_F (aux-aux / aux-chg / chg-chg) is AO-permutation
invariant, so a block mismatch isolates the buggy layer without
needing the L=1 (py,pz,px)↔(px,py,pz) reorder.

PySCF is invoked out-of-process per CLAUDE.md §10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import vibeqc as vq
from vibeqc.aux_basis import (
    _compcell_aft_correction,
    fuse_transform_matrix,
    make_aux_basis_set,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
)
from vibeqc._vibeqc_core import compute_2c_eri_lattice


ANG2BOHR = 1.0 / 0.529177210903
A_LIH_ANG = 4.084                    # FCC conventional-cell parameter
AUX_NAME_VIBEQC = "def2-svp-jk"
AUX_NAME_PYSCF = "def2-svp-jkfit"    # PySCF's name for the same basis
ETA = 0.25
LAT_CUTOFF_BOHR = 30.0


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.df import gdf_builder

A_CONV = {a_conv}
AUX = "{aux}"
ETA = {eta}

a = A_CONV
# LiH primitive FCC (rhombohedral basis), Angstrom.
lattice = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
]) * a
cell = pbc_gto.M(
    atom=[("Li", (0.0, 0.0, 0.0)),
          ("H",  (0.5*a, 0.5*a, 0.5*a))],
    a=lattice,
    unit="A",
    basis="sto-3g",
    precision=1e-12,
    verbose=0,
)

from pyscf.pbc.df import df as pbc_df
from pyscf.pbc.df import ft_ao as _ft_ao_mod
auxcell = pbc_df.make_modrho_basis(cell, AUX, drop_eta=0.0)
fused_cell, fuse = gdf_builder.fuse_auxcell(auxcell, ETA)

# Probe F̂_fused at 3 specific reciprocal-lattice points so the
# caller can diff against vibe-qc's _aft_fourier_transform_libcint_
# convention element-wise. b = reciprocal vectors; probe b0, b1,
# b0+b1 (guaranteed present in any G-mesh of this lattice).
b_recip = fused_cell.reciprocal_vectors()
G_probe = np.array([b_recip[0], b_recip[1], b_recip[0] + b_recip[1]])
ft_probe = _ft_ao_mod.ft_ao(fused_cell, G_probe).T   # (nfused, 3)

naux = auxcell.nao_nr()
nfused = fused_cell.nao_nr()
nchg = nfused - naux

# (1) BARE 2c metric on fused.
j2c_bare = fused_cell.pbc_intor("int2c2e", hermi=0, kpts=np.zeros((1,3)))[0]
j2c_bare = 0.5 * (j2c_bare + j2c_bare.conj().T)

# (1b) MOLECULAR 2c metric on fused — single cell, NO lattice sum.
# fused_cell.intor (vs pbc_intor) gives the home-cell-only integral.
# Comparing the per-L grid of this against vibe-qc's compute_2c_eri
# isolates whether the cross-L bug is in the basic libint call or in
# the periodic image-sum loop.
j2c_mol = np.asarray(fused_cell.intor("int2c2e"))
j2c_mol = 0.5 * (j2c_mol + j2c_mol.T)

# (1c) MOLECULAR 3c tensor on (ao_basis × fused) — single cell.
# 3c analog of (1b). The 2026-05-25 M1 attempt 1 (pair-FT conversion
# in `_compcell_aft_correction_3c` libint path) had ZERO effect on
# the LiH +11 686 Ha SCF blowup, ruling out that as the convention
# bug. Per the handover, the next data needed is whether
# `compute_3c_eri(ao_basis, fused)` bit-matches PySCF
# `aux_e2(cell, fused_cell, intor='int3c2e')` (analogous to the
# MOLECULAR 2c bit-exact confirmation).
from pyscf.df.incore import aux_e2
# aux_e2 returns shape (n_orb, n_orb, n_aux) for aosym='s1'; transpose
# to vibe-qc's (n_aux, n_orb, n_orb) convention for comparison.
T_mol_py = aux_e2(cell, fused_cell, intor='int3c2e', aosym='s1')
T_mol_py = T_mol_py.transpose(2, 0, 1)  # (n_fused, n_orb, n_orb)

# (2) AFT j2c_p — replicate _CCGDFBuilder.get_2c2e inner loop.
from pyscf import lib
from pyscf.pbc.df import aft, ft_ao
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc.df.gdf_builder import estimate_ke_cutoff_for_eta

precision = cell.precision ** 2
ke = estimate_ke_cutoff_for_eta(auxcell, ETA, precision)
mesh = auxcell.cutoff_to_mesh(ke)
mesh = cell.symmetrize_mesh(mesh)

Gv, Gvbase, kws = fused_cell.get_Gv_weights(mesh)
b = fused_cell.reciprocal_vectors()
gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
ngrids = Gv.shape[0]
kpt = np.zeros(3)
coulG = pbc_tools.get_coulG(cell, kpt, False, None, mesh, Gv) * kws

auxG = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
auxGR = np.asarray(auxG.real, order="C")
auxGI = np.asarray(auxG.imag, order="C")

j2c_p  = (auxGR[naux:] * coulG) @ auxGR.T
j2c_p += (auxGI[naux:] * coulG) @ auxGI.T

# (3) Compensated j2c.
j2c_corrected = j2c_bare.copy()
j2c_corrected[naux:] -= j2c_p
j2c_corrected[:naux, naux:] -= j2c_p[:, :naux].conj().T
j2c_corrected = 0.5 * (j2c_corrected + j2c_corrected.conj().T)
j2c_compensated = fuse(fuse(j2c_corrected), axis=1)

def fro(M): return float(np.linalg.norm(M))
def diag5(M): return [float(x) for x in np.diag(M)[:8]]

shells = []
ao_loc = fused_cell.ao_loc_nr()
for i in range(fused_cell.nbas):
    # bas_ctr_coeff gives the (normalized) contraction coefficients
    # libcint uses internally for this shell.
    ctr = fused_cell.bas_ctr_coeff(i)
    shells.append({{
        "ish": i,
        "atom": int(fused_cell.bas_atom(i)),
        "l": int(fused_cell.bas_angular(i)),
        "n_prims": int(fused_cell.bas_nprim(i)),
        "ao_start": int(ao_loc[i]),
        "ao_end": int(ao_loc[i+1]),
        "exps": [float(x) for x in fused_cell.bas_exp(i)],
        "coefs": [float(x) for x in np.asarray(ctr).ravel()],
    }})

# Normalization quantities for the j2c_p G-sum comparison.
cell_vol = float(cell.vol)
Gnorms = np.linalg.norm(Gv, axis=1)
Gnz = Gnorms[Gnorms > 1e-10]
kws_arr = np.asarray(kws).ravel()

print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "naux": naux, "nchg": nchg, "nfused": nfused,
    "mesh": [int(x) for x in mesh],
    "ngrids": int(ngrids),
    "ke_cutoff": float(ke),
    "ETA": float(ETA),
    "cell_vol": cell_vol,
    "kws_first": float(kws_arr[0]),
    "kws_uniform": bool(np.allclose(kws_arr, kws_arr[0])),
    "n_G_nonzero": int(Gnz.size),
    "G_max": float(Gnorms.max()),
    "G_min_nonzero": float(Gnz.min()),
    "coulG_kws_sum": float((coulG[Gnorms > 1e-10]).sum()),
    "shells_fused": shells,
    # F̂_fused probe at G = b0, b1, b0+b1.
    "G_probe": G_probe.tolist(),
    "ft_probe_real": ft_probe.real.tolist(),
    "ft_probe_imag": ft_probe.imag.tolist(),
    "M_bare_fro": fro(j2c_bare),
    "M_bare_aux_aux_fro": fro(j2c_bare[:naux, :naux]),
    "M_bare_aux_chg_fro": fro(j2c_bare[:naux, naux:]),
    "M_bare_chg_chg_fro": fro(j2c_bare[naux:, naux:]),
    "M_bare_diag5": diag5(j2c_bare),
    "M_bare_chg_chg_diag5": diag5(j2c_bare[naux:, naux:]),
    "j2c_p_fro": fro(j2c_p),
    "j2c_p_chg_aux_fro": fro(j2c_p[:, :naux]),
    "j2c_p_chg_chg_fro": fro(j2c_p[:, naux:]),
    "j2c_p_chg_chg_diag5": diag5(j2c_p[:, naux:]),
    "M_compensated_fro": fro(j2c_compensated),
    "M_compensated_diag5": diag5(j2c_compensated),
    "M_compensated_min_eig": float(np.linalg.eigvalsh(j2c_compensated).min()),
    "M_compensated_max_eig": float(np.linalg.eigvalsh(j2c_compensated).max()),
    # Full bare 2c metric for the per-(L_P,L_Q) ratio grid.
    "j2c_bare_full": j2c_bare.real.tolist(),
    # Molecular (no lattice sum) 2c metric on the same fused basis.
    "j2c_mol_full": j2c_mol.real.tolist(),
    # Molecular (no lattice sum) 3c tensor on (ao_basis × fused).
    # Shape (n_fused, n_orb, n_orb). For LiH primitive + STO-3G this
    # is 94 × 6 × 6 = 3384 doubles — small enough for JSON.
    "T_mol_full": T_mol_py.real.tolist(),
    "n_orb_py": int(cell.nao_nr()),
}}))
'''


def run_pyscf() -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(a_conv=A_LIH_ANG, aux=AUX_NAME_PYSCF, eta=ETA)
    proc = subprocess.run(
        [py, "-c", script], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    print("PySCF subprocess emitted no result marker", file=sys.stderr)
    print("STDOUT:", proc.stdout[:2000], file=sys.stderr)
    print("STDERR:", proc.stderr[:2000], file=sys.stderr)
    sys.exit(2)


def _lih_primitive_system():
    a = A_LIH_ANG * ANG2BOHR
    lattice = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]) * a
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.5 * a, 0.5 * a, 0.5 * a]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def main() -> int:
    print(f"vibeqc {vq.__version__}: LiH primitive compcell-AFT vs PySCF")
    print(f"  Li-H FCC, a_conv={A_LIH_ANG} Å, aux={AUX_NAME_VIBEQC}, η={ETA}")
    print()

    # vibe-qc side
    system = _lih_primitive_system()
    mol = system.unit_cell_molecule()
    ao_basis = vq.BasisSet(mol, "sto-3g")  # orbital basis for the 3c MOL probe
    aux = make_aux_basis_set(mol, aux_name=AUX_NAME_VIBEQC)
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=ETA)
    fused = make_fused_basis(modrho, chg, mol)
    A = fuse_transform_matrix(modrho, chg)
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    n_fused = fused.nbasis

    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = LAT_CUTOFF_BOHR
    lo.nuclear_cutoff_bohr = LAT_CUTOFF_BOHR

    M_bare = np.asarray(compute_2c_eri_lattice(fused, system, lo))
    M_bare = 0.5 * (M_bare + M_bare.T)

    # Compute j2c_p with BOTH conventions and BOTH precisions so the
    # M1 diagnostic can isolate (a) the FT convention bug (Bug 4a:
    # libcint-conv off by √((2L+1)/(4π)) per L vs PySCF ft_ao) from
    # (b) the G-mesh truncation issue (vibe-qc precision=1e-10 gives
    # n_G=724; PySCF uses 4912 at cell.precision=1e-12 ** 2 = 1e-24).
    variants = {}
    for ft_conv in ("libcint", "libint"):
        for prec in (1e-10, 1e-14):
            j2c_p_v = _compcell_aft_correction(
                fused, n_aux, system, eta=ETA, precision=prec,
                ft_convention=ft_conv,
            )
            M_aft = M_bare.copy()
            M_aft[n_aux:, :]      -= j2c_p_v
            M_aft[:n_aux, n_aux:] -= j2c_p_v[:, :n_aux].T
            M_aft = 0.5 * (M_aft + M_aft.T)
            M_comp = A @ M_aft @ A.T
            M_comp = 0.5 * (M_comp + M_comp.T)
            eigs = np.linalg.eigvalsh(M_comp)
            variants[(ft_conv, prec)] = dict(
                j2c_p=j2c_p_v, M_comp=M_comp, eigs=eigs,
            )

    # Keep the original ft_convention="libcint", precision=1e-10 as the
    # primary thing the rest of the script compares (matches the prior
    # output layout — minimises diff churn).
    j2c_p_vqc = variants[("libcint", 1e-10)]["j2c_p"]
    M_compensated = variants[("libcint", 1e-10)]["M_comp"]
    eigs_vqc = variants[("libcint", 1e-10)]["eigs"]

    # PySCF side
    py = run_pyscf()

    # vibe-qc G-mesh normalization quantities, for the j2c_p comparison.
    from vibeqc.aux_basis import _aft_g_mesh
    G_vqc = _aft_g_mesh(system, ETA, precision=1e-10)
    G_vqc_norms = np.linalg.norm(G_vqc, axis=1)
    cell_vol_vqc = float(abs(np.linalg.det(np.asarray(system.lattice))))

    print(f"=== AFT G-mesh normalization ===")
    print(f"  {'quantity':<22s}  {'vibe-qc':>16s}  {'PySCF':>16s}")
    print(f"  {'cell volume':<22s}  {cell_vol_vqc:>16.6f}  "
          f"{py['cell_vol']:>16.6f}")
    print(f"  {'n_G (nonzero)':<22s}  {G_vqc.shape[0]:>16d}  "
          f"{py['n_G_nonzero']:>16d}")
    print(f"  {'G_max':<22s}  {G_vqc_norms.max():>16.4f}  "
          f"{py['G_max']:>16.4f}")
    print(f"  {'G_min nonzero':<22s}  {G_vqc_norms.min():>16.4f}  "
          f"{py['G_min_nonzero']:>16.4f}")
    print(f"  PySCF kws[0]={py['kws_first']:.6e}  "
          f"uniform={py['kws_uniform']}  "
          f"(vibe-qc weight = 4π/G²/V)")
    print(f"  PySCF Σ(coulG·kws) over G≠0 = {py['coulG_kws_sum']:.6e}")
    # vibe-qc equivalent: Σ 4π/G²/V
    coul_sum_vqc = float(((4.0 * np.pi)
                          / (G_vqc_norms ** 2) / cell_vol_vqc).sum())
    print(f"  vibe-qc       Σ(4π/G²/V) over G≠0 = {coul_sum_vqc:.6e}")
    print()

    # F̂_fused element-wise probe vs PySCF's ft_ao at G = b0, b1, b0+b1.
    from vibeqc.aux_basis import (
        _aft_fourier_transform_libcint_convention,
        rsgdf_aux_fourier_transform,
    )
    G_probe = np.array(py["G_probe"])
    ft_libcint = _aft_fourier_transform_libcint_convention(fused, G_probe)
    ft_libint = rsgdf_aux_fourier_transform(fused, G_probe)
    ft_py = (np.array(py["ft_probe_real"])
             + 1j * np.array(py["ft_probe_imag"]))  # (n_fused, 3)
    print(f"=== F̂_fused element-wise vs PySCF ft_ao (G = b0, b1, b0+b1) ===")
    print(f"  per-G ||diff||/||PySCF|| for both vibe-qc FT routines:")
    print(f"  {'G_idx':>5s}  {'|G|':>8s}  "
          f"{'libcint-conv rel':>18s}  {'libint-conv rel':>18s}")
    for g in range(3):
        Gn = float(np.linalg.norm(G_probe[g]))
        py_col = ft_py[:, g]
        npy = float(np.linalg.norm(py_col))
        rel_c = float(np.linalg.norm(ft_libcint[:, g] - py_col)) / max(npy, 1e-30)
        rel_i = float(np.linalg.norm(ft_libint[:, g] - py_col)) / max(npy, 1e-30)
        print(f"  {g:>5d}  {Gn:>8.4f}  {rel_c:>18.4e}  {rel_i:>18.4e}")
    # Per-shell-L COMPLEX ratio at G_idx=0 — for both routines.
    # NOTE: vibe-qc's L=1 AO order is libint (py,pz,px) = (m=-1,0,+1);
    # PySCF/libcint's L=1 order is (px,py,pz). To compare like-for-like
    # we reorder vibe-qc's L=1 triple via [2,0,1] → (px,py,pz). L≥2
    # both use m=-l..+l and agree.
    def _perm_l1(col, L):
        if L != 1:
            return col
        return col[[2, 0, 1]]

    print(f"  per-L complex ratio (vqc/PySCF) of F̂ at G_idx=0 "
          f"(L=1 reordered to libcint px,py,pz):")
    print(f"  {'AO':>9s}  {'L':>2s}  {'libcint-conv':>26s}  "
          f"{'libint-conv':>26s}")
    bf = 0
    for sh in fused.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        p = ft_py[bf:bf + nc, 0]
        v_c = _perm_l1(ft_libcint[bf:bf + nc, 0], L)
        v_i = _perm_l1(ft_libint[bf:bf + nc, 0], L)
        mask = np.abs(p) > 1e-10
        if mask.any():
            rc = np.mean(v_c[mask] / p[mask])
            ri = np.mean(v_i[mask] / p[mask])
            print(f"  {bf:>3d}-{bf+nc-1:<3d}  L={L}  "
                  f"{rc.real:>+11.5f}{rc.imag:>+11.5f}j  "
                  f"{ri.real:>+11.5f}{ri.imag:>+11.5f}j")
        bf += nc
    print()

    # Fused-basis construction parity: shell-by-shell exponent +
    # coefficient comparison vs PySCF's make_modrho_basis +
    # fuse_auxcell. The per-L coef fix (commit 043efdd) rescales
    # vibe-qc's coefficients by √((2L+1)/(4π)) per L, so compare
    # coefficients after dividing that out.
    print(f"=== Fused-basis construction parity (shell-by-shell) ===")
    vqc_shells = list(fused.shells())
    py_shells = py["shells_fused"]
    sqrt4pi = float(np.sqrt(4.0 * np.pi))
    print(f"  {'ish':>3s}  {'L':>2s}  {'np':>3s}  {'exp match':>10s}  "
          f"{'coef ratio (vqc/PySCF, perL-corrected)':>40s}")
    n_exp_mismatch = 0
    n_coef_mismatch = 0
    for i, (vs, ps) in enumerate(zip(vqc_shells, py_shells)):
        L = int(vs.l)
        v_exps = np.asarray(vs.exponents, dtype=float)
        p_exps = np.asarray(ps["exps"], dtype=float)
        v_coefs = np.asarray(vs.coefficients, dtype=float)
        p_coefs = np.asarray(ps["coefs"], dtype=float)
        exp_ok = (v_exps.shape == p_exps.shape
                  and np.allclose(np.sort(v_exps), np.sort(p_exps),
                                  rtol=1e-6))
        # vibe-qc coef = libcint coef × √((2L+1)/4π)  (per-L coef fix).
        perL = np.sqrt(2 * L + 1) / sqrt4pi
        if v_coefs.shape == p_coefs.shape and p_coefs.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (v_coefs / perL) / p_coefs
            ratio_str = "  ".join(f"{x:+.5f}" for x in ratio[:4])
            coef_ok = np.allclose(ratio[np.isfinite(ratio)], 1.0,
                                  rtol=1e-4)
        else:
            ratio_str = f"SHAPE {v_coefs.shape} vs {p_coefs.shape}"
            coef_ok = False
        if not exp_ok:
            n_exp_mismatch += 1
        if not coef_ok:
            n_coef_mismatch += 1
        flag = "" if (exp_ok and coef_ok) else "  ←"
        print(f"  {i:>3d}  {L:>2d}  {v_exps.size:>3d}  "
              f"{str(exp_ok):>10s}  {ratio_str:>40s}{flag}")
    print(f"  → {n_exp_mismatch} exp mismatches, "
          f"{n_coef_mismatch} coef mismatches out of {len(vqc_shells)} "
          f"shells")
    print()

    # Per-(region, L) block ratio grid of the bare 2c metric. Split
    # the fused basis into aux (modrho) vs chg regions and by L, so a
    # region-specific or L-specific mismatch shows cleanly. Frobenius
    # norm per block is AO-permutation invariant.
    j2c_py = np.array(py["j2c_bare_full"])
    # Per-AO (region, L) label from vibe-qc's shell walk. Keep as a
    # plain Python list — np.array collapses a list of 2-tuples into
    # a 2D array.
    ao_tag = []
    bf = 0
    for sh in fused.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        region = "aux" if bf < n_aux else "chg"
        ao_tag.extend([(region, L)] * nc)
        bf += nc
    tags = []
    for region in ("aux", "chg"):
        for L in sorted({t[1] for t in ao_tag if t[0] == region}):
            tags.append((region, L))
    masks = {t: np.array([x == t for x in ao_tag], dtype=bool)
             for t in tags}

    def _print_grid(title, M_v, M_p):
        print(f"=== {title} ===")
        print(f"  {'P \\ Q':>10s}  "
              + "".join(f"{str(t):>13s}" for t in tags))
        for tp in tags:
            row = f"  {str(tp):>10s}  "
            for tq in tags:
                blk_v = M_v[np.ix_(masks[tp], masks[tq])]
                blk_p = M_p[np.ix_(masks[tp], masks[tq])]
                npy = float(np.linalg.norm(blk_p))
                r = (float(np.linalg.norm(blk_v)) / npy
                     if npy > 1e-12 else float("nan"))
                row += f"{r:>13.5f}"
            print(row)
        print()

    print(f"  block tags: {tags}")
    _print_grid("LATTICE M_fused per-(region,L) block ratio (vqc/PySCF)",
                M_bare, j2c_py)

    # MOLECULAR (single-cell) 2c metric — isolates the cross-L bug to
    # the basic libint call (if the molecular grid is also weird) vs
    # the periodic image-sum loop (if molecular is clean ~1.0).
    from vibeqc._vibeqc_core import compute_2c_eri
    M_mol_v = np.asarray(compute_2c_eri(fused))
    M_mol_v = 0.5 * (M_mol_v + M_mol_v.T)
    j2c_mol_py = np.array(py["j2c_mol_full"])
    _print_grid("MOLECULAR 2c metric per-(region,L) block ratio (vqc/PySCF)",
                M_mol_v, j2c_mol_py)

    # MOLECULAR 3c tensor — analog of the 2c MOLECULAR check, isolates
    # whether `compute_3c_eri(ao_basis, fused)` matches PySCF
    # `aux_e2(cell, fused_cell, intor='int3c2e')` on the single-cell
    # integral. If MOLECULAR 3c is bit-exact, the +11 686 Ha LiH SCF
    # blowup must come from j3c_p (AFT correction). If MOLECULAR 3c is
    # NOT bit-exact, the bug is in `compute_3c_eri` itself.
    from vibeqc._vibeqc_core import compute_3c_eri
    T_mol_v = np.asarray(compute_3c_eri(ao_basis, fused))
    # shape (n_fused, n_orb, n_orb)
    T_mol_p = np.array(py["T_mol_full"])
    n_orb_py = int(py["n_orb_py"])
    if T_mol_v.shape != T_mol_p.shape:
        print(f"  ⚠️  MOLECULAR 3c shape mismatch: "
              f"vqc {T_mol_v.shape} vs PySCF {T_mol_p.shape}")

    # AO-pair tag walk for ao_basis (just L per AO; "region" doesn't
    # apply to the orbital cell).
    ao_L_tag = []
    for sh in ao_basis.shells():
        L_ao = int(sh.l)
        ao_L_tag.extend([L_ao] * (2 * L_ao + 1))
    ao_L_set = sorted(set(ao_L_tag))
    ao_L_masks = {L: np.array([t == L for t in ao_L_tag], dtype=bool)
                  for L in ao_L_set}

    # Aux/chg AO tag walk for fused — re-use the existing `masks` /
    # `tags` from the 2c block grid (already constructed above).

    print(f"=== MOLECULAR 3c tensor per-(fused-region/L × ao-L × ao-L) "
          f"||·||_F ratio (vqc/PySCF) ===")
    print(f"  shape: vqc {T_mol_v.shape} vs PySCF {T_mol_p.shape}")
    if T_mol_v.shape == T_mol_p.shape:
        # Overall ratio:
        overall = (float(np.linalg.norm(T_mol_v))
                   / max(float(np.linalg.norm(T_mol_p)), 1e-30))
        print(f"  overall ||T_v||_F / ||T_py||_F = {overall:>9.5f}")
        # Per-(fused_tag) × (ao_L_μ, ao_L_ν) ratio table.
        for tp in tags:
            for L_mu in ao_L_set:
                for L_nu in ao_L_set:
                    if L_nu < L_mu:
                        continue  # symmetric in (μ, ν); skip lower triangle
                    blk_v = T_mol_v[np.ix_(masks[tp],
                                           ao_L_masks[L_mu],
                                           ao_L_masks[L_nu])]
                    blk_p = T_mol_p[np.ix_(masks[tp],
                                           ao_L_masks[L_mu],
                                           ao_L_masks[L_nu])]
                    npy = float(np.linalg.norm(blk_p))
                    if npy < 1e-12:
                        continue
                    r = float(np.linalg.norm(blk_v)) / npy
                    flag = ""
                    if abs(r - 1.0) > 1e-3:
                        flag = "  ← MISMATCH"
                    print(f"  fused {str(tp):>10s}  ao L={L_mu},L={L_nu}: "
                          f"||v||_F={np.linalg.norm(blk_v):>10.4f}  "
                          f"||py||_F={npy:>10.4f}  ratio={r:>8.5f}{flag}")
    print()

    print(f"=== Sizes ===")
    print(f"  vibe-qc: n_aux={n_aux}, n_chg={n_chg}, n_fused={n_fused}")
    print(f"  PySCF:   n_aux={py['naux']}, n_chg={py['nchg']}, "
          f"n_fused={py['nfused']}")
    print(f"  PySCF mesh: {py['mesh']} ({py['ngrids']} G-vectors), "
          f"ke_cutoff={py['ke_cutoff']:.2f}")
    print()

    print(f"  fused shell layout (PySCF):")
    print(f"  {'ish':>3s}  {'atom':>4s}  {'l':>2s}  {'nprim':>5s}  "
          f"{'ao':>8s}  {'first_exp':>10s}")
    for s in py["shells_fused"]:
        print(f"  {s['ish']:>3d}  {s['atom']:>4d}  {s['l']:>2d}  "
              f"{s['n_prims']:>5d}  {s['ao_start']:>3d}-{s['ao_end']:<3d}  "
              f"{s['exps'][0]:>10.4f}")
    print()

    print(f"=== Bare M_fused per-block ||·||_F ===")
    print(f"  {'block':<12s}  {'vibe-qc':>14s}  {'PySCF':>14s}  {'ratio':>10s}")
    pairs = [
        ("all",     float(np.linalg.norm(M_bare)),                  py["M_bare_fro"]),
        ("aux-aux", float(np.linalg.norm(M_bare[:n_aux, :n_aux])),   py["M_bare_aux_aux_fro"]),
        ("aux-chg", float(np.linalg.norm(M_bare[:n_aux, n_aux:])),   py["M_bare_aux_chg_fro"]),
        ("chg-chg", float(np.linalg.norm(M_bare[n_aux:, n_aux:])),   py["M_bare_chg_chg_fro"]),
    ]
    for name, v, p in pairs:
        r = v / p if p else float("inf")
        flag = "  ← MISMATCH" if abs(r - 1.0) > 1e-3 else ""
        print(f"  {name:<12s}  {v:>14.6f}  {p:>14.6f}  {r:>10.4f}{flag}")
    print()

    print(f"=== j2c_p (AFT correction) per-block ||·||_F ===")
    print(f"  {'block':<12s}  {'vibe-qc':>14s}  {'PySCF':>14s}  {'ratio':>10s}")
    pairs = [
        ("all",     float(np.linalg.norm(j2c_p_vqc)),               py["j2c_p_fro"]),
        ("chg-aux", float(np.linalg.norm(j2c_p_vqc[:, :n_aux])),     py["j2c_p_chg_aux_fro"]),
        ("chg-chg", float(np.linalg.norm(j2c_p_vqc[:, n_aux:])),     py["j2c_p_chg_chg_fro"]),
    ]
    for name, v, p in pairs:
        r = v / p if p else float("inf")
        flag = "  ← MISMATCH" if abs(r - 1.0) > 1e-3 else ""
        print(f"  {name:<12s}  {v:>14.6f}  {p:>14.6f}  {r:>10.4f}{flag}")
    print()

    print(f"=== Compensated M (after AFT + fuse) — convention × precision sweep ===")
    print(f"  PySCF target: ||M||_F = {py['M_compensated_fro']:.6f}, "
          f"min_eig={py['M_compensated_min_eig']:+.4e}, "
          f"max_eig={py['M_compensated_max_eig']:.4e}")
    print(f"  {'ft_conv':<10s}  {'precision':>10s}  {'||M||_F':>12s}  "
          f"{'min_eig':>14s}  {'max_eig':>12s}  {'ratio_F':>9s}  {'pd?':>4s}")
    for (ft_conv, prec), v in variants.items():
        Mn = float(np.linalg.norm(v["M_comp"]))
        emin = float(v["eigs"].min())
        emax = float(v["eigs"].max())
        r = Mn / py["M_compensated_fro"] if py["M_compensated_fro"] else float("nan")
        pd = "YES" if emin > 0 else "NO"
        print(f"  {ft_conv:<10s}  {prec:>10.0e}  {Mn:>12.6f}  "
              f"{emin:>+14.4e}  {emax:>+12.4e}  {r:>9.4f}  {pd:>4s}")
    print()

    # Per-(region,L) block ratio of the COMPENSATED metric vs PySCF.
    # The handover (§Bug 4b reframing) says this — not the bare metric —
    # is the cutoff-independent / convergent target. A block stuck away
    # from 1.0 here is the actual residue source.
    py_M_comp = None
    # PySCF doesn't currently emit the full compensated metric in the
    # subprocess result — only the Frobenius norm and eigenvalues. If
    # added later this block fills in. For now we limit to a self-
    # consistency check: how the compensated metric changes across
    # convention × precision (does either choice make it positive
    # definite and match PySCF's norm + eigenvalues?).
    if py_M_comp is None:
        pass  # block ratios deferred until PySCF side emits j2c_compensated
    print()

    print(f"=== Diagonal samples (first 8 AOs) ===")
    print(f"  Bare M_fused[i,i]:")
    print(f"    vibe-qc: {[round(x, 4) for x in np.diag(M_bare)[:8]]}")
    print(f"    PySCF:   {[round(x, 4) for x in py['M_bare_diag5']]}")
    print(f"  Bare M_fused[chg,chg]:")
    print(f"    vibe-qc: {[round(x, 4) for x in np.diag(M_bare[n_aux:, n_aux:])[:8]]}")
    print(f"    PySCF:   {[round(x, 4) for x in py['M_bare_chg_chg_diag5']]}")
    print(f"  j2c_p[chg,chg]:")
    print(f"    vibe-qc: {[round(x, 4) for x in np.diag(j2c_p_vqc[:, n_aux:])[:8]]}")
    print(f"    PySCF:   {[round(x, 4) for x in py['j2c_p_chg_chg_diag5']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
