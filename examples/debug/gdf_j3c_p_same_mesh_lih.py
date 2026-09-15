"""Compare vibe-qc j3c_p to PySCF j3c_p on the SAME G-mesh — LiH primitive.

Built during the 2026-05-26 M1 investigation, after the MOLECULAR 3c
"bit-exact" diagnostic in `gdf_aft_pyscf_diff_lih.py` was found to use
Frobenius norms per block — which are permutation-invariant and miss
element-wise sign/ordering differences in L≥1 components.

This probe:
* Pulls PySCF's exact G-mesh (4913 G-vectors for LiH at η=0.25,
  precision=1e-12²) AND the resulting j3c_p out of a subprocess.
* Computes vibe-qc's j3c_p ON THE SAME MESH (and with PySCF's coulG·kws)
  under several variants: as-is, with rho_pair libint→libcint rescale,
  with L=1 m-permutation, etc.
* Diff's element-wise.

2026-05-26 finding:
* Even on PySCF's EXACT mesh, vibe-qc's j3c_p is **63% different** from
  PySCF — the G-mesh truncation is NOT the bug. CLAUDE.md §7-style
  unmasked, this is a real convention or sign error somewhere in the
  vibe-qc → PySCF translation that the existing diagnostics (which all
  use Frobenius-norm-per-block) systematically miss.
* |F_chg| magnitudes per chg AO match PySCF to 4 decimal places, but
  the complex values differ — phase or sign issue.
* L=1 m-permutation (libint py,pz,px → libcint px,py,pz) does NOT close
  the gap.
* `Gpq` (the pair FT) differs by ~0.7 in Frobenius norm. The earlier
  pair-FT-bit-exact claim in `gdf_pair_ft_pyscf_diff.py` was on a
  specific calibration test; on PySCF's full production mesh the FT
  values aren't bit-exact.

Next investigation steps (left for the M1 chat after this session):
* Sign of the complex `F_chg` per-G — is it a `(+i)^L` vs `(-i)^L` issue
  on the chg side? (`cartesian_gaussian_product_ft` had the analogous
  fix in 2026-05-18 for the pair-FT; chg single-AO FT may need the
  same audit.)
* Phase of `rho_pair` per-G for L=1 ao_basis components.
* Whether vibe-qc's solid-harmonic to spherical-harmonic rotation
  matches libcint's (subtle for L≥2 — Condon-Shortley phase, real-
  vs complex-spherical-harmonic convention).
"""
import json
import os
import subprocess
import sys

import numpy as np
import vibeqc as vq
from vibeqc.aux_basis import (
    _per_ao_libcint_to_libint_scale,
    _per_ao_pair_libint_to_libcint_scale,
    make_aux_basis_set,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
    rsgdf_aux_fourier_transform,
)
from vibeqc._aopair_ft import ao_pair_fourier_transform


ANG2BOHR = 1.0 / 0.529177210903
A_LIH_ANG = 4.084
ETA = 0.25


# PySCF script: emits its exact G-mesh + j3c_p + n_chg + n_orb so vibe-qc
# can replay on the same grid.
PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.df import gdf_builder, ft_ao
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc.df.gdf_builder import estimate_ke_cutoff_for_eta
from pyscf import lib

A_CONV = {a_conv}
ETA = {eta}
a = A_CONV
lattice = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
]) * a
cell = pbc_gto.M(
    atom=[("Li", (0.0, 0.0, 0.0)), ("H",  (0.5*a, 0.5*a, 0.5*a))],
    a=lattice, unit="A", basis="sto-3g",
    precision=1e-12, verbose=0,
)

from pyscf.pbc.df import df as pbc_df
auxcell = pbc_df.make_modrho_basis(cell, "def2-svp-jkfit", drop_eta=0.0)
fused_cell, fuse = gdf_builder.fuse_auxcell(auxcell, ETA)
naux = auxcell.nao_nr()
nfused = fused_cell.nao_nr()
nchg = nfused - naux
n_orb = cell.nao_nr()

precision = cell.precision ** 2
ke = estimate_ke_cutoff_for_eta(auxcell, ETA, precision)
mesh = auxcell.cutoff_to_mesh(ke)
mesh = cell.symmetrize_mesh(mesh)
Gv, Gvbase, kws = fused_cell.get_Gv_weights(mesh)
b = fused_cell.reciprocal_vectors()
gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
kpt = np.zeros(3)
coulG_kws = pbc_tools.get_coulG(cell, kpt, False, None, mesh, Gv) * kws

# Gchg & Gpq on the same mesh
shls_slice = (auxcell.nbas, fused_cell.nbas)
auxG = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T  # (n_chg, n_G)
Gchg_unweighted = lib.transpose(auxG)  # (n_G, n_chg), NO coulG yet
auxG_weighted = auxG * coulG_kws
Gchg = lib.transpose(auxG_weighted)  # (n_G, n_chg) with coulG·kws

Gpq = ft_ao.ft_aopair(cell, Gv, shls_slice=None, aosym='s1', b=b, gxyz=gxyz, Gvbase=Gvbase, kpti_kptj=np.zeros((2,3)), q=kpt)
# Gpq shape: PySCF returns (n_G, n_orb_pair) or (n_G, n_orb, n_orb).
Gpq = np.asarray(Gpq).reshape(Gv.shape[0], n_orb, n_orb)

j3c_p_py = np.einsum('Gc,Gmn->cmn', Gchg.conj(), Gpq).real
# Or equivalently the dot-product form from add_ft_j3c:
# GchgR.T @ GpqR + GchgI.T @ GpqI (gives same answer)

print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "Gv": Gv.tolist(),  # (n_G, 3) — vibe-qc will replay on this
    "coulG_kws": coulG_kws.tolist(),  # (n_G,)
    "j3c_p_py": j3c_p_py.tolist(),
    # F_chg unweighted for diagnostic comparison (NOT multiplied by coulG·kws)
    "F_chg_pyscf_unweighted": Gchg_unweighted.real.tolist(),  # (n_G, n_chg)
    "F_chg_pyscf_unweighted_imag": Gchg_unweighted.imag.tolist(),
    # PySCF Gpq for diagnostic comparison
    "Gpq_real": Gpq.real.tolist(),
    "Gpq_imag": Gpq.imag.tolist(),
    "n_chg": nchg, "n_orb": n_orb,
    "mesh": [int(m) for m in mesh],
}}))
'''


def lih_setup():
    a = A_LIH_ANG * ANG2BOHR
    lattice = np.array([
        [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
    ]) * a
    atoms = [vq.Atom(3, [0,0,0]), vq.Atom(1, [0.5*a, 0.5*a, 0.5*a])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    mol = system.unit_cell_molecule()
    ao_basis = vq.BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=ETA)
    fused = make_fused_basis(modrho, chg, mol)
    return system, ao_basis, modrho, chg, fused


def run_pyscf():
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(a_conv=A_LIH_ANG, eta=ETA)
    p = subprocess.run([py, "-c", script], capture_output=True, text=True)
    if p.returncode != 0:
        print("PySCF subprocess failed:", file=sys.stderr)
        print(p.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    for line in p.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    sys.exit(2)


def main():
    system, ao_basis, modrho, chg, fused = lih_setup()
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    n_orb = ao_basis.nbasis

    print("Running PySCF reference...")
    py = run_pyscf()
    Gv_py = np.array(py["Gv"])
    coulG_py = np.array(py["coulG_kws"])  # already weighted
    j3c_p_py = np.array(py["j3c_p_py"])
    F_chg_py = np.array(py["F_chg_pyscf_unweighted"]) + 1j * np.array(py["F_chg_pyscf_unweighted_imag"])
    Gpq_py = np.array(py["Gpq_real"]) + 1j * np.array(py["Gpq_imag"])
    n_G = Gv_py.shape[0]
    print(f"  mesh = {py['mesh']}, n_G = {n_G}")
    print(f"  PySCF j3c_p ||·||_F = {np.linalg.norm(j3c_p_py):.6e}")
    print()

    # ─── Step 1: vibe-qc on PySCF's exact G-mesh, libint mode (no rescales) ───
    # No rescales — same as production "libint" path but on PySCF G-mesh.
    F_v = rsgdf_aux_fourier_transform(fused, Gv_py)  # (n_fused, n_G)
    F_chg_v = F_v[n_aux:]
    rho_pair_v = ao_pair_fourier_transform(ao_basis, Gv_py)

    # Apply L=1 m-permutation (libint py,pz,px → libcint px,py,pz)
    # to F_chg's L=1 chg AOs AND rho_pair's L=1 ao_basis AOs.
    # vibe-qc internal: L=1 = (py, pz, px) = m order (-1, 0, +1) in libint solid-harm
    # PySCF/libcint:    L=1 = (px, py, pz) = m order (+1, -1, 0) in libcint
    # Permutation libint→libcint per axis: [2, 0, 1]  (px=libint[2], py=libint[0], pz=libint[1])
    def make_perm_l1(basis):
        """Build a full-AO permutation that reorders L=1 triples [2,0,1]."""
        n = basis.nbasis
        perm = list(range(n))
        bf = 0
        for sh in basis.shells():
            L = int(sh.l)
            nc = 2 * L + 1
            if L == 1:
                # Replace [bf, bf+1, bf+2] with [bf+2, bf, bf+1]
                perm[bf], perm[bf+1], perm[bf+2] = bf+2, bf, bf+1
            bf += nc
        return np.array(perm)

    # Permutation for the chg block of fused (chg axis of j3c_p)
    perm_chg_full = make_perm_l1(fused)[n_aux:] - n_aux  # local to chg block
    perm_ao = make_perm_l1(ao_basis)
    print(f"  chg L=1 m-perm exists: {(perm_chg_full != np.arange(len(perm_chg_full))).any()}")
    print(f"  ao L=1 m-perm exists:  {(perm_ao != np.arange(len(perm_ao))).any()}")

    # Apply permutation to F_chg (chg axis) and rho_pair (both AO axes).
    F_chg_v_perm = F_chg_v[perm_chg_full]
    rho_pair_v_perm = rho_pair_v[perm_ao][:, perm_ao]
    print(f"  After permutation:")
    # j3c_p with PySCF coulG_kws (weighted Coulomb), using PERMUTED arrays
    F_chg_w = np.conj(F_chg_v_perm) * coulG_py[None, :]
    rho_flat = rho_pair_v_perm.reshape(n_orb * n_orb, n_G).T
    j3c_p_vqc_pymesh = (F_chg_w @ rho_flat).reshape(n_chg, n_orb, n_orb).real

    rel = np.linalg.norm(j3c_p_vqc_pymesh - j3c_p_py) / np.linalg.norm(j3c_p_py)
    print(f"=== vibe-qc j3c_p on PySCF's mesh + m-permutation (libint→libcint AO ordering) ===")
    print(f"  ||vqc_perm||_F = {np.linalg.norm(j3c_p_vqc_pymesh):.6e}")
    print(f"  ||PySCF||_F     = {np.linalg.norm(j3c_p_py):.6e}")
    print(f"  ||vqc - pyscf||_F / ||pyscf||_F = {rel:.4e}")
    print()

    # ─── Step 2: same with rho_pair libint→libcint rescale + permutation ───
    scales_full = _per_ao_pair_libint_to_libcint_scale(ao_basis)
    inv = 1.0 / scales_full
    rho_pair_c = rho_pair_v_perm * inv[:, None, None] * inv[None, :, None]
    rho_flat_c = rho_pair_c.reshape(n_orb * n_orb, n_G).T
    j3c_p_vqc_pymesh_C = (F_chg_w @ rho_flat_c).reshape(n_chg, n_orb, n_orb).real
    rel = np.linalg.norm(j3c_p_vqc_pymesh_C - j3c_p_py) / np.linalg.norm(j3c_p_py)
    print(f"=== with perm + rho_pair libint→libcint rescale ===")
    print(f"  ||vqc||_F = {np.linalg.norm(j3c_p_vqc_pymesh_C):.6e}")
    print(f"  ||vqc - pyscf||_F / ||pyscf||_F = {rel:.4e}")
    print()

    # ─── Step 3: F_chg parity on PySCF mesh ───
    # F_chg_pyscf vs vibe-qc F_chg on same mesh:
    F_chg_v_complex = F_chg_v.T  # (n_G, n_chg)
    rel_Fchg = np.linalg.norm(F_chg_v_complex - F_chg_py) / np.linalg.norm(F_chg_py)
    print(f"=== F_chg parity on PySCF mesh (vibe-qc rsgdf vs PySCF ft_ao chg block) ===")
    print(f"  ||vqc - pyscf||_F / ||pyscf||_F = {rel_Fchg:.4e}")
    if rel_Fchg > 1e-6:
        # Per-shell diagnostic
        print(f"  ← F_chg DIFFERS bit-significantly. Per-shell shape check:")
        print(f"    F_chg_v shape: {F_chg_v_complex.shape}")
        print(f"    F_chg_py shape: {F_chg_py.shape}")
        # Per-chg-AO ratio at G=Gv_py[0] (first non-zero G)
        for chg_i in range(min(5, n_chg)):
            v = F_chg_v_complex[1, chg_i]  # G index 1
            p = F_chg_py[1, chg_i]
            r = abs(v) / abs(p) if abs(p) > 1e-30 else float("nan")
            print(f"    chg AO {chg_i}: |vqc|={abs(v):.4e}, |py|={abs(p):.4e}, ratio={r:.4f}")
    print()

    # ─── Step 4: Gpq parity on PySCF mesh ───
    rho_pair_v_pq = rho_pair_v.transpose(2, 0, 1)  # (n_G, n_orb, n_orb)
    rel_Gpq = np.linalg.norm(rho_pair_v_pq - Gpq_py) / np.linalg.norm(Gpq_py)
    print(f"=== Gpq parity on PySCF mesh (vibe-qc ao_pair_FT vs PySCF ft_aopair) ===")
    print(f"  ||vqc - pyscf||_F / ||pyscf||_F = {rel_Gpq:.4e}")
    print()

    # ─── Step 5: with libint→libcint rescale on rho_pair ───
    rho_pair_c_pq = rho_pair_c.transpose(2, 0, 1)
    rel_Gpq_c = np.linalg.norm(rho_pair_c_pq - Gpq_py) / np.linalg.norm(Gpq_py)
    print(f"=== Gpq parity (with rho_pair rescale) ===")
    print(f"  ||vqc - pyscf||_F / ||pyscf||_F = {rel_Gpq_c:.4e}")
    print()

    # ─── Step 6: per-G complex F_chg dump for selected chg AOs (2026-05-26 add) ───
    # Goal: find the phase / sign pattern. Magnitudes match per AO; complex
    # values differ. Look for: constant phase (e^{iθ}·1) per AO; per-L i^L sign;
    # complex conjugation across all G.
    #
    # chg basis layout on Li (at origin) + H (at 0.5,0.5,0.5*a):
    #   Li chg shells: L=0 (AO 0), L=1 (1..3), L=2 (4..8), L=3 (9..15)
    #   H  chg shells: L=0 (AO 16), L=1 (17..19), L=2 (20..24)
    # Li at origin: e^{-iG·R}=1, so F_chg[Li L=0] is purely real.
    # H  at R_H:    e^{-iG·R_H} ≠ 1, so F_chg[H L=0] is complex.

    def fmt_c(z):
        return f"({z.real:+.4e}{z.imag:+.4e}j)"

    print("=== per-G complex F_chg — Li chg L=0 (AO 0, at origin → must be REAL) ===")
    print(f"{'G_idx':>5}  {'G_vec':>30}  {'F_vqc':>28}  {'F_pyscf':>28}  {'vqc/py':>20}")
    for g in [1, 2, 3, 5, 10, 50, 100, 500, 1000]:
        Gv = Gv_py[g]
        v = F_chg_v_complex[g, 0]
        p = F_chg_py[g, 0]
        if abs(p) > 1e-30:
            r = v / p
        else:
            r = float("nan")
        Gstr = f"({Gv[0]:+.3f},{Gv[1]:+.3f},{Gv[2]:+.3f})"
        print(f"{g:>5}  {Gstr:>30}  {fmt_c(v):>28}  {fmt_c(p):>28}  {fmt_c(r):>20}")
    print()

    print("=== per-G complex F_chg — H chg L=0 (AO 16, at R_H) ===")
    print(f"{'G_idx':>5}  {'G_vec':>30}  {'F_vqc':>28}  {'F_pyscf':>28}  {'vqc/py':>20}")
    for g in [1, 2, 3, 5, 10, 50, 100, 500, 1000]:
        Gv = Gv_py[g]
        v = F_chg_v_complex[g, 16]
        p = F_chg_py[g, 16]
        if abs(p) > 1e-30:
            r = v / p
        else:
            r = float("nan")
        Gstr = f"({Gv[0]:+.3f},{Gv[1]:+.3f},{Gv[2]:+.3f})"
        print(f"{g:>5}  {Gstr:>30}  {fmt_c(v):>28}  {fmt_c(p):>28}  {fmt_c(r):>20}")
    print()

    print("=== per-G complex F_chg — Li chg L=1 (AOs 1,2,3; libint=py,pz,px order) ===")
    print(f"{'G_idx':>5}  {'AO':>3}  {'G_vec':>30}  {'F_vqc':>28}  {'F_pyscf':>28}")
    for g in [1, 2, 5, 10]:
        Gv = Gv_py[g]
        Gstr = f"({Gv[0]:+.3f},{Gv[1]:+.3f},{Gv[2]:+.3f})"
        for chg_i in [1, 2, 3]:
            v = F_chg_v_complex[g, chg_i]
            p = F_chg_py[g, chg_i]
            print(f"{g:>5}  {chg_i:>3}  {Gstr:>30}  {fmt_c(v):>28}  {fmt_c(p):>28}")
    print()

    # ─── Step 7: scan across all G — what's the per-G ratio statistic? ───
    print("=== ratio F_chg_vqc / F_chg_py across all G for selected chg AOs ===")
    print("(if constant per AO → simple scale; if e^{iθ}·1 with θ depending on G → phase bug)")
    for chg_i in [0, 1, 16, 17]:
        ratios = []
        for g in range(1, n_G):
            p = F_chg_py[g, chg_i]
            v = F_chg_v_complex[g, chg_i]
            if abs(p) > 1e-12:
                ratios.append(v / p)
        ratios = np.array(ratios)
        # If ratio is constant, std is tiny. If ratio is conjugation, real parts agree but imag part flips.
        print(f"  chg AO {chg_i}: mean ratio = {ratios.mean():+.4f}{ratios.mean().imag:+.4f}j  "
              f"std = {ratios.std():.4f}  "
              f"|mean ratio| = {abs(ratios.mean()):.4f}  "
              f"angle(mean) = {np.angle(ratios.mean()):+.4f} rad")
        # Sample ratios at first 5 G to look for the per-G phase
        sample = ratios[:5]
        print(f"    sample first 5 G ratios: {[fmt_c(r) for r in sample]}")
    print()

    # ─── Step 7b: VERIFY post-perm F_chg parity (was the bug really just L=1 m-order?) ───
    F_chg_v_perm_complex = F_chg_v_perm.T  # (n_G, n_chg)
    rel_Fchg_perm = np.linalg.norm(F_chg_v_perm_complex - F_chg_py) / np.linalg.norm(F_chg_py)
    print(f"=== POST-PERM F_chg parity (after L=1 m-perm libint→libcint on chg axis) ===")
    print(f"  pre-perm  ‖v − py‖/‖py‖ = {rel_Fchg:.4e}")
    print(f"  post-perm ‖v − py‖/‖py‖ = {rel_Fchg_perm:.4e}")
    if rel_Fchg_perm < 1e-6:
        print("  ✅ F_chg AGREES element-wise after L=1 m-perm. The chg-side bug IS just the m-order.")
    else:
        print(f"  ⚠ still off — residue is NOT only L=1 m-order")
    print()

    # post-perm rho_pair parity
    rel_Gpq_perm = np.linalg.norm(rho_pair_v_perm.transpose(2,0,1) - Gpq_py) / np.linalg.norm(Gpq_py)
    print(f"=== POST-PERM rho_pair parity (after L=1 m-perm on both ao axes) ===")
    print(f"  pre-perm  ‖v − py‖/‖py‖ = {rel_Gpq:.4e}")
    print(f"  post-perm ‖v − py‖/‖py‖ = {rel_Gpq_perm:.4e}")
    if rel_Gpq_perm < 1e-6:
        print("  ✅ rho_pair AGREES element-wise after L=1 m-perm. The ao-side bug IS just the m-order.")
    else:
        print(f"  ⚠ still off — residue is NOT only L=1 m-order")
        # Per-AO-pair direct vs conjugate
        ao_label = []
        for sh in ao_basis.shells():
            L = int(sh.l); nc = 2*L+1
            for k in range(nc):
                ao_label.append(f"L={L}")
        print(f"  ao_basis labels: {ao_label}")
        # Per-AO-pair sum of |diff|² for first few rows
        for mu in range(n_orb):
            for nu in range(mu, n_orb):
                Vv = rho_pair_v_perm[mu, nu, :]
                Vp = Gpq_py[:, mu, nu]
                diff = np.linalg.norm(Vv - Vp)
                denom = np.linalg.norm(Vp)
                if denom > 1e-12 and diff/denom > 1e-6:
                    print(f"    ({mu:>2},{nu:>2}) {ao_label[mu]}×{ao_label[nu]}: ‖v−py‖/‖py‖ = {diff/denom:.4e}")
    print()

    # Per-pair complex value dump for selected (mu, nu) pairs across 5 G's
    print(f"=== rho_pair per-element dump for diagnostic pairs ===")
    for mu, nu, label in [(0,0,'Li1s×Li1s'), (1,1,'Li2s×Li2s'),
                          (0,1,'Li1s×Li2s'), (5,5,'H1s×H1s'),
                          (0,5,'Li1s×H1s')]:
        print(f"--- ({mu},{nu}) {label} ---")
        for g_idx in [1, 2, 5, 10]:
            Gv = Gv_py[g_idx]
            v = rho_pair_v_perm[mu, nu, g_idx]
            p = Gpq_py[g_idx, mu, nu]
            r = v/p if abs(p) > 1e-30 else float('nan')
            print(f"  G={Gv}  vqc={fmt_c(v)}  py={fmt_c(p)}  ratio={fmt_c(r) if isinstance(r,complex) else r}")
    print()

    # ─── Step 7c: j3c_p with FULL m-perm consistency (chg axis AND ao axes) ───
    # The "Step 2" of the original script already applied perm; print again clearly.
    print(f"=== j3c_p comparison summary ===")
    print(f"  baseline (no perm, no rescale):                    {rel:.4e}")
    print(f"  with perm + rho_pair libint→libcint rescale:       {np.linalg.norm(j3c_p_vqc_pymesh_C - j3c_p_py) / np.linalg.norm(j3c_p_py):.4e}")
    print()

    # ─── Step 8: test the conjugate hypothesis ───
    # If vqc = conj(py), then F_chg_v == conj(F_chg_py).
    # The Frobenius diff of (vqc - py) compared to (vqc - conj(py))
    # tells us if conjugation closes the gap.
    rel_conj = np.linalg.norm(F_chg_v_complex - np.conj(F_chg_py)) / np.linalg.norm(F_chg_py)
    print(f"=== conjugate hypothesis: ‖F_chg_v − conj(F_chg_py)‖/‖py‖ = {rel_conj:.4e} ===")
    print(f"    (baseline      ‖F_chg_v − F_chg_py‖/‖py‖       = {rel_Fchg:.4e})")
    print()

    # ─── Step 9: test e^{iG·R} per-atom-position phase hypothesis ───
    # If vibe-qc uses e^{+iG·R} instead of e^{-iG·R}, then F_vqc[at_origin_atom] = F_py
    # but F_vqc[H] = conj(F_py[H]).
    # Test per-AO: vqc/py - 1 should be 0 for Li (at origin) and 2i·sin(G·R_H)/...
    # Check sign of imaginary part of vqc/py for AO 16 (H L=0).
    print("=== per-atom phase hypothesis: F_vqc[H L=0] vs conj(F_py[H L=0]) ===")
    # H is at index 16. If conjugation closes for H but Li (0) is already matching:
    F_v_H = F_chg_v_complex[:, 16]
    F_py_H = F_chg_py[:, 16]
    rel_H_direct = np.linalg.norm(F_v_H - F_py_H) / np.linalg.norm(F_py_H)
    rel_H_conj = np.linalg.norm(F_v_H - np.conj(F_py_H)) / np.linalg.norm(F_py_H)
    print(f"    H chg L=0 (AO 16):  ‖v−py‖/‖py‖={rel_H_direct:.4e}, ‖v−conj(py)‖/‖py‖={rel_H_conj:.4e}")

    F_v_Li = F_chg_v_complex[:, 0]
    F_py_Li = F_chg_py[:, 0]
    rel_Li_direct = np.linalg.norm(F_v_Li - F_py_Li) / np.linalg.norm(F_py_Li)
    rel_Li_conj = np.linalg.norm(F_v_Li - np.conj(F_py_Li)) / np.linalg.norm(F_py_Li)
    print(f"    Li chg L=0 (AO 0):  ‖v−py‖/‖py‖={rel_Li_direct:.4e}, ‖v−conj(py)‖/‖py‖={rel_Li_conj:.4e}")

    # Per-AO conjugate-closes ratio
    print()
    print("=== per-AO direct vs conjugate diff for ALL chg AOs ===")
    print(f"{'AO':>3}  {'shell':>15}  {'rel(direct)':>14}  {'rel(conj)':>14}  {'closer':>8}")
    chg_shell_id = [(0,'Li L=0'),(1,'Li L=1'),(2,'Li L=1'),(3,'Li L=1'),
                    (4,'Li L=2'),(5,'Li L=2'),(6,'Li L=2'),(7,'Li L=2'),(8,'Li L=2'),
                    (9,'Li L=3'),(10,'Li L=3'),(11,'Li L=3'),(12,'Li L=3'),(13,'Li L=3'),(14,'Li L=3'),(15,'Li L=3'),
                    (16,'H L=0'),(17,'H L=1'),(18,'H L=1'),(19,'H L=1'),
                    (20,'H L=2'),(21,'H L=2'),(22,'H L=2'),(23,'H L=2'),(24,'H L=2')]
    for ao_i, label in chg_shell_id:
        if ao_i >= n_chg:
            break
        Fv = F_chg_v_complex[:, ao_i]
        Fp = F_chg_py[:, ao_i]
        denom = np.linalg.norm(Fp)
        if denom < 1e-30:
            continue
        rel_d = np.linalg.norm(Fv - Fp) / denom
        rel_c = np.linalg.norm(Fv - np.conj(Fp)) / denom
        closer = "direct" if rel_d < rel_c else ("conj" if rel_c < rel_d else "tie")
        print(f"{ao_i:>3}  {label:>15}  {rel_d:>14.4e}  {rel_c:>14.4e}  {closer:>8}")


if __name__ == "__main__":
    main()
