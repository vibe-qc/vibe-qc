"""Element-wise PySCF↔vibe-qc comparison of the compcell AFT pipeline.

Goal: isolate the bug that makes vibe-qc's apply_aft_correction=True
give H2 SCF energies +0.67 Ha off PySCF.

The compcell GDF Lpq construction has these intermediate quantities,
each of which we compare:

  (1) Bare 2c metric on the fused (modrho_aux ∪ chg) basis:
      PySCF:   fused_cell.pbc_intor('int2c2e', kpts=[0])
      vibeqc:  compute_2c_eri_lattice(fused, system, lat_opts)

  (2) AFT j2c_p correction (chg-anything LR Coulomb in G-space):
      PySCF:   internal `j2c_p` from _CCGDFBuilder.get_2c2e[0]
      vibeqc:  _compcell_aft_correction(fused, n_aux, system, eta)

  (3) Compensated 2c metric after AFT subtraction + fuse transform:
      PySCF:   _CCGDFBuilder.get_2c2e() output (after fuse(fuse(j2c)))
      vibeqc:  A @ M_fused_with_aft @ A.T  (no fuse() — done as matmul)

For each, report the per-block ||.||_F (aux-aux, aux-chg, chg-chg)
and the diagonal of the first L=0 shell — both invariant under AO
permutation (the L=1 (py,pz,px)↔(px,py,pz) thing). If those agree,
dig into element-wise diff with the permutation applied.

PySCF is invoked out-of-process per CLAUDE.md §10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

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


# H2 / 12-bohr cubic / def2-svp-jk, η=0.25 (the same setup that fails
# under apply_aft_correction=True).
BOX_BOHR = 12.0
SEP_BOHR = 1.4
AUX_NAME_VIBEQC = "def2-svp-jk"
AUX_NAME_PYSCF = "def2-svp-jkfit"     # PySCF's name for the same basis
ETA = 0.25
LAT_CUTOFF_BOHR = 30.0


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.df import gdf_builder

BOX = {box}
SEP = {sep}
AUX = "{aux}"
ETA = {eta}

# H2 / 12-bohr cubic
cell = pbc_gto.M(
    atom=[("H", (0.0, 0.0, -SEP/2)), ("H", (0.0, 0.0, SEP/2))],
    a=np.eye(3) * BOX,
    unit="B",                # bohr
    basis="sto-3g",
    precision=1e-12,
    verbose=0,
)

# Build the modrho-rescaled aux and the compensating basis exactly as
# PySCF's _CCGDFBuilder does internally.
from pyscf.pbc.df import df as pbc_df
auxcell = pbc_df.make_modrho_basis(cell, AUX, drop_eta=0.0)
fused_cell, fuse = gdf_builder.fuse_auxcell(auxcell, ETA)

naux = auxcell.nao_nr()
nfused = fused_cell.nao_nr()
nchg = nfused - naux

# (1) BARE 2c metric on fused.
j2c_bare = fused_cell.pbc_intor("int2c2e", hermi=0, kpts=np.zeros((1,3)))[0]
j2c_bare = 0.5 * (j2c_bare + j2c_bare.conj().T)
# (We force hermitian-symmetrise so the comparison isn't sensitive to
# PySCF's "may not be hermitian without symmetric lattice vectors"
# quirk noted in gdf_builder.py.)

# (2) AFT j2c_p. We replicate the inner loop of _CCGDFBuilder.get_2c2e
# so the value is identical to what PySCF subtracts internally.
from pyscf import lib
from pyscf.pbc.df import aft, ft_ao
from pyscf.pbc import tools as pbc_tools

# Match PySCF's mesh-sizing logic from get_2c2e
precision = cell.precision ** 2  # auxcell.precision squared per PySCF
from pyscf.pbc.df.gdf_builder import estimate_ke_cutoff_for_eta
ke = estimate_ke_cutoff_for_eta(auxcell, ETA, precision)
mesh = auxcell.cutoff_to_mesh(ke)
mesh = cell.symmetrize_mesh(mesh)

Gv, Gvbase, kws = fused_cell.get_Gv_weights(mesh)
b = fused_cell.reciprocal_vectors()
gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
ngrids = Gv.shape[0]
kpt = np.zeros(3)
# weighted_coulG returns 4π/G² · kws with G=0 zeroed (False = no exxdiv)
coulG = aft.weighted_coulG(None, kpt, False, mesh, Gv=Gv, omega=0.0) if False else None
# That signature doesn't exist; use the simple form:
coulG = pbc_tools.get_coulG(cell, kpt, False, None, mesh, Gv) * kws

auxG = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T  # (nfused, ngrids)
auxGR = np.asarray(auxG.real, order="C")
auxGI = np.asarray(auxG.imag, order="C")

j2c_p  = (auxGR[naux:] * coulG) @ auxGR.T
j2c_p += (auxGI[naux:] * coulG) @ auxGI.T
# j2c_p shape: (nchg, nfused), real-valued at Γ.

# (3) Compensated j2c: subtract j2c_p, symmetrise, fuse both axes.
j2c_corrected = j2c_bare.copy()
j2c_corrected[naux:] -= j2c_p
j2c_corrected[:naux, naux:] -= j2c_p[:, :naux].conj().T
j2c_corrected = 0.5 * (j2c_corrected + j2c_corrected.conj().T)
j2c_compensated = fuse(fuse(j2c_corrected), axis=1)        # (naux, naux)

# Frobenius norms per block (permutation-invariant).
def fro(M): return float(np.linalg.norm(M))
def diag5(M): return [float(x) for x in np.diag(M)[:8]]

# Per-shell ordering for AO alignment with vibe-qc.
shells = []
ao_loc = fused_cell.ao_loc_nr()
for i in range(fused_cell.nbas):
    shells.append({{
        "ish": i,
        "atom": int(fused_cell.bas_atom(i)),
        "l": int(fused_cell.bas_angular(i)),
        "n_prims": int(fused_cell.bas_nprim(i)),
        "ao_start": int(ao_loc[i]),
        "ao_end": int(ao_loc[i+1]),
        "exps": [float(x) for x in fused_cell.bas_exp(i)],
    }})

print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "naux": naux, "nchg": nchg, "nfused": nfused,
    "mesh": [int(x) for x in mesh],
    "ngrids": int(ngrids),
    "ke_cutoff": float(ke),
    "ETA": float(ETA),
    "shells_fused": shells,
    # Bare metric block norms + first 8 diagonal entries
    "M_bare_fro": fro(j2c_bare),
    "M_bare_aux_aux_fro": fro(j2c_bare[:naux, :naux]),
    "M_bare_aux_chg_fro": fro(j2c_bare[:naux, naux:]),
    "M_bare_chg_chg_fro": fro(j2c_bare[naux:, naux:]),
    "M_bare_diag5": diag5(j2c_bare),
    "M_bare_chg_chg_diag5": diag5(j2c_bare[naux:, naux:]),
    # j2c_p block norms + first 8 diagonal entries (chg-fused tensor)
    "j2c_p_fro": fro(j2c_p),
    "j2c_p_chg_aux_fro": fro(j2c_p[:, :naux]),
    "j2c_p_chg_chg_fro": fro(j2c_p[:, naux:]),
    "j2c_p_chg_chg_diag5": diag5(j2c_p[:, naux:]),
    # Compensated j2c (after fuse) — naux × naux
    "M_compensated_fro": fro(j2c_compensated),
    "M_compensated_diag5": diag5(j2c_compensated),
    "M_compensated_min_eig": float(np.linalg.eigvalsh(j2c_compensated).min()),
    "M_compensated_max_eig": float(np.linalg.eigvalsh(j2c_compensated).max()),
}}))
'''


def run_pyscf() -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(
        box=BOX_BOHR, sep=SEP_BOHR, aux=AUX_NAME_PYSCF, eta=ETA,
    )
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


def main() -> int:
    # ------------------------------------------------------------------
    # vibe-qc side
    # ------------------------------------------------------------------
    half = 0.5 * SEP_BOHR
    system = vq.PeriodicSystem(
        3, np.diag([BOX_BOHR, BOX_BOHR, BOX_BOHR]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    mol = system.unit_cell_molecule()
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
    j2c_p_vqc = _compcell_aft_correction(fused, n_aux, system, eta=ETA,
                                         precision=1e-10)
    M_with_aft = M_bare.copy()
    M_with_aft[n_aux:, :]      -= j2c_p_vqc
    M_with_aft[:n_aux, n_aux:] -= j2c_p_vqc[:, :n_aux].T
    M_with_aft = 0.5 * (M_with_aft + M_with_aft.T)
    M_compensated = A @ M_with_aft @ A.T
    M_compensated = 0.5 * (M_compensated + M_compensated.T)
    eigs_vqc = np.linalg.eigvalsh(M_compensated)

    # ------------------------------------------------------------------
    # PySCF side
    # ------------------------------------------------------------------
    py = run_pyscf()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print(f"=== Sizes ===")
    print(f"  vibe-qc: n_aux={n_aux}, n_chg={n_chg}, n_fused={n_fused}")
    print(f"  PySCF:   n_aux={py['naux']}, n_chg={py['nchg']}, "
          f"n_fused={py['nfused']}")
    print(f"  PySCF mesh: {py['mesh']} ({py['ngrids']} G-vectors), "
          f"ke_cutoff={py['ke_cutoff']:.2f}")
    print()

    print(f"=== Bare M_fused per-block ||.||_F ===")
    print(f"  {'block':<12s}  {'vibe-qc':>14s}  {'PySCF':>14s}  {'ratio':>10s}")
    pairs = [
        ("all",      float(np.linalg.norm(M_bare)),               py["M_bare_fro"]),
        ("aux-aux",  float(np.linalg.norm(M_bare[:n_aux, :n_aux])), py["M_bare_aux_aux_fro"]),
        ("aux-chg",  float(np.linalg.norm(M_bare[:n_aux, n_aux:])), py["M_bare_aux_chg_fro"]),
        ("chg-chg",  float(np.linalg.norm(M_bare[n_aux:, n_aux:])), py["M_bare_chg_chg_fro"]),
    ]
    for name, v, p in pairs:
        print(f"  {name:<12s}  {v:>14.6f}  {p:>14.6f}  {v/p if p else float('inf'):>10.4f}")
    print()

    print(f"=== j2c_p (AFT correction) per-block ||.||_F ===")
    print(f"  {'block':<12s}  {'vibe-qc':>14s}  {'PySCF':>14s}  {'ratio':>10s}")
    pairs = [
        ("all",          float(np.linalg.norm(j2c_p_vqc)),                py["j2c_p_fro"]),
        ("chg-aux",      float(np.linalg.norm(j2c_p_vqc[:, :n_aux])),     py["j2c_p_chg_aux_fro"]),
        ("chg-chg",      float(np.linalg.norm(j2c_p_vqc[:, n_aux:])),     py["j2c_p_chg_chg_fro"]),
    ]
    for name, v, p in pairs:
        print(f"  {name:<12s}  {v:>14.6f}  {p:>14.6f}  "
              f"{v/p if p else float('inf'):>10.4f}")
    print()

    print(f"=== Compensated M (after AFT + fuse) ===")
    print(f"  vibe-qc: ||M||_F = {np.linalg.norm(M_compensated):.6f}, "
          f"min_eig={eigs_vqc.min():+.4e}, max_eig={eigs_vqc.max():.4e}")
    print(f"  PySCF:   ||M||_F = {py['M_compensated_fro']:.6f}, "
          f"min_eig={py['M_compensated_min_eig']:+.4e}, "
          f"max_eig={py['M_compensated_max_eig']:.4e}")
    print(f"  ratio:   {np.linalg.norm(M_compensated)/py['M_compensated_fro']:.4f}")
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
