"""M1 steps 1+2 — j3c_p parity + vibe-qc T_lat vs T_mol comparison.

Step 1: vibe-qc lattice 3c = molecular 3c + lattice images. Verify that
the lattice contribution is reasonable and matches expectations.

Step 2: j3c_p element-wise parity with PySCF (Bloch pair-FT).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import vibeqc as vq
from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch
from vibeqc._vibeqc_core import (
    compute_3c_eri,
    compute_3c_eri_lattice,
    direct_lattice_cells,
)
from vibeqc.aux_basis import (
    _per_ao_pair_libint_to_libcint_scale,
    make_aux_basis_set,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
    rsgdf_aux_fourier_transform,
)

ANG2BOHR = 1.0 / 0.529177210903
A_LIH_ANG = 4.084
ETA = 0.25
CUTOFF_BOHR = 30.0

PYSCF_SCRIPT = """
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto, tools as pbc_tools
from pyscf.pbc.df import gdf_builder, ft_ao
from pyscf.pbc.df import df as pbc_df
from pyscf.pbc.df.gdf_builder import estimate_ke_cutoff_for_eta
from pyscf import lib

A_CONV = 4.084
ETA = 0.25
a = A_CONV
lattice = np.array([
    [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
]) * a
cell = pbc_gto.M(
    atom=[("Li", (0.0, 0.0, 0.0)), ("H", (0.5*a, 0.5*a, 0.5*a))],
    a=lattice, unit="A", basis="sto-3g",
    precision=1e-12, verbose=0,
)

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

# Chg FT
shls_slice = (auxcell.nbas, fused_cell.nbas)
auxG = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
F_chg_py = lib.transpose(auxG)

# Lattice-summed pair FT
Gpq = ft_ao.ft_aopair(cell, Gv, shls_slice=None, aosym='s1', b=b,
                       gxyz=gxyz, Gvbase=Gvbase,
                       kpti_kptj=np.zeros((2,3)), q=kpt)
Gpq = np.asarray(Gpq).reshape(Gv.shape[0], n_orb, n_orb)

# j3c_p
F_chg_w = auxG * coulG_kws
j3c_p_py = np.einsum('cG,Gmn->cmn', F_chg_w.conj(), Gpq).real

# Per-chg-AO j3c_p norms
chg_norms_py = [float(np.linalg.norm(j3c_p_py[i])) for i in range(nchg)]

result = {
    "Gv": Gv.tolist(),
    "coulG_kws": coulG_kws.tolist(),
    "j3c_p_py": j3c_p_py.tolist(),
    "F_chg_py_real": F_chg_py.real.tolist(),
    "F_chg_py_imag": F_chg_py.imag.tolist(),
    "Gpq_real": Gpq.real.tolist(),
    "Gpq_imag": Gpq.imag.tolist(),
    "chg_norms_py": chg_norms_py,
    "n_chg": nchg, "n_orb": n_orb, "n_aux": naux,
    "mesh": [int(m) for m in mesh],
}
print("VIBEQC-PYSCF-RESULT:" + json.dumps(result))
"""


def lih_setup():
    a = A_LIH_ANG * ANG2BOHR
    lattice = (
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        * a
    )
    atoms = [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0.5 * a] * 3)]
    system = vq.PeriodicSystem(3, lattice, atoms)
    mol = system.unit_cell_molecule()
    ao_basis = vq.BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=ETA)
    fused = make_fused_basis(modrho, chg, mol)
    return system, mol, ao_basis, modrho, chg, fused


def run_pyscf():
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(PYSCF_SCRIPT)
        fname = f.name
    try:
        p = subprocess.run([py, fname], capture_output=True, text=True)
        if p.returncode != 0:
            print("PySCF subprocess failed:", file=sys.stderr)
            print(p.stderr[-3000:], file=sys.stderr)
            sys.exit(2)
        for line in p.stdout.splitlines():
            if line.startswith("VIBEQC-PYSCF-RESULT:"):
                return json.loads(line[len("VIBEQC-PYSCF-RESULT:") :])
        sys.exit(2)
    finally:
        os.unlink(fname)


def make_perm_l1(basis):
    n = basis.nbasis
    perm = list(range(n))
    bf = 0
    for sh in basis.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        if L == 1:
            perm[bf], perm[bf + 1], perm[bf + 2] = bf + 2, bf, bf + 1
        bf += nc
    return np.array(perm)


def main():
    print("M1 Steps 1+2: T_lat vs T_mol (vibe-qc internal) + j3c_p vs PySCF")
    print("=" * 72)
    system, mol, ao_basis, modrho, chg, fused = lih_setup()
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    n_fused = fused.nbasis
    n_orb = ao_basis.nbasis

    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = CUTOFF_BOHR
    cells = direct_lattice_cells(system, lo.cutoff_bohr)
    print(f"  n_cells = {len(cells)}")
    print(f"  n_orb={n_orb}, n_aux={n_aux}, n_chg={n_chg}, n_fused={n_fused}")

    # ===========================================================
    # Step 1: vibe-qc internal T_lat vs T_mol comparison
    # ===========================================================
    print("\n--- Step 1: T_lat vs T_mol (vibe-qc internal) ---")
    T_mol = np.asarray(compute_3c_eri(ao_basis, fused))
    T_mol = T_mol.reshape(n_orb, n_orb, n_fused).transpose(2, 0, 1)
    T_mol = 0.5 * (T_mol + np.swapaxes(T_mol, 1, 2))
    T_lat = np.asarray(compute_3c_eri_lattice(ao_basis, fused, system, lo))

    T_diff = T_lat - T_mol
    print(f"  ‖T_mol‖_F = {np.linalg.norm(T_mol):.4e}")
    print(f"  ‖T_lat‖_F = {np.linalg.norm(T_lat):.4e}")
    print(
        f"  ‖T_lat−T_mol‖_F / ‖T_lat‖_F = {np.linalg.norm(T_diff) / np.linalg.norm(T_lat):.4e}"
    )

    # Per-region
    for region, start, end in [("aux", 0, n_aux), ("chg", n_aux, n_fused)]:
        print(f"\n  {region} block:")
        print(f"    ‖T_mol‖_F = {np.linalg.norm(T_mol[start:end]):.4e}")
        print(f"    ‖T_lat‖_F = {np.linalg.norm(T_lat[start:end]):.4e}")
        d = T_lat[start:end] - T_mol[start:end]
        print(
            f"    ‖diff‖_F / ‖T_lat‖_F = {np.linalg.norm(d) / np.linalg.norm(T_lat[start:end]):.4e}"
        )

    # Per-chg-AO: ratio of T_lat to T_mol norms
    print(f"\n  Per-chg-AO T_lat/T_mol norm ratios:")
    chg_labels = []
    for sh in chg.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        chg_labels.extend([f"L={L}"] * nc)
    for i in range(n_chg):
        nl = np.linalg.norm(T_lat[n_aux + i])
        nm = np.linalg.norm(T_mol[n_aux + i])
        r = nl / nm if nm > 1e-30 else float("nan")
        print(
            f"    chg AO {i:>2} ({chg_labels[i]}): lat={nl:.4e}, mol={nm:.4e}, ratio={r:.4f}"
        )

    # Per-chg-AO T_lat per-orb-pair (μ,ν) values for the first cell image
    # The dominant cell image is the home cell (T=0). Print a few slices.
    print(f"\n  T_lat−T_mol per-chg-AO max element:")
    for i in range(min(5, n_chg)):
        d = np.abs(T_diff[n_aux + i])
        print(
            f"    chg AO {i}: max|diff| = {np.max(d):.4e} at ({np.unravel_index(np.argmax(d), d.shape)})"
        )

    # ===========================================================
    # Step 2: j3c_p element-wise vs PySCF
    # ===========================================================
    print("\n\n--- Step 2: j3c_p vs PySCF (Bloch pair-FT) ---")
    print("Running PySCF reference...")
    py = run_pyscf()
    Gv_py = np.array(py["Gv"])
    coulG_py = np.array(py["coulG_kws"])
    j3c_p_py = np.array(py["j3c_p_py"])
    F_chg_py = np.array(py["F_chg_py_real"]) + 1j * np.array(py["F_chg_py_imag"])
    Gpq_py = np.array(py["Gpq_real"]) + 1j * np.array(py["Gpq_imag"])
    chg_norms_py = py["chg_norms_py"]
    n_G = Gv_py.shape[0]
    print(f"  mesh = {py['mesh']}, n_G = {n_G}")
    print(f"  ‖j3c_p(py)‖_F = {np.linalg.norm(j3c_p_py):.4e}")
    print(f"  ‖F_chg(py)‖_F = {np.linalg.norm(F_chg_py):.4e}")
    print(f"  ‖Gpq(py)‖_F  = {np.linalg.norm(Gpq_py):.4e}")

    # F_chg (vibe-qc)
    F_v = rsgdf_aux_fourier_transform(fused, Gv_py)
    F_chg_v = F_v[n_aux:]
    perm_chg = make_perm_l1(fused)[n_aux:] - n_aux
    F_chg_v_p = F_chg_v[perm_chg]
    rel_Fchg = np.linalg.norm(F_chg_v_p.T - F_chg_py) / np.linalg.norm(F_chg_py)
    print(f"\n  F_chg:       ‖v−py‖/‖py‖ = {rel_Fchg:.4e}")

    # Bloch pair-FT (vibe-qc, on PySCF mesh)
    R_g_list = np.array([list(c.r_cart) for c in cells], dtype=float)
    rpb = ao_pair_fourier_transform_bloch(ao_basis, Gv_py, R_g_list, np.zeros(3))
    perm_ao = make_perm_l1(ao_basis)
    rpb_p = rpb[perm_ao][:, perm_ao]
    scales = _per_ao_pair_libint_to_libcint_scale(ao_basis)
    inv = 1.0 / scales
    rpb_pr = rpb_p * inv[:, None, None] * inv[None, :, None]
    rho_pq = rpb_pr.transpose(2, 0, 1)
    rel_Gpq = np.linalg.norm(rho_pq - Gpq_py) / np.linalg.norm(Gpq_py)
    print(f"  Gpq:         ‖v−py‖/‖py‖ = {rel_Gpq:.4e}")

    # j3c_p (vibe-qc, on PySCF mesh, Bloch pair-FT)
    F_chg_w = np.conj(F_chg_v_p) * coulG_py[None, :]
    rho_flat = rpb_pr.reshape(n_orb * n_orb, n_G).T
    j3c_p_v = (F_chg_w @ rho_flat).reshape(n_chg, n_orb, n_orb).real
    rel_j3c = np.linalg.norm(j3c_p_v - j3c_p_py) / np.linalg.norm(j3c_p_py)
    print(f"  j3c_p:       ‖v−py‖/‖py‖ = {rel_j3c:.4e}")

    # Per-chg-AO j3c_p norm comparison
    print(f"\n  Per-chg-AO j3c_p norms:")
    for i in range(min(5, n_chg)):
        vn = np.linalg.norm(j3c_p_v[i])
        pn = chg_norms_py[i]
        r = vn / pn if pn > 1e-30 else float("nan")
        print(f"    chg AO {i}: ‖v‖={vn:.4e}, ‖py‖={pn:.4e}, ratio={r:.4f}")
        diff = np.max(np.abs(j3c_p_v[i] - j3c_p_py[i]))
        if diff > 1e-8:
            print(f"      max|diff|={diff:.4e}")

    # ===========================================================
    # Key insight: compare T_chg vs j3c_p per-chg-AO
    # ===========================================================
    print(f"\n\n--- Per-chg-AO T_lat(chg) vs j3c_p comparison ---")
    print(
        f"  {'AO':>3}  {'‖T_chg‖':>12}  {'‖j3c_p‖':>12}  {'‖T/j3c_p‖ ratio':>16}  {'‖T_py/j3c_p_py‖':>16}"
    )
    for i in range(min(5, n_chg)):
        tn = np.linalg.norm(T_lat[n_aux + i])
        jn = np.linalg.norm(j3c_p_v[i])
        jn_py = chg_norms_py[i]
        print(
            f"  {i:>3}  {tn:>12.4e}  {jn:>12.4e}  "
            f"{tn / jn if jn > 1e-30 else float('nan'):>16.4f}  "
            f"{np.linalg.norm(T_lat[n_aux + i]) / jn_py if jn_py > 1e-30 else float('nan'):>16.4f}"
        )

    # Summary
    print()
    print("=" * 72)
    print(f"  F_chg:  {'✅' if rel_Fchg < 1e-9 else '❌'} {rel_Fchg:.1e}")
    print(f"  Gpq:    {'✅' if rel_Gpq < 1e-8 else '❌'} {rel_Gpq:.1e}")
    print(f"  j3c_p:  {'✅' if rel_j3c < 1e-8 else '❌'} {rel_j3c:.1e}")
    print(
        f"  T_lat vs T_mol diff: {np.linalg.norm(T_diff) / np.linalg.norm(T_lat):.4e}"
    )


if __name__ == "__main__":
    main()
