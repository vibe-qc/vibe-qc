"""M1 Step 1 proper — verify compute_3c_eri_lattice vs PySCF pbc_intor.

Constructs a combined PySCF cell: orbital basis (sto-3g on Li, H)
plus fused aux basis (modrho + chg on a ghost atom at origin).
Then runs pbc_intor('int3c2e') with shell slices and compares
element-by-element with vibe-qc's compute_3c_eri_lattice.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903
A_LIH_ANG = 4.084
ETA = 0.25
CUTOFF_BOHR = 30.0

PYSCF_SCRIPT = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.df import gdf_builder, df as pbc_df
from pyscf import lib

A_CONV = 4.084
ETA = 0.25
a = A_CONV
lattice = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
]) * a

# ----- original cell (orbital basis) -----
cell = pbc_gto.M(
    atom=[("Li", (0.0, 0.0, 0.0)), ("H", (0.5*a, 0.5*a, 0.5*a))],
    a=lattice, unit="A", basis="sto-3g",
    precision=1e-12, verbose=0,
)
n_orb_shells = cell.nbas
n_orb = cell.nao_nr()

# ----- fused aux basis (modrho + chg) -----
auxcell = pbc_df.make_modrho_basis(cell, "def2-svp-jkfit", drop_eta=0.0)
fused_cell, fuse = gdf_builder.fuse_auxcell(auxcell, ETA)
n_aux = auxcell.nao_nr()
n_fused = fused_cell.nao_nr()
n_chg = n_fused - n_aux

# Build a custom basis dict for ghost atom X that carries the fused basis.
# PySCF custom basis format:
#   "X": [[L, [exps, coef_ctr0, coef_ctr1, ...]], ...]
# We flatten fused_cell's shells into this format.
# For contracted shells with multiple contractions: each contraction
# becomes a separate [L, [exps, coefs_single_ctr]] entry.
ghost_basis = []
for i in range(fused_cell.nbas):
    L = int(fused_cell.bas_angular(i))
    exps = [float(x) for x in fused_cell.bas_exp(i)]
    coefs = fused_cell.bas_ctr_coeff(i)  # (nprim, nctr) column-major
    # Flatten: for nctr=1 this is (nprim, 1), take [:, 0]
    # For nctr>1, each column is a separate contraction
    if coefs.ndim == 1:
        ctr_coefs = [float(c) for c in coefs]
        ghost_basis.append([L, [exps, ctr_coefs]])
    else:
        # nctr columns; PySCF format: [L, [exps, ctr0, ctr1, ...]]
        entry = [exps]
        for j in range(coefs.shape[1]):
            entry.append([float(c) for c in coefs[:, j]])
        ghost_basis.append([L, entry])

# Create the combined cell: orbital atoms + ghost atom at origin with fused basis.
combined_atom = list(cell._atom) + [["X", (0.0, 0.0, 0.0)]]
combined = pbc_gto.M(
    atom=combined_atom,
    a=lattice, unit="A",
    basis={"default": "sto-3g", "X": ghost_basis},
    precision=1e-12, verbose=0,
)
# Set the lattice correctly (pbc_gto.M with ghost atoms may reset some attrs)
combined.a = lattice
combined.unit = "A"
combined.build(dump_input=False)

# Verify shell counts
n_combined_shells = combined.nbas
n_fused_shells = len(ghost_basis)
# Orbital shells should be first n_orb_shells, then fused shells
# Verify by checking angular momentum of representative shells
fused_start = n_orb_shells

# pbc_intor('int3c2e') on combined cell:
# Shell indices: i,j from orbital (0..n_orb_shells), k from fused aux
# (fused_start..n_combined_shells)
T_lat_py = combined.pbc_intor(
    'int3c2e',
    shls_slice=(0, n_orb_shells, 0, n_orb_shells, fused_start, n_combined_shells)
)

# pbc_intor returns (n_i, n_j, n_k) row-major
# n_i = n_orb, n_j = n_orb, n_k = fused_nao (number of fused AOs)
# But wait — the combined cell has the fused basis on a ghost atom,
# and the orbital basis on Li+H. The total nao is cell.nao_nr + fused_nao.
# For the k-slice (fused_start..n_combined_shells), n_k is the number of
# AOs for those shells.
k_ao_start = sum(2*combined.bas_angular(i)+1 for i in range(fused_start))
k_ao_end = combined.nao_nr
k_count = k_ao_end - k_ao_start

# Reshape: pbc_intor returns (n_i, n_j, n_k) row-major
# Actually wait — pbc_intor returns a flat array. Need to figure out the ordering.
# pbc_intor('int3c2e') returns integrals in the order (i,j,k) with
# i fastest, then j, then k? Let me check PySCF's convention.
# Usually it's (i*nj*nk + j*nk + k) for (i,j,k) or C-contiguous.
# Let me try reshape:
T_flat = np.asarray(T_lat_py)
# Try (n_orb, n_orb, k_count)
T_lat_py = T_flat.reshape(n_orb, n_orb, k_count)
# Transpose to (k, i, j) to match vibe-qc: (n_fused, n_orb, n_orb)
T_lat_py = T_lat_py.transpose(2, 0, 1)

# Permute L=1 within the fused basis block: PySCF uses libcint ordering
# (px, py, pz). After T_lat_py is assembled, L=1 is already in that order.
# But for vibe-qc comparison we also need to compare only the fused rows,
# and the fused basis order may differ from vibe-qc's fused basis.
# For now, compare per-basis-norm (Frobenius per AO).

# Store the raw tensor + metadata
result = {
    "T_lat_py": T_lat_py.tolist(),
    "n_orb": n_orb,
    "n_aux": n_aux,
    "n_chg": n_chg,
    "n_fused": n_fused,
    "n_orb_shells": n_orb_shells,
    "n_combined_shells": n_combined_shells,
    "fused_start": fused_start,
    "k_count": k_count,
    # Per-fused-shell metadata for vibe-qc mapping
    "fused_shells_py": [{"L": int(combined.bas_angular(i)),
                         "nprim": int(combined.bas_nprim(i)),
                         "nctr": int(combined.bas_nctr(i)),
                         "origin": [float(x) for x in combined.bas_coord(i)]}
                        for i in range(fused_start, n_combined_shells)],
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
    from vibeqc.aux_basis import (
        make_aux_basis_set,
        make_compensating_basis,
        make_fused_basis,
        make_modrho_aux_basis,
    )

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
    print("M1 Step 1: T_lat element-wise vs PySCF pbc_intor")
    print("=" * 72)
    system, mol, ao_basis, modrho, chg, fused = lih_setup()
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    n_fused = fused.nbasis
    n_orb = ao_basis.nbasis

    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = CUTOFF_BOHR

    # vibe-qc lattice 3c
    from vibeqc._vibeqc_core import compute_3c_eri_lattice

    T_v = np.asarray(compute_3c_eri_lattice(ao_basis, fused, system, lo))
    print(f"vibe-qc T_lat shape: {T_v.shape}")
    print(f"  ‖T‖_F = {np.linalg.norm(T_v):.4e}")

    print("Running PySCF reference...")
    py = run_pyscf()
    T_py = np.array(py["T_lat_py"])
    print(f"PySCF T_lat shape: {T_py.shape}")
    print(f"  ‖T‖_F = {np.linalg.norm(T_py):.4e}")
    print(f"  n_orb={py['n_orb']}, n_aux={py['n_aux']}, n_chg={py['n_chg']}")

    # Permute vibe-qc's T for L=1 m-ordering comparison
    perm_fused = make_perm_l1(fused)
    perm_ao = make_perm_l1(ao_basis)
    T_v_p = T_v[perm_fused][:, perm_ao][:, :, perm_ao]

    # Both should now be in libcint ordering for L=1.
    # But the fused basis ordering in PySCF's ghost atom might differ from
    # vibe-qc's fused basis. Compare per-region norms first.

    # Align: vibe-qc's aux block vs PySCF's first n_aux fused rows
    rel_total = np.linalg.norm(
        T_v_p[:, :n_orb, :n_orb] - T_py[:, :n_orb, :n_orb]
    ) / np.linalg.norm(T_py)
    print(f"\n  Total rel diff (vibe-qc vs PySCF): {rel_total:.4e}")

    # Per-fused-AO norm comparison
    print(f"\n  Per-fused-AO norm comparison (first 10):")
    print(f"  {'AO':>3}  {'‖vqc‖':>12}  {'‖py‖':>12}  {'ratio':>10}")
    for i in range(min(10, n_fused)):
        vn = np.linalg.norm(T_v_p[i])
        pn = np.linalg.norm(T_py[i])
        r = vn / pn if pn > 1e-30 else float("nan")
        region = "aux" if i < n_aux else "chg"
        print(f"  {i:>3}  {vn:>12.4e}  {pn:>12.4e}  {r:>10.4f}  ({region})")

    # Detailed per-L breakdown
    from vibeqc.aux_basis import _per_ao_pair_libint_to_libcint_scale

    scales_fused = _per_ao_pair_libint_to_libcint_scale(fused)
    scales_ao = _per_ao_pair_libint_to_libcint_scale(ao_basis)

    print(f"\n  Per-L scale factors:")
    print(f"  fused scales: {scales_fused[:5]}...")
    print(f"  ao scales:    {scales_ao}")

    # Compare T_chg block where j3c_p subtractions happen
    print(f"\n  T_chg block (rows {n_aux}:{n_fused}):")
    T_v_chg = T_v_p[n_aux:]
    T_py_chg = T_py[n_aux:]
    rel_chg = np.linalg.norm(T_v_chg - T_py_chg) / np.linalg.norm(T_py_chg)
    print(f"  ‖v−py‖/‖py‖ (chg block) = {rel_chg:.4e}")

    # Per-chg-AO comparison
    for i in range(min(5, n_chg)):
        vn = np.linalg.norm(T_v_chg[i])
        pn = np.linalg.norm(T_py_chg[i])
        r = vn / pn if pn > 1e-30 else float("nan")
        print(f"    chg AO {i}: ‖v‖={vn:.4e}, ‖py‖={pn:.4e}, ratio={r:.4f}")
        diff = np.max(np.abs(T_v_chg[i] - T_py_chg[i]))
        if diff > 1e-8:
            print(f"      max|diff|={diff:.4e}")

    print()
    print("=" * 72)
    print(f"  T_lat parity: {'✅' if rel_total < 1e-6 else '❌'} {rel_total:.1e}")
    print(f"  T_chg parity: {'✅' if rel_chg < 1e-6 else '❌'} {rel_chg:.1e}")


if __name__ == "__main__":
    main()
