"""Probe PySCF's GDF construction on MgO/sto-3g to understand:
- which aux basis it picks for sto-3g (the auto-resolution path),
- the modrho-compensated auxcell shell list,
- the shape and structure of the (P|Q) metric and the (P|μν) tensor
  it ultimately serialises to mf.with_df._cderi.

We need this BEFORE designing the vibe-qc aux-basis loader: we must
match what PySCF emits, primitive-for-primitive, to get sub-µHa
parity on Lpq later.
"""
from __future__ import annotations

import numpy as np

from pyscf.pbc import df, gto

# Same MgO setup as examples/periodic/MgO-rocksalt/MgO-rocksalt-RHF-sto3g.py
A = 4.211
mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
atom_lines = []
for fx, fy, fz in mg_frac:
    atom_lines.append(f"Mg {fx*A:.6f} {fy*A:.6f} {fz*A:.6f}")
for fx, fy, fz in o_frac:
    atom_lines.append(f"O {fx*A:.6f} {fy*A:.6f} {fz*A:.6f}")

cell = gto.M(
    atom="; ".join(atom_lines),
    a=np.diag([A, A, A]),
    basis="sto-3g",
    unit="A",
    verbose=0,
    precision=1e-8,
)
print(f"PySCF cell: nao = {cell.nao}, n_atoms = {cell.natm}")

with_df = df.GDF(cell)
with_df.build()

print(f"\nGDF auxbasis: {with_df.auxbasis!r}")
auxcell = with_df.auxcell
print(f"auxcell.nao = {auxcell.nao}, n_shells = {auxcell.nbas}")
print("auxcell shells:")
for i in range(auxcell.nbas):
    atm = int(auxcell.bas_atom(i))
    l = int(auxcell.bas_angular(i))
    nprim = int(auxcell.bas_nprim(i))
    nctr = int(auxcell.bas_nctr(i))
    expnts = list(auxcell.bas_exp(i))
    print(f"  shell {i}: atom={atm}  L={l}  nprim={nprim}  nctr={nctr}  exps={expnts[:3]}{'...' if nprim > 3 else ''}")

# What integrals does GDF actually compute?
# - 2c metric: (P|Q) on the auxcell, periodic image-summed.
# - 3c tensor: (P|μν) on auxcell × cell × cell, ket image-summed.
# Both are charge-compensated (modrho): each aux primitive has a
# matching smooth Gaussian compensating its monopole.

# Pull the 2c metric directly via PySCF's int2c2e for comparison.
# This is the analytic Coulomb metric *without* the modrho compensation.
# Useful as a reference for what an uncompensated 2c looks like.
m2c_raw = auxcell.pbc_intor("int2c2e", hermi=1)
print(f"\nauxcell.pbc_intor('int2c2e'):  shape = {m2c_raw.shape}, "
      f"||M||_max = {float(np.max(np.abs(m2c_raw))):.3e}, "
      f"||M_diag - M_diag^T||_max trivially 0")
print(f"  diag(M_raw)[:5] = {[float(x) for x in m2c_raw.diagonal()[:5]]}")

# Pull the modrho-compensated 2c metric. This is what GDF actually uses.
# Method: int2c2e on the auxcell + smooth chgs (the augcell), then
# shave off the comp block. PySCF's helper is `with_df.get_2c2e()` in
# some versions; here we do it the long way to be explicit.
try:
    fused_cell, fused_kpts = with_df.fuse_auxcell(auxcell, kpt=np.zeros(3))
    print(f"\nfused auxcell + comp: nao = {fused_cell.nao}")
except Exception as e:
    print(f"\nfuse_auxcell unavailable: {e}")

# Read back the actual Lpq stored in the GDF object.
import h5py
print(f"\n_cderi file: {with_df._cderi}")
with h5py.File(with_df._cderi, "r") as h5:
    print(f"  groups: {list(h5.keys())}")
    if "j3c" in h5:
        for k in h5["j3c"]:
            arr = h5["j3c"][k]
            print(f"  j3c/{k}: shape={arr.shape}, dtype={arr.dtype}")

# What does the (compact) tril-packed Lpq look like?
nao = cell.nao
kpts_ii = np.zeros((2, 3))
print(f"\nLpq via sr_loop:")
total_naux = 0
for LpqR_packed, _LpqI, sign in with_df.sr_loop(kpts_ii, max_memory=2000, compact=True):
    total_naux += LpqR_packed.shape[0]
    print(f"  chunk: shape = {LpqR_packed.shape}, sign = {sign}, "
          f"max|elem| = {float(np.max(np.abs(LpqR_packed))):.3e}")
print(f"  total naux = {total_naux}")
