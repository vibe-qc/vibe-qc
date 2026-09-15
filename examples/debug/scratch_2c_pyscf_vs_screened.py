"""Post-hoc test: do per-shell rcut bounds reproduce PySCF's 2c metric?

vibe-qc currently uses a uniform image cutoff (LatticeSumOptions.cutoff_bohr).
PySCF uses per-shell rcut. This script computes vibe-qc's 2c metric at
a large global cutoff, then ZEROES OUT element groups belonging to
shell pairs (sP, sQ) where the lattice-sum truncation should have
excluded that shell pair. If the post-hoc-screened result matches
PySCF, the slice-3c implementation is "push per-shell rcut into the
C++ kernel".

This isn't perfectly accurate because the screening should be applied
PER CELL inside the integral sum, not after summing — but it's a
useful smoke. If it gets us close to PySCF, we know the direction.
"""
from __future__ import annotations

import numpy as np

from pyscf.pbc import df, gto

import vibeqc as vq
from vibeqc.aux_basis import make_aux_basis_set


def estimate_rcut(alpha: np.ndarray, l: int, c: np.ndarray,
                  precision: float = 1e-8) -> np.ndarray:
    """Port of pyscf.pbc.gto.cell._estimate_rcut."""
    alpha = np.asarray(alpha, dtype=float)
    c = np.asarray(c, dtype=float)
    theta = alpha * 0.5
    a1 = (alpha * 2) ** -0.5
    norm_ang = (2 * l + 1) / (4 * np.pi)
    fac = 2 * np.pi * c**2 * norm_ang / theta / precision
    fac *= 4 * alpha**2
    r0 = 20.0
    r0 = (np.log(fac * r0 * (r0 * 0.5 + a1) ** (2 * l + 2) + 1.0) / theta) ** 0.5
    r0 = (np.log(fac * r0 * (r0 * 0.5 + a1) ** (2 * l + 2) + 1.0) / theta) ** 0.5
    return r0


# --- Setup ---
ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211
A = A_ANG * ANG2BOHR

mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]

atoms_vq = [vq.Atom(12, [f * A for f in p]) for p in mg_frac] + \
           [vq.Atom(8, [f * A for f in p]) for p in o_frac]
system = vq.PeriodicSystem(3, np.diag([A, A, A]), atoms_vq)
mol_vq = system.unit_cell_molecule()

atom_lines = []
for fx, fy, fz in mg_frac:
    atom_lines.append(f"Mg {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
for fx, fy, fz in o_frac:
    atom_lines.append(f"O {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
cell = gto.M(
    atom="; ".join(atom_lines),
    a=np.diag([A_ANG, A_ANG, A_ANG]),
    basis="sto-3g",
    unit="A",
    verbose=0,
    precision=1e-8,
)
with_df = df.GDF(cell)
with_df.build()
auxcell = with_df.auxcell

M_pyscf = np.asarray(auxcell.pbc_intor("int2c2e", hermi=1))
print(f"PySCF: ‖M‖_F={float(np.linalg.norm(M_pyscf)):.4e}, "
      f"max|M|={float(np.max(np.abs(M_pyscf))):.4e}")

aux_vq = make_aux_basis_set(mol_vq, aux_name="def2-universal-jkfit", compensate=True)
shells = aux_vq.shells()

# Per-shell rcut + AO-range mapping
shell_rcuts = []
shell_offsets = []   # AO-index range per shell
off = 0
for sh in shells:
    es = np.asarray(sh.exponents, dtype=float)
    cs = np.abs(np.asarray(sh.coefficients, dtype=float))
    rc = float(estimate_rcut(es, sh.l, cs, 1e-8).max())
    shell_rcuts.append(rc)
    sz = 2 * sh.l + 1
    shell_offsets.append((off, off + sz))
    off += sz
shell_rcuts = np.array(shell_rcuts)

print(f"\nshell rcuts: min={shell_rcuts.min():.2f}, max={shell_rcuts.max():.2f}, "
      f"median={np.median(shell_rcuts):.2f} bohr")

opts = vq.LatticeSumOptions()
opts.cutoff_bohr = float(shell_rcuts.max())
M_vq_full = np.asarray(vq.compute_2c_eri_lattice(aux_vq, system, opts))
M_vq_full = 0.5 * (M_vq_full + M_vq_full.T)
print(f"vibe-qc full sum (cut={opts.cutoff_bohr:.2f}): ‖M‖_F={float(np.linalg.norm(M_vq_full)):.4e}, "
      f"max|M|={float(np.max(np.abs(M_vq_full))):.4e}")

# This isn't a per-cell screen (we already summed all cells), but as a
# sanity check we apply a SHELL-PAIR ZEROING: for each pair (sP, sQ) where
# rcut_P + rcut_Q is "small" (smaller than typical inter-cell distance),
# zero out the corresponding M block. This is a conservative bound — real
# PySCF screening is per-cell so it's tighter.
#
# Better proxy: for each (sP, sQ), if |T| > rcut_P + rcut_Q for ALL T except
# T=0, that pair's contribution should be just the molecular (T=0) value.
# But the full lattice-summed value can still be close to molecular if all
# but the T=0 term were screened.
#
# We don't have per-cell decomposition here; this script is a stub for
# the proper C++-level test (which lands when slice-3c implementation
# adds rcut to compute_2c_eri_lattice).

print()
print("Direct comparison shows PySCF and vibe-qc differ in shape — element-wise")
print("alignment requires AO-permutation between libint and libcint orders.")
print("This script is a stub: the rigorous test waits on the slice-3c C++ work")
print("(per-shell-pair screening inside the cell loop).")
