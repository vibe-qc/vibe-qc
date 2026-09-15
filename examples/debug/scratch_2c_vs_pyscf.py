"""Compare vibe-qc's compute_2c_eri_lattice against PySCF's
auxcell.pbc_intor('int2c2e') element-by-element.

Setup: MgO/sto-3g, aux=def2-universal-jkfit, modrho applied on both
sides. We need to align the basis-function orderings (libint within-
p-shell convention vs libcint, plus possibly across-shell ordering).

Question: does my modrho 2c kernel match PySCF's int2c2e at any
cutoff? If yes, the slice-3 path is "use the same per-shell
truncation as PySCF" rather than the full compcell algorithm. If no,
the kernels differ structurally and the lit-sweep findings on the
PySCF algorithm are needed.
"""
from __future__ import annotations

import numpy as np

from pyscf.pbc import df, gto

import vibeqc as vq
from vibeqc.aux_basis import make_aux_basis_set, modrho_renormalise
from vibeqc._vibeqc_core import BasisSet, ShellInfo

# --- MgO setup (matching PySCF + vibe-qc) ---
ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211
A = A_ANG * ANG2BOHR

mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]

# vibe-qc system
atoms_vq = [vq.Atom(12, [f * A for f in p]) for p in mg_frac] + \
           [vq.Atom(8, [f * A for f in p]) for p in o_frac]
system = vq.PeriodicSystem(3, np.diag([A, A, A]), atoms_vq)
mol_vq = system.unit_cell_molecule()

# PySCF cell (Å units)
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
auxcell = with_df.auxcell  # PySCF's modrho'd auxcell

# PySCF's 2c metric (modrho aux, periodic):
M_pyscf = np.asarray(auxcell.pbc_intor("int2c2e", hermi=1))
print(f"PySCF auxcell: nao={auxcell.nao}, ‖M‖_F = {float(np.linalg.norm(M_pyscf)):.4e}, "
      f"max|M| = {float(np.max(np.abs(M_pyscf))):.4e}")
eigs_p = np.linalg.eigvalsh(0.5 * (M_pyscf + M_pyscf.T))
print(f"  PySCF eig spectrum: min={eigs_p[0]:+.3e}, max={eigs_p[-1]:.3e}")

# vibe-qc auxcell (modrho)
aux_vq = make_aux_basis_set(mol_vq, aux_name="def2-universal-jkfit", compensate=True)
print(f"\nvibe-qc auxcell: nao={aux_vq.nbasis}")

# Try cutoffs
for cut in [4.0, 8.0, 16.0, float(auxcell.rcut) if hasattr(auxcell, 'rcut') else 25.0]:
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cut
    M_vq = np.asarray(vq.compute_2c_eri_lattice(aux_vq, system, opts))
    M_vq = 0.5 * (M_vq + M_vq.T)
    eigs_v = np.linalg.eigvalsh(M_vq)
    print(f"\n  vibe-qc cut={cut:5.2f}: ‖M‖_F={float(np.linalg.norm(M_vq)):.4e}  "
          f"max|M|={float(np.max(np.abs(M_vq))):.4e}  "
          f"min_eig={eigs_v[0]:+.3e}  max_eig={eigs_v[-1]:.3e}")
    # Naïve elementwise comparison (no AO-permutation alignment yet —
    # the absolute values won't match, but we can compare MAGNITUDES)
    ratio = float(np.linalg.norm(M_vq)) / float(np.linalg.norm(M_pyscf))
    print(f"     ‖M_vq‖ / ‖M_pyscf‖ = {ratio:.3f}")

# Check what auxcell.rcut is (PySCF's internal cutoff for this aux).
rcut_pyscf = float(auxcell.rcut) if hasattr(auxcell, "rcut") else None
print(f"\nPySCF auxcell.rcut = {rcut_pyscf}")
print("(This is the auto-set per-aux-cell rcut PySCF uses for pbc_intor.)")
