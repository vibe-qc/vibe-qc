"""Symmetry-reduced periodic calculations — end-to-end demo.

Walks through every symmetry feature available in vibe-qc v0.8.x:

  1. ``symmetry="auto"`` — one-keyword symmetry enable
  2. ``reduce_to_primitive=True`` — explicit cell reduction
  3. Manual ``attach_symmetry`` + compute-reduced S/T integrals
  4. Symmetrization operators (``vibeqc.symmetry_scf``)
  5. K-point mesh reduction via ``symmetry_reduce()``

Uses NaCl rocksalt as the running example (Fm-3m, |G|=48).
Run with::

    .venv/bin/python examples/periodic/input-symmetry-demo.py
"""

from __future__ import annotations

import time

import numpy as np
import vibeqc as vq
from vibeqc.symmetry_integrals import symmorphic_operations
from vibeqc.symmetry_integrals_reduced import (
    compression_summary,
    compute_kinetic_lattice_reduced,
    compute_overlap_lattice_reduced,
)
from vibeqc.symmetry_scf import (
    build_ao_permutation_cache,
    symmetrize_matrix,
)


def banner(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


# ---------------------------------------------------------------------------
# Build the test system: NaCl primitive (conventional 8-atom cell)
# ---------------------------------------------------------------------------

A_ANG = 5.64  # lattice constant in Å
A_BOHR = A_ANG / 0.529177210903

# Conventional rocksalt: 8 atoms
atoms_conv = [
    vq.Atom(11, [0, 0, 0]),
    vq.Atom(11, [A_BOHR / 2, A_BOHR / 2, 0]),
    vq.Atom(11, [A_BOHR / 2, 0, A_BOHR / 2]),
    vq.Atom(11, [0, A_BOHR / 2, A_BOHR / 2]),
    vq.Atom(17, [A_BOHR / 2, 0, 0]),
    vq.Atom(17, [0, A_BOHR / 2, 0]),
    vq.Atom(17, [0, 0, A_BOHR / 2]),
    vq.Atom(17, [A_BOHR / 2, A_BOHR / 2, A_BOHR / 2]),
]
sys_conv = vq.PeriodicSystem(3, np.eye(3) * A_BOHR, atoms_conv)
basis_conv = vq.BasisSet(sys_conv.unit_cell_molecule(), "sto-3g")

print(
    f"Conventional cell: {len(sys_conv.unit_cell)} atoms, "
    f"{basis_conv.nbasis} basis functions"
)

# ---------------------------------------------------------------------------
# 1. symmetry="auto" — one keyword
# ---------------------------------------------------------------------------

banner("1. symmetry='auto' — auto-attach + reduce + reduced S/T")
t0 = time.perf_counter()
r_auto = vq.run_periodic_job(
    sys_conv,
    basis_conv,
    method="RHF",
    jk_method="gdf",
    output="/tmp/symmetry_demo_auto",
    symmetry="auto",
    max_iter=15,
    conv_tol_energy=1e-5,
    write_molden_file=False,
    write_xyz_file=False,
    write_poscar_file=False,
    write_xsf_structure_file=False,
    write_cif_file=False,
    write_population_file=False,
    citations=False,
)
t_auto = time.perf_counter() - t0
print(f"  Energy: {r_auto.energy:.10f} Ha  ({r_auto.n_iter} iters, {t_auto:.1f}s)")

# ---------------------------------------------------------------------------
# 2. Manual control — attach_symmetry + reduce_to_primitive
# ---------------------------------------------------------------------------

banner("2. Manual: attach_symmetry + reduce_to_primitive")
sys2 = vq.PeriodicSystem(3, np.eye(3) * A_BOHR, atoms_conv)
vq.attach_symmetry(sys2, symprec=1e-4)
sg = sys2.symmetry
print(f"  Space group: {sg.international_symbol} (No. {sg.number})")
print(f"  Point group: {sg.point_group}, order: {sg.order}")
print(f"  Equivalent atoms: {list(sg.equivalent_atoms)}")

t0 = time.perf_counter()
r2 = vq.run_periodic_job(
    sys2,
    basis_conv,
    method="RHF",
    jk_method="gdf",
    output="/tmp/symmetry_demo_manual",
    reduce_to_primitive=True,
    max_iter=15,
    conv_tol_energy=1e-5,
    write_molden_file=False,
    write_xyz_file=False,
    write_poscar_file=False,
    write_xsf_structure_file=False,
    write_cif_file=False,
    write_population_file=False,
    citations=False,
)
t2 = time.perf_counter() - t0
print(f"  Energy: {r2.energy:.10f} Ha  ({r2.n_iter} iters, {t2:.1f}s)")
assert abs(r_auto.energy - r2.energy) < 1e-8, "Auto vs manual mismatch!"

# ---------------------------------------------------------------------------
# 3. Compute-reduced S/T integrals manually
# ---------------------------------------------------------------------------

banner("3. Compute-reduced S/T integrals (M2)")
sys3 = vq.PeriodicSystem(
    3,
    np.eye(3) * A_BOHR,
    [vq.Atom(11, [0, 0, 0]), vq.Atom(17, [A_BOHR / 2, A_BOHR / 2, A_BOHR / 2])],
)
vq.attach_symmetry(sys3)
basis3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
opts3 = vq.LatticeSumOptions()
opts3.cutoff_bohr = 12.0

S_full = vq.compute_overlap_lattice(basis3, sys3, opts3)
S_red, S_rec = compute_overlap_lattice_reduced(
    basis3,
    sys3,
    opts3,
    sys3.symmetry.operations,
)
summary = compression_summary(S_red.orbits)
err_S = max(
    float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    for a, b in zip(S_full.blocks, S_rec)
)
print(
    f"  Cells: {summary['n_cells_full']} full → "
    f"{summary['n_cells_reduced']} reduced "
    f"({summary['compression_cells']:.1f}×)"
)
print(f"  S round-trip error: {err_S:.2e}")
assert err_S < 1e-12

# ---------------------------------------------------------------------------
# 4. Symmetrization operators (M5)
# ---------------------------------------------------------------------------

banner("4. Symmetrization (group averaging)")
ops4 = symmorphic_operations(sys3.symmetry.operations)
P_cache = build_ao_permutation_cache(sys3, basis3, ops4)
print(f"  Symmorphic operators: {len(ops4)}")

# Build a random matrix and symmetrize
rng = np.random.default_rng(123)
M = rng.normal(size=(basis3.nbasis, basis3.nbasis))
M_sym = symmetrize_matrix(M, P_cache)
M_sym2 = symmetrize_matrix(M_sym, P_cache)
idem_err = float(np.linalg.norm(M_sym - M_sym2))
print(f"  Symmetrization idempotent: ||M_sym - M_sym2|| = {idem_err:.2e}")
assert idem_err < 1e-12

# Verify overlap is invariant (single-atom Mg, all atoms origin-fixed)
sys_mg = vq.PeriodicSystem(3, np.eye(3) * 5.0, [vq.Atom(12, [0, 0, 0])])
vq.attach_symmetry(sys_mg)
basis_mg = vq.BasisSet(sys_mg.unit_cell_molecule(), "sto-3g")
ops_mg = symmorphic_operations(sys_mg.symmetry.operations)
P_mg = build_ao_permutation_cache(sys_mg, basis_mg, ops_mg)
S_mg = np.asarray(vq.compute_overlap(basis_mg))
S_mg_sym = symmetrize_matrix(S_mg, P_mg)
inv_err = float(np.linalg.norm(S_mg - S_mg_sym))
print(f"  Overlap invariance (Mg): ||S - S_sym|| = {inv_err:.2e}")
assert inv_err < 1e-12

# ---------------------------------------------------------------------------
# 5. K-point symmetry reduction
# ---------------------------------------------------------------------------

banner("5. K-point mesh symmetry reduction")
kp_full = vq.KPoints.monkhorst_pack(sys3, [4, 4, 4], symmetry=False)
kp_ibz = vq.KPoints.monkhorst_pack(sys3, [4, 4, 4], symmetry=True)
print(f"  Full mesh: {kp_full.n_kpoints} k-points")
print(f"  IBZ mesh:  {kp_ibz.n_kpoints} k-points")
print(f"  Reduction: {kp_full.n_kpoints / kp_ibz.n_kpoints:.1f}×")
if kp_ibz.ir_mapping.size > 0:
    # Unique k-point count from ir_mapping
    n_unique = len(set(int(x) for x in kp_ibz.ir_mapping))
    print(f"  Unique representatives: {n_unique}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

banner("All symmetry features verified")
print("""
Summary of available features:
  run_periodic_job(system, basis, symmetry=True)       — one-keyword enable
  run_periodic_job(system, basis, symmetry='attach')   — attach only
  run_periodic_job(system, basis, reduce_to_primitive=True)  — explicit
  compute_overlap_lattice_reduced(...)                  — M2 compute reduction
  compute_kinetic_lattice_reduced(...)
  symmetrize_matrix(M, P_cache)                         — M5 group averaging
  KPoints.monkhorst_pack(system, mesh, symmetry=True)   — IBZ k-points
""")
