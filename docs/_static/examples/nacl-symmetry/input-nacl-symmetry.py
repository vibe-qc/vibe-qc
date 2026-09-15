"""Space-group symmetry infrastructure on NaCl.

Run:
    .venv/bin/python input-nacl-symmetry.py

Demonstrates the Phase SYM machinery that landed in v0.2.5: Wigner-D
matrices for real-basis AO rotation (SYM1), lattice-cell orbit
identification plus LatticeMatrixSet compression for origin-fixed
structures (SYM2b), and atom-pair-resolved orbits for cells with more
than one atom (SYM2c). The SCF-side integration (SYM3) that turns
these orbits into wall-clock savings hasn't landed yet, so this
script showcases the raw compression ratios and the exact
round-trip property.

Three demos on the same NaCl primitive cubic cell:

1. Attach the Pm-3m (48-operation) space group to the PeriodicSystem
   via ``attach_symmetry`` and print a few rotations.
2. Build the overlap LatticeMatrixSet with a 20 bohr cutoff, then
   compress-and-reconstruct it via the atom-pair-orbit path (SYM2c).
   Report the compression ratio and the max block error.
3. Re-do the simpler single-atom cubic cell (Mg) with the
   origin-fixed path (SYM2b) for comparison — the orbit count per
   cell should match SYM2c up to the atom-pair multiplicity.

And one direct showcase of the underlying representation layer:

4. Pick a 90°-rotation operation and print its real-basis Wigner-D
   matrix at l=2 (d-orbitals). The 5×5 block shows the non-trivial
   mixing among (xy, yz, z², xz, x²-y²); orthogonality
   ``D Dᵀ = I`` holds to machine precision.
"""

import numpy as np

import vibeqc as vq


def banner(title: str) -> None:
    bar = "-" * 68
    print(f"\n{bar}\n  {title}\n{bar}")


# ---- 1. NaCl primitive cubic with Pm-3m -------------------------------
A = 4.0
sys_ = vq.PeriodicSystem(
    3, np.eye(3) * A,
    [vq.Atom(11, [0, 0, 0]),          # Na
     vq.Atom(17, [A / 2, A / 2, A / 2])],   # Cl
)
vq.attach_symmetry(sys_)

banner(f"NaCl Pm-3m (a = {A} bohr) — {len(sys_.symmetry.operations)} operations")
print("  First three rotations (fractional basis):")
for i in range(3):
    r = sys_.symmetry.operations[i].rotation
    print(f"    op {i}:  {r.tolist()}")

# ---- 2. SYM2c: atom-pair orbits + lattice-matrix compression ----------
basis = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
opts = vq.LatticeSumOptions()
opts.cutoff_bohr = 20.0

S = vq.compute_overlap_lattice(basis, sys_, opts)
banner(f"SYM2c — overlap LatticeMatrixSet ({len(S.cells)} cells, "
       f"{basis.nbasis} AOs)")

pair_orbits = vq.identify_atom_pair_orbits(
    S.cells, sys_.symmetry.operations, sys_, require_closed=False,
)
print(f"  n_triples        = {pair_orbits.n_triples}    "
      f"(cells × atom-pairs)")
print(f"  n_orbits         = {pair_orbits.n_orbits}")
print(f"  compression      = {pair_orbits.compression_ratio:.2f}×")

# Orbit-size distribution
sizes = [o.size for o in pair_orbits.orbits]
print(f"  orbit sizes      = min {min(sizes)}, max {max(sizes)}, "
      f"mean {np.mean(sizes):.1f}")

reps = vq.compress_lattice_matrix_set_c(S, pair_orbits, basis)
rec = vq.reconstruct_lattice_matrix_set_c(
    reps, pair_orbits, basis, sys_, sys_.symmetry.operations, cells=S.cells,
)
err = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
          for a, b in zip(rec, S.blocks))
print(f"  max block error  = {err:.3e}   "
      f"(S = reconstruct(compress(S)) to machine precision)")

# ---- 3. SYM2b: simpler origin-fixed lattice-cell orbits ---------------
sys_mg = vq.PeriodicSystem(
    3, np.eye(3) * A, [vq.Atom(12, [0, 0, 0])],     # Mg at origin
)
vq.attach_symmetry(sys_mg)
basis_mg = vq.BasisSet(sys_mg.unit_cell_molecule(), "sto-3g")
S_mg = vq.compute_overlap_lattice(basis_mg, sys_mg, opts)
lattice_orbits = vq.identify_lattice_orbits(
    S_mg.cells, sys_mg.symmetry.operations, require_closed=False,
)
banner("SYM2b — single-atom Mg primitive cubic (origin-fixed path)")
print(f"  n_cells          = {len(S_mg.cells)}")
print(f"  n_orbits         = {lattice_orbits.n_orbits}")
print(f"  compression      = {lattice_orbits.compression_ratio:.2f}×")

reps_mg = vq.compress_lattice_matrix_set(S_mg, lattice_orbits)
rec_mg = vq.reconstruct_lattice_matrix_set(
    reps_mg, lattice_orbits, basis_mg, sys_mg, sys_mg.symmetry.operations,
)
err_mg = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
             for a, b in zip(rec_mg, S_mg.blocks))
print(f"  max block error  = {err_mg:.3e}")
print(f"  (SYM2b's compression equals SYM2c on this cell because the "
      f"single-atom primitive has only the trivial atom-pair.)")

# ---- 4. Wigner-D direct showcase --------------------------------------
banner("SYM1 — real-basis Wigner-D at l=2, 90°-about-z rotation")
op = sys_.symmetry.operations[2]         # 90°-about-z for this setup
R_cart = vq.lattice_to_cartesian_rotation(
    op.rotation.astype(float), sys_.lattice,
)
print("  Cartesian rotation matrix:")
for row in R_cart:
    print("    [" + "  ".join(f"{x:+.2f}" for x in row) + "]")
D = vq.wigner_d_real(l=2, R=R_cart)
print("  D (real d-basis, rows/cols in order d_{-2}, d_{-1}, d_0, d_{+1}, d_{+2}):")
for row in D:
    print("    [" + "  ".join(f"{x:+.4f}" for x in row) + "]")
ortho_err = float(np.max(np.abs(D @ D.T - np.eye(5))))
print(f"  orthogonality ‖D Dᵀ − I‖_∞ = {ortho_err:.3e}")
