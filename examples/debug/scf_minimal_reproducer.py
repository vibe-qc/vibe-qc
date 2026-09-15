"""Smallest-possible periodic-RKS reproducer — single He atom, cubic.

If your debug session needs the absolute minimum periodic-DFT
calculation that exercises EWALD_3D + RKS-LDA, this is it:

  - 1 atom per cell (He)
  - sto-3g basis → 1 basis function per cell
  - cubic 3 Å cell (orthorhombic, EWALD_3D legal)
  - Γ-only k-mesh → 1 k-point
  - RKS / LDA (Slater + VWN5)
  - max_iter = 20

Wall: ≈30 s on a laptop with v0.5.5 Schwarz screening on. If THIS
calculation diverges or hangs, the bug is upstream of any
chemistry — it's in the Fock build, density mixing, or DIIS
machinery itself.

Use case: bisecting whether a bug is system-size-dependent. If
this minimum case reproduces a divergence the other examples
also show, work it here — it'll iterate fastest.

Run:
    .venv/bin/python examples/debug/scf_minimal_reproducer.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "minimal-he"
OUT.parent.mkdir(exist_ok=True)

# Single He atom in a 3 Å cubic cell — closed-shell, smallest
# possible 3D periodic SCF. EWALD_3D requires orthorhombic, which
# the cubic cell trivially satisfies.
A_BOHR = 3.0 / 0.529177210903
system = vq.PeriodicSystem(
    dim=3,
    lattice=A_BOHR * np.eye(3),
    unit_cell=[vq.Atom(2, [0.0, 0.0, 0.0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
print(f"He cubic: 1 atom, {basis.nbasis} bf per cell, a = 3.0 Å")

opts = vq.PeriodicKSOptions()
opts.functional = "LDA"
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr = 8.0           # tight; cell is small
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.conv_tol_energy = 1e-7
opts.max_iter = 20

plog = ProgressLogger(log_path=str(OUT) + ".out", verbose=4)

t0 = time.perf_counter()
with vq.crash_dump_context(OUT):
    with vq.perf_log(str(OUT) + ".perf"):
        result = vq.run_rks_periodic_scf(
            system, basis, vq.KPoints.gamma(system),
            opts, progress=plog, spacing_bohr=0.4,
        )
wall = time.perf_counter() - t0

print()
print(f"E      = {result.energy:.10f} Ha")
print(f"iters  = {result.n_iter} ({'converged' if result.converged else 'NOT'})")
print(f"wall   = {wall:.1f} s")
print()
print(f".out:  {OUT}.out")
print(f".perf: {OUT}.perf")
print(f".dump: {OUT}.dump  (if SCF failed)")
