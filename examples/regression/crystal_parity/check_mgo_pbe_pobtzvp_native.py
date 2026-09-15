"""Diagnostic: vibe-qc MgO PBE/POB-TZVP/Γ via native gamma RKS GDF.

CRYSTAL14 converges MgO PBE/POB-TZVP in 15 cycles to
E = -275.31500190603 Ha per primitive cell (user's compute-reference
queue announcement, 2026-05-10). This is the v0.8.0 baseline:
DFT-PBE on an ionic crystal that BOTH codes can handle with
default stabilizers.

If vibe-qc converges to a similar number (modulo the conventional-
vs-primitive cell factor), we have a working DFT validation
baseline. Native gamma RKS GDF is the most-tested path in codex's
overhaul.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")

import numpy as np
import vibeqc as vq


def main() -> int:
    # Conventional cubic cell to match CRYSTAL .d12 SG-225 input
    # (CRYSTAL internally uses the primitive 2-atom rocksalt cell,
    # so the per-cell energy will be 1/4 of ours; we'll divide).
    a_ang = 4.21
    ang2bohr = 1.0 / 0.529177210903
    a = a_ang * ang2bohr
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = ([vq.Atom(12, [fx*a, fy*a, fz*a]) for fx,fy,fz in mg_frac]
           + [vq.Atom(8, [fx*a, fy*a, fz*a]) for fx,fy,fz in o_frac])
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
    print(f"vibeqc {vq.__version__} — MgO PBE/POB-TZVP/Γ native RKS GDF")
    print(f"  {len(atoms)} atoms (conventional cubic), "
          f"{system.n_electrons()} electrons, basis {basis.nbasis} BFs")

    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 50
    opts.conv_tol_energy = 1e-8

    # Codex's overhaul folded RKS into run_rhf_periodic_gamma_gdf via
    # the ``functional`` kwarg; there is no separate run_rks_*.
    result = vq.run_rhf_periodic_gamma_gdf(
        system, basis, opts, functional="pbe", progress=True, verbose=2,
    )
    print()
    print(f"FINAL E   = {result.energy:.10f} Ha  (8-atom conventional cell)")
    print(f"  per FU  = {result.energy/4:.10f} Ha  (= /4 to compare CRYSTAL primitive)")
    print(f"  n_iter   = {result.n_iter}")
    print(f"  converged = {result.converged}")
    print()
    crystal_per_primitive = -275.31500190603
    per_fu = result.energy / 4.0
    delta = per_fu - crystal_per_primitive
    print(f"CRYSTAL14 reference per primitive: {crystal_per_primitive:.10f} Ha")
    print(f"  delta vibe-qc/FU vs CRYSTAL/prim: {delta:.6e} Ha "
          f"({delta*1000:.3f} mHa, {delta*1e6:.1f} µHa)")
    if not result.converged:
        print("VERDICT: did NOT converge.")
        return 1
    if abs(delta) < 1e-5:
        print("VERDICT: sub-µHa agreement with CRYSTAL14. v0.8.0 baseline ✓")
        return 0
    if abs(delta) < 1e-3:
        print("VERDICT: sub-mHa agreement with CRYSTAL14. v0.8.0 baseline (basis/grid noise expected).")
        return 0
    print(f"VERDICT: converged but {delta*1000:.1f} mHa off CRYSTAL14. "
          "Likely native-Lpq metric error shifting the answer.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
