"""Diagnostic: re-run the MgO ``MgO-rocksalt-RHF-sto3g.py``
recipe via ``run_rhf_periodic_gamma_gdf`` and report energy +
convergence.

This is the bug-confirmation companion to the CRYSTAL14 input:
historical runs documented E = −1085.23 Ha in 9 SCF iters with plain
DIIS (no FMIXING, no level shift). If a fresh run on compute-host-d reproduces
that, the local-Mac divergence is a build issue. If a fresh run on
compute-host-d also diverges, the codebase has regressed from the sealed
reference value hard-coded below.

Run via vq:

    vq submit examples/regression/crystal_parity/check_mgo_gamma_rhf_native.py
"""
from __future__ import annotations

import os
import sys

# Be explicit about /tmp on compute-host-d (tmpfs); PySCF reference
# scratch would blow it up, but we're native-only here so it
# matters less. Harmless to set.
os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")

import numpy as np
import vibeqc as vq


def main() -> int:
    print(f"vibeqc {vq.__version__}")
    print(f"vibeqc.__file__: {vq.__file__}")
    print(f"_vibeqc_core: {vq._vibeqc_core.__file__}")
    print()

    a_ang = 4.211
    ang2bohr = 1.0 / 0.529177210903
    a = a_ang * ang2bohr
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = []
    for fx, fy, fz in mg_frac:
        atoms.append(vq.Atom(12, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        atoms.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    print(f"MgO rocksalt, a = {a_ang} Å, "
          f"{len(atoms)} atoms, {system.n_electrons()} electrons")
    print(f"basis: sto-3g, {basis.nbasis} basis functions")
    print()

    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-8
    print("SCF options: DIIS on, damping 0, no FMIXING, no level_shift")
    print(f"  (matches the committed examples/periodic/MgO-rocksalt/"
          f"MgO-rocksalt-RHF-sto3g.py recipe)")
    print()

    result = vq.run_rhf_periodic_gamma_gdf(
        system, basis, opts, progress=True, verbose=2,
    )
    print()
    print(f"FINAL E    = {result.energy:.10f} Ha")
    print(f"  n_iter   = {result.n_iter}")
    print(f"  converged = {result.converged}")
    print()
    print(f"Expected (sealed reference): -1085.2315464583 Ha, 9 iters, "
          f"converged=True")
    delta = float(result.energy) - (-1085.2315464583)
    print(f"  delta vs committed: {delta:.3e} Ha")
    if not result.converged:
        print()
        print("VERDICT: compute-host-d run did NOT converge — codebase regression "
              "since the committed .out was produced.")
        return 1
    if abs(delta) > 1e-6:
        print()
        print(f"VERDICT: compute-host-d run converged but to a different energy "
              f"({delta:.3e} Ha off). Either the sealed reference is stale "
              f"or there's a precision regression.")
        return 2
    print()
    print("VERDICT: compute-host-d run matches the sealed reference — local-Mac "
          "build is the source of the divergence I saw, NOT the codebase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
