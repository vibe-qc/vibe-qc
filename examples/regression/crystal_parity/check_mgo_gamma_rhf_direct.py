"""Diagnostic: bypass the GDF path entirely; run MgO/sto-3g/Γ RHF
via the C++ direct-lattice-sum ``run_rhf_periodic_gamma``.

If this converges to ≈ -1085.23 Ha, we have a working "no fitting"
foundation (the user's prerequisite to layering DF and RIJCOSX on
top). The native GDF metric-error regression then becomes a
follow-up rather than a v0.8.0 blocker.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")

import numpy as np
import vibeqc as vq


def main() -> int:
    print(f"vibeqc {vq.__version__}")
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
    print(f"MgO rocksalt conventional cubic, "
          f"{len(atoms)} atoms, {system.n_electrons()} electrons, "
          f"basis sto-3g ({basis.nbasis} BFs)")
    print()

    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-8
    # Explicit DIRECT (no fitting) — bypass GDF entirely.
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.DIRECT_TRUNCATED

    print("Coulomb method: DIRECT_TRUNCATED (no fitting; bare libint)")
    print(f"cutoff_bohr: {opts.lattice_opts.cutoff_bohr}")
    print(f"nuclear_cutoff_bohr: {opts.lattice_opts.nuclear_cutoff_bohr}")
    print()

    result = vq.run_rhf_periodic_gamma(system, basis, opts)
    print()
    print(f"FINAL E    = {result.energy:.10f} Ha")
    print(f"  n_iter   = {result.n_iter}")
    print(f"  converged = {result.converged}")
    print()
    print(f"Expected (committed .out via gamma GDF):"
          f" -1085.2315464583 Ha, 9 iters, converged=True")
    delta = float(result.energy) - (-1085.2315464583)
    print(f"  delta vs committed (GDF)   : {delta:.3e} Ha")
    if result.converged:
        if abs(delta) < 1e-3:
            print("VERDICT: standard (no fitting) converges to the GDF "
                  "reference. We have a working baseline.")
        else:
            print(f"VERDICT: standard converges but to a different energy "
                  f"({delta:.3e} Ha off GDF). Either GDF was wrong, or "
                  f"DIRECT_TRUNCATED is in a different gauge.")
        return 0
    print("VERDICT: standard did NOT converge. Independent of GDF.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
