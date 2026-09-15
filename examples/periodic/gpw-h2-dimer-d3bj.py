"""Two H2 molecules in a 20-bohr cube — PBE-D3(BJ) via the GPW route.

Demonstrates Grimme's D3 dispersion (Becke-Johnson damping) on the GPW
RKS path: ``run_periodic_rhf_gpw(functional="pbe", dispersion="d3-bj",
dispersion_functional="pbe")``. The D3-BJ correction is summed into
``result.energy`` and exposed on the breakdown as ``e_dispersion``.

Two well-separated H2 molecules (~6 bohr centre-to-centre) give a small
but non-zero *intermolecular* dispersion attraction — unlike a single H2,
where the intramolecular D3 term is ~0. Requires the ``[dispersion]``
extra (``dftd3``); the example skips gracefully if it is absent.

NOTE: the Γ-only RKS GPW entry is ``run_periodic_rhf_gpw`` with
``functional=`` (there is no ``run_periodic_rks_gpw``; see
``docs/user_guide/gapw.md``).

Run:
    .venv/bin/python examples/periodic/gpw-h2-dimer-d3bj.py
"""

import warnings

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

L = 20.0  # bohr — vacuum-padded
c = L / 2
# Two H2 molecules along x, centres at x = c-3 and c+3 (6 bohr apart).
atoms = [
    vq.Atom(1, [c - 3 - 0.7, c, c]), vq.Atom(1, [c - 3 + 0.7, c, c]),
    vq.Atom(1, [c + 3 - 0.7, c, c]), vq.Atom(1, [c + 3 + 0.7, c, c]),
]
system = vq.PeriodicSystem(3, L * np.eye(3), atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

try:
    result = vq.run_periodic_rhf_gpw(
        system, basis,
        functional="pbe",
        dispersion="d3-bj",
        dispersion_functional="pbe",
        cutoff_ha=300.0,
        max_iter=60,
        conv_tol_energy=1e-7,
    )
except NotImplementedError as exc:
    print(f"  dispersion unavailable (skipping): {exc}")
    raise SystemExit(0)

e_disp = getattr(result.breakdown, "e_dispersion", None)
print(f"  E_total (PBE-D3BJ) = {result.energy:.8f} Ha   converged={result.converged}")
if e_disp is not None:
    print(f"  e_dispersion       = {e_disp:+.8f} Ha   (additive D3-BJ correction)")
    assert e_disp < 0.0, "intermolecular dispersion should be attractive (< 0)"
    print("  dispersion is attractive (< 0) ✓")
