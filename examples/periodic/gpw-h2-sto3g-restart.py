"""H2 in a 16-bohr cube — save / load / describe a GPW SCF result.

Demonstrates the v0.12-prep restart format for the GPW route:

* :func:`vibeqc.save_gpw_result` — write a converged ``GpwScfResult`` (energy,
  density, MO coefficients + energies, per-term breakdown, lattice / atoms /
  basis) to a single compressed ``.npz`` archive.
* :func:`vibeqc.load_gpw_result` — read it back into a plain ``dict``.
* :func:`vibeqc.describe_gpw_result` — a short human-readable summary.

Useful any time you want to avoid re-running the SCF — band paths / DOS
sweeps against a converged density, sharing a result, or resuming analysis.

NOTE: the Γ-only RKS GPW entry is ``run_periodic_rhf_gpw`` with a
``functional=`` argument (there is no ``run_periodic_rks_gpw``; see
``docs/user_guide/gapw.md``).

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-restart.py
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

L = 16.0  # bohr — vacuum padding
lattice = L * np.eye(3)
atoms = [
    vq.Atom(1, [-0.7, 0.0, 0.0]),
    vq.Atom(1, [+0.7, 0.0, 0.0]),
]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_rhf_gpw(
    system, basis, functional="lda", cutoff_ha=300.0,
    max_iter=50, conv_tol_energy=1e-8,
)
print(f"  SCF       E = {result.energy:.8f} Ha   converged={result.converged}")

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "h2_gpw.npz"
    written = vq.save_gpw_result(path, result, basis, system)
    print(f"  saved     {written.name} ({written.stat().st_size} bytes)")
    print("  describe  " + vq.describe_gpw_result(written).replace("\n", "\n            "))

    # Resume in-process (in practice: a fresh process days later).
    data = vq.load_gpw_result(written)
    print(f"  loaded    E = {data['energy']:.8f} Ha   density {data['density'].shape}")
    assert abs(data["energy"] - result.energy) < 1e-12, "restart energy drift"
    print("  round-trip energy matches to < 1e-12 Ha ✓")
