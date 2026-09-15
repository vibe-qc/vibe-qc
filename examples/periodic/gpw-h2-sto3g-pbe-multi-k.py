"""H2 in a 12-bohr cube — multi-k RKS / PBE / STO-3G via the GPW route (M3e).

Drives the multi-k GPW SCF directly through
:func:`vibeqc.run_periodic_rks_gpw_multi_k` on a 2×2×2 Monkhorst-Pack
mesh. The GPW J is built from the Γ-summed density on the FFT grid;
T, S, V_ne are Bloch-summed per k-point. Multi-k GPW is closed-shell
pure-DFT only at v0.10.x — hybrids and HF stay on the Γ-only path —
and is not yet exposed through ``run_periodic_job``.

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-pbe-multi-k.py
"""

import warnings
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

HERE = Path(__file__).resolve().parent
OUT = HERE / Path(__file__).stem

L = 12.0  # bohr
lattice = L * np.eye(3)
atoms = [
    vq.Atom(1, [-0.7, 0.0, 0.0]),
    vq.Atom(1, [+0.7, 0.0, 0.0]),
]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(system, [2, 2, 2])

result = vq.run_periodic_rks_gpw_multi_k(
    system,
    basis,
    kmesh,
    functional="pbe",
    cutoff_ha=300.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    quiet=True,
)

print(f"  E_total   = {result.energy:.8f} Ha")
print(f"  converged = {result.converged}   n_iter = {result.n_iter}")
print(f"  k-points  = {len(result.mo_energies_k)}")
print("  Per-k MO eigenvalues (Hartree):")
for ik, eps_k in enumerate(result.mo_energies_k):
    eps_str = "  ".join(f"{e:+.5f}" for e in np.asarray(eps_k).real)
    print(f"    k[{ik}]: {eps_str}")
print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   n_iter={result.n_iter}")
