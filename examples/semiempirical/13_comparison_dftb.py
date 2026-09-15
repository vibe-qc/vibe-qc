"""Three-tier comparison: DFTB0 and SCC-DFTB on H2O.

Part of the vibe-qc semiempirical-vs-MLIP comparison study
(manuscript §5, SI §S4).  Runs DFTB0 and SCC-DFTB on water —
fastest native electronic route in vibe-qc.

Note: the shipped DFTB parameter set is an IN-HOUSE baseline
(91 elements), NOT a published DFTB parameter set with fitted
repulsive potentials. Energies are method-internal; do not
compare across method families.

Run:
    python examples/semiempirical/13_comparison_dftb.py

Produces:
    output-h2o-dftb0.out  / .bibtex  / .references  / .xyz
    output-h2o-scc.out    / .bibtex  / .references  / .xyz
    output-h2o-dftb0-opt.out  / .traj  / ...
    output-h2o-scc-opt.out    / .traj  / ...
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).resolve().parent
mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

# --- DFTB0: non-self-consistent, exact analytic gradient ---
run_job(mol, method="dftb0", output=HERE / "output-h2o-dftb0")
run_job(
    mol,
    method="dftb0",
    optimize=True,
    fmax=0.02,
    output=HERE / "output-h2o-dftb0-opt",
)

# --- SCC-DFTB: self-consistent charge, fixed-charge approximate gradient ---
run_job(mol, method="scc_dftb", output=HERE / "output-h2o-scc")
run_job(
    mol,
    method="scc_dftb",
    optimize=True,
    fmax=0.02,
    output=HERE / "output-h2o-scc-opt",
)

print("\nDone. Compare the .out files:")
print(f"  {HERE / 'output-h2o-dftb0.out'}")
print(f"  {HERE / 'output-h2o-scc.out'}")
print(f"  {HERE / 'output-h2o-dftb0-opt.out'}")
print(f"  {HERE / 'output-h2o-scc-opt.out'}")
