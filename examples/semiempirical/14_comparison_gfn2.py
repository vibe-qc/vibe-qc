"""Three-tier comparison: GFN2-xTB on H2O.

Part of the vibe-qc semiempirical-vs-MLIP comparison study
(manuscript §5, SI §S4).  Runs GFN2-xTB on water — the most
ambitious native fast method in the stack, with AES dipole+quadrupole,
GAM3 third-order, and post-SCF native D4 dispersion.

Note: GFN2-xTB remains under EXPERIMENTAL gate. The published
Grimme-group parameter set is fetched on demand at first use
(LGPL-3.0 → cached at ~/.cache/vibeqc/). GFN2-xTB has not yet
closed the external parity matrix against the xtb reference.

Run:
    python examples/semiempirical/14_comparison_gfn2.py

Produces:
    output-h2o-gfn2.out  / .bibtex  / .references  / .xyz
    output-h2o-gfn2-opt.out  / .traj  / ...
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).resolve().parent
mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

# Single-point energy (default Broyden mixer, post-SCF D4).
# On first run this fetches the GFN2-xTB parameter set from upstream.
run_job(mol, method="gfn2_xtb", output=HERE / "output-h2o-gfn2")

# Geometry optimization.
run_job(
    mol,
    method="gfn2_xtb",
    optimize=True,
    fmax=0.02,
    output=HERE / "output-h2o-gfn2-opt",
)

print(f"\nGFN2-xTB energy: see {HERE / 'output-h2o-gfn2.out'}")
print(f"GFN2-xTB optimized geometry: see {HERE / 'output-h2o-gfn2-opt.out'}")
