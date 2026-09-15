"""Three-tier comparison: PM6 (NDDO) on H2O and H2.

Part of the vibe-qc semiempirical-vs-MLIP comparison study
(manuscript §5, SI §S4).  Runs PM6 on water and hydrogen —
MOPAC parameter set (Apache 2.0), 75 chemical elements, 933 diatomic-
pair core-core corrections. Finite-difference gradient.

Run:
    python examples/semiempirical/15_comparison_pm6.py

Produces:
    output-h2o-pm6.out  / .bibtex  / .references  / .xyz
    output-h2-pm6.out   / .bibtex  / .references  / .xyz
    output-h2o-pm6-opt.out  / .traj  / ...
    output-h2-pm6-opt.out   / .traj  / ...
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).resolve().parent

for name in ("h2o", "h2"):
    if name == "h2":
        mol = Molecule.from_xyz(HERE.parent / "h2.xyz")
    else:
        mol = Molecule.from_xyz(HERE.parent / f"{name}.xyz")
    stem = HERE / f"output-{name}-pm6"
    opt_stem = HERE / f"output-{name}-pm6-opt"

    run_job(mol, method="pm6", output=stem)
    run_job(mol, method="pm6", optimize=True, fmax=0.02, output=opt_stem)

    print(f"{name}: single-point → {stem}.out")
    print(f"{name}: optimized  → {opt_stem}.out")
