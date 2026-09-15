"""Three-tier comparison: OM1, OM2, OM3 on H2O.

Part of the vibe-qc semiempirical-vs-MLIP comparison study
(manuscript §5, SI §S4).  Runs all three orthogonalization-corrected
NDDO variants (OM1/OM2/OM3) on water. The Dral-2016 parameter tables
(Kolb--Thiel 1993 for OM1; Weber--Thiel 2000 for OM2/OM3) are loaded
on H, C, N, O, F. Validation against published reference data is
still thin — energies are reported as approximate.

Run:
    python examples/semiempirical/17_comparison_omx.py

Produces:
    output-h2o-om1.out  / .bibtex  / .references  / .xyz
    output-h2o-om2.out  / .bibtex  / .references  / .xyz
    output-h2o-om3.out  / .bibtex  / .references  / .xyz
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).resolve().parent
mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

for variant in ("om1", "om2", "om3"):
    stem = HERE / f"output-h2o-{variant}"
    run_job(mol, method=variant, output=stem)
    print(f"OM{variant[-1]}: {stem}.out")
