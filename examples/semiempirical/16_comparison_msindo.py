"""Three-tier comparison: MSINDO on H2O and H2S.

Part of the vibe-qc semiempirical-vs-MLIP comparison study
(manuscript §5, SI §S4).  Runs MSINDO — the strongest production
semiempirical path within its validated s/p/d scope (9 elements:
H, He, C, N, O, F, Si, P, S).  Oracle parity against the
reference MSINDO Fortran closed within documented accuracy.

H2S is the canonical d-shell test for MSINDO (S carries a 3d
polarisation shell + Ne frozen core).

Run:
    python examples/semiempirical/16_comparison_msindo.py

Produces:
    output-h2o-msindo.out  / .bibtex  / .references  / .xyz
    output-h2s-msindo.out  / .bibtex  / .references  / .xyz
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).resolve().parent

# H2O: canonical small-molecule closed-shell test.
mol_h2o = Molecule.from_xyz(HERE.parent / "h2o.xyz")
run_job(mol_h2o, method="msindo", output=HERE / "output-h2o-msindo")
print(f"H2O: {HERE / 'output-h2o-msindo.out'}")

# H2S: third-row d-shell test (S 3d polarisation + Ne frozen core).
mol_h2s = Molecule.from_xyz(HERE.parent / "h2s.xyz")
run_job(mol_h2s, method="msindo", output=HERE / "output-h2s-msindo")
print(f"H2S: {HERE / 'output-h2s-msindo.out'}")
