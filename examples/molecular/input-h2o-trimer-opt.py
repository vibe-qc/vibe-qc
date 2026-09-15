"""Geometry optimization of the water trimer (cyclic) at HF/6-31G*.

Run:
    .venv/bin/python input-h2o-trimer-opt.py

Produces:
    output-h2o-trimer-opt.out     — SCF + optimizer log, final energy
    output-h2o-trimer-opt.molden  — MOs at the minimum
    output-h2o-trimer-opt.traj    — trajectory frames for animation

Starting geometry: approximate cyclic "uuu" arrangement with three
water oxygens at the vertices of an equilateral triangle (R_OO =
2.9 Å) and each water donating an H around the ring. BFGS relaxes
to the nearest stationary point — typically the "uud" ring minimum
once the free H's tilt asymmetrically.

Runtime: ~30 s to a few minutes depending on optimizer path.
"""

from pathlib import Path

from vibeqc import InitialGuess, Molecule, RHFOptions, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o_trimer.xyz")

# H-bonded clusters can land on a hard-to-converge geometry mid-BFGS;
# bump SCF max_iter to keep the optimization robust.
rhf_opts = RHFOptions()
rhf_opts.max_iter = 200
rhf_opts.initial_guess = InitialGuess.SAD

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o-trimer-opt",
    optimize=True,
    # H-bonded PESs are very flat — fmax=0.2 eV/A is realistic at the
    # default SCF tolerance. Tighten rhf_opts.conv_tol_grad if you want
    # fmax < 0.05.
    fmax=0.2,
    max_opt_steps=40,
    rhf_options=rhf_opts,
)
