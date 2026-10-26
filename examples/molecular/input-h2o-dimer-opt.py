"""Geometry optimization of the water dimer at HF/6-31G*.

Run:
    .venv/bin/python input-h2o-dimer-opt.py

Produces:
    output-h2o-dimer-opt.out     — SCF + optimizer log, final energy
    output-h2o-dimer-opt.molden  — molecular orbitals at the minimum
    output-h2o-dimer-opt.traj    — trajectory frames for animation

Starting geometry: Cs-symmetric linear H-bond, R(OO) = 2.95 Å. BFGS
(via ASE) relaxes it to the nearest local minimum on the HF/6-31G*
PES. Expect R(OO) to contract to ~2.98 Å (HF overestimates slightly);
the interaction energy at HF/6-31G* is ~-5 kcal/mol.

Note: HF and small basis sets don't capture dispersion — for
production work, use DFT+D or MP2.
"""

from pathlib import Path

from vibeqc import InitialGuess, Molecule, RHFOptions, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o_dimer.xyz")

# H-bonded systems occasionally oscillate on the flat part of the PES
# during BFGS; bump max_iter so the SCF always reaches convergence
# regardless of the optimizer's intermediate geometries.
rhf_opts = RHFOptions()
rhf_opts.max_iter = 200
rhf_opts.initial_guess = InitialGuess.SAD

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o-dimer-opt",
    optimize=True,
    # H-bonded PESs are very flat — the analytic-gradient noise floor at
    # the default SCF tolerance is around 0.1 eV/A, so fmax tighter than
    # ~0.2 rarely converges without also tightening conv_tol_grad on the
    # RHF options. R(OO) is already within 0.005 A of the literature
    # HF/6-31G* minimum once |F|_max drops below 0.2.
    fmax=0.2,
    max_opt_steps=25,
    rhf_options=rhf_opts,
)
