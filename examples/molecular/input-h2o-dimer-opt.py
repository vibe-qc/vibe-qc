"""Geometry optimization of the water dimer at HF/6-31G*.

Run:
    .venv/bin/python input-h2o-dimer-opt.py

Produces:
    output-h2o-dimer-opt.out     — SCF + optimizer log, final energy
    output-h2o-dimer-opt.molden  — molecular orbitals at the minimum
    output-h2o-dimer-opt.traj    — trajectory frames for animation

Starting geometry: Cs-symmetric linear H-bond, R(OO) = 2.95 Å. BFGS
(via ASE) searches for a stationary geometry on the HF/6-31G* PES. The loose force target below demonstrates the workflow; check the
final forces and convergence before interpreting a distance or interaction
energy. Tighten the geometry and SCF thresholds for quantitative work.

Note: HF and small basis sets don't capture dispersion — for
production work, use DFT+D or MP2.
"""

from pathlib import Path

from vibeqc import InitialGuess, Molecule, RHFOptions, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o_dimer.xyz")

# H-bonded systems occasionally oscillate on the flat part of the PES
# during BFGS. A larger SCF iteration budget can help a progressing solve,
# but does not guarantee convergence at every trial geometry.
rhf_opts = RHFOptions()
rhf_opts.max_iter = 200
rhf_opts.initial_guess = InitialGuess.SAD

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o-dimer-opt",
    optimize=True,
    optimizer_backend="ase",  # the documented .traj is an ASE trajectory
    write_molden_file=True,
    # Loose demonstration threshold, not a measured force-accuracy limit.
    # Inspect the final force norm and test tighter fmax/SCF tolerances before
    # claiming a quantitative geometry or interaction energy.
    fmax=0.2,
    max_opt_steps=25,
    rhf_options=rhf_opts,
)
