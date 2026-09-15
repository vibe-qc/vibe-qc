"""H2O geometry optimization with the Brent steepest-descent backend.

Demonstrates optimizer_backend="brent" — a conservative, gradient-driven
optimiser that uses steepest-descent direction + Brent 1-D line search
per step.  Never takes uphill steps; needs no Hessian approximation.
Useful for flat/dispersion-bound surfaces where L-BFGS-B or BFGSLineSearch
misbehave.

The ``output_qvf=True`` flag (default for ``run_job`` from v0.14) writes a
self-contained QVF archive that vibe-view can open as an animated
optimization trajectory — same user experience as the ASE backend.

Compare with examples/molecular/input-h2o-opt.py (ASE BFGS) and
examples/molecular/input-h2o-rks-pbe-d3-opt.py (ASE BFGS + dispersion).

Run:
    .venv/bin/python examples/molecular/input-h2o-brent-opt.py

Produces:
    output-h2o-brent-opt.out     — banner, SCF trace, optimised geometry
    output-h2o-brent-opt.molden  — MOs at the optimised geometry
    output-h2o-brent-opt.qvf     — QVF archive with trajectory, structure,
                                    density, MO metadata; open with vibe-view
    output-h2o-brent-opt.system  — host/build/runtime manifest (TOML)
"""

from pathlib import Path

from vibeqc import Atom, Molecule, run_job

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem.replace("input-", "output-")

mol = Molecule(
    [
        Atom(8, [0.0, 0.00, 0.00]),
        Atom(1, [0.0, 1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ]
)

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    optimize=True,
    optimizer_backend="brent",  # ← steepest-descent + Brent line search
    output=HERE / STEM,
    output_qvf=True,  # ← write .qvf archive for vibe-view
    fmax=0.05,
    max_opt_steps=50,
)
