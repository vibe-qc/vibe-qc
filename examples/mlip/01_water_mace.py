"""MACE machine-learning interatomic potential on H2O — single point + opt.

Requires the optional ``[mace]`` extra (PyTorch + e3nn) and Python <= 3.13:

    pip install 'vibe-qc[mace]'

Run:

    python examples/mlip/01_water_mace.py

Produces (sharing each call's output stem):

    output-h2o-mace.out / .bibtex / .references / .xyz   — single point
    output-h2o-mace-opt.out / .traj / ...                — geometry opt

``method="mace"`` drives ACEsuit MACE's pre-trained forward pass
(``CLAUDE.md`` §10). The default model is the MIT-licensed MACE-MPA-0
(materials). The energy is on a model-specific reference scale, **not** a
vibe-qc total electronic energy — the ``.out`` prints a provenance block
(model, license, energy-scale caveat) and the run cites the MACE papers.
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).parent
mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

# 1. Single-point energy + forces with the default MIT model (MACE-MPA-0).
run_job(mol, method="mace", output=HERE / "output-h2o-mace")

# 2. Geometry optimization (ASE BFGS driving the MACE calculator).
run_job(
    mol, method="mace", optimize=True, fmax=0.02,
    output=HERE / "output-h2o-mace-opt",
)

# 3. Another model — the organic MACE-OFF23 (ASL: academic, NON-COMMERCIAL;
#    requires acknowledgment). Uncomment to run:
# from vibeqc.mlip import MLIPOptions
# run_job(
#     mol, method="mace",
#     mlip_options=MLIPOptions(model="off23-medium", accept_academic_license=True),
#     output=HERE / "output-h2o-mace-off23",
# )
