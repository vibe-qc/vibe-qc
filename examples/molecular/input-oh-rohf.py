"""Restricted open-shell HF (ROHF) single-point on the OH radical (doublet).

Run:
    .venv/bin/python input-oh-rohf.py

Produces:
    output-oh-rohf.out      — SCF trace, orbital tables, energy summary
    output-oh-rohf.molden   — MO blocks (identical alpha/beta spatial
                              orbitals; occupations 1+1 alpha, 1 beta core)

ROHF keeps a single spin-restricted orbital set (doubly-occupied +
singly-occupied + virtual), so the determinant is **spin-pure**:
``<S^2> = S(S+1) = 0.75`` exactly for a doublet — no spin contamination,
unlike UHF (compare ``input-oh-radical.py``). This is the standard
reference for spin-pure post-HF and a clean CAS starting point.

Theory: Roothaan's single effective Fock operator,
Roothaan, Rev. Mod. Phys. 32, 179 (1960), doi:10.1103/RevModPhys.32.179.
"""

from pathlib import Path

from vibeqc import Atom, Molecule, ROHFOptions, run_job

HERE = Path(__file__).parent

# OH radical: doublet, 9 electrons. Bond length 0.97 A = 1.833 bohr.
mol = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [1.833, 0.0, 0.0]),
    ],
    multiplicity=2,
)

rohf_opts = ROHFOptions()
rohf_opts.conv_tol_grad = 1e-8

run_job(
    mol,
    basis="6-31g*",
    method="rohf",
    output=HERE / "output-oh-rohf",
    rohf_options=rohf_opts,
)
