"""Open-shell UHF single-point on OH radical (doublet).

Run:
    .venv/bin/python input-oh-radical.py

Produces:
    output-oh-radical.out     — SCF trace, alpha and beta orbital tables,
                                 HOMO/LUMO markers for each spin
    output-oh-radical.molden  — separate Alpha / Beta MO blocks

Uses the SAD initial guess — a couple of ASCII-level iterations rarely
converge open-shell atoms / small radicals from Hcore, so SAD is set as
the default in ``UHFOptions`` below.
"""

from pathlib import Path

from vibeqc import Atom, InitialGuess, Molecule, UHFOptions, run_job

HERE = Path(__file__).parent

# OH radical: doublet, 9 electrons. Bond length 0.97 Å = 1.833 bohr.
mol = Molecule(
    [
        Atom(8, [0.0,   0.0, 0.0]),
        Atom(1, [1.833, 0.0, 0.0]),
    ],
    multiplicity=2,
)

uhf_opts = UHFOptions()
uhf_opts.initial_guess = InitialGuess.SAD

run_job(
    mol,
    basis="6-31g*",
    method="uhf",
    output=HERE / "output-oh-radical",
    uhf_options=uhf_opts,
)
