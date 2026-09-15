"""Restricted open-shell Kohn--Sham (ROKS) single-point on the methyl
radical CH3 (doublet) with the PBE functional.

Run:
    .venv/bin/python input-ch3-roks.py

Produces:
    output-ch3-roks.out      — SCF trace, orbital tables, energy components
                               (incl. the exchange-correlation term), and
                               the spin-pure <S^2> = 0.75
    output-ch3-roks.molden   — MO blocks for visualisation

ROKS is the DFT counterpart of ROHF: a spin-restricted open-shell
Kohn--Sham determinant (spin-pure density). It reuses ROHF's Roothaan
coupling with the spin-polarised XC potential. Supports LDA, GGA and
global-hybrid functionals (here PBE; try ``functional="b3lyp"`` for a
hybrid). Meta-GGA / range-separated / double hybrids are not yet wired.

Theory: Roothaan coupling (Rev. Mod. Phys. 32, 179 (1960)) + libxc XC.
"""

from pathlib import Path

from vibeqc import Atom, Molecule, ROKSOptions, run_job

HERE = Path(__file__).parent

# Planar methyl radical CH3 (D3h), doublet, 9 electrons. Bohr coordinates;
# C-H ~ 1.079 A = 2.039 bohr.
mol = Molecule(
    [
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(1, [2.039, 0.0, 0.0]),
        Atom(1, [-1.020, 1.766, 0.0]),
        Atom(1, [-1.020, -1.766, 0.0]),
    ],
    multiplicity=2,
)

roks_opts = ROKSOptions()
roks_opts.conv_tol_grad = 1e-7

run_job(
    mol,
    basis="6-31g*",
    method="roks",
    functional="pbe",
    output=HERE / "output-ch3-roks",
    roks_options=roks_opts,
)
