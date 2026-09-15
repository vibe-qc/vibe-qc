"""RHF on glycine / def2-SVP with an explicit DIRECT Fock build.

Glycine at def2-SVP has 95 basis functions, below the AUTO threshold of 140
in this version. AUTO would therefore choose CONVENTIONAL and store the full
tensor: 95⁴ x 8 bytes = 621.42 MiB, before other process memory. This example
selects DIRECT explicitly to demonstrate the memory-bounded alternative.

For automatic selection under a tighter memory policy, set
``scf_mode_auto_threshold`` to 107 to cap the dense tensor below 1 GiB. See
input-h2o-rhf-direct.py / input-h2o-rhf-conventional.py for a minimal pair.

Run:
    .venv/bin/python input-glycine-rhf-direct.py
"""

from __future__ import annotations

from vibeqc import Atom, BasisSet, Molecule, RHFOptions, SCFMode, run_rhf

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def atom_from_angstrom(z, xyz):
    """Construct a low-level Atom after converting angstrom to bohr."""
    return Atom(z, [value * ANGSTROM_TO_BOHR for value in xyz])


# Glycine geometry in angstrom (C₂H₅NO₂, 10 atoms, closed-shell),
# generated from an MMFF94 pre-optimisation. Atom coordinates are stored in
# bohr, so every row is converted at construction.
GLYCINE_ATOMS = [
    atom_from_angstrom(6, [-2.14912117, 0.61490945, -0.04623979]),
    atom_from_angstrom(6, [-0.69532565, 1.08777284, 0.46529739]),
    atom_from_angstrom(8, [2.07136293, -0.20398383, -0.40288843]),
    atom_from_angstrom(8, [-0.47627582, 0.75493601, 1.83890424]),
    atom_from_angstrom(7, [0.25131369, 0.36737660, -0.41062234]),
    atom_from_angstrom(1, [-2.42040617, 1.13995698, -0.95870224]),
    atom_from_angstrom(1, [-2.26895158, -0.45332391, -0.11091677]),
    atom_from_angstrom(1, [-0.61604639, 2.17228635, 0.34163710]),
    atom_from_angstrom(1, [1.35683503, 1.20732242, -0.25452337]),
    atom_from_angstrom(1, [0.90108897, -0.53527126, -0.96850826]),
]

mol = Molecule(GLYCINE_ATOMS)

opts = RHFOptions()
opts.scf_mode = SCFMode.DIRECT  # explicit: integral-driven at any basis size
opts.conv_tol_energy = 1e-8

basis = BasisSet(mol, "def2-svp")
result = run_rhf(mol, basis, opts)

print(f"Energy: {result.energy:.10f} Ha")
print(f"Converged in {result.n_iter} iterations")
print(f"DIRECT Fock build: {basis.nbasis} basis functions")
