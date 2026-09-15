#!/usr/bin/env python
"""CASSCF natural orbitals, and the QVF archive that carries them.

Writes ``h2_natural_orbitals.qvf`` in the current directory by default (or
at the output stem passed as the first argument). It reproduces the fixture
``tests/data/h2_natural_orbitals.qvf`` owned by the separate vibe-view
repository, which pins its HONO / LUNO path against computed data. It does
not write into the viewer checkout automatically.

H2 at 2.4 A in 6-31G with CAS(2,2) is the textbook two-configuration case:
as the bond stretches the wavefunction stops being a single determinant and
the natural occupations leave {0, 2}. They come out 1.885 and 0.115 here,
substantial diradical character, and exactly the regime where an occupancy
is a measured real number rather than something you could read off the
orbital index.

A CASSCF run now writes a ``wavefunction.gto`` section holding those natural
orbitals, with nothing to configure. That was not true before the solver
exposed its orbitals: ``run_job(method="casscf", output_qvf=True)`` used to
write structure, citations and run.record and no wavefunction at all,
because the QVF payload was built from mean-field ``mo_coeffs`` that a
correlated solver does not have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import vibeqc as vq

BOND = 2.4  # Angstrom
BASIS = "6-31g"
ACTIVE = (2, 2)  # (electrons, orbitals)


def main() -> int:
    stem = Path(sys.argv[1] if len(sys.argv) > 1 else "h2_natural_orbitals")

    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, BOND])])
    # record_hostname=False: this archive is committed as a test fixture, and
    # the machine name has no business travelling with it. An earlier
    # regeneration of this file did leak one, caught by the repo-wide
    # redaction audit rather than by anything here.
    result = vq.run_job(
        mol, basis=BASIS, method="casscf", active_space=ACTIVE,
        output=str(stem), output_qvf=True, record_hostname=False,
    )

    occ = np.asarray(result.natural_occupations)
    print(f"CASSCF({ACTIVE[0]}e,{ACTIVE[1]}o)/{BASIS} on H2 at {BOND} A")
    print(f"  E           = {result.energy:.10f} Ha")
    print(f"  occupations = {np.array2string(occ, precision=6)}")
    print(f"  sum         = {occ.sum():.8f} electrons")
    print(f"  in the Pulay 0.02-1.98 window: {int(((occ > 0.02) & (occ < 1.98)).sum())}")

    # The orbitals must be orthonormal and their density must carry the right
    # number of electrons, or the archive describes no wavefunction at all.
    basis = vq.BasisSet(mol, BASIS)
    overlap = np.asarray(vq.compute_overlap(basis))
    coeffs = np.asarray(result.natural_orbitals)
    gram = coeffs.T @ overlap @ coeffs
    density = (coeffs * occ) @ coeffs.T
    print(f"  max |C^T S C - I| = {np.abs(gram - np.eye(gram.shape[0])).max():.2e}")
    print(f"  tr(P S)           = {np.trace(density @ overlap):.8f}")

    print(f"wrote {stem.with_suffix('.qvf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
