"""Phase G1e — ASE bridge for periodic forces.

Pinned contracts:

1. **Public API** — ``vibeqc.ase_periodic.atoms_to_periodic_system``
   and ``vibeqc.ase_periodic.periodic_forces`` are importable.

2. **Atoms → PeriodicSystem conversion** — Å → bohr, lattice +
   atomic positions both rescaled.

3. **End-to-end periodic LDA forces** — H₂ in a 20-Å cubic box
   returns ASE-units forces (eV/Å) with Newton's 3rd law obeyed.

4. **Pure-DFT only for open-shell** — multiplicity > 1 + HF
   refused (path through periodic UHF gradient is v0.6.x).

5. **Refusal on non-3D-periodic Atoms** — slabs / wires not yet
   supported.
"""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms

import vibeqc as vq
from vibeqc.ase_periodic import (
    atoms_to_periodic_system,
    periodic_forces,
)


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_api_exposed():
    from vibeqc.ase_periodic import (
        atoms_to_periodic_system,
        periodic_forces,
    )
    assert callable(atoms_to_periodic_system)
    assert callable(periodic_forces)


# ---------------------------------------------------------------------------
# 2. Atoms → PeriodicSystem conversion
# ---------------------------------------------------------------------------

def test_atoms_to_periodic_system_converts_units():
    atoms = Atoms('H2',
                   positions=[[0, 0, 0], [0, 0, 1.0]],
                   cell=[20, 20, 20], pbc=True)
    sys = atoms_to_periodic_system(atoms)
    assert sys.dim == 3
    # Å → bohr conversion on positions (atom 1 at z = 1.0 Å)
    np.testing.assert_allclose(sys.unit_cell[1].xyz[2],
                                  1.0 * ANGSTROM_TO_BOHR, atol=1e-10)
    # Cell vectors also scaled
    lat = np.asarray(sys.lattice)
    np.testing.assert_allclose(lat[0, 0], 20.0 * ANGSTROM_TO_BOHR,
                                  atol=1e-10)


def test_atoms_to_periodic_system_refuses_non_3d():
    """2D-periodic (slab) Atoms should raise — full slab support is v0.6.x."""
    atoms = Atoms('H2',
                   positions=[[0, 0, 0], [0, 0, 1.0]],
                   cell=[20, 20, 20], pbc=[True, True, False])
    with pytest.raises(NotImplementedError, match="3D-periodic"):
        atoms_to_periodic_system(atoms)


# ---------------------------------------------------------------------------
# 3. End-to-end periodic LDA forces
# ---------------------------------------------------------------------------

def test_periodic_forces_lda_h2_box():
    """H₂ in 20-Å cubic box, LDA periodic SCF → eV/Å forces."""
    atoms = Atoms('H2',
                   positions=[[0, 0, 0], [0, 0, 1.0]],
                   cell=[20, 20, 20], pbc=True)
    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    F = periodic_forces(atoms, basis, kpts=[1, 1, 1],
                          functional="lda",
                          cutoff_bohr=25.0, nuclear_cutoff_bohr=25.0,
                          conv_tol_energy=1e-12)
    assert F.shape == (2, 3)
    # Newton's-3rd-law along z (the bond axis).
    np.testing.assert_allclose(F[0, 2], -F[1, 2], atol=1e-9)
    # Transverse forces vanish.
    np.testing.assert_allclose(F[:, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(F[:, 1], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Open-shell HF refused (pure-DFT-only for now)
# ---------------------------------------------------------------------------

def test_periodic_forces_open_shell_hf_refused():
    """OH-radical-style open-shell HF periodic — periodic UHF gradient
    deferred to v0.6.x; should raise NotImplementedError."""
    atoms = Atoms('OH',
                   positions=[[0, 0, 0], [0, 0, 1.0]],
                   cell=[20, 20, 20], pbc=True)
    sys = atoms_to_periodic_system(atoms, multiplicity=2)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="UHF"):
        periodic_forces(atoms, basis, kpts=[1, 1, 1],
                         functional=None,  # HF
                         multiplicity=2)
