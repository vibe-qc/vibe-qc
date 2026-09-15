"""Tests for the ASE embedded-surface calculator."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc.periodic_embedding.ase_calc import EmbeddedSurfaceCalculator


@pytest.mark.skipif(
    not hasattr(EmbeddedSurfaceCalculator, "calculate"),
    reason="ASE not installed or calculator class is a stub",
)
def test_calculator_energy():
    """Calculator returns a finite energy for a simple chain."""
    from ase import Atoms

    # Build a simple chain: 4 H atoms in a box.
    box = 30.0
    sp = 2.0
    atoms = Atoms(
        "H4",
        positions=[(box / 2, box / 2, 2 + i * sp) for i in range(4)],
        cell=[box, box, box],
        pbc=[True, True, True],
    )
    atoms.set_tags([0, 0, 1, 1])  # bottom 2 = substrate, top 2 = region I

    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g",
        surface_k_mesh=(1, 1),
        contour_n_nodes=24,
        e_bottom=-2.2,
        e_fermi=-1.0,
        lat_cutoff_bohr=15.0,
    )
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert np.isfinite(energy)

    # The last result should be accessible.
    assert calc.last_result is not None
    assert calc.last_result.occupation > 0


@pytest.mark.skipif(
    not hasattr(EmbeddedSurfaceCalculator, "calculate"),
    reason="ASE not installed or calculator class is a stub",
)
def test_atoms_to_system_lattice_transpose():
    """ASE cell rows (lattice vectors) map to PeriodicSystem columns.

    A non-orthogonal (sheared) in-plane cell exposes the transpose:
    cubic cells are transpose-invariant and would hide a bug here.
    """
    from ase import Atoms
    from ase.units import Bohr

    # Sheared in-plane cell + long c-axis (a slab-like geometry).
    cell_ang = np.array(
        [[4.0, 0.7, 0.0], [0.3, 4.0, 0.0], [0.0, 0.0, 20.0]]
    )
    atoms = Atoms(
        "H2",
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
        cell=cell_ang,
        pbc=[True, True, True],
    )
    calc = EmbeddedSurfaceCalculator(basis_name="sto-3g")
    system = calc._atoms_to_system(atoms)

    lattice = np.array(system.lattice)  # bohr; columns = lattice vectors
    expected = (cell_ang / Bohr).T  # ASE rows -> PeriodicSystem columns
    assert np.allclose(lattice, expected)
    # Column j is lattice vector a_j (not row j).
    assert np.allclose(lattice[:, 0], cell_ang[0] / Bohr)
    assert np.allclose(lattice[:, 1], cell_ang[1] / Bohr)
    # Long c-axis -> slab dimensionality.
    assert system.dim == 2


def _h4_chain():
    """An on-axis H4 chain in a 30-bohr box: 2 substrate + 2 region I.

    Centred in (x, y) so the in-plane forces must vanish by symmetry.
    """
    from ase import Atoms

    box, sp = 30.0, 2.0
    atoms = Atoms(
        "H4",
        positions=[(box / 2, box / 2, 2 + i * sp) for i in range(4)],
        cell=[box, box, box],
        pbc=[True, True, True],
    )
    atoms.set_tags([0, 0, 1, 1])
    return atoms


@pytest.mark.skipif(
    not hasattr(EmbeddedSurfaceCalculator, "calculate"),
    reason="ASE not installed or calculator class is a stub",
)
def test_calculator_forces_shape_mask_and_symmetry():
    """Numerical forces: shape (N, 3), masked atoms zero, in-plane
    components vanish by the chain's x/y symmetry, z-force is real."""
    atoms = _h4_chain()
    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g",
        surface_k_mesh=(1, 1),
        contour_n_nodes=20,
        e_bottom=-2.2,
        e_fermi=-1.0,
        lat_cutoff_bohr=15.0,
        force_atom_indices=[2, 3],  # only region-I atoms are mobile
    )
    atoms.calc = calc

    forces = atoms.get_forces()
    assert forces.shape == (4, 3)
    assert np.all(np.isfinite(forces))

    # Non-active (substrate) atoms keep exactly zero force.
    assert np.allclose(forces[:2], 0.0)

    # On-axis chain: x and y forces vanish by symmetry; z is nonzero.
    assert np.allclose(forces[2:, 0], 0.0, atol=1e-6)
    assert np.allclose(forces[2:, 1], 0.0, atol=1e-6)
    assert np.abs(forces[3, 2]) > 1e-3


@pytest.mark.skipif(
    not hasattr(EmbeddedSurfaceCalculator, "calculate"),
    reason="ASE not installed or calculator class is a stub",
)
def test_calculator_force_matches_independent_fd():
    """The calculator's force equals an independent finite difference
    of get_potential_energy (different step) — pins sign, eV/Å units,
    and atom/axis indexing."""
    from ase.units import Bohr

    atoms = _h4_chain()
    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g",
        surface_k_mesh=(1, 1),
        contour_n_nodes=20,
        e_bottom=-2.2,
        e_fermi=-1.0,
        lat_cutoff_bohr=15.0,
        force_atom_indices=[3],
    )
    atoms.calc = calc
    forces = atoms.get_forces()

    # Independent central difference on atom 3, z-axis, with a *different*
    # step, holding the contour window fixed (explicit e_fermi/e_bottom).
    delta_bohr = 0.02
    delta_ang = delta_bohr * Bohr
    base = atoms.get_positions().copy()

    def energy_at(pos):
        a = atoms.copy()
        a.set_positions(pos)
        c = EmbeddedSurfaceCalculator(
            basis_name="sto-3g",
            surface_k_mesh=(1, 1),
            contour_n_nodes=20,
            e_bottom=-2.2,
            e_fermi=-1.0,
            lat_cutoff_bohr=15.0,
        )
        a.calc = c
        return a.get_potential_energy()  # eV

    pos = base.copy()
    pos[3, 2] += delta_ang
    e_plus = energy_at(pos)
    pos = base.copy()
    pos[3, 2] -= delta_ang
    e_minus = energy_at(pos)
    f_indep = -(e_plus - e_minus) / (2.0 * delta_ang)  # eV/Å

    # Agreement is limited only by the O(h²) truncation difference
    # between the two step sizes; a sign / unit / index bug would be
    # off by ~×2, ×27, or land on the wrong component entirely.
    assert abs(forces[3, 2] - f_indep) < 0.05


@pytest.mark.skipif(
    not hasattr(EmbeddedSurfaceCalculator, "calculate"),
    reason="ASE not installed or calculator class is a stub",
)
def test_open_shell_odd_electron_cell():
    """Odd-electron cell (H3, 3 e⁻) runs energy + forces with
    multiplicity=2 + scf_method="uhf".

    Pre-fix the calculator hardcoded charge=0, multiplicity=1, so
    ``unit_cell_molecule()`` raised "n_electrons and multiplicity are
    inconsistent" before any embedding work ran.  Threading charge /
    multiplicity through ``_atoms_to_system`` lets the open-shell path
    run.
    """
    from ase import Atoms

    box, sp = 30.0, 2.0
    atoms = Atoms(
        "H3",
        positions=[(box / 2, box / 2, 2 + i * sp) for i in range(3)],
        cell=[box, box, box],
        pbc=[True, True, True],
    )
    atoms.set_tags([0, 0, 1])  # bottom 1 = substrate, top = region I

    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g",
        surface_k_mesh=(1, 1),
        contour_n_nodes=16,  # coarse contour: this is a plumbing test
        e_bottom=-2.2,
        e_fermi=-1.0,
        lat_cutoff_bohr=15.0,
        multiplicity=2,  # doublet — 3 electrons, odd
        scf_method="uhf",  # spin-polarised density route
        force_atom_indices=[2],
    )
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert np.isfinite(energy)

    forces = atoms.get_forces()
    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
    # Masked (substrate) atoms keep exactly zero force.
    assert np.allclose(forces[:2], 0.0)


@pytest.mark.skipif(
    not hasattr(EmbeddedSurfaceCalculator, "calculate"),
    reason="ASE not installed or calculator class is a stub",
)
def test_charge_multiplicity_threaded_into_system():
    """``charge`` / ``multiplicity`` reach the built PeriodicSystem;
    defaults stay charge=0, multiplicity=1."""
    from ase import Atoms

    atoms = Atoms(
        "H3",
        positions=[(0.0, 0.0, 2.0 * i) for i in range(3)],
        cell=[30.0, 30.0, 30.0],
        pbc=[True, True, True],
    )

    # Default: closed-shell singlet.
    default_calc = EmbeddedSurfaceCalculator(basis_name="sto-3g")
    assert default_calc.charge == 0
    assert default_calc.multiplicity == 1

    # Open-shell: a doublet cell builds without the parity error.
    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g", multiplicity=2
    )
    system = calc._atoms_to_system(atoms)
    assert system.charge == 0
    assert system.multiplicity == 2
    # The molecule view (which runs the parity check) is now consistent.
    mol = system.unit_cell_molecule()
    assert mol.multiplicity == 2
