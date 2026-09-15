"""Band-structure / DOS smoke tests on a 1D H2 chain.

Goal: verify that the periodic-bands pipeline (k-path construction,
Bloch summation, diagonalisation, and Gaussian DOS broadening) returns
sensible shapes and obeys the basic sum rules. Quantitative
ground-truth eigenvalues live in dedicated solid-state tests later;
here we assert the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


@pytest.fixture
def h2_chain():
    a = 6.0  # bohr
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])]
    system = vq.PeriodicSystem(
        1,
        [[a, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        atoms,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_kpath_shapes_and_distances(h2_chain):
    system, _ = h2_chain
    kpath = vq.kpath_from_segments(
        system,
        [
            ((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X"),
            ((0.5, 0.0, 0.0), "X", (0.0, 0.0, 0.0), "Γ"),
        ],
        points_per_segment=10,
    )
    # 11 + 10 (start of seg 2 dropped to avoid duplicate)
    assert kpath.n_points == 21
    assert kpath.kpoints_cart.shape == (21, 3)
    assert kpath.kpoints_frac.shape == (21, 3)
    assert kpath.distances[0] == pytest.approx(0.0)
    assert np.all(np.diff(kpath.distances) >= -1e-12)
    # Going Γ→X→Γ ends at 2× the Γ→X distance.
    assert kpath.distances[-1] == pytest.approx(2.0 * kpath.distances[10])


def test_band_structure_hcore_shape_and_fermi(h2_chain):
    system, basis = h2_chain
    kpath = vq.kpath_from_segments(
        system,
        [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")],
        points_per_segment=20,
    )
    bs = vq.band_structure_hcore(
        system, basis, kpath, n_electrons_per_cell=2,
    )
    assert bs.energies.shape == (21, basis.nbasis)
    assert bs.n_bands == basis.nbasis
    # Eigenvalues at each k must be sorted ascending.
    assert np.all(np.diff(bs.energies, axis=1) >= -1e-10)
    # e_fermi is the highest occupied eigenvalue (HOMO) across all k.
    assert bs.e_fermi is not None
    n_occ = 1
    assert bs.e_fermi == pytest.approx(bs.energies[:, :n_occ].max())


def test_dos_area_equals_n_bands(h2_chain):
    system, basis = h2_chain
    dos = vq.density_of_states_hcore(
        system, basis, [40, 1, 1], sigma=0.05, n_electrons_per_cell=2,
    )
    area = float(np.trapezoid(dos.dos, dos.energies))
    # Each (k, band) contributes a unit-area Gaussian, weighted so
    # weights sum to 1 → total integral = n_bands.
    assert area == pytest.approx(basis.nbasis, rel=5e-3)


def test_dos_e_fermi_only_for_closed_shell(h2_chain):
    system, basis = h2_chain
    dos_open = vq.density_of_states_hcore(
        system, basis, [10, 1, 1], sigma=0.05, n_electrons_per_cell=1,
    )
    assert dos_open.e_fermi is None  # odd electron count → unset
    dos_closed = vq.density_of_states_hcore(
        system, basis, [10, 1, 1], sigma=0.05, n_electrons_per_cell=2,
    )
    assert dos_closed.e_fermi is not None
