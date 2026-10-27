"""The atomic mass table reaches the elements an ECP basis exists for.

Frequencies, thermochemistry and molecular dynamics mass-weight, and the table
used to stop at krypton, so every job with a heavy atom died on "no built-in
mass" before it computed anything.
"""

from __future__ import annotations

import pytest

import vibeqc as vq
from vibeqc.hessian import _resolve_masses
from vibeqc.properties import _ATOMIC_MASSES


def test_every_element_through_plutonium_has_a_mass():
    assert len(_ATOMIC_MASSES) == 95
    assert _ATOMIC_MASSES[0] == 0.0
    assert all(m > 1.0 for m in _ATOMIC_MASSES[1:])


@pytest.mark.parametrize(
    "z, mass",
    [
        (35, 79.904),     # Br, the old table's last-but-one
        (36, 83.798),     # Kr, the old table's last
        (37, 85.468),     # Rb, the first one past it
        (46, 106.42),     # Pd
        (53, 126.904),    # I
        (54, 131.293),    # Xe
        (78, 195.084),    # Pt
        (79, 196.967),    # Au
        (82, 207.2),      # Pb
        (92, 238.029),    # U
    ],
)
def test_anchor_masses(z, mass):
    assert _ATOMIC_MASSES[z] == pytest.approx(mass, abs=1e-3)


def test_the_known_inversions_of_the_periodic_order_are_kept():
    """Mass is not monotonic in Z at Ar/K, Co/Ni, Te/I, Th/Pa, U/Np: a table
    typed in element order that "sorted itself" would be wrong exactly there."""
    m = _ATOMIC_MASSES
    assert m[18] > m[19]  # Ar > K
    assert m[27] > m[28]  # Co > Ni
    assert m[52] > m[53]  # Te > I
    assert m[90] > m[91]  # Th > Pa
    # and everywhere else the mass grows with Z
    inversions = {i for i in range(2, len(m)) if m[i] < m[i - 1]}
    assert inversions == {19, 28, 53, 91, 93}


def test_a_molecule_with_iodine_gets_its_masses():
    mol = vq.Molecule([vq.Atom(53, (0.0, 0.0, 0.0)), vq.Atom(1, (0.0, 0.0, 3.0))], 0, 1)
    masses = _resolve_masses(mol, None)
    assert list(masses) == pytest.approx([126.904, 1.008], abs=1e-3)


def test_an_element_beyond_the_table_still_asks_for_an_explicit_mass():
    mol = vq.Molecule([vq.Atom(101, (0.0, 0.0, 0.0))], 0, 2)
    with pytest.raises(ValueError, match="no built-in mass for Z=101"):
        _resolve_masses(mol, None)
