"""The periodic BUG 99 guard decides per element, like the molecular one.

``run_periodic_job`` refuses a cell whose basis is valence-only on some atom
when no ECP is attached for it, because the run would be an all-electron
calculation in a valence-only basis (energies off by hundreds of Hartree).
Until 2026-10 it asked the name heuristic ``is_ecp_paired_basis``, so a
light cell in LANL2DZ or dhf-SVP (both all-electron there) and any cell in
x2c-TZVPall (all-electron everywhere) were refused with "designed for use
with an effective core potential". The guard now goes through
``vibeqc.ecp_metadata.validate_ecp_required`` and the basis registry.
These cases build no integrals; each checks the guard alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.ecp_metadata import is_ecp_paired_basis
from vibeqc.periodic_runner import _require_periodic_ecp_where_needed


class _Options:
    """The two ECP fields the guard reads, as on the periodic SCF options."""

    def __init__(self, blocks=()):
        self.ecp_primitive_blocks = list(blocks)
        self.ecp_centers = []


def _cell(*zs):
    atoms = [vq.Atom(int(z), [0.0, 0.0, 1.6 * i]) for i, z in enumerate(zs)]
    return vq.PeriodicSystem(3, 6.0 * np.eye(3), atoms)


@pytest.mark.parametrize("basis", ["lanl2dz", "dhf-svp", "x2c-tzvpall"])
def test_all_electron_light_cells_are_no_longer_refused_by_name(basis):
    assert is_ecp_paired_basis(basis)  # what the old guard keyed on
    _require_periodic_ecp_where_needed(_Options(), _cell(3, 1), basis)


@pytest.mark.parametrize("basis, z", [("lanl2dz", 11), ("pob-tzvp-rev2", 47), ("def2-svp", 53)])
def test_a_valence_only_atom_without_its_ecp_is_still_refused(basis, z):
    with pytest.raises(ValueError, match="ECP"):
        _require_periodic_ecp_where_needed(_Options(), _cell(z, 17), basis)


def test_an_attached_ecp_satisfies_the_guard():
    _require_periodic_ecp_where_needed(_Options(blocks=[object()]), _cell(47, 17), "pob-tzvp-rev2")


def test_a_valence_only_family_with_no_name_marker_is_refused():
    """def2-mSVP carries the def2-ECP beyond Kr but no ECP-like name, so
    the old heuristic let it through without one."""
    assert not is_ecp_paired_basis("def2-msvp")
    with pytest.raises(ValueError, match="ECP"):
        _require_periodic_ecp_where_needed(_Options(), _cell(53, 53), "def2-msvp")
