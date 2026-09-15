"""Periodic D3-BJ dispersion tests.

The builtin backend uses the constant-C6 starter table shipped with
vibe-qc (only H, C, N, O, F at present) so tests focus on those
elements. The dftd3 backend is exercised when the optional package is
importable; tests skip cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.dispersion import dftd3_available
from vibeqc.dispersion_periodic import (
    PeriodicDispersionResult,
    _auto_supercell,
    compute_d3bj_periodic,
)


_ANG_TO_BOHR = 1.0 / 0.529177210903


def _h2_in_box(box_ang: float) -> vq.PeriodicSystem:
    A = box_ang * _ANG_TO_BOHR
    lat = np.diag([A, A, A])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.74 * _ANG_TO_BOHR, 0.0, 0.0]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def _graphene_unit_cell() -> vq.PeriodicSystem:
    a = 2.46 * _ANG_TO_BOHR
    c = 15.0 * _ANG_TO_BOHR
    lat = np.array(
        [[a, -a / 2, 0.0],
         [0.0, a * np.sqrt(3) / 2, 0.0],
         [0.0, 0.0, c]]
    )
    z = c / 2.0
    atoms = [
        vq.Atom(6, [0.0, 0.0, z]),
        vq.Atom(6, [a / 2.0, a * np.sqrt(3) / 6.0, z]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def test_builtin_h2_converges_with_supercell_size():
    """Dispersion energy magnitude should grow monotonically with
    supercell size as more shells of images contribute, then plateau."""
    sys = _h2_in_box(box_ang=4.0)
    energies = []
    for ns in (1, 3, 5, 7):
        res = compute_d3bj_periodic(
            sys, "pbe", backend="builtin", supercell=(ns, ns, ns),
        )
        energies.append(res.energy)
    # Monotonic (more negative) and bounded (the increments shrink).
    diffs = np.diff(energies)
    assert np.all(diffs <= 1e-12), f"non-monotone: {energies}"
    assert abs(diffs[-1]) < abs(diffs[0])


def test_auto_supercell_grows_with_small_box():
    sys_small = _h2_in_box(box_ang=3.0)
    sys_large = _h2_in_box(box_ang=20.0)
    small = _auto_supercell(sys_small, cutoff_bohr=30.0)
    large = _auto_supercell(sys_large, cutoff_bohr=30.0)
    assert all(s >= l for s, l in zip(small, large))
    # All entries odd (centred supercell).
    for n in small + large:
        assert n % 2 == 1


def test_builtin_graphene_unit_cell():
    """C2 graphene unit cell: dispersion energy is negative and finite."""
    sys = _graphene_unit_cell()
    res = compute_d3bj_periodic(
        sys, "pbe", backend="builtin", cutoff_bohr=30.0,
    )
    assert isinstance(res, PeriodicDispersionResult)
    assert res.backend == "builtin"
    assert res.energy < 0.0
    # Sanity: |E| < 1 Ha/cell (way below physical: this is just bounding).
    assert abs(res.energy) < 0.1


def test_builtin_gradient_shape_matches_central_cell():
    sys = _h2_in_box(box_ang=5.0)
    res = compute_d3bj_periodic(
        sys, "pbe", backend="builtin", supercell=(3, 3, 3),
        with_gradient=True,
    )
    assert res.gradient.shape == (2, 3)
    # Newton's third law: net force on the cell should be zero for
    # the central cell when summed over all atoms.
    net = res.gradient.sum(axis=0)
    assert np.allclose(net, 0.0, atol=1e-10)


def test_unknown_functional_raises():
    sys = _h2_in_box(box_ang=5.0)
    with pytest.raises(ValueError, match="no D3-BJ parameters"):
        compute_d3bj_periodic(
            sys, "totally-not-a-functional", backend="builtin",
        )


def test_passing_explicit_params_object():
    sys = _h2_in_box(box_ang=5.0)
    params = vq.D3BJParams(s6=1.0, s8=0.722, a1=0.4289, a2=4.4407)  # PBE
    res = compute_d3bj_periodic(
        sys, params, backend="builtin", supercell=(3, 3, 3),
    )
    assert res.energy < 0.0


@pytest.mark.skipif(not dftd3_available(),
                    reason="dftd3 optional package not installed")
def test_dftd3_backend_matches_reference():
    """dftd3's periodic D3-BJ on graphene matches its molecular result
    when the box is huge (no images contribute)."""
    # Big enough box that dftd3's lattice sum sees no images.
    A = 50.0 * _ANG_TO_BOHR
    lat = np.diag([A, A, A])
    atoms = [vq.Atom(6, [0.0, 0.0, 0.0]), vq.Atom(6, [1.4 * _ANG_TO_BOHR, 0.0, 0.0])]
    sys = vq.PeriodicSystem(3, lat, atoms)
    res_periodic = compute_d3bj_periodic(sys, "pbe", backend="dftd3")

    # Compare against the molecular dftd3 result on the same 2 atoms.
    mol = vq.Molecule(atoms, 0, 1)
    res_mol = vq.compute_d3bj(mol, "pbe", backend="dftd3")
    assert np.isclose(res_periodic.energy, res_mol.energy, atol=1e-8)


@pytest.mark.skipif(not dftd3_available(),
                    reason="dftd3 optional package not installed")
def test_dftd3_periodic_below_molecular_for_dense_packing():
    """In a dense periodic system the dispersion stabilisation should
    be greater (more negative) than for the isolated unit cell."""
    sys = _graphene_unit_cell()
    res_periodic = compute_d3bj_periodic(sys, "pbe", backend="dftd3")
    mol = vq.Molecule(list(sys.unit_cell), 0, 1)
    res_mol = vq.compute_d3bj(mol, "pbe", backend="dftd3")
    assert res_periodic.energy < res_mol.energy < 0.0


def test_auto_dispatch_picks_a_real_backend():
    sys = _h2_in_box(box_ang=5.0)
    res = compute_d3bj_periodic(sys, "pbe", backend="auto")
    assert res.backend in ("dftd3", "builtin")
    assert res.energy < 0.0


@pytest.mark.skipif(not dftd3_available(),
                    reason="dftd3 optional package not installed")
def test_slab_with_vacuum_runs_dftd3():
    """A slab (3D system with vacuum along c) should compute periodic
    D3-BJ without choking on the long vacuum direction."""
    sys, _info = vq.slab("Ni", facet=(1, 1, 1), n_layers=3,
                         vacuum=12.0, periodic_z=True)
    res = compute_d3bj_periodic(sys, "pbe", backend="dftd3",
                                 cutoff_bohr=30.0)
    # Dispersion is attractive for a metallic slab.
    assert res.energy < 0.0
