"""Tests for the BIPOLE Phase 1 C++ kernel:
``compute_multipole_moments_lattice``.

Validates:
  * Shape conventions (n_cells, n_components, nbf, nbf).
  * Overlap component at g=0 matches the existing
    ``compute_overlap``.
  * Dipole components at g=0 match the existing
    ``compute_dipole``.
  * Per-cell Frobenius norm decays with |R_g| (Gaussian-overlap
    screening).
  * `cartesian_multipole_n_components` returns 4 / 10 / 20 for
    L_max = 1 / 2 / 3 respectively.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import CoulombMethod, LatticeSumOptions
from vibeqc._vibeqc_core import (
    cartesian_multipole_n_components,
    compute_dipole,
    compute_multipole_moments_lattice,
    compute_overlap,
    compute_overlap_lattice,
)


ANG2BOHR = 1.0 / 0.529177210903


@pytest.fixture
def lih_primitive():
    """Small periodic system for fast tests."""
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


@pytest.fixture
def lat_opts():
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    opts.nuclear_cutoff_bohr = 8.0
    opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED
    return opts


# ---------------------------------------------------------------------
# Component counting
# ---------------------------------------------------------------------
def test_cartesian_n_components():
    assert cartesian_multipole_n_components(1) == 4    # S + xyz
    assert cartesian_multipole_n_components(2) == 10   # + 6 quadrupole
    assert cartesian_multipole_n_components(3) == 20   # + 10 octupole


# ---------------------------------------------------------------------
# Shape + basic structure
# ---------------------------------------------------------------------
def test_multipole_set_shape(lih_primitive, lat_opts):
    system, basis = lih_primitive
    set_ = compute_multipole_moments_lattice(basis, system, lat_opts, L_max=2)
    n_cells = len(list(set_.cells))
    n_comp = cartesian_multipole_n_components(2)
    assert set_.nbf == basis.nbasis
    assert set_.L_max == 2
    assert len(set_.blocks) == n_cells
    for c in range(n_cells):
        assert len(set_.blocks[c]) == n_comp
        for comp in range(n_comp):
            blk = np.asarray(set_.blocks[c][comp])
            assert blk.shape == (basis.nbasis, basis.nbasis)


def test_multipole_set_L_max_3_has_20_components(lih_primitive, lat_opts):
    system, basis = lih_primitive
    set_ = compute_multipole_moments_lattice(basis, system, lat_opts, L_max=3)
    assert set_.L_max == 3
    assert len(set_.blocks[0]) == 20


def test_invalid_L_max_raises(lih_primitive, lat_opts):
    system, basis = lih_primitive
    for L in [0, 4, -1, 100]:
        with pytest.raises(Exception):
            compute_multipole_moments_lattice(basis, system, lat_opts, L_max=L)


# ---------------------------------------------------------------------
# Overlap component at g=0 matches compute_overlap
# ---------------------------------------------------------------------
def test_overlap_component_at_g0_matches_compute_overlap(lih_primitive, lat_opts):
    """Component 0 of the multipole set at g=0 is the standard overlap."""
    system, basis = lih_primitive
    set_ = compute_multipole_moments_lattice(basis, system, lat_opts, L_max=2)
    # Find g=0 cell.
    g0_idx = next(i for i, c in enumerate(set_.cells)
                   if (np.asarray(c.index) == np.array([0, 0, 0])).all())
    S_lattice_g0 = np.asarray(set_.blocks[g0_idx][0])

    S_molecular = compute_overlap(basis)
    np.testing.assert_allclose(S_lattice_g0, S_molecular, atol=1e-12)


def test_overlap_component_at_g0_matches_overlap_lattice(lih_primitive, lat_opts):
    """Cross-check: also matches compute_overlap_lattice at g=0."""
    system, basis = lih_primitive
    set_ = compute_multipole_moments_lattice(basis, system, lat_opts, L_max=2)
    S_lat = compute_overlap_lattice(basis, system, lat_opts)

    # Find g=0 in both.
    g0_set = next(i for i, c in enumerate(set_.cells)
                   if (np.asarray(c.index) == np.array([0, 0, 0])).all())
    g0_lat = next(i for i, c in enumerate(S_lat.cells)
                   if (np.asarray(c.index) == np.array([0, 0, 0])).all())
    np.testing.assert_allclose(
        np.asarray(set_.blocks[g0_set][0]),
        np.asarray(S_lat.blocks[g0_lat]),
        atol=1e-12,
    )


# ---------------------------------------------------------------------
# Dipole components at g=0 match compute_dipole
# ---------------------------------------------------------------------
def test_dipole_components_at_g0_match_compute_dipole(lih_primitive, lat_opts):
    """Components 1, 2, 3 (μ_x, μ_y, μ_z) at g=0 match compute_dipole."""
    system, basis = lih_primitive
    origin = [0.0, 0.0, 0.0]
    set_ = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=2, origin=origin,
    )
    g0_idx = next(i for i, c in enumerate(set_.cells)
                   if (np.asarray(c.index) == np.array([0, 0, 0])).all())
    mu_x = np.asarray(set_.blocks[g0_idx][1])
    mu_y = np.asarray(set_.blocks[g0_idx][2])
    mu_z = np.asarray(set_.blocks[g0_idx][3])

    dip = compute_dipole(basis, origin)
    np.testing.assert_allclose(mu_x, np.asarray(dip.x), atol=1e-12)
    np.testing.assert_allclose(mu_y, np.asarray(dip.y), atol=1e-12)
    np.testing.assert_allclose(mu_z, np.asarray(dip.z), atol=1e-12)


# ---------------------------------------------------------------------
# Per-cell decay: bra-pair overlap dies exponentially with |R_g|
# ---------------------------------------------------------------------
def test_blocks_decay_with_distance(lih_primitive, lat_opts):
    """The Frobenius norm of each per-cell moment block should decrease
    on average with |R_g| (Gaussian-product overlap screening)."""
    system, basis = lih_primitive
    set_ = compute_multipole_moments_lattice(basis, system, lat_opts, L_max=2)
    # For each cell, get |R_g| and the overlap-component norm.
    data = []
    for c_idx, c in enumerate(set_.cells):
        R = float(np.linalg.norm(np.asarray(c.r_cart)))
        norm_S = float(np.linalg.norm(np.asarray(set_.blocks[c_idx][0])))
        data.append((R, norm_S))
    data.sort(key=lambda x: x[0])
    # First entry is g=0 (R=0); should have the largest norm.
    R0, norm0 = data[0]
    assert R0 == 0.0
    # The maximum norm over R > 5 bohr should be significantly less.
    norms_far = [n for r, n in data if r > 5.0]
    if norms_far:
        assert max(norms_far) < norm0


# ---------------------------------------------------------------------
# Origin-shift consistency at g=0 (sanity check)
# ---------------------------------------------------------------------
def test_dipole_origin_shift_at_g0(lih_primitive, lat_opts):
    """⟨μ | x − O_x | ν⟩ = ⟨μ | x | ν⟩ − O_x · S_μν.

    Shifting the origin by ΔO changes the dipole matrix by
    M_x(O+ΔO) = M_x(O) − ΔO_x · S.
    """
    system, basis = lih_primitive
    set_O0 = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=1, origin=[0.0, 0.0, 0.0],
    )
    set_O1 = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=1, origin=[1.5, 0.0, 0.0],
    )
    g0_idx = next(i for i, c in enumerate(set_O0.cells)
                   if (np.asarray(c.index) == np.array([0, 0, 0])).all())
    S = np.asarray(set_O0.blocks[g0_idx][0])
    mu_x_O0 = np.asarray(set_O0.blocks[g0_idx][1])
    mu_x_O1 = np.asarray(set_O1.blocks[g0_idx][1])

    # ⟨μ | x - 1.5 | ν⟩ = ⟨μ | x | ν⟩ - 1.5 · S
    expected = mu_x_O0 - 1.5 * S
    np.testing.assert_allclose(mu_x_O1, expected, atol=1e-12)
