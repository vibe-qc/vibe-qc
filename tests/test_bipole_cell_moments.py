"""Tests for BIPOLE Phase 4a: cell-level multipole moments from
``LatticeMatrixSet`` density + ``LatticeMultipoleSet`` shell-pair moments."""
from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import CoulombMethod, InitialGuess, LatticeSumOptions
from vibeqc._vibeqc_core import (
    compute_multipole_moments_lattice,
    compute_overlap_lattice,
)
from vibeqc.bipole_cell_moments import (
    CellMultipoleMoments,
    cartesian_component_indices,
    cartesian_component_label,
    compute_cell_multipole_moments,
)
from vibeqc.guess import initial_density_closed_shell


ANG2BOHR = 1.0 / 0.529177210903


@pytest.fixture
def lih_primitive():
    """LiH FCC primitive — smallest available periodic system."""
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
# Component label helpers
# ---------------------------------------------------------------------
def test_cartesian_component_label():
    assert cartesian_component_label(0, 0, 0) == "S"
    assert cartesian_component_label(1, 0, 0) == "x"
    assert cartesian_component_label(0, 1, 0) == "y"
    assert cartesian_component_label(0, 0, 1) == "z"
    assert cartesian_component_label(2, 0, 0) == "xx"
    assert cartesian_component_label(1, 1, 0) == "xy"
    assert cartesian_component_label(1, 1, 1) == "xyz"
    assert cartesian_component_label(0, 0, 2) == "zz"


def test_cartesian_component_indices_L2_count():
    indices = cartesian_component_indices(2)
    assert len(indices) == 10
    assert indices[0] == (0, 0, 0)
    assert indices[1] == (1, 0, 0)
    assert indices[-1] == (0, 0, 2)


def test_cartesian_component_indices_invalid():
    # L=0 is valid (monopole): a single (0, 0, 0) component.
    assert cartesian_component_indices(0) == [(0, 0, 0)]
    # Negative L returns empty list (silently).
    assert cartesian_component_indices(-1) == []
    # Out-of-range L returns empty beyond L_max components.


# ---------------------------------------------------------------------
# CellMultipoleMoments dataclass behaviour
# ---------------------------------------------------------------------
def test_cell_multipole_moments_indexing():
    moments = np.arange(10, dtype=float)
    cell_M = CellMultipoleMoments(
        L_max=2, n_components=10, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    assert cell_M["S"] == 0.0
    assert cell_M["x"] == 1.0
    assert cell_M["xx"] == 4.0
    assert cell_M["zz"] == 9.0
    assert cell_M[5] == 5.0  # xy
    with pytest.raises(KeyError):
        _ = cell_M["xxxx"]


def test_spheropole_trace_L_max_lt_2_returns_none():
    moments = np.array([1.0, 0.5, 0.5, 0.5])
    cell_M = CellMultipoleMoments(
        L_max=1, n_components=4, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    assert cell_M.get_spheropole_trace() is None


def test_spheropole_trace_L_max_ge_2():
    # moments = [S, x, y, z, xx, xy, xz, yy, yz, zz]
    moments = np.array([
        20.0,        # S (= electron count for neutral cell)
        0.0, 0.0, 0.0,  # dipole (zero by symmetry for centered)
        4.0,         # xx
        0.0, 0.0,    # xy, xz
        5.0,         # yy
        0.0,         # yz
        7.0,         # zz
    ])
    cell_M = CellMultipoleMoments(
        L_max=2, n_components=10, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    assert math.isclose(cell_M.get_spheropole_trace(), 4.0 + 5.0 + 7.0)


# ---------------------------------------------------------------------
# Monopole sanity: M_S = N_electrons for SAD on a closed-shell neutral cell
# ---------------------------------------------------------------------
def test_monopole_at_iter1_SAD_equals_electron_count(lih_primitive, lat_opts):
    """The overlap component ``M_S = Σ_μν P_μν · S_μν = tr(D·S)`` is
    exactly the cell's electron count. For LiH STO-3G (Li + H) we expect
    M_S = 4 electrons at iter 1 SAD."""
    system, basis = lih_primitive
    n_occ = system.n_electrons() // 2

    # SAD density placed at g=0 only — typical iter 1 BIPOLE setup.
    M_lattice = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=2,
    )
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(), basis, n_occ,
        InitialGuess.SAD, is_periodic=True,
    )
    # Build a LatticeMatrixSet with SAD at g=0, zeros elsewhere.
    P_real = S_lat
    for g_idx in range(len(P_real.cells)):
        is_g0 = (np.asarray(P_real.cells[g_idx].index) == np.array([0, 0, 0])).all()
        if is_g0:
            P_real.set_block(g_idx, np.asarray(D_sad, dtype=float))
        else:
            P_real.set_block(
                g_idx, np.zeros_like(np.asarray(D_sad), dtype=float),
            )

    cell_M = compute_cell_multipole_moments(P_real, M_lattice)
    n_electrons_actual = cell_M["S"]
    # SAD over-normalises slightly (atomic densities don't sum exactly to
    # integer molecular count). Allow up to 0.5 e- discrepancy.
    print(f"  LiH cell electron count via cell-multipole M_S = {n_electrons_actual:.6f}")
    print(f"  expected (n_electrons = 4) — SAD tolerance ~0.5")
    assert abs(n_electrons_actual - 4.0) < 0.5


# ---------------------------------------------------------------------
# Shape + origin propagation
# ---------------------------------------------------------------------
def test_origin_propagates(lih_primitive, lat_opts):
    """The origin of the returned CellMultipoleMoments matches the
    LatticeMultipoleSet origin used in construction."""
    system, basis = lih_primitive
    origin = (1.5, -2.0, 3.7)
    M_lattice = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=2, origin=list(origin),
    )
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(), basis, n_occ,
        InitialGuess.SAD, is_periodic=True,
    )
    P_real = S_lat
    for g_idx in range(len(P_real.cells)):
        is_g0 = (np.asarray(P_real.cells[g_idx].index) == np.array([0, 0, 0])).all()
        if is_g0:
            P_real.set_block(g_idx, np.asarray(D_sad, dtype=float))
        else:
            P_real.set_block(
                g_idx, np.zeros_like(np.asarray(D_sad), dtype=float),
            )
    cell_M = compute_cell_multipole_moments(P_real, M_lattice)
    assert cell_M.origin == origin
    assert cell_M.L_max == 2
    assert cell_M.n_components == 10


# ---------------------------------------------------------------------
# Origin-shift identity: monopole is invariant, dipole shifts by O·N
# ---------------------------------------------------------------------
def test_origin_shift_invariants(lih_primitive, lat_opts):
    """Shifting the expansion origin doesn't change the electron count
    (monopole component) — only the higher-multipole components change
    via the standard polynomial-shift relation."""
    system, basis = lih_primitive
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(), basis, n_occ,
        InitialGuess.SAD, is_periodic=True,
    )
    P_real = S_lat
    for g_idx in range(len(P_real.cells)):
        is_g0 = (np.asarray(P_real.cells[g_idx].index) == np.array([0, 0, 0])).all()
        if is_g0:
            P_real.set_block(g_idx, np.asarray(D_sad, dtype=float))
        else:
            P_real.set_block(
                g_idx, np.zeros_like(np.asarray(D_sad), dtype=float),
            )

    M_O0 = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=1, origin=[0.0, 0.0, 0.0],
    )
    M_O1 = compute_multipole_moments_lattice(
        basis, system, lat_opts, L_max=1, origin=[2.0, 0.0, 0.0],
    )
    cell_M_O0 = compute_cell_multipole_moments(P_real, M_O0)
    cell_M_O1 = compute_cell_multipole_moments(P_real, M_O1)

    # Monopole invariant under origin shift.
    assert math.isclose(cell_M_O0["S"], cell_M_O1["S"], rel_tol=1e-12)
    # Dipole-x shift: M_x(O+ΔO) = M_x(O) - ΔO_x · M_S
    expected_M_x_O1 = cell_M_O0["x"] - 2.0 * cell_M_O0["S"]
    assert math.isclose(cell_M_O1["x"], expected_M_x_O1, rel_tol=1e-10)
