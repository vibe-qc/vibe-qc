"""Partial Hessian (frozen atoms) — PHVA for slab TS characterisation.

``HessianFDOptions(frozen_indices=[...])`` holds the listed atoms fixed,
so only the unfrozen atoms are displaced and the result is the
``(3M, 3M)`` sub-block of the full Hessian over the M unfrozen atoms —
partial-Hessian vibrational analysis. The frozen atoms anchor the
system, so the 3M modes are all vibrational (no trans/rot zero modes)
and downstream thermochemistry is vibrational-only.

These pin: the sub-block equals the corresponding block of the full
Hessian; the partial frequencies have no spurious zero modes; the
displacement count drops to 6M; thermochemistry zeros out the
gas-phase translational/rotational partition functions; and the full
(unfrozen) Hessian + thermo are unchanged (regression).
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.hessian import HessianFDOptions, compute_hessian_fd
from vibeqc.thermo import ThermoOptions, compute_thermochemistry


def _h2o() -> vq.Molecule:
    # Bent H2O in bohr (not a stationary point — fine: we test the
    # machinery, and the sub-block identity holds at any geometry).
    return vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.22]),
         vq.Atom(1, [0.0, 1.43, -0.9]),
         vq.Atom(1, [0.0, -1.43, -0.9])], 0, 1)


@pytest.fixture(scope="module")
def hessians():
    mol = _h2o()
    full = compute_hessian_fd(mol, "sto-3g", method="RHF")
    partial = compute_hessian_fd(
        mol, "sto-3g", method="RHF",
        hessian_options=HessianFDOptions(frozen_indices=[0]))  # freeze O
    return mol, full, partial


# ---------------------------------------------------------------------------
# Sub-block identity + shape
# ---------------------------------------------------------------------------


def test_partial_hessian_matches_full_subblock(hessians):
    _, full, partial = hessians
    assert partial.hessian.shape == (6, 6)
    assert full.hessian.shape == (9, 9)
    # The partial Hessian IS the full Hessian's unfrozen×unfrozen block.
    uf = [3 * i + c for i in (1, 2) for c in range(3)]
    sub = full.hessian[np.ix_(uf, uf)]
    assert np.max(np.abs(partial.hessian - sub)) < 1e-6


def test_partial_hessian_metadata(hessians):
    _, full, partial = hessians
    assert list(partial.unfrozen_indices) == [1, 2]
    assert full.unfrozen_indices is None
    assert partial.n_displacements == 12      # 6M, M=2
    assert full.n_displacements == 18         # 6N, N=3
    assert partial.masses_amu.shape == (2,)   # unfrozen masses only


def test_partial_hessian_has_no_zero_modes(hessians):
    _, full, partial = hessians
    # Anchored system: all 6 modes are vibrational, none ~0.
    assert len(partial.frequencies_cm1) == 6
    assert int(np.sum(np.abs(partial.frequencies_cm1) < 1.0)) == 0
    # Full molecule: 6 trans/rot zero modes (projected) out of 9.
    assert int(np.sum(np.abs(full.frequencies_cm1) < 1.0)) == 6


# ---------------------------------------------------------------------------
# Thermochemistry: vibrational-only for a frozen system
# ---------------------------------------------------------------------------


def test_partial_thermo_is_vibrational_only(hessians):
    mol, _, partial = hessians
    th = compute_thermochemistry(mol, partial,
                                 options=ThermoOptions(temperature=298.15))
    assert th.rotor_type == "frozen"
    assert th.e_trans == 0.0 and th.e_rot == 0.0
    assert th.s_trans == 0.0 and th.s_rot == 0.0
    assert th.cv_trans == 0.0 and th.cv_rot == 0.0
    # No gas-phase PV term for an anchored system: H == U.
    assert th.h_thermal == pytest.approx(th.u_thermal, abs=1e-12)
    # Vibrational ZPE/entropy still contribute.
    assert th.zpe > 0.0


def test_full_thermo_unchanged(hessians):
    # Regression: a full (unfrozen) Hessian still gets gas-phase trans/rot.
    mol, full, _ = hessians
    th = compute_thermochemistry(
        mol, full, options=ThermoOptions(temperature=298.15, symmetry_number=2))
    assert th.rotor_type == "nonlinear"
    assert th.e_trans > 0.0 and th.e_rot > 0.0 and th.s_trans > 0.0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_frozen_indices_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        compute_hessian_fd(_h2o(), "sto-3g", method="RHF",
                           hessian_options=HessianFDOptions(frozen_indices=[5]))


def test_all_atoms_frozen_raises():
    with pytest.raises(ValueError, match="every atom is frozen"):
        compute_hessian_fd(_h2o(), "sto-3g", method="RHF",
                           hessian_options=HessianFDOptions(frozen_indices=[0, 1, 2]))
