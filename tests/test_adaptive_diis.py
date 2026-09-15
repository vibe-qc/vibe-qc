"""Adaptive-depth commutator-DIIS — R-CDIIS + AD-CDIIS.

Chupin, Dupuy, Legendre & Séré, "Convergence analysis of adaptive DIIS
algorithms with application to electronic ground state calculations",
ESAIM: M2AN 55, 2785 (2021), doi:10.1051/m2an/2021069.

Both variants reuse Pulay's commutator-DIIS extrapolation unchanged; only
the *history depth* is chosen adaptively. These tests pin:

  1. The two new option fields exist with the paper's default (1e-4) on
     every molecular SCF options struct.

  2. The user-facing keyword strings resolve to the right enum values.

  3. The C++ DIIS depth policies behave per the algorithms:
       * FIXED    — unchanged FIFO cap (regression guard).
       * RESTART  — Algorithm 3: colinear error differences trigger a
                    restart (depth collapses to 1); independent ones grow.
       * ADAPTIVE — Algorithm 4: a stored residual more than 1/δ times the
                    current one is dropped; comparable residuals are kept.

  4. R-CDIIS and AD-CDIIS SCFs converge to the *same* fixed point as plain
     DIIS across RHF / UHF / RKS / UKS. The adaptive depth changes the path
     through the SCF manifold, not the converged energy.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    BasisSet,
    DIIS,
    DIISDepthPolicy,
    RHFOptions,
    RKSOptions,
    SCFAccelerator,
    UHFOptions,
    UKSOptions,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)
from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_atoms():
    return [
        (8, [0.0, 0.0, 0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0, 0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


def _oh_radical_atoms():
    return [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ]


# ---------------------------------------------------------------------------
# 1. Option-field defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RHFOptions, UHFOptions, RKSOptions, UKSOptions])
def test_adaptive_diis_option_defaults(cls):
    opts = cls()
    assert opts.diis_restart_tau == pytest.approx(1e-4)
    assert opts.diis_adaptive_delta == pytest.approx(1e-4)
    # Fields are writable.
    opts.diis_restart_tau = 1e-3
    opts.diis_adaptive_delta = 1e-2
    assert opts.diis_restart_tau == pytest.approx(1e-3)
    assert opts.diis_adaptive_delta == pytest.approx(1e-2)


# ---------------------------------------------------------------------------
# 2. Keyword-string resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["r_cdiis", "R_CDIIS", "rcdiis", "r-cdiis", "restarted_cdiis",
     "restarted_diis"],
)
def test_r_cdiis_keyword_resolves(name):
    assert vq.scf_accelerator_from_string(name) == SCFAccelerator.R_CDIIS


@pytest.mark.parametrize(
    "name",
    ["ad_cdiis", "AD_CDIIS", "adcdiis", "ad-cdiis", "adaptive_cdiis",
     "adaptive_diis"],
)
def test_ad_cdiis_keyword_resolves(name):
    assert vq.scf_accelerator_from_string(name) == SCFAccelerator.AD_CDIIS


# ---------------------------------------------------------------------------
# 3. C++ depth-policy behaviour
# ---------------------------------------------------------------------------

def _sym(n, rng):
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


def test_fixed_policy_is_default_and_caps_history():
    d = DIIS(max_subspace=4)
    assert d.policy == DIISDepthPolicy.FIXED
    rng = np.random.default_rng(0)
    F = _sym(3, rng)
    for _ in range(8):
        d.extrapolate(F, _sym(3, rng))
    assert d.subspace_size == 4


def test_restart_policy_restarts_on_colinear_error_differences():
    """Algorithm 3: when the stored error differences are colinear, the
    newest difference lies in their span, the orthogonal residual is ~0,
    and τ‖s‖ > ‖(id−Π)s‖ fires — the history restarts to depth 1."""
    V = np.array([[1.0, 0.5], [0.5, -1.0]])
    F = np.eye(2)
    d = DIIS(max_subspace=8, policy=DIISDepthPolicy.RESTART,
             adaptive_param=1e-4)
    d.extrapolate(F, 3.0 * V)   # n=1
    d.extrapolate(F, 2.0 * V)   # n=2 (no test yet)
    assert d.subspace_size == 2
    d.extrapolate(F, 1.0 * V)   # n=3: differences colinear -> restart
    assert d.subspace_size == 1


def test_restart_policy_grows_on_independent_errors():
    """Independent error directions are not linearly dependent, so no
    restart fires and the depth grows."""
    rng = np.random.default_rng(3)
    n = 6  # big enough that random symmetric errors are affinely independent
    d = DIIS(max_subspace=16, policy=DIISDepthPolicy.RESTART,
             adaptive_param=1e-4)
    for _ in range(4):
        d.extrapolate(np.eye(n), _sym(n, rng))
    assert d.subspace_size == 4


def test_adaptive_policy_drops_stale_large_residuals():
    """Algorithm 4: a stored residual larger than 1/δ times the current
    one violates δ‖r_i‖ < ‖r_new‖ and is dropped."""
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    F = np.eye(2)
    d = DIIS(max_subspace=8, policy=DIISDepthPolicy.ADAPTIVE,
             adaptive_param=1e-4)
    d.extrapolate(F, 1.0 * V)      # ‖r‖ = sqrt(2)
    d.extrapolate(F, 1.0 * V)
    # Current residual 1e-6 * ‖V‖; old residuals are ~1e6x larger than
    # current, so δ‖r_old‖ = 1e-4 * O(1) is NOT < 1e-6 -> both dropped.
    d.extrapolate(F, 1.0e-6 * V)
    assert d.subspace_size == 1


def test_adaptive_policy_keeps_comparable_residuals():
    """When residuals are comparable, δ‖r_i‖ < ‖r_new‖ holds for all of
    them (δ = 1e-4 « 1) and the window keeps growing."""
    rng = np.random.default_rng(5)
    d = DIIS(max_subspace=16, policy=DIISDepthPolicy.ADAPTIVE,
             adaptive_param=1e-4)
    base = _sym(4, rng)
    for _ in range(5):
        # Norms within a factor of ~2 of each other -> all retained.
        e = base + 0.1 * _sym(4, rng)
        d.extrapolate(np.eye(4), e)
    assert d.subspace_size == 5


def test_adaptive_param_must_be_positive():
    with pytest.raises((ValueError, RuntimeError)):
        DIIS(max_subspace=8, policy=DIISDepthPolicy.RESTART,
             adaptive_param=0.0)


# ---------------------------------------------------------------------------
# 4. SCF parity — same fixed point as plain DIIS
# ---------------------------------------------------------------------------

@pytest.fixture
def h2o_basis():
    mol = make_molecule(_h2o_atoms())
    return mol, BasisSet(mol, "sto-3g")


def _rhf(mol, basis, method):
    opts = RHFOptions()
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.scf_accelerator = method
    return run_rhf(mol, basis, opts)


@pytest.mark.parametrize(
    "variant", [SCFAccelerator.R_CDIIS, SCFAccelerator.AD_CDIIS]
)
def test_rhf_h2o_adaptive_matches_diis(h2o_basis, variant):
    mol, basis = h2o_basis
    ref = _rhf(mol, basis, SCFAccelerator.DIIS)
    got = _rhf(mol, basis, variant)
    assert ref.converged and got.converged
    assert got.energy == pytest.approx(ref.energy, abs=1e-9)


@pytest.mark.parametrize(
    "variant", [SCFAccelerator.R_CDIIS, SCFAccelerator.AD_CDIIS]
)
def test_uhf_oh_adaptive_matches_diis(variant):
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UHFOptions()
    common.max_iter = 200
    common.conv_tol_energy = 1e-10
    common.conv_tol_grad = 1e-8
    common.scf_accelerator = SCFAccelerator.DIIS
    ref = run_uhf(mol, basis, common)
    common.scf_accelerator = variant
    got = run_uhf(mol, basis, common)
    assert ref.converged and got.converged
    assert got.energy == pytest.approx(ref.energy, abs=1e-9)
    assert got.s_squared == pytest.approx(ref.s_squared, abs=1e-6)


@pytest.mark.parametrize(
    "variant", [SCFAccelerator.R_CDIIS, SCFAccelerator.AD_CDIIS]
)
def test_rks_h2o_pbe_adaptive_matches_diis(h2o_basis, variant):
    mol, basis = h2o_basis
    common = RKSOptions()
    common.functional = "PBE"
    common.max_iter = 100
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    ref = run_rks(mol, basis, common)
    common.scf_accelerator = variant
    got = run_rks(mol, basis, common)
    assert ref.converged and got.converged
    assert got.energy == pytest.approx(ref.energy, abs=1e-8)


@pytest.mark.parametrize(
    "variant", [SCFAccelerator.R_CDIIS, SCFAccelerator.AD_CDIIS]
)
def test_uks_oh_pbe_adaptive_matches_diis(variant):
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    common = UKSOptions()
    common.functional = "PBE"
    common.max_iter = 200
    common.conv_tol_energy = 1e-9
    common.conv_tol_grad = 1e-7
    common.scf_accelerator = SCFAccelerator.DIIS
    ref = run_uks(mol, basis, common)
    common.scf_accelerator = variant
    got = run_uks(mol, basis, common)
    assert ref.converged and got.converged
    assert got.energy == pytest.approx(ref.energy, abs=1e-8)
