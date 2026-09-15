"""Regressions for #289/#297: screen CCM overlap without misdiagnosing it.

Before the fix, every CCM SCF driver gated ``lindep_tol`` as a hard refusal
(``ValueError`` as soon as the threshold exceeded the overlap's smallest
eigenvalue), so the option could never screen a direction and sweeping it
produced a two-valued step instead of a screening curve. The drivers now hand
``lindep_tol`` to the canonical orthogonaliser, which projects below-threshold
overlap eigenvectors out of the Fock subspace; the run fails closed only when
screening retains fewer directions than the occupation count.

Pins:

* the dense driver sweeps: a threshold above part of the overlap spectrum
  returns a finite, converged solution whose energy differs from the
  unscreened one (the screening curve, not a two-valued step);
* the shared Python SCF loop fails closed with a named error when the
  retained subspace cannot hold the occupied orbitals;
* the C++-driver routes (DFT / scalable) actually forward the threshold
  instead of leaving it commented out (probed via the fail-closed path).
* the genuine LiH/pob-DZVP-rev2 ``(2, 2, 2)`` overlap is classified as
  indefinite, while the native host screens its occupied-rank-safe subspace;
* material indefiniteness, numerical non-positivity, positive
  near-singularity, and non-finite diagnostic inputs remain distinct.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq

from vibeqc import Atom, PeriodicSystem
from vibeqc.linear_dependence import DEFAULT_NEGATIVE_THRESHOLD
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.dft import run_ccm_rks
from vibeqc.periodic.ccm.integrals import ccm_overlap
from vibeqc.periodic.ccm.ri import _rhf_loop
from vibeqc.periodic.ccm.scf import (
    run_ccm_rhf,
    run_ccm_rhf_scalable,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _h4_unit():
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    return PeriodicSystem(3, np.diag([4.0 * BOHR, 40.0, 40.0]),
                          [Atom(1, p) for p in pos], charge=0, multiplicity=1)


def _h2_unit():
    return PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])],
                          charge=0, multiplicity=1)


def _lih_unit():
    """Primitive rocksalt LiH cell used by issue #297."""
    a = 7.72
    lattice = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    return PeriodicSystem(
        3,
        lattice,
        [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.5 * a] * 3)],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture(scope="module")
def lih_pob_indefinite_overlap():
    ccm = CCMSystem(_lih_unit(), (2, 2, 2), "pob-DZVP-rev2")
    overlap = np.asarray(ccm_overlap(ccm), dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (overlap + overlap.T))
    return ccm, overlap, eigenvalues


class _ZeroJK(vq.JKBuilder):
    """Cheap native-host probe: the overlap is the only live operator."""

    def __init__(self):
        super().__init__()

    def build_J(self, density):
        return np.zeros_like(np.asarray(density))

    def build_K(self, density):
        return np.zeros_like(np.asarray(density))

    def build_g_rhf(self, density, alpha_hf=1.0):
        del alpha_hf
        return np.zeros_like(np.asarray(density))


def _run_zero_jk_host(ccm, overlap, lindep_tol):
    options = vq.RHFOptions()
    options.max_iter = 8
    options.conv_tol_energy = 1e-12
    options.conv_tol_grad = 1e-10
    options.linear_dep_threshold = float(lindep_tol)
    return vq.run_rhf_scf_with_jk(
        ccm.basis,
        ccm.supercell.n_electrons(),
        overlap,
        -overlap,
        0.0,
        _ZeroJK(),
        options,
        np.empty((0, 0)),
    )


def test_driver_screens_instead_of_refusing():
    """A threshold above the overlap's smallest eigenvalue screens that
    direction and returns a converged solution (it used to raise ValueError).
    The screened energy departs from the unscreened one, so the knob is a
    live screening curve rather than a two-valued step."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    w = np.linalg.eigvalsh(ccm_overlap(ccm))
    e_ref = run_ccm_rhf(ccm).energy
    tol = 0.5 * (w[0] + w[1])
    res = run_ccm_rhf(ccm, lindep_tol=tol)
    assert res.converged
    assert np.isfinite(res.energy)
    assert res.energy != pytest.approx(e_ref, abs=1e-9)


def test_loop_fails_closed_below_occupation():
    """Screening everything (threshold above the whole overlap spectrum)
    leaves no occupied subspace: the shared loop raises a named error
    instead of silently truncating the occupied block."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    S = 1e-12 * np.eye(ccm.nbf)  # every direction below lindep_tol=1e-7
    h = np.eye(ccm.nbf)
    zero = lambda D: np.zeros_like(D)
    with pytest.raises(
        ValueError,
        match="screens the CCM overlap below the occupied-orbital count",
    ):
        _rhf_loop(ccm, S, h, 0.0, zero, zero, max_iter=8, conv_tol=1e-9,
                  conv_tol_grad=1e-6, diis_dim=8, lindep_tol=1e-7)


def test_dft_route_forwards_threshold_to_cpp():
    """The DFT (C++-driver) route forwards ``lindep_tol`` into
    ``RKSOptions.linear_dep_threshold``: an above-spectrum threshold makes the
    C++ canonical orthogonaliser fail closed (it used to be a no-op because
    the option was commented out and a Python guard did the refusing)."""
    ccm = CCMSystem(_h2_unit(), (1, 1, 1), "sto-3g")
    with pytest.raises(
        RuntimeError,
        match="(no non-null directions|dropped too many basis directions)",
    ):
        run_ccm_rks(ccm, "pbe", lindep_tol=10.0)


def test_lih_pob_dzvp_overlap_is_indefinite_but_occupied_rank_safe(
    lih_pob_indefinite_overlap,
):
    """The exact issue #297 spectrum is indefinite but retains 76 directions
    for 16 occupied orbitals at the production screening threshold."""
    ccm, _, eigenvalues = lih_pob_indefinite_overlap
    n_occ = ccm.supercell.n_electrons() // 2

    assert ccm.nbf == 88
    assert eigenvalues[0] == pytest.approx(-0.3535115803113, abs=1e-12)
    assert np.count_nonzero(eigenvalues < DEFAULT_NEGATIVE_THRESHOLD) == 12
    assert np.count_nonzero(eigenvalues > 1e-7) == 76
    assert n_occ == 16


def test_indefinite_overlap_diagnostic_is_not_near_singular(
    lih_pob_indefinite_overlap, monkeypatch,
):
    """A material negative mode is the paper's indefinite C-point failure,
    not ordinary positive near-linear dependence."""
    from vibeqc.periodic.ccm import integrals

    ccm, overlap, _ = lih_pob_indefinite_overlap
    monkeypatch.setattr(integrals, "ccm_overlap", lambda _: overlap)

    with pytest.raises(ValueError, match="indefinite") as exc_info:
        ccm.check_overlap_spectrum(threshold=1e-7)
    message = str(exc_info.value)
    assert "near-singular" not in message
    assert "canonical" in message


@pytest.mark.parametrize(
    ("minimum", "classification"),
    [
        (DEFAULT_NEGATIVE_THRESHOLD, "numerically non-positive"),
        (-1e-12, "numerically non-positive"),
        (0.0, "numerically non-positive"),
        (1e-9, "near-singular"),
    ],
)
def test_overlap_diagnostic_distinguishes_boundary_classes(
    minimum, classification, monkeypatch,
):
    from vibeqc.periodic.ccm import integrals

    ccm = CCMSystem(_h2_unit(), (1, 1, 1), "sto-3g")
    overlap = np.eye(ccm.nbf)
    overlap[0, 0] = minimum
    monkeypatch.setattr(integrals, "ccm_overlap", lambda _: overlap)

    with pytest.raises(ValueError, match=classification):
        ccm.check_overlap_spectrum(threshold=1e-7)


@pytest.mark.parametrize(
    "threshold",
    [True, False, -1e-7, np.nan, np.inf, -np.inf, "1e-7", 1e-7j, None],
)
def test_overlap_diagnostic_rejects_invalid_threshold_before_integrals(
    threshold, monkeypatch,
):
    from vibeqc.periodic.ccm import integrals

    ccm = CCMSystem(_h2_unit(), (1, 1, 1), "sto-3g")

    def unexpected_overlap(_):
        pytest.fail("invalid threshold reached overlap construction")

    monkeypatch.setattr(integrals, "ccm_overlap", unexpected_overlap)
    with pytest.raises(ValueError, match="finite non-negative real"):
        ccm.check_overlap_spectrum(threshold=threshold)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_overlap_diagnostic_rejects_nonfinite_matrix(bad_value, monkeypatch):
    from vibeqc.periodic.ccm import integrals

    ccm = CCMSystem(_h2_unit(), (1, 1, 1), "sto-3g")
    overlap = np.eye(ccm.nbf)
    overlap[0, 0] = bad_value
    monkeypatch.setattr(integrals, "ccm_overlap", lambda _: overlap)

    with pytest.raises(ValueError, match="non-finite entries"):
        ccm.check_overlap_spectrum(threshold=1e-7)


def test_cpp_host_screens_real_lih_indefinite_overlap(
    lih_pob_indefinite_overlap,
):
    """The production RHF host canonically screens the genuine issue #297
    metric and keeps enough rank for the occupied manifold."""
    ccm, overlap, _ = lih_pob_indefinite_overlap
    result = _run_zero_jk_host(ccm, overlap, lindep_tol=1e-7)

    assert result.converged
    assert result.n_iter == 2
    assert result.energy == pytest.approx(-32.0, abs=1e-10)
    assert np.asarray(result.mo_coeffs).shape == (88, 76)
    density = np.asarray(result.density)
    assert np.max(np.abs(density @ overlap @ density - 2.0 * density)) < 3e-12


def test_cpp_host_fails_when_real_lih_screening_loses_occupied_rank(
    lih_pob_indefinite_overlap,
):
    """Canonical screening remains fail-closed when only 15 directions are
    retained for the 16 occupied LiH orbitals."""
    ccm, overlap, eigenvalues = lih_pob_indefinite_overlap
    threshold = 0.5 * (eigenvalues[-16] + eigenvalues[-15])
    assert np.count_nonzero(eigenvalues > threshold) == 15

    with pytest.raises(
        RuntimeError,
        match=r"n_occ = 16, n_kept = 15",
    ):
        _run_zero_jk_host(ccm, overlap, lindep_tol=threshold)


def test_well_conditioned_cpp_host_screening_is_bitwise_inactive():
    """A positive-definite STO-3G control is byte-identical with the default
    threshold enabled or disabled."""
    ccm = CCMSystem(_lih_unit(), (2, 2, 2), "sto-3g")
    overlap = np.asarray(ccm_overlap(ccm), dtype=float)
    assert np.linalg.eigvalsh(overlap)[0] == pytest.approx(
        0.03973354798448, abs=1e-13
    )

    unscreened = _run_zero_jk_host(ccm, overlap, lindep_tol=0.0)
    screened = _run_zero_jk_host(ccm, overlap, lindep_tol=1e-7)
    assert unscreened.converged and screened.converged
    assert unscreened.energy == screened.energy
    assert unscreened.n_iter == screened.n_iter
    for field in ("density", "fock", "mo_coeffs", "mo_energies"):
        assert np.array_equal(
            np.asarray(getattr(unscreened, field)),
            np.asarray(getattr(screened, field)),
        )


@pytest.mark.parametrize(
    ("method", "cxx_method", "backend"),
    [
        (
            "union12",
            "bra_home_full-direct",
            "ccm-fourcenter-direct-union12-rhf",
        ),
        (
            "aiccm2026dev-a",
            "aiccm2026dev-a-direct",
            "ccm-fourcenter-direct-aiccm2026dev-a-rhf",
        ),
    ],
)
def test_both_four_center_weightings_forward_screening_to_native_host(
    method,
    cxx_method,
    backend,
    lih_pob_indefinite_overlap,
    monkeypatch,
):
    """Both Γ four-center weightings forward a non-default screening cutoff.

    The real overlap reaches the production native SCF host. Only the costly
    physical hcore, nuclear repulsion, and J/K construction are replaced by a
    synthetic ``h=-S``, zero-J/K probe; no physical energy is claimed.
    """
    from vibeqc.periodic.ccm import integrals, padded, scf

    ccm, overlap, eigenvalues = lih_pob_indefinite_overlap
    recorded_methods = []

    monkeypatch.setattr(integrals, "ccm_overlap", lambda _: overlap)
    monkeypatch.setattr(padded, "ccm_hcore", lambda _: (-overlap, None, None))
    monkeypatch.setattr(padded, "ccm_nuclear_repulsion", lambda _: 0.0)

    def synthetic_jk_builder(_, selected_method, __):
        recorded_methods.append(selected_method)
        return _ZeroJK()

    monkeypatch.setattr(scf, "_make_ccm_jk_builder", synthetic_jk_builder)

    lindep_tol = 0.1
    expected_rank = int(np.count_nonzero(eigenvalues > lindep_tol))
    assert expected_rank == 67
    result = run_ccm_rhf_scalable(
        ccm,
        method=method,
        four_center="direct",
        lindep_tol=lindep_tol,
        max_iter=8,
        conv_tol=1e-12,
        conv_tol_grad=1e-10,
    )

    assert result.converged
    assert result.n_iter == 2
    assert result.energy == pytest.approx(-32.0, abs=1e-10)
    assert result.mo_coeffs.shape == (88, expected_rank)
    assert result.backend == backend
    assert recorded_methods == [cxx_method]
