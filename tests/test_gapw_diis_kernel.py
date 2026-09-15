"""The GAPW / GPW / RSGAPW drivers extrapolate with the canonical C++ DIIS.

Until v0.15.x the ``run_periodic_*_gapw`` / ``_gpw`` / ``_rsgapw`` family kept a
private numpy Pulay history and rebuilt the bordered B-matrix by hand at six
call sites. This module pins the two things that had to hold for those six
sites to be replaceable by ``vibeqc::DIIS``:

* ``DIIS.extrapolate`` reproduces the retired closed-shell B-matrix, and
  ``DIIS.extrapolate_spin_coupled`` reproduces the retired *joint* open-shell
  B-matrix (the one that summed the α and β Frobenius products into a single
  coefficient set), over a realistic converging history.
* No hand-rolled Pulay solve has come back into the GAPW family.

The equivalence is exact rather than statistical: for real symmetric error
matrices ``np.sum(e_i * e_j)`` *is* the Frobenius inner product ``Tr(e_iᵀ e_j)``
that the kernel accumulates in its Gram matrix, and stacking the spins
vertically makes the kernel's single B-matrix accumulate
``Tr(e_α_iᵀ e_α_j) + Tr(e_β_iᵀ e_β_j)`` -- termwise the retired joint build.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from vibeqc import _vibeqc_core as _core

# Agreement between the kernel and the retired numpy build over a 10-iterate
# history. Both solve the same bordered system; they differ only in the
# factorisation (Eigen full-pivot LU vs LAPACK gesv), so the gap is pure
# round-off on a well-conditioned solve.
TOL = 1.0e-11

SUBSPACE = 8
GAPW_DRIVER_FILES = (
    "periodic_gapw_j.py",
    "periodic_gapw_augment.py",
    "periodic_gapw_open_shell.py",
    "periodic_gapw_range_sep.py",
)


# --------------------------------------------------------------------------
# The retired numpy Pulay build, transcribed verbatim from the six call sites
# it is replacing (periodic_gapw_j.py:2318-2334 and siblings, as of ff44c66e).
# --------------------------------------------------------------------------
def _retired_pulay_closed(diis_F, diis_err, F, err, subspace):
    diis_F.append(F.copy())
    diis_err.append(err.copy())
    if len(diis_F) > subspace:
        diis_F.pop(0)
        diis_err.pop(0)
    n_diis = len(diis_F)
    B = np.zeros((n_diis + 1, n_diis + 1))
    for i in range(n_diis):
        for j in range(n_diis):
            B[i, j] = float(np.sum(diis_err[i] * diis_err[j]))
        B[i, -1] = -1.0
        B[-1, i] = -1.0
    rhs = np.zeros(n_diis + 1)
    rhs[-1] = -1.0
    try:
        coeffs = np.linalg.solve(B, rhs)
        return sum(coeffs[i] * diis_F[i] for i in range(n_diis))
    except np.linalg.LinAlgError:
        return F  # Skip DIIS this iter; use raw F.


def _retired_pulay_open(dF_a, de_a, dF_b, de_b, F_a, F_b, e_a, e_b, subspace):
    dF_a.append(F_a.copy())
    de_a.append(e_a.copy())
    dF_b.append(F_b.copy())
    de_b.append(e_b.copy())
    if len(dF_a) > subspace:
        dF_a.pop(0)
        de_a.pop(0)
        dF_b.pop(0)
        de_b.pop(0)
    n_diis = len(dF_a)
    B = np.zeros((n_diis + 1, n_diis + 1))
    for i in range(n_diis):
        for j in range(n_diis):
            B[i, j] = float(
                np.sum(de_a[i] * de_a[j]) + np.sum(de_b[i] * de_b[j])
            )
        B[i, -1] = -1.0
        B[-1, i] = -1.0
    rhs = np.zeros(n_diis + 1)
    rhs[-1] = -1.0
    try:
        coeffs = np.linalg.solve(B, rhs)
        return (
            sum(coeffs[i] * dF_a[i] for i in range(n_diis)),
            sum(coeffs[i] * dF_b[i] for i in range(n_diis)),
        )
    except np.linalg.LinAlgError:
        return F_a, F_b


def _sym(rng, n):
    a = rng.standard_normal((n, n))
    return a + a.T


def _converging_history(rng, n_bf, n_iter):
    """A history whose error decays geometrically but stays full rank.

    A history of *collinear* errors (e_i = eps_i * E for one fixed E) makes the
    bordered system singular for n >= 2, which is precisely the branch where
    the kernel's blow-up guard and numpy's LinAlgError take deliberately
    different exits. Independent directions per iterate keep the comparison on
    the healthy branch, which is the one every converging SCF actually walks.
    """
    F_conv = _sym(rng, n_bf)
    out = []
    for k in range(n_iter):
        eps = 0.5**k
        out.append((F_conv + eps * _sym(rng, n_bf), eps * _sym(rng, n_bf)))
    return out


@pytest.mark.parametrize("n_bf, n_iter", [(6, 10), (11, 12)])
def test_closed_shell_matches_retired_numpy_build(n_bf, n_iter):
    """DIIS.extrapolate reproduces the retired closed-shell B-matrix."""
    rng = np.random.default_rng(20260710)
    history = _converging_history(rng, n_bf, n_iter)

    diis = _core.DIIS(SUBSPACE)
    ref_F: list[np.ndarray] = []
    ref_e: list[np.ndarray] = []

    for F, err in history:
        got = diis.extrapolate(F, err)
        want = _retired_pulay_closed(ref_F, ref_e, F, err, SUBSPACE)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=TOL)


@pytest.mark.parametrize("n_bf, n_iter", [(6, 10), (11, 12)])
def test_spin_coupled_matches_retired_joint_b_matrix(n_bf, n_iter):
    """extrapolate_spin_coupled reproduces the retired joint (α+β) B-matrix.

    One coefficient set drives both spins, exactly as the hand-rolled build did.
    """
    rng = np.random.default_rng(20260711)
    hist_a = _converging_history(rng, n_bf, n_iter)
    hist_b = _converging_history(rng, n_bf, n_iter)

    diis = _core.DIIS(SUBSPACE)
    rF_a: list[np.ndarray] = []
    re_a: list[np.ndarray] = []
    rF_b: list[np.ndarray] = []
    re_b: list[np.ndarray] = []

    for (F_a, e_a), (F_b, e_b) in zip(hist_a, hist_b):
        got_a, got_b = diis.extrapolate_spin_coupled(F_a, F_b, e_a, e_b)
        want_a, want_b = _retired_pulay_open(
            rF_a, re_a, rF_b, re_b, F_a, F_b, e_a, e_b, SUBSPACE
        )
        np.testing.assert_allclose(got_a, want_a, rtol=0.0, atol=TOL)
        np.testing.assert_allclose(got_b, want_b, rtol=0.0, atol=TOL)


def test_spin_coupling_is_not_two_independent_diis():
    """The joint B-matrix is what couples the spins; pin that it is used.

    Extrapolating α and β through *separate* DIIS instances yields a different
    Fock pair than the coupled call, so a future refactor that quietly swaps in
    two independent histories cannot pass silently.
    """
    rng = np.random.default_rng(20260712)
    hist_a = _converging_history(rng, 7, 6)
    hist_b = _converging_history(rng, 7, 6)

    coupled = _core.DIIS(SUBSPACE)
    solo_a = _core.DIIS(SUBSPACE)
    solo_b = _core.DIIS(SUBSPACE)

    diverged = False
    for (F_a, e_a), (F_b, e_b) in zip(hist_a, hist_b):
        c_a, _c_b = coupled.extrapolate_spin_coupled(F_a, F_b, e_a, e_b)
        s_a = solo_a.extrapolate(F_a, e_a)
        solo_b.extrapolate(F_b, e_b)
        if not np.allclose(c_a, s_a, rtol=0.0, atol=1e-8):
            diverged = True
    assert diverged, "coupled and independent DIIS must not coincide"


def test_first_iterate_is_returned_unchanged():
    """A one-deep history extrapolates to the raw Fock (c_0 = 1)."""
    rng = np.random.default_rng(20260713)
    F, err = _converging_history(rng, 5, 1)[0]
    np.testing.assert_allclose(
        _core.DIIS(SUBSPACE).extrapolate(F, err), F, rtol=0.0, atol=0.0
    )


def test_subspace_cap_drops_the_oldest_iterate():
    """Beyond the cap the kernel keeps a FIFO window, as the numpy code did."""
    rng = np.random.default_rng(20260714)
    history = _converging_history(rng, 5, 9)
    small = 3

    diis = _core.DIIS(small)
    ref_F: list[np.ndarray] = []
    ref_e: list[np.ndarray] = []
    for F, err in history:
        got = diis.extrapolate(F, err)
        want = _retired_pulay_closed(ref_F, ref_e, F, err, small)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=TOL)
    assert diis.subspace_size == small


def test_gapw_drivers_hold_no_hand_rolled_pulay_solve():
    """Guard the retirement: no GAPW driver rebuilds a bordered B-matrix.

    Pins the CLAUDE.md section 10 invariant (one canonical DIIS) mechanically,
    so a future edit cannot reintroduce a private Pulay history without a
    failing test.
    """
    pkg = pathlib.Path(__file__).resolve().parents[1] / "python" / "vibeqc"
    # The border row/column assignment of a bordered Pulay system, and the
    # right-hand side that goes with it. Either is a hand-rolled solve.
    border = re.compile(r"B\[\s*-1\s*,\s*i\s*\]|B\[\s*i\s*,\s*-1\s*\]")
    rhs = re.compile(r"rhs\[\s*-1\s*\]\s*=\s*-1\.0")

    offenders = []
    for name in GAPW_DRIVER_FILES:
        src = (pkg / name).read_text()
        if border.search(src) or rhs.search(src):
            offenders.append(name)
    assert not offenders, (
        "hand-rolled Pulay B-matrix reintroduced in: "
        + ", ".join(offenders)
        + " -- route through vibeqc::DIIS (_core.DIIS) instead"
    )
