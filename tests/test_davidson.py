"""Unit tests for the Davidson iterative diagonalizer (Phase D3).

Covers:
- Real-symmetric Davidson on small Fock-like matrices
- Edge cases: identity matrix, tiny matrices
- End-to-end SCF parity: molecular RHF with STO-3G and def2-SVP
- Python bindings roundtrip

Note: the Davidson diagonal preconditioner requires strong diagonal
dominance (like the Fock matrix in an orthogonalised basis). Random
or weakly-diagonally-dominant matrices may not converge — this is
expected behaviour, not a bug.  The SCF integration tests are the
real measure of correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc._vibeqc_core as _core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fock_like_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Build a real symmetric matrix with Fock-like structure:
    quadratic diagonal growth (like orbital energies) and small
    off-diagonal couplings that decay with index separation."""
    diag = (np.arange(1, n + 1, dtype=float) ** 2) * 0.5 + 2.0
    A = np.diag(diag)
    for i in range(n):
        for j in range(i + 1, n):
            val = rng.normal(0, 0.01 / ((abs(i - j) + 1) ** 2))
            A[i, j] = val
            A[j, i] = val
    return A


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_davidson_small_fock_like() -> None:
    """Davidson on a small Fock-like matrix matches numpy eigh."""
    rng = np.random.default_rng(42)
    n = 20
    A = _fock_like_matrix(n, rng)

    opts = _core.DavidsonOptions()
    opts.n_eig = 5
    opts.conv_tol = 1e-10
    opts.max_iter = 100

    res = _core.davidson_solve(A, opts)
    assert res.converged, f"Davidson did not converge (n_iter={res.n_iter})"

    np_eigs = np.linalg.eigh(A)[0][:5]
    assert np.max(np.abs(res.eigenvalues - np_eigs)) < 1e-8

    # Residual norms
    for i in range(5):
        r = A @ res.eigenvectors[:, i] - res.eigenvalues[i] * res.eigenvectors[:, i]
        assert np.linalg.norm(r) < 1e-7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_davidson_identity_matrix() -> None:
    """Extracting 3 eigenvalues from the identity matrix."""
    n = 20
    A = np.eye(n)
    opts = _core.DavidsonOptions()
    opts.n_eig = 3
    opts.conv_tol = 1e-12
    res = _core.davidson_solve(A, opts)
    assert res.converged
    assert np.max(np.abs(res.eigenvalues - 1.0)) < 1e-10


def test_davidson_tiny_matrix() -> None:
    """A 3x3 matrix: Davidson should still converge."""
    A = np.array([[2.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 3.0]])
    opts = _core.DavidsonOptions()
    opts.n_eig = 2
    opts.conv_tol = 1e-12
    res = _core.davidson_solve(A, opts)
    assert res.converged
    np_e = np.linalg.eigh(A)[0][:2]
    assert np.max(np.abs(res.eigenvalues - np_e)) < 1e-10


# ---------------------------------------------------------------------------
# SCF integration smoke tests -- the real measure of correctness
# ---------------------------------------------------------------------------

def test_rhf_davidson_matches_full_diag_sto3g() -> None:
    """Molecular RHF with Davidson gives same energy as full diagonalization
    on STO-3G (CO molecule)."""
    import tempfile

    from vibeqc.molecule import Atom, Molecule
    from vibeqc.runner import run_job

    atoms = [Atom(6, (0.0, 0.0, 0.0)), Atom(8, (0.0, 0.0, 1.2))]
    mol = Molecule(atoms=atoms, charge=0, multiplicity=1)

    opts = _core.RHFOptions()
    opts.use_davidson = True
    opts.davidson_min_dim = 1

    with tempfile.TemporaryDirectory() as tmp:
        r_std = run_job(mol, basis="sto-3g", method="rhf", output=tmp, verbose=0)
        r_dav = run_job(
            mol, basis="sto-3g", method="rhf", output=tmp,
            verbose=0, rhf_options=opts,
        )

    assert r_std.converged
    assert r_dav.converged
    assert abs(r_std.energy - r_dav.energy) < 1e-10


def test_rhf_davidson_matches_full_diag_def2svp() -> None:
    """Davidson RHF on def2-SVP (24 AOs for water) matches full diag."""
    import tempfile

    from vibeqc.molecule import Atom, Molecule
    from vibeqc.runner import run_job

    atoms = [
        Atom(8, (0.0, 0.0, 0.1173)),
        Atom(1, (0.0, 0.7572, -0.4692)),
        Atom(1, (0.0, -0.7572, -0.4692)),
    ]
    mol = Molecule(atoms=atoms, charge=0, multiplicity=1)

    opts = _core.RHFOptions()
    opts.use_davidson = True
    opts.davidson_min_dim = 1

    with tempfile.TemporaryDirectory() as tmp:
        r_std = run_job(mol, basis="def2-svp", method="rhf", output=tmp, verbose=0)
        r_dav = run_job(
            mol, basis="def2-svp", method="rhf", output=tmp,
            verbose=0, rhf_options=opts,
        )

    assert r_std.converged
    assert r_dav.converged
    assert abs(r_std.energy - r_dav.energy) < 1e-8


def test_davidson_options_roundtrip() -> None:
    """DavidsonOptions Python bindings work correctly."""
    opts = _core.DavidsonOptions()
    assert opts.n_eig == 0
    assert opts.conv_tol == 1e-7
    assert opts.max_iter == 200
    assert opts.preshift == 1e-6

    opts.n_eig = 10
    opts.conv_tol = 1e-9
    opts.max_iter = 50
    assert opts.n_eig == 10
    assert opts.conv_tol == 1e-9
    assert opts.max_iter == 50


# ---------------------------------------------------------------------------
# GitLab #503 -- the partial-spectrum path.
#
# The tests above all request the FULL spectrum (n_eig = n), which converges in
# one iteration and never exercises subspace expansion.  Two defects therefore
# lived in the expansion path unnoticed:
#
#   1. Correction vectors were orthogonalised against the existing subspace V
#      but not against each other.  Corrections for different unconverged roots
#      are preconditioned residuals of neighbouring Ritz pairs and are often
#      near-parallel, so the basis degenerated, H_proj = V^T A V acquired
#      spurious near-zero eigenvalues, and -- because lambda is sorted
#      ascending -- those sorted BELOW the true roots.  The solver returned
#      zeros and null eigenvectors in place of the lowest eigenpairs.
#
#   2. The correction-acceptance test was `dnorm > tol`, coupling it to the
#      convergence tolerance.  As residuals approached tol the corrections
#      needed to reduce them were rejected as already-converged, so the
#      iteration stalled at a small multiple of tol at every tolerance and
#      never reported convergence.
#
# The two interacted: (2) hid (1) by preventing convergence from ever being
# signalled, so a corrupted basis usually looked "unconverged" rather than
# "wrong".  These pin both.
# ---------------------------------------------------------------------------


def _diagonally_dominant(n: int, seed: int, spacing: float, offset: float):
    """Fock-like operator: well-separated diagonal plus small coupling."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) * (0.02 if spacing >= 1.0 else 0.01)
    A = 0.5 * (A + A.T)
    A[np.diag_indices(n)] = np.arange(n) * spacing + offset
    return A


def test_partial_spectrum_returns_the_true_lowest_roots() -> None:
    """Requesting a few roots must give the same ones dense eigh does.

    Pre-fix this returned eigenvalues of 4.0e-15..2.9e-12 with eigenvector
    column norms of ~1e-8 -- five null columns padded into an (n, 8) result,
    indistinguishable from converged roots by norm, eigenvalue or residual.
    """
    A = _diagonally_dominant(500, seed=3, spacing=0.1, offset=0.2)
    reference = np.sort(np.linalg.eigvalsh(A))[:8]

    opts = _core.DavidsonOptions()
    opts.n_eig = 8
    opts.conv_tol = 1e-6
    opts.max_iter = 200
    res = _core.davidson_solve(A, opts)

    evals = np.sort(np.asarray(res.eigenvalues))
    evecs = np.asarray(res.eigenvectors)

    # No dead columns: every returned eigenvector is a unit vector.
    norms = np.linalg.norm(evecs, axis=0)
    assert np.all(norms > 0.5), f"null eigenvector columns returned: {norms}"
    assert np.abs(evals - reference).max() < 1e-8


def test_partial_spectrum_signals_convergence_and_stops() -> None:
    """It must converge, say so, and not burn the whole iteration budget.

    Pre-fix the stalled residual tracked the requested tolerance at 3.0x, 4.7x,
    3.1x and 3.5x for tol = 1e-3, 1e-5, 1e-6 and 1e-8 -- `converged` was False
    at every tolerance and `n_iter` was always `max_iter`.
    """
    A = _diagonally_dominant(500, seed=3, spacing=0.1, offset=0.2)

    for tol in (1e-4, 1e-6, 1e-8):
        opts = _core.DavidsonOptions()
        opts.n_eig = 8
        opts.conv_tol = tol
        opts.max_iter = 200
        res = _core.davidson_solve(A, opts)

        assert res.converged, f"did not converge at tol={tol:g}"
        assert res.n_iter < 200, f"ran to max_iter at tol={tol:g}"

        evecs = np.asarray(res.eigenvectors)
        evals = np.asarray(res.eigenvalues)
        residual = max(
            np.linalg.norm(A @ evecs[:, k] - evals[k] * evecs[:, k])
            for k in range(evals.shape[0])
        )
        # Converged means converged: the residual honours the tolerance it
        # was given, rather than stalling at a multiple of it.
        assert residual < tol * 10.0, (
            f"tol={tol:g} reported converged at residual {residual:.2e}"
        )


def test_partial_spectrum_does_not_report_wrong_roots_as_converged() -> None:
    """The exact case that returned `converged=True` while 4.0 Ha wrong.

    n=200, n_eig=4, no caller-supplied guess, tol=1e-6: pre-fix this reported
    success after 6 iterations with every eigenvalue displaced by exactly one
    level -- it locked onto the wrong subset and declared victory.
    """
    A = _diagonally_dominant(200, seed=2, spacing=1.0, offset=1.0)
    reference = np.sort(np.linalg.eigvalsh(A))[:4]

    opts = _core.DavidsonOptions()
    opts.n_eig = 4
    opts.conv_tol = 1e-6
    opts.max_iter = 300
    res = _core.davidson_solve(A, opts)

    evals = np.sort(np.asarray(res.eigenvalues))
    if res.converged:
        assert np.abs(evals - reference).max() < 1e-6, (
            "reported convergence on the wrong roots"
        )
    assert np.abs(evals - reference).max() < 1e-8


def test_full_spectrum_path_is_unchanged() -> None:
    """The production periodic-SCF configuration must not regress.

    `periodic_{rhf,rks,uhf}_ewald.py` set `n_eig = Fp.shape[0]` and raise on
    `not converged`, so they take the one-iteration full-spectrum path.  That
    path was never affected by either defect and must stay that way.
    """
    A = _diagonally_dominant(200, seed=2, spacing=1.0, offset=1.0)
    reference = np.sort(np.linalg.eigvalsh(A))

    opts = _core.DavidsonOptions()
    opts.n_eig = A.shape[0]
    opts.conv_tol = 1e-6
    opts.max_iter = 300
    res = _core.davidson_solve(A, opts)

    assert res.converged
    assert res.n_iter == 1
    assert np.abs(np.sort(np.asarray(res.eigenvalues)) - reference).max() < 1e-9


# ---------------------------------------------------------------------------
# GitLab #506 -- the matrix-free entry point must actually be matrix-free.
#
# `davidson_solve_matvec` used to apply the caller's operator to all n unit
# vectors, assemble the dense matrix and hand it to the explicit kernel.  That
# cost the n x n allocation the caller asked to avoid, plus n callbacks, and
# discarded the supplied diagonal.  The eigenvalues were right, so only an
# operator-application count catches it.
# ---------------------------------------------------------------------------


def test_matvec_path_applies_the_operator_a_few_times_not_n_times() -> None:
    """Operator applications must scale with the roots, not the dimension."""
    n = 400
    A = _diagonally_dominant(n, seed=5, spacing=0.1, offset=0.3)
    diag = np.diag(A).copy()

    calls = {"n": 0}

    def matvec(v):
        calls["n"] += 1
        return A @ np.asarray(v)

    opts = _core.DavidsonOptions()
    opts.n_eig = 6
    opts.conv_tol = 1e-6
    opts.max_iter = 100
    res = _core.davidson_solve_matvec(n, matvec, diag, opts)

    reference = np.sort(np.linalg.eigvalsh(A))[:6]
    assert res.converged
    assert np.abs(np.sort(np.asarray(res.eigenvalues)) - reference).max() < 1e-8

    # Pre-fix this was exactly n (400) regardless of n_eig or convergence.
    # A genuine matrix-free solve needs O(n_roots) per iteration.
    assert calls["n"] < n, (
        f"applied the operator {calls['n']} times for n={n}: "
        "the matrix-free path is densifying again"
    )


def test_matvec_path_honours_the_supplied_diagonal() -> None:
    """The diagonal is the preconditioner; it must not be ignored.

    Pre-fix the kernel carried `(void)diag;` and recovered the diagonal from
    the densified matrix instead.  A wrong diagonal must therefore now change
    the iteration count -- if it does not, the argument is still being
    discarded.
    """
    n = 300
    A = _diagonally_dominant(n, seed=6, spacing=0.1, offset=0.3)
    true_diag = np.diag(A).copy()

    def solve_with(diag, max_iter=100):
        calls = {"n": 0}

        def matvec(v):
            calls["n"] += 1
            return A @ np.asarray(v)

        opts = _core.DavidsonOptions()
        opts.n_eig = 4
        opts.conv_tol = 1e-6
        opts.max_iter = max_iter
        res = _core.davidson_solve_matvec(n, matvec, diag, opts)
        return res, calls["n"]

    good, calls_good = solve_with(true_diag)
    # A deliberately useless preconditioner: constant, so it cannot separate
    # the roots.  The solve should cost more than with the true diagonal.
    bad, calls_bad = solve_with(np.zeros(n))

    reference = np.sort(np.linalg.eigvalsh(A))[:4]
    assert good.converged
    assert np.abs(np.sort(np.asarray(good.eigenvalues)) - reference).max() < 1e-8
    assert calls_bad > calls_good, (
        f"a constant diagonal cost the same as the true one "
        f"({calls_bad} vs {calls_good} applications): diag is being ignored"
    )
