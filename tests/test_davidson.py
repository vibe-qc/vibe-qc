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


# ---------------------------------------------------------------------------
# GitLab #123 -- the subspace-collapse branch indexed past the subspace.
#
# `keep = min(n_eig + 5, n)` chose how many Ritz vectors a restart carries
# over, without bounding it by the subspace actually held.  V, AV and X all
# have exactly `m_sub` columns at that point, so any budget that forces a
# collapse while `m_sub < n_eig + 5` read and wrote past the end of all
# three.  Three public-API routes reach it: a small `max_subspace`, an
# explicit `n_guess` below `n_eig + 5`, and a warm-start `guess_vectors` of
# exactly `n_eig` columns -- which is what the SCF drivers recycle.
#
# Release builds define NDEBUG, so Eigen's bounds assertions were off and the
# overrun was silent.  Measured before the fix on the n=120 Hermitian case
# below, over ten identical runs: five hard process kills (SIGTRAP from
# libmalloc, "memory corruption of free block", raised inside ZGEMM when the
# corrupted heap was next touched), four returns with null eigenvector
# columns, and one `converged=True` carrying eigenvalues of -2.5e-14,
# -3.2e-15 and 3.0e-31 in place of 0.988, 2.022 and 2.999.
#
# That last outcome is the defect's whole point: a Ritz vector that collapsed
# to zero has residual ||A.x - lambda.x|| = 0, so it passes a residual-only
# convergence test.  The fix bounds the restart by the held subspace, gives
# the restart room to expand again, re-seeds a collapsed column in the
# Hermitian kernel as the real one always did, and refuses to call a null
# Ritz pair converged.
# ---------------------------------------------------------------------------


def _hermitian_ladder(n: int, seed: int):
    """Complex Hermitian operator with a unit-spaced diagonal ladder."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, n)) * 0.02 + 1j * rng.normal(size=(n, n)) * 0.02
    return np.diag(np.arange(1.0, n + 1.0)) + (z + z.conj().T) / 2


def test_collapse_does_not_overrun_a_narrow_subspace() -> None:
    """Real kernel, the configuration AddressSanitizer caught first.

    n_eig=4 with n_guess=8 and max_subspace=12 collapses on the first sweep
    while the subspace holds 8 columns, and pre-fix wrote `X.leftCols(9)`
    into a `V` with 8 allocated columns -- 60 doubles past the end of the
    buffer.  Eigen's own bounds assertion fires on this input in a debug
    build; in the shipped build it corrupted the heap instead.
    """
    n = 60
    A = _diagonally_dominant(n, seed=0, spacing=1.0, offset=1.0)
    reference = np.sort(np.linalg.eigvalsh(A))[:4]

    opts = _core.DavidsonOptions()
    opts.n_eig = 4
    opts.n_guess = 8
    opts.max_subspace = 12
    opts.conv_tol = 1e-6
    opts.max_iter = 200
    res = _core.davidson_solve(A, opts)

    evecs = np.asarray(res.eigenvectors)
    norms = np.linalg.norm(evecs, axis=0)
    assert np.all(norms > 0.5), f"null eigenvector columns returned: {norms}"
    assert res.subspace_dim <= n, (
        f"reported a subspace of {res.subspace_dim} columns on an {n}-column "
        "operator: the collapse indexed past what it held"
    )
    assert np.abs(np.sort(np.asarray(res.eigenvalues)) - reference).max() < 1e-6


def test_hermitian_collapse_returns_the_true_roots_not_zeros() -> None:
    """The Hermitian case that reported success while returning zeros.

    n=120, three roots, n_guess=6, max_subspace=7.  Pre-fix this returned
    `converged=True` with all three eigenvalues at ~1e-14 and eigenvector
    column norms of 1e-14..1e-16, or killed the process outright.  The
    repeats pin determinism: the pre-fix answer varied run to run because it
    depended on whatever memory followed the buffer.
    """
    A = _hermitian_ladder(120, seed=0)
    reference = np.linalg.eigvalsh(A)[:3]

    seen = []
    for _ in range(3):
        opts = _core.DavidsonOptions()
        opts.n_eig = 3
        opts.n_guess = 6
        opts.max_subspace = 7
        opts.conv_tol = 1e-6
        opts.max_iter = 100
        res = _core.davidson_solve_hermitian(A, opts)

        evals = np.asarray(res.eigenvalues)
        evecs = np.asarray(res.eigenvectors)
        norms = np.linalg.norm(evecs, axis=0)

        assert np.all(norms > 0.5), f"null eigenvector columns: {norms}"
        assert res.converged, "a budget this size is workable and must converge"
        assert np.abs(evals - reference).max() < 1e-6, (
            f"returned {evals} for true roots {reference}"
        )
        seen.append(evals.copy())

    for repeat in seen[1:]:
        assert np.array_equal(seen[0], repeat), (
            "identical input gave different answers across runs"
        )


def test_warm_start_guess_of_exactly_n_eig_columns_is_safe() -> None:
    """The shape the SCF drivers recycle: guess_vectors with n_eig columns.

    `rhf.cpp`, `uhf.cpp`, `rks.cpp` and `uks.cpp` feed the previous
    iteration's eigenvectors back in, so the subspace starts at exactly
    `n_eig` columns -- below the `n_eig + 5` the restart used to keep.
    """
    n = 40
    A = _diagonally_dominant(n, seed=4, spacing=1.0, offset=1.0)
    reference = np.sort(np.linalg.eigvalsh(A))[:6]
    guess = np.linalg.eigh(A)[1][:, :6] + 0.01

    opts = _core.DavidsonOptions()
    opts.n_eig = 6
    opts.guess_vectors = guess
    opts.max_subspace = 10
    opts.conv_tol = 1e-6
    opts.max_iter = 200
    res = _core.davidson_solve(A, opts)

    norms = np.linalg.norm(np.asarray(res.eigenvectors), axis=0)
    assert np.all(norms > 0.5), f"null eigenvector columns: {norms}"
    assert np.abs(np.sort(np.asarray(res.eigenvalues)) - reference).max() < 1e-6


def test_a_guess_narrower_than_the_requested_roots_is_refused() -> None:
    """Fewer guess vectors than roots indexed past the Ritz block.

    The subspace starts at the guess width, but every Ritz quantity is read
    up to `n_eig`, so this used to read past the end of X and AX rather than
    fail.  Both kernels now refuse it.
    """
    n = 30
    A = _diagonally_dominant(n, seed=5, spacing=1.0, offset=1.0)

    opts = _core.DavidsonOptions()
    opts.n_eig = 6
    opts.guess_vectors = np.linalg.eigh(A)[1][:, :2]
    opts.conv_tol = 1e-6
    with pytest.raises(ValueError, match="fewer than n_eig"):
        _core.davidson_solve(A, opts)

    H = _hermitian_ladder(n, seed=5)
    copts = _core.DavidsonOptions()
    copts.n_eig = 6
    copts.guess_vectors_cplx = np.linalg.eigh(H)[1][:, :2]
    copts.conv_tol = 1e-6
    with pytest.raises(ValueError, match="fewer than n_eig"):
        _core.davidson_solve_hermitian(H, copts)


def test_hermitian_kernel_refuses_n_guess_below_n_eig_like_the_real_one() -> None:
    """The real kernel has always refused this; the Hermitian one accepted it.

    It then extracted `n_eig` Ritz pairs from an `n_guess`-wide subspace,
    reading past the end of both Ritz blocks.
    """
    H = _hermitian_ladder(30, seed=1)
    opts = _core.DavidsonOptions()
    opts.n_eig = 8
    opts.n_guess = 3
    opts.conv_tol = 1e-6
    with pytest.raises(ValueError, match="n_guess"):
        _core.davidson_solve_hermitian(H, opts)


def test_n_converged_reports_how_much_of_the_spectrum_was_reached() -> None:
    """A partial spectrum has to be usable, which needs a count (#123).

    `converged` is all-or-nothing; without a count a caller cannot tell a run
    that found most of its roots from one that found none.
    """
    A = _diagonally_dominant(120, seed=2, spacing=1.0, offset=1.0)

    opts = _core.DavidsonOptions()
    opts.n_eig = 4
    opts.conv_tol = 1e-8
    opts.max_iter = 300
    res = _core.davidson_solve(A, opts)
    assert res.converged
    assert res.n_converged == 4

    # One iteration cannot converge four roots from a diagonal seed; the
    # count must be honest about that rather than reporting the full block.
    starved = _core.DavidsonOptions()
    starved.n_eig = 4
    starved.conv_tol = 1e-10
    starved.max_iter = 1
    res2 = _core.davidson_solve(A, starved)
    assert not res2.converged
    assert 0 <= res2.n_converged < 4


@pytest.mark.parametrize("n_eig", [1, 2, 4, 8])
@pytest.mark.parametrize("max_subspace", [7, 12, 40])
def test_no_budget_returns_a_null_ritz_pair(n_eig: int, max_subspace: int) -> None:
    """Sweep the collapse budget: no setting may return a null eigenvector.

    A null Ritz vector has a null residual, so a residual-only convergence
    test accepts it.  This walks the budgets that force early collapses in
    both kernels and pins that none of them produces one.
    """
    n = 60
    real = _diagonally_dominant(n, seed=1, spacing=1.0, offset=1.0)
    cplx = _hermitian_ladder(n, seed=1)

    for A, solve in ((real, _core.davidson_solve),
                     (cplx, _core.davidson_solve_hermitian)):
        reference = np.linalg.eigvalsh(A)[:n_eig]
        opts = _core.DavidsonOptions()
        opts.n_eig = n_eig
        opts.max_subspace = max_subspace
        opts.conv_tol = 1e-6
        opts.max_iter = 200
        res = solve(A, opts)

        evecs = np.asarray(res.eigenvectors)
        norms = np.linalg.norm(evecs, axis=0)
        assert np.all(norms > 0.5), (
            f"n_eig={n_eig} max_subspace={max_subspace}: null columns {norms}"
        )
        if res.converged:
            assert np.abs(np.asarray(res.eigenvalues) - reference).max() < 1e-6, (
                f"n_eig={n_eig} max_subspace={max_subspace}: reported "
                "convergence on the wrong roots"
            )
