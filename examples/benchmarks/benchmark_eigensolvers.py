"""Benchmark: eigensolver comparison on Fock-like matrices.

Compares dense, davidson, lobpcg, and lanczos on:
- Runtime (wall-clock)
- Iteration count / matrix-vector products
- Eigenvalue accuracy vs numpy reference
- Scaling with matrix size
- Matrix-free vs explicit matrix modes
- Warm-start benefit

Usage::

    python examples/benchmarks/benchmark_eigensolvers.py
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from vibeqc.solvers.eigensolver import (
    EigenProblem,
    SolverOptions,
    available_solvers,
    solve_eigenproblem,
)

# ── Matrix generation ────────────────────────────────────────────────────────


def make_fock_like(n: int, rng: np.random.Generator) -> np.ndarray:
    """Build a diagonally-dominant real symmetric matrix mimicking a Fock matrix.

    The diagonal grows as i² (orbital-energy-like), and off-diagonals
    decay as 1 / |i-j|² with Gaussian noise.  The resulting matrix is
    strongly diagonally dominant — the regime where iterative eigensolvers
    (Davidson, LOBPCG) excel.
    """
    diag = 0.5 * np.arange(1, n + 1, dtype=np.float64) ** 2 + 2.0
    A = np.diag(diag)
    # Off-diagonals: distance-decaying noise, symmetric
    i_grid, j_grid = np.triu_indices(n, k=1)
    dist = np.abs(i_grid - j_grid).astype(np.float64)
    noise = rng.normal(0.0, 0.01 / ((dist + 1.0) ** 2))
    A[i_grid, j_grid] = noise
    A[j_grid, i_grid] = noise
    return A


# ── Solver benchmarking ──────────────────────────────────────────────────────


def benchmark_solver(
    name: str,
    A: np.ndarray,
    n_roots: int,
    n_runs: int = 3,
    *,
    matrix_free: bool = False,
) -> dict[str, Any]:
    """Run solver *n_runs* times, return median stats.

    Parameters
    ----------
    name : str
        Solver key (``"dense"``, ``"davidson"``, ``"lobpcg"``, ``"lanczos"``).
    A : (n, n) ndarray
        The explicit matrix.
    n_roots : int
        Number of lowest eigenvalues to extract.
    n_runs : int
        Repeat count for timing stability.
    matrix_free : bool
        If True, provide only ``matvec`` (matrix-free path).
        If False, pass the full ``matrix``.

    Returns
    -------
    dict
        Keys: ``"time_s"``, ``"iters"``, ``"n_matvec"``, ``"error"``,
        ``"converged"``.
    """
    n = A.shape[0]
    if matrix_free:
        _A = A  # capture for closure
        problem = EigenProblem(matvec=lambda v: _A @ v, n=n, dtype=np.float64)
    else:
        problem = EigenProblem(matrix=A, n=n)

    options = SolverOptions(n_roots=n_roots, which="SA", tol=1e-8, max_iter=200)

    times: list[float] = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = solve_eigenproblem(problem, options, solver=name)
        times.append(time.perf_counter() - t0)

    assert result is not None
    # Reference: full dense diagonalisation
    ref_eigs = np.sort(np.linalg.eigh(A)[0])[:n_roots]
    err = float(np.max(np.abs(np.sort(result.eigenvalues) - ref_eigs)))

    return {
        "time_s": float(np.median(times)),
        "iters": result.n_iter,
        "n_matvec": result.n_matvec,
        "error": err,
        "converged": result.converged,
    }


# ── Warm-start benchmark ─────────────────────────────────────────────────────


def benchmark_warm_start(
    name: str,
    A: np.ndarray,
    n_roots: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Measure the benefit of eigenvector recycling (warm-start).

    Solves the eigenproblem cold, then solves a *perturbed* version
    of the matrix using the cold eigenvectors as ``guess_vectors``.

    Returns
    -------
    dict
        Keys: ``"cold_time_s"``, ``"cold_iters"``, ``"warm_time_s"``,
        ``"warm_iters"``, ``"speedup"``.
    """
    n = A.shape[0]
    problem = EigenProblem(matrix=A, n=n)
    opts = SolverOptions(n_roots=n_roots, which="SA", tol=1e-8, max_iter=200)

    # Cold run
    t0 = time.perf_counter()
    cold = solve_eigenproblem(problem, opts, solver=name)
    t_cold = time.perf_counter() - t0

    # Perturb the matrix
    dA = rng.normal(0.0, 0.001, (n, n))
    dA = 0.5 * (dA + dA.T) * 0.01
    A2 = A + dA

    # Warm run: recycle cold eigenvectors as initial guess
    problem2 = EigenProblem(matrix=A2, n=n)
    opts2 = SolverOptions(
        n_roots=n_roots,
        which="SA",
        tol=1e-8,
        max_iter=200,
        guess_vectors=cold.eigenvectors.copy(),
    )
    t0 = time.perf_counter()
    warm = solve_eigenproblem(problem2, opts2, solver=name)
    t_warm = time.perf_counter() - t0

    return {
        "cold_time_s": t_cold,
        "cold_iters": cold.n_iter,
        "warm_time_s": t_warm,
        "warm_iters": warm.n_iter,
        "speedup": t_cold / t_warm if t_warm > 0 else float("inf"),
        "converged": cold.converged and warm.converged,
    }


# ── Formatting ───────────────────────────────────────────────────────────────


def _format_row(
    solver: str,
    mode: str,
    stats: dict[str, Any],
    width_solver: int = 12,
    width_mode: int = 10,
) -> str:
    conv = "yes" if stats["converged"] else "NO"
    return (
        f"  {solver:<{width_solver}s}  {mode:<{width_mode}s}"
        f"  {stats['time_s']:>8.4f}"
        f"  {stats['iters']:>6d}"
        f"  {stats['n_matvec']:>8d}"
        f"  {stats['error']:>10.2e}"
        f"  {conv:>6s}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(42)

    # Solver keys the user wants to compare
    desired = ["dense", "davidson", "lobpcg", "lanczos"]
    available = available_solvers()
    solvers = [s for s in desired if s in available]
    missing = set(desired) - set(solvers)
    if missing:
        print(f"Note: solvers not available, skipping: {sorted(missing)}")

    # Which solvers support matrix-free operation
    MF_CAPABLE = {"davidson", "lobpcg", "lanczos"}

    # ── Header ──────────────────────────────────────────────────────────────
    print("=" * 90)
    print("Eigensolver Benchmark — Fock-like matrices")
    print("=" * 90)
    print(f"  Solvers tested: {', '.join(solvers)}")
    print(f"  n_runs per cell: 3 (median reported)")
    print(f"  tol: 1e-8  |  n_roots: 5  |  which: SA")
    print(f"  Modes: 'explicit' = full matrix; 'matfree' = A@v only")

    # ── Per-size tables ─────────────────────────────────────────────────────
    sizes = [50, 100, 200, 500]
    n_roots = 5

    for n in sizes:
        print(f"\n{'─' * 90}")
        print(f"  n = {n}   (n_roots = {n_roots})")
        print(f"  {'─' * 90}")

        A = make_fock_like(n, rng)
        ref_eigs = np.sort(np.linalg.eigh(A)[0])[:n_roots]
        print(
            f"  Reference eigenvalues: {np.array2string(ref_eigs, precision=6, suppress_small=True)}"
        )
        print()

        # Column headers
        hdr_solver = "Solver"
        hdr_mode = "Mode"
        w_s = max(len(hdr_solver), max((len(s) for s in solvers), default=8))
        w_m = max(len(hdr_mode), 8)
        print(
            f"  {'Solver':<{w_s}s}  {'Mode':<{w_m}s}"
            f"  {'Time(s)':>8s}  {'Iters':>6s}  {'Matvec':>8s}"
            f"  {'Error':>10s}  {'Conv':>6s}"
        )
        print(
            f"  {'─' * w_s}  {'─' * w_m}  {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 10}  {'─' * 6}"
        )

        for name in solvers:
            # Explicit matrix mode
            stats = benchmark_solver(name, A, n_roots, matrix_free=False)
            print(_format_row(name, "explicit", stats, w_s, w_m))

            # Matrix-free mode (where supported)
            if name in MF_CAPABLE:
                stats_mf = benchmark_solver(name, A, n_roots, matrix_free=True)
                print(_format_row(name, "matfree", stats_mf, w_s, w_m))

    # ── Warm-start benchmark ────────────────────────────────────────────────
    print(f"\n{'─' * 90}")
    print("  Warm-start benefit (eigenvector recycling)")
    print(f"  {'─' * 90}")

    A100 = make_fock_like(100, rng)

    w_warm = max(len("Solver"), max((len(s) for s in solvers), 8))
    print(
        f"  {'Solver':<{w_warm}s}"
        f"  {'Cold(s)':>9s}  {'Cold iter':>10s}"
        f"  {'Warm(s)':>9s}  {'Warm iter':>10s}"
        f"  {'Speedup':>8s}"
    )
    print(f"  {'─' * w_warm}  {'─' * 9}  {'─' * 10}  {'─' * 9}  {'─' * 10}  {'─' * 8}")

    for name in solvers:
        if name == "dense":
            # Dense solver is single-shot — warm-start is meaningless
            print(f"  {name:<{w_warm}s}  {'(single-shot, skipped)':>42s}")
            continue
        ws = benchmark_warm_start(name, A100, n_roots, rng)
        print(
            f"  {name:<{w_warm}s}"
            f"  {ws['cold_time_s']:9.4f}  {ws['cold_iters']:10d}"
            f"  {ws['warm_time_s']:9.4f}  {ws['warm_iters']:10d}"
            f"  {ws['speedup']:7.1f}x"
        )

    # ── Scaling summary ─────────────────────────────────────────────────────
    print(f"\n{'─' * 90}")
    print("  Scaling summary — explicit matrix, median time (s)")
    print(f"  {'─' * 90}")

    size_headers = [str(s) for s in sizes]
    w_scale = max(max((len(s) for s in solvers), len("Solver")), 8)
    col_w = max(max((len(h) for h in size_headers), 6), 8)

    # Header
    line = f"  {'Solver':<{w_scale}s}"
    for h in size_headers:
        line += f"  {'n=' + h:>{col_w}s}"
    print(line)
    print(f"  {'─' * w_scale}" + f"  {'─' * col_w}" * len(sizes))

    # Pre-compute scaling data
    scaling: dict[str, list[float]] = {name: [] for name in solvers}
    for n_s in sizes:
        A_s = make_fock_like(n_s, rng)
        for name in solvers:
            s = benchmark_solver(name, A_s, n_roots, matrix_free=False)
            scaling[name].append(s["time_s"])

    for name in solvers:
        line = f"  {name:<{w_scale}s}"
        for t in scaling[name]:
            line += f"  {t:>{col_w}.4f}"
        print(line)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
