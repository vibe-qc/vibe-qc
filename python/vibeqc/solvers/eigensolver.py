"""Unified eigensolver framework for vibe-qc.

Provides a common interface for all diagonalisation backends used in SCF
cycles and post-SCF analysis, with problem descriptors, solver strategies,
a solver registry, and a unified `solve_eigenproblem` entry point.

Solvers currently available:
- ``"dense"`` -- full dense diagonalisation via `np.linalg.eigh` / `scipy.linalg.eigh`
- ``"davidson"`` -- blocked Davidson (C++ backend, real symmetric)
- ``"hermitian_davidson"`` -- blocked Davidson (C++ backend, complex Hermitian)
- ``"lanczos"`` -- Lanczos tridiagonalisation with full reorthogonalisation (Python)

Usage::

    from vibeqc.solvers.eigensolver import (
        EigenProblem, SolverOptions, solve_eigenproblem,
    )
    problem = EigenProblem(matrix=F, n=n_basis)
    options = SolverOptions(n_roots=5, which="SA", tol=1e-8)
    result = solve_eigenproblem(problem, options, solver="davidson")
    C = result.eigenvectors  # n x 5
    eps = result.eigenvalues # (5,)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

# ── Lazy import helpers ────────────────────────────────────────────────────


def _get_core():
    """Lazy-load the C++ native core to avoid import-time cost."""
    # Import is deferred because the native extension may not be built
    # during doc-generation / linting.
    import vibeqc._vibeqc_core as _core

    return _core


# ── Problem descriptors ────────────────────────────────────────────────────


@dataclass
class EigenProblem:
    """Standard Hermitian eigenproblem: H x = l x.

    At least one of ``matrix`` or ``matvec`` must be provided.  When both
    are given, ``matvec`` takes precedence for matrix-free solvers and
    ``matrix`` is used for explicit-diagonalisation solvers.

    Attributes
    ----------
    matrix : (n, n) ndarray or None
        Explicit real symmetric / complex Hermitian matrix.
    matvec : callable(v) -> ndarray or None
        Matrix-vector product H @ v for matrix-free operation.
    n : int
        Dimension.  Auto-detected from ``matrix`` when zero.
    dtype : type
        Element type (``np.float64`` for real, ``np.complex128`` for complex).
    diagonal : (n,) ndarray or None
        The operator's diagonal, when the caller already knows it in closed
        form.  Optional and purely an optimisation: matrix-free solvers need
        the diagonal for their preconditioner, and recover it otherwise by
        applying ``matvec`` to ``n`` unit vectors (:func:`_matvec_diagonal`).
        That probe is ``O(n)`` operator applications, so for a problem whose
        ``matvec`` is ``O(n**2)`` it costs ``O(n**3)`` -- as much as the dense
        diagonalisation the matrix-free path exists to avoid.  Supply this
        whenever the diagonal is analytic (TDA/Casida response, CI, Fock).
    """

    matrix: np.ndarray | None = None
    matvec: Callable[[np.ndarray], np.ndarray] | None = None
    n: int = 0
    dtype: type = np.float64
    diagonal: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n == 0 and self.matrix is not None:
            self.n = self.matrix.shape[0]
        if self.matrix is not None:
            self.dtype = self.matrix.dtype.type
        if self.n <= 0:
            raise ValueError("EigenProblem: n must be > 0, or provide matrix.")


@dataclass
class GeneralizedEigenProblem:
    """Generalized Hermitian eigenproblem: H x = l S x.

    Attributes
    ----------
    matrix : (n, n) ndarray or None
        Hamiltonian / Fock matrix H.
    overlap : (n, n) ndarray or None
        Overlap / metric matrix S (must be positive definite).
    matvec : callable(v) -> ndarray or None
        Matrix-vector product H @ v (S is always explicit).
    n : int
        Dimension.
    """

    matrix: np.ndarray | None = None
    overlap: np.ndarray | None = None
    matvec: Callable[[np.ndarray], np.ndarray] | None = None
    n: int = 0

    def __post_init__(self) -> None:
        if self.n == 0:
            if self.matrix is not None:
                self.n = self.matrix.shape[0]
            elif self.overlap is not None:
                self.n = self.overlap.shape[0]
        if self.n <= 0:
            raise ValueError(
                "GeneralizedEigenProblem: n must be > 0, or provide matrix/overlap."
            )


@dataclass
class InteriorEigenProblem:
    """Interior eigenvalue problem: find roots near a target shift s.

    Wraps a standard eigenproblem with a shift so that solvers targeting
    extremal eigenvalues can find interior ones via spectral transformation
    (shift-and-invert).

    Attributes
    ----------
    problem : EigenProblem
        The underlying Hermitian problem.
    sigma : float
        Target shift.  Solvers that support shift-and-invert will find
        eigenvalues closest to s.
    """

    problem: EigenProblem
    sigma: float = 0.0


@dataclass
class SCFStep:
    """Orbital optimisation step descriptor for SCF diagonalisation.

    Carries the Fock/Kohn-Sham matrix together with optional warm-start
    information (previous MO coefficients, canonical orthogonaliser) so
    that iterative eigensolvers can recycle subspaces across SCF cycles.

    Attributes
    ----------
    fock : (n, n) ndarray
        Fock / Kohn-Sham matrix in the current orbital basis.
    X : (n, n) ndarray or None
        Canonical-orthogonalisation matrix (S^(-1/2) or similar).
        When present, the eigensolver operates in the orthogonalised basis
        and the eigenvectors are transformed back.
    n_occ : int
        Number of occupied orbitals (roots to extract).
    prev_C : (n, n_occ) ndarray or None
        Previous-iteration MO coefficients for subspace recycling.
    prev_eps : (n_occ,) ndarray or None
        Previous-iteration orbital energies.
    """

    fock: np.ndarray
    X: np.ndarray | None = None
    n_occ: int = 0
    prev_C: np.ndarray | None = None
    prev_eps: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n_occ == 0:
            self.n_occ = self.fock.shape[0]
        if self.fock.ndim != 2:
            raise ValueError("SCFStep.fock must be a 2-D array.")


# ── Solver options ─────────────────────────────────────────────────────────


@dataclass
class SolverOptions:
    """Unified options for all eigensolvers.

    Attributes
    ----------
    n_roots : int
        Number of eigenpairs to extract.  ``0`` = all (for dense solvers).
    which : str
        Which part of the spectrum:
        - ``"SA"`` -- smallest algebraic (ascending order)
        - ``"LA"`` -- largest algebraic (descending order)
        - ``"SM"`` -- smallest magnitude (closest to zero)
        - ``"LM"`` -- largest magnitude (furthest from zero)
        - ``"BE"`` -- both ends (extremal eigenvalues from both sides)
    sigma : float or None
        Target shift for interior / shift-and-invert modes.
    max_iter : int
        Maximum subspace-expansion (outer) iterations.
    tol : float
        Convergence threshold on residual norm ‖r_i‖₂.
    ncv : int or None
        Maximum subspace / Arnoldi dimension.  ``None`` = auto.
    block_size : int
        Block size for block methods.  ``0`` = auto.
    max_subspace : int
        Maximum subspace dimension before collapse.  ``0`` = auto.
    preconditioner : str
        Preconditioner type: ``"diagonal"``, ``"none"``, or ``"identity"``.
    preshift : float
        Diagonal shift added to the Davidson preconditioner denominator
        to avoid near-zero divisors when l_i ≈ diag(A)_i.
    max_restarts : int
        Maximum restarts (for restarted Lanczos / Arnoldi).
    verbosity : int
        0 = silent, 1 = summary, 2 = per-iteration detail.
    guess_vectors : (n, k) ndarray or None
        Optional warm-start subspace columns.
    n_guess : int
        Initial Davidson subspace size when no ``guess_vectors`` are supplied.
        ``0`` uses the native automatic size, ``max(n_roots + 5, 2*n_roots)``,
        capped at the problem dimension. A positive value must cover all roots.
    """

    n_roots: int = 0
    which: str = "SA"
    sigma: float | None = None
    max_iter: int = 200
    tol: float = 1e-7
    ncv: int | None = None
    block_size: int = 0
    max_subspace: int = 0
    preconditioner: str = "diagonal"
    preshift: float = 1e-6
    max_restarts: int = 5
    verbosity: int = 0
    guess_vectors: np.ndarray | None = None
    n_guess: int = 0


# ── Solver result ──────────────────────────────────────────────────────────


@dataclass
class SolverResult:
    """Result container for an eigensolver solve.

    Attributes
    ----------
    eigenvalues : (n_roots,) ndarray
        Converged eigenvalues in ascending order.
    eigenvectors : (n, n_roots) ndarray
        Corresponding eigenvectors as columns.
    n_iter : int
        Number of outer (subspace-expansion / restart) iterations.
    converged : bool
        Whether all requested roots met the convergence threshold.
    residuals : (n_roots,) ndarray or None
        Residual norms ‖A x_i - l_i x_i‖₂ at termination.
    n_matvec : int
        Total number of matrix-vector products performed.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_iter: int
    converged: bool
    residuals: np.ndarray | None = None
    n_matvec: int = 0


_UPPER_ROOT_MODES = {"LA", "LM"}


def _wants_upper_roots(options: SolverOptions) -> bool:
    """Whether an iterative lowest-root backend must be sign-flipped."""
    return options.which.upper() in _UPPER_ROOT_MODES


def _lowest_root_options(options: SolverOptions) -> SolverOptions:
    """Map an upper-root request onto the backend's lowest-root contract."""
    return replace(options, which="SA") if _wants_upper_roots(options) else options


def _flip_upper_root_result(result: SolverResult) -> SolverResult:
    """Convert a solve of ``-A`` back to ascending upper roots of ``A``."""
    eigenvalues = -np.asarray(result.eigenvalues)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = np.asarray(result.eigenvectors)[:, order]
    residuals = None
    if result.residuals is not None:
        residuals = np.asarray(result.residuals)[order]
    return SolverResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_iter=result.n_iter,
        converged=result.converged,
        residuals=residuals,
        n_matvec=result.n_matvec,
    )


def _negated_problem(problem: EigenProblem) -> EigenProblem:
    """Return the matrix-free/explicit problem for ``-A``."""
    matrix = -np.asarray(problem.matrix) if problem.matrix is not None else None
    matvec = None
    if problem.matvec is not None:
        matvec = lambda v: -problem.matvec(v)
    # Carry the diagonal through: -A's diagonal is -diag(A).  Dropping it here
    # would silently send an upper-root matrix-free solve back to the O(n)
    # probe in _matvec_diagonal.
    diagonal = None
    if problem.diagonal is not None:
        diagonal = -np.asarray(problem.diagonal, dtype=np.float64)
    return EigenProblem(
        matrix=matrix,
        matvec=matvec,
        n=problem.n,
        dtype=problem.dtype,
        diagonal=diagonal,
    )


def _matvec_diagonal(problem: EigenProblem, *, dtype: type = np.float64) -> np.ndarray:
    """Diagonal for matrix-free preconditioners.

    Returns ``problem.diagonal`` when the caller supplied it; otherwise probes
    ``matvec(e_i)[i]`` for each ``i``.  The probe is ``O(n)`` operator
    applications and is a fallback, not the intended path for large problems --
    see :class:`EigenProblem`.
    """
    if problem.diagonal is not None:
        diag = np.asarray(problem.diagonal, dtype=np.float64).reshape(-1)
        if diag.shape[0] != problem.n:
            raise ValueError(
                f"EigenProblem.diagonal has length {diag.shape[0]}, "
                f"expected n={problem.n}."
            )
        return diag
    if problem.matvec is None:
        raise ValueError("_matvec_diagonal requires problem.matvec")
    diag = np.zeros(problem.n, dtype=np.float64)
    for i in range(problem.n):
        e_i = np.zeros(problem.n, dtype=dtype)
        e_i[i] = 1.0
        diag[i] = np.asarray(problem.matvec(e_i))[i].real
    return diag


# ── Solver registry ────────────────────────────────────────────────────────

_SOLVER_REGISTRY: dict[str, type[SolverStrategy]] = {}


def available_solvers() -> list[str]:
    """Return the names of all registered solver strategies."""
    return list(_SOLVER_REGISTRY.keys())


def get_solver(name: str) -> SolverStrategy:
    """Instantiate a solver strategy by name.

    Parameters
    ----------
    name : str
        Solver name (e.g. ``"davidson"``, ``"lanczos"``, ``"dense"``).

    Returns
    -------
    SolverStrategy
        A fresh solver instance.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in _SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver '{name}'. Available: {available_solvers()}")
    return _SOLVER_REGISTRY[name]()


# ── Solver strategy base class ─────────────────────────────────────────────


class SolverStrategy(ABC):
    """Abstract base for all eigensolvers.

    Subclasses declare their capabilities via class-level flags and
    implement ``solve``.  Registration into ``_SOLVER_REGISTRY`` is
    automatic for any non-private subclass (class name must not start
    with ``'_'``).
    """

    # Capability flags -- subclasses override as needed.
    supports_extremal: bool = True
    supports_interior: bool = False
    supports_generalized: bool = False
    supports_block: bool = True
    supports_matrix_free: bool = False
    supports_explicit: bool = True
    supports_scf_step: bool = False

    @abstractmethod
    def solve(
        self,
        problem: EigenProblem,
        options: SolverOptions,
    ) -> SolverResult:
        """Solve an eigenproblem.

        Parameters
        ----------
        problem : EigenProblem
            The problem description (matrix or matvec, dimension, dtype).
        options : SolverOptions
            Convergence and performance knobs.

        Returns
        -------
        SolverResult
        """
        ...

    @classmethod
    def name(cls) -> str:
        """Canonical short keyword: davidson, lobpcg, dense, lanczos."""
        return cls.__name__.removesuffix("Solver").lower()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__name__.startswith("_"):
            name = cls.name()
            if name not in _SOLVER_REGISTRY:
                _SOLVER_REGISTRY[name] = cls
            for alias in getattr(cls, "_aliases", []):
                if alias not in _SOLVER_REGISTRY:
                    _SOLVER_REGISTRY[alias] = cls


# ── Unified entry point ────────────────────────────────────────────────────


def solve_eigenproblem(
    problem: EigenProblem | GeneralizedEigenProblem | InteriorEigenProblem,
    options: SolverOptions | None = None,
    *,
    solver: str = "davidson",
) -> SolverResult:
    """Solve an eigenproblem via a named solver strategy.

    This is the single entry point for all diagonalisation in vibe-qc.
    It dispatches to the registered solver, handling interior and
    generalised problems by reduction to the standard Hermitian form
    when possible.

    Parameters
    ----------
    problem : EigenProblem or GeneralizedEigenProblem or InteriorEigenProblem
        The eigenproblem to solve.
    options : SolverOptions, optional
        Solver knobs.  Defaults to ``SolverOptions()``.
    solver : str
        Which registered solver to use (``"davidson"``, ``"dense"``,
        ``"lanczos"``, ``"hermitian_davidson"``).

    Returns
    -------
    SolverResult
    """
    if options is None:
        options = SolverOptions()

    strategy = get_solver(solver)

    # Dispatch interior problems
    if isinstance(problem, InteriorEigenProblem):
        if not strategy.supports_interior:
            raise NotImplementedError(
                f"Solver '{solver}' does not support interior eigenvalue "
                f"problems.  Available interior-capable solvers: "
                f"{[n for n, c in _SOLVER_REGISTRY.items() if c.supports_interior]}"
            )
        # Propagate the interior shift into solver options
        if options.sigma is None:
            options = replace(options, sigma=problem.sigma)
        return strategy.solve(problem.problem, options)

    # Dispatch generalised problems
    if isinstance(problem, GeneralizedEigenProblem):
        if not strategy.supports_generalized:
            # Canonical reduction: H' = S^(-1/2) H S^(-1/2)
            if problem.overlap is None:
                raise ValueError(
                    "GeneralizedEigenProblem requires an overlap matrix "
                    "when the solver does not natively support generalised forms."
                )
            # Symmetric orthogonalisation via S^(-1/2), reduced to the
            # numerically independent overlap subspace. Keeping the discarded
            # dimensions as zero rows/columns creates ghost roots, especially
            # for upper-spectrum requests.
            s_eigvals, s_eigvecs = np.linalg.eigh(problem.overlap)
            s_cutoff = max(1e-12, 1e-12 * float(np.max(np.abs(s_eigvals))))
            mask = s_eigvals > s_cutoff
            if not np.any(mask):
                raise ValueError(
                    "GeneralizedEigenProblem overlap has no numerically "
                    "positive directions."
                )
            s_inv_sqrt = s_eigvecs[:, mask] / np.sqrt(s_eigvals[mask])
            if problem.matrix is not None:
                H_prime = s_inv_sqrt.T @ problem.matrix @ s_inv_sqrt
                reduced = EigenProblem(matrix=H_prime, n=H_prime.shape[0])
            else:
                # Matrix-free: wrap matvec
                S_is = s_inv_sqrt
                _matvec = problem.matvec
                if _matvec is None:
                    raise ValueError(
                        "GeneralizedEigenProblem: matvec required when matrix is None."
                    )

                def wrapped_matvec(v):
                    return S_is.T @ _matvec(S_is @ v)

                reduced = EigenProblem(matvec=wrapped_matvec, n=S_is.shape[1])
            result = strategy.solve(reduced, options)
            # Back-transform eigenvectors: C = S^(-1/2) C'
            result.eigenvectors = s_inv_sqrt @ result.eigenvectors
            return result
        return strategy.solve(problem, options)  # type: ignore[arg-type]

    # Standard Hermitian problem
    if problem.n <= 0:
        raise ValueError("EigenProblem.n must be > 0.")
    return strategy.solve(problem, options)


# ═══════════════════════════════════════════════════════════════════════════
# Concrete solver implementations
# ═══════════════════════════════════════════════════════════════════════════


# ── Dense solver ───────────────────────────────────────────────────────────


class DenseSolver(SolverStrategy):
    _aliases = ["dense", "eigh"]
    """Full dense diagonalisation via `numpy.linalg.eigh` or `scipy.linalg.eigh`.

    This is the reference solver: exact (to machine precision for the given
    matrix), always converges, but O(N^3).  Suitable for small systems and
    as a fallback / validation baseline.
    """

    supports_extremal = True
    supports_generalized = False  # handled by solve_eigenproblem reduction
    supports_block = False
    supports_matrix_free = False
    supports_explicit = True
    supports_scf_step = True

    def solve(
        self,
        problem: EigenProblem,
        options: SolverOptions,
    ) -> SolverResult:
        if problem.matrix is None:
            raise ValueError(
                "DenseSolver requires an explicit matrix; "
                "use a matrix-free-capable solver instead."
            )
        A = problem.matrix
        n = A.shape[0]
        n_roots = options.n_roots if options.n_roots > 0 else n
        n_roots = min(n_roots, n)

        # Use scipy when available (supports a subset-by-index parameter);
        # fall back to numpy eigh (always available).
        try:
            import scipy.linalg  # noqa: F401

            _has_scipy = True
        except ImportError:
            _has_scipy = False

        if np.iscomplexobj(A):
            # Hermitian case
            if _has_scipy:
                import scipy.linalg

                if options.which in ("SA", "SM"):
                    w, v = scipy.linalg.eigh(A, subset_by_index=[0, n_roots - 1])
                elif options.which in ("LA", "LM"):
                    w, v = scipy.linalg.eigh(A, subset_by_index=[n - n_roots, n - 1])
                elif options.which == "BE":
                    # Both ends: get first and last
                    k_half = n_roots // 2
                    w_lo, v_lo = scipy.linalg.eigh(A, subset_by_index=[0, k_half - 1])
                    w_hi, v_hi = scipy.linalg.eigh(
                        A, subset_by_index=[n - (n_roots - k_half), n - 1]
                    )
                    w = np.concatenate([w_lo, w_hi])
                    v = np.concatenate([v_lo, v_hi], axis=1)
                else:
                    w, v = scipy.linalg.eigh(A)
            else:
                w, v = np.linalg.eigh(A)
        else:
            # Real symmetric case
            if _has_scipy:
                import scipy.linalg

                if options.which in ("SA", "SM"):
                    w, v = scipy.linalg.eigh(A, subset_by_index=[0, n_roots - 1])
                elif options.which in ("LA", "LM"):
                    w, v = scipy.linalg.eigh(A, subset_by_index=[n - n_roots, n - 1])
                elif options.which == "BE":
                    k_half = n_roots // 2
                    w_lo, v_lo = scipy.linalg.eigh(A, subset_by_index=[0, k_half - 1])
                    w_hi, v_hi = scipy.linalg.eigh(
                        A, subset_by_index=[n - (n_roots - k_half), n - 1]
                    )
                    w = np.concatenate([w_lo, w_hi])
                    v = np.concatenate([v_lo, v_hi], axis=1)
                else:
                    w, v = scipy.linalg.eigh(A)
            else:
                w, v = np.linalg.eigh(A)

        # Slice to requested roots if full diagonalisation was done
        if options.n_roots > 0 and w.size > n_roots:
            if options.which in ("LA", "LM"):
                w = w[-n_roots:]
                v = v[:, -n_roots:]
            else:
                w = w[:n_roots]
                v = v[:, :n_roots]

        # Residuals
        residuals = np.zeros(w.size)
        for i in range(w.size):
            r = A @ v[:, i] - w[i] * v[:, i]
            residuals[i] = np.linalg.norm(r)

        return SolverResult(
            eigenvalues=w,
            eigenvectors=v,
            n_iter=1,  # dense: single-shot
            converged=True,
            residuals=residuals,
            n_matvec=0,
        )


# ── Davidson solver (C++ backend, real symmetric) ──────────────────────────


class DavidsonSolver(SolverStrategy):
    _aliases = ["davidson", "block_davidson"]
    """Blocked Davidson diagonalisation via the C++ native core.

    Wraps ``_vibeqc_core.davidson_solve`` for real symmetric matrices.
    Uses the diagonal (Davidson) preconditioner; requires strong diagonal
    dominance for efficiency (typical of Fock matrices in an orthonormal
    basis).

    References
    ----------
    Davidson, E. R. J. Comput. Phys. 17, 87-94 (1975).
    Liu, B. NRCC Workshop LBL-8158 (1978).
    Kresse, G. & Furthmüller, J. Phys. Rev. B 54, 11169 (1996).
    """

    supports_extremal = True
    supports_interior = False
    supports_generalized = False
    supports_block = True
    supports_matrix_free = True  # via davidson_solve_matvec
    supports_explicit = True
    supports_scf_step = True

    def solve(
        self,
        problem: EigenProblem,
        options: SolverOptions,
    ) -> SolverResult:
        if problem.n <= 0:
            raise ValueError("DavidsonSolver: problem dimension must be > 0.")

        upper_roots = _wants_upper_roots(options)
        solve_problem = _negated_problem(problem) if upper_roots else problem
        solve_options = _lowest_root_options(options)

        _core = _get_core()
        c_opts = _core.DavidsonOptions()

        # Map unified options -> C++ options
        c_opts.n_eig = (
            solve_options.n_roots if solve_options.n_roots > 0 else solve_problem.n
        )
        c_opts.conv_tol = solve_options.tol
        c_opts.n_guess = solve_options.n_guess
        if c_opts.n_guess < 0 or 0 < c_opts.n_guess < c_opts.n_eig:
            raise ValueError("Davidson n_guess must be 0 (auto) or at least n_roots")
        c_opts.max_iter = solve_options.max_iter
        c_opts.preshift = solve_options.preshift
        c_opts.verbosity = solve_options.verbosity
        if solve_options.max_subspace > 0:
            c_opts.max_subspace = solve_options.max_subspace
        if solve_options.guess_vectors is not None:
            c_opts.guess_vectors = np.asarray(
                solve_options.guess_vectors, dtype=np.float64
            )

        # Dispatch: matrix-free vs explicit
        if solve_problem.matvec is not None and solve_problem.matrix is None:
            # Matrix-free path: need diagonal for preconditioner
            # When we only have matvec, compute diag via probing
            n = solve_problem.n
            diag = _matvec_diagonal(solve_problem, dtype=np.float64)
            res = _core.davidson_solve_matvec(n, solve_problem.matvec, diag, c_opts)
        elif solve_problem.matrix is not None:
            A = np.asarray(solve_problem.matrix, dtype=np.float64)
            res = _core.davidson_solve(A, c_opts)
        else:
            raise ValueError("DavidsonSolver: requires either matrix or matvec.")

        eigenvalues = np.asarray(res.eigenvalues)
        eigenvectors = np.asarray(res.eigenvectors)
        if upper_roots:
            eigenvalues = -eigenvalues
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

        # Residuals
        if problem.matrix is not None:
            residuals = np.zeros(len(eigenvalues))
            for i in range(len(eigenvalues)):
                r = (
                    problem.matrix @ eigenvectors[:, i]
                    - eigenvalues[i] * eigenvectors[:, i]
                )
                residuals[i] = np.linalg.norm(r)
        else:
            residuals = None

        return SolverResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_iter=res.n_iter,
            converged=res.converged,
            residuals=residuals,
            n_matvec=res.n_iter * (getattr(res, "subspace_dim", 1) + 1),
        )


# ── Hermitian Davidson solver (C++ backend, complex) ───────────────────────


class HermitianDavidsonSolver(SolverStrategy):
    _aliases = ["hermitian_davidson"]
    """Blocked Davidson for complex Hermitian matrices via the C++ core.

    Wraps ``_vibeqc_core.davidson_solve_hermitian``.  Used for k != Γ
    periodic Fock matrices where the Hamiltonian is complex-symmetric
    (Hermitian) but not purely real.
    """

    supports_extremal = True
    supports_interior = False
    supports_generalized = False
    supports_block = True
    supports_matrix_free = True  # via davidson_solve_hermitian_matvec
    supports_explicit = True
    supports_scf_step = True

    def solve(
        self,
        problem: EigenProblem,
        options: SolverOptions,
    ) -> SolverResult:
        if problem.n <= 0:
            raise ValueError("HermitianDavidsonSolver: problem dimension must be > 0.")

        upper_roots = _wants_upper_roots(options)
        solve_problem = _negated_problem(problem) if upper_roots else problem
        solve_options = _lowest_root_options(options)

        _core = _get_core()
        c_opts = _core.DavidsonOptions()

        c_opts.n_eig = (
            solve_options.n_roots if solve_options.n_roots > 0 else solve_problem.n
        )
        c_opts.conv_tol = solve_options.tol
        c_opts.n_guess = solve_options.n_guess
        if c_opts.n_guess < 0 or 0 < c_opts.n_guess < c_opts.n_eig:
            raise ValueError("Davidson n_guess must be 0 (auto) or at least n_roots")
        c_opts.max_iter = solve_options.max_iter
        c_opts.preshift = solve_options.preshift
        c_opts.verbosity = solve_options.verbosity
        if solve_options.max_subspace > 0:
            c_opts.max_subspace = solve_options.max_subspace
        if solve_options.guess_vectors is not None:
            c_opts.guess_vectors_cplx = np.asarray(
                solve_options.guess_vectors, dtype=np.complex128
            )

        # Dispatch: matrix-free vs explicit
        if solve_problem.matvec is not None and solve_problem.matrix is None:
            n = solve_problem.n
            diag = _matvec_diagonal(solve_problem, dtype=np.complex128)
            res = _core.davidson_solve_hermitian_matvec(
                n, solve_problem.matvec, diag, c_opts
            )
        elif solve_problem.matrix is not None:
            A = np.asarray(solve_problem.matrix, dtype=np.complex128)
            res = _core.davidson_solve_hermitian(A, c_opts)
        else:
            raise ValueError(
                "HermitianDavidsonSolver: requires either matrix or matvec."
            )

        eigenvalues = np.asarray(res.eigenvalues)
        eigenvectors = np.asarray(res.eigenvectors)
        if upper_roots:
            eigenvalues = -eigenvalues
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

        if problem.matrix is not None:
            residuals = np.zeros(len(eigenvalues))
            for i in range(len(eigenvalues)):
                r = (
                    problem.matrix @ eigenvectors[:, i]
                    - eigenvalues[i] * eigenvectors[:, i]
                )
                residuals[i] = np.linalg.norm(r)
        else:
            residuals = None

        return SolverResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_iter=res.n_iter,
            converged=res.converged,
            residuals=residuals,
            n_matvec=res.n_iter * (getattr(res, "subspace_dim", 1) + 1),
        )


# ── Lanczos solver (Python, full reorthogonalisation) ──────────────────────


class LOBPCGSolver(SolverStrategy):
    """LOBPCG -- Locally Optimal Block Preconditioned Conjugate Gradient.

    Reference: Knyazev, A. V. "Toward the Optimal Preconditioned
    Eigensolver." SIAM J. Sci. Comput. 23, 517-541 (2001).
    DOI: 10.1137/S1064827500366124
    """

    _aliases = ["lobpcg"]

    supports_extremal: bool = True
    supports_interior: bool = False
    supports_generalized: bool = False
    supports_block: bool = True
    supports_matrix_free: bool = True
    supports_explicit: bool = True
    supports_scf_step: bool = True

    def solve(self, problem: EigenProblem, options: SolverOptions) -> SolverResult:
        if options.n_roots <= 0:
            raise ValueError("LOBPCG requires n_roots > 0")
        upper_roots = _wants_upper_roots(options)
        solve_problem = _negated_problem(problem) if upper_roots else problem
        solve_options = _lowest_root_options(options)

        core = _get_core()
        lopts = core.LOBPCGOptions()
        lopts.n_eig = solve_options.n_roots
        lopts.tol = solve_options.tol
        lopts.max_iter = solve_options.max_iter
        lopts.preshift = solve_options.preshift
        if solve_options.block_size > 0:
            lopts.block_size = solve_options.block_size
        if solve_options.verbosity > 0:
            lopts.verbosity = solve_options.verbosity
        if solve_problem.matrix is not None:
            res = core.lobpcg_solve(solve_problem.matrix, lopts)
        elif solve_problem.matvec is not None:
            diag = _matvec_diagonal(solve_problem, dtype=solve_problem.dtype)
            res = core.lobpcg_solve_matvec(
                solve_problem.n, solve_problem.matvec, diag, lopts
            )
        else:
            raise ValueError("LOBPCG needs either matrix or matvec")
        result = SolverResult(
            eigenvalues=res.eigenvalues,
            eigenvectors=res.eigenvectors,
            n_iter=res.n_iter,
            converged=res.converged,
        )
        return _flip_upper_root_result(result) if upper_roots else result


class JDSolver(SolverStrategy):
    """Jacobi-Davidson with MINRES correction-equation solver.

    Reference: Sleijpen, G. L. G. & Van der Vorst, H. A.
    "A Jacobi-Davidson iteration method for linear eigenvalue
    problems." SIAM J. Matrix Anal. Appl. 17, 401-425 (1996).
    DOI: 10.1137/S0895479894270427
    """

    _aliases = ["jd", "jacobi_davidson"]

    supports_extremal: bool = True
    supports_interior: bool = (
        True  # experimental: MINRES correction needs full recurrence
    )
    supports_generalized: bool = False
    supports_block: bool = False
    supports_matrix_free: bool = False
    supports_explicit: bool = True
    supports_scf_step: bool = False

    def solve(self, problem: EigenProblem, options: SolverOptions) -> SolverResult:
        if options.n_roots <= 0:
            raise ValueError("JD requires n_roots > 0")
        upper_roots = _wants_upper_roots(options)
        solve_problem = _negated_problem(problem) if upper_roots else problem
        solve_options = _lowest_root_options(options)

        core = _get_core()
        jopts = core.JDOptions()
        jopts.n_eig = solve_options.n_roots
        jopts.tol = solve_options.tol
        jopts.max_iter = solve_options.max_iter
        if solve_options.verbosity > 0:
            jopts.verbosity = solve_options.verbosity
        if solve_options.sigma is not None:
            jopts.sigma = -solve_options.sigma if upper_roots else solve_options.sigma
        if solve_problem.matrix is not None:
            res = core.jd_solve(solve_problem.matrix, jopts)
        else:
            raise ValueError("JD requires an explicit matrix")
        result = SolverResult(
            eigenvalues=res.eigenvalues,
            eigenvectors=res.eigenvectors,
            n_iter=res.n_iter,
            converged=res.converged,
        )
        return _flip_upper_root_result(result) if upper_roots else result


class GPLHRSolver(SolverStrategy):
    """GPLHR -- Generalized Preconditioned Locally Harmonic Residual.

    Interior eigenvalue solver using harmonic Ritz extraction
    and shift-and-project strategy. Targets eigenvalues nearest
    to a specified sigma.

    Reference: Zuev, D. et al. "New algorithms for iterative
    matrix-free eigensolvers in quantum chemistry."
    J. Comput. Chem. 36, 273-284 (2015).
    DOI: 10.1002/jcc.23800
    """

    _aliases = ["gplhr"]

    supports_extremal: bool = False
    supports_interior: bool = True
    supports_generalized: bool = False
    supports_block: bool = True
    supports_matrix_free: bool = False
    supports_explicit: bool = True
    supports_scf_step: bool = False

    def solve(self, problem: EigenProblem, options: SolverOptions) -> SolverResult:
        if options.n_roots <= 0:
            raise ValueError("GPLHR requires n_roots > 0")
        if options.sigma is None:
            raise ValueError("GPLHR requires sigma (target shift)")
        core = _get_core()
        gopts = core.GPLHROptions()
        gopts.n_eig = options.n_roots
        gopts.sigma = options.sigma
        gopts.tol = options.tol
        gopts.max_iter = options.max_iter
        if options.block_size > 0:
            gopts.block_size = options.block_size
        if options.verbosity > 0:
            gopts.verbosity = options.verbosity
        if problem.matrix is not None:
            res = core.gplhr_solve(problem.matrix, gopts)
        else:
            raise ValueError("GPLHR requires an explicit matrix")
        return SolverResult(
            eigenvalues=res.eigenvalues,
            eigenvectors=res.eigenvectors,
            n_iter=res.n_iter,
            converged=res.converged,
        )


class LanczosSolver(SolverStrategy):
    _aliases = ["lanczos"]
    """Lanczos tridiagonalisation with full reorthogonalisation.

    A minimal Python implementation for small / medium matrices, producing
    the tridiagonal T and the Lanczos basis Q, then diagonalising T to
    extract Ritz pairs.

    Full reorthogonalisation (Gram-Schmidt against all previous Lanczos
    vectors) suppresses the loss of orthogonality that plagues the
    classical Lanczos algorithm in finite precision.

    References
    ----------
    Lanczos, C. J. Res. Nat. Bur. Stand. 45, 255 (1950).
    Paige, C. C. J. Inst. Math. Appl. 10, 373 (1972).
    """

    supports_extremal = True
    supports_interior = False
    supports_generalized = False
    supports_block = False
    supports_matrix_free = True
    supports_explicit = True
    supports_scf_step = False

    def solve(
        self,
        problem: EigenProblem,
        options: SolverOptions,
    ) -> SolverResult:
        n = problem.n
        n_krylov = options.ncv or min(n, max(2 * (options.n_roots or 10), 20))
        n_krylov = min(n_krylov, n)
        n_roots = options.n_roots if options.n_roots > 0 else n_krylov
        n_roots = min(n_roots, n_krylov)

        # Build matvec from problem
        if problem.matvec is not None:
            matvec = problem.matvec
        elif problem.matrix is not None:
            A = np.asarray(problem.matrix, dtype=problem.dtype)
            matvec = lambda v: A @ v
        else:
            raise ValueError("LanczosSolver: requires matrix or matvec.")

        # --- Initialise Lanczos ---
        Q = np.zeros((n, n_krylov), dtype=problem.dtype)
        alpha = np.zeros(n_krylov)
        beta = np.zeros(n_krylov)

        # Random initial vector
        rng = np.random.default_rng(42)
        v = rng.normal(size=n).astype(problem.dtype)
        if problem.dtype == np.complex128:
            v = v + 1j * rng.normal(size=n).astype(np.float64)
        beta[0] = np.linalg.norm(v)
        Q[:, 0] = v / beta[0]

        n_matvec = 0
        j = 0
        k = 0

        # --- Lanczos iteration ---
        converged = False
        for j in range(n_krylov - 1):
            # Matrix-vector product
            w = matvec(Q[:, j])
            n_matvec += 1

            alpha[j] = np.real(np.vdot(Q[:, j], w))

            # Residual: r = (A - a_j) q_j - b_j q_{j-1}
            r = w - alpha[j] * Q[:, j]
            if j > 0:
                r = r - beta[j] * Q[:, j - 1]

            # Full reorthogonalisation (Paige 1972)
            for _ in range(2):  # two passes for numerical stability
                for i in range(j + 1):
                    proj = np.vdot(Q[:, i], r)
                    r = r - proj * Q[:, i]

            beta_next = np.linalg.norm(r)
            beta[j + 1] = beta_next
            k = j + 2  # we now have vectors 0..j+1

            if beta_next < options.tol * 1e-2:
                # Breakdown -- stop, T is (j+1)x(j+1)
                k = j + 1
                break

            # Store next Lanczos vector
            Q[:, j + 1] = r / beta_next

            # Optional early convergence check (every 10 steps)
            if (j + 1) % 10 == 0 and j + 1 > n_roots:
                T_small = (
                    np.diag(alpha[: j + 2])
                    + np.diag(beta[1 : j + 2], -1)
                    + np.diag(beta[1 : j + 2], 1)
                )
                theta = np.linalg.eigh(T_small)[0]
                if getattr(self, "_prev_theta", None) is not None:
                    delta = np.max(np.abs(theta[:n_roots] - self._prev_theta[:n_roots]))
                    if delta < options.tol and j + 1 > n_roots + 5:
                        converged = True
                        break
                self._prev_theta = theta.copy()
        else:
            # Loop completed without break -- all n_krylov vectors built.
            # Need one more matvec to get a for the last stored vector.
            k = n_krylov
            j = n_krylov - 1  # update for final-alpha logic below

        # Compute alpha for the final Lanczos vector if needed.
        # When the loop broke early (breakdown or convergence), alpha up to
        # k-1 are already set.  When the loop ran to completion, we need
        # alpha[n_krylov - 1] (the last stored vector's Rayleigh quotient).
        if k == n_krylov and n_krylov > 1:
            w_last = matvec(Q[:, n_krylov - 1])
            n_matvec += 1
            alpha[n_krylov - 1] = np.real(np.vdot(Q[:, n_krylov - 1], w_last))

        # --- Diagonalise tridiagonal T ---
        k_use = min(k, n_krylov)
        T = (
            np.diag(alpha[:k_use])
            + np.diag(beta[1:k_use], -1)
            + np.diag(beta[1:k_use], 1)
        )
        theta, y = np.linalg.eigh(T)

        # Sort based on which parameter
        if options.which in ("LA", "LM"):
            idx = np.argsort(-theta)  # descending
        else:
            idx = np.argsort(theta)  # ascending (SA, SM, BE)
        theta = theta[idx]
        y = y[:, idx]

        # Extract Ritz vectors: x_i = Q_k y_i
        n_out = min(n_roots, k_use)
        eigenvalues = theta[:n_out]
        eigenvectors = Q[:, :k_use] @ y[:, :n_out]

        # Residuals
        residuals = np.zeros(n_out)
        for i in range(n_out):
            r = matvec(eigenvectors[:, i]) - eigenvalues[i] * eigenvectors[:, i]
            residuals[i] = np.linalg.norm(r)
            n_matvec += 1

        return SolverResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_iter=k_use,
            converged=bool(converged or np.all(residuals < options.tol)),
            residuals=residuals,
            n_matvec=n_matvec,
        )


# ── Convenience: SCF step solver ───────────────────────────────────────────


def solve_scf_step(
    scf_step: SCFStep,
    options: SolverOptions | None = None,
    *,
    solver: str = "davidson",
) -> SolverResult:
    """Convenience: solve a Fock diagonalisation step directly from an
    ``SCFStep`` descriptor.

    This handles the orthogonaliser transformation (when ``X`` is
    provided) and subspace recycling (``prev_C``) automatically.

    Parameters
    ----------
    scf_step : SCFStep
        The Fock matrix and optional orthogonaliser / previous coefficients.
    options : SolverOptions, optional
        Solver knobs.
    solver : str
        Which solver to use.

    Returns
    -------
    SolverResult
        Eigenvalues and eigenvectors in the **original** (non-orthogonal)
        basis.
    """
    if options is None:
        options = SolverOptions(n_roots=scf_step.n_occ, which="SA")

    F = np.asarray(scf_step.fock, dtype=np.float64)

    # Build guess from previous coefficients if available
    if scf_step.prev_C is not None and options.guess_vectors is None:
        if scf_step.X is not None:
            X_g = np.asarray(scf_step.X)
            # In the orthonormal basis, the previous eigenvectors are
            # X^T S C_prev = X^{-1} C_prev (since X^T S X = I, S X = X^{-T})
            # For canonical orthogonalisation: X = U s^{-1/2} -> X^T S = X^{-1}
            guess = np.linalg.solve(X_g.T, scf_step.prev_C)
        else:
            guess = scf_step.prev_C
        options.guess_vectors = guess[:, : scf_step.n_occ]

    # Apply orthogonaliser
    if scf_step.X is not None:
        X_mat = np.asarray(scf_step.X)
        F_orth = X_mat.T @ F @ X_mat
        problem = EigenProblem(matrix=F_orth, n=F_orth.shape[0])
        result = solve_eigenproblem(problem, options, solver=solver)
        # Transform back: C = X @ C_orth
        result.eigenvectors = X_mat @ result.eigenvectors
    else:
        problem = EigenProblem(matrix=F, n=F.shape[0])
        result = solve_eigenproblem(problem, options, solver=solver)

    return result


def _scf_diagonalize(
    F: np.ndarray,
    X: np.ndarray,
    n_occ: int,
    *,
    solver: str = "dense",
    prev_C_orth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize F in orthogonalized basis X^T F X.

    Returns (C, eps) where C = X @ C_orth are the MO coefficients
    and eps are the eigenvalues, all in ascending order.
    """
    step = SCFStep(fock=F, X=X, n_occ=n_occ)
    if prev_C_orth is not None:
        step.prev_C = prev_C_orth
    opts = SolverOptions(n_roots=F.shape[0], tol=1e-10, max_iter=100)
    result = solve_scf_step(step, opts, solver=solver)
    return result.eigenvectors, result.eigenvalues
