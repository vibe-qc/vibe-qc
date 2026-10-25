"""Hermitian operator snapshots and explicit solver capabilities.

The native boundary is ``hermitian_operator.hpp``. This Python boundary also
accepts sparse storage and callbacks, without converting them to dense arrays.
See ``docs/design/operator_contract.md`` for ownership and numerical contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from typing import Callable
from types import MappingProxyType

import numpy as np


def _snapshot(value):
    array = np.asarray(value)
    if array.dtype.kind not in "fc":
        array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Operator arrays must be finite")
    # An immutable bytes owner also prevents setflags(write=True). A frozen
    # dataclass or a read-only flag on a caller-owned buffer is insufficient.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _hermitian(matrix, name):
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if np.linalg.norm(matrix - matrix.conj().T) > 1e-10 * max(1., np.linalg.norm(matrix)):
        raise ValueError(f"{name} must be Hermitian")


@dataclass(frozen=True, kw_only=True)
class OperatorFrame:
    """Reusable owned metric and retained coordinates, validated once.

    Supply both S and X for AO coordinates, or neither for an orthonormal
    frame. Basis/order/image identities are producer declarations. The digest
    also covers the actual S/X bytes, so a changed retained space cannot reuse
    a previous subspace merely by keeping the same basis label.
    """
    basis_id: str
    n: int
    ordering: str
    image_gauge: str
    overlap: np.ndarray | None = None
    X: np.ndarray | None = None
    identity: str = field(init=False)

    def __post_init__(self):
        if not self.basis_id or not self.ordering or not self.image_gauge:
            raise ValueError("Frames require basis, ordering and image-gauge identities")
        if not isinstance(self.n, (int, np.integer)) or self.n <= 0:
            raise ValueError("Frame dimension must be a positive integer")
        if (self.overlap is None) != (self.X is None):
            raise ValueError("An AO frame requires both overlap and X")
        digest = hashlib.sha256(json.dumps(
            [self.basis_id, int(self.n), self.ordering, self.image_gauge]
        ).encode())
        if self.overlap is not None:
            s, x = _snapshot(self.overlap), _snapshot(self.X)
            _hermitian(s, "Frame overlap")
            if s.shape != (self.n, self.n):
                raise ValueError("Frame overlap has incompatible shape")
            if x.ndim != 2 or x.shape[0] != self.n or not 0 < x.shape[1] <= self.n:
                raise ValueError("Frame X must have shape (n, retained_rank)")
            if not np.allclose(x.conj().T @ s @ x, np.eye(x.shape[1]), atol=1e-8, rtol=0):
                raise ValueError("Frame X is not orthonormal in the supplied metric")
            object.__setattr__(self, "overlap", s)
            object.__setattr__(self, "X", x)
            for a in (s, x):
                digest.update(str((a.shape, a.dtype.str)).encode())
                digest.update(memoryview(a).cast("B"))
        object.__setattr__(self, "identity", digest.hexdigest())

    @property
    def retained_rank(self):
        return self.n if self.X is None else self.X.shape[1]

    @property
    def storage_nbytes(self):
        return 0 if self.X is None else self.X.nbytes + self.overlap.nbytes


@dataclass(frozen=True)
class OperatorMetadata:
    """Annotations, never instructions to fill or rotate an eigenspace.

    ``basis_id`` identifies the coordinate frame; ``generation`` identifies a
    builder snapshot. Callback owners must create a new object when they change
    captured state. No density-dependent cache is held by this module.
    """
    kind: str = "hermitian"
    basis_id: str = ""
    generation: int = 0
    spin: str | None = None
    kpoint: tuple[float, float, float] | None = None
    symmetry_block: str | None = None
    occupations: tuple[float, ...] | None = None
    frame_id: str | None = None
    build_id: str | None = None
    density_id: str | None = None
    role: str = "physical"
    modification: str | None = None

    def __post_init__(self):
        if self.role not in {"physical", "effective", "solver"}:
            raise ValueError("Unknown operator role")
        if (self.role == "solver") != (self.modification is not None):
            raise ValueError("A solver-modified operator requires a modification description")
        if self.kind not in {"hermitian", "fock", "ks", "one_electron", "ci", "response"}:
            raise ValueError("Unsupported Hermitian operator kind; non-Hermitian operators need a separate solver")
        if self.kpoint is not None:
            if len(self.kpoint) != 3 or not np.all(np.isfinite(self.kpoint)):
                raise ValueError("kpoint must contain three finite coordinates")
            object.__setattr__(self, "kpoint", tuple(self.kpoint))
        if self.occupations is not None:
            if not np.all(np.isfinite(self.occupations)):
                raise ValueError("occupations must be finite")
            object.__setattr__(self, "occupations", tuple(self.occupations))


@dataclass(frozen=True, kw_only=True)
class HermitianOperator:
    """One immutable operator snapshot in a declared coordinate frame.

    Supply either ``matrix`` (dense or SciPy sparse), or ``matvec``/``matmat``
    with an explicit dimension and dtype. Callbacks promise Hermiticity and
    stable captured state for the entire solve. They are never basis-probed.
    Dense/sparse arrays, diagonal, metric and X are copied on construction.
    ``X`` may be rectangular; when supplied with overlap, X^H S X must be I.
    A metric without X uses canonical rank reduction with the stated cutoff.
    Transformed sparse/callback Davidson requires explicit ``X`` and
    ``reduced_diagonal = diag(X^H A X)`` in that exact retained frame.
    """
    matrix: object | None = None
    matvec: Callable | None = None
    matmat: Callable | None = None
    n: int = 0
    dtype: object = np.float64
    diagonal: np.ndarray | None = None
    reduced_diagonal: np.ndarray | None = None
    overlap: np.ndarray | None = None
    X: np.ndarray | None = None
    overlap_cutoff: float = 1e-12
    metadata: OperatorMetadata = field(default_factory=OperatorMetadata)
    frame: OperatorFrame | None = None
    storage: str = field(init=False)

    def __post_init__(self):
        if self.matrix is not None:
            if self.matvec is not None or self.matmat is not None:
                raise ValueError("Provide one representation: matrix or callbacks")
            if hasattr(self.matrix, "tocsr"):
                # Optional dependency, imported only for sparse input.
                from scipy.sparse import csr_matrix
                a = csr_matrix(self.matrix, copy=True)
                a.sum_duplicates()
                a.sort_indices()
                if a.dtype.kind not in "fc":
                    a = a.astype(float)
                if not np.all(np.isfinite(a.data)):
                    raise ValueError("Sparse operator must be finite")
                if a.shape[0] == 0 or a.shape[0] != a.shape[1]:
                    raise ValueError("Sparse operator must be nonempty and square")
                if np.linalg.norm((a - a.conj().T).data) > 1e-10 * max(1., np.linalg.norm(a.data)):
                    raise ValueError("Sparse operator must be Hermitian")
                for array in (a.data, a.indices, a.indptr):
                    array.setflags(write=False)
                storage = "sparse"
                diag = a.diagonal()
            else:
                a = _snapshot(self.matrix)
                _hermitian(a, "Operator")
                storage = "dense"
                diag = np.diag(a)
            if self.n not in (0, a.shape[0]):
                raise ValueError("Operator dimension disagrees with matrix")
            object.__setattr__(self, "matrix", a)
            object.__setattr__(self, "n", a.shape[0])
            object.__setattr__(self, "dtype", np.dtype(a.dtype))
            if self.diagonal is not None and not np.allclose(self.diagonal, diag):
                raise ValueError("Provided diagonal disagrees with stored matrix")
            object.__setattr__(self, "diagonal", _snapshot(diag.real))
        else:
            if not isinstance(self.n, (int, np.integer)) or self.n <= 0 or (self.matvec is None and self.matmat is None):
                raise ValueError("Callback operator needs n > 0 and matvec or matmat")
            dtype = np.dtype(self.dtype)
            if dtype.kind not in "fc":
                raise ValueError("Callback dtype must be real or complex floating point")
            object.__setattr__(self, "dtype", dtype)
            storage = "callback"
            if self.diagonal is not None:
                diag = _snapshot(self.diagonal)
                if diag.shape != (self.n,) or np.any(np.abs(diag.imag) > 1e-12):
                    raise ValueError("Hermitian diagonal must be real with shape (n,)")
                object.__setattr__(self, "diagonal", _snapshot(diag.real))
        object.__setattr__(self, "storage", storage)
        if self.frame is not None:
            frame = self.frame
            if not isinstance(frame, OperatorFrame) or frame.n != self.n:
                raise ValueError("Operator dimension disagrees with its frame")
            if (self.overlap is not None and self.overlap is not frame.overlap
                    or self.X is not None and self.X is not frame.X):
                raise ValueError("Supply a shared frame or separate metric arrays, not both")
            if self.metadata.basis_id not in ("", frame.basis_id):
                raise ValueError("Operator basis identity disagrees with its frame")
            if self.metadata.frame_id not in (None, frame.identity):
                raise ValueError("Operator frame identity disagrees with its metric")
            object.__setattr__(self, "overlap", frame.overlap)
            object.__setattr__(self, "X", frame.X)
            object.__setattr__(self, "metadata", replace(
                self.metadata, basis_id=frame.basis_id, frame_id=frame.identity))
        if not np.isfinite(self.overlap_cutoff) or self.overlap_cutoff < 0:
            raise ValueError("overlap_cutoff must be finite and nonnegative")
        if self.overlap is not None and self.frame is None:
            s = _snapshot(self.overlap)
            _hermitian(s, "Overlap")
            if s.shape != (self.n, self.n):
                raise ValueError("Overlap shape disagrees with operator")
            object.__setattr__(self, "overlap", s)
        if self.X is not None and self.frame is None:
            x = _snapshot(self.X)
            if x.ndim != 2 or x.shape[0] != self.n or not 0 < x.shape[1] <= self.n:
                raise ValueError("X must have shape (n, retained_rank)")
            gram = x.conj().T @ (x if self.overlap is None else self.overlap @ x)
            if self.overlap is not None and not np.allclose(gram, np.eye(x.shape[1]), atol=1e-8, rtol=0):
                raise ValueError("X is not orthonormal in the supplied metric")
            if np.linalg.matrix_rank(x) != x.shape[1]:
                raise ValueError("X must have independent columns")
            object.__setattr__(self, "X", x)
        if self.reduced_diagonal is not None:
            if self.storage == "dense":
                raise ValueError("reduced_diagonal is only for sparse/callback operators; dense solves compute it")
            if self.X is None:
                raise ValueError("reduced_diagonal requires explicit X to identify its retained frame")
            diag = _snapshot(self.reduced_diagonal)
            if diag.shape != (self.X.shape[1],) or np.any(np.abs(diag.imag) > 1e-12):
                raise ValueError("reduced_diagonal must be real with shape (retained_rank,)")
            object.__setattr__(self, "reduced_diagonal", _snapshot(diag.real))

    def apply(self, vectors):
        """Return an owned vector/block action; no implicit dense conversion.

        Callbacks may return temporary workspace, valid until their next call.
        Copy it before another callback can reuse that storage.
        """
        v = np.asarray(vectors)
        if v.ndim not in (1, 2) or v.shape[0] != self.n:
            raise ValueError("Operator input has incompatible shape")
        if self.matrix is not None:
            out = self.matrix @ v
        elif v.ndim == 2 and self.matmat is not None:
            out = self.matmat(v)
        elif v.ndim == 1 and self.matvec is not None:
            out = self.matvec(v)
        elif v.ndim == 1:
            out = np.asarray(self.matmat(v[:, None]))[:, 0]
        else:
            # Each column must be copied before the next matvec can reuse its
            # buffer. Collecting borrowed callback arrays first aliases them.
            return np.column_stack([self.apply(c) for c in v.T])
        out = np.asarray(out)
        if out.shape != v.shape or not np.all(np.isfinite(out)):
            raise ValueError("Operator callback returned invalid shape or nonfinite values")
        if self.dtype.kind != "c" and not np.iscomplexobj(v) and np.iscomplexobj(out) and np.any(out.imag != 0):
            raise ValueError("Real operator callback returned complex values")
        return out.copy() if self.storage == "callback" else out

    def block(self, rows, columns):
        """Explicit storage block for a preconditioner; callbacks must supply their own."""
        if self.matrix is None:
            raise NotImplementedError("Callback block access is not available; provide a preconditioner separately")
        return self.matrix[np.asarray(rows)[:, None], np.asarray(columns)].copy()


@dataclass(frozen=True, kw_only=True)
class FockOperator(HermitianOperator):
    """A one-particle Fock or Kohn-Sham snapshot, with the same algebra."""
    metadata: OperatorMetadata = field(default_factory=lambda: OperatorMetadata(kind="fock"))

    def __post_init__(self):
        if self.metadata.kind not in {"fock", "ks", "one_electron"}:
            raise ValueError("A CI or response operator is not a Fock operator")
        super().__post_init__()


@dataclass(frozen=True)
class SolverCapabilities:
    complex: bool
    matrix_free: bool
    selectors: frozenset[str]
    warm_start: bool = False
    interior: bool = False
    experimental: bool = False


def solver_capabilities(solver):
    """Capabilities of the strict operator entry point, not aspirational flags."""
    from .eigensolver import get_solver
    name = get_solver(solver).name()
    capabilities = {
        "dense": SolverCapabilities(True, False, frozenset({"SA", "LA", "SM", "LM", "BE"}), interior=True),
        "davidson": SolverCapabilities(False, True, frozenset({"SA", "LA"}), warm_start=True),
        "hermitiandavidson": SolverCapabilities(True, True, frozenset({"SA", "LA"}), warm_start=True),
        "lobpcg": SolverCapabilities(False, False, frozenset({"SA", "LA"})),
        "jd": SolverCapabilities(False, False, frozenset({"SA", "LA"}), interior=True, experimental=True),
        "gplhr": SolverCapabilities(False, False, frozenset({"SA"}), interior=True, experimental=True),
        "lanczos": SolverCapabilities(True, True, frozenset({"SA", "LA"}), experimental=True),
    }
    if name not in capabilities:
        raise NotImplementedError(f"{solver} has no strict operator capability contract")
    return capabilities[name]


class SolverConvergenceError(RuntimeError):
    """Refused eigensolution with bounded, immutable diagnostic evidence.

    ``diagnostics`` contains scalar status and limits; ``arrays`` contains
    at most eight MiB of numeric snapshots, or is empty with an explicit
    omission reason. These are failed solver data, never an accepted SCF state.
    No callback is invoked and no file is written to collect this evidence.
    """
    def __init__(self, solver, diagnostics, arrays):
        self.diagnostics = MappingProxyType(dict(diagnostics))
        self.arrays = MappingProxyType(dict(arrays))
        super().__init__(
            f"{solver} failed the requested residual/orthogonality contract "
            f"(backend_converged={diagnostics['backend_converged']}, "
            f"max_projected_residual={diagnostics['max_projected_residual']}, "
            f"tol={diagnostics['residual_tolerance']}, "
            f"orthogonality_error={diagnostics['orthogonality_error']}, "
            f"orthogonality_tolerance={diagnostics['orthogonality_tolerance']})"
        )


_FAILURE_EVIDENCE_MAX_BYTES = 8 * 1024**2


def _solver_convergence_error(solver, operator, options, result,
                              backend_converged, backend_residuals, arrays):
    """Capture only values already evaluated by the original contract."""
    def finite_number(value):
        return float(value) if np.isfinite(value) else None

    def maximum(values):
        return None if values is None or not np.size(values) else finite_number(np.max(values))

    diagnostics = {
        "schema": "vibeqc.solver-refusal.v1",
        "solver": str(solver), "converged": False, "usable_as_solution": False,
        "backend_converged": bool(backend_converged),
        "max_backend_residual": maximum(backend_residuals),
        "max_projected_residual": maximum(result.projected_residuals),
        "projected_residuals_finite": bool(np.all(np.isfinite(result.projected_residuals))),
        "residual_tolerance": float(options.tol),
        "orthogonality_error": finite_number(result.orthogonality_error),
        "orthogonality_tolerance": 1e-7,
        "n_converged": int(result.n_converged), "requested_roots": int(options.n_roots),
        "retained_rank": int(result.retained_rank), "operator_dimension": int(operator.n),
        "operator_storage": operator.storage, "operator_dtype": str(operator.dtype),
        "operator_kind": operator.metadata.kind,
        "residual_frame": result.residual_frame,
        "overlap_supplied": operator.overlap is not None,
        "orthogonalizer_supplied": operator.X is not None,
        "matrix_free_operator_materialized": False,
        "evidence_byte_limit": _FAILURE_EVIDENCE_MAX_BYTES,
    }
    available = {key: value for key, value in arrays.items() if value is not None}
    required = sum(value.nbytes for value in available.values())
    diagnostics["evidence_required_bytes"] = int(required)
    saved = {}
    if required > _FAILURE_EVIDENCE_MAX_BYTES:
        diagnostics["evidence_status"] = "omitted_byte_limit"
    else:
        try:
            # Unlike accepted operator snapshots, failed results may contain
            # NaN/Inf. Preserve those bytes for diagnosis, without promoting them.
            for key, value in available.items():
                saved[key] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            diagnostics["evidence_status"] = "captured"
        except MemoryError:
            saved.clear()
            diagnostics["evidence_status"] = "omitted_allocation_failure"
    diagnostics["evidence_bytes"] = sum(value.nbytes for value in saved.values())
    return SolverConvergenceError(solver, diagnostics, saved)


def solve_operator(operator, options=None, *, solver="dense", previous=None,
                   require_selection=False):
    """Solve a snapshot with independently checked residual/metric diagnostics.

    ``previous`` is a prior SolverResult in the same coordinate frame. A guess
    is a subspace seed, not a root identity or completeness certificate.
    ``require_selection`` refuses iterative backends without a certificate;
    dense selection is certified by enumerating the full finite spectrum.
    """
    from .eigensolver import EigenProblem, SolverOptions, get_solver, solve_eigenproblem
    options = replace(options) if options is not None else SolverOptions()
    strategy = get_solver(solver)
    name = strategy.name()
    caps = solver_capabilities(solver)
    if require_selection and name != "dense":
        raise NotImplementedError("Requested-root selection is not certified by this iterative backend")
    if operator.dtype.kind == "c" and not caps.complex:
        raise NotImplementedError(f"{solver} does not support complex operators")
    if operator.storage != "dense" and not caps.matrix_free:
        raise NotImplementedError(f"{solver} does not support matrix-free/sparse storage without materialization")
    if options.which not in caps.selectors:
        raise NotImplementedError(f"{solver} does not support selector {options.which}")
    if options.sigma is not None and not caps.interior:
        raise NotImplementedError(f"{solver} does not support an interior shift")
    if options.guess_vectors is not None and previous is not None:
        raise ValueError("Supply previous or guess_vectors, not both")
    if (previous is not None or options.guess_vectors is not None) and not caps.warm_start:
        raise NotImplementedError(f"{solver} does not support warm starts")
    accepted = {
        "dense": set(),
        "davidson": {"max_iter", "max_subspace", "preshift", "verbosity", "n_guess", "guess_vectors"},
        "hermitiandavidson": {"max_iter", "max_subspace", "preshift", "n_guess", "guess_vectors"},
        "lobpcg": {"max_iter", "block_size", "preshift", "verbosity"},
        "jd": {"max_iter", "verbosity"},
        "gplhr": {"max_iter", "max_subspace", "block_size", "preshift", "verbosity"},
        "lanczos": {"ncv"},
    }[name] | {"n_roots", "which", "sigma", "tol"}
    defaults = SolverOptions()
    for item in fields(options):
        if item.name not in accepted and getattr(options, item.name) != getattr(defaults, item.name):
            raise NotImplementedError(f"{solver} does not implement option {item.name}")
    if options.max_iter <= 0 or not np.isfinite(options.tol) or options.tol <= 0:
        raise ValueError("max_iter and tol must be positive and finite")
    if options.sigma is not None and not np.isfinite(options.sigma):
        raise ValueError("sigma must be finite")

    x = operator.X
    if x is None and operator.overlap is not None:
        w, u = np.linalg.eigh(operator.overlap)
        cutoff = max(operator.overlap_cutoff, operator.overlap_cutoff * np.max(np.abs(w)))
        if w[0] < -cutoff:
            raise ValueError("Overlap is indefinite")
        keep = w > cutoff
        if not np.any(keep):
            raise ValueError("Overlap has no positive retained directions")
        x = u[:, keep] / np.sqrt(w[keep])
    n = operator.n if x is None else x.shape[1]
    effective_dtype = np.result_type(operator.dtype, x.dtype) if x is not None else operator.dtype
    if effective_dtype.kind == "c" and not caps.complex:
        raise NotImplementedError(f"{solver} does not support complex reduced operators")
    if not isinstance(options.n_roots, (int, np.integer)):
        raise ValueError("n_roots must be an integer")
    k = options.n_roots or n
    if not 1 <= k <= n:
        raise ValueError("n_roots must fit the retained operator rank")
    options.n_roots = k
    if name == "lanczos" and options.ncv is not None:
        if not isinstance(options.ncv, (int, np.integer)) or options.ncv < k:
            raise ValueError("ncv must be an integer at least n_roots")
    if name in {"davidson", "hermitiandavidson"} and options.max_subspace:
        floor = min(n, k + min(5, n-k) + k + 1)
        if options.max_subspace < floor:
            raise ValueError("max_subspace is below the native Davidson restart floor")
    if operator.storage != "dense" and name in {"davidson", "hermitiandavidson"} and k < n < 2*k + 6:
        raise NotImplementedError("Davidson below its restart floor would materialize the callback operator")
    if operator.storage != "dense" and x is not None and name in {"davidson", "hermitiandavidson"}:
        if operator.reduced_diagonal is None:
            raise NotImplementedError("Transformed sparse/callback Davidson requires explicit X and reduced_diagonal = diag(X^H A X)")

    guess = options.guess_vectors
    if previous is not None:
        if previous.operator_metadata is None or (
            previous.operator_metadata.kind,
            previous.operator_metadata.basis_id,
            previous.operator_metadata.spin,
            previous.operator_metadata.kpoint,
            previous.operator_metadata.symmetry_block,
            previous.operator_metadata.frame_id,
        ) != (operator.metadata.kind, operator.metadata.basis_id, operator.metadata.spin,
              operator.metadata.kpoint, operator.metadata.symmetry_block, operator.metadata.frame_id):
            raise ValueError("Previous subspace belongs to a different operator frame")
        guess = previous.eigenvectors
    if guess is not None:
        guess = np.asarray(guess)
        if guess.ndim != 2 or guess.shape[0] != operator.n or guess.shape[1] < k or not np.all(np.isfinite(guess)):
            raise ValueError("Warm-start vectors must have shape (n, at least n_roots) and be finite")
        # Coordinates C = X Y. Handles rectangular and nonsymmetric X too.
        if x is not None:
            guess = np.linalg.lstsq(x, guess, rcond=None)[0]
        if not caps.complex and np.iscomplexobj(guess):
            raise NotImplementedError(f"{solver} does not support complex warm-start vectors")
        options.guess_vectors = guess

    count = 0
    def apply(v):
        nonlocal count
        count += 1 if v.ndim == 1 else v.shape[1]
        return operator.apply(v)
    def reduced_apply(v):
        return apply(v) if x is None else x.conj().T @ apply(x @ v)
    if operator.storage == "dense":
        a = operator.matrix if x is None else x.conj().T @ operator.matrix @ x
        problem = EigenProblem(matrix=a)
    else:
        diagonal = operator.diagonal if x is None else operator.reduced_diagonal
        if name in {"davidson", "hermitiandavidson"} and diagonal is None:
            raise ValueError("Callback Davidson requires a supplied diagonal; implicit probing is disabled")
        problem = EigenProblem(matvec=reduced_apply, n=n, dtype=effective_dtype.type,
                               diagonal=diagonal, matmat=reduced_apply)
    result = solve_eigenproblem(problem, options, solver=solver)
    if result.eigenvalues.shape != (k,) or result.eigenvectors.shape != (n, k):
        raise RuntimeError(f"{solver} returned fewer than the requested roots")
    backend_converged = result.converged
    backend_residuals = result.residuals
    order = np.argsort(result.eigenvalues, kind="stable")
    result.eigenvalues = result.eigenvalues[order]
    result.eigenvectors = result.eigenvectors[:, order]
    y = result.eigenvectors
    c = y if x is None else x @ y
    ac = apply(c)
    sc = c if operator.overlap is None else operator.overlap @ c
    # With X alone, raw AO generalized residuals are undefined without S.
    residual = ac - sc * result.eigenvalues
    reduced_residual = reduced_apply(y) - y * result.eigenvalues
    result.residuals = np.linalg.norm(residual if operator.overlap is not None or x is None else reduced_residual, axis=0)
    result.projected_residuals = np.linalg.norm(reduced_residual, axis=0)
    result.residual_frame = "original" if operator.overlap is not None or x is None else "orthogonal"
    gram = c.conj().T @ sc if operator.overlap is not None or x is None else y.conj().T @ y
    result.orthogonality_error = float(np.linalg.norm(gram - np.eye(k)))
    result.converged = bool(result.converged and len(result.eigenvalues) == k
                            and np.all(result.projected_residuals <= options.tol)
                            and result.orthogonality_error <= 1e-7)
    result.n_converged = int(np.count_nonzero(result.projected_residuals <= options.tol))
    result.eigenvectors = c
    result.retained_rank = n
    result.selection_verified = name == "dense"
    result.operator_metadata = operator.metadata
    # Only the callback count is observable; never estimate a native count.
    result.n_matvec = count if operator.storage != "dense" else None
    if not result.converged:
        raise _solver_convergence_error(
            solver, operator, options, result, backend_converged, backend_residuals,
            {
                "operator_matrix": operator.matrix if operator.storage == "dense" else None,
                "overlap": operator.overlap, "orthogonalizer": x,
                "reduced_matrix": a if operator.storage == "dense" else None,
                "eigenvalues": result.eigenvalues, "ritz_vectors": y,
                "eigenvectors": c, "projected_residual_vectors": reduced_residual,
                "original_residual_vectors": residual if operator.overlap is not None or x is None else None,
                "projected_residuals": result.projected_residuals,
                "reported_residuals": result.residuals,
                "metric_gram": gram,
                # Backend residuals precede the stable presentation ordering.
                "backend_residuals": backend_residuals, "backend_sort_order": order,
            },
        )
    return result


def subspace_overlap(previous, current, *, overlap=None):
    """Return overlap amplitudes and principal cosines for root tracking.

    Inputs must contain metric-orthonormal columns in the same AO frame. Use
    the singular values for a degenerate cluster: individual columns inside
    that cluster have no unique identity. This does not reorder occupations.
    """
    a, b = np.asarray(previous), np.asarray(current)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Subspaces must have the same row dimension")
    sa = a if overlap is None else np.asarray(overlap) @ a
    sb = b if overlap is None else np.asarray(overlap) @ b
    if not np.allclose(a.conj().T @ sa, np.eye(a.shape[1]), atol=1e-7, rtol=0) or not np.allclose(
        b.conj().T @ sb, np.eye(b.shape[1]), atol=1e-7, rtol=0
    ):
        raise ValueError("Subspaces must be metric orthonormal")
    amplitudes = a.conj().T @ sb
    return amplitudes, np.linalg.svd(amplitudes, compute_uv=False)
