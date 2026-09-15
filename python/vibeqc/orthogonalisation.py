"""Orthogonalisation methods for near-singular overlap matrices --
explicit opt-in fallbacks when the user has decided to run SCF
despite a critical-severity preflight diagnosis.

Three methods, in order of conditioning robustness:

1. **Canonical orthogonalisation** (Löwdin) -- diagonalise S, drop
   eigenvectors with eigenvalue below ``threshold``, build
   ``X = U Λ^{-1/2}`` on the kept subspace. Standard since 1970,
   adequate for moderate ill-conditioning. Already used internally
   in the Γ-only and multi-k SCF drivers via
   ``_canonical_orthogonalizer``.

2. **Pivoted Cholesky orthogonalisation** (Lehtola) -- replaces the
   eigendecomposition with a pivoted Cholesky decomposition of S,
   which is numerically more robust at extreme overcompleteness.
   Yields an explicit selection of which basis functions to keep
   (rather than which linear combinations) -- the surviving basis is
   a strict subset of the original, useful for analysis. Reference:
   Lehtola, S. *J. Chem. Phys.* **151**, 241102 (2019);
   *Phys. Rev. A* **101**, 032504 (2020).

3. **Symmetric (Löwdin) orthogonalisation** ``S^{-1/2}`` -- the
   non-truncating reference for benchmarking; if S is well-
   conditioned this is the cheapest option. Provided here for
   completeness; not recommended on near-singular S since the
   inverse-sqrt of the smallest eigenvalue dominates the
   conditioning of X.

Selection policy
----------------

* CRYSTAL's stance -- refuse to silently truncate; push the problem
  to basis-set design -- is vibe-qc's default (item 1+4 in
  ``bbe2b2a``).
* PySCF's stance -- drop near-zero eigenvectors and run anyway,
  optionally with pivoted Cholesky if cond(S) is extreme -- is
  available here as **opt-in** behaviour.

Users who hit a critical-severity preflight diagnosis must
explicitly choose to use these methods; they're not auto-applied.

Design -- ``orthogonalise_overlap``
----------------------------------

.. code-block:: python

    X, info = vq.orthogonalise_overlap(
        S,
        method="auto",            # canonical | cholesky | symmetric | auto
        threshold=1e-7,
        cholesky_threshold=1e-9,
        cholesky_trigger_cond=1e15,   # auto: switch from canonical to
                                      # cholesky when cond(S) > this
        normalize_diag_first=True,    # PySCF-aligned, item 3
    )

    # X is (n_bf, n_kept); X^T S X = I_{n_kept}.
    # info is an OrthogonalisationInfo carrying which method was
    # used, n_kept, condition number, etc.

When ``method='auto'``: pick canonical if cond(S) < ``cholesky_trigger_cond``
(default 1e15); otherwise pivoted Cholesky. This mirrors PySCF's
``remove_linear_dep_`` graceful-degradation logic without making it
the default SCF behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


__all__ = [
    "OrthogonalisationInfo",
    "orthogonalise_overlap",
    "canonical_orth",
    "pivoted_cholesky_orth",
    "symmetric_orth",
]


@dataclass
class OrthogonalisationInfo:
    """Metadata returned alongside the orthogonaliser."""

    method: str                       # "canonical" | "cholesky" | "symmetric"
    n_basis: int
    n_kept: int
    threshold: float
    condition_number: float
    normalize_diag_first: bool
    # Pivoted-Cholesky only: indices of basis functions kept
    # (sorted ascending).
    cholesky_indices: Optional[np.ndarray] = None


def _diag_normalisation(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(scale, S_normalised)`` for the PySCF-style
    pre-conditioning. ``scale[i] = 1 / sqrt(S[i, i])``; ``S_norm =
    diag(scale) @ S @ diag(scale)`` has unit diagonal.

    Returns ``(None, S)`` when the diagonal has non-positive entries
    (caller should fall back to raw-S path).
    """
    d = np.real(np.asarray(S).diagonal())
    if np.any(d <= 0):
        return None, S
    scale = 1.0 / np.sqrt(d)
    S_norm = S * np.outer(scale, scale)
    return scale, S_norm


# ---------------------------------------------------------------------------
# Method 1: canonical (Löwdin)
# ---------------------------------------------------------------------------

def canonical_orth(
    S: np.ndarray,
    threshold: float = 1e-7,
    *,
    normalize_diag_first: bool = True,
) -> Tuple[np.ndarray, OrthogonalisationInfo]:
    """Löwdin canonical orthogonalisation. Returns ``(X, info)`` with
    ``X^+ S X = I_{n_kept}``.

    Mirrors :func:`vibeqc.periodic_rhf_ewald._canonical_orthogonalizer`
    exposed publicly with the ``OrthogonalisationInfo`` wrapper, so
    users can opt in / inspect the method choice / record the
    n_kept reduction post hoc.
    """
    n = S.shape[0]
    if normalize_diag_first:
        scale, S_use = _diag_normalisation(S)
        if scale is None:
            normalize_diag_first = False
            S_use = S
    else:
        scale = None
        S_use = S

    eigvals, eigvecs = np.linalg.eigh(S_use)
    mask = eigvals >= threshold
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"canonical_orth: no eigenvalue above threshold {threshold:.1e}; "
            "basis is fully linearly dependent."
        )
    kept_vals = eigvals[mask]
    kept_vecs = eigvecs[:, mask]
    X_norm = kept_vecs / np.sqrt(kept_vals)
    if scale is not None:
        X = scale[:, None] * X_norm
    else:
        X = X_norm

    cond = (
        kept_vals.max() / kept_vals.min()
        if kept_vals.min() > 0 else float("inf")
    )
    info = OrthogonalisationInfo(
        method="canonical",
        n_basis=n,
        n_kept=n_kept,
        threshold=float(threshold),
        condition_number=float(cond),
        normalize_diag_first=bool(normalize_diag_first),
    )
    return X, info


# ---------------------------------------------------------------------------
# Method 2: pivoted Cholesky (Lehtola)
# ---------------------------------------------------------------------------

def pivoted_cholesky_orth(
    S: np.ndarray,
    threshold: float = 1e-7,
    *,
    cholesky_threshold: float = 1e-9,
    normalize_diag_first: bool = True,
) -> Tuple[np.ndarray, OrthogonalisationInfo]:
    """Lehtola's pivoted-Cholesky orthogonalisation.

    Replaces the eigendecomposition with a pivoted Cholesky
    decomposition of S, then runs canonical orthogonalisation on the
    surviving sub-basis. Numerically robust at extreme overcompleteness
    where the bare eigendecomposition is itself unstable.

    The pivot indices identify which AOs to keep; the rest are
    dropped (rather than linear combinations of all AOs being
    dropped, as canonical orth does). Useful when the user wants to
    know *which* basis functions are responsible for the
    overcompleteness.

    Reference
    ---------
    Lehtola, S. *J. Chem. Phys.* **151**, 241102 (2019);
    *Phys. Rev. A* **101**, 032504 (2020).

    Implementation note
    -------------------
    SciPy >= 1.10 ships ``scipy.linalg.lapack.dpstrf`` for pivoted
    Cholesky. We implement a numpy-only version here to avoid the
    SciPy dependency shape; for production-grade performance on
    large bases the SciPy path can be substituted later.

    The numpy-only pivoted Cholesky:
      - Initialise pivot perm = [0, 1, ..., n-1]; diag d = diag(S).
      - For step k = 0 .. r-1:
          j = argmax(d[k:]) + k
          if d[j] < cholesky_threshold: break.
          swap perm[k] <-> perm[j], swap rows/cols of S accordingly.
          L[k, k] = sqrt(d[k]).
          For i = k+1 .. n-1:
            L[i, k] = (S[perm[i], perm[k]] - S_m<k L[i, m] * L[k, m]) / L[k, k].
            d[i] -= L[i, k]**2.
      - Return permuted indices [perm[0], ..., perm[r-1]] as the
        kept-AO subset.
    """
    n = S.shape[0]
    if normalize_diag_first:
        scale, S_use = _diag_normalisation(S)
        if scale is None:
            normalize_diag_first = False
            S_use = S
    else:
        scale = None
        S_use = S

    # Pivoted Cholesky: pick the most-significant pivot at each step.
    # Numpy-only implementation -- adequate for testing + research; a
    # SciPy LAPACK call would be the production-grade swap.
    A = np.asarray(S_use, dtype=float).copy()
    perm = np.arange(n)
    L = np.zeros((n, n), dtype=float)
    rank = 0
    diag = A.diagonal().copy()

    for k in range(n):
        # Pick max remaining diagonal as pivot.
        j_local = int(np.argmax(diag[k:n]))
        j = k + j_local
        if diag[j] < cholesky_threshold:
            break
        # Swap k <-> j in perm and in A's rows/cols.
        if j != k:
            perm[[k, j]] = perm[[j, k]]
            A[[k, j], :] = A[[j, k], :]
            A[:, [k, j]] = A[:, [j, k]]
            diag[[k, j]] = diag[[j, k]]
            L[[k, j], :] = L[[j, k], :]
        # L[k,k] = sqrt of remaining diagonal.
        L[k, k] = np.sqrt(diag[k])
        if k + 1 < n:
            # L[k+1:, k] = (A[k+1:, k] - L[k+1:, :k] @ L[k, :k]) / L[k, k]
            update = A[k + 1:, k] - L[k + 1:, :k] @ L[k, :k]
            L[k + 1:, k] = update / L[k, k]
            diag[k + 1:] -= L[k + 1:, k] ** 2
        rank += 1

    if rank == 0:
        raise RuntimeError(
            f"pivoted_cholesky_orth: every diagonal pivot below threshold "
            f"{cholesky_threshold:.1e}; basis is fully linearly dependent."
        )

    # Kept AO indices (sorted for deterministic output).
    kept_idx = np.sort(perm[:rank])

    # Run canonical orth on the sub-basis selected by Cholesky pivots.
    S_sub = np.asarray(S)[np.ix_(kept_idx, kept_idx)]
    X_sub, _info = canonical_orth(
        S_sub, threshold=threshold,
        normalize_diag_first=normalize_diag_first,
    )

    # Lift back to full basis: X has shape (n, n_kept_sub) with
    # zeros in the discarded rows.
    X = np.zeros((n, X_sub.shape[1]), dtype=X_sub.dtype)
    X[kept_idx, :] = X_sub

    eigvals_sub = np.linalg.eigvalsh(S_sub)
    cond_sub = (
        eigvals_sub.max() / eigvals_sub.min()
        if eigvals_sub.min() > 0 else float("inf")
    )
    info = OrthogonalisationInfo(
        method="cholesky",
        n_basis=n,
        n_kept=int(X.shape[1]),
        threshold=float(threshold),
        condition_number=float(cond_sub),
        normalize_diag_first=bool(normalize_diag_first),
        cholesky_indices=kept_idx,
    )
    return X, info


# ---------------------------------------------------------------------------
# Method 3: symmetric (Löwdin S^{-1/2})
# ---------------------------------------------------------------------------

def symmetric_orth(
    S: np.ndarray,
) -> Tuple[np.ndarray, OrthogonalisationInfo]:
    """Plain ``S^{-1/2}`` (Löwdin symmetric orthogonalisation).

    Non-truncating. Recommended only when ``S`` is well-conditioned
    (cond < 1/eps_machine ≈ 1e15); on near-singular S the smallest
    eigenvalue dominates the conditioning of ``X``. Provided for
    completeness and benchmarking against the truncating methods.
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    if eigvals.min() <= 0:
        raise ValueError(
            "symmetric_orth: S is not positive-definite "
            f"(min eigenvalue = {eigvals.min():.3e}). Use canonical_orth "
            "or pivoted_cholesky_orth on near-singular S, or fix the "
            "basis-set / screening conditioning first."
        )
    n = S.shape[0]
    inv_sqrt_eigs = 1.0 / np.sqrt(eigvals)
    X = eigvecs @ np.diag(inv_sqrt_eigs) @ eigvecs.conj().T
    cond = float(eigvals.max() / eigvals.min())
    info = OrthogonalisationInfo(
        method="symmetric",
        n_basis=n,
        n_kept=n,
        threshold=0.0,
        condition_number=cond,
        normalize_diag_first=False,
    )
    return X, info


# ---------------------------------------------------------------------------
# Auto-dispatch entry point
# ---------------------------------------------------------------------------

def orthogonalise_overlap(
    S: np.ndarray,
    *,
    method: str = "auto",
    threshold: float = 1e-7,
    cholesky_threshold: float = 1e-9,
    cholesky_trigger_cond: float = 1e15,
    normalize_diag_first: bool = True,
) -> Tuple[np.ndarray, OrthogonalisationInfo]:
    """Build the AO orthogonaliser X with explicit method choice.

    Parameters
    ----------
    S
        ``(n, n)`` overlap matrix (real symmetric or Hermitian
        complex). Caller is expected to have already passed the
        preflight check or explicitly opted past it via
        ``raise_if_severe(allow_critical=True)``.
    method
        ``"canonical"``: Löwdin canonical orthogonalisation with
        eigenvalue truncation. Standard and adequate for moderate
        ill-conditioning.

        ``"cholesky"``: Lehtola pivoted Cholesky orthogonalisation.
        Numerically robust at extreme overcompleteness where the
        eigendecomposition itself is unstable.

        ``"symmetric"``: plain ``S^{-1/2}``. Non-truncating;
        recommended only on well-conditioned S.

        ``"auto"`` (default): pick canonical if cond(S) below
        ``cholesky_trigger_cond``, else cholesky. Mirrors PySCF's
        ``remove_linear_dep_`` graceful-degradation logic.
    threshold
        Eigenvalue truncation threshold for canonical / cholesky
        methods. ``1e-7`` matches PySCF's default.
    cholesky_threshold
        Pivot-magnitude threshold for the Cholesky method. ``1e-9``
        matches PySCF's default.
    cholesky_trigger_cond
        ``"auto"`` mode switches from canonical to cholesky when
        ``cond(S) > cholesky_trigger_cond``. Default ``1e15`` ≈
        ``1 / eps_machine``.
    normalize_diag_first
        PySCF-style sqrt(diag(S)) pre-conditioning before the
        eigendecomposition. Default True (recommended).

    Returns
    -------
    X : np.ndarray of shape (n_bf, n_kept)
    info : :class:`OrthogonalisationInfo`
    """
    if method == "symmetric":
        return symmetric_orth(S)
    if method == "canonical":
        return canonical_orth(
            S, threshold=threshold,
            normalize_diag_first=normalize_diag_first,
        )
    if method == "cholesky":
        return pivoted_cholesky_orth(
            S, threshold=threshold,
            cholesky_threshold=cholesky_threshold,
            normalize_diag_first=normalize_diag_first,
        )
    if method == "auto":
        # Compute cond(S) cheaply via eigvals.
        eigvals = np.linalg.eigvalsh(S)
        if eigvals.min() > 0 and np.isfinite(eigvals.max() / eigvals.min()):
            cond = float(eigvals.max() / eigvals.min())
        else:
            cond = float("inf")
        if cond < cholesky_trigger_cond:
            return canonical_orth(
                S, threshold=threshold,
                normalize_diag_first=normalize_diag_first,
            )
        return pivoted_cholesky_orth(
            S, threshold=threshold,
            cholesky_threshold=cholesky_threshold,
            normalize_diag_first=normalize_diag_first,
        )
    raise ValueError(
        f"orthogonalise_overlap: unknown method {method!r}. "
        "Choose from 'canonical', 'cholesky', 'symmetric', 'auto'."
    )
