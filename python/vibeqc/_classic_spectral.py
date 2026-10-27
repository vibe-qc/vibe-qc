"""Opt-in precision control for the unchanged classic fitting metric.

Sun et al., JCP 147, 164119 (2017), Sec. II B, specifies spectral removal
below the absolute metric threshold. This module changes arithmetic only:
it never edits the matrix, moves the threshold or restores discarded modes.
"""
from __future__ import annotations

import numbers
import numpy as np


def spectral_workspace_bytes(size, digits):
    """Conservative numerical workspace reserve, not a process RSS guarantee."""
    if isinstance(digits, bool) or not isinstance(digits, numbers.Integral) or not 30 <= digits <= 100:
        raise ValueError('metric_precision_digits must be an integer from 30 to 100')
    # Two precisions, eigensolver scratch and matrix functions. Object and
    # allocator overhead remain outside the source numerical-payload budget.
    words = (int(np.ceil((digits+20)*np.log2(10)))+63)//64
    return 64*int(size)**2*(256+16*words)


def retained_spectral_construction(metric, threshold, *, digits, memory_byte_cap):
    """Return a checked eigensystem, canonical inverse root and receipt.

    Both solves consume the identical binary Hermitian matrix. Compare whole
    projectors and matrix functions, not eigenvectors in degenerate spaces.
    The strict > threshold mask must equal the original double-precision
    mask at both precisions. Any ambiguous boundary or failed check refuses.
    A private mpmath context avoids changing application-wide precision.
    """
    m = np.asarray(metric)
    if (m.ndim != 2 or m.shape[0] != m.shape[1] or not len(m)
            or not np.isfinite(m).all() or not np.array_equal(m, m.conj().T)):
        raise ValueError('precision construction requires a finite Hermitian metric')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('metric threshold must be finite and nonnegative')
    reserve = spectral_workspace_bytes(len(m), digits)
    if memory_byte_cap < reserve:
        raise MemoryError('precision-controlled metric exceeds its numerical workspace budget')
    try:
        import mpmath
    except ImportError as exc:
        raise ImportError('metric_precision_digits requires the optional vibeqc[chi-precision] extra') from exc
    baseline = np.linalg.eigh(m)[0] > threshold
    ctx = mpmath.mp.clone()
    results = []
    for precision in (int(digits), int(digits)+20):
        ctx.dps = precision
        a = ctx.matrix(m.tolist())
        values, vectors = ctx.eighe(a)
        keep = np.array([value > ctx.mpf(float(threshold)) for value in values])
        if not np.array_equal(keep, baseline) or not keep.any():
            raise ValueError('precision construction would change the retained metric rank')
        residual = max(abs(x) for x in a*vectors-vectors*ctx.diag(values))
        orthogonality = max(abs(x) for x in vectors.H*vectors-ctx.eye(len(m)))
        scale = max(ctx.mpf(1), max(abs(x) for x in a))
        accuracy = ctx.power(10, -precision+8)
        if residual > accuracy*scale or orthogonality > accuracy:
            raise ValueError('precision-controlled metric eigensystem did not converge')
        margin = min(abs(value-ctx.mpf(float(threshold))) for value in values)
        if margin <= 100*len(m)*max(residual, accuracy*scale):
            raise ValueError('metric rank boundary is unresolved at requested precision')
        selected = [i for i, flag in enumerate(keep) if flag]
        u = vectors[:, selected[0]:selected[-1]+1]
        p = u*u.H
        w = u*ctx.diag([1/ctx.sqrt(values[i]) for i in selected])*u.H
        results.append((values, vectors, p, w, residual, orthogonality, margin))
    low, high = results
    projector_delta = max(abs(x) for x in low[2]-high[2])
    inverse_root_delta = max(abs(x) for x in low[3]-high[3])
    # Well below binary64 output error and the unchanged 1e-10 entry gate.
    if max(projector_delta, inverse_root_delta) > ctx.mpf('1e-20'):
        raise ValueError('retained metric matrix functions are not precision-converged')
    values = np.array([float(x) for x in high[0]])
    vectors = np.array(high[1].tolist(), dtype=complex)
    inverse_root = np.array(high[3].tolist(), dtype=complex)
    if not np.array_equal(values > threshold, baseline):
        raise ValueError('binary64 conversion changes the retained metric rank')
    receipt = dict(algorithm='mpmath-eighe-two-precision-v1', digits=int(digits),
                   verification_digits=int(digits)+20, dependency_version=mpmath.__version__,
                   rank=int(baseline.sum()), numerical_workspace_bytes=reserve,
                   projector_precision_difference=float(projector_delta),
                   inverse_root_precision_difference=float(inverse_root_delta),
                   eigen_residual=float(high[4]), orthogonality=float(high[5]),
                   threshold_distance=float(high[6]), matrix_unchanged=True,
                   original_double_rank_preserved=True)
    return values, vectors, inverse_root, receipt
