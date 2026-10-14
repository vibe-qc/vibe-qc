"""Admission shared by the experimental chi finite-character transforms.

This module depends only on NumPy. A normalized weighted sum and a complete
finite-group Fourier transform have different weight contracts.
"""
from __future__ import annotations

from math import prod
import numpy as np


def integer_array(values, *, name: str, shape=None) -> np.ndarray:
    """Accept integral real values only, before any narrowing conversion."""
    raw = np.asarray(values)
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    # Object conversion preserves Python/uint64 integers, including overflow,
    # and booleans in mixed lists which a normal NumPy coercion would erase.
    objects = np.asarray(values, dtype=object)
    checked = []
    for value in objects.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer, float, np.floating)
        ):
            raise ValueError(f"{name} must contain finite integral real values")
        if isinstance(value, (float, np.floating)) and (
            not np.isfinite(value) or value != np.floor(value)
        ):
            raise ValueError(f"{name} must contain finite integral real values")
        integer = int(value)
        if not -(2**63) <= integer < 2**63:
            raise ValueError(f"{name} exceeds the signed int64 range")
        checked.append(integer)
    return np.asarray(checked, dtype=np.int64).reshape(objects.shape)


def real_array(values, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(f"{name} must contain finite real values")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values")
    return array


def translations_array(values) -> np.ndarray:
    result = integer_array(values, name='translations')
    if result.ndim != 2 or result.shape[1] != 3 or not result.shape[0]:
        raise ValueError('translations must have nonempty shape (n, 3)')
    return result


def inverse_inputs(matrices_k, kpoints_frac, weights=None):
    matrices = np.asarray(matrices_k, dtype=np.complex128)
    kfrac = real_array(kpoints_frac, name='kpoints_frac')
    if kfrac.ndim != 2 or kfrac.shape[1] != 3 or not kfrac.shape[0]:
        raise ValueError('kpoints_frac must have nonempty shape (n_k, 3)')
    if (matrices.ndim != 3 or matrices.shape[0] != len(kfrac)
            or matrices.shape[1] != matrices.shape[2] or matrices.shape[1] == 0):
        raise ValueError('inverse_bloch_transform requires nonempty square matrices (n_k, nbf, nbf)')
    if not np.all(np.isfinite(matrices)):
        raise ValueError('inverse_bloch_transform matrices must be finite')
    w = (np.full(len(kfrac), 1.0 / len(kfrac)) if weights is None
         else real_array(weights, name='weights'))
    if w.shape != (len(kfrac),):
        raise ValueError('inverse_bloch_transform weight count/shape does not match k-points')
    if np.any(w < 0) or np.any(w > 1) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('inverse_bloch_transform weights must be nonnegative and sum to one')
    return matrices, kfrac, w


def complete_characters(kpoints_frac, weights, extension_matrix):
    """Validate the full dual of Z^3 / S Z^3 (columns of S extend the cell).

    Haar weights are 1/abs(det S). Distinct dual characters are identified by
    their integer numerators modulo abs(det S), without an O(N^2) Gram matrix.
    This algebraic admission does not expand the SCF solver's support.
    """
    s = integer_array(extension_matrix, name='extension_matrix', shape=(3, 3))
    a, b, c, d, e, f, g, h, i = map(int, s.flat)
    order = abs(a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g))
    if order == 0 or order > np.iinfo(np.intp).max // 24:
        raise ValueError('extension_matrix must be nonsingular with addressable group order')
    k = real_array(kpoints_frac, name='kpoints_frac')
    w = real_array(weights, name='weights')
    if k.shape != (order, 3) or w.shape != (order,):
        raise ValueError('complete-character count must equal abs(det extension_matrix)')
    if np.any(w < 0) or not np.allclose(w, 1. / order, atol=1e-12, rtol=0):
        raise ValueError('complete characters require uniform finite-group weights')
    # Reducing modulo primitive reciprocal translations also avoids overflow
    # from large equivalent character representatives.
    reduced = np.remainder(k, 1.)
    labels = reduced * order
    if not np.allclose(labels, np.rint(labels), atol=1e-10, rtol=0):
        raise ValueError('k-points are not finite-group characters')
    numerators = np.rint(labels).astype(object)
    for row in numerators:
        if any(sum(int(row[j]) * int(s[j, axis]) for j in range(3)) % order for axis in range(3)):
            raise ValueError('k-points are not characters of extension_matrix')
    keys = {tuple(int(x) % order for x in row) for row in numerators}
    if len(keys) != order:
        raise ValueError('k-points must cover the complete distinct character group')
    return s


def checked_mesh_size(mesh):
    """Check address arithmetic before enumeration or NumPy products."""
    size = prod(int(x) for x in mesh)
    if any(int(x) <= 0 or int(x) >= 2**63 for x in mesh) or size > np.iinfo(np.intp).max // 24:
        raise ValueError('chi mesh exceeds addressable array bounds')
    return size


def phase_arguments(kfrac, translations):
    """Return k.R modulo one without integer narrowing or large products."""
    k = np.remainder(kfrac, 1.)
    result = np.zeros((len(translations), len(k)))
    for row, translation in enumerate(translations):
        for axis in range(3):
            n = int(translation[axis])
            term = k[:, axis].copy()
            sign = -1 if n < 0 else 1
            n = abs(n)
            accum = np.zeros(len(k))
            while n:
                if n & 1:
                    accum = np.remainder(accum + term, 1.)
                term = np.remainder(2. * term, 1.)
                n >>= 1
            result[row] = np.remainder(result[row] + sign * accum, 1.)
    return result
