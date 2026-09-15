"""Phase SYM5a: symmetry-averaged density and Fock matrices.

Provides group-averaging operators that remove symmetry-breaking
numerical noise from SCF density and Fock matrices:

    D_sym  =  (1/|G|) . S_R  P(R) . D . P(R)ᵀ
    F_sym  =  (1/|G|) . S_R  P(R) . F . P(R)ᵀ

where P(R) is the AO permutation matrix from :mod:`vibeqc.symmetry_ao`
and the sum runs over all symmorphic operators.  This is a pure
stability enhancement -- it doesn't change the physical fixed point
of the SCF, only the convergence path.

Public API
----------
  - :func:`build_ao_permutation_cache` -- pre-compute P(R) for all ops.
  - :func:`symmetrize_matrix` -- group-average a single (nbf, nbf) matrix.
  - :func:`symmetrize_density` -- group-average with optional diagnostics.
  - :func:`symmetrize_fock` -- group-average with optional diagnostics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ._vibeqc_core import BasisSet, PeriodicSystem, SymmetryOp
from .progress import resolve_progress
from .symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix
from .symmetry_integrals import symmorphic_operations
from .symmetry_lattice import lattice_to_cartesian_rotation

__all__ = [
    "build_ao_permutation_cache",
    "symmetrize_matrix",
    "symmetrize_density",
    "symmetrize_fock",
]


# ---------------------------------------------------------------------------
# Permutation cache
# ---------------------------------------------------------------------------


def build_ao_permutation_cache(
    system: PeriodicSystem,
    basis: BasisSet,
    operations: Sequence[SymmetryOp],
) -> List[np.ndarray]:
    """Pre-compute the AO permutation matrix P(R) for every operator.

    Parameters
    ----------
    system
        :class:`PeriodicSystem` with symmetry attached.
    basis
        AO basis set.
    operations
        Sequence of :class:`SymmetryOp` (from ``system.symmetry.operations``
        or the symmorphic subset).

    Returns
    -------
    List of ``(nbf, nbf)`` real orthogonal matrices, one per operator.
    """
    L = np.asarray(system.lattice, dtype=float)
    cache: List[np.ndarray] = []
    for op in operations:
        R_cart = lattice_to_cartesian_rotation(op.rotation, L)
        t_cart = L @ np.asarray(op.translation, dtype=float)
        ap = atom_permutation_under_op(system, R_cart, t_cart)
        P = build_ao_permutation_matrix(basis, R_cart, ap)
        cache.append(P)
    return cache


# ---------------------------------------------------------------------------
# Group averaging
# ---------------------------------------------------------------------------


def symmetrize_matrix(
    M: np.ndarray,
    P_cache: Sequence[np.ndarray],
) -> np.ndarray:
    """Group-average a single (nbf, nbf) matrix over all operators.

    Returns ``(1/|G|) . S_R P(R) . M . P(R)ᵀ``.
    """
    M = np.asarray(M, dtype=float)
    n_ops = len(P_cache)
    if n_ops == 0:
        return M.copy()
    result = np.zeros_like(M)
    for P in P_cache:
        result += P @ M @ P.T
    result /= n_ops
    return result


def symmetrize_density(
    D: np.ndarray,
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    operations: Optional[Sequence[SymmetryOp]] = None,
    P_cache: Optional[Sequence[np.ndarray]] = None,
    report_asymmetry: bool = False,
) -> np.ndarray:
    """Symmetrize a density matrix under the space group.

    Parameters
    ----------
    D
        (nbf, nbf) density matrix.
    system, basis
        Periodic system and AO basis.
    operations
        If provided, use these operators directly (must be symmorphic).
        If None, fall back to ``system.symmetry.operations`` (full set),
        filtered through :func:`symmorphic_operations`.
    P_cache
        Pre-computed permutation cache from :func:`build_ao_permutation_cache`.
        If None, computed on the fly.
    report_asymmetry
        If True, emit the ||D - D_sym|| Frobenius norm as live progress.

    Returns
    -------
    Symmetrized density ``D_sym``.
    """
    D = np.asarray(D, dtype=float)
    if operations is None:
        if system.symmetry is None:
            return D.copy()
        operations = symmorphic_operations(system.symmetry.operations)
    if P_cache is None:
        P_cache = build_ao_permutation_cache(system, basis, operations)

    D_sym = symmetrize_matrix(D, P_cache)

    if report_asymmetry and len(P_cache) > 0:
        asym = float(np.linalg.norm(D - D_sym))
        resolve_progress(True, verbose=2).write_raw(
            f"  [symmetrize_density] ||D - D_sym|| = {asym:.3e}"
        )

    return D_sym


def symmetrize_fock(
    F: np.ndarray,
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    operations: Optional[Sequence[SymmetryOp]] = None,
    P_cache: Optional[Sequence[np.ndarray]] = None,
    report_asymmetry: bool = False,
) -> np.ndarray:
    """Symmetrize a Fock matrix under the space group.

    Parameters
    ----------
    F
        (nbf, nbf) Fock matrix.
    system, basis
        Periodic system and AO basis.
    operations
        Symmetry operators. If None, derived from system.
    P_cache
        Pre-computed permutation cache. If None, computed on the fly.
    report_asymmetry
        If True, emit the ||F - F_sym|| Frobenius norm as live progress.

    Returns
    -------
    Symmetrized Fock matrix ``F_sym``.
    """
    F = np.asarray(F, dtype=float)
    if operations is None:
        if system.symmetry is None:
            return F.copy()
        operations = symmorphic_operations(system.symmetry.operations)
    if P_cache is None:
        P_cache = build_ao_permutation_cache(system, basis, operations)

    F_sym = symmetrize_matrix(F, P_cache)

    if report_asymmetry and len(P_cache) > 0:
        asym = float(np.linalg.norm(F - F_sym))
        resolve_progress(True, verbose=2).write_raw(
            f"  [symmetrize_fock] ||F - F_sym|| = {asym:.3e}"
        )

    return F_sym


def symmetrize_forces(
    forces: np.ndarray,
    system: PeriodicSystem,
    operations: Optional[Sequence[SymmetryOp]] = None,
) -> np.ndarray:
    """Symmetrize nuclear forces under the space group.

    For each symmetry operation R with atom permutation \u03c0:
        F_sym[a] = (1/|G|) * \u03a3_R  R^{-1} * F_raw[\u03c0(a)]

    Only the Cartesian rotation and atom permutation are needed
    (forces are vectors, not matrices).
    """
    from .symmetry_ao import atom_permutation_under_op
    from .symmetry_integrals import symmorphic_operations
    from .symmetry_lattice import lattice_to_cartesian_rotation

    forces = np.asarray(forces, dtype=float)
    if operations is None:
        if system.symmetry is None:
            return forces.copy()
        operations = symmorphic_operations(system.symmetry.operations)
    if len(operations) == 0:
        return forces.copy()

    L = np.asarray(system.lattice, dtype=float)
    n_atoms = len(system.unit_cell)
    result = np.zeros_like(forces)

    for op in operations:
        R_lat = np.asarray(op.rotation, dtype=int)
        R_cart = lattice_to_cartesian_rotation(R_lat, L)
        ap = atom_permutation_under_op(system, R_cart, np.zeros(3))
        R_inv = R_cart.T
        for a in range(n_atoms):
            b = int(ap.perm[a])
            result[a] += R_inv @ forces[b]

    result /= len(operations)
    return result
