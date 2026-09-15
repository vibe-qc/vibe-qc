"""Periodic projected-atomic-orbital (PAO) virtual space at the Γ-point -- Stage 3.

Builds the virtual-space foundation for periodic local correlation
([`handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md`], Stage 3): project the occupied
manifold out of the AO space to obtain the virtual-space projector
``Q = 1 - C_occ C_occᵀ S``, and form an S-orthonormal, Fock-diagonal virtual
basis over the home cell.

At the Γ-point this is exactly the molecular PAO construction applied to the
periodic Γ quantities, so it reuses :mod:`vibeqc.dlpno.pao` directly --
``build_projection_matrix`` and ``semicanonical_pao_basis``. The occupied
projector depends only on the occupied *subspace*, so canonical or
Wannier-localised occupied orbitals give the same ``Q``; the localisation
matters for per-pair domains (Stage 5), not for this global virtual space.

Per-pair PAO domains (``select_domain_atoms_mulliken`` / ``build_pao_coeffs`` in
``vibeqc.dlpno.pao``) are wired in at Stage 5, once the occupied Wannier pairs
are defined.

Scope: Γ-point. The full-domain semicanonical PAO basis here equals the
canonical virtual MOs (it diagonalises the Fock matrix over the whole virtual
space); the point of Stage 3 is the *reusable redundant-PAO + projector*
machinery that Stage 5 restricts to local domains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dlpno.pao import build_projection_matrix, semicanonical_pao_basis

__all__ = ["PeriodicPAOResult", "build_pao_periodic_gamma"]


@dataclass
class PeriodicPAOResult:
    """Periodic Γ-point PAO virtual space.

    Attributes
    ----------
    Q_vir : ndarray, shape (nbf, nbf)
        Virtual-space projector ``1 - C_occ C_occᵀ S``. Columns are the
        (redundant) projected atomic orbitals; this is what Stage 5 restricts
        to per-pair domains.
    V_semi : ndarray, shape (nbf, n_vir)
        S-orthonormal, Fock-diagonal virtual basis over the whole cell
        (full-domain semicanonical PAOs).
    eps_vir : ndarray, shape (n_vir,)
        Fock eigenvalues of ``V_semi`` (the virtual orbital energies).
    n_occ, n_vir : int
        Occupied and virtual counts (``n_occ + n_vir = nbf`` minus any
        linear-dependence drop).
    """

    Q_vir: np.ndarray
    V_semi: np.ndarray
    eps_vir: np.ndarray
    n_occ: int
    n_vir: int


def build_pao_periodic_gamma(
    result,
    basis,
    system,
    *,
    n_occ: int | None = None,
    lindep_thresh: float = 1e-8,
) -> PeriodicPAOResult:
    """Build the periodic Γ-point PAO virtual space from a converged SCF.

    Parameters
    ----------
    result
        Converged Γ-point periodic RHF result. Duck-typed: needs ``mo_coeffs``
        (nbf x norb), ``overlap`` (S at Γ) and ``fock`` (AO Fock at Γ).
    basis
        The ``BasisSet`` (kept in the signature for symmetry with the rest of
        the Stage pipeline / future per-domain work; not needed for the global
        projector).
    system
        The ``PeriodicSystem`` -- supplies the electron count for ``n_occ``.
    n_occ
        Occupied-orbital count. Defaults to
        ``system.unit_cell_molecule().n_electrons() // 2``.
    lindep_thresh
        Overlap-eigenvalue threshold for removing PAO redundancy.

    Returns
    -------
    PeriodicPAOResult
    """
    C_all = np.asarray(result.mo_coeffs)
    if np.iscomplexobj(C_all):
        C_all = np.real_if_close(C_all, tol=1000).real
    if n_occ is None:
        n_occ = int(system.unit_cell_molecule().n_electrons()) // 2
    if n_occ < 1:
        raise ValueError(f"n_occ must be >= 1, got {n_occ}")

    C_occ = np.array(C_all[:, :n_occ], dtype=float)
    S = np.asarray(result.overlap, dtype=float)
    F = np.asarray(result.fock, dtype=float)
    nbf = S.shape[0]

    # Q = 1 - C_occ C_occᵀ S  (reused molecular construction).
    Q_vir = build_projection_matrix(C_occ, S)

    # Full-domain (all AOs) S-orthonormal, Fock-diagonal virtual basis.
    all_ao = np.arange(nbf, dtype=int)
    V_semi, eps_vir = semicanonical_pao_basis(
        F, S, Q_vir, all_ao, lindep_thresh=lindep_thresh
    )

    return PeriodicPAOResult(
        Q_vir=Q_vir,
        V_semi=V_semi,
        eps_vir=eps_vir,
        n_occ=n_occ,
        n_vir=int(V_semi.shape[1]),
    )
