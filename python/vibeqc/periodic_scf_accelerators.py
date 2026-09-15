"""Shared SCF accelerator + dynamic-damping helpers for the Python
periodic backends.

Why this module exists
----------------------
``PeriodicRHFOptions`` / ``PeriodicSCFOptions`` / ``PeriodicKSOptions``
expose ``scf_accelerator`` (DIIS / KDIIS / EDIIS / EDIIS_DIIS / ADIIS)
and ``dynamic_damping`` to match the molecular options surface. The C++
direct-truncated periodic kernels honour them; the Python periodic
backends (Γ-Ewald, multi-k Ewald, GDF, BIPOLE) historically only
implemented Pulay DIIS + static damping, with a
``_reject_unsupported_python_accelerator`` helper that raised
``NotImplementedError`` on the unsupported combinations.

This module is the canonical wiring of the full accelerator family for
the Γ-point Python backends. It binds against the C++
``vibeqc::DIIS / KDIIS / EDIIS / ADIIS`` classes (pybind11 bindings in
``cpp/src/bindings.cpp``) so the numerics live in one place -- the same
classes the molecular and C++ periodic drivers use.

The multi-k variants (per-k Fock list, KDIIS per-k MO-basis design) are
deferred to a follow-up milestone (M2) and ship their own helper to
avoid coupling Γ and multi-k call sites.

The dynamic-damping helper mirrors
``cpp/include/vibeqc/dynamic_damping.hpp`` (Zerner-Hehenberger 1979)
exactly so the static and dynamic flavours produce identical a
trajectories across the C++ and Python backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    ADIIS,
    DIIS,
    DIISDepthPolicy,
    EDIIS,
    KDIIS,
    SCFAccelerator,
)


# Commutator-DIIS accelerators whose extrapolation is plain Pulay DIIS but
# whose history depth is chosen adaptively (Chupin, Dupuy, Legendre & Séré,
# ESAIM: M2AN 55, 2785 (2021)). They share every SCF code path with
# SCFAccelerator.DIIS; only the retained (F, error) window differs.
_DIIS_LIKE = (
    SCFAccelerator.DIIS,
    SCFAccelerator.R_CDIIS,
    SCFAccelerator.AD_CDIIS,
)


def _diis_with_policy(opts, max_sub: int) -> DIIS:
    """Build a C++ DIIS whose depth policy comes from ``opts``.

    Mirrors the C++ ``make_diis`` helper (cpp/include/vibeqc/ediis.hpp):
    R_CDIIS selects the restart policy with τ = ``diis_restart_tau``;
    AD_CDIIS selects the adaptive-depth policy with δ =
    ``diis_adaptive_delta``; every other accelerator yields a
    fixed-depth history.
    """
    mode = getattr(opts, "scf_accelerator", SCFAccelerator.DIIS)
    if mode == SCFAccelerator.R_CDIIS:
        return DIIS(max_sub, DIISDepthPolicy.RESTART,
                    float(getattr(opts, "diis_restart_tau", 1.0e-4)))
    if mode == SCFAccelerator.AD_CDIIS:
        return DIIS(max_sub, DIISDepthPolicy.ADAPTIVE,
                    float(getattr(opts, "diis_adaptive_delta", 1.0e-4)))
    return DIIS(max_sub)


def _ediis_diis_switch_metric(
    err_norm: float, dim: int, n_blocks: int = 1
) -> float:
    """Intensive RMS switch metric for the EDIIS_DIIS / ADIIS_DIIS hybrids.

    Python mirror of the C++ ``ediis_diis_switch_metric``
    (cpp/include/vibeqc/ediis.hpp): the public
    ``ediis_diis_switch_threshold`` is intensive, while a raw Frobenius
    commutator norm scales with the AO matrix dimension (and the square
    root of the number of spin blocks), so the hybrid must compare an
    RMS matrix-element norm against the threshold. The molecular C++
    drivers switched to this metric in ddd8d325 (S22 water dimer stuck
    in EDIIS for 100 iterations); the periodic dispatch sites kept the
    raw norm, which pinned c-diamond KRHF-GDF kmesh (2,2,2)/STO-3G in
    the EDIIS regime (raw 0.19 > 0.1, intensive 0.019 < 0.1) for 8
    frozen cycles once EDIIS_DIIS became the periodic default
    (cb712e21).

    For the multi-k callers ``err_norm`` is the k-weighted RMS
    ``sqrt(S_k w_k ||e(k)||_F^2)`` with ``S_k w_k = 1``, which is
    already intensive in the k index and reduces to the molecular
    Frobenius norm at Γ, so the same ``dim`` normalisation applies.
    """
    denom = float(np.sqrt(float(max(1, n_blocks)))) * float(dim)
    return float(err_norm) / max(denom, 1.0)


def _k_weighted_error_norm(
    error_k_list: Sequence[np.ndarray], weights: Sequence[float]
) -> float:
    """k-weighted RMS commutator norm ``sqrt(S_k w_k ||e(k)||_F^2)``.

    ``S_k w_k = 1``, so this is intensive in the k index and reduces to
    the plain Frobenius norm at Γ -- the multi-k input to
    :func:`_ediis_diis_switch_metric`.
    """
    err_norm_sq = 0.0
    for w_k, e_k in zip(weights, error_k_list):
        err_norm_sq += float(w_k) * float(
            np.real(np.vdot(e_k.ravel(), e_k.ravel()))
        )
    return float(np.sqrt(max(err_norm_sq, 0.0)))


# Zero-commutator extrapolation floor. On symmetry-locked fixtures (e.g.
# H2 in a cubic box / sto-3g: the sigma_g/sigma_u eigenvectors are fixed
# by symmetry) the FDS-SDF commutator vanishes IDENTICALLY from iteration
# 1 for ANY occupations, so the Pulay B matrix assembled from those error
# vectors is exactly singular and the solved extrapolation weights are
# unconstrained noise. Under finite-temperature smearing the resulting
# garbage-mixed Fock's eigenvalues freeze, the occupations computed from
# them stop moving, and the SCF "converges" (dE = 0 exactly) at a state
# whose occupations are NOT the Fermi filling of its own Fock (measured
# 2026-07-30 on H2 2.4 bohr / 12-bohr box / kBT = 0.05 Ha: stored
# occ_alpha [0.998955, 0.001045] vs the converged Fock's own filling
# [0.999126, 0.000874]; free energy off ~1e-4 Ha). An error at this floor
# carries no extrapolation information for ANY accelerator of the family,
# so every extrapolate_* method returns the input Fock unchanged and
# records nothing. A healthy SCF only reaches this floor at (or beyond)
# its convergence tolerance, where skipping extrapolation is a no-op.
_EXTRAPOLATION_ERROR_FLOOR = 1.0e-12


def _below_extrapolation_floor(err_norm: float, fock_norm: float) -> bool:
    """True when the commutator error is numerical noise on the Fock scale."""
    return float(err_norm) <= _EXTRAPOLATION_ERROR_FLOOR * max(
        1.0, float(fock_norm)
    )


__all__ = [
    "PeriodicSCFAccelerator",
    "MultiKPeriodicSCFAccelerator",
    "MultiKPeriodicUHFAccelerator",
    "DynamicDamping",
    "blocks_from_per_k",
    "per_k_from_blocks",
    "per_k_to_stacked_real_blocks",
    "stacked_real_blocks_to_per_k",
]


# ---------------------------------------------------------------------------
# per-k <-> per-cell Bloch bridge
# ---------------------------------------------------------------------------
#
# Multi-k periodic SCF in vibe-qc represents the Fock matrix in two equivalent
# bases:
#
#   * Per-k F(k) -- complex Hermitian, one (n_bf, n_bf) block per k-point in
#     the irreducible BZ mesh. This is what the SCF loop's diagonaliser, the
#     Bloch-summed integral builders, and the per-k DIIS implementation
#     (:class:`vibeqc.periodic_rhf_multi_k_ewald._MultiKPulayDIIS`) consume.
#
#   * Per-cell F(g) -- real (n_bf, n_bf) block per real-space lattice vector
#     in the cell list of a :class:`vibeqc.LatticeMatrixSet`. The C++
#     multi-k SCF dispatch (``cpp/src/periodic_scf.cpp:412-440``) uses this
#     representation when feeding the block-vector EDIIS / ADIIS kernel,
#     and so do the parts of the Python multi-k driver that build Fock via
#     ``build_periodic_fock_ewald3d_k`` (which Bloch-sums a per-cell density
#     to k-space inside the integral build).
#
# The two are related by Bloch summation:
#
#   F(k) = S_g  exp(+i k . R_g)            . F(g)
#   F(g) = S_k  w_k . Re[ exp(-i k . R_g)  . F(k) ]
#
# where the Re[] in the inverse direction encodes time-reversal averaging:
# F(-k) = F(k)* for a time-reversal-symmetric Hamiltonian, so the (k, -k)
# pair contribution to F(g) is real. Monkhorst-Pack meshes are
# time-reversal symmetric by construction (and BlochKMesh is built that
# way). The forward direction mirrors
# ``vibeqc.periodic_fock_multi_k._bloch_sum_blocks``; the inverse mirrors
# ``vibeqc::real_space_density_from_kpoints`` in
# ``cpp/src/periodic_scf.cpp`` but generalised from C.diag(occ).C+ to an
# arbitrary Hermitian per-k matrix list.


def blocks_from_per_k(
    M_per_k: Sequence[np.ndarray],
    cells: Sequence,
    kpoints: Sequence[np.ndarray],
    weights: Sequence[float],
) -> List[np.ndarray]:
    """Inverse Bloch summation per-k -> per-cell.

    Returns the per-cell real-block list
    ``M(g) = S_k w_k . Re[ exp(-i k . R_g) . M(k) ]`` for an input list
    of complex Hermitian per-k matrices. Mirrors
    ``vibeqc::real_space_density_from_kpoints`` but takes the per-k
    matrices directly instead of reconstructing them from MO
    coefficients, so it generalises to Fock matrices and any other
    per-k Hermitian operator.

    Parameters
    ----------
    M_per_k
        Sequence of ``(n_bf, n_bf)`` complex Hermitian arrays, one per
        k-point. Hermiticity is taken on trust; the imaginary part of
        the final per-cell block is dropped by ``Re[]``.
    cells
        Sequence of :class:`vibeqc.LatticeCell` (each carries
        ``r_cart``, the Cartesian real-space lattice vector for the
        cell).
    kpoints, weights
        Cartesian k-points and integration weights from a
        :class:`vibeqc.BlochKMesh`. ``weights`` must sum to 1 for the
        round-trip ``per_k_from_blocks(blocks_from_per_k(...))`` to be
        the identity on Hermitian-symmetric meshes.

    Returns
    -------
    list[np.ndarray]
        ``[M(g_0), M(g_1), ...]`` -- one real ``(n_bf, n_bf)`` array per
        cell, in the same order as ``cells``.
    """
    if len(kpoints) != len(M_per_k) or len(weights) != len(M_per_k):
        raise ValueError(
            "blocks_from_per_k: M_per_k, kpoints, weights must all have "
            f"the same length; got {len(M_per_k)}, {len(kpoints)}, "
            f"{len(weights)}"
        )
    if not M_per_k:
        return []
    n_bf = M_per_k[0].shape[0]
    blocks: List[np.ndarray] = []
    for cell in cells:
        R_g = np.asarray(cell.r_cart, dtype=float).reshape(3)
        acc = np.zeros((n_bf, n_bf), dtype=complex)
        for w_k, k_arr, M_k in zip(weights, kpoints, M_per_k):
            k = np.asarray(k_arr, dtype=float).reshape(3)
            phase = np.exp(-1j * float(np.dot(k, R_g)))
            acc = acc + float(w_k) * phase * np.asarray(M_k)
        blocks.append(np.ascontiguousarray(acc.real))
    return blocks


def per_k_from_blocks(
    blocks: Sequence[np.ndarray],
    cells: Sequence,
    kpoints: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Forward Bloch summation per-cell -> per-k.

    Returns ``M(k) = S_g exp(+i k . R_g) . M(g)`` for an input list of
    real per-cell blocks. Mirrors
    ``vibeqc.periodic_fock_multi_k._bloch_sum_blocks`` lifted to a
    per-k list and used as the inverse of :func:`blocks_from_per_k`.

    Hermiticity of the returned ``M(k)`` is guaranteed when the input
    cell list is closed under negation (``{R_g} = {-R_g}``), which is
    the standard real-space-cell convention for vibe-qc's periodic
    lattice-matrix sets.
    """
    if not blocks:
        return [np.zeros((0, 0), dtype=complex) for _ in kpoints]
    n_bf = blocks[0].shape[0]
    out: List[np.ndarray] = []
    for k_arr in kpoints:
        k = np.asarray(k_arr, dtype=float).reshape(3)
        M_k = np.zeros((n_bf, n_bf), dtype=complex)
        for g_idx, block in enumerate(blocks):
            R_g = np.asarray(cells[g_idx].r_cart, dtype=float).reshape(3)
            phase = np.exp(1j * float(np.dot(k, R_g)))
            M_k = M_k + phase * np.asarray(block)
        out.append(M_k)
    return out


# ---------------------------------------------------------------------------
# per-k <-> stacked-real-block bridge (for the C++ block-vector EDIIS / ADIIS)
# ---------------------------------------------------------------------------
#
# The Bloch bridge above (per-k <-> per-cell) round-trips to the identity
# on Bloch-Floquet matched cell lists, but on a generic cutoff-determined
# cell list the per-cell bilinear form
# ``S_g (F(g) odot D(g)).sum() = S_g Tr[F(g) D(g)]`` does *not* reduce to a
# clean rescaling of the per-k bilinear form
# ``S_k w_k Re Tr[F(k) D(k)]``. Even on matched cells the round-trip
# weight pattern is ``S_k w_k^2 ...`` rather than ``S_k w_k ...``, which
# is a *structural* (not uniform-scale) reweighting of the EDIIS / ADIIS
# quadratic cross-term. Composed against the unchanged linear energy
# term ``S_i c_i E_i`` (in Hartree), the QP minimum lands at the wrong
# coefficient set and the SCF fails to converge on an H₂ chain with
# kmesh=[2,1,1] (commit ``5c333372`` in M2d documents this end-to-end).
#
# The helpers below sidestep the Bloch bridge entirely. For each per-k
# Hermitian matrix ``M(k) = M_R(k) + i M_I(k)`` (M_R symmetric, M_I
# antisymmetric) we emit *two* real blocks scaled by ``a(k) = √w_k``:
#
#   block(2k)   = √w_k . M_R(k)        (n_bf x n_bf, symmetric)
#   block(2k+1) = √w_k . M_I(k)        (n_bf x n_bf, antisymmetric)
#
# The C++ kernel's per-block sum-of-Frobenius bilinear form on a paired
# stack {F̃, D̃} then equals
#
#   S_b (F̃[b] odot D̃[b]).sum()
#     = S_k a(k)^2 ( (F_RodotD_R).sum() + (F_IodotD_I).sum() )
#     = S_k w_k ( Tr[F_R D_R] - Tr[F_I D_I] )
#     = S_k w_k Re Tr[F(k) D(k)]
#
# which is exactly the per-k energy bilinear form the Python multi-k
# Ewald driver uses for ``E_elec``. So the C++ EDIIS / ADIIS QP linear
# term (E in Hartree) and quadratic cross-term are now in matching
# units, the QP minimum corresponds to a true energy minimiser on the
# simplex, and the linear-combo output unstacks per-k correctly:
#
#   F̃_ex[2k] = S_i c_i √w_k F_R_i(k) = √w_k F_R_ex(k)  (and similarly Im)
#
# so dividing by ``√w_k`` recovers ``F_ex(k) = S_i c_i F_i(k)`` per k.
# Zero-weight k-points contribute zero blocks both ways; we just return
# a zero per-k matrix at those indices (matches the SCF semantics --
# a zero-weight k carries no electrons through the energy or density).


def per_k_to_stacked_real_blocks(
    M_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
) -> List[np.ndarray]:
    """Per-k Hermitian list -> 2.N_k real blocks weighted by ``a(k) = √w_k``.

    See this module's "per-k <-> stacked-real-block bridge" section for
    the derivation. Used by :class:`MultiKPeriodicSCFAccelerator` /
    :class:`MultiKPeriodicUHFAccelerator` to feed the C++
    ``EDIIS::extrapolate_blocks`` / ``ADIIS::extrapolate_blocks``
    kernels with a representation in which the kernel's
    sum-of-Frobenius bilinear form matches the periodic per-k energy
    bilinear form ``S_k w_k Re Tr[F(k) D(k)]``.
    """
    if len(M_per_k) != len(weights):
        raise ValueError(
            "per_k_to_stacked_real_blocks: M_per_k and weights must "
            f"have the same length; got {len(M_per_k)} vs {len(weights)}"
        )
    blocks: List[np.ndarray] = []
    for w_k, M_k in zip(weights, M_per_k):
        alpha = float(np.sqrt(max(float(w_k), 0.0)))
        Mk = np.asarray(M_k)
        blocks.append(np.ascontiguousarray(alpha * Mk.real))
        blocks.append(np.ascontiguousarray(alpha * Mk.imag))
    return blocks


def stacked_real_blocks_to_per_k(
    blocks: Sequence[np.ndarray],
    weights: Sequence[float],
) -> List[np.ndarray]:
    """Inverse of :func:`per_k_to_stacked_real_blocks`.

    Reads pairs ``(a F_R, a F_I)`` per k, divides by ``a = √w_k`` and
    recombines ``F(k) = F_R + i F_I``. Zero-weight k-points return a
    zero per-k block (the linear-combo over a zero-input history is
    itself zero, so the only consistent recovery is zero).
    """
    if 2 * len(weights) != len(blocks):
        raise ValueError(
            "stacked_real_blocks_to_per_k: expected 2.N_k blocks, "
            f"got {len(blocks)} for {len(weights)} k-points"
        )
    out: List[np.ndarray] = []
    for k_idx, w_k in enumerate(weights):
        w = float(w_k)
        F_R = np.asarray(blocks[2 * k_idx])
        F_I = np.asarray(blocks[2 * k_idx + 1])
        if w <= 0.0:
            out.append(np.zeros(F_R.shape, dtype=complex))
        else:
            inv_alpha = 1.0 / float(np.sqrt(w))
            out.append(np.ascontiguousarray(inv_alpha * (F_R + 1j * F_I)))
    return out


@dataclass
class _AcceleratorState:
    # Pulay DIIS instance. Closed shell feeds it the plain (F, e) pair;
    # UHF feeds it the vertically stacked (F_a; F_b) / (e_a; e_b) pair,
    # so the open-shell history is spin-coupled: one B matrix built from
    # the stacked error, one coefficient set applied to both Focks. F_a
    # and F_b are coupled through the shared Coulomb term J(D_a + D_b),
    # so two independent per-spin histories let each extrapolation
    # optimise its own residual against a moving other-spin field --
    # that stalled the molecular OH/def2-TZVP UHF tail (2026-07 M02
    # regression) and the same coupling argument holds per k-point in
    # the periodic drivers.
    diis: Optional[DIIS]
    kdiis: Optional[KDIIS]
    ediis: Optional[EDIIS]
    adiis: Optional[ADIIS]
    mode: SCFAccelerator
    switch_threshold: float


class PeriodicSCFAccelerator:
    """Γ-point SCF Fock-extrapolation accelerator.

    Dispatches on ``opts.scf_accelerator`` and exposes a uniform
    ``extrapolate_rhf`` / ``extrapolate_uhf`` API to the periodic SCF
    drivers. Mirrors the C++ ``run_rhf_periodic`` accelerator dispatch
    (``cpp/src/periodic_scf.cpp``) at Γ where the Γ-folded Fock equals
    the home-cell Fock.

    Parameters
    ----------
    opts
        A periodic SCF options struct (``PeriodicRHFOptions``,
        ``PeriodicSCFOptions``, ``PeriodicKSOptions``). The accessed
        fields are ``scf_accelerator``, ``diis_subspace_size``, and
        ``ediis_diis_switch_threshold`` (the latter only consulted in
        the EDIIS_DIIS hybrid).

    Notes
    -----
    For the EDIIS_DIIS hybrid both an EDIIS and a DIIS history are kept
    warm in parallel. The extrapolated Fock returned by ``extrapolate_*``
    is EDIIS while the commutator-error norm is above
    ``opts.ediis_diis_switch_threshold`` and DIIS once it drops below --
    matching ``cpp/src/periodic_scf.cpp`` lines 419-440.

    KDIIS uses the orbital-rotation gradient in the canonical MO basis
    (Kollmar 1997). Pass the *previous* iteration's ``mo_coeffs`` and
    ``mo_energies`` -- the C e pair that produced the current density
    D from which F was built. The first call has no history and
    ``extrapolate_rhf`` returns ``fock`` unchanged regardless of the
    selected accelerator.
    """

    def __init__(self, opts):
        mode = getattr(opts, "scf_accelerator", SCFAccelerator.DIIS)
        max_sub = int(opts.diis_subspace_size)
        switch = float(getattr(opts, "ediis_diis_switch_threshold", 1.0e-2))

        diis = None
        kdiis = None
        ediis = None
        adiis = None
        if mode in _DIIS_LIKE:
            # DIIS / R_CDIIS / AD_CDIIS: same extrapolation, depth policy
            # baked into the C++ object (see _diis_with_policy).
            diis = _diis_with_policy(opts, max_sub)
        elif mode == SCFAccelerator.KDIIS:
            kdiis = KDIIS(max_sub)
        elif mode == SCFAccelerator.EDIIS:
            ediis = EDIIS(max_sub)
        elif mode == SCFAccelerator.ADIIS:
            adiis = ADIIS(max_sub)
        elif mode == SCFAccelerator.EDIIS_DIIS:
            ediis = EDIIS(max_sub)
            diis = DIIS(max_sub)
        elif mode == SCFAccelerator.ADIIS_DIIS:
            adiis = ADIIS(max_sub)
            diis = DIIS(max_sub)
        else:
            raise ValueError(
                f"PeriodicSCFAccelerator: unknown scf_accelerator "
                f"{mode!r}"
            )

        self._state = _AcceleratorState(
            diis=diis,
            kdiis=kdiis,
            ediis=ediis,
            adiis=adiis,
            mode=mode,
            switch_threshold=switch,
        )

    @property
    def subspace_size(self) -> int:
        """Current rolling-history depth of the active accelerator.

        For the EDIIS_DIIS hybrid this returns the DIIS history depth
        (both branches are kept in lockstep so either is informative).
        """
        s = self._state
        if s.mode in _DIIS_LIKE:
            return int(s.diis.subspace_size)
        if s.mode == SCFAccelerator.KDIIS:
            return int(s.kdiis.subspace_size)
        if s.mode == SCFAccelerator.EDIIS:
            return int(s.ediis.subspace_size())
        if s.mode == SCFAccelerator.ADIIS:
            return int(s.adiis.subspace_size())
        # EDIIS_DIIS
        return int(s.diis.subspace_size)

    def extrapolate_rhf(
        self,
        fock: np.ndarray,
        *,
        error: np.ndarray,
        density: np.ndarray,
        energy: float,
        mo_coeffs: np.ndarray,
        mo_energies: np.ndarray,
        n_occ: int,
    ) -> np.ndarray:
        """Closed-shell extrapolation.

        All inputs are required even when the active accelerator only
        consumes a subset -- the unused ones are ignored. This keeps the
        call site uniform across drivers and lets a future change of
        ``opts.scf_accelerator`` not need a corresponding wiring edit.

        Parameters
        ----------
        fock
            ``(n_bf, n_bf)`` Γ-folded Fock matrix to extrapolate.
        error
            ``(n_bf, n_bf)`` commutator error ``F.D.S - S.D.F``
            (consumed by DIIS and the EDIIS_DIIS switch).
        density
            ``(n_bf, n_bf)`` density matrix at the iterate that
            produced ``fock`` (consumed by EDIIS and ADIIS).
        energy
            Total electronic + nuclear energy at the iterate (consumed
            by EDIIS).
        mo_coeffs
            ``(n_bf, n_mo)`` canonical MO coefficients from the
            previous diagonalisation (consumed by KDIIS).
        mo_energies
            ``(n_mo,)`` canonical MO energies (consumed by KDIIS).
        n_occ
            Number of doubly-occupied orbitals (consumed by KDIIS).

        Returns
        -------
        np.ndarray
            ``(n_bf, n_bf)`` extrapolated Fock matrix. Returned
            unchanged before the rolling history has at least two
            entries (every accelerator obeys this convention), and
            whenever the commutator error is at the numerical floor
            (see ``_below_extrapolation_floor``), in which case nothing
            is recorded either.
        """
        if _below_extrapolation_floor(
            float(np.linalg.norm(error)), float(np.linalg.norm(fock))
        ):
            return np.asarray(fock)
        s = self._state
        if s.mode in _DIIS_LIKE:
            return np.asarray(s.diis.extrapolate(fock, error))
        if s.mode == SCFAccelerator.KDIIS:
            return np.asarray(
                s.kdiis.extrapolate(fock, mo_coeffs, mo_energies, int(n_occ))
            )
        if s.mode == SCFAccelerator.EDIIS:
            return np.asarray(s.ediis.extrapolate(fock, density, float(energy)))
        if s.mode == SCFAccelerator.ADIIS:
            return np.asarray(s.adiis.extrapolate(fock, density))
        if s.mode == SCFAccelerator.ADIIS_DIIS:
            # keep both warm; switch on the intensive commutator metric.
            F_d = np.asarray(s.diis.extrapolate(fock, error))
            F_a = np.asarray(s.adiis.extrapolate(fock, density))
            metric = _ediis_diis_switch_metric(
                float(np.linalg.norm(error)), int(np.asarray(fock).shape[0])
            )
            if metric > s.switch_threshold:
                return F_a
            # DIIS branch: F_a never reaches the SCF, so retract it from
            # the anti-replay guard's record of consumed returns (see
            # EDIIS::finish_extrapolation in cpp/src/ediis.cpp).
            s.adiis.discard_last_extrapolation()
            return F_d
        # EDIIS_DIIS -- keep both warm; switch on the intensive
        # commutator metric (same convention as the molecular C++
        # drivers, rhf.cpp / rks.cpp).
        F_d = np.asarray(s.diis.extrapolate(fock, error))
        F_e = np.asarray(s.ediis.extrapolate(fock, density, float(energy)))
        metric = _ediis_diis_switch_metric(
            float(np.linalg.norm(error)), int(np.asarray(fock).shape[0])
        )
        if metric > s.switch_threshold:
            return F_e
        s.ediis.discard_last_extrapolation()
        return F_d

    def extrapolate_uhf(
        self,
        fock_alpha: np.ndarray,
        fock_beta: np.ndarray,
        *,
        error_alpha: np.ndarray,
        error_beta: np.ndarray,
        density_alpha: np.ndarray,
        density_beta: np.ndarray,
        energy: float,
        mo_coeffs_alpha: np.ndarray,
        mo_coeffs_beta: np.ndarray,
        mo_energies_alpha: np.ndarray,
        mo_energies_beta: np.ndarray,
        n_alpha: int,
        n_beta: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Open-shell extrapolation.

        Same calling convention as :meth:`extrapolate_rhf` with per-spin
        Fock + density + MO + commutator-error inputs. Every
        accelerator runs a single spin-coupled history whose coefficient
        set applies to both Focks: DIIS stacks ``(F_a; F_b)`` /
        ``(e_a; e_b)`` vertically into one Pulay history (one B matrix
        from the stacked error -- see the ``_AcceleratorState`` note);
        EDIIS / ADIIS / KDIIS use their native spin-coupled overloads.
        The EDIIS_DIIS switch uses ``‖error_a‖ + ‖error_b‖`` against
        ``opts.ediis_diis_switch_threshold``.
        """
        if _below_extrapolation_floor(
            max(
                float(np.linalg.norm(error_alpha)),
                float(np.linalg.norm(error_beta)),
            ),
            max(
                float(np.linalg.norm(fock_alpha)),
                float(np.linalg.norm(fock_beta)),
            ),
        ):
            return np.asarray(fock_alpha), np.asarray(fock_beta)
        s = self._state
        n_bf = int(np.asarray(fock_alpha).shape[0])
        if s.mode in _DIIS_LIKE:
            F_ex = np.asarray(
                s.diis.extrapolate(
                    np.vstack([fock_alpha, fock_beta]),
                    np.vstack([error_alpha, error_beta]),
                )
            )
            return (
                np.ascontiguousarray(F_ex[:n_bf]),
                np.ascontiguousarray(F_ex[n_bf:]),
            )
        if s.mode == SCFAccelerator.KDIIS:
            pair = s.kdiis.extrapolate_uhf(
                fock_alpha, fock_beta,
                mo_coeffs_alpha, mo_coeffs_beta,
                mo_energies_alpha, mo_energies_beta,
                int(n_alpha), int(n_beta),
            )
            return np.asarray(pair[0]), np.asarray(pair[1])
        if s.mode == SCFAccelerator.EDIIS:
            pair = s.ediis.extrapolate_uhf(
                fock_alpha, fock_beta,
                density_alpha, density_beta,
                float(energy),
            )
            return np.asarray(pair[0]), np.asarray(pair[1])
        if s.mode == SCFAccelerator.ADIIS:
            pair = s.adiis.extrapolate_uhf(
                fock_alpha, fock_beta,
                density_alpha, density_beta,
            )
            return np.asarray(pair[0]), np.asarray(pair[1])
        if s.mode == SCFAccelerator.ADIIS_DIIS:
            # ADIIS analogue of the EDIIS_DIIS hybrid below.
            F_d = np.asarray(
                s.diis.extrapolate(
                    np.vstack([fock_alpha, fock_beta]),
                    np.vstack([error_alpha, error_beta]),
                )
            )
            Fa_d = np.ascontiguousarray(F_d[:n_bf])
            Fb_d = np.ascontiguousarray(F_d[n_bf:])
            F_a_a, F_a_b = s.adiis.extrapolate_uhf(
                fock_alpha, fock_beta,
                density_alpha, density_beta,
            )
            # Max-over-spins norm + n_blocks=2, mirroring uhf.cpp.
            metric = _ediis_diis_switch_metric(
                max(
                    float(np.linalg.norm(error_alpha)),
                    float(np.linalg.norm(error_beta)),
                ),
                int(n_bf),
                n_blocks=2,
            )
            if metric > s.switch_threshold:
                return np.asarray(F_a_a), np.asarray(F_a_b)
            # DIIS branch: the ADIIS pair never reaches the SCF, so
            # retract it from the anti-replay guard's record.
            s.adiis.discard_last_extrapolation()
            return Fa_d, Fb_d
        # EDIIS_DIIS hybrid -- the DIIS branch is spin-coupled via the
        # same vertical stacking as the plain-DIIS branch above.
        F_d = np.asarray(
            s.diis.extrapolate(
                np.vstack([fock_alpha, fock_beta]),
                np.vstack([error_alpha, error_beta]),
            )
        )
        Fa_d = np.ascontiguousarray(F_d[:n_bf])
        Fb_d = np.ascontiguousarray(F_d[n_bf:])
        F_e_a, F_e_b = s.ediis.extrapolate_uhf(
            fock_alpha, fock_beta,
            density_alpha, density_beta,
            float(energy),
        )
        F_e_a = np.asarray(F_e_a)
        F_e_b = np.asarray(F_e_b)
        # Max-over-spins norm + n_blocks=2, mirroring uhf.cpp.
        metric = _ediis_diis_switch_metric(
            max(
                float(np.linalg.norm(error_alpha)),
                float(np.linalg.norm(error_beta)),
            ),
            int(n_bf),
            n_blocks=2,
        )
        if metric > s.switch_threshold:
            return F_e_a, F_e_b
        # DIIS branch: the EDIIS pair never reaches the SCF, so retract
        # it from the anti-replay guard's record.
        s.ediis.discard_last_extrapolation()
        return Fa_d, Fb_d


class DynamicDamping:
    """Adaptive density-mixing a (Zerner-Hehenberger 1979).

    Python port of ``cpp/include/vibeqc/dynamic_damping.hpp``. Caller
    constructs once per SCF run with the initial a and the
    ``[a_min, a_max]`` window, then calls ``update(E_new)`` once per
    iteration to advance a. The current a is read from the ``alpha``
    attribute.

    The heuristic:
      * First call: no update; alpha unchanged.
      * dE = E_new - E_prev. If dE > 0 (SCF diverging / oscillating),
        bump a toward ``alpha_max`` by ``step``. If dE is a substantial
        decrease (more negative than ``-decrease_threshold``), ease a
        toward ``alpha_min`` by ``1/2 . step``. Otherwise leave a
        unchanged -- small fluctuations near convergence shouldn't pump
        a up and down.
      * a is clamped to ``[alpha_min, alpha_max]`` on every update.

    Parameters
    ----------
    initial_alpha
        Starting damping factor. Use ``opts.damping`` for backwards
        compatibility with static-damping call sites.
    alpha_min, alpha_max
        Window inside which a is allowed to move. From
        ``opts.dynamic_damping_min`` / ``opts.dynamic_damping_max``;
        defaults match the C++ header (0.0 / 0.95).
    step
        Per-iteration a update magnitude on a divergent step
        (1/2 step on a substantial decrease).
    decrease_threshold
        Minimum |dE| (per Hartree) to count as a "substantial decrease"
        and back off the damping. Below this the SCF is considered
        stationary to noise; a is left alone.

    Notes
    -----
    Dynamic damping composes freely with the
    :class:`PeriodicSCFAccelerator` family -- it modulates the density
    *before* the next Fock build, while the accelerator operates on the
    Fock *after* the build. The C++ header has the same composition
    rule (`dynamic_damping.hpp` Sec. "Composes freely with FMIXING and the
    SCF-accelerator family").
    """

    def __init__(
        self,
        initial_alpha: float,
        *,
        alpha_min: float = 0.0,
        alpha_max: float = 0.95,
        step: float = 0.1,
        decrease_threshold: float = 1.0e-4,
    ) -> None:
        if not (0.0 <= alpha_min <= alpha_max <= 1.0):
            raise ValueError(
                "DynamicDamping: alpha_min / alpha_max must satisfy "
                f"0 <= alpha_min <= alpha_max <= 1; got "
                f"({alpha_min}, {alpha_max})"
            )
        self.alpha = float(np.clip(initial_alpha, alpha_min, alpha_max))
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._step = float(step)
        self._decrease_threshold = float(decrease_threshold)
        self._e_prev: Optional[float] = None

    def update(self, e_new: float) -> float:
        """Advance a one iteration and return the updated value.

        First call: stores ``e_new`` and returns the unchanged a. From
        the second call onward, applies the Zerner-Hehenberger heuristic
        on ``dE = e_new - e_prev``.
        """
        if self._e_prev is None:
            self._e_prev = float(e_new)
            return self.alpha
        dE = float(e_new) - self._e_prev
        if dE > 0.0:
            self.alpha = min(self._alpha_max, self.alpha + self._step)
        elif dE < -self._decrease_threshold:
            self.alpha = max(self._alpha_min, self.alpha - 0.5 * self._step)
        # Else leave alpha unchanged -- small fluctuations don't move it.
        self.alpha = float(np.clip(self.alpha, self._alpha_min, self._alpha_max))
        self._e_prev = float(e_new)
        return self.alpha


# ===========================================================================
# Multi-k SCF accelerators
# ===========================================================================
#
# The Γ-point :class:`PeriodicSCFAccelerator` above wires DIIS / KDIIS /
# EDIIS / EDIIS_DIIS / ADIIS + dynamic damping for SCF kernels whose
# Fock matrices are real-valued and live on a single k-point. The
# multi-k counterparts in this section handle a per-k *list* of complex
# Hermitian Fock matrices and dispatch the same accelerator family:
#
#   * DIIS / KDIIS -- operate natively on the per-k representation
#     (per-k Pulay or per-k orbital-rotation gradients with a
#     k-weighted Frobenius B-matrix).
#   * EDIIS / ADIIS / EDIIS_DIIS -- bridge to a stacked real-block
#     representation via :func:`per_k_to_stacked_real_blocks`, call
#     into the C++ block-vector ``EDIIS::extrapolate_blocks`` /
#     ``ADIIS::extrapolate_blocks`` overloads (single source of truth
#     with the molecular and C++ multi-k SCF kernels), and recombine
#     back per-k via :func:`stacked_real_blocks_to_per_k`. The stacked
#     bridge is linear and ``a = √w_k``-weighted so the kernel's
#     sum-of-Frobenius bilinear form matches the periodic per-k energy
#     bilinear form ``S_k w_k Re Tr[F(k) D(k)]`` exactly -- see this
#     module's "per-k <-> stacked-real-block bridge" section for the
#     derivation and the rejected per-cell-Bloch-bridge alternative.
#
# KDIIS multi-k design (this module's own contribution). Kollmar's
# error metric in the molecular case is the canonical-MO-basis
# orbital-rotation gradient ``g_{ai} = (C^T F C)_{ai}``. For multi-k
# we extend per k:
#
#   g(k)_{ai}  =  (C(k)^+  F(k)  C(k))_{ai}    a in vir(k), i in occ(k)
#
# The Pulay least-squares is augmented by a k-weighted Frobenius inner
# product on the stacked per-k errors:
#
#   B_{ij}  =  S_k  w_k . Re Tr( g_i(k)^+  g_j(k) )
#
# Closed-shell n_occ is k-independent (filled bands); open-shell adds
# per-k n_alpha / n_beta with a stacked a + b error vector and a single
# coefficient set applied to both spins. At the SCF fixed point all
# g(k) vanish (the per-k Brillouin condition), so the error norm is a
# valid convergence indicator just like the AO-basis commutator.


def _restore_zero_weight_k(
    F_ex_list: List[np.ndarray],
    F_k_list: Sequence[np.ndarray],
    weights: Sequence[float],
) -> List[np.ndarray]:
    """Hand back the raw Fock at zero-weight k-points.

    ``stacked_real_blocks_to_per_k`` cannot invert the ``√w_k`` scaling at
    ``w_k = 0`` and returns a zero block there. A zero-weight k carries no
    electrons through the energy or the density, but its bands are still
    diagonalized (band-structure k-paths ride along on zero weight), and
    diagonalizing a zeroed Fock would hand back meaningless orbitals. The
    unextrapolated Fock is the honest answer: DIIS has no information about a
    k-point it never saw in the B-matrix.
    """
    return [
        np.asarray(F_k_list[k]).astype(complex, copy=True)
        if float(w) <= 0.0 else F_ex
        for k, (F_ex, w) in enumerate(zip(F_ex_list, weights))
    ]


class _MultiKPulayDIIS:
    """Pulay DIIS on a per-k Fock-matrix list (closed-shell).

    Error vector per k:  e(k) = F(k) D(k) S(k) - S(k) D(k) F(k).
    B-matrix:           B_{ij} = S_k w_k . Re Tr(e_i(k)^+ e_j(k)).
    Extrapolation:       F_ex(k) = S_i c_i F_i(k) with the Pulay
                         simplex constraint S c_i = 1.

    A thin adapter over the canonical C++ :class:`DIIS`. The per-k complex
    lists are mapped onto ``√w_k``-scaled real blocks (see this module's
    "per-k <-> stacked-real-block bridge" section), under which the C++
    kernel's Euclidean inner product *is* the k-weighted one above. So the
    history, the Gram-matrix cache, the Chupin adaptive-depth policies and
    the Pulay solve all live in one place rather than being re-derived in
    numpy here.
    """

    def __init__(
        self,
        max_subspace: int = 8,
        policy: "DIISDepthPolicy" = DIISDepthPolicy.FIXED,
        adaptive_param: float = 1.0e-4,
    ):
        self.max_subspace = int(max_subspace)
        self.policy = policy
        self.adaptive_param = float(adaptive_param)
        self._diis = DIIS(self.max_subspace, policy, self.adaptive_param)

    @property
    def subspace_size(self) -> int:
        return self._diis.subspace_size

    def clear(self) -> None:
        self._diis.clear()

    def extrapolate(
        self,
        F_k_list: Sequence[np.ndarray],
        error_k_list: Sequence[np.ndarray],
        weights: Sequence[float],
    ) -> List[np.ndarray]:
        F_ex = stacked_real_blocks_to_per_k(
            self._diis.extrapolate_blocks(
                per_k_to_stacked_real_blocks(F_k_list, weights),
                per_k_to_stacked_real_blocks(error_k_list, weights),
            ),
            weights,
        )
        return _restore_zero_weight_k(F_ex, F_k_list, weights)


def _multik_diis_with_policy(opts, max_sub: int) -> _MultiKPulayDIIS:
    """Build a per-k weighted DIIS whose depth policy comes from ``opts``.

    Multi-k analogue of :func:`_diis_with_policy` — R_CDIIS selects the
    restart policy with τ = ``diis_restart_tau``; AD_CDIIS the
    adaptive-depth policy with δ = ``diis_adaptive_delta``; anything else
    a fixed-depth history.
    """
    mode = getattr(opts, "scf_accelerator", SCFAccelerator.DIIS)
    if mode == SCFAccelerator.R_CDIIS:
        return _MultiKPulayDIIS(max_sub, DIISDepthPolicy.RESTART,
                                float(getattr(opts, "diis_restart_tau", 1.0e-4)))
    if mode == SCFAccelerator.AD_CDIIS:
        return _MultiKPulayDIIS(
            max_sub, DIISDepthPolicy.ADAPTIVE,
            float(getattr(opts, "diis_adaptive_delta", 1.0e-4)))
    return _MultiKPulayDIIS(max_sub)


def _kdiis_orbital_gradient(
    F_k: np.ndarray,
    C_k: np.ndarray,
    n_occ_k: int,
) -> np.ndarray:
    """Per-k orbital-rotation gradient.

    Returns the occ-vir block ``g_{ai} = (C(k)^+ F(k) C(k))_{ai}`` of
    the canonical-MO-basis Fock matrix at one k-point. The returned
    array has shape ``(n_vir, n_occ)``.
    """
    F_mo = C_k.conj().T @ F_k @ C_k
    return F_mo[n_occ_k:, :n_occ_k]


class _MultiKKDIIS:
    """KDIIS on a per-k Fock-matrix list (closed-shell).

    Per-k orbital-rotation-gradient error metric:
        g(k)_{ai} = (C(k)^+ F(k) C(k))_{ai},  a in vir, i in occ.

    Pulay least-squares with k-weighted Frobenius B-matrix:
        B_{ij} = S_k w_k . Re Tr(g_i(k)^+ g_j(k)).

    Coefficients apply to the per-k AO-basis Fock list to produce
    ``F_ex(k) = S_i c_i F_i(k)``. At the SCF fixed point every g(k)
    vanishes (per-k Brillouin condition).

    Like :class:`_MultiKPulayDIIS`, a thin adapter over the canonical C++
    :class:`DIIS`: only the error metric differs (orbital-rotation gradient
    instead of the commutator), and ``extrapolate_blocks`` accepts the
    n_vir×n_occ gradient blocks alongside the nbf×nbf Fock blocks because the
    two lists are independent histories.
    """

    def __init__(self, max_subspace: int = 8):
        self.max_subspace = int(max_subspace)
        self._diis = DIIS(self.max_subspace)

    @property
    def subspace_size(self) -> int:
        return self._diis.subspace_size

    def clear(self) -> None:
        self._diis.clear()

    def extrapolate(
        self,
        F_k_list: Sequence[np.ndarray],
        C_k_list: Sequence[np.ndarray],
        n_occ: int,
        weights: Sequence[float],
    ) -> List[np.ndarray]:
        g_k_list = [
            _kdiis_orbital_gradient(F_k, C_k, n_occ)
            for F_k, C_k in zip(F_k_list, C_k_list)
        ]
        F_ex = stacked_real_blocks_to_per_k(
            self._diis.extrapolate_blocks(
                per_k_to_stacked_real_blocks(F_k_list, weights),
                per_k_to_stacked_real_blocks(g_k_list, weights),
            ),
            weights,
        )
        return _restore_zero_weight_k(F_ex, F_k_list, weights)


class _MultiKKDIISOpenShell:
    """KDIIS on a per-k Fock-matrix list (open-shell).

    Stacks per-spin orbital-rotation gradients (a + b) into a single
    error vector and applies one coefficient set to both spins -- the
    spin-coupled convention used by ``cpp/src/kdiis.cpp`` and by the
    open-shell EDIIS / ADIIS overloads.

    Spin coupling falls out of the block representation for free: appending
    the β blocks to the same Fock and error lists makes the C++ kernel's
    per-block Frobenius sum accumulate ``⟨g_α,i, g_α,j⟩ + ⟨g_β,i, g_β,j⟩``
    into one B-matrix, and the single coefficient set it returns applies to
    both spins.
    """

    def __init__(self, max_subspace: int = 8):
        self.max_subspace = int(max_subspace)
        self._diis = DIIS(self.max_subspace)

    @property
    def subspace_size(self) -> int:
        return self._diis.subspace_size

    def clear(self) -> None:
        self._diis.clear()

    def extrapolate(
        self,
        Fa_k_list: Sequence[np.ndarray],
        Fb_k_list: Sequence[np.ndarray],
        Ca_k_list: Sequence[np.ndarray],
        Cb_k_list: Sequence[np.ndarray],
        n_alpha: int,
        n_beta: int,
        weights: Sequence[float],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ga = [
            _kdiis_orbital_gradient(F_k, C_k, n_alpha)
            for F_k, C_k in zip(Fa_k_list, Ca_k_list)
        ]
        gb = [
            _kdiis_orbital_gradient(F_k, C_k, n_beta)
            for F_k, C_k in zip(Fb_k_list, Cb_k_list)
        ]
        F_blocks = (per_k_to_stacked_real_blocks(Fa_k_list, weights)
                    + per_k_to_stacked_real_blocks(Fb_k_list, weights))
        g_blocks = (per_k_to_stacked_real_blocks(ga, weights)
                    + per_k_to_stacked_real_blocks(gb, weights))
        out = self._diis.extrapolate_blocks(F_blocks, g_blocks)

        n_half = 2 * len(weights)   # 2 real blocks (Re, Im) per k, per spin
        Fa_ex = stacked_real_blocks_to_per_k(out[:n_half], weights)
        Fb_ex = stacked_real_blocks_to_per_k(out[n_half:], weights)
        return (_restore_zero_weight_k(Fa_ex, Fa_k_list, weights),
                _restore_zero_weight_k(Fb_ex, Fb_k_list, weights))


class MultiKPeriodicSCFAccelerator:
    """Multi-k SCF Fock-extrapolation accelerator (closed-shell).

    Dispatches on ``opts.scf_accelerator`` and exposes
    :meth:`extrapolate_rhf` returning the extrapolated per-k Fock list.
    DIIS / KDIIS operate natively on per-k Hermitian matrices; EDIIS /
    ADIIS / EDIIS_DIIS bridge through the per-cell representation and
    the C++ ``extrapolate_blocks`` overloads (see this module's
    "per-k <-> per-cell Bloch bridge" section).

    Drivers stash ``cells`` and ``kpoints`` / ``weights`` once at SCF
    start and pass them to every :meth:`extrapolate_rhf` call -- the
    same metadata the SCF loop already needs for its Fock build.
    """

    def __init__(self, opts):
        mode = getattr(opts, "scf_accelerator", SCFAccelerator.DIIS)
        max_sub = int(opts.diis_subspace_size)
        switch = float(getattr(opts, "ediis_diis_switch_threshold", 1.0e-2))
        self._mode = mode
        self._switch_threshold = switch
        self._diis: Optional[_MultiKPulayDIIS] = None
        self._kdiis: Optional[_MultiKKDIIS] = None
        self._ediis: Optional[EDIIS] = None
        self._adiis: Optional[ADIIS] = None
        if mode in _DIIS_LIKE:
            # DIIS / R_CDIIS / AD_CDIIS: per-k extrapolation, depth policy
            # baked into the _MultiKPulayDIIS object.
            self._diis = _multik_diis_with_policy(opts, max_sub)
        elif mode == SCFAccelerator.KDIIS:
            self._kdiis = _MultiKKDIIS(max_sub)
        elif mode == SCFAccelerator.EDIIS:
            self._ediis = EDIIS(max_sub)
        elif mode == SCFAccelerator.ADIIS:
            self._adiis = ADIIS(max_sub)
        elif mode == SCFAccelerator.EDIIS_DIIS:
            self._ediis = EDIIS(max_sub)
            self._diis = _MultiKPulayDIIS(max_sub)
        elif mode == SCFAccelerator.ADIIS_DIIS:
            self._adiis = ADIIS(max_sub)
            self._diis = _MultiKPulayDIIS(max_sub)
        else:
            raise ValueError(
                f"MultiKPeriodicSCFAccelerator: unknown scf_accelerator "
                f"{mode!r}"
            )

    @property
    def subspace_size(self) -> int:
        if self._mode in _DIIS_LIKE:
            return int(self._diis.subspace_size)
        if self._mode == SCFAccelerator.KDIIS:
            return int(self._kdiis.subspace_size)
        if self._mode == SCFAccelerator.EDIIS:
            return int(self._ediis.subspace_size())
        if self._mode == SCFAccelerator.ADIIS:
            return int(self._adiis.subspace_size())
        return int(self._diis.subspace_size)  # EDIIS_DIIS

    def extrapolate_rhf(
        self,
        F_k_list: Sequence[np.ndarray],
        *,
        error_k_list: Sequence[np.ndarray],
        density_k_list: Sequence[np.ndarray],
        energy: float,
        mo_coeffs_k_list: Sequence[np.ndarray],
        n_occ: int,
        weights: Sequence[float],
        cells,
        kpoints: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """Closed-shell multi-k extrapolation. See class docstring for
        per-mode dispatch."""
        if _below_extrapolation_floor(
            _k_weighted_error_norm(error_k_list, weights),
            max(float(np.linalg.norm(F)) for F in F_k_list),
        ):
            return [np.asarray(F) for F in F_k_list]
        if self._mode in _DIIS_LIKE:
            return self._diis.extrapolate(F_k_list, error_k_list, weights)
        if self._mode == SCFAccelerator.KDIIS:
            return self._kdiis.extrapolate(
                F_k_list, mo_coeffs_k_list, n_occ, weights
            )
        if self._mode == SCFAccelerator.EDIIS:
            F_blocks = per_k_to_stacked_real_blocks(F_k_list, weights)
            D_blocks = per_k_to_stacked_real_blocks(density_k_list, weights)
            F_ex_blocks = self._ediis.extrapolate_blocks(
                F_blocks, D_blocks, float(energy)
            )
            F_ex_np = [np.asarray(b) for b in F_ex_blocks]
            return stacked_real_blocks_to_per_k(F_ex_np, weights)
        if self._mode == SCFAccelerator.ADIIS:
            F_blocks = per_k_to_stacked_real_blocks(F_k_list, weights)
            D_blocks = per_k_to_stacked_real_blocks(density_k_list, weights)
            F_ex_blocks = self._adiis.extrapolate_blocks(F_blocks, D_blocks)
            F_ex_np = [np.asarray(b) for b in F_ex_blocks]
            return stacked_real_blocks_to_per_k(F_ex_np, weights)
        if self._mode == SCFAccelerator.ADIIS_DIIS:
            # ADIIS analogue of the EDIIS_DIIS hybrid below.
            F_diis = self._diis.extrapolate(F_k_list, error_k_list, weights)
            F_blocks = per_k_to_stacked_real_blocks(F_k_list, weights)
            D_blocks = per_k_to_stacked_real_blocks(density_k_list, weights)
            F_adiis_blocks = self._adiis.extrapolate_blocks(F_blocks, D_blocks)
            F_adiis = stacked_real_blocks_to_per_k(
                [np.asarray(b) for b in F_adiis_blocks], weights
            )
            metric = _ediis_diis_switch_metric(
                _k_weighted_error_norm(error_k_list, weights),
                int(np.asarray(F_k_list[0]).shape[0]),
            )
            if metric > self._switch_threshold:
                return F_adiis
            # DIIS branch: F_adiis never reaches the SCF, so retract it
            # from the anti-replay guard's record of consumed returns.
            self._adiis.discard_last_extrapolation()
            return F_diis
        # EDIIS_DIIS hybrid -- keep both warm; switch on the intensive
        # k-weighted commutator metric against
        # ``ediis_diis_switch_threshold`` (see _ediis_diis_switch_metric).
        F_diis = self._diis.extrapolate(F_k_list, error_k_list, weights)
        F_blocks = per_k_to_stacked_real_blocks(F_k_list, weights)
        D_blocks = per_k_to_stacked_real_blocks(density_k_list, weights)
        F_ediis_blocks = self._ediis.extrapolate_blocks(
            F_blocks, D_blocks, float(energy)
        )
        F_ediis = stacked_real_blocks_to_per_k(
            [np.asarray(b) for b in F_ediis_blocks], weights
        )
        metric = _ediis_diis_switch_metric(
            _k_weighted_error_norm(error_k_list, weights),
            int(np.asarray(F_k_list[0]).shape[0]),
        )
        if metric > self._switch_threshold:
            return F_ediis
        # DIIS branch: F_ediis never reaches the SCF, so retract it from
        # the anti-replay guard's record of consumed returns.
        self._ediis.discard_last_extrapolation()
        return F_diis


class MultiKPeriodicUHFAccelerator:
    """Multi-k SCF Fock-extrapolation accelerator (open-shell).

    Same dispatch as :class:`MultiKPeriodicSCFAccelerator` with per-spin
    Fock / density / MO inputs. Every accelerator uses a single
    spin-coupled history whose coefficient set applies to both a and b:
    DIIS concatenates the a and b per-k block lists (with duplicated
    k-weights) into one Pulay history, so its B matrix is
    ``S_k w_k (Re Tr(e_a_i(k)^+ e_a_j(k)) + Re Tr(e_b_i(k)^+ e_b_j(k)))``
    -- the same spin-coupled metric as :class:`_MultiKKDIISOpenShell`
    and the ``_bridge_uhf_blocks`` EDIIS / ADIIS stacking. See the
    ``_AcceleratorState`` note for why per-spin histories are wrong for
    the J(D_a + D_b)-coupled open-shell Fock pair.
    """

    def __init__(self, opts):
        mode = getattr(opts, "scf_accelerator", SCFAccelerator.DIIS)
        max_sub = int(opts.diis_subspace_size)
        switch = float(getattr(opts, "ediis_diis_switch_threshold", 1.0e-2))
        self._mode = mode
        self._switch_threshold = switch
        self._diis: Optional[_MultiKPulayDIIS] = None
        self._kdiis: Optional[_MultiKKDIISOpenShell] = None
        self._ediis: Optional[EDIIS] = None
        self._adiis: Optional[ADIIS] = None
        if mode in _DIIS_LIKE:
            # DIIS / R_CDIIS / AD_CDIIS: per-k extrapolation, depth policy
            # baked into the _MultiKPulayDIIS object.
            self._diis = _multik_diis_with_policy(opts, max_sub)
        elif mode == SCFAccelerator.KDIIS:
            self._kdiis = _MultiKKDIISOpenShell(max_sub)
        elif mode == SCFAccelerator.EDIIS:
            self._ediis = EDIIS(max_sub)
        elif mode == SCFAccelerator.ADIIS:
            self._adiis = ADIIS(max_sub)
        elif mode == SCFAccelerator.EDIIS_DIIS:
            self._ediis = EDIIS(max_sub)
            self._diis = _MultiKPulayDIIS(max_sub)
        elif mode == SCFAccelerator.ADIIS_DIIS:
            self._adiis = ADIIS(max_sub)
            self._diis = _MultiKPulayDIIS(max_sub)
        else:
            raise ValueError(
                f"MultiKPeriodicUHFAccelerator: unknown scf_accelerator "
                f"{mode!r}"
            )

    @property
    def subspace_size(self) -> int:
        if self._mode in _DIIS_LIKE:
            return int(self._diis.subspace_size)
        if self._mode == SCFAccelerator.KDIIS:
            return int(self._kdiis.subspace_size)
        if self._mode == SCFAccelerator.EDIIS:
            return int(self._ediis.subspace_size())
        if self._mode == SCFAccelerator.ADIIS:
            return int(self._adiis.subspace_size())
        return int(self._diis.subspace_size)  # EDIIS_DIIS

    def _bridge_uhf_blocks(
        self,
        Fa_k_list, Fb_k_list,
        Da_k_list, Db_k_list,
        weights,
    ):
        """Concatenate a and b stacked-real-block lists for the C++
        block-vector kernels (one coefficient set across both spins,
        mirroring the open-shell EDIIS / ADIIS spin-coupled convention).
        """
        Fa_blocks = per_k_to_stacked_real_blocks(Fa_k_list, weights)
        Fb_blocks = per_k_to_stacked_real_blocks(Fb_k_list, weights)
        Da_blocks = per_k_to_stacked_real_blocks(Da_k_list, weights)
        Db_blocks = per_k_to_stacked_real_blocks(Db_k_list, weights)
        F_blocks = list(Fa_blocks) + list(Fb_blocks)
        D_blocks = list(Da_blocks) + list(Db_blocks)
        return F_blocks, D_blocks, len(Fa_blocks)

    def extrapolate_uhf(
        self,
        Fa_k_list: Sequence[np.ndarray],
        Fb_k_list: Sequence[np.ndarray],
        *,
        error_alpha_k_list: Sequence[np.ndarray],
        error_beta_k_list: Sequence[np.ndarray],
        density_alpha_k_list: Sequence[np.ndarray],
        density_beta_k_list: Sequence[np.ndarray],
        energy: float,
        mo_coeffs_alpha_k_list: Sequence[np.ndarray],
        mo_coeffs_beta_k_list: Sequence[np.ndarray],
        n_alpha: int,
        n_beta: int,
        weights: Sequence[float],
        cells,
        kpoints: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Open-shell multi-k extrapolation. See class docstring."""
        if _below_extrapolation_floor(
            max(
                _k_weighted_error_norm(error_alpha_k_list, weights),
                _k_weighted_error_norm(error_beta_k_list, weights),
            ),
            max(
                max(float(np.linalg.norm(F)) for F in Fa_k_list),
                max(float(np.linalg.norm(F)) for F in Fb_k_list),
            ),
        ):
            return (
                [np.asarray(F) for F in Fa_k_list],
                [np.asarray(F) for F in Fb_k_list],
            )
        n_k = len(Fa_k_list)
        if self._mode in _DIIS_LIKE:
            # Spin-coupled: one history over the concatenated a + b
            # per-k block lists with duplicated k-weights (mirrors
            # ``_bridge_uhf_blocks``). One coefficient set extrapolates
            # both spins.
            F_ex = self._diis.extrapolate(
                list(Fa_k_list) + list(Fb_k_list),
                list(error_alpha_k_list) + list(error_beta_k_list),
                list(weights) + list(weights),
            )
            return F_ex[:n_k], F_ex[n_k:]
        if self._mode == SCFAccelerator.KDIIS:
            return self._kdiis.extrapolate(
                Fa_k_list, Fb_k_list,
                mo_coeffs_alpha_k_list, mo_coeffs_beta_k_list,
                n_alpha, n_beta, weights,
            )
        if self._mode == SCFAccelerator.EDIIS:
            F_blocks, D_blocks, n_split = self._bridge_uhf_blocks(
                Fa_k_list, Fb_k_list,
                density_alpha_k_list, density_beta_k_list,
                weights,
            )
            F_ex_blocks = self._ediis.extrapolate_blocks(
                F_blocks, D_blocks, float(energy)
            )
            F_ex_blocks_np = [np.asarray(b) for b in F_ex_blocks]
            Fa_ex = stacked_real_blocks_to_per_k(
                F_ex_blocks_np[:n_split], weights
            )
            Fb_ex = stacked_real_blocks_to_per_k(
                F_ex_blocks_np[n_split:], weights
            )
            return Fa_ex, Fb_ex
        if self._mode == SCFAccelerator.ADIIS:
            F_blocks, D_blocks, n_split = self._bridge_uhf_blocks(
                Fa_k_list, Fb_k_list,
                density_alpha_k_list, density_beta_k_list,
                weights,
            )
            F_ex_blocks = self._adiis.extrapolate_blocks(F_blocks, D_blocks)
            F_ex_blocks_np = [np.asarray(b) for b in F_ex_blocks]
            Fa_ex = stacked_real_blocks_to_per_k(
                F_ex_blocks_np[:n_split], weights
            )
            Fb_ex = stacked_real_blocks_to_per_k(
                F_ex_blocks_np[n_split:], weights
            )
            return Fa_ex, Fb_ex
        if self._mode == SCFAccelerator.ADIIS_DIIS:
            # ADIIS analogue of the EDIIS_DIIS hybrid below.
            F_d = self._diis.extrapolate(
                list(Fa_k_list) + list(Fb_k_list),
                list(error_alpha_k_list) + list(error_beta_k_list),
                list(weights) + list(weights),
            )
            Fa_d = F_d[:n_k]
            Fb_d = F_d[n_k:]
            F_blocks, D_blocks, n_split = self._bridge_uhf_blocks(
                Fa_k_list, Fb_k_list,
                density_alpha_k_list, density_beta_k_list,
                weights,
            )
            F_a_blocks = self._adiis.extrapolate_blocks(F_blocks, D_blocks)
            F_a_blocks_np = [np.asarray(b) for b in F_a_blocks]
            Fa_a = stacked_real_blocks_to_per_k(
                F_a_blocks_np[:n_split], weights
            )
            Fb_a = stacked_real_blocks_to_per_k(
                F_a_blocks_np[n_split:], weights
            )
            # Max-over-spins k-weighted norm + n_blocks=2 (uhf.cpp
            # convention lifted per k).
            metric = _ediis_diis_switch_metric(
                max(
                    _k_weighted_error_norm(error_alpha_k_list, weights),
                    _k_weighted_error_norm(error_beta_k_list, weights),
                ),
                int(np.asarray(Fa_k_list[0]).shape[0]),
                n_blocks=2,
            )
            if metric > self._switch_threshold:
                return Fa_a, Fb_a
            # DIIS branch: the ADIIS pair never reaches the SCF, so
            # retract it from the anti-replay guard's record.
            self._adiis.discard_last_extrapolation()
            return Fa_d, Fb_d
        # EDIIS_DIIS hybrid -- the DIIS branch is spin-coupled via the
        # same a + b list concatenation as the plain-DIIS branch above.
        F_d = self._diis.extrapolate(
            list(Fa_k_list) + list(Fb_k_list),
            list(error_alpha_k_list) + list(error_beta_k_list),
            list(weights) + list(weights),
        )
        Fa_d = F_d[:n_k]
        Fb_d = F_d[n_k:]
        F_blocks, D_blocks, n_split = self._bridge_uhf_blocks(
            Fa_k_list, Fb_k_list,
            density_alpha_k_list, density_beta_k_list,
            weights,
        )
        F_e_blocks = self._ediis.extrapolate_blocks(
            F_blocks, D_blocks, float(energy)
        )
        F_e_blocks_np = [np.asarray(b) for b in F_e_blocks]
        Fa_e = stacked_real_blocks_to_per_k(
            F_e_blocks_np[:n_split], weights
        )
        Fb_e = stacked_real_blocks_to_per_k(
            F_e_blocks_np[n_split:], weights
        )
        # Switch on the intensive commutator metric: max-over-spins
        # k-weighted norm + n_blocks=2 (uhf.cpp convention lifted per k).
        metric = _ediis_diis_switch_metric(
            max(
                _k_weighted_error_norm(error_alpha_k_list, weights),
                _k_weighted_error_norm(error_beta_k_list, weights),
            ),
            int(np.asarray(Fa_k_list[0]).shape[0]),
            n_blocks=2,
        )
        if metric > self._switch_threshold:
            return Fa_e, Fb_e
        # DIIS branch: the EDIIS pair never reaches the SCF, so retract
        # it from the anti-replay guard's record.
        self._ediis.discard_last_extrapolation()
        return Fa_d, Fb_d
