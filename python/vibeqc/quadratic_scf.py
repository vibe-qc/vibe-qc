"""Phase C1c -- second-order ("quadratic") SCF fallback.

When standard SCF (damping + DIIS + level shift) fails to converge,
switch from "diagonalize F" to a **Newton step in MO space** with
the orbital-rotation Hessian approximated by its diagonal:

    κ_{ai}  =  -F_{ai}^{MO} / (e_a - e_i + l)
    C_new   =  C_prev . exp(κ)            (skew-symmetric κ)
    D_new   =  2 . C_new[:, :n_occ] . C_new[:, :n_occ]^T

The denominator is the diagonal of the (occ, vir | vir, occ) block
of the orbital-rotation Hessian; ``l > 0`` regularises near-
degenerate cases. The step is trust-region capped at ``max_step``
to keep the matrix exponential of κ well-conditioned (the Taylor
series converges quadratically for ``‖κ‖ < 1`` and very fast for
``‖κ‖ ~ 0.1``).

This module ships:

* :func:`expm_skew` -- matrix exponential of a real skew-symmetric
  matrix via scaling-and-squaring + truncated Taylor. Avoids a hard
  ``scipy`` dependency for one operation we use in a hot SCF loop.

* :func:`quadratic_step` -- one Newton step taking ``(F, C, e,
  n_occ)`` and returning ``(C_new, e_new)``. Drop-in replacement
  for the ``diagonalize(F_diag)`` step in the Ewald SCF drivers
  when the C1c fallback is active.

Activation logic lives in the SCF drivers themselves: typically
``opts.quadratic_fallback_iter > 0 and iter_idx > opts.quadratic_fallback_iter``,
with DIIS and level shift skipped while the quadratic phase is
active.

References
----------
The diagonal-Hessian / orbital-energy-difference preconditioner is
the standard "augmented Hessian" approach used by every quantum-
chemistry SCF code (PySCF :func:`pyscf.scf.newton`, ORCA
``! NRSCF``, Q-Chem ``GDM``). Vanilla Newton without preconditioning
is unstable far from convergence; with the energy-difference
denominator it becomes a damped quasi-Newton method that's
demonstrably robust on small-gap insulators where plain DIIS
oscillates.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


__all__ = ["expm_skew", "quadratic_step"]


def expm_skew(K: np.ndarray, max_norm_per_step: float = 0.5) -> np.ndarray:
    """Matrix exponential of a real skew-symmetric matrix.

    Implementation: scaling-and-squaring + truncated Taylor series.
    For ``‖K‖ <= max_norm_per_step``, the Taylor series converges in
    ~10 terms to ~10⁻¹⁵; outside, ``K`` is repeatedly halved until
    the tail is small enough, then the result is squared back.

    The function does **not** check that ``K`` is genuinely skew-
    symmetric -- it just exponentiates whatever you give it. The
    name reflects the use case (we feed it skew-symmetric κ from
    the orbital-rotation Newton step) and lets the caller skip the
    redundant symmetry check on hot paths.

    Parameters
    ----------
    K : (n, n) ndarray
        Real matrix. Skew-symmetric in the SCF use case.
    max_norm_per_step : float
        Frobenius-norm threshold above which to scale-and-square.
        Default 0.5 -- Taylor at order 12 is accurate to 4.10⁻¹⁴ at
        ‖K‖ = 0.5.

    Returns
    -------
    (n, n) ndarray
        Real matrix ``exp(K)``.
    """
    n = K.shape[0]

    # Scaling-and-squaring: find s with ‖K / 2^s‖ <= max_norm_per_step.
    s = 0
    K_scaled = K
    while np.linalg.norm(K_scaled) > max_norm_per_step:
        K_scaled = K_scaled * 0.5
        s += 1

    # Truncated Taylor series. Order 12 is overkill for ‖K‖ <= 0.5
    # (the order-12 term has norm <= 0.5^12 / 12! ≈ 6e-13) but the
    # cost is trivial for the small SCF matrices we work with.
    #
    # Use K's dtype for the identity so complex-K (multi-k SCF, per-k
    # skew-Hermitian rotations) doesn't trigger a real->complex
    # promotion mid-loop. For real K this is just float64.
    dtype = np.result_type(K, np.float64)
    out = np.eye(n, dtype=dtype)
    term = np.eye(n, dtype=dtype)
    for k in range(1, 13):
        term = (term @ K_scaled) / k
        out = out + term

    # Square back: exp(K) = exp(K/2^s)^(2^s).
    for _ in range(s):
        out = out @ out

    return out


def quadratic_step(
    F: np.ndarray,
    C_prev: np.ndarray,
    eps_prev: np.ndarray,
    n_occ: int,
    *,
    shift: float = 0.1,
    max_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """One C1c Newton step in MO space.

    Replaces the ``C, e = diag(F_diag)`` line of the Ewald SCF
    drivers when standard convergence aids fail.

    Parameters
    ----------
    F : (n_bf, n_bf) ndarray
        Current AO-basis Fock matrix. Symmetric (caller's
        responsibility -- typically ``F = 1/2(F + F^T)`` is applied
        before this call).
    C_prev : (n_bf, n_kept) ndarray
        Current MO coefficients (in the canonical-orth or
        full-rank basis; the rotation stays inside whichever
        subspace ``C_prev`` spans).
    eps_prev : (n_kept,) ndarray
        Current MO energies. Used as the diagonal Hessian
        preconditioner.
    n_occ : int
        Number of doubly-occupied MOs (RHF / RKS convention).
        For UHF / UKS, call this once per spin with the per-spin
        ``n_occ_s``.
    shift : float, keyword-only
        Damping l in ``(e_a - e_i + l)``. Default 0.1 Hartree.
    max_step : float, keyword-only
        Trust-region cap on the Frobenius norm of the occ-vir
        rotation κ. Default 0.1.

    Returns
    -------
    C_new : (n_bf, n_kept) ndarray
        Rotated MO coefficients.
    eps_new : (n_kept,) ndarray
        New MO energies -- diagonal of ``C_new^T F C_new``.

    Notes
    -----
    The function is robust to the canonical-orthogonalisation case
    where ``n_kept < n_bf``: the rotation lives entirely in the
    kept-direction subspace, so the ``(n_bf, n_kept)`` shape of
    ``C_new`` is preserved.

    For systems where ``n_occ == n_kept`` (no virtual MOs in the
    subspace), the rotation is the identity and the step is a no-
    op -- this can happen for very small bases on small molecules
    with the canonical-orth threshold pruning aggressively. The
    function returns the input unchanged in that case.
    """
    n_kept = C_prev.shape[1]
    n_vir = n_kept - n_occ

    # Use the conjugate transpose throughout so the function works
    # for complex C / F (multi-k SCF) and reduces to plain transpose
    # for real input (Γ-only SCF).
    Ch = C_prev.conj().T

    if n_vir <= 0:
        # No virtual subspace; rotation is trivially the identity.
        # Refresh e from the diagonal of F in the current MO basis,
        # but C is unchanged.
        eps_refreshed = np.real(np.diag(Ch @ F @ C_prev))
        return C_prev, eps_refreshed

    # F in current MO basis.
    F_mo = Ch @ F @ C_prev
    F_mo = 0.5 * (F_mo + F_mo.conj().T)   # Hermitise residual drift

    # Occ-vir block: F_mo[a, i] for i in occ, a in vir.
    F_ov = F_mo[n_occ:, :n_occ]   # (n_vir, n_occ)

    # Diagonal Hessian denominator. Add ``shift`` to regularise
    # near-degenerate occ-vir pairs.
    eps_o = np.real(eps_prev[:n_occ])
    eps_v = np.real(eps_prev[n_occ:])
    Delta = (eps_v[:, None] - eps_o[None, :]) + float(shift)

    # Newton step on κ.
    kappa_ov = -F_ov / Delta

    # Trust region: cap ‖κ_ov‖_F at max_step.
    kappa_norm = float(np.linalg.norm(kappa_ov))
    if kappa_norm > max_step:
        kappa_ov = kappa_ov * (max_step / kappa_norm)

    # Build full skew-Hermitian κ (n_kept x n_kept). For real input
    # this is skew-symmetric (κ.T = -κ); for complex input it's
    # skew-Hermitian (κ+ = -κ). Either way, exp(κ) is unitary
    # (orthogonal in the real case) so the rotation preserves the
    # MO orthonormality.
    kappa = np.zeros((n_kept, n_kept), dtype=F_ov.dtype)
    kappa[n_occ:, :n_occ] = kappa_ov
    kappa[:n_occ, n_occ:] = -kappa_ov.conj().T

    # Apply orbital rotation: C_new = C_prev . exp(κ).
    U = expm_skew(kappa)
    C_new = C_prev @ U

    # New MO energies = diag(C_new+ F C_new).
    eps_new = np.real(np.diag(C_new.conj().T @ F @ C_new))

    return C_new, eps_new
