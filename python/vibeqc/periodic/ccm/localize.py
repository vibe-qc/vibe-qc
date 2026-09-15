"""Post-SCF localization of the CCM canonical crystalline orbitals (Wannier).

Task 1 of the aiccm2026dev-a follow-on. It localizes the occupied MOs ``psi``
of the explicitly selected finite Γ-CCM Hamiltonian by a deliberate unitary
rotation *within that occupied space*. The density-invariance result does not
depend on identifying Γ-CCM with a neutral GDF control or with χ-CCM.

Equivalence to the Wannier back-transform
-----------------------------------------
The textbook cell-periodic Wannier construction is the discrete back-transform
over the cyclic-cluster net,

    w_n(r - R) = (1/N) S_k e^{-i k.R} psi_{n,k}(r)            (N = N₁N₂N₃ cells),

giving N translationally-equivalent Wannier functions per band. For one
specified block-circulant finite Hamiltonian, its real-supercell occupied space
and character blocks are related by the unitary block-Fourier matrix above.
Localizing the real-supercell occupieds then selects a localized gauge within
that same space, with no explicit k-sum. This conditional same-H statement is
not a Γ-CCM/χ-CCM construction theorem. (A literal multi-k back-transform plus
``M(k,b)`` gauge is deferred; see the decisions log.)

Method
------
Reuses the validated localizer
:func:`vibeqc.periodic_localise.localise_periodic_gamma` on the cluster supercell:

* ``"pipek-mezey"`` -- Mulliken-population objective; **PBC-safe in every regime**
  (no position operator), including orbitals that wrap the cluster boundary.
* ``"boys"`` -- Foster-Boys ``S_i <i|r|i>^2`` from the home-cell dipole integrals;
  faithful (along with the reported centroids/spreads) only for orbitals that do
  **not** wrap the boundary -- i.e. molecule-in-a-box / dilute / large-cell.

Marzari-Vanderbilt per-orbital spreads ``Ω_i = <i|r^2> - <i|r>^2`` and Wannier
centers are returned in :class:`~vibeqc.periodic_localise.PeriodicWannierResult`.

The unitary rotation is verified to leave the occupied density matrix -- hence the
total energy -- invariant (:func:`localization_density_residual`); a non-zero
residual would mean the rotation left the occupied space, i.e. a bug.

Open items (decisions log): an explicit multi-k U(k)/M(k,b) Wannier route, the
Resta periodic position operator for boundary-wrapping crystals (so Boys/spreads
apply to dense ionic cells), the IAO/IBO path, and aliasing diagnostics when the
localization length exceeds the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "localise_ccm",
    "localization_density_residual",
    "WannierAliasingReport",
    "localization_aliasing",
]


def _occ_coeffs(ccm_result, n_occ: int) -> np.ndarray:
    """Real occupied MO coefficients ``C_occ`` (nbf x n_occ) from a CCM result."""
    C = np.asarray(ccm_result.mo_coeffs)
    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real  # Γ coefficients are real up to phase
    return np.array(C[:, :n_occ], dtype=float)


def localise_ccm(
    ccm_result,
    ccm,
    *,
    method: str = "pipek-mezey",
    n_occ: int | None = None,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> "PeriodicWannierResult":
    """Localize the occupied CCM crystalline orbitals into Wannier functions.

    Parameters
    ----------
    ccm_result
        A converged CCM SCF result (``CCMSCFResult`` from ``run_ccm_rhf`` /
        ``run_ccm_rks``). Duck-typed: needs ``mo_coeffs`` and ``overlap``
        (``S^CCM``).
    ccm
        The :class:`~vibeqc.periodic.ccm.CCMSystem` (supplies the supercell basis,
        geometry, and electron count).
    method
        ``"pipek-mezey"`` (default, PBC-safe) or ``"boys"`` (non-wrapping only).
    n_occ
        Number of doubly-occupied orbitals to localize. Defaults to the supercell
        closed-shell count ``n_electrons // 2`` -- i.e. all N.(occ per cell)
        Wannier functions of the cluster.

    Returns
    -------
    PeriodicWannierResult
        ``C_loc`` (the localized occupieds in the rep Task 2 consumes), the
        unitary ``U``, Wannier ``centers`` / ``centroids`` / ``spreads``,
        Mulliken ``charges`` and ``localization`` metric.
    """
    from ...periodic_localise import localise_periodic_gamma

    if n_occ is None:
        n_occ = int(ccm.supercell.n_electrons()) // 2
    # Localize the selected finite Hamiltonian's occupied space on its cluster.
    return localise_periodic_gamma(
        ccm_result, ccm.basis, ccm.cluster_system,
        method=method, n_occ=n_occ, max_iter=max_iter, conv_tol=conv_tol,
    )


def localization_density_residual(ccm_result, wannier) -> float:
    """``‖P_loc - P_canonical‖_F`` of the occupied density matrices (validation).

    Localization is a unitary rotation *within* the occupied space, so the
    occupied density ``P = S_i C_i C_iᵀ`` is invariant -- and the total energy,
    a functional of ``P``, with it. This Frobenius residual is therefore ~0 to
    SCF tolerance; a non-zero value flags a rotation that left the occupied
    space (a bug). (Uses the closed-shell ``P = C_occ C_occᵀ``; the factor of 2
    cancels in the comparison.)
    """
    n = int(wannier.n_occ)
    C_can = _occ_coeffs(ccm_result, n)
    C_loc = np.asarray(wannier.C_loc, dtype=float)
    return float(np.linalg.norm(C_loc @ C_loc.T - C_can @ C_can.T))


@dataclass
class WannierAliasingReport:
    """Per-orbital Wannier-aliasing diagnostic for a CCM cluster (Task 1 M2).

    ``aliased[i]`` — orbital ``i``'s spread tail wraps the BvK torus;
    ``ratios[i] = √Ω_i / R_wsc``; ``max_ratio`` the worst orbital; ``any_aliased``
    the cluster-level verdict; ``wsc_inscribed_radius`` (bohr) and the ``safety``
    threshold used.
    """

    aliased: list
    ratios: np.ndarray
    max_ratio: float
    any_aliased: bool
    wsc_inscribed_radius: float
    safety: float


def localization_aliasing(ccm, wannier, *, safety: float = 0.5) -> WannierAliasingReport:
    """Flag Wannier functions whose spread approaches the CCM cluster size.

    The CCM is a finite Born–von-Kármán torus, so a localized orbital decaying over
    its RMS spread ``σ_i = √Ω_i`` (``Ω_i`` the Marzari–Vanderbilt variance
    ``wannier.spreads``) wraps around the torus once ``σ_i`` is comparable to the
    Wigner–Seitz **inscribed radius** ``R_wsc`` (the largest sphere that fits in the
    cluster cell without crossing a WSC face). When that happens the position
    operator folds the wrapped tail and the reported centroid/spread is corrupted —
    the canonical CCM pitfall (the cluster must exceed the localization length).

    Orbital ``i`` is flagged when ``σ_i > safety · R_wsc`` (default
    ``safety = 0.5`` → fires when ~2σ reaches the WSC boundary). This is the
    companion diagnostic to running :func:`localise_ccm` over a cluster-size ladder:
    ``any_aliased`` should clear and the per-orbital spreads should converge once
    the cluster exceeds the localization length (Task 1 M2). It does not modify the
    localization — it only reports whether the result can be trusted.
    """
    if safety <= 0.0:
        raise ValueError(f"safety must be > 0; got {safety}")
    sp = np.asarray(wannier.spreads, dtype=float)
    R = float(ccm.wsc_inscribed_radius)
    sigma = np.sqrt(np.clip(sp, 0.0, None))
    ratios = sigma / R if R > 0.0 else np.full_like(sigma, np.inf)
    aliased = [bool(r > safety) for r in ratios]
    return WannierAliasingReport(
        aliased=aliased,
        ratios=ratios,
        max_ratio=float(ratios.max()) if ratios.size else 0.0,
        any_aliased=bool(any(aliased)),
        wsc_inscribed_radius=R,
        safety=float(safety),
    )
