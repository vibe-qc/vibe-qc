"""Phase G1-GDF -- analytic Γ-only periodic GDF (compcell) atomic gradient.

The GDF two-electron energy at Γ with density D:

  E_2e = 1/2 tr(D . J) - 1/4 a_HF . tr(D . K)

where J and K are built from the compcell cderi L via
:func:`vibeqc.pbc_gdf._build_j_from_lpq` / :func:`_build_k_from_lpq`.

**J gradient (Coulomb).** The DF-J gradient formula:

  dE_J/dR = S_P g̃_P S_muν D_muν dT[P,muν]/dR - 1/2 S_PQ g̃_P g̃_Q dM_PQ/dR

where g̃ = M^+ r, r_P = S_muν T[P,muν] D_muν, and (M, T) are the
compensated 2c metric and 3c tensor. ``M^+`` is the exact thresholded
pseudoinverse used to build the SCF cderi. Its derivative includes the
Fréchet response of the retained eigenspace, not only the full-rank
``-g̃ g̃^T`` term.

The compensation matrix A maps the fused basis (modrho-aux + compensating
charges) to the physical aux space: M = A.M_fused.A^T, T = A.T_fused.
The gradient maps back onto the fused basis via g̃_fused = A^T.g̃.

**K gradient (exchange).** The DF-K gradient:

  dE_K/dR = +a_HF S_PQ w_PQ dM_PQ/dR - 2 a_HF S_{P,muν} Y^P_{muν} d(P|muν)/dR

with w = (η^P : η^Q), η = M^+ K_occ, K^P_occ,ij = C_occ^T T^P C_occ,
Y^P = C_occ . η^P . C_occ^T. For pure DFT (a_HF = 0) this term is zero.
When the fit drops metric modes, the J and K metric weights use the spectral
divided differences of the thresholded inverse; the displayed outer-product
forms are their full-rank limit.

**Current scope:** Gamma-point closed-shell RHF with the compcell cderi and
without the AFT correction.  The J and K derivatives use the C++
lattice-summed gradient kernels below.  The FT-based Ewald ``V_ne``
derivative includes its real-space, structure-factor, AO-centre, and
``G = 0`` terms.  Production RSGDF, RKS/open-shell, multi-k, and optimizer
wiring remain separate increments.

- :func:`vibeqc._vibeqc_core.compute_2c_eri_lattice_gradient_weighted`
- :func:`vibeqc._vibeqc_core.compute_3c_eri_lattice_gradient_weighted`
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    CoulombMethod,
    LatticeSumOptions,
    PeriodicSystem,
    compute_2c_eri_lattice_gradient_weighted,
    compute_3c_eri_lattice_gradient_weighted,
    compute_overlap_lattice,
    kinetic_lattice_gradient_contribution,
    nuclear_repulsion_gradient_per_cell,
    overlap_lattice_gradient_contribution,
)
from .aux_basis import (
    _CompcellFitState,
    _basis_fingerprint,
    _build_lpq_compcell_state,
)

__all__ = ["compute_gdf_gradient_rhf_gamma", "compute_gdf_gradient"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gamma_density_lattice_set(template_lat, D: np.ndarray) -> np.ndarray:
    """Build a Γ-only lattice-resolved density. For the Γ-point, the
    real-space density is identical in every image cell (k=0 Bloch
    phase = 1)."""
    # Actually, for the GDF route at Γ, the density is the Γ-folded
    # single-cell density. The lattice-resolved D(g) = D_Γ for all g.
    # But for the gradient, the ERI lattice gradient uses a
    # LatticeMatrixSet. Since we compute the 2e gradient through the
    # DF path (not through the 4c ERI path), we don't need D(g).
    # The D passed to the 3c gradient kernel is the single-cell density.
    return np.asarray(D, dtype=np.float64)


@dataclass
class _CompcellGradientCache:
    """Cached intermediates from the compcell pipeline for gradient use.

    The compensation matrix A and the fused basis integrals M_fused
    and T_fused depend only on the geometry, not on the density. They
    are computed once and reused for every gradient evaluation.
    """

    A: np.ndarray  # (n_aux, n_fused) compensation matrix
    M_fused: np.ndarray  # (n_fused, n_fused) pre-compensation 2c metric
    T_fused: np.ndarray  # (n_fused, n_orb, n_orb) pre-compensation 3c tensor
    fused_basis: BasisSet  # the fused basis (modrho-aux + chg)
    lat_opts_2c: LatticeSumOptions
    lat_opts_3c: LatticeSumOptions
    eigvals: np.ndarray
    eigvecs: np.ndarray
    keep_mask: np.ndarray
    inverse_eigvals: np.ndarray
    inverse_frechet: np.ndarray
    linear_dep_thr: float
    system_fingerprint: str
    ao_basis_fingerprint: str
    aux_basis_fingerprint: str
    compcell_eta: float
    lattice_cutoff_bohr: float
    nuclear_cutoff_bohr: float
    rcut_strategy: Optional[str]
    rcut_precision: float
    n_fused: int
    n_aux: int
    n_orb: int
    n_fit: int
    # AFT provenance (G-PBC-002 milestone 3b): when the SCF fit
    # subtracted the reciprocal-space AFT corrections, the J/K gradient
    # assemblies add the matching weighted AFT centre derivatives.
    apply_aft_correction: bool = False
    aft_precision: float = 1e-10
    aft_ft_convention: str = "libint"


def _cache_from_compcell_fit_state(
    state: _CompcellFitState,
    *,
    system_fingerprint: str,
    ao_basis_fingerprint: str,
    aux_basis_fingerprint: str,
    compcell_eta: float,
    lattice_cutoff_bohr: float,
    nuclear_cutoff_bohr: float,
    rcut_strategy: Optional[str],
    rcut_precision: float,
) -> _CompcellGradientCache:
    """Prepare the exact thresholded inverse response used by the SCF fit."""
    eigvals = np.asarray(state.eigvals, dtype=np.float64)
    eigvecs = np.asarray(state.eigvecs, dtype=np.float64)
    keep = np.asarray(state.keep_mask, dtype=bool)
    if eigvals.ndim != 1 or eigvecs.shape != (eigvals.size, eigvals.size):
        raise ValueError("compcell gradient: malformed metric eigensystem")
    if keep.shape != eigvals.shape or not np.any(keep):
        raise ValueError("compcell gradient: malformed retained-mode mask")

    max_eig = float(eigvals[-1])
    threshold = float(state.linear_dep_thr) * max_eig
    eig_scale = max(abs(max_eig), 1.0)
    spectral_tol = 128.0 * np.finfo(np.float64).eps * eig_scale
    if np.any(np.abs(eigvals - threshold) <= spectral_tol):
        raise RuntimeError(
            "compcell gradient: an auxiliary metric eigenvalue lies on the "
            "linear-dependence threshold, so the fit derivative is undefined"
        )

    inverse_eigvals = np.zeros_like(eigvals)
    inverse_eigvals[keep] = 1.0 / eigvals[keep]

    # Divided differences for the Fréchet derivative of the thresholded
    # inverse f(M), f(lambda)=1/lambda on kept modes and zero otherwise.
    inverse_frechet = np.zeros((eigvals.size, eigvals.size), dtype=np.float64)
    kept_pair = keep[:, None] & keep[None, :]
    inverse_frechet[kept_pair] = -(
        inverse_eigvals[:, None] * inverse_eigvals[None, :]
    )[kept_pair]
    mixed_pair = keep[:, None] ^ keep[None, :]
    eig_diff = eigvals[:, None] - eigvals[None, :]
    if np.any(np.abs(eig_diff[mixed_pair]) <= spectral_tol):
        raise RuntimeError(
            "compcell gradient: retained and dropped auxiliary modes are "
            "numerically degenerate, so the fit derivative is undefined"
        )
    value_diff = inverse_eigvals[:, None] - inverse_eigvals[None, :]
    inverse_frechet[mixed_pair] = (
        value_diff[mixed_pair] / eig_diff[mixed_pair]
    )

    n_aux = int(state.A.shape[0])
    n_fused = int(state.A.shape[1])
    n_orb = int(state.T_fused.shape[1])
    return _CompcellGradientCache(
        A=np.asarray(state.A, dtype=np.float64),
        M_fused=np.asarray(state.M_fused, dtype=np.float64),
        T_fused=np.asarray(state.T_fused, dtype=np.float64),
        fused_basis=state.fused_basis,
        lat_opts_2c=state.lat_opts_2c,
        lat_opts_3c=state.lat_opts_3c,
        eigvals=eigvals,
        eigvecs=eigvecs,
        keep_mask=keep,
        inverse_eigvals=inverse_eigvals,
        inverse_frechet=inverse_frechet,
        linear_dep_thr=float(state.linear_dep_thr),
        system_fingerprint=system_fingerprint,
        ao_basis_fingerprint=ao_basis_fingerprint,
        aux_basis_fingerprint=aux_basis_fingerprint,
        compcell_eta=float(compcell_eta),
        lattice_cutoff_bohr=float(lattice_cutoff_bohr),
        nuclear_cutoff_bohr=float(nuclear_cutoff_bohr),
        rcut_strategy=rcut_strategy,
        rcut_precision=float(rcut_precision),
        n_fused=n_fused,
        n_aux=n_aux,
        n_orb=n_orb,
        n_fit=int(np.count_nonzero(keep)),
        apply_aft_correction=bool(
            getattr(state, "apply_aft_correction", False)
        ),
        aft_precision=float(getattr(state, "aft_precision", 1e-10)),
        aft_ft_convention=str(
            getattr(state, "aft_ft_convention", "libint")
        ),
    )


def _system_fingerprint(system: PeriodicSystem) -> str:
    """Return a stable fingerprint for gradient-relevant cell geometry."""
    digest = hashlib.sha256(b"vibeqc-periodic-gradient-system-v1\0")
    metadata = (int(system.dim), int(system.charge), int(system.multiplicity))
    digest.update(":".join(str(value) for value in metadata).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        np.asarray(system.lattice, dtype="<f8").tobytes(order="C")
    )
    for atom in system.unit_cell:
        digest.update(str(int(atom.Z)).encode("ascii"))
        digest.update(b":")
        digest.update(
            np.asarray(atom.xyz, dtype="<f8").tobytes(order="C")
        )
    return digest.hexdigest()


def _rcut_strategy_value(strategy: Optional[object]) -> Optional[str]:
    if strategy is None:
        return None
    return str(getattr(strategy, "value", strategy))


def _build_compcell_gradient_cache(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_basis: BasisSet,
    compcell_eta: float,
    lattice_opts: LatticeSumOptions,
    *,
    linear_dep_thr: float = 1e-9,
    rcut_strategy: Optional[object] = None,
    rcut_precision: float = 1e-8,
    fit_state: Optional[_CompcellFitState] = None,
    apply_aft_correction: bool = False,
    aft_precision: float = 1e-10,
    aft_ft_convention: str = "libint",
) -> _CompcellGradientCache:
    """Build the geometry-dependent (density-independent) cache for
    compcell GDF gradient evaluation.

    Uses the same private fit builder as the SCF so the resolved 2c/3c
    cutoffs, compensated integrals, threshold, retained eigenspace, and
    AFT subtraction are identical. ``fit_state`` avoids rebuilding those
    intermediates when the gradient is requested directly by the SCF
    driver (its retained AFT provenance then wins over the keyword
    defaults here).
    """
    if fit_state is None:
        fit_state = _build_lpq_compcell_state(
            system,
            ao_basis,
            aux_basis,
            molecule=system.unit_cell_molecule(),
            lat_opts=lattice_opts,
            linear_dep_thr=float(linear_dep_thr),
            compcell_eta=float(compcell_eta),
            apply_aft_correction=bool(apply_aft_correction),
            aft_precision=float(aft_precision),
            aft_ft_convention=str(aft_ft_convention),
            rcut_strategy=rcut_strategy,
            rcut_precision=float(rcut_precision),
        )
    return _cache_from_compcell_fit_state(
        fit_state,
        system_fingerprint=_system_fingerprint(system),
        ao_basis_fingerprint=_basis_fingerprint(ao_basis),
        aux_basis_fingerprint=_basis_fingerprint(aux_basis),
        compcell_eta=float(compcell_eta),
        lattice_cutoff_bohr=float(lattice_opts.cutoff_bohr),
        nuclear_cutoff_bohr=float(lattice_opts.nuclear_cutoff_bohr),
        rcut_strategy=_rcut_strategy_value(rcut_strategy),
        rcut_precision=float(rcut_precision),
    )


def _validated_result_madelung(
    system: PeriodicSystem,
    result,
    *,
    caller: str,
) -> float:
    """Validate and return the exchange-divergence constant used by SCF."""
    exxdiv = str(getattr(result, "exxdiv", ""))
    retained = float(getattr(result, "madelung_constant", float("nan")))
    if exxdiv not in {"ewald", "none"} or not np.isfinite(retained):
        raise ValueError(f"{caller}: invalid exxdiv/Madelung provenance")
    if exxdiv == "none":
        if abs(retained) > 1e-15:
            raise ValueError(
                f"{caller}: exxdiv='none' requires a zero retained "
                "Madelung constant"
            )
        return retained

    from .madelung import madelung_constant_for_cell

    expected = float(madelung_constant_for_cell(system))
    if retained <= 0.0 or not np.isclose(
        retained,
        expected,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(
            f"{caller}: retained Ewald Madelung constant does not match "
            "the periodic cell"
        )
    return retained


def _validate_supported_gradient_result(
    result, *, caller: str, allow_ks: bool = False
) -> None:
    """Fail closed unless ``result`` is from an implemented compcell route.

    ``allow_ks=False`` (the historical contract) accepts exactly the Γ
    RHF compcell result. ``allow_ks=True`` accepts exactly the Γ
    closed-shell KS compcell result (backend ``pbc-gdf-compcell-rks``
    with a non-empty functional) — the G-PBC-002 milestone-2 envelope.
    """
    if not bool(getattr(result, "converged", False)):
        raise ValueError(f"{caller}: SCF result is not converged")
    backend = str(getattr(result, "backend", ""))
    functional = str(getattr(result, "functional", ""))
    if allow_ks:
        if backend != "pbc-gdf-compcell-rks" or not functional:
            raise NotImplementedError(
                f"{caller}: only a Gamma closed-shell KS compcell result "
                f"is supported on this path; got backend={backend!r}, "
                f"functional={functional!r}."
            )
    elif backend != "pbc-gdf-compcell" or functional:
        raise NotImplementedError(
            f"{caller}: only a Gamma RHF compcell result is supported; "
            f"got backend={backend!r}, functional={functional!r}."
        )
    # AFT-corrected fits are supported since G-PBC-002 milestone 3b (the
    # 2c/3c AFT centre derivatives); both AFT-on and AFT-off results are
    # differentiated exactly.
    if str(getattr(result, "v_ne_backend", "")) != "analytic_ft":
        raise NotImplementedError(
            f"{caller}: the converged result must use the analytical FT "
            "Ewald V_ne backend."
        )


def _aft_2c_gradient_term(
    system: PeriodicSystem,
    cache: _CompcellGradientCache,
    omega_fused: np.ndarray,
) -> np.ndarray:
    """Weighted 2c AFT derivative for one metric-weight contraction.

    The SCF fit subtracts the j2c_p correction from the chg rows of the
    bare fused metric and its transpose from the (aux-row, chg-column)
    cross block (``_build_lpq_compcell_state``), so for a weight matrix
    ``omega`` the total-metric contraction splits as

        Σ_ij ω_ij dM_total_ij = Σ_ij ω_ij dM_bare_ij - Σ_cf V_cf dj2c_cf

    with ``V[c, f] = ω[n_aux + c, f] + (f < n_aux) ω[f, n_aux + c]``
    (exact pattern mapping — no symmetry assumption on ω). The C++
    lattice kernels supply the bare piece; this returns the AFT piece
    with its subtraction sign already applied, so callers simply add it.
    """
    from .aux_basis import _compcell_aft_correction_2c_gradient_weighted

    n_aux = cache.n_aux
    V = np.array(omega_fused[n_aux:, :], dtype=np.float64, copy=True)
    V[:, :n_aux] += omega_fused[:n_aux, n_aux:].T
    return -_compcell_aft_correction_2c_gradient_weighted(
        cache.fused_basis,
        n_aux,
        system,
        eta=float(cache.compcell_eta),
        weight=V,
        precision=float(cache.aft_precision),
        ft_convention=str(cache.aft_ft_convention),
    )


def _aft_3c_gradient_term(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    cache: _CompcellGradientCache,
    W_fused_flat: np.ndarray,
) -> np.ndarray:
    """Weighted 3c AFT derivative for one 3c-weight contraction.

    The SCF fit subtracts j3c_p from the chg rows of the bare fused 3c
    tensor, so for a weight ``W`` over ``(n_fused, n_orb^2)`` the
    total-tensor contraction gains ``-Σ W[n_aux:, ...] dj3c/dR``.
    Returns that term with the subtraction sign applied.
    """
    from .aux_basis import _compcell_aft_correction_3c_gradient_weighted

    n_aux = cache.n_aux
    n_orb = cache.n_orb
    W3 = np.ascontiguousarray(
        np.asarray(W_fused_flat, dtype=np.float64)[n_aux:].reshape(
            -1, n_orb, n_orb
        )
    )
    return -_compcell_aft_correction_3c_gradient_weighted(
        cache.fused_basis,
        ao_basis,
        n_aux,
        system,
        eta=float(cache.compcell_eta),
        weight=W3,
        precision=float(cache.aft_precision),
        ft_convention=str(cache.aft_ft_convention),
        lat_opts=cache.lat_opts_3c,
    )


def _compute_j_gradient_compcell(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D: np.ndarray,
    cache: _CompcellGradientCache,
) -> np.ndarray:
    """Compute the DF-J (Coulomb) analytic gradient via the compcell path.

    Uses the standard DF gradient formula mapped onto the fused basis
    via the compensation matrix A.

    Parameters
    ----------
    system, ao_basis, D
        Periodic system, orbital basis, and converged density (n_orb, n_orb).
    cache
        Pre-built geometry cache from :func:`_build_compcell_gradient_cache`.

    Returns
    -------
    grad_J : np.ndarray of shape (n_atoms, 3) in Hartree/bohr.
    """
    n_aux = cache.n_aux
    n_fused = cache.n_fused
    n_orb = cache.n_orb

    # Step 1: Compensate T. The metric eigensystem in the cache was built
    # from M_comp = A @ M_fused @ A.T by the exact SCF fit route.
    A = cache.A
    T_comp = np.einsum("iP,Pmn->imn", A, cache.T_fused, optimize=True)

    # Step 2: Contract T with D to get r.
    # r_P = S_muν T_comp[P, mu, ν] . D[mu, ν]
    T_flat = T_comp.reshape(n_aux, n_orb * n_orb)
    D_flat = np.asarray(D, dtype=np.float64).ravel()
    rho = T_flat @ D_flat  # (n_aux,)

    # Step 3: Apply the exact thresholded inverse used by the SCF cderi.
    U = cache.eigvecs
    rho_eig = U.T @ rho
    gamma_tilde = U @ (cache.inverse_eigvals * rho_eig)

    # Step 4: Project g̃ back onto the fused basis.
    # g̃_fused = A^T . g̃  (n_fused,)
    gamma_tilde_fused = A.T @ gamma_tilde  # (n_fused,)

    # Step 5: Differentiate the thresholded inverse. The mixed retained /
    # dropped blocks are the moving-projector response and cannot be written
    # as only -1/2 g g.T when the fit dropped a mode.
    omega_eig = 0.5 * cache.inverse_frechet * np.outer(rho_eig, rho_eig)
    omega_aux = U @ omega_eig @ U.T
    omega_fused = A.T @ omega_aux @ A

    # Step 6: Build the 3c weight W.
    # W[P, muν] = g̃_fused[P] . D[muν]  (n_fused, n_orb x n_orb) row-major
    W = np.outer(gamma_tilde_fused, D_flat)  # (n_fused, n_orb^2)
    W = np.ascontiguousarray(W)  # ensure row-major for C++

    # Step 7: Call the C++ lattice-summed gradient kernels.
    grad_2c = np.asarray(
        compute_2c_eri_lattice_gradient_weighted(
            cache.fused_basis, system, cache.lat_opts_2c, omega_fused
        ),
        dtype=np.float64,
    )
    grad_3c = np.asarray(
        compute_3c_eri_lattice_gradient_weighted(
            ao_basis, cache.fused_basis, system, cache.lat_opts_3c, W
        ),
        dtype=np.float64,
    )

    grad = grad_2c + grad_3c
    if cache.apply_aft_correction:
        grad = grad + _aft_2c_gradient_term(system, cache, omega_fused)
        grad = grad + _aft_3c_gradient_term(system, ao_basis, cache, W)
    return grad


def _compute_k_gradient_compcell(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D: np.ndarray,
    C_occ: np.ndarray,
    alpha_hf: float,
    cache: _CompcellGradientCache,
) -> np.ndarray:
    """Compute the DF-K (exchange) analytic gradient via the compcell path.

    For pure DFT (alpha_hf = 0) returns zero.

    Parameters
    ----------
    system, ao_basis, D, C_occ, alpha_hf
        Same conventions as the molecular DF-K gradient.
    cache
        Same as :func:`_compute_j_gradient_compcell`.

    Returns
    -------
    grad_K : np.ndarray of shape (n_atoms, 3) in Hartree/bohr.
    """
    if alpha_hf == 0.0:
        return np.zeros((len(system.unit_cell), 3), dtype=np.float64)

    n_aux = cache.n_aux
    n_fused = cache.n_fused
    n_orb = cache.n_orb
    n_occ = C_occ.shape[1]

    A = cache.A
    T_comp = np.einsum("iP,Pmn->imn", A, cache.T_fused, optimize=True)

    # Step 1: Build M^P_ij = C_occ^T . T^P . C_occ  (K_occ)
    # M_occ[P, i, j] for each aux function P.
    M_occ = np.einsum("Pmn,mi,nj->Pij", T_comp, C_occ, C_occ, optimize=True)
    # M_occ shape: (n_aux, n_occ, n_occ)

    # Step 2: Apply the SCF's thresholded inverse column-wise.
    M_occ_flat = M_occ.reshape(n_aux, n_occ * n_occ)
    U = cache.eigvecs
    M_occ_eig = U.T @ M_occ_flat
    eta_flat = U @ (cache.inverse_eigvals[:, None] * M_occ_eig)
    eta = eta_flat.reshape(n_aux, n_occ, n_occ)

    # Step 3: Exact metric weight, including the retained/dropped projector
    # response. For a full-rank fit this reduces to +alpha * eta @ eta.T.
    rhs_gram_eig = M_occ_eig @ M_occ_eig.T
    omega_eig = -float(alpha_hf) * cache.inverse_frechet * rhs_gram_eig
    omega_aux = U @ omega_eig @ U.T
    omega_fused = A.T @ omega_aux @ A

    # Step 4: Build Y^P_{muν} = (C_occ . η^P . C_occ^T)_{muν}.
    Y = np.einsum("Pij,mi,nj->Pmn", eta, C_occ, C_occ, optimize=True)
    # Y shape: (n_aux, n_orb, n_orb)
    Y_fused = np.einsum("Pi,Pmn->imn", A, Y, optimize=True)
    # Y_fused shape: (n_fused, n_orb, n_orb)
    Y_flat = Y_fused.reshape(n_fused, n_orb * n_orb)
    Y_flat = np.ascontiguousarray(Y_flat)

    # Step 5: Gradient contract.
    # dE_K/dR = +a_HF S_PQ w_PQ dM_PQ/dR - 2 a_HF S_{P,muν} Y^P_{muν} d(P|muν)/dR
    grad_2c = np.asarray(
        compute_2c_eri_lattice_gradient_weighted(
            cache.fused_basis, system, cache.lat_opts_2c, omega_fused
        ),
        dtype=np.float64,
    )
    grad_3c = np.asarray(
        compute_3c_eri_lattice_gradient_weighted(
            ao_basis, cache.fused_basis, system, cache.lat_opts_3c, Y_flat
        ),
        dtype=np.float64,
    )

    grad = grad_2c - 2.0 * alpha_hf * grad_3c
    if cache.apply_aft_correction:
        grad = grad + _aft_2c_gradient_term(system, cache, omega_fused)
        grad = grad - 2.0 * alpha_hf * _aft_3c_gradient_term(
            system, ao_basis, cache, Y_flat
        )
    return grad


# ---------------------------------------------------------------------------
# rsgdf gradient cache + J/K assemblies (G-PBC-002 milestone 6, rung 3)
# ---------------------------------------------------------------------------


@dataclass
class _RsgdfGradientCache:
    """Geometry-dependent cache for the Γ rsgdf gradient.

    The aux-frame sibling of :class:`_CompcellGradientCache`: rsgdf works
    directly in the modrho auxiliary basis (identity compensation, no
    fused/chg split), its 2c metric eigendecomposition uses the ABSOLUTE
    linear-dependence threshold of :func:`build_lpq_native_fft`, and its
    M/T centre derivatives are the reciprocal-space weighted kernels
    ``_rsgdf_weighted_2c_metric_gradient`` /
    ``_rsgdf_weighted_3c_tensor_gradient`` instead of the C++ lattice-sum
    kernels.
    """

    aux_basis: BasisSet  # modrho-rescaled
    T: np.ndarray  # (n_aux, n_orb, n_orb) dense-mesh 3c tensor
    eigvals: np.ndarray
    eigvecs: np.ndarray
    keep_mask: np.ndarray
    inverse_eigvals: np.ndarray
    inverse_frechet: np.ndarray
    linear_dep_thr: float  # ABSOLUTE (build_lpq_native_fft convention)
    ke_cutoff: float
    tail_ke_cutoff: Optional[float]
    tail_pair_ft_screen: float
    lat_opts: LatticeSumOptions
    n_aux: int
    n_orb: int
    n_fit: int
    fit_screen_threshold: float  # 0.0 = unscreened (exact) fit
    # Schwarz AO-pair keep mask of the screened fit ((n_orb, n_orb)
    # bool; None = every pair kept). Masked pairs are hard-zeroed in T,
    # so their derivative AT FIXED MASK is exactly zero — the J/K
    # assemblies zero the corresponding weight entries before the 3c
    # derivative kernels.
    pair_keep_ao: Optional[np.ndarray]


def _build_rsgdf_gradient_cache(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_modrho: BasisSet,
    *,
    ke_cutoff: float,
    tail_ke_cutoff: Optional[float],
    lat_opts: LatticeSumOptions,
    linear_dep_thr: float,
    tail_pair_ft_screen: float = 0.0,
    fit_screen_threshold: float = 0.0,
) -> _RsgdfGradientCache:
    """Rebuild the rsgdf dense-mesh M/T and prepare the fit response.

    Reconstructs exactly the :func:`build_lpq_native_fft` base + tail
    metric and 3c tensor (chunked), eigendecomposes with the ABSOLUTE
    threshold, and prepares the same divided-difference Fréchet inverse
    the compcell cache uses — with the threshold guard evaluated in the
    absolute convention.

    ``fit_screen_threshold > 0`` mirrors the SCF's Schwarz-screened fit
    instead: ``build_lpq_native_fft`` delegates screened builds to the
    ``(k_bra, k_ket) = (0, 0)`` Bloch builder (ONE screened build path),
    so the rebuild follows that route bit-for-bit — the shifted q = 0
    mesh, complex chunked M/T accumulation, the SAME
    ``_rsgdf_fit_pair_keep_mask`` from the base-mesh metric, masked
    pair-FT zeroing at the identical point of the T sweep (base AND
    tail), q = 0 real projection of M, and NO T symmetrisation (the
    Bloch builder never symmetrises T). T is stored as its real part:
    at Γ the eigenvectors are real, so the SCF's post-fit
    ``np.real(Lpq)`` commutes exactly with the fit contraction (the
    zero-imaginary products of a complex GEMM cannot perturb the real
    accumulation; verified bit-identical against the SCF Lpq in
    ``tests/test_periodic_gdf_gradient.py``).
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells
    from .aux_basis import (
        _ao_scales_for_rsgdf,
        _rsgdf_fit_pair_keep_mask,
        _rsgdf_shifted_dense_g_mesh,
        rsgdf_aux_fourier_transform,
        rsgdf_dense_g_mesh,
    )

    if float(fit_screen_threshold) < 0.0:
        raise ValueError(
            "rsgdf gradient cache: fit_screen_threshold must be >= 0; "
            f"got {fit_screen_threshold}"
        )
    if float(fit_screen_threshold) > 0.0 and float(tail_pair_ft_screen) > 0.0:
        # The screened SCF fit rejects the native tail pair screen
        # (build_lpq_native_fft raises on the Bloch-delegated path);
        # there is nothing consistent to mirror.
        raise ValueError(
            "rsgdf gradient cache: tail_pair_ft_screen is not part of "
            "the Schwarz-screened SCF fit; pass 0.0 with "
            "fit_screen_threshold > 0."
        )

    n_aux = int(aux_modrho.nbasis)
    n_orb = int(ao_basis.nbasis)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(scales, scales)

    chunk = 65536
    pair_keep_ao: Optional[np.ndarray] = None
    if float(fit_screen_threshold) > 0.0:
        # ---- Screened branch: mirror the Bloch builder at Γ ----------
        # (see the docstring). Everything below matches
        # build_lpq_bloch_native_fft at k_bra = k_ket = 0 with the
        # driver's default tail_chunk_g = 65536: same shifted mesh and
        # zero filter, same complex ZGEMM/einsum accumulation order,
        # same mask derived from the base-mesh metric BEFORE the tail.
        q0 = np.zeros(3)
        Gq = _rsgdf_shifted_dense_g_mesh(system, q0, float(ke_cutoff))
        Gq2 = (Gq**2).sum(axis=1)
        nz = Gq2 > 1e-12
        Gq = Gq[nz]
        Gq2 = Gq2[nz]
        coul = (4.0 * np.pi) / Gq2 / V

        M_c = np.zeros((n_aux, n_aux), dtype=np.complex128)
        for lo in range(0, Gq.shape[0], chunk):
            aux_ft_c = rsgdf_aux_fourier_transform(
                aux_modrho, Gq[lo : lo + chunk]
            )
            M_c += (
                aux_ft_c.conj() * coul[lo : lo + chunk][None, :]
            ) @ aux_ft_c.T
        # THE SAME mask builder the SCF fit uses, on the same base-mesh
        # metric (structural consistency, not a copy). The mask depends
        # on the basis, the geometry (shell origins + lattice cells),
        # the fit mesh, and the threshold only.
        pair_keep_ao = _rsgdf_fit_pair_keep_mask(
            ao_basis,
            R_g,
            Gq2,
            coul,
            M_c,
            fit_screen_threshold=float(fit_screen_threshold),
            fit_pair_list=None,
            progress=None,
        )
        T_c = np.zeros((n_aux, n_orb, n_orb), dtype=np.complex128)
        for lo in range(0, Gq.shape[0], chunk):
            aux_ft_c = rsgdf_aux_fourier_transform(
                aux_modrho, Gq[lo : lo + chunk]
            )
            pair_ft_c = ao_pair_fourier_transform_bloch(
                ao_basis, Gq[lo : lo + chunk], R_g, k_cart=q0
            )
            pair_ft_c *= pair_scales[:, :, None]
            if pair_keep_ao is not None:
                pair_ft_c[~pair_keep_ao] = 0.0
            aux_w_c = aux_ft_c.conj() * coul[lo : lo + chunk][None, :]
            T_c += np.einsum("Pk,mnk->Pmn", aux_w_c, pair_ft_c)
        if (
            tail_ke_cutoff is not None
            and float(tail_ke_cutoff) > float(ke_cutoff)
        ):
            # Shifted tail partition; base-boundary tolerance is zero
            # at q = 0, and the tail completes exactly the pairs the
            # base kept (mask applied identically).
            Gq_tail_all = _rsgdf_shifted_dense_g_mesh(
                system, q0, float(tail_ke_cutoff)
            )
            G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
            tail_norms = np.linalg.norm(Gq_tail_all, axis=1)
            Gq_tail = Gq_tail_all[tail_norms > G_base_max]
            for lo in range(0, Gq_tail.shape[0], chunk):
                Gqc = Gq_tail[lo : lo + chunk]
                Gq2c = (Gqc**2).sum(axis=1)
                nzc = Gq2c > 1e-12
                Gqc = Gqc[nzc]
                if Gqc.shape[0] == 0:
                    continue
                coul_c = (4.0 * np.pi) / Gq2c[nzc] / V
                aux_ft_c = rsgdf_aux_fourier_transform(aux_modrho, Gqc)
                pair_ft_c = ao_pair_fourier_transform_bloch(
                    ao_basis, Gqc, R_g, k_cart=q0
                )
                pair_ft_c *= pair_scales[:, :, None]
                if pair_keep_ao is not None:
                    pair_ft_c[~pair_keep_ao] = 0.0
                aux_w_c = aux_ft_c.conj() * coul_c[None, :]
                M_c += aux_w_c @ aux_ft_c.T
                T_c += np.einsum("Pk,mnk->Pmn", aux_w_c, pair_ft_c)
        # q = 0: real-project M like the Bloch builder, then hermitize
        # (conj is a no-op on the projected M — kept for the mirror).
        M = np.real(M_c)
        M = 0.5 * (M + M.conj().T)
        # Γ real projection of T; the Bloch builder projects the Lpq
        # AFTER the fit, which commutes bit-exactly (docstring). NO
        # symmetrisation — the Bloch builder never symmetrises T.
        T = np.real(T_c)
    else:
        M = np.zeros((n_aux, n_aux), dtype=np.float64)
        T = np.zeros((n_aux, n_orb, n_orb), dtype=np.float64)

        def _accumulate(G_block: np.ndarray, screen_tol: float = 0.0) -> None:
            G2 = (G_block**2).sum(axis=1)
            nz = G2 > 0
            Gc = np.ascontiguousarray(G_block[nz])
            if Gc.shape[0] == 0:
                return
            coul = (4.0 * np.pi) / G2[nz] / V
            F = rsgdf_aux_fourier_transform(aux_modrho, Gc)
            rho = ao_pair_fourier_transform_bloch(
                ao_basis, Gc, R_g, k_cart=np.zeros(3),
                screen_tol=float(screen_tol),
            ) * pair_scales[:, :, None]
            aux_w = F.conj() * coul[None, :]
            M[:, :] += np.real(aux_w @ F.T)
            T[:, :, :] += np.real(
                np.einsum("Pk,mnk->Pmn", aux_w, rho, optimize=True)
            )

        G_base = rsgdf_dense_g_mesh(system, float(ke_cutoff))
        for lo in range(0, G_base.shape[0], chunk):
            _accumulate(G_base[lo : lo + chunk])
        if (
            tail_ke_cutoff is not None
            and float(tail_ke_cutoff) > float(ke_cutoff)
        ):
            # Bit-consistency with the SCF fit (M6 rung 9): the SCF's
            # tail sweep applies the native pair-FT screen, and the
            # eigensystem / Fréchet response amplifies any
            # SCF-vs-rebuild inconsistency by 1/(kept-dropped gap) —
            # measured 1.6e9 on the MgO anchor — so the rebuild's TAIL
            # chunks must screen identically (the base sweep is
            # unscreened on both sides; M is aux-FT-only and never
            # screened).
            G_tail_all = rsgdf_dense_g_mesh(system, float(tail_ke_cutoff))
            G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
            norms = np.linalg.norm(G_tail_all, axis=1)
            G_tail = G_tail_all[norms > G_base_max]
            for lo in range(0, G_tail.shape[0], chunk):
                _accumulate(
                    G_tail[lo : lo + chunk],
                    screen_tol=float(tail_pair_ft_screen),
                )

        M = 0.5 * (M + M.T)
        T = 0.5 * (T + T.transpose(0, 2, 1))

    eigvals, U = np.linalg.eigh(M)
    max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
    if max_eig <= 0:
        raise RuntimeError(
            "rsgdf gradient cache: 2c metric has no positive eigenvalue"
        )
    keep = eigvals > float(linear_dep_thr)
    if not np.any(keep):
        raise RuntimeError(
            "rsgdf gradient cache: no eigenvalues above threshold"
        )
    threshold = float(linear_dep_thr)  # ABSOLUTE convention
    eig_scale = max(abs(max_eig), 1.0)
    spectral_tol = 128.0 * np.finfo(np.float64).eps * eig_scale
    if np.any(np.abs(eigvals - threshold) <= spectral_tol):
        raise RuntimeError(
            "rsgdf gradient cache: an auxiliary metric eigenvalue lies "
            "on the linear-dependence threshold; the fit derivative is "
            "undefined"
        )
    inverse_eigvals = np.zeros_like(eigvals)
    inverse_eigvals[keep] = 1.0 / eigvals[keep]
    inverse_frechet = np.zeros((eigvals.size, eigvals.size), dtype=np.float64)
    kept_pair = keep[:, None] & keep[None, :]
    inverse_frechet[kept_pair] = -(
        inverse_eigvals[:, None] * inverse_eigvals[None, :]
    )[kept_pair]
    mixed_pair = keep[:, None] ^ keep[None, :]
    eig_diff = eigvals[:, None] - eigvals[None, :]
    if np.any(np.abs(eig_diff[mixed_pair]) <= spectral_tol):
        raise RuntimeError(
            "rsgdf gradient cache: retained and dropped auxiliary modes "
            "are numerically degenerate; the fit derivative is undefined"
        )
    value_diff = inverse_eigvals[:, None] - inverse_eigvals[None, :]
    inverse_frechet[mixed_pair] = (
        value_diff[mixed_pair] / eig_diff[mixed_pair]
    )

    return _RsgdfGradientCache(
        aux_basis=aux_modrho,
        T=T,
        eigvals=eigvals,
        eigvecs=U,
        keep_mask=keep,
        inverse_eigvals=inverse_eigvals,
        inverse_frechet=inverse_frechet,
        linear_dep_thr=float(linear_dep_thr),
        ke_cutoff=float(ke_cutoff),
        tail_ke_cutoff=(
            float(tail_ke_cutoff) if tail_ke_cutoff is not None else None
        ),
        tail_pair_ft_screen=float(tail_pair_ft_screen),
        lat_opts=lat_opts,
        n_aux=n_aux,
        n_orb=n_orb,
        n_fit=int(np.count_nonzero(keep)),
        fit_screen_threshold=float(fit_screen_threshold),
        pair_keep_ao=pair_keep_ao,
    )


def rsgdf_fit_response_conditioning(cache: _RsgdfGradientCache) -> float:
    """Relative sensitivity estimate of the fit-response gradient terms.

    ``eps · λ_max / λ_min_kept``: float64 eigenvectors of a cluster with
    small eigenvalues are accurate only to this, and the
    truncated-inverse response inherits it as a RELATIVE bound on its
    near-mode contribution. Measured: MgO/STO-3G at the driver threshold
    gives 3e-7, the H2 vacuum anchor 1.3e-11 -- a clean dense-core class
    detector. Informational only: the historical ~2.5e-5 dense-core
    analytic-vs-FD discrepancy this once tried to explain was an e_nuc
    truncation artefact in the driver ENERGY, fixed 2026-07-29 (see the
    RESOLVED dense-core entry, HANDOVER_OPEN_BUGS_V015.md); post-fix the
    dense-core full-SCF FD gate passes at < 5e-9 Ha/bohr.
    """
    lam = np.asarray(cache.eigvals, dtype=np.float64)
    kept = lam[np.asarray(cache.keep_mask, dtype=bool)]
    if kept.size == 0 or lam.size == 0:
        return 0.0
    return float(
        np.finfo(np.float64).eps * float(lam.max()) / float(kept.min())
    )


# ---------------------------------------------------------------------------
# Multi-k rsgdf gradient cache (G-PBC-002 Item 4, rung 2)
# ---------------------------------------------------------------------------


@dataclass
class _MultikRsgdfQGroup:
    """Per-q fit state mirroring ONE shared-q SCF builder call.

    Everything the future multi-k J/K gradient assembly (Item-4 rung
    3/4) consumes about one canonical momentum transfer: the metric
    eigensystem it must respond through and the per-bra 3c tensors it
    contracts. ``M``/``eigvecs`` keep the SCF dtype flow — complex128
    at q != 0, real-projected float64 at q = 0 — because the LAPACK
    driver (zheevd vs dsyevd) is dtype-dispatched and the eigensystem
    must be the SCF fit's bit-for-bit (M6 rung 9: any fit mismatch is
    Fréchet-amplified near threshold). ``inverse_frechet`` is the same
    divided-difference response as the Γ caches, complex128 so the q!=0
    weight algebra composes without a cast at every use site.
    """

    q: np.ndarray  # canonical Cartesian transfer (bohr^-1)
    pairs: list  # (k_i, k_j) index pairs served by this q, driver order
    bra_k_indices: list  # bra k index per T_list entry (pairs' first slots)
    M: np.ndarray  # (n_aux, n_aux) Hermitized; float64 at q=0
    T_list: list  # per bra: (n_aux, n_orb, n_orb) complex128
    eigvals: np.ndarray  # (n_aux,) real ascending
    eigvecs: np.ndarray  # (n_aux, n_aux); complex128 at q!=0, float64 at q=0
    keep_mask: np.ndarray  # ABSOLUTE-threshold retained modes
    inverse_eigvals: np.ndarray  # 1/eigval on kept modes, 0 on dropped
    inverse_frechet: np.ndarray  # (n_aux, n_aux) complex128
    n_fit: int
    # Per-q Schwarz AO-pair keep mask of the screened fit ((n_orb,
    # n_orb) bool; None = every pair kept). Masked pairs are hard-zeroed
    # in every T of this group, so their derivative AT FIXED MASK is
    # exactly zero — the J/K assemblies zero the corresponding weight
    # entries before the Bloch 3c derivative kernel.
    pair_keep_ao: Optional[np.ndarray]


@dataclass
class _MultikRsgdfGradientCache:
    """Geometry-dependent cache for the multi-k rsgdf gradient.

    The shared-q sibling of :class:`_RsgdfGradientCache`: one
    :class:`_MultikRsgdfQGroup` per canonical momentum transfer of the
    FULL ``(k_i, k_j)`` pair set of ``k_cart_list`` (J consumes the
    q = 0 diagonal group; K consumes every group), keyed by the same
    14-decimal q key the SCF's ``_build_rsgdf_lpq_cache_shared_q``
    groups by. No ``tail_pair_ft_screen`` field: unlike the Γ builder's
    tail, the shared-q SCF builder never screens its batched pair-FT,
    so there is nothing to mirror.
    """

    aux_basis: BasisSet  # modrho-rescaled
    k_cart_list: np.ndarray  # (n_k, 3) Cartesian
    groups: dict  # 14-decimal q key -> _MultikRsgdfQGroup
    linear_dep_thr: float  # ABSOLUTE (build_lpq_bloch_native_fft convention)
    ke_cutoff: float
    tail_ke_cutoff: Optional[float]
    lat_opts: LatticeSumOptions
    n_aux: int
    n_orb: int
    fit_screen_threshold: float  # 0.0 = unscreened (exact) fit


def _build_multik_rsgdf_gradient_cache(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_modrho: BasisSet,
    k_cart_list: np.ndarray,
    *,
    ke_cutoff: float,
    tail_ke_cutoff: Optional[float],
    lat_opts: LatticeSumOptions,
    linear_dep_thr: float,
    fit_screen_threshold: float = 0.0,
) -> _MultikRsgdfGradientCache:
    """Rebuild the shared-q rsgdf M(q)/T(q,k) and prepare fit responses.

    Bit-for-bit mirror of :func:`vibeqc.aux_basis.
    build_lpq_bloch_native_fft_shared_q` on every unique canonical q of
    the full ``(k_i, k_j)`` pair set, grouped exactly like the SCF's
    ``_build_rsgdf_lpq_cache_shared_q`` (14-decimal canonical-q key,
    bra list in pair order, pairs enumerated i-outer/j-inner). The
    mirrored behaviours that bit-consistency depends on (M6 rung 9):

    * shifted mesh ``0 < |G+q| <= sqrt(2 ke)`` via
      ``_rsgdf_shifted_dense_g_mesh`` (base filtered before chunking,
      tail filtered per chunk after slicing);
    * G-chunk width ``65536 // n_batch`` (the SCF's default
      ``tail_chunk_g`` divided by the q group's bra count), so the
      accumulation boundaries of M and T match;
    * Sweep-A metric accumulation ``M += (aux_ft.conj() * coul)
      @ aux_ft.T`` with the budgeted aux-FT chunk cache reused by
      Sweep B (recomputed chunks are bit-identical, but the budget is
      mirrored so peak memory follows the SCF build);
    * Sweep-B batched ket-Bloch pair FT at ``k_bra + q_canonical``
      phased momenta with ``outer(ao_scales, ao_scales)`` pair scaling
      and a per-k ZGEMM in the same chunk order;
    * tail partition beyond ``sqrt(2 ke)`` with the shifted-sphere
      base-boundary tolerance (zero at q = 0), unscreened like the SCF
      tail;
    * real projection of M at ``|q| < 1e-12`` (dtype-dispatches eigh to
      the same LAPACK driver), Hermitization, ``np.linalg.eigh``, and
      the ABSOLUTE keep threshold per q;
    * for ``fit_screen_threshold > 0``, the per-q Schwarz AO-pair mask
      from THE SAME ``_rsgdf_fit_pair_keep_mask`` the SCF builder calls
      (same base-mesh metric input, BEFORE the tail contributions),
      with masked pairs hard-zeroed in the batched pair FT at the
      identical point of the base and tail T sweeps. The mask is stored
      per q group so the J/K weight algebra can zero the masked-pair
      derivative weights (exactly zero at fixed mask).

    Scope guards: ``fit_pair_list = None`` (space-group pair reduction
    has no cache counterpart) and ``omega_screen = 0``; the signature
    exposes neither. The threshold-proximity and retained/dropped
    degeneracy guards of the Γ caches apply per q.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch_multi
    from ._vibeqc_core import direct_lattice_cells
    from .aux_basis import (
        _ao_scales_for_rsgdf,
        _canonical_reciprocal_transfer,
        _rsgdf_aux_ft_cache_budget_bytes,
        _rsgdf_fit_pair_keep_mask,
        _rsgdf_shifted_dense_g_mesh,
        _rsgdf_shifted_sphere_boundary_tolerance,
        rsgdf_aux_fourier_transform,
    )

    kpts = np.ascontiguousarray(k_cart_list, dtype=float).reshape(-1, 3)
    n_k = int(kpts.shape[0])
    if n_k == 0:
        raise ValueError(
            "multik rsgdf gradient cache: k_cart_list has zero k-points"
        )
    n_aux = int(aux_modrho.nbasis)
    n_orb = int(ao_basis.nbasis)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)

    # Full pair set: J needs the q=0 diagonal, K all q. Same grouping
    # (canonical q, 14-decimal key, insertion order) as the SCF's
    # _build_rsgdf_lpq_cache_shared_q with need_k_pairs=True; its
    # diagonal-only J/COSX grouping is the q=0 subset of this one on a
    # regular mesh, so both SCF shapes are covered.
    groups_meta: dict = {}
    for i in range(n_k):
        for j in range(n_k):
            q_pair = _canonical_reciprocal_transfer(system, kpts[j] - kpts[i])
            q_key = tuple(float(v) for v in np.round(q_pair, 14))
            meta = groups_meta.setdefault(q_key, {"q": q_pair, "pairs": []})
            meta["pairs"].append((i, j))

    groups: dict = {}
    for q_key in sorted(groups_meta):
        meta = groups_meta[q_key]
        group_pairs = list(meta["pairs"])
        bra_indices = [i for i, _ in group_pairs]
        k_bras = np.asarray([kpts[i] for i in bra_indices], dtype=float)
        n_batch = int(k_bras.shape[0])
        # Re-canonicalise like the builder does on entry (idempotent for
        # an already-canonical q, including BZ-boundary labels where
        # +b/2 and -b/2 share the canonical -b/2 representative).
        q = _canonical_reciprocal_transfer(
            system, np.asarray(meta["q"], dtype=float).reshape(3)
        )
        k_kets = k_bras + q[None, :]

        Gq = _rsgdf_shifted_dense_g_mesh(system, q, float(ke_cutoff))
        Gq2 = (Gq**2).sum(axis=1)
        nz = Gq2 > 1e-12
        Gq = Gq[nz]
        Gq2 = Gq2[nz]
        coul = (4.0 * np.pi) / Gq2 / V

        chunk = max(65536 // max(n_batch, 1), 1)
        n_base = int(Gq.shape[0])

        # Sweep A -- M(q), with the SCF builder's budgeted per-chunk
        # aux-FT cache handed to Sweep B (bit-neutral dedup; over-budget
        # chunks are recomputed there, same deterministic function).
        cache_budget = _rsgdf_aux_ft_cache_budget_bytes()
        cached_bytes = 0
        M = np.zeros((n_aux, n_aux), dtype=np.complex128)
        aux_ft_cache: list = []
        for lo in range(0, n_base, chunk):
            aux_ft_c = rsgdf_aux_fourier_transform(
                aux_modrho, Gq[lo : lo + chunk]
            )
            if cached_bytes + aux_ft_c.nbytes <= cache_budget:
                aux_ft_cache.append(aux_ft_c)
                cached_bytes += aux_ft_c.nbytes
            else:
                aux_ft_cache.append(None)
            M += (
                aux_ft_c.conj() * coul[lo : lo + chunk][None, :]
            ) @ aux_ft_c.T

        # Per-q Schwarz pair mask from THE SAME mask builder the SCF's
        # shared-q builder calls, at the same point: base-mesh metric
        # only (the SCF derives the mask before the tail accumulates
        # into M). Returns None when nothing is dropped, so the
        # unscreened path is byte-unchanged.
        pair_keep_ao = _rsgdf_fit_pair_keep_mask(
            ao_basis,
            R_g,
            Gq2,
            coul,
            M,
            fit_screen_threshold=float(fit_screen_threshold),
            fit_pair_list=None,
            progress=None,
        )

        T_list = [
            np.zeros((n_aux, n_orb, n_orb), dtype=np.complex128)
            for _ in range(n_batch)
        ]
        # Per-AO-pair store weight, exactly as the SCF builder forms it
        # (vibeqc.aux_basis.build_lpq_bloch_native_fft_shared_q): the
        # calibration scale with the fit-screen mask folded in as an
        # exact 0.0, applied inside the kernel's own store instead of in
        # two single-threaded NumPy passes over the batch here. Same
        # bit-identical contract, same reason -- see the note there.
        pair_weights = pair_scales
        if pair_keep_ao is not None:
            pair_weights = np.where(pair_keep_ao, pair_scales, 0.0)

        def _accumulate_chunk(
            Gq_chunk: np.ndarray,
            coul_chunk: np.ndarray,
            aux_ft_c: Optional[np.ndarray] = None,
        ) -> None:
            if aux_ft_c is None:
                aux_ft_c = rsgdf_aux_fourier_transform(aux_modrho, Gq_chunk)
            aux_w_c = aux_ft_c.conj() * coul_chunk[None, :]
            # Returns the pair FT already scaled, and zeroed where the
            # fit screen masks a pair (base AND tail chunks route here).
            pair_ft_batch = ao_pair_fourier_transform_bloch_multi(
                ao_basis, Gq_chunk, R_g, k_kets, pair_weights=pair_weights
            )
            n_chunk_g = int(Gq_chunk.shape[0])
            for i_k, pair_ft_c in enumerate(pair_ft_batch):
                T_list[i_k] += (
                    aux_w_c @ pair_ft_c.reshape(n_orb * n_orb, n_chunk_g).T
                ).reshape(n_aux, n_orb, n_orb)

        for chunk_idx, lo in enumerate(range(0, n_base, chunk)):
            _accumulate_chunk(
                Gq[lo : lo + chunk],
                coul[lo : lo + chunk],
                aux_ft_c=aux_ft_cache[chunk_idx],
            )

        if (
            tail_ke_cutoff is not None
            and float(tail_ke_cutoff) > float(ke_cutoff)
        ):
            Gq_tail_all = _rsgdf_shifted_dense_g_mesh(
                system, q, float(tail_ke_cutoff)
            )
            G_base_max = float(np.sqrt(2.0 * float(ke_cutoff)))
            tail_norms = np.linalg.norm(Gq_tail_all, axis=1)
            base_boundary_tolerance = (
                0.0
                if float(np.linalg.norm(q)) < 1.0e-14
                else _rsgdf_shifted_sphere_boundary_tolerance(
                    system, G_base_max, q
                )
            )
            Gq_tail = Gq_tail_all[
                tail_norms > G_base_max + base_boundary_tolerance
            ]
            n_tail = int(Gq_tail.shape[0])
            for lo in range(0, n_tail, chunk):
                Gqc = Gq_tail[lo : lo + chunk]
                Gq2c = (Gqc**2).sum(axis=1)
                nzc = Gq2c > 1e-12
                Gqc = Gqc[nzc]
                if Gqc.shape[0] == 0:
                    continue
                coul_c = (4.0 * np.pi) / Gq2c[nzc] / V
                tail_aux_ft_c = rsgdf_aux_fourier_transform(aux_modrho, Gqc)
                M += (
                    tail_aux_ft_c.conj() * coul_c[None, :]
                ) @ tail_aux_ft_c.T
                _accumulate_chunk(Gqc, coul_c, aux_ft_c=tail_aux_ft_c)

        if float(np.linalg.norm(q)) < 1e-12:
            M = np.real(M)
        M = 0.5 * (M + M.conj().T)

        eigvals, U = np.linalg.eigh(M)
        max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0.0
        if max_eig <= 0:
            raise RuntimeError(
                "multik rsgdf gradient cache: 2c metric has no positive "
                f"eigenvalue at q={q}"
            )
        keep = eigvals > float(linear_dep_thr)
        if not bool(np.any(keep)):
            raise RuntimeError(
                "multik rsgdf gradient cache: no eigenvalues above "
                f"threshold at q={q}"
            )
        threshold = float(linear_dep_thr)  # ABSOLUTE convention
        eig_scale = max(abs(max_eig), 1.0)
        spectral_tol = 128.0 * np.finfo(np.float64).eps * eig_scale
        if np.any(np.abs(eigvals - threshold) <= spectral_tol):
            raise RuntimeError(
                "multik rsgdf gradient cache: an auxiliary metric "
                f"eigenvalue lies on the linear-dependence threshold at "
                f"q={q}; the fit derivative is undefined"
            )
        inverse_eigvals = np.zeros_like(eigvals)
        inverse_eigvals[keep] = 1.0 / eigvals[keep]
        # Same divided-difference response as the Γ caches (eigvals are
        # real for Hermitian M(q)), complex dtype for the q!=0 algebra.
        inverse_frechet = np.zeros(
            (eigvals.size, eigvals.size), dtype=np.complex128
        )
        kept_pair = keep[:, None] & keep[None, :]
        inverse_frechet[kept_pair] = -(
            inverse_eigvals[:, None] * inverse_eigvals[None, :]
        )[kept_pair]
        mixed_pair = keep[:, None] ^ keep[None, :]
        eig_diff = eigvals[:, None] - eigvals[None, :]
        if np.any(np.abs(eig_diff[mixed_pair]) <= spectral_tol):
            raise RuntimeError(
                "multik rsgdf gradient cache: retained and dropped "
                f"auxiliary modes are numerically degenerate at q={q}; "
                "the fit derivative is undefined"
            )
        value_diff = inverse_eigvals[:, None] - inverse_eigvals[None, :]
        inverse_frechet[mixed_pair] = (
            value_diff[mixed_pair] / eig_diff[mixed_pair]
        )

        groups[q_key] = _MultikRsgdfQGroup(
            q=q,
            pairs=group_pairs,
            bra_k_indices=bra_indices,
            M=M,
            T_list=T_list,
            eigvals=eigvals,
            eigvecs=U,
            keep_mask=keep,
            inverse_eigvals=inverse_eigvals,
            inverse_frechet=inverse_frechet,
            n_fit=int(np.count_nonzero(keep)),
            pair_keep_ao=pair_keep_ao,
        )

    return _MultikRsgdfGradientCache(
        aux_basis=aux_modrho,
        k_cart_list=kpts,
        groups=groups,
        linear_dep_thr=float(linear_dep_thr),
        ke_cutoff=float(ke_cutoff),
        tail_ke_cutoff=(
            float(tail_ke_cutoff) if tail_ke_cutoff is not None else None
        ),
        lat_opts=lat_opts,
        n_aux=n_aux,
        n_orb=n_orb,
        fit_screen_threshold=float(fit_screen_threshold),
    )


def _multik_rsgdf_q0_group(cache: _MultikRsgdfGradientCache):
    """Return the cache's q = 0 group (the diagonal cderi blocks J
    consumes). On a regular mesh of distinct k-points the group's pairs
    are exactly the diagonal ``(i, i)`` set; assert that so a future
    exotic mesh fails loudly instead of mis-weighting the J density."""
    for group in cache.groups.values():
        if float(np.linalg.norm(group.q)) < 1e-12:
            if any(i != j for i, j in group.pairs):
                raise NotImplementedError(
                    "multik rsgdf J gradient: the q=0 group contains "
                    "off-diagonal (k_i, k_j) pairs (coincident k-points "
                    "in k_cart_list?); the diagonal-only J weight "
                    "algebra below does not cover that."
                )
            return group
    raise RuntimeError(
        "multik rsgdf J gradient: cache has no q=0 group; "
        "J consumes the diagonal cderi blocks and cannot proceed."
    )


def _compute_j_gradient_multik_rsgdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    weights,
    cache: _MultikRsgdfGradientCache,
) -> np.ndarray:
    """Differentiate the shared auxiliary Gram Coulomb contraction.

    With v_P=sum_k w_k sum_mn T(k)_Pmn D(k)_nm and the thresholded
    Hermitian inverse H=U h(lambda) U^H, E_J=0.5 v^H H v. Therefore
    gamma=H v and dE_J=Re[gamma^H dv]+0.5 v^H dH v. The metric
    response includes retained/dropped divided differences, with no
    real-coordinate assumption. Its unconjugated derivative weights are

      W2 = [0.5 U (h^[1] * outer(U^H v, conj(U^H v))) U^H]^T,
      W3(k)_Pmn = w_k conj(gamma_P) D(k)_nm.

    This is the derivative of the same operator used by Gamma and
    multi-k Hartree assembly, including a complex auxiliary gauge.
    Exchange, one-electron, Pulay and Madelung terms are separate.
    """
    from .aux_basis import (
        _rsgdf_weighted_2c_metric_gradient_bloch,
        _rsgdf_weighted_3c_tensor_gradient_bloch,
    )

    from .periodic_k_gdf import _RangeSeparatedGdfCache

    if isinstance(cache, _RangeSeparatedGdfCache):
        return _compute_range_separated_cache_gradient(
            system, ao_basis, cache.aux_basis, cache, D_k_list, weights,
            memory_byte_cap=(cache.memory_byte_cap - cache.retained_cache_bytes),
            native_workspace_byte_cap=cache.native_workspace_byte_cap,
            coulomb_scale=1.0, exchange_scale=0.0,
        )

    n_aux, n_orb = cache.n_aux, cache.n_orb
    kpts = np.asarray(cache.k_cart_list, dtype=float).reshape(-1, 3)
    n_k = int(kpts.shape[0])
    if len(D_k_list) != n_k:
        raise ValueError(
            f"multik rsgdf J gradient: {len(D_k_list)} density blocks "
            f"for {n_k} cached k-points."
        )
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n_k:
        raise ValueError(
            f"multik rsgdf J gradient: {w.shape[0]} k-weights for "
            f"{n_k} cached k-points."
        )
    group = _multik_rsgdf_q0_group(cache)
    if sorted(group.bra_k_indices) != list(range(n_k)):
        raise RuntimeError(
            "multik rsgdf J gradient: q=0 group does not cover every "
            f"k-point (bras {group.bra_k_indices} of {n_k})."
        )
    U = group.eigvecs
    # t_P = S_k w_k S_ls T0(k)[P,l,s] D(k)[s,l]  (flat: T @ D^T.ravel).
    t = np.zeros(n_aux, dtype=np.complex128)
    D_T_flat = {}
    for bra_pos, k_idx in enumerate(group.bra_k_indices):
        T_flat = group.T_list[bra_pos].reshape(n_aux, n_orb * n_orb)
        Dt = np.ascontiguousarray(
            np.asarray(D_k_list[k_idx], dtype=np.complex128).T
        ).reshape(-1)
        D_T_flat[k_idx] = Dt
        t += w[k_idx] * (T_flat @ Dt)

    t_eig = U.conj().T @ t
    gamma = U @ (group.inverse_eigvals * t_eig)
    omega_eig = 0.5 * group.inverse_frechet * np.outer(t_eig, t_eig.conj())
    omega_aux = (U @ omega_eig @ U.conj().T).T
    grad = _rsgdf_weighted_2c_metric_gradient_bloch(
        cache.aux_basis,
        system,
        ke_cutoff=cache.ke_cutoff,
        q_cart=group.q,
        weight=omega_aux,
        tail_ke_cutoff=cache.tail_ke_cutoff,
    )
    for bra_pos, k_idx in enumerate(group.bra_k_indices):
        W3 = (w[k_idx] * np.outer(gamma.conj(), D_T_flat[k_idx])).reshape(
            n_aux, n_orb, n_orb
        )
        if group.pair_keep_ao is not None:
            # Screened fit, fixed mask: masked AO pairs are hard-zeroed
            # in T(q, k), so their derivative is exactly zero. Both
            # pieces of the Bloch weighted-3c kernel are linear in the
            # per-pair weight entries (piece A contracts W3 with the
            # pair-FT values; piece B folds conj(W3) into the per-pair
            # kernel weight Q), so zeroing here removes the masked
            # pairs from the derivative exactly. The 2c metric weight
            # is aux-aux only — no AO-pair dependence, no mask.
            W3[:, ~group.pair_keep_ao] = 0.0
        grad = grad + _rsgdf_weighted_3c_tensor_gradient_bloch(
            cache.aux_basis,
            ao_basis,
            system,
            ke_cutoff=cache.ke_cutoff,
            weight=W3,
            k_ket=kpts[k_idx] + group.q,
            q_cart=group.q,
            lat_opts=cache.lat_opts,
            tail_ke_cutoff=cache.tail_ke_cutoff,
        )
    return grad


def _compute_k_gradient_multik_rsgdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    weights,
    cache: _MultikRsgdfGradientCache,
) -> np.ndarray:
    """Multi-k KRHF DF-K fit-derivative gradient over ALL q groups.

    G-PBC-002 Item-4 rung 4: the Γ :func:`_compute_k_gradient_rsgdf`
    weight algebra generalized to the shared-q multi-k fit. The SCF
    exchange (``_build_k_from_lpq_cache``, periodic_k_gdf.py) is

        K(k_i) = S_j w_j S_L L(i,j)·D(k_j)·conj-contract,
        E_K    = -0.25 S_i w_i Re Tr[D(k_i) K(k_i)]

    (the per-k Hermitization of K is trace-neutral against Hermitian
    D). In unfactorized cache terms — per q group,
    ``L(k_i, k_i + q) = U_keep(q)^H T(q, k_i) / sqrt(λ_keep(q))`` (the
    shared-q builder's exact factorization) — the double contraction
    collapses onto the truncated inverse
    ``H(q) = U_keep Λ_keep^{-1} U_keep^H`` of the Hermitian metric
    eigensystem ``(U, Λ)``:

        E_K = -0.25 S_q Re Tr[ G(q) H(q) ],
        G(q)_PQ = S_{(i,j) in q} w_i w_j
                  Tr[ T(q,k_i)_P D(k_j) T(q,k_i)_Q^H D(k_i) ]

    with ``G(q)`` Hermitian for Hermitian D (so ``Re`` is a no-op in
    exact arithmetic — the inputs are Hermitized on entry, which is
    EXACT with respect to the SCF's ``Re[...]``, not an approximation:
    the anti-Hermitian parts of D and G contract to purely imaginary
    traces). Differentiating:

    * **2c metric weight Ω_K(q).** With the divided-difference Fréchet
      response ``dH = U (h^[1] ∘ (U^H dM U)) U^H`` (``h^[1]`` =
      ``cache.inverse_frechet``: kept/kept ``-1/(λ_a λ_b)``,
      retained/dropped divided difference, dropped/dropped 0),

          -0.25 Tr[G dH] = S_PQ Ω_K(q)_PQ dM_PQ,
          Ω_K(q) = -0.25 conj( U (h^[1] ∘ Ĝ) U^H ),  Ĝ = U^H G(q) U

      — complex Hermitian at q != 0 (h^[1] real symmetric, Ĝ
      Hermitian, and the conj of a Hermitian matrix is Hermitian), so
      ``Re S Ω_K dM`` is real term-by-term and is dispatched to
      :func:`vibeqc.aux_basis._rsgdf_weighted_2c_metric_gradient_bloch`
      (whose docstring shows the ``Re[...]`` contraction is exact
      against the SCF's Hermitized / q=0-real-projected metric for a
      Hermitian weight). At Γ (1 k-point, w = 1, real U, D = 2CC^T)
      ``G = 4·outer-gram(A)`` with ``A = C^T T C``, so
      ``Ω_K = -h^[1] ∘ (U^T A A^T U)``-transformed — exactly the Γ
      template's ``omega_aux`` at ``alpha_hf = 1``.

    * **3c weights (two T appearances -> product rule).** T enters G
      once un-conjugated (bra side) and once conjugated (ket side);
      for Hermitian D and H the two product-rule terms are complex
      conjugates, so ``-0.25 Tr[dG H] = Re S_P,pt dT_P,pt ·
      W3K_P,pt`` with ONE kernel dispatch per (i, j) pair:

          W3K(q, i, j) = -0.5 w_i w_j conj( D(k_i) η(q,k_i)_P D(k_j) ),
          η(q,k_i) = H(q) T(q,k_i)

      contracted by :func:`vibeqc.aux_basis.
      _rsgdf_weighted_3c_tensor_gradient_bloch` at ket momentum
      ``k_i + q_canonical`` (the builder's ket, which at the BZ
      boundary differs from ``k_j`` by a reciprocal vector — lattice
      Bloch phases are invariant). At Γ this is
      ``-2·C (C^T η C) C^T = -2 alpha Y`` — the Γ template's
      ``- 2 alpha grad_3c`` at ``alpha_hf = 1``.

    Symmetries exploited (each exact, none assumed of the mesh):
    Hermitization of D and of G(q) under the SCF's ``Re[...]``; the
    conjugate-pair collapse of the 3c product rule (halves the kernel
    dispatches); Hermiticity of Ω_K(q) making the 2c contraction
    real. No k <-> -k or q <-> -q mesh symmetry is used — every
    (i, j) pair contributes its own 3c term, so IBZ-style reductions
    stay out of scope (cache guards, handover Item-4 rung 6).

    Pure KRHF exchange: the ``-0.25`` SCF prefactor is baked in; the
    hybrid ``alpha_hf`` scaling is the caller's (rung-5) concern.
    NO exxdiv shift here — that is
    :func:`_compute_exxdiv_w_gradient_multik`.
    """
    from .aux_basis import (
        _rsgdf_weighted_2c_metric_gradient_bloch,
        _rsgdf_weighted_3c_tensor_gradient_bloch,
    )

    from .periodic_k_gdf import _RangeSeparatedGdfCache

    if isinstance(cache, _RangeSeparatedGdfCache):
        return _compute_range_separated_cache_gradient(
            system, ao_basis, cache.aux_basis, cache, D_k_list, weights,
            memory_byte_cap=(cache.memory_byte_cap - cache.retained_cache_bytes),
            native_workspace_byte_cap=cache.native_workspace_byte_cap,
            coulomb_scale=0.0, exchange_scale=1.0,
        )

    n_aux, n_orb = cache.n_aux, cache.n_orb
    n_atoms = len(system.unit_cell)
    kpts = np.asarray(cache.k_cart_list, dtype=float).reshape(-1, 3)
    n_k = int(kpts.shape[0])
    if len(D_k_list) != n_k:
        raise ValueError(
            f"multik rsgdf K gradient: {len(D_k_list)} density blocks "
            f"for {n_k} cached k-points."
        )
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n_k:
        raise ValueError(
            f"multik rsgdf K gradient: {w.shape[0]} k-weights for "
            f"{n_k} cached k-points."
        )
    # Hermitize on entry — exact wrt the SCF's Re[...] (see docstring).
    D_h = [
        0.5
        * (
            np.asarray(D, dtype=np.complex128)
            + np.asarray(D, dtype=np.complex128).conj().T
        )
        for D in D_k_list
    ]

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for q_key in sorted(cache.groups):
        group = cache.groups[q_key]
        U = np.asarray(group.eigvecs)

        # ---- G(q): per-pair exchange Gram in the aux frame --------
        # G_PQ = Tr[T_P D_j T_Q^H D_i] = S_ps (T_P D_j)_ps
        #        conj((D_i T_Q)_ps) using conj(D^T) = D (Hermitian).
        G_tot = np.zeros((n_aux, n_aux), dtype=np.complex128)
        for pair_pos, (i, j) in enumerate(group.pairs):
            T = group.T_list[pair_pos]
            TDj = np.matmul(T, D_h[j]).reshape(n_aux, n_orb * n_orb)
            DiT = np.matmul(D_h[i], T).reshape(n_aux, n_orb * n_orb)
            G_tot += (w[i] * w[j]) * (TDj @ DiT.conj().T)
        # Hermitize — exact wrt Re Tr[G H] with Hermitian H.
        G_tot = 0.5 * (G_tot + G_tot.conj().T)

        # ---- 2c: Ω_K(q) = -0.25 conj(U (h^[1] ∘ Ĝ) U^H) -----------
        G_hat = U.conj().T @ G_tot @ U
        omega_eig = group.inverse_frechet * G_hat
        omega_k = -0.25 * np.conj(U @ omega_eig @ U.conj().T)
        grad += _rsgdf_weighted_2c_metric_gradient_bloch(
            cache.aux_basis,
            system,
            ke_cutoff=cache.ke_cutoff,
            q_cart=group.q,
            weight=omega_k,
            tail_ke_cutoff=cache.tail_ke_cutoff,
        )

        # ---- 3c: one conjugate-pair-collapsed dispatch per pair ----
        for pair_pos, (i, j) in enumerate(group.pairs):
            T_flat = group.T_list[pair_pos].reshape(n_aux, n_orb * n_orb)
            eta = (
                U @ (group.inverse_eigvals[:, None] * (U.conj().T @ T_flat))
            ).reshape(n_aux, n_orb, n_orb)
            W3 = (
                -0.5
                * (w[i] * w[j])
                * np.conj(np.matmul(D_h[i], np.matmul(eta, D_h[j])))
            )
            if group.pair_keep_ao is not None:
                # Fixed-mask screened fit: zero the masked-pair 3c
                # derivative weights (the D eta D sandwich repopulates
                # masked entries even though eta inherits T's zeros;
                # see _compute_j_gradient_multik_rsgdf for the
                # linearity argument).
                W3[:, ~group.pair_keep_ao] = 0.0
            grad += _rsgdf_weighted_3c_tensor_gradient_bloch(
                cache.aux_basis,
                ao_basis,
                system,
                ke_cutoff=cache.ke_cutoff,
                weight=W3,
                k_ket=kpts[i] + group.q,
                q_cart=group.q,
                lat_opts=cache.lat_opts,
                tail_ke_cutoff=cache.tail_ke_cutoff,
            )
    return grad


# ---------------------------------------------------------------------------
# Slab (dim=2) truncated-Coulomb GDF gradient
# (HANDOVER_GDF_GRADIENT_DEFERRED.md § 6, rung 4)
# ---------------------------------------------------------------------------


def _compute_range_separated_cache_gradient(
    system, orbital, auxiliary, cache, densities, weights, *,
    memory_byte_cap: int, native_workspace_byte_cap: int,
    coulomb_scale: float = 1.0, exchange_scale: float = 1.0,
):
    """Differentiate a fitted cache with one live shared-q source batch.

    The energy cache owns the physical source parameters and geometry
    signature. Its resident factors and caller-owned density are outside
    this incremental workspace budget. Rebuilding raw T is necessary:
    whitening discarded metric directions cannot recover their response.
    The Hartree source is accumulated across *all* diagonal k points before
    differentiating individual batches. No full-mesh raw-T cache is kept.
    """
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError,
        _build_lpq_range_separated_shared_q,
        _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature,
    )

    if (len(cache.source_parameters) != 8 or cache.source_signature
            != _range_separated_gdf_source_signature(system, orbital, auxiliary)):
        raise ValueError("SR/LR gradient requires the energy cache's geometry and source")
    (omega, pair_cutoff, auxiliary_cutoff, ke_cutoff, threshold,
     image_cap, reciprocal_cap, screen) = cache.source_parameters
    n_k = len(cache.kpoints_cart)
    n_aux, n_orb = int(auxiliary.nbasis), int(orbital.nbasis)
    # Include the input conversion/Hermitian density plus q/pair metadata
    # and the global Hartree source. Both native source construction and
    # response contraction have independent, overlapping reservations.
    fixed = 48*n_k*n_orb*n_orb + 512*n_k*n_k + 64*n_aux + 8192
    remaining = int(memory_byte_cap) - fixed
    source_cap = remaining // 2
    response_cap = remaining - source_cap
    if min(source_cap, response_cap) <= int(native_workspace_byte_cap):
        raise MemoryError("SR/LR gradient source and response exceed workspace cap")
    if (native_workspace_byte_cap <= 0 or not np.isfinite(coulomb_scale)
            or not np.isfinite(exchange_scale)):
        raise ValueError("SR/LR gradient requires a positive workspace and finite scales")
    kpoints = np.asarray(cache.kpoints_cart, dtype=float)
    density = np.asarray(densities)
    w = np.asarray(weights, dtype=float)
    if (density.shape != (n_k, n_orb, n_orb) or w.shape != (n_k,)
            or not np.isfinite(density).all() or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("SR/LR gradient requires finite full-mesh density and weights")
    groups = {}
    for i in range(n_k):
        for j in range(n_k):
            if not exchange_scale and i != j:
                continue
            if (i, j) not in cache.factors:
                raise ValueError("SR/LR gradient requires the full energy-cache pair set")
            q = _canonical_reciprocal_transfer(system, kpoints[j] - kpoints[i])
            key = tuple(float(v) for v in np.round(q, 14))
            groups.setdefault(key, (q, []))[1].append((i, j))
    result = np.zeros((len(system.unit_cell), 3))
    density = np.asarray(density, dtype=complex)
    density = .5 * (density + density.conj().transpose(0, 2, 1))

    for key in sorted(groups):
        q, pairs = groups[key]
        metric_state = {}

        def batches():
            start, size = 0, len(pairs)
            while start < len(pairs):
                selected = pairs[start:start+size]
                # The response algebra retains several T-sized scratch
                # arrays. Reserve these before performing a source build.
                response_bound = (256*n_aux*n_aux
                                  + 96*len(selected)*n_aux*n_orb*n_orb
                                  + 96*n_k*n_orb*n_orb + 8192)
                if response_bound + native_workspace_byte_cap > response_cap:
                    if len(selected) == 1:
                        raise MemoryError("SR/LR gradient single-pair response exceeds workspace cap")
                    size = max(1, len(selected)//2)
                    continue
                try:
                    batch = _build_lpq_range_separated_shared_q(
                        system, orbital, auxiliary,
                        np.asarray([kpoints[i] for i, _ in selected]), q,
                        omega=omega, pair_cutoff=pair_cutoff,
                        auxiliary_cutoff=auxiliary_cutoff, ke_cutoff=ke_cutoff,
                        linear_dep_thr=threshold, memory_byte_cap=source_cap,
                        native_workspace_byte_cap=native_workspace_byte_cap,
                        image_candidate_cap=image_cap,
                        reciprocal_candidate_cap=reciprocal_cap,
                        integral_screen_error=screen, retain_fit_state=True,
                        _metric_state=metric_state,
                    )
                except _RangeSeparatedGdfAdmissionError as exc:
                    if not exc.retry_with_fewer_kpoints or len(selected) == 1:
                        raise
                    size = max(1, len(selected)//2)
                    continue
                for pair, factor in zip(selected, batch.factors):
                    cached = cache.factors[pair]
                    if cache.factor_frame == 'canonical':
                        if cached.shape != (n_aux, n_orb, n_orb):
                            raise RuntimeError('SR/LR gradient canonical factor dimensions differ')
                        state = batch.fit_state
                        kept = np.flatnonzero(state.keep)
                        # The canonical cache retains original auxiliary
                        # coordinates, including annihilated metric modes.
                        # Compare its unique M^(-1/2) T against this source
                        # rebuild rather than treating n_aux as metric rank.
                        canonical = (state.eigenvectors[:, int(kept[0]):]
                                     @ factor.reshape(len(factor), -1)).reshape(cached.shape)
                        scale = max(1., float(np.max(np.abs(cached))))
                        if np.max(np.abs(canonical-cached)) > 256*np.finfo(float).eps*n_aux*scale:
                            raise RuntimeError('SR/LR gradient canonical factors differ from the energy source')
                        del canonical
                    elif cache.factor_frame != 'eigenmode':
                        raise ValueError('SR/LR gradient received an unknown auxiliary frame')
                    elif factor.shape != cached.shape:
                        raise RuntimeError("SR/LR gradient metric rank differs from the energy cache")
                yield selected, batch
                del batch
                start += len(selected)

        has_coulomb = bool(coulomb_scale) and any(i == j for i, j in pairs)
        source = None
        if has_coulomb:
            source = np.zeros(n_aux, dtype=complex)
            for selected, batch in batches():
                for position, (i, j) in enumerate(selected):
                    if i != j:
                        raise ValueError("SR/LR Hartree source requires distinct diagonal mesh points")
                    source += w[i] * np.einsum(
                        'pmn,nm->p', batch.fit_state.three_center[position], density[i],
                    )
                del batch
        first = True
        for selected, batch in batches():
            result += _compute_range_separated_jk_gradient(
                system, orbital, auxiliary, batch, selected, density, w,
                memory_byte_cap=response_cap,
                native_workspace_byte_cap=native_workspace_byte_cap,
                coulomb_scale=coulomb_scale if has_coulomb else 0.,
                exchange_scale=exchange_scale, coulomb_source=source,
                include_coulomb_metric=first,
            )
            first = False
            del batch
    return result


def _compute_range_separated_jk_gradient(
    system, orbital, auxiliary, batch, pairs, densities, weights, *,
    memory_byte_cap: int, native_workspace_byte_cap: int,
    coulomb_scale: float = 1.0, exchange_scale: float = 1.0,
    coulomb_source: Optional[np.ndarray] = None,
    include_coulomb_metric: bool = True,
):
    """Contract SR/LR derivatives on the exact source of one fitted q group.

    All physical integration parameters come from the retained SCF source.
    The caller supplies memory budgets only; geometry or normalized-basis
    changes require a new fit. Borrowed source/factor/density storage is
    separate from this derivative-workspace cap.
    """
    from .aux_basis import _range_separated_gdf_source_signature
    from ._vibeqc_core import compute_gdf_range_separated_gradient_weighted

    state = batch.fit_state
    if state is None or state.source_parameters is None:
        raise ValueError("SR/LR gradient requires retained source provenance")
    if state.source_signature != _range_separated_gdf_source_signature(
        system, orbital, auxiliary,
    ):
        raise ValueError("SR/LR gradient geometry or basis differs from the retained fit")
    omega, pair_cutoff, auxiliary_cutoff, image_cap, screen, threshold = state.source_parameters
    output_bytes = 24 * len(system.unit_cell)
    # The binding borrows the F-contiguous metric weight and C-contiguous
    # tensor weight, but Eigen copies the reciprocal vectors and k points.
    binding_bytes = state.vectors.nbytes + state.ket_kpoints.nbytes + 4096
    response_cap = (int(memory_byte_cap) - int(native_workspace_byte_cap)
                    - output_bytes - binding_bytes)
    if native_workspace_byte_cap <= 0 or response_cap <= 0:
        raise MemoryError("SR/LR gradient workspace exceeds memory cap")
    metric_weight, tensor_weight = _range_separated_jk_response_weights(
        batch, pairs, densities, weights, linear_dep_thr=threshold,
        workspace_byte_cap=response_cap, coulomb_scale=coulomb_scale,
        exchange_scale=exchange_scale,
        coulomb_source=coulomb_source,
        include_coulomb_metric=include_coulomb_metric,
    )
    return compute_gdf_range_separated_gradient_weighted(
        orbital, auxiliary, system, batch.transfer, state.ket_kpoints, state.vectors,
        omega, pair_cutoff, auxiliary_cutoff, metric_weight, tensor_weight,
        output_bytes, int(native_workspace_byte_cap), image_cap, screen,
    )


def _range_separated_jk_response_weights(
    batch,
    pairs,
    densities,
    weights,
    *,
    linear_dep_thr: float,
    workspace_byte_cap: int,
    coulomb_scale: float = 1.0,
    exchange_scale: float = 1.0,
    coulomb_source: Optional[np.ndarray] = None,
    include_coulomb_metric: bool = True,
):
    """Differentiate the exact fitted J/K energy in one SR/LR q group.

    Returns unconjugated M/T weights for the native combined derivative.
    The restricted exchange convention is -1/4 sum_i w_i Tr(D_i K_i).
    An unrestricted caller adds J of the total density and exchange of
    each spin density with ``exchange_scale=2*alpha_hf``. The retained and
    dropped metric modes respond together through the divided differences
    already used by the GDF gradients; no separate integral source or
    eigensystem is reconstructed here.

    A subdivided q=0 group supplies its full raw auxiliary density
    ``coulomb_source = sum_k w_k T(k):D(k).T`` to every batch. Include
    the Coulomb metric response once across those batches. Using a
    batch-local density would lose cross-batch Hartree terms; recovering
    this vector from whitened factors would lose dropped-mode response.

    The explicit cap admits returned weights and numerical temporaries.
    Borrowed source arrays, densities, factors and native derivative
    scratch are separate. This is not a whole-process memory limit.
    """
    state = batch.fit_state
    if state is None:
        raise ValueError("SR/LR fit response requires retained source state")
    source = state.three_center
    n_pairs, n_aux, n_orb, last = source.shape
    n_k = len(densities)
    if (last != n_orb or len(pairs) != n_pairs or len(weights) != n_k
            or any(len(pair) != 2 or any(
                int(index) != index or index < 0 or index >= n_k
                for index in pair) for pair in pairs)
            or len(set(tuple(pair) for pair in pairs)) != n_pairs):
        raise ValueError("SR/LR fit response has inconsistent k pairs or shapes")
    if (not np.isfinite(linear_dep_thr) or linear_dep_thr < 0
            or not np.isfinite([coulomb_scale, exchange_scale]).all()):
        raise ValueError("SR/LR fit response requires finite scales and metric threshold")
    if coulomb_scale:
        if np.linalg.norm(batch.transfer) > 1e-12 or any(i != j for i, j in pairs):
            raise ValueError("SR/LR Coulomb response requires q=0 diagonal pairs")
        if coulomb_source is None and (
            sorted(tuple(pair) for pair in pairs) != [(k, k) for k in range(n_k)]
        ):
            raise ValueError("SR/LR Coulomb response requires every q=0 diagonal or its full source")
    if coulomb_source is not None:
        coulomb_source = np.asarray(coulomb_source)
        if (coulomb_source.shape != (n_aux,)
                or not np.isfinite(coulomb_source).all()):
            raise ValueError("SR/LR Coulomb source must be a finite auxiliary vector")
    # The Frechet helper uses dense auxiliary masks/differences. Charge
    # those, both matrix-product operands, the source-sized sandwiches
    # and their complex conjugates before allocating the first weight.
    reservation = (16 * 16 * n_aux**2 + 6 * source.nbytes
                   + 96 * n_k * n_orb**2 + 4096 + 128 * (n_aux + n_k))
    if reservation > int(workspace_byte_cap):
        raise MemoryError("SR/LR fit response workspace exceeds memory cap")
    w = np.asarray(weights)
    if np.iscomplexobj(w) or w.shape != (n_k,) or not np.isfinite(w).all():
        raise ValueError("SR/LR fit response requires finite real k weights")
    density = [np.asarray(d) for d in densities]
    for d in density:
        if (d.shape != (n_orb, n_orb) or not np.isfinite(d).all()
                or not np.allclose(d, d.conj().T, rtol=0, atol=(
                    128 * np.finfo(float).eps * n_orb * max(1., float(np.max(np.abs(d))))
                ))):
            raise ValueError("SR/LR fit response requires finite Hermitian densities")
    eigenvalues, U = batch.metric_eigenvalues, state.eigenvectors
    spectral_tol = (64 * np.finfo(float).eps * n_aux
                    * max(1., float(np.max(np.abs(eigenvalues)))))
    if np.any(np.abs(eigenvalues - linear_dep_thr) <= spectral_tol):
        raise RuntimeError("SR/LR metric mode lies on the fit threshold; derivative undefined")
    expected_keep = eigenvalues > linear_dep_thr
    if not np.array_equal(state.keep, expected_keep):
        raise ValueError("SR/LR fit response threshold differs from its source state")
    inverse, frechet = _signed_truncated_inverse_frechet(
        eigenvalues, state.keep, spectral_tol=spectral_tol, where="SR/LR fit response",
    )
    metric_weight = np.zeros((n_aux, n_aux), dtype=np.complex128, order="F")
    tensor_weight = np.zeros(source.shape, dtype=np.complex128)
    if coulomb_scale:
        v = coulomb_source
        if v is None:
            v = np.zeros(n_aux, dtype=np.complex128)
            for position, (i, _) in enumerate(pairs):
                v += w[i] * (source[position].reshape(n_aux, -1) @ density[i].T.reshape(-1))
        projected = U.conj().T @ v
        gamma = U @ (inverse * projected)
        if include_coulomb_metric:
            response = .5 * coulomb_scale * frechet * np.outer(projected, projected.conj())
            metric_weight += (U @ response @ U.conj().T).T
            del response
        for position, (i, _) in enumerate(pairs):
            tensor_weight[position] = (
                coulomb_scale * w[i] * gamma.conj()[:, None, None] * density[i].T
            )
    if exchange_scale:
        gram = np.zeros((n_aux, n_aux), dtype=np.complex128)
        for position, (i, j) in enumerate(pairs):
            right = np.matmul(source[position], density[j]).reshape(n_aux, -1)
            left = np.matmul(density[i], source[position]).reshape(n_aux, -1)
            gram += w[i] * w[j] * (right @ left.conj().T)
            del right, left
        gram = .5 * (gram + gram.conj().T)
        response = frechet * (U.conj().T @ gram @ U)
        metric_weight -= .25 * exchange_scale * (U @ response @ U.conj().T).conj()
        del gram, response
        for position, (i, j) in enumerate(pairs):
            projected = U.conj().T @ source[position].reshape(n_aux, -1)
            projected *= inverse[:, None]
            eta = (U @ projected).reshape(n_aux, n_orb, n_orb)
            tensor_weight[position] -= (
                .5 * exchange_scale * w[i] * w[j]
                * np.matmul(density[i], np.matmul(eta, density[j])).conj()
            )
            del projected, eta
    return metric_weight, tensor_weight


def _signed_truncated_inverse_frechet(
    eigvals: np.ndarray,
    keep: np.ndarray,
    *,
    spectral_tol: float,
    where: str,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Divided-difference response of the SIGNED truncated inverse.

    The slab-truncated metric fit
    (:func:`vibeqc.aux_basis._factor_slab_truncated_gdf_metric`) keeps
    modes by ``|λ| > thr`` of an INDEFINITE Hermitian metric — the
    Sundararaman-Arias kernel's finite negative ``K(0) = -πL²/2`` zero
    mode makes one eigenvalue negative — and factors them as
    ``L = |λ|^{-1/2} U_keep^H T`` with the signature returned
    separately. Every signed contraction ``Σ_L s_L L_L (·) L_L^H``
    therefore collapses onto

        f(M) = U_keep diag(s_L / |λ_L|) U_keep^H
             = U_keep diag(1 / λ_L) U_keep^H,

    because ``sign(λ)/|λ| = 1/λ`` for BOTH signs: the signed factor
    contract IS the ordinary spectral truncated inverse, evaluated on
    an indefinite spectrum

        h(λ) = 1/λ   for |λ| > thr (kept),
        h(λ) = 0     otherwise (dropped).

    Daleckii-Krein then gives, for Hermitian ``M = U Λ U^H`` and a
    Hermitian perturbation ``dM``,

        df = U ( h^[1] ∘ (U^H dM U) ) U^H,
        h^[1]_ab = (h(λ_a) - h(λ_b)) / (λ_a - λ_b)   (a != b),
        h^[1]_aa = h'(λ_a),

    whose blocks on the signed spectrum are

    * kept/kept (incl. the diagonal):
      ``(1/λ_a - 1/λ_b)/(λ_a - λ_b) = -1/(λ_a λ_b)``, with the a = b
      limit ``-1/λ_a²``. Equivalently in the signed parametrization,
      ``d(s/|λ|) = -(s/|λ|²) d|λ|`` with ``d|λ| = s·dλ`` — the same
      value, since ``s² = 1``. Note the OPPOSITE-sign kept pair gives a
      POSITIVE entry ``-1/(λ_a λ_b) > 0`` — the definite-metric
      intuition that the kept/kept response is negative does not
      survive the indefinite spectrum.
    * kept/dropped: ``(1/λ_a - 0)/(λ_a - λ_b)`` — the
      retained/dropped projector-rotation response. Well defined
      because ``|λ_a| > thr >= |λ_b|`` forbids ``λ_a = λ_b`` exactly;
      numerical degeneracy is guarded below.
    * dropped/dropped: 0.

    Verified directly against a dense central difference of ``f(M)``
    under a random Hermitian perturbation on an indefinite matrix with
    kept AND dropped modes of both signs (gate <= 1e-9,
    ``test_signed_truncated_inverse_frechet_matches_dense_fd`` in
    ``tests/test_slab_2d_routing.py``) before any J/K assembly consumed
    it. Returns ``(inverse_eigvals, inverse_frechet)``:
    ``inverse_eigvals = h(λ)`` per mode and the full ``h^[1]`` table in
    complex128 so the q != 0 weight algebra composes without casts.
    """
    lam = np.asarray(eigvals, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    inverse_eigvals = np.zeros_like(lam)
    inverse_eigvals[keep] = 1.0 / lam[keep]
    inverse_frechet = np.zeros((lam.size, lam.size), dtype=np.complex128)
    kept_pair = keep[:, None] & keep[None, :]
    inverse_frechet[kept_pair] = -(
        inverse_eigvals[:, None] * inverse_eigvals[None, :]
    )[kept_pair]
    mixed_pair = keep[:, None] ^ keep[None, :]
    lam_diff = lam[:, None] - lam[None, :]
    if np.any(np.abs(lam_diff[mixed_pair]) <= float(spectral_tol)):
        raise RuntimeError(
            f"{where}: retained and dropped metric modes are numerically "
            "degenerate; the signed fit derivative is undefined"
        )
    value_diff = inverse_eigvals[:, None] - inverse_eigvals[None, :]
    inverse_frechet[mixed_pair] = (
        value_diff[mixed_pair] / lam_diff[mixed_pair]
    )
    return inverse_eigvals, inverse_frechet


@dataclass
class _SlabGdfQGroup:
    """Per-q slab fit state mirroring the SCF's per-(i, j) fit builds.

    The slab SCF (``_run_krhf_periodic_slab_gdf``) builds one
    :func:`vibeqc.aux_basis._build_lpq_bloch_slab_truncated` fit per
    ``(k_i, k_j)`` pair — there is no shared-q builder to mirror.
    Grouping by canonical q here is bit-neutral, not an approximation:
    the metric M(q) is a pure deterministic function of the q-shifted
    mesh (the identical floating-point op sequence for every pair of
    the group), so one eigensystem per q reproduces every pair's
    factors bit-for-bit while avoiding n_pairs duplicate zheevd calls —
    pinned per pair by ``np.array_equal`` in
    ``test_slab_gdf_gradient_cache_bit_consistent_with_scf_fit``.
    ``eigvecs`` stays complex128 even at q = 0: unlike the bulk
    shared-q builder, ``_factor_slab_truncated_gdf_metric`` casts the
    (real-projected) metric back to complex128 before ``eigh``, so the
    SCF's LAPACK driver is zheevd at every q and the cache must
    dispatch identically.
    """

    q: np.ndarray  # canonical Cartesian transfer (bohr^-1)
    pairs: list  # (k_i, k_j) pairs served by this q, driver order
    bra_k_indices: list  # bra k index per T_list entry
    M: np.ndarray  # (n_aux, n_aux) the factorizer's exact eigh input
    T_list: list  # per pair: (n_aux, n_orb, n_orb) complex128
    eigvals: np.ndarray  # (n_aux,) real ascending (indefinite spectrum)
    eigvecs: np.ndarray  # (n_aux, n_aux) complex128 (zheevd at every q)
    keep_mask: np.ndarray  # |λ| > thr retained modes
    signs: np.ndarray  # int8 signature of the kept modes, SCF order
    inverse_eigvals: np.ndarray  # h(λ) = 1/λ on kept modes, 0 dropped
    inverse_frechet: np.ndarray  # (n_aux, n_aux) complex128 h^[1] table
    n_fit: int
    g_mesh: np.ndarray  # (n_G, 3) shifted physical mesh incl. zero mode
    kernel_weights: np.ndarray  # K(G+q)/V per mesh point; finite K(0)


@dataclass
class _SlabGdfGradientCache:
    """Geometry-dependent cache for the slab (dim=2) GDF gradient.

    One :class:`_SlabGdfQGroup` per canonical momentum transfer of the
    full ``(k_i, k_j)`` pair set (J consumes the q = 0 diagonal group;
    K every group), keyed by the 14-decimal canonical-q key like the
    bulk multi-k cache. The kernel/mesh facts that make the aux-phase +
    pair-centre derivative COMPLETE carry over from the bulk case
    unchanged: the Sundararaman-Arias weights ``K(G+q)/V``, the mesh,
    and the per-AO pair scales are lattice/exponent/cutoff-only — atom
    positions enter the fit ONLY through the aux-shell phases and the
    Bloch AO-pair Fourier transform.
    """

    aux_basis: BasisSet  # modrho-rescaled
    k_cart_list: np.ndarray  # (n_k, 3) Cartesian
    groups: dict  # 14-decimal q key -> _SlabGdfQGroup
    linear_dep_thr: float  # |λ| threshold (slab signed convention)
    ke_cutoff: float
    lat_opts: LatticeSumOptions
    n_aux: int
    n_orb: int


def _build_slab_gdf_gradient_cache(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    aux_modrho: BasisSet,
    k_cart_list: np.ndarray,
    *,
    ke_cutoff: float,
    lat_opts: LatticeSumOptions,
    linear_dep_thr: float,
    g_chunk: int = 65536,
) -> _SlabGdfGradientCache:
    """Rebuild the slab-truncated M(q)/T(q, k) and prepare fit responses.

    Bit-for-bit mirror of
    :func:`vibeqc.aux_basis._build_lpq_bloch_slab_truncated` on every
    ``(k_i, k_j)`` pair of ``k_cart_list`` (the slab SCF's exact fit
    set), grouped by canonical q per :class:`_SlabGdfQGroup`. The
    mirrored behaviours bit-consistency depends on (the M6-rung-9
    lesson — any fit mismatch is Fréchet-amplified near threshold):

    * the full three-axis ``_slab_truncated_dense_g_mesh`` at
      ``q = canonical(k_j - k_i)`` INCLUDING the exact zero point;
    * the SCF's 65536-wide G chunking with
      ``_slab_truncated_gdf_contractions`` per chunk (per-chunk
      Hermitized metric, einsum 3c contraction, Sundararaman-Arias
      weights + 1/V applied inside);
    * the per-pair (NOT batched) ket-Bloch pair FT at
      ``k_ket = k_bra + q`` over the ``lat_opts`` cell list, with
      ``outer(ao_scales, ao_scales)`` pair scaling — the slab SCF calls
      the single-k ``ao_pair_fourier_transform_bloch``, so the cache
      does too (the batched multi kernel is documented as
      non-bit-identical);
    * real projection of M at ``|q| < 1e-12``, final Hermitization,
      then ``_factor_slab_truncated_gdf_metric``'s exact
      eigendecomposition path: cast to complex128, Hermiticity check,
      re-Hermitization, ``np.linalg.eigh`` (zheevd at every q), and the
      SIGNED ``|λ| > thr`` keep mask.

    The shared aux-FT per chunk (computed once per group instead of
    once per pair) is bit-neutral: ``rsgdf_aux_fourier_transform`` is
    deterministic on identical input, so every pair of a q group sees
    the identical array the SCF's per-pair call would produce.

    Fit responses: ``inverse_eigvals``/``inverse_frechet`` from
    :func:`_signed_truncated_inverse_frechet` (signed truncated
    inverse; see its docstring for the derivation), with the Γ caches'
    threshold-proximity guard applied to ``|λ|`` — the slab keep
    criterion — so a mode sitting numerically ON the threshold fails
    loudly instead of producing an undefined derivative.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells
    from .aux_basis import (
        _ao_scales_for_rsgdf,
        _canonical_reciprocal_transfer,
        _slab_truncated_coulomb_kernel,
        _slab_truncated_dense_g_mesh,
        _slab_truncated_gdf_contractions,
        _slab_truncation_geometry,
        rsgdf_aux_fourier_transform,
    )

    # Validates dim == 2 + the synthesized-normal embedding up front.
    _slab_truncation_geometry(system)

    kpts = np.ascontiguousarray(k_cart_list, dtype=float).reshape(-1, 3)
    n_k = int(kpts.shape[0])
    if n_k == 0:
        raise ValueError(
            "slab GDF gradient cache: k_cart_list has zero k-points"
        )
    threshold = float(linear_dep_thr)
    if threshold <= 0.0:
        raise ValueError(
            "slab GDF gradient cache: linear_dep_thr must be > 0"
        )
    n_aux = int(aux_modrho.nbasis)
    n_orb = int(ao_basis.nbasis)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    chunk = max(int(g_chunk), 1)

    # Full pair set in the slab driver's i-outer/j-inner order.
    groups_meta: dict = {}
    for i in range(n_k):
        for j in range(n_k):
            q_pair = _canonical_reciprocal_transfer(system, kpts[j] - kpts[i])
            q_key = tuple(float(v) for v in np.round(q_pair, 14))
            meta = groups_meta.setdefault(q_key, {"q": q_pair, "pairs": []})
            meta["pairs"].append((i, j))

    groups: dict = {}
    for q_key in sorted(groups_meta):
        meta = groups_meta[q_key]
        group_pairs = list(meta["pairs"])
        bra_indices = [i for i, _ in group_pairs]
        q = _canonical_reciprocal_transfer(
            system, np.asarray(meta["q"], dtype=float).reshape(3)
        )
        g_mesh = _slab_truncated_dense_g_mesh(system, float(ke_cutoff), q_cart=q)
        # The gradient kernels consume the same per-point weights the
        # SCF contractions apply internally (K(G+q)/V, finite zero
        # mode). Chunk-independent: K is evaluated per point.
        kernel_weights = (
            _slab_truncated_coulomb_kernel(system, g_mesh) / V
        )

        M = np.zeros((n_aux, n_aux), dtype=np.complex128)
        T_list = [
            np.zeros((n_aux, n_orb, n_orb), dtype=np.complex128)
            for _ in range(len(group_pairs))
        ]
        for lo in range(0, g_mesh.shape[0], chunk):
            vectors = g_mesh[lo : lo + chunk]
            aux_ft = rsgdf_aux_fourier_transform(aux_modrho, vectors)
            for pair_pos, (i, _j) in enumerate(group_pairs):
                ket = kpts[i] + q
                pair_ft = ao_pair_fourier_transform_bloch(
                    ao_basis, vectors, R_g, k_cart=ket
                )
                pair_ft *= pair_scales[:, :, None]
                metric_chunk, three_center_chunk = (
                    _slab_truncated_gdf_contractions(
                        system, vectors, aux_ft, pair_ft
                    )
                )
                if pair_pos == 0:
                    M += metric_chunk
                T_list[pair_pos] += three_center_chunk

        if float(np.linalg.norm(q)) < 1e-12:
            M = np.real(M)
        M = 0.5 * (M + M.conj().T)

        # _factor_slab_truncated_gdf_metric's exact eigh path, mirrored
        # line for line (complex cast BEFORE eigh at every q).
        M_eigh = np.asarray(M, dtype=np.complex128)
        hermitian_error = (
            float(np.max(np.abs(M_eigh - M_eigh.conj().T)))
            if M_eigh.size
            else 0.0
        )
        matrix_scale = float(np.max(np.abs(M_eigh))) if M_eigh.size else 0.0
        if hermitian_error > 1e-11 * max(matrix_scale, 1.0):
            raise RuntimeError(
                "slab GDF gradient cache: non-Hermitian metric at "
                f"q={q}; max residual={hermitian_error:.3e}"
            )
        M_eigh = 0.5 * (M_eigh + M_eigh.conj().T)
        eigvals, U = np.linalg.eigh(M_eigh)
        max_abs_eigenvalue = (
            float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        )
        if max_abs_eigenvalue <= threshold:
            raise RuntimeError(
                "slab GDF gradient cache: metric has no eigenvalue "
                f"outside the linear dependency threshold at q={q}"
            )
        keep = np.abs(eigvals) > threshold
        eig_scale = max(max_abs_eigenvalue, 1.0)
        spectral_tol = 128.0 * np.finfo(np.float64).eps * eig_scale
        if np.any(np.abs(np.abs(eigvals) - threshold) <= spectral_tol):
            raise RuntimeError(
                "slab GDF gradient cache: a metric eigenvalue lies on "
                f"the |λ| linear-dependence threshold at q={q}; the fit "
                "derivative is undefined"
            )
        inverse_eigvals, inverse_frechet = _signed_truncated_inverse_frechet(
            eigvals,
            keep,
            spectral_tol=spectral_tol,
            where=f"slab GDF gradient cache (q={q})",
        )

        groups[q_key] = _SlabGdfQGroup(
            q=q,
            pairs=group_pairs,
            bra_k_indices=bra_indices,
            M=M_eigh,
            T_list=T_list,
            eigvals=eigvals,
            eigvecs=U,
            keep_mask=keep,
            signs=np.sign(eigvals[keep]).astype(np.int8),
            inverse_eigvals=inverse_eigvals,
            inverse_frechet=inverse_frechet,
            n_fit=int(np.count_nonzero(keep)),
            g_mesh=g_mesh,
            kernel_weights=kernel_weights,
        )

    return _SlabGdfGradientCache(
        aux_basis=aux_modrho,
        k_cart_list=kpts,
        groups=groups,
        linear_dep_thr=threshold,
        ke_cutoff=float(ke_cutoff),
        lat_opts=lat_opts,
        n_aux=n_aux,
        n_orb=n_orb,
    )


def _slab_gdf_q0_group(cache: _SlabGdfGradientCache) -> _SlabGdfQGroup:
    """Return the cache's q = 0 group (the diagonal fits J consumes),
    failing loudly on exotic meshes with off-diagonal q = 0 pairs."""
    for group in cache.groups.values():
        if float(np.linalg.norm(group.q)) < 1e-12:
            if any(i != j for i, j in group.pairs):
                raise NotImplementedError(
                    "slab GDF J gradient: the q=0 group contains "
                    "off-diagonal (k_i, k_j) pairs (coincident k-points "
                    "in k_cart_list?); the diagonal-only J weight "
                    "algebra does not cover that."
                )
            return group
    raise RuntimeError(
        "slab GDF J gradient: cache has no q=0 group; J consumes the "
        "diagonal fit blocks and cannot proceed."
    )


def _compute_j_gradient_slab_gdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    weights,
    cache: _SlabGdfGradientCache,
) -> np.ndarray:
    """Slab KRHF DF-J fit-derivative gradient (q = 0 diagonal fits).

    Differentiates, at fixed density, the slab driver's Coulomb energy
    (``_run_krhf_periodic_slab_gdf``'s ``build_j`` closure +
    ``E_coulomb = 0.5 Σ_i w_i Re Tr[D(k_i) J(k_i)]``): with the signed
    factors ``L(k) = |λ|^{-1/2} U_keep^H T(k)`` and the signature
    ``s_L`` applied in the rho contraction (no conjugation anywhere on
    L — the driver mirrors the PySCF J convention), the double
    contraction collapses onto the signed truncated inverse
    ``H = U_keep Λ_keep^{-1} U_keep^H`` of the q = 0 metric
    (:func:`_signed_truncated_inverse_frechet`):

        E_J = 0.5 Re[ t^T conj(H) t ],
        t_P = Σ_k w_k Σ_{ls} T(k)_{P,l,s} D(k)_{s,l}.

    ``conj(H)`` (not H) because both t factors enter UNconjugated; the
    expression is invariant under per-column eigenvector phases, which
    matters here because the slab factorizer eigh-decomposes in
    complex128 even at q = 0 (zheevd phases are arbitrary), so unlike
    the bulk multi-k J the algebra must not assume real U. With
    ``t_eig = U^H t`` and ``s_eig = U^T t``:

    * 3c: ``dE = Re[ γ^T dt ]`` with ``γ = H t = U (h ∘ t_eig)`` —
      per diagonal fit the weight ``W3(k)[P,l,s] = w_k γ_P D(k)[s,l]``
      dispatched to the Bloch 3c kernel with the SLAB mesh + kernel
      weights (finite zero mode included; it contributes through the
      pair-centre piece only).
    * 2c: ``dE = Re Σ_PQ Ω_PQ dM_PQ`` with

          Ω = 0.5 conj( U (h^[1] ∘ outer(t_eig, s_eig)) U^H )

      (derivation: write ``Re[t^T conj(dH) t] = Re[ū^T dH ū]`` with
      ``ū = conj(t)``, insert the Daleckii-Krein ``dH``, and read off
      the coefficient of ``dM_PQ``; for real U this reduces to the
      bulk J weight). Exact against the SCF's real-projected q = 0
      metric: the slab q = 0 mesh is inversion symmetric, so
      ``Im M ≡ 0`` identically as a function of geometry and
      contracting the raw complex accumulation under ``Re[...]`` loses
      nothing.

    Pure fit derivative: NO one-electron/W terms, NO exchange, NO
    probe-charge (xi) shift — those are the composition rung's terms.
    """
    from .aux_basis import (
        _rsgdf_weighted_2c_metric_gradient_bloch,
        _rsgdf_weighted_3c_tensor_gradient_bloch,
    )

    n_aux, n_orb = cache.n_aux, cache.n_orb
    kpts = np.asarray(cache.k_cart_list, dtype=float).reshape(-1, 3)
    n_k = int(kpts.shape[0])
    if len(D_k_list) != n_k:
        raise ValueError(
            f"slab GDF J gradient: {len(D_k_list)} density blocks for "
            f"{n_k} cached k-points."
        )
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n_k:
        raise ValueError(
            f"slab GDF J gradient: {w.shape[0]} k-weights for "
            f"{n_k} cached k-points."
        )
    group = _slab_gdf_q0_group(cache)
    if sorted(group.bra_k_indices) != list(range(n_k)):
        raise RuntimeError(
            "slab GDF J gradient: q=0 group does not cover every "
            f"k-point (bras {group.bra_k_indices} of {n_k})."
        )
    U = np.asarray(group.eigvecs)

    # t_P = Σ_k w_k Σ_ls T(k)[P,l,s] D(k)[s,l]  (flat: T @ D^T.ravel).
    t = np.zeros(n_aux, dtype=np.complex128)
    D_T_flat = {}
    for pair_pos, k_idx in enumerate(group.bra_k_indices):
        T_flat = group.T_list[pair_pos].reshape(n_aux, n_orb * n_orb)
        Dt = np.ascontiguousarray(
            np.asarray(D_k_list[k_idx], dtype=np.complex128).T
        ).reshape(-1)
        D_T_flat[k_idx] = Dt
        t += w[k_idx] * (T_flat @ Dt)

    t_eig = U.conj().T @ t
    s_eig = U.T @ t
    gamma = U @ (group.inverse_eigvals * t_eig)

    omega = 0.5 * np.conj(
        U @ (group.inverse_frechet * np.outer(t_eig, s_eig)) @ U.conj().T
    )

    grad = _rsgdf_weighted_2c_metric_gradient_bloch(
        cache.aux_basis,
        system,
        ke_cutoff=cache.ke_cutoff,
        q_cart=group.q,
        weight=omega,
        g_mesh=group.g_mesh,
        kernel_weights=group.kernel_weights,
    )
    for _pair_pos, k_idx in enumerate(group.bra_k_indices):
        W3 = (w[k_idx] * np.outer(gamma, D_T_flat[k_idx])).reshape(
            n_aux, n_orb, n_orb
        )
        grad = grad + _rsgdf_weighted_3c_tensor_gradient_bloch(
            cache.aux_basis,
            ao_basis,
            system,
            ke_cutoff=cache.ke_cutoff,
            weight=W3,
            k_ket=kpts[k_idx] + group.q,
            q_cart=group.q,
            lat_opts=cache.lat_opts,
            g_mesh=group.g_mesh,
            kernel_weights=group.kernel_weights,
        )
    return grad


def _compute_k_gradient_slab_gdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    weights,
    cache: _SlabGdfGradientCache,
) -> np.ndarray:
    """Slab KRHF DF-K fit-derivative gradient over ALL q groups.

    Differentiates, at fixed density, the FITTED part of the slab
    driver's exchange (``_contract_slab_gdf_multik_exchange`` minus its
    ``xi·S D S`` probe-charge shift): per fit block,
    ``K(k_i) += w_j Σ_L s_L L_L D(k_j) L_L^H``, and

        E_K^fit = -0.25 Σ_i w_i Re Tr[D(k_i) K^fit(k_i)]
                = -0.25 Σ_q Re Tr[ G(q) H(q) ],
        G(q)_PQ = Σ_{(i,j) in q} w_i w_j
                  Tr[ T(q,k_i)_P D(k_j) T(q,k_i)_Q^H D(k_i) ]

    — exactly the bulk multi-k K structure
    (:func:`_compute_k_gradient_multik_rsgdf`, see its docstring for
    the weight-algebra derivation, Hermitization-exactness arguments,
    and the conjugate-pair collapse), with three slab substitutions and
    nothing else:

    * ``H(q)`` is the SIGNED truncated inverse of the indefinite
      Sundararaman-Arias metric and ``h^[1]`` its signed
      divided-difference table (:func:`_signed_truncated_inverse_
      frechet`) — the formulas are unchanged because ``sign/|λ| =
      1/λ``; only the keep criterion (``|λ| > thr``) and the sign
      pattern of the products differ;
    * the 2c/3c dispatches carry the slab mesh + ``K(G+q)/V`` kernel
      weights (finite zero mode; 2c receives no zero-mode term, the
      3c pair-centre piece does);
    * there is no high-|G| tail (the slab builder has none).

    The probe-charge shift is NOT differentiated here: its constant
    ``xi`` is a pure lattice functional (``_slab_probe_charge_
    madelung_for_kmesh`` consumes ``system.lattice`` only, so
    ``dxi/dR = 0`` exactly) and its ``S D S`` overlap response belongs
    to the composition rung's one-electron/W assembly, mirroring the
    bulk ``_compute_exxdiv_w_gradient_multik`` split. Pure KRHF
    exchange: the ``-0.25`` SCF prefactor is baked in; hybrid
    ``alpha_hf`` scaling is the caller's concern.
    """
    from .aux_basis import (
        _rsgdf_weighted_2c_metric_gradient_bloch,
        _rsgdf_weighted_3c_tensor_gradient_bloch,
    )

    n_aux, n_orb = cache.n_aux, cache.n_orb
    n_atoms = len(system.unit_cell)
    kpts = np.asarray(cache.k_cart_list, dtype=float).reshape(-1, 3)
    n_k = int(kpts.shape[0])
    if len(D_k_list) != n_k:
        raise ValueError(
            f"slab GDF K gradient: {len(D_k_list)} density blocks for "
            f"{n_k} cached k-points."
        )
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n_k:
        raise ValueError(
            f"slab GDF K gradient: {w.shape[0]} k-weights for "
            f"{n_k} cached k-points."
        )
    # Hermitize on entry — exact wrt the SCF's Re[...] (bulk docstring).
    D_h = [
        0.5
        * (
            np.asarray(D, dtype=np.complex128)
            + np.asarray(D, dtype=np.complex128).conj().T
        )
        for D in D_k_list
    ]

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for q_key in sorted(cache.groups):
        group = cache.groups[q_key]
        U = np.asarray(group.eigvecs)

        # ---- G(q): per-pair exchange Gram in the aux frame --------
        G_tot = np.zeros((n_aux, n_aux), dtype=np.complex128)
        for pair_pos, (i, j) in enumerate(group.pairs):
            T = group.T_list[pair_pos]
            TDj = np.matmul(T, D_h[j]).reshape(n_aux, n_orb * n_orb)
            DiT = np.matmul(D_h[i], T).reshape(n_aux, n_orb * n_orb)
            G_tot += (w[i] * w[j]) * (TDj @ DiT.conj().T)
        G_tot = 0.5 * (G_tot + G_tot.conj().T)

        # ---- 2c: Ω_K(q) = -0.25 conj(U (h^[1] ∘ Ĝ) U^H) -----------
        G_hat = U.conj().T @ G_tot @ U
        omega_k = -0.25 * np.conj(
            U @ (group.inverse_frechet * G_hat) @ U.conj().T
        )
        grad += _rsgdf_weighted_2c_metric_gradient_bloch(
            cache.aux_basis,
            system,
            ke_cutoff=cache.ke_cutoff,
            q_cart=group.q,
            weight=omega_k,
            g_mesh=group.g_mesh,
            kernel_weights=group.kernel_weights,
        )

        # ---- 3c: one conjugate-pair-collapsed dispatch per pair ----
        for pair_pos, (i, j) in enumerate(group.pairs):
            T_flat = group.T_list[pair_pos].reshape(n_aux, n_orb * n_orb)
            eta = (
                U @ (group.inverse_eigvals[:, None] * (U.conj().T @ T_flat))
            ).reshape(n_aux, n_orb, n_orb)
            W3 = (
                -0.5
                * (w[i] * w[j])
                * np.conj(np.matmul(D_h[i], np.matmul(eta, D_h[j])))
            )
            grad += _rsgdf_weighted_3c_tensor_gradient_bloch(
                cache.aux_basis,
                ao_basis,
                system,
                ke_cutoff=cache.ke_cutoff,
                weight=W3,
                k_ket=kpts[i] + group.q,
                q_cart=group.q,
                lat_opts=cache.lat_opts,
                g_mesh=group.g_mesh,
                kernel_weights=group.kernel_weights,
            )
    return grad


def _fold_hermitian_k_matrices_to_lattice(
    M_k_list,
    weights,
    k_cart_list: np.ndarray,
    template,
):
    """Fold per-k Hermitian matrices onto a real-space lattice set:

        M(g) = S_k w_k Re[ exp(-i k.g) M(k) ]

    (the :func:`vibeqc.periodic_gradient_multi_k._bloch_fold_w_per_k`
    convention). The k-blind lattice gradient kernels
    (``overlap/kinetic/nuclear_erfc_lattice_gradient_contribution``)
    contract ELEMENTWISE, S_g S_ij M(g)_ij dX(g)_ij, and the exact
    elementwise weight for S_k w_k Re Tr[M(k) X(k)] with
    X(k) = S_g exp(+ik.g) X(g) is S_k w_k Re[exp(+ik.g) M(k)^T].
    For Hermitian M(k) (M^T = conj(M)) that is identical to the fold
    above, with no cell-list inversion-symmetry assumption.

    ``template`` (a freshly built ``compute_overlap_lattice`` result on
    the SAME LatticeSumOptions the energy term uses) is consumed: its
    blocks are overwritten and the set is returned.
    """
    n_cells = len(template.cells)
    nbf = int(template.nbf)
    blocks = [np.zeros((nbf, nbf), dtype=np.float64) for _ in range(n_cells)]
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    kpts = np.asarray(k_cart_list, dtype=float).reshape(-1, 3)
    for ik in range(kpts.shape[0]):
        M = np.asarray(M_k_list[ik], dtype=np.complex128)
        w_k = float(w_arr[ik])
        for c, cell in enumerate(template.cells):
            phase = float(np.dot(kpts[ik], np.asarray(cell.r_cart)))
            blocks[c] += w_k * (
                np.cos(phase) * M.real + np.sin(phase) * M.imag
            )
    for c in range(n_cells):
        template.set_block(c, blocks[c])
    return template


def _compute_kinetic_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    weights,
    k_cart_list: np.ndarray,
    oneel_lat_opts: LatticeSumOptions,
) -> np.ndarray:
    """d/dR of the multi-k kinetic energy S_k w_k Re Tr[D(k) T(k)]
    at fixed D(k), where T(k) = bloch_sum(T_lat, k) on the SCF's
    one-electron lattice options (``_oneel_lattice_opts`` output —
    NOT the base/gauge opts; S/T may use a grown cutoff)."""
    D_set = _fold_hermitian_k_matrices_to_lattice(
        D_k_list,
        weights,
        k_cart_list,
        compute_overlap_lattice(ao_basis, system, oneel_lat_opts),
    )
    return np.asarray(
        kinetic_lattice_gradient_contribution(
            ao_basis, system, D_set, oneel_lat_opts
        ),
        dtype=np.float64,
    )


def _compute_overlap_w_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    W_k_list,
    weights,
    k_cart_list: np.ndarray,
    oneel_lat_opts: LatticeSumOptions,
) -> np.ndarray:
    """Overlap-Lagrangian term -S_k w_k Re Tr[W(k) dS(k)/dR] with
    W(k) = 2 C_occ(k) diag(eps_occ(k)) C_occ(k)^H (the
    ``_bloch_fold_w_per_k`` closed-shell convention; caller builds
    W(k), so smearing/per-spin variants stay caller-side). S(k) is the
    SCF's bloch_sum(S_lat, k) on the one-electron lattice options."""
    W_set = _fold_hermitian_k_matrices_to_lattice(
        W_k_list,
        weights,
        k_cart_list,
        compute_overlap_lattice(ao_basis, system, oneel_lat_opts),
    )
    return np.asarray(
        overlap_lattice_gradient_contribution(
            ao_basis, system, W_set, oneel_lat_opts
        ),
        dtype=np.float64,
    )


def _compute_exxdiv_w_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    alpha_hf: float,
    madelung: float,
    oneel_lat_opts: LatticeSumOptions,
) -> np.ndarray:
    """Fixed-density gradient of the multi-k exxdiv='ewald' K-shift
    energy (only meaningful when the SCF ran exact exchange,
    ``alpha_hf > 0`` with unscreened K — rung-4 composition; unit-gated
    here so rung 4 only has to wire it).

    The SCF shifts K(k) -> K(k) + xi S(k) D(k) S(k)
    (:func:`vibeqc.madelung.apply_exxdiv_ewald_to_K`), entering the
    energy as

        E_shift = -1/4 a S_k w_k xi Re Tr[D(k) S(k) D(k) S(k)].

    xi = xi_BvK from ``_madelung_for_kmesh`` depends on the lattice
    geometry only (``madelung_constant_for_cell`` ignores atom
    positions), so d xi/dR = 0 and the fixed-D derivative is purely the
    overlap response:

        dE_shift/dR = -1/2 a xi S_k w_k Re Tr[D(k) S(k) D(k) dS(k)/dR]

    i.e. the Gamma assembly's W_exx = 1/2 a xi D S D generalized per k
    (compute_gdf_gradient_rhf_gamma, exxdiv block) and folded against
    dS with the k-blind overlap kernel (which returns -Tr[W dS])."""
    xi = float(madelung)
    a = float(alpha_hf)
    if a == 0.0 or xi == 0.0:
        return np.zeros((len(system.unit_cell), 3), dtype=np.float64)
    W_exx_k = [
        0.5
        * a
        * xi
        * (
            np.asarray(D, dtype=np.complex128)
            @ np.asarray(S, dtype=np.complex128)
            @ np.asarray(D, dtype=np.complex128)
        )
        for D, S in zip(D_k_list, S_k_list)
    ]
    W_set = _fold_hermitian_k_matrices_to_lattice(
        W_exx_k,
        weights,
        k_cart_list,
        compute_overlap_lattice(ao_basis, system, oneel_lat_opts),
    )
    return np.asarray(
        overlap_lattice_gradient_contribution(
            ao_basis, system, W_set, oneel_lat_opts
        ),
        dtype=np.float64,
    )


def _compute_v_ne_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    weights,
    k_cart_list: np.ndarray,
    gauge_lat_opts: LatticeSumOptions,
    *,
    ke_cutoff: Optional[float] = None,
    ewald_options=None,
) -> np.ndarray:
    """Fixed-density gradient of the multi-k Ewald-gauge V_ne energy
    S_k w_k Re Tr[D(k) V(k)], V(k) = bloch_sum(V_lat, k).

    Differentiates EXACTLY the split lattice build the multi-k GDF SCF
    uses (``compute_nuclear_lattice_dispatch`` under the forced
    EWALD_3D gauge -> :func:`vibeqc.periodic_v_ne.
    compute_v_ne_ewald_3d_ft_lattice`): per cell g,

        V(g) = V_short(g) + V_long(g) + corr(g)

        V_short(g): libint erfc(alpha r)/r nuclear attraction lattice
                    sum (real-space, nuclear_cutoff point charges);
        V_long(g)_ab  = (1/O) s_ab S_{G!=0} v_long(G) conj(FT_ab(G; g)),
          v_long(G) = -S_I Z_I (4pi/G^2) exp(-G^2/4a^2) exp(-iG.R_I);
        corr(g)   = -v_short(G=0) S(g),
          v_short(G=0) = -pi S_I Z_I / (a^2 O)   (PySCF get_nuc gauge).

    Derivative strategy per piece:

    * V_short — fold D(k) to the real-space density and contract with
      the k-blind C++ erfc lattice gradient kernel
      (``nuclear_erfc_lattice_gradient_contribution``: AO-centre +
      nuclear point-charge derivatives, elementwise contraction).
    * corr — overlap kernel with weight v_short(G=0) . D_fold (the
      kernel returns -Tr[W dS], and the value term is
      -v_short(G=0) Tr[D S]), matching the Gamma route.
    * V_long — reciprocal per-k form. Using the Bloch pair-FT
      convention FT^{(+k')}(G) = S_g exp(+ik'.g) FT(G; g) and the
      full-mesh reality of the per-cell G sum (G <-> -G conjugate
      pairs),

        Tr[D(k) V_long(k)]
          = (1/O) S_G v_long(G) S_ab (D(k)^T)_ab s_ab
                                     conj( FT^{(-k)}_ab(G) )

      so the AO-side derivative is the rung-1 Bloch gweighted kernel
      (:func:`vibeqc._aopair_ft.
      ao_pair_fourier_transform_bloch_gradient_gweighted`) at
      k_cart = -k with complex weights
      Q_ab(G) = (w_k/O) v_long(G) (D(k)^T)_ab s_ab, and the
      nuclear-side term contracts d v_long(G)/dR_I (the structure-
      factor phase derivative) against the density-contracted pair FT.
      The identity was pinned numerically at 2e-13 on the H2 (2,1,1)
      anchor before this landed.

    Deliberate value-vs-derivative seam (documented, sub-gate): the
    value builder zeroes V_long(g) for cells whose overlap norm is
    below ``screen_rel = 1e-12`` of the largest block; this derivative
    keeps every cell (like the Gamma gradient). The induced FD
    discrepancy is bounded by that relative screen and sits orders of
    magnitude below the 1e-7 gates.

    ``ke_cutoff = None`` resolves like the dispatch: env
    ``VIBEQC_VNE_EWALD3D_KE``, else 200.0. ``ewald_options = None``
    mirrors the dispatch default (real_cutoff = nuclear cutoff,
    alpha <= 0 -> 2.0). Fails loud if the legacy grid backend is
    forced via env (this differentiates the analytic-FT build only).
    """
    import os

    from ._aopair_ft import (
        ao_pair_fourier_transform_bloch,
        ao_pair_fourier_transform_bloch_gradient_gweighted,
    )
    from ._vibeqc_core import EwaldOptions, direct_lattice_cells
    from .aux_basis import _ao_scales_for_rsgdf, rsgdf_dense_g_mesh
    from .periodic_v_ne import _vne_ft_g_chunk_size

    if int(system.dim) != 3:
        raise NotImplementedError(
            "_compute_v_ne_gradient_multik: EWALD_3D V_ne requires "
            f"dim == 3 (got dim = {int(system.dim)})."
        )
    if gauge_lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        raise ValueError(
            "_compute_v_ne_gradient_multik: gauge_lat_opts must carry "
            "the forced EWALD_3D gauge the multi-k SCF uses for V_ne."
        )
    backend = os.environ.get(
        "VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft"
    ).lower()
    if backend != "analytic_ft":
        raise NotImplementedError(
            "_compute_v_ne_gradient_multik: differentiates the "
            "analytic-FT V_ne build only; the legacy grid backend "
            f"(VIBEQC_VNE_EWALD3D_BACKEND={backend!r}) has no matching "
            "derivative."
        )
    if ke_cutoff is None:
        ke_cutoff = float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0"))
    ke = float(ke_cutoff)

    if ewald_options is None:
        ewald_options = EwaldOptions()
        ewald_options.real_cutoff_bohr = gauge_lat_opts.nuclear_cutoff_bohr
    alpha = float(ewald_options.alpha)
    if alpha <= 0.0:
        alpha = 2.0

    from ._vibeqc_core import nuclear_erfc_lattice_gradient_contribution

    kpts = np.asarray(k_cart_list, dtype=float).reshape(-1, 3)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    n_atoms = len(system.unit_cell)
    n_orb = int(ao_basis.nbasis)
    cell_volume = float(
        abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
    )

    # ---- V_short: erfc real-space lattice sum on the folded density.
    D_fold = _fold_hermitian_k_matrices_to_lattice(
        D_k_list,
        w_arr,
        kpts,
        compute_overlap_lattice(ao_basis, system, gauge_lat_opts),
    )
    gradient = np.asarray(
        nuclear_erfc_lattice_gradient_contribution(
            ao_basis, system, D_fold, gauge_lat_opts, alpha
        ),
        dtype=np.float64,
    )

    nuclei_z = np.array(
        [atom.Z for atom in system.unit_cell], dtype=np.float64
    )
    nuclei_r = np.array(
        [list(atom.xyz) for atom in system.unit_cell], dtype=np.float64
    )

    # ---- G = 0 correction: value adds -v_short(G=0) S(k); the overlap
    # kernel returns -Tr[W dS], so the weight is +v_short(G=0) D_fold.
    total_nuclear_charge = float(nuclei_z.sum())
    if abs(total_nuclear_charge) > 1.0e-12:
        v_short_g0 = (
            -(np.pi / (alpha**2 * cell_volume)) * total_nuclear_charge
        )
        g0_weight_set = _fold_hermitian_k_matrices_to_lattice(
            [v_short_g0 * np.asarray(D) for D in D_k_list],
            w_arr,
            kpts,
            compute_overlap_lattice(ao_basis, system, gauge_lat_opts),
        )
        gradient += np.asarray(
            overlap_lattice_gradient_contribution(
                ao_basis, system, g0_weight_set, gauge_lat_opts
            ),
            dtype=np.float64,
        )

    # ---- V_long: reciprocal form, same mesh policy as the value
    # builder (dense |G| <= sqrt(2 ke) sphere, G = 0 dropped, Gaussian
    # damping filter at 1e-14).
    G_all = rsgdf_dense_g_mesh(system, ke)
    G2_all = np.einsum("gx,gx->g", G_all, G_all)
    nonzero = G2_all > 0.0
    G = np.asarray(G_all[nonzero], dtype=np.float64)
    G2 = np.asarray(G2_all[nonzero], dtype=np.float64)
    damping = np.exp(-G2 / (4.0 * alpha**2))
    significant = damping > 1.0e-14
    G = G[significant]
    G2 = G2[significant]
    damping = damping[significant]

    phases = np.exp(-1j * (nuclei_r @ G.T))          # (n_atoms, n_G)
    kernel = damping * (4.0 * np.pi / G2)            # (n_G,)
    v_long = -np.einsum(
        "a,ag,g->g", nuclei_z, phases, kernel, optimize=True
    )

    cells = direct_lattice_cells(system, float(gauge_lat_opts.cutoff_bohr))
    cell_vectors = np.array(
        [list(cell.r_cart) for cell in cells], dtype=np.float64
    )
    if cell_vectors.size == 0:
        cell_vectors = np.zeros((1, 3), dtype=np.float64)

    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    g_chunk = _vne_ft_g_chunk_size(n_orb)

    for ik in range(kpts.shape[0]):
        w_k = float(w_arr[ik])
        # Elementwise reciprocal weight (D(k)^T)_ab s_ab — for the
        # Hermitian SCF density D^T = conj(D).
        D_weight = (
            np.asarray(D_k_list[ik], dtype=np.complex128).T * pair_scales
        )
        minus_k = -kpts[ik]
        for start in range(0, len(G), g_chunk):
            stop = min(start + g_chunk, len(G))
            G_chunk = G[start:stop]
            kernel_chunk = kernel[start:stop]
            phases_chunk = phases[:, start:stop]
            v_long_chunk = v_long[start:stop]

            # Nuclear structure-factor derivative:
            #   d v_long(G)/dR_A = -Z_A kernel(G) (-iG) exp(-iG.R_A),
            # contracted with the density-weighted Bloch pair FT at -k.
            pair_ft = ao_pair_fourier_transform_bloch(
                ao_basis, G_chunk, cell_vectors, k_cart=minus_k
            )
            density_pair_ft = np.einsum(
                "ab,abg->g", D_weight, pair_ft.conj(), optimize=True
            )
            del pair_ft
            d_structure = (
                -nuclei_z[:, None, None]
                * kernel_chunk[None, :, None]
                * phases_chunk[:, :, None]
                * (-1j * G_chunk[None, :, :])
            )
            gradient += (
                w_k
                * np.real(
                    np.einsum(
                        "agx,g->ax",
                        d_structure,
                        density_pair_ft,
                        optimize=True,
                    )
                )
                / cell_volume
            )

            # AO-side pair-FT centre derivative (rung-1 Bloch kernel):
            # d/dR Re S_G S_ab Q_ab(G) conj(FT^{(-k)}_ab(G)).
            Q = (
                (w_k / cell_volume)
                * D_weight[:, :, None]
                * v_long_chunk[None, None, :]
            )
            gradient += ao_pair_fourier_transform_bloch_gradient_gweighted(
                ao_basis,
                G_chunk,
                cell_vectors,
                minus_k,
                Q,
                n_atoms,
            )

    return gradient


def _compute_oneel_w_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    W_k_list,
    weights,
    k_cart_list: np.ndarray,
    *,
    oneel_lat_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    v_ne_ke_cutoff: Optional[float] = None,
    v_ne_ewald_options=None,
    ecp_context=None,
    ecp_lat_opts=None,
) -> np.ndarray:
    """Multi-k GDF KRHF one-electron + overlap-Lagrangian + nuclear
    gradient assembly (G-PBC-002 Item-4 rung 3b) at fixed D(k)/W(k):

        grad = d e_nuc/dR                     (converged Ewald pair sum)
             + S_k w_k Re Tr[D(k) dT(k)/dR]   (kinetic Pulay)
             - S_k w_k Re Tr[W(k) dS(k)/dR]   (overlap Lagrangian)
             + S_k w_k Re Tr[D(k) dV(k)/dR]   (split Ewald-gauge V_ne)

    mirroring the Gamma assembly's term structure
    (:func:`compute_gdf_gradient_rhf_gamma`) so the rung-4 full
    composition only adds the DF-J/K fit terms
    (:func:`_compute_j_gradient_multik_rsgdf` + the K rung) and the
    exxdiv shift (:func:`_compute_exxdiv_w_gradient_multik`).

    ``oneel_lat_opts`` MUST be the SCF's ``_oneel_lattice_opts`` output
    (S/T lattice sums; possibly a grown cutoff) and ``gauge_lat_opts``
    the SCF's ``_gauge_lat_opts_for_v_ne_and_e_nuc`` output (V_ne) —
    each term differentiates exactly the matrix build the multi-k SCF
    energy used (the e_nuc gauge lesson, HANDOVER_OPEN_BUGS_V015.md
    dense-core entry). e_nuc is the converged
    ``ewald_nuclear_repulsion``; its exact derivative partner is
    ``ewald_nuclear_repulsion_gradient``.
    """
    from ._vibeqc_core import ewald_nuclear_repulsion_gradient

    nuclear_system = system if ecp_context is None else ecp_context[4]
    grad = np.asarray(
        ewald_nuclear_repulsion_gradient(nuclear_system), dtype=np.float64
    )
    grad += _compute_kinetic_gradient_multik(
        system, ao_basis, D_k_list, weights, k_cart_list, oneel_lat_opts
    )
    grad += _compute_overlap_w_gradient_multik(
        system, ao_basis, W_k_list, weights, k_cart_list, oneel_lat_opts
    )
    grad += _compute_v_ne_gradient_multik(
        nuclear_system,
        ao_basis,
        D_k_list,
        weights,
        k_cart_list,
        gauge_lat_opts,
        ke_cutoff=v_ne_ke_cutoff,
        ewald_options=v_ne_ewald_options,
    )
    if ecp_context is not None:
        from ._vibeqc_core import ecp_lattice_gradient_contribution_from_primitives

        if ecp_lat_opts is None:
            raise ValueError("Periodic ECP force requires the SCF projector image options")
        density = _fold_hermitian_k_matrices_to_lattice(
            D_k_list, weights, k_cart_list,
            compute_overlap_lattice(ao_basis, system, ecp_lat_opts),
        )
        blocks, centers, *_ = ecp_context
        grad += np.asarray(ecp_lattice_gradient_contribution_from_primitives(
            ao_basis, system, density, ecp_lat_opts, centers, blocks,
        ))
    return grad


def _compute_krhf_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    W_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    cache: _MultikRsgdfGradientCache,
    *,
    alpha_hf: float,
    madelung: float,
    oneel_lat_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    v_ne_ke_cutoff: Optional[float] = None,
    v_ne_ewald_options=None,
    ecp_context=None,
    ecp_lat_opts=None,
) -> np.ndarray:
    """Full multi-k KRHF GDF gradient assembly at fixed D(k)/W(k).

    G-PBC-002 Item-4 rung-4 composition — every term the multi-k SCF
    total energy differentiates, each through its FD-gated helper:

        grad = oneel/W/nn        (:func:`_compute_oneel_w_gradient_multik`)
             + DF-J fit          (:func:`_compute_j_gradient_multik_rsgdf`)
             + alpha · DF-K fit  (:func:`_compute_k_gradient_multik_rsgdf`;
                                  the helper carries the SCF's -0.25,
                                  alpha scales it to E = -0.25 alpha ...)
             + exxdiv W-shift    (:func:`_compute_exxdiv_w_gradient_multik`)

    Private on purpose: the composition gate
    (``tests/test_periodic_gdf_gradient.py``,
    ``test_multik_krhf_full_scf_gradient_vs_fd``) consumes it directly;
    the public route is ``run_krhf_periodic_gdf(compute_gradient=True)``
    (Item-4 rung 6, landed 2026-07-30), whose entry guards
    (``periodic_k_gdf._reject_unsupported_multik_gradient``) pin the
    supported envelope and whose converged-state block hands the SCF's
    exact provenance here.
    ``W_k_list`` is the closed-shell energy-weighted density
    ``2 C_occ diag(eps_occ) C_occ^H`` per k -- or, for a Fermi-Dirac
    smeared SCF, the fractional-occupation
    ``sum_i f_i(k) eps_i(k) c_i(k) c_i(k)^H`` with ``D_k_list`` the
    matching fractional densities (the assembly is occupation-agnostic
    at fixed D/W; the Mermin free-energy force theorem makes the
    occupation/mu responses vanish -- see the driver's gradient
    block); ``S_k_list`` the SCF's
    overlap blocks; ``madelung`` the SCF's k-mesh-aware BvK ``ξ``
    (``_madelung_for_kmesh``). ``oneel_lat_opts`` /
    ``gauge_lat_opts`` MUST be the SCF's own resolved options (the
    e_nuc gauge lesson — see :func:`_compute_oneel_w_gradient_multik`).

    An explicitly supplied private MDF cache uses its combined SR/LR/PW
    fitted derivative. The remaining terms use the same SCF one-electron
    options, W/S matrices and Madelung value. Public MDF gradient routing
    remains disabled pending numerical acceptance.
    """
    kpts = np.asarray(k_cart_list, dtype=float).reshape(-1, 3)
    if kpts.shape != np.asarray(cache.k_cart_list).shape or not np.allclose(
        kpts, np.asarray(cache.k_cart_list), rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "multik KRHF gradient: k_cart_list does not match the "
            "cache's k-point set; the fit terms would mix meshes."
        )
    from .periodic_mdf import _MdfCache, _compute_mdf_mean_field_fit_gradient

    mdf_gradient = None
    if isinstance(cache, _MdfCache):
        # Validate/admit the private source before one-electron work. This
        # path is reachable only with an explicitly built private cache.
        mdf_gradient = _compute_mdf_mean_field_fit_gradient(
            cache, system, ao_basis, (D_k_list,), weights, kpts, alpha_hf=float(alpha_hf),
        )
    grad = _compute_oneel_w_gradient_multik(
        system,
        ao_basis,
        D_k_list,
        W_k_list,
        weights,
        kpts,
        oneel_lat_opts=oneel_lat_opts,
        gauge_lat_opts=gauge_lat_opts,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
        v_ne_ewald_options=v_ne_ewald_options,
        ecp_context=ecp_context,
        ecp_lat_opts=ecp_lat_opts,
    )
    if mdf_gradient is not None:
        grad += mdf_gradient
    else:
        grad += _compute_j_gradient_multik_rsgdf(
            system, ao_basis, D_k_list, weights, cache
        )
        if float(alpha_hf) != 0.0:
            grad += float(alpha_hf) * _compute_k_gradient_multik_rsgdf(
                system, ao_basis, D_k_list, weights, cache
            )
    if float(alpha_hf) != 0.0:
        grad += _compute_exxdiv_w_gradient_multik(
            system,
            ao_basis,
            D_k_list,
            S_k_list,
            weights,
            kpts,
            float(alpha_hf),
            float(madelung),
            oneel_lat_opts,
        )
    return grad


def _compute_kuhf_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_alpha_k_list,
    D_beta_k_list,
    W_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    cache: _MultikRsgdfGradientCache,
    *,
    alpha_hf: float,
    madelung: float,
    oneel_lat_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    v_ne_ke_cutoff: Optional[float] = None,
    v_ne_ewald_options=None,
    ecp_context=None,
    ecp_lat_opts=None,
) -> np.ndarray:
    """Full multi-k KUHF GDF gradient assembly at fixed per-spin
    D_s(k) / W(k).

    G-PBC-002 Item-4 rung-5 open-shell composition: the per-spin energy
    conventions of ``run_kuhf_periodic_gdf`` (periodic_k_gdf.py) mapped
    onto the FD-gated closed-shell helpers, mirroring the Γ ladder's
    :func:`compute_gdf_gradient_uhf_gamma` /
    :func:`compute_gdf_gradient_rsgdf_uhf_gamma` term structure. The
    SCF total energy is (a = ``alpha_hf``; UHF runs at a = 1)

        E = S_k w_k Re( Tr[D_t(k) Hcore(k)]
                        + 0.5 S_s Tr[D_s(k) F_2e_s(k)] ) + E_nn,
        F_2e_s = J[D_t] - a (K[D_s] + xi S D_s S),  D_t = D_a + D_b,

    so at fixed density, term by term:

    * **one-electron / W / nn** — the closed-shell helper on the
      spin-summed density; ``W_k_list`` is the occupation-1 open-shell
      energy-weighted density
      ``W(k) = S_s C_s,occ(k) diag(eps_s,occ(k)) C_s,occ(k)^H``
      (caller-built; NO closed-shell factor 2 — the Γ UHF convention;
      a Fermi-Dirac smeared caller builds the fractional-occupation
      ``sum_s sum_i f_s,i(k) eps_s,i(k) c c^H`` with matching
      fractional ``D_s`` — the assembly is occupation-agnostic at
      fixed D/W).
    * **DF-J** — quadratic in D_t
      (``E_J = 0.5 S_k w_k Re Tr[D_t J[D_t]]``, the same
      ``_build_j_from_lpq`` objective as KRHF), so the closed-shell
      fit derivative applies verbatim on D_t.
    * **DF-K** — ``E_K = -0.5 a S_s S_k w_k Re Tr[D_s K[D_s]]``
      (no closed-shell 1/2: occupation-1 spin orbitals), while
      :func:`_compute_k_gradient_multik_rsgdf` bakes in the SCF's
      closed-shell ``-0.25``; each spin therefore contributes
      ``2 a x`` that helper on D_s. M = 1 (D_s = D/2, helper
      quadratic in D) collapses exactly onto the KRHF assembler's
      ``a x helper(D)``, and the Γ reduction reproduces the Γ
      template's per-spin ``0.5 x kernel(alpha)`` weight.
    * **exxdiv 'ewald' shift** — the SCF applies
      ``apply_exxdiv_ewald_to_K`` PER SPIN
      (``K_s -> K_s + xi S D_s S``), so
      ``E_shift = -0.5 a xi S_s S_k w_k Re Tr[D_s S D_s S]``; the
      closed-shell helper carries ``-0.25``, so each spin contributes
      2x the helper on D_s (Γ template:
      ``W_exx = a xi S_s D_s S D_s``). ``d xi/dR = 0`` as before.

    ``alpha_hf = 0`` (pure-DFT KUKS caller) skips K and the shift; the
    KUKS caller adds the spin-polarised XC Pulay itself
    (:func:`_compute_kuks_gradient_multik`). An empty spin channel
    (n_beta = 0, D_b = 0) is skipped rather than dispatched with zero
    weights.

    A private MDF cache instead receives both density channels in one
    fitted-derivative call, including the spin factors internally. The
    separate Madelung contributions below keep the same factors of two.
    Public MDF SCF/gradient selectors remain gated.
    """
    kpts = np.asarray(k_cart_list, dtype=float).reshape(-1, 3)
    if kpts.shape != np.asarray(cache.k_cart_list).shape or not np.allclose(
        kpts, np.asarray(cache.k_cart_list), rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "multik KUHF gradient: k_cart_list does not match the "
            "cache's k-point set; the fit terms would mix meshes."
        )
    if len(D_alpha_k_list) != len(D_beta_k_list):
        raise ValueError(
            "multik KUHF gradient: alpha and beta density lists differ "
            f"in length ({len(D_alpha_k_list)} != {len(D_beta_k_list)})."
        )
    from .periodic_mdf import _MdfCache, _compute_mdf_mean_field_fit_gradient

    mdf_gradient = None
    if isinstance(cache, _MdfCache):
        # Keep the separately allocated one-electron spin sum out of the
        # fitted-derivative phase, whose own spin total is already admitted.
        mdf_gradient = _compute_mdf_mean_field_fit_gradient(
            cache, system, ao_basis, (D_alpha_k_list, D_beta_k_list), weights, kpts,
            alpha_hf=float(alpha_hf),
        )
    D_t = [
        np.asarray(Da, dtype=np.complex128)
        + np.asarray(Db, dtype=np.complex128)
        for Da, Db in zip(D_alpha_k_list, D_beta_k_list)
    ]

    grad = _compute_oneel_w_gradient_multik(
        system,
        ao_basis,
        D_t,
        W_k_list,
        weights,
        kpts,
        oneel_lat_opts=oneel_lat_opts,
        gauge_lat_opts=gauge_lat_opts,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
        v_ne_ewald_options=v_ne_ewald_options,
        ecp_context=ecp_context,
        ecp_lat_opts=ecp_lat_opts,
    )
    if mdf_gradient is not None:
        grad += mdf_gradient
    else:
        grad += _compute_j_gradient_multik_rsgdf(
            system, ao_basis, D_t, weights, cache
        )
    if float(alpha_hf) != 0.0:
        for D_s in (D_alpha_k_list, D_beta_k_list):
            if all(
                float(np.abs(np.asarray(D)).max(initial=0.0)) == 0.0
                for D in D_s
            ):
                continue  # empty spin channel (e.g. n_beta = 0)
            if not isinstance(cache, _MdfCache):
                grad += (2.0 * float(alpha_hf)) * _compute_k_gradient_multik_rsgdf(
                    system, ao_basis, D_s, weights, cache
                )
            grad += 2.0 * _compute_exxdiv_w_gradient_multik(
                system,
                ao_basis,
                D_s,
                S_k_list,
                weights,
                kpts,
                float(alpha_hf),
                float(madelung),
                oneel_lat_opts,
            )
    return grad


def _gdf_xc_grid_motion_correction(
    system, basis, densities, functional, lattice_opts, grid_options,
    use_periodic_becke, becke_image_radius_bohr, fixed_grid_gradient,
    *, density_domain, step_bohr=1e-3,
):
    """Differentiate the moving quadrature on the SCF's exact density domain.

    Only the XC energy is differenced at fixed accepted lattice density.
    Subtract the fixed-grid AO derivative already included by the caller.
    Both terms use the same shells, cell list and explicit bra-image domain.
    """
    from ._vibeqc_core import (
        Atom, Functional, build_xc_periodic, build_xc_periodic_uks,
    )
    from .bipole_gradient import (
        _build_ks_grid, _recenter_basis_on_periodic_system,
    )

    h = float(step_bohr)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("XC grid response requires a positive finite step")
    if len(densities) not in (1, 2):
        raise ValueError("XC grid response requires one or two spin densities")
    call = build_xc_periodic if len(densities) == 1 else build_xc_periodic_uks
    func = Functional(str(functional), len(densities))
    atoms = list(system.unit_cell)
    moving = np.zeros((len(atoms), 3))
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                position = list(displaced[atom].xyz)
                position[axis] += sign * h
                displaced[atom] = Atom(displaced[atom].Z, position)
                shifted = PeriodicSystem(system.dim, system.lattice, displaced)
                shifted.charge = system.charge
                shifted.multiplicity = system.multiplicity
                shifted_basis = _recenter_basis_on_periodic_system(basis, shifted)
                grid = _build_ks_grid(
                    shifted, grid_options, use_periodic_becke,
                    becke_image_radius_bohr,
                )
                energies.append(call(
                    shifted_basis, shifted, grid, func, *densities,
                    lattice_opts, density_domain,
                ).e_xc)
            moving[atom, axis] = (energies[0] - energies[1]) / (2.0 * h)
    return moving - np.asarray(fixed_grid_gradient, dtype=np.float64)


def _compute_krks_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    W_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    cache: _MultikRsgdfGradientCache,
    *,
    functional: str,
    xc_grid,
    kmesh_bloch,
    xc_lat_opts: LatticeSumOptions,
    madelung: float,
    oneel_lat_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    xc_density_cells=None,
    xc_density_domain=None,
    xc_grid_options=None,
    xc_use_periodic_becke: Optional[bool] = None,
    xc_becke_image_radius_bohr: Optional[float] = None,
    v_ne_ke_cutoff: Optional[float] = None,
    v_ne_ewald_options=None,
    ecp_context=None,
    ecp_lat_opts=None,
) -> np.ndarray:
    """Full multi-k KRKS GDF gradient assembly at fixed D(k)/W(k).

    G-PBC-002 Item-4 rung-5 closed-shell KS composition: the KRHF
    assembler with the functional's exact-exchange fraction (pure DFT
    a_x = 0 skips the K and exxdiv-shift terms; global hybrids scale
    both — exactly the SCF's ``F_2e = J - 0.5 a K^shifted``) plus the
    XC Pulay of the multi-k GDF KS energy. The SCF's E_xc
    (``_build_xc_k_from_density``) is a functional of the inverse-Bloch
    folded real-space density

        D(g) = S_k w_k Re[ e^{-i k.g} D(k) ]

    on the ``lat_opts``-cutoff cell list, evaluated by
    ``build_xc_periodic`` on the driver's quadrature. At fixed D(k) the
    fold's phases depend on lattice vectors only, so the fixed-grid
    dE_xc/dR is the lattice-summed periodic XC Pulay primitive
    ``xc_lattice_gradient_contribution`` (AO-centre terms, full LDA +
    GGA s-piece) on that folded density. When ``xc_grid_options`` is
    supplied, the assembly also adds the atom-centred quadrature motion
    correction already exercised by the multi-k BIPOLE KS gradient (G1c,
    :func:`vibeqc.periodic_gradient_multi_k.
    compute_gradient_periodic_rks_multi_k`). No new quadrature scheme is
    introduced.

    ``xc_grid`` MUST be the SCF's own quadrature (the multi-k KS driver
    builds ``build_periodic_becke_grid(system, grid_options,
    image_radius_bohr)`` when ``opts.use_periodic_becke`` — the
    PeriodicKSOptions default — else the molecular
    ``build_grid(system.unit_cell_molecule(), grid_options)``), and
    ``kmesh_bloch`` / ``xc_lat_opts`` the driver's ``kmesh_bloch`` /
    base ``lat_opts`` (its XC cell list is
    ``direct_lattice_cells(system, lat_opts.cutoff_bohr)``): the
    gradient must differentiate the energy the SCF actually converged
    (the e_nuc gauge lesson).
    """
    from ._vibeqc_core import (
        Functional,
        direct_lattice_cells,
        xc_lattice_gradient_contribution,
    )
    from .periodic_k_density import real_space_density_from_per_k_density

    func = Functional(str(functional), 1)
    if bool(getattr(func, "is_range_separated", False)):
        raise NotImplementedError(
            "_compute_krks_gradient_multik: range-separated hybrids are "
            "not supported (the multi-k GDF fitted-K route rejects them; "
            "the screened-COSX exchange path has no fit derivative)."
        )
    alpha_hf = float(func.hf_exchange_fraction)

    grad = _compute_krhf_gradient_multik(
        system,
        ao_basis,
        D_k_list,
        W_k_list,
        S_k_list,
        weights,
        k_cart_list,
        cache,
        alpha_hf=alpha_hf,
        madelung=madelung,
        oneel_lat_opts=oneel_lat_opts,
        gauge_lat_opts=gauge_lat_opts,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
        v_ne_ewald_options=v_ne_ewald_options,
        ecp_context=ecp_context,
        ecp_lat_opts=ecp_lat_opts,
    )

    from ._vibeqc_core import PeriodicXCDensityDomain

    if xc_density_domain is None:
        xc_density_domain = PeriodicXCDensityDomain.AUTO
    cells = (
        direct_lattice_cells(system, float(xc_lat_opts.cutoff_bohr))
        if xc_density_cells is None else xc_density_cells
    )
    D_real = real_space_density_from_per_k_density(
        [np.asarray(D, dtype=np.complex128) for D in D_k_list],
        kmesh_bloch,
        cells,
    )
    xc_fixed = np.asarray(
        xc_lattice_gradient_contribution(
            ao_basis, system, xc_grid, func, D_real, xc_lat_opts,
            xc_density_domain
        ),
        dtype=np.float64,
    )
    grad = grad + xc_fixed
    if xc_grid_options is not None:
        if (
            xc_use_periodic_becke is None
            or xc_becke_image_radius_bohr is None
        ):
            raise ValueError(
                "_compute_krks_gradient_multik: moving-grid XC response "
                "requires the SCF's use_periodic_becke and "
                "becke_image_radius_bohr provenance."
            )
        grad = grad + _gdf_xc_grid_motion_correction(
            system,
            ao_basis,
            [D_real],
            str(functional),
            xc_lat_opts,
            xc_grid_options,
            bool(xc_use_periodic_becke),
            float(xc_becke_image_radius_bohr),
            xc_fixed,
            density_domain=xc_density_domain,
        )
    return grad


def _compute_kuks_gradient_multik(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_alpha_k_list,
    D_beta_k_list,
    W_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    cache: _MultikRsgdfGradientCache,
    *,
    functional: str,
    xc_grid,
    kmesh_bloch,
    xc_lat_opts: LatticeSumOptions,
    madelung: float,
    oneel_lat_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    xc_density_cells=None,
    xc_density_domain=None,
    xc_grid_options=None,
    xc_use_periodic_becke: Optional[bool] = None,
    xc_becke_image_radius_bohr: Optional[float] = None,
    v_ne_ke_cutoff: Optional[float] = None,
    v_ne_ewald_options=None,
    ecp_context=None,
    ecp_lat_opts=None,
) -> np.ndarray:
    """Full multi-k KUKS GDF gradient assembly at fixed per-spin
    D_s(k) / W(k).

    G-PBC-002 Item-4 rung-5 open-shell KS composition: the KUHF
    assembler at the functional's exact-exchange fraction plus the
    spin-polarised XC Pulay. The SCF's E_xc
    (``_build_xc_k_from_density_uks``) is a functional of the
    per-spin inverse-Bloch folded densities
    ``D_s(g) = S_k w_k Re[e^{-ik.g} D_s(k)]`` on the ``lat_opts``
    cell list, so its fixed-grid, fixed-D(k) derivative is
    ``xc_lattice_gradient_contribution_uks`` on the two folded sets.
    When ``xc_grid_options`` is supplied, the assembly also adds the
    atom-centred quadrature motion correction used by the open-shell
    BIPOLE gradient. Together these are the two XC-gradient terms of
    Reine et al., J. Chem. Phys. 133, 034102 (2010), Eq. 27.
    With ``D_a = D_b = D/2`` the folded spin densities reproduce the
    closed-shell fold pointwise, so KUKS(M=1) collapses onto
    :func:`_compute_krks_gradient_multik` by construction (the
    fixed-density algebra gate). Grid/kmesh/lat-opts provenance rules
    as in :func:`_compute_krks_gradient_multik`.
    """
    from ._vibeqc_core import (
        Functional,
        direct_lattice_cells,
        xc_lattice_gradient_contribution_uks,
    )
    from .periodic_k_density import real_space_density_from_per_k_density

    # Spin-polarised functional handle (the KUHF/KUKS driver's spin=2
    # construction) for both the alpha fraction and the XC kernel.
    func = Functional(str(functional), 2)
    if bool(getattr(func, "is_range_separated", False)):
        raise NotImplementedError(
            "_compute_kuks_gradient_multik: range-separated hybrids are "
            "not supported (the multi-k GDF fitted-K route rejects them; "
            "the screened-COSX exchange path has no fit derivative)."
        )
    alpha_hf = float(func.hf_exchange_fraction)

    grad = _compute_kuhf_gradient_multik(
        system,
        ao_basis,
        D_alpha_k_list,
        D_beta_k_list,
        W_k_list,
        S_k_list,
        weights,
        k_cart_list,
        cache,
        alpha_hf=alpha_hf,
        madelung=madelung,
        oneel_lat_opts=oneel_lat_opts,
        gauge_lat_opts=gauge_lat_opts,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
        v_ne_ewald_options=v_ne_ewald_options,
        ecp_context=ecp_context,
        ecp_lat_opts=ecp_lat_opts,
    )

    from ._vibeqc_core import PeriodicXCDensityDomain

    if xc_density_domain is None:
        xc_density_domain = PeriodicXCDensityDomain.AUTO
    cells = (
        direct_lattice_cells(system, float(xc_lat_opts.cutoff_bohr))
        if xc_density_cells is None else xc_density_cells
    )
    D_a_real = real_space_density_from_per_k_density(
        [np.asarray(D, dtype=np.complex128) for D in D_alpha_k_list],
        kmesh_bloch,
        cells,
    )
    D_b_real = real_space_density_from_per_k_density(
        [np.asarray(D, dtype=np.complex128) for D in D_beta_k_list],
        kmesh_bloch,
        cells,
    )
    xc_fixed = np.asarray(
        xc_lattice_gradient_contribution_uks(
            ao_basis, system, xc_grid, func, D_a_real, D_b_real, xc_lat_opts,
            xc_density_domain
        ),
        dtype=np.float64,
    )
    grad = grad + xc_fixed
    if xc_grid_options is not None:
        if (
            xc_use_periodic_becke is None
            or xc_becke_image_radius_bohr is None
        ):
            raise ValueError(
                "_compute_kuks_gradient_multik: moving-grid XC response "
                "requires the SCF's use_periodic_becke and "
                "becke_image_radius_bohr provenance."
            )
        grad = grad + _gdf_xc_grid_motion_correction(
            system,
            ao_basis,
            [D_a_real, D_b_real],
            str(functional),
            xc_lat_opts,
            xc_grid_options,
            bool(xc_use_periodic_becke),
            float(xc_becke_image_radius_bohr),
            xc_fixed,
            density_domain=xc_density_domain,
        )
    return grad


def _compute_krhf_gradient_slab_gdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    W_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    cache: _SlabGdfGradientCache,
    *,
    alpha_hf: float,
    bvk_probe_charge_madelung: float,
    lat_opts: LatticeSumOptions,
) -> np.ndarray:
    """Full slab (dim=2) KRHF GDF gradient assembly at fixed D(k)/W(k).

    G-PBC-002 § 6 rung-5 composition — every term of the slab driver's
    two-gauge total energy (``_run_krhf_periodic_slab_gdf``; pinned
    from parts, bit-exact, by ``test_slab_energy_identity_from_parts``),
    each through its FD-gated helper:

        grad = d e_nuc/dR                  (bare 2D-Ewald block; rung 2,
                                            ``nuclear_repulsion_slab_
                                            ewald_2d_gradient``)
             + kinetic Pulay              (S/T lattice fold; the slab
                                            SCF Bloch-sums plain
                                            ``compute_kinetic_lattice``
                                            on the base ``lat_opts``)
             - overlap Lagrangian          (W(k) fold against dS)
             + slab V_ne                   (rung 3, ``compute_v_ne_slab_
                                            ewald_2d_gradient``)
             + DF-J fit                    (rung 4, q = 0 signed fits)
             + alpha · DF-K fit            (rung 4; the helper carries
                                            the SCF's -0.25)
             + exxdiv W-shift              (xi_BvK · S D S overlap
                                            response; ``d xi/dR = 0`` —
                                            pinned in rung 4 — so the
                                            bulk helper applies verbatim)

    Gauge discipline (the e_nuc-gauge lesson): every term
    differentiates exactly the matrix build the slab SCF used —
    ``lat_opts`` MUST be the adapter's cloned ``SLAB_EWALD_2D`` options
    (S/T/V_ne/e_nuc/XC all build on that one object in the slab
    driver; there is no grown one-electron cutoff and no separate
    V_ne gauge object on this route). The V_ne term passes
    ``alpha = 0.0`` because that is the driver's literal call
    (resolved to the forced slab default 0.4 inside the cache
    builder); e_nuc's alpha is ``slab_ewald_alpha`` (or 0.4) — each
    block is self-consistently differentiated at its own alpha, and
    the composition is exact because each block is alpha-invariant
    in total (the rung-2/3 alpha-invariance gates).

    ``W_k_list`` is the closed-shell zero-temperature energy-weighted
    density ``2 C_occ(k) diag(eps_occ(k)) C_occ(k)^H`` (the slab route
    supports no smearing); ``bvk_probe_charge_madelung`` the SCF's
    ``_slab_probe_charge_madelung_for_kmesh`` constant. Private on
    purpose: the public route is
    ``run_krhf_periodic_gdf(compute_gradient=True)`` on a ``dim=2``
    system, whose adapter hands the SCF's exact provenance here.
    """
    from ._vibeqc_core import nuclear_repulsion_slab_ewald_2d_gradient
    from .periodic_v_ne_slab_gradient import (
        compute_v_ne_slab_ewald_2d_gradient,
    )

    if int(system.dim) != 2:
        raise ValueError(
            "slab KRHF gradient: requires a dim=2 system "
            f"(got dim={int(system.dim)})."
        )
    if lat_opts.coulomb_method != CoulombMethod.SLAB_EWALD_2D:
        raise ValueError(
            "slab KRHF gradient: lat_opts must carry the SLAB_EWALD_2D "
            "gauge the slab SCF ran under."
        )
    kpts = np.asarray(k_cart_list, dtype=float).reshape(-1, 3)
    if kpts.shape != np.asarray(cache.k_cart_list).shape or not np.allclose(
        kpts, np.asarray(cache.k_cart_list), rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "slab KRHF gradient: k_cart_list does not match the cache's "
            "k-point set; the fit terms would mix meshes."
        )

    grad = np.asarray(
        nuclear_repulsion_slab_ewald_2d_gradient(system, lat_opts),
        dtype=np.float64,
    )
    grad += _compute_kinetic_gradient_multik(
        system, ao_basis, D_k_list, weights, kpts, lat_opts
    )
    grad += _compute_overlap_w_gradient_multik(
        system, ao_basis, W_k_list, weights, kpts, lat_opts
    )
    grad += compute_v_ne_slab_ewald_2d_gradient(
        ao_basis, system, lat_opts, D_k_list, weights, kpts, alpha=0.0
    )
    grad += _compute_j_gradient_slab_gdf(
        system, ao_basis, D_k_list, weights, cache
    )
    if float(alpha_hf) != 0.0:
        grad += float(alpha_hf) * _compute_k_gradient_slab_gdf(
            system, ao_basis, D_k_list, weights, cache
        )
        grad += _compute_exxdiv_w_gradient_multik(
            system,
            ao_basis,
            D_k_list,
            S_k_list,
            weights,
            kpts,
            float(alpha_hf),
            float(bvk_probe_charge_madelung),
            lat_opts,
        )
    return grad


def _compute_krks_gradient_slab_gdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D_k_list,
    W_k_list,
    S_k_list,
    weights,
    k_cart_list: np.ndarray,
    cache: _SlabGdfGradientCache,
    *,
    functional: str,
    xc_grid,
    kmesh_bloch,
    bvk_probe_charge_madelung: float,
    lat_opts: LatticeSumOptions,
) -> np.ndarray:
    """Full slab (dim=2) KRKS GDF gradient assembly at fixed D(k)/W(k).

    The KRHF assembler at the functional's exact-exchange fraction
    (pure DFT ``a_x = 0`` skips the K and exxdiv-shift terms; global
    hybrids scale both — exactly the slab SCF's
    ``F_2e = J - 0.5 a K^shifted``) plus the XC Pulay. The slab
    driver's E_xc is a functional of the inverse-Bloch folded
    real-space density on the ``lat_opts``-cutoff cell list
    (``real_space_density_from_per_k_density`` + ``build_xc_periodic``
    — the identical build the 3D multi-k KS driver uses; only the
    Coulomb gauge differs, and E_xc never sees the Coulomb kernel), so
    at fixed D(k) its derivative is the same lattice-summed
    ``xc_lattice_gradient_contribution`` primitive the bulk KRKS
    assembler dispatches — the quadrature machinery is dim-agnostic
    and no new scheme is introduced.

    ``xc_grid`` MUST be the slab SCF's own quadrature (periodic-Becke
    when ``opts.use_periodic_becke``, else the molecular grid on
    ``system.unit_cell_molecule()``) and ``kmesh_bloch`` the driver's
    native Monkhorst-Pack object — the caller (the slab adapter)
    rebuilds both exactly as the driver did. Range-separated
    functionals are rejected upstream by the slab SCF itself
    (``reject_unscreened_range_separated``); the check here keeps the
    assembler fail-closed when called directly.
    """
    from ._vibeqc_core import (
        Functional,
        direct_lattice_cells,
        xc_lattice_gradient_contribution,
    )
    from .periodic_k_density import real_space_density_from_per_k_density

    func = Functional(str(functional), 1)
    if bool(getattr(func, "is_range_separated", False)):
        raise NotImplementedError(
            "_compute_krks_gradient_slab_gdf: range-separated hybrids "
            "are not supported (the slab fitted-K route is full-range "
            "only and rejects them at SCF entry)."
        )
    alpha_hf = float(func.hf_exchange_fraction)

    grad = _compute_krhf_gradient_slab_gdf(
        system,
        ao_basis,
        D_k_list,
        W_k_list,
        S_k_list,
        weights,
        k_cart_list,
        cache,
        alpha_hf=alpha_hf,
        bvk_probe_charge_madelung=bvk_probe_charge_madelung,
        lat_opts=lat_opts,
    )

    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    D_real = real_space_density_from_per_k_density(
        [np.asarray(D, dtype=np.complex128) for D in D_k_list],
        kmesh_bloch,
        cells,
    )
    grad = grad + np.asarray(
        xc_lattice_gradient_contribution(
            ao_basis, system, xc_grid, func, D_real, lat_opts
        ),
        dtype=np.float64,
    )
    return grad


def _compute_j_gradient_rsgdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D: np.ndarray,
    cache: _RsgdfGradientCache,
) -> np.ndarray:
    """DF-J gradient on the rsgdf fit: the compcell weight algebra with
    identity compensation, contracted with the reciprocal-space M/T
    centre-derivative kernels."""
    from .aux_basis import (
        _rsgdf_weighted_2c_metric_gradient,
        _rsgdf_weighted_3c_tensor_gradient,
    )

    n_aux, n_orb = cache.n_aux, cache.n_orb
    T_flat = cache.T.reshape(n_aux, n_orb * n_orb)
    D_flat = np.asarray(D, dtype=np.float64).ravel()
    rho = T_flat @ D_flat
    U = cache.eigvecs
    rho_eig = U.T @ rho
    gamma = U @ (cache.inverse_eigvals * rho_eig)

    omega_eig = 0.5 * cache.inverse_frechet * np.outer(rho_eig, rho_eig)
    omega_aux = U @ omega_eig @ U.T

    W3 = np.outer(gamma, D_flat).reshape(n_aux, n_orb, n_orb)
    if cache.pair_keep_ao is not None:
        # Screened fit, fixed mask: masked AO pairs are hard-zeroed in
        # the SCF's T, so their derivative is exactly zero. Both pieces
        # of the weighted 3c kernel are linear in the per-pair weight
        # entries (piece A contracts W3 with the pair-FT values, piece
        # B folds W3 into the per-pair G-resolved kernel weight Q), so
        # zeroing the masked entries here removes them from the
        # derivative exactly. The 2c metric weight is aux-aux only —
        # the metric has no AO-pair dependence and needs no mask.
        W3[:, ~cache.pair_keep_ao] = 0.0

    grad = _rsgdf_weighted_2c_metric_gradient(
        cache.aux_basis,
        system,
        ke_cutoff=cache.ke_cutoff,
        weight=omega_aux,
        tail_ke_cutoff=cache.tail_ke_cutoff,
    )
    grad = grad + _rsgdf_weighted_3c_tensor_gradient(
        cache.aux_basis,
        ao_basis,
        system,
        ke_cutoff=cache.ke_cutoff,
        weight=W3,
        lat_opts=cache.lat_opts,
        tail_ke_cutoff=cache.tail_ke_cutoff,
        tail_pair_ft_screen=cache.tail_pair_ft_screen,
    )
    return grad


def _compute_k_gradient_rsgdf(
    system: PeriodicSystem,
    ao_basis: BasisSet,
    D: np.ndarray,
    C_occ: np.ndarray,
    alpha_hf: float,
    cache: _RsgdfGradientCache,
) -> np.ndarray:
    """DF-K gradient on the rsgdf fit (identity-compensation weight
    algebra + reciprocal-space derivative kernels). Zero at
    ``alpha_hf = 0``."""
    from .aux_basis import (
        _rsgdf_weighted_2c_metric_gradient,
        _rsgdf_weighted_3c_tensor_gradient,
    )

    if alpha_hf == 0.0:
        return np.zeros((len(system.unit_cell), 3), dtype=np.float64)
    n_aux, n_orb = cache.n_aux, cache.n_orb
    n_occ = C_occ.shape[1]
    M_occ = np.einsum(
        "Pmn,mi,nj->Pij", cache.T, C_occ, C_occ, optimize=True
    )
    M_occ_flat = M_occ.reshape(n_aux, n_occ * n_occ)
    U = cache.eigvecs
    M_occ_eig = U.T @ M_occ_flat
    eta_flat = U @ (cache.inverse_eigvals[:, None] * M_occ_eig)
    eta = eta_flat.reshape(n_aux, n_occ, n_occ)

    rhs_gram_eig = M_occ_eig @ M_occ_eig.T
    omega_eig = -float(alpha_hf) * cache.inverse_frechet * rhs_gram_eig
    omega_aux = U @ omega_eig @ U.T

    Y = np.einsum("Pij,mi,nj->Pmn", eta, C_occ, C_occ, optimize=True)
    if cache.pair_keep_ao is not None:
        # Fixed-mask screened fit: zero the masked-pair 3c derivative
        # weights (see _compute_j_gradient_rsgdf — the same linearity
        # argument; the 2c metric weight has no AO-pair dependence).
        Y[:, ~cache.pair_keep_ao] = 0.0

    grad = _rsgdf_weighted_2c_metric_gradient(
        cache.aux_basis,
        system,
        ke_cutoff=cache.ke_cutoff,
        weight=omega_aux,
        tail_ke_cutoff=cache.tail_ke_cutoff,
    )
    grad_3c = _rsgdf_weighted_3c_tensor_gradient(
        cache.aux_basis,
        ao_basis,
        system,
        ke_cutoff=cache.ke_cutoff,
        weight=Y,
        lat_opts=cache.lat_opts,
        tail_ke_cutoff=cache.tail_ke_cutoff,
        tail_pair_ft_screen=cache.tail_pair_ft_screen,
    )
    return grad - 2.0 * float(alpha_hf) * grad_3c


def compute_gdf_gradient_rsgdf_rhf_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    D: np.ndarray,
    C_occ: np.ndarray,
    eps_occ: np.ndarray,
    S: np.ndarray,
    cache: _RsgdfGradientCache,
    lattice_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    madelung: float,
    v_ne_ke_cutoff: float,
    alpha_hf: float = 1.0,
) -> np.ndarray:
    """Analytic Γ-only rsgdf closed-shell atomic gradient assembly.

    In-driver assembly (the caller owns provenance, exactly like
    :func:`compute_gdf_gradient_uhf_gamma`): the one-electron /
    overlap-Lagrangian / Ewald-nuclear / V_ne terms are the same
    primitives as the compcell assemblies, and the two-electron DF
    terms are :func:`_compute_j_gradient_rsgdf` /
    :func:`_compute_k_gradient_rsgdf` on the rung-3 cache. The exxdiv
    'ewald' shift term is the ``W_exx = 1/2 a_x ξ D S D`` overlap
    contribution. ``alpha_hf`` (M6 rung 5) is the functional's
    exact-exchange fraction for the KS caller — 1.0 for RHF, 0 (pure
    DFT) skips the exchange gradient and the shift term; the driver
    adds the XC Pulay itself.
    """
    from ._vibeqc_core import ewald_nuclear_repulsion_gradient

    n_atoms = len(system.unit_cell)
    n_occ = int(C_occ.shape[1])
    D = np.asarray(D, dtype=np.float64)

    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    home_cell_only = all(
        tuple(int(v) for v in np.asarray(c.index).reshape(3)) == (0, 0, 0)
        for c in D_set.cells
    )
    zero_block = np.zeros_like(D)
    for c_idx in range(len(D_set.cells)):
        idx = tuple(int(v) for v in np.asarray(D_set.cells[c_idx].index).reshape(3))
        D_set.set_block(
            c_idx, D if (idx == (0, 0, 0) or not home_cell_only) else zero_block
        )

    eps = np.asarray(eps_occ, dtype=np.float64)[:n_occ]
    C = np.asarray(C_occ, dtype=np.float64)
    W_gamma = 2.0 * (C * eps[None, :]) @ C.T
    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    for c_idx in range(len(W_set.cells)):
        idx = tuple(int(v) for v in np.asarray(W_set.cells[c_idx].index).reshape(3))
        W_set.set_block(
            c_idx,
            W_gamma if (idx == (0, 0, 0) or not home_cell_only) else zero_block,
        )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    grad += np.asarray(ewald_nuclear_repulsion_gradient(system))
    grad += np.asarray(
        overlap_lattice_gradient_contribution(basis, system, W_set, lattice_opts)
    )
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(basis, system, D_set, lattice_opts)
    )

    from .periodic_v_ne_gradient import compute_v_ne_ewald_3d_ft_gamma_gradient

    grad += compute_v_ne_ewald_3d_ft_gamma_gradient(
        basis,
        system,
        gauge_lat_opts,
        D,
        ke_cutoff=float(v_ne_ke_cutoff),
    )

    from .periodic_k_gdf import _RangeSeparatedGdfCache

    if isinstance(cache, _RangeSeparatedGdfCache):
        grad += _compute_range_separated_cache_gradient(
            system, basis, cache.aux_basis, cache, [D], [1.0],
            memory_byte_cap=cache.memory_byte_cap - cache.retained_cache_bytes,
            native_workspace_byte_cap=cache.native_workspace_byte_cap,
            coulomb_scale=1.0, exchange_scale=float(alpha_hf),
        )
    else:
        grad += _compute_j_gradient_rsgdf(system, basis, D, cache)
        grad += _compute_k_gradient_rsgdf(
            system, basis, D, C, float(alpha_hf), cache
        )

    if abs(float(alpha_hf) * float(madelung)) > 0.0:
        S_arr = np.asarray(S, dtype=np.float64)
        W_exx_gamma = (
            0.5 * float(alpha_hf) * float(madelung) * (D @ S_arr @ D)
        )
        W_exx_set = compute_overlap_lattice(basis, system, lattice_opts)
        for c_idx in range(len(W_exx_set.cells)):
            idx = tuple(
                int(v) for v in np.asarray(W_exx_set.cells[c_idx].index).reshape(3)
            )
            W_exx_set.set_block(
                c_idx,
                W_exx_gamma
                if (idx == (0, 0, 0) or not home_cell_only)
                else np.zeros_like(W_exx_gamma),
            )
        grad += np.asarray(
            overlap_lattice_gradient_contribution(
                basis, system, W_exx_set, lattice_opts
            )
        )
    return grad


def compute_gdf_gradient_rsgdf_uhf_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    D_alpha: np.ndarray,
    D_beta: np.ndarray,
    C_alpha_occ: np.ndarray,
    C_beta_occ: np.ndarray,
    eps_alpha_occ: np.ndarray,
    eps_beta_occ: np.ndarray,
    S: np.ndarray,
    cache: _RsgdfGradientCache,
    lattice_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    madelung: float,
    v_ne_ke_cutoff: float,
    alpha_hf: float = 1.0,
) -> np.ndarray:
    """Analytic Γ-only rsgdf open-shell gradient assembly (M6 rung 5b).

    The rsgdf sibling of :func:`compute_gdf_gradient_uhf_gamma`
    (identical per-spin energy conventions): Coulomb from the total
    density, per-spin ``alpha_hf``-scaled exchange (each spin at half
    the closed-shell kernel weight), occupation-1 energy-weighted
    density, and the per-spin ``a_x``-scaled exxdiv shift. The UKS
    driver adds the spin-polarised XC Pulay itself.
    """
    from ._vibeqc_core import ewald_nuclear_repulsion_gradient

    n_atoms = len(system.unit_cell)
    D_a = np.asarray(D_alpha, dtype=np.float64)
    D_b = np.asarray(D_beta, dtype=np.float64)
    D_t = D_a + D_b

    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    home_cell_only = all(
        tuple(int(v) for v in np.asarray(c.index).reshape(3)) == (0, 0, 0)
        for c in D_set.cells
    )
    zero_block = np.zeros_like(D_t)
    for c_idx in range(len(D_set.cells)):
        idx = tuple(int(v) for v in np.asarray(D_set.cells[c_idx].index).reshape(3))
        D_set.set_block(
            c_idx, D_t if (idx == (0, 0, 0) or not home_cell_only) else zero_block
        )

    W_gamma = (
        (np.asarray(C_alpha_occ) * np.asarray(eps_alpha_occ)[None, :])
        @ np.asarray(C_alpha_occ).T
        + (np.asarray(C_beta_occ) * np.asarray(eps_beta_occ)[None, :])
        @ np.asarray(C_beta_occ).T
    )
    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    for c_idx in range(len(W_set.cells)):
        idx = tuple(int(v) for v in np.asarray(W_set.cells[c_idx].index).reshape(3))
        W_set.set_block(
            c_idx,
            W_gamma if (idx == (0, 0, 0) or not home_cell_only) else zero_block,
        )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    grad += np.asarray(ewald_nuclear_repulsion_gradient(system))
    grad += np.asarray(
        overlap_lattice_gradient_contribution(basis, system, W_set, lattice_opts)
    )
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(basis, system, D_set, lattice_opts)
    )

    from .periodic_v_ne_gradient import compute_v_ne_ewald_3d_ft_gamma_gradient

    grad += compute_v_ne_ewald_3d_ft_gamma_gradient(
        basis,
        system,
        gauge_lat_opts,
        D_t,
        ke_cutoff=float(v_ne_ke_cutoff),
    )

    from .periodic_k_gdf import _RangeSeparatedGdfCache

    if isinstance(cache, _RangeSeparatedGdfCache):
        budgets = dict(
            memory_byte_cap=cache.memory_byte_cap - cache.retained_cache_bytes,
            native_workspace_byte_cap=cache.native_workspace_byte_cap,
        )
        grad += _compute_range_separated_cache_gradient(
            system, basis, cache.aux_basis, cache, [D_t], [1.0],
            coulomb_scale=1.0, exchange_scale=0.0, **budgets,
        )
        if alpha_hf:
            for density in (D_a, D_b):
                # Per-spin exchange energy is -alpha/2 tr(D_s K_s),
                # twice the restricted-density kernel's -alpha/4.
                grad += _compute_range_separated_cache_gradient(
                    system, basis, cache.aux_basis, cache, [density], [1.0],
                    coulomb_scale=0.0, exchange_scale=2.0*float(alpha_hf), **budgets,
                )
    else:
        grad += _compute_j_gradient_rsgdf(system, basis, D_t, cache)

        for D_s, C_s in ((D_a, C_alpha_occ), (D_b, C_beta_occ)):
            if C_s is None or np.asarray(C_s).shape[1] == 0:
                continue
            grad += 0.5 * _compute_k_gradient_rsgdf(
                system,
                basis,
                D_s,
                np.asarray(C_s, dtype=np.float64),
                float(alpha_hf),
                cache,
            )

    if abs(float(alpha_hf) * float(madelung)) > 0.0:
        S_arr = np.asarray(S, dtype=np.float64)
        W_exx_gamma = float(alpha_hf) * float(madelung) * (
            D_a @ S_arr @ D_a + D_b @ S_arr @ D_b
        )
        W_exx_set = compute_overlap_lattice(basis, system, lattice_opts)
        for c_idx in range(len(W_exx_set.cells)):
            idx = tuple(
                int(v) for v in np.asarray(W_exx_set.cells[c_idx].index).reshape(3)
            )
            W_exx_set.set_block(
                c_idx,
                W_exx_gamma
                if (idx == (0, 0, 0) or not home_cell_only)
                else np.zeros_like(W_exx_gamma),
            )
        grad += np.asarray(
            overlap_lattice_gradient_contribution(
                basis, system, W_exx_set, lattice_opts
            )
        )
    return grad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_gdf_gradient_rhf_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    result,  # PBCGDFResult
    *,
    aux_basis: BasisSet,
    compcell_eta: Optional[float] = None,
    alpha_hf: float = 1.0,
    lattice_opts: Optional[LatticeSumOptions] = None,
    cache: Optional[_CompcellGradientCache] = None,
    madelung: Optional[float] = None,
    gauge_lat_opts: Optional[LatticeSumOptions] = None,
    v_ne_ke_cutoff: Optional[float] = None,
    _ks_result_ok: bool = False,
) -> Tuple[np.ndarray, _CompcellGradientCache]:
    """Analytic Γ-only GDF (compcell) RHF atomic gradient.

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    result
        Converged :class:`PBCGDFResult` from :func:`vibeqc.run_pbc_gdf_rhf`.
    aux_basis
        Auxiliary basis (unscaled -- the gradient pipeline handles modrho
        rescaling internally, matching the SCF).
    compcell_eta
        Smooth-Gaussian exponent for the compensating charges. ``None`` uses
        the value retained on ``result``; an explicit value must match it.
    alpha_hf
        HF-exchange fraction. The supported RHF route requires ``1.0``.
    lattice_opts
        Lattice-sum options. Must match the SCF. If None, reconstructed from
        the scalar cutoffs retained on ``result``.
    cache
        Optional pre-built gradient cache from the identical system, AO and
        auxiliary bases, compensation exponent, and cutoff policy. If None,
        built fresh; incompatible provenance is rejected.
    madelung
        Ewald exchange-divergence constant. If None, uses the value retained
        on ``result``; an explicit value must match it.
    v_ne_ke_cutoff
        Reciprocal kinetic-energy cutoff used by the SCF's Ewald ``V_ne``.
        ``None`` uses the value retained on ``result``; an explicit value
        must match it.

    Returns
    -------
    grad : np.ndarray of shape (n_atoms, 3) in Hartree/bohr.
    cache : _CompcellGradientCache
        The cache built (or passed in). Caller can reuse across
        multiple gradient evaluations at the same geometry.
    """
    if int(system.dim) != 3:
        raise NotImplementedError(
            "compute_gdf_gradient_rhf_gamma: the analytic Ewald V_ne "
            "derivative currently supports 3D periodic systems only."
        )
    _validate_supported_gradient_result(
        result,
        caller="compute_gdf_gradient_rhf_gamma",
        allow_ks=bool(_ks_result_ok),
    )
    if _ks_result_ok:
        # KS caller (compute_gdf_gradient_rks_gamma): the functional's
        # exact-exchange fraction. Pure DFT (0.0) skips the K gradient
        # entirely below; global hybrids scale K and the exxdiv shift.
        if not (0.0 <= float(alpha_hf) <= 1.0):
            raise ValueError(
                "compute_gdf_gradient_rhf_gamma: alpha_hf must be in "
                f"[0, 1] for a KS result (got {float(alpha_hf)!r})"
            )
    elif not np.isclose(float(alpha_hf), 1.0, rtol=0.0, atol=0.0):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: alpha_hf=1.0 is required for "
            "the supported RHF result"
        )
    retained_madelung = _validated_result_madelung(
        system,
        result,
        caller="compute_gdf_gradient_rhf_gamma",
    )
    if madelung is None:
        madelung = retained_madelung
    elif not np.isclose(
        float(madelung), retained_madelung, rtol=0.0, atol=1e-15
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: madelung must match the "
            "converged SCF result"
        )

    missing = object()
    result_rcut_strategy = getattr(result, "gdf_rcut_strategy", missing)
    if result_rcut_strategy is missing:
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: missing compcell rcut provenance"
        )
    result_linear_dep_thr = float(
        getattr(result, "gdf_linear_dep_threshold", float("nan"))
    )
    result_rcut_precision = float(
        getattr(result, "gdf_rcut_precision", float("nan"))
    )
    result_fit_cutoff_2c = float(
        getattr(result, "gdf_fit_cutoff_2c", float("nan"))
    )
    result_fit_cutoff_3c = float(
        getattr(result, "gdf_fit_cutoff_3c", float("nan"))
    )
    result_aux_fingerprint = str(
        getattr(result, "aux_basis_fingerprint", "")
    )
    result_compcell_eta = float(
        getattr(result, "compcell_eta", float("nan"))
    )
    result_v_ne_ke_cutoff = float(
        getattr(result, "v_ne_ke_cutoff", float("nan"))
    )
    result_lattice_cutoff = float(
        getattr(result, "gdf_lattice_cutoff_bohr", float("nan"))
    )
    result_nuclear_cutoff = float(
        getattr(result, "gdf_nuclear_cutoff_bohr", float("nan"))
    )
    if not (
        np.isfinite(result_linear_dep_thr)
        and result_linear_dep_thr >= 0.0
        and np.isfinite(result_rcut_precision)
        and result_rcut_precision > 0.0
        and np.isfinite(result_fit_cutoff_2c)
        and result_fit_cutoff_2c > 0.0
        and np.isfinite(result_fit_cutoff_3c)
        and result_fit_cutoff_3c > 0.0
        and np.isfinite(result_lattice_cutoff)
        and result_lattice_cutoff > 0.0
        and np.isfinite(result_nuclear_cutoff)
        and result_nuclear_cutoff > 0.0
        and bool(result_aux_fingerprint)
        and np.isfinite(result_compcell_eta)
        and result_compcell_eta > 0.0
        and np.isfinite(result_v_ne_ke_cutoff)
        and result_v_ne_ke_cutoff > 0.0
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: invalid compcell fit provenance"
        )
    if _basis_fingerprint(aux_basis) != result_aux_fingerprint:
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: auxiliary basis content does "
            "not match the converged SCF result"
        )
    if compcell_eta is None:
        compcell_eta = result_compcell_eta
    elif not np.isclose(
        float(compcell_eta), result_compcell_eta, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: compcell_eta must match the "
            "converged SCF result"
        )
    if v_ne_ke_cutoff is None:
        v_ne_ke_cutoff = result_v_ne_ke_cutoff
    elif not np.isclose(
        float(v_ne_ke_cutoff), result_v_ne_ke_cutoff, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: v_ne_ke_cutoff must match the "
            "converged SCF result"
        )

    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
        lattice_opts.cutoff_bohr = result_lattice_cutoff
        lattice_opts.nuclear_cutoff_bohr = result_nuclear_cutoff
    elif not (
        np.isclose(
            float(lattice_opts.cutoff_bohr),
            result_lattice_cutoff,
            rtol=0.0,
            atol=1e-14,
        )
        and np.isclose(
            float(lattice_opts.nuclear_cutoff_bohr),
            result_nuclear_cutoff,
            rtol=0.0,
            atol=1e-14,
        )
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: lattice_opts cutoffs must "
            "match the converged SCF result"
        )
    if gauge_lat_opts is None:
        from .pbc_gdf import _gauge_lat_opts_ewald_3d

        gauge_lat_opts = _gauge_lat_opts_ewald_3d(lattice_opts, system)
    else:
        if gauge_lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
            raise ValueError(
                "compute_gdf_gradient_rhf_gamma: gauge_lat_opts must use "
                "EWALD_3D"
            )
        if not (
            np.isclose(
                float(gauge_lat_opts.cutoff_bohr),
                result_lattice_cutoff,
                rtol=0.0,
                atol=1e-14,
            )
            and np.isclose(
                float(gauge_lat_opts.nuclear_cutoff_bohr),
                result_nuclear_cutoff,
                rtol=0.0,
                atol=1e-14,
            )
        ):
            raise ValueError(
                "compute_gdf_gradient_rhf_gamma: gauge_lat_opts cutoffs "
                "must match the converged SCF result"
            )

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: closed-shell only "
            f"(got {n_elec} electrons)"
        )
    n_occ = n_elec // 2

    D = np.asarray(result.density, dtype=np.float64)
    C_occ = np.asarray(result.mo_coeffs[:, :n_occ], dtype=np.float64)
    n_atoms = len(system.unit_cell)
    n_orb = basis.nbasis

    # Build or reuse the geometry-dependent (density-independent) cache.
    # The AFT settings come from the converged result so the rebuilt fit
    # subtracts exactly what the SCF subtracted (milestone 3b).
    result_apply_aft = bool(getattr(result, "apply_aft_correction", False))
    result_aft_precision = float(getattr(result, "aft_precision", 1e-10))
    result_aft_convention = str(
        getattr(result, "aft_ft_convention", "libint")
    )
    if cache is None:
        cache = _build_compcell_gradient_cache(
            system,
            basis,
            aux_basis,
            compcell_eta=float(compcell_eta),
            lattice_opts=lattice_opts,
            linear_dep_thr=result_linear_dep_thr,
            rcut_strategy=result_rcut_strategy,
            rcut_precision=result_rcut_precision,
            apply_aft_correction=result_apply_aft,
            aft_precision=result_aft_precision,
            aft_ft_convention=result_aft_convention,
        )
    if (
        bool(cache.apply_aft_correction) != result_apply_aft
        or (
            result_apply_aft
            and (
                float(cache.aft_precision) != result_aft_precision
                or str(cache.aft_ft_convention) != result_aft_convention
            )
        )
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: cache AFT provenance does "
            "not match the converged SCF result"
        )
    cache_provenance_matches = (
        str(getattr(cache, "system_fingerprint", ""))
        == _system_fingerprint(system)
        and str(getattr(cache, "ao_basis_fingerprint", ""))
        == _basis_fingerprint(basis)
        and str(getattr(cache, "aux_basis_fingerprint", ""))
        == result_aux_fingerprint
        and np.isclose(
            float(getattr(cache, "compcell_eta", float("nan"))),
            result_compcell_eta,
            rtol=0.0,
            atol=0.0,
        )
        and np.isclose(
            float(getattr(cache, "lattice_cutoff_bohr", float("nan"))),
            result_lattice_cutoff,
            rtol=0.0,
            atol=0.0,
        )
        and np.isclose(
            float(getattr(cache, "nuclear_cutoff_bohr", float("nan"))),
            result_nuclear_cutoff,
            rtol=0.0,
            atol=0.0,
        )
        and getattr(cache, "rcut_strategy", missing)
        == _rcut_strategy_value(result_rcut_strategy)
        and np.isclose(
            float(getattr(cache, "rcut_precision", float("nan"))),
            result_rcut_precision,
            rtol=0.0,
            atol=0.0,
        )
    )
    if (
        not cache_provenance_matches
        or cache.n_fit != int(getattr(result, "n_fit", -1))
        or cache.linear_dep_thr != result_linear_dep_thr
        or not np.isclose(
            float(cache.lat_opts_2c.cutoff_bohr),
            result_fit_cutoff_2c,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(cache.lat_opts_3c.cutoff_bohr),
            result_fit_cutoff_3c,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "compute_gdf_gradient_rhf_gamma: compcell cache provenance "
            "does not match the converged SCF fit"
        )

    # ---- 1-electron Pulay + overlap + nuclear terms ---------------------
    # These are the same as the existing periodic gradient (G1a) for the
    # Ewald path. We build the lattice-resolved density and W matrices
    # using the GDF SCF's hcore (not rebuilt molecule Fock, since at
    # Γ the GDF Fock doesn't have the G=0 gauge issue of EWALD_3D).

    # Build the overlap lattice for the cell list.
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)

    # Lattice-resolved density: at Γ, D(g) = D_Γ for all images.
    D_gamma = np.asarray(D, dtype=np.float64)
    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    # Homogeneous Γ density: D(g) = D_Γ for all image cells.
    # At Γ, k=0 Bloch phase is 1 in every cell, so the real-space
    # density is identical in all images. The SCF energy is built from
    # this homogeneous density; the gradient must use it too.
    home_cell_only = all(
        tuple(int(v) for v in np.asarray(c.index).reshape(3)) == (0, 0, 0)
        for c in D_set.cells
    )
    zero_block = np.zeros_like(D_gamma)
    for c_idx in range(len(D_set.cells)):
        idx = tuple(int(v) for v in np.asarray(D_set.cells[c_idx].index).reshape(3))
        D_set.set_block(
            c_idx, D_gamma if (idx == (0, 0, 0) or not home_cell_only) else zero_block
        )

    # Energy-weighted density W for the overlap-Lagrangian.
    # Use the converged Fock eigenvalues directly -- the GDF route doesn't
    # have the EWALD_3D G=0 gauge shift, so e from the SCF is fine.
    eps = np.asarray(result.mo_energies, dtype=np.float64)
    C = np.asarray(result.mo_coeffs, dtype=np.float64)
    W_gamma = 2.0 * (C[:, :n_occ] * eps[:n_occ][None, :]) @ C[:, :n_occ].T
    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    for c_idx in range(len(W_set.cells)):
        idx = tuple(int(v) for v in np.asarray(W_set.cells[c_idx].index).reshape(3))
        W_set.set_block(
            c_idx,
            W_gamma if (idx == (0, 0, 0) or not home_cell_only) else zero_block,
        )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    # Nuclear repulsion. The GDF setup builds e_nuc in the Ewald gauge
    # (``_pbc_gdf_gamma_setup`` forces EWALD_3D to match PySCF's
    # exxdiv='ewald'), so the matching derivative is the Ewald nuclear
    # gradient. The legacy ``nuclear_repulsion_gradient_per_cell``
    # differentiates the truncated *direct* 1/r sum instead — the
    # long-undiagnosed 3.394e-3 Ha/bohr flat residual on the H2 FD
    # control (G-PBC-002 milestone 3b identification, 2026-07-29; the
    # BIPOLE route hit and fixed the same inconsistency independently).
    # Both routines are alpha-invariant at convergence, and
    # ``ewald_nuclear_repulsion_gradient`` matches FD of the retained
    # e_nuc at 2.6e-9 on that control.
    from ._vibeqc_core import ewald_nuclear_repulsion_gradient

    grad += np.asarray(ewald_nuclear_repulsion_gradient(system))

    # Overlap Lagrangian: -tr(W dS/dR).
    grad += np.asarray(
        overlap_lattice_gradient_contribution(basis, system, W_set, lattice_opts)
    )

    # Kinetic + nuclear-attraction Pulay: tr(D d(T+V)/dR).
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(basis, system, D_set, lattice_opts)
    )
    # V_ne Hellmann-Feynman and AO-centre response in the exact
    # FT-based Ewald gauge used by the SCF.
    from .periodic_v_ne_gradient import (
        compute_v_ne_ewald_3d_ft_gamma_gradient,
    )

    grad += compute_v_ne_ewald_3d_ft_gamma_gradient(
        basis,
        system,
        gauge_lat_opts,
        D,
        ke_cutoff=float(v_ne_ke_cutoff),
    )

    # ---- 2-electron DF gradient -----------------------------------------
    grad_J = _compute_j_gradient_compcell(
        system,
        basis,
        D,
        cache,
    )

    if alpha_hf > 0.0:
        grad_K = _compute_k_gradient_compcell(
            system,
            basis,
            D,
            C_occ,
            float(alpha_hf),
            cache,
        )
        # E_2e = E_J - 1/4 a_HF tr(D.K_shifted)
        #      = E_J - 1/4 a_HF tr(D.K_raw) - 1/4 a_HF ξ tr(D.S.D.S)
        # dE_2e/dR = dE_J/dR - 1/4 a_HF dE_K_raw/dR - 1/2 a_HF ξ tr(D.S.D.dS/dR)
        grad += grad_J + grad_K

        # Exxdiv shift gradient: K_shifted = K_raw + ξ.S.D.S (ADDED in
        # apply_exxdiv_ewald_to_K). So E_K = -1/4 a Tr(D.K_raw) - 1/4 a ξ
        # Tr(D.S.D.S) with a = alpha_hf (the SCF scales the *shifted* K by
        # a, so the shift term carries a too).
        # d(-1/4 a ξ Tr(D.S.D.S))/dR at fixed D = -1/2 a ξ Tr(D.S.D.dS/dR).
        # overlap_lattice_gradient_contribution computes -Tr(W.dS/dR),
        # so pass W_exx = +1/2 a ξ D.S.D. (a = 1.0 exactly on the RHF
        # path, where this multiplication is a bit-identical no-op.)
        if abs(madelung) > 0.0:
            S = np.asarray(result.overlap, dtype=np.float64)
            DSD = D @ S @ D
            W_exx_gamma = 0.5 * float(alpha_hf) * madelung * DSD
            W_exx_set = compute_overlap_lattice(basis, system, lattice_opts)
            for c_idx in range(len(W_exx_set.cells)):
                idx = tuple(
                    int(v) for v in np.asarray(W_exx_set.cells[c_idx].index).reshape(3)
                )
                W_exx_set.set_block(
                    c_idx,
                    W_exx_gamma
                    if (idx == (0, 0, 0) or not home_cell_only)
                    else np.zeros_like(W_exx_gamma),
                )
            grad += np.asarray(
                overlap_lattice_gradient_contribution(
                    basis, system, W_exx_set, lattice_opts
                )
            )
    else:
        grad += grad_J

    return grad, cache


def compute_gdf_gradient_rks_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    result,  # PBCGDFResult with a functional (backend pbc-gdf-compcell-rks)
    *,
    aux_basis: BasisSet,
    grid_options=None,
    use_periodic_becke: bool = False,
    becke_image_radius_bohr: float = 0.0,
    compcell_eta: Optional[float] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
    cache: Optional[_CompcellGradientCache] = None,
    madelung: Optional[float] = None,
    gauge_lat_opts: Optional[LatticeSumOptions] = None,
    v_ne_ke_cutoff: Optional[float] = None,
) -> Tuple[np.ndarray, _CompcellGradientCache]:
    """Analytic Γ-only GDF (compcell) closed-shell KS atomic gradient.

    G-PBC-002 milestone 2: composes the general-``alpha_hf`` compcell
    J/K/1e/overlap-Lagrangian machinery of
    :func:`compute_gdf_gradient_rhf_gamma` (pure DFT skips the K
    gradient; global hybrids scale K and the exxdiv shift by the
    functional's exact-exchange fraction) with the separately validated
    lattice-summed XC Pulay primitive
    ``xc_lattice_gradient_contribution`` (the G1b periodic-RKS
    machinery). The GDF KS SCF evaluates its XC on exactly the
    quadrature this primitive differentiates — ``build_xc_periodic`` on
    either the molecular grid + home-cell Γ density
    (``use_periodic_becke=False``) or the periodic-Becke grid + Γ-torus
    density (``use_periodic_becke=True``) — so the three grid/density
    parameters below MUST match the converged SCF for the gradient to be
    the derivative of its energy.

    The energy-weighted density uses the converged KS Fock eigenvalues
    directly: the GDF Fock includes ``V_xc``, so ``result.mo_energies``
    are already the variational KS eigenvalues (no EWALD_3D-style G=0
    gauge threading is needed on this route; see the RHF kernel's note).

    Range-separated functionals never reach this function — the SCF
    driver fails closed on them before converging a result — but a
    defensive check below rejects them anyway for standalone callers.

    Parameters mirror :func:`compute_gdf_gradient_rhf_gamma`, plus
    ``grid_options`` / ``use_periodic_becke`` / ``becke_image_radius_bohr``
    (the SCF's XC quadrature provenance).

    Returns
    -------
    grad : np.ndarray of shape (n_atoms, 3) in Hartree/bohr.
    cache : _CompcellGradientCache
        Reusable geometry-dependent gradient cache.
    """
    from ._vibeqc_core import (
        Functional,
        GridOptions,
        build_grid,
        xc_lattice_gradient_contribution,
    )

    _validate_supported_gradient_result(
        result,
        caller="compute_gdf_gradient_rks_gamma",
        allow_ks=True,
    )
    func = Functional(str(result.functional), 1)
    if bool(getattr(func, "is_range_separated", False)):
        raise NotImplementedError(
            "compute_gdf_gradient_rks_gamma: range-separated hybrids are "
            "not supported (the Γ GDF SCF fails closed on them; this "
            "result did not come from the supported route)."
        )
    alpha_hf = float(func.hf_exchange_fraction)

    grad, cache = compute_gdf_gradient_rhf_gamma(
        system,
        basis,
        result,
        aux_basis=aux_basis,
        compcell_eta=compcell_eta,
        alpha_hf=alpha_hf,
        lattice_opts=lattice_opts,
        cache=cache,
        madelung=madelung,
        gauge_lat_opts=gauge_lat_opts,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
        _ks_result_ok=True,
    )

    # ---- XC Pulay -------------------------------------------------------
    # Rebuild the SCF's exact XC quadrature (grid + Γ density-set
    # convention) and differentiate it with the lattice-summed periodic
    # primitive. The reconstructed-lattice_opts case mirrors the RHF
    # kernel: cutoffs were validated against the result's retained
    # provenance there, so reconstruct identically here.
    from .periodic_rhf_gdf import _density_set_gamma, _density_set_torus_gamma

    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
        lattice_opts.cutoff_bohr = float(result.gdf_lattice_cutoff_bohr)
        lattice_opts.nuclear_cutoff_bohr = float(
            result.gdf_nuclear_cutoff_bohr
        )
    grid_options = grid_options if grid_options is not None else GridOptions()
    if bool(use_periodic_becke):
        from .periodic_grid import build_periodic_becke_grid

        grid = build_periodic_becke_grid(
            system,
            grid_options=grid_options,
            image_radius_bohr=float(becke_image_radius_bohr),
        )
        set_xc_density = _density_set_torus_gamma
    else:
        grid = build_grid(system.unit_cell_molecule(), grid_options)
        set_xc_density = _density_set_gamma
    D = np.asarray(result.density, dtype=np.float64)
    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    set_xc_density(D_set, D)
    grad = grad + np.asarray(
        xc_lattice_gradient_contribution(
            basis, system, grid, func, D_set, lattice_opts
        )
    )
    return grad, cache


def compute_gdf_gradient_uhf_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    D_alpha: np.ndarray,
    D_beta: np.ndarray,
    C_alpha_occ: np.ndarray,
    C_beta_occ: np.ndarray,
    eps_alpha_occ: np.ndarray,
    eps_beta_occ: np.ndarray,
    S: np.ndarray,
    cache: _CompcellGradientCache,
    lattice_opts: LatticeSumOptions,
    gauge_lat_opts: LatticeSumOptions,
    madelung: float,
    v_ne_ke_cutoff: float,
    alpha_hf: float = 1.0,
) -> np.ndarray:
    """Analytic Γ-only GDF (compcell) UHF atomic gradient assembly.

    ``alpha_hf`` (G-PBC-002 milestone 4) scales the per-spin exchange
    and its exxdiv-shift term by the functional's exact-exchange
    fraction — 1.0 is the UHF value; the UKS driver passes its
    functional's fraction (0.0 for pure DFT skips both) and adds the
    spin-polarised XC Pulay itself.

    In-driver open-shell sibling of the assembly half of
    :func:`compute_gdf_gradient_rhf_gamma`. The caller
    (:func:`vibeqc.run_pbc_gdf_uhf` with ``compute_gradient=True``) owns
    provenance: it passes the converged per-spin quantities together with
    the gradient cache built from the SCF's own retained compcell fit
    state, so every setting matches the converged fit by construction.
    There is no standalone re-derivation wrapper for UHF yet -- that
    (mirroring :func:`compute_gdf_gradient`'s scalar-provenance
    validation) is a G-PBC-002 follow-up.

    Per-spin generalisation of the closed-shell terms (UHF energy
    convention of ``run_pbc_gdf_uhf``:
    ``E_2e = 1/2 Tr[D_t J] - 1/2 Σ_σ Tr[D_σ K_σ^shifted]`` with
    one-particle spin densities ``D_σ = C_σ C_σᵀ``):

    * Coulomb: quadratic in the total density, so the closed-shell DF-J
      gradient applies verbatim with ``D_t = D_α + D_β``.
    * Exchange: ``-1/2 Tr[D_σ K_σ[D_σ]] = -1/2 Σ_{ij∈σ} (ij|ji)``.
      :func:`_compute_k_gradient_compcell` differentiates
      ``-α Σ_{ij}^{ncols(C)} (ij|ji)`` for the supplied orbital block, so
      each spin contributes ``1/2 ×`` that kernel at ``α = 1`` (the
      closed-shell call is recovered exactly at M = 1, where
      ``C_α = C_β = C_occ`` and the two halves sum to one kernel call).
    * Energy-weighted density: occupation-1 orbitals give
      ``W = Σ_σ C_σ ε_σ C_σᵀ`` (no closed-shell factor 2). A
      Fermi-Dirac smeared caller passes sqrt(f)-scaled columns
      (``pbc_gdf._gamma_gradient_orbital_blocks``), which evaluates
      the fractional-occupation ``W = Σ f ε c cᵀ`` and the
      f_i f_j-weighted exchange through this same occupation-agnostic
      algebra (Mermin free-energy forces).
    * exxdiv='ewald' shift: ``K_σ += ξ S D_σ S`` per spin
      (``apply_exxdiv_ewald_to_K``), so
      ``E_shift = -1/2 ξ Σ_σ Tr[D_σ S D_σ S]`` and, at fixed ``D_σ``,
      ``dE_shift/dR = -ξ Σ_σ Tr[D_σ S D_σ dS/dR]``. With the
      ``-Tr(W dS/dR)`` convention of
      ``overlap_lattice_gradient_contribution`` that is
      ``W_exx = ξ Σ_σ D_σ S D_σ`` (closed shell: ``D_σ = D/2`` recovers
      the RHF ``1/2 ξ D S D``).

    Returns ``(n_atoms, 3)`` in Hartree/bohr.
    """
    n_atoms = len(system.unit_cell)
    D_a = np.asarray(D_alpha, dtype=np.float64)
    D_b = np.asarray(D_beta, dtype=np.float64)
    D_t = D_a + D_b

    # Lattice-resolved density / W sets: at Γ the k=0 Bloch phase is 1 in
    # every image cell, so the home-cell block is replicated exactly as in
    # the closed-shell assembly.
    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    home_cell_only = all(
        tuple(int(v) for v in np.asarray(c.index).reshape(3)) == (0, 0, 0)
        for c in D_set.cells
    )
    zero_block = np.zeros_like(D_t)
    for c_idx in range(len(D_set.cells)):
        idx = tuple(int(v) for v in np.asarray(D_set.cells[c_idx].index).reshape(3))
        D_set.set_block(
            c_idx, D_t if (idx == (0, 0, 0) or not home_cell_only) else zero_block
        )

    # Energy-weighted density, occupation 1 per spin orbital.
    W_gamma = (
        (np.asarray(C_alpha_occ) * np.asarray(eps_alpha_occ)[None, :])
        @ np.asarray(C_alpha_occ).T
        + (np.asarray(C_beta_occ) * np.asarray(eps_beta_occ)[None, :])
        @ np.asarray(C_beta_occ).T
    )
    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    for c_idx in range(len(W_set.cells)):
        idx = tuple(int(v) for v in np.asarray(W_set.cells[c_idx].index).reshape(3))
        W_set.set_block(
            c_idx,
            W_gamma if (idx == (0, 0, 0) or not home_cell_only) else zero_block,
        )

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    # Ewald-gauge nuclear gradient — matches the Ewald e_nuc the GDF
    # setup builds; see the closed-shell assembly's note on the retired
    # truncated-direct-sum helper (the 3.394e-3 residual).
    from ._vibeqc_core import ewald_nuclear_repulsion_gradient

    grad += np.asarray(ewald_nuclear_repulsion_gradient(system))
    grad += np.asarray(
        overlap_lattice_gradient_contribution(basis, system, W_set, lattice_opts)
    )
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(basis, system, D_set, lattice_opts)
    )

    from .periodic_v_ne_gradient import (
        compute_v_ne_ewald_3d_ft_gamma_gradient,
    )

    grad += compute_v_ne_ewald_3d_ft_gamma_gradient(
        basis,
        system,
        gauge_lat_opts,
        D_t,
        ke_cutoff=float(v_ne_ke_cutoff),
    )

    grad += _compute_j_gradient_compcell(system, basis, D_t, cache)

    for D_s, C_s in ((D_a, C_alpha_occ), (D_b, C_beta_occ)):
        if C_s is None or np.asarray(C_s).shape[1] == 0:
            continue  # e.g. n_beta = 0 (hydrogen-atom class)
        grad += 0.5 * _compute_k_gradient_compcell(
            system,
            basis,
            D_s,
            np.asarray(C_s, dtype=np.float64),
            float(alpha_hf),
            cache,
        )

    if abs(float(alpha_hf) * float(madelung)) > 0.0:
        S_arr = np.asarray(S, dtype=np.float64)
        W_exx_gamma = float(alpha_hf) * float(madelung) * (
            D_a @ S_arr @ D_a + D_b @ S_arr @ D_b
        )
        W_exx_set = compute_overlap_lattice(basis, system, lattice_opts)
        for c_idx in range(len(W_exx_set.cells)):
            idx = tuple(
                int(v) for v in np.asarray(W_exx_set.cells[c_idx].index).reshape(3)
            )
            W_exx_set.set_block(
                c_idx,
                W_exx_gamma
                if (idx == (0, 0, 0) or not home_cell_only)
                else np.zeros_like(W_exx_gamma),
            )
        grad += np.asarray(
            overlap_lattice_gradient_contribution(
                basis, system, W_exx_set, lattice_opts
            )
        )

    return grad


def compute_gdf_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    result,  # PBCGDFResult
    *,
    aux_basis: Optional[BasisSet] = None,
    aux_basis_name: Optional[str] = None,
    compcell_eta: Optional[float] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
    v_ne_ke_cutoff: Optional[float] = None,
    grid_options=None,
    use_periodic_becke: bool = False,
    becke_image_radius_bohr: float = 0.0,
    _fit_state: Optional[_CompcellFitState] = None,
) -> np.ndarray:
    """Compute the GDF analytic gradient from a converged PBCGDFResult.

    Convenience wrapper around :func:`compute_gdf_gradient_rhf_gamma`
    (RHF results) or :func:`compute_gdf_gradient_rks_gamma` (closed-shell
    KS results, detected by a non-empty ``result.functional``). Handles
    gauge setup, gradient cache construction, and validation of the
    exchange-divergence constant retained by the converged SCF result.
    For KS results, ``grid_options`` / ``use_periodic_becke`` /
    ``becke_image_radius_bohr`` must reproduce the SCF's XC quadrature
    (the driver passes its own values through; standalone callers must
    match their SCF call).

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    result
        Converged :class:`PBCGDFResult` from :func:`vibeqc.run_pbc_gdf_rhf`.
    aux_basis
        Auxiliary basis. If None, built from ``aux_basis_name``.
    aux_basis_name
        Aux basis name for auto-construction. ``None`` uses the name retained
        on ``result``; an explicit name must match the converged SCF value.
    compcell_eta
        Compensation exponent. ``None`` uses the value retained on ``result``;
        an explicit value must match the converged SCF value.
    lattice_opts
        Lattice-sum options. If None, reconstructs the base and nuclear
        cutoffs retained by the converged SCF result.
    v_ne_ke_cutoff
        Ewald ``V_ne`` reciprocal cutoff. ``None`` uses the cutoff retained
        on ``result``; an explicit value must match that converged SCF value.

    Returns
    -------
    grad : (n_atoms, 3) ndarray in Hartree/bohr.
    """
    if int(system.dim) != 3:
        raise NotImplementedError(
            "compute_gdf_gradient: the analytic Ewald V_ne derivative "
            "currently supports 3D periodic systems only."
        )
    is_ks = bool(str(getattr(result, "functional", "") or ""))
    _validate_supported_gradient_result(
        result, caller="compute_gdf_gradient", allow_ks=is_ks
    )
    result_v_ne_ke_cutoff = float(
        getattr(result, "v_ne_ke_cutoff", float("nan"))
    )
    if not np.isfinite(result_v_ne_ke_cutoff):
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "its Ewald V_ne reciprocal cutoff."
        )
    if v_ne_ke_cutoff is None:
        v_ne_ke_cutoff = result_v_ne_ke_cutoff
    elif float(v_ne_ke_cutoff) != result_v_ne_ke_cutoff:
        raise ValueError(
            "compute_gdf_gradient: v_ne_ke_cutoff must match the converged "
            f"SCF value ({result_v_ne_ke_cutoff:g} Ha)."
        )

    result_aux_basis_name = str(getattr(result, "aux_basis_name", ""))
    if not result_aux_basis_name:
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "its auxiliary-basis name."
        )
    if aux_basis_name is None:
        aux_basis_name = result_aux_basis_name
    elif str(aux_basis_name) != result_aux_basis_name:
        raise ValueError(
            "compute_gdf_gradient: aux_basis_name must match the converged "
            f"SCF value ({result_aux_basis_name!r})."
        )
    if aux_basis is not None:
        supplied_aux_basis_name = str(getattr(aux_basis, "name", ""))
        if (
            supplied_aux_basis_name
            and supplied_aux_basis_name != result_aux_basis_name
        ):
            raise ValueError(
                "compute_gdf_gradient: aux_basis must match the converged "
                f"SCF value ({result_aux_basis_name!r})."
            )

    result_aux_fingerprint = str(
        getattr(result, "aux_basis_fingerprint", "")
    )
    if not result_aux_fingerprint:
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "its auxiliary-basis fingerprint."
        )

    result_compcell_eta = float(
        getattr(result, "compcell_eta", float("nan"))
    )
    if not np.isfinite(result_compcell_eta):
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "its compcell_eta value."
        )
    if compcell_eta is None:
        compcell_eta = result_compcell_eta
    elif float(compcell_eta) != result_compcell_eta:
        raise ValueError(
            "compute_gdf_gradient: compcell_eta must match the converged "
            f"SCF value ({result_compcell_eta:g})."
        )

    missing = object()
    result_rcut_strategy = getattr(result, "gdf_rcut_strategy", missing)
    if result_rcut_strategy is missing:
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "its compcell rcut strategy."
        )
    if result_rcut_strategy is not None:
        result_rcut_strategy = str(result_rcut_strategy)

    result_rcut_precision = float(
        getattr(result, "gdf_rcut_precision", float("nan"))
    )
    result_linear_dep_thr = float(
        getattr(result, "gdf_linear_dep_threshold", float("nan"))
    )
    if not np.isfinite(result_rcut_precision) or result_rcut_precision <= 0.0:
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "a valid compcell rcut precision."
        )
    if not np.isfinite(result_linear_dep_thr) or result_linear_dep_thr < 0.0:
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "a valid GDF linear-dependence threshold."
        )

    result_lattice_cutoff = float(
        getattr(result, "gdf_lattice_cutoff_bohr", float("nan"))
    )
    result_nuclear_cutoff = float(
        getattr(result, "gdf_nuclear_cutoff_bohr", float("nan"))
    )
    result_fit_cutoff_2c = float(
        getattr(result, "gdf_fit_cutoff_2c", float("nan"))
    )
    result_fit_cutoff_3c = float(
        getattr(result, "gdf_fit_cutoff_3c", float("nan"))
    )
    retained_cutoffs = (
        result_lattice_cutoff,
        result_nuclear_cutoff,
        result_fit_cutoff_2c,
        result_fit_cutoff_3c,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in retained_cutoffs):
        raise ValueError(
            "compute_gdf_gradient: the converged result does not retain "
            "valid base and resolved compcell lattice cutoffs."
        )

    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
        lattice_opts.cutoff_bohr = result_lattice_cutoff
        lattice_opts.nuclear_cutoff_bohr = result_nuclear_cutoff
    elif not (
        np.isclose(
            float(lattice_opts.cutoff_bohr),
            result_lattice_cutoff,
            rtol=0.0,
            atol=1e-14,
        )
        and np.isclose(
            float(lattice_opts.nuclear_cutoff_bohr),
            result_nuclear_cutoff,
            rtol=0.0,
            atol=1e-14,
        )
    ):
        raise ValueError(
            "compute_gdf_gradient: lattice_opts cutoffs must match the "
            "converged SCF values."
        )

    result_madelung = _validated_result_madelung(
        system,
        result,
        caller="compute_gdf_gradient",
    )

    from .pbc_gdf import _gauge_lat_opts_ewald_3d
    from .aux_basis import make_aux_basis_set

    gauge_opts = _gauge_lat_opts_ewald_3d(lattice_opts, system)
    mol = system.unit_cell_molecule()

    if aux_basis is None:
        aux_basis = make_aux_basis_set(mol, aux_name=aux_basis_name)
    if _basis_fingerprint(aux_basis) != result_aux_fingerprint:
        raise ValueError(
            "compute_gdf_gradient: auxiliary basis content must match the "
            "converged SCF result."
        )

    # Build gradient cache (expensive, geometry-dependent, density-independent)
    cache = _build_compcell_gradient_cache(
        system,
        basis,
        aux_basis,
        compcell_eta=float(compcell_eta),
        lattice_opts=lattice_opts,
        linear_dep_thr=result_linear_dep_thr,
        rcut_strategy=result_rcut_strategy,
        rcut_precision=result_rcut_precision,
        fit_state=_fit_state,
        apply_aft_correction=bool(
            getattr(result, "apply_aft_correction", False)
        ),
        aft_precision=float(getattr(result, "aft_precision", 1e-10)),
        aft_ft_convention=str(
            getattr(result, "aft_ft_convention", "libint")
        ),
    )
    result_n_fit = int(getattr(result, "n_fit", -1))
    if cache.n_fit != result_n_fit:
        raise ValueError(
            "compute_gdf_gradient: rebuilt compcell fit rank does not match "
            f"the converged SCF result ({cache.n_fit} != {result_n_fit})."
        )
    if not np.isclose(
        cache.linear_dep_thr,
        result_linear_dep_thr,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "compute_gdf_gradient: rebuilt compcell fit threshold does not "
            "match the converged SCF result."
        )
    if not np.isclose(
        float(cache.lat_opts_2c.cutoff_bohr),
        result_fit_cutoff_2c,
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        float(cache.lat_opts_3c.cutoff_bohr),
        result_fit_cutoff_3c,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "compute_gdf_gradient: rebuilt compcell fit cutoffs do not match "
            "the converged SCF result."
        )

    if is_ks:
        grad, _ = compute_gdf_gradient_rks_gamma(
            system, basis, result,
            aux_basis=aux_basis,
            grid_options=grid_options,
            use_periodic_becke=bool(use_periodic_becke),
            becke_image_radius_bohr=float(becke_image_radius_bohr),
            compcell_eta=float(compcell_eta),
            lattice_opts=lattice_opts,
            cache=cache,
            madelung=result_madelung,
            gauge_lat_opts=gauge_opts,
            v_ne_ke_cutoff=v_ne_ke_cutoff,
        )
        return grad
    grad, _ = compute_gdf_gradient_rhf_gamma(
        system, basis, result,
        aux_basis=aux_basis,
        compcell_eta=float(compcell_eta),
        alpha_hf=1.0,
        lattice_opts=lattice_opts,
        cache=cache,
        madelung=result_madelung,
        gauge_lat_opts=gauge_opts,
        v_ne_ke_cutoff=v_ne_ke_cutoff,
    )
    return grad
