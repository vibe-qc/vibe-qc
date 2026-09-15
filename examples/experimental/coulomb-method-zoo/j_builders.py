"""Alternative J-builders for the periodic-Gaussian-AO Hartree matrix.

Each builder takes ``(basis, system, D, **kwargs)`` and returns a
``(n_bf, n_bf)`` symmetric J matrix in Hartree.

EWALD3D — reference, vibe-qc release path. erfc(ω·r)/r real-space +
          erf(ω·r)/r FFT-Poisson long-range.
WOLF    — Wolf cutoff Coulomb. Real-space-only, smooth cutoff at R_c.
PLAIN_EWALD — same Ewald split as EWALD3D, but the long-range piece is
          done as an analytic G-vector sum on Gaussian-product Fourier
          transforms (no FFT).
ADFT    — variational Coulomb fitting on an auxiliary basis.

The four methods only differ in how they build J. Exchange K is built
the same way (full-range real-space, ``omega=0``) for all of them so
the comparison is purely about the Coulomb math.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

import vibeqc as vq
from vibeqc import (
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
    build_jk_gamma_molecular_limit,
    compute_2c_eri,
    compute_3c_eri,
)
from vibeqc.ewald_composed import build_j_ewald_3d
from vibeqc.ewald_j import auto_grid


__all__ = [
    "JBuilder",
    "build_j_ewald3d",
    "build_j_wolf",
    "build_j_plain_ewald",
    "build_j_adft",
    "build_k_real_space",
    "build_jk",
]


# Type alias: J-builder takes density, returns J. K is independent.
JBuilder = Callable[[np.ndarray], np.ndarray]


def build_k_real_space(
    basis: BasisSet,
    system: PeriodicSystem,
    lattice_opts: LatticeSumOptions,
    D: np.ndarray,
) -> np.ndarray:
    """Full-range real-space K matrix at Γ (ω=0)."""
    jk = build_jk_gamma_molecular_limit(
        basis, system, lattice_opts, D, 0.0,
    )
    return np.asarray(jk.K)


# ---------------------------------------------------------------------
# Method 1: EWALD3D (reference, vibe-qc release path)
# ---------------------------------------------------------------------

def build_j_ewald3d(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    *,
    lattice_opts: LatticeSumOptions,
    omega: float = 0.5,
    spacing_bohr: float = 0.3,
) -> np.ndarray:
    """vibe-qc release-path Ewald-3D J. Reference."""
    return build_j_ewald_3d(
        basis, system, D, omega,
        lattice_opts=lattice_opts,
        spacing_bohr=spacing_bohr,
    )


# ---------------------------------------------------------------------
# Method 2: WOLF
# ---------------------------------------------------------------------
#
# Wolf et al., JCP 110, 8254 (1999): truncate the real-space Coulomb
# kernel at R_c and apply a smoothed shift so the kernel and its first
# derivative both vanish at R_c. For a damped Coulomb erfc(α·r)/r the
# Wolf form is
#
#     K_W(r) = erfc(α·r)/r - erfc(α·R_c)/R_c     (r < R_c)
#            = 0                                  (r ≥ R_c)
#
# For Gaussian-AO products the J matrix induced by erfc(α·r)/r is what
# vibe-qc's ``build_jk_gamma_molecular_limit(..., omega=α)`` returns
# (its J_SR component). The Wolf shift is constant in r, so it produces
# a scalar correction on the AO overlap matrix:
#
#     ΔJ_μν = -(erfc(α·R_c)/R_c) · ∫χ_μ χ_ν dr · ⟨ρ⟩
#
# where ⟨ρ⟩ = tr(D·S) / V_cell ≈ Q/V_cell is the average density. For
# net-neutral cells the overlap shift cancels the matching nuclear
# Wolf shift; here we drop it (consistent with what vibe-qc's
# Ewald-3D builder does at G=0) and rely on the cell being neutral.
#
# Practically: we approximate Wolf with a real-space-only J at large
# enough α that erfc(α·R_c) ≪ 10⁻⁸ within the lattice cutoff R_c. That
# is exactly ``build_jk_gamma_molecular_limit`` with α = α_wolf.

def build_j_wolf(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    *,
    lattice_opts: LatticeSumOptions,
    alpha_wolf: float = 0.5,
) -> np.ndarray:
    """Wolf-style real-space-only J via erfc(α·r)/r truncation.

    The cutoff comes from ``lattice_opts.cutoff_bohr``; at large enough
    α the residual erfc(α·R_c) is below the tolerance and the shift
    correction is negligible.
    """
    jk = build_jk_gamma_molecular_limit(
        basis, system, lattice_opts, D, float(alpha_wolf),
    )
    return np.asarray(jk.J)


# ---------------------------------------------------------------------
# Method 3: PLAIN_EWALD (analytic reciprocal-space)
# ---------------------------------------------------------------------
#
# Same erf/erfc Ewald split as EWALD3D, but instead of FFT-Poisson on
# a real-space density grid, the long-range piece is computed as a
# direct analytic sum over reciprocal-lattice G-vectors:
#
#     J_LR_μν = (4π / V_cell) Σ_{G ≠ 0} e^{-|G|²/(4ω²)} / |G|²
#                · [Σ_κλ D_κλ ρ̂_κλ(-G)] · ρ̂_μν(G)
#
# where for a contracted Gaussian-AO pair (μ at A_μ exponent α_μ;
# ν at A_ν exponent α_ν), the analytic Fourier transform of the AO
# product is, per primitive pair (s only — sto-3g):
#
#     ρ̂_μν(G) = Σ_{ij} c_μi c_νj * (π / γ_ij)^{3/2} *
#                e^{-α_μi α_νj / γ_ij · |A_μ - A_ν|²} *
#                e^{-|G|²/(4 γ_ij)} *
#                e^{-i G · P_ij}
#
# with γ_ij = α_μi + α_νj, P_ij = (α_μi A_μ + α_νj A_ν) / γ_ij.
# This is the Gaussian-product theorem applied to χ_μ * χ_ν.
#
# **For an AO basis with non-s shells, the analytic FT picks up
# polynomial-in-G factors.** For sto-3g (H, Li, ..., Ne all-s except
# B-Ne which need p), implement s-only first, document the limit.

def build_j_plain_ewald(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    *,
    lattice_opts: LatticeSumOptions,
    omega: float = 0.5,
    g_cutoff_factor: float = 8.0,
) -> np.ndarray:
    """Analytic reciprocal-space Ewald J for s-only Gaussian basis.

    G-cutoff: G_max = g_cutoff_factor · ω. This is the exponent at which
    the e^{-G²/(4ω²)} envelope is below e^{-g_cutoff_factor²} ~ 1e-28
    for g_cutoff_factor=8.
    """
    # Short-range: erfc(ω r)/r real-space, same as EWALD3D's J_SR.
    jk_short = build_jk_gamma_molecular_limit(
        basis, system, lattice_opts, D, float(omega),
    )
    J_sr = np.asarray(jk_short.J)

    # Build the analytic-FT AO density on G vectors.
    # rho_hat_G[G_idx, mu, nu] for each G.
    # The long-range J_μν[D] = (4π/V) * Σ_G [e^{-G²/(4ω²)} / G²]
    #                              * Re[ rho_hat(-G).D ]_κλ * conj(rho_hat_G[G,μ,ν])
    # Because χ_μ χ_ν is real, ρ̂_μν(-G) = conj(ρ̂_μν(G)).
    #
    # Closed form below uses primitive-pair decomposition.
    primitives = _s_only_primitive_info(basis)
    if primitives is None:
        raise NotImplementedError(
            "build_j_plain_ewald: only s-shells supported in the "
            "experimental implementation. Higher angular momentum "
            "needs polynomial-in-G factors."
        )

    lat = np.asarray(system.lattice, dtype=float)
    V_cell = abs(np.linalg.det(lat))

    G_vecs = _build_g_vectors(lat, omega, g_cutoff_factor)
    if G_vecs.size == 0:
        # No G-vectors in cutoff → only G=0, which we drop. J_LR ≡ 0.
        return J_sr

    # Build ρ̂_μν(G) primitive-by-primitive. Memory: (n_G, n_bf, n_bf)
    # complex128. For sto-3g H₂ (n_bf=2, n_G ~ a few hundred) this is
    # tiny; for larger systems we'd batch.
    n_bf = D.shape[0]
    rho_hat = np.zeros((G_vecs.shape[0], n_bf, n_bf), dtype=np.complex128)
    _accum_rho_hat_s(primitives, G_vecs, rho_hat)

    # Long-range Coulomb kernel on G: 4π e^{-G²/(4ω²)} / |G|² (G≠0)
    G_sq = np.sum(G_vecs * G_vecs, axis=1)
    kernel = (4.0 * np.pi) * np.exp(-G_sq / (4.0 * omega**2)) / G_sq

    # ρ̂(G) = Σ_κλ D_κλ ρ̂_κλ(G)  (scalar per G).
    rho_G = np.einsum("Gkl,kl->G", rho_hat, D, optimize=True)
    # J_LR_μν(G) = kernel(G) * conj(ρ̂(G)) * ρ̂_μν(G) summed over G,
    # times 1/V_cell and a factor of 1 for the (4π/V) normalisation
    # already in `kernel`.
    # Note: J is REAL because for every G there is also -G in the
    # cutoff sphere (G_vecs is symmetric); the imaginary parts cancel.
    weighted = kernel * np.conj(rho_G)  # (n_G,)
    J_lr = np.einsum("G,Gmn->mn", weighted, rho_hat, optimize=True)
    J_lr = (J_lr / V_cell).real
    # Symmetrise.
    J_lr = 0.5 * (J_lr + J_lr.T)

    return J_sr + J_lr


def _build_g_vectors(lattice: np.ndarray, omega: float, factor: float) -> np.ndarray:
    """Reciprocal-lattice vectors with |G|/(2ω) ≤ factor, G ≠ 0.

    For an orthorhombic cell with side ``a_i``, b_i = 2π/a_i (1/bohr).
    Build the box of indices (i,j,k) such that the corresponding
    cartesian G has |G| ≤ G_max = 2 ω · factor.
    """
    a = lattice[0, 0]
    b = lattice[1, 1]
    c = lattice[2, 2]
    bx = 2.0 * np.pi / a
    by = 2.0 * np.pi / b
    bz = 2.0 * np.pi / c
    G_max = 2.0 * omega * factor
    n_x = int(np.ceil(G_max / bx))
    n_y = int(np.ceil(G_max / by))
    n_z = int(np.ceil(G_max / bz))
    ix = np.arange(-n_x, n_x + 1)
    iy = np.arange(-n_y, n_y + 1)
    iz = np.arange(-n_z, n_z + 1)
    IX, IY, IZ = np.meshgrid(ix, iy, iz, indexing="ij")
    Gx = IX.ravel() * bx
    Gy = IY.ravel() * by
    Gz = IZ.ravel() * bz
    G = np.column_stack([Gx, Gy, Gz])
    G_sq = np.sum(G * G, axis=1)
    mask = (G_sq > 0.0) & (G_sq <= G_max * G_max)
    return G[mask]


@dataclass(frozen=True)
class _SPrim:
    mu: int
    alpha: float
    coeff: float  # contraction coefficient × normalisation
    center: Tuple[float, float, float]


def _s_only_primitive_info(basis: BasisSet):
    """Return per-AO s-primitive list for analytic Plain Ewald.

    Each entry is a list of ``_SPrim(alpha, coeff, center)`` for an
    s-AO. ``coeff`` is libint's already-normalised primitive contraction
    coefficient (i.e., χ(r) = Σ_i coeff_i * e^{-α_i r²} integrates to
    unit norm without an extra primitive-normalisation factor).

    Returns ``None`` if any non-s shell is encountered.
    """
    shells = list(basis.shells())
    per_ao: list[list[_SPrim]] = []
    mu = 0
    for sh in shells:
        if sh.l != 0:
            return None
        center = tuple(float(x) for x in sh.origin)
        prims = []
        for alpha, coeff in zip(sh.exponents, sh.coefficients):
            prims.append(_SPrim(
                mu=mu, alpha=float(alpha), coeff=float(coeff),
                center=center,
            ))
        per_ao.append(prims)
        mu += 1
    return per_ao


def _accum_rho_hat_s(per_ao: list, G_vecs: np.ndarray,
                     rho_hat: np.ndarray) -> None:
    """Fill ``rho_hat[G, μ, ν]`` for s-only contracted AOs.

    For two contracted s-AOs centered at A_μ and A_ν:
      χ_μ(r)χ_ν(r) = Σ_ij c_μi c_νj e^{-α_μi(r-A_μ)²} e^{-α_νj(r-A_ν)²}
                   = Σ_ij K_ij e^{-γ_ij(r-P_ij)²}
    with γ_ij = α_μi + α_νj,
          P_ij = (α_μi A_μ + α_νj A_ν)/γ_ij,
          K_ij = c_μi c_νj e^{-α_μi α_νj / γ_ij · |A_μ-A_ν|²}.

    FT of e^{-γ(r-P)²}: (π/γ)^{3/2} e^{-G²/(4γ)} e^{-iG·P}.
    """
    n_bf = len(per_ao)
    # Precompute G·G and (loop-invariant) G factors.
    G_sq = np.sum(G_vecs * G_vecs, axis=1)
    for mu in range(n_bf):
        for nu in range(mu, n_bf):
            for pmu in per_ao[mu]:
                for pnu in per_ao[nu]:
                    a, b = pmu.alpha, pnu.alpha
                    gamma = a + b
                    A = np.asarray(pmu.center)
                    B = np.asarray(pnu.center)
                    P = (a * A + b * B) / gamma
                    AB2 = np.sum((A - B) ** 2)
                    K = pmu.coeff * pnu.coeff * np.exp(-a * b / gamma * AB2)
                    pre = K * (np.pi / gamma) ** 1.5
                    env = np.exp(-G_sq / (4.0 * gamma))
                    phase = np.exp(-1j * (G_vecs @ P))
                    contrib = pre * env * phase
                    rho_hat[:, mu, nu] += contrib
                    if nu != mu:
                        rho_hat[:, nu, mu] += np.conj(contrib)


# ---------------------------------------------------------------------
# Method 4: ADFT (variational Coulomb fitting on auxiliary basis)
# ---------------------------------------------------------------------
#
# Given an auxiliary Gaussian basis {ξ_P}, fit the AO density:
#
#     ρ(r) ≈ ρ̃(r) = Σ_P c_P ξ_P(r)
#
# Variational fitting (Mintmire-Dunlap, Köster) minimizes the Coulomb
# self-energy of the residual:
#
#     E_err = ½ ∫∫ (ρ-ρ̃)(r)(ρ-ρ̃)(r')/|r-r'| dr dr' = min over c
#
# Setting dE_err/dc_P = 0:  A c = b  with
#     A_PQ = (P|Q),    b_P = (P|μν) D_μν
#
# The variational Coulomb energy at the fit optimum is
#     E_J = ½ c·b
# (by symmetry this equals ½ ∫∫ ρ̃ ρ̃ / |r-r'| at the optimum), and the
# Fock contribution is
#     J̃_μν = (∂E_J / ∂D_μν) = Σ_P c_P (P|μν)
#
# For PERIODIC at molecular-limit, vibe-qc's molecular DF integrals
# (compute_2c_eri, compute_3c_eri) are sufficient because the AO
# density doesn't overlap its periodic image. For tight bulk this
# would need a periodic 3c-ERI binding.

@dataclass
class ADFTContext:
    """Cached pieces of the ADFT linear system: A^{-1} via Cholesky,
    plus the 3c tensor T_{P,μν} = (P|μν).

    Build once per (orbital, aux) pair. Iterate cheaply per SCF step.
    """
    A_inv_chol: np.ndarray  # lower-triangular Cholesky of A
    T: np.ndarray           # (n_aux, n_orb, n_orb)
    n_aux: int

    @classmethod
    def build(cls, orbital_basis: BasisSet, aux_basis: BasisSet) -> "ADFTContext":
        A = compute_2c_eri(aux_basis)
        # Tikhonov-regularised Cholesky in case of tiny eigenvalues.
        # 1e-12 is a numerical lower bound; this is the same threshold
        # vibeqc.density_fitting uses internally.
        from scipy.linalg import cho_factor
        try:
            cho, lower = cho_factor(A, lower=True)
        except np.linalg.LinAlgError:
            # Reg + retry. Throwaway.
            A = A + 1e-10 * np.eye(A.shape[0])
            cho, lower = cho_factor(A, lower=True)
        T = compute_3c_eri(orbital_basis, aux_basis)
        return cls(A_inv_chol=cho, T=T, n_aux=A.shape[0])


def build_j_adft(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    *,
    ctx: ADFTContext,
) -> np.ndarray:
    """ADFT-fitted J = Σ_P c_P (P|μν), with c solved from A c = b.

    ``ctx`` is cached per-geometry — see ADFTContext.build.
    """
    from scipy.linalg import cho_solve
    # b_P = T_{P,μν} D_{μν}.
    b = np.einsum("Pmn,mn->P", ctx.T, D, optimize=True)
    c = cho_solve((ctx.A_inv_chol, True), b)
    # J̃_μν = T_{P,μν} c_P.
    J = np.einsum("Pmn,P->mn", ctx.T, c, optimize=True)
    # Symmetrise (T is symmetric in (μ,ν), but tiny FP drift possible).
    return 0.5 * (J + J.T)


# ---------------------------------------------------------------------
# Convenience: full JK build for a given method.
# ---------------------------------------------------------------------

def build_jk(
    method: str,
    basis: BasisSet,
    system: PeriodicSystem,
    lattice_opts: LatticeSumOptions,
    D: np.ndarray,
    *,
    omega: float = 0.5,
    spacing_bohr: float = 0.3,
    alpha_wolf: float = 0.5,
    g_cutoff_factor: float = 8.0,
    adft_ctx: Optional[ADFTContext] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build (J, K, timings) at Γ for the given method label.

    K is always full-range real-space. J varies by method.
    Returns timings dict with per-piece walls in seconds.
    """
    timings: dict[str, float] = {}

    t = time.perf_counter()
    K = build_k_real_space(basis, system, lattice_opts, D)
    timings["K"] = time.perf_counter() - t

    t = time.perf_counter()
    if method == "EWALD3D":
        J = build_j_ewald3d(
            basis, system, D,
            lattice_opts=lattice_opts, omega=omega,
            spacing_bohr=spacing_bohr,
        )
    elif method == "WOLF":
        J = build_j_wolf(
            basis, system, D,
            lattice_opts=lattice_opts, alpha_wolf=alpha_wolf,
        )
    elif method == "PLAIN_EWALD":
        J = build_j_plain_ewald(
            basis, system, D,
            lattice_opts=lattice_opts, omega=omega,
            g_cutoff_factor=g_cutoff_factor,
        )
    elif method == "ADFT":
        if adft_ctx is None:
            raise ValueError("ADFT requires an ADFTContext")
        J = build_j_adft(basis, system, D, ctx=adft_ctx)
    else:
        raise ValueError(f"Unknown method '{method}'")
    timings["J"] = time.perf_counter() - t

    return J, K, timings
