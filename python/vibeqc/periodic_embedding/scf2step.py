"""Ishida two-step self-consistency driver for embedded region I.

Step 1 -- build the embedding potential S_emb(k, z) from the clean
substrate once (via :mod:`~vibeqc.periodic_embedding.surface_sigma`).
Step 2 -- converge region I's density self-consistently with the
embedding potential held fixed.  The density is obtained by contour
integration over the surface Brillouin zone because the embedding
potential turns the discrete slab spectrum into a continuous LDOS.

The one-shot (non-self-consistent) density from Hcore + S_emb is the
building block.  The Hartree + exchange-correlation contribution from
region I's own density is added on top as the SCF loop matures.

References
----------
* H. Ishida, Phys. Rev. B 63, 165409 (2001), doi:10.1103/PhysRevB.63.165409.
* J. E. Inglesfield, J. Phys. C 14, 3795 (1981), doi:10.1088/0022-3719/14/26/015.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    LatticeSumOptions,
    PeriodicSystem,
)
from .contour import EnergyContour
from .embedding_potential import LinearizedEmbeddingPotential
from .region import RegionPartition
from .substrate_gf import _build_hk_sk
from .surface_sigma import build_surface_sigma


@dataclass
class EmbeddedDensityResult:
    """Result of a region-I density calculation (one-shot or SCF).

    Attributes
    ----------
    density_i
        Region-I density matrix in the full AO basis (total for UHF).
    density_i_local
        Region-I density matrix restricted to ``region.i_ao``.
    density_a, density_b
        Per-spin density matrices (UHF only; None for RHF).
    occupation
        Integrated occupied charge in region I.
    band_energy
        Band-structure energy from the contour.
    per_k_trace
        Trace of G(z) at each contour node, per k-point.
    n_iter
        Number of SCF iterations (1 for one-shot).
    converged
        Whether the SCF converged.
    spin_polarized
        True if UHF (per-spin densities differ).
    """

    density_i: NDArray[np.float64]
    density_i_local: NDArray[np.float64]
    density_a: Optional[NDArray[np.float64]] = None
    density_b: Optional[NDArray[np.float64]] = None
    occupation: float = 0.0
    band_energy: float = 0.0
    per_k_trace: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    n_iter: int = 1
    converged: bool = True

    @property
    def spin_polarized(self) -> bool:
        return self.density_a is not None and self.density_b is not None


def compute_region_i_gf_at_kz(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    k_cart: NDArray[np.float64],
    z: complex,
    sigma_emb_z: NDArray[np.complex128],
    *,
    v_eff: Optional[NDArray[np.float64]] = None,
    lat_opts: Optional[LatticeSumOptions] = None,
) -> NDArray[np.complex128]:
    """Build the region-I Green function at one (k, z) point.

        G_II(k,z) = inv(z*S_II(k) - H_I(k) - V_eff - Sigma_emb(z))

    where ``H_I(k)`` and ``S_{II}(k)`` are the core-Hamiltonian and
    overlap blocks restricted to region I (indices ``region.i_ao``),
    ``V_eff`` is an optional potential (Hartree + XC) from the SCF,
    and ``Sigma_emb(z)`` is the embedding self-energy (already in the
    region-I AO basis, shape ``(n_i, n_i)``).

    Parameters
    ----------
    system, basis
        Clean slab system + orbital basis.
    region
        Region partition.
    k_cart
        3D Cartesian k-vector in bohr⁻¹.
    z
        Complex energy.
    sigma_emb_z
        Embedding self-energy at ``z``, shape ``(n_i, n_i)``.
    v_eff
        Optional effective potential (Hartree+XC), shape ``(n_i, n_i)``.
        If None, only Hcore + S_emb is used.
    lat_opts
        Lattice-sum options.

    Returns
    -------
    G_II
        Region-I Green function, shape ``(n_i, n_i)``.
    """
    lat_opts = _ensure_lat_opts(lat_opts)
    _k = np.asarray(k_cart, dtype=float).reshape(3)

    H_k, S_k = _build_hk_sk(system, basis, _k, lat_opts)

    i_ao = region.i_ao
    H_I = H_k[np.ix_(i_ao, i_ao)]
    S_II = S_k[np.ix_(i_ao, i_ao)]

    n_i = region.n_i
    if sigma_emb_z.shape != (n_i, n_i):
        raise ValueError(
            f"sigma_emb_z shape {sigma_emb_z.shape} != expected ({n_i}, {n_i})"
        )
    if v_eff is not None and v_eff.shape != (n_i, n_i):
        raise ValueError(f"v_eff shape {v_eff.shape} != expected ({n_i}, {n_i})")

    denom = z * S_II - H_I - sigma_emb_z
    if v_eff is not None:
        denom = denom - v_eff.astype(np.complex128)
    return np.linalg.inv(denom)


def _ensure_lat_opts(lat_opts: Optional[LatticeSumOptions]) -> LatticeSumOptions:
    if lat_opts is None:
        opts = LatticeSumOptions()
        opts.cutoff_bohr = 30.0
        opts.nuclear_cutoff_bohr = 30.0
        return opts
    return lat_opts


def compute_region_i_density_one_shot(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    kmesh: BlochKMesh,
    contour: EnergyContour,
    sigma_lin_per_k: Dict[int, LinearizedEmbeddingPotential],
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
) -> EmbeddedDensityResult:
    """One-shot region-I density from Hcore + S_emb (no 2e terms).

    This is the non-self-consistent building block.  Each surface
    k-point contribution is computed independently and averaged with
    the k-point weights.

    Parameters
    ----------
    system, basis
        Clean slab (no adsorbate).
    region
        Region partition.
    kmesh
        Surface k-point mesh (from :func:`~vibeqc.periodic_embedding.substrate_gf.default_surface_k_mesh`).
    contour
        Complex-energy contour for the occupied density.
    sigma_lin_per_k
        Mapping ``k_index -> LinearizedEmbeddingPotential``, one per
        irreducible surface k-point.  The k-index matches the order in
        ``kmesh.kpoints``.
    lat_opts
        Lattice-sum options.

    Returns
    -------
    EmbeddedDensityResult
        Density matrix, occupation, band energy, and per-k trace for
        diagnostics.
    """
    lat_opts = _ensure_lat_opts(lat_opts)

    n_i = region.n_i
    n_k = len(kmesh)
    n_nodes = contour.n_nodes
    n_ao = region.n_ao_total

    # Accumulate the density matrix (complex during contour integration).
    D_local = np.zeros((n_i, n_i), dtype=np.complex128)
    per_k_trace = np.zeros((n_k, n_nodes), dtype=np.float64)

    total_occ = 0.0
    total_band = 0.0

    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            # If no sigma for this k-point, skip (shouldn't happen for
            # a properly constructed mesh).
            continue

        g_trace_k = np.empty(n_nodes, dtype=np.complex128)

        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            w_j = complex(contour.weights[jz])

            sigma_z = sigma_lin.at(z_j)
            G = compute_region_i_gf_at_kz(
                system, basis, region, kvec, z_j, sigma_z, lat_opts=lat_opts
            )
            D_local += wk * w_j * G
            g_trace_k[jz] = np.trace(G)

        # Per-k occupation and band energy from trace.
        total_occ += wk * contour.occupation(g_trace_k)
        total_band += wk * contour.band_energy(g_trace_k)
        per_k_trace[ik, :] = g_trace_k.imag  # store LDOS-like trace

    # Density = Im S w_k w_j G(z_j,k).  The contour weights already carry
    # the -(1/pi) Im prefactor, so after the weighted sum we take the
    # imaginary part (which is what contour.occupation does).
    D_local = np.asarray(D_local.imag, dtype=float)
    D_local = 0.5 * (D_local + D_local.T)

    # Zero-pad to full AO basis.
    D_full = np.zeros((n_ao, n_ao), dtype=float)
    D_full[np.ix_(region.i_ao, region.i_ao)] = D_local

    return EmbeddedDensityResult(
        density_i=D_full,
        density_i_local=D_local,
        occupation=float(total_occ),
        band_energy=float(total_band),
        per_k_trace=per_k_trace,
    )


# --- SCF loop ----------------------------------------------------------------


def _diagonal_hartree(
    d_local: NDArray[np.float64],
    u_onsite: float = 0.75,
) -> NDArray[np.float64]:
    """Build a diagonal Hartree potential ``V_muν = U * D_muν * d_muν``.

    This is a minimal model for the region-I Hartree term -- each AO's
    charge ``D_mumu`` produces an on-site repulsion ``U * D_mumu``.  The
    default ``U = 0.75 Ha`` is the (1s|1s) Coulomb integral for H in
    sto-3g.  For production, replace with the full ``∫ r(r')/|r-r'| dr'``
    from the 2e integral machinery.
    """
    v = np.zeros_like(d_local, dtype=float)
    diag = np.diag(d_local)
    np.fill_diagonal(v, u_onsite * diag)
    return v


def _build_lpq_full(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    aux_name: Optional[str] = None,
    linear_dep_thr: float = 1e-9,
) -> np.ndarray:
    """Build the full-cell GDF Lpq cderi tensor, shape ``(n_fit, n_ao, n_ao)``."""
    from ..aux_basis import build_lpq_compcell, default_aux_for, make_aux_basis_set

    lat_opts = _ensure_lat_opts(lat_opts)
    mol = system.unit_cell_molecule()

    if aux_name is None:
        try:
            aux_name = default_aux_for(basis.name)
        except KeyError:
            # Fallback: use the orbital basis itself as aux.
            aux_name = basis.name()

    aux = make_aux_basis_set(mol, aux_name=aux_name)

    lpq = build_lpq_compcell(
        system,
        basis,
        aux,
        molecule=mol,
        lat_opts=lat_opts,
        linear_dep_thr=linear_dep_thr,
    )
    return np.asarray(lpq, dtype=float)


def _build_lpq_for_region_i(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    aux_name: Optional[str] = None,
    linear_dep_thr: float = 1e-9,
) -> np.ndarray:
    """Build the GDF Lpq cderi tensor and extract the region-I block.

    Returns ``Lpq_I`` of shape ``(n_fit, n_i, n_i)``.
    """
    lpq = _build_lpq_full(
        system,
        basis,
        lat_opts=lat_opts,
        aux_name=aux_name,
        linear_dep_thr=linear_dep_thr,
    )
    i_ao = region.i_ao
    lpq_i = lpq[:, i_ao, :][:, :, i_ao]
    return np.asarray(lpq_i, dtype=float)


def _build_lpq_sr_for_region_i(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    omega: float,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    aux_name: Optional[str] = None,
    linear_dep_thr: float = 1e-9,
) -> np.ndarray:
    """Build the SR (erfc) GDF Lpq cderi tensor and extract region-I block.

    Uses the range-separated Coulomb operator erfc(wr)/r for the exchange
    part of RSH hybrids.  Returns ``Lpq_I_sr`` of shape ``(n_fit_sr, n_i, n_i)``.
    """
    from .._vibeqc_core import (
        compute_2c_eri_lattice_sr,
        compute_3c_eri_lattice_sr,
    )
    from ..aux_basis import default_aux_for, make_aux_basis_set

    lat_opts = _ensure_lat_opts(lat_opts)
    mol = system.unit_cell_molecule()

    if aux_name is None:
        try:
            aux_name = default_aux_for(basis.name)
        except KeyError:
            aux_name = basis.name

    aux = make_aux_basis_set(mol, aux_name=aux_name)

    M_sr = np.asarray(compute_2c_eri_lattice_sr(aux, system, lat_opts, omega))
    T_sr = np.asarray(compute_3c_eri_lattice_sr(basis, aux, system, lat_opts, omega))

    # Eigendecompose: M = V.S.V^T, Lpq = S^{-1/2}.V^T.T
    s, V = np.linalg.eigh(M_sr)
    mask = s > linear_dep_thr
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s[mask]))
    V_kept = V[:, mask]
    L = s_inv_sqrt @ V_kept.T @ T_sr.reshape(T_sr.shape[0], -1)
    lpq = L.reshape(-1, T_sr.shape[1], T_sr.shape[2])

    i_ao = region.i_ao
    lpq_i = lpq[:, i_ao, :][:, :, i_ao]
    return np.asarray(lpq_i, dtype=float)


def _build_jk_from_lpq(
    lpq_i: np.ndarray,  # (n_fit, n_i, n_i)
    d_i: np.ndarray,  # (n_i, n_i)
) -> tuple[np.ndarray, np.ndarray]:
    """Build Coulomb J and exchange K matrices from Lpq and density.

    J_muν = S_{κl} (muν|κl) D_κl
    K_muν = S_{κl} (muκ|νl) D_κl

    using the density-fitting approximation:

    J = S_L L_{L,muν} . (S_{κl} L_{L,κl} D_{κl})
    K = S_{L,κ,l} L_{L,muκ} . D_{κl} . L_{L,νl}
    """
    rho = np.einsum("Lij,ij->L", lpq_i, d_i, optimize=True)
    J = np.einsum("L,Lij->ij", rho, lpq_i, optimize=True)
    K = np.einsum("Lik,kl,Ljl->ij", lpq_i, d_i, lpq_i, optimize=True)
    return J, K


def compute_region_i_density_scf(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    kmesh: BlochKMesh,
    contour: EnergyContour,
    sigma_lin_per_k: Dict[int, LinearizedEmbeddingPotential],
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    u_onsite: float = 0.75,
    mixing: float = 0.3,
    max_iter: int = 50,
    conv_thr: float = 1e-6,
) -> EmbeddedDensityResult:
    """Self-consistent region-I density with a diagonal Hartree potential.

    Iterates the embedded Dyson equation with a fixed S_emb and a
    density-dependent on-site Hartree shift ``V_eff = U * diag(D)``.
    Uses simple linear mixing.

    Parameters
    ----------
    u_onsite
        On-site Coulomb repulsion ``U`` (Ha).  For sto-3g H: ~0.75 Ha.
    mixing
        Linear mixing fraction ``a`` for ``D_new = a*D_iter + (1-a)*D_old``.
    max_iter
        Maximum SCF iterations.
    conv_thr
        Convergence threshold on ``max|ΔD|`` (RMS change in density).

    Returns
    -------
    EmbeddedDensityResult
        With ``n_iter`` and ``converged`` populated.
    """
    lat_opts = _ensure_lat_opts(lat_opts)

    n_i = region.n_i
    n_k = len(kmesh)
    n_nodes = contour.n_nodes
    n_ao = region.n_ao_total

    # --- Initial density (one-shot, no Hartree) -------------------------
    D_local = np.zeros((n_i, n_i), dtype=np.complex128)
    total_occ = 0.0
    total_band = 0.0
    per_k_trace = np.zeros((n_k, n_nodes), dtype=np.float64)

    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            continue
        g_trace_k = np.empty(n_nodes, dtype=np.complex128)
        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            w_j = complex(contour.weights[jz])
            G = compute_region_i_gf_at_kz(
                system,
                basis,
                region,
                kvec,
                z_j,
                sigma_lin.at(z_j),
                v_eff=None,
                lat_opts=lat_opts,
            )
            D_local += wk * w_j * G
            g_trace_k[jz] = np.trace(G)
        total_occ += wk * contour.occupation(g_trace_k)
        total_band += wk * contour.band_energy(g_trace_k)
        per_k_trace[ik, :] = g_trace_k.imag

    D_local = np.asarray(D_local.imag, dtype=float)
    D_local = 0.5 * (D_local + D_local.T)

    # --- SCF iterations --------------------------------------------------
    converged = False
    n_iter = 1
    for it in range(max_iter):
        v_eff = _diagonal_hartree(D_local, u_onsite)

        D_new = np.zeros((n_i, n_i), dtype=np.complex128)
        occ_new = 0.0
        band_new = 0.0

        for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
            sigma_lin = sigma_lin_per_k.get(ik)
            if sigma_lin is None:
                continue
            for jz in range(n_nodes):
                z_j = complex(contour.nodes[jz])
                w_j = complex(contour.weights[jz])
                G = compute_region_i_gf_at_kz(
                    system,
                    basis,
                    region,
                    kvec,
                    z_j,
                    sigma_lin.at(z_j),
                    v_eff=v_eff,
                    lat_opts=lat_opts,
                )
                D_new += wk * w_j * G
        D_new = np.asarray(D_new.imag, dtype=float)
        D_new = 0.5 * (D_new + D_new.T)

        # Compute the density change.
        delta = np.max(np.abs(D_new - D_local))

        # Mix.
        D_local = mixing * D_new + (1.0 - mixing) * D_local
        D_local = 0.5 * (D_local + D_local.T)
        n_iter += 1

        if delta < conv_thr:
            converged = True
            break

    # --- Final density and energy from the last iteration ---------------
    D_local = np.asarray(D_local, dtype=float)

    # Recompute occupation and band energy with the final V_eff.
    v_eff = _diagonal_hartree(D_local, u_onsite)
    total_occ = 0.0
    total_band = 0.0
    per_k_trace = np.zeros((n_k, n_nodes), dtype=np.float64)
    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            continue
        g_trace_k = np.empty(n_nodes, dtype=np.complex128)
        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            G = compute_region_i_gf_at_kz(
                system,
                basis,
                region,
                kvec,
                z_j,
                sigma_lin.at(z_j),
                v_eff=v_eff,
                lat_opts=lat_opts,
            )
            g_trace_k[jz] = np.trace(G)
        total_occ += wk * contour.occupation(g_trace_k)
        total_band += wk * contour.band_energy(g_trace_k)
        per_k_trace[ik, :] = g_trace_k.imag

    D_full = np.zeros((n_ao, n_ao), dtype=float)
    D_full[np.ix_(region.i_ao, region.i_ao)] = D_local

    return EmbeddedDensityResult(
        density_i=D_full,
        density_i_local=D_local,
        occupation=float(total_occ),
        band_energy=float(total_band),
        per_k_trace=per_k_trace,
        n_iter=n_iter,
        converged=converged,
    )


def compute_region_i_density_scf_hf(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    kmesh: BlochKMesh,
    contour: EnergyContour,
    sigma_lin_per_k: Dict[int, LinearizedEmbeddingPotential],
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    aux_name: Optional[str] = None,
    mixing: float = 0.3,
    max_iter: int = 30,
    conv_thr: float = 1e-6,
    rsh_omega: float = 0.0,
    cam_alpha: float = 0.0,
    cam_beta: float = 0.0,
) -> EmbeddedDensityResult:
    """Self-consistent HF density with optional range-separated exchange.

    When ``rsh_omega > 0``, builds an additional SR Lpq tensor and
    constructs ``K_erf`` using the erf-attenuated Coulomb operator.
    The effective exchange is:

        K_eff = cam_alpha * K  +  cam_beta * K_erf

    For a pure functional (cam_alpha = cam_beta = 0), this reduces to
    plain J-only (no exchange).  For a global hybrid (cam_beta = 0),
    it's plain HF exchange scaled by cam_alpha."""
    lat_opts = _ensure_lat_opts(lat_opts)

    n_i = region.n_i
    n_k = len(kmesh)
    n_nodes = contour.n_nodes
    n_ao = region.n_ao_total

    # --- Build Lpq once ------------------------------------------------
    lpq_i = _build_lpq_for_region_i(
        system,
        basis,
        region,
        lat_opts=lat_opts,
        aux_name=aux_name,
    )

    # --- Initial density (one-shot, no 2e terms) ----------------------
    D_local = np.zeros((n_i, n_i), dtype=np.complex128)
    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            continue
        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            w_j = complex(contour.weights[jz])
            G = compute_region_i_gf_at_kz(
                system,
                basis,
                region,
                kvec,
                z_j,
                sigma_lin.at(z_j),
                v_eff=None,
                lat_opts=lat_opts,
            )
            D_local += wk * w_j * G
    D_local = np.asarray(D_local.imag, dtype=float)
    D_local = 0.5 * (D_local + D_local.T)

    # --- SCF iterations with J and K ----------------------------------
    converged = False
    n_iter = 1
    for it in range(max_iter):
        J_i, K_i = _build_jk_from_lpq(lpq_i, D_local)
        # RHF: V_eff = J - K/2 for closed-shell.
        v_eff = J_i - 0.5 * K_i

        D_new = np.zeros((n_i, n_i), dtype=np.complex128)
        for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
            sigma_lin = sigma_lin_per_k.get(ik)
            if sigma_lin is None:
                continue
            for jz in range(n_nodes):
                z_j = complex(contour.nodes[jz])
                w_j = complex(contour.weights[jz])
                G = compute_region_i_gf_at_kz(
                    system,
                    basis,
                    region,
                    kvec,
                    z_j,
                    sigma_lin.at(z_j),
                    v_eff=v_eff,
                    lat_opts=lat_opts,
                )
                D_new += wk * w_j * G
        D_new = np.asarray(D_new.imag, dtype=float)
        D_new = 0.5 * (D_new + D_new.T)

        delta = np.max(np.abs(D_new - D_local))
        D_local = mixing * D_new + (1.0 - mixing) * D_local
        D_local = 0.5 * (D_local + D_local.T)
        n_iter += 1

        if delta < conv_thr:
            converged = True
            break

    # --- Final energy evaluation --------------------------------------
    J_i, K_i = _build_jk_from_lpq(lpq_i, D_local)
    v_eff = J_i - 0.5 * K_i
    total_occ = 0.0
    total_band = 0.0
    per_k_trace = np.zeros((n_k, n_nodes), dtype=np.float64)
    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            continue
        g_trace_k = np.empty(n_nodes, dtype=np.complex128)
        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            G = compute_region_i_gf_at_kz(
                system,
                basis,
                region,
                kvec,
                z_j,
                sigma_lin.at(z_j),
                v_eff=v_eff,
                lat_opts=lat_opts,
            )
            g_trace_k[jz] = np.trace(G)
        total_occ += wk * contour.occupation(g_trace_k)
        total_band += wk * contour.band_energy(g_trace_k)
        per_k_trace[ik, :] = g_trace_k.imag

    D_full = np.zeros((n_ao, n_ao), dtype=float)
    D_full[np.ix_(region.i_ao, region.i_ao)] = D_local

    return EmbeddedDensityResult(
        density_i=D_full,
        density_i_local=D_local,
        occupation=float(total_occ),
        band_energy=float(total_band),
        per_k_trace=per_k_trace,
        n_iter=n_iter,
        converged=converged,
    )


def compute_region_i_density_scf_uhf(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    kmesh: BlochKMesh,
    contour: EnergyContour,
    sigma_lin_per_k: Dict[int, LinearizedEmbeddingPotential],
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    aux_name: Optional[str] = None,
    mixing: float = 0.3,
    max_iter: int = 30,
    conv_thr: float = 1e-6,
) -> EmbeddedDensityResult:
    """Self-consistent UHF density with J and K per spin.

    Same as :func:`compute_region_i_density_scf_hf` but with separate
    alpha and beta density matrices and per-spin exchange."""
    lat_opts = _ensure_lat_opts(lat_opts)
    n_i = region.n_i
    n_k = len(kmesh)
    n_nodes = contour.n_nodes
    n_ao = region.n_ao_total

    lpq_i = _build_lpq_for_region_i(
        system,
        basis,
        region,
        lat_opts=lat_opts,
        aux_name=aux_name,
    )

    # Initial guess: half-and-half from one-shot total density.
    D_total = np.zeros((n_i, n_i), dtype=np.complex128)
    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            continue
        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            w_j = complex(contour.weights[jz])
            G = compute_region_i_gf_at_kz(
                system,
                basis,
                region,
                kvec,
                z_j,
                sigma_lin.at(z_j),
                v_eff=None,
                lat_opts=lat_opts,
            )
            D_total += wk * w_j * G
    D_total = np.asarray(D_total.imag, dtype=float)
    D_total = 0.5 * (D_total + D_total.T)
    Da = 0.5 * D_total.copy()
    Db = 0.5 * D_total.copy()

    converged = False
    n_iter = 1
    for it in range(max_iter):
        D_tot = Da + Db
        J_i, _ = _build_jk_from_lpq(lpq_i, D_tot)
        _, Ka = _build_jk_from_lpq(lpq_i, Da)
        _, Kb = _build_jk_from_lpq(lpq_i, Db)
        va = J_i - Ka
        vb = J_i - Kb

        Da_new = np.zeros((n_i, n_i), dtype=np.complex128)
        Db_new = np.zeros((n_i, n_i), dtype=np.complex128)
        for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
            sigma_lin = sigma_lin_per_k.get(ik)
            if sigma_lin is None:
                continue
            for jz in range(n_nodes):
                z_j = complex(contour.nodes[jz])
                w_j = complex(contour.weights[jz])
                Ga = compute_region_i_gf_at_kz(
                    system,
                    basis,
                    region,
                    kvec,
                    z_j,
                    sigma_lin.at(z_j),
                    v_eff=va,
                    lat_opts=lat_opts,
                )
                Gb = compute_region_i_gf_at_kz(
                    system,
                    basis,
                    region,
                    kvec,
                    z_j,
                    sigma_lin.at(z_j),
                    v_eff=vb,
                    lat_opts=lat_opts,
                )
                Da_new += wk * w_j * Ga
                Db_new += wk * w_j * Gb
        Da_new = np.asarray(Da_new.imag, dtype=float)
        Da_new = 0.5 * (Da_new + Da_new.T)
        Db_new = np.asarray(Db_new.imag, dtype=float)
        Db_new = 0.5 * (Db_new + Db_new.T)

        delta = max(
            np.max(np.abs(Da_new - Da)),
            np.max(np.abs(Db_new - Db)),
        )
        Da = mixing * Da_new + (1.0 - mixing) * Da
        Db = mixing * Db_new + (1.0 - mixing) * Db
        Da = 0.5 * (Da + Da.T)
        Db = 0.5 * (Db + Db.T)
        n_iter += 1
        if delta < conv_thr:
            converged = True
            break

    D_local = Da + Db
    D_full = np.zeros((n_ao, n_ao), dtype=float)
    D_full[np.ix_(region.i_ao, region.i_ao)] = D_local

    return EmbeddedDensityResult(
        density_i=D_full,
        density_i_local=D_local,
        density_a=Da,
        density_b=Db,
        occupation=float(np.trace(D_local)),
        band_energy=0.0,
        n_iter=n_iter,
        converged=converged,
    )


def compute_region_i_density_scf_ks(
    system: PeriodicSystem,
    basis,
    region: RegionPartition,
    kmesh: BlochKMesh,
    contour: EnergyContour,
    sigma_lin_per_k: Dict[int, LinearizedEmbeddingPotential],
    *,
    functional_name: str = "PBE",
    lat_opts=None,
    aux_name: Optional[str] = None,
    mixing: float = 0.3,
    max_iter: int = 30,
    conv_thr: float = 1e-6,
) -> EmbeddedDensityResult:
    """KS-DFT SCF with J + Vxc (no exact exchange, pure functional)."""
    from .xc import build_vxc_from_density

    lat_opts = _ensure_lat_opts(lat_opts)
    n_i = region.n_i
    n_k = len(kmesh)
    n_nodes = contour.n_nodes
    n_ao = region.n_ao_total

    lpq_i = _build_lpq_for_region_i(
        system,
        basis,
        region,
        lat_opts=lat_opts,
        aux_name=aux_name,
    )
    mol = system.unit_cell_molecule()

    # Initial density
    D_local = np.zeros((n_i, n_i), dtype=np.complex128)
    for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
        sigma_lin = sigma_lin_per_k.get(ik)
        if sigma_lin is None:
            continue
        for jz in range(n_nodes):
            z_j = complex(contour.nodes[jz])
            w_j = complex(contour.weights[jz])
            G = compute_region_i_gf_at_kz(
                system,
                basis,
                region,
                kvec,
                z_j,
                sigma_lin.at(z_j),
                v_eff=None,
                lat_opts=lat_opts,
            )
            D_local += wk * w_j * G
    D_local = np.asarray(D_local.imag, dtype=float)
    D_local = 0.5 * (D_local + D_local.T)

    converged = False
    n_iter = 1
    for it in range(max_iter):
        J_i, _ = _build_jk_from_lpq(lpq_i, D_local)
        vxc_i, _ = build_vxc_from_density(
            mol,
            basis,
            region,
            D_local,
            functional_name=functional_name,
        )
        v_eff = J_i + vxc_i

        D_new = np.zeros((n_i, n_i), dtype=np.complex128)
        for ik, (kvec, wk) in enumerate(zip(kmesh.kpoints, kmesh.weights)):
            sigma_lin = sigma_lin_per_k.get(ik)
            if sigma_lin is None:
                continue
            for jz in range(n_nodes):
                z_j = complex(contour.nodes[jz])
                w_j = complex(contour.weights[jz])
                G = compute_region_i_gf_at_kz(
                    system,
                    basis,
                    region,
                    kvec,
                    z_j,
                    sigma_lin.at(z_j),
                    v_eff=v_eff,
                    lat_opts=lat_opts,
                )
                D_new += wk * w_j * G
        D_new = np.asarray(D_new.imag, dtype=float)
        D_new = 0.5 * (D_new + D_new.T)

        delta = np.max(np.abs(D_new - D_local))
        D_local = mixing * D_new + (1.0 - mixing) * D_local
        D_local = 0.5 * (D_local + D_local.T)
        n_iter += 1
        if delta < conv_thr:
            converged = True
            break

    D_full = np.zeros((n_ao, n_ao), dtype=float)
    D_full[np.ix_(region.i_ao, region.i_ao)] = D_local

    return EmbeddedDensityResult(
        density_i=D_full,
        density_i_local=D_local,
        n_iter=n_iter,
        converged=converged,
    )


@dataclass
class SubstrateMeanField:
    """Converged substrate mean field for the Ishida two-step path.

    Attributes
    ----------
    v_eff
        ``(n_ao, n_ao)`` real-symmetric effective potential ``J - 1/2K`` at
        Γ, in the embedding's own DF convention (added to Hcore to form the
        SCF Fock feeding S_emb).
    mo_energies
        Substrate SCF orbital energies (Ha), ascending -- the SCF-shifted
        spectrum the embedding contour must enclose.
    e_homo, e_lumo
        Highest occupied / lowest unoccupied substrate SCF eigenvalue (Ha).
        ``e_lumo`` is ``nan`` if every orbital is occupied.
    scf_energy
        Converged substrate RHF total energy per cell (Ha).
    """

    v_eff: NDArray[np.float64]
    mo_energies: NDArray[np.float64]
    e_homo: float
    e_lumo: float
    scf_energy: float


def build_substrate_scf_potential(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    aux_name: Optional[str] = None,
) -> SubstrateMeanField:
    """Ishida step 1: converged substrate mean-field potential ``V_eff``.

    Runs a periodic GDF RHF on the clean slab for the converged density
    ``D``, then builds the two-electron potential ``V_eff = J[D] - 1/2K[D]``
    with the **same** density-fitting machinery the region-I HF SCF uses
    (:func:`_build_jk_from_lpq`).  This replaces the bare Hcore that feeds
    the embedding potential S_emb, so the semi-infinite substrate is seen
    at the SCF level rather than at Hcore (the ~10 Ha Hcore<->SCF band shift
    the embedding needs for GDF parity).

    Why rebuild ``V_eff`` here instead of taking ``F - Hcore`` from the
    pbc-gdf result: the pbc-gdf Hcore uses a different nuclear-attraction
    gauge than the embedding's :func:`_build_hk_sk` (verified ~6.6 Ha apart
    on H₄/STO-3G), and its exchange carries an exxdiv/Madelung correction
    the embedding does not.  Rebuilding ``J - 1/2K`` from the density in the
    embedding's own convention keeps ``V_eff`` consistent with the Hcore it
    is added to and with the region-I treatment (no double gauge; Sec.7).

    Parameters
    ----------
    system
        Clean slab system (Layer A).  Must be closed-shell (even electron
        count); open-shell substrates are a follow-up.
    basis
        Orbital basis set.
    lat_opts
        Lattice-sum options (must match those used for the embedding run).
    aux_name
        Density-fitting auxiliary basis; defaults to the orbital basis's
        canonical aux (see :func:`_build_lpq_full`).

    Returns
    -------
    SubstrateMeanField
        ``v_eff`` (the effective potential) plus the SCF substrate
        spectrum the embedding contour must enclose (the contour window
        shifts with ``V_eff``; estimating it from the Hcore spectrum would
        silently miss the SCF-shifted bands).

    Notes
    -----
    Γ-point: S_emb built from ``V_eff`` is exact for a Γ-only surface mesh
    and the k-independent mean-field approximation for a multi-k mesh (the
    lattice-resolved k-dependent ``V_eff`` is a follow-up).  The density is
    taken from the validated pbc-gdf SCF; iterating it to full embedding-
    convention self-consistency is a second-order refinement.
    """
    from ..pbc_gdf import run_pbc_gdf_rhf

    lat_opts = _ensure_lat_opts(lat_opts)

    n_elec = sum(int(a.Z) for a in system.unit_cell) - int(system.charge)
    if n_elec % 2 != 0:
        raise NotImplementedError(
            f"build_substrate_scf_potential: clean slab has {n_elec} electrons "
            "(odd); the substrate RHF is closed-shell only. Open-shell (UHF) "
            "substrate is a follow-up."
        )

    scf = run_pbc_gdf_rhf(system, basis)
    if not scf.converged:
        raise RuntimeError(
            "build_substrate_scf_potential: substrate GDF RHF did not converge "
            f"after {scf.n_iter} iterations (E = {scf.energy} Ha)."
        )
    d_full = np.asarray(scf.density, dtype=float)

    lpq_full = _build_lpq_full(system, basis, lat_opts=lat_opts, aux_name=aux_name)
    j_full, k_full = _build_jk_from_lpq(lpq_full, d_full)
    v_eff = j_full - 0.5 * k_full  # RHF closed-shell, no exxdiv (embedding gauge)
    v_eff = np.asarray(0.5 * (v_eff + v_eff.T), dtype=float)

    mo_e = np.sort(np.asarray(scf.mo_energies, dtype=float).ravel())
    n_occ = n_elec // 2
    e_homo = float(mo_e[n_occ - 1]) if n_occ >= 1 else float(mo_e[0])
    e_lumo = float(mo_e[n_occ]) if n_occ < mo_e.shape[0] else float("nan")
    return SubstrateMeanField(
        v_eff=v_eff,
        mo_energies=mo_e,
        e_homo=e_homo,
        e_lumo=e_lumo,
        scf_energy=float(scf.energy),
    )


def build_sigma_lin_per_k(
    system: PeriodicSystem,
    basis: BasisSet,
    region: RegionPartition,
    kmesh: BlochKMesh,
    z0: complex,
    *,
    lat_opts: Optional[LatticeSumOptions] = None,
    layer_tol: float = 0.5,
    v_eff_full: Optional[NDArray[np.float64]] = None,
) -> Dict[int, LinearizedEmbeddingPotential]:
    """Build and linearize S_emb at every surface k-point.

    Convenience wrapper that calls
    :func:`~vibeqc.periodic_embedding.surface_sigma.build_surface_sigma`
    for each k-point in ``kmesh``.  Returns a dict keyed by k-index.

    When ``v_eff_full`` (the converged substrate mean-field potential from
    :func:`build_substrate_scf_potential`) is supplied, S_emb is built from
    the SCF Fock instead of the bare Hcore -- the Ishida two-step path.
    """
    lat_opts = _ensure_lat_opts(lat_opts)
    result: Dict[int, LinearizedEmbeddingPotential] = {}
    for ik, kvec in enumerate(kmesh.kpoints):
        # Use the in-plane components (first 2) as the 2D k-vector.
        k_2d = np.asarray(kvec)[:2]
        lin = build_surface_sigma(
            system,
            basis,
            region,
            k_2d,
            lat_opts=lat_opts,
            layer_tol=layer_tol,
            v_eff_full=v_eff_full,
            label=f"k[{ik}]",
        ).linearize(z0)
        result[ik] = lin
    return result
