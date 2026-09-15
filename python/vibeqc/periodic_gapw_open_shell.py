"""Open-shell (UHF / UKS / ROHF / ROKS) GPW periodic SCF on the smooth FFT grid.

The v0.12 R2 sibling of :func:`vibeqc.periodic_gapw_j.run_periodic_rhf_gpw`
that lifts the closed-shell GPW route to spin-polarised densities.
Closed-shell RHF / RKS forms the periodic AO density as ``D = 2 .
C_occ C_occ^T`` and folds a + b into one matrix; this module keeps
``D_a`` and ``D_b`` separate, builds J from ``D_a + D_b``, and per-
spin K (UHF) or per-spin V_xc (UKS) on the same Γ-point FFT-Poisson
grid the closed-shell driver already uses.

The ROHF and ROKS entry points couple those same per-spin Fock matrices
through the shared Roothaan effective-Fock loop. They retain one spatial
orbital set with integer closed/open/virtual occupations (2/1/0). ROKS also
supports the pure-DFT multi-k GPW envelope; multi-k exact exchange and
electronic smearing are separate follow-ups.

Open-shell conventions match the rest of vibe-qc:

* ``D_s = C_s_occ @ C_s_occ.T`` (no factor of 2; each spin counts its
  own electrons).
* ``D_total = D_a + D_b``.
* ``F_s = Hcore + J(D_total) - a . K(D_s) + V_xc^s`` (UKS).
  Pure UHF drops V_xc and fixes ``a = 1``.
* ``E_elec = tr(D_total . Hcore) + 1/2 tr(D_total . J) - 1/2 a (tr(D_a
  . K_a) + tr(D_b . K_b)) + E_xc``.

K is built via the existing
:func:`vibeqc._vibeqc_core.build_jk_gamma_molecular_limit` per spin --
the same primitive the closed-shell driver uses, fed a per-spin
density and the resulting K divided by 2 (the builder follows the
closed-shell ``2.D`` convention internally; ``K(2.D_s) = 2 K(D_s)``).

J uses :class:`vibeqc.periodic_gapw_j.GpwJBuilder` on the total
density.

V_xc (UKS) is evaluated via libxc's polarised path
(:meth:`Functional.eval_polarised`) on the FFT grid.

**Single-k (Γ-only) for UHF / hybrid UKS.** Pure-DFT multi-k UKS ships via
:func:`run_periodic_uks_gpw_multi_k` (molecular-limit + compact Bloch
regimes, mirroring the closed-shell multi-k driver); multi-k UHF / hybrids
still need per-k exact-exchange builders and remain Γ-only.

**Experimental** -- the GAPW path itself is experimental (see
:mod:`vibeqc.periodic_gapw_j`); open-shell GPW inherits all the
caveats of the closed-shell driver plus the additional fact that
the UKS XC evaluation drops to the unpolarised path when the input
density is bit-for-bit closed shell, which is how the parity-with-
RHF / RKS tests close.

Examples
--------
Hydrogen atom (UHF, single unpaired electron)::

    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core

    L = 16.0
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(1, [L/2, L/2, L/2])]
    mol = vq.Molecule(list(system.unit_cell), 0, 2)  # doublet
    basis = vq.BasisSet(mol, "sto-3g")

    result = vq.run_periodic_uhf_gpw(
        system, basis, n_alpha=1, n_beta=0, cutoff_ha=300.0, quiet=True,
    )
    print(f"H atom UHF/GPW energy = {result.energy:.6f} Ha")
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import _vibeqc_core as _core
from .periodic_gapw_grid import PlaneWaveGrid
from . import periodic_gapw_j as _gapw_j
from .periodic_gapw_j import (
    GpwCollocationCache,
    GpwEnergyBreakdown,
    GpwJBuilder,
    _COMPACT_GPW_LINDEP_THRESHOLD,
    _build_v_ne,
    _collocate_bloch_density_streaming,
    _compute_bloch_kinetic_energy_density_streaming,
    _compute_density_gradient_fft,
    _eigh_safe,
    _kinetic_lattice_gamma,
    _make_diis,
    _multik_gpw_is_molecular_limit,
    _overlap_lattice_gamma,
    _project_potential_to_bloch_ao_streaming,
    _resolve_recip,
    _warn_experimental,
    bloch_ao_on_grid,
    build_gpw_collocation_cache,
    collocate_bloch_density_on_grid,
    collocate_density_on_grid,
    compute_bloch_kinetic_energy_density,
    project_potential_to_ao,
    project_potential_to_bloch_ao,
    project_vtau_to_bloch_ao,
)
from .periodic_screened_exchange import reject_unscreened_range_separated
from .progress import ProgressLogger, resolve_progress
from .rohf import (
    ROHFOptions,
    _make_hf_fock_builder,
    _orthonormaliser,
    _roothaan_occupations,
    roothaan_effective_fock,
    run_roothaan_scf,
)


__all__ = [
    "GpwRohfScfResult",
    "GpwRoksScfResult",
    "GpwUhfScfResult",
    "GpwUksScfResult",
    "GpwUksMultiKScfResult",
    "GpwRoksMultiKScfResult",
    "run_periodic_rohf_gpw",
    "run_periodic_roks_gpw",
    "run_periodic_uhf_gpw",
    "run_periodic_uks_gpw",
    "run_periodic_uks_gpw_multi_k",
    "run_periodic_roks_gpw_multi_k",
    "infer_alpha_beta_from_system",
]


# ============================================================
# Result dataclasses
# ============================================================


@dataclass(frozen=True)
class GpwRohfScfResult:
    """Converged Gamma-point GPW restricted-open-shell HF state.

    The result combines the single-orbital-set ROHF contract with the GPW
    route's auditable energy breakdown and grid.  Alpha and beta expose the
    same spatial orbitals; only their occupations and density matrices differ.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    breakdown: GpwEnergyBreakdown
    density: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_coeffs: np.ndarray
    mo_energies: np.ndarray
    mo_occupations: np.ndarray
    fock: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    overlap: np.ndarray
    n_alpha: int
    n_beta: int
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    s_squared: float
    s_squared_ideal: float
    scf_trace: tuple = ()
    e_dft_plus_u: float = 0.0
    e_xc: float = 0.0
    functional: Optional[str] = None
    method: str = "rohf"

    @property
    def mo_coefficients(self) -> np.ndarray:
        """Long-form alias used by orbital artefact consumers."""
        return self.mo_coeffs

    @property
    def mo_coeffs_alpha(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coeffs_beta(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coefficients_alpha(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coefficients_beta(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_energies_alpha(self) -> np.ndarray:
        return self.mo_energies

    @property
    def mo_energies_beta(self) -> np.ndarray:
        return self.mo_energies

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


@dataclass(frozen=True)
class GpwRoksScfResult(GpwRohfScfResult):
    """Converged Gamma-point GPW restricted-open-shell KS state."""

    method: str = "roks"


@dataclass(frozen=True)
class GpwUhfScfResult:
    """Result of a converged single-point GPW periodic UHF SCF.

    Mirrors :class:`vibeqc.periodic_gapw_j.GpwScfResult` but carries
    per-spin densities, MO coefficients, and MO energies. The
    ``breakdown`` field already reflects the open-shell energy
    accounting; ``energy == breakdown.e_total``.

    Attributes
    ----------
    energy
        Total periodic UHF energy in Hartree (``breakdown.e_total``).
    breakdown
        :class:`GpwEnergyBreakdown` at the converged spin densities.
    density_alpha, density_beta
        ``(n_basis, n_basis)`` converged per-spin AO density matrices
        with the convention ``D_s = C_s_occ C_s_occ^T``.
    mo_coeffs_alpha, mo_coeffs_beta
        ``(n_basis, n_basis)`` MO coefficient matrices per spin.
    mo_energies_alpha, mo_energies_beta
        ``(n_basis,)`` MO eigenvalues per spin (Hartree).
    n_alpha, n_beta
        Number of occupied a / b MOs.
    converged
        True iff both energy and density criteria were satisfied
        within ``max_iter``.
    n_iter
        Number of SCF iterations executed.
    grid
        :class:`PlaneWaveGrid` used for the FFT-Poisson J.
    scf_trace
        Per-iter records (dicts) with the same keys as the closed-
        shell driver, except ``grad_norm`` aggregates both spins.
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    n_alpha: int
    n_beta: int
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    e_dft_plus_u: float = 0.0
    dft_plus_u_sites: tuple = ()
    scf_trace: tuple = ()

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


@dataclass(frozen=True)
class GpwUksScfResult:
    """Result of a converged single-point GPW periodic UKS SCF.

    Same shape as :class:`GpwUhfScfResult`; the only practical
    difference is that ``breakdown.e_xc`` is non-zero and
    ``breakdown.functional`` carries the libxc functional name.
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    n_alpha: int
    n_beta: int
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    e_dft_plus_u: float = 0.0
    dft_plus_u_sites: tuple = ()
    scf_trace: tuple = ()

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


# ============================================================
# Helpers
# ============================================================


def infer_alpha_beta_from_system(
    system,
    *,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve ``(n_alpha, n_beta)`` for a periodic ``system``.

    If the user supplies both, they are validated and returned. If
    only ``system`` is supplied, the electron count comes from
    ``system.n_electrons()`` (including the declared cell charge) and falls
    back to ``n_alpha = n_beta = n_elec // 2`` for an even-electron cell
    (closed-shell-forced-into-UHF). For odd-electron cells the
    caller must supply ``n_alpha`` / ``n_beta`` explicitly.

    Parameters
    ----------
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem`.
    n_alpha, n_beta
        Optional explicit per-spin occupations. When both are
        supplied, ``n_alpha + n_beta`` must equal the cell's
        electron count.
    """
    n_elec = int(system.n_electrons())
    if n_alpha is None and n_beta is None:
        # Default: respect system.multiplicity if it's defined and >1
        # (so a doublet H atom gets n_alpha=1, n_beta=0 from the
        # default path, matching the BIPOLE-UHF convention). Even-
        # electron singlets degenerate to n_alpha = n_beta = n_elec/2.
        mult = int(getattr(system, "multiplicity", 1) or 1)
        if mult < 1:
            raise ValueError(
                f"infer_alpha_beta_from_system: system.multiplicity"
                f" must be >= 1; got {mult}."
            )
        if (n_elec + mult - 1) % 2 != 0 or (n_elec - mult + 1) % 2 != 0:
            raise ValueError(
                f"infer_alpha_beta_from_system: (n_electrons="
                f"{n_elec}, multiplicity={mult}) cannot be split "
                f"into integer alpha / beta counts."
            )
        n_a = (n_elec + mult - 1) // 2
        n_b = (n_elec - mult + 1) // 2
        if n_b < 0:
            raise ValueError(
                f"infer_alpha_beta_from_system: multiplicity={mult}"
                f" is too large for {n_elec} electrons."
            )
        return n_a, n_b
    if n_alpha is None or n_beta is None:
        raise ValueError(
            "infer_alpha_beta_from_system: pass both n_alpha and "
            "n_beta together, or neither (the default falls back to "
            "n_alpha = n_beta = n_elec // 2)."
        )
    n_alpha = int(n_alpha)
    n_beta = int(n_beta)
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(
            f"infer_alpha_beta_from_system: n_alpha={n_alpha}, "
            f"n_beta={n_beta} must be non-negative."
        )
    if n_alpha + n_beta != n_elec:
        raise ValueError(
            f"infer_alpha_beta_from_system: n_alpha + n_beta = "
            f"{n_alpha + n_beta} doesn't match the cell's electron "
            f"count {n_elec}."
        )
    return n_alpha, n_beta


def _evaluate_xc_polarised_on_grid(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    grid: PlaneWaveGrid,
    functional,
    *,
    basis=None,
    density_matrix_alpha=None,
    density_matrix_beta=None,
    tau_alpha: Optional[np.ndarray] = None,
    tau_beta: Optional[np.ndarray] = None,
    cache: Optional[GpwCollocationCache] = None,
) -> tuple:
    """Spin-polarised XC evaluation on the FFT grid.

    Returns ``(e_xc_total, v_xc_a_grid, v_xc_b_grid,
    v_sigma_aa_grid, v_sigma_ab_grid, v_sigma_bb_grid,
    grad_rho_a, grad_rho_b, v_tau_a_grid, v_tau_b_grid)``. The s pieces
    and gradients are ``None`` for LDA; the ``v_tau`` pieces are ``None``
    for LDA/GGA.

    Convention mirrors the closed-shell ``_evaluate_xc_on_grid``:
    libxc's binding returns ``exc`` as the energy density per unit
    volume, so the total exchange-correlation energy is
    ``S_g exc . ΔV`` with no extra r factor. For meta-GGA the caller
    either passes ``basis`` + the per-spin density matrices so the
    per-spin kinetic-energy densities t_s can be built (the Gamma /
    molecular-limit path), or passes precomputed ``tau_alpha=`` /
    ``tau_beta=`` grids directly (the compact multi-k Bloch path, where
    t_s is built from per-k Bloch AO gradients). The von Weizsacker
    floor, constrained chain rule, density screen, and fail-closed
    stiffness guard of the closed-shell path are applied per spin
    channel regardless of how t_s was obtained.

    The ``functional`` must be constructed with ``spin=2``.
    """
    from .periodic_gapw_j import (
        _MGGA_DENSITY_SCREEN,
        _MGGA_VSIGMA_STIFF_MAX,
        _MGGA_VTAU_STIFF_MAX,
        _compute_kinetic_energy_density_fft,
    )

    is_mgga = functional.kind == _core.XCKind.MGGA
    tau_supplied = tau_alpha is not None and tau_beta is not None
    if is_mgga and not tau_supplied and (
        basis is None
        or density_matrix_alpha is None
        or density_matrix_beta is None
    ):
        raise NotImplementedError(
            "open-shell GPW: meta-GGA needs basis + per-spin density "
            "matrices to build t_alpha / t_beta on this call path, or "
            "precomputed tau_alpha= / tau_beta= grids (compact multi-k "
            "Bloch path). Use an LDA/GGA functional here, or the Gamma "
            "open-shell GPW driver / jk_method='gdf'."
        )
    rho_a_flat = np.maximum(rho_a.reshape(-1), 0.0)
    rho_b_flat = np.maximum(rho_b.reshape(-1), 0.0)
    is_gga = functional.kind != _core.XCKind.LDA and not is_mgga

    if is_gga or is_mgga:
        recip = _resolve_recip(cache, grid) if cache is not None else None
        grad_a = _compute_density_gradient_fft(rho_a, grid, recip=recip)
        grad_b = _compute_density_gradient_fft(rho_b, grid, recip=recip)
        sigma_aa_flat = (grad_a ** 2).sum(axis=-1).reshape(-1)
        sigma_ab_flat = (grad_a * grad_b).sum(axis=-1).reshape(-1)
        sigma_bb_flat = (grad_b ** 2).sum(axis=-1).reshape(-1)
    else:
        grad_a = None
        grad_b = None
        sigma_aa_flat = np.zeros_like(rho_a_flat)
        sigma_ab_flat = np.zeros_like(rho_a_flat)
        sigma_bb_flat = np.zeros_like(rho_a_flat)

    v_tau_a_grid = None
    v_tau_b_grid = None
    if is_mgga:
        # Per-spin t with the same regularisation chain as the
        # closed-shell path (periodic_gapw_j._evaluate_xc_on_grid):
        # von Weizsacker floor t_s >= sigma_ss / (8 rho_s), constrained
        # chain rule at clamp-active points, density screen for the
        # uniform grid, and the fail-closed stiffness guard.
        if tau_supplied:
            tau_a = np.asarray(tau_alpha, dtype=float).reshape(-1)
            tau_b = np.asarray(tau_beta, dtype=float).reshape(-1)
        else:
            tau_a = _compute_kinetic_energy_density_fft(
                basis, density_matrix_alpha, grid, cache=cache
            ).reshape(-1)
            tau_b = _compute_kinetic_energy_density_fft(
                basis, density_matrix_beta, grid, cache=cache
            ).reshape(-1)

        def _floor(tau_s, rho_s, sigma_ss):
            tau_w = np.zeros_like(rho_s)
            np.divide(sigma_ss, 8.0 * rho_s, out=tau_w, where=rho_s > 0.0)
            clamp = (tau_s < tau_w) & (rho_s > 0.0)
            return np.where(rho_s > 0.0, np.maximum(tau_s, tau_w), 0.0), clamp

        tau_a_c, clamp_a = _floor(tau_a, rho_a_flat, sigma_aa_flat)
        tau_b_c, clamp_b = _floor(tau_b, rho_b_flat, sigma_bb_flat)

        (
            exc,
            v_rho_a,
            v_rho_b,
            v_sigma_aa,
            v_sigma_ab,
            v_sigma_bb,
            v_tau_a,
            v_tau_b,
        ) = functional.eval_polarised_mgga(
            rho_a_flat,
            rho_b_flat,
            sigma_aa_flat,
            sigma_ab_flat,
            sigma_bb_flat,
            tau_a_c,
            tau_b_c,
        )
        exc = np.asarray(exc, dtype=float)
        v_rho_a = np.asarray(v_rho_a, dtype=float)
        v_rho_b = np.asarray(v_rho_b, dtype=float)
        v_sigma_aa = np.asarray(v_sigma_aa, dtype=float)
        v_sigma_ab = np.asarray(v_sigma_ab, dtype=float)
        v_sigma_bb = np.asarray(v_sigma_bb, dtype=float)
        v_tau_a = np.asarray(v_tau_a, dtype=float)
        v_tau_b = np.asarray(v_tau_b, dtype=float)

        # Constrained chain rule at clamp-active points, per spin (see
        # the closed-shell block for the derivation: there the energy is
        # E(..., t_W(rho_s, sigma_ss)), so the t derivative flows through
        # sigma_ss and rho_s and the direct t channel is zero).
        for clamp, rho_s, sig_ss, vt, vs_name, vr_name in (
            (clamp_a, rho_a_flat, sigma_aa_flat, v_tau_a, "aa", "a"),
            (clamp_b, rho_b_flat, sigma_bb_flat, v_tau_b, "bb", "b"),
        ):
            if not np.any(clamp):
                continue
            rho_c = np.maximum(rho_s, 1e-300)
            if vs_name == "aa":
                v_sigma_aa = np.where(
                    clamp, v_sigma_aa + vt / (8.0 * rho_c), v_sigma_aa
                )
                v_rho_a = np.where(
                    clamp, v_rho_a - vt * sig_ss / (8.0 * rho_c * rho_c), v_rho_a
                )
                v_tau_a = np.where(clamp, 0.0, v_tau_a)
            else:
                v_sigma_bb = np.where(
                    clamp, v_sigma_bb + vt / (8.0 * rho_c), v_sigma_bb
                )
                v_rho_b = np.where(
                    clamp, v_rho_b - vt * sig_ss / (8.0 * rho_c * rho_c), v_rho_b
                )
                v_tau_b = np.where(clamp, 0.0, v_tau_b)

        # Density screen. exc is a TOTAL energy density: zero it only
        # where BOTH spins are below the cutoff -- a fully-polarised
        # region (rho_b == 0 everywhere for a doublet) must keep the
        # majority spin's exchange. Per-spin potential channels screen on
        # their own spin; the cross v_sigma_ab needs both spins present.
        screen_a = rho_a_flat < _MGGA_DENSITY_SCREEN
        screen_b = rho_b_flat < _MGGA_DENSITY_SCREEN
        exc[screen_a & screen_b] = 0.0
        v_sigma_ab[screen_a | screen_b] = 0.0
        v_rho_a[screen_a] = 0.0
        v_sigma_aa[screen_a] = 0.0
        v_tau_a[screen_a] = 0.0
        v_rho_b[screen_b] = 0.0
        v_sigma_bb[screen_b] = 0.0
        v_tau_b[screen_b] = 0.0

        # Fail-closed stiffness guard (same bounds as closed shell).
        max_vs = max(
            float(np.abs(v_sigma_aa).max()) if v_sigma_aa.size else 0.0,
            float(np.abs(v_sigma_ab).max()) if v_sigma_ab.size else 0.0,
            float(np.abs(v_sigma_bb).max()) if v_sigma_bb.size else 0.0,
        )
        max_vt = max(
            float(np.abs(v_tau_a).max()) if v_tau_a.size else 0.0,
            float(np.abs(v_tau_b).max()) if v_tau_b.size else 0.0,
        )
        if max_vs > _MGGA_VSIGMA_STIFF_MAX or max_vt > _MGGA_VTAU_STIFF_MAX:
            raise NotImplementedError(
                "open-shell GPW meta-GGA: the XC potential on the uniform "
                f"grid is stiffness-dominated (max|v_sigma| = {max_vs:.2e}, "
                f"max|v_tau| = {max_vt:.2e}) -- this functional's "
                "iso-orbital-boundary derivatives diverge at orbital "
                "critical points. Use a self-regularising meta-GGA "
                "(r2scan / scan) or jk_method='gdf'."
            )
        v_tau_a_grid = v_tau_a.reshape(grid.shape)
        v_tau_b_grid = v_tau_b.reshape(grid.shape)
    else:
        (exc, v_rho_a, v_rho_b,
         v_sigma_aa, v_sigma_ab, v_sigma_bb) = functional.eval_polarised(
            rho_a_flat, rho_b_flat,
            sigma_aa_flat, sigma_ab_flat, sigma_bb_flat,
        )
    dv = grid.voxel_volume_bohr3
    e_xc_total = float(np.asarray(exc).sum() * dv)

    v_xc_a_grid = np.asarray(v_rho_a).reshape(grid.shape)
    v_xc_b_grid = np.asarray(v_rho_b).reshape(grid.shape)
    if is_gga or is_mgga:
        v_sigma_aa_grid = np.asarray(v_sigma_aa).reshape(grid.shape)
        v_sigma_ab_grid = np.asarray(v_sigma_ab).reshape(grid.shape)
        v_sigma_bb_grid = np.asarray(v_sigma_bb).reshape(grid.shape)
    else:
        v_sigma_aa_grid = None
        v_sigma_ab_grid = None
        v_sigma_bb_grid = None
    return (
        e_xc_total,
        v_xc_a_grid, v_xc_b_grid,
        v_sigma_aa_grid, v_sigma_ab_grid, v_sigma_bb_grid,
        grad_a, grad_b,
        v_tau_a_grid, v_tau_b_grid,
    )


def _project_vxc_polarised_to_ao(
    basis,
    spin: str,
    v_xc_grid: np.ndarray,
    grid: PlaneWaveGrid,
    *,
    v_sigma_self: Optional[np.ndarray] = None,
    v_sigma_cross: Optional[np.ndarray] = None,
    grad_rho_self: Optional[np.ndarray] = None,
    grad_rho_other: Optional[np.ndarray] = None,
    cache: Optional[GpwCollocationCache] = None,
) -> np.ndarray:
    """Project the per-spin V_xc onto the AO basis.

    For LDA: ``V_xc_s[muν] = ∫ chi_mu chi_ν . v_r_s dr``.

    For GGA the gradient piece is

        ``V_xc_s_GGA[muν] = - ∫ chi_mu chi_ν . grad.(2 v_ss gradr_s
                                            + v_ss' gradr_s') dr``

    where ``s'`` is the opposite spin. Implemented via FFT
    divergence + the same AO projection primitive as the closed-
    shell GGA path.

    Symmetry is enforced at the end.
    """
    V_lda = project_potential_to_ao(basis, v_xc_grid, grid, cache=cache)
    if v_sigma_self is None or grad_rho_self is None:
        return V_lda
    # GGA: flux = (2 v_ss gradr_s + v_ss' gradr_s')
    flux = 2.0 * v_sigma_self[..., None] * grad_rho_self
    if v_sigma_cross is not None and grad_rho_other is not None:
        flux = flux + v_sigma_cross[..., None] * grad_rho_other
    divergence = np.zeros(grid.shape, dtype=float)
    G = _resolve_recip(cache, grid)
    for d in range(3):
        flux_k = np.fft.fftn(flux[..., d])
        divergence += np.real(np.fft.ifftn(1j * G[..., d] * flux_k))
    V_gga = -project_potential_to_ao(basis, divergence, grid, cache=cache)
    V_total = V_lda + V_gga
    return 0.5 * (V_total + V_total.T)


def _k_per_spin(basis, system, D_spin: np.ndarray, lat_opts) -> np.ndarray:
    """Per-spin K matrix via the closed-shell builder convention.

    ``build_jk_gamma_molecular_limit`` implements the closed-shell
    convention internally -- given a density D it returns K(D)
    consistent with ``F = h + J - 1/2 K`` and ``D = 2 P``. For an
    open-shell per-spin density ``D_s = C_s C_s^T`` (no factor 2),
    the corresponding open-shell K (the one entering
    ``F_s = h + J - K_s``) is ``K(2.D_s) / 2``.
    """
    jk = _core.build_jk_gamma_molecular_limit(
        basis, system, lat_opts, 2.0 * D_spin,
    )
    return 0.5 * np.asarray(jk.K)


# ============================================================
# UHF / UKS drivers
# ============================================================


def _run_open_shell_gpw(
    system,
    basis,
    *,
    n_alpha: Optional[int],
    n_beta: Optional[int],
    grid: Optional[PlaneWaveGrid],
    cutoff_ha: Optional[float],
    max_iter: int,
    conv_tol_energy: float,
    conv_tol_density: float,
    damping: float,
    initial_density: Optional[tuple],
    initial_guess: Optional[Union[str, _core.InitialGuess]],
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str,
    smearing_alpha: Optional[float],
    functional: Optional[str],
    dft_plus_u_sites: Optional[object],
    use_diis: bool,
    diis_subspace_size: int,
    diis_start_iter: int,
    quiet: bool,
    label: str,
    progress: Union[bool, ProgressLogger, None] = None,
):
    """Shared open-shell SCF body -- called by both UHF and UKS
    entry points with ``functional=None`` selecting pure UHF.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    from .guess import _coerce_periodic_driver_guess

    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver=label,
        supported=periodic_guess_capabilities('gpw', 'UHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
    )
    if initial_density is not None:
        from .guess_read import _real_density
        if len(initial_density) != 2:
            raise ValueError("READ: expected an alpha/beta density pair")
        initial_density = tuple(_real_density(d) for d in initial_density)


    if not quiet:
        _warn_experimental(
            f"{label}: minimal Python open-shell SCF loop, Γ-only "
            f"(pure-DFT multi-k UKS ships separately via "
            f"run_periodic_uks_gpw_multi_k; multi-k UHF / hybrids need "
            f"per-k exact exchange and remain Γ-only)"
        )

    plog = resolve_progress(progress)

    if system.dim != 3:
        raise ValueError(
            f"{label}: only dim == 3 is supported (got dim="
            f"{system.dim}); 1D/2D open-shell GPW is later work."
        )

    n_alpha_resolved, n_beta_resolved = infer_alpha_beta_from_system(
        system, n_alpha=n_alpha, n_beta=n_beta,
    )

    # Build grid if needed.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid
        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(
            np.asarray(system.lattice, dtype=float),
            cutoff_ha=cutoff_ha,
        )

    # One-electron integrals -- same gauge as the closed-shell driver.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(basis, system, v_ne_convention, smearing_alpha, grid)
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne

    # Ewald nuclear repulsion.
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    # Dudarev DFT+U setup. Open shell uses the per-spin convention
    # directly: E_U = E_U_alpha + E_U_beta and each spin Fock receives
    # S V_AO_spin S.
    from .dft_plus_u import (
        _v_ao_per_spin as _dftu_v_ao_per_spin,
        ao_group_indices as _dftu_ao_group_indices,
        compute_dudarev_energy as _dftu_compute_dudarev_energy,
        compute_occupation_matrices as _dftu_compute_occupation_matrices,
    )

    _dftu_sites = tuple(dft_plus_u_sites or ())
    if _dftu_sites:
        _dftu_ao_groups = _dftu_ao_group_indices(basis)
        for _site in _dftu_sites:
            _key = (int(_site.atom_index), int(_site.l))
            if _key not in _dftu_ao_groups:
                raise ValueError(
                    f"{label}: HubbardSite{_key} has no AOs in the basis. "
                    f"Available channels: {sorted(_dftu_ao_groups.keys())}"
                )
    else:
        _dftu_ao_groups = {}

    # Builders. Build chi (+ G-mesh) once and share it across the J builder
    # and every per-iteration XC collocate / project below (UKS rebuilds chi
    # ~4x/iter -- r_a, r_b, V_xc_a, V_xc_b -- all from this one table now).
    collocation_cache = build_gpw_collocation_cache(basis, grid)
    gpw = GpwJBuilder(basis, grid, collocation_cache=collocation_cache)
    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0

    # Optional libxc functional (spin=2 for the polarised path).
    if functional is not None:
        func = _core.Functional(functional, 2)
        # Full-range-only per-spin K on this route.
        reject_unscreened_range_separated(
            func, where="_run_open_shell_gpw"
        )
        ex_frac = float(func.hf_exchange_fraction)
    else:
        func = None
        ex_frac = 1.0  # Pure UHF

    # Canonical-orthonormaliser S^{-1/2}.
    s_eigs, U = _eigh_safe(S)
    S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T

    # Density-mode guesses use the shared periodic adapter. An explicit
    # restart wins and HCORE deliberately retains this route's local fallback.
    if initial_density is None:
        from .guess import initial_densities_open_shell

        initial_density = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha_resolved,
            n_beta_resolved,
            (_core.InitialGuess.SAD if initial_guess == _core.InitialGuess.PATOM
             else initial_guess),
            is_periodic=True,
            periodic_system=system,
            lattice_opts=_core.LatticeSumOptions(),
            overlap=S,
                              atomic_spins=atomic_spins,
        )
    if initial_density is not None:
        D_alpha_init, D_beta_init = initial_density
        D_alpha = np.asarray(D_alpha_init, dtype=float).copy()
        D_beta = np.asarray(D_beta_init, dtype=float).copy()
        if D_alpha.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density[alpha] shape {D_alpha.shape} "
                f"doesn't match basis ({basis.nbasis}, {basis.nbasis})"
            )
        if D_beta.shape != (basis.nbasis, basis.nbasis):
            raise ValueError(
                f"initial_density[beta] shape {D_beta.shape} "
                f"doesn't match basis ({basis.nbasis}, {basis.nbasis})"
            )
    else:
        # Hcore guess -- diagonalise Hcore and Aufbau-fill per spin.
        # Symmetry-broken initial guess: bias a / b slightly off the
        # closed-shell solution when n_alpha != n_beta so the SCF
        # doesn't get trapped at the symmetric saddle. For
        # n_alpha == n_beta we leave the densities equal -- the UHF
        # solution reduces to the RHF one bit-for-bit, which is the
        # parity-with-RHF test pin.
        e_init, C_orth_init = _eigh_safe(S_half_inv @ Hcore @ S_half_inv)
        C_init = S_half_inv @ C_orth_init
        D_alpha = (
            C_init[:, :n_alpha_resolved]
            @ C_init[:, :n_alpha_resolved].T
        )
        D_beta = (
            C_init[:, :n_beta_resolved]
            @ C_init[:, :n_beta_resolved].T
        )

    from .guess import normalize_spin_density_guess
    D_alpha, D_beta = normalize_spin_density_guess(
        D_alpha, D_beta, S, n_alpha_resolved, n_beta_resolved,
    )

    if initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_spin_density_step
        D_alpha, D_beta = patom_spin_density_step(
            D_alpha, D_beta, Hcore, S, n_alpha_resolved, n_beta_resolved,
            gpw.build_J, lambda d: _k_per_spin(basis, system, d, lo),
        )

    # Pulay-DIIS state (one joint history couples the spins).
    diis = _make_diis(use_diis, diis_subspace_size)

    # Per-iter SCF trace.
    scf_trace: list[dict] = []

    # SCF iteration.
    E_prev = 0.0
    converged = False
    n_iter = 0
    n_basis = basis.nbasis
    C_alpha = np.zeros((n_basis, n_basis))
    C_beta = np.zeros((n_basis, n_basis))
    e_a = np.zeros(n_basis)
    e_b = np.zeros(n_basis)
    e_xc_iter = 0.0

    for it in range(1, max_iter + 1):
        D_total = D_alpha + D_beta
        J = gpw.build_J(D_total)

        # Per-spin K (open-shell-K convention).
        if ex_frac > 0.0:
            K_alpha = _k_per_spin(basis, system, D_alpha, lo)
            K_beta = _k_per_spin(basis, system, D_beta, lo)
        else:
            K_alpha = np.zeros_like(D_alpha)
            K_beta = np.zeros_like(D_beta)

        F_alpha = Hcore + J - ex_frac * K_alpha
        F_beta = Hcore + J - ex_frac * K_beta

        e_dft_plus_u_iter = 0.0
        if _dftu_sites:
            n_alpha_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                D_alpha,
                S,
                _dftu_ao_groups,
            )
            n_beta_map = _dftu_compute_occupation_matrices(
                _dftu_sites,
                D_beta,
                S,
                _dftu_ao_groups,
            )
            e_dft_plus_u_iter = float(
                _dftu_compute_dudarev_energy(_dftu_sites, n_alpha_map)
                + _dftu_compute_dudarev_energy(_dftu_sites, n_beta_map)
            )
            V_U_alpha = S @ _dftu_v_ao_per_spin(
                _dftu_sites,
                D_alpha,
                S,
                _dftu_ao_groups,
            ) @ S
            V_U_beta = S @ _dftu_v_ao_per_spin(
                _dftu_sites,
                D_beta,
                S,
                _dftu_ao_groups,
            ) @ S
            F_alpha = F_alpha + V_U_alpha
            F_beta = F_beta + V_U_beta

        # XC piece (UKS). All grid primitives reuse the per-SCF chi cache.
        if func is not None:
            rho_a_grid = collocate_density_on_grid(
                basis, D_alpha, grid, cache=collocation_cache
            )
            rho_b_grid = collocate_density_on_grid(
                basis, D_beta, grid, cache=collocation_cache
            )
            (
                e_xc_iter,
                v_xc_a_grid, v_xc_b_grid,
                v_sigma_aa_g, v_sigma_ab_g, v_sigma_bb_g,
                grad_a, grad_b,
                v_tau_a_g, v_tau_b_g,
            ) = _evaluate_xc_polarised_on_grid(
                rho_a_grid, rho_b_grid, grid, func,
                basis=basis,
                density_matrix_alpha=D_alpha,
                density_matrix_beta=D_beta,
                cache=collocation_cache,
            )
            V_xc_alpha = _project_vxc_polarised_to_ao(
                basis, "alpha", v_xc_a_grid, grid,
                v_sigma_self=v_sigma_aa_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_a,
                grad_rho_other=grad_b,
                cache=collocation_cache,
            )
            V_xc_beta = _project_vxc_polarised_to_ao(
                basis, "beta", v_xc_b_grid, grid,
                v_sigma_self=v_sigma_bb_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_b,
                grad_rho_other=grad_a,
                cache=collocation_cache,
            )
            # Meta-GGA: per-spin t Fock term 1/2 int v_tau_s gradchi.gradchi
            # (the generalized-KS matrix element; see
            # periodic_gapw_j._project_vtau_to_ao).
            if v_tau_a_g is not None:
                from .periodic_gapw_j import _project_vtau_to_ao

                V_xc_alpha = V_xc_alpha + _project_vtau_to_ao(
                    basis, v_tau_a_g, grid, cache=collocation_cache
                )
                V_xc_beta = V_xc_beta + _project_vtau_to_ao(
                    basis, v_tau_b_g, grid, cache=collocation_cache
                )
            F_alpha = F_alpha + V_xc_alpha
            F_beta = F_beta + V_xc_beta
        else:
            e_xc_iter = 0.0

        # Electronic energy (open-shell convention; see module docstring).
        # E_elec = tr(D_total . Hcore)
        #          + 1/2 tr(D_total . J)
        #          - 1/2 ex_frac (tr(D_alpha . K_alpha) + tr(D_beta . K_beta))
        #          + e_xc
        E_core = float(np.einsum("ij,ij->", D_total, Hcore))
        E_J = 0.5 * float(np.einsum("ij,ij->", D_total, J))
        E_K_a = float(np.einsum("ij,ij->", D_alpha, K_alpha))
        E_K_b = float(np.einsum("ij,ij->", D_beta, K_beta))
        E_K = -0.5 * ex_frac * (E_K_a + E_K_b)
        E_elec = E_core + E_J + E_K + e_xc_iter
        E = E_elec + E_nn
        E = E + e_dft_plus_u_iter

        # DIIS error per spin -- r_s = F_s D_s S - S D_s F_s in the
        # canonical-orthonormal basis. Stacking α and β makes the kernel
        # accumulate the joint B-matrix Tr(e_α_iᵀ e_α_j) + Tr(e_β_iᵀ e_β_j),
        # so a single coefficient set drives both Fock matrices.
        if diis is not None and it >= diis_start_iter:
            err_a = F_alpha @ D_alpha @ S - S @ D_alpha @ F_alpha
            err_a = S_half_inv @ err_a @ S_half_inv
            err_b = F_beta @ D_beta @ S - S @ D_beta @ F_beta
            err_b = S_half_inv @ err_b @ S_half_inv
            F_alpha, F_beta = diis.extrapolate_spin_coupled(
                F_alpha, F_beta, err_a, err_b
            )

        # Solve F_s C_s = e_s S C_s per spin in the orthonormal basis.
        e_a, C_orth_a = _eigh_safe(S_half_inv @ F_alpha @ S_half_inv)
        C_alpha = S_half_inv @ C_orth_a
        e_b, C_orth_b = _eigh_safe(S_half_inv @ F_beta @ S_half_inv)
        C_beta = S_half_inv @ C_orth_b

        # Aufbau per spin.
        if n_alpha_resolved > 0:
            D_alpha_new = (
                C_alpha[:, :n_alpha_resolved]
                @ C_alpha[:, :n_alpha_resolved].T
            )
        else:
            D_alpha_new = np.zeros_like(D_alpha)
        if n_beta_resolved > 0:
            D_beta_new = (
                C_beta[:, :n_beta_resolved]
                @ C_beta[:, :n_beta_resolved].T
            )
        else:
            D_beta_new = np.zeros_like(D_beta)

        if damping > 0.0:
            D_alpha_new = (1.0 - damping) * D_alpha_new + damping * D_alpha
            D_beta_new = (1.0 - damping) * D_beta_new + damping * D_beta

        dE = E - E_prev
        dD = float(
            np.sqrt(
                np.linalg.norm(D_alpha_new - D_alpha) ** 2
                + np.linalg.norm(D_beta_new - D_beta) ** 2
            )
        )
        n_iter = it
        scf_trace.append({
            "iter": it, "energy": float(E), "delta_e": float(dE),
            "grad_norm": float(dD), "e_xc": float(e_xc_iter),
            "e_dft_plus_u": float(e_dft_plus_u_iter),
        })
        # Per-iteration progress hook -> live QVF checkpoint cadence when the
        # runner passes a checkpoint-wrapped ``plog`` (see run_periodic_rhf_gpw).
        plog.iteration(it, energy=float(E), dE=float(dE), grad=float(dD))
        if (abs(dE) < conv_tol_energy
                and dD < conv_tol_density
                and it > 1):
            converged = True
            D_alpha = D_alpha_new
            D_beta = D_beta_new
            break
        D_alpha = D_alpha_new
        D_beta = D_beta_new
        E_prev = E

    # Final breakdown at the converged densities (open-shell variant).
    D_total = D_alpha + D_beta
    J = gpw.build_J(D_total)
    if ex_frac > 0.0:
        K_alpha = _k_per_spin(basis, system, D_alpha, lo)
        K_beta = _k_per_spin(basis, system, D_beta, lo)
    else:
        K_alpha = np.zeros_like(D_alpha)
        K_beta = np.zeros_like(D_beta)
    if func is not None:
        rho_a_grid = collocate_density_on_grid(
            basis, D_alpha, grid, cache=collocation_cache
        )
        rho_b_grid = collocate_density_on_grid(
            basis, D_beta, grid, cache=collocation_cache
        )
        (e_xc_final, *_rest) = _evaluate_xc_polarised_on_grid(
            rho_a_grid, rho_b_grid, grid, func,
            basis=basis,
            density_matrix_alpha=D_alpha,
            density_matrix_beta=D_beta,
            cache=collocation_cache,
        )
    else:
        e_xc_final = 0.0

    E_kin = float(np.einsum("ij,ij->", D_total, T))
    E_ne = float(np.einsum("ij,ij->", D_total, V_ne))
    E_H = 0.5 * float(np.einsum("ij,ij->", D_total, J))
    E_K_final = -0.5 * ex_frac * (
        float(np.einsum("ij,ij->", D_alpha, K_alpha))
        + float(np.einsum("ij,ij->", D_beta, K_beta))
    )
    e_dft_plus_u_final = 0.0
    if _dftu_sites:
        n_alpha_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            D_alpha,
            S,
            _dftu_ao_groups,
        )
        n_beta_final = _dftu_compute_occupation_matrices(
            _dftu_sites,
            D_beta,
            S,
            _dftu_ao_groups,
        )
        e_dft_plus_u_final = float(
            _dftu_compute_dudarev_energy(_dftu_sites, n_alpha_final)
            + _dftu_compute_dudarev_energy(_dftu_sites, n_beta_final)
        )
    e_total = (
        E_kin + E_ne + E_H + E_K_final + e_xc_final
        + e_dft_plus_u_final + E_nn
    )

    breakdown = GpwEnergyBreakdown(
        e_kinetic=E_kin,
        e_nuclear_attraction=E_ne,
        e_hartree=E_H,
        e_hf_exchange=E_K_final,
        e_nuclear_repulsion=E_nn,
        e_total=e_total,
        grid=grid,
        e_xc=float(e_xc_final),
        functional=str(functional) if functional is not None else None,
        e_dft_plus_u=e_dft_plus_u_final,
    )

    return (
        breakdown,
        D_alpha, D_beta,
        C_alpha, C_beta,
        e_a, e_b,
        n_alpha_resolved, n_beta_resolved,
        converged, n_iter, grid,
        tuple(scf_trace),
    )


def _run_periodic_restricted_open_gpw(
    system,
    basis,
    *,
    functional: Optional[str],
    label: str,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_grad: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    linear_dep_threshold: float = 1e-8,
    level_shift: float = 0.0,
    fractional_open_shell: bool = False,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
) -> Union[GpwRohfScfResult, GpwRoksScfResult]:
    """Restricted-open-shell HF/KS implementation for the Gamma GPW route.

    Hartree J, per-spin exact exchange, the one-electron Hamiltonian, and
    nuclear repulsion use the same GPW/Ewald machinery as
    :func:`run_periodic_uhf_gpw`.  The two spin Focks are coupled through
    the shared Roothaan single-effective-Fock loop, producing one spatial
    orbital set with closed/open/virtual occupations ``2/1/0``.

    This route is currently limited to three-dimensional Gamma-point HF/KS.
    ``smearing_alpha`` controls the existing nuclear-charge smoothing
    convention; it is not electronic Fermi smearing. ``initial_guess="AUTO"``
    resolves to SAD through the shared periodic guess engine. Explicit
    selectors use the same capability validation as the public runner.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    input_restart_supplied = initial_density is not None
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver=label,
        supported=periodic_guess_capabilities('gpw', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        restart_supplied=initial_density is not None,
    )
    if initial_density is not None:
        from .guess_read import _real_density
        if len(initial_density) != 2:
            raise ValueError("READ: expected an alpha/beta density pair")
        initial_density = tuple(_real_density(d) for d in initial_density)


    if not quiet:
        _warn_experimental(
            f"{label}: restricted-open-shell GPW SCF, Gamma-only"
        )

    plog = resolve_progress(progress)
    if system.dim != 3:
        raise ValueError(
            f"{label}: only dim == 3 is supported (got dim={system.dim}); "
            "1D/2D ROHF GPW is later work."
        )

    n_alpha_resolved, n_beta_resolved = infer_alpha_beta_from_system(
        system,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
    if n_alpha_resolved < n_beta_resolved:
        raise ValueError(
            f"{label}: restricted-open-shell SCF requires n_alpha >= n_beta; got "
            f"n_alpha={n_alpha_resolved}, n_beta={n_beta_resolved}."
        )

    func = None
    ex_frac = 1.0
    if functional is not None:
        if not functional:
            raise ValueError(f"{label}: functional is required for periodic ROKS.")
        func = _core.Functional(str(functional), 2)
        reject_unscreened_range_separated(func, where=label)
        if getattr(func, "is_double_hybrid", False):
            raise NotImplementedError(
                f"{label}: a direct periodic ROKS double-hybrid calculation "
                "would omit its perturbative correlation term. Use a "
                "non-double-hybrid functional."
            )
        ex_frac = float(func.hf_exchange_fraction)

    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(
            np.asarray(system.lattice, dtype=float),
            cutoff_ha=cutoff_ha,
        )

    # Keep the one-electron and nuclear terms in the same periodic gauge as
    # the established RHF/UHF GPW routes.
    kinetic = _kinetic_lattice_gamma(basis, system)
    nuclear_attraction = _build_v_ne(
        basis,
        system,
        v_ne_convention,
        smearing_alpha,
        grid,
    )
    overlap = _overlap_lattice_gamma(basis, system)
    hcore = kinetic + nuclear_attraction
    e_nuclear = float(
        _core.ewald_nuclear_repulsion(system, _core.EwaldOptions())
    )

    collocation_cache = build_gpw_collocation_cache(basis, grid)
    gpw = GpwJBuilder(
        basis,
        grid,
        collocation_cache=collocation_cache,
    )
    lattice_options = _core.LatticeSumOptions()
    lattice_options.cutoff_bohr = 25.0

    def build_k(density_spin: np.ndarray) -> np.ndarray:
        return _k_per_spin(basis, system, density_spin, lattice_options)

    if func is None:
        fock_builder = _make_hf_fock_builder(
            hcore,
            gpw.build_J,
            build_k,
        )
    else:

        def fock_builder(
            density_alpha: np.ndarray,
            density_beta: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
            density_total = density_alpha + density_beta
            coulomb = gpw.build_J(density_total)
            if ex_frac > 0.0:
                exchange_alpha = build_k(density_alpha)
                exchange_beta = build_k(density_beta)
            else:
                exchange_alpha = np.zeros_like(density_alpha)
                exchange_beta = np.zeros_like(density_beta)

            rho_alpha = collocate_density_on_grid(
                basis, density_alpha, grid, cache=collocation_cache
            )
            rho_beta = collocate_density_on_grid(
                basis, density_beta, grid, cache=collocation_cache
            )
            (
                e_xc,
                v_xc_alpha_grid,
                v_xc_beta_grid,
                v_sigma_aa,
                v_sigma_ab,
                v_sigma_bb,
                grad_alpha,
                grad_beta,
                v_tau_alpha,
                v_tau_beta,
            ) = _evaluate_xc_polarised_on_grid(
                rho_alpha,
                rho_beta,
                grid,
                func,
                basis=basis,
                density_matrix_alpha=density_alpha,
                density_matrix_beta=density_beta,
                cache=collocation_cache,
            )
            v_xc_alpha = _project_vxc_polarised_to_ao(
                basis,
                "alpha",
                v_xc_alpha_grid,
                grid,
                v_sigma_self=v_sigma_aa,
                v_sigma_cross=v_sigma_ab,
                grad_rho_self=grad_alpha,
                grad_rho_other=grad_beta,
                cache=collocation_cache,
            )
            v_xc_beta = _project_vxc_polarised_to_ao(
                basis,
                "beta",
                v_xc_beta_grid,
                grid,
                v_sigma_self=v_sigma_bb,
                v_sigma_cross=v_sigma_ab,
                grad_rho_self=grad_beta,
                grad_rho_other=grad_alpha,
                cache=collocation_cache,
            )
            if v_tau_alpha is not None:
                from .periodic_gapw_j import _project_vtau_to_ao

                v_xc_alpha = v_xc_alpha + _project_vtau_to_ao(
                    basis, v_tau_alpha, grid, cache=collocation_cache
                )
                v_xc_beta = v_xc_beta + _project_vtau_to_ao(
                    basis, v_tau_beta, grid, cache=collocation_cache
                )

            fock_alpha = (
                hcore + coulomb - ex_frac * exchange_alpha + v_xc_alpha
            )
            fock_beta = (
                hcore + coulomb - ex_frac * exchange_beta + v_xc_beta
            )
            e_one = float(np.einsum("ij,ij->", density_total, hcore))
            e_hartree = 0.5 * float(
                np.einsum("ij,ij->", density_total, coulomb)
            )
            e_exchange = -0.5 * ex_frac * (
                float(np.einsum("ij,ij->", density_alpha, exchange_alpha))
                + float(np.einsum("ij,ij->", density_beta, exchange_beta))
            )
            return (
                fock_alpha,
                fock_beta,
                e_one + e_hartree + e_exchange + float(e_xc),
            )

    n_basis = int(basis.nbasis)
    expected_shape = (n_basis, n_basis)
    if initial_density is not None:
        if len(initial_density) != 2:
            raise ValueError(
                f"{label}: initial_density must be a (D_alpha, D_beta) pair."
            )
        init_alpha = np.asarray(initial_density[0], dtype=float).copy()
        init_beta = np.asarray(initial_density[1], dtype=float).copy()
        if init_alpha.shape != expected_shape:
            raise ValueError(
                f"{label}: initial alpha density shape {init_alpha.shape} "
                f"doesn't match basis {expected_shape}."
            )
        if init_beta.shape != expected_shape:
            raise ValueError(
                f"{label}: initial beta density shape {init_beta.shape} "
                f"doesn't match basis {expected_shape}."
            )
    else:
        from .guess import initial_densities_open_shell

        # The exchange builder deliberately uses a longer 25-bohr cell list,
        # but this route's S/T pencil comes from the default one-electron
        # lattice options. Build every periodic guess in that same metric;
        # using the exchange cutoff here would normalise the seed against a
        # different overlap from the one consumed by the SCF.
        guess_lattice_options = _core.LatticeSumOptions()
        guess_densities = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha_resolved,
            n_beta_resolved,
            (_core.InitialGuess.SAD if initial_guess == _core.InitialGuess.PATOM
             else initial_guess),
            is_periodic=True,
            periodic_system=system,
            lattice_opts=guess_lattice_options,
            overlap=overlap,
                              atomic_spins=atomic_spins,
        )
        if guess_densities is not None:
            init_alpha = np.asarray(guess_densities[0], dtype=float).copy()
            init_beta = np.asarray(guess_densities[1], dtype=float).copy()
        else:
            x = _orthonormaliser(overlap, float(linear_dep_threshold))
            if n_alpha_resolved > x.shape[1]:
                raise RuntimeError(
                    f"{label}: canonical orthogonalisation retained "
                    f"{x.shape[1]} orbitals, but n_alpha={n_alpha_resolved}."
                )
            _, coeffs_orth = np.linalg.eigh(x.T @ hcore @ x)
            coeffs = x @ coeffs_orth
            coeffs_alpha = coeffs[:, :n_alpha_resolved]
            coeffs_beta = coeffs[:, :n_beta_resolved]
            init_alpha = coeffs_alpha @ coeffs_alpha.T
            init_beta = coeffs_beta @ coeffs_beta.T

    from .guess import normalize_spin_density_guess
    init_alpha, init_beta = normalize_spin_density_guess(
        init_alpha, init_beta, overlap, n_alpha_resolved, n_beta_resolved,
    )

    if initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_spin_density_step
        init_alpha, init_beta = patom_spin_density_step(
            init_alpha, init_beta, hcore, overlap,
            n_alpha_resolved, n_beta_resolved, gpw.build_J, build_k,
            threshold=float(linear_dep_threshold),
        )

    options = ROHFOptions(
        max_iter=int(max_iter),
        conv_tol_energy=float(conv_tol_energy),
        conv_tol_grad=float(conv_tol_grad),
        use_diis=bool(use_diis),
        diis_start_iter=int(diis_start_iter),
        diis_subspace_size=int(diis_subspace_size),
        damping=float(damping),
        level_shift=float(level_shift),
        linear_dep_threshold=float(linear_dep_threshold),
        fractional_open_shell=bool(fractional_open_shell),
    )

    def on_iteration(step) -> None:
        plog.iteration(
            step.iter,
            energy=float(step.energy),
            dE=float(step.delta_e),
            grad=float(step.grad_norm),
            diis=int(step.diis_subspace),
        )

    base = run_roothaan_scf(
        overlap,
        hcore,
        e_nuclear,
        n_alpha_resolved,
        n_beta_resolved,
        fock_builder,
        options,
        init_alpha,
        init_beta,
        iteration_callback=on_iteration,
    )

    # Rebuild the component energies once at the returned density so the
    # route-specific breakdown, total, and Focks all describe one state.
    density_total = np.asarray(base.density)
    density_alpha = np.asarray(base.density_alpha)
    density_beta = np.asarray(base.density_beta)
    coulomb = gpw.build_J(density_total)
    if ex_frac > 0.0:
        exchange_alpha = build_k(density_alpha)
        exchange_beta = build_k(density_beta)
    else:
        exchange_alpha = np.zeros_like(density_alpha)
        exchange_beta = np.zeros_like(density_beta)

    if func is not None:
        rho_alpha = collocate_density_on_grid(
            basis, density_alpha, grid, cache=collocation_cache
        )
        rho_beta = collocate_density_on_grid(
            basis, density_beta, grid, cache=collocation_cache
        )
        e_xc, *_ = _evaluate_xc_polarised_on_grid(
            rho_alpha,
            rho_beta,
            grid,
            func,
            basis=basis,
            density_matrix_alpha=density_alpha,
            density_matrix_beta=density_beta,
            cache=collocation_cache,
        )
    else:
        e_xc = 0.0

    e_kinetic = float(np.einsum("ij,ij->", density_total, kinetic))
    e_nuclear_attraction = float(
        np.einsum("ij,ij->", density_total, nuclear_attraction)
    )
    e_hartree = 0.5 * float(
        np.einsum("ij,ij->", density_total, coulomb)
    )
    e_hf_exchange = -0.5 * ex_frac * (
        float(np.einsum("ij,ij->", density_alpha, exchange_alpha))
        + float(np.einsum("ij,ij->", density_beta, exchange_beta))
    )
    e_electronic = (
        e_kinetic
        + e_nuclear_attraction
        + e_hartree
        + e_hf_exchange
        + float(e_xc)
    )
    e_total = e_electronic + e_nuclear
    breakdown = GpwEnergyBreakdown(
        e_kinetic=e_kinetic,
        e_nuclear_attraction=e_nuclear_attraction,
        e_hartree=e_hartree,
        e_hf_exchange=e_hf_exchange,
        e_nuclear_repulsion=e_nuclear,
        e_total=e_total,
        grid=grid,
        e_xc=float(e_xc),
        functional=str(functional) if functional is not None else None,
    )

    result_type = GpwRoksScfResult if func is not None else GpwRohfScfResult
    return result_type(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=e_total,
        e_electronic=e_electronic,
        e_nuclear=e_nuclear,
        breakdown=breakdown,
        density=density_total,
        density_alpha=density_alpha,
        density_beta=density_beta,
        mo_coeffs=np.asarray(base.mo_coeffs),
        mo_energies=np.asarray(base.mo_energies),
        mo_occupations=np.asarray(base.mo_occupations),
        fock=np.asarray(base.fock),
        fock_alpha=np.asarray(base.fock_alpha),
        fock_beta=np.asarray(base.fock_beta),
        overlap=np.asarray(overlap),
        n_alpha=n_alpha_resolved,
        n_beta=n_beta_resolved,
        converged=bool(base.converged),
        n_iter=int(base.n_iter),
        grid=grid,
        s_squared=float(base.s_squared),
        s_squared_ideal=float(base.s_squared_ideal),
        scf_trace=tuple(base.scf_trace),
        e_xc=float(e_xc),
        functional=str(functional) if functional is not None else None,
    )


def run_periodic_rohf_gpw(
    system,
    basis,
    *,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_grad: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    linear_dep_threshold: float = 1e-8,
    level_shift: float = 0.0,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
) -> GpwRohfScfResult:
    """Restricted-open-shell HF on the three-dimensional Gamma GPW route."""
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    return _run_periodic_restricted_open_gpw(
        system,
        basis,
        functional=None,
        label="run_periodic_rohf_gpw",
        n_alpha=n_alpha,
        n_beta=n_beta,
        grid=grid,
        cutoff_ha=cutoff_ha,
        max_iter=max_iter,
        conv_tol_energy=conv_tol_energy,
        conv_tol_grad=conv_tol_grad,
        damping=damping,
        initial_density=initial_density,
        initial_guess=initial_guess,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        use_diis=use_diis,
        diis_subspace_size=diis_subspace_size,
        diis_start_iter=diis_start_iter,
        linear_dep_threshold=linear_dep_threshold,
        level_shift=level_shift,
        fractional_open_shell=False,
        quiet=quiet,
        progress=progress,
               atomic_spins=atomic_spins,
    )


def run_periodic_roks_gpw(
    system,
    basis,
    *,
    functional: str,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_grad: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    linear_dep_threshold: float = 1e-8,
    level_shift: float = 0.0,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
) -> GpwRoksScfResult:
    """Restricted-open-shell KS on the three-dimensional Gamma GPW route.

    The route evaluates spin-polarised XC on the established open-shell GPW
    grid and couples the two spin Focks through the shared Roothaan effective
    Fock. It supports the same LDA, GGA, meta-GGA, and global-hybrid envelope
    as Gamma GPW UKS; range-separated and double-hybrid functionals remain
    outside the GPW backend's implemented exchange/correlation contract.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    if not functional:
        raise ValueError(
            "run_periodic_roks_gpw: functional is required (for example "
            "'lda', 'pbe', or 'b3lyp')."
        )
    return _run_periodic_restricted_open_gpw(
        system,
        basis,
        functional=str(functional),
        label="run_periodic_roks_gpw",
        n_alpha=n_alpha,
        n_beta=n_beta,
        grid=grid,
        cutoff_ha=cutoff_ha,
        max_iter=max_iter,
        conv_tol_energy=conv_tol_energy,
        conv_tol_grad=conv_tol_grad,
        damping=damping,
        initial_density=initial_density,
        initial_guess=initial_guess,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        use_diis=use_diis,
        diis_subspace_size=diis_subspace_size,
        diis_start_iter=diis_start_iter,
        linear_dep_threshold=linear_dep_threshold,
        level_shift=level_shift,
        fractional_open_shell=False,
        quiet=quiet,
        progress=progress,
               atomic_spins=atomic_spins,
    )


def run_periodic_uhf_gpw(
    system,
    basis,
    *,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    dft_plus_u_sites: Optional[object] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
) -> GpwUhfScfResult:
    """Open-shell periodic UHF on the GPW route (Γ-only).

    Pure Hartree-Fock -- no V_xc or D3. The minimal open-shell SCF
    lives in :func:`_run_open_shell_gpw`; this function is the
    user-facing wrapper that fixes ``functional=None``. Optional
    ``dft_plus_u_sites`` applies a Dudarev per-spin +U correction.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis (same conventions as
        :func:`run_periodic_rhf_gpw`).
    n_alpha, n_beta
        Per-spin occupations. Default ``None`` falls back to
        ``n_alpha = n_beta = n_elec // 2`` for even-electron cells
        (closed-shell forced into the UHF path; reduces to RHF
        bit-for-bit, which is the parity test pin). Open-shell
        cells require both to be supplied.
    grid, cutoff_ha
        Optional pre-built grid + plane-wave cutoff. Default
        ``cutoff_ha=300.0``.
    max_iter, conv_tol_energy, conv_tol_density
        SCF convergence knobs.
    damping
        Linear-density damping in ``[0, 1)`` applied per spin.
    initial_density
        Optional ``(D_alpha_init, D_beta_init)`` AO densities.
        Default ``None`` uses an Hcore guess Aufbau-filled per
        spin (and bit-for-bit closed-shell symmetric when
        ``n_alpha == n_beta``).
    v_ne_convention, smearing_alpha
        See :func:`run_periodic_rhf_gpw`.
    use_diis, diis_subspace_size, diis_start_iter
        Pulay-DIIS knobs. The DIIS error vector combines both
        spins so the joint B-matrix mixes a + b residuals.
    quiet
        Suppress the ``GAPWExperimentalWarning``.

    Returns
    -------
    result
        :class:`GpwUhfScfResult` with per-spin densities, MO
        coefficients, MO energies, the breakdown, and convergence
        flag.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    input_restart_supplied = initial_density is not None
    (breakdown,
     D_alpha, D_beta,
     C_alpha, C_beta,
     e_a, e_b,
     n_alpha_resolved, n_beta_resolved,
     converged, n_iter, grid_out,
     scf_trace) = _run_open_shell_gpw(
        system, basis,
        n_alpha=n_alpha, n_beta=n_beta,
        grid=grid, cutoff_ha=cutoff_ha,
        max_iter=max_iter,
        conv_tol_energy=conv_tol_energy,
        conv_tol_density=conv_tol_density,
        damping=damping,
        initial_density=initial_density,
        initial_guess=initial_guess,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=None,
        dft_plus_u_sites=dft_plus_u_sites,
        use_diis=use_diis,
        diis_subspace_size=diis_subspace_size,
        diis_start_iter=diis_start_iter,
        quiet=quiet,
        label="run_periodic_uhf_gpw",
        progress=progress,
                      atomic_spins=atomic_spins,
    )
    return GpwUhfScfResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=breakdown.e_total,
        breakdown=breakdown,
        density_alpha=D_alpha,
        density_beta=D_beta,
        mo_coeffs_alpha=C_alpha,
        mo_coeffs_beta=C_beta,
        mo_energies_alpha=e_a,
        mo_energies_beta=e_b,
        n_alpha=n_alpha_resolved,
        n_beta=n_beta_resolved,
        converged=converged,
        n_iter=n_iter,
        grid=grid_out,
        e_dft_plus_u=float(getattr(breakdown, "e_dft_plus_u", 0.0)),
        dft_plus_u_sites=tuple(dft_plus_u_sites or ()),
        scf_trace=scf_trace,
    )


def run_periodic_uks_gpw(
    system,
    basis,
    *,
    functional: str,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[tuple] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    dft_plus_u_sites: Optional[object] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
) -> GpwUksScfResult:
    """Open-shell periodic UKS on the GPW route (Γ-only).

    Single-k spin-polarised KS-DFT. ``functional`` is required and
    is passed straight to :class:`vibeqc._vibeqc_core.Functional`
    with ``spin=2`` (the polarised libxc evaluation path). For
    hybrids the HF exchange fraction is read off the functional
    and the per-spin K is built from
    :func:`vibeqc._vibeqc_core.build_jk_gamma_molecular_limit`.

    See :func:`run_periodic_uhf_gpw` for the parameter conventions
    that aren't ``functional``. Optional ``dft_plus_u_sites`` applies
    a Dudarev per-spin +U correction in the same convention as UHF.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density is not None,
    )
    input_restart_supplied = initial_density is not None
    if not functional:
        raise ValueError(
            "run_periodic_uks_gpw: functional is required (got "
            f"functional={functional!r}). For pure UHF use "
            "run_periodic_uhf_gpw."
        )
    (breakdown,
     D_alpha, D_beta,
     C_alpha, C_beta,
     e_a, e_b,
     n_alpha_resolved, n_beta_resolved,
     converged, n_iter, grid_out,
     scf_trace) = _run_open_shell_gpw(
        system, basis,
        n_alpha=n_alpha, n_beta=n_beta,
        grid=grid, cutoff_ha=cutoff_ha,
        max_iter=max_iter,
        conv_tol_energy=conv_tol_energy,
        conv_tol_density=conv_tol_density,
        damping=damping,
        initial_density=initial_density,
        initial_guess=initial_guess,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=str(functional),
        dft_plus_u_sites=dft_plus_u_sites,
        use_diis=use_diis,
        diis_subspace_size=diis_subspace_size,
        diis_start_iter=diis_start_iter,
        quiet=quiet,
        label="run_periodic_uks_gpw",
        progress=progress,
                      atomic_spins=atomic_spins,
    )
    return GpwUksScfResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, initial_guess, restarted=input_restart_supplied),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=breakdown.e_total,
        breakdown=breakdown,
        density_alpha=D_alpha,
        density_beta=D_beta,
        mo_coeffs_alpha=C_alpha,
        mo_coeffs_beta=C_beta,
        mo_energies_alpha=e_a,
        mo_energies_beta=e_b,
        n_alpha=n_alpha_resolved,
        n_beta=n_beta_resolved,
        converged=converged,
        n_iter=n_iter,
        grid=grid_out,
        e_dft_plus_u=float(getattr(breakdown, "e_dft_plus_u", 0.0)),
        dft_plus_u_sites=tuple(dft_plus_u_sites or ()),
        scf_trace=scf_trace,
    )


# ============================================================
# Multi-k UKS (pure DFT)
# ============================================================


@dataclass(frozen=True)
class GpwUksMultiKScfResult:
    """Result of a multi-k GPW periodic UKS SCF (pure DFT).

    Per-spin, per-k analog of
    :class:`vibeqc.periodic_gapw_j.GpwMultiKScfResult`. ``mo_coeffs_*_k``
    and ``mo_energies_*_k`` are length-``n_k`` tuples; on the compact
    Bloch path the per-k MO count can be below ``n_basis`` (the canonical
    orthogonaliser projects out linearly-dependent Bloch-AO directions).
    ``density_alpha`` / ``density_beta`` are the Gamma-summed real AO
    density matrices ``sum_k w_k Re D_s(k)``.
    """

    energy: float
    breakdown: GpwEnergyBreakdown
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_coeffs_alpha_k: tuple
    mo_coeffs_beta_k: tuple
    mo_energies_alpha_k: tuple
    mo_energies_beta_k: tuple
    n_alpha: int
    n_beta: int
    converged: bool
    n_iter: int
    grid: PlaneWaveGrid
    kmesh: object = None
    scf_trace: tuple = ()
    method: str = "uks"

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


@dataclass(frozen=True)
class GpwRoksMultiKScfResult(GpwUksMultiKScfResult):
    """Result of a multi-k pure-DFT GPW periodic ROKS SCF."""

    method: str = "roks"


def _xc_effective_potentials_polarised_grid(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    grid: PlaneWaveGrid,
    functional,
    *,
    tau_alpha: Optional[np.ndarray] = None,
    tau_beta: Optional[np.ndarray] = None,
    cache: Optional[GpwCollocationCache] = None,
) -> tuple:
    """Per-spin local XC potentials as scalar grid fields, plus E_xc.

    Spin-polarised sibling of
    :func:`vibeqc.periodic_gapw_j._xc_effective_potential_grid`: assembles
    the LDA piece and -- for GGAs -- the integrated-by-parts gradient
    correction into one real potential per spin,

        ``v_eff_s(r) = v_rho_s - div(2 v_ss gradrho_s + v_ab gradrho_s')``

    so the compact multi-k Bloch path can project the full per-spin local
    Fock block per k in a single pass. For meta-GGA the caller passes the
    precomputed per-spin Bloch kinetic-energy densities ``tau_alpha=`` /
    ``tau_beta=``; the two ``v_tau`` grids are returned so the driver can
    project the generalised-KS ``½∫v_tau ∇χ·∇χ`` Fock block per k (that
    piece is NOT a multiplicative scalar field, so it stays out of
    ``v_eff`` and is handled with :func:`project_vtau_to_bloch_ao`).

    Returns ``(e_xc_total, v_eff_a, v_eff_b, v_tau_a, v_tau_b)`` -- the
    ``v_tau`` grids are ``None`` for LDA/GGA.
    """
    (
        e_xc,
        v_xc_a,
        v_xc_b,
        v_sigma_aa,
        v_sigma_ab,
        v_sigma_bb,
        grad_a,
        grad_b,
        v_tau_a_grid,
        v_tau_b_grid,
    ) = _evaluate_xc_polarised_on_grid(
        rho_a,
        rho_b,
        grid,
        functional,
        tau_alpha=tau_alpha,
        tau_beta=tau_beta,
        cache=cache,
    )
    v_eff_a = np.array(v_xc_a, dtype=float)
    v_eff_b = np.array(v_xc_b, dtype=float)
    if v_sigma_aa is not None and grad_a is not None:
        G = _resolve_recip(cache, grid)

        def _div(flux):
            out = np.zeros(grid.shape, dtype=float)
            for d in range(3):
                flux_k = np.fft.fftn(flux[..., d])
                out += np.real(np.fft.ifftn(1j * G[..., d] * flux_k))
            return out

        flux_a = 2.0 * v_sigma_aa[..., None] * grad_a + v_sigma_ab[..., None] * grad_b
        flux_b = 2.0 * v_sigma_bb[..., None] * grad_b + v_sigma_ab[..., None] * grad_a
        v_eff_a -= _div(flux_a)
        v_eff_b -= _div(flux_b)
    return e_xc, v_eff_a, v_eff_b, v_tau_a_grid, v_tau_b_grid


def run_periodic_uks_gpw_multi_k(
    system,
    basis,
    kmesh,
    *,
    functional: str,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 80,
    conv_tol_energy: float = 1e-8,
    conv_tol_density: float = 1e-6,
    initial_density_k: Optional[
        Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]
    ] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
    _restricted_open: bool = False,
) -> GpwUksMultiKScfResult:
    """Multi-k GPW periodic UKS SCF (spin-polarised, pure DFT only).

    Open-shell closure of the "GPW multi-k only supports RKS" gap. Mirrors
    :func:`vibeqc.periodic_gapw_j.run_periodic_rks_gpw_multi_k` with two
    spin channels:

    * **Pure DFT (LDA / GGA) only** -- hybrids raise (per-k exact-exchange
      builders are not wired); meta-GGA raises from the polarised XC
      helper (per-spin t is not wired on the GPW route).
    * **Two collocation regimes**, dispatched exactly like the RKS driver:
      molecular-limit cells fold each spin density to a Gamma-summed real
      density matrix on the periodic Gamma AOs; compact crystals build the
      per-spin Bloch density ``rho_s(r) = sum_k w_k sum_mn D_s,mn(k)
      chi_mk(r) chi_nk(r)*`` and project ``V_H + v_eff_s`` back per k onto
      Bloch AOs (Hermitian per-k blocks).
    * **Per-k Pulay commutator DIIS in the orthonormal subspace**, run
      jointly over both spins (one B matrix, alpha + beta error blocks) on
      both regimes -- the driver is new, so there is no historical linear-
      mixing behaviour to preserve.
    * **T = 0 hard aufbau per spin** -- ``n_alpha`` / ``n_beta`` electrons
      at every k (resolved from ``system.multiplicity`` when not given).
      Smearing / DFT+U / dispersion are not wired at this milestone.
    """
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density_k is not None,
    )
    input_restart_supplied = initial_density_k is not None
    label = (
        "run_periodic_roks_gpw_multi_k"
        if _restricted_open
        else "run_periodic_uks_gpw_multi_k"
    )
    from .guess import _coerce_periodic_driver_guess

    requested_initial_guess = initial_guess
    initial_guess = _coerce_periodic_driver_guess(
        initial_guess,
        driver=label,
        supported=periodic_guess_capabilities('gpw', 'UKS', dim=getattr(system, "dim", 3), multi_k=True, transport='k'),
        restart_supplied=initial_density_k is not None,
    )

    if not functional:
        raise ValueError(
            f"{label}: functional is required (got "
            f"functional={functional!r})."
        )
    if not quiet:
        _warn_experimental(
            f"{label}: multi-k "
            + ("restricted-open-shell " if _restricted_open else "spin-polarised ")
            + "pure-DFT GPW"
        )

    plog = resolve_progress(progress)

    if system.dim != 3:
        raise ValueError(
            f"{label}: dim == 3 only; got dim={system.dim}"
        )

    func = _core.Functional(str(functional), 2)
    if float(func.hf_exchange_fraction) != 0.0:
        raise NotImplementedError(
            f"{label}: hybrids (hf_exchange_fraction "
            f"= {func.hf_exchange_fraction}) are not supported -- per-k "
            f"exact-exchange builders are not wired. Use a pure functional "
            f"like 'lda' or 'pbe'."
        )
    is_mgga = func.kind == _core.XCKind.MGGA

    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = kpoints.shape[0]
    use_bloch_compact = n_k > 1 and not _multik_gpw_is_molecular_limit(system)

    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    collocation_cache = build_gpw_collocation_cache(basis, grid)
    n_alpha_res, n_beta_res = infer_alpha_beta_from_system(
        system, n_alpha=n_alpha, n_beta=n_beta
    )
    n_basis = basis.nbasis

    # One-electron lattice integrals + per-k Bloch sums (same gauge as the
    # RKS multi-k driver). S, T and V_ne MUST share one R-set -- the pencil
    # H(k) C = S(k) C eps is only consistent when every operator drops the
    # same lattice tail (Pisani & Dovesi 1980, doi:10.1002/qua.560170311,
    # p. 510; see multik_v_ne_lattice_options for the measured -18 Ha
    # failure mode of mixing R-sets).
    from .periodic_gapw_j import (
        multik_one_electron_lattice_options,
        multik_v_ne_lattice_options,
    )

    lat_opts = multik_one_electron_lattice_options(basis, system)
    T_lat = _core.compute_kinetic_lattice(basis, system, lat_opts)
    S_lat = _core.compute_overlap_lattice(basis, system, lat_opts)
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    lat_opts_v = multik_v_ne_lattice_options(basis, system)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts_v)
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    if use_bloch_compact:
        from .periodic_rhf_multi_k_ewald import (
            _canonical_orthogonalizer_complex,
        )

    T_k: list = []
    S_k: list = []
    V_ne_k: list = []
    X_k: list = []
    for ik in range(n_k):
        k = kpoints[ik]
        Tk = np.asarray(_core.bloch_sum(T_lat, k))
        Sk = np.asarray(_core.bloch_sum(S_lat, k))
        Vk = np.asarray(_core.bloch_sum(V_lat, k))
        Sk = 0.5 * (Sk + Sk.conj().T)
        Tk = 0.5 * (Tk + Tk.conj().T)
        Vk = 0.5 * (Vk + Vk.conj().T)
        if use_bloch_compact:
            # Fail closed on an indefinite metric (a truncation defect of
            # the overlap lattice sum, never a basis property; see the RKS
            # multi-k driver).
            from .periodic_gapw_j import _assert_bloch_overlap_psd

            _assert_bloch_overlap_psd(Sk, k, lat_opts.cutoff_bohr)
            # Project out linearly-dependent Bloch-AO directions (see the
            # RKS multi-k driver for the rationale; X is (n_bf, n_kept)).
            X, _n_kept = _canonical_orthogonalizer_complex(
                Sk, threshold=_COMPACT_GPW_LINDEP_THRESHOLD
            )
        else:
            s_eigs, U = np.linalg.eigh(Sk)
            X = U @ np.diag(1.0 / np.sqrt(np.maximum(s_eigs, 1e-12))) @ U.conj().T
        T_k.append(Tk)
        S_k.append(Sk)
        V_ne_k.append(Vk)
        X_k.append(X)
        n_mo_k = X.shape[1]
        if n_alpha_res > n_mo_k or n_beta_res > n_mo_k:
            raise ValueError(
                f"{label}: k-point {ik} keeps only "
                f"{n_mo_k} independent Bloch-AO directions but "
                f"n_alpha={n_alpha_res} / n_beta={n_beta_res} occupations "
                f"were requested."
            )

    # Per-k Bloch AO tables for the compact density build (iteration-
    # invariant; shared by both spins). Retained only while the whole all-k
    # cache fits the closed-shell driver's target; above it, the density
    # collocation and the per-k projections stream bounded point batches
    # exactly as run_periodic_rks_gpw_multi_k does (d17aae954), and the
    # meta-GGA spectral tau retains one complete k table at a time. Before
    # this the open-shell branch kept n_k x n_grid x n_ao complex tables
    # unconditionally: 702.7 GiB on the P16 NiO/DFT+U cell (232 AOs, 128^3
    # grid, 64 k) against 1.2 GiB for the same closed-shell cell (issue #89).
    # The cache target is read from periodic_gapw_j at call time so one
    # setting (and one test monkeypatch) governs both drivers.
    chi_k_list: Optional[list] = None
    direct_cells = None
    lattice_translations = None
    grid_pts = None
    if use_bloch_compact:
        direct_cells = _core.direct_lattice_cells(system, 25.0)
        lattice_translations = np.array(
            [c.r_cart for c in direct_cells], dtype=float
        )
        grid_pts = grid.cartesian_coords().reshape(-1, 3)
        compact_cache_bytes = (
            n_k * len(grid_pts) * n_basis * np.dtype(np.complex128).itemsize
        )
        if compact_cache_bytes <= _gapw_j._COMPACT_BLOCH_AO_CACHE_BYTES:
            chi_k_list = [
                bloch_ao_on_grid(
                    basis, grid_pts, kpoints[ik], lattice_translations
                )
                for ik in range(n_k)
            ]

    def _bloch_tau(D_k_list):
        """Per-spin meta-GGA tau on the compact route, cached or streamed."""
        if chi_k_list is not None:
            return compute_bloch_kinetic_energy_density(
                chi_k_list, D_k_list, weights, kpoints, grid_pts, grid,
                recip=collocation_cache.recip,
            )
        return _compute_bloch_kinetic_energy_density_streaming(
            basis, D_k_list, weights, kpoints, grid_pts, lattice_translations,
            grid, recip=collocation_cache.recip,
        )

    def _bloch_local_blocks(ik, v_loc_a, v_loc_b, v_tau_a_grid, v_tau_b_grid):
        """Project the per-spin local (+ v_tau) potentials onto the Bloch AOs
        at k-point ``ik`` with at most one k table live at once."""
        if chi_k_list is not None:
            chi_k = chi_k_list[ik]
        elif v_tau_a_grid is None and v_tau_b_grid is None:
            return (
                _project_potential_to_bloch_ao_streaming(
                    basis, grid_pts, kpoints[ik], lattice_translations,
                    v_loc_a, grid,
                ),
                _project_potential_to_bloch_ao_streaming(
                    basis, grid_pts, kpoints[ik], lattice_translations,
                    v_loc_b, grid,
                ),
            )
        else:
            # Spectral meta-GGA gradients need the complete grid, but only
            # this k-point's table is live; reuse it for all four
            # projections before releasing it.
            chi_k = bloch_ao_on_grid(
                basis, grid_pts, kpoints[ik], lattice_translations
            )
        V_a = project_potential_to_bloch_ao(chi_k, v_loc_a, grid)
        V_b = project_potential_to_bloch_ao(chi_k, v_loc_b, grid)
        if v_tau_a_grid is not None:
            # Per-spin t Fock term per k: 1/2 int v_tau gradchi*.gradchi.
            V_a = V_a + project_vtau_to_bloch_ao(
                chi_k, kpoints[ik], v_tau_a_grid, grid_pts, grid,
                recip=collocation_cache.recip,
            )
            V_b = V_b + project_vtau_to_bloch_ao(
                chi_k, kpoints[ik], v_tau_b_grid, grid_pts, grid,
                recip=collocation_cache.recip,
            )
        return V_a, V_b

    def _build_D_k(C_list, n_occ):
        return [
            (
                C_list[ik][:, :n_occ] @ C_list[ik][:, :n_occ].conj().T
                if n_occ > 0
                else np.zeros((n_basis, n_basis), dtype=complex)
            )
            for ik in range(n_k)
        ]

    def _gamma_sum(D_k_list):
        D = np.zeros((n_basis, n_basis), dtype=float)
        for ik in range(n_k):
            D += weights[ik] * np.real(D_k_list[ik])
        return 0.5 * (D + D.T)

    def _collocate_spin(D_gamma, D_k_list):
        if use_bloch_compact:
            if chi_k_list is not None:
                return collocate_bloch_density_on_grid(
                    chi_k_list, D_k_list, weights, grid
                )
            return _collocate_bloch_density_streaming(
                basis, grid_pts, kpoints, lattice_translations,
                D_k_list, weights, grid,
            )
        return collocate_density_on_grid(
            basis, D_gamma, grid, cache=collocation_cache
        )

    # Fock-mode guesses retain this route's per-k occupations. Density-mode
    # guesses are produced once by the shared adapter and injected as g=0
    # blocks at every k point. Explicit restart blocks always take precedence.
    guess_fock_k = None
    if initial_density_k is None and initial_guess == _core.InitialGuess.PATOM:
        from .guess import patom_full_hf_focks_k
        focks = patom_full_hf_focks_k(
            system, basis, kmesh, [t + v for t, v in zip(T_k, V_ne_k)],
            S_k, n_alpha_res, n_beta_res, lat_opts, atomic_spins=atomic_spins)
        initial_density_k = []
        for spin_focks, count in zip(focks, (n_alpha_res, n_beta_res)):
            blocks = []
            for fock, x in zip(spin_focks, X_k):
                _, vectors = np.linalg.eigh(x.conj().T @ ((fock + fock.conj().T) * .5) @ x)
                occupied = (x @ vectors)[:, :count]
                blocks.append(occupied @ occupied.conj().T)
            initial_density_k.append(blocks)
    if initial_density_k is None:
        from .guess import initial_densities_open_shell, periodic_fock_guess_k

        guess_fock_k = periodic_fock_guess_k(
            system,
            basis,
            kpoints,
            initial_guess,
            lattice_opts=lat_opts,
            kinetic_lattice=T_lat,
            overlap_lattice=S_lat,
        )
        if guess_fock_k is None:
            density_g0 = initial_densities_open_shell(
                system.unit_cell_molecule(),
                basis,
                n_alpha_res,
                n_beta_res,
                initial_guess,
                is_periodic=True,
                periodic_system=system,
                lattice_opts=lat_opts,
                overlap=S_k, weights=weights,
                             atomic_spins=atomic_spins,
            )
            if density_g0 is not None:
                density_alpha_g0, density_beta_g0 = density_g0
                initial_density_k = (
                    [
                        np.asarray(density_alpha_g0, dtype=complex).copy()
                        for _ in range(n_k)
                    ],
                    [
                        np.asarray(density_beta_g0, dtype=complex).copy()
                        for _ in range(n_k)
                    ],
                )

    # Initial guess: diagonalise the selected per-k one-particle Hamiltonian
    # and Aufbau-fill per spin. Spin symmetry is broken by the occupation
    # counts when n_alpha != n_beta, matching the Gamma open-shell driver.
    C_a_k: list = []
    C_b_k: list = []
    for ik in range(n_k):
        guess_fock = (
            guess_fock_k[ik]
            if guess_fock_k is not None
            else T_k[ik] + V_ne_k[ik]
        )
        Fo = X_k[ik].conj().T @ guess_fock @ X_k[ik]
        Fo = 0.5 * (Fo + Fo.conj().T)
        _eps, C_orth = np.linalg.eigh(Fo)
        C = X_k[ik] @ C_orth
        C_a_k.append(C)
        C_b_k.append(C)
    D_a_k = _build_D_k(C_a_k, n_alpha_res)
    D_b_k = _build_D_k(C_b_k, n_beta_res)
    if initial_density_k is not None:
        if len(initial_density_k) != 2:
            raise ValueError(
                f"{label}: initial_density_k must be an "
                "(alpha_blocks, beta_blocks) pair."
            )
        from .guess import normalize_spin_density_k_guess
        D_a_k, D_b_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k, weights, n_alpha_res, n_beta_res,
        )
    D_a = _gamma_sum(D_a_k)
    D_b = _gamma_sum(D_b_k)

    from .periodic_rhf_multi_k_ewald import _MultiKPulayDIIS

    diis = _MultiKPulayDIIS(max_subspace=8)

    scf_trace: list = []
    E_prev = 0.0
    converged = False
    n_iter = 0
    e_a_k: list = [np.zeros(X_k[ik].shape[1]) for ik in range(n_k)]
    e_b_k: list = [np.zeros(X_k[ik].shape[1]) for ik in range(n_k)]
    e_hartree = 0.0
    e_xc_iter = 0.0

    for it in range(1, max_iter + 1):
        rho_a = _collocate_spin(D_a, D_a_k)
        rho_b = _collocate_spin(D_b, D_b_k)
        rho_tot = rho_a + rho_b
        V_H = _core.solve_poisson_coulomb(rho_tot, grid.lattice_bohr)

        # Local Fock blocks per spin (Hartree shared, XC per spin).
        V_loc_a_per_k: list = [None] * n_k
        V_loc_b_per_k: list = [None] * n_k
        J_ao = None
        V_xc_a = None
        V_xc_b = None
        if use_bloch_compact:
            e_hartree = 0.5 * float(
                np.sum(rho_tot * V_H) * grid.voxel_volume_bohr3
            )
            # Meta-GGA: per-spin t from the per-k Bloch AO gradients (the
            # Gamma t builder drops the inter-cell Bloch phases on a compact
            # cell, exactly like the density collocation it mirrors). Mirrors
            # the closed-shell compact multi-k path in periodic_gapw_j.
            tau_a_bloch = None
            tau_b_bloch = None
            if is_mgga:
                tau_a_bloch = _bloch_tau(D_a_k)
                tau_b_bloch = _bloch_tau(D_b_k)
            (
                e_xc_iter,
                v_eff_a,
                v_eff_b,
                v_tau_a_grid,
                v_tau_b_grid,
            ) = _xc_effective_potentials_polarised_grid(
                rho_a, rho_b, grid, func,
                tau_alpha=tau_a_bloch, tau_beta=tau_b_bloch,
                cache=collocation_cache,
            )
            v_loc_a = np.asarray(V_H, dtype=float) + v_eff_a
            v_loc_b = np.asarray(V_H, dtype=float) + v_eff_b
            for ik in range(n_k):
                V_loc_a_per_k[ik], V_loc_b_per_k[ik] = _bloch_local_blocks(
                    ik, v_loc_a, v_loc_b, v_tau_a_grid, v_tau_b_grid
                )
        else:
            J_ao = project_potential_to_ao(basis, V_H, grid, cache=collocation_cache)
            e_hartree = 0.5 * float(np.einsum("ij,ij->", D_a + D_b, J_ao))
            (
                e_xc_iter,
                v_xc_a_grid,
                v_xc_b_grid,
                v_sigma_aa_g,
                v_sigma_ab_g,
                v_sigma_bb_g,
                grad_a,
                grad_b,
                v_tau_a_grid,
                v_tau_b_grid,
            ) = _evaluate_xc_polarised_on_grid(
                rho_a, rho_b, grid, func,
                basis=basis,
                density_matrix_alpha=D_a,
                density_matrix_beta=D_b,
                cache=collocation_cache,
            )
            V_xc_a = _project_vxc_polarised_to_ao(
                basis, "alpha", v_xc_a_grid, grid,
                v_sigma_self=v_sigma_aa_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_a,
                grad_rho_other=grad_b,
                cache=collocation_cache,
            )
            V_xc_b = _project_vxc_polarised_to_ao(
                basis, "beta", v_xc_b_grid, grid,
                v_sigma_self=v_sigma_bb_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_b,
                grad_rho_other=grad_a,
                cache=collocation_cache,
            )
            if v_tau_a_grid is not None:
                # Per-spin t Fock term: 1/2 int v_tau gradchi.gradchi
                # (generalised-KS; see periodic_gapw_j._project_vtau_to_ao).
                from .periodic_gapw_j import _project_vtau_to_ao

                V_xc_a = V_xc_a + _project_vtau_to_ao(
                    basis, v_tau_a_grid, grid, cache=collocation_cache
                )
                V_xc_b = V_xc_b + _project_vtau_to_ao(
                    basis, v_tau_b_grid, grid, cache=collocation_cache
                )

        # Raw per-k per-spin Fock, Hermitised.
        F_a_raw: list = []
        F_b_raw: list = []
        for ik in range(n_k):
            Hcore_k = T_k[ik] + V_ne_k[ik]
            if use_bloch_compact:
                Fa = Hcore_k + V_loc_a_per_k[ik]
                Fb = Hcore_k + V_loc_b_per_k[ik]
            else:
                Fa = Hcore_k + J_ao + V_xc_a
                Fb = Hcore_k + J_ao + V_xc_b
            F_a_raw.append(0.5 * (Fa + Fa.conj().T))
            F_b_raw.append(0.5 * (Fb + Fb.conj().T))

        # UKS extrapolates both spin Focks jointly. ROKS first composes one
        # Roothaan effective Fock at each k and extrapolates that restricted
        # operator against the total-density commutator.
        Forth_list: list = []
        err_list: list = []
        spin_pairs = (
            ((F_a_raw, D_a_k), (F_b_raw, D_b_k))
            if not _restricted_open
            else ()
        )
        for spin_F, spin_D in spin_pairs:
            for ik in range(n_k):
                X = X_k[ik]
                Sk = S_k[ik]
                Fo = X.conj().T @ spin_F[ik] @ X
                Fo = 0.5 * (Fo + Fo.conj().T)
                Do = X.conj().T @ (Sk @ spin_D[ik] @ Sk) @ X
                Do = 0.5 * (Do + Do.conj().T)
                Forth_list.append(Fo)
                err_list.append(Fo @ Do - Do @ Fo)
        if _restricted_open:
            for ik in range(n_k):
                X = X_k[ik]
                Sk = S_k[ik]
                F_eff = roothaan_effective_fock(
                    F_a_raw[ik], F_b_raw[ik], D_a_k[ik], D_b_k[ik], Sk
                )
                Fo = X.conj().T @ F_eff @ X
                Fo = 0.5 * (Fo + Fo.conj().T)
                D_total_k = D_a_k[ik] + D_b_k[ik]
                Do = X.conj().T @ (Sk @ D_total_k @ Sk) @ X
                Do = 0.5 * (Do + Do.conj().T)
                Forth_list.append(Fo)
                err_list.append(Fo @ Do - Do @ Fo)
            Forth_use = diis.extrapolate(Forth_list, err_list, list(weights))
        else:
            Forth_use = diis.extrapolate(
                Forth_list, err_list, list(weights) + list(weights)
            )

        new_C_a: list = []
        new_C_b: list = []
        new_e_a: list = []
        new_e_b: list = []
        for ik in range(n_k):
            if _restricted_open:
                Fo = 0.5 * (Forth_use[ik] + Forth_use[ik].conj().T)
                eps, C_orth = np.linalg.eigh(Fo)
                C = X_k[ik] @ C_orth
                eps_alpha = np.einsum(
                    "pi,pq,qi->i", C.conj(), F_a_raw[ik], C
                ).real
                occupations = _roothaan_occupations(
                    eps, eps_alpha, n_alpha_res, n_beta_res
                )
                order = np.argsort(-occupations, kind="stable")
                C = C[:, order]
                eps = eps[order]
                new_C_a.append(C)
                new_C_b.append(C)
                new_e_a.append(eps)
                new_e_b.append(eps)
                continue
            Fo_a = Forth_use[ik]
            Fo_b = Forth_use[n_k + ik]
            Fo_a = 0.5 * (Fo_a + Fo_a.conj().T)
            Fo_b = 0.5 * (Fo_b + Fo_b.conj().T)
            eps_a, C_orth_a = np.linalg.eigh(Fo_a)
            eps_b, C_orth_b = np.linalg.eigh(Fo_b)
            new_C_a.append(X_k[ik] @ C_orth_a)
            new_C_b.append(X_k[ik] @ C_orth_b)
            new_e_a.append(eps_a)
            new_e_b.append(eps_b)

        D_a_k_new = _build_D_k(new_C_a, n_alpha_res)
        D_b_k_new = _build_D_k(new_C_b, n_beta_res)
        D_a_new = _gamma_sum(D_a_k_new)
        D_b_new = _gamma_sum(D_b_k_new)

        # Total energy: E = S_s S_k w_k tr(D_s(k) Hcore(k)) + E_H + E_xc + E_nn.
        e_hcore = 0.0
        for ik in range(n_k):
            Hcore_k = T_k[ik] + V_ne_k[ik]
            e_hcore += weights[ik] * float(
                np.real(np.einsum("ij,ji->", D_a_k_new[ik], Hcore_k))
                + np.real(np.einsum("ij,ji->", D_b_k_new[ik], Hcore_k))
            )
        E = e_hcore + e_hartree + e_xc_iter + E_nn

        dE = E - E_prev
        dD = float(
            np.sqrt(
                np.linalg.norm(D_a_new - D_a) ** 2
                + np.linalg.norm(D_b_new - D_b) ** 2
            )
        )
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_iter),
            }
        )
        # Per-iteration progress hook -> live QVF checkpoint cadence when the
        # runner passes a checkpoint-wrapped ``plog`` (see run_periodic_rhf_gpw).
        plog.iteration(it, energy=float(E), dE=float(dE), grad=float(dD))
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D_a_k, D_b_k = D_a_k_new, D_b_k_new
            D_a, D_b = D_a_new, D_b_new
            C_a_k, C_b_k = new_C_a, new_C_b
            e_a_k, e_b_k = new_e_a, new_e_b
            break
        D_a_k, D_b_k = D_a_k_new, D_b_k_new
        D_a, D_b = D_a_new, D_b_new
        C_a_k, C_b_k = new_C_a, new_C_b
        e_a_k, e_b_k = new_e_a, new_e_b
        E_prev = E

    # ROKS rebuilds the nonlinear Hartree and XC energies once at the returned
    # restricted density. The SCF iteration evaluates those terms at D_n while
    # diagonalisation produces D_(n+1); convergence makes the difference tiny,
    # but the public result and last trace row must describe exactly one state.
    if _restricted_open:
        rho_a = _collocate_spin(D_a, D_a_k)
        rho_b = _collocate_spin(D_b, D_b_k)
        rho_tot = rho_a + rho_b
        V_H = _core.solve_poisson_coulomb(rho_tot, grid.lattice_bohr)
        if use_bloch_compact:
            e_hartree = 0.5 * float(
                np.sum(rho_tot * V_H) * grid.voxel_volume_bohr3
            )
            tau_a_bloch = None
            tau_b_bloch = None
            if is_mgga:
                tau_a_bloch = _bloch_tau(D_a_k)
                tau_b_bloch = _bloch_tau(D_b_k)
            e_xc_iter, *_ = _xc_effective_potentials_polarised_grid(
                rho_a,
                rho_b,
                grid,
                func,
                tau_alpha=tau_a_bloch,
                tau_beta=tau_b_bloch,
                cache=collocation_cache,
            )
        else:
            J_final = project_potential_to_ao(
                basis, V_H, grid, cache=collocation_cache
            )
            e_hartree = 0.5 * float(
                np.einsum("ij,ij->", D_a + D_b, J_final)
            )
            e_xc_iter, *_ = _evaluate_xc_polarised_on_grid(
                rho_a,
                rho_b,
                grid,
                func,
                basis=basis,
                density_matrix_alpha=D_a,
                density_matrix_beta=D_b,
                cache=collocation_cache,
            )

    # Breakdown from the per-k Bloch-summed Hcore(k) pieces (same sum that
    # produced E; a Gamma-only re-evaluation drops the per-k phases).
    e_kin = 0.0
    e_ne = 0.0
    for ik in range(n_k):
        e_kin += weights[ik] * float(
            np.real(np.einsum("ij,ji->", D_a_k[ik], T_k[ik]))
            + np.real(np.einsum("ij,ji->", D_b_k[ik], T_k[ik]))
        )
        e_ne += weights[ik] * float(
            np.real(np.einsum("ij,ji->", D_a_k[ik], V_ne_k[ik]))
            + np.real(np.einsum("ij,ji->", D_b_k[ik], V_ne_k[ik]))
        )
    breakdown = GpwEnergyBreakdown(
        e_kinetic=e_kin,
        e_nuclear_attraction=e_ne,
        e_hartree=e_hartree,
        e_hf_exchange=0.0,
        e_nuclear_repulsion=E_nn,
        e_total=e_kin + e_ne + e_hartree + e_xc_iter + E_nn,
        grid=grid,
        e_xc=e_xc_iter,
        functional=str(functional),
    )
    if _restricted_open and scf_trace:
        scf_trace[-1] = {
            **scf_trace[-1],
            "energy": float(breakdown.e_total),
            "e_xc": float(e_xc_iter),
        }

    result_type = (
        GpwRoksMultiKScfResult if _restricted_open else GpwUksMultiKScfResult
    )
    return result_type(
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, requested_initial_guess, restarted=input_restart_supplied),
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=breakdown.e_total,
        breakdown=breakdown,
        density_alpha=D_a,
        density_beta=D_b,
        mo_coeffs_alpha_k=tuple(C_a_k),
        mo_coeffs_beta_k=tuple(C_b_k),
        mo_energies_alpha_k=tuple(e_a_k),
        mo_energies_beta_k=tuple(e_b_k),
        n_alpha=n_alpha_res,
        n_beta=n_beta_res,
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        kmesh=kmesh,
        scf_trace=tuple(scf_trace),
    )


def run_periodic_roks_gpw_multi_k(
    system,
    basis,
    kmesh,
    *,
    functional: str,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 80,
    conv_tol_energy: float = 1e-8,
    conv_tol_density: float = 1e-6,
    initial_density_k: Optional[
        Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]
    ] = None,
    initial_guess: Optional[Union[str, _core.InitialGuess]] = "AUTO",
    atomic_spins: Optional[Sequence[int]] = None,
    quiet: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
) -> GpwRoksMultiKScfResult:
    """Multi-k pure-DFT GPW ROKS with one restricted orbital set per k."""
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=True, atomic_spins=atomic_spins,
        restart_supplied=initial_density_k is not None,
    )
    return run_periodic_uks_gpw_multi_k(
        system,
        basis,
        kmesh,
        functional=functional,
        n_alpha=n_alpha,
        n_beta=n_beta,
        grid=grid,
        cutoff_ha=cutoff_ha,
        max_iter=max_iter,
        conv_tol_energy=conv_tol_energy,
        conv_tol_density=conv_tol_density,
        initial_density_k=initial_density_k,
        initial_guess=initial_guess,
        quiet=quiet,
        progress=progress,
        _restricted_open=True,
               atomic_spins=atomic_spins,
    )
