"""Restricted open-shell Kohn--Sham (ROKS).

The DFT counterpart of :mod:`vibeqc.rohf`: a spin-restricted open-shell
Kohn--Sham determinant (one set of spatial orbitals; doubly-occupied +
singly-occupied + virtual) giving a spin-pure density.  It reuses ROHF's
Roothaan single-effective-Fock coupling (:func:`vibeqc.rohf.run_roothaan_scf`)
verbatim --- the *only* difference from ROHF is the per-spin Fock build:

    F_sigma = Hcore + J(D_a + D_b) - K_xc(D_sigma) + V_xc_sigma

with the spin-polarised XC potential ``V_xc_sigma`` from libxc and
``K_xc = cam_alpha*K + cam_beta*K_erf`` for global and range-separated
hybrids.  Energy::

    E = Tr[Dt Hcore] + 1/2 Tr[Dt J]
        - 1/2 (Tr[Da Kxc_a] + Tr[Db Kxc_b]) + E_xc[rho_a, rho_b] + E_nuc

The Roothaan coupling that follows (closed/open/virtual blocks) is
identical to ROHF; see :func:`vibeqc.rohf.roothaan_effective_fock`.

Scope: molecular LDA, GGA, meta-GGA, global-hybrid and range-separated
hybrid functionals.  Direct ``run_roks`` on a double hybrid raises
:class:`NotImplementedError`; use the double-hybrid dispatcher so the
ROKS SCF half is combined with the required perturbative correlation.
See ``docs/user_guide/functionals.md`` for the supported functional
families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .rohf import (
    ROHFOptions,
    ROHFResult,
    _orthonormaliser,
    _spin_partition,
    require_nonnegative_max_iter,
    run_roothaan_scf,
)

__all__ = ["ROKSOptions", "ROKSResult", "run_roks", "build_uks_xc_potential"]


@dataclass
class ROKSOptions(ROHFOptions):
    """:class:`ROHFOptions` plus the KS functional and integration grid."""

    functional: str = ""
    grid: Any = None  # _vibeqc_core.GridOptions, or None for the default
    # ROKS uses the fractional frontier-shell path by default so degenerate
    # terms such as OH(2Pi) do not collapse onto one arbitrary SOMO component.
    fractional_open_shell: bool = True


@dataclass
class ROKSResult(ROHFResult):
    """ROHF result surface + KS energy components (``e_xc`` triggers the
    energy-components block in the SCF-log writer)."""

    e_xc: float = 0.0
    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    e_nuclear: float = 0.0
    functional: str = ""
    method: str = "roks"


# ---------------------------------------------------------------------------
# Spin-polarised XC potential matrix (LDA + GGA + meta-GGA)
# ---------------------------------------------------------------------------


def build_uks_xc_potential(
    basis: Any,
    dma: np.ndarray,
    dmb: np.ndarray,
    func: Any,
    grid: Any,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(V_xc_alpha, V_xc_beta, E_xc)`` from the spin densities on a grid.

    Mirrors the validated UKS assembly in
    :mod:`vibeqc.hessian_analytic_uks` (libxc ``eval_polarised`` + AO
    values on the integration grid).  Handles LDA, GGA and meta-GGA
    functionals, including the per-spin kinetic-energy-density potential
    and VV10 nonlocal correlation when requested by the functional.

    ``E_xc = sum_g w_g e_xc(g)`` follows vibe-qc's grid convention (the
    ``eval_*`` energy density is already per unit volume; see
    :func:`vibeqc.parity._xc_energy_unrestricted`).
    """
    from . import _vibeqc_core as _core

    # Full-grid providers return adjoints of one integrated energy and cannot
    # be evaluated through the pointwise libxc methods below.  Reuse the same
    # native spin-resolved projector as UKS; ROKS differs only in how its two
    # Fock matrices are coupled into one restricted orbital update.
    if bool(getattr(func, "is_external", False)):
        v_alpha, v_beta, e_xc = _core.evaluate_uks_xc_potential(
            func,
            basis,
            grid,
            np.asarray(dma, dtype=float),
            np.asarray(dmb, dtype=float),
        )
        return (
            np.asarray(v_alpha, dtype=float),
            np.asarray(v_beta, dtype=float),
            float(e_xc),
        )

    pts = np.asarray(grid.points)
    weights = np.asarray(grid.weights)
    chi, dchi_x, dchi_y, dchi_z = _core.evaluate_ao_with_gradient(basis, pts)
    chi = np.asarray(chi)

    chiD_a = chi @ dma
    chiD_b = chi @ dmb
    rho_a = (chiD_a * chi).sum(axis=1)
    rho_b = (chiD_b * chi).sum(axis=1)

    is_gga = func.kind == _core.XCKind.GGA
    is_mgga = func.kind == _core.XCKind.MGGA
    needs_grad = is_gga or is_mgga or bool(getattr(func, "needs_vv10", False))
    dchi = [np.asarray(dchi_x), np.asarray(dchi_y), np.asarray(dchi_z)]
    if needs_grad:
        gax = 2.0 * (chiD_a * dchi[0]).sum(axis=1)
        gay = 2.0 * (chiD_a * dchi[1]).sum(axis=1)
        gaz = 2.0 * (chiD_a * dchi[2]).sum(axis=1)
        gbx = 2.0 * (chiD_b * dchi[0]).sum(axis=1)
        gby = 2.0 * (chiD_b * dchi[1]).sum(axis=1)
        gbz = 2.0 * (chiD_b * dchi[2]).sum(axis=1)
        sigma_aa = gax**2 + gay**2 + gaz**2
        sigma_ab = gax * gbx + gay * gby + gaz * gbz
        sigma_bb = gbx**2 + gby**2 + gbz**2
    else:
        zero = np.zeros_like(rho_a)
        sigma_aa = sigma_ab = sigma_bb = zero
        gax = gay = gaz = gbx = gby = gbz = None

    if is_mgga:
        tau_a = np.zeros_like(rho_a)
        tau_b = np.zeros_like(rho_b)
        for dchi_c in dchi:
            tau_a += (dchi_c @ dma * dchi_c).sum(axis=1)
            tau_b += (dchi_c @ dmb * dchi_c).sum(axis=1)
        tau_a *= 0.5
        tau_b *= 0.5
        (
            exc,
            v_rho_a,
            v_rho_b,
            v_sigma_aa,
            v_sigma_ab,
            v_sigma_bb,
            v_tau_a,
            v_tau_b,
        ) = func.eval_polarised_mgga(
            rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b
        )
    else:
        (
            exc,
            v_rho_a,
            v_rho_b,
            v_sigma_aa,
            v_sigma_ab,
            v_sigma_bb,
        ) = func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb)
        v_tau_a = v_tau_b = None

    exc = np.asarray(exc)
    e_xc = float(np.sum(weights * exc))

    if bool(getattr(func, "needs_vv10", False)):
        rho_tot = rho_a + rho_b
        sigma_tot = sigma_aa + 2.0 * sigma_ab + sigma_bb
        e_vv10, v_rho_vv10, v_sigma_vv10 = _core.compute_vv10(
            pts,
            weights,
            rho_tot,
            sigma_tot,
            float(getattr(func, "vv10_b", 0.0)),
            float(getattr(func, "vv10_C", 0.0)),
        )
        e_xc += float(e_vv10)
        v_rho_a = np.asarray(v_rho_a) + np.asarray(v_rho_vv10)
        v_rho_b = np.asarray(v_rho_b) + np.asarray(v_rho_vv10)
        v_sigma_aa = np.asarray(v_sigma_aa) + np.asarray(v_sigma_vv10)
        v_sigma_ab = np.asarray(v_sigma_ab) + 2.0 * np.asarray(v_sigma_vv10)
        v_sigma_bb = np.asarray(v_sigma_bb) + np.asarray(v_sigma_vv10)

    # LDA piece, per spin: V_xc_sigma_uv = sum_g w(g) v_rho_sigma(g) chi_u(g) chi_v(g)
    w_va = weights * np.asarray(v_rho_a)
    w_vb = weights * np.asarray(v_rho_b)
    v_xc_a = chi.T @ (w_va[:, None] * chi)
    v_xc_b = chi.T @ (w_vb[:, None] * chi)

    if needs_grad:
        # GGA sigma piece (chain rule on sigma_aa/ab/bb), symmetrised.
        u_a = 2.0 * weights * np.asarray(v_sigma_aa)
        u_ab = weights * np.asarray(v_sigma_ab)
        u_b = 2.0 * weights * np.asarray(v_sigma_bb)
        flux_a = (
            (u_a[:, None] * dchi[0]) * gax[:, None]
            + (u_a[:, None] * dchi[1]) * gay[:, None]
            + (u_a[:, None] * dchi[2]) * gaz[:, None]
        )
        flux_a += (
            (u_ab[:, None] * dchi[0]) * gbx[:, None]
            + (u_ab[:, None] * dchi[1]) * gby[:, None]
            + (u_ab[:, None] * dchi[2]) * gbz[:, None]
        )
        flux_b = (
            (u_b[:, None] * dchi[0]) * gbx[:, None]
            + (u_b[:, None] * dchi[1]) * gby[:, None]
            + (u_b[:, None] * dchi[2]) * gbz[:, None]
        )
        flux_b += (
            (u_ab[:, None] * dchi[0]) * gax[:, None]
            + (u_ab[:, None] * dchi[1]) * gay[:, None]
            + (u_ab[:, None] * dchi[2]) * gaz[:, None]
        )
        fu_a = flux_a.T @ chi
        fu_b = flux_b.T @ chi
        v_xc_a = v_xc_a + fu_a + fu_a.T
        v_xc_b = v_xc_b + fu_b + fu_b.T

    if is_mgga:
        # Kinetic-energy-density piece:
        # V_tau = 1/2 sum_c dchi_c^T diag(w * v_tau) dchi_c.
        w_tau_a = 0.5 * weights * np.asarray(v_tau_a)
        w_tau_b = 0.5 * weights * np.asarray(v_tau_b)
        for dchi_c in dchi:
            v_xc_a = v_xc_a + dchi_c.T @ (w_tau_a[:, None] * dchi_c)
            v_xc_b = v_xc_b + dchi_c.T @ (w_tau_b[:, None] * dchi_c)

    return v_xc_a, v_xc_b, e_xc


# ---------------------------------------------------------------------------
# KS Fock builder + driver
# ---------------------------------------------------------------------------


def _refuse_plain_double_hybrid(
    func: Any,
    *,
    route: str,
    allow_double_hybrid_scf: bool = False,
) -> None:
    """Reject public KS routes that return only a double hybrid's SCF part."""
    if getattr(func, "is_double_hybrid", False) and not allow_double_hybrid_scf:
        raise NotImplementedError(
            f"{route} with the double hybrid functional {func.name!r} would "
            "return only its SCF component and omit the required perturbative "
            "correlation. Use vibeqc.run_double_hybrid(..., "
            f"{func.name!r}) or the matching complete wrapper "
            "(run_b2plyp, run_dsd_pbep86, run_revdsd_pbep86, or run_pwpb95)."
        )


def _refuse_unsupported(func: Any, *, allow_double_hybrid_scf: bool = False) -> None:
    """Reject direct double-hybrid ROKS calls.

    ``allow_double_hybrid_scf`` lets the open-shell double-hybrid dispatcher
    (:func:`vibeqc.run_double_hybrid`) run the *SCF piece* of a double hybrid
    through ROKS; it then adds the ROHF-MP2 correction itself. A direct
    ``run_roks`` on a double-hybrid functional still raises (the SCF energy
    alone is an incomplete double-hybrid result).
    """
    _refuse_plain_double_hybrid(
        func,
        route="ROKS",
        allow_double_hybrid_scf=allow_double_hybrid_scf,
    )


def _make_ks_fock_builder(
    basis: Any,
    hcore: np.ndarray,
    build_j: Callable[[np.ndarray], np.ndarray],
    build_k: Callable[[np.ndarray], np.ndarray],
    build_k_erf: Optional[Callable[[np.ndarray, float], np.ndarray]],
    func: Any,
    grid: Any,
    cam_alpha: float,
    cam_beta: float,
    rsh_omega: float,
):
    """Spin-polarised KS per-spin Fock builder for the Roothaan loop.

    Returns ``builder(dma, dmb) -> (focka, fockb, e_elec)`` plus, stashed
    on the closure, the last-evaluated KS energy components (so the driver
    can report them without a second Fock build)."""
    hcore = np.asarray(hcore, dtype=float)
    comps: dict = {}
    need_k = (cam_alpha != 0.0) or (cam_beta != 0.0)

    def builder(
        dma: np.ndarray, dmb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        dm_total = dma + dmb
        j = np.asarray(build_j(dm_total))
        e_coulomb = 0.5 * float(np.einsum("ij,ij->", dm_total, j))
        if need_k:
            ka = cam_alpha * np.asarray(build_k(dma))
            kb = cam_alpha * np.asarray(build_k(dmb))
            if cam_beta != 0.0:
                if build_k_erf is None:
                    raise RuntimeError(
                        "ROKS range-separated exchange needs build_K_erf; "
                        "use the direct molecular JK builder."
                    )
                ka = ka + cam_beta * np.asarray(build_k_erf(dma, rsh_omega))
                kb = kb + cam_beta * np.asarray(build_k_erf(dmb, rsh_omega))
            e_hf_exchange = (
                -0.5
                * (
                    float(np.einsum("ij,ij->", dma, ka))
                    + float(np.einsum("ij,ij->", dmb, kb))
                )
            )
            fock_x_a = -ka
            fock_x_b = -kb
        else:
            e_hf_exchange = 0.0
            fock_x_a = fock_x_b = np.zeros_like(j)
        v_xc_a, v_xc_b, e_xc = build_uks_xc_potential(basis, dma, dmb, func, grid)
        focka = hcore + j + fock_x_a + v_xc_a
        fockb = hcore + j + fock_x_b + v_xc_b
        e_one = float(np.einsum("ij,ij->", dm_total, hcore))
        e_elec = e_one + e_coulomb + e_hf_exchange + e_xc
        comps.update(e_xc=e_xc, e_coulomb=e_coulomb, e_hf_exchange=e_hf_exchange)
        return focka, fockb, e_elec

    builder.components = comps  # type: ignore[attr-defined]
    return builder


def run_roks(
    molecule: Any,
    basis: Any,
    options: Optional[ROKSOptions] = None,
    *,
    functional: Optional[str] = None,
    initial_density: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    read_from: Optional[object] = None,
    fragments: Optional[object] = None,
    _allow_double_hybrid_scf: bool = False,
) -> ROKSResult:
    """Restricted open-shell Kohn--Sham single point.

    Open-shell counterpart of :func:`vibeqc.run_rks` with a single
    spin-restricted orbital set (spin-pure density).  ``functional`` may
    be passed explicitly or set on ``options.functional``.

    ``_allow_double_hybrid_scf`` is internal: it lets
    :func:`vibeqc.run_double_hybrid` run the SCF half of an open-shell
    double hybrid through ROKS (it adds the ROHF-MP2 correction itself).
    """
    if options is None:
        options = ROKSOptions()
    require_nonnegative_max_iter(options.max_iter, route="run_roks")
    from .rohf import (
        _prepare_ecp_options,
        _prepare_initial_guess,
        _stamp_ecp_provenance,
    )

    _prepare_ecp_options(options, molecule, basis, route="run_roks")
    if read_from is not None or fragments is not None:
        if initial_density is not None:
            raise ValueError("ROKS: pass either initial_density or a READ/FRAGMO source")
        from .guess import prepare_molecular_guess_source
        prepare_molecular_guess_source(
            "roks", options, molecule, basis, read_from=read_from, fragments=fragments,
        )
    selection = _prepare_initial_guess(molecule, options, initial_density)
    from .guess import guess_ecp_context, validate_guess_ecp
    validate_guess_ecp(selection.effective, guess_ecp_context(options), molecule)
    fname = functional or options.functional or "lda"
    options.functional = fname

    from . import _vibeqc_core as _core

    func = _core.Functional(fname, 2)  # spin-polarised
    _refuse_unsupported(func, allow_double_hybrid_scf=_allow_double_hybrid_scf)
    cam_alpha = float(getattr(func, "cam_alpha", 0.0))
    cam_beta = float(getattr(func, "cam_beta", 0.0))
    rsh_omega = float(getattr(func, "rsh_omega", 0.0))

    s = np.asarray(_core.compute_overlap(basis), dtype=float)
    from .ecp_metadata import one_electron_hamiltonian
    from .rohf import _initial_densities, _resolve_jk_builder

    hcore, e_nuc, n_valence, _z_eff = one_electron_hamiltonian(
        molecule, basis, options
    )
    hcore = np.asarray(hcore, dtype=float)
    e_nuc = float(e_nuc)
    n_alpha, n_beta = _spin_partition(int(n_valence), molecule.multiplicity)

    grid_opts = options.grid if options.grid is not None else _core.GridOptions()
    grid = _core.build_grid(molecule, grid_opts)

    if cam_beta != 0.0:
        # The RI/COSX builders do not expose erf-attenuated exchange; mirror
        # the native RKS/UKS policy and force the direct molecular JK builder
        # for CAM/RSH functionals.
        jk = _core.make_direct_jk_builder(basis)
        build_j = lambda d: np.asarray(jk.build_J(d))
        build_k = lambda d: np.asarray(jk.build_K(d))
        build_k_erf = lambda d, omega: np.asarray(jk.build_K_erf(d, omega))
    else:
        build_j, build_k = _resolve_jk_builder(molecule, basis, options)
        build_k_erf = None
    fock_builder = _make_ks_fock_builder(
        basis,
        hcore,
        build_j,
        build_k,
        build_k_erf,
        func,
        grid,
        cam_alpha,
        cam_beta,
        rsh_omega,
    )

    x = _orthonormaliser(s, options.linear_dep_threshold)
    if initial_density is not None:
        init_a = np.asarray(initial_density[0], dtype=float)
        init_b = np.asarray(initial_density[1], dtype=float)
    else:
        init_a, init_b = _initial_densities(
            molecule, basis, n_alpha, n_beta, options, s, hcore, x
        )

    base = run_roothaan_scf(
        s, hcore, e_nuc, n_alpha, n_beta, fock_builder, options, init_a, init_b
    )
    comps = fock_builder.components  # consistent with the converged density

    result = ROKSResult(
        restart_basis=basis,
        energy=base.energy,
        guess_selection=selection,
        e_electronic=base.e_electronic,
        n_iter=base.n_iter,
        converged=base.converged,
        mo_energies=base.mo_energies,
        mo_coeffs=base.mo_coeffs,
        mo_occupations=base.mo_occupations,
        density=base.density,
        density_alpha=base.density_alpha,
        density_beta=base.density_beta,
        fock=base.fock,
        fock_alpha=base.fock_alpha,
        fock_beta=base.fock_beta,
        s_squared=base.s_squared,
        s_squared_ideal=base.s_squared_ideal,
        n_alpha=base.n_alpha,
        n_beta=base.n_beta,
        scf_trace=base.scf_trace,
        # Canonicalisation is selected inside run_roothaan_scf: integer
        # shells use Guest-Saunders, while a detected fractional frontier
        # shell retains its symmetry-preserving Roothaan eigenpairs. Carry
        # that provenance and the explicit Roothaan fields across so the
        # result and .system manifest describe the same convention.
        rohf_canonicalization=base.rohf_canonicalization,
        mo_energies_roothaan=base.mo_energies_roothaan,
        mo_coeffs_roothaan=base.mo_coeffs_roothaan,
        e_xc=comps.get("e_xc", 0.0),
        e_coulomb=comps.get("e_coulomb", 0.0),
        e_hf_exchange=comps.get("e_hf_exchange", 0.0),
        e_nuclear=e_nuc,
        functional=fname,
    )
    _stamp_ecp_provenance(result, molecule, options)
    return result
