"""BIPOLE periodic atomic gradient.

⚠️  RESEARCH PREVIEW -- the **RHF and UHF Γ-only** hybrid gradients are
complete for **general crystals** (symmetric or asymmetric, 1-cell or
multi-cell): they match the exact finite-difference gradient to ~1e-4-1e-7
Ha/bohr, with the full local-energy orbital relaxation included via the
Bloch-CPHF Z-vector (``_bloch_cphf_relaxation`` for RHF;
``_bloch_cphf_relaxation_open`` -- coupled two-spin -- for UHF). RHF multi-k
now has the corrected W/J^LR convention plus a diagonal Z-response pinned on
the maintained [2,1,1] symmetric/asymmetric regressions, and UHF multi-k has
the analogous coupled-spin Z response pinned on the maintained high-spin and
asymmetric [2,1,1] regressions. "Hybrid" means the four Ewald
electrostatic kernels are analytic, while the EXT EL-SPHEROPOLE
Hellmann-Feynman term is central-differenced at fixed density because the
native libint derivative path is disabled until it is safe. Gamma-local RKS/UKS
now also have fixed-density XC Pulay terms plus semi-numerical KS Bloch-CPHF
responses pinned on maintained LDA asymmetric cells. The broader path (general
KS beyond the maintained diagnostics and general UHF/KS multi-k) is **not yet
certified**. For production forces / geometry optimisation use the exact
finite-difference gradient
:func:`compute_bipole_gradient_fd` -- it central-differences the real total
energy, so it is correct by construction (cost: 6N+1 SCFs).
:mod:`vibeqc.bipole_optimize` and the periodic NEB driver both default to
the FD path.

The gauge-consistent hybrid gradient (RHF/UHF Γ -- research preview)
-------------------------------------------------------------------
The BIPOLE *energy* is built in CRYSTAL's Ewald electrostatic gauge --
``E_nn`` via ``ewald_nuclear_repulsion``, ``V_ne`` via screened-erfc +
reciprocal AO-pair-FT + neutralising background, ``J = J_SR(w) +
J_LR(w)``, plus a post-SCF EXT EL-SPHEROPOLE term. An earlier draft
differentiated full-Coulomb direct-space kernels in a *different* gauge
from the energy and the residual had the wrong sign even at Γ. That is
now replaced by Ewald-gauge derivative kernels that match the energy
term-by-term. Four terms are analytic; the EXT EL-SPHEROPOLE term is
central-differenced at fixed density while the native derivative path is
disabled:

- ``ewald_nuclear_repulsion_gradient`` -- Ewald ``E_nn`` gradient.
- the Ewald-``V_ne`` reciprocal gradient (screened-erfc + AO-pair-FT).
- the screened ``J_SR(w)`` gradient.
- the ``J_LR(w)`` reciprocal-space gradient (mixed Bloch/local density
  convention, ``_j_long_range_ewald_gradient``).
- the EXT EL-SPHEROPOLE gradient, central-differenced at fixed density.
  The native libint ``emultipole2`` derivative engine is not called from
  this path because the current vendored build can segfault in that kernel.

plus the correct neutralising-background ``V_bg.S`` derivative and the
local-energy Pulay term ``W' = 2.C_occ(C_occᵀ.dE/dP(0).C_occ).C_occᵀ``
built from the home-cell block of ``dE/dP`` (NOT the diagonalised Bloch
``F(Γ)`` -- the Γ-only density-locality projection makes the energy a
local contraction ``Tr[D(0).H(0)]``).

UHF Γ (done)
------------
UHF uses the same five Ewald-gauge kernels with two open-shell
adjustments: the exchange gradient is spin-resolved
(``2.(dE_x[P_a] + dE_x[P_b])`` -- exact closed-shell reduction when
``P_a=P_b``), and the Pulay energy-weighted density is the per-spin
local-energy block ``W = W_a + W_b`` with
``dE/dP_s(0) = shared - a_HF.K[P_s](0)``
(``_corrected_w_gamma_open``). Exact (~1e-7) for symmetric/well-localised
spin-polarised cells (e.g. triplet H₂).

RHF general crystals (done) -- the Bloch-CPHF Z-vector
-----------------------------------------------------
The BIPOLE Γ SCF diagonalises the Bloch sum ``F(Γ)=S_g F(g)`` while the
energy is the LOCAL contraction ``Tr[D(0).H(0)]``, so for an *asymmetric
multi-cell* crystal ``F(0) != F(Γ)``: the converged density is not
stationary for the local energy and ``occ-virt(dE_local/dP(0)) != 0`` -- both
the SCF-Fock mismatch (~8e-2 Ha/bohr) AND the post-SCF EXT EL-SPHEROPOLE
(~1e-3 on asymmetric 1-cell). The no-CPHF local-energy Pulay misses the
matching orbital relaxation; ``_bloch_cphf_relaxation`` (a full Bloch-CPHF
Z-vector -- analytic Hessian + solve) recovers it. The CPHF right-hand side
``dB0/dR`` is selected by ``cphf_rhs=`` (see ``_bloch_cphf_rhs_analytic``):
``"hybrid"`` (default) is fully analytic except a 6N cheap-build local
renormalisation and matches FD to ~3e-5; ``"analytic"`` is fully analytic
(no FD) at ~2e-3; ``"seminumeric"`` is the original 6N-full-Fock-build FD
reference. The decisive ingredient is the **Bloch metric**: the SCF
diagonalises ``F(Γ)=S_g F(g)``, so the response density is contracted
*broadcast into every cell* (not home-only). Exact across all RHF regimes:
symmetric multi-cell ~1e-7, asymmetric 1-cell ~1e-7, asymmetric multi-cell
~4e-5 (the general low-symmetry crystal). **UHF** uses the coupled two-spin
counterpart ``_bloch_cphf_relaxation_open`` (the Coulomb response couples
a<->b; the dense Hessian is solved by pseudo-inverse to absorb
degenerate-shell null modes), same ``cphf_rhs`` modes, exact to ~4e-5 on
asymmetric open-shell multi-cell cells (BeH doublet).

Still gated (research preview)
------------------------------
- Corrected-gauge integer-occupation RHF now differentiates the production
  padded radial M5 short-range domain at multi-k. The finite-domain density
  adjoint and full real-linear k-coupled response are pinned on asymmetric
  BeH2 [2,1,1] to 2.6e-5 Ha/bohr versus the converged-energy finite
  difference. The corresponding UHF route preserves the full BvK support of
  ``D_alpha + D_beta`` and applies spin-resolved adjoints plus the coupled
  alpha/beta response; asymmetric BeH [2,1,1] is FD-clean to 6.9e-8
  Ha/bohr. Padded multi-k KS, pair-resolved, and fractional routes remain
  fail-closed.
- Multi-k KS-CPHF: the legacy-gauge multi-k KS path uses the diagonal-Z
  approximation (warns); the full complex k-space coupled-perturbed
  response is deferred. The corrected-gauge multi-k KS path is
  variational (needs no CPHF) and already lands.
- RKS/UKS Gamma-local have fixed-grid/moving-grid LDA/GGA/meta-GGA XC Pulay
  force kernels and semi-numerical KS Bloch-CPHF responses pinned on maintained
  LDA asymmetric cells. Broader KS certification remains pending.
- Fractional-occupation corrected-gauge gradients are maintained on the
  historical radial domain and at multi-k. They fail closed on the padded M5
  Gamma domain because its finite-domain J/K map is not variational with
  respect to fractional occupations; use the exact finite-difference path.
- The multi-k Pulay energy-weighted density is inverse-Bloch folded
  ``W(g) = S_k w_k Re[e^{-ik.g} W(k)]``. Pass ``kmesh=`` to enable the
  multi-k fold; without it the term falls back to the Γ broadcast and
  warns.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    EwaldOptions,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    ShellInfo,
    build_jk_2e_real_space_domains_gamma_derivative,
    build_jk_2e_real_space_domains_density_derivative,
    compute_overlap_lattice,
    direct_lattice_cells,
    eri_lattice_gradient_contribution,
    ewald_nuclear_repulsion_gradient,
    kinetic_lattice_gradient_contribution,
    make_lattice_matrix_set,
    nuclear_erfc_lattice_gradient_contribution,
    nuclear_lattice_gradient_contribution,
    nuclear_repulsion_gradient_per_cell,
    overlap_lattice_gradient_contribution,
)
from .pbc_bipole_fock import (
    _sr_internal_cells, _sr_padded_lat_opts, _reciprocal_blocks_on_output,
)
from .pbc_bipole import PBCBipoleRHFResult
from .pbc_bipole_rks import PBCBipoleRKSResult
from .pbc_bipole_uhf import PBCBipoleUHFResult
from .pbc_bipole_uks import PBCBipoleUKSResult

__all__ = [
    "compute_bipole_gradient_rhf",
    "compute_bipole_gradient_uhf",
    "compute_bipole_gradient_rks",
    "compute_bipole_gradient_uks",
    "compute_bipole_gradient_fd",
    "compute_stress_tensor",
]


def _reject_m5_domain_analytic_gradient(result, method: str) -> None:
    """Fail closed until derivatives support the M5 Fock domains."""
    # BIPOLE-EXACT-ZONE increment 1: a restricted exact output zone has
    # no derivative closure -- refuse before the domain checks below so
    # the caller lands on the FD production force path.
    zone = getattr(result, "exact_zone_bohr", None)
    if zone is not None:
        raise NotImplementedError(
            f"compute_bipole_gradient_{method}: the SCF energy used a "
            f"restricted exact bielectronic zone (exact_zone_bohr = "
            f"{float(zone):.6g}), which has no analytic-derivative "
            "closure; use compute_bipole_gradient_fd (the production "
            "force path) or omit exact_zone_bohr"
        )
    extent = getattr(result, "sr_image_extent_bohr", None)
    pair_domain = bool(getattr(result, "pair_resolved_fock_domain", False))
    mo_coeffs = getattr(
        result,
        "mo_coeffs" if method in {"rhf", "rks"} else "mo_coeffs_alpha",
        (),
    )
    fractional_uhf = method == "uhf" and (
        float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0
        or any(
            np.asarray(occ).size > 0
            and bool(
                np.any(
                    (np.asarray(occ, dtype=float) > 1.0e-12)
                    & (np.asarray(occ, dtype=float) < 1.0 - 1.0e-12)
                )
            )
            for occ in list(getattr(result, "occupations_alpha", ()))
            + list(getattr(result, "occupations_beta", ()))
        )
    )
    if fractional_uhf:
        raise NotImplementedError(
            "compute_bipole_gradient_uhf: finite-temperature or fractional-"
            "occupation UHF analytic gradients are not implemented. The "
            "current analytic energy-weighted density uses integer Aufbau "
            "occupations and therefore is not the derivative of the Mermin "
            "free energy; use compute_bipole_gradient_fd."
        )
    fractional_uks = method == "uks" and _is_fractional_ks_occupation(
        result, "uks"
    )
    fractional_rks = method == "rks" and _is_fractional_ks_occupation(
        result, "rks"
    )
    padded_corrected_gamma = (
        method in {"rhf", "uhf", "rks", "uks"}
        and extent is not None
        and bool(getattr(result, "exchange_ewald_split", False))
        and len(mo_coeffs) == 1
        and not fractional_rks
        and not fractional_uks
    )
    padded_corrected_multik_hf = (
        method in {"rhf", "uhf"}
        and extent is not None
        and bool(getattr(result, "exchange_ewald_split", False))
        and len(mo_coeffs) > 1
        and not pair_domain
    )
    pair_supported = (
        pair_domain
        and padded_corrected_gamma
        and method in {"rhf", "uhf", "rks", "uks"}
    )
    if (pair_domain and not pair_supported) or (
        extent is not None
        and not (padded_corrected_gamma or padded_corrected_multik_hf)
    ):
        details = []
        if extent is not None:
            details.append(
                f"a padded erfc SR ket-image domain ({float(extent):.6g} bohr)"
            )
        if pair_domain:
            details.append("the group-invariant pair-resolved Fock domain")
        if fractional_rks or fractional_uks:
            details.append("fractional occupations")
        raise NotImplementedError(
            f"compute_bipole_gradient_{method}: the SCF energy used "
            f"{' and '.join(details)}, which is outside this analytic "
            "driver's validated M5 envelope. Corrected-gauge RHF/UHF and "
            "integer-occupation RKS/UKS Gamma support padded radial and "
            "pair-resolved domains; corrected-gauge RHF/UHF also support "
            "the padded radial multi-k domain. Other "
            "M5 combinations remain fail-closed. Use "
            "compute_bipole_gradient_fd "
            "for the production energy, or run the historical diagnostic "
            "with sr_image_precision=None and "
            "use_fock_symmetry_reduce=False."
        )


def _m5_gamma_gradient_domain(
    result,
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    density_cells,
) -> dict:
    """Resolve the padded Gamma derivative domains used by an SCF result."""
    extent = getattr(result, "sr_image_extent_bohr", None)
    if extent is None:
        return {}
    resolved = {
        "sr_image_extent_bohr": float(extent),
        "sr_density_cells": list(density_cells),
    }
    if not bool(getattr(result, "pair_resolved_fock_domain", False)):
        return resolved

    from .bipole_symmetry_fock import pair_resolved_fock_mapping

    mapping = pair_resolved_fock_mapping(system, basis, lattice_opts)
    if not mapping.pair_resolved or mapping.output_masks is None:
        raise RuntimeError(
            "the SCF reports a pair-resolved Fock domain, but its SYM3b "
            "mapping could not be reconstructed for the analytic gradient"
        )
    reach = max(
        (
            float(np.linalg.norm(np.asarray(cell.r_cart, dtype=float)))
            for cell in mapping.cells
        ),
        default=0.0,
    )
    resolved.update(
        sr_image_extent_bohr=max(float(extent), reach + 1.0e-6),
        sr_output_cells=list(mapping.cells),
        sr_output_shell_masks=list(mapping.output_masks),
        sr_density_domain=mapping.density_domain,
        lr_output_cells=list(mapping.cells),
    )
    return resolved


def _m5_gamma_jk_adjoint_correction(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    density_block: np.ndarray,
    *,
    ewald_alpha: float,
    sr_image_extent_bohr: float,
    sr_density_cells: Sequence[object],
    compute_exchange: bool,
    sr_output_cells: Optional[Sequence[object]] = None,
    sr_output_shell_masks: Optional[Sequence[object]] = None,
    sr_density_domain: Optional[object] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-domain ``(dJ, dK)`` corrections at Gamma.

    The padded SR builder maps a wide ``P(h)`` support onto the ordinary
    Fock-output cells. Its forward Bloch sums are therefore not, separately,
    the derivatives of ``1/2 P:J[P]`` and ``1/2 P:K[P]``. The native domain
    traversal evaluates the missing transpose action with exactly the same
    quartet screening decisions. This helper subtracts the forward matrices,
    leaving the correction to add to the occupied-space Pulay operator.

    Pair-resolved domains pass both projector families explicitly: output
    masks restrict the first density occurrence in the energy contraction,
    while density masks restrict the transpose contribution.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    internal_cells = list(
        _sr_internal_cells(basis, system, lattice_opts, sr_image_extent_bohr)
    )
    output_cells = (
        list(sr_output_cells)
        if sr_output_cells is not None
        else (internal_cells if lattice_opts.pair_complete_1e
              else list(compute_overlap_lattice(basis, system, lattice_opts).cells))
    )
    output_masks = (
        list(sr_output_shell_masks)
        if sr_output_shell_masks is not None
        else []
    )
    internal_index = {
        tuple(int(x) for x in cell.index): i
        for i, cell in enumerate(internal_cells)
    }
    try:
        output_indices = [
            internal_index[tuple(int(x) for x in cell.index)]
            for cell in output_cells
        ]
    except KeyError as exc:
        raise ValueError(
            "the padded SR internal domain does not contain every radial "
            "Fock-output cell"
        ) from exc

    density_block = np.asarray(density_block, dtype=np.float64)
    density = make_lattice_matrix_set(
        int(basis.nbasis),
        list(sr_density_cells),
        [density_block.copy() for _ in sr_density_cells],
    )
    density_masks = []
    if sr_density_domain is not None:
        from .pair_resolved_truncation import (
            atom_pair_shell_masks,
            mask_lattice_blocks_to_domain,
        )

        mask_lattice_blocks_to_domain(basis, sr_density_domain, density)
        density_masks = atom_pair_shell_masks(
            basis, sr_density_domain.pairs_by_cell
        )
    jk = build_jk_2e_real_space_domains_gamma_derivative(
        basis,
        system,
        lattice_opts,
        density,
        internal_cells,
        output_indices,
        float(ewald_alpha),
        bool(compute_exchange),
        output_masks,
        density_masks,
        density_block,
    )

    def _forward(blocks) -> np.ndarray:
        matrix = sum(
            (np.asarray(item, dtype=np.float64) for item in blocks),
            np.zeros_like(density_block),
        )
        return 0.5 * (matrix + matrix.T)

    d_j = np.asarray(jk.J_gamma_energy_derivative, dtype=np.float64) - _forward(
        jk.J.blocks
    )
    if compute_exchange:
        d_k = np.asarray(
            jk.K_gamma_energy_derivative, dtype=np.float64
        ) - _forward(jk.K.blocks)
    else:
        d_k = np.zeros_like(d_j)
    return d_j, d_k


def _m5_gamma_open_w_adjoint(
    W_gamma: np.ndarray,
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    total_block: np.ndarray,
    spin_blocks: Sequence[np.ndarray],
    spin_coeffs: Sequence[np.ndarray],
    spin_nocc: Sequence[int],
    *,
    alpha_hf: float,
    ewald_alpha: float,
    sr_image_extent_bohr: float,
    sr_density_cells: Sequence[object],
    sr_output_cells: Optional[Sequence[object]] = None,
    sr_output_shell_masks: Optional[Sequence[object]] = None,
    sr_density_domain: Optional[object] = None,
) -> np.ndarray:
    """Add the padded radial or pair-domain J/K transpose action to ``W``."""
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    d_j, _ = _m5_gamma_jk_adjoint_correction(
        system,
        basis,
        lattice_opts,
        total_block,
        ewald_alpha=ewald_alpha,
        sr_image_extent_bohr=sr_image_extent_bohr,
        sr_density_cells=sr_density_cells,
        compute_exchange=False,
        sr_output_cells=sr_output_cells,
        sr_output_shell_masks=sr_output_shell_masks,
        sr_density_domain=sr_density_domain,
    )
    corrected = np.asarray(W_gamma, dtype=np.float64).copy()
    for block, coeff, n_occ in zip(spin_blocks, spin_coeffs, spin_nocc):
        if int(n_occ) <= 0:
            continue
        _, d_k = _m5_gamma_jk_adjoint_correction(
            system,
            basis,
            lattice_opts,
            block,
            ewald_alpha=ewald_alpha,
            sr_image_extent_bohr=sr_image_extent_bohr,
            sr_density_cells=sr_density_cells,
            compute_exchange=abs(float(alpha_hf)) > 1.0e-14,
            sr_output_cells=sr_output_cells,
            sr_output_shell_masks=sr_output_shell_masks,
            sr_density_domain=sr_density_domain,
        )
        delta = d_j - float(alpha_hf) * d_k
        C_occ = np.asarray(coeff)[:, : int(n_occ)]
        corrected += np.real(
            C_occ @ (C_occ.conj().T @ delta @ C_occ) @ C_occ.conj().T
        )
    return corrected


def _m5_multi_k_jk_adjoint_delta(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    density: LatticeMatrixSet,
    kmesh,
    *,
    ewald_alpha: float,
    sr_image_extent_bohr: float,
    compute_exchange: bool,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Return padded finite-domain ``(delta J(k), delta K(k))`` operators.

    The finite M5 map sends the wide BvK real-space density support into the
    narrower radial Fock-output domain. Its energy derivative is therefore
    one half forward action plus one half transpose action, cell by cell.
    The native traversal returns that derivative on the density support.
    Subtract the forward Fock blocks and Bloch-fold the remainder into the
    usual ``Tr[F(k) dD(k)]`` matrix convention.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    internal_cells = list(
        _sr_internal_cells(basis, system, lattice_opts, sr_image_extent_bohr)
    )
    output_cells = (internal_cells if lattice_opts.pair_complete_1e else list(
        compute_overlap_lattice(basis, system, lattice_opts).cells
    ))
    internal_index = {
        tuple(int(x) for x in cell.index): i
        for i, cell in enumerate(internal_cells)
    }
    output_indices = [
        internal_index[tuple(int(x) for x in cell.index)]
        for cell in output_cells
    ]
    jk = build_jk_2e_real_space_domains_density_derivative(
        basis,
        system,
        lattice_opts,
        density,
        internal_cells,
        output_indices,
        float(ewald_alpha),
        bool(compute_exchange),
    )
    output_keys = {
        tuple(int(x) for x in cell.index)
        for cell in output_cells
    }
    j_forward = {
        tuple(int(x) for x in cell.index): np.asarray(
            jk.J.blocks[internal_index[tuple(int(x) for x in cell.index)]],
            dtype=np.float64,
        )
        for cell in output_cells
    }
    k_forward = {
        tuple(int(x) for x in cell.index): np.asarray(
            jk.K.blocks[internal_index[tuple(int(x) for x in cell.index)]],
            dtype=np.float64,
        )
        for cell in output_cells
    }
    delta_j_blocks = []
    delta_k_blocks = []
    zero = np.zeros((int(basis.nbasis), int(basis.nbasis)), dtype=np.float64)
    for cell, d_j, d_k in zip(
        density.cells,
        jk.J_density_energy_derivative.blocks,
        jk.K_density_energy_derivative.blocks,
    ):
        key = tuple(int(x) for x in cell.index)
        f_j = j_forward[key] if key in output_keys else zero
        f_k = k_forward[key] if key in output_keys else zero
        delta_j_blocks.append(np.asarray(d_j, dtype=np.float64) - f_j)
        delta_k_blocks.append(
            np.asarray(d_k, dtype=np.float64) - f_k
            if compute_exchange
            else zero.copy()
        )

    def _fold(delta_blocks: Sequence[np.ndarray]) -> List[np.ndarray]:
        out = []
        for k_cart in kmesh.kpoints:
            delta_f = sum(
                (
                    np.exp(
                        -1j
                        * float(np.dot(np.asarray(k_cart), cell.r_cart))
                    )
                    * block
                    for cell, block in zip(density.cells, delta_blocks)
                ),
                np.zeros(
                    (int(basis.nbasis), int(basis.nbasis)),
                    dtype=np.complex128,
                ),
            ).T
            out.append(0.5 * (delta_f + delta_f.conj().T))
        return out

    return _fold(delta_j_blocks), _fold(delta_k_blocks)


def _m5_multi_k_closed_w_adjoint(
    W_k_list: Sequence[np.ndarray],
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    density: LatticeMatrixSet,
    mo_coeffs: Sequence[np.ndarray],
    n_occ: int,
    kmesh,
    *,
    alpha_hf: float,
    ewald_alpha: float,
    sr_image_extent_bohr: float,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Add the padded radial J/K transpose action to closed-shell ``W(k)``."""
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    delta_j_k, delta_k_k = _m5_multi_k_jk_adjoint_delta(
        system,
        basis,
        lattice_opts,
        density,
        kmesh,
        ewald_alpha=ewald_alpha,
        sr_image_extent_bohr=sr_image_extent_bohr,
        compute_exchange=abs(float(alpha_hf)) > 1.0e-14,
    )
    delta_f_k = [
        d_j - 0.5 * float(alpha_hf) * d_k
        for d_j, d_k in zip(delta_j_k, delta_k_k)
    ]
    corrected = []
    for W_k, C_k, delta_f in zip(W_k_list, mo_coeffs, delta_f_k):
        # The lattice energy uses an elementwise D(g):F(g) contraction.
        # With D(g) = sum_k w_k Re[e^{-ik.g} D(k)], its matrix derivative
        # is [sum_g e^{-ik.g} dE/dD(g)]^T (the transpose converts the
        # elementwise derivative to the usual Tr[F(k) dD(k)] convention).
        C_occ = np.asarray(C_k, dtype=np.complex128)[:, : int(n_occ)]
        delta_occ = C_occ.conj().T @ delta_f @ C_occ
        corrected.append(
            np.asarray(W_k, dtype=np.complex128)
            + 2.0 * C_occ @ delta_occ @ C_occ.conj().T
        )
    return corrected, delta_f_k


def _m5_multi_k_open_w_adjoint(
    W_k_list: Sequence[np.ndarray],
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    total_density: LatticeMatrixSet,
    spin_densities: Sequence[LatticeMatrixSet],
    spin_coeffs: Sequence[Sequence[np.ndarray]],
    spin_nocc: Sequence[int],
    kmesh,
    *,
    alpha_hf: float,
    ewald_alpha: float,
    sr_image_extent_bohr: float,
) -> tuple[List[np.ndarray], tuple[List[np.ndarray], List[np.ndarray]]]:
    """Add the padded radial J/K transpose action to open-shell ``W(k)``."""
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    delta_j_k, _ = _m5_multi_k_jk_adjoint_delta(
        system,
        basis,
        lattice_opts,
        total_density,
        kmesh,
        ewald_alpha=ewald_alpha,
        sr_image_extent_bohr=sr_image_extent_bohr,
        compute_exchange=False,
    )
    spin_delta_k = []
    for density in spin_densities:
        _, delta_k = _m5_multi_k_jk_adjoint_delta(
            system,
            basis,
            lattice_opts,
            density,
            kmesh,
            ewald_alpha=ewald_alpha,
            sr_image_extent_bohr=sr_image_extent_bohr,
            compute_exchange=abs(float(alpha_hf)) > 1.0e-14,
        )
        spin_delta_k.append(
            [
                d_j - float(alpha_hf) * d_k
                for d_j, d_k in zip(delta_j_k, delta_k)
            ]
        )

    corrected = []
    for k_idx, W_k in enumerate(W_k_list):
        W_corr = np.asarray(W_k, dtype=np.complex128).copy()
        for coeffs, n_occ, delta_spin in zip(
            spin_coeffs, spin_nocc, spin_delta_k
        ):
            C_occ = np.asarray(
                coeffs[k_idx], dtype=np.complex128
            )[:, : int(n_occ)]
            delta_f = delta_spin[k_idx]
            delta_occ = C_occ.conj().T @ delta_f @ C_occ
            W_corr += C_occ @ delta_occ @ C_occ.conj().T
        corrected.append(W_corr)
    return corrected, (spin_delta_k[0], spin_delta_k[1])


def _sum_lattice_densities_on_wide_support(
    density_alpha: LatticeMatrixSet,
    density_beta: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Return ``D_alpha + D_beta`` without narrowing the density cell list.

    ``pbc_bipole_common._combine_density_sets`` deliberately projects onto
    the ordinary overlap/Fock-output domain for energy contractions.  The
    padded M5 short-range traversal instead consumes the wider BvK density
    support.  Preserve that support here and match the beta blocks by cell
    key so the result is independent of cell-list tie ordering.
    """
    beta_by_cell = {
        tuple(int(x) for x in cell.index): np.asarray(block, dtype=np.float64)
        for cell, block in zip(density_beta.cells, density_beta.blocks)
    }
    blocks = []
    for cell, alpha_block in zip(
        density_alpha.cells, density_alpha.blocks
    ):
        key = tuple(int(x) for x in cell.index)
        if key not in beta_by_cell:
            raise ValueError(
                "padded open-shell spin densities have different cell "
                f"support; beta is missing {key}"
            )
        blocks.append(
            np.asarray(alpha_block, dtype=np.float64) + beta_by_cell[key]
        )
    if len(beta_by_cell) != len(blocks):
        raise ValueError(
            "padded open-shell spin densities have different cell support"
        )
    return make_lattice_matrix_set(
        int(density_alpha.nbf),
        list(density_alpha.cells),
        blocks,
    )


def _build_energy_weighted_density_closed(
    mo_coeffs: List[np.ndarray],
    mo_energies: List[np.ndarray],
    n_occ: int,
) -> List[np.ndarray]:
    """Build per-k *complex* W(k) = 2.S_i e_i.C_i.C_i+(k) for
    closed-shell.

    The imaginary part is retained -- the multi-k inverse-Bloch fold
    ``W(g) = S_k w_k Re[e^{-ik.g} W(k)]`` needs it (Im W(k) vanishes
    only at Γ, where the broadcast is exact).
    """
    W_k_list: List[np.ndarray] = []
    for C_k, eps_k in zip(mo_coeffs, mo_energies):
        C = np.asarray(C_k, dtype=np.complex128)
        eps = np.asarray(np.real(eps_k))
        if n_occ > C.shape[1]:
            raise ValueError(f"n_occ={n_occ} exceeds n_mo={C.shape[1]}")
        eps_occ = eps[:n_occ]
        C_occ = C[:, :n_occ]
        W_k = 2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T
        W_k_list.append(W_k)
    return W_k_list


def _build_energy_weighted_density_open(
    mo_coeffs_alpha: List[np.ndarray],
    mo_energies_alpha: List[np.ndarray],
    mo_coeffs_beta: List[np.ndarray],
    mo_energies_beta: List[np.ndarray],
    n_alpha: int,
    n_beta: int,
) -> List[np.ndarray]:
    """Build per-k *complex* total W(k) = W_a(k) + W_b(k) for
    open-shell. The imaginary part is retained for the multi-k
    inverse-Bloch fold (see :func:`_build_energy_weighted_density_closed`).
    """
    W_k_list: List[np.ndarray] = []
    for C_a, eps_a, C_b, eps_b in zip(
        mo_coeffs_alpha,
        mo_energies_alpha,
        mo_coeffs_beta,
        mo_energies_beta,
    ):
        # Multi-k MO coefficients are complex; the accumulator must
        # carry their dtype (np.zeros_like(..., dtype=float) crashes
        # the casting rule when we add a complex product). The complex
        # W(k) is kept; Re[] is applied by the inverse-Bloch fold (or
        # the Γ broadcast) downstream.
        C_a_arr = np.asarray(C_a, dtype=np.complex128)
        C_b_arr = np.asarray(C_b, dtype=np.complex128)
        eps_a_arr = np.asarray(np.real(eps_a))
        eps_b_arr = np.asarray(np.real(eps_b))
        W_k = np.zeros_like(C_a_arr)
        if n_alpha > 0:
            eps_a_occ = eps_a_arr[:n_alpha]
            Ca_occ = C_a_arr[:, :n_alpha]
            W_k += (Ca_occ * eps_a_occ[None, :]) @ Ca_occ.conj().T
        if n_beta > 0:
            eps_b_occ = eps_b_arr[:n_beta]
            Cb_occ = C_b_arr[:, :n_beta]
            W_k += (Cb_occ * eps_b_occ[None, :]) @ Cb_occ.conj().T
        W_k_list.append(W_k)
    return W_k_list


def _build_energy_weighted_density_closed_frac(
    mo_coeffs: Sequence[np.ndarray],
    mo_energies: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Fractional-occupation closed-shell W(k) = S_i f_i e_i C_i C_i+(k).

    The free-energy (Mermin) analytic gradient for a finite-temperature /
    smeared KS run differs from the integer-Aufbau form only by the
    occupation weights: the density is D(k) = S_i f_i C_iC_i+ (already built
    by the SCF) and the energy-weighted density carries the SAME f_i,
    ``W(k) = S_i f_i e_i C_iC_i+`` (closed-shell f_i in [0, 2]). Because A =
    E - TS is stationary w.r.t. both the orbitals and the occupations at
    convergence, no occupation-response term survives -- dA/dR is the standard
    Hellmann-Feynman + Pulay expression with these fractional D and W. The
    per-orbital jellium shift carried by e_i still supplies the cancelling
    overlap term (see :func:`_compute_bipole_gradient_corrected_gamma`).

    Reduces exactly to :func:`_build_energy_weighted_density_closed` for the
    integer-Aufbau occupation vector f = [2, ..., 2, 0, ..., 0].
    """
    W_k_list: List[np.ndarray] = []
    for C_k, eps_k, occ_k in zip(mo_coeffs, mo_energies, occupations):
        C = np.asarray(C_k, dtype=np.complex128)
        eps = np.asarray(np.real(eps_k), dtype=np.float64)
        f = np.asarray(np.real(occ_k), dtype=np.float64)
        n = min(C.shape[1], eps.size, f.size)
        Cn = C[:, :n]
        W_k_list.append((Cn * (f[:n] * eps[:n])[None, :]) @ Cn.conj().T)
    return W_k_list


def _build_energy_weighted_density_open_frac(
    mo_coeffs_alpha: Sequence[np.ndarray],
    mo_energies_alpha: Sequence[np.ndarray],
    occupations_alpha: Sequence[np.ndarray],
    mo_coeffs_beta: Sequence[np.ndarray],
    mo_energies_beta: Sequence[np.ndarray],
    occupations_beta: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Fractional-occupation total W(k) = S_i f_i^a e_i^a C_iC_i+ + (b).

    Open-shell companion of :func:`_build_energy_weighted_density_closed_frac`
    (per-spin f_i^s in [0, 1]). Reduces to
    :func:`_build_energy_weighted_density_open` for integer Aufbau.
    """
    W_k_list: List[np.ndarray] = []
    for Ca_k, ea_k, fa_k, Cb_k, eb_k, fb_k in zip(
        mo_coeffs_alpha,
        mo_energies_alpha,
        occupations_alpha,
        mo_coeffs_beta,
        mo_energies_beta,
        occupations_beta,
    ):
        Ca = np.asarray(Ca_k, dtype=np.complex128)
        Cb = np.asarray(Cb_k, dtype=np.complex128)
        ea = np.asarray(np.real(ea_k), dtype=np.float64)
        eb = np.asarray(np.real(eb_k), dtype=np.float64)
        fa = np.asarray(np.real(fa_k), dtype=np.float64)
        fb = np.asarray(np.real(fb_k), dtype=np.float64)
        W_k = np.zeros((Ca.shape[0], Ca.shape[0]), dtype=np.complex128)
        na = min(Ca.shape[1], ea.size, fa.size)
        W_k += (Ca[:, :na] * (fa[:na] * ea[:na])[None, :]) @ Ca[:, :na].conj().T
        nb = min(Cb.shape[1], eb.size, fb.size)
        W_k += (Cb[:, :nb] * (fb[:nb] * eb[:nb])[None, :]) @ Cb[:, :nb].conj().T
        W_k_list.append(W_k)
    return W_k_list


def _gamma_lattice_set(template: LatticeMatrixSet, M: np.ndarray) -> LatticeMatrixSet:
    """Fill a LatticeMatrixSet with (the real part of) M in every cell
    block. Correct for the Pulay term only at Γ, where ``W(g) = W(Γ)``
    for every ``g``."""
    M_arr = np.real(np.asarray(M)).astype(np.float64)
    for c in range(len(template.cells)):
        template.set_block(c, M_arr)
    return template


def _bloch_fold_w_matrices(
    W_k_list: Sequence[np.ndarray],
    kmesh,
    template: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Inverse-Bloch-fold per-k energy-weighted densities into the
    real-space cell list of ``template``:

        W(g) = S_k w_k Re[ exp(-i k.g) . W(k) ]

    Same convention as ``real_space_density_from_kpoints`` and
    :func:`vibeqc.periodic_gradient_multi_k._bloch_fold_w_per_k` -- so
    feeding the result to ``overlap_lattice_gradient_contribution``
    contracts ``W(g)`` against ``dS(g)/dR`` exactly the way the SCF
    contracts the real-space density against the one-electron operators.
    Reduces to the Γ broadcast when there is a single k-point.

    ``template`` is consumed (its blocks are overwritten).
    """
    weights = list(kmesh.weights)
    kpts = list(kmesh.kpoints)
    if len(weights) != len(W_k_list):
        raise ValueError(
            f"_bloch_fold_w_matrices: kmesh has {len(weights)} k-points "
            f"but got {len(W_k_list)} W(k) matrices"
        )
    n_cells = len(template.cells)
    nbf = int(template.nbf)
    blocks = [np.zeros((nbf, nbf), dtype=np.float64) for _ in range(n_cells)]
    for ik, W_k in enumerate(W_k_list):
        Wk = np.asarray(W_k, dtype=np.complex128)
        Wk_re = Wk.real
        Wk_im = Wk.imag
        w_k = float(weights[ik])
        k_cart = np.asarray(kpts[ik], dtype=np.float64)
        for c, cell in enumerate(template.cells):
            r = np.asarray(cell.r_cart, dtype=np.float64)
            phase = float(np.dot(k_cart, r))
            # exp(-i k.g) . W(k) -> Re part = cos.Re(W) + sin.Im(W)
            blocks[c] += w_k * (np.cos(phase) * Wk_re + np.sin(phase) * Wk_im)
    for c in range(n_cells):
        template.set_block(c, blocks[c])
    return template


_RESEARCH_PREVIEW_MSG = (
    "compute_bipole_gradient_{kind}: the BIPOLE analytic gradient is a "
    "maintained preview. Corrected-gauge (Ewald-exchange-split) RHF/UHF "
    "multi-k is FD-validated (pass kmesh=); padded radial multi-k RHF/UHF "
    "include the full finite-domain response (~3e-5/~7e-8 vs FD). Legacy-gauge "
    "RHF/UHF (use_exchange_ewald_split=False) is maintained on symmetric "
    "controls; asymmetric legacy-HF SCF is unsupported. For production forces use "
    "compute_bipole_gradient_fd (bipole_optimize and periodic NEB default to it)."
)


_RKS_UKS_PREVIEW_MSG = (
    "compute_bipole_gradient_{kind}: the BIPOLE KS analytic gradient is a "
    "maintained preview. Gamma-local RKS/UKS include LDA/GGA/meta-GGA/PBE/B3LYP XC "
    "Pulay, moving-grid correction, and KS Bloch-CPHF on maintained asymmetric "
    "cells. Fractional-occupation (finite-T smeared) analytic gradients are "
    "landed in the corrected gauge. Multi-k KS runs with diagonal-Z + corrected "
    "W + J^LR + XC Pulay (warns); multi-k KS CPHF "
    "remains gated. For production forces use compute_bipole_gradient_fd."
)


def _warn_research_preview(kind: str) -> None:
    msg = (
        _RKS_UKS_PREVIEW_MSG if kind in ("rks", "uks") else _RESEARCH_PREVIEW_MSG
    ).format(kind=kind)
    warnings.warn(
        msg,
        UserWarning,
        stacklevel=3,
    )


def _occupation_arrays(result, attr: str) -> List[np.ndarray]:
    """Per-k occupation arrays for ``attr`` (``occupations`` /
    ``occupations_alpha`` / ``occupations_beta``), normalised to a list of
    1-D arrays. Empty list when the attribute is absent."""
    raw = getattr(result, attr, None)
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return [np.asarray(raw, dtype=np.float64)]
    try:
        return [np.asarray(o, dtype=np.float64) for o in raw]
    except TypeError:
        return [np.asarray(raw, dtype=np.float64)]


def _is_fractional_ks_occupation(result, kind: str, *, tol: float = 1e-8) -> bool:
    """True if the KS result carries finite-temperature smearing or any
    genuinely fractional occupation, i.e. the analytic gradient must use the
    Mermin free-energy form (fractional density + fractional energy-weighted
    density). False on the integer-Aufbau surface.

    ``result.energy`` for a smeared run is the Helmholtz free energy
    ``A = E - T.S`` (``pbc_bipole_rks``/``_uks``), so the production FD path
    differentiates ``A``; the analytic gradient matches it with fractional
    ``D`` and ``W = S_i f_i e_i C_iC_i+`` and NO occupation-response term --
    ``A`` is stationary w.r.t. the occupations at convergence.
    """
    if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
        return True
    specs = (
        (("occupations", 2.0),)
        if kind == "rks"
        else (("occupations_alpha", 1.0), ("occupations_beta", 1.0))
    )
    for attr, max_occ in specs:
        for arr in _occupation_arrays(result, attr):
            if arr.size == 0:
                continue
            dist = np.minimum(np.abs(arr), np.abs(arr - max_occ))
            if bool(np.any(dist > tol)):
                return True
    return False


def _reject_fractional_ks_analytic_gradient(result, kind: str) -> None:
    """Reject KS analytic gradients outside the integer-occupation surface.

    Used by the gauges that still require integer occupations (legacy-gauge
    Γ-local + the legacy/diagonal-Z multi-k path). The corrected
    (Ewald-exchange-split) gauge handles fractional occupations directly via
    the Mermin free-energy form and does NOT call this -- see
    :func:`_is_fractional_ks_occupation`.
    """
    if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            f"compute_bipole_gradient_{kind}: finite-temperature KS analytic "
            "BIPOLE gradients are only implemented in the corrected "
            "(use_ewald_j_split=True) gauge. Use compute_bipole_gradient_fd "
            "for production forces in this gauge."
        )
    if _is_fractional_ks_occupation(result, kind):
        raise NotImplementedError(
            f"compute_bipole_gradient_{kind}: fractional-occupation KS analytic "
            "BIPOLE gradients are only implemented in the corrected "
            "(use_ewald_j_split=True) gauge. Use compute_bipole_gradient_fd "
            "for production forces in this gauge."
        )


def _count_k_points_from_attr(result, attr: str) -> int:
    raw = getattr(result, attr, None)
    if raw is None or isinstance(raw, np.ndarray):
        return 1
    try:
        return len(raw)
    except TypeError:
        return 1


def _warn_multi_k_ks_analytic_gradient(result, kind: str, kmesh=None) -> bool:
    """Return True if this is a multi-k KS run (needs xc pulay + corrected W).

    Multi-k KS analytic gradients now use the per-k corrected W via
    _corrected_w_multi_k_closed/open plus the multi-k J^LR convention.
    KS Bloch-CPHF remains gated for multi-k (requires complex coupled
    k-space response).
    """
    attrs = ("mo_coeffs",) if kind == "rks" else ("mo_coeffs_alpha", "mo_coeffs_beta")
    n_k = max(_count_k_points_from_attr(result, attr) for attr in attrs)
    if kmesh is not None:
        try:
            ir_mapping = np.asarray(
                getattr(kmesh, "ir_mapping", []),
                dtype=int,
            ).reshape(-1)
            if ir_mapping.size > 0:
                n_k = max(n_k, int(ir_mapping.size))
            else:
                n_k = max(n_k, len(list(kmesh.kpoints)))
        except Exception:
            pass
    return n_k > 1


def _reject_multi_k_ks_analytic_gradient(result, kind: str, kmesh=None) -> None:
    """Backward-compat: now warns instead of raising; use _warn_ variant directly."""
    if _warn_multi_k_ks_analytic_gradient(result, kind, kmesh):
        warnings.warn(
            f"compute_bipole_gradient_{kind}: multi-k KS analytic gradient "
            "is a maintained preview (corrected W + multi-k J^LR, "
            "KS CPHF gated). Use compute_bipole_gradient_fd for production.",
            UserWarning,
            stacklevel=2,
        )


def _matching_ewald_options(
    system: PeriodicSystem,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: Optional[float],
    *,
    reciprocal_cutoff_bohr_inv: Optional[float] = None,
    tolerance: float = 1.0e-8,
) -> Optional[EwaldOptions]:
    """Rebuild the exact ``EwaldOptions`` the BIPOLE energy used for E_nn.

    Returns ``None`` when the run was not in the 3D Ewald gauge (no
    ``ewald_alpha`` recorded, or a 1D / 2D system), so the caller falls
    back to the direct-sum gradient.

    The energy path (``pbc_bipole._crystal_ewald_options`` +
    ``run_pbc_bipole_*``) fixes a, the real-space cutoff and the reciprocal
    cutoff (``bipole_ewald_reciprocal_cutoff(V_cell, alpha)``, CRYSTAL's
    volume-only envelope scaled with the alpha in use, #674). With a and
    both cutoffs pinned the Ewald sum is fully determined, so the gradient
    differentiates exactly the energy that was evaluated.

    This DELEGATES to ``_crystal_ewald_options`` rather than rebuilding the
    fields by hand (#478). The hand-rolled copy set ``real_cutoff_bohr``
    straight from ``lattice_opts.nuclear_cutoff_bohr``, and when the energy
    path started flooring that at the alpha-consistent radius the copy kept
    the old value -- desyncing the analytic gradient from the energy it is
    supposed to differentiate. Measured on the legacy-gauge multi-k triplet
    H2/STO-3G [2,1,1] fixture: analytic-vs-FD ``max|dg|`` went 1.3e-3 with
    the duplicate against a 1e-3 gate, and back to agreement once this
    delegated. A rule with two implementations is a rule that will drift;
    keep this a delegation.

    ``tolerance`` must match the run's ``ewald_precision`` because it now
    sets that real-space floor. Public gradient entry points read it from
    the SCF result; older results without this field retain the 1e-8 default.
    """
    if ewald_alpha is None or ewald_alpha <= 0.0 or system.dim != 3:
        return None
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .pbc_bipole_common import _crystal_ewald_options

    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    recip = (
        float(reciprocal_cutoff_bohr_inv)
        if reciprocal_cutoff_bohr_inv is not None
        and float(reciprocal_cutoff_bohr_inv) > 0.0
        else float(bipole_ewald_reciprocal_cutoff(V_cell, float(ewald_alpha)))
    )
    return _crystal_ewald_options(
        lattice_opts,
        alpha_bohr_inv=float(ewald_alpha),
        tolerance=float(tolerance),
        recip_cutoff_bohr_inv=recip,
    )


def _ao_to_atom_map(system: PeriodicSystem, basis: BasisSet) -> np.ndarray:
    """Per-AO unit-cell atom index, by matching each shell origin to the
    nearest atom. Used to scatter AO-pair-FT centre derivatives onto the
    atom each AO sits on."""
    atom_pos = np.array(
        [[float(x) for x in a.xyz] for a in system.unit_cell], dtype=float
    )
    ao2atom: list[int] = []
    for sh in basis.shells():
        origin = np.asarray(sh.origin, dtype=float)
        atom = int(np.argmin(((atom_pos - origin) ** 2).sum(axis=1)))
        ao2atom.extend([atom] * (2 * int(sh.l) + 1))
    return np.asarray(ao2atom, dtype=int)


def _pair_ft_gradient_at_cells(basis, vectors, shifts, *, lattice_opts=None):
    """Differentiate the same finite AO-pair support as the reciprocal value."""
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from .lattice_screening import ao_pair_support_mask

    bra, ket = ao_pair_fourier_transform_grad_at_cells(basis, vectors, shifts)
    if lattice_opts is not None and lattice_opts.pair_complete_1e:
        for cell, shift in enumerate(shifts):
            mask = ao_pair_support_mask(basis, shift, lattice_opts.cutoff_bohr)
            bra[cell] *= mask[:, :, None, None]
            ket[cell] *= mask[:, :, None, None]
    return bra, ket


def _v_ne_ewald_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    *,
    precision: float = 1e-8,
    reciprocal_cutoff_bohr_inv: Optional[float] = None,
) -> np.ndarray:
    """Hellmann-Feynman gradient of the BIPOLE Ewald V_ne energy.

    Returns ``S_g S_muν D(g)_muν dV_ne(g)_muν/dR_A`` as ``(n_atoms, 3)``,
    the gauge-correct replacement for the truncated full-Coulomb
    ``nuclear_lattice_gradient_contribution``. The V_ne matrix is the one
    ``pbc_bipole._compute_nuclear_lattice_ewald_reciprocal_ft`` builds:

        V_ne(g) = V_short(g) + Re[v_lr(g)] + background . S(g),
        v_lr(g)_muν = -S_K kernel(K) . r_nuc(K) . FT_muν(K; g)*,
        r_nuc(K)  = S_A Z_A e^{-iK.R_A},
        background = pi Q_nuc / (a^2 V).

    Three additive pieces, each differentiated in the energy's gauge:

    1. **V_short** -- erfc-screened nuclear attraction; libint integral
       gradient (`nuclear_erfc_lattice_gradient_contribution`).
    2. **V_long (reciprocal)** -- analytic. Splits into the r_nuc
       structure-factor derivative ``dr_nuc/dR_C = -iK Z_C e^{-iK.R_C}``
       (trivial, reuses the cached FT) and the AO-pair-FT centre
       derivative (the `_aopair_ft` gradient linchpin), scattered onto the
       bra/ket atoms.
    3. **background.S** -- the scalar ``background`` times the overlap
       (Pulay-type) derivative ``dS/dR``.
    """
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import _build_j_long_range_cache

    n_atoms = len(system.unit_cell)
    alpha = float(ewald_alpha)
    a_lat = np.asarray(system.lattice, dtype=float)
    V_cell = float(abs(np.linalg.det(a_lat)))
    K_max = (
        float(reciprocal_cutoff_bohr_inv)
        if reciprocal_cutoff_bohr_inv is not None
        and float(reciprocal_cutoff_bohr_inv) > 0.0
        else float(bipole_ewald_reciprocal_cutoff(V_cell, alpha))
    )

    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in D_real.cells], dtype=float
    )
    D_blocks = np.array(
        [np.asarray(b, dtype=float) for b in D_real.blocks], dtype=float
    )  # (n_g, nbf, nbf), real

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    # --- Piece 1: V_short (erfc-screened) ---
    # #478/#704: match the BIPOLE energy's nuclear image ball, including
    # alpha, Gaussian smearing and displaced product centres. Sharing
    # ewald_erfc_lattice_options with the energy's compute_nuclear_erfc_lattice
    # is what keeps this an exact derivative partner.
    from .pbc_bipole_common import ewald_erfc_lattice_options

    grad += np.asarray(
        nuclear_erfc_lattice_gradient_contribution(
            basis,
            system,
            D_real,
            ewald_erfc_lattice_options(
                lattice_opts, alpha, precision, basis=basis, system=system,
            ),
            alpha,
        )
    )

    # --- Piece 3: background . S ---
    # The Hellmann-Feynman term is +background.S_g D(g).dS(g)/dR, but
    # ``overlap_lattice_gradient_contribution`` returns -S M.dS/dR (the
    # Pulay sign convention used by the W-term above), so negate.
    Q_nuc = float(sum(float(at.Z) for at in system.unit_cell))
    background = np.pi * Q_nuc / (alpha * alpha * V_cell)
    grad += -background * np.asarray(
        overlap_lattice_gradient_contribution(basis, system, D_real, lattice_opts)
    )

    # --- Piece 2: V_long (reciprocal) ---
    if lattice_opts.pair_complete_1e:
        from .lattice_screening import ao_pair_support_mask

        for c, shift in enumerate(cells_r_cart):
            D_blocks[c][~ao_pair_support_mask(
                basis, shift, lattice_opts.cutoff_bohr,
            )] = 0.0
    cache = _build_j_long_range_cache(
        basis, system, cells_r_cart, alpha, precision, K_max=K_max, lattice_opts=lattice_opts)
    K_vec = cache.K_vectors  # (n_K, 3)
    kernel = cache.kernel  # (n_K,)
    ft = cache.ft_per_cell  # (n_g, nbf, nbf, n_K), corrected
    atom_pos = np.array(
        [[float(x) for x in at.xyz] for at in system.unit_cell], dtype=float
    )
    atom_z = np.array([float(at.Z) for at in system.unit_cell], dtype=float)
    phases = np.exp(-1j * (atom_pos @ K_vec.T))  # (n_atoms, n_K)
    rho_nuc = atom_z @ phases  # (n_K,)
    weighted = kernel * rho_nuc  # (n_K,)

    # D-contracted FT (matches the energy's -Re S_K weighted DFT* form).
    DFT = np.einsum("gmn,gmnk->k", D_blocks, ft)  # (n_K,) complex

    # (2a) structure-factor derivative: dr_nuc/dR_C = -iK Z_C e^{-iK.R_C}.
    #   dE_long^(2a)/dR_C = -Re S_K kernel(K) (-iK Z_C e^{-iK.R_C}) DFT(K)*
    for C in range(n_atoms):
        phase_C = np.exp(-1j * (K_vec @ atom_pos[C]))  # (n_K,)
        # vector over axes mu: -Re S_K kernel . (-i K_mu Z_C phase_C) . DFT*
        contrib = -np.real(
            (kernel * atom_z[C] * phase_C * np.conj(DFT))[:, None] * (-1j * K_vec)
        ).sum(axis=0)  # (3,)
        grad[C] += contrib

    # (2b) AO-pair-FT centre derivative. grad_bra/grad_ket: (n_g,nbf,nbf,3,n_K).
    grad_bra, grad_ket = _pair_ft_gradient_at_cells(
        basis, K_vec, cells_r_cart, lattice_opts=lattice_opts)
    corr = _libint_ylm_correction_per_ao(basis)  # (nbf,)
    cc = corr[:, None] * corr[None, :]  # (nbf,nbf)
    grad_bra = grad_bra * cc[None, :, :, None, None]
    grad_ket = grad_ket * cc[None, :, :, None, None]

    #   dE_long^(2b)/dR_C = -Re S_K weighted(K) (dDFT(K)/dR_C)*
    # with dDFT/dR_C the D-contraction of the FT centre derivative,
    # scattered onto bra atom (AO mu) and ket atom (AO ν).
    bra_per_m = np.einsum("gmn,gmnxk->mxk", D_blocks, grad_bra)  # (nbf,3,n_K)
    ket_per_n = np.einsum("gmn,gmnxk->nxk", D_blocks, grad_ket)  # (nbf,3,n_K)
    ao2atom = _ao_to_atom_map(system, basis)
    dDFT = np.zeros((n_atoms, 3, K_vec.shape[0]), dtype=np.complex128)
    np.add.at(dDFT, ao2atom, bra_per_m)
    np.add.at(dDFT, ao2atom, ket_per_n)
    for C in range(n_atoms):
        grad[C] += -np.real((weighted[None, :] * np.conj(dDFT[C]))).sum(axis=1)  # (3,)

    return grad


def _j_long_range_ewald_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    ewald_alpha: float,
    *,
    precision: float = 1e-8,
    gamma_local: bool = False,
    lattice_opts: Optional[LatticeSumOptions] = None,
) -> np.ndarray:
    """Hellmann-Feynman gradient of the BIPOLE long-range Coulomb energy.

    Returns ``(n_atoms, 3)``. The reciprocal (long-range) Hartree energy
    uses a **mixed** Bloch/local convention (the SCF builds the J^LR
    operator with a Bloch-k-density ``r̂_bloch`` but contracts it against
    the local projected density):

        E_J^LR = 1/2 S_K kernel(K) . Re[ r̂_bloch(K) . r̂_local(K)* ],
        r̂_x(K) = S_g S_muν D_x(g)_muν FT_muν(K; R_g),

    verified to 1e-16 against ``e_j_long_range``. For a Γ-only run
    (``gamma_local=True``) ``D_local`` is the projected density (``D_real``,
    ``D(g!=0)=0``) and ``D_bloch(g)=P(Γ)`` is its home block broadcast into
    every cell (the inverse-Bloch density at Γ). For multi-k there is no
    projection, ``r̂_bloch=r̂_local`` and the form reduces to ``1/2|r̂|^2``.

    Holding the density fixed,

        dE/dR_C = 1/2 S_K kernel(K) . Re[ dr̂_b/dR_C.r̂_l* + r̂_b.dr̂_l*/dR_C ],

    with dr̂_x/dR_C the AO-pair-FT centre derivative (the `_aopair_ft`
    linchpin) contracted with D_x and scattered onto the bra/ket atoms.
    """
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import _build_j_long_range_cache

    n_atoms = len(system.unit_cell)
    alpha = float(ewald_alpha)
    a_lat = np.asarray(system.lattice, dtype=float)
    V_cell = float(abs(np.linalg.det(a_lat)))
    K_max = float(bipole_ewald_reciprocal_cutoff(V_cell, alpha))

    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in D_real.cells], dtype=float
    )
    D_local = np.array([np.asarray(b, dtype=float) for b in D_real.blocks], dtype=float)
    if gamma_local:
        home = _home_cell_index(D_real.cells)
        D_bloch = np.broadcast_to(D_local[home], D_local.shape).copy()
    else:
        D_bloch = D_local

    cache = _build_j_long_range_cache(
        basis, system, cells_r_cart, alpha, precision, K_max=K_max, lattice_opts=lattice_opts)
    K_vec = cache.K_vectors
    kernel = cache.kernel
    ft = cache.ft_per_cell  # (n_g,nbf,nbf,n_K), corrected
    rho_l = np.einsum("gmn,gmnk->k", D_local, ft)  # (n_K,) complex
    rho_b = np.einsum("gmn,gmnk->k", D_bloch, ft)

    grad_bra, grad_ket = _pair_ft_gradient_at_cells(
        basis, K_vec, cells_r_cart, lattice_opts=lattice_opts)
    corr = _libint_ylm_correction_per_ao(basis)
    cc = corr[:, None] * corr[None, :]
    grad_bra = grad_bra * cc[None, :, :, None, None]
    grad_ket = grad_ket * cc[None, :, :, None, None]
    ao2atom = _ao_to_atom_map(system, basis)

    def _drho(D_blk):
        bra = np.einsum("gmn,gmnxk->mxk", D_blk, grad_bra)  # (nbf,3,n_K)
        ket = np.einsum("gmn,gmnxk->nxk", D_blk, grad_ket)
        out = np.zeros((n_atoms, 3, K_vec.shape[0]), dtype=np.complex128)
        np.add.at(out, ao2atom, bra)
        np.add.at(out, ao2atom, ket)
        return out

    drho_l = _drho(D_local)
    drho_b = _drho(D_bloch)

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for C in range(n_atoms):
        term = drho_b[C] * np.conj(rho_l)[None, :] + rho_b[None, :] * np.conj(
            drho_l[C]
        )  # (3, n_K)
        grad[C] = 0.5 * np.real((kernel[None, :] * term).sum(axis=1))
    return grad


def _k_long_range_ewald_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    ewald_alpha: float,
    *,
    precision: float = 1e-8,
    lattice_opts: Optional[LatticeSumOptions] = None,
) -> np.ndarray:
    """Hellmann-Feynman gradient of the Γ reciprocal long-range EXCHANGE.

    The corrected (Ewald-exchange-split) gauge replaces the legacy
    full-Coulomb K with ``K = K_SR(erfc) + K_LR(erf) + Madelung.SDS``
    (``bipole_fock_ewald`` module docstring). The reciprocal long-range
    exchange operator (``compute_K_long_range_gamma``) is the exchange
    analogue of ``J^LR`` -- it sandwiches the density between two pair FTs
    instead of tracing against it::

        K^LR_muν = S_{K!=0} kernel(K) . A*_mul(K) D_ls A_νs(K),
        A_mul(K) = S_g FT_mul(K; R_g)   (Γ Bloch-summed shifted-ν pair FT).

    The exchange-energy contribution is ``E = -1/4 Tr[D K^LR]`` (the SCF's
    ``e_2e_k_correction`` at n_k=1; ΔF = -1/2K_corr => ΔE = -1/4 Tr[D K_corr]).
    Holding the density fixed and differentiating the two pair-FT factors,

        dE/dR_C = -1/2 Re S_K kernel(K) S_muν dA*_muν(K)/dR_C . G_muν(K),
        G(K) = D . A(K) . D,

    with ``dA/dR`` the AO-pair-FT centre derivative (the same `_aopair_ft`
    linchpin as ``_j_long_range_ewald_gradient``) scattered onto the
    bra/ket atoms. ``D`` is the Γ density (= the BvK home-cell block, the
    density ``compute_K_long_range_gamma`` is contracted against).

    Validated against the central-difference of ``-1/4 Tr[D K^LR(D)]`` at a
    fixed density to ~2.5e-12 Ha/bohr on MgO/STO-3G (2026-06-15).
    """
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import _build_j_long_range_cache

    alpha = float(ewald_alpha)
    a_lat = np.asarray(system.lattice, dtype=float)
    V_cell = float(abs(np.linalg.det(a_lat)))
    K_max = float(bipole_ewald_reciprocal_cutoff(V_cell, alpha))
    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in D_real.cells], dtype=float
    )
    home = _home_cell_index(D_real.cells)
    D_home = np.asarray(D_real.blocks[home], dtype=float)

    cache = _build_j_long_range_cache(
        basis, system, cells_r_cart, alpha, precision, K_max=K_max, lattice_opts=lattice_opts)
    kernel = cache.kernel
    # A(K) = S_g FT(K; R_g) -- the Γ Bloch sum (k=0 phase 1), corr baked in.
    A = cache.ft_per_cell.sum(axis=0)  # (nbf, nbf, n_K)
    G = np.einsum("ma,abk,bn->mnk", D_home, A, D_home, optimize=True)

    grad_bra, grad_ket = _pair_ft_gradient_at_cells(
        basis, cache.K_vectors, cells_r_cart, lattice_opts=lattice_opts)
    corr = _libint_ylm_correction_per_ao(basis)
    cc = corr[:, None] * corr[None, :]
    dA_bra = (grad_bra * cc[None, :, :, None, None]).sum(axis=0)  # (nbf,nbf,3,n_K)
    dA_ket = (grad_ket * cc[None, :, :, None, None]).sum(axis=0)
    ao2atom = _ao_to_atom_map(system, basis)

    n_atoms = len(system.unit_cell)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    # dA_muν wrt bra atom (mu) and ket atom (ν); contract with G and kernel.
    bcontr = np.einsum("k,mnxk,mnk->mx", kernel, dA_bra.conj(), G, optimize=True)
    kcontr = np.einsum("k,mnxk,mnk->nx", kernel, dA_ket.conj(), G, optimize=True)
    np.add.at(grad, ao2atom, -0.5 * np.real(bcontr))
    np.add.at(grad, ao2atom, -0.5 * np.real(kcontr))
    return grad


def _spheropole_dedp_lattice_blocks(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
) -> list[np.ndarray]:
    """``dE_sph/dP(g) = prefactor.K(g)`` for EXT EL-SPHEROPOLE.

    Replicates the v3 (emultipole2) energy kernel of
    ``compute_ext_el_spheropole``: ``K_muν(g) = 2.Tr<r^2> - 2(A_mu+B_ν).<r> +
    (|A_mu|^2+|B_ν|^2).<1>`` with the moments from
    ``compute_multipole_moments_lattice`` and prefactor ``pi.N_e/(6V)``.  The
    energy is linear in ``P(g)``, so these are the exact density derivatives.
    """
    from ._vibeqc_core import compute_multipole_moments_lattice

    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    n_elec = int(system.n_electrons())
    prefactor = math.pi * (n_elec / 6.0) / V
    M_lat = compute_multipole_moments_lattice(
        basis, system, lattice_opts, 2, (0.0, 0.0, 0.0)
    )
    centers = []
    for sh in basis.shells():
        for _ in range(2 * int(sh.l) + 1):
            centers.append(tuple(sh.origin))
    A = np.asarray(centers, dtype=float)
    A2 = np.einsum("mi,mi->m", A, A)
    blocks: list[np.ndarray] = []
    for cell, blk in zip(M_lat.cells, M_lat.blocks):
        M0 = np.asarray(blk[0], dtype=float)
        M1 = [np.asarray(blk[k], dtype=float) for k in (1, 2, 3)]
        TrM2 = (
            np.asarray(blk[4], dtype=float)
            + np.asarray(blk[7], dtype=float)
            + np.asarray(blk[9], dtype=float)
        )
        g = np.asarray(cell.r_cart, dtype=float)
        Bk = A + g
        Bk2 = np.einsum("ni,ni->n", Bk, Bk)
        AdotM1 = sum(A[:, i][:, None] * M1[i] for i in range(3))
        BdotM1 = sum(Bk[:, i][None, :] * M1[i] for i in range(3))
        K_g = 2.0 * TrM2 - 2.0 * (AdotM1 + BdotM1) + (A2[:, None] + Bk2[None, :]) * M0
        blocks.append(prefactor * K_g)
    return blocks


def _spheropole_dedp_home_block(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
) -> np.ndarray:
    """Home-cell ``dE_sph/dP(0)`` block for the Γ-local corrected W."""
    S = compute_overlap_lattice(basis, system, lattice_opts)
    home = _home_cell_index(S.cells)
    return _spheropole_dedp_lattice_blocks(system, basis, lattice_opts)[home]


def _spheropole_ewald_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    lattice_opts: LatticeSumOptions,
) -> np.ndarray:
    """Hellmann-Feynman gradient of the EXT EL-SPHEROPOLE energy.

    The energy is the exact bond-symmetrised second moment via libint
    ``emultipole2`` (``compute_ext_el_spheropole`` v3, 2026-06-01
    ``fcd16eb5``). The native derivative engine currently segfaults on
    some valid zero-component shell pairs in the vendored libint build, so
    this path uses an exact central finite difference of that energy at
    fixed density. This keeps the public BIPOLE gradient process-safe while
    the native derivative port is repaired.
    """
    from .bipole_ext_el_pole import compute_ext_el_spheropole

    if system.dim != 3:
        return np.zeros((len(system.unit_cell), 3), dtype=np.float64)

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = [Atom(int(a.Z), list(a.xyz)) for a in system.unit_cell]
    base_xyz = [np.asarray(a.xyz, dtype=float) for a in atoms]
    base_shells = list(basis.shells())
    density_blocks = [np.asarray(b, dtype=float).copy() for b in D_real.blocks]
    n_cells = len(density_blocks)
    h = 1.0e-5
    # Build a cell-index -> block map from the reference geometry so
    # we can re-match blocks when the displaced cell list reorders them.
    _cell_idx_to_block = {
        tuple(int(x) for x in c.index): np.asarray(b, dtype=float).copy()
        for c, b in zip(D_real.cells, density_blocks)
    }

    def _periodic_system_for(pert_atoms: Sequence[Atom]) -> PeriodicSystem:
        s = PeriodicSystem(3, lattice, list(pert_atoms))
        s.charge = int(system.charge)
        s.multiplicity = int(system.multiplicity)
        return s

    def _basis_for(pert_atoms: Sequence[Atom]) -> BasisSet:
        deltas = [
            np.asarray(a.xyz, dtype=float) - base_xyz[i]
            for i, a in enumerate(pert_atoms)
        ]
        shells: list[ShellInfo] = []
        for sh in base_shells:
            atom_index = int(sh.atom_index)
            origin = np.asarray(sh.origin, dtype=float) + deltas[atom_index]
            shells.append(
                ShellInfo(
                    atom_index,
                    int(sh.l),
                    bool(sh.pure),
                    list(sh.exponents),
                    list(sh.coefficients),
                    [float(x) for x in origin],
                )
            )
        mol = _periodic_system_for(pert_atoms).unit_cell_molecule()
        return BasisSet(mol, shells, str(basis.name), True)

    def _energy_at(atom_index: int, axis: int, sign: float) -> float:
        pert = [Atom(int(a.Z), list(a.xyz)) for a in atoms]
        xyz = list(pert[atom_index].xyz)
        xyz[axis] += sign * h
        pert[atom_index] = Atom(int(pert[atom_index].Z), xyz)
        s = _periodic_system_for(pert)
        b = _basis_for(pert)
        Dp = compute_overlap_lattice(b, s, lattice_opts)
        # Re-match blocks by cell index (a tiny displacement reorders
        # lattice cells near the cutoff edge but the home cell and its
        # neighbours are stable).  Cells missing from the reference set
        # get a zero block.
        n_matched = 0
        for c_idx in range(len(Dp.cells)):
            key = tuple(int(x) for x in Dp.cells[c_idx].index)
            block = _cell_idx_to_block.get(key)
            if block is not None:
                Dp.set_block(c_idx, block)
                n_matched += 1
            else:
                Dp.set_block(c_idx, np.zeros_like(density_blocks[0], dtype=float))
        if n_matched == 0:
            raise RuntimeError(
                "_spheropole_ewald_gradient: displaced density matched "
                "zero cells -- the finite-difference displacement is too "
                "large for this cell list"
            )
        return float(compute_ext_el_spheropole(Dp, b, s, lattice_opts))

    grad = np.zeros((len(atoms), 3), dtype=np.float64)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            grad[atom_index, axis] = (
                _energy_at(atom_index, axis, +1.0) - _energy_at(atom_index, axis, -1.0)
            ) / (2.0 * h)
    return grad


def _home_cell_index(cells) -> int:
    """Index of the (0,0,0) lattice cell in a cell list."""
    for i, c in enumerate(cells):
        if tuple(int(x) for x in c.index) == (0, 0, 0):
            return i
    raise ValueError("no (0,0,0) home cell in lattice cell list")


def _bipole_de_dp_home_block(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    *,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """The home-cell block of dE_total/dP(Γ) for a Γ-only BIPOLE run.

    Under the Γ-only density-locality projection (``_zero_cross_cell_density``)
    the BIPOLE energy is the LOCAL contraction ``Tr[D(0).H(0)] +
    1/2Tr[D(0).F2e(0)] + E_sph``, so the matrix the energy differentiates
    w.r.t. the density is the *home-cell* (g=0) block of

        dE/dP = Hcore(0) + J_SR(0) + J^LR(0) + 1/2.v_bg.S(0)
                - 1/2.a_HF.K(0) + prefactor.K_sph(0).

    Two subtleties vs the SCF Fock that is *diagonalised*: the J^LR jellium
    background enters the energy at the Coulomb 1/2 (not the full v_bg.S the
    Fock carries), and the spheropole (energy-only, absent from the Fock)
    contributes ``prefactor.K_sph``. This block -- NOT the Bloch-summed
    F(Γ) -- gives the correct energy-weighted density for the Pulay term
    (its occ-virt block vanishes at the converged density, so no CPHF is
    needed).
    """
    from ._vibeqc_core import (
        build_fock_2e_real_space,
        compute_kinetic_lattice,
    )
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import _build_j_long_range_cache
    from .pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )

    alpha = float(ewald_alpha)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    n_elec = int(system.n_electrons())
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    v_bg = -np.pi * float(n_elec) / (alpha * alpha * V)

    cells = list(D_real.cells)
    home = _home_cell_index(cells)

    def b0(latset_or_blocks):
        blocks = getattr(latset_or_blocks, "blocks", latset_or_blocks)
        return np.asarray(blocks[home], dtype=float)

    S = compute_overlap_lattice(basis, system, lattice_opts)
    T0 = b0(compute_kinetic_lattice(basis, system, lattice_opts))
    ew = _crystal_ewald_options(
        lattice_opts,
        alpha_bohr_inv=alpha,
        tolerance=ewald_precision,
        recip_cutoff_bohr_inv=K_max,
    )
    Vne, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        lattice_opts,
        ew,
        S,
        precision=ewald_precision,
        K_max=K_max,
    )
    Vne0 = b0(Vne)
    Jsr0 = b0(build_fock_2e_real_space(basis, system, lattice_opts, D_real, 0.0, alpha))
    # dE_jlr/dP(0) for the MIXED Bloch/local J^LR energy
    # E_jlr = 1/2 S_K kernel.Re[r̂_b.r̂_l*] (both linear in P(Γ)):
    #   d/dP(0)_muν = 1/2 S_K kernel.Re[ (S_g FT(K;g))_muν.r̂_l* + r̂_b.FT(K;0)_muν* ]
    # (the 2nd term is 1/2.F_LR_scf(0); both FTs carry the cache's per-AO
    # correction). D_local = D_real (projected, D(g!=0)=0); D_bloch(g)=P(Γ).
    crc = np.array([np.asarray(c.r_cart, dtype=float) for c in cells], dtype=float)
    cache = _build_j_long_range_cache(basis, system, crc, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
    ft = cache.ft_per_cell
    kern = cache.kernel
    D_local = np.array(
        [np.asarray(D_real.blocks[c], dtype=float) for c in range(len(cells))]
    )
    D_bloch = np.broadcast_to(D_local[home], D_local.shape)
    rho_l = np.einsum("gmn,gmnk->k", D_local, ft)
    rho_b = np.einsum("gmn,gmnk->k", D_bloch, ft)
    ft_sum = ft.sum(axis=0)  # (nbf, nbf, n_K) = S_g FT(K;g)
    ft_home = ft[home]  # FT(K; 0)
    Jlr0 = 0.5 * np.real(
        np.einsum("k,mnk->mn", kern * np.conj(rho_l), ft_sum)
        + np.einsum("k,mnk->mn", kern * rho_b, np.conj(ft_home))
    )
    # -1/2.a_HF.K(0) = [J - 1/2a_HF K](0) - J(0)
    minus_half_K0 = b0(
        build_fock_2e_real_space(
            basis, system, lattice_opts, D_real, float(alpha_hf), 0.0
        )
    ) - b0(build_fock_2e_real_space(basis, system, lattice_opts, D_real, 0.0, 0.0))
    # Spheropole dE/dP(0) = prefactor.K(0) (emultipole2 v3 energy; linear in P).
    sph0 = _spheropole_dedp_home_block(system, basis, lattice_opts)

    return T0 + Vne0 + Jsr0 + Jlr0 + 0.5 * v_bg * b0(S) + minus_half_K0 + sph0


def _corrected_w_gamma_closed(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    mo_coeffs_gamma: np.ndarray,
    n_occ: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    extra_home_block: Optional[np.ndarray] = None,
    *,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Γ-only closed-shell energy-weighted density consistent with the
    LOCAL BIPOLE energy: ``W = 2.C_occ.(C_occ+ dE/dP(0) C_occ).C_occ+``.

    Replaces the naive ``W = 2S_i e_i c_i c_i+`` (which uses the
    *diagonalised* Bloch F(Γ) eigenvalues and over-counts the Pulay term
    because the energy is a local home-cell contraction, not the Bloch
    energy). Returns a real ``(nbf, nbf)`` matrix.

    ``extra_home_block`` (KS path): a home-cell matrix added to ``dE/dP(0)``
    -- the ``V_xc(0)`` block, so the KS energy-weighted density uses the full
    KS Fock ``F_KS = Hcore + J + a_HF.K + V_xc`` rather than the HF part.
    """
    dEdP0 = _bipole_de_dp_home_block(
        system,
        basis,
        D_real,
        lattice_opts,
        float(ewald_alpha),
        float(alpha_hf),
        ewald_precision=ewald_precision,
    )
    if extra_home_block is not None:
        dEdP0 = dEdP0 + np.asarray(extra_home_block, dtype=float)
    C = np.asarray(mo_coeffs_gamma)
    C_occ = C[:, :n_occ]
    F_occ = C_occ.conj().T @ dEdP0 @ C_occ
    return np.real(2.0 * (C_occ @ F_occ @ C_occ.conj().T))


def _corrected_w_multi_k_closed(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs: Sequence[np.ndarray],
    mo_energies: Sequence[np.ndarray],
    n_occ: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
) -> List[np.ndarray]:
    """Closed-shell multi-k W consistent with BIPOLE's Ewald energy.

    The diagonalised SCF Fock carries the full J^LR jellium background
    ``v_bg * S(k)`` and omits the post-SCF EXT EL-SPHEROPOLE operator.  The
    energy derivative entering the Pulay term instead carries
    ``0.5 * v_bg * S(k) + dE_sph/dP(k)``.  Starting from the standard
    ``2*C*eps*C+`` W, add the occupied-space projection of
    ``-0.5*v_bg*S(k) + dE_sph/dP(k)`` at each k point.
    """
    from .pbc_bipole_common import _bloch_sum_blocks

    W_k_list = _build_energy_weighted_density_closed(
        list(mo_coeffs), list(mo_energies), n_occ
    )
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    v_bg = -np.pi * float(system.n_electrons()) / (float(ewald_alpha) ** 2 * V)
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    sph_blocks = _spheropole_dedp_lattice_blocks(system, basis, lattice_opts)
    if len(sph_blocks) != len(S_lat.cells):
        raise ValueError(
            "_corrected_w_multi_k_closed: spheropole and overlap cell lists "
            f"differ ({len(sph_blocks)} vs {len(S_lat.cells)})"
        )

    delta_blocks = [
        np.asarray(sph_blocks[c], dtype=float)
        - 0.5 * v_bg * np.asarray(S_lat.blocks[c], dtype=float)
        for c in range(len(S_lat.cells))
    ]
    corrected: List[np.ndarray] = []
    for C_k, W_k, k_arr in zip(mo_coeffs, W_k_list, kmesh.kpoints):
        C = np.asarray(C_k, dtype=np.complex128)
        C_occ = C[:, :n_occ]
        delta_k = _bloch_sum_blocks(delta_blocks, S_lat.cells, np.asarray(k_arr))
        delta_occ = C_occ.conj().T @ delta_k @ C_occ
        W_corr = np.asarray(W_k, dtype=np.complex128) + 2.0 * (
            C_occ @ delta_occ @ C_occ.conj().T
        )
        corrected.append(0.5 * (W_corr + W_corr.conj().T))
    return corrected


def _corrected_w_multi_k_open(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs_alpha: Sequence[np.ndarray],
    mo_energies_alpha: Sequence[np.ndarray],
    mo_coeffs_beta: Sequence[np.ndarray],
    mo_energies_beta: Sequence[np.ndarray],
    n_alpha: int,
    n_beta: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
) -> List[np.ndarray]:
    """Open-shell multi-k W consistent with BIPOLE's Ewald energy."""
    from .pbc_bipole_common import _bloch_sum_blocks

    W_k_list = _build_energy_weighted_density_open(
        list(mo_coeffs_alpha),
        list(mo_energies_alpha),
        list(mo_coeffs_beta),
        list(mo_energies_beta),
        n_alpha,
        n_beta,
    )
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    v_bg = -np.pi * float(system.n_electrons()) / (float(ewald_alpha) ** 2 * V)
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    sph_blocks = _spheropole_dedp_lattice_blocks(system, basis, lattice_opts)
    if len(sph_blocks) != len(S_lat.cells):
        raise ValueError(
            "_corrected_w_multi_k_open: spheropole and overlap cell lists "
            f"differ ({len(sph_blocks)} vs {len(S_lat.cells)})"
        )
    delta_blocks = [
        np.asarray(sph_blocks[c], dtype=float)
        - 0.5 * v_bg * np.asarray(S_lat.blocks[c], dtype=float)
        for c in range(len(S_lat.cells))
    ]

    corrected: List[np.ndarray] = []
    for Ca_k, Cb_k, W_k, k_arr in zip(
        mo_coeffs_alpha, mo_coeffs_beta, W_k_list, kmesh.kpoints
    ):
        delta_k = _bloch_sum_blocks(delta_blocks, S_lat.cells, np.asarray(k_arr))
        W_corr = np.asarray(W_k, dtype=np.complex128).copy()
        if n_alpha > 0:
            Ca_occ = np.asarray(Ca_k, dtype=np.complex128)[:, :n_alpha]
            delta_a = Ca_occ.conj().T @ delta_k @ Ca_occ
            W_corr += Ca_occ @ delta_a @ Ca_occ.conj().T
        if n_beta > 0:
            Cb_occ = np.asarray(Cb_k, dtype=np.complex128)[:, :n_beta]
            delta_b = Cb_occ.conj().T @ delta_k @ Cb_occ
            W_corr += Cb_occ @ delta_b @ Cb_occ.conj().T
        corrected.append(0.5 * (W_corr + W_corr.conj().T))
    return corrected


def _density_set_from_k_density_matrices(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    kmesh,
    D_k_list: Sequence[np.ndarray],
    *,
    cells: Optional[Sequence[object]] = None,
) -> LatticeMatrixSet:
    """Inverse-Bloch fold arbitrary per-k densities into a real lattice set."""
    if cells is None:
        template = compute_overlap_lattice(basis, system, lattice_opts)
    else:
        template = make_lattice_matrix_set(
            int(basis.nbasis),
            list(cells),
            [
                np.zeros((int(basis.nbasis), int(basis.nbasis)), dtype=float)
                for _ in cells
            ],
        )
    weights = list(kmesh.weights)
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]
    if len(D_k_list) != len(kpts):
        raise ValueError(
            "_density_set_from_k_density_matrices: density/kmesh length "
            f"mismatch ({len(D_k_list)} vs {len(kpts)})"
        )
    for c, cell in enumerate(template.cells):
        R = np.asarray(cell.r_cart, dtype=float)
        block = np.zeros_like(np.asarray(D_k_list[0]).real)
        for w_k, k_arr, D_k in zip(weights, kpts, D_k_list):
            phase = np.exp(-1j * float(np.dot(k_arr, R)))
            block += float(w_k) * np.real(phase * np.asarray(D_k))
        template.set_block(c, block)
    return template


def _build_multi_k_bipole_b0_closed(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs: Sequence[np.ndarray],
    mo_energies: Sequence[np.ndarray],
    n_occ: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    *,
    exchange_ewald_split: bool = False,
    sr_image_extent_bohr: Optional[float] = None,
    sr_density_cells: Optional[Sequence[object]] = None,
    ewald_precision: float = 1e-8,
) -> List[np.ndarray]:
    """Fixed-C multi-k SCF orbital gradient B0(k) for RHF.

    The reference ``C(k)`` is renormalised against each displaced ``S(k)`` so
    the occupied block remains S-orthonormal while differentiating ``B0``.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    from ._vibeqc_core import (
        build_fock_2e_real_space,
        build_jk_2e_real_space,
        build_jk_2e_real_space_domains,
        compute_kinetic_lattice,
    )
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        build_k_exchange_long_range_cache,
        compute_K_long_range_at_k,
        compute_J_long_range_real_space_blocks,
        compute_rho_hat_from_k_density,
        probe_charge_madelung_supercell,
        exchange_q0_gauge_constant,
    )
    from .pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )
    from .pbc_bipole_common import _bloch_sum_blocks

    alpha = float(ewald_alpha)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    T_lat = compute_kinetic_lattice(basis, system, lattice_opts)
    ew = _crystal_ewald_options(
        lattice_opts,
        alpha_bohr_inv=alpha,
        tolerance=ewald_precision,
        recip_cutoff_bohr_inv=K_max,
    )
    V_lat, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        lattice_opts,
        ew,
        S_lat,
        precision=ewald_precision,
        K_max=K_max,
    )
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]

    D_k_list: List[np.ndarray] = []
    for C_k, k_arr in zip(mo_coeffs, kpts):
        C = np.asarray(C_k, dtype=np.complex128)
        C_occ = C[:, :n_occ]
        S_k = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
        metric = C_occ.conj().T @ S_k @ C_occ
        D_k_list.append(2.0 * C_occ @ np.linalg.inv(metric) @ C_occ.conj().T)
    D_output = _density_set_from_k_density_matrices(
        system, basis, lattice_opts, kmesh, D_k_list
    )
    D_real = (
        D_output
        if sr_density_cells is None
        else _density_set_from_k_density_matrices(
            system,
            basis,
            lattice_opts,
            kmesh,
            D_k_list,
            cells=sr_density_cells,
        )
    )
    output_cells = list(S_lat.cells)
    k_corr_per_k = None
    if exchange_ewald_split:
        if sr_image_extent_bohr is None or sr_density_cells is None:
            raise ValueError(
                "_build_multi_k_bipole_b0_closed: the padded corrected "
                "route requires its internal and density-support domains"
            )
        internal_cells = list(
            _sr_internal_cells(basis, system, lattice_opts, sr_image_extent_bohr)
        )
        internal_index = {
            tuple(int(x) for x in cell.index): i
            for i, cell in enumerate(internal_cells)
        }
        if lattice_opts.pair_complete_1e:
            output_cells = internal_cells
        output_indices = [
            internal_index[tuple(int(x) for x in cell.index)]
            for cell in output_cells
        ]
        jk_sr = build_jk_2e_real_space_domains(
            basis,
            system,
            lattice_opts,
            D_real,
            internal_cells,
            output_indices,
            [],
            alpha,
            True,
        )
        j_sr_blocks = [
            np.asarray(jk_sr.J.blocks[i], dtype=np.float64)
            for i in output_indices
        ]
        k_sr_blocks = [
            np.asarray(jk_sr.K.blocks[i], dtype=np.float64)
            for i in output_indices
        ]
    else:
        J_sr = build_fock_2e_real_space(
            basis,
            system,
            lattice_opts,
            D_output,
            0.0,
            alpha,
        )
        JK_full = build_jk_2e_real_space(
            basis, system, lattice_opts, D_output, 0.0
        )
        j_sr_blocks = [
            np.asarray(block, dtype=np.float64) for block in J_sr.blocks
        ]
        k_sr_blocks = [
            np.asarray(block, dtype=np.float64) for block in JK_full.K.blocks
        ]
    cells_r = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
    )
    cache = _build_j_long_range_cache(basis, system, cells_r, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
    rho_hat = compute_rho_hat_from_k_density(D_k_list, kpts, kmesh.weights, cache)
    J_lr_blocks = compute_J_long_range_real_space_blocks(
        D_output,
        basis,
        system,
        alpha,
        cache=cache,
        rho_hat=rho_hat,
        precision=ewald_precision,
    )
    if lattice_opts.pair_complete_1e:
        J_lr_blocks = _reciprocal_blocks_on_output(J_lr_blocks, cache, system, output_cells)
    overlap_by_cell = {tuple(int(x) for x in cell.index): np.asarray(block, dtype=float)
                       for cell, block in zip(S_lat.cells, S_lat.blocks)}
    overlap_blocks = [overlap_by_cell.get(tuple(int(x) for x in cell.index),
                      np.zeros((basis.nbasis, basis.nbasis))) for cell in output_cells]
    v_bg = -np.pi * float(system.n_electrons()) / (alpha * alpha * V)
    f2e_blocks: list[np.ndarray] = []
    for c in range(len(output_cells)):
        f2e_blocks.append(
            j_sr_blocks[c]
            - 0.5 * k_sr_blocks[c]
            + J_lr_blocks[c]
            + v_bg * overlap_blocks[c]
        )
    if exchange_ewald_split:
        x_cache = build_k_exchange_long_range_cache(
            basis, system, cache, K_max=K_max
        )
        c_g0 = exchange_q0_gauge_constant(
            probe_charge_madelung_supercell(system, list(kmesh.mesh)),
            alpha,
            V,
            len(kpts),
        )
        k_corr_per_k = []
        for k_idx, k_arr in enumerate(kpts):
            k_lr = compute_K_long_range_at_k(
                x_cache,
                k_arr,
                kpts,
                list(kmesh.weights),
                D_k_list,
            )
            S_k = _bloch_sum_blocks(
                S_lat.blocks, S_lat.cells, k_arr
            )
            k_corr_per_k.append(
                k_lr + c_g0 * (S_k @ D_k_list[k_idx] @ S_k)
            )

    out: List[np.ndarray] = []
    for k_idx, (C_k, eps_k, k_arr) in enumerate(
        zip(mo_coeffs, mo_energies, kpts)
    ):
        C = np.asarray(C_k, dtype=np.complex128)
        eps = np.asarray(eps_k, dtype=float)
        C_occ = C[:, :n_occ]
        C_vir = C[:, n_occ:]
        S_k = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
        H_k = _bloch_sum_blocks(T_lat.blocks, S_lat.cells, k_arr) + _bloch_sum_blocks(
            V_lat.blocks, S_lat.cells, k_arr
        )
        F_k = H_k + _bloch_sum_blocks(f2e_blocks, output_cells, k_arr)
        if k_corr_per_k is not None:
            F_k = F_k - 0.5 * k_corr_per_k[k_idx]
        out.append(
            C_occ.conj().T @ F_k @ C_vir
            - (C_occ.conj().T @ S_k @ C_vir) * eps[:n_occ, None]
        )
    return out


def _build_multi_k_bipole_b0_open(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs_alpha: Sequence[np.ndarray],
    mo_energies_alpha: Sequence[np.ndarray],
    mo_coeffs_beta: Sequence[np.ndarray],
    mo_energies_beta: Sequence[np.ndarray],
    n_alpha: int,
    n_beta: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    *,
    exchange_ewald_split: bool = False,
    sr_image_extent_bohr: Optional[float] = None,
    sr_density_cells: Optional[Sequence[object]] = None,
    ewald_precision: float = 1e-8,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Fixed-C multi-k SCF orbital gradients ``B0_s(k)`` for UHF."""
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    from ._vibeqc_core import (
        build_fock_2e_real_space,
        build_jk_2e_real_space,
        build_jk_2e_real_space_domains,
        compute_kinetic_lattice,
    )
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        build_k_exchange_long_range_cache,
        compute_K_long_range_at_k,
        compute_J_long_range_real_space_blocks,
        compute_rho_hat_from_k_density,
        probe_charge_madelung_supercell,
        exchange_q0_gauge_constant,
    )
    from .pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )
    from .pbc_bipole_common import _bloch_sum_blocks

    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    T_lat = compute_kinetic_lattice(basis, system, lattice_opts)
    ew = _crystal_ewald_options(
        lattice_opts,
        alpha_bohr_inv=alpha,
        tolerance=ewald_precision,
        recip_cutoff_bohr_inv=K_max,
    )
    V_lat, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        lattice_opts,
        ew,
        S_lat,
        precision=ewald_precision,
        K_max=K_max,
    )
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]

    def _spin_density(C_k: np.ndarray, n_occ: int, S_k: np.ndarray) -> np.ndarray:
        C = np.asarray(C_k, dtype=np.complex128)
        D = np.zeros((C.shape[0], C.shape[0]), dtype=np.complex128)
        if n_occ > 0:
            C_occ = C[:, :n_occ]
            metric = C_occ.conj().T @ S_k @ C_occ
            D = C_occ @ np.linalg.inv(metric) @ C_occ.conj().T
        return D

    D_alpha_k: List[np.ndarray] = []
    D_beta_k: List[np.ndarray] = []
    D_total_k: List[np.ndarray] = []
    for Ca_k, Cb_k, k_arr in zip(mo_coeffs_alpha, mo_coeffs_beta, kpts):
        S_k = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
        Da = _spin_density(np.asarray(Ca_k), n_alpha, S_k)
        Db = _spin_density(np.asarray(Cb_k), n_beta, S_k)
        D_alpha_k.append(Da)
        D_beta_k.append(Db)
        D_total_k.append(Da + Db)

    D_total_output = _density_set_from_k_density_matrices(
        system, basis, lattice_opts, kmesh, D_total_k
    )
    D_alpha_output = _density_set_from_k_density_matrices(
        system, basis, lattice_opts, kmesh, D_alpha_k
    )
    D_beta_output = _density_set_from_k_density_matrices(
        system, basis, lattice_opts, kmesh, D_beta_k
    )
    if sr_density_cells is None:
        D_total_real = D_total_output
        D_alpha_real = D_alpha_output
        D_beta_real = D_beta_output
    else:
        D_total_real = _density_set_from_k_density_matrices(
            system,
            basis,
            lattice_opts,
            kmesh,
            D_total_k,
            cells=sr_density_cells,
        )
        D_alpha_real = _density_set_from_k_density_matrices(
            system,
            basis,
            lattice_opts,
            kmesh,
            D_alpha_k,
            cells=sr_density_cells,
        )
        D_beta_real = _density_set_from_k_density_matrices(
            system,
            basis,
            lattice_opts,
            kmesh,
            D_beta_k,
            cells=sr_density_cells,
        )

    output_cells = list(S_lat.cells)
    k_corr_alpha = None
    k_corr_beta = None
    if exchange_ewald_split:
        if sr_image_extent_bohr is None or sr_density_cells is None:
            raise ValueError(
                "_build_multi_k_bipole_b0_open: the padded corrected route "
                "requires its internal and density-support domains"
            )
        internal_cells = list(
            _sr_internal_cells(basis, system, lattice_opts, sr_image_extent_bohr)
        )
        internal_index = {
            tuple(int(x) for x in cell.index): i
            for i, cell in enumerate(internal_cells)
        }
        if lattice_opts.pair_complete_1e:
            output_cells = internal_cells
        output_indices = [
            internal_index[tuple(int(x) for x in cell.index)]
            for cell in output_cells
        ]
        jk_total = build_jk_2e_real_space_domains(
            basis,
            system,
            lattice_opts,
            D_total_real,
            internal_cells,
            output_indices,
            [],
            alpha,
            False,
        )
        jk_alpha = build_jk_2e_real_space_domains(
            basis,
            system,
            lattice_opts,
            D_alpha_real,
            internal_cells,
            output_indices,
            [],
            alpha,
            True,
        )
        jk_beta = build_jk_2e_real_space_domains(
            basis,
            system,
            lattice_opts,
            D_beta_real,
            internal_cells,
            output_indices,
            [],
            alpha,
            True,
        )
        j_sr_blocks = [
            np.asarray(jk_total.J.blocks[i], dtype=np.float64)
            for i in output_indices
        ]
        k_alpha_blocks = [
            np.asarray(jk_alpha.K.blocks[i], dtype=np.float64)
            for i in output_indices
        ]
        k_beta_blocks = [
            np.asarray(jk_beta.K.blocks[i], dtype=np.float64)
            for i in output_indices
        ]
    else:
        J_sr = build_fock_2e_real_space(
            basis,
            system,
            lattice_opts,
            D_total_output,
            0.0,
            alpha,
        )
        K_alpha = build_jk_2e_real_space(
            basis, system, lattice_opts, D_alpha_output, 0.0
        ).K
        K_beta = build_jk_2e_real_space(
            basis, system, lattice_opts, D_beta_output, 0.0
        ).K
        j_sr_blocks = [
            np.asarray(block, dtype=np.float64) for block in J_sr.blocks
        ]
        k_alpha_blocks = [
            np.asarray(block, dtype=np.float64) for block in K_alpha.blocks
        ]
        k_beta_blocks = [
            np.asarray(block, dtype=np.float64) for block in K_beta.blocks
        ]
    cells_r = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
    )
    cache = _build_j_long_range_cache(basis, system, cells_r, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
    rho_hat = compute_rho_hat_from_k_density(D_total_k, kpts, kmesh.weights, cache)
    J_lr_blocks = compute_J_long_range_real_space_blocks(
        D_total_output,
        basis,
        system,
        alpha,
        cache=cache,
        rho_hat=rho_hat,
        precision=ewald_precision,
    )
    if lattice_opts.pair_complete_1e:
        J_lr_blocks = _reciprocal_blocks_on_output(J_lr_blocks, cache, system, output_cells)
    overlap_by_cell = {tuple(int(x) for x in cell.index): np.asarray(block, dtype=float)
                       for cell, block in zip(S_lat.cells, S_lat.blocks)}
    overlap_blocks = [overlap_by_cell.get(tuple(int(x) for x in cell.index),
                      np.zeros((basis.nbasis, basis.nbasis))) for cell in output_cells]
    v_bg = -np.pi * float(system.n_electrons()) / (alpha * alpha * V)
    f_alpha_blocks: list[np.ndarray] = []
    f_beta_blocks: list[np.ndarray] = []
    for c in range(len(output_cells)):
        common = (
            j_sr_blocks[c]
            + J_lr_blocks[c]
            + v_bg * overlap_blocks[c]
        )
        f_alpha_blocks.append(common - a_hf * k_alpha_blocks[c])
        f_beta_blocks.append(common - a_hf * k_beta_blocks[c])
    if exchange_ewald_split:
        x_cache = build_k_exchange_long_range_cache(
            basis, system, cache, K_max=K_max
        )
        c_g0 = exchange_q0_gauge_constant(
            probe_charge_madelung_supercell(system, list(kmesh.mesh)),
            alpha,
            V,
            len(kpts),
        )
        k_corr_alpha = []
        k_corr_beta = []
        for k_idx, k_arr in enumerate(kpts):
            S_k = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
            k_corr_alpha.append(
                compute_K_long_range_at_k(
                    x_cache,
                    k_arr,
                    kpts,
                    list(kmesh.weights),
                    D_alpha_k,
                )
                + c_g0 * (S_k @ D_alpha_k[k_idx] @ S_k)
            )
            k_corr_beta.append(
                compute_K_long_range_at_k(
                    x_cache,
                    k_arr,
                    kpts,
                    list(kmesh.weights),
                    D_beta_k,
                )
                + c_g0 * (S_k @ D_beta_k[k_idx] @ S_k)
            )

    out_alpha: List[np.ndarray] = []
    out_beta: List[np.ndarray] = []
    for k_idx, (Ca_k, ea_k, Cb_k, eb_k, k_arr) in enumerate(zip(
        mo_coeffs_alpha,
        mo_energies_alpha,
        mo_coeffs_beta,
        mo_energies_beta,
        kpts,
    )):
        S_k = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
        H_k = _bloch_sum_blocks(T_lat.blocks, S_lat.cells, k_arr) + _bloch_sum_blocks(
            V_lat.blocks, S_lat.cells, k_arr
        )
        Ca = np.asarray(Ca_k, dtype=np.complex128)
        Cb = np.asarray(Cb_k, dtype=np.complex128)
        if Ca.shape[1] > n_alpha:
            Ca_occ = Ca[:, :n_alpha]
            Ca_vir = Ca[:, n_alpha:]
            F_a = H_k + _bloch_sum_blocks(f_alpha_blocks, output_cells, k_arr)
            if k_corr_alpha is not None:
                F_a = F_a - a_hf * k_corr_alpha[k_idx]
            out_alpha.append(
                Ca_occ.conj().T @ F_a @ Ca_vir
                - (Ca_occ.conj().T @ S_k @ Ca_vir)
                * np.asarray(ea_k, dtype=float)[:n_alpha, None]
            )
        else:
            out_alpha.append(np.zeros((n_alpha, 0), dtype=np.complex128))
        if Cb.shape[1] > n_beta:
            Cb_occ = Cb[:, :n_beta]
            Cb_vir = Cb[:, n_beta:]
            F_b = H_k + _bloch_sum_blocks(f_beta_blocks, output_cells, k_arr)
            if k_corr_beta is not None:
                F_b = F_b - a_hf * k_corr_beta[k_idx]
            out_beta.append(
                Cb_occ.conj().T @ F_b @ Cb_vir
                - (Cb_occ.conj().T @ S_k @ Cb_vir)
                * np.asarray(eb_k, dtype=float)[:n_beta, None]
            )
        else:
            out_beta.append(np.zeros((n_beta, 0), dtype=np.complex128))
    return out_alpha, out_beta


def _multi_k_orbital_relaxation_closed_diag(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs: Sequence[np.ndarray],
    mo_energies: Sequence[np.ndarray],
    n_occ: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    *,
    step_bohr: float = 1e-4,
    energy_fock_delta_k: Optional[Sequence[np.ndarray]] = None,
    sr_image_extent_bohr: Optional[float] = None,
    sr_density_cells: Optional[Sequence[object]] = None,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Multi-k RHF response for the BIPOLE energy-vs-Fock delta.

    The historical-domain route retains its documented diagonal orbital
    Hessian. When ``energy_fock_delta_k`` is supplied for the padded M5 route,
    the finite-domain map couples real and imaginary orbital rotations across
    k points. Its unequal domains need not give a self-adjoint SCF response.
    Solve the transpose of that full real-linear Jacobian, with k weights in
    the energy RHS, and fail closed if its residual does not converge.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    from .pbc_bipole_common import _bloch_sum_blocks

    n_atoms = len(system.unit_cell)
    if system.dim != 3 or kmesh is None:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    C0 = [np.asarray(C, dtype=np.complex128) for C in mo_coeffs]
    eps0 = [np.asarray(e, dtype=float) for e in mo_energies]
    if not C0 or C0[0].shape[1] <= n_occ:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    delta_k_input = (
        None
        if energy_fock_delta_k is None
        else [np.asarray(item, dtype=np.complex128) for item in energy_fock_delta_k]
    )
    if delta_k_input is None:
        V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        v_bg = -np.pi * float(system.n_electrons()) / (
            float(ewald_alpha) ** 2 * V
        )
        sph_blocks = _spheropole_dedp_lattice_blocks(
            system, basis, lattice_opts
        )
        delta_blocks = [
            np.asarray(sph_blocks[c], dtype=float)
            - 0.5 * v_bg * np.asarray(S_lat.blocks[c], dtype=float)
            for c in range(len(S_lat.cells))
        ]
    elif len(delta_k_input) != len(C0):
        raise ValueError(
            "_multi_k_orbital_relaxation_closed_diag: energy/Fock delta "
            "length does not match the k mesh"
        )
    rhs_list: List[np.ndarray] = []
    ediff_list: List[np.ndarray] = []
    for k_idx, (C, eps, k_arr) in enumerate(
        zip(C0, eps0, kmesh.kpoints)
    ):
        C_occ = C[:, :n_occ]
        C_vir = C[:, n_occ:]
        delta_k = (
            _bloch_sum_blocks(delta_blocks, S_lat.cells, np.asarray(k_arr))
            if delta_k_input is None
            else delta_k_input[k_idx]
        )
        rhs_list.append(C_occ.conj().T @ delta_k @ C_vir)
        ediff_list.append(eps[n_occ:][None, :] - eps[:n_occ, None])

    if delta_k_input is None:
        z_list = [
            rhs / ediff for rhs, ediff in zip(rhs_list, ediff_list)
        ]
    else:
        sizes = [item.size for item in rhs_list]
        offsets = np.cumsum([0] + sizes)
        n_complex = int(offsets[-1])

        def _unpack_complex(flat):
            return [
                np.asarray(flat[offsets[i] : offsets[i + 1]]).reshape(
                    rhs_list[i].shape
                )
                for i in range(len(rhs_list))
            ]

        def _pack_real(items):
            flat = np.concatenate(
                [np.asarray(item, dtype=np.complex128).ravel() for item in items]
            )
            return np.concatenate((flat.real, flat.imag))

        def _unpack_real(flat):
            arr = np.asarray(flat, dtype=np.float64)
            complex_flat = arr[:n_complex] + 1j * arr[n_complex:]
            return _unpack_complex(complex_flat)

        def _rotated_coeffs(v_list, sign, eta):
            out = []
            for C, v in zip(C0, v_list):
                C_occ = C[:, :n_occ]
                C_vir = C[:, n_occ:]
                # Use the B_ov convention: dE = 4 Re <v, delta_ov>.
                # Conjugating the occupied rotation also makes the gap term
                # positive in both real and imaginary coordinates.
                occ = C_occ + sign * eta * (C_vir @ v.conj().T)
                vir = C_vir - sign * eta * (C_occ @ v)
                out.append(np.concatenate((occ, vir), axis=1))
            return out

        rotation_step = 1.0e-3

        def _b0_for_coeffs(coeffs):
            return _build_multi_k_bipole_b0_closed(
                system,
                basis,
                coeffs,
                eps0,
                n_occ,
                kmesh,
                lattice_opts,
                float(ewald_alpha),
                exchange_ewald_split=True,
                sr_image_extent_bohr=sr_image_extent_bohr,
                sr_density_cells=sr_density_cells,
                ewald_precision=ewald_precision,
            )

        def _aop(flat):
            v_list = _unpack_real(flat)
            bp = _b0_for_coeffs(
                _rotated_coeffs(v_list, +1.0, rotation_step)
            )
            bm = _b0_for_coeffs(
                _rotated_coeffs(v_list, -1.0, rotation_step)
            )
            return _pack_real(
                [(p - m) / (2.0 * rotation_step) for p, m in zip(bp, bm)]
            )

        rhs = _pack_real([
            float(w) * item for w, item in zip(kmesh.weights, rhs_list)
        ])
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return np.zeros((n_atoms, 3), dtype=np.float64)
        jacobian = np.empty((rhs.size, rhs.size), dtype=np.float64)
        for col in range(rhs.size):
            direction = np.zeros(rhs.size, dtype=np.float64)
            direction[col] = 1.0
            jacobian[:, col] = _aop(direction)
        try:
            # A forward or symmetrized solve differentiates a different SCF
            # constraint. Keep k weights in z, including for nonuniform meshes.
            zflat, *_ = np.linalg.lstsq(jacobian.T, rhs, rcond=1.0e-10)
            residual = float(np.linalg.norm(jacobian.T @ zflat - rhs))
            if not np.isfinite(residual) or residual > 1.0e-9 * rhs_norm:
                raise np.linalg.LinAlgError(
                    "transpose CPHF residual exceeds tolerance "
                    f"({residual / rhs_norm:.3e} relative)"
                )
        except np.linalg.LinAlgError as exc:
            raise NotImplementedError(
                "padded multi-k RHF analytic gradient could not solve the "
                f"finite-domain orbital response ({exc}); use "
                "compute_bipole_gradient_fd"
            ) from exc
        z_list = _unpack_real(zflat)

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    h = float(step_bohr)
    weights = [float(w) for w in kmesh.weights]
    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):

            def _disp(sign):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(displaced[a].xyz)
                xyz[d] += sign * h
                displaced[a] = Atom(displaced[a].Z, xyz)
                sd = PeriodicSystem(system.dim, lattice, displaced)
                sd.charge = system.charge
                sd.multiplicity = system.multiplicity
                bd = _recenter_basis_on_periodic_system(basis, sd)
                return _build_multi_k_bipole_b0_closed(
                    sd,
                    bd,
                    C0,
                    eps0,
                    n_occ,
                    kmesh,
                    lattice_opts,
                    float(ewald_alpha),
                    exchange_ewald_split=delta_k_input is not None,
                    sr_image_extent_bohr=sr_image_extent_bohr,
                    sr_density_cells=sr_density_cells,
                    ewald_precision=ewald_precision,
                )

            Bp = _disp(+1.0)
            Bm = _disp(-1.0)
            total = 0.0
            for w_k, z_k, bp, bm in zip(weights, z_list, Bp, Bm):
                dB = (bp - bm) / (2.0 * h)
                contraction = (
                    np.sum(z_k * dB)
                    if delta_k_input is None
                    else np.vdot(z_k, dB)
                )
                # Padded z already contains the energy's k-point weights.
                weight = w_k if delta_k_input is None else 1.0
                total += weight * float(np.real(contraction))
            relax[a, d] = -4.0 * total
    return relax


def _multi_k_orbital_relaxation_ks_closed_diag(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs: Sequence[np.ndarray],
    mo_energies: Sequence[np.ndarray],
    n_occ: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    func_name: str,
    *,
    step_bohr: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Diagonal multi-k Z-vector for the RKS BIPOLE energy-vs-Fock delta.

    Recovers the leading orbital relaxation from the spheropole + jellium
    background terms that are absent from the KS Fock. Uses the same delta(k)
    as the RHF closed-shell case; V_xc is already in mo_energies from the SCF
    Fock diagonalisation.
    """
    from ._vibeqc_core import (
        Functional,
        build_fock_2e_real_space,
        build_jk_2e_real_space,
        build_xc_periodic,
        compute_kinetic_lattice,
    )
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_J_long_range_real_space_blocks,
        compute_rho_hat_from_k_density,
    )
    from .pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )
    from .pbc_bipole_common import _bloch_sum_blocks
    from .periodic_grid import build_periodic_becke_grid

    n_atoms = len(system.unit_cell)
    if system.dim != 3 or kmesh is None:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    C0 = [np.asarray(C, dtype=np.complex128) for C in mo_coeffs]
    eps0 = [np.asarray(e, dtype=float) for e in mo_energies]
    if not C0 or C0[0].shape[1] <= n_occ:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    alpha = float(ewald_alpha)
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    v_bg = -np.pi * float(system.n_electrons()) / (alpha**2 * V)
    sph_blocks = _spheropole_dedp_lattice_blocks(system, basis, lattice_opts)
    delta_blocks = [
        np.asarray(sph_blocks[c], dtype=float)
        - 0.5 * v_bg * np.asarray(S_lat.blocks[c], dtype=float)
        for c in range(len(S_lat.cells))
    ]
    z_list: List[np.ndarray] = []
    for C, eps, k_arr in zip(C0, eps0, kmesh.kpoints):
        C_occ = C[:, :n_occ]
        C_vir = C[:, n_occ:]
        if C_vir.shape[1] == 0:
            z_list.append(np.zeros((n_occ, 0), dtype=np.complex128))
            continue
        delta_k = _bloch_sum_blocks(delta_blocks, S_lat.cells, np.asarray(k_arr))
        b = C_occ.conj().T @ delta_k @ C_vir
        ediff = eps[n_occ:][None, :] - eps[:n_occ, None]
        z_list.append(b / ediff)

    # Build KS grid and Fock for B0 differentiation.
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    grid = build_periodic_becke_grid(system, image_radius_bohr=10.0)
    func = Functional(func_name, 1)
    T_lat = compute_kinetic_lattice(basis, system, lattice_opts)
    ew = _crystal_ewald_options(
        lattice_opts,
        alpha_bohr_inv=alpha,
        tolerance=ewald_precision,
        recip_cutoff_bohr_inv=K_max,
    )
    Vn_lat, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        lattice_opts,
        ew,
        S_lat,
        precision=ewald_precision,
        K_max=K_max,
    )
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]
    weights = [float(w) for w in kmesh.weights]

    def _build_b0_at(sd, bd):
        """Build per-k B0(k) for a displaced system at fixed density."""
        D_k_list: List[np.ndarray] = []
        for C_k, k_arr in zip(C0, kpts):
            Ck = np.asarray(C_k, dtype=np.complex128)
            Cko = Ck[:, :n_occ]
            Sk = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
            metric = Cko.conj().T @ Sk @ Cko
            D_k_list.append(2.0 * Cko @ np.linalg.inv(metric) @ Cko.conj().T)
        D_real = _density_set_from_k_density_matrices(
            sd, bd, lattice_opts, kmesh, D_k_list
        )
        J_sr = build_fock_2e_real_space(bd, sd, lattice_opts, D_real, 0.0, alpha)
        JK_full = build_jk_2e_real_space(bd, sd, lattice_opts, D_real, 0.0)
        cells_r = np.array(
            [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
        )
        cache = _build_j_long_range_cache(bd, sd, cells_r, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
        rho_hat = compute_rho_hat_from_k_density(D_k_list, kpts, weights, cache)
        J_lr_blocks = compute_J_long_range_real_space_blocks(
            D_real,
            bd,
            sd,
            alpha,
            cache=cache,
            rho_hat=rho_hat,
            precision=ewald_precision,
        )
        v_bg_l = -np.pi * float(sd.n_electrons()) / (alpha * alpha * V)
        vxc = build_xc_periodic(bd, sd, grid, func, D_real, lattice_opts)
        f2e_blocks: list[np.ndarray] = []
        for c in range(len(S_lat.cells)):
            block = (
                np.asarray(J_sr.blocks[c], dtype=float)
                - 0.5 * np.asarray(JK_full.K.blocks[c], dtype=float)
                + J_lr_blocks[c]
                + v_bg_l * np.asarray(S_lat.blocks[c], dtype=float)
                + np.asarray(vxc.V_xc.blocks[c], dtype=float)
            )
            f2e_blocks.append(block)
        out: List[np.ndarray] = []
        for C_k, eps_k, k_arr in zip(C0, eps0, kpts):
            Ck = np.asarray(C_k, dtype=np.complex128)
            Cko = Ck[:, :n_occ]
            Ckv = Ck[:, n_occ:]
            Sk = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
            Hk = _bloch_sum_blocks(
                T_lat.blocks, S_lat.cells, k_arr
            ) + _bloch_sum_blocks(Vn_lat.blocks, S_lat.cells, k_arr)
            Fk = Hk + _bloch_sum_blocks(f2e_blocks, S_lat.cells, k_arr)
            eps = np.asarray(eps_k, dtype=float)
            out.append(
                Cko.conj().T @ Fk @ Ckv - (Cko.conj().T @ Sk @ Ckv) * eps[:n_occ, None]
            )
        return out

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    h = float(step_bohr)
    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):

            def _disp(sign):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(displaced[a].xyz)
                xyz[d] += sign * h
                displaced[a] = Atom(displaced[a].Z, xyz)
                sd = PeriodicSystem(system.dim, lattice, displaced)
                sd.charge = system.charge
                sd.multiplicity = system.multiplicity
                bd = _recenter_basis_on_periodic_system(basis, sd)
                return _build_b0_at(sd, bd)

            Bp = _disp(+1.0)
            Bm = _disp(-1.0)
            total = 0.0
            for w_k, z_k, bp, bm in zip(weights, z_list, Bp, Bm):
                dB = (bp - bm) / (2.0 * h)
                total += w_k * float(np.real(np.sum(z_k * dB)))
            relax[a, d] = -4.0 * total
    return relax


def _multi_k_orbital_relaxation_ks_open_diag(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs_alpha: Sequence[np.ndarray],
    mo_energies_alpha: Sequence[np.ndarray],
    n_alpha: int,
    mo_coeffs_beta: Sequence[np.ndarray],
    mo_energies_beta: Sequence[np.ndarray],
    n_beta: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    func_name: str,
    *,
    step_bohr: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Diagonal per-spin multi-k Z-vector for UKS BIPOLE."""
    from ._vibeqc_core import (
        Functional,
        build_fock_2e_real_space,
        build_jk_2e_real_space,
        build_xc_periodic_uks,
        compute_kinetic_lattice,
    )
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_J_long_range_real_space_blocks,
        compute_rho_hat_from_k_density,
    )
    from .pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )
    from .pbc_bipole_common import _bloch_sum_blocks
    from .periodic_grid import build_periodic_becke_grid

    n_atoms = len(system.unit_cell)
    if system.dim != 3 or kmesh is None:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    Ca0 = [np.asarray(C, dtype=np.complex128) for C in mo_coeffs_alpha]
    Cb0 = [np.asarray(C, dtype=np.complex128) for C in mo_coeffs_beta]
    eA = [np.asarray(e, dtype=float) for e in mo_energies_alpha]
    eB = [np.asarray(e, dtype=float) for e in mo_energies_beta]
    if not Ca0:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    alpha = float(ewald_alpha)
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    v_bg = -np.pi * float(system.n_electrons()) / (alpha**2 * V)
    sph_blocks = _spheropole_dedp_lattice_blocks(system, basis, lattice_opts)
    delta_blocks = [
        np.asarray(sph_blocks[c], dtype=float)
        - 0.5 * v_bg * np.asarray(S_lat.blocks[c], dtype=float)
        for c in range(len(S_lat.cells))
    ]
    za_list: List[np.ndarray] = []
    zb_list: List[np.ndarray] = []
    for Ca, Cb, eA_, eB_, k_arr in zip(Ca0, Cb0, eA, eB, kmesh.kpoints):
        delta_k = _bloch_sum_blocks(delta_blocks, S_lat.cells, np.asarray(k_arr))
        for Cc, nocc, evec, zstore in [
            (Ca, n_alpha, eA_, za_list),
            (Cb, n_beta, eB_, zb_list),
        ]:
            if Cc.shape[1] <= nocc:
                zstore.append(np.zeros((nocc, 0), dtype=np.complex128))
                continue
            C_occ = Cc[:, :nocc]
            C_vir = Cc[:, nocc:]
            b = C_occ.conj().T @ delta_k @ C_vir
            ediff = evec[nocc:][None, :] - evec[:nocc, None]
            zstore.append(b / ediff)

    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    grid = build_periodic_becke_grid(system, image_radius_bohr=10.0)
    func = Functional(func_name, 2)
    T_lat = compute_kinetic_lattice(basis, system, lattice_opts)
    ew = _crystal_ewald_options(
        lattice_opts,
        alpha_bohr_inv=alpha,
        tolerance=ewald_precision,
        recip_cutoff_bohr_inv=K_max,
    )
    Vn_lat, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        lattice_opts,
        ew,
        S_lat,
        precision=ewald_precision,
        K_max=K_max,
    )
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]
    weights = [float(w) for w in kmesh.weights]

    def _build_b0_at(sd, bd):
        Dt_k: List[np.ndarray] = []
        Da_k: List[np.ndarray] = []
        Db_k: List[np.ndarray] = []
        for Ck_a, Ck_b, k_arr in zip(Ca0, Cb0, kpts):
            for Ck, nocc, Dstore, fac in [
                (Ck_a, n_alpha, Da_k, 1.0),
                (Ck_b, n_beta, Db_k, 1.0),
            ]:
                Cko = Ck[:, :nocc]
                Sk = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
                metric = Cko.conj().T @ Sk @ Cko
                Dstore.append(fac * Cko @ np.linalg.inv(metric) @ Cko.conj().T)
            Dt_k.append(Da_k[-1] + Db_k[-1])
        D_real = _density_set_from_k_density_matrices(sd, bd, lattice_opts, kmesh, Dt_k)
        Da_real = _density_set_from_k_density_matrices(
            sd, bd, lattice_opts, kmesh, Da_k
        )
        Db_real = _density_set_from_k_density_matrices(
            sd, bd, lattice_opts, kmesh, Db_k
        )
        J_sr = build_fock_2e_real_space(bd, sd, lattice_opts, D_real, 0.0, alpha)
        JK_full = build_jk_2e_real_space(bd, sd, lattice_opts, D_real, 0.0)
        K_a = build_jk_2e_real_space(bd, sd, lattice_opts, Da_real, 0.0).K
        K_b = build_jk_2e_real_space(bd, sd, lattice_opts, Db_real, 0.0).K
        cells_r = np.array(
            [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
        )
        cache = _build_j_long_range_cache(bd, sd, cells_r, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
        rho_hat = compute_rho_hat_from_k_density(Dt_k, kpts, weights, cache)
        J_lr_blocks = compute_J_long_range_real_space_blocks(
            D_real,
            bd,
            sd,
            alpha,
            cache=cache,
            rho_hat=rho_hat,
            precision=ewald_precision,
        )
        v_bg_l = -np.pi * float(sd.n_electrons()) / (alpha * alpha * V)
        vxc = build_xc_periodic_uks(bd, sd, grid, func, Da_real, Db_real, lattice_opts)
        fa_blocks: list[np.ndarray] = []
        fb_blocks: list[np.ndarray] = []
        for c in range(len(S_lat.cells)):
            common = (
                np.asarray(J_sr.blocks[c], dtype=float)
                + J_lr_blocks[c]
                + v_bg_l * np.asarray(S_lat.blocks[c], dtype=float)
            )
            fa_blocks.append(
                common
                - np.asarray(JK_full.K.blocks[c], dtype=float)
                + np.asarray(K_a.blocks[c], dtype=float)
                + np.asarray(vxc.V_alpha.blocks[c], dtype=float)
            )
            fb_blocks.append(
                common
                - np.asarray(JK_full.K.blocks[c], dtype=float)
                + np.asarray(K_b.blocks[c], dtype=float)
                + np.asarray(vxc.V_beta.blocks[c], dtype=float)
            )
        out_a: List[np.ndarray] = []
        out_b: List[np.ndarray] = []
        for Ck_a, Ck_b, eA_, eB_, k_arr in zip(Ca0, Cb0, eA, eB, kpts):
            Sk = _bloch_sum_blocks(S_lat.blocks, S_lat.cells, k_arr)
            Hk = _bloch_sum_blocks(
                T_lat.blocks, S_lat.cells, k_arr
            ) + _bloch_sum_blocks(Vn_lat.blocks, S_lat.cells, k_arr)
            for Ck, nocc, blocks, out in [
                (Ck_a, n_alpha, fa_blocks, out_a),
                (Ck_b, n_beta, fb_blocks, out_b),
            ]:
                if Ck.shape[1] <= nocc:
                    out.append(np.zeros((nocc, 0), dtype=np.complex128))
                    continue
                Cko = Ck[:, :nocc]
                Ckv = Ck[:, nocc:]
                Fk = Hk + _bloch_sum_blocks(blocks, S_lat.cells, k_arr)
                eps = (
                    np.asarray(eA_, dtype=float)
                    if out is out_a
                    else np.asarray(eB_, dtype=float)
                )
                out.append(
                    Cko.conj().T @ Fk @ Ckv
                    - (Cko.conj().T @ Sk @ Ckv) * eps[:nocc, None]
                )
        return out_a, out_b

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    h = float(step_bohr)
    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):

            def _disp(sign):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(displaced[a].xyz)
                xyz[d] += sign * h
                displaced[a] = Atom(displaced[a].Z, xyz)
                sd = PeriodicSystem(system.dim, lattice, displaced)
                sd.charge = system.charge
                sd.multiplicity = system.multiplicity
                bd = _recenter_basis_on_periodic_system(basis, sd)
                return _build_b0_at(sd, bd)

            Bpa, Bpb = _disp(+1.0)
            Bma, Bmb = _disp(-1.0)
            total = 0.0
            for w_k, za_k, zb_k, bpa, bma, bpb, bmb in zip(
                weights, za_list, zb_list, Bpa, Bma, Bpb, Bmb
            ):
                dBa = (bpa - bma) / (2.0 * h)
                dBb = (bpb - bmb) / (2.0 * h)
                total += w_k * float(
                    np.real(np.sum(za_k * dBa)) + np.real(np.sum(zb_k * dBb))
                )
            relax[a, d] = -2.0 * total
    return relax


def _multi_k_orbital_relaxation_open(
    system: PeriodicSystem,
    basis: BasisSet,
    mo_coeffs_alpha: Sequence[np.ndarray],
    mo_energies_alpha: Sequence[np.ndarray],
    n_alpha: int,
    mo_coeffs_beta: Sequence[np.ndarray],
    mo_energies_beta: Sequence[np.ndarray],
    n_beta: int,
    kmesh,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    *,
    step_bohr: float = 1e-4,
    energy_fock_delta_k: Optional[
        tuple[Sequence[np.ndarray], Sequence[np.ndarray]]
    ] = None,
    sr_image_extent_bohr: Optional[float] = None,
    sr_density_cells: Optional[Sequence[object]] = None,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Coupled-spin multi-k UHF Z-vector for the BIPOLE energy-vs-Fock delta.

    The legacy route uses its existing analytic coupled-spin Hessian. For the
    padded corrected-gauge route, ``energy_fock_delta_k`` supplies the
    spin-resolved finite-domain energy-vs-Fock mismatch and the complete
    real-linear k/spin-coupled Hessian is evaluated from the corrected fixed-C
    Fock builder.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    from ._vibeqc_core import build_fock_2e_real_space, build_jk_2e_real_space
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_J_long_range_real_space_blocks,
        compute_rho_hat_from_k_density,
    )
    from .pbc_bipole_common import _bloch_sum_blocks

    n_atoms = len(system.unit_cell)
    if system.dim != 3 or kmesh is None:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    Ca0 = [np.asarray(C, dtype=np.complex128) for C in mo_coeffs_alpha]
    Cb0 = [np.asarray(C, dtype=np.complex128) for C in mo_coeffs_beta]
    eps_a = [np.asarray(e, dtype=float) for e in mo_energies_alpha]
    eps_b = [np.asarray(e, dtype=float) for e in mo_energies_beta]
    if not Ca0:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    nva = [C.shape[1] - n_alpha for C in Ca0]
    nvb = [C.shape[1] - n_beta for C in Cb0]
    if all(n <= 0 for n in nva) and all(n <= 0 for n in nvb):
        return np.zeros((n_atoms, 3), dtype=np.float64)

    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    v_bg = -np.pi * float(system.n_electrons()) / (alpha * alpha * V)
    delta_spin_input = (
        None
        if energy_fock_delta_k is None
        else (
            [
                np.asarray(item, dtype=np.complex128)
                for item in energy_fock_delta_k[0]
            ],
            [
                np.asarray(item, dtype=np.complex128)
                for item in energy_fock_delta_k[1]
            ],
        )
    )
    if delta_spin_input is None:
        sph_blocks = _spheropole_dedp_lattice_blocks(
            system, basis, lattice_opts
        )
        delta_blocks = [
            np.asarray(sph_blocks[c], dtype=float)
            - 0.5 * v_bg * np.asarray(S_lat.blocks[c], dtype=float)
            for c in range(len(S_lat.cells))
        ]
    elif (
        len(delta_spin_input[0]) != len(Ca0)
        or len(delta_spin_input[1]) != len(Cb0)
    ):
        raise ValueError(
            "_multi_k_orbital_relaxation_open: energy/Fock delta lengths "
            "do not match the k mesh"
        )
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh.kpoints]
    weights = [float(w) for w in kmesh.weights]

    rhs_alpha: List[np.ndarray] = []
    rhs_beta: List[np.ndarray] = []
    for k_idx, (Ca, Cb, k_arr) in enumerate(zip(Ca0, Cb0, kpts)):
        delta_alpha = (
            _bloch_sum_blocks(delta_blocks, S_lat.cells, k_arr)
            if delta_spin_input is None
            else delta_spin_input[0][k_idx]
        )
        delta_beta = (
            delta_alpha
            if delta_spin_input is None
            else delta_spin_input[1][k_idx]
        )
        rhs_alpha.append(
            Ca[:, :n_alpha].conj().T @ delta_alpha @ Ca[:, n_alpha:]
        )
        rhs_beta.append(
            Cb[:, :n_beta].conj().T @ delta_beta @ Cb[:, n_beta:]
        )

    shapes_alpha = [(n_alpha, max(0, nv)) for nv in nva]
    shapes_beta = [(n_beta, max(0, nv)) for nv in nvb]
    n_complex = sum(a * b for a, b in shapes_alpha + shapes_beta)
    if n_complex == 0:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    def _flatten_complex(
        a_blocks: Sequence[np.ndarray],
        b_blocks: Sequence[np.ndarray],
    ) -> np.ndarray:
        pieces = [np.asarray(M, dtype=np.complex128).reshape(-1) for M in a_blocks]
        pieces.extend(np.asarray(M, dtype=np.complex128).reshape(-1) for M in b_blocks)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.complex128)

    def _pack_real(
        a_blocks: Sequence[np.ndarray],
        b_blocks: Sequence[np.ndarray],
    ) -> np.ndarray:
        z = _flatten_complex(a_blocks, b_blocks)
        return np.concatenate([z.real, z.imag])

    def _unpack_real(x: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        z = np.asarray(x[:n_complex], dtype=float) + 1j * np.asarray(
            x[n_complex:], dtype=float
        )
        pos = 0
        out_a: List[np.ndarray] = []
        for shape in shapes_alpha:
            size = shape[0] * shape[1]
            out_a.append(z[pos : pos + size].reshape(shape))
            pos += size
        out_b: List[np.ndarray] = []
        for shape in shapes_beta:
            size = shape[0] * shape[1]
            out_b.append(z[pos : pos + size].reshape(shape))
            pos += size
        return out_a, out_b

    rhs = _pack_real(rhs_alpha, rhs_beta)
    if float(np.linalg.norm(rhs)) < 1e-14:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    nbf = int(basis.nbasis)
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    cells_r = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
    )
    cache = _build_j_long_range_cache(basis, system, cells_r, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)

    def _density_response(
        C: np.ndarray,
        n_occ: int,
        v: np.ndarray,
    ) -> np.ndarray:
        if n_occ == 0 or v.size == 0:
            return np.zeros((nbf, nbf), dtype=np.complex128)
        C_occ = C[:, :n_occ]
        C_vir = C[:, n_occ:]
        return C_vir @ v.T @ C_occ.conj().T + C_occ @ v.conj() @ C_vir.conj().T

    def _response_blocks(
        va_list: Sequence[np.ndarray],
        vb_list: Sequence[np.ndarray],
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        dDa_k: List[np.ndarray] = []
        dDb_k: List[np.ndarray] = []
        dDt_k: List[np.ndarray] = []
        for Ca, Cb, va, vb in zip(Ca0, Cb0, va_list, vb_list):
            dDa = _density_response(Ca, n_alpha, np.asarray(va))
            dDb = _density_response(Cb, n_beta, np.asarray(vb))
            dDa_k.append(dDa)
            dDb_k.append(dDb)
            dDt_k.append(dDa + dDb)

        dDt_real = _density_set_from_k_density_matrices(
            system, basis, lattice_opts, kmesh, dDt_k
        )
        dDa_real = _density_set_from_k_density_matrices(
            system, basis, lattice_opts, kmesh, dDa_k
        )
        dDb_real = _density_set_from_k_density_matrices(
            system, basis, lattice_opts, kmesh, dDb_k
        )
        J_sr = build_fock_2e_real_space(
            basis, system, lattice_opts, dDt_real, 0.0, alpha
        )
        K_alpha = build_jk_2e_real_space(basis, system, lattice_opts, dDa_real, 0.0).K
        K_beta = build_jk_2e_real_space(basis, system, lattice_opts, dDb_real, 0.0).K
        rho_hat = compute_rho_hat_from_k_density(dDt_k, kpts, kmesh.weights, cache)
        J_lr = compute_J_long_range_real_space_blocks(
            dDt_real,
            basis,
            system,
            alpha,
            cache=cache,
            rho_hat=rho_hat,
            precision=ewald_precision,
        )
        g_alpha_blocks: List[np.ndarray] = []
        g_beta_blocks: List[np.ndarray] = []
        for c in range(len(S_lat.cells)):
            common = np.asarray(J_sr.blocks[c], dtype=float) + J_lr[c]
            g_alpha_blocks.append(
                common - a_hf * np.asarray(K_alpha.blocks[c], dtype=float)
            )
            g_beta_blocks.append(
                common - a_hf * np.asarray(K_beta.blocks[c], dtype=float)
            )

        out_a: List[np.ndarray] = []
        out_b: List[np.ndarray] = []
        for Ca, Cb, k_arr in zip(Ca0, Cb0, kpts):
            if Ca.shape[1] > n_alpha:
                out_a.append(
                    Ca[:, :n_alpha].conj().T
                    @ _bloch_sum_blocks(g_alpha_blocks, S_lat.cells, k_arr)
                    @ Ca[:, n_alpha:]
                )
            else:
                out_a.append(np.zeros((n_alpha, 0), dtype=np.complex128))
            if Cb.shape[1] > n_beta:
                out_b.append(
                    Cb[:, :n_beta].conj().T
                    @ _bloch_sum_blocks(g_beta_blocks, S_lat.cells, k_arr)
                    @ Cb[:, n_beta:]
                )
            else:
                out_b.append(np.zeros((n_beta, 0), dtype=np.complex128))
        return out_a, out_b

    def _Aop(x: np.ndarray) -> np.ndarray:
        va_list, vb_list = _unpack_real(x)
        Ga, Gb = _response_blocks(va_list, vb_list)
        out_a: List[np.ndarray] = []
        for va, g, eps, nv in zip(va_list, Ga, eps_a, nva):
            if nv > 0:
                ediff = eps[n_alpha:][None, :] - eps[:n_alpha, None]
                out_a.append(ediff * va + g)
            else:
                out_a.append(np.zeros((n_alpha, 0), dtype=np.complex128))
        out_b: List[np.ndarray] = []
        for vb, g, eps, nv in zip(vb_list, Gb, eps_b, nvb):
            if nv > 0:
                ediff = eps[n_beta:][None, :] - eps[:n_beta, None]
                out_b.append(ediff * vb + g)
            else:
                out_b.append(np.zeros((n_beta, 0), dtype=np.complex128))
        return _pack_real(out_a, out_b)

    ndim = 2 * n_complex
    if delta_spin_input is None:
        H = np.zeros((ndim, ndim), dtype=float)
        for col in range(ndim):
            basis_vec = np.zeros(ndim, dtype=float)
            basis_vec[col] = 1.0
            H[:, col] = _Aop(basis_vec)
        H = 0.5 * (H + H.T)
        try:
            z_vec, *_ = np.linalg.lstsq(H, rhs, rcond=1e-10)
        except np.linalg.LinAlgError as exc:
            warnings.warn(
                "compute_bipole_gradient_uhf: multi-k UHF Z-vector solve "
                f"failed ({exc}); skipping the orbital relaxation. Use "
                "compute_bipole_gradient_fd for reliable forces.",
                UserWarning,
                stacklevel=3,
            )
            return np.zeros((n_atoms, 3), dtype=np.float64)
    else:
        def _rotated_coeffs(
            coeffs: Sequence[np.ndarray],
            n_occ: int,
            rotations: Sequence[np.ndarray],
            sign: float,
            eta: float,
        ) -> List[np.ndarray]:
            out = []
            for C, v in zip(coeffs, rotations):
                C_occ = C[:, :n_occ]
                C_vir = C[:, n_occ:]
                occ = C_occ + sign * eta * (C_vir @ v.T)
                vir = C_vir - sign * eta * (C_occ @ v.conj())
                out.append(np.concatenate((occ, vir), axis=1))
            return out

        rotation_step = 1.0e-3

        def _aop_padded(x: np.ndarray) -> np.ndarray:
            va_list, vb_list = _unpack_real(x)
            Ca_plus = _rotated_coeffs(
                Ca0, n_alpha, va_list, +1.0, rotation_step
            )
            Ca_minus = _rotated_coeffs(
                Ca0, n_alpha, va_list, -1.0, rotation_step
            )
            Cb_plus = _rotated_coeffs(
                Cb0, n_beta, vb_list, +1.0, rotation_step
            )
            Cb_minus = _rotated_coeffs(
                Cb0, n_beta, vb_list, -1.0, rotation_step
            )
            Bpa, Bpb = _build_multi_k_bipole_b0_open(
                system,
                basis,
                Ca_plus,
                eps_a,
                Cb_plus,
                eps_b,
                n_alpha,
                n_beta,
                kmesh,
                lattice_opts,
                alpha,
                a_hf,
                exchange_ewald_split=True,
                sr_image_extent_bohr=sr_image_extent_bohr,
                sr_density_cells=sr_density_cells,
                ewald_precision=ewald_precision,
            )
            Bma, Bmb = _build_multi_k_bipole_b0_open(
                system,
                basis,
                Ca_minus,
                eps_a,
                Cb_minus,
                eps_b,
                n_alpha,
                n_beta,
                kmesh,
                lattice_opts,
                alpha,
                a_hf,
                exchange_ewald_split=True,
                sr_image_extent_bohr=sr_image_extent_bohr,
                sr_density_cells=sr_density_cells,
                ewald_precision=ewald_precision,
            )
            return _pack_real(
                [
                    (plus - minus) / (2.0 * rotation_step)
                    for plus, minus in zip(Bpa, Bma)
                ],
                [
                    (plus - minus) / (2.0 * rotation_step)
                    for plus, minus in zip(Bpb, Bmb)
                ],
            )

        H = np.zeros((ndim, ndim), dtype=float)
        for col in range(ndim):
            basis_vec = np.zeros(ndim, dtype=float)
            basis_vec[col] = 1.0
            H[:, col] = _aop_padded(basis_vec)
        try:
            # The unequal M5 domains make the SCF forward map genuinely
            # non-self-adjoint.  The Z-vector therefore solves the transpose
            # response equation B_C.T z = dE/dC, not a symmetrized surrogate.
            z_vec, *_ = np.linalg.lstsq(H.T, rhs, rcond=1.0e-10)
        except np.linalg.LinAlgError as exc:
            raise NotImplementedError(
                "padded multi-k UHF analytic gradient could not solve the "
                f"finite-domain orbital response ({exc}); use "
                "compute_bipole_gradient_fd"
            ) from exc
    za_list, zb_list = _unpack_real(z_vec)

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    h = float(step_bohr)
    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):

            def _disp(sign):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(displaced[a].xyz)
                xyz[d] += sign * h
                displaced[a] = Atom(displaced[a].Z, xyz)
                sd = PeriodicSystem(system.dim, lattice, displaced)
                sd.charge = system.charge
                sd.multiplicity = system.multiplicity
                bd = _recenter_basis_on_periodic_system(basis, sd)
                return _build_multi_k_bipole_b0_open(
                    sd,
                    bd,
                    Ca0,
                    eps_a,
                    Cb0,
                    eps_b,
                    n_alpha,
                    n_beta,
                    kmesh,
                    lattice_opts,
                    alpha,
                    a_hf,
                    exchange_ewald_split=delta_spin_input is not None,
                    sr_image_extent_bohr=sr_image_extent_bohr,
                    sr_density_cells=sr_density_cells,
                    ewald_precision=ewald_precision,
                )

            Bpa, Bpb = _disp(+1.0)
            Bma, Bmb = _disp(-1.0)
            total = 0.0
            for w_k, z_k, bp, bm in zip(weights, za_list, Bpa, Bma):
                dB = (bp - bm) / (2.0 * h)
                contraction = (
                    np.sum(z_k * dB)
                    if delta_spin_input is None
                    else np.vdot(z_k, dB)
                )
                total += w_k * float(np.real(contraction))
            for w_k, z_k, bp, bm in zip(weights, zb_list, Bpb, Bmb):
                dB = (bp - bm) / (2.0 * h)
                contraction = (
                    np.sum(z_k * dB)
                    if delta_spin_input is None
                    else np.vdot(z_k, dB)
                )
                total += w_k * float(np.real(contraction))
            relax[a, d] = -2.0 * total
    return relax


def _corrected_w_gamma_open(
    system: PeriodicSystem,
    basis: BasisSet,
    D_total: LatticeMatrixSet,
    D_alpha: LatticeMatrixSet,
    D_beta: LatticeMatrixSet,
    mo_coeffs_alpha_gamma: np.ndarray,
    n_alpha: int,
    mo_coeffs_beta_gamma: np.ndarray,
    n_beta: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    extra_home_block_alpha: Optional[np.ndarray] = None,
    extra_home_block_beta: Optional[np.ndarray] = None,
    *,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Γ-only open-shell energy-weighted density consistent with the LOCAL
    BIPOLE energy: ``W = W_a + W_b`` with
    ``W_s = C_occ,s.(C_occ,s+ dE/dP_s(0) C_occ,s).C_occ,s+`` (occupation 1
    per spin -- no factor 2).

    Only the exchange is spin-dependent. The spin-independent part of the
    home-cell dE/dP is the BIPOLE energy's *total*-density block

        shared = T0 + V_ne0 + J_SR0 + J^LR0 + 1/2.v_bg.S0 + spheropole0,

    obtained by reusing :func:`_bipole_de_dp_home_block` at ``alpha_hf=0``
    (its -1/2a_HF.K term then vanishes). Per spin we add the full-Coulomb
    exchange derivative ``dE_x/dP_s = -a_HF.K_full[P_s](0)``:

        dE/dP_s(0) = shared - a_HF.K_full[P_s](0).

    ``extra_home_block_alpha`` / ``extra_home_block_beta`` (UKS path):
    per-spin home-cell matrices added to ``dE/dP_s(0)`` -- the ``V_xc,s(0)``
    blocks, so the KS energy-weighted density uses the full KS Fock per
    spin rather than the HF part.

    (For a closed-shell singlet -- n_a=n_b, P_a=P_b=1/2P_total -- this reduces
    *exactly* to ``2.`` the closed-shell ``_corrected_w_gamma_closed``.)
    Returns a real ``(nbf, nbf)`` matrix.
    """
    from ._vibeqc_core import build_fock_2e_real_space

    shared = _bipole_de_dp_home_block(
        system,
        basis,
        D_total,
        lattice_opts,
        float(ewald_alpha),
        0.0,
        ewald_precision=ewald_precision,
    )
    home = _home_cell_index(list(D_total.cells))

    def _minus_alpha_K0(D_spin: LatticeMatrixSet) -> np.ndarray:
        # 2.([J - 1/2a_HF K] - J)(0) = -a_HF.K_full[P_s](0). The full-Coulomb
        # exchange (w=0) matches BIPOLE's K; dE_x/dP_s carries the full a_HF
        # (factor 2 vs the closed-shell -1/2a_HF, since each spin density is
        # occupied once).
        fk = np.asarray(
            build_fock_2e_real_space(
                basis, system, lattice_opts, D_spin, float(alpha_hf), 0.0
            ).blocks[home],
            dtype=float,
        )
        fj = np.asarray(
            build_fock_2e_real_space(
                basis, system, lattice_opts, D_spin, 0.0, 0.0
            ).blocks[home],
            dtype=float,
        )
        return 2.0 * (fk - fj)

    W = np.zeros_like(shared)
    if n_alpha > 0:
        dEdP_a = shared + _minus_alpha_K0(D_alpha)
        if extra_home_block_alpha is not None:
            dEdP_a = dEdP_a + np.asarray(extra_home_block_alpha, dtype=float)
        Ca = np.asarray(mo_coeffs_alpha_gamma)[:, :n_alpha]
        W = W + np.real(Ca @ (Ca.conj().T @ dEdP_a @ Ca) @ Ca.conj().T)
    if n_beta > 0:
        dEdP_b = shared + _minus_alpha_K0(D_beta)
        if extra_home_block_beta is not None:
            dEdP_b = dEdP_b + np.asarray(extra_home_block_beta, dtype=float)
        Cb = np.asarray(mo_coeffs_beta_gamma)[:, :n_beta]
        W = W + np.real(Cb @ (Cb.conj().T @ dEdP_b @ Cb) @ Cb.conj().T)
    return W


def _reconstruct_bipole_fock_gamma_builder(
    system, basis, lattice_opts, ewald_alpha, alpha_hf: float = 1.0,
    *,
    ewald_precision: float = 1e-8,
):
    """Return ``(S(Γ), Hcore(Γ), f2e(P_home, D_k))`` reproducing the BIPOLE
    SCF's Bloch-summed Fock at Γ for a given geometry -- the foundation of the
    Bloch-CPHF Hessian/RHS.

    ``Hcore(Γ) = S_g [T(g)+V_ne(g)]`` (kinetic + Ewald nuclear-attraction).
    ``f2e(P_home, D_k) = S_g [J_SR(g) - 1/2a_HF K(g) + J^LR(g) + v_bg.S(g)]``
    with ``J^LR`` built from ``rho_hat`` of the Γ k-density ``D_k`` (the SCF's
    mixed convention -- NOT ``build_fock_2e_ewald_j_split_gamma``, whose J^LR
    uses the wrong real-space r̂). The returned Bloch-folded matrices are
    Hermitian, matching the SCF driver convention after it symmetrises
    ``F(k)`` before diagonalisation."""
    from ._vibeqc_core import (
        build_fock_2e_real_space,
        build_jk_2e_real_space,
        compute_kinetic_lattice,
    )
    from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_J_long_range_real_space_blocks,
        compute_rho_hat_from_k_density,
    )
    from .pbc_bipole import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _crystal_ewald_options,
    )

    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    nbf = int(basis.nbasis)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    S = compute_overlap_lattice(basis, system, lattice_opts)
    cells = list(S.cells)
    n_cells = len(cells)
    home = _home_cell_index(cells)
    rc = np.array([np.asarray(c.r_cart, dtype=float) for c in cells], dtype=float)
    n_elec = int(system.n_electrons())
    v_bg = -np.pi * float(n_elec) / (alpha * alpha * V)

    def _hermitian(M: np.ndarray) -> np.ndarray:
        A = np.asarray(M)
        return 0.5 * (A + A.conj().T)

    S_g = _hermitian(sum(np.asarray(S.blocks[i], dtype=float) for i in range(n_cells)))
    Tg = sum(
        np.asarray(
            compute_kinetic_lattice(basis, system, lattice_opts).blocks[i], dtype=float
        )
        for i in range(n_cells)
    )
    ew = _crystal_ewald_options(
        lattice_opts, alpha_bohr_inv=alpha, tolerance=ewald_precision, recip_cutoff_bohr_inv=K_max
    )
    Vne, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis, system, lattice_opts, ew, S, precision=ewald_precision, K_max=K_max
    )
    Vne_g = sum(np.asarray(Vne.blocks[i], dtype=float) for i in range(n_cells))
    Hcore_g = _hermitian(Tg + Vne_g)
    cache = _build_j_long_range_cache(basis, system, rc, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)

    def _ls(M):
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            s.set_block(c, M if c == home else np.zeros((nbf, nbf)))
        return s

    def j_build(P_home, D_k):
        """Bloch-summed Coulomb ``S_g [J_SR(g) + J^LR(g) + v_bg.S(g)]`` from the
        TOTAL density (``P_home`` real-space home block, ``D_k`` Γ k-density for
        the J^LR rho_hat)."""
        jsr = build_fock_2e_real_space(
            basis, system, lattice_opts, _ls(P_home), 0.0, alpha
        )
        rho = compute_rho_hat_from_k_density([D_k], [np.zeros(3)], [1.0], cache)
        flr = compute_J_long_range_real_space_blocks(
            _ls(P_home), basis, system, alpha, precision=ewald_precision, cache=cache, rho_hat=rho
        )
        return _hermitian(
            sum(
                np.asarray(jsr.blocks[i], dtype=float)
                + np.asarray(flr[i], dtype=float)
                + v_bg * np.asarray(S.blocks[i], dtype=float)
                for i in range(n_cells)
            )
        )

    def k_build(P_spin):
        """Bloch-summed full-Coulomb exchange ``S_g K[P_spin](g)``."""
        kk = build_jk_2e_real_space(basis, system, lattice_opts, _ls(P_spin), 0.0).K
        return _hermitian(
            sum(np.asarray(kk.blocks[i], dtype=float) for i in range(n_cells))
        )

    def f2e(P_home, D_k):
        # Closed-shell: J(total) - 1/2a_HF K(total). Numerically identical to the
        # original single-pass build for RHF and pure/hybrid KS when a_HF is
        # set to the functional's exact-exchange fraction.
        return j_build(P_home, D_k) - 0.5 * a_hf * k_build(P_home)

    return S_g, Hcore_g, f2e, j_build, k_build


def _bloch_cphf_rhs_analytic(
    system: PeriodicSystem,
    basis: BasisSet,
    P_home: np.ndarray,
    z: np.ndarray,
    C_occ: np.ndarray,
    C_vir: np.ndarray,
    eps: np.ndarray,
    n_occ: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    builders: tuple,
    *,
    mode: str = "hybrid",
    step_bohr: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Analytic / hybrid Bloch-CPHF right-hand-side gradient ``-4.S z.dB0/dR``
    -- the fast replacement for the 6N-full-Fock-build semi-numerical RHS.

    The Bloch orbital gradient is differentiated **in the Bloch metric**: the
    SCF diagonalises ``F(Γ)=S_g F(g)``, so every kernel must contract the
    response density ``Pz`` *broadcast into every cell* (not home-only) to pick
    up the full ``S_g dOp(g)/dR`` rather than just the ``g=0`` block. With that
    fix the skeleton (``term1+3``) and the J^LR renormalisation response are
    exact analytic kernels.

    ``dB0/dR`` splits as

      * **term1+3** (skeleton + ``dS``): kinetic + Ewald V_ne + screened J_SR +
        full-Coulomb -1/2a_HF K + Fock-convention J^LR (both-Bloch r̂) +
        electronic ``v_bg.S`` + the energy-weighted ``Wsym`` overlap term -- all
        analytic, ``Pz``/``Wsym`` broadcast. Validated ~1e-4 vs FD.
      * **term2** (renormalisation response of the fixed-C occupied block):
          - **J^LR part** -- self-adjoint, exact analytic
            (``-4.og((Cocc.(Cocc+.G_Jlr[Pz].Cocc).Cocc+)_bcast)``).
          - **local (J_SR-1/2K) part** -- the reconstruction's lattice cutoff
            breaks the 4-index symmetry, so the self-adjoint shortcut is
            ~2e-3 off. Two modes:
              * ``mode="analytic"`` (A): self-adjoint shortcut for the *whole*
                ``G[Pz]`` -- fully analytic, no FD, ~6x faster than semi-num,
                accurate to ~2e-3 vs FD.
              * ``mode="hybrid"`` (B, default): the local renorm is taken
                exactly via 6N cheap ``J_SR``+``K`` builds of
                ``G_local[dD_eff/dR_c]`` (``dD_eff/dR`` from a 6N overlap-only
                FD of ``S(Γ)``); J^LR/V_ne stay analytic. ~2x faster than
                semi-num, matches it to ~3e-5.

    Returns ``(n_atoms, 3)``."""
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from ._vibeqc_core import build_fock_2e_real_space
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import _build_j_long_range_cache

    S_g, _Hcore_g, f2e0, _j_build, k_build = builders
    a_hf = float(alpha_hf)
    n_atoms = len(system.unit_cell)
    nbf = int(basis.nbasis)
    alpha = float(ewald_alpha)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    n_elec = int(system.n_electrons())
    v_bg = -np.pi * float(n_elec) / (alpha * alpha * V)
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))

    # MO-derived response densities (home blocks).
    Pz = C_vir @ z.T @ C_occ.T
    Pz = Pz + Pz.T
    Wsym = C_vir @ (z * eps[:n_occ][:, None]).T @ C_occ.T
    Wsym = Wsym + Wsym.T

    # Cell template + home index from the overlap lattice.
    S_set = compute_overlap_lattice(basis, system, lattice_opts)
    cells = list(S_set.cells)
    n_cells = len(cells)
    home = _home_cell_index(cells)
    rc = np.array([np.asarray(c.r_cart, dtype=float) for c in cells], dtype=float)

    def _lsb(M):  # M broadcast into EVERY cell (the Bloch metric)
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            s.set_block(c, M)
        return s

    def _lsh(M):  # M in the home cell only
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            s.set_block(c, M if c == home else np.zeros((nbf, nbf)))
        return s

    def _kin(D_set):
        return np.asarray(
            kinetic_lattice_gradient_contribution(basis, system, D_set, lattice_opts)
        )

    def _og(D_set):
        return np.asarray(
            overlap_lattice_gradient_contribution(basis, system, D_set, lattice_opts)
        )

    def _eri(D_set, ahf, j_scale, omega):
        return np.asarray(
            eri_lattice_gradient_contribution(
                basis,
                system,
                D_set,
                lattice_opts,
                ahf,
                j_scale,
                omega,
                exchange_energy_convention=(
                    ahf != 0.0 and j_scale == 0.0 and omega == 0.0
                ),
            )
        )

    def _eri_cross(ahf, j_scale, omega):
        # bilinear cross Tr[Pz_bloch . d(2e[P_home])/dR]: P home, Pz broadcast.
        comb = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            comb.set_block(c, (P_home + Pz) if c == home else Pz)
        return (
            _eri(comb, ahf, j_scale, omega)
            - _eri(_lsh(P_home), ahf, j_scale, omega)
            - _eri(_lsb(Pz), ahf, j_scale, omega)
        )

    # --- Fock-convention J^LR cross (both-Bloch r̂[P], r̂[Pz]) ---
    cache = _build_j_long_range_cache(basis, system, rc, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
    K_vec = cache.K_vectors
    kernel = cache.kernel
    ft = cache.ft_per_cell  # (n_g,nbf,nbf,n_K)
    ft_sum = ft.sum(axis=0)  # (nbf,nbf,n_K), Bloch FT
    grad_bra, grad_ket = _pair_ft_gradient_at_cells(basis, K_vec, rc, lattice_opts=lattice_opts)
    corr = _libint_ylm_correction_per_ao(basis)
    cc = corr[:, None] * corr[None, :]
    grad_bra = grad_bra * cc[None, :, :, None, None]
    grad_ket = grad_ket * cc[None, :, :, None, None]
    ao2atom = _ao_to_atom_map(system, basis)

    def _rho_bloch(M):  # r̂_b[M] = S_g M.FT(g) = M.FTsum  (M broadcast)
        return np.einsum("mn,mnk->k", M, ft_sum)

    def _drho_bloch(M):  # dr̂_b[M]/dR scattered onto atoms (M broadcast)
        bra = np.einsum("mn,gmnxk->mxk", M, grad_bra)
        ket = np.einsum("mn,gmnxk->nxk", M, grad_ket)
        out = np.zeros((n_atoms, 3, K_vec.shape[0]), dtype=np.complex128)
        np.add.at(out, ao2atom, bra)
        np.add.at(out, ao2atom, ket)
        return out

    rb_P, rb_Pz = _rho_bloch(P_home), _rho_bloch(Pz)
    drb_P, drb_Pz = _drho_bloch(P_home), _drho_bloch(Pz)
    cross_Jlr = np.zeros((n_atoms, 3), dtype=np.float64)
    for Cc in range(n_atoms):
        term = drb_P[Cc] * np.conj(rb_Pz)[None, :] + rb_P[None, :] * np.conj(drb_Pz[Cc])
        cross_Jlr[Cc] = np.real((kernel[None, :] * term).sum(axis=1))

    onee = _kin(_lsb(Pz)) + _v_ne_ewald_gradient(
        system, basis, _lsb(Pz), lattice_opts, alpha,
        precision=ewald_precision,
    )
    term13 = (
        -2.0 * onee
        + 2.0 * v_bg * _og(_lsb(Pz))
        - 2.0
        * (
            _eri_cross(a_hf, 0.0, 0.0)  # full-Coulomb -1/2a_HF K
            + _eri_cross(0.0, 1.0, alpha)  # screened J_SR
            + cross_Jlr
        )  # Fock J^LR
        - 2.0 * _og(_lsb(Wsym))
    )

    # --- term2: renormalisation response ---
    f2e_zero = f2e0(np.zeros((nbf, nbf)), np.zeros((nbf, nbf)))
    # G[Pz] (J_SR + J^LR - 1/2a_HF K).
    G_Pz = 0.5 * (f2e0(2.0 * Pz, 2.0 * Pz) - f2e_zero)

    def _W_bcast(W_occ):
        return _lsb(C_occ @ W_occ @ C_occ.T)

    if mode == "analytic":
        # self-adjoint shortcut for the whole G[Pz] (fast; ~2e-3 local error)
        term2 = -4.0 * _og(_W_bcast(C_occ.T @ G_Pz @ C_occ))
    elif mode == "hybrid":
        # local J_SR-1/2a_HF K renorm exact (6N cheap builds);
        # J^LR renorm analytic.
        def _G_local(M_home):  # bloch(J_SR[M] - 1/2a_HF K[M]) from home builders
            jsr = build_fock_2e_real_space(
                basis, system, lattice_opts, _lsh(M_home), 0.0, alpha
            )
            jsr_g = sum(np.asarray(jsr.blocks[i], dtype=float) for i in range(n_cells))
            return jsr_g - 0.5 * a_hf * k_build(M_home)

        G_local_Pz = _G_local(Pz)
        G_Jlr_Pz = G_Pz - G_local_Pz
        term2 = -4.0 * _og(_W_bcast(C_occ.T @ G_Jlr_Pz @ C_occ))  # J^LR renorm
        # local renorm: -4.S z.Cocc+.G_local[dD_eff/dR_c].Cvir, dD_eff via dS(Γ).
        lattice = np.asarray(system.lattice, dtype=float)
        atoms = list(system.unit_cell)
        h = float(step_bohr)

        def _S_gamma(disp_atoms):
            sd = PeriodicSystem(3, lattice, disp_atoms)
            # Propagate charge + multiplicity so a charged closed shell isn't
            # rejected by unit_cell_molecule()'s n_e/mult check (see _D_eff_spin
            # in the open-shell path). Neither affects the overlap numerics.
            sd.charge = system.charge
            sd.multiplicity = system.multiplicity
            bd = _recenter_basis_on_periodic_system(basis, sd)
            sset = compute_overlap_lattice(bd, sd, lattice_opts)
            return sum(
                np.asarray(sset.blocks[i], dtype=float)
                for i in range(len(list(sset.cells)))
            )

        def _D_eff(S_gamma):
            M = C_occ.T @ S_gamma @ C_occ
            return 2.0 * C_occ @ np.linalg.inv(M) @ C_occ.T

        term2_local = np.zeros((n_atoms, 3), dtype=np.float64)
        for a in range(n_atoms):
            for d in range(3):
                ap = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xp = list(ap[a].xyz)
                xp[d] += h
                ap[a] = Atom(ap[a].Z, xp)
                am = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xm = list(am[a].xyz)
                xm[d] -= h
                am[a] = Atom(am[a].Z, xm)
                dDeff = (_D_eff(_S_gamma(ap)) - _D_eff(_S_gamma(am))) / (2.0 * h)
                resp = C_occ.T @ _G_local(dDeff) @ C_vir
                term2_local[a, d] = -4.0 * float(np.sum(z * resp))
        term2 = term2 + term2_local
    else:
        raise ValueError(
            f"_bloch_cphf_rhs_analytic: unknown mode {mode!r} "
            "(expected 'analytic' or 'hybrid')"
        )

    return term13 + term2


def _bloch_cphf_relaxation(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    mo_coeffs_gamma: np.ndarray,
    mo_energies_gamma: np.ndarray,
    n_occ: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    *,
    cphf_rhs: str = "hybrid",
    step_bohr: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Full orbital-relaxation (Bloch CPHF) gradient for the Γ-only local-energy
    BIPOLE gauge -- the **general-crystal** replacement for the spheropole-only
    Z-vector.

    The BIPOLE Γ SCF diagonalises the Bloch sum ``F(Γ)=S_g F(g)`` while the
    energy is the LOCAL contraction ``Tr[D(0).H(0)]``; for an asymmetric
    multi-cell crystal ``F(0) != F(Γ)`` so the converged density is not
    stationary for the local energy and ``occ-virt(dE_local/dP(0)) != 0`` (both
    the SCF-Fock mismatch AND the post-SCF spheropole). The missing orbital
    relaxation ``S_ai (dE_local/dth_ai)(dth_ai/dR)`` is recovered by a Z-vector:

      * **Hessian** ``A(v) = (e_a-e_i).v + Cocc.G(dD(v)).Cvir`` with
        ``e = mo_energies`` (Bloch F(Γ) eigenvalues -- required; the home-block
        ``diag(CᵀF_scf C)`` is unordered for multi-cell) and ``G`` the
        Bloch-summed BIPOLE 2e Fock response, ``dD(v)=2(Cvir.vᵀ.Coccᵀ + h.c.)``.
        Positive-definite for a stable SCF.
      * **RHS** ``b = Cocc.dE_local/dP(0).Cvir`` (the full home-block
        ``_bipole_de_dp_home_block`` occ-virt). Solve ``A.z = b`` (PCG).
      * **Gradient** ``-4.S_ia z_ia.dB0_ia/dR`` where the Bloch orbital
        gradient ``B0 = Cocc.F(Γ).Cvir - (Cocc.S(Γ).Cvir).e`` at the
        **renormalised fixed-C density** ``2Cocc(CoccᵀS(Γ)Cocc)⁻¹Coccᵀ`` is
        differentiated by ``_bloch_cphf_rhs_analytic`` (``cphf_rhs="hybrid"``
        default: analytic skeleton + analytic J^LR renorm + 6N cheap-build
        local renorm, ~3e-5 vs FD; ``"analytic"``: fully analytic, ~2e-3;
        ``"seminumeric"``: the original 6N-full-Fock-build FD reference).
        Validated on asymmetric multi-cell BeH₂ (a=8, 7 cells; base 2.8e-2 ->
        hybrid 4.5e-5 / analytic 2.2e-3).

    Returns ``(n_atoms, 3)``; zero when there are no virtuals or not 3D-Ewald.
    Reduces to the spheropole-only relaxation on symmetric / 1-cell systems
    (where ``occ-virt(F_scf)=0``)."""
    from .cphf import CPHFConvergenceError, _pcg

    n_atoms = len(system.unit_cell)
    if system.dim != 3:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    C = np.asarray(mo_coeffs_gamma)
    C = np.real(C) if np.iscomplexobj(C) else C
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    if C_vir.shape[1] == 0:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    eps = np.real(np.asarray(mo_energies_gamma))
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    nbf = int(basis.nbasis)

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)

    # --- Bloch Hessian action (home geometry) ---
    builders = _reconstruct_bipole_fock_gamma_builder(
        system, basis, lattice_opts, alpha, a_hf,
        ewald_precision=ewald_precision,
    )
    f2e0 = builders[2]
    f2e_zero = f2e0(np.zeros((nbf, nbf)), np.zeros((nbf, nbf)))

    def _Gresp(vmat):
        T = C_vir @ vmat.T
        dD = 2.0 * (T @ C_occ.T)
        dD = dD + dD.T
        return f2e0(dD, dD) - f2e_zero

    ediff = eps[n_occ:][None, :] - eps[:n_occ][:, None]

    def _Aop(vflat):
        vm = vflat.reshape(C_occ.shape[1], C_vir.shape[1])
        return (ediff * vm + (C_occ.T @ _Gresp(vm) @ C_vir)).ravel()

    b = (
        C_occ.T
        @ _bipole_de_dp_home_block(system, basis, D_real, lattice_opts, alpha, a_hf, ewald_precision=ewald_precision)
        @ C_vir
    ).ravel()
    try:
        zflat, _, _ = _pcg(
            _Aop, b, lambda x: (x.reshape(ediff.shape) / ediff).ravel(), 1e-9, 400
        )
    except CPHFConvergenceError as exc:
        warnings.warn(
            "compute_bipole_gradient: Bloch-CPHF Z-vector solve failed "
            f"({exc} -- likely an SCF instability); skipping the orbital "
            "relaxation. Use compute_bipole_gradient_fd for reliable forces.",
            UserWarning,
            stacklevel=3,
        )
        return np.zeros((n_atoms, 3), dtype=np.float64)
    z = zflat.reshape(ediff.shape)

    # --- dB0/dR (the CPHF right-hand side) ---
    if cphf_rhs in ("analytic", "hybrid"):
        P_home = np.asarray(
            D_real.blocks[_home_cell_index(list(D_real.cells))], dtype=float
        )
        return _bloch_cphf_rhs_analytic(
            system,
            basis,
            P_home,
            z,
            C_occ,
            C_vir,
            eps,
            n_occ,
            lattice_opts,
            alpha,
            a_hf,
            builders,
            mode=cphf_rhs,
            step_bohr=step_bohr,
            ewald_precision=ewald_precision,
        )
    if cphf_rhs != "seminumeric":
        raise ValueError(
            f"_bloch_cphf_relaxation: unknown cphf_rhs {cphf_rhs!r} "
            "(expected 'hybrid', 'analytic', or 'seminumeric')"
        )

    # --- semi-numerical dB0/dR (6N full Fock builds; reference / fallback) ---
    def _B0(disp_atoms):
        sd = PeriodicSystem(3, lattice, disp_atoms)
        # Propagate charge + multiplicity (see _D_eff_spin): a charged closed
        # shell would otherwise fail unit_cell_molecule()'s n_e/mult check.
        sd.charge = system.charge
        sd.multiplicity = system.multiplicity
        bd = _recenter_basis_on_periodic_system(basis, sd)
        S_g, Hcore_g, f2e, _, _ = _reconstruct_bipole_fock_gamma_builder(
            sd, bd, lattice_opts, alpha, a_hf,
            ewald_precision=ewald_precision,
        )
        # renormalise the fixed-C occupied block to stay S(Γ)-orthonormal
        M = C_occ.T @ S_g @ C_occ
        D_eff = 2.0 * C_occ @ np.linalg.inv(M) @ C_occ.T
        F = Hcore_g + f2e(D_eff, D_eff)
        return (C_occ.T @ F @ C_vir) - (C_occ.T @ S_g @ C_vir) * eps[:n_occ][:, None]

    h = float(step_bohr)
    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            ap = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xp = list(ap[a].xyz)
            xp[d] += h
            ap[a] = Atom(ap[a].Z, xp)
            am = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xm = list(am[a].xyz)
            xm[d] -= h
            am[a] = Atom(am[a].Z, xm)
            dB0 = (_B0(ap) - _B0(am)) / (2.0 * h)
            relax[a, d] = -4.0 * float(np.sum(z * dB0))
    return relax


def _bloch_cphf_relaxation_ks_closed(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    mo_coeffs_gamma: np.ndarray,
    mo_energies_gamma: np.ndarray,
    n_occ: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    functional_name: str,
    grid_options,
    use_periodic_becke: bool,
    becke_image_radius_bohr: float,
    *,
    step_bohr: float = 1e-4,
    fxc_step: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Closed-shell Gamma KS Bloch-CPHF orbital-relaxation force.

    KS differs from the RHF helper in two important ways: the Hessian needs the
    XC kernel response, and the finite lattice/gauge convention makes the
    orbital Hessian visibly non-self-adjoint.  The Z-vector therefore solves
    ``A.T z = b`` with a dense matrix assembled from finite-difference
    ``V_xc[P ± h*dP]`` responses.  The right-hand side uses the local-energy
    home-cell derivative ``dE/dP(0) + V_xc(0)``; ``dB0/dR`` is differentiated
    semi-numerically with the same moving periodic Becke grid as the SCF.
    """
    n_atoms = len(system.unit_cell)
    if system.dim != 3:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    C = np.asarray(mo_coeffs_gamma)
    C = np.real(C) if np.iscomplexobj(C) else C
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    if C_vir.shape[1] == 0:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    from ._vibeqc_core import Functional, build_xc_periodic

    eps = np.real(np.asarray(mo_energies_gamma))
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    nbf = int(basis.nbasis)
    home = _home_cell_index(list(D_real.cells))
    P_home = np.asarray(D_real.blocks[home], dtype=float)
    func = Functional(functional_name, 1)

    builders = _reconstruct_bipole_fock_gamma_builder(
        system, basis, lattice_opts, alpha, a_hf,
        ewald_precision=ewald_precision,
    )
    f2e = builders[2]
    f2e_zero = f2e(np.zeros((nbf, nbf)), np.zeros((nbf, nbf)))
    grid = _build_ks_grid(
        system, grid_options, use_periodic_becke, becke_image_radius_bohr
    )

    def _density_home_set(sys, bas, P):
        s = compute_overlap_lattice(bas, sys, lattice_opts)
        hidx = _home_cell_index(list(s.cells))
        zero = np.zeros_like(P)
        for c in range(len(s.cells)):
            s.set_block(c, P if c == hidx else zero)
        return s

    def _vxc_blocks(sys, bas, grd, P):
        xc = build_xc_periodic(
            bas,
            sys,
            grd,
            func,
            _density_home_set(sys, bas, P),
            lattice_opts,
        )
        home_idx = _home_cell_index(list(xc.V_xc.cells))
        V_home = np.asarray(xc.V_xc.blocks[home_idx], dtype=float)
        V_bloch = sum(np.asarray(block, dtype=float) for block in xc.V_xc.blocks)
        V_bloch = 0.5 * (V_bloch + V_bloch.T)
        return V_home, V_bloch

    def _dD_from_rotation(vmat):
        T = C_vir @ vmat.T
        dD = 2.0 * (T @ C_occ.T)
        return dD + dD.T

    ediff = eps[n_occ:][None, :] - eps[:n_occ][:, None]
    ndim = C_occ.shape[1] * C_vir.shape[1]
    A = np.zeros((ndim, ndim), dtype=np.float64)
    hxc = float(fxc_step)
    for col in range(ndim):
        e = np.zeros(ndim, dtype=np.float64)
        e[col] = 1.0
        v = e.reshape(C_occ.shape[1], C_vir.shape[1])
        dD = _dD_from_rotation(v)
        G2e = f2e(dD, dD) - f2e_zero
        _, Vp = _vxc_blocks(system, basis, grid, P_home + hxc * dD)
        _, Vm = _vxc_blocks(system, basis, grid, P_home - hxc * dD)
        Gxc = (Vp - Vm) / (2.0 * hxc)
        A[:, col] = (ediff * v + C_occ.T @ (G2e + Gxc) @ C_vir).ravel()

    Vxc_home, _ = _vxc_blocks(system, basis, grid, P_home)
    rhs = (
        C_occ.T
        @ (
            _bipole_de_dp_home_block(system, basis, D_real, lattice_opts, alpha, a_hf, ewald_precision=ewald_precision)
            + Vxc_home
        )
        @ C_vir
    ).ravel()
    try:
        zflat, *_ = np.linalg.lstsq(A.T, rhs, rcond=1e-10)
    except np.linalg.LinAlgError as exc:
        warnings.warn(
            "compute_bipole_gradient_rks: KS Bloch-CPHF solve failed "
            f"({exc}); skipping the orbital relaxation. Use "
            "compute_bipole_gradient_fd for reliable forces.",
            UserWarning,
            stacklevel=3,
        )
        return np.zeros((n_atoms, 3), dtype=np.float64)
    z = zflat.reshape(C_occ.shape[1], C_vir.shape[1])

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    h = float(step_bohr)

    def _B0(disp_atoms):
        sd = PeriodicSystem(3, lattice, disp_atoms)
        sd.charge = system.charge
        sd.multiplicity = system.multiplicity
        bd = _recenter_basis_on_periodic_system(basis, sd)
        S_g, Hcore_g, f2e_d, _, _ = _reconstruct_bipole_fock_gamma_builder(
            sd, bd, lattice_opts, alpha, a_hf,
            ewald_precision=ewald_precision,
        )
        M = C_occ.T @ S_g @ C_occ
        D_eff = 2.0 * C_occ @ np.linalg.inv(M) @ C_occ.T
        gd = _build_ks_grid(
            sd, grid_options, use_periodic_becke, becke_image_radius_bohr
        )
        _, Vxc_g = _vxc_blocks(sd, bd, gd, D_eff)
        F = Hcore_g + f2e_d(D_eff, D_eff) + Vxc_g
        return (C_occ.T @ F @ C_vir) - (C_occ.T @ S_g @ C_vir) * eps[:n_occ][:, None]

    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            ap = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xp = list(ap[a].xyz)
            xp[d] += h
            ap[a] = Atom(ap[a].Z, xp)
            am = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xm = list(am[a].xyz)
            xm[d] -= h
            am[a] = Atom(am[a].Z, xm)
            dB0 = (_B0(ap) - _B0(am)) / (2.0 * h)
            relax[a, d] = -4.0 * float(np.sum(z * dB0))
    return relax


def _bloch_cphf_relaxation_ks_open(
    system: PeriodicSystem,
    basis: BasisSet,
    D_total: LatticeMatrixSet,
    D_alpha: LatticeMatrixSet,
    D_beta: LatticeMatrixSet,
    mo_coeffs_alpha: np.ndarray,
    mo_energies_alpha: np.ndarray,
    n_alpha: int,
    mo_coeffs_beta: np.ndarray,
    mo_energies_beta: np.ndarray,
    n_beta: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    functional_name: str,
    grid_options,
    use_periodic_becke: bool,
    becke_image_radius_bohr: float,
    *,
    step_bohr: float = 1e-4,
    fxc_step: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Open-shell Gamma UKS Bloch-CPHF orbital-relaxation force."""
    n_atoms = len(system.unit_cell)
    if system.dim != 3:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    Ca = np.asarray(mo_coeffs_alpha)
    Ca = np.real(Ca) if np.iscomplexobj(Ca) else Ca
    Cb = np.asarray(mo_coeffs_beta)
    Cb = np.real(Cb) if np.iscomplexobj(Cb) else Cb
    Caocc = Ca[:, :n_alpha]
    Cavir = Ca[:, n_alpha:]
    Cbocc = Cb[:, :n_beta]
    Cbvir = Cb[:, n_beta:]
    nva = Cavir.shape[1]
    nvb = Cbvir.shape[1]
    if nva == 0 and nvb == 0:
        return np.zeros((n_atoms, 3), dtype=np.float64)

    from ._vibeqc_core import (
        Functional,
        build_fock_2e_real_space,
        build_xc_periodic_uks,
    )

    eA = np.real(np.asarray(mo_energies_alpha))
    eB = np.real(np.asarray(mo_energies_beta))
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    nbf = int(basis.nbasis)
    home = _home_cell_index(list(D_total.cells))
    Pa = np.asarray(D_alpha.blocks[home], dtype=float)
    Pb = np.asarray(D_beta.blocks[home], dtype=float)
    func = Functional(functional_name, 2)

    _S_g, _Hcore_g, _f2e, j_build, k_build = _reconstruct_bipole_fock_gamma_builder(
        system, basis, lattice_opts, alpha, a_hf,
        ewald_precision=ewald_precision,
    )
    j_zero = j_build(np.zeros((nbf, nbf)), np.zeros((nbf, nbf)))
    k_zero = k_build(np.zeros((nbf, nbf)))
    grid = _build_ks_grid(
        system, grid_options, use_periodic_becke, becke_image_radius_bohr
    )

    def _density_home_set(sys, bas, P):
        s = compute_overlap_lattice(bas, sys, lattice_opts)
        hidx = _home_cell_index(list(s.cells))
        zero = np.zeros_like(P)
        for c in range(len(s.cells)):
            s.set_block(c, P if c == hidx else zero)
        return s

    def _vxc_blocks(sys, bas, grd, Pa_home, Pb_home):
        xc = build_xc_periodic_uks(
            bas,
            sys,
            grd,
            func,
            _density_home_set(sys, bas, Pa_home),
            _density_home_set(sys, bas, Pb_home),
            lattice_opts,
        )
        home_idx = _home_cell_index(list(xc.V_alpha.cells))
        Va_home = np.asarray(xc.V_alpha.blocks[home_idx], dtype=float)
        Vb_home = np.asarray(xc.V_beta.blocks[home_idx], dtype=float)
        Va_bloch = sum(np.asarray(block, dtype=float) for block in xc.V_alpha.blocks)
        Vb_bloch = sum(np.asarray(block, dtype=float) for block in xc.V_beta.blocks)
        Va_bloch = 0.5 * (Va_bloch + Va_bloch.T)
        Vb_bloch = 0.5 * (Vb_bloch + Vb_bloch.T)
        return Va_home, Vb_home, Va_bloch, Vb_bloch

    def _dD_from_rotation(vmat, Cocc, Cvir):
        if vmat.size == 0:
            return np.zeros((nbf, nbf))
        T = Cvir @ vmat.T
        dD = T @ Cocc.T
        return dD + dD.T

    eda = (
        eA[n_alpha:][None, :] - eA[:n_alpha][:, None] if nva else np.zeros((n_alpha, 0))
    )
    edb = eB[n_beta:][None, :] - eB[:n_beta][:, None] if nvb else np.zeros((n_beta, 0))
    na_dim = n_alpha * nva
    nb_dim = n_beta * nvb
    ndim = na_dim + nb_dim
    A = np.zeros((ndim, ndim), dtype=np.float64)
    hxc = float(fxc_step)

    def _pack(ma, mb):
        out = []
        if nva:
            out.append(ma.ravel())
        if nvb:
            out.append(mb.ravel())
        return np.concatenate(out) if out else np.zeros(0)

    for col in range(ndim):
        e = np.zeros(ndim, dtype=np.float64)
        e[col] = 1.0
        va = e[:na_dim].reshape(n_alpha, nva) if nva else np.zeros((n_alpha, 0))
        vb = e[na_dim:].reshape(n_beta, nvb) if nvb else np.zeros((n_beta, 0))
        dDa = _dD_from_rotation(va, Caocc, Cavir)
        dDb = _dD_from_rotation(vb, Cbocc, Cbvir)
        dDt = dDa + dDb
        GJ = j_build(dDt, dDt) - j_zero
        GKa = k_build(dDa) - k_zero
        GKb = k_build(dDb) - k_zero
        _, _, Vap, Vbp = _vxc_blocks(
            system, basis, grid, Pa + hxc * dDa, Pb + hxc * dDb
        )
        _, _, Vam, Vbm = _vxc_blocks(
            system, basis, grid, Pa - hxc * dDa, Pb - hxc * dDb
        )
        Gxca = (Vap - Vam) / (2.0 * hxc)
        Gxcb = (Vbp - Vbm) / (2.0 * hxc)
        out_a = (
            eda * va + Caocc.T @ (GJ - a_hf * GKa + Gxca) @ Cavir
            if nva
            else np.zeros((n_alpha, 0))
        )
        out_b = (
            edb * vb + Cbocc.T @ (GJ - a_hf * GKb + Gxcb) @ Cbvir
            if nvb
            else np.zeros((n_beta, 0))
        )
        A[:, col] = _pack(out_a, out_b)

    shared = _bipole_de_dp_home_block(system, basis, D_total, lattice_opts, alpha, 0.0, ewald_precision=ewald_precision)

    def _ls0(M):
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(len(s.cells)):
            s.set_block(c, M if c == home else np.zeros((nbf, nbf)))
        return s

    def _minus_alpha_K0(P_spin):
        fk = np.asarray(
            build_fock_2e_real_space(
                basis, system, lattice_opts, _ls0(P_spin), a_hf, 0.0
            ).blocks[home],
            dtype=float,
        )
        fj = np.asarray(
            build_fock_2e_real_space(
                basis, system, lattice_opts, _ls0(P_spin), 0.0, 0.0
            ).blocks[home],
            dtype=float,
        )
        return 2.0 * (fk - fj)

    Vxa_home, Vxb_home, _, _ = _vxc_blocks(system, basis, grid, Pa, Pb)
    Ma = shared + _minus_alpha_K0(Pa) + Vxa_home
    Mb = shared + _minus_alpha_K0(Pb) + Vxb_home
    rhs_a = (Caocc.T @ Ma @ Cavir).ravel() if nva else np.zeros(0)
    rhs_b = (Cbocc.T @ Mb @ Cbvir).ravel() if nvb else np.zeros(0)
    rhs = np.concatenate([rhs_a, rhs_b])
    try:
        zflat, *_ = np.linalg.lstsq(A.T, rhs, rcond=1e-10)
    except np.linalg.LinAlgError as exc:
        warnings.warn(
            "compute_bipole_gradient_uks: KS Bloch-CPHF solve failed "
            f"({exc}); skipping the orbital relaxation. Use "
            "compute_bipole_gradient_fd for reliable forces.",
            UserWarning,
            stacklevel=3,
        )
        return np.zeros((n_atoms, 3), dtype=np.float64)
    za = zflat[:na_dim].reshape(n_alpha, nva) if nva else np.zeros((n_alpha, 0))
    zb = zflat[na_dim:].reshape(n_beta, nvb) if nvb else np.zeros((n_beta, 0))

    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    h = float(step_bohr)

    def _B0(disp_atoms):
        sd = PeriodicSystem(system.dim, lattice, disp_atoms)
        sd.charge = system.charge
        sd.multiplicity = system.multiplicity
        bd = _recenter_basis_on_periodic_system(basis, sd)
        S_d, Hcore_d, _f2e_d, jd, kd = _reconstruct_bipole_fock_gamma_builder(
            sd, bd, lattice_opts, alpha, a_hf,
            ewald_precision=ewald_precision,
        )

        def _D_eff(Cocc):
            if Cocc.shape[1] == 0:
                return np.zeros((nbf, nbf))
            M = Cocc.T @ S_d @ Cocc
            return Cocc @ np.linalg.inv(M) @ Cocc.T

        Da_eff = _D_eff(Caocc)
        Db_eff = _D_eff(Cbocc)
        gd = _build_ks_grid(
            sd, grid_options, use_periodic_becke, becke_image_radius_bohr
        )
        _, _, Vxa_d, Vxb_d = _vxc_blocks(sd, bd, gd, Da_eff, Db_eff)
        Jd = jd(Da_eff + Db_eff, Da_eff + Db_eff)
        Fa = Hcore_d + Jd - a_hf * kd(Da_eff) + Vxa_d
        Fb = Hcore_d + Jd - a_hf * kd(Db_eff) + Vxb_d
        Ba = (
            (Caocc.T @ Fa @ Cavir) - (Caocc.T @ S_d @ Cavir) * eA[:n_alpha, None]
            if nva
            else np.zeros((n_alpha, 0))
        )
        Bb = (
            (Cbocc.T @ Fb @ Cbvir) - (Cbocc.T @ S_d @ Cbvir) * eB[:n_beta, None]
            if nvb
            else np.zeros((n_beta, 0))
        )
        return Ba, Bb

    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            ap = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xp = list(ap[a].xyz)
            xp[d] += h
            ap[a] = Atom(ap[a].Z, xp)
            am = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xm = list(am[a].xyz)
            xm[d] -= h
            am[a] = Atom(am[a].Z, xm)
            Bpa, Bpb = _B0(ap)
            Bma, Bmb = _B0(am)
            total = 0.0
            if nva:
                total += float(np.sum(za * ((Bpa - Bma) / (2.0 * h))))
            if nvb:
                total += float(np.sum(zb * ((Bpb - Bmb) / (2.0 * h))))
            relax[a, d] = -2.0 * total
    return relax


def _bloch_cphf_rhs_analytic_open(
    system: PeriodicSystem,
    basis: BasisSet,
    Dtot_home: np.ndarray,
    Pa: np.ndarray,
    Pb: np.ndarray,
    za: np.ndarray,
    zb: np.ndarray,
    Caocc: np.ndarray,
    Cavir: np.ndarray,
    Cbocc: np.ndarray,
    Cbvir: np.ndarray,
    eA: np.ndarray,
    eB: np.ndarray,
    n_alpha: int,
    n_beta: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    builders: tuple,
    *,
    mode: str = "hybrid",
    step_bohr: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Analytic / hybrid UHF coupled-spin Bloch-CPHF RHS gradient
    ``-2.S_s z_s.dB0_s/dR`` -- the open-shell counterpart of
    :func:`_bloch_cphf_rhs_analytic`.

    Per spin ``F_s = Hcore + J[P_tot] - a_HF.K[P_s] + v_bg.S``, so each kernel
    takes the **total** density for the Coulomb cross and the **same-spin**
    density for exchange, contracted with the spin-resolved broadcast response
    ``Pz_s`` (occupation 1 -> factor -1 per spin per 1/2-symmetrised trace). The
    renormalisation response uses the per-spin 2e operator
    ``G_s[Pz] = J_resp[Pz_a+Pz_b] - a_HF.K[Pz_s]``; ``mode`` selects the
    self-adjoint shortcut (``"analytic"``) or the exact 6N-build local renorm
    (``"hybrid"``, J^LR renorm analytic)."""
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from ._vibeqc_core import build_fock_2e_real_space
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import _build_j_long_range_cache

    _S_g, _Hc, f2e0, j_build, k_build = builders
    n_atoms = len(system.unit_cell)
    nbf = int(basis.nbasis)
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    n_elec = int(system.n_electrons())
    v_bg = -np.pi * float(n_elec) / (alpha * alpha * V)
    K_max = float(bipole_ewald_reciprocal_cutoff(V, alpha))
    nva = Cavir.shape[1]
    nvb = Cbvir.shape[1]

    # Spin-resolved response densities (home blocks, occupation 1).
    def _mk_Pz(z, Cocc, Cvir):
        if z.size == 0:
            return np.zeros((nbf, nbf))
        P = Cvir @ z.T @ Cocc.T
        return P + P.T

    def _mk_W(z, Cocc, Cvir, eps, n):
        if z.size == 0:
            return np.zeros((nbf, nbf))
        W = Cvir @ (z * eps[:n][:, None]).T @ Cocc.T
        return W + W.T

    Pza = _mk_Pz(za, Caocc, Cavir)
    Pzb = _mk_Pz(zb, Cbocc, Cbvir)
    Pz_tot = Pza + Pzb
    Wa = _mk_W(za, Caocc, Cavir, eA, n_alpha)
    Wb = _mk_W(zb, Cbocc, Cbvir, eB, n_beta)

    S_set = compute_overlap_lattice(basis, system, lattice_opts)
    cells = list(S_set.cells)
    n_cells = len(cells)
    home = _home_cell_index(cells)
    rc = np.array([np.asarray(c.r_cart, dtype=float) for c in cells], dtype=float)

    def _lsb(M):
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            s.set_block(c, M)
        return s

    def _lsh(M):
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            s.set_block(c, M if c == home else np.zeros((nbf, nbf)))
        return s

    def _kin(M):
        return np.asarray(
            kinetic_lattice_gradient_contribution(basis, system, _lsb(M), lattice_opts)
        )

    def _og(M):
        return np.asarray(
            overlap_lattice_gradient_contribution(basis, system, _lsb(M), lattice_opts)
        )

    def _eri(D_set, ahf, j_scale, omega):
        return np.asarray(
            eri_lattice_gradient_contribution(
                basis,
                system,
                D_set,
                lattice_opts,
                ahf,
                j_scale,
                omega,
                exchange_energy_convention=(
                    ahf != 0.0 and j_scale == 0.0 and omega == 0.0
                ),
            )
        )

    def _eri_cross(P_home, Pz_b, ahf, j_scale, omega):
        comb = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(n_cells):
            comb.set_block(c, (P_home + Pz_b) if c == home else Pz_b)
        return (
            _eri(comb, ahf, j_scale, omega)
            - _eri(_lsh(P_home), ahf, j_scale, omega)
            - _eri(_lsb(Pz_b), ahf, j_scale, omega)
        )

    # Fock-convention J^LR cross machinery (both-Bloch r̂).
    cache = _build_j_long_range_cache(basis, system, rc, alpha, ewald_precision, K_max=K_max, lattice_opts=lattice_opts)
    K_vec = cache.K_vectors
    kernel = cache.kernel
    ft_sum = cache.ft_per_cell.sum(axis=0)
    grad_bra, grad_ket = _pair_ft_gradient_at_cells(basis, K_vec, rc, lattice_opts=lattice_opts)
    corr = _libint_ylm_correction_per_ao(basis)
    cc = corr[:, None] * corr[None, :]
    grad_bra = grad_bra * cc[None, :, :, None, None]
    grad_ket = grad_ket * cc[None, :, :, None, None]
    ao2atom = _ao_to_atom_map(system, basis)

    def _rho_b(M):
        return np.einsum("mn,mnk->k", M, ft_sum)

    def _drho_b(M):
        bra = np.einsum("mn,gmnxk->mxk", M, grad_bra)
        ket = np.einsum("mn,gmnxk->nxk", M, grad_ket)
        out = np.zeros((n_atoms, 3, K_vec.shape[0]), dtype=np.complex128)
        np.add.at(out, ao2atom, bra)
        np.add.at(out, ao2atom, ket)
        return out

    def _jlr_cross(P_home, Pz_b):
        rbP, rbPz = _rho_b(P_home), _rho_b(Pz_b)
        drbP, drbPz = _drho_b(P_home), _drho_b(Pz_b)
        g = np.zeros((n_atoms, 3), dtype=np.float64)
        for Cc in range(n_atoms):
            term = drbP[Cc] * np.conj(rbPz)[None, :] + rbP[None, :] * np.conj(drbPz[Cc])
            g[Cc] = np.real((kernel[None, :] * term).sum(axis=1))
        return g

    # --- term1+3 (skeleton + dS), summed over spins ---
    term13 = np.zeros((n_atoms, 3), dtype=np.float64)
    for Pz_s, W_s, P_s in ((Pza, Wa, Pa), (Pzb, Wb, Pb)):
        if not np.any(Pz_s):
            continue
        onee = _kin(Pz_s) + _v_ne_ewald_gradient(
            system, basis, _lsb(Pz_s), lattice_opts, alpha,
            precision=ewald_precision,
        )
        term13 += (
            -onee
            + v_bg * _og(Pz_s)
            - _eri_cross(Dtot_home, Pz_s, 0.0, 1.0, alpha)  # J_SR[P_tot]
            - _jlr_cross(Dtot_home, Pz_s)  # J^LR[P_tot]
            - 2.0 * a_hf * _eri_cross(P_s, Pz_s, 1.0, 0.0, 0.0)  # -a_HF K[P_s]
            - _og(W_s)
        )

    # --- term2: coupled renormalisation response ---
    def _Gjlr_tot(M):  # J^LR-only response to total density M
        return (
            j_build(M, M)
            - j_build(np.zeros((nbf, nbf)), np.zeros((nbf, nbf)))
            - _jsr_bloch(M)
        )

    def _jsr_bloch(M):
        jsr = build_fock_2e_real_space(basis, system, lattice_opts, _lsh(M), 0.0, alpha)
        return sum(np.asarray(jsr.blocks[i], dtype=float) for i in range(n_cells))

    if mode == "analytic":
        # self-adjoint shortcut: term2 = -S_s og(W_s), W_s from G_s[Pz].
        jresp_tot = j_build(Pz_tot, Pz_tot) - j_build(
            np.zeros((nbf, nbf)), np.zeros((nbf, nbf))
        )
        term2 = np.zeros((n_atoms, 3), dtype=np.float64)
        for Cocc, Pz_s in ((Caocc, Pza), (Cbocc, Pzb)):
            if Cocc.shape[1] == 0:
                continue
            G_s = jresp_tot - a_hf * k_build(Pz_s)
            term2 += -_og(Cocc @ (Cocc.T @ G_s @ Cocc) @ Cocc.T)
    elif mode == "hybrid":
        # J^LR renorm analytic (self-adjoint, exact); local J_SR/K renorm via
        # 6N cheap builds of dD_eff (per-spin renormalised density derivative).
        Gjlr_tot_Pz = _Gjlr_tot(Pz_tot)  # J^LR response to total Pz
        term2 = np.zeros((n_atoms, 3), dtype=np.float64)
        for Cocc, Pz_s in ((Caocc, Pza), (Cbocc, Pzb)):
            if Cocc.shape[1] == 0:
                continue
            term2 += -_og(Cocc @ (Cocc.T @ Gjlr_tot_Pz @ Cocc) @ Cocc.T)
        lattice = np.asarray(system.lattice, dtype=float)
        atoms = list(system.unit_cell)
        h = float(step_bohr)

        def _D_eff_spin(disp_atoms, Cocc):
            sd = PeriodicSystem(system.dim, lattice, disp_atoms)
            # Propagate BOTH charge and multiplicity: unit_cell_molecule()
            # builds a Molecule that validates n_electrons vs multiplicity,
            # so a charged open shell (e.g. H₂⁺ doublet) is rejected if the
            # charge is left at its default 0. Neither attribute affects the
            # overlap numerics below -- they only gate that validation.
            sd.charge = system.charge
            sd.multiplicity = system.multiplicity
            bd = _recenter_basis_on_periodic_system(basis, sd)
            sset = compute_overlap_lattice(bd, sd, lattice_opts)
            S_gamma = sum(
                np.asarray(sset.blocks[i], dtype=float)
                for i in range(len(list(sset.cells)))
            )
            M = Cocc.T @ S_gamma @ Cocc
            return Cocc @ np.linalg.inv(M) @ Cocc.T

        term2_local = np.zeros((n_atoms, 3), dtype=np.float64)
        for a in range(n_atoms):
            for d in range(3):
                ap = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xp = list(ap[a].xyz)
                xp[d] += h
                ap[a] = Atom(ap[a].Z, xp)
                am = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xm = list(am[a].xyz)
                xm[d] -= h
                am[a] = Atom(am[a].Z, xm)
                dDa = (
                    (_D_eff_spin(ap, Caocc) - _D_eff_spin(am, Caocc)) / (2.0 * h)
                    if nva or n_alpha
                    else np.zeros((nbf, nbf))
                )
                dDb = (
                    (_D_eff_spin(ap, Cbocc) - _D_eff_spin(am, Cbocc)) / (2.0 * h)
                    if nvb or n_beta
                    else np.zeros((nbf, nbf))
                )
                jsr_resp = _jsr_bloch(dDa + dDb)  # J_SR[dD_tot]
                tot = 0.0
                if nva:
                    Ga = jsr_resp - a_hf * k_build(dDa)
                    tot += float(np.sum(za * (Caocc.T @ Ga @ Cavir)))
                if nvb:
                    Gb = jsr_resp - a_hf * k_build(dDb)
                    tot += float(np.sum(zb * (Cbocc.T @ Gb @ Cbvir)))
                term2_local[a, d] = -2.0 * tot
        term2 = term2 + term2_local
    else:
        raise ValueError(
            f"_bloch_cphf_rhs_analytic_open: unknown mode {mode!r} "
            "(expected 'analytic' or 'hybrid')"
        )

    return term13 + term2


def _bloch_cphf_relaxation_open(
    system: PeriodicSystem,
    basis: BasisSet,
    D_total: LatticeMatrixSet,
    D_alpha: LatticeMatrixSet,
    D_beta: LatticeMatrixSet,
    mo_coeffs_alpha: np.ndarray,
    mo_energies_alpha: np.ndarray,
    n_alpha: int,
    mo_coeffs_beta: np.ndarray,
    mo_energies_beta: np.ndarray,
    n_beta: int,
    lattice_opts: LatticeSumOptions,
    ewald_alpha: float,
    alpha_hf: float,
    *,
    cphf_rhs: str = "hybrid",
    step_bohr: float = 1e-4,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """UHF coupled-spin Bloch-CPHF orbital-relaxation gradient -- the open-shell
    counterpart of :func:`_bloch_cphf_relaxation`.

    The UHF SCF diagonalises the per-spin Bloch Fock
    ``F_s(Γ) = Hcore + J[P_total] - a_HF.K[P_s] + v_bg.S`` (Coulomb from the
    total density, exchange same-spin). The orbital Hessian is therefore a
    **coupled two-spin** system (the Coulomb response couples a<->b):
    ``A_s(v_a,v_b) = (e_s,a-e_s,i)v_s + Cocc,s.[J_resp(dD_a+dD_b) -
    a_HF.K(dD_s)].Cvir,s`` with ``e_s = mo_energies_s`` and
    ``dD_s = Cvir,s.v_sᵀ.Cocc,sᵀ + h.c.`` (occupation 1). FD-validated to ~1e-6.
    The dense Hessian is solved by least-squares (pseudo-inverse) to absorb
    near-null modes (e.g. a degenerate singly-occupied shell).

    RHS ``b_s = Cocc,s.dE_local/dP_s.Cvir,s`` (per spin, the
    ``_corrected_w_gamma_open`` home block). Gradient
    ``-2.S_s S_ia z_s,ia.dB0_s,ia/dR`` (factor 2 vs RHF's 4 -- occupation 1 vs
    2), with ``B0_s`` the renormalised-fixed-C per-spin Bloch orbital gradient,
    differentiated semi-numerically. Reduces to the RHF result on a
    closed-shell singlet. Returns ``(n_atoms, 3)``."""
    from ._vibeqc_core import build_fock_2e_real_space
    from .cphf import CPHFConvergenceError  # noqa: F401 (parity with RHF path)

    n_atoms = len(system.unit_cell)
    if system.dim != 3:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    Ca = np.asarray(mo_coeffs_alpha)
    Ca = np.real(Ca) if np.iscomplexobj(Ca) else Ca
    Cb = np.asarray(mo_coeffs_beta)
    Cb = np.real(Cb) if np.iscomplexobj(Cb) else Cb
    nva = Ca.shape[1] - n_alpha
    nvb = Cb.shape[1] - n_beta
    if nva == 0 and nvb == 0:
        return np.zeros((n_atoms, 3), dtype=np.float64)
    eA = np.real(np.asarray(mo_energies_alpha))
    eB = np.real(np.asarray(mo_energies_beta))
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    nbf = int(basis.nbasis)
    home = _home_cell_index(list(D_total.cells))
    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)

    Caocc = Ca[:, :n_alpha]
    Cavir = Ca[:, n_alpha:]
    Cbocc = Cb[:, :n_beta]
    Cbvir = Cb[:, n_beta:]
    Pa = np.asarray(D_alpha.blocks[home], dtype=float)
    Pb = np.asarray(D_beta.blocks[home], dtype=float)

    # --- RHS: per-spin local-energy orbital gradient ---
    shared = _bipole_de_dp_home_block(system, basis, D_total, lattice_opts, alpha, 0.0, ewald_precision=ewald_precision)

    def _ls0(M):
        s = compute_overlap_lattice(basis, system, lattice_opts)
        for c in range(len(s.cells)):
            s.set_block(c, M if c == home else np.zeros((nbf, nbf)))
        return s

    def _mK_home(P_spin):  # -a_HF.K[P_spin] home block
        fk = np.asarray(
            build_fock_2e_real_space(
                basis, system, lattice_opts, _ls0(P_spin), a_hf, 0.0
            ).blocks[home],
            dtype=float,
        )
        fj = np.asarray(
            build_fock_2e_real_space(
                basis, system, lattice_opts, _ls0(P_spin), 0.0, 0.0
            ).blocks[home],
            dtype=float,
        )
        return 2.0 * (fk - fj)

    Ma = shared + _mK_home(Pa)
    Mb = shared + _mK_home(Pb)
    ba = (Caocc.T @ Ma @ Cavir).ravel() if nva else np.zeros(0)
    bb = (Cbocc.T @ Mb @ Cbvir).ravel() if nvb else np.zeros(0)

    # --- coupled Bloch Hessian (home geometry; cache built once) ---
    builders = _reconstruct_bipole_fock_gamma_builder(
        system, basis, lattice_opts, alpha, a_hf,
        ewald_precision=ewald_precision,
    )
    j_build, k_build = builders[3], builders[4]

    def _Jresp(dDt):
        return j_build(dDt, dDt) - j_build(np.zeros((nbf, nbf)), np.zeros((nbf, nbf)))

    def _dDsig(v, Cocc, Cvir):
        T = Cvir @ v.T
        D = T @ Cocc.T
        return D + D.T

    eda = (
        (eA[n_alpha:][None, :] - eA[:n_alpha][:, None])
        if nva
        else np.zeros((n_alpha, 0))
    )
    edb = (
        (eB[n_beta:][None, :] - eB[:n_beta][:, None]) if nvb else np.zeros((n_beta, 0))
    )
    na_dim = n_alpha * nva
    nb_dim = n_beta * nvb

    def _Aop(x):
        va = x[:na_dim].reshape(n_alpha, nva) if nva else np.zeros((n_alpha, 0))
        vb = x[na_dim:].reshape(n_beta, nvb) if nvb else np.zeros((n_beta, 0))
        dDa = _dDsig(va, Caocc, Cavir) if nva else np.zeros((nbf, nbf))
        dDb = _dDsig(vb, Cbocc, Cbvir) if nvb else np.zeros((nbf, nbf))
        jr = _Jresp(dDa + dDb)
        out = []
        if nva:
            Ga = jr - a_hf * k_build(dDa)
            out.append((eda * va + (Caocc.T @ Ga @ Cavir)).ravel())
        if nvb:
            Gb = jr - a_hf * k_build(dDb)
            out.append((edb * vb + (Cbocc.T @ Gb @ Cbvir)).ravel())
        return np.concatenate(out) if out else np.zeros(0)

    ndim = na_dim + nb_dim
    H = np.zeros((ndim, ndim))
    for k in range(ndim):
        e = np.zeros(ndim)
        e[k] = 1.0
        H[:, k] = _Aop(e)
    H = 0.5 * (H + H.T)
    rhs = np.concatenate([ba, bb])
    zx, *_ = np.linalg.lstsq(H, rhs, rcond=1e-10)
    za = zx[:na_dim].reshape(n_alpha, nva) if nva else np.zeros((n_alpha, 0))
    zb = zx[na_dim:].reshape(n_beta, nvb) if nvb else np.zeros((n_beta, 0))

    # --- dB0_s/dR (the coupled CPHF right-hand side) ---
    if cphf_rhs in ("analytic", "hybrid"):
        Dtot_home = np.asarray(D_total.blocks[home], dtype=float)
        return _bloch_cphf_rhs_analytic_open(
            system,
            basis,
            Dtot_home,
            Pa,
            Pb,
            za,
            zb,
            Caocc,
            Cavir,
            Cbocc,
            Cbvir,
            eA,
            eB,
            n_alpha,
            n_beta,
            lattice_opts,
            alpha,
            a_hf,
            builders,
            mode=cphf_rhs,
            step_bohr=step_bohr,
            ewald_precision=ewald_precision,
        )
    if cphf_rhs != "seminumeric":
        raise ValueError(
            f"_bloch_cphf_relaxation_open: unknown cphf_rhs {cphf_rhs!r} "
            "(expected 'hybrid', 'analytic', or 'seminumeric')"
        )

    # --- semi-numerical dB0_s/dR (renormalised fixed-C per-spin Bloch grad) ---
    def _B0(disp_atoms, Cocc, Cvir, n, eps, which):
        sd = PeriodicSystem(system.dim, lattice, disp_atoms)
        # Propagate charge as well as multiplicity (see _D_eff_spin): a charged
        # open shell would otherwise fail unit_cell_molecule()'s n_e/mult check.
        sd.charge = system.charge
        sd.multiplicity = system.multiplicity
        bd = _recenter_basis_on_periodic_system(basis, sd)
        S_g, Hcore_g, _, jb, kb = _reconstruct_bipole_fock_gamma_builder(
            sd, bd, lattice_opts, alpha, a_hf,
            ewald_precision=ewald_precision,
        )
        Da = Caocc @ np.linalg.inv(Caocc.T @ S_g @ Caocc) @ Caocc.T
        Db = Cbocc @ np.linalg.inv(Cbocc.T @ S_g @ Cbocc) @ Cbocc.T
        Ps = Da if which == "a" else Db
        F = Hcore_g + jb(Da + Db, Da + Db) - a_hf * kb(Ps)
        return (Cocc.T @ F @ Cvir) - (Cocc.T @ S_g @ Cvir) * eps[:n][:, None]

    h = float(step_bohr)
    relax = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            ap = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xp = list(ap[a].xyz)
            xp[d] += h
            ap[a] = Atom(ap[a].Z, xp)
            am = [Atom(at.Z, list(at.xyz)) for at in atoms]
            xm = list(am[a].xyz)
            xm[d] -= h
            am[a] = Atom(am[a].Z, xm)
            tot = 0.0
            if nva:
                dB0a = (
                    _B0(ap, Caocc, Cavir, n_alpha, eA, "a")
                    - _B0(am, Caocc, Cavir, n_alpha, eA, "a")
                ) / (2.0 * h)
                tot += float(np.sum(za * dB0a))
            if nvb:
                dB0b = (
                    _B0(ap, Cbocc, Cbvir, n_beta, eB, "b")
                    - _B0(am, Cbocc, Cbvir, n_beta, eB, "b")
                ) / (2.0 * h)
                tot += float(np.sum(zb * dB0b))
            relax[a, d] = -2.0 * tot
    return relax


def _j_long_range_ewald_gradient_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    kmesh,
    ewald_alpha: float,
    per_k_density_matrices: Sequence[np.ndarray],
    *,
    precision: float = 1e-8,
    reciprocal_cutoff_bohr_inv: Optional[float] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
) -> np.ndarray:
    """J^LR gradient for multi-k: weighted total r̂*.dr̂/dR.

    The multi-k BIPOLE energy evaluates J^LR per k-point:
        E_JLR = 1/2 S_K kernel(K) . |S_k w_k r̂(k,K)|^2
    so the Hellmann-Feynman gradient (fixed D(k)) is:
        dE/dR = S_K kernel . Re[r̂_tot(K)* . S_k w_k dr̂(k)/dR]

    ``per_k_density_matrices``: list of complex k-space density matrices
    D(k) = 2.C_occ(k).C_occ(k)^+ for each k-point (raw, unweighted).

    This mirrors the SCF's ``compute_rho_hat_from_k_density`` convention:
    one weighted total ``r̂_tot`` builds every ``J^LR(k)`` Fock matrix and the
    real-space block energy contracts to ``1/2 |r̂_tot|^2``.
    """
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import _build_j_long_range_cache

    n_atoms = len(system.unit_cell)
    alpha = float(ewald_alpha)
    a_lat = np.asarray(system.lattice, dtype=float)
    V_cell = float(abs(np.linalg.det(a_lat)))
    K_max = (
        float(reciprocal_cutoff_bohr_inv)
        if reciprocal_cutoff_bohr_inv is not None
        and float(reciprocal_cutoff_bohr_inv) > 0.0
        else float(bipole_ewald_reciprocal_cutoff(V_cell, alpha))
    )

    cells = list(D_real.cells)
    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in cells], dtype=float
    )
    cache = _build_j_long_range_cache(
        basis, system, cells_r_cart, alpha, precision, K_max=K_max, lattice_opts=lattice_opts)
    K_vec = cache.K_vectors
    kernel = cache.kernel
    ft = cache.ft_per_cell  # (n_g, nbf, nbf, n_K)

    # FT centre gradients (shared across k-points)
    grad_bra, grad_ket = _pair_ft_gradient_at_cells(
        basis, K_vec, cells_r_cart, lattice_opts=lattice_opts)
    corr = _libint_ylm_correction_per_ao(basis)
    cc = corr[:, None] * corr[None, :]
    grad_bra = grad_bra * cc[None, :, :, None, None]
    grad_ket = grad_ket * cc[None, :, :, None, None]
    ao2atom = _ao_to_atom_map(system, basis)

    kpts = list(kmesh.kpoints)
    weights = list(kmesh.weights)
    n_g = len(cells)
    n_K = K_vec.shape[0]
    nbf = int(basis.nbasis)

    rho_total = np.zeros(n_K, dtype=np.complex128)
    drho_total = np.zeros((n_atoms, 3, n_K), dtype=np.complex128)

    for ik in range(len(kpts)):
        w_k = float(weights[ik])
        k_arr = np.asarray(kpts[ik], dtype=float).reshape(3)
        Dk = np.asarray(per_k_density_matrices[ik], dtype=np.complex128)

        # Bloch-summed FT: FT^{(-k)}(K) = S_g exp(-ik.R_g) . FT(K; R_g)
        # This matches compute_rho_hat_from_k_density's convention -- uses the
        # full complex Bloch phase, NOT the Re[] used in real-space density.
        phases_k = np.exp(-1j * (cells_r_cart @ k_arr))  # (n_g,)
        ft_bloch = np.einsum("g,gmnk->mnk", phases_k, ft)  # (nbf,nbf,n_K)

        # r̂(k,K) = S_{muν} D(k)_muν . FT^{(-k)}_muν(K)
        rho_k = np.einsum("mn,mnk->k", Dk, ft_bloch)  # (n_K,) complex

        # dr̂(k,K)/dR_C: Bloch-summed FT centre derivative, scattered
        # to atoms via the bra/ket AO-to-atom map.
        # dFT^{(-k)}/dR = S_g exp(-ik.R_g) . dFT(K; R_g)/dR
        grad_bra_bloch = np.einsum("g,gmnxk->mnxk", phases_k, grad_bra)
        grad_ket_bloch = np.einsum("g,gmnxk->mnxk", phases_k, grad_ket)

        # dr̂(k)/dR_C = S_{muinC} S_ν D_muν . dFT^{(-k)}_muν/dR
        #                 + S_{νinC} S_mu D_muν . dFT^{(-k)}_muν/dR.
        bra_m = np.einsum("mn,mnxk->mxk", Dk, grad_bra_bloch)  # (nbf,3,n_K)
        ket_n = np.einsum("mn,mnxk->nxk", Dk, grad_ket_bloch)  # (nbf,3,n_K)
        drho_k = np.zeros((n_atoms, 3, n_K), dtype=np.complex128)
        np.add.at(drho_k, (ao2atom, slice(None), slice(None)), bra_m)
        np.add.at(drho_k, (ao2atom, slice(None), slice(None)), ket_n)

        rho_total += w_k * rho_k
        drho_total += w_k * drho_k

    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for C in range(n_atoms):
        for axis in range(3):
            grad[C, axis] = np.real(
                kernel * np.conj(rho_total) * drho_total[C, axis]
            ).sum()
    return grad


def _build_per_k_density_matrices(
    mo_coeffs: Sequence[np.ndarray],
    n_occ: int,
    n_k: int,
) -> List[np.ndarray]:
    """Build per-k k-space density matrices D(k) = 2.C_occ(k).C_occ(k)+.

    Returns a list of complex (nbf, nbf) matrices, one per k-point.
    These are the raw (unweighted) k-space densities, used for the
    multi-k J^LR gradient (``_j_long_range_ewald_gradient_multi_k``).
    """
    per_k: List[np.ndarray] = []
    for ik in range(n_k):
        C_k = np.asarray(mo_coeffs[ik], dtype=np.complex128)
        C_occ = C_k[:, :n_occ]
        D_k = 2.0 * (C_occ @ C_occ.conj().T)
        per_k.append(D_k)
    return per_k


def _build_per_k_density_matrices_open(
    mo_coeffs_alpha: Sequence[np.ndarray],
    mo_coeffs_beta: Sequence[np.ndarray],
    n_alpha: int,
    n_beta: int,
    n_k: int,
) -> List[np.ndarray]:
    """Build total per-k UHF densities D(k) = D_alpha(k) + D_beta(k)."""
    per_k: List[np.ndarray] = []
    for ik in range(n_k):
        Ca = np.asarray(mo_coeffs_alpha[ik], dtype=np.complex128)
        Cb = np.asarray(mo_coeffs_beta[ik], dtype=np.complex128)
        D_k = np.zeros((Ca.shape[0], Ca.shape[0]), dtype=np.complex128)
        if n_alpha > 0:
            Ca_occ = Ca[:, :n_alpha]
            D_k += Ca_occ @ Ca_occ.conj().T
        if n_beta > 0:
            Cb_occ = Cb[:, :n_beta]
            D_k += Cb_occ @ Cb_occ.conj().T
        per_k.append(D_k)
    return per_k


def _build_per_k_density_matrices_frac(
    mo_coeffs: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
    n_k: int,
) -> List[np.ndarray]:
    """Per-k closed-shell density D(k) = S_i f_i(k) C_i C_i+ with fractional
    occupations (f_i in [0, 2]). Reduces to
    :func:`_build_per_k_density_matrices` for integer Aufbau."""
    per_k: List[np.ndarray] = []
    for ik in range(n_k):
        C_k = np.asarray(mo_coeffs[ik], dtype=np.complex128)
        f = np.asarray(np.real(occupations[ik]), dtype=np.float64)
        n = min(C_k.shape[1], f.size)
        Cn = C_k[:, :n]
        per_k.append((Cn * f[:n][None, :]) @ Cn.conj().T)
    return per_k


def _build_per_k_spin_density_frac(
    mo_coeffs: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
    n_k: int,
) -> List[np.ndarray]:
    """Per-k single-spin density D_s(k) = S_i f_i^s(k) C_i C_i+ (f_i^s in
    [0, 1]) for the open-shell fractional-occupation gradient."""
    per_k: List[np.ndarray] = []
    for ik in range(n_k):
        C_k = np.asarray(mo_coeffs[ik], dtype=np.complex128)
        f = np.asarray(np.real(occupations[ik]), dtype=np.float64)
        n = min(C_k.shape[1], f.size)
        Cn = C_k[:, :n]
        per_k.append((Cn * f[:n][None, :]) @ Cn.conj().T)
    return per_k


def _compute_bipole_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    W_k_list: List[np.ndarray],
    n_elec: int,
    *,
    lattice_opts: LatticeSumOptions,
    alpha_hf: float,
    ewald_alpha: Optional[float],
    kmesh=None,
    D_alpha: Optional[LatticeMatrixSet] = None,
    D_beta: Optional[LatticeMatrixSet] = None,
    per_k_jlr_densities: Optional[Sequence[np.ndarray]] = None,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Shared gradient computation for all BIPOLE methods.

    ``D_alpha`` / ``D_beta``: for open-shell (UHF/UKS) the exchange energy
    is spin-resolved, ``E_x = -1/2a_HF S_s Tr[P_s K[P_s]]``, so its gradient
    must contract per spin. When both are supplied the exchange term is
    built as ``2.(dE_x[P_a] + dE_x[P_b])`` (which reduces *exactly* to the
    closed-shell ``dE_x[P_total]`` single call when ``P_a = P_b = 1/2P_total``,
    since the exchange-gradient kernel is quadratic in the density). When
    they are ``None`` (RHF/RKS closed-shell) the total-density exchange is
    used.

    Only the kinetic + overlap (Pulay) terms are gauge-correct; the
    electrostatic terms use truncated direct full-Coulomb kernels that
    do not match the energy's Ewald split (see module docstring). The
    Pulay energy-weighted density ``W`` is inverse-Bloch folded over the
    k-mesh when ``kmesh`` is supplied and the run is multi-k; otherwise
    it falls back to the Γ broadcast (exact at Γ, a warning is emitted
    at multi-k).

    ``per_k_jlr_densities``: optional list of complex k-space density
    matrices. When supplied for a multi-k run, the J^LR gradient uses the
    same weighted total ``rho_hat`` convention as the SCF Fock builder,
    ``rho_tot = sum_k w_k rho_k``.
    """
    n_atoms = len(system.unit_cell)
    n_k = len(W_k_list)
    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    if n_k > 1 and kmesh is not None:
        _bloch_fold_w_matrices(W_k_list, kmesh, W_set)
    else:
        if n_k > 1 and kmesh is None:
            warnings.warn(
                "compute_bipole_gradient: multi-k run but no kmesh= was "
                "passed; the Pulay energy-weighted-density term falls back "
                "to the Γ broadcast W(Γ), which is incorrect off-Γ. Pass "
                "kmesh= (the BlochKMesh the SCF used) for the multi-k "
                "inverse-Bloch fold W(g)=S_k w_k Re[e^{-ik.g}W(k)].",
                UserWarning,
                stacklevel=3,
            )
        _gamma_lattice_set(W_set, W_k_list[0])

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    # Nuclear repulsion E_nn gradient.
    #
    # The BIPOLE *energy* builds E_nn in CRYSTAL's Ewald gauge
    # (``ewald_nuclear_repulsion``), so the matching gradient is the Ewald
    # gradient dE_nn/dR_A with the *same* a and real / reciprocal cutoffs.
    # The legacy ``nuclear_repulsion_gradient_per_cell`` differentiates the
    # truncated *direct* 1/r sum instead -- gauge-inconsistent on 3D
    # crystals (CLAUDE.md Sec.7), kept only as the 1D / 2D / non-Ewald path.
    ewald_opts = _matching_ewald_options(system, lattice_opts, ewald_alpha, tolerance=ewald_precision)
    if ewald_opts is not None:
        grad += np.asarray(ewald_nuclear_repulsion_gradient(system, ewald_opts))
    else:
        grad += np.asarray(nuclear_repulsion_gradient_per_cell(system, lattice_opts))
    grad += np.asarray(
        overlap_lattice_gradient_contribution(basis, system, W_set, lattice_opts)
    )
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(basis, system, D_real, lattice_opts)
    )

    # V_ne gradient. In the 3D Ewald gauge the energy uses screened-erfc +
    # reciprocal AO-pair-FT V_ne, so the matching gradient is the Ewald
    # V_ne gradient (erfc V_short + reciprocal V_long + background).
    # ``nuclear_lattice_gradient_contribution`` (truncated full-Coulomb) is
    # the non-Ewald fallback.
    if ewald_opts is not None:
        grad += _v_ne_ewald_gradient(
            system, basis, D_real, lattice_opts, float(ewald_alpha),
            precision=ewald_precision,
        )
    else:
        grad += np.asarray(
            nuclear_lattice_gradient_contribution(basis, system, D_real, lattice_opts)
        )

    # J^LR electron-electron jellium background.
    #
    # The energy adds the potential v_bg.S(g), v_bg = -piN_e/(a^2V), to the
    # J^LR Fock operator, so its energy is the Coulomb 1/2.S_g tr[D(g) v_bg S(g)]
    # (the 1/2 is the Hartree double-counting factor; cf. pbc_bipole
    # ``e_j_long_range = 0.5.contract(D, F_LR)``). Its derivative is therefore
    # +1/2 v_bg S_g D(g).dS(g)/dR. Since
    # ``overlap_lattice_gradient_contribution(M)`` returns -S M.dS/dR, that is
    # -1/2.overlap_grad(v_bg.D). (The V_ne +piQ/(a^2V).S background is a separate
    # full-weight Hcore term handled in _v_ne_ewald_gradient; the E_nn jellium
    # term is position-independent.)
    if ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        v_bg = -np.pi * float(n_elec) / (float(ewald_alpha) ** 2 * V_cell)
        # Build a v_bg-scaled density set using D_real's own cell list
        # so the loop below never runs past the available blocks.
        D_bg_set = compute_overlap_lattice(basis, system, lattice_opts)
        n_bg_cells = len(D_bg_set.cells)
        for c in range(min(n_bg_cells, len(D_real.blocks))):
            D_bg_set.set_block(c, v_bg * np.asarray(D_real.blocks[c], dtype=float))
        grad += -0.5 * np.asarray(
            overlap_lattice_gradient_contribution(basis, system, D_bg_set, lattice_opts)
        )

    # 2-electron J + K gradient.
    #
    # The BIPOLE energy builds the Coulomb J in the Ewald split
    # J = J_SR(w) + J^LR(w) (screened short-range + reciprocal long-range)
    # while the exchange K stays full-Coulomb. So in the Ewald gauge the
    # 2e gradient is dJ_SR + dJ^LR - 1/2dK_full, assembled from:
    #   * exchange-only full-Coulomb K  (j_scale=0, w=0)
    #   * screened J_SR only            (alpha_hf=0, j_scale=1, w)
    #   * reciprocal J^LR               (Python, _j_long_range_ewald_gradient)
    # The legacy single full-Coulomb J+K call is the non-Ewald fallback.
    open_shell = D_alpha is not None and D_beta is not None
    if ewald_opts is not None:
        # Exchange (K only, full Coulomb). Closed-shell: one total-density
        # call. Open-shell: spin-resolved 2.(dE_x[P_a] + dE_x[P_b]) (see the
        # docstring -- exact closed-shell reduction when P_a=P_b).
        if open_shell:
            for D_sigma in (D_alpha, D_beta):
                grad += 2.0 * np.asarray(
                    eri_lattice_gradient_contribution(
                        basis,
                        system,
                        D_sigma,
                        lattice_opts,
                        float(alpha_hf),
                        0.0,
                        0.0,
                        exchange_energy_convention=True,
                    )
                )
        else:
            grad += np.asarray(
                eri_lattice_gradient_contribution(
                    basis,
                    system,
                    D_real,
                    lattice_opts,
                    float(alpha_hf),
                    0.0,
                    0.0,  # K only (full Coulomb)
                    exchange_energy_convention=True,
                )
            )
        grad += np.asarray(
            eri_lattice_gradient_contribution(
                basis,
                system,
                D_real,
                lattice_opts,
                0.0,
                1.0,
                float(ewald_alpha),  # J_SR only (erfc-screened)
            )
        )
        if per_k_jlr_densities is not None and n_k > 1:
            # Multi-k J^LR: per-k sum avoids cross-term error.
            grad += _j_long_range_ewald_gradient_multi_k(
                system,
                basis,
                D_real,
                kmesh,
                float(ewald_alpha),
                per_k_jlr_densities, lattice_opts=lattice_opts, precision=ewald_precision)
        else:
            grad += _j_long_range_ewald_gradient(
                system,
                basis,
                D_real,
                float(ewald_alpha),
                gamma_local=(n_k == 1), lattice_opts=lattice_opts, precision=ewald_precision)
        # Post-SCF EXT EL-SPHEROPOLE term (K=0 spheropole coupling), part of
        # the BIPOLE total energy and therefore of its gradient.
        grad += _spheropole_ewald_gradient(system, basis, D_real, lattice_opts)
    elif open_shell:
        # Legacy (non-Ewald) fallback, open-shell: full-Coulomb J from the
        # total density + spin-resolved exchange.
        grad += np.asarray(
            eri_lattice_gradient_contribution(
                basis,
                system,
                D_real,
                lattice_opts,
                0.0,
                1.0,
                0.0,  # J only
            )
        )
        for D_sigma in (D_alpha, D_beta):
            grad += 2.0 * np.asarray(
                eri_lattice_gradient_contribution(
                    basis,
                    system,
                    D_sigma,
                    lattice_opts,
                    float(alpha_hf),
                    0.0,
                    0.0,  # K per spin
                    exchange_energy_convention=True,
                )
            )
    else:
        grad += np.asarray(
            eri_lattice_gradient_contribution(
                basis,
                system,
                D_real,
                lattice_opts,
                float(alpha_hf),
                exchange_energy_convention=True,
            )
        )
    return grad


def _corrected_gamma_homogeneous_density(
    system: PeriodicSystem,
    basis: BasisSet,
    block: np.ndarray,
    lattice_opts: LatticeSumOptions,
) -> LatticeMatrixSet:
    """Density set with ``block`` in every cell of the gradient template.

    At Γ the corrected-gauge real-space density is ``D(g)=D_Γ`` for every
    cell -- and that is the density the corrected-gauge SCF energy is built
    from, so the analytic gradient kernels must consume it too. Passing
    the SCF density set directly over-counts by the cell-count (the SCF
    density lives on a different cell list). Mirrors the periodic-SCF
    chat's ``_gamma_density_lattice_set(homogeneous=True)``.
    """
    D = compute_overlap_lattice(basis, system, lattice_opts)
    blk = np.asarray(block, dtype=np.float64)
    for c in range(len(D.cells)):
        D.set_block(c, blk)
    return D


def _compute_bipole_gradient_corrected_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    D_home: np.ndarray,
    W_gamma: np.ndarray,
    n_elec: int,
    *,
    lattice_opts: LatticeSumOptions,
    alpha_hf: float,
    ewald_alpha: float,
    spin_home_blocks: Optional[Sequence[np.ndarray]] = None,
    sr_image_extent_bohr: Optional[float] = None,
    sr_density_cells: Optional[Sequence[object]] = None,
    sr_density_domain: Optional[object] = None,
    sr_output_cells: Optional[Sequence[object]] = None,
    sr_output_shell_masks: Optional[Sequence[np.ndarray]] = None,
    lr_output_cells: Optional[Sequence[object]] = None,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Γ-only corrected-gauge (Ewald-exchange-split) BIPOLE gradient.

    Closed-shell (``spin_home_blocks=None``) and open-shell (RHF/RKS vs
    UHF/UKS) share everything except the exchange contraction. ``D_home``
    is always the TOTAL Γ density home block (1-electron + Coulomb +
    jellium terms); for open-shell the exchange is spin-resolved,
    ``2.S_s dE_x[P_s]``, with ``spin_home_blocks = (P_a^home, P_b^home)``
    (the closed-shell total-density call is its ``P_a=P_b=1/2P`` reduction).
    ``W_gamma`` is the energy-weighted density (closed: ``2 S_i e_i C_iC_i+``;
    open: ``W_a + W_b``).

    The corrected gauge is a *standard* variational HF gradient -- the full
    Bloch density carries no Γ-locality projection, so (unlike the legacy
    Γ-local gauge) there is NO Bloch-CPHF orbital relaxation. It is the
    legacy assembly with the exchange block swapped: full-Coulomb K +
    spheropole -> ``K_SR(erfc) + K_LR(recip) + (ξ_M - pi/Vw^2).S.D.S``, and
    ``W`` taken from the variational Fock eigenvalues.

    Two corrected-gauge specifics vs ``_compute_bipole_gradient``:

    * the gradient kernels consume a density built HOMOGENEOUS on the
      gradient template (see ``_corrected_gamma_homogeneous_density``).
    * the jellium background ``-pi N_e^2/(2w^2V)`` is quadratic in
      ``N_e = Tr[D S]``, so its fixed-density gradient is FULL
      (``v_bg.Tr[DdS]``), not the legacy half -- the ``W`` (built from the
      variational eigenvalues, which carry the ``+v_bg`` per-orbital
      jellium shift) supplies the cancelling ``-v_bg.Tr[DdS]``, so the net
      jellium force is the correct ~0.

    ``sr_image_extent_bohr`` extends the internal lambda/sigma image traversal
    of the erfc J_SR/K_SR derivatives. ``sr_density_cells`` supplies the SCF's
    2x-cutoff P(h) support while c_g remains on the ordinary gradient template;
    ``sr_density_domain`` applies the same private pair mask as the direct SR
    builder. ``sr_output_cells`` and ``sr_output_shell_masks`` replace the
    ordinary output template with the pair-resolved SYM3b cell/sub-block domain.
    ``lr_output_cells`` independently selects the reciprocal J/K pair template;
    the uniform background remains on the radial SCF overlap template.

    Validated against ``compute_bipole_gradient_fd`` to ~1e-8 Ha/bohr on the
    historical domain (MgO/STO-3G, 2026-06-15); the M5 padded-domain gate is
    maintained separately.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    from .bipole_fock_ewald import (
        exchange_q0_gauge_constant,
        probe_charge_madelung,
    )

    lat = lattice_opts
    nbf = basis.nbasis
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    D_home = np.asarray(D_home, dtype=np.float64)
    sr_internal_cells = (
        []
        if sr_image_extent_bohr is None
        else list(_sr_internal_cells(basis, system, lattice_opts, sr_image_extent_bohr))
    )
    if sr_image_extent_bohr is not None and not sr_density_cells:
        raise ValueError(
            "_compute_bipole_gradient_corrected_gamma: padded SR traversal "
            "requires the SCF density-support cells"
        )
    if sr_output_shell_masks is not None and sr_output_cells is None:
        raise ValueError(
            "_compute_bipole_gradient_corrected_gamma: output shell masks "
            "require explicit output cells"
        )

    Dg = _corrected_gamma_homogeneous_density(system, basis, D_home, lat)
    Dg_lr = (
        Dg
        if lr_output_cells is None
        else make_lattice_matrix_set(
            int(nbf),
            list(lr_output_cells),
            [D_home.copy() for _ in lr_output_cells],
        )
    )
    if sr_image_extent_bohr is None:
        Dg_sr = Dg
        sr_output_cells = []
    else:
        density_cells = list(sr_density_cells)
        Dg_sr = make_lattice_matrix_set(
            int(nbf),
            density_cells,
            [D_home.copy() for _ in density_cells],
        )
        if sr_density_domain is not None:
            from .pair_resolved_truncation import (
                mask_lattice_blocks_to_domain,
            )

            mask_lattice_blocks_to_domain(
                basis, sr_density_domain, Dg_sr
            )
        sr_output_cells = (
            (sr_internal_cells if lattice_opts.pair_complete_1e else list(Dg.cells))
            if sr_output_cells is None
            else list(sr_output_cells)
        )
    sr_output_shell_masks = (
        []
        if sr_output_shell_masks is None
        else [np.asarray(mask, dtype=np.uint8) for mask in sr_output_shell_masks]
    )

    # --- E_nn + overlap-Lagrangian (Pulay) + kinetic + V_ne ---
    W_set = compute_overlap_lattice(basis, system, lat)
    _gamma_lattice_set(W_set, np.asarray(W_gamma, dtype=np.float64))
    ewald_opts = _matching_ewald_options(system, lat, alpha, tolerance=ewald_precision)
    grad = np.asarray(ewald_nuclear_repulsion_gradient(system, ewald_opts))
    grad += np.asarray(overlap_lattice_gradient_contribution(basis, system, W_set, lat))
    grad += np.asarray(kinetic_lattice_gradient_contribution(basis, system, Dg, lat))
    grad += _v_ne_ewald_gradient(system, basis, Dg, lat, alpha, precision=ewald_precision)

    # --- jellium background (FULL: v_bg ∝ N_e = Tr[D S], quadratic term) ---
    v_bg = -math.pi * float(n_elec) / (alpha * alpha * V_cell)
    D_bg = _corrected_gamma_homogeneous_density(
        system, basis, v_bg * D_home, lat
    )
    grad += -1.0 * np.asarray(
        overlap_lattice_gradient_contribution(basis, system, D_bg, lat)
    )

    # --- corrected-gauge exchange: K_SR(erfc) + K_LR(recip) + Madelung ---
    # dE_x[D'] for one density block = d(-1/4 Tr[D' (K_SR+K_LR+Madelung)[D']]).
    # Closed: one total-density call. Open: 2.(dE_x[P_a] + dE_x[P_b]).
    if abs(a_hf) > 1.0e-14:
        c_g0 = exchange_q0_gauge_constant(
            probe_charge_madelung(system), alpha, V_cell, 1
        )
        S_set = compute_overlap_lattice(basis, system, lat)
        S_gamma = np.zeros((nbf, nbf), dtype=np.float64)
        for c in range(len(S_set.cells)):
            S_gamma += np.asarray(S_set.blocks[c], dtype=np.float64)

        def _exchange_grad(block: np.ndarray) -> np.ndarray:
            blk = np.asarray(block, dtype=np.float64)
            Dgx_lr = (
                _corrected_gamma_homogeneous_density(system, basis, blk, lat)
                if lr_output_cells is None
                else make_lattice_matrix_set(
                    int(nbf),
                    list(lr_output_cells),
                    [blk.copy() for _ in lr_output_cells],
                )
            )
            if sr_image_extent_bohr is None:
                Dgx_sr = Dgx_lr
            else:
                density_cells = list(sr_density_cells)
                Dgx_sr = make_lattice_matrix_set(
                    int(nbf),
                    density_cells,
                    [blk.copy() for _ in density_cells],
                )
                if sr_density_domain is not None:
                    from .pair_resolved_truncation import (
                        mask_lattice_blocks_to_domain,
                    )

                    mask_lattice_blocks_to_domain(
                        basis, sr_density_domain, Dgx_sr
                    )
            gx = np.asarray(
                eri_lattice_gradient_contribution(
                    basis,
                    system,
                    Dgx_sr,
                    lat,
                    a_hf,
                    0.0,
                    alpha,
                    False,
                    sr_internal_cells,
                    sr_output_cells,
                    sr_output_shell_masks,
                )
            )
            gx = gx + a_hf * _k_long_range_ewald_gradient(
                system, basis, Dgx_lr, alpha, lattice_opts=lattice_opts, precision=ewald_precision)
            Mx_set = compute_overlap_lattice(basis, system, lat)
            _gamma_lattice_set(Mx_set, blk @ S_gamma @ blk)
            gx = gx + a_hf * 0.5 * c_g0 * np.asarray(
                overlap_lattice_gradient_contribution(basis, system, Mx_set, lat)
            )
            return gx

        if spin_home_blocks is None:
            grad += _exchange_grad(D_home)
        else:
            for spin_block in spin_home_blocks:
                grad += 2.0 * _exchange_grad(spin_block)

    # --- Coulomb J_SR(erfc) + J_LR(recip) ---
    grad += np.asarray(
        eri_lattice_gradient_contribution(
            basis,
            system,
            Dg_sr,
            lat,
            0.0,
            1.0,
            alpha,
            False,
            sr_internal_cells,
            sr_output_cells,
            sr_output_shell_masks,
        )
    )
    grad += _j_long_range_ewald_gradient(
        system, basis, Dg_lr, alpha, gamma_local=True, lattice_opts=lattice_opts, precision=ewald_precision)
    return grad


def _k_long_range_ewald_gradient_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    per_k_density: Sequence[np.ndarray],
    kmesh,
    ewald_alpha: float,
    lattice_opts: LatticeSumOptions,
    *,
    precision: float = 1e-8,
    reciprocal_cutoff_bohr_inv: Optional[float] = None,
) -> np.ndarray:
    """Multi-k per-q reciprocal long-range EXCHANGE gradient (corrected gauge).

    Multi-k generalisation of :func:`_k_long_range_ewald_gradient`. The
    reciprocal exchange couples every ordered ``(k, k′)`` pair through the
    momentum transfer ``q = k - k′`` -- the gradient analogue of
    :func:`vibeqc.bipole_fock_ewald.compute_K_long_range_at_k`::

        E_K^LR  = -1/4 S_k w_k Tr[D(k) K^LR(k)],
        K^LR(k) = S_{k′} w_{k′} S_{q+G!=0} kernel . B^{(k′)}*.D(k′).B^{(k′)}.

    Holding ``D(k)`` fixed and differentiating the two pair-FT factors,

        dE = -1/2 S_k S_{k′} w_k w_{k′} Re S_K kernel S_muν dB*_muν(q+G).C1_muν,
        C1(q+G) = D(k).B^{(k′)}(q+G).D(k′),

    with ``dB^{(k′)}`` the q-shifted, ``k′``-phased Bloch-summed AO-pair-FT
    centre derivative. ``B^{(k′)}`` and the per-channel ``q+G`` vectors come
    from the SCF's :class:`KExchangeLongRangeCache`. Reduces to
    :func:`_k_long_range_ewald_gradient` at ``n_k = 1`` (q == 0). Validated
    against the central difference of ``-1/4S_k w_k Tr[D(k)K^LR(k)]`` to
    7.1e-9 Ha/bohr (asymmetric BeH₂ [2,1,1]/STO-3G, 2026-06-17).
    """
    from ._aopair_ft import ao_pair_fourier_transform_grad_at_cells
    from .bipole_ext_el_pole import (
        _libint_ylm_correction_per_ao,
        bipole_ewald_reciprocal_cutoff,
    )
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        build_k_exchange_long_range_cache,
    )

    alpha = float(ewald_alpha)
    a_lat = np.asarray(system.lattice, dtype=float)
    V_cell = float(abs(np.linalg.det(a_lat)))
    K_max = (
        float(reciprocal_cutoff_bohr_inv)
        if reciprocal_cutoff_bohr_inv is not None
        and float(reciprocal_cutoff_bohr_inv) > 0.0
        else float(bipole_ewald_reciprocal_cutoff(V_cell, alpha))
    )
    cells_r = np.array(
        [
            np.asarray(c.r_cart, dtype=float)
            for c in compute_overlap_lattice(basis, system, lattice_opts).cells
        ],
        dtype=float,
    )
    j_cache = _build_j_long_range_cache(
        basis, system, cells_r, alpha, precision, K_max=K_max, lattice_opts=lattice_opts)
    x_cache = build_k_exchange_long_range_cache(basis, system, j_cache, K_max=K_max)
    corr = _libint_ylm_correction_per_ao(basis)
    cc = corr[:, None] * corr[None, :]
    ao2atom = _ao_to_atom_map(system, basis)
    kpts = [np.asarray(k, dtype=float) for k in kmesh.kpoints]
    weights = [float(w) for w in kmesh.weights]
    n_atoms = len(system.unit_cell)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for ik, k in enumerate(kpts):
        Dk = np.asarray(per_k_density[ik], dtype=np.complex128)
        for jk, kp in enumerate(kpts):
            Dkp = np.asarray(per_k_density[jk], dtype=np.complex128)
            kernel, B = x_cache.channel_tables(k - kp, kp)
            key, _ = x_cache._canonical_q(k - kp)
            if not np.any(np.frombuffer(key, dtype=float)):
                K_vec = j_cache.K_vectors
            else:
                K_vec = x_cache.q_channels[key].K_vectors
            grad_bra, grad_ket = _pair_ft_gradient_at_cells(
                basis, K_vec, cells_r, lattice_opts=lattice_opts)
            phases = np.exp(-1j * (cells_r @ kp))
            dB_bra = np.einsum("g,gmnxk->mnxk", phases, grad_bra) * cc[:, :, None, None]
            dB_ket = np.einsum("g,gmnxk->mnxk", phases, grad_ket) * cc[:, :, None, None]
            C1 = np.einsum("ma,abk,bn->mnk", Dk, B, Dkp, optimize=True)
            w = weights[ik] * weights[jk]
            bcontr = np.einsum(
                "k,mnxk,mnk->mx", kernel, dB_bra.conj(), C1, optimize=True
            )
            kcontr = np.einsum(
                "k,mnxk,mnk->nx", kernel, dB_ket.conj(), C1, optimize=True
            )
            np.add.at(grad, ao2atom, -0.5 * w * np.real(bcontr))
            np.add.at(grad, ao2atom, -0.5 * w * np.real(kcontr))
    return grad


def _madelung_ewald_gradient_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    per_k_density: Sequence[np.ndarray],
    kmesh,
    ewald_alpha: float,
    lattice_opts: LatticeSumOptions,
) -> np.ndarray:
    """Multi-k per-k Madelung (G=0 exchange) gradient (corrected gauge).

    The supercell G=0 exchange correction is ``c_g0.S(k)D(k)S(k)`` per k
    (``compute_K_long_range_at_k`` G=0 handling), with
    ``c_g0 = ξ_M(supercell) - pi/(w^2.V.n_k)``. Its energy is
    ``-1/4 S_k w_k c_g0 Tr[D(k)S(k)D(k)S(k)]``; holding D(k) fixed,
    ``dE = -1/2 c_g0 S_k w_k Tr[M(k) dS(k)]`` with ``M(k)=D(k)S(k)D(k)``.
    Inverse-Bloch-folding ``M(k)`` to real space and using
    ``overlap_lattice_gradient_contribution(M̃) = -S_g M̃(g).dS(g)``, the term
    is ``+1/2.c_g0.overlap_grad(M̃)``. ``c_g0`` is constant under atom
    displacement (ξ_M, V, n_k fixed). Validated FD 1.8e-8 (BeH₂ [2,1,1],
    2026-06-17).
    """
    from .bipole_fock_ewald import (
        exchange_q0_gauge_constant,
        probe_charge_madelung_supercell,
    )
    from .periodic_k_symmetry import density_set_from_k_matrices

    alpha = float(ewald_alpha)
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    n_k = len(list(kmesh.kpoints))
    mesh = list(kmesh.mesh)
    c_g0 = exchange_q0_gauge_constant(
        probe_charge_madelung_supercell(system, mesh), alpha, V_cell, n_k
    )
    S_set = compute_overlap_lattice(basis, system, lattice_opts)
    cells_r = [np.asarray(c.r_cart, dtype=float) for c in S_set.cells]
    s_blocks = [np.asarray(b, dtype=float) for b in S_set.blocks]
    M_k = []
    for ik, k in enumerate(kmesh.kpoints):
        k_arr = np.asarray(k, dtype=float)
        S_k = sum(
            np.exp(1j * float(np.dot(k_arr, R))) * Sg
            for R, Sg in zip(cells_r, s_blocks)
        )
        S_k = 0.5 * (S_k + S_k.conj().T)
        Dk = np.asarray(per_k_density[ik], dtype=np.complex128)
        M_k.append(Dk @ S_k @ Dk)
    M_set = density_set_from_k_matrices(system, basis, lattice_opts, kmesh, M_k)
    return (
        0.5
        * c_g0
        * np.asarray(
            overlap_lattice_gradient_contribution(basis, system, M_set, lattice_opts)
        )
    )


def _compute_bipole_gradient_corrected_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    result,
    kmesh,
    *,
    lattice_opts: LatticeSumOptions,
    alpha_hf: float,
    ewald_alpha: float,
    sr_image_extent_bohr: Optional[float] = None,
    sr_density: Optional[LatticeMatrixSet] = None,
    sr_spin_density: Optional[
        tuple[LatticeMatrixSet, LatticeMatrixSet]
    ] = None,
    reciprocal_cutoff_bohr_inv: Optional[float] = None,
    j_sr_alpha_image_ball: bool = False,
    ewald_precision: float = 1e-8,
) -> np.ndarray:
    """Multi-k corrected-gauge (Ewald-exchange-split) BIPOLE RHF/UHF gradient.

    Multi-k generalisation of :func:`_compute_bipole_gradient_corrected_gamma`.
    A *standard* variational HF gradient (full Bloch density -> no Bloch-CPHF):
    the legacy assembly with the exchange block swapped to ``K_SR(erfc) +
    K_LR(recip, per-q) + Madelung(per-k)``.

    **Density convention (the multi-k analogue of the Γ homogeneous-on-template
    trap).** Every real-space-density term -- kinetic, V_ne, jellium, J_SR,
    K_SR, *and* J^LR -- consumes ``D_grad``, the per-k density inverse-Bloch-
    folded onto the GRADIENT template (``density_set_from_k_matrices``), NOT
    ``result.density`` (whose 2x-cutoff cell list over-counts the lattice-
    summed 2e gradient -- feeding J^LR ``result.density`` was a 2.78e-3 error).
    The per-q K_LR + per-k Madelung kernels take the raw per-k ``D(k)``.

    Handles closed-shell (RHF / RKS-HF-part) and open-shell (UHF / UKS-HF-part)
    automatically from the result's shell type: open-shell builds per-spin per-k
    densities and a spin-resolved exchange ``2.S_s dE_x[P_s]`` with the open
    energy-weighted ``W``. ``alpha_hf`` is the functional's HF fraction. RKS/UKS
    XC Pulay is layered on by their drivers. Reduces EXACTLY to the Γ corrected
    core at ``n_k = 1`` (RHF 1.8e-18); FD-clean at ``n_k = 2`` (RHF 6.3e-8 vs
    ``compute_bipole_gradient_fd``, asymmetric BeH₂/STO-3G, 2026-06-17).

    ``j_sr_alpha_image_ball`` (#575) is for the Ewald multi-k SCF energies
    whose Hartree term is the exact analytic-FT J (every image present, no
    real-space erfc pass): it extends the lambda/sigma image traversal of
    the ``J_SR`` derivative to the radius where ``erfc(alpha R)`` has decayed,
    keeping the density support as the output domain. Ignored when a BIPOLE
    ``sr_image_extent_bohr`` padded traversal is supplied.
    """
    if sr_image_extent_bohr is not None and lattice_opts.pair_complete_1e:
        lattice_opts = _sr_padded_lat_opts(lattice_opts, sr_image_extent_bohr)
    from .periodic_k_symmetry import density_set_from_k_matrices

    lat = lattice_opts
    alpha = float(ewald_alpha)
    a_hf = float(alpha_hf)
    n_elec = system.n_electrons()
    n_k = len(list(kmesh.kpoints))
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))

    # Closed-shell (RHF/RKS) vs open-shell (UHF/UKS): the TOTAL density drives
    # the 1-electron / Coulomb / jellium / J^LR terms and the overlap-Lagrangian
    # W; the exchange is spin-resolved for open shell (2.S_s dE_x[P_s]).
    open_shell = hasattr(result, "mo_coeffs_alpha")
    # Fractional occupation (finite-T / smearing): per-k density and W carry
    # the SCF occupations f_i(k); the free-energy force is the standard
    # multi-k Pulay/HF gradient with these fractional weights (see the Γ
    # corrected core). False for RHF/UHF (no occupations) and integer Aufbau.
    frac = _is_fractional_ks_occupation(result, "uks" if open_shell else "rks")
    if open_shell:
        n_alpha = (n_elec + system.multiplicity - 1) // 2
        n_beta = (n_elec - system.multiplicity + 1) // 2
        if frac:
            per_k_Da = _build_per_k_spin_density_frac(
                result.mo_coeffs_alpha, result.occupations_alpha, n_k
            )
            per_k_Db = _build_per_k_spin_density_frac(
                result.mo_coeffs_beta, result.occupations_beta, n_k
            )
            W_k_list = _build_energy_weighted_density_open_frac(
                result.mo_coeffs_alpha,
                result.mo_energies_alpha,
                result.occupations_alpha,
                result.mo_coeffs_beta,
                result.mo_energies_beta,
                result.occupations_beta,
            )
        else:
            per_k_Da = [
                np.asarray(Ca, dtype=np.complex128)[:, :n_alpha]
                @ np.asarray(Ca, dtype=np.complex128)[:, :n_alpha].conj().T
                for Ca in result.mo_coeffs_alpha
            ]
            per_k_Db = [
                np.asarray(Cb, dtype=np.complex128)[:, :n_beta]
                @ np.asarray(Cb, dtype=np.complex128)[:, :n_beta].conj().T
                for Cb in result.mo_coeffs_beta
            ]
            W_k_list = _build_energy_weighted_density_open(
                result.mo_coeffs_alpha,
                result.mo_energies_alpha,
                result.mo_coeffs_beta,
                result.mo_energies_beta,
                n_alpha,
                n_beta,
            )
        per_k_D_total = [da + db for da, db in zip(per_k_Da, per_k_Db)]
        spin_per_k = (per_k_Da, per_k_Db)
    else:
        nocc = n_elec // 2
        if frac:
            per_k_D_total = _build_per_k_density_matrices_frac(
                result.mo_coeffs, result.occupations, n_k
            )
            W_k_list = _build_energy_weighted_density_closed_frac(
                result.mo_coeffs, result.mo_energies, result.occupations
            )
        else:
            per_k_D_total = _build_per_k_density_matrices(result.mo_coeffs, nocc, n_k)
            W_k_list = _build_energy_weighted_density_closed(
                result.mo_coeffs, result.mo_energies, nocc
            )
        spin_per_k = None

    m5_delta_f_k = None
    if sr_image_extent_bohr is not None:
        if open_shell:
            if frac:
                raise NotImplementedError(
                    "fractional-occupation padded multi-k open-shell "
                    "gradients remain gated"
                )
            if sr_spin_density is None:
                raise ValueError(
                    "_compute_bipole_gradient_corrected_multi_k: padded "
                    "open-shell traversal requires both spin densities"
                )
            # The ordinary UHF density combiner intentionally returns the
            # narrow overlap/Fock-output domain.  M5 J_SR and its density
            # adjoint need D_alpha + D_beta on the full padded BvK support.
            sr_density = _sum_lattice_densities_on_wide_support(
                sr_spin_density[0], sr_spin_density[1]
            )
            W_k_list, m5_delta_f_k = _m5_multi_k_open_w_adjoint(
                W_k_list,
                system,
                basis,
                lat,
                sr_density,
                sr_spin_density,
                (result.mo_coeffs_alpha, result.mo_coeffs_beta),
                (n_alpha, n_beta),
                kmesh,
                alpha_hf=a_hf,
                ewald_alpha=alpha,
                sr_image_extent_bohr=float(sr_image_extent_bohr),
            )
        else:
            if sr_density is None:
                raise ValueError(
                    "_compute_bipole_gradient_corrected_multi_k: padded SR "
                    "traversal requires the SCF real-space density support"
                )
            W_k_list, m5_delta_f_k = _m5_multi_k_closed_w_adjoint(
                W_k_list,
                system,
                basis,
                lat,
                sr_density,
                result.mo_coeffs,
                nocc,
                kmesh,
                alpha_hf=a_hf,
                ewald_alpha=alpha,
                sr_image_extent_bohr=float(sr_image_extent_bohr),
            )

    D_grad = density_set_from_k_matrices(system, basis, lat, kmesh, per_k_D_total)
    sr_internal_cells = (
        []
        if sr_image_extent_bohr is None
        else list(_sr_internal_cells(basis, system, lattice_opts, sr_image_extent_bohr))
    )
    if sr_image_extent_bohr is None:
        D_sr_total = D_grad
        sr_output_cells = []
    else:
        if sr_density is None:
            raise ValueError(
                "_compute_bipole_gradient_corrected_multi_k: padded SR "
                "traversal requires the SCF real-space density support"
            )
        D_sr_total = sr_density
        sr_output_cells = (sr_internal_cells if lat.pair_complete_1e else list(
            compute_overlap_lattice(basis, system, lat).cells
        ))

    # #575. The Ewald multi-k SCF energies (RHF/RKS/UHF/UKS and the RHF
    # single-Gamma molecular-limit branch) evaluate the Hartree term with the
    # exact analytic-FT J, v_G = 4 pi / G^2 over every G != 0
    # (ewald_composed.compute_j_ewald_3d_ft_{gamma,lattice}); every image
    # interaction of the cell density is in that energy and no real-space
    # erfc pass exists there. This assembly differentiates the same J as
    # J_SR(erfc, real space) + J_LR(erf, reciprocal) + jellium, the Ewald
    # split (Ewald, Ann. Phys. 369, 253 (1921), as used throughout
    # pbc_bipole_common), an identity that holds only when the erfc image
    # sum runs out to where erfc(alpha R) has decayed. With
    # alpha = 2.8 / V^(1/3) (crystal_default_ewald_alpha) the nearest image
    # sits at erfc(2.8) = 7.5e-5 whatever the cell size, so a box whose
    # density support (cutoff_bohr) holds only the home cell dropped the
    # image erfc Hartree derivative: H2/STO-3G in a 37.8 bohr box,
    # analytic - FD = -1.03e-5 Ha/bohr scaling as 1/L, of which the dropped
    # term measured +1.026e-5 (frozen-density FD of the image erfc Hartree
    # energy, +2.8e-5 Ha over 18 image cells). Size the lambda/sigma image
    # traversal of the J_SR derivative from alpha, exactly as #478 sized the
    # original V_ne point-charge bound (ewald_real_cutoff_for_alpha:
    # sqrt(-ln tol)/alpha,
    # tol = ewald_precision), and keep the density support as the output domain so the
    # K_SR contraction and the Fock-output cells are untouched. The
    # exchange needs no image ball: its image terms carry an AO overlap
    # across the cell length. BIPOLE routes keep their own M5 padded
    # traversal (sr_image_extent_bohr) and are not affected.
    j_sr_internal_cells = sr_internal_cells
    j_sr_output_cells = sr_output_cells
    if sr_image_extent_bohr is None and j_sr_alpha_image_ball:
        from .pbc_bipole_common import ewald_real_cutoff_for_alpha

        j_sr_radius = max(
            float(lat.cutoff_bohr),
            float(ewald_real_cutoff_for_alpha(alpha, ewald_precision)),
        )
        j_sr_internal_cells = list(direct_lattice_cells(system, j_sr_radius))
        j_sr_output_cells = list(D_grad.cells)

    grad = np.zeros((len(system.unit_cell), 3), dtype=np.float64)
    ewald_opts = _matching_ewald_options(
        system,
        lat,
        alpha,
        reciprocal_cutoff_bohr_inv=reciprocal_cutoff_bohr_inv,
        tolerance=ewald_precision,
    )
    grad += np.asarray(ewald_nuclear_repulsion_gradient(system, ewald_opts))
    # overlap-Lagrangian -- inverse-Bloch fold of the standard energy-weighted
    # density (the corrected-gauge KS/HF Fock eigenvalues are gauge-free).
    W_set = _bloch_fold_w_matrices(
        W_k_list, kmesh, compute_overlap_lattice(basis, system, lat)
    )
    grad += np.asarray(overlap_lattice_gradient_contribution(basis, system, W_set, lat))
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(basis, system, D_grad, lat)
    )
    grad += _v_ne_ewald_gradient(
        system,
        basis,
        D_grad,
        lat,
        alpha,
        reciprocal_cutoff_bohr_inv=reciprocal_cutoff_bohr_inv,
        precision=ewald_precision,
    )
    # FULL jellium (-pi N_e^2/(2w^2V) is quadratic in N_e=Tr[DS]; the W carries the
    # +v_bg per-orbital shift that supplies the cancelling overlap term).
    v_bg = -math.pi * float(n_elec) / (alpha * alpha * V_cell)
    D_bg = compute_overlap_lattice(basis, system, lat)
    for c in range(len(D_bg.cells)):
        D_bg.set_block(c, v_bg * np.asarray(D_grad.blocks[c], dtype=float))
    grad += -1.0 * np.asarray(
        overlap_lattice_gradient_contribution(basis, system, D_bg, lat)
    )

    # corrected-gauge exchange: K_SR(erfc) + per-q K_LR(recip) + per-k Madelung,
    # each on the gradient-template fold of the contracting density. Closed
    # shell: one total-density call. Open shell: 2.S_s dE_x[P_s] (the exact
    # closed-shell reduction at P_a=P_b=1/2P, the exchange kernels being quadratic).
    if abs(a_hf) > 1.0e-14:
        def _exchange_grad_multi_k(per_k_Dx, sr_density_override=None):
            Dx_grad = density_set_from_k_matrices(system, basis, lat, kmesh, per_k_Dx)
            Dx_sr = Dx_grad
            if sr_image_extent_bohr is not None:
                if sr_density_override is None:
                    raise ValueError(
                        "padded multi-k exchange requires its matching "
                        "real-space spin density"
                    )
                Dx_sr = sr_density_override
            gx = np.asarray(
                eri_lattice_gradient_contribution(
                    basis,
                    system,
                    Dx_sr,
                    lat,
                    a_hf,
                    0.0,
                    alpha,
                    False,
                    sr_internal_cells,
                    sr_output_cells,
                )
            )
            gx = gx + a_hf * _k_long_range_ewald_gradient_multi_k(
                system,
                basis,
                per_k_Dx,
                kmesh,
                alpha,
                lat,
                reciprocal_cutoff_bohr_inv=reciprocal_cutoff_bohr_inv,
                precision=ewald_precision,
            )
            gx = gx + a_hf * _madelung_ewald_gradient_multi_k(
                system, basis, per_k_Dx, kmesh, alpha, lat
            )
            return gx

        if spin_per_k is None:
            grad += _exchange_grad_multi_k(
                per_k_D_total,
                D_sr_total if sr_image_extent_bohr is not None else None,
            )
        else:
            for spin_idx, per_k_Ds in enumerate(spin_per_k):
                grad += 2.0 * _exchange_grad_multi_k(
                    per_k_Ds,
                    (
                        sr_spin_density[spin_idx]
                        if sr_spin_density is not None
                        else None
                    ),
                )
    # Coulomb: J_SR(erfc) + J^LR(reciprocal, multi-k) -- total density.
    grad += np.asarray(
        eri_lattice_gradient_contribution(
            basis,
            system,
            D_sr_total,
            lat,
            0.0,
            1.0,
            alpha,
            False,
            j_sr_internal_cells,
            j_sr_output_cells,
        )
    )
    grad += _j_long_range_ewald_gradient_multi_k(
        system,
        basis,
        D_grad,
        kmesh,
        alpha,
        per_k_D_total,
        reciprocal_cutoff_bohr_inv=reciprocal_cutoff_bohr_inv, lattice_opts=lattice_opts, precision=ewald_precision)
    if m5_delta_f_k is not None:
        if open_shell:
            grad += _multi_k_orbital_relaxation_open(
                system,
                basis,
                result.mo_coeffs_alpha,
                result.mo_energies_alpha,
                n_alpha,
                result.mo_coeffs_beta,
                result.mo_energies_beta,
                n_beta,
                kmesh,
                lat,
                alpha,
                a_hf,
                energy_fock_delta_k=m5_delta_f_k,
                sr_image_extent_bohr=float(sr_image_extent_bohr),
                sr_density_cells=list(sr_density.cells),
                ewald_precision=ewald_precision,
            )
        else:
            grad += _multi_k_orbital_relaxation_closed_diag(
                system,
                basis,
                result.mo_coeffs,
                result.mo_energies,
                nocc,
                kmesh,
                lat,
                alpha,
                energy_fock_delta_k=m5_delta_f_k,
                sr_image_extent_bohr=float(sr_image_extent_bohr),
                sr_density_cells=list(sr_density.cells),
                ewald_precision=ewald_precision,
            )
    return grad


def compute_stress_tensor(
    system: PeriodicSystem,
    gradient: np.ndarray,
) -> np.ndarray:
    """Force virial -- NOT the true periodic stress.

    .. warning::
       This is only the atomic-force virial
       ``s_{ij} = -(1/V).S_A R_{A,i}.F_{A,j}``. It is **not** the periodic
       stress ``(1/V).dE/de``: it omits the explicit lattice/Ewald strain
       dependence and the Gaussian-basis Pulay stress, and it assumes the
       atoms scale with the strain. On H₂/STO-3G it comes out *opposite in
       sign* to the true ``dE/de``. **Do not use it to relax the cell or
       report a stress.** No certified BIPOLE variable-cell optimizer is
       currently available: ``relax_cell``, ``relax_cell_gradient``, and
       ``relax_full`` fail closed. Use a converged equation-of-state scan or
       another validated code. This helper is retained only for the
       force-virial diagnostic.

    Computes the 3x3 force virial in Ha/bohr^3:
        s_{ij} = -(1/V) . S_A R_{A,i} . F_{A,j}

    where R_A are atomic positions and F_A = -dE/dR_A.

    Parameters
    ----------
    system : PeriodicSystem
        The periodic system (provides lattice + atomic positions).
    gradient : (n_atoms, 3) ndarray
        Atomic gradient in Ha/bohr (negative of forces).

    Returns
    -------
    ndarray shape (3, 3)
        Stress tensor in Ha/bohr^3.
    """
    lattice = np.asarray(system.lattice, dtype=float)
    V = float(abs(np.linalg.det(lattice)))
    if V < 1e-14:
        raise ValueError(f"Degenerate lattice (V={V})")
    grad = np.asarray(gradient, dtype=float)
    n_atoms = len(system.unit_cell)
    if grad.shape != (n_atoms, 3):
        raise ValueError(f"Gradient shape {grad.shape} != ({n_atoms}, 3)")
    stress = np.zeros((3, 3), dtype=float)
    for a in range(n_atoms):
        R = np.asarray(system.unit_cell[a].xyz, dtype=float)
        for i in range(3):
            for j in range(3):
                stress[i, j] -= R[i] * grad[a, j]
    stress /= V
    return stress


def _bloch_sum_density_per_k(
    D_real: LatticeMatrixSet,
    kmesh,
) -> List[np.ndarray]:
    """Bloch-sum a real-space :class:`LatticeMatrixSet` to a list of
    complex per-k density matrices ``P(k) = S_g e^{i k.g} P(g)``.

    Mirrors :func:`vibeqc.pbc_bipole._bloch_sum_blocks` (which takes a
    LatticeMatrixSet's blocks + cells directly) for each kpoint in
    ``kmesh``.
    """
    from .pbc_bipole import _bloch_sum_blocks

    P_k_list: List[np.ndarray] = []
    kpts = list(kmesh.kpoints)
    for k in kpts:
        k_arr = np.asarray(k, dtype=float)
        P_k = _bloch_sum_blocks(D_real.blocks, D_real.cells, k_arr)
        # Hermitise (the SCF result is exact, but defensive against
        # round-off).
        P_k = 0.5 * (P_k + P_k.conj().T)
        P_k_list.append(P_k)
    return P_k_list


def _add_dft_plus_u_pulay(
    grad: np.ndarray,
    *,
    system: PeriodicSystem,
    basis: BasisSet,
    sites: Sequence["object"],
    kmesh,
    S_k_list: Sequence[np.ndarray],
    P_total_k_list: Optional[Sequence[np.ndarray]] = None,
    P_alpha_k_list: Optional[Sequence[np.ndarray]] = None,
    P_beta_k_list: Optional[Sequence[np.ndarray]] = None,
    lattice_opts: LatticeSumOptions,
) -> np.ndarray:
    """Add the multi-k +U Pulay overlap-derivative contribution
    ``dE_U/dR|_C = 2 S_k w_k tr(V_AO_s S(k) P_s(k) dS(k)/dR)`` to a
    BIPOLE gradient. No-op if ``sites`` is empty.
    """
    if not sites:
        return grad
    from .dft_plus_u import (
        _compute_dft_plus_u_gradient_periodic_multi_k,
    )

    extra = _compute_dft_plus_u_gradient_periodic_multi_k(
        basis,
        system,
        sites,
        kmesh=kmesh,
        S_k_list=S_k_list,
        P_total_k_list=P_total_k_list,
        P_alpha_k_list=P_alpha_k_list,
        P_beta_k_list=P_beta_k_list,
        lattice_opts=lattice_opts,
    )
    return grad + np.asarray(extra, dtype=np.float64)


def compute_bipole_gradient_rhf(
    system: PeriodicSystem,
    basis: BasisSet,
    result: PBCBipoleRHFResult,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha_hf: float = 1.0,
    kmesh=None,
    dft_plus_u: Optional[Sequence["object"]] = None,
    cphf_rhs: str = "hybrid",
) -> np.ndarray:
    """BIPOLE RHF atomic gradient.

    ``cphf_rhs`` selects how the Bloch-CPHF orbital-relaxation right-hand side
    ``dB0/dR`` is evaluated (the Γ-only local-energy gauge needs it; see
    :func:`_bloch_cphf_relaxation`):

      * ``"hybrid"`` (default) -- analytic skeleton + analytic J^LR renorm, with
        the local (J_SR-1/2K) renormalisation taken exactly via 6N cheap
        ``J_SR``+``K`` builds. ~2x faster than the semi-numerical reference and
        matches it to ~3e-5 vs FD.
      * ``"analytic"`` -- fully analytic (no FD); ~6x faster but ~2e-3 vs FD
        (the reconstruction's lattice cutoff breaks the local-renorm 4-index
        symmetry). Use for fast pre-screening / large cells.
      * ``"seminumeric"`` -- the original 6N-full-Fock-build FD reference.

    ``dft_plus_u``: optional list of :class:`HubbardSite`. When set,
    adds the multi-k +U Pulay overlap-gradient term
    ``2 S_k w_k tr(V_AO_s S(k) P_s(k) dS(k)/dR)``. Requires
    ``kmesh=`` -- the same :class:`BlochKMesh` the SCF used.

    .. warning::
       Research preview -- maintained RHF/UHF Gamma and multi-k slices are
       finite-difference validated. Corrected-gauge padded radial multi-k
       RHF/UHF are included; other padded multi-k combinations remain gated. Use
       :func:`compute_bipole_gradient_fd` for production forces.
    """
    ewald_precision = float(getattr(result, "ewald_precision", 1e-8))
    _reject_m5_domain_analytic_gradient(result, "rhf")
    _warn_research_preview("rhf")
    if getattr(result, "exchange_ewald_split", False):
        # Corrected (Ewald-exchange-split) gauge. Gamma is a standard
        # variational HF gradient (full Bloch density -> no Bloch-CPHF);
        # the multi-k route also includes per-q K_LR and the supercell
        # Madelung term. Padded radial multi-k adds its finite-domain response.
        _ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
        if (
            len(result.mo_coeffs) == 1
            and _ewald_alpha is not None
            and _ewald_alpha > 0
            and system.dim == 3
        ):
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_rhf: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            n_elec = system.n_electrons()
            W_gamma = _build_energy_weighted_density_closed(
                result.mo_coeffs, result.mo_energies, n_elec // 2
            )[0]
            home = _home_cell_index(result.density.cells)
            D_home = np.asarray(result.density.blocks[home], dtype=np.float64)
            return _compute_bipole_gradient_corrected_gamma(
                system,
                basis,
                D_home,
                np.real(W_gamma),
                n_elec,
                lattice_opts=lattice_opts,
                alpha_hf=alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                **_m5_gamma_gradient_domain(
                    result,
                    system,
                    basis,
                    lattice_opts,
                    result.density.cells,
                ),
                ewald_precision=ewald_precision,
            )
        # Multi-k corrected gauge (n_k > 1): per-q reciprocal exchange (K_LR) +
        # per-k Madelung, with the density inverse-Bloch-folded onto the
        # gradient template. Reduces EXACTLY to the Γ core at n_k=1 (1.8e-18)
        # and is FD-clean at n_k=2 (6.3e-8). See
        # _compute_bipole_gradient_corrected_multi_k.
        if _ewald_alpha is not None and _ewald_alpha > 0 and system.dim == 3:
            if kmesh is None:
                raise ValueError(
                    "compute_bipole_gradient_rhf: the multi-k corrected-gauge "
                    "gradient requires kmesh= (the BlochKMesh the SCF used)."
                )
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_rhf: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            return _compute_bipole_gradient_corrected_multi_k(
                system,
                basis,
                result,
                kmesh,
                lattice_opts=lattice_opts,
                alpha_hf=alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                sr_image_extent_bohr=getattr(
                    result, "sr_image_extent_bohr", None
                ),
                sr_density=result.density,
                ewald_precision=ewald_precision,
            )
        raise ValueError(
            "compute_bipole_gradient_rhf: corrected-gauge analytic gradient "
            "needs a 3D system with a positive Ewald alpha (got "
            f"dim={system.dim}, ewald_alpha={_ewald_alpha}). Use "
            "compute_bipole_gradient_fd."
        )
    if not result.converged:
        warnings.warn(
            f"compute_bipole_gradient_rhf: result not converged "
            f"(n_iter={result.n_iter}). Gradient may be inaccurate."
        )
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
    n_elec = system.n_electrons()
    n_occ = n_elec // 2
    ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
    # Γ-only BIPOLE evaluates the energy as a LOCAL home-cell contraction,
    # so the Pulay energy-weighted density must use dE/dP(0) (NOT the
    # diagonalised Bloch F(Γ) eigenvalues -- see _corrected_w_gamma_closed).
    # Multi-k carries the real Bloch density (no locality projection), so
    # the standard mo_energy W + inverse-Bloch fold is used there.
    n_k = len(result.mo_coeffs)
    if n_k == 1 and ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        W_k_list = [
            _corrected_w_gamma_closed(
                system,
                basis,
                result.density,
                np.asarray(result.mo_coeffs[0]),
                n_occ,
                lattice_opts,
                float(ewald_alpha),
                alpha_hf,
                ewald_precision=ewald_precision,
            )
        ]
    elif (
        n_k > 1
        and kmesh is not None
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
    ):
        W_k_list = _corrected_w_multi_k_closed(
            system,
            basis,
            result.mo_coeffs,
            result.mo_energies,
            n_occ,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
        )
    else:
        W_k_list = _build_energy_weighted_density_closed(
            result.mo_coeffs, result.mo_energies, n_occ
        )
    per_k_jlr = None
    if n_k > 1 and ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        per_k_jlr = _build_per_k_density_matrices(result.mo_coeffs, n_occ, n_k)
    grad = _compute_bipole_gradient(
        system,
        basis,
        result.density,
        W_k_list,
        n_elec,
        lattice_opts=lattice_opts,
        alpha_hf=alpha_hf,
        ewald_alpha=ewald_alpha,
        kmesh=kmesh,
        per_k_jlr_densities=per_k_jlr,
        ewald_precision=ewald_precision,
    )
    # Local-energy orbital-relaxation (Bloch CPHF Z-vector) -- recovers the
    # orbital response the no-CPHF local-energy Pulay misses. Covers BOTH the
    # diagonalise-Bloch/contract-local F_scf mismatch (asymmetric multi-cell
    # crystals -- ~8e-2 Ha/bohr without it) AND the post-SCF spheropole
    # (asymmetric 1-cell -- ~1e-3). Zero by symmetry on symmetric cells.
    if n_k == 1 and ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        grad = grad + _bloch_cphf_relaxation(
            system,
            basis,
            result.density,
            np.asarray(result.mo_coeffs[0]),
            np.asarray(result.mo_energies[0]),
            n_occ,
            lattice_opts,
            float(ewald_alpha),
            alpha_hf,
            cphf_rhs=cphf_rhs,
            ewald_precision=ewald_precision,
        )
    elif (
        n_k > 1
        and kmesh is not None
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
    ):
        grad = grad + _multi_k_orbital_relaxation_closed_diag(
            system,
            basis,
            result.mo_coeffs,
            result.mo_energies,
            n_occ,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
            ewald_precision=ewald_precision,
        )
    if dft_plus_u:
        if kmesh is None:
            raise ValueError(
                "compute_bipole_gradient_rhf: dft_plus_u=[...] requires "
                "kmesh= (the BlochKMesh the SCF was run on)."
            )
        P_k_list = _bloch_sum_density_per_k(result.density, kmesh)
        S_k_list = list(result.overlap)
        grad = _add_dft_plus_u_pulay(
            grad,
            system=system,
            basis=basis,
            sites=dft_plus_u,
            kmesh=kmesh,
            S_k_list=S_k_list,
            P_total_k_list=P_k_list,
            lattice_opts=lattice_opts,
        )
    return grad


def compute_bipole_gradient_uhf(
    system: PeriodicSystem,
    basis: BasisSet,
    result: PBCBipoleUHFResult,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha_hf: float = 1.0,
    kmesh=None,
    dft_plus_u: Optional[Sequence["object"]] = None,
    cphf_rhs: str = "hybrid",
) -> np.ndarray:
    """BIPOLE UHF atomic gradient.

    See :func:`compute_bipole_gradient_rhf` for the ``dft_plus_u`` /
    ``kmesh`` / ``cphf_rhs`` kwargs -- ``cphf_rhs`` selects the coupled-spin
    Bloch-CPHF RHS method (``"hybrid"`` default / ``"analytic"`` /
    ``"seminumeric"``), same meaning as RHF.

    .. warning::
       Research preview -- RHF/UHF Γ-only hybrid gradients are certified
       against FD, but RKS/UKS and multi-k are not. Use
       :func:`compute_bipole_gradient_fd` for production forces.
    """
    ewald_precision = float(getattr(result, "ewald_precision", 1e-8))
    _reject_m5_domain_analytic_gradient(result, "uhf")
    _warn_research_preview("uhf")
    if getattr(result, "exchange_ewald_split", False):
        # Corrected (Ewald-exchange-split) gauge at Γ: standard variational
        # UHF gradient (no Bloch-CPHF), exchange spin-resolved. Multi-k
        # (per-k q-channel K_LR + supercell Madelung) is not yet wired.
        _ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
        if (
            len(result.mo_coeffs_alpha) == 1
            and _ewald_alpha is not None
            and _ewald_alpha > 0
            and system.dim == 3
        ):
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_uhf: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            n_elec = system.n_electrons()
            n_alpha = (n_elec + system.multiplicity - 1) // 2
            n_beta = (n_elec - system.multiplicity + 1) // 2
            W_gamma = _build_energy_weighted_density_open(
                result.mo_coeffs_alpha,
                result.mo_energies_alpha,
                result.mo_coeffs_beta,
                result.mo_energies_beta,
                n_alpha,
                n_beta,
            )[0]
            ha = _home_cell_index(result.density_alpha.cells)
            hb = _home_cell_index(result.density_beta.cells)
            Da_home = np.asarray(result.density_alpha.blocks[ha], dtype=np.float64)
            Db_home = np.asarray(result.density_beta.blocks[hb], dtype=np.float64)
            m5_domain = _m5_gamma_gradient_domain(
                result,
                system,
                basis,
                lattice_opts,
                result.density_alpha.cells,
            )
            if m5_domain:
                W_gamma = _m5_gamma_open_w_adjoint(
                    np.real(W_gamma),
                    system,
                    basis,
                    lattice_opts,
                    Da_home + Db_home,
                    (Da_home, Db_home),
                    (
                        np.asarray(result.mo_coeffs_alpha[0]),
                        np.asarray(result.mo_coeffs_beta[0]),
                    ),
                    (n_alpha, n_beta),
                    alpha_hf=alpha_hf,
                    ewald_alpha=float(_ewald_alpha),
                    sr_image_extent_bohr=m5_domain["sr_image_extent_bohr"],
                    sr_density_cells=m5_domain["sr_density_cells"],
                    sr_output_cells=m5_domain.get("sr_output_cells"),
                    sr_output_shell_masks=m5_domain.get(
                        "sr_output_shell_masks"
                    ),
                    sr_density_domain=m5_domain.get("sr_density_domain"),
                )
            return _compute_bipole_gradient_corrected_gamma(
                system,
                basis,
                Da_home + Db_home,
                np.real(W_gamma),
                n_elec,
                lattice_opts=lattice_opts,
                alpha_hf=alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                spin_home_blocks=(Da_home, Db_home),
                **m5_domain,
                ewald_precision=ewald_precision,
            )
        # Multi-k corrected gauge (n_k > 1): the shared multi-k core builds the
        # spin-resolved exchange (2.S_s dE_x[P_s]) from the open-shell result.
        if _ewald_alpha is not None and _ewald_alpha > 0 and system.dim == 3:
            if kmesh is None:
                raise ValueError(
                    "compute_bipole_gradient_uhf: the multi-k corrected-gauge "
                    "gradient requires kmesh= (the BlochKMesh the SCF used)."
                )
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_uhf: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            from .pbc_bipole_uhf import _combine_density_sets

            D_total = _combine_density_sets(
                basis,
                system,
                lattice_opts,
                result.density_alpha,
                result.density_beta,
            )
            return _compute_bipole_gradient_corrected_multi_k(
                system,
                basis,
                result,
                kmesh,
                lattice_opts=lattice_opts,
                alpha_hf=alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                sr_image_extent_bohr=getattr(
                    result, "sr_image_extent_bohr", None
                ),
                sr_density=D_total,
                sr_spin_density=(
                    result.density_alpha,
                    result.density_beta,
                ),
                ewald_precision=ewald_precision,
            )
        raise ValueError(
            "compute_bipole_gradient_uhf: corrected-gauge analytic gradient "
            "needs a 3D system with a positive Ewald alpha (got "
            f"dim={system.dim}, ewald_alpha={_ewald_alpha}). Use "
            "compute_bipole_gradient_fd."
        )
    if not result.converged:
        warnings.warn("compute_bipole_gradient_uhf: not converged")
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
    n_elec = system.n_electrons()
    n_alpha = (n_elec + system.multiplicity - 1) // 2
    n_beta = (n_elec - system.multiplicity + 1) // 2
    from .pbc_bipole_uhf import _combine_density_sets

    D_total = _combine_density_sets(
        basis, system, lattice_opts, result.density_alpha, result.density_beta
    )
    ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
    # Γ-only BIPOLE evaluates the energy as a LOCAL home-cell contraction,
    # so the per-spin Pulay density uses dE/dP_s(0) (NOT the diagonalised
    # Bloch F_s(Γ) eigenvalues -- see _corrected_w_gamma_open). Multi-k
    # carries the real Bloch density (no locality projection), so the
    # standard mo_energy W + inverse-Bloch fold is used there.
    n_k = len(result.mo_coeffs_alpha)
    if n_k == 1 and ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        W_k_list = [
            _corrected_w_gamma_open(
                system,
                basis,
                D_total,
                result.density_alpha,
                result.density_beta,
                np.asarray(result.mo_coeffs_alpha[0]),
                n_alpha,
                np.asarray(result.mo_coeffs_beta[0]),
                n_beta,
                lattice_opts,
                float(ewald_alpha),
                alpha_hf,
                ewald_precision=ewald_precision,
            )
        ]
    elif (
        n_k > 1
        and kmesh is not None
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
    ):
        W_k_list = _corrected_w_multi_k_open(
            system,
            basis,
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
            n_alpha,
            n_beta,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
        )
    else:
        W_k_list = _build_energy_weighted_density_open(
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
            n_alpha,
            n_beta,
        )
    per_k_jlr = None
    if n_k > 1 and ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        per_k_jlr = _build_per_k_density_matrices_open(
            result.mo_coeffs_alpha,
            result.mo_coeffs_beta,
            n_alpha,
            n_beta,
            n_k,
        )
    grad = _compute_bipole_gradient(
        system,
        basis,
        D_total,
        W_k_list,
        n_elec,
        lattice_opts=lattice_opts,
        alpha_hf=alpha_hf,
        ewald_alpha=ewald_alpha,
        kmesh=kmesh,
        D_alpha=result.density_alpha,
        D_beta=result.density_beta,
        per_k_jlr_densities=per_k_jlr,
        ewald_precision=ewald_precision,
    )
    # Local-energy orbital-relaxation (UHF coupled-spin Bloch CPHF Z-vector) --
    # recovers the diagonalise-Bloch/contract-local F_scf mismatch (asymmetric
    # multi-cell) + the post-SCF spheropole, per spin. Zero by symmetry on
    # symmetric cells; reduces to the RHF result on a closed-shell singlet.
    if n_k == 1 and ewald_alpha is not None and ewald_alpha > 0 and system.dim == 3:
        grad = grad + _bloch_cphf_relaxation_open(
            system,
            basis,
            D_total,
            result.density_alpha,
            result.density_beta,
            np.asarray(result.mo_coeffs_alpha[0]),
            np.asarray(result.mo_energies_alpha[0]),
            n_alpha,
            np.asarray(result.mo_coeffs_beta[0]),
            np.asarray(result.mo_energies_beta[0]),
            n_beta,
            lattice_opts,
            float(ewald_alpha),
            alpha_hf,
            cphf_rhs=cphf_rhs,
            ewald_precision=ewald_precision,
        )
    elif (
        n_k > 1
        and kmesh is not None
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
    ):
        grad = grad + _multi_k_orbital_relaxation_open(
            system,
            basis,
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
            n_alpha,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
            n_beta,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
            alpha_hf,
            ewald_precision=ewald_precision,
        )
    if dft_plus_u:
        if kmesh is None:
            raise ValueError(
                "compute_bipole_gradient_uhf: dft_plus_u=[...] requires "
                "kmesh= (the BlochKMesh the SCF was run on)."
            )
        Pa_k = _bloch_sum_density_per_k(result.density_alpha, kmesh)
        Pb_k = _bloch_sum_density_per_k(result.density_beta, kmesh)
        S_k_list = list(result.overlap)
        grad = _add_dft_plus_u_pulay(
            grad,
            system=system,
            basis=basis,
            sites=dft_plus_u,
            kmesh=kmesh,
            S_k_list=S_k_list,
            P_alpha_k_list=Pa_k,
            P_beta_k_list=Pb_k,
            lattice_opts=lattice_opts,
        )
    return grad


def _build_ks_grid(system, grid_options, use_periodic_becke, image_radius_bohr):
    """Build the DFT integration grid exactly as the periodic KS SCF does
    (``pbc_bipole_rks``): periodic Becke partition when ``use_periodic_becke``,
    else the molecular unit-cell grid. The analytic XC gradient must use the
    same grid the energy was integrated on."""
    from ._vibeqc_core import GridOptions, build_grid

    if grid_options is None:
        grid_options = GridOptions()
    if use_periodic_becke:
        from .periodic_grid import build_periodic_becke_grid

        return build_periodic_becke_grid(
            system,
            grid_options=grid_options,
            image_radius_bohr=float(image_radius_bohr),
        )
    return build_grid(system.unit_cell_molecule(), grid_options)


def _recenter_basis_on_periodic_system(
    basis: BasisSet, system: PeriodicSystem
) -> BasisSet:
    """Copy ``basis`` exactly while moving each shell to its owning atom.

    Displaced orbital-response and grid-motion builds must preserve imported,
    programmatic and optimized basis content. Reloading ``basis.name`` can
    fail for an in-memory basis or silently substitute library shells for a modified basis
    that retained a loadable name. ``BasisSet.shells()`` is the native
    round-trip representation and carries normalized coefficients, so only
    each shell origin changes here.
    """
    atoms = list(system.unit_cell)
    shells: list[ShellInfo] = []
    for shell in basis.shells():
        atom_index = int(shell.atom_index)
        if not 0 <= atom_index < len(atoms):
            raise ValueError(
                "_recenter_basis_on_periodic_system: shell atom_index "
                f"{atom_index} is outside the displaced unit cell."
            )
        shells.append(
            ShellInfo(
                atom_index,
                int(shell.l),
                bool(shell.pure),
                list(shell.exponents),
                list(shell.coefficients),
                [float(x) for x in atoms[atom_index].xyz],
            )
        )
    return BasisSet(
        system.unit_cell_molecule(), shells, str(basis.name), True
    )


def _periodic_xc_pulay_gradient(
    system, basis, density, functional_name, lattice_opts, grid, spin
):
    """Analytic periodic XC Pulay atomic gradient via the C++
    ``xc_lattice_gradient_contribution`` kernel. ``spin`` is 1 (RKS) -- the
    UKS path uses the open-shell kernel. LDA, GGA sigma-Pulay, and meta-GGA
    tau-Pulay terms are included."""
    from ._vibeqc_core import Functional, xc_lattice_gradient_contribution

    func = Functional(functional_name, spin)
    return np.asarray(
        xc_lattice_gradient_contribution(
            basis, system, grid, func, density, lattice_opts
        ),
        dtype=np.float64,
    )


def _periodic_xc_grid_motion_correction(
    system,
    basis,
    density,
    functional_name,
    lattice_opts,
    grid_options,
    use_periodic_becke,
    becke_image_radius_bohr,
    fixed_grid_gradient,
    *,
    step_bohr: float = 1e-3,
):
    """Fixed-density correction from moving the atom-centred KS grid.

    ``xc_lattice_gradient_contribution`` differentiates the AO basis on a
    fixed grid. The SCF/FD energy rebuilds the periodic Becke grid after each
    displacement, so add ``dE_xc(moving grid)/dR - dE_xc(fixed grid)/dR``.
    The fixed-grid derivative is the analytic kernel already added by the
    caller; the moving-grid derivative is a cheap central difference of the
    XC grid energy only, not a 6N SCF.
    """
    from ._vibeqc_core import Functional, build_xc_periodic

    n_atoms = len(system.unit_cell)
    h = float(step_bohr)
    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    func = Functional(functional_name, 1)
    moving = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            energies = []
            for sign in (+1.0, -1.0):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(displaced[a].xyz)
                xyz[d] += sign * h
                displaced[a] = Atom(displaced[a].Z, xyz)
                sd = PeriodicSystem(system.dim, lattice, displaced)
                sd.charge = system.charge
                sd.multiplicity = system.multiplicity
                bd = _recenter_basis_on_periodic_system(basis, sd)
                gd = _build_ks_grid(
                    sd,
                    grid_options,
                    use_periodic_becke,
                    becke_image_radius_bohr,
                )
                energies.append(
                    build_xc_periodic(
                        bd,
                        sd,
                        gd,
                        func,
                        density,
                        lattice_opts,
                    ).e_xc
                )
            moving[a, d] = (float(energies[0]) - float(energies[1])) / (2.0 * h)
    return moving - np.asarray(fixed_grid_gradient, dtype=np.float64)


def _periodic_xc_pulay_gradient_uks(
    system, basis, density_alpha, density_beta, functional_name, lattice_opts, grid
):
    """Analytic spin-polarized periodic XC Pulay atomic gradient.

    LDA, GGA sigma-Pulay, and meta-GGA tau-Pulay terms are included.
    """
    from ._vibeqc_core import Functional, xc_lattice_gradient_contribution_uks

    return np.asarray(
        xc_lattice_gradient_contribution_uks(
            basis,
            system,
            grid,
            Functional(functional_name, 2),
            density_alpha,
            density_beta,
            lattice_opts,
        ),
        dtype=np.float64,
    )


def _periodic_xc_grid_motion_correction_uks(
    system,
    basis,
    density_alpha,
    density_beta,
    functional_name,
    lattice_opts,
    grid_options,
    use_periodic_becke,
    becke_image_radius_bohr,
    fixed_grid_gradient,
    *,
    step_bohr: float = 1e-3,
):
    """Open-shell companion of ``_periodic_xc_grid_motion_correction``."""
    from ._vibeqc_core import Functional, build_xc_periodic_uks

    n_atoms = len(system.unit_cell)
    h = float(step_bohr)
    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    func = Functional(functional_name, 2)
    moving = np.zeros((n_atoms, 3), dtype=np.float64)
    for a in range(n_atoms):
        for d in range(3):
            energies = []
            for sign in (+1.0, -1.0):
                displaced = [Atom(at.Z, list(at.xyz)) for at in atoms]
                xyz = list(displaced[a].xyz)
                xyz[d] += sign * h
                displaced[a] = Atom(displaced[a].Z, xyz)
                sd = PeriodicSystem(system.dim, lattice, displaced)
                sd.charge = system.charge
                sd.multiplicity = system.multiplicity
                bd = _recenter_basis_on_periodic_system(basis, sd)
                gd = _build_ks_grid(
                    sd,
                    grid_options,
                    use_periodic_becke,
                    becke_image_radius_bohr,
                )
                energies.append(
                    build_xc_periodic_uks(
                        bd,
                        sd,
                        gd,
                        func,
                        density_alpha,
                        density_beta,
                        lattice_opts,
                    ).e_xc
                )
            moving[a, d] = (float(energies[0]) - float(energies[1])) / (2.0 * h)
    return moving - np.asarray(fixed_grid_gradient, dtype=np.float64)


def _ks_gradient_lattice_options(lattice_opts, use_periodic_becke, image_radius):
    """Match the SCF's periodic XC reach without mutating caller options."""
    if not use_periodic_becke:
        return lattice_opts
    from .pbc_bipole_common import _lattice_options_passthrough

    source = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    out = _lattice_options_passthrough(source)
    out.coulomb_method = source.coulomb_method
    out.becke_image_radius_bohr = float(image_radius)
    return out


def compute_bipole_gradient_rks(
    system: PeriodicSystem,
    basis: BasisSet,
    result: PBCBipoleRKSResult,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    kmesh=None,
    dft_plus_u: Optional[Sequence["object"]] = None,
    grid_options=None,
    use_periodic_becke: bool = True,
    becke_image_radius_bohr: float = 10.0,
) -> np.ndarray:
    """BIPOLE RKS (DFT) atomic gradient. Uses alpha_hf from functional.

    The Γ-only path includes the fixed-density **XC Pulay force** (analytic
    LDA/GGA fixed-grid AO Pulay plus a cheap central-difference correction for
    the moving atom-centred grid), the local-energy ``V_xc``-augmented
    ``dE/dP(0)`` Pulay density, and a semi-numerical KS Bloch-CPHF orbital
    relaxation using a finite-difference ``f_xc`` response.

    ``grid_options`` / ``use_periodic_becke`` / ``becke_image_radius_bohr``
    MUST match the SCF's DFT-grid settings (the analytic XC gradient is
    integrated on the same grid the energy used). The defaults match
    :class:`PeriodicKSOptions` (``use_periodic_becke=True``).

    See :func:`compute_bipole_gradient_rhf` for ``dft_plus_u`` /
    ``kmesh`` kwargs.

    .. warning::
       Research preview. Gamma-local RKS now includes the KS Bloch-CPHF
       orbital relaxation and is pinned on maintained LDA asymmetric cells.
       Corrected-gauge fractional occupations are supported on the historical
       radial and multi-k domains, while the padded M5 Gamma composition fails
       closed pending a variational finite-domain J/K operator. Broader KS
       certification remains open. Use
       :func:`compute_bipole_gradient_fd` for production forces.
    """
    ewald_precision = float(getattr(result, "ewald_precision", 1e-8))
    _reject_m5_domain_analytic_gradient(result, "rks")
    _warn_research_preview("rks")
    lattice_opts = _ks_gradient_lattice_options(
        lattice_opts, use_periodic_becke, becke_image_radius_bohr
    )
    if getattr(result, "exchange_ewald_split", False):
        # Corrected (Ewald-exchange-split) gauge at Γ: the standard
        # variational HF/KS gradient core + the XC Pulay; no Bloch-CPHF.
        # The exchange block scales by the functional's HF fraction
        # (0 for pure DFT). Multi-k still refuses.
        _ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
        _func_name = getattr(result, "functional", "")
        if (
            len(result.mo_coeffs) == 1
            and _ewald_alpha is not None
            and _ewald_alpha > 0
            and system.dim == 3
            and _func_name
        ):
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_rks: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            from ._vibeqc_core import Functional

            try:
                _alpha_hf = float(Functional(_func_name, 1).hf_exchange_fraction)
            except Exception:
                _alpha_hf = 0.0
            n_elec = system.n_electrons()
            # Fractional occupation (finite-T / smearing): the free-energy
            # gradient uses the same fractional D (result.density) and the
            # occupation-weighted W = S_i f_i e_i C_iC_i+.
            if _is_fractional_ks_occupation(result, "rks"):
                W_gamma = _build_energy_weighted_density_closed_frac(
                    result.mo_coeffs, result.mo_energies, result.occupations
                )[0]
            else:
                W_gamma = _build_energy_weighted_density_closed(
                    result.mo_coeffs, result.mo_energies, n_elec // 2
                )[0]
            home = _home_cell_index(result.density.cells)
            D_home = np.asarray(result.density.blocks[home], dtype=np.float64)
            grad = _compute_bipole_gradient_corrected_gamma(
                system,
                basis,
                D_home,
                np.real(W_gamma),
                n_elec,
                lattice_opts=lattice_opts,
                alpha_hf=_alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                **_m5_gamma_gradient_domain(
                    result,
                    system,
                    basis,
                    lattice_opts,
                    result.density.cells,
                ),
                ewald_precision=ewald_precision,
            )
            grid = _build_ks_grid(
                system, grid_options, use_periodic_becke, becke_image_radius_bohr
            )
            xc_fixed = _periodic_xc_pulay_gradient(
                system, basis, result.density, _func_name, lattice_opts, grid, 1
            )
            grad = grad + xc_fixed
            grad = grad + _periodic_xc_grid_motion_correction(
                system,
                basis,
                result.density,
                _func_name,
                lattice_opts,
                grid_options,
                use_periodic_becke,
                becke_image_radius_bohr,
                xc_fixed,
            )
            return grad
        # Multi-k corrected gauge: HF-ish part via the shared multi-k core
        # (exchange scaled by the functional's HF fraction; 0 for pure DFT) +
        # the multi-k XC Pulay (the same lattice kernel as Γ, on the Bloch-
        # folded result.density) + the grid-motion correction.
        if _ewald_alpha is not None and _ewald_alpha > 0 and system.dim == 3:
            if kmesh is None:
                raise ValueError(
                    "compute_bipole_gradient_rks: the multi-k corrected-gauge "
                    "gradient requires kmesh= (the BlochKMesh the SCF used)."
                )
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_rks: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            # Fractional occupation handled by the multi-k corrected core
            # (per-k fractional density + occupation-weighted W).
            from ._vibeqc_core import Functional

            try:
                _alpha_hf = float(Functional(_func_name, 1).hf_exchange_fraction)
            except Exception:
                _alpha_hf = 0.0
            grad = _compute_bipole_gradient_corrected_multi_k(
                system,
                basis,
                result,
                kmesh,
                lattice_opts=lattice_opts,
                alpha_hf=_alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                ewald_precision=ewald_precision,
            )
            grid = _build_ks_grid(
                system, grid_options, use_periodic_becke, becke_image_radius_bohr
            )
            xc_fixed = _periodic_xc_pulay_gradient(
                system, basis, result.density, _func_name, lattice_opts, grid, 1
            )
            grad = grad + xc_fixed
            grad = grad + _periodic_xc_grid_motion_correction(
                system,
                basis,
                result.density,
                _func_name,
                lattice_opts,
                grid_options,
                use_periodic_becke,
                becke_image_radius_bohr,
                xc_fixed,
            )
            return grad
        raise ValueError(
            "compute_bipole_gradient_rks: corrected-gauge analytic gradient "
            "needs a 3D system with a positive Ewald alpha (got "
            f"dim={system.dim}, ewald_alpha={_ewald_alpha}). Use "
            "compute_bipole_gradient_fd."
        )
    if not result.converged:
        warnings.warn("compute_bipole_gradient_rks: not converged")
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
    _reject_fractional_ks_analytic_gradient(result, "rks")
    _reject_multi_k_ks_analytic_gradient(result, "rks", kmesh)
    n_elec = system.n_electrons()
    n_occ = n_elec // 2
    # Extract HF exchange fraction from functional name stored in result
    alpha_hf = 0.0
    func_name = getattr(result, "functional", "")
    if func_name:
        from ._vibeqc_core import Functional

        try:
            alpha_hf = float(Functional(func_name, 1).hf_exchange_fraction)
        except Exception:
            alpha_hf = 0.0
    ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
    n_k = len(result.mo_coeffs)
    is_gamma_local = (
        n_k == 1
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
        and func_name
    )
    is_multik_ks = (
        not is_gamma_local
        and n_k > 1
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
        and func_name
    )
    grid = None
    vxc_home = None
    if is_gamma_local or is_multik_ks:
        from ._vibeqc_core import Functional, build_xc_periodic

        grid = _build_ks_grid(
            system, grid_options, use_periodic_becke, becke_image_radius_bohr
        )
    if is_gamma_local:
        vxc = build_xc_periodic(
            basis, system, grid, Functional(func_name, 1), result.density, lattice_opts
        )
        home = _home_cell_index(list(result.density.cells))
        vxc_home = np.asarray(vxc.V_xc.blocks[home], dtype=float)
    if is_gamma_local:
        W_k_list = [
            _corrected_w_gamma_closed(
                system,
                basis,
                result.density,
                np.asarray(result.mo_coeffs[0]),
                n_occ,
                lattice_opts,
                float(ewald_alpha),
                alpha_hf,
                extra_home_block=vxc_home,
                ewald_precision=ewald_precision,
            )
        ]
    elif is_multik_ks:
        # Multi-k KS: corrected W per k with spheropole + jellium (V_xc
        # is already in mo_energies from the SCF Fock diagonalisation).
        W_k_list = _corrected_w_multi_k_closed(
            system,
            basis,
            result.mo_coeffs,
            result.mo_energies,
            n_occ,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
        )
    else:
        W_k_list = _build_energy_weighted_density_closed(
            result.mo_coeffs, result.mo_energies, n_occ
        )
    # Multi-k per-k J^LR densities for the multi-k J^LR gradient convention.
    per_k_jlr = None
    if is_multik_ks and kmesh is not None:
        per_k_jlr = _build_per_k_density_matrices(result.mo_coeffs, n_occ, n_k)
    grad = _compute_bipole_gradient(
        system,
        basis,
        result.density,
        W_k_list,
        n_elec,
        lattice_opts=lattice_opts,
        alpha_hf=alpha_hf,
        ewald_alpha=ewald_alpha,
        kmesh=kmesh,
        per_k_jlr_densities=per_k_jlr,
        ewald_precision=ewald_precision,
    )
    # XC Pulay force (analytic V_xc gradient on periodic Becke grid).
    if is_gamma_local or is_multik_ks:
        xc_fixed = _periodic_xc_pulay_gradient(
            system, basis, result.density, func_name, lattice_opts, grid, 1
        )
        grad = grad + xc_fixed
        grad = grad + _periodic_xc_grid_motion_correction(
            system,
            basis,
            result.density,
            func_name,
            lattice_opts,
            grid_options,
            use_periodic_becke,
            becke_image_radius_bohr,
            xc_fixed,
        )
    if is_gamma_local:
        grad = grad + _bloch_cphf_relaxation_ks_closed(
            system,
            basis,
            result.density,
            np.asarray(result.mo_coeffs[0]),
            np.asarray(result.mo_energies[0]),
            n_occ,
            lattice_opts,
            float(ewald_alpha),
            alpha_hf,
            func_name,
            grid_options,
            use_periodic_becke,
            becke_image_radius_bohr,
            ewald_precision=ewald_precision,
        )
    elif is_multik_ks and kmesh is not None:
        grad = grad + _multi_k_orbital_relaxation_ks_closed_diag(
            system,
            basis,
            result.mo_coeffs,
            result.mo_energies,
            n_occ,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
            func_name,
            ewald_precision=ewald_precision,
        )
    if dft_plus_u:
        if kmesh is None:
            raise ValueError(
                "compute_bipole_gradient_rks: dft_plus_u=[...] requires "
                "kmesh= (the BlochKMesh the SCF was run on)."
            )
        P_k_list = _bloch_sum_density_per_k(result.density, kmesh)
        S_k_list = list(result.overlap)
        grad = _add_dft_plus_u_pulay(
            grad,
            system=system,
            basis=basis,
            sites=dft_plus_u,
            kmesh=kmesh,
            S_k_list=S_k_list,
            P_total_k_list=P_k_list,
            lattice_opts=lattice_opts,
        )
    return grad


def compute_bipole_gradient_uks(
    system: PeriodicSystem,
    basis: BasisSet,
    result: PBCBipoleUKSResult,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    kmesh=None,
    dft_plus_u: Optional[Sequence["object"]] = None,
    grid_options=None,
    use_periodic_becke: bool = True,
    becke_image_radius_bohr: float = 10.0,
) -> np.ndarray:
    """BIPOLE UKS (spin-DFT) atomic gradient.

    The Γ-only path includes the fixed-density **XC Pulay force** (analytic
    LDA/GGA fixed-grid AO Pulay plus a cheap central-difference correction for
    the moving atom-centred grid, using the per-spin local-energy
    ``V_xc,s``-augmented ``dE/dP_s(0)``) and a semi-numerical coupled-spin KS
    Bloch-CPHF orbital relaxation using a finite-difference spin-``f_xc``
    response.

    ``grid_options`` / ``use_periodic_becke`` / ``becke_image_radius_bohr``
    MUST match the SCF's DFT-grid settings (the analytic XC gradient is
    integrated on the same grid the energy used). The defaults match
    :class:`PeriodicKSOptions` (``use_periodic_becke=True``).

    See :func:`compute_bipole_gradient_rhf` for ``dft_plus_u`` /
    ``kmesh`` kwargs.

    .. warning::
       Research preview. Gamma-local UKS now includes the coupled-spin KS
       Bloch-CPHF orbital relaxation and is pinned on a maintained LDA
       asymmetric cell, but multi-k and finite-temperature/fractional-occupation
       KS analytic calls raise :class:`NotImplementedError`; broader KS
       certification remains open. Use
       :func:`compute_bipole_gradient_fd` for production forces.
    """
    ewald_precision = float(getattr(result, "ewald_precision", 1e-8))
    _reject_m5_domain_analytic_gradient(result, "uks")
    _warn_research_preview("uks")
    lattice_opts = _ks_gradient_lattice_options(
        lattice_opts, use_periodic_becke, becke_image_radius_bohr
    )
    if getattr(result, "exchange_ewald_split", False):
        # Corrected (Ewald-exchange-split) gauge at Γ: standard variational
        # UKS gradient (no Bloch-CPHF) -- spin-resolved exchange + per-spin
        # XC Pulay. Multi-k still refuses.
        _ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
        _func_name = getattr(result, "functional", "")
        if (
            len(result.mo_coeffs_alpha) == 1
            and _ewald_alpha is not None
            and _ewald_alpha > 0
            and system.dim == 3
            and _func_name
        ):
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_uks: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            from ._vibeqc_core import Functional

            try:
                _alpha_hf = float(Functional(_func_name, 2).hf_exchange_fraction)
            except Exception:
                _alpha_hf = 0.0
            n_elec = system.n_electrons()
            n_alpha = (n_elec + system.multiplicity - 1) // 2
            n_beta = (n_elec - system.multiplicity + 1) // 2
            # Fractional occupation (finite-T / smearing): free-energy gradient
            # with fractional per-spin D (result.density_alpha/beta) and the
            # occupation-weighted W = S_i f_i^a e_i^a C_iC_i+ + (b).
            if _is_fractional_ks_occupation(result, "uks"):
                W_gamma = _build_energy_weighted_density_open_frac(
                    result.mo_coeffs_alpha,
                    result.mo_energies_alpha,
                    result.occupations_alpha,
                    result.mo_coeffs_beta,
                    result.mo_energies_beta,
                    result.occupations_beta,
                )[0]
            else:
                W_gamma = _build_energy_weighted_density_open(
                    result.mo_coeffs_alpha,
                    result.mo_energies_alpha,
                    result.mo_coeffs_beta,
                    result.mo_energies_beta,
                    n_alpha,
                    n_beta,
                )[0]
            ha = _home_cell_index(result.density_alpha.cells)
            hb = _home_cell_index(result.density_beta.cells)
            Da_home = np.asarray(result.density_alpha.blocks[ha], dtype=np.float64)
            Db_home = np.asarray(result.density_beta.blocks[hb], dtype=np.float64)
            m5_domain = _m5_gamma_gradient_domain(
                result,
                system,
                basis,
                lattice_opts,
                result.density_alpha.cells,
            )
            if m5_domain:
                W_gamma = _m5_gamma_open_w_adjoint(
                    np.real(W_gamma),
                    system,
                    basis,
                    lattice_opts,
                    Da_home + Db_home,
                    (Da_home, Db_home),
                    (
                        np.asarray(result.mo_coeffs_alpha[0]),
                        np.asarray(result.mo_coeffs_beta[0]),
                    ),
                    (n_alpha, n_beta),
                    alpha_hf=_alpha_hf,
                    ewald_alpha=float(_ewald_alpha),
                    sr_image_extent_bohr=m5_domain["sr_image_extent_bohr"],
                    sr_density_cells=m5_domain["sr_density_cells"],
                    sr_output_cells=m5_domain.get("sr_output_cells"),
                    sr_output_shell_masks=m5_domain.get(
                        "sr_output_shell_masks"
                    ),
                    sr_density_domain=m5_domain.get("sr_density_domain"),
                )
            grad = _compute_bipole_gradient_corrected_gamma(
                system,
                basis,
                Da_home + Db_home,
                np.real(W_gamma),
                n_elec,
                lattice_opts=lattice_opts,
                alpha_hf=_alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                spin_home_blocks=(Da_home, Db_home),
                **m5_domain,
                ewald_precision=ewald_precision,
            )
            grid = _build_ks_grid(
                system, grid_options, use_periodic_becke, becke_image_radius_bohr
            )
            xc_fixed = _periodic_xc_pulay_gradient_uks(
                system,
                basis,
                result.density_alpha,
                result.density_beta,
                _func_name,
                lattice_opts,
                grid,
            )
            grad = grad + xc_fixed
            grad = grad + _periodic_xc_grid_motion_correction_uks(
                system,
                basis,
                result.density_alpha,
                result.density_beta,
                _func_name,
                lattice_opts,
                grid_options,
                use_periodic_becke,
                becke_image_radius_bohr,
                xc_fixed,
            )
            return grad
        # Multi-k corrected gauge: spin-resolved HF-ish part via the shared
        # multi-k core (auto-detects the open-shell result; exchange scaled by
        # the functional's HF fraction, 0 for pure DFT) + the per-spin multi-k
        # XC Pulay + grid-motion correction.
        if (
            _ewald_alpha is not None
            and _ewald_alpha > 0
            and system.dim == 3
            and _func_name
        ):
            if kmesh is None:
                raise ValueError(
                    "compute_bipole_gradient_uks: the multi-k corrected-gauge "
                    "gradient requires kmesh= (the BlochKMesh the SCF used)."
                )
            if not result.converged:
                warnings.warn(
                    f"compute_bipole_gradient_uks: result not converged "
                    f"(n_iter={result.n_iter}). Gradient may be inaccurate."
                )
            if lattice_opts is None:
                lattice_opts = LatticeSumOptions()
            # Fractional occupation handled by the multi-k corrected core
            # (per-spin per-k fractional density + occupation-weighted W).
            from ._vibeqc_core import Functional

            try:
                _alpha_hf = float(Functional(_func_name, 2).hf_exchange_fraction)
            except Exception:
                _alpha_hf = 0.0
            grad = _compute_bipole_gradient_corrected_multi_k(
                system,
                basis,
                result,
                kmesh,
                lattice_opts=lattice_opts,
                alpha_hf=_alpha_hf,
                ewald_alpha=float(_ewald_alpha),
                ewald_precision=ewald_precision,
            )
            grid = _build_ks_grid(
                system, grid_options, use_periodic_becke, becke_image_radius_bohr
            )
            xc_fixed = _periodic_xc_pulay_gradient_uks(
                system,
                basis,
                result.density_alpha,
                result.density_beta,
                _func_name,
                lattice_opts,
                grid,
            )
            grad = grad + xc_fixed
            grad = grad + _periodic_xc_grid_motion_correction_uks(
                system,
                basis,
                result.density_alpha,
                result.density_beta,
                _func_name,
                lattice_opts,
                grid_options,
                use_periodic_becke,
                becke_image_radius_bohr,
                xc_fixed,
            )
            return grad
        raise ValueError(
            "compute_bipole_gradient_uks: corrected-gauge analytic gradient "
            "needs a 3D system with a positive Ewald alpha and a functional "
            f"(got dim={system.dim}, ewald_alpha={_ewald_alpha}). Use "
            "compute_bipole_gradient_fd."
        )
    if not result.converged:
        warnings.warn("compute_bipole_gradient_uks: not converged")
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
    _reject_fractional_ks_analytic_gradient(result, "uks")
    _reject_multi_k_ks_analytic_gradient(result, "uks", kmesh)  # warns, not raises
    n_elec = system.n_electrons()
    n_alpha = (n_elec + system.multiplicity - 1) // 2
    n_beta = (n_elec - system.multiplicity + 1) // 2
    from .pbc_bipole_uhf import _combine_density_sets

    D_total = _combine_density_sets(
        basis,
        system,
        lattice_opts,
        result.density_alpha,
        result.density_beta,
    )

    # Extract HF exchange fraction from functional name
    alpha_hf = 0.0
    func_name = getattr(result, "functional", "")
    if func_name:
        from ._vibeqc_core import Functional

        try:
            alpha_hf = float(Functional(func_name, 2).hf_exchange_fraction)
        except Exception:
            alpha_hf = 0.0

    ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
    n_k = len(result.mo_coeffs_alpha)
    is_gamma_local = (
        n_k == 1
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
        and func_name
    )
    is_multik_ks = (
        not is_gamma_local
        and n_k > 1
        and ewald_alpha is not None
        and ewald_alpha > 0
        and system.dim == 3
        and func_name
    )
    grid = None
    vxc_alpha_home = None
    vxc_beta_home = None
    if is_gamma_local or is_multik_ks:
        from ._vibeqc_core import Functional as _Functional
        from ._vibeqc_core import build_xc_periodic_uks

        grid = _build_ks_grid(
            system, grid_options, use_periodic_becke, becke_image_radius_bohr
        )
    if is_gamma_local:
        func_s = _Functional(func_name, 2)
        xc = build_xc_periodic_uks(
            basis,
            system,
            grid,
            func_s,
            result.density_alpha,
            result.density_beta,
            lattice_opts,
        )
        home = _home_cell_index(list(D_total.cells))
        vxc_alpha_home = np.asarray(xc.V_alpha.blocks[home], dtype=float)
        vxc_beta_home = np.asarray(xc.V_beta.blocks[home], dtype=float)

    if is_gamma_local:
        W_k_list = [
            _corrected_w_gamma_open(
                system,
                basis,
                D_total,
                result.density_alpha,
                result.density_beta,
                np.asarray(result.mo_coeffs_alpha[0]),
                n_alpha,
                np.asarray(result.mo_coeffs_beta[0]),
                n_beta,
                lattice_opts,
                float(ewald_alpha),
                alpha_hf,
                extra_home_block_alpha=vxc_alpha_home,
                extra_home_block_beta=vxc_beta_home,
                ewald_precision=ewald_precision,
            )
        ]
    elif is_multik_ks:
        W_k_list = _corrected_w_multi_k_open(
            system,
            basis,
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
            n_alpha,
            n_beta,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
        )
    else:
        W_k_list = _build_energy_weighted_density_open(
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
            n_alpha,
            n_beta,
        )

    per_k_jlr = None
    if is_multik_ks and kmesh is not None:
        per_k_jlr = _build_per_k_density_matrices_open(
            result.mo_coeffs_alpha,
            result.mo_coeffs_beta,
            n_alpha,
            n_beta,
            n_k,
        )

    grad = _compute_bipole_gradient(
        system,
        basis,
        D_total,
        W_k_list,
        n_elec,
        lattice_opts=lattice_opts,
        alpha_hf=alpha_hf,
        ewald_alpha=ewald_alpha,
        kmesh=kmesh,
        D_alpha=result.density_alpha,
        D_beta=result.density_beta,
        per_k_jlr_densities=per_k_jlr,
        ewald_precision=ewald_precision,
    )

    # XC Pulay force (analytic per-spin V_xc gradient).
    if is_gamma_local or is_multik_ks:
        xc_fixed = _periodic_xc_pulay_gradient_uks(
            system,
            basis,
            result.density_alpha,
            result.density_beta,
            func_name,
            lattice_opts,
            grid,
        )
        grad = grad + xc_fixed
        grad = grad + _periodic_xc_grid_motion_correction_uks(
            system,
            basis,
            result.density_alpha,
            result.density_beta,
            func_name,
            lattice_opts,
            grid_options,
            use_periodic_becke,
            becke_image_radius_bohr,
            xc_fixed,
        )
    if is_gamma_local:
        grad = grad + _bloch_cphf_relaxation_ks_open(
            system,
            basis,
            D_total,
            result.density_alpha,
            result.density_beta,
            np.asarray(result.mo_coeffs_alpha[0]),
            np.asarray(result.mo_energies_alpha[0]),
            n_alpha,
            np.asarray(result.mo_coeffs_beta[0]),
            np.asarray(result.mo_energies_beta[0]),
            n_beta,
            lattice_opts,
            float(ewald_alpha),
            alpha_hf,
            func_name,
            grid_options,
            use_periodic_becke,
            becke_image_radius_bohr,
            ewald_precision=ewald_precision,
        )
    elif is_multik_ks and kmesh is not None:
        grad = grad + _multi_k_orbital_relaxation_ks_open_diag(
            system,
            basis,
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
            n_alpha,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
            n_beta,
            kmesh,
            lattice_opts,
            float(ewald_alpha),
            func_name,
            ewald_precision=ewald_precision,
        )

    if dft_plus_u:
        if kmesh is None:
            raise ValueError(
                "compute_bipole_gradient_uks: dft_plus_u=[...] requires "
                "kmesh= (the BlochKMesh the SCF was run on)."
            )
        Pa_k = _bloch_sum_density_per_k(result.density_alpha, kmesh)
        Pb_k = _bloch_sum_density_per_k(result.density_beta, kmesh)
        S_k_list = list(result.overlap)
        grad = _add_dft_plus_u_pulay(
            grad,
            system=system,
            basis=basis,
            sites=dft_plus_u,
            kmesh=kmesh,
            S_k_list=S_k_list,
            P_alpha_k_list=Pa_k,
            P_beta_k_list=Pb_k,
            lattice_opts=lattice_opts,
        )
    return grad


def compute_bipole_gradient_fd(
    system: PeriodicSystem,
    basis_name: str | BasisSet,
    kmesh,
    options=None,
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    step_bohr: float = 1e-3,
    require_converged: bool = True,
    **bipole_kwargs,
) -> np.ndarray:
    """Central-difference BIPOLE gradient via repeated SCF.

    This is the **exact, production** BIPOLE gradient: it
    central-differences the real total energy, so every gauge piece
    (Ewald ``E_nn`` / ``V_ne``, ``J_SR`` + ``J_LR``, exchange,
    spheropole) is differentiated consistently. Correct in the limit
    ``step_bohr -> 0``; cost is ``6N + ...`` full SCFs (vs. one for the
    analytic path). By default every displaced SCF point must converge;
    otherwise the function raises instead of differentiating a failed SCF
    iterate. Prefer this over :func:`compute_bipole_gradient_rhf` et al. for
    production forces; the analytic drivers are still a narrow/hybrid
    research preview.

    Parameters
    ----------
    system : PeriodicSystem
        Reference geometry.
    basis_name : str or BasisSet
        Name to load once, or the actual reference basis. Displacements
        preserve its shells while moving them with their owning atoms.
    kmesh : BlochKMesh
        k-point mesh.
    options : PeriodicRHFOptions / PeriodicKSOptions, optional
        SCF options.
    method : str
        ``"RHF"``, ``"UHF"``, ``"RKS"`` or ``"UKS"`` (default ``"RHF"``).
    functional : str, optional
        XC functional name for the ``"RKS"`` / ``"UKS"`` methods.
    step_bohr : float
        Half-step for central difference (default 1e-3 bohr).
    require_converged : bool, default True
        If True, raise ``RuntimeError`` when any displaced SCF result reports
        ``converged=False``. Set False only for failure-surface diagnostics.
    **bipole_kwargs
        Forwarded to the ``run_pbc_bipole_*`` driver (``ewald_precision``,
        ``use_ewald_j_split``, ...).

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr.
    """
    from ._vibeqc_core import Atom, attach_symmetry
    from ._vibeqc_core import BasisSet as _BasisSet

    method_upper = method.upper()

    def _run_displaced(sys_disp, basis_disp, displacement_label: str) -> float:
        if method_upper == "RHF":
            from .pbc_bipole import run_pbc_bipole_rhf

            res = run_pbc_bipole_rhf(
                sys_disp,
                basis_disp,
                kmesh,
                options,
                progress=False,
                **bipole_kwargs,
            )
        elif method_upper == "UHF":
            from .pbc_bipole_uhf import run_pbc_bipole_uhf

            res = run_pbc_bipole_uhf(
                sys_disp,
                basis_disp,
                kmesh,
                options,
                progress=False,
                **bipole_kwargs,
            )
        elif method_upper == "RKS":
            from .pbc_bipole_rks import run_pbc_bipole_rks

            res = run_pbc_bipole_rks(
                sys_disp,
                basis_disp,
                kmesh,
                options,
                functional=functional,
                progress=False,
                **bipole_kwargs,
            )
        elif method_upper == "UKS":
            from .pbc_bipole_uks import run_pbc_bipole_uks

            res = run_pbc_bipole_uks(
                sys_disp,
                basis_disp,
                kmesh,
                options,
                functional=functional,
                progress=False,
                **bipole_kwargs,
            )
        else:
            raise ValueError(
                f"compute_bipole_gradient_fd: unknown method {method!r} "
                "(expected RHF, UHF, RKS or UKS)"
            )
        if require_converged and not bool(getattr(res, "converged", True)):
            n_iter = getattr(res, "n_iter", "unknown")
            energy = getattr(res, "energy", None)
            energy_txt = "unknown" if energy is None else f"{float(energy):.12g}"
            raise RuntimeError(
                "compute_bipole_gradient_fd: "
                f"{method_upper} SCF did not converge for {displacement_label} "
                f"(n_iter={n_iter}, energy={energy_txt}). Refusing to "
                "finite-difference a non-converged energy; loosen the SCF "
                "options or pass require_converged=False for diagnostics."
            )
        # Finite-temperature (Fermi-smeared) KS: the production atomic force is
        # the Mermin FREE-energy derivative -dA/dR (A = E_total - T.S), the
        # variationally-consistent finite-T force (Wentzcovitch 1992; Marzari).
        # ``res.energy`` is the bare total energy E_total, whose derivative
        # carries the non-variational occupation-response term T.dS/dR; FD over
        # E_total would therefore NOT match the analytic free-energy gradient.
        # At T = 0 (and for HF, which rejects smearing) free_energy == energy,
        # so integer-occupation runs are unaffected.
        if float(getattr(res, "smearing_temperature", 0.0) or 0.0) > 0.0:
            return float(getattr(res, "free_energy", res.energy))
        return float(res.energy)

    n_atoms = len(system.unit_cell)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    template = (_BasisSet(system.unit_cell_molecule(), basis_name)
                if isinstance(basis_name, str) else basis_name)

    for a in range(n_atoms):
        for cart in range(3):
            e_plus = None
            e_minus = None
            for sign in (+1, -1):
                delta = sign * step_bohr
                new_atoms = []
                for i, atom in enumerate(system.unit_cell):
                    xyz = list(atom.xyz)
                    if i == a:
                        xyz[cart] += delta
                    new_atoms.append(Atom(int(atom.Z), xyz))
                sys_disp = PeriodicSystem(
                    system.dim,
                    np.asarray(system.lattice, dtype=np.float64),
                    new_atoms,
                    charge=system.charge,
                    multiplicity=system.multiplicity,
                )
                # Preserve the caller's attached-symmetry calculation mode.
                # Re-detect at each displaced geometry instead of copying the
                # reference operations: an individual Cartesian displacement
                # can lower the point group, but even its valid C1 operation
                # keeps the group-invariant pair-resolved Fock domain active.
                # Dropping symmetry entirely switched every displaced M5
                # point back to the radial domain, so the "FD" compared two
                # different energy surfaces (most visibly for UHF/KS).
                symmetry = getattr(system, "symmetry", None)
                if getattr(symmetry, "operations", None):
                    attach_symmetry(sys_disp)
                basis_disp = _recenter_basis_on_periodic_system(
                    template, sys_disp,
                )
                coord = "xyz"[cart]
                sign_label = "+" if sign > 0 else "-"
                label = f"atom {a}, coord {coord}, step {sign_label}{step_bohr:g} bohr"
                e = _run_displaced(sys_disp, basis_disp, label)
                if sign == +1:
                    e_plus = e
                else:
                    e_minus = e
            grad[a, cart] = (e_plus - e_minus) / (2.0 * step_bohr)

    return grad
