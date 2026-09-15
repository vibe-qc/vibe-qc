"""Phase G1d -- open-shell (UHF / UKS) periodic atomic gradient.

Per-spin extension of :func:`vibeqc.compute_gradient_periodic_rhf_multi_k`
(G1c) and :func:`vibeqc.compute_gradient_periodic_rks_multi_k`.

For closed-shell RHF / RKS, the lattice-summed gradient primitives
take the closed-shell density ``D = 2 C_occ C_occ+`` and the
closed-shell W = ``2 S_i e_i C_i C_i+``. For UHF / UKS the analogues
are:

  D_total(g) = D_a(g) + D_b(g)
  W_total(g) = W_a(g) + W_b(g)

with each spin's D_s(g), W_s(g) Bloch-folded from the per-k C_s(k),
e_s(k) using the same convention as G1c (real-space density via
``real_space_density_from_kpoints``, energy-weighted density via the
``_bloch_fold_w_per_k`` helper).

The 1-electron + nuclear-rep + overlap pieces consume D_total / W_total
unchanged (closed-shell-like). The 2-electron J piece also consumes
D_total. The 2-electron K piece needs per-spin densities -- and the
existing ``eri_lattice_gradient_contribution(... alpha_hf=...)``
contracts with the *closed-shell* Γ formula, which is wrong for UHF
(the spin-summed K is ``D_a K(D_a) + D_b K(D_b)``, not ``D_total K(D_total)``).

So today this module ships:

  - **Pure-DFT UKS** (LDA, PBE, BLYP -- a_HF = 0): 3D Ewald jobs reuse
    the same energy-matched Ewald electrostatic derivative as the SCF;
    non-Ewald jobs retain the D_total lattice path below.
  - **UHF / hybrid UKS** (a_HF != 0): not exposed yet. The corrected-
    Ewald derivative core is spin resolved, but this public surface keeps
    its existing fail-closed contract pending dedicated hybrid regressions.

Validation status:

  - **Pure-DFT UKS multi-k** matches the RKS multi-k path bit-for-bit
    when the SCF reduces to closed-shell (e.g. zero spin polarization).
  - **Spin-polarized cases** are FD-validated (no analytic-vs-analytic
    cross-check yet -- the molecular UKS-gradient reference + FD
    reference both agree on small periodic test systems).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    Functional,
    GridOptions,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    compute_overlap_lattice,
    eri_lattice_gradient_contribution,
    kinetic_lattice_gradient_contribution,
    nuclear_lattice_gradient_contribution,
    nuclear_repulsion_gradient_per_cell,
    overlap_lattice_gradient_contribution,
)
from .periodic_gradient_multi_k import _bloch_fold_w_per_k


__all__ = [
    "compute_gradient_periodic_uks_multi_k",
]


def _fold_density_total(
    D_alpha_set: LatticeMatrixSet, D_beta_set: LatticeMatrixSet,
    template: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Build D_total(g) = D_a(g) + D_b(g) on the cell list of
    ``template``. Both input sets must share the same cell list.

    Mutates and returns ``template``."""
    if len(D_alpha_set.cells) != len(D_beta_set.cells):
        raise ValueError(
            "_fold_density_total: a and b density cell lists differ.")
    if len(D_alpha_set.cells) != len(template.cells):
        raise ValueError(
            "_fold_density_total: density cell lists differ from "
            "template.")
    for c in range(len(template.cells)):
        D_a = np.asarray(D_alpha_set.blocks[c], dtype=np.float64)
        D_b = np.asarray(D_beta_set.blocks[c], dtype=np.float64)
        template.set_block(c, D_a + D_b)
    return template


def _compute_uks_xc_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    result,
    lattice_opts: LatticeSumOptions,
    grid_options: GridOptions,
) -> np.ndarray:
    """Spin-resolved periodic XC Pulay contribution."""
    from ._vibeqc_core import xc_lattice_gradient_contribution_uks
    from .periodic_grid import build_periodic_becke_grid

    grid = build_periodic_becke_grid(system, grid_options=grid_options)
    func_uks = Functional(result.functional, 2)
    return np.asarray(
        xc_lattice_gradient_contribution_uks(
            basis,
            system,
            grid,
            func_uks,
            result.density_alpha,
            result.density_beta,
            lattice_opts,
        )
    )


def compute_gradient_periodic_uks_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    result,
    kmesh: BlochKMesh,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_options: Optional[GridOptions] = None,
) -> np.ndarray:
    """Multi-k open-shell-DFT periodic atomic gradient.

    Parameters
    ----------
    system, basis, kmesh
        Periodic system, AO basis, k-mesh -- same as for the closed-
        shell drivers.
    result
        Converged :class:`PeriodicUKSMultiKEwaldResult`. Must expose
        ``converged``, ``functional`` (string), per-spin
        ``mo_coeffs_alpha`` / ``mo_coeffs_beta`` (each a list of
        per-k complex (nbf, nbf) arrays), per-spin
        ``mo_energies_alpha`` / ``mo_energies_beta`` (lists of per-k
        real (nbf,) arrays), and per-spin ``density_alpha`` /
        ``density_beta`` LatticeMatrixSets.
    lattice_opts, grid_options
        Lattice-sum and DFT-grid options; must match the SCF.

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr.

    Notes
    -----
    **Scope (G1d-1, this commit)**: pure-DFT UKS only (a_HF = 0).
    Hybrid UKS remains fail closed pending dedicated full-range and screened-
    exchange regressions. Fractional-occupation UKS remains fail closed until
    the result retains the per-k spin occupations needed by the Pulay density.
    """
    if not result.converged:
        raise ValueError(
            "compute_gradient_periodic_uks_multi_k: result is not converged."
        )
    from .periodic_gradient_multi_k import (
        _resolve_gradient_lattice_opts,
        _validate_corrected_ewald_gradient_mesh,
    )

    lattice_opts = _resolve_gradient_lattice_opts(
        system,
        result,
        lattice_opts,
        where="compute_gradient_periodic_uks_multi_k",
    )
    if grid_options is None:
        grid_options = GridOptions()

    func = Functional(result.functional)
    alpha_hf = func.hf_exchange_fraction
    if alpha_hf != 0.0:
        # The corrected core is spin resolved, but screened/range-separated
        # exchange needs its own derivative mapping and global hybrids still
        # need dedicated public-surface FD regressions. Keep the existing
        # hybrid contract fail closed in this focused pure-DFT repair.
        raise NotImplementedError(
            f"compute_gradient_periodic_uks_multi_k: hybrid functional "
            f"'{result.functional}' (a_HF = {alpha_hf}) remains gated "
            f"pending full-range and screened-exchange analytic-gradient "
            f"regressions. Use a pure-DFT functional (LDA, PBE, BLYP, ...) "
            f"or the finite-difference gradient driver."
        )
    uses_fractional_occupations = (
        float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0
        or getattr(result, "bz_integration", None) == "gilat"
    )
    if uses_fractional_occupations:
        raise NotImplementedError(
            "compute_gradient_periodic_uks_multi_k: fractional-occupation "
            "analytic gradients require the per-k alpha/beta occupations "
            "used by the SCF, which this result surface does not retain; "
            "use the finite-difference gradient driver."
        )

    ewald_alpha = float(getattr(result, "omega", 0.0) or 0.0)
    is_corrected_ewald = (
        int(system.dim) == 3
        and lattice_opts.coulomb_method == CoulombMethod.EWALD_3D
        and ewald_alpha > 0.0
    )
    if is_corrected_ewald:
        _validate_corrected_ewald_gradient_mesh(kmesh)
        from .bipole_gradient import _compute_bipole_gradient_corrected_multi_k
        from .periodic_corrected_exchange import (
            corrected_ewald_reciprocal_cutoff,
        )

        volume = float(
            abs(np.linalg.det(np.asarray(system.lattice, dtype=np.float64)))
        )
        reciprocal_cutoff = corrected_ewald_reciprocal_cutoff(
            volume, ewald_alpha
        )
        grad = _compute_bipole_gradient_corrected_multi_k(
            system,
            basis,
            result,
            kmesh,
            lattice_opts=lattice_opts,
            alpha_hf=0.0,
            ewald_alpha=ewald_alpha,
            reciprocal_cutoff_bohr_inv=reciprocal_cutoff,
            # The SCF's J is the exact analytic-FT Hartree matrix (#575).
            j_sr_alpha_image_ball=True,
        )
        return np.asarray(grad) + _compute_uks_xc_gradient(
            system,
            basis,
            result,
            lattice_opts,
            grid_options,
        )

    # Per-spin densities are already Bloch-folded on the SCF side.
    D_alpha_set = result.density_alpha
    D_beta_set = result.density_beta

    # D_total(g) = D_a(g) + D_b(g) -- what the closed-shell-ERI primitives
    # see. Build by populating a fresh template's blocks.
    template = compute_overlap_lattice(basis, system, lattice_opts)
    D_total_set = _fold_density_total(D_alpha_set, D_beta_set, template)

    # W_total(g) = W_a(g) + W_b(g) -- Bloch-folded per spin then summed.
    # Build per-spin W on its own template, sum into a final set.
    n_k = len(kmesh.kpoints)
    n_alpha = _count_alpha_electrons(system, result)
    n_beta = system.n_electrons() - n_alpha
    occ_alpha_per_k = [n_alpha] * n_k
    occ_beta_per_k = [n_beta] * n_k

    template_W_a = compute_overlap_lattice(basis, system, lattice_opts)
    W_alpha_set = _bloch_fold_w_per_k(
        result.mo_coeffs_alpha, result.mo_energies_alpha,
        occ_alpha_per_k, kmesh, template_W_a,
    )
    template_W_b = compute_overlap_lattice(basis, system, lattice_opts)
    W_beta_set = _bloch_fold_w_per_k(
        result.mo_coeffs_beta, result.mo_energies_beta,
        occ_beta_per_k, kmesh, template_W_b,
    )

    # The W bloch-fold helper bakes in the closed-shell factor 2 (the
    # n_occ-int branch multiplies by 2 inside W_k). For UKS we want
    # *single-spin* W, no factor 2. Compensate by halving each set.
    for c in range(len(W_alpha_set.cells)):
        W_alpha_set.set_block(c, 0.5 * np.asarray(W_alpha_set.blocks[c]))
        W_beta_set.set_block(c, 0.5 * np.asarray(W_beta_set.blocks[c]))

    # Sum into W_total.
    template_W_total = compute_overlap_lattice(basis, system, lattice_opts)
    for c in range(len(template_W_total.cells)):
        W_a = np.asarray(W_alpha_set.blocks[c], dtype=np.float64)
        W_b = np.asarray(W_beta_set.blocks[c], dtype=np.float64)
        template_W_total.set_block(c, W_a + W_b)
    W_total_set = template_W_total

    # Now sum the four contributions through the existing C++
    # primitives. These all consume the closed-shell-shaped density /
    # W matrices, which is what D_total / W_total are.
    grad = np.zeros((len(system.unit_cell), 3), dtype=np.float64)
    grad += np.asarray(
        nuclear_repulsion_gradient_per_cell(system, lattice_opts))
    grad += np.asarray(
        overlap_lattice_gradient_contribution(
            basis, system, W_total_set, lattice_opts))
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(
            basis, system, D_total_set, lattice_opts))
    grad += np.asarray(
        nuclear_lattice_gradient_contribution(
            basis, system, D_total_set, lattice_opts))
    # 2-e J with a_HF = 0 (pure DFT -- no K piece).
    grad += np.asarray(
        eri_lattice_gradient_contribution(
            basis, system, D_total_set, lattice_opts, 0.0))

    # XC Pulay via the open-shell lattice-summed periodic primitive (full
    # LDA + GGA s-piece) -- the spin-polarized companion of the kernel the
    # Γ-only RKS driver uses. The per-spin Bloch-folded densities are
    # already real-space D_s(g) on the SCF cell list, the convention
    # ``xc_lattice_gradient_contribution_uks`` consumes. Replaces the former
    # molecular-grid fallback that dropped the GGA s-coupled piece and the
    # image-cell density.
    grad_xc = _compute_uks_xc_gradient(
        system,
        basis,
        result,
        lattice_opts,
        grid_options,
    )

    return np.asarray(grad) + grad_xc


def _count_alpha_electrons(system, result) -> int:
    """Resolve the a-electron count from the periodic system + the
    UKS result. Falls back to (N_e + 2S) / 2 = (N_e + (mult-1)) / 2
    if the result doesn't expose it directly."""
    if hasattr(result, "n_alpha"):
        return int(result.n_alpha)
    n_elec = system.n_electrons()
    # Multiplicity 2S+1 -> 2S = mult - 1 = N_a - N_b.
    # N_a = (N_e + 2S) / 2.
    two_s = system.multiplicity - 1
    n_alpha = (n_elec + two_s) // 2
    return n_alpha
