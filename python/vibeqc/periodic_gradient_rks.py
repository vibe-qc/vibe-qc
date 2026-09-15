"""Phase G1b -- analytic Γ-only periodic RKS atomic gradient.

Closed-shell-DFT extension of :func:`vibeqc.compute_gradient_periodic_rhf_gamma`
(G1a). Adds the **XC Pulay** term to the four contributions assembled
there:

  F_A = dE_nn^pc/dR_A
      + S_g  tr(D(g) . dh_core(g)/dR_A)              (1-e Pulay)
      - S_g  tr(W(g) . dS(g)/dR_A)                   (overlap-Lagrangian)
      + S_g S_h S_h'  Γ(g, h, h') . d(muν0|lh sh')/dR  (2-e J + a_HF.K Pulay)
      + S_g  dE_xc[r_D]/dR_A                          (XC Pulay)

**Validation status (G1b series)**

  - **Molecular-limit periodic (LDA / GGA / hybrids)**: matches the
    molecular analytic gradient -- ~1e-9 (LDA / PBE), ~1e-14 (B3LYP /
    PBE0) on H₂ / H₂O in a 20-Å box. The XC Pulay is the lattice-
    summed periodic primitive ``xc_lattice_gradient_contribution``
    (full LDA + GGA s piece), and the overlap-Lagrangian uses the
    gauge-free variational KS Fock eigenvalues -- the threaded
    ``V_xc`` removes the a_HF-scaled exchange-G=0 self-image gauge
    that otherwise corrupts hybrids (see :mod:`vibeqc.periodic_gradient`
    *Gauge-free overlap-Lagrangian*).

  - **True periodic (cells overlap)**: the XC Pulay is the same
    lattice-summed primitive (no longer a molecular fallback), so
    pure DFT (a_HF = 0) is exact. Hybrids' HF-exchange **K** Pulay
    path is refused for DIRECT_TRUNCATED tight cells until a validated
    true-periodic exchange gradient route lands.

Use cases that work today via this module:

  - Periodic LDA / GGA / hybrid geometry optimization in molecular-
    limit unit cells -- surfaces with vacuum padding, defects in large
    supercells, molecular crystals -- exact.
  - Pure-DFT tight-packed bulk -- exact XC + J Pulay; hybrid bulk is
    gated on this DIRECT_TRUNCATED Γ path and should use finite
    differences or a validated Ewald/GDF gradient route.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    Functional,
    GridOptions,
    LatticeSumOptions,
    PeriodicKSResult,
    PeriodicSystem,
)
from .periodic_gradient import compute_gradient_periodic_rhf_gamma


__all__ = ["compute_gradient_periodic_rks_gamma"]


def _gamma_density_set(basis, system, result, lattice_opts):
    """Γ-only density as a :class:`LatticeMatrixSet` (home cell = D, image
    blocks zero) -- the periodic-XC input shared by the lattice XC Pulay
    gradient and the variational-V_xc rebuild.

    ``result.density`` is either an ``(nbf, nbf)`` array (Γ-only result) or
    a :class:`LatticeMatrixSet` (multi-k); both expose the g=0 block.
    """
    from ._vibeqc_core import compute_overlap_lattice

    if hasattr(result.density, "blocks"):
        D = None
        for c in range(len(result.density.cells)):
            if tuple(int(v) for v in result.density.cells[c].index) == (0, 0, 0):
                D = np.asarray(result.density.blocks[c], dtype=np.float64)
                break
    else:
        D = np.asarray(result.density, dtype=np.float64)

    d_set = compute_overlap_lattice(basis, system, lattice_opts)
    zero = np.zeros_like(D)
    for c, cell in enumerate(d_set.cells):
        idx = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
        d_set.set_block(c, D if idx == (0, 0, 0) else zero)
    return d_set


def compute_gradient_periodic_rks_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    result: PeriodicKSResult,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_options: Optional[GridOptions] = None,
    dft_plus_u: Optional[Sequence["HubbardSite"]] = None,
) -> np.ndarray:
    """Analytic Γ-only periodic RKS atomic gradient (closed-shell DFT).

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    result
        Converged :class:`PeriodicKSResult` from
        :func:`vibeqc.run_rks_periodic` (must have ``converged=True``).
    lattice_opts
        :class:`LatticeSumOptions` for the lattice-sum cutoffs. Must
        match the values used during the SCF for the gradient to
        equal the energy gradient.
    grid_options
        :class:`GridOptions` controlling the DFT quadrature grid.
        Must match the SCF grid for the XC Pulay term to be consistent
        with the SCF energy.

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr.

    Notes
    -----
    **Assembly.**

    1. The Hartree-Fock-ish part (nuclear-rep + 1-e Pulay + overlap-
       Lagrangian + 2-e J/K Pulay) is computed via
       :func:`compute_gradient_periodic_rhf_gamma` with the Hartree-
       fraction set by the functional (a_HF = 0 for pure DFT; a_HF != 0
       for hybrids). The converged ``V_xc`` is threaded in so the
       overlap-Lagrangian uses the gauge-free variational KS Fock
       eigenvalues (see that function's *Gauge-free overlap-Lagrangian*
       note). Hybrid true-periodic DIRECT_TRUNCATED gradients fail
       closed in the shared RHF helper; the molecular limit is exact.
    2. The XC Pulay term uses the lattice-summed periodic primitive
       :func:`xc_lattice_gradient_contribution` (libxc on the periodic
       Becke grid + analytic d_c chi), exact for LDA / GGA / hybrid.
       Combined with (1) the **molecular-limit** RKS gradient matches
       the molecular analytic gradient to ~1e-9 (LDA / PBE) and ~1e-14
       (B3LYP / PBE0).
    """
    if not result.converged:
        raise ValueError(
            "compute_gradient_periodic_rks_gamma: PeriodicKSResult "
            "is not converged."
        )
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
    if grid_options is None:
        grid_options = GridOptions()

    # The Hartree-Fock-ish part reuses run_rhf_periodic-style data with the
    # functional's a_HF for the J/K weights (PeriodicKSResult duck-types
    # through). The periodic Becke grid + Γ density set are shared by the
    # XC Pulay gradient and the variational V_xc below.
    from ._vibeqc_core import (
        Functional,
        build_xc_periodic,
        xc_lattice_gradient_contribution,
    )
    from .periodic_grid import build_periodic_becke_grid

    func = Functional(result.functional)
    # The analytic-gradient exchange Pulay term is full-range only; a
    # screened-K SCF (hse06) would silently get full-range forces.
    from .periodic_screened_exchange import reject_unscreened_range_separated

    reject_unscreened_range_separated(
        func, where="compute_gradient_periodic_rks_gamma"
    )
    alpha_hf = func.hf_exchange_fraction
    grid = build_periodic_becke_grid(system, grid_options=grid_options)
    d_set = _gamma_density_set(basis, system, result, lattice_opts)

    # Gauge-free overlap-Lagrangian: thread the converged V_xc (Bloch-folded
    # to Γ) so the kernel builds the energy-weighted density from the full
    # variational KS Fock eigenvalues. Skipped for DFT+U, where the kernel
    # keeps the driver eigenvalues (which carry the variational +U shift).
    # Pure DFT is a no-op (V_xc restores the eigenvalues the driver already
    # reports, offset = 0); hybrids get the a_HF-scaled exchange-G=0
    # self-image gauge removed.
    variational_xc_fock = None
    if dft_plus_u is None:
        v_xc_set = build_xc_periodic(
            basis, system, grid, func, d_set, lattice_opts).V_xc
        variational_xc_fock = sum(
            np.asarray(b, dtype=np.float64) for b in v_xc_set.blocks)

    grad_hf_part = compute_gradient_periodic_rhf_gamma(
        system, basis, result, lattice_opts=lattice_opts,
        alpha_hf=alpha_hf,
        dft_plus_u=dft_plus_u,
        variational_xc_fock=variational_xc_fock,
    )

    # XC Pulay gradient via the lattice-summed periodic primitive (libxc on
    # the periodic Becke grid + analytic d_c chi). Exact for LDA / GGA /
    # hybrid; replaces the former molecular LDA-only fallback that skipped
    # the GGA s-piece. With the gauge-free overlap-Lagrangian above, the
    # molecular-limit RKS gradient now matches the molecular analytic
    # gradient to ~1e-9 (LDA/PBE) and ~1e-14 (B3LYP/PBE0).
    grad_xc = xc_lattice_gradient_contribution(
        basis, system, grid, func, d_set, lattice_opts)

    return np.asarray(grad_hf_part) + np.asarray(grad_xc)
