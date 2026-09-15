"""Phase G1a -- analytic Γ-only periodic RHF / RKS atomic gradient.

Closed-shell extension of :func:`vibeqc.compute_gradient` (molecular)
to a periodic system. Assembles the four contributions to the per-
cell gradient:

  F_A = dE_nn^pc/dR_A
      + S_g  tr(D(g) . dh_core(g)/dR_A)              (1-e Pulay)
      - S_g  tr(W(g) . dS(g)/dR_A)                   (overlap-Lagrangian)
      + S_g S_h S_h'  Γ(g, h, h') . d(muν0|lh sh')/dR_A  (2-e Pulay)

**Validation status (G1a series).**

Molecular limit (cell list reduces to a single cell, ``cutoff_bohr <
inter-image distance``):

  - All four contributions match the molecular gradient bit-for-bit
    (<= 1e-7 Ha/bohr); validated against
    :func:`vibeqc.compute_gradient` on H₂ in 20-Å cubic box.
  - 27x speedup vs the FD reference at this regime.

True periodic (cells overlap; cross-cell density / ERIs significant):

  - **Nuclear-rep + 1-e Pulay + overlap-Lagrangian**: matches FD to
    Newton's-3rd-law precision (1e-13). These three are the bulk of
    the gradient on systems with negligible a_HF.K (pure DFT -- LDA,
    PBE, BLYP, ...).
  - **2-e J piece** (exchange_scale = 0): matches FD to N3-precision
    (~1e-13). This is what pure-DFT periodic gradients use.
  - **Full HF/hybrid gradient** (a_HF != 0): refused for true-periodic
    DIRECT_TRUNCATED cells. The 2026-06-13/30 diagnosis (1D H chain,
    a=2 Å, STO-3G, DIRECT_TRUNCATED) showed a remaining, cutoff-oscillatory
    HF K residual after the periodic density-contraction fix. Returning that
    force as a production analytic gradient is unsafe, so the public driver
    fails closed unless the overlap lattice has reduced to the molecular
    limit. Low-level K-contraction regressions stay active in the test suite.

So today this module produces:

  - ✅ Exact analytic gradients in the **molecular limit** (large cells)
    for HF, pure DFT, and hybrids (see G1b / the gauge-free overlap-
    Lagrangian below).
  - ✅ Exact pure-DFT true-periodic gradients (a_HF = 0; J + 1-e + overlap
    to N3 precision).
  - ⚠️  HF/hybrid **true-periodic** DIRECT_TRUNCATED gradients are gated:
    use :func:`compute_gradient_periodic_rhf_fd` (slow but tied to the
    same energy route) for diagnostics, or a validated Ewald/GDF gradient
    route when available.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicRHFResult,
    PeriodicSystem,
    build_jk_gamma_molecular_limit,
    compute_kinetic,
    compute_nuclear,
    compute_overlap_lattice,
    eri_lattice_gradient_contribution,
    kinetic_lattice_gradient_contribution,
    nuclear_lattice_gradient_contribution,
    nuclear_repulsion_gradient_per_cell,
    overlap_lattice_gradient_contribution,
    two_electron_gradient_contribution,
)


__all__ = ["compute_gradient_periodic_rhf_gamma"]


def _fold_gamma_real(M_set: LatticeMatrixSet) -> np.ndarray:
    """S_g M(g) -- the Γ-point Bloch fold (k = 0 -> all cells equally
    weighted). For a single-cell mesh this returns the g=0 block;
    for a true multi-cell list, sums them."""
    nbf = int(M_set.nbf)
    out = np.zeros((nbf, nbf), dtype=np.float64)
    for blk in M_set.blocks:
        out += np.asarray(blk, dtype=np.float64)
    return out


def _gamma_density_lattice_set(
    template: LatticeMatrixSet, D: np.ndarray, homogeneous: bool = False
) -> LatticeMatrixSet:
    """Build a Γ-only lattice-resolved density on ``template``'s cell list.

    Two conventions, selected by ``homogeneous``:

    * ``False`` (molecular limit): home-cell block ``D``, image blocks 0.
      Correct when the cell is large enough that only the unit-cell density
      is non-negligible.
    * ``True`` (true-periodic Γ): ``D(g) = D_Γ`` in **every** cell. At a
      single Γ k-point the k=0 Bloch phase is 1 in every image, so the
      real-space density is identical in every cell -- and that is the
      density the SCF energy is built from, so the analytic gradient must
      use it too. Using the molecular-limit (home-only) convention for a
      true-periodic cell was the G1a-2 bug (1D H chain, a=2 Å: the analytic
      gradient sat ~0.22 Ha/bohr off FD vs ~0.099 with the homogeneous
      density).

    Multi-k generalisation (G1c): replace this with the proper
    ``real_space_density_from_kpoints`` call so D(g) carries its correct
    k-dependent Bloch phase (the residual on the homogeneous-Γ convention
    is the SCF<->gradient lattice-sum consistency the multi-k path closes).
    """
    D_arr = np.asarray(D, dtype=np.float64)
    zero = np.zeros_like(D_arr)
    if homogeneous:
        for c in range(len(template.cells)):
            template.set_block(c, D_arr)
        return template
    for c, cell in enumerate(template.cells):
        idx = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
        template.set_block(c, D_arr if idx == (0, 0, 0) else zero)
    return template


def compute_gradient_periodic_rhf_gamma(
    system: PeriodicSystem,
    basis: BasisSet,
    result: PeriodicRHFResult,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha_hf: float = 1.0,
    dft_plus_u: Optional[Sequence["HubbardSite"]] = None,
    variational_xc_fock: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Analytic Γ-only periodic RHF atomic gradient.

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    result
        Converged :class:`PeriodicRHFResult` from
        :func:`vibeqc.run_rhf_periodic` (must have ``converged=True``).
    lattice_opts
        :class:`LatticeSumOptions` controlling the lattice-sum
        cutoffs. If ``None``, defaults from :class:`LatticeSumOptions()`.
        For the gradient to match the SCF energy gradient, these
        cutoffs **must** match the values used during the SCF (i.e.
        the same ``opts.lattice_opts`` you passed to
        ``run_rhf_periodic``).
    variational_xc_fock
        Optional ``(nbf, nbf)`` converged XC potential matrix ``V_xc``
        in the AO basis. Supplied by the RKS caller so the energy-
        weighted density is built from the *full* variational KS Fock
        ``F_var = T + V_ne + J - 1/2.a_HF.K + V_xc`` (see Notes on the
        gauge-free overlap-Lagrangian). ``None`` (the default) is the
        pure-HF case, where ``V_xc = 0``.

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr.

    Notes
    -----
    **Current scope (G1a-1)**: 1-electron Pulay + nuclear-repulsion +
    overlap-Lagrangian terms via the new lattice-summed C++
    primitives. The 2-electron Pulay term falls back to the molecular
    code path on the Γ-folded total density. This is exact in the
    molecular limit (single cell, AO overlap between cells negligible)
    and approximate for truly periodic systems where cross-cell ERIs
    contribute. **G1a-2** will replace the 2-e fallback with the
    full lattice-summed periodic ERI gradient.

    **Gauge-free overlap-Lagrangian.** The EWALD_3D Γ driver reports
    ``mo_energies`` / ``fock`` on a shifted orbital-energy reference
    (the real-space exact-exchange G=0 self-image; a_HF-scaled, cell-
    size-independent -- pure DFT is unaffected). Using those eigenvalues
    in the energy-weighted density ``W = 2 S_i e_i C_mui C_νi`` injects a
    spurious ``-Δ.tr(D dS/dR)`` force. In the molecular limit this
    function instead rebuilds the variational Fock ``F_var = dE/dD`` from
    the converged density and takes ``e`` from it, so the overlap term is
    gauge-consistent with the SCF energy (and the FD reference).
    """
    if not result.converged:
        raise ValueError(
            "compute_gradient_periodic_rhf_gamma: PeriodicRHFResult "
            "is not converged."
        )
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()

    # Fetch the Γ-folded RHF reference data.
    D_gamma = np.asarray(result.density, dtype=np.float64)
    C = np.asarray(result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(result.mo_energies, dtype=np.float64)
    F_result = np.asarray(getattr(result, "fock", np.empty((0, 0))), dtype=np.float64)
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "compute_gradient_periodic_rhf_gamma: open-shell RHF not "
            "supported (closed-shell only). Use the periodic UHF "
            "gradient driver (G1d, post-G1a) for open-shell systems."
        )
    nocc = n_elec // 2

    # Wrap D_gamma into a LatticeMatrixSet (cell layout matches the SCF)
    # and detect the molecular limit (single home cell within cutoff). The
    # density convention then follows the regime: molecular-limit -> home-only
    # D(g!=0)=0; true-periodic Γ -> homogeneous D(g)=D_Γ (the density the SCF
    # energy is built from). See ``_gamma_density_lattice_set``.
    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    home_cell_only = all(
        tuple(int(v) for v in np.asarray(cell.index).reshape(3)) == (0, 0, 0)
        for cell in D_set.cells
    )
    if not home_cell_only and abs(float(alpha_hf)) > 1.0e-14:
        raise ValueError(
            "compute_gradient_periodic_rhf_gamma: true-periodic "
            "DIRECT_TRUNCATED HF/hybrid analytic gradients are not "
            "validated because the bare-Coulomb exchange Pulay residual "
            "is cutoff dependent. Use compute_gradient_periodic_rhf_fd "
            "for this route, or a validated Ewald/GDF periodic gradient. "
            "Pure-DFT periodic gradients (alpha_hf=0) remain supported."
        )
    _gamma_density_lattice_set(D_set, D_gamma, homogeneous=not home_cell_only)

    # Energy-weighted density W = 2 S_i e_i C_mui C_νi for the overlap-
    # Lagrangian (Pulay) term -- the ONLY gradient contribution that
    # consumes orbital energies.
    #
    # The EWALD_3D Γ driver reports mo_energies / fock on a *shifted*
    # orbital-energy reference: the real-space exact-exchange carries a
    # G=0 self-image gauge that raises occupied (and lowers virtual)
    # orbital energies by an a_HF-scaled, cell-size-independent constant
    # (≈ +0.106 Ha on H₂/STO-3G HF; exactly 0 for pure DFT, a_HF=0). The
    # density and total energy are gauge-invariant, but W is not -- the
    # offset injects a spurious -Δ.tr(D dS/dR) force (H₂/STO-3G/20-Å box
    # HF: analytic 0.1927 vs FD 0.1486 Ha/bohr). The fix is to build W
    # from the *variational* Fock F_var = dE/dD rebuilt from the
    # converged density (gauge-free by construction), not from the
    # driver's reported eigenvalues. In the molecular limit
    #   F_var = T + V_ne + J(D) - 1/2.a_HF.K(D) (+ V_xc),
    # where V_xc is threaded in from the RKS caller (None => pure HF).
    # True-periodic (cross-cell) F_var is G1a-2 scope; its gradient is
    # separately xfailed, so it stays on the driver eigenvalues. DFT+U
    # also stays on the driver eigenvalues (which already carry the
    # variational +U shift, see compute_gradient docstring).
    #
    # The rebuild only fires when F_var can be assembled in full: pure HF
    # (no functional on the result => V_xc = 0), or a caller that supplies
    # the converged V_xc via ``variational_xc_fock``. A KS result with no
    # V_xc supplied keeps the driver eigenvalues -- correct for pure DFT,
    # whose exchange-free F carries no G=0 self-image gauge (offset = 0).
    _rebuild_w = (
        home_cell_only
        and dft_plus_u is None
        and (variational_xc_fock is not None
             or not hasattr(result, "functional"))
    )
    if _rebuild_w:
        mol_cell = system.unit_cell_molecule()
        h_core = (np.asarray(compute_kinetic(basis), dtype=np.float64)
                  + np.asarray(compute_nuclear(basis, mol_cell), dtype=np.float64))
        jk = build_jk_gamma_molecular_limit(
            basis, system, lattice_opts, D_gamma, 0.0)
        f_var = (h_core + np.asarray(jk.J, dtype=np.float64)
                 - 0.5 * float(alpha_hf) * np.asarray(jk.K, dtype=np.float64))
        if variational_xc_fock is not None:
            f_var = f_var + np.asarray(variational_xc_fock, dtype=np.float64)
        eps_for_w = np.einsum("ui,uv,vi->i", C, f_var, C, optimize=True)
    elif F_result.shape == (C.shape[0], C.shape[0]):
        # Some periodic result paths stored pre-Fock orbital energies;
        # recover e_i = C_i^T F C_i from the converged Fock when present.
        eps_for_w = np.einsum("ui,uv,vi->i", C, F_result, C, optimize=True)
    else:
        eps_for_w = eps
    W_gamma = 2.0 * (C[:, :nocc] * eps_for_w[:nocc][None, :]) @ C[:, :nocc].T

    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    _gamma_density_lattice_set(W_set, W_gamma, homogeneous=not home_cell_only)

    # Sum the four contributions.
    grad = np.zeros((len(system.unit_cell), 3), dtype=np.float64)

    grad += np.asarray(
        nuclear_repulsion_gradient_per_cell(system, lattice_opts))
    grad += np.asarray(
        overlap_lattice_gradient_contribution(
            basis, system, W_set, lattice_opts))
    grad += np.asarray(
        kinetic_lattice_gradient_contribution(
            basis, system, D_set, lattice_opts))
    grad += np.asarray(
        nuclear_lattice_gradient_contribution(
            basis, system, D_set, lattice_opts))

    if home_cell_only:
        grad += np.asarray(
            two_electron_gradient_contribution(
                basis,
                system.unit_cell_molecule(),
                D_gamma,
                float(alpha_hf),
            )
        )
    else:
        # 2-electron Pulay -- full lattice-summed periodic ERI gradient.
        # alpha_hf=1 -> plain HF (J - 1/2 K). alpha_hf=0 -> pure DFT (J only).
        # Hybrids inherit the K-piece bug (G1a-2 patch).
        grad += np.asarray(
            eri_lattice_gradient_contribution(
                basis, system, D_set, lattice_opts, float(alpha_hf)))

    # DFT+U Pulay overlap-gradient contribution -- Γ-only periodic only.
    # The orbital-response piece is captured by the W-Lagrangian term
    # above (the converged e already includes the +U shift via the
    # variational ``V_U_fock = S V_AO S``); what we add here is the
    # explicit ``2 tr(V_AO_s S P_s dS/dR)`` at fixed C.
    if dft_plus_u:
        from .dft_plus_u import _compute_dft_plus_u_gradient_periodic_gamma
        # Recover the home-cell overlap S from D_set (the cell list is
        # the same and the home-cell index is (0,0,0)).
        S_set = compute_overlap_lattice(basis, system, lattice_opts)
        S_gamma = None
        for c, cell in enumerate(S_set.cells):
            idx = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
            if idx == (0, 0, 0):
                S_gamma = np.asarray(S_set.blocks[c], dtype=np.float64)
                break
        if S_gamma is None:
            raise RuntimeError(
                "compute_gradient_periodic_rhf_gamma: home cell missing "
                "from overlap lattice -- internal invariant violation."
            )
        grad += np.asarray(
            _compute_dft_plus_u_gradient_periodic_gamma(
                basis, system, dft_plus_u,
                S_gamma=S_gamma,
                P_total_gamma=D_gamma,
                lattice_opts=lattice_opts,
            )
        )

    return grad
