"""Phase G1c -- multi-k periodic RHF / RKS atomic gradient.

Multi-k extension of :func:`vibeqc.compute_gradient_periodic_rhf_gamma`
(G1a) and :func:`vibeqc.compute_gradient_periodic_rks_gamma` (G1b).

The Γ-only path stores a single density block ``D(0) = C.C+`` and
replicates it across every cell in the lattice list. Multi-k extends
this to a proper Bloch-folded real-space density:

  D(g) = S_k w_k Re[ exp(-i k.g) . 2.C_occ(k).C_occ(k)+ ]
  W(g) = S_k w_k Re[ exp(-i k.g) . 2.C_occ(k).diag(e_occ(k)).C_occ(k)+ ]

(D conjugate matches ``real_space_density_from_kpoints``'s convention.)

The G1a lattice-summed gradient primitives don't care whether the
density was built Γ-only or multi-k -- they consume any
``LatticeMatrixSet`` and return per-atom gradients. So G1c is a
thin Python wrapper that:

  1. Reuses ``result.density`` directly (already a Bloch-folded
     ``LatticeMatrixSet`` from
     :func:`real_space_density_from_kpoints`).
  2. Builds W(g) inline from per-k C and e.
  3. Wires both into the existing G1a/G1b lattice-summed primitives.

**Validation status (G1c)**

  - 3D Ewald RHF and global-hybrid gradients reuse the energy-matched
    Ewald derivative assembly: Ewald nuclear / electron-nuclear / Hartree,
    short-range exchange, reciprocal exchange, and the ``G=0`` correction.
    The RHF single-Γ compatibility branch is treated separately because its
    SCF deliberately uses molecular full-range exchange.
  - Each finite k mesh is validated against the derivative of its own SCF
    energy.  Two finite meshes are not required to agree: finite-k exchange
    is periodic on a mesh-dependent Born-von-Karman supercell (Sundararaman
    and Arias, Phys. Rev. B 87, 165122 (2013), Eqs. 10-13).
  - Pure-DFT (LDA / GGA) adds the lattice-summed periodic XC Pulay primitive
    ``xc_lattice_gradient_contribution`` (full LDA + GGA s-piece), shared
    with the Γ-only G1b path.
  - Symmetry-reduced and explicit non-Monkhorst-Pack Ewald meshes fail closed
    until their gradient-side full-star unfolding is implemented.

  Non-Ewald routes retain the historical lattice-summed assembly below.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

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
    two_electron_gradient_contribution,
)


__all__ = [
    "compute_gradient_periodic_rhf_multi_k",
    "compute_gradient_periodic_rks_multi_k",
]


# ----------------------------------------------------------------------
# Helpers -- Bloch fold of the per-k MO data into a real-space matrix set
# ----------------------------------------------------------------------

def _bloch_fold_w_per_k(
    C_per_k: Sequence[np.ndarray],
    eps_per_k: Sequence[np.ndarray],
    occ_per_k: Sequence[Union[int, np.ndarray]],
    kmesh: BlochKMesh,
    template: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Build the energy-weighted density on the same cell list as
    ``template``:

        W(g) = S_k w_k Re[ exp(-i k.g) . 2 C_occ(k) . diag(e_occ(k)) . C_occ(k)+ ]

    Mirrors ``real_space_density_from_kpoints``'s convention.
    ``occ_per_k`` may be either an integer (closed-shell Aufbau,
    n_occ orbitals filled) or a per-orbital occupation array
    (Fermi-Dirac smearing).

    The ``template`` LatticeMatrixSet is consumed (its blocks are
    overwritten) -- caller passes a freshly built
    ``compute_overlap_lattice`` result and uses the returned set.
    """
    n_cells = len(template.cells)
    nbf = int(template.nbf)
    blocks = [np.zeros((nbf, nbf), dtype=np.float64) for _ in range(n_cells)]
    weights = list(kmesh.weights)
    kpts = list(kmesh.kpoints)

    for ik in range(len(C_per_k)):
        C = np.asarray(C_per_k[ik], dtype=np.complex128)
        eps = np.asarray(eps_per_k[ik], dtype=np.float64)
        occ = occ_per_k[ik]
        w_k = float(weights[ik])
        k_cart = np.asarray(kpts[ik], dtype=np.float64)

        if np.isscalar(occ) or (hasattr(occ, "shape") and occ.shape == ()):
            n_occ = int(occ)
            if n_occ == 0:
                continue
            C_occ = C[:, :n_occ]
            eps_occ = eps[:n_occ]
            # W(k) = 2 . C_occ . diag(e) . C_occ+ (closed-shell factor 2)
            W_k = 2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T
        else:
            occ_arr = np.asarray(occ, dtype=np.float64)
            # W(k) = S_i occ_i . e_i . C_mui . C_νi+
            W_k = (C * (occ_arr * eps)[None, :]) @ C.conj().T

        Wk_re = W_k.real
        Wk_im = W_k.imag
        for c, cell in enumerate(template.cells):
            r = np.asarray(cell.r_cart, dtype=np.float64)
            phase = float(np.dot(k_cart, r))
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            # exp(-i k.g) . W(k) -> Re part = cos.Re(W) + sin.Im(W)
            blocks[c] += w_k * (cos_p * Wk_re + sin_p * Wk_im)

    for c in range(n_cells):
        template.set_block(c, blocks[c])
    return template


def _validate_corrected_ewald_gradient_mesh(kmesh: BlochKMesh) -> None:
    """Fail closed outside the full uniform Monkhorst-Pack domain.

    The corrected-exchange SCF can unfold an irreducible wedge through its
    symmetry operations, but the derivative helper currently consumes the
    supplied per-k matrices directly.  Treating wedge representatives as the
    full star would give the wrong density and energy-weighted-density folds.
    Explicit weighted k lists likewise carry no Born-von-Karman dimensions
    for the reciprocal exchange correction.
    """
    n_k = len(list(kmesh.kpoints))
    mesh = tuple(int(x) for x in getattr(kmesh, "mesh", ()))
    if len(mesh) != 3 or any(n <= 0 for n in mesh):
        raise NotImplementedError(
            "compute_gradient_periodic_rhf_multi_k: the corrected-Ewald "
            "analytic gradient requires Monkhorst-Pack mesh dimensions; "
            "explicit k-point lists require a defined Born-von-Karman "
            "supercell and are not supported. Use a full Monkhorst-Pack "
            "mesh or finite differences."
        )
    n_full = int(np.prod(mesh))
    weights = np.asarray(list(kmesh.weights), dtype=np.float64)
    if n_k != n_full:
        raise NotImplementedError(
            "compute_gradient_periodic_rhf_multi_k: the corrected-Ewald "
            "analytic gradient currently requires a full Monkhorst-Pack "
            f"mesh; got {n_k} supplied k-points for mesh={mesh}. "
            "Symmetry-reduced and explicit k-point lists require full-star "
            "gradient unfolding; use the full mesh or finite differences."
        )
    if n_k == 0 or weights.size != n_k:
        raise ValueError(
            "compute_gradient_periodic_rhf_multi_k: kmesh has inconsistent "
            "k-point and weight counts."
        )
    if not np.allclose(weights, 1.0 / float(n_k), atol=1.0e-12):
        raise NotImplementedError(
            "compute_gradient_periodic_rhf_multi_k: the corrected-Ewald "
            "analytic gradient requires uniform full-mesh weights; use a "
            "Monkhorst-Pack mesh or finite differences."
        )
    if mesh == (1, 1, 1) and not _is_single_gamma_mesh(kmesh):
        raise NotImplementedError(
            "compute_gradient_periodic_rhf_multi_k: a one-point "
            "corrected-Ewald mesh must be Gamma. An explicit twisted "
            "k point has no recoverable Monkhorst-Pack/Born-von-Karman "
            "metadata; use Gamma, a full mesh, or finite differences."
        )


def _is_single_gamma_mesh(kmesh: BlochKMesh) -> bool:
    """Match the RHF SCF driver's molecular-exchange Γ discriminator."""
    kpoints = list(kmesh.kpoints)
    weights = list(kmesh.weights)
    return (
        len(kpoints) == 1
        and len(weights) == 1
        and np.isclose(float(weights[0]), 1.0)
        and np.allclose(
            np.asarray(kpoints[0], dtype=np.float64), 0.0, atol=1.0e-12
        )
    )


def _resolve_gradient_lattice_opts(
    system: PeriodicSystem,
    result,
    supplied: Optional[LatticeSumOptions],
    *,
    where: str,
) -> LatticeSumOptions:
    """Use the SCF's effective lattice support or fail closed.

    Multi-k Ewald SCF can replace the caller's options with an optimized
    copy. New results retain a snapshot of that effective support. Older or
    external result objects must still receive explicit matching options;
    constructing a default DIRECT_TRUNCATED object for an Ewald result would
    silently select the known-wrong legacy derivative.
    """
    recorded = getattr(result, "effective_lattice_opts", None)
    resolved = recorded if recorded is not None else supplied
    ewald_alpha = float(getattr(result, "omega", 0.0) or 0.0)
    is_ewald_result = int(system.dim) == 3 and ewald_alpha > 0.0

    if resolved is None:
        if is_ewald_result:
            raise ValueError(
                f"{where}: this Ewald result does not record its effective "
                "lattice options; pass the matching lattice_opts used by "
                "the SCF."
            )
        return LatticeSumOptions()

    if (
        is_ewald_result
        and resolved.coulomb_method != CoulombMethod.EWALD_3D
    ):
        source = "recorded" if recorded is not None else "supplied"
        raise ValueError(
            f"{where}: the result records a positive 3D Ewald alpha but "
            f"the {source} lattice options select "
            f"{resolved.coulomb_method!r}; refusing a mixed-gauge gradient."
        )
    return resolved


def _compute_rhf_single_gamma_ewald_gradient(
    system: PeriodicSystem,
    basis: BasisSet,
    result,
    kmesh: BlochKMesh,
    *,
    lattice_opts: LatticeSumOptions,
    alpha_hf: float,
    ewald_alpha: float,
    reciprocal_cutoff_bohr_inv: float,
) -> np.ndarray:
    """Differentiate the RHF single-Γ Ewald/molecular-K energy.

    The shared corrected-Ewald derivative with ``alpha_hf=0`` supplies the
    Ewald nuclear, electron-nuclear, Hartree, background, and Pulay pieces.
    The RHF SCF's compatibility branch uses a home-cell molecular full-range
    K rather than the periodic corrected-exchange split, so add exactly the
    molecular exchange derivative by subtracting the J-only contraction.
    """
    from .bipole_gradient import (
        _compute_bipole_gradient_corrected_multi_k,
        _home_cell_index,
    )

    grad = _compute_bipole_gradient_corrected_multi_k(
        system,
        basis,
        result,
        kmesh,
        lattice_opts=lattice_opts,
        alpha_hf=0.0,
        ewald_alpha=float(ewald_alpha),
        reciprocal_cutoff_bohr_inv=float(reciprocal_cutoff_bohr_inv),
        # The SCF's J is the exact analytic-FT Hartree matrix (#575).
        j_sr_alpha_image_ball=True,
    )
    if abs(float(alpha_hf)) <= 1.0e-14:
        return np.asarray(grad)

    home = _home_cell_index(result.density.cells)
    D_home = np.asarray(result.density.blocks[home], dtype=np.float64)
    molecule = system.unit_cell_molecule()
    grad += np.asarray(
        two_electron_gradient_contribution(
            basis, molecule, D_home, float(alpha_hf)
        )
    )
    grad -= np.asarray(
        two_electron_gradient_contribution(basis, molecule, D_home, 0.0)
    )
    return np.asarray(grad)


# ----------------------------------------------------------------------
# Public driver -- multi-k periodic RHF gradient
# ----------------------------------------------------------------------

def compute_gradient_periodic_rhf_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    result,
    kmesh: BlochKMesh,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha_hf: float = 1.0,
) -> np.ndarray:
    """Multi-k closed-shell periodic RHF/RKS-HF-part atomic gradient.

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    result
        Converged multi-k SCF result (e.g.
        :class:`PeriodicRHFMultiKEwaldResult`,
        :class:`PeriodicRKSMultiKEwaldResult`). Must expose
        ``converged``, ``mo_coeffs`` (per-k complex (nbf, nbf)),
        ``mo_energies`` (per-k real (nbf,)), and ``density`` (a
        ``LatticeMatrixSet``).
    kmesh
        :class:`BlochKMesh` from
        :func:`vibeqc.monkhorst_pack` -- same one passed to the SCF.
    lattice_opts
        Fallback :class:`LatticeSumOptions` for result objects that predate
        effective-support provenance. Current Ewald results record and reuse
        the post-optimization lattice options actually used by the SCF.
    alpha_hf
        HF-exchange fraction. ``1.0`` for plain RHF; for hybrid DFT
        the RKS driver passes the functional's HF fraction.

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr.
    """
    if not result.converged:
        raise ValueError(
            "compute_gradient_periodic_rhf_multi_k: result is not converged."
        )
    lattice_opts = _resolve_gradient_lattice_opts(
        system,
        result,
        lattice_opts,
        where="compute_gradient_periodic_rhf_multi_k",
    )

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "compute_gradient_periodic_rhf_multi_k: open-shell not "
            "supported (closed-shell only). Use the periodic UHF/UKS "
            "driver (G1d) for open-shell systems."
        )
    n_k = len(kmesh.kpoints)

    # Doll et al., Comput. Phys. Commun. 137, 299 (2001), Eqs. 34,
    # 43-44: the Ewald energy gauge and the orbital-energy gauge in the
    # Pulay density must be differentiated together.  The historical
    # G1c assembly below instead mixed raw corrected-Ewald eigenvalues with
    # direct-truncated nuclear/V_ne/J/K derivatives.  Reuse the complete,
    # independently FD-validated Ewald derivative rather than duplicating
    # those gauge-sensitive terms here.
    ewald_alpha = float(getattr(result, "omega", 0.0) or 0.0)
    is_corrected_ewald = (
        int(system.dim) == 3
        and lattice_opts.coulomb_method == CoulombMethod.EWALD_3D
        and ewald_alpha > 0.0
    )
    if is_corrected_ewald:
        _validate_corrected_ewald_gradient_mesh(kmesh)
        from .periodic_corrected_exchange import (
            corrected_ewald_reciprocal_cutoff,
        )

        volume = float(
            abs(np.linalg.det(np.asarray(system.lattice, dtype=np.float64)))
        )
        reciprocal_cutoff = corrected_ewald_reciprocal_cutoff(
            volume, ewald_alpha
        )

        # Plain RHF has one deliberate exception: a genuine single Γ point
        # uses molecular full-range K for compatibility with the molecular
        # limit. RKS has no such exception; its global-hybrid full-range arm
        # uses corrected periodic exchange even at one k point.
        if not hasattr(result, "functional") and _is_single_gamma_mesh(kmesh):
            return _compute_rhf_single_gamma_ewald_gradient(
                system,
                basis,
                result,
                kmesh,
                lattice_opts=lattice_opts,
                alpha_hf=float(alpha_hf),
                ewald_alpha=ewald_alpha,
                reciprocal_cutoff_bohr_inv=reciprocal_cutoff,
            )

        from .bipole_gradient import _compute_bipole_gradient_corrected_multi_k

        return _compute_bipole_gradient_corrected_multi_k(
            system,
            basis,
            result,
            kmesh,
            lattice_opts=lattice_opts,
            alpha_hf=float(alpha_hf),
            ewald_alpha=ewald_alpha,
            reciprocal_cutoff_bohr_inv=reciprocal_cutoff,
            # The SCF's J is the exact analytic-FT Hartree matrix (#575).
            j_sr_alpha_image_ball=True,
        )

    # Non-Ewald compatibility path. The SCF result already has the
    # Bloch-folded D as a LatticeMatrixSet. Build W(g) on the same cell list
    # and retain the historical lattice-summed derivative functional.
    D_set = result.density
    template = compute_overlap_lattice(basis, system, lattice_opts)
    nocc = n_elec // 2
    occ_per_k = [nocc] * n_k

    W_set = _bloch_fold_w_per_k(
        result.mo_coeffs, result.mo_energies, occ_per_k, kmesh, template,
    )

    # Sum the four contributions through the C++ lattice-summed
    # primitives. D_set + W_set are now properly Bloch-folded for the
    # given k-mesh.
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
    grad += np.asarray(
        eri_lattice_gradient_contribution(
            basis, system, D_set, lattice_opts, float(alpha_hf)))

    return grad


# ----------------------------------------------------------------------
# Public driver -- multi-k periodic RKS gradient (LDA / pure DFT path)
# ----------------------------------------------------------------------

def compute_gradient_periodic_rks_multi_k(
    system: PeriodicSystem,
    basis: BasisSet,
    result,
    kmesh: BlochKMesh,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_options: Optional[GridOptions] = None,
) -> np.ndarray:
    """Multi-k closed-shell-DFT periodic atomic gradient.

    Parameters
    ----------
    system, basis, kmesh
        As in :func:`compute_gradient_periodic_rhf_multi_k`.
    result
        Converged :class:`PeriodicRKSMultiKEwaldResult`. Must expose
        ``functional`` (string) so the HF-exchange fraction can be
        read for the J/K Pulay piece.
    lattice_opts, grid_options
        Lattice-sum and DFT-grid options; must match the SCF.

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr.

    Notes
    -----
    Same scope caveats as the Γ-only G1b driver:

      - **LDA / GGA**: XC Pulay via the lattice-summed periodic
        primitive ``xc_lattice_gradient_contribution`` (full LDA + GGA
        s-piece), shared with the Γ-only path. Pure DFT (a_HF = 0) is
        exact in the molecular limit and for true-periodic cells.
      - **Global hybrid DFT (PBE0, B3LYP, ...)**: the 3D Ewald path uses
        the same corrected full-range exchange split as the SCF. Range-
        separated functionals remain fail-closed on this full-range-only
        gradient surface.
    """
    if not result.converged:
        raise ValueError(
            "compute_gradient_periodic_rks_multi_k: result is not converged."
        )
    lattice_opts = _resolve_gradient_lattice_opts(
        system,
        result,
        lattice_opts,
        where="compute_gradient_periodic_rks_multi_k",
    )
    if grid_options is None:
        grid_options = GridOptions()

    func = Functional(result.functional)
    # Full-range-only exchange Pulay term (see the Γ RKS gradient).
    from .periodic_screened_exchange import reject_unscreened_range_separated

    reject_unscreened_range_separated(
        func, where="compute_gradient_periodic_rks_multi_k"
    )
    alpha_hf = func.hf_exchange_fraction

    # HF-ish part via the multi-k driver.
    grad_hf = compute_gradient_periodic_rhf_multi_k(
        system, basis, result, kmesh,
        lattice_opts=lattice_opts, alpha_hf=alpha_hf,
    )

    # XC Pulay via the lattice-summed periodic primitive (full LDA + GGA
    # s-piece) -- the same kernel the Γ-only G1b driver uses. The multi-k
    # ``result.density`` is already a Bloch-folded real-space D(g) on the
    # SCF cell list, which is exactly the convention
    # ``xc_lattice_gradient_contribution`` consumes
    # (r(r) = S_g chi_ref.D(g).chi_g on the periodic Becke grid). This replaces
    # the former molecular-grid fallback that dropped the GGA s-coupled
    # piece and the image-cell density.
    from ._vibeqc_core import xc_lattice_gradient_contribution
    from .periodic_grid import build_periodic_becke_grid

    grid = build_periodic_becke_grid(system, grid_options=grid_options)
    grad_xc = np.asarray(
        xc_lattice_gradient_contribution(
            basis, system, grid, func, result.density, lattice_opts))

    return np.asarray(grad_hf) + grad_xc
