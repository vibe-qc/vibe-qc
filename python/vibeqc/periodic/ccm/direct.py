"""Real-Γ-supercell SCF for the neutral fitted-torus representation control.

This module and :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` evaluate one
specified block-circulant neutral-torus Hamiltonian in Fourier-related
real-Γ and Bloch representations. This module supplies the k-free side: one
real Γ-supercell SCF -- a single real
generalized eigenproblem over the ``N_c·n_μ`` supercell AOs, no k-point mesh, no
complex Bloch blocks, no per-k diagonalisation -- that reproduces the GDF energy
to numerical implementation tolerance. Any RI fitting error is common to both
representations. Ewald sums *inside* integral evaluation remain (they are the
physics of the neutral torus kernel ``v_E``, not band-structure machinery).

The finite Fourier theorem used here applies to any specified block-circulant
torus Hamiltonian. It establishes representation equivalence for this control;
it does not equate the union-and-weight/Wigner--Seitz Γ-CCM construction with
the finite-translation-group χ-CCM construction. Accordingly, this module is
not selected by ``aiccm2026dev-a`` or ``gamma-ccm`` and is not Γ-CCM approach
evidence.

The Hamiltonian, term by term (all real at Γ)
---------------------------------------------
* ``S``, ``T``, ``V_ne`` -- the multi-k GDF driver's **unit-cell** lattice
  sums, folded onto the torus (:func:`_fold_lattice_matrix_to_torus`): the
  folded supercell matrix regroups exactly the truncated blocks the per-k
  Bloch matrices are built from, so the one-electron side equals the multi-k
  Hamiltonian *by construction* at any cutoff (re-summing on the supercell
  lattice instead would truncate at supercell granularity -- a ~10 mHa-class
  mismatch for the conditionally-convergent bare ``dim < 3`` nuclear sums).
* ``V_ne``, ``E_nn`` gauge -- **Ewald** (``G = 0`` dropped, compensating
  background), forced for 3-D by
  :func:`~vibeqc.periodic_rhf_gdf._gauge_lat_opts_for_v_ne_and_e_nuc`, so the
  gauges match term by term against the neutral ``J``/``K``.
* ``dim < 3`` -- this whole 3-D-torus construction does **not** apply, and the
  route **fails closed** (2026-07-10). Its bare ``V_ne``/``E_nn`` sums are
  conditionally convergent and its neutral cderi is not a Coulomb kernel below
  three dimensions. The gauge-consistent alternatives are named in the error and
  in :func:`_reject_low_dimensional_direct`, which carries the algebra and the
  measurements. This supersedes the 2026-07-09 "KNOWN DEFECT" note that stood
  here: the defect is fixed, and its stated cause was incomplete.
* **measured** dimensionality (2026-08-25, IID 291) -- the declared-dimension
  refusal above was evadable by padding a low-dimensional system with
  transverse vacuum and declaring the cell 3-D. Every entry point now also
  measures the physical periodicity from the atom positions
  (:func:`~vibeqc.periodic.ccm.system.ccm_vacuum_directions`) and fails
  closed with the vacuum directions named
  (:func:`_reject_vacuum_padded_direct`); a closed-shell neutral SCF that
  converges to a positive total energy fails closed independently of the gate.
* ``J``, ``K`` -- pure real RI contractions of the **neutral Γ-supercell cderi**
  ``L`` (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`, real at Γ;
  the ``G+q=0`` mode dropped = the charge-neutral ``v_E``), via
  :func:`~vibeqc.periodic.ccm.ri.ccm_ri_j_neutral` /
  :func:`~vibeqc.periodic.ccm.ri.ccm_ri_k_neutral`.
* the **exchange-``q=0`` seam** -- the term this module derives; below.

Why the supercell-Γ fit equals the multi-k fit exactly: the supercell reciprocal
lattice at a given ``ke_cutoff`` is *exactly* the set ``{q + G}`` of the unit
cell's mesh-``q`` channels at the same cutoff, and the supercell auxiliary space
is the unit-cell auxiliary space replicated over the ``N_c`` cells -- so the
Γ-supercell density fit block-diagonalises into the per-``q`` unit-cell fits
(band folding at the RI level). The residual direct-vs-GDF difference is
therefore FP accumulation, not physics -- measured ``1.7e-11`` Ha/cell on the
vacuum-padded H₂ anchor (external KRHF ``-1.1182352381`` Ha/cell).

The exchange-``q=0`` seam, derived as a Fock operator
-----------------------------------------------------
The neutral kernel drops the single ``G+q=0`` mode of the supercell. In the
*Coulomb* channel that drop is compensated exactly by the nuclear side (the
Ewald gauge of ``V_ne``/``E_nn``): for a neutral cell the background terms
cancel term by term. In the *exchange* channel nothing compensates: the dropped
mode leaves each occupied orbital interacting with its own BvK images, an
``O(1/N_c)`` self-energy that does not vanish on a finite torus. The
BvK-Madelung (``exxdiv="ewald"``) convention restores it with the probe-charge
Ewald constant of the **supercell**, in the form used by periodic GDF:

    # Sec. 2.3 / Eq. 24 context of Sun, Berkelbach, McClain & Chan,
    # J. Chem. Phys. 147, 164119 (2017), doi:10.1063/1.4998644 (periodic GDF;
    # exxdiv='ewald' Madelung correction of the exchange G+q=0 divergence):
    #   K_muν  ->  K_muν + ξ_N (S D S)_muν ,
    #   ξ_N = _madelung_for_kmesh(unit_system, nrep)
    # (the BvK-supercell probe-charge Ewald constant -- for dim=3 equal to
    #  madelung_constant_for_cell(cluster_system); reused verbatim from the
    #  multi-k driver so both routes carry one constant by construction).

This is an **operator**, not an energy patch, and it satisfies ``F = ∂E/∂D``
exactly: the seam is linear and symmetric in ``D`` (like ``J`` and ``K``
themselves), so with

    F = h + J(D) - 1/2 [ K(D) + ξ_N S D S ],
    E = 1/2 Tr[D (h + F)] + E_nn

the quadratic seam energy ``-(ξ_N/4)·Tr[(DS)²]`` has density derivative
``-(ξ_N/2)·S D S`` -- precisely the seam's Fock contribution. At a converged
idempotent closed-shell density (``D S D = 2 D``) the seam energy is
``-(ξ_N/4)·Tr[(DS)²] = -(ξ_N/2)·N_e`` per supercell -- the standard exxdiv
energy shift, here *emerging from the variational functional* rather than
patched in. A deliberate ``exxdiv=None`` run (:func:`run_ccm_rhf_direct` with
``exxdiv=None``) therefore sits above the GDF by exactly ``ξ_N·N_e/(2 N_c)``
per unit cell -- the measured, pinned demonstration that the seam term is what
closes the documented gauge gap of the standalone neutral route
(:mod:`~vibeqc.periodic.ccm.neutral`; the D2 finding).

Convention bookkeeping: the route records ``exchange_q0`` on its result
(``"BvK-ewald"`` for ``exxdiv="ewald"``, ``"strict-zero-mode"`` for ``None``;
:mod:`vibeqc.periodic.exchange_convention`), and every parity comparison must
assert matched conventions (``assert_matched_exchange_q0``).

Scope: like the lean neutral SCF this builds only ``L`` (never any ``n⁴``
object); the supercell FFT fit is the memory/cost ceiling. Cite Peintinger &
Bredow, J. Comput. Chem. 35, 839 (2014) for the CCM and Sun et al. (2017) for
the periodic GDF / exxdiv convention.
"""

from __future__ import annotations

from .scf import _ccm_initial_guess, _with_ccm_guess

import numpy as np

from .experimental import _warn_experimental
from .neutral import ccm_neutral_cderi, ccm_neutral_tail_ke_cutoff
from .ri import (
    _rhf_loop,
    _rks_loop,
    ccm_ri_j_neutral,
    ccm_ri_k_neutral,
    ccm_ri_k_neutral_factored,
    psd_density_factor,
)
from .scf import _validate_conv_tol_grad


def _k_neutral_fast(L, D, S=None, eta=0.0):
    """Exchange (+ optional ``eta S D S`` seam/zero-mode term) through the
    occupied-rank factored contraction (efficiency-program K lever,
    2026-08-21). The SCF loops feed only aufbau densities
    (``D = c X X^T``, PSD -- DIIS extrapolates the Fock, never D), so the
    factor is exact; machine-precision agreement with the dense
    :func:`ccm_ri_k_neutral` is gated in ``tests/test_ccm_direct.py``.
    ``eta != 0`` folds the rank-1-per-column seam through the same factor
    (``S D S = (S X)(S X)^T``).
    """
    X = psd_density_factor(D)
    K = ccm_ri_k_neutral_factored(L, X)
    if eta:
        SX = S @ X
        K = K + eta * (SX @ SX.T)
    return K

__all__ = [
    "ccm_exchange_q0_madelung",
    "ccm_direct_oneelectron",
    "run_ccm_rhf_direct",
    "run_ccm_uhf_direct",
    "run_ccm_rks_direct",
    "run_ccm_uks_direct",
    "run_ccm_double_hybrid_direct",
    "run_ccm_direct_gradient",
    "run_ccm_direct_optimize",
    "run_ccm_dlpno_mp2_direct",
    "run_ccm_dlpno_ump2_direct",
    "run_ccm_dlpno_ccsd_direct",
    "run_ccm_rhf_direct_rijcosx",
]


def ccm_exchange_q0_madelung(ccm) -> float:
    """The exchange-``q=0`` seam constant ``ξ_N`` of the cluster (BvK) supercell.

    Delegates to the multi-k GDF driver's own ``_madelung_for_kmesh`` on
    ``(unit_system, nrep)`` -- the probe-charge Ewald-Madelung constant of the
    BvK supercell -- so the direct route's seam and the GDF's ``exxdiv='ewald'``
    shift are **one constant by construction** for the declared 3-D control.
    Every direct neutral-control SCF entry point rejects ``dim != 3`` before
    using this value; it does not define a wire or slab exchange convention.
    """
    from vibeqc.periodic_k_gdf import _madelung_for_kmesh

    return float(_madelung_for_kmesh(ccm.unit_system, list(ccm.nrep)))


def _fold_lattice_matrix_to_torus(m_lat, nrep, n_mu):
    """Fold a unit-cell lattice matrix set onto the BvK torus (real, Γ).

    ``m_lat`` holds home-bra blocks ``M(g)[mu,ν] = <mu,0|Ô|ν,g>`` over the
    truncated unit-cell image list. The torus (cluster supercell) matrix at Γ
    regroups exactly those blocks -- no re-summation on the supercell lattice,
    so the truncation set is *identical* to the multi-k driver's and the fold
    is the inverse mesh-Fourier transform of the per-k Bloch matrices
    ``M(k) = Σ_g e^{ik·g} M(g)`` (same Hamiltonian, diagonal representation):

        M^SC[(mu,c),(ν,c')] = Σ_{g ≡ c'−c (mod N)} M(g)

    with supercell cells ordered ``((i·n2)+j)·n3+k`` and per-cell contiguous
    AO blocks (the :class:`CCMSystem` supercell layout).
    """
    n1, n2, n3 = (int(n) for n in nrep)
    n_c = n1 * n2 * n3
    # Linear supercell-cell index for each (i, j, k), CCMSystem ordering.
    grid = np.array(
        [(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)],
        dtype=int)
    lin = np.arange(n_c)
    lin_of = np.empty((n1, n2, n3), dtype=int)
    lin_of[grid[:, 0], grid[:, 1], grid[:, 2]] = lin

    out = np.zeros((n_c * n_mu, n_c * n_mu))
    nvec = np.array([n1, n2, n3])
    for cell, blk in zip(m_lat.cells, m_lat.blocks):
        g = np.asarray(cell.index, dtype=int)
        blk = np.asarray(blk)
        cp = (grid + g[None, :]) % nvec[None, :]
        cp_lin = lin_of[cp[:, 0], cp[:, 1], cp[:, 2]]
        for c, cpl in zip(lin, cp_lin):
            out[c * n_mu:(c + 1) * n_mu, cpl * n_mu:(cpl + 1) * n_mu] += blk
    return out


def ccm_direct_oneelectron(ccm, *, lat_opts=None):
    """One-electron matrices + nuclear energy of the BvK torus at Γ.

    Returns ``(S, h, e_nn)`` -- real symmetric overlap and core Hamiltonian
    (``T + V_ne``) over the cluster-supercell AOs, and the nuclear repulsion
    per supercell -- built from the **unit-cell** lattice sums of the multi-k
    GDF driver, folded onto the torus (:func:`_fold_lattice_matrix_to_torus`):

    * ``S``/``T``: :func:`compute_overlap_lattice` / :func:`compute_kinetic_lattice`
      with the diffuse-basis cutoff widening of the GDF path
      (``_oneel_lattice_opts`` at the same mesh k-points).
    * ``V_ne``/``e_nn``: :func:`compute_nuclear_lattice_dispatch` /
      ``N_c ×`` :func:`nuclear_repulsion_per_cell` under
      ``_gauge_lat_opts_for_v_ne_and_e_nuc`` (Ewald gauge -- the GDF driver's
      exact dispatch).

    Folding the *unit-cell* sums (rather than re-summing on the supercell
    lattice) makes the one-electron/nuclear side equal to the multi-k
    Hamiltonian **by construction**, at any cutoff: re-summation would truncate
    the image list at supercell granularity. Real algebra throughout; no Bloch
    phases enter (Γ).

    **3-D only.** The Ewald gauge this builder forces exists only for a
    3-D-periodic cell; a ``dim < 3`` cell has no gauge-consistent neutral-torus
    Hamiltonian in this route (see :func:`_reject_low_dimensional_direct`).
    """
    if int(ccm.unit_system.dim) != 3:
        raise NotImplementedError(
            "ccm_direct_oneelectron: 3-D-periodic cells only. A dim < 3 cell's "
            "bare V_ne/E_nn lattice sums are conditionally convergent and do not "
            "share a gauge with any J. The direct neutral-control SCF also fails "
            "closed for every dim != 3; no wire or slab convention is dispatched."
        )
    from vibeqc._vibeqc_core import (
        BasisSet,
        LatticeSumOptions,
        nuclear_repulsion_per_cell,
        compute_kinetic_lattice,
        compute_overlap_lattice,
    )
    from vibeqc.periodic_k_gdf import (
        _kmesh_to_kpoints_weights, _oneel_lattice_opts,
        _preflight_gdf_oneel_memory,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    unit = ccm.unit_system
    ubasis = BasisSet(unit.unit_cell_molecule(), ccm.basis_name)
    base = lat_opts if lat_opts is not None else LatticeSumOptions()
    # Resolve the same converged AO domain as the GDF run for S/T/V.
    # Positive overlap alone does not certify image-tail convergence.
    kpts_cart, _ = _kmesh_to_kpoints_weights(unit, list(ccm.nrep))
    oneel = _oneel_lattice_opts(
        unit, ubasis, base, rcut_strategy="pyscf_auto", k_points_cart=kpts_cart
    )
    _preflight_gdf_oneel_memory(unit, ubasis, oneel, n_kpoints=len(kpts_cart))
    S_lat = compute_overlap_lattice(ubasis, unit, oneel)
    T_lat = compute_kinetic_lattice(ubasis, unit, oneel)
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(oneel, unit)
    V_lat = compute_nuclear_lattice_dispatch(ubasis, unit, gauge)

    n_mu = int(ubasis.nbasis)
    S = _fold_lattice_matrix_to_torus(S_lat, ccm.nrep, n_mu)
    T = _fold_lattice_matrix_to_torus(T_lat, ccm.nrep, n_mu)
    V = _fold_lattice_matrix_to_torus(V_lat, ccm.nrep, n_mu)
    S = 0.5 * (S + S.T)
    h = T + V
    h = 0.5 * (h + h.T)
    e_nn = float(ccm.n_cells) * float(nuclear_repulsion_per_cell(unit, gauge))
    return S, h, e_nn


def _neutral_cderi_for(ccm, cderi, cderi_build, ke_cutoff, aux_basis,
                       cderi_symmetry=None, omega_screen=0.0,
                       tail_ke_cutoff=None):
    """The neutral Γ-supercell cderi ``L`` per the shared ``cderi``/``cderi_build``
    keyword contract of the direct-route drivers (see :func:`run_ccm_rhf_direct`).

    ``cderi_symmetry`` is forwarded to the builder's ``symmetry`` -- the
    space-group pair reduction (supercell build) / k-pair star reduction
    (fold build); ignored when a prebuilt ``cderi`` is passed.

    ``aux_basis`` together with a prebuilt ``cderi`` is a contradiction and
    fails closed: the cderi already fixes the fitting basis, so a silently
    dropped ``aux_basis`` would report a run the caller did not ask for.

    ``tail_ke_cutoff`` is forwarded to the builder's RSGDF high-``|G|`` tail
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_tail_ke_cutoff`); like
    ``aux_basis`` it cannot take effect on a prebuilt ``cderi``, whose tail is
    already baked in, and fails closed there for the same reason (IID 307).

    ``omega_screen > 0`` builds the screened erfc-SR sibling instead (the
    HSE-class exchange kernel; experimental) -- exchange-only, and missing
    the finite zero mode the caller restores via
    :func:`~vibeqc.periodic.ccm.neutral.ccm_sr_exchange_zero_mode_constant`.
    """
    if cderi is not None:
        if aux_basis is not None:
            raise ValueError(
                "aux_basis is ignored when a prebuilt cderi is supplied: "
                "the cderi already fixes the fitting basis. Omit aux_basis, "
                "or drop cderi and let the builder construct it at aux_basis."
            )
        if tail_ke_cutoff is not None:
            raise ValueError(
                "tail_ke_cutoff is ignored when a prebuilt cderi is supplied: "
                "the cderi already fixes the high-|G| tail. Omit "
                "tail_ke_cutoff, or drop cderi and let the builder construct "
                "it at tail_ke_cutoff."
            )
        return np.asarray(cderi, dtype=float)
    if cderi_build == "fold":
        from .neutral import ccm_neutral_cderi_fold

        return ccm_neutral_cderi_fold(ccm, ke_cutoff=ke_cutoff,
                                      tail_ke_cutoff=tail_ke_cutoff,
                                      aux_basis=aux_basis,
                                      symmetry=cderi_symmetry,
                                      omega_screen=float(omega_screen))
    return np.asarray(
        ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff,
                          tail_ke_cutoff=tail_ke_cutoff,
                          aux_basis=aux_basis,
                          symmetry=cderi_symmetry,
                          omega_screen=float(omega_screen)),
        dtype=float)


def _supercell_density_to_lattice_blocks(D_sc, nrep, n_mu, cells):
    """Cell-averaged unit-cell density blocks from a supercell-Γ density -- the
    inverse of :func:`_fold_lattice_matrix_to_torus`.

    For each unit-cell lattice vector ``g`` in ``cells`` returns

        D(g) = (1/N_c) . S_c  D_sc[(c), (c+g mod N)]                     (*)

    the block average over the ``N_c`` torus cells at separation class
    ``g mod N``. For a block-circulant (translation-invariant) ``D_sc`` the
    average is exact block extraction, and (*) equals the multi-k driver's
    real-space density ``D(g) = (1/N_c) S_k e^{-ik.g} D(k)`` -- which is
    mesh-periodic in ``g``, so every ``g`` in the (cutoff-truncated) ``cells``
    list maps to its torus class exactly as the multi-k fold does. For a
    non-circulant iterate, (*) is the circulant **projection** P(D): P is
    linear and self-adjoint, so an XC potential folded back onto the torus
    from these blocks is exactly ``dE_xc[P(D)]/dD`` -- the variational
    gradient of the composed functional (F = dE/dD holds through the SCF,
    and on the translation-invariant manifold P(D) = D the composition is
    the plain XC functional; CCM_THEORY.md Sec. 10's constrained-density
    equivalence).
    """
    n1, n2, n3 = (int(n) for n in nrep)
    n_c = n1 * n2 * n3
    grid = np.array(
        [(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)],
        dtype=int)
    lin_of = np.empty((n1, n2, n3), dtype=int)
    lin_of[grid[:, 0], grid[:, 1], grid[:, 2]] = np.arange(n_c)
    nvec = np.array([n1, n2, n3])

    # One averaged block per torus residue class (N_c classes), then map each
    # lattice cell in `cells` to its class -- mesh-periodic extension, as in
    # the multi-k inverse Bloch fold.
    cls_blocks = {}
    for r_idx in range(n_c):
        r = grid[r_idx]
        blk = np.zeros((n_mu, n_mu))
        for c_idx in range(n_c):
            cp = (grid[c_idx] + r) % nvec
            cpl = int(lin_of[cp[0], cp[1], cp[2]])
            blk += D_sc[c_idx * n_mu:(c_idx + 1) * n_mu,
                        cpl * n_mu:(cpl + 1) * n_mu]
        cls_blocks[tuple(int(x) for x in r)] = blk / float(n_c)
    return [
        cls_blocks[tuple(int(x) for x in
                         np.mod(np.asarray(cell.index, dtype=int), nvec))]
        for cell in cells
    ]


def _external_xc_grid_options(func, grid_options, *, who):
    """Resolve an external provider's grid-profile capability before SCF.

    A full-grid provider may pin one complete atom-grid protocol.  When the
    caller did not supply :class:`GridOptions`, construct that required
    profile automatically; an explicit incompatible profile fails here,
    before the neutral cderi or any SCF work is built.  Libxc functionals and
    external providers without a profile requirement keep the existing
    default behaviour.
    """
    if not bool(getattr(func, "is_external", False)):
        return grid_options
    capabilities = getattr(func, "external_capabilities", None) or {}
    required = str(capabilities.get("required_grid_profile", "") or "")
    if not required:
        return grid_options

    from vibeqc._vibeqc_core import GridOptions

    required_options = GridOptions()
    required_options.atomic_grid_profile = required
    if grid_options is None:
        return required_options
    if grid_options.atomic_grid_profile != required_options.atomic_grid_profile:
        actual = str(grid_options.atomic_grid_profile).rsplit(".", 1)[-1]
        raise ValueError(
            f"{who}: external XC functional {func.name!r} requires grid "
            f"profile {required!r}, but the supplied GridOptions selects "
            f"{actual!r}."
        )
    return grid_options


def _reject_external_xc_neutral_hamiltonian(ccm, func, *, who):
    """Fail closed outside the real-Gamma adapter's validated Hamiltonian."""
    if not bool(getattr(func, "is_external", False)):
        return
    if int(ccm.unit_system.charge) != 0:
        raise NotImplementedError(
            f"{who}: charged-cell external XC is not supported by the "
            "neutral fitted-torus Hamiltonian; use a neutral unit cell"
        )

    from vibeqc.pbc_bipole_common import reject_bipole_ecp_options

    try:
        reject_bipole_ecp_options(
            object(),
            driver=f"{who} external XC",
            basis=ccm.basis,
            system=ccm.unit_system,
        )
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{who}: ECP-bearing bases are not implemented with external "
            "XC; use an all-electron basis"
        ) from exc


def _xc_lattice_options(lat_opts, becke_image_radius_bohr):
    """Copy lattice options and synchronize the periodic-XC image reach.

    ``build_periodic_becke_grid`` uses ``becke_image_radius_bohr`` for the
    partition denominator.  ``build_xc_periodic`` independently reads the
    same-named field from :class:`LatticeSumOptions` to select shifted AO
    images.  They must describe one finite image convention.  Pybind value
    objects cannot be copied with :mod:`copy`, so mirror their public fields
    without mutating a caller-owned object.
    """
    from vibeqc._vibeqc_core import LatticeSumOptions

    out = LatticeSumOptions()
    if lat_opts is not None:
        for name in dir(lat_opts):
            if name.startswith("_"):
                continue
            try:
                value = getattr(lat_opts, name)
            except Exception:
                continue
            if callable(value):
                continue
            try:
                setattr(out, name, value)
            except Exception:
                pass
    image_radius = float(becke_image_radius_bohr)
    if not np.isfinite(image_radius) or image_radius < 0.0:
        raise ValueError(
            "real-Gamma periodic XC becke_image_radius_bohr must be "
            "finite and >= 0"
        )
    cutoff = float(out.cutoff_bohr)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(
            "real-Gamma periodic XC cutoff_bohr must be finite and > 0"
        )
    if cutoff < image_radius:
        raise ValueError(
            "real-Gamma periodic XC cutoff_bohr must be at least the "
            "periodic Becke image radius; got "
            f"{cutoff:.6g} < {image_radius:.6g} bohr"
        )
    out.becke_image_radius_bohr = image_radius
    return out


def _xc_difference_closed_cells(
    system,
    cutoff_bohr,
    image_radius_bohr=10.0,
):
    """Return the atom-pair-complete periodic-XC difference domain."""
    from vibeqc.periodic_external_xc import (
        periodic_xc_difference_closed_cells,
    )

    return periodic_xc_difference_closed_cells(
        system,
        float(cutoff_bohr),
        float(image_radius_bohr),
    )


def _real_gamma_rks_xc_builder(
    ccm, func, *, grid_options=None, becke_image_radius_bohr=10.0,
    lat_opts=None,
):
    """Return the variational periodic-XC closure used by real-Gamma RKS.

    The reference-cell grid plus translated-basis density contributions follow
    Janetzko et al., J. Chem. Phys. 128, 024102 (2008),
    doi:10.1063/1.2817582.  This neutral finite-BvK-torus representation is
    distinct from the literal four-centre WSSC ``aiccm2026dev-a`` lineage.
    """
    from vibeqc._vibeqc_core import (
        BasisSet,
        PeriodicXCDensityDomain,
        build_xc_periodic,
        make_lattice_matrix_set,
    )
    from vibeqc.periodic_grid import build_periodic_becke_grid

    grid_options = _external_xc_grid_options(
        func, grid_options, who="run_ccm_rks_direct")
    _external_image_radius = float(becke_image_radius_bohr)
    if bool(getattr(func, "is_external", False)) and (
        not np.isfinite(_external_image_radius)
        or _external_image_radius <= 0.0
    ):
        raise ValueError(
            "run_ccm_rks_direct: external XC requires "
            "becke_image_radius_bohr > 0"
        )
    unit = ccm.unit_system
    ubasis = BasisSet(unit.unit_cell_molecule(), ccm.basis_name)
    n_mu = int(ubasis.nbasis)
    base = _xc_lattice_options(lat_opts, becke_image_radius_bohr)
    grid = build_periodic_becke_grid(
        unit,
        grid_options=grid_options,
        image_radius_bohr=float(becke_image_radius_bohr),
    )
    uses_external_xc = bool(getattr(func, "is_external", False))
    if uses_external_xc:
        # AO images a,s are active inside cutoff_bohr, while the variational
        # density/potential block is indexed by g=s-a. Its support can
        # therefore reach twice that radius and must not be truncated to the
        # AO-image ball.
        cells = _xc_difference_closed_cells(
            unit,
            base.cutoff_bohr,
            base.becke_image_radius_bohr,
        )
    else:
        # Preserve the validated historical libxc real-Gamma construction.
        # Its AUTO density-domain behavior and radial cell list are part of
        # the existing direct-vs-GDF parity contract.
        from vibeqc._vibeqc_core import direct_lattice_cells

        cells = direct_lattice_cells(unit, float(base.cutoff_bohr))

    def xc_of_D(D):
        blocks = _supercell_density_to_lattice_blocks(
            D, ccm.nrep, n_mu, cells)
        density_real = make_lattice_matrix_set(n_mu, cells, blocks)
        xc = build_xc_periodic(
            ubasis,
            unit,
            grid,
            func,
            density_real,
            base,
            (
                PeriodicXCDensityDomain.PERIODIC_LATTICE
                if uses_external_xc
                else PeriodicXCDensityDomain.AUTO
            ),
        )
        V_sc = _fold_lattice_matrix_to_torus(xc.V_xc, ccm.nrep, n_mu)
        # build_xc_periodic returns E_xc per primitive unit cell.  The folded
        # potential is d(N_c E_xc)/dD_sc, so the scalar follows that same
        # supercell convention without another quadrature-weight factor.
        return (
            float(xc.e_xc) * float(ccm.n_cells),
            0.5 * (V_sc + V_sc.T),
        )

    return xc_of_D


def _real_gamma_uks_xc_builder(
    ccm, func, *, grid_options=None, becke_image_radius_bohr=10.0,
    lat_opts=None,
):
    """Return the variational periodic-XC closure used by real-Gamma UKS.

    Uses the same Janetzko et al. reference-cell/image-density construction as
    :func:`_real_gamma_rks_xc_builder` (doi:10.1063/1.2817582), on the neutral
    BvK torus rather than the four-centre WSSC lineage.
    """
    from vibeqc._vibeqc_core import (
        BasisSet,
        PeriodicXCDensityDomain,
        build_xc_periodic_uks,
        make_lattice_matrix_set,
    )
    from vibeqc.periodic_grid import build_periodic_becke_grid

    grid_options = _external_xc_grid_options(
        func, grid_options, who="run_ccm_uks_direct")
    _external_image_radius = float(becke_image_radius_bohr)
    if bool(getattr(func, "is_external", False)) and (
        not np.isfinite(_external_image_radius)
        or _external_image_radius <= 0.0
    ):
        raise ValueError(
            "run_ccm_uks_direct: external XC requires "
            "becke_image_radius_bohr > 0"
        )
    unit = ccm.unit_system
    ubasis = BasisSet(unit.unit_cell_molecule(), ccm.basis_name)
    n_mu = int(ubasis.nbasis)
    base = _xc_lattice_options(lat_opts, becke_image_radius_bohr)
    grid = build_periodic_becke_grid(
        unit,
        grid_options=grid_options,
        image_radius_bohr=float(becke_image_radius_bohr),
    )
    uses_external_xc = bool(getattr(func, "is_external", False))
    if uses_external_xc:
        cells = _xc_difference_closed_cells(
            unit,
            base.cutoff_bohr,
            base.becke_image_radius_bohr,
        )
    else:
        from vibeqc._vibeqc_core import direct_lattice_cells

        cells = direct_lattice_cells(unit, float(base.cutoff_bohr))

    def xc_of_spins(Da, Db):
        blocks_a = _supercell_density_to_lattice_blocks(
            Da, ccm.nrep, n_mu, cells)
        blocks_b = _supercell_density_to_lattice_blocks(
            Db, ccm.nrep, n_mu, cells)
        density_a = make_lattice_matrix_set(n_mu, cells, blocks_a)
        density_b = make_lattice_matrix_set(n_mu, cells, blocks_b)
        xc = build_xc_periodic_uks(
            ubasis,
            unit,
            grid,
            func,
            density_a,
            density_b,
            base,
            (
                PeriodicXCDensityDomain.PERIODIC_LATTICE
                if uses_external_xc
                else PeriodicXCDensityDomain.AUTO
            ),
        )
        Va = _fold_lattice_matrix_to_torus(
            xc.V_alpha, ccm.nrep, n_mu)
        Vb = _fold_lattice_matrix_to_torus(
            xc.V_beta, ccm.nrep, n_mu)
        return (
            float(xc.e_xc) * float(ccm.n_cells),
            0.5 * (Va + Va.T),
            0.5 * (Vb + Vb.T),
        )

    return xc_of_spins


def run_ccm_rks_direct(
    ccm, functional="pbe", *, initial_guess: object = "AUTO", cderi=None, cderi_sr=None, cderi_build="fold",
    ke_cutoff=200.0, tail_ke_cutoff=None,
    aux_basis=None, cderi_symmetry=None, exxdiv="ewald", lat_opts=None,
    grid_options=None, becke_image_radius_bohr=10.0,
    max_iter=128, conv_tol=1e-8, conv_tol_grad=1e-6, diis_dim=8, lindep_tol=1e-7,
    _allow_double_hybrid_scf=False,
):
    """Closed-shell KS for the neutral fitted-torus **real-Γ control**.

    This is one real Γ-supercell KS-SCF (no k-mesh), the DFT sibling of
    :func:`run_ccm_rhf_direct`,
    reproducing :func:`~vibeqc.periodic.ccm.ri.run_ccm_rks_gdf` to numerical
    implementation tolerance for pure **and** hybrid functionals. Any RI
    fitting error is common to both representations.

    Kohn-Sham on the torus (CCM_THEORY.md Sec. 11.2):

        F = h + J(D) + V_xc[rho_D] - (a_x/2) [ K(D) + ξ_N S D S ]

    with ``a_x`` the functional's exact-exchange fraction (libxc
    ``hf_exchange_fraction``; 0 for pure LDA/GGA/meta-GGA), ``J``/``K`` the
    neutral-cderi RI contractions of :func:`run_ccm_rhf_direct`, and the
    exchange-``q=0`` seam applied to the **exact-exchange channel only**,
    scaled by ``a_x`` exactly as the multi-k driver applies its
    ``exxdiv='ewald'`` shift (hybrid exchange under the identical convention,
    so the finite-size electrostatics stay consistent across the Coulomb,
    exchange, and XC channels).

    The XC term is built **periodically, by the same construction as the
    one-electron fold**: the supercell density is cell-averaged to unit-cell
    lattice blocks (:func:`_supercell_density_to_lattice_blocks`), fed to the
    multi-k driver's own periodic XC kernel (``build_xc_periodic``) on the
    **periodic Becke grid** of the unit cell
    (:func:`~vibeqc.periodic_grid.build_periodic_becke_grid` -- home-cell
    atoms own all grid points, partition denominator extended over image
    atoms; the ``run_krks_periodic_gdf`` production default,
    ``use_periodic_becke=True``), and the returned ``V_xc(g)`` lattice
    matrix is folded onto the torus with :func:`_fold_lattice_matrix_to_torus`.
    Same grid, same cells list, same XC kernel as the multi-k KS reference --
    parity by construction, like ``S``/``h``. (A *molecular* Becke grid over
    the cluster -- the four-center :func:`~vibeqc.periodic.ccm.dft.run_ccm_rks`
    choice -- integrates the mesh-periodic density over all space instead of
    one cell and over-counts E_xc grossly on periodic densities.)

    Parameters are those of :func:`run_ccm_rhf_direct` plus ``functional``
    (any libxc name -- pure or global hybrid), ``grid_options``
    (:class:`~vibeqc.GridOptions` for the XC quadrature; default = the
    multi-k driver's default tier) and ``becke_image_radius_bohr`` (periodic
    Becke partition image radius; default 10 bohr = the
    ``PeriodicKSOptions.becke_image_radius_bohr`` default). ``exxdiv``
    governs the seam on the exact-exchange fraction; for a pure functional
    it never enters (recorded on the result regardless, as the declared
    convention).

    **Screened (HSE-class) hybrids -- EXPERIMENTAL (2026-07-17).**
    Range-separated functionals resolve through
    :func:`~vibeqc.periodic_screened_exchange.resolve_periodic_exchange`
    (the shared periodic RSH policy): short-range-only screened hybrids
    (``hse06``-class, no full-range exact-exchange arm) run, with

        K_sr(D) = K[L_sr](D) + sigma_0 . S D S,
        sigma_0 = pi / (omega_screen^2 . V_sc)

    where ``L_sr`` is the erfc-attenuated sibling of the neutral cderi
    (same ``cderi_build``/``cderi_symmetry`` machinery,
    ``omega_screen`` = the functional's physical screening parameter) and
    ``sigma_0 S D S`` restores the screened kernel's FINITE ``G+q=0``
    mode, which the fit excludes
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_sr_exchange_zero_mode_constant`).
    F = h + J + V_xc - (c_sr/2) K_sr, E_x = -(c_sr/4) Tr[D K_sr] -- the
    aiccm2026dev-b binding-contract form. **No exchange-q=0 seam attaches
    to the screened kernel** (its zero mode is finite; ``exxdiv`` never
    enters, the b-contract's eta = 0), so
    ``.exchange_q0_applicability = "inactive"`` on such runs.
    Range-separated functionals *with* a full-range arm (wb97x,
    cam-b3lyp, lc-wpbe, ...) keep failing closed, per the maintainer
    policy recorded in the resolver. ``cderi_sr`` optionally passes a
    prebuilt screened cderi (the ``cderi`` contract's SR sibling).

    Returns a :class:`~vibeqc.periodic.ccm.dft.CCMKSResult` (energies **per
    supercell**) with ``.exchange_q0`` + ``.exchange_q0_applicability``
    recorded (``"active"`` iff a full-range exact-exchange arm is present,
    so the q=0 convention materially selects the Hamiltonian; ``"inactive"``
    for pure functionals and screened-only hybrids).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rks_direct',
    )
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rks_direct"
    )
    _warn_experimental()
    from vibeqc._vibeqc_core import Functional
    from vibeqc.periodic.exchange_convention import exchange_q0_label

    if exxdiv not in ("ewald", None):
        raise ValueError(
            f"run_ccm_rks_direct: exxdiv must be 'ewald' or None, got {exxdiv!r}"
        )
    dim = int(ccm.unit_system.dim)
    if dim != 3:
        _reject_low_dimensional_direct(dim, who="run_ccm_rks_direct")
    _reject_vacuum_padded_direct(ccm, who="run_ccm_rks_direct")
    if cderi_build not in ("supercell", "fold"):
        raise ValueError(
            "run_ccm_rks_direct: cderi_build must be 'supercell' or 'fold', "
            f"got {cderi_build!r}"
        )
    func = Functional(functional, 1)
    _reject_external_xc_neutral_hamiltonian(
        ccm, func, who="run_ccm_rks_direct"
    )
    if (
        bool(getattr(func, "is_external", False))
        and float(becke_image_radius_bohr) <= 0.0
    ):
        raise ValueError(
            "run_ccm_rks_direct: external XC requires "
            "becke_image_radius_bohr > 0"
        )
    grid_options = _external_xc_grid_options(
        func, grid_options, who="run_ccm_rks_direct")
    exx_screened = None
    if bool(func.is_range_separated):
        from vibeqc.periodic_screened_exchange import resolve_periodic_exchange

        # Shared periodic RSH policy: HSE-class (screened, no full-range
        # exact-exchange arm) resolves to (c_full = 0, c_sr, omega_screen);
        # full-range-arm RSH (wb97x, cam-b3lyp, ...) raises here.
        exx_screened = resolve_periodic_exchange(
            func, where="run_ccm_rks_direct")
    if (bool(getattr(func, "is_double_hybrid", False))
            and not _allow_double_hybrid_scf):
        raise NotImplementedError(
            "run_ccm_rks_direct: double hybrids need the scaled-MP2 step on "
            "top of the SCF, which this SCF driver does not add; use "
            "run_ccm_double_hybrid_direct (EXPERIMENTAL, closed shell) or a "
            "plain global hybrid."
        )
    alpha = float(func.hf_exchange_fraction)

    L = _neutral_cderi_for(ccm, cderi, cderi_build, ke_cutoff, aux_basis,
                           cderi_symmetry,
                           tail_ke_cutoff=tail_ke_cutoff)
    S, h, e_nn = ccm_direct_oneelectron(ccm, lat_opts=lat_opts)

    # Periodic XC on the unit cell, folded onto the torus (multi-k parity by
    # construction: same grid default, same cells list, same C++ XC kernel).
    xc_of_D = _real_gamma_rks_xc_builder(
        ccm,
        func,
        grid_options=grid_options,
        becke_image_radius_bohr=becke_image_radius_bohr,
        lat_opts=lat_opts,
    )

    k_of_D = None
    alpha_loop = alpha
    if exx_screened is not None and exx_screened.is_screened:
        # HSE-class screened hybrid (EXPERIMENTAL): the resolver guarantees
        # c_full == 0 here, so the whole exact exchange rides the erfc-SR
        # kernel -- fitted (L_sr) plus its finite zero mode restored
        # analytically. No exchange-q=0 seam attaches (finite kernel; the
        # aiccm2026dev-b contract's eta = 0). Coefficients are folded into
        # k_of_D with alpha_loop = 1, so the loop's -(1/2) K contraction /
        # -(1/4) Tr[D K] energy realise F -= (c_sr/2) K_sr and
        # E_x = -(c_sr/4) Tr[D K_sr].
        from .neutral import ccm_sr_exchange_zero_mode_constant

        L_sr = _neutral_cderi_for(
            ccm, cderi_sr, cderi_build, ke_cutoff, aux_basis, cderi_symmetry,
            omega_screen=exx_screened.omega_screen,
            tail_ke_cutoff=(None if cderi_sr is not None
                            else tail_ke_cutoff))
        sigma0 = ccm_sr_exchange_zero_mode_constant(
            ccm, exx_screened.omega_screen)
        c_sr = float(exx_screened.c_sr)
        k_of_D = lambda D: c_sr * _k_neutral_fast(L_sr, D, S, sigma0)
        alpha_loop = 1.0
    elif alpha != 0.0:
        if exxdiv == "ewald":
            xi = ccm_exchange_q0_madelung(ccm)
            # Seam on the exact-exchange channel (K -> K + ξ_N S D S), scaled
            # by a_x through the -a_x/2 hybrid contraction in the loop --
            # identical convention to apply_exxdiv_ewald_to_K in the multi-k
            # driver (CCM_THEORY.md Sec. 11.1-11.2).
            k_of_D = lambda D: _k_neutral_fast(L, D, S, xi)
        else:
            k_of_D = lambda D: _k_neutral_fast(L, D)

    res = _rks_loop(
        ccm, S, h, e_nn,
        lambda D: ccm_ri_j_neutral(L, D), k_of_D, xc_of_D, alpha_loop,
        functional,
        max_iter=max_iter, conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis_dim=diis_dim, lindep_tol=lindep_tol)
    res.backend = "ccm-neutral-direct-rks"  # result-backend identity (IID 344)
    res.rsgdf_tail_ke_cutoff = (
        None if cderi is not None
        else ccm_neutral_tail_ke_cutoff(
            ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff))
    res.exchange_q0 = exchange_q0_label(exxdiv)
    # "active" iff a full-range exact-exchange arm is present (the q=0
    # convention materially selects the Hamiltonian); pure functionals and
    # screened-only (HSE-class) hybrids carry no seam -- eta = 0 in the
    # aiccm2026dev-b binding contract.
    res.exchange_q0_applicability = (
        "active" if (exx_screened is None and alpha != 0.0) else "inactive")
    return _with_ccm_guess(res, guess_selection)


def _reject_low_dimensional_direct(dim, who="run_ccm_rhf_direct"):
    """Refuse a ``dim < 3`` cell: the 3-D-torus construction of
    :func:`run_ccm_rhf_direct` (and its KS sibling :func:`run_ccm_rks_direct`,
    which passes its own name as ``who``) has no gauge in which its three
    Coulomb channels agree, so there is no energy for it to return.

    Why the 3-D-torus route cannot be reused below three dimensions
    ---------------------------------------------------------------
    Every periodic Coulomb kernel is fixed only up to a conditional constant ``c``
    (the ``G+q=0`` mode). For a **neutral** cell the three Coulomb-bearing channels
    each pick up a power of ``c``, and they must cancel::

        E_nn        ->  +½ c Z²
        Tr[D·V_ne]  ->  -  c Z N_e
        ½Tr[D·J]    ->  +½ c N_e²
        ------------------------------------------------
        sum         =  ½ c (N_e - Z)²  =  0     (neutral cell)

    The cancellation is exact *only* when one ``c`` is shared by all three. Before
    2026-07-10 the ``dim < 3`` path violated this twice over:

    1. ``V_ne``/``E_nn`` came from **bare** lattice sums (the gauge helper passed
       its input through for ``dim != 3``). In 1-D and 2-D those sums are only
       conditionally convergent, so their ``c`` grows without bound with the
       lattice-sum cutoff. Measured on polyethylene (dim=1, C₂H₄, sto-3g, (4,1,1)),
       ``nuclear_cutoff_bohr`` 45 -> 90 -> 180::

           E_nn         +766.07  ->  +907.69  ->  +1057.84
           e_electronic -1992.85 -> -2276.10  -> -2576.39   (2x faster: the
                                                             -c·Z·N_e cross term)
           E_total      -1226.78 -> -1368.41  -> -1518.55   <- diverges

    2. ``J``/``K`` came from the neutral cderi, which for ``dim < 3`` is not a
       Coulomb kernel at all: ``rsgdf_dense_g_mesh`` collapses every non-periodic
       axis to the single ``G_perp = 0`` plane (35 G-points for a 1-D chain where
       the same lattice at ``dim=3`` carries 65281), so the AO-pair density is
       replaced by its transverse average and the kernel degenerates to a uniform
       sheet term ``∝ 1/V``. Measured: ``|g_neutral|_max · V`` is constant (92.46)
       across transverse vacuum D = 12/18/24 bohr, i.e. the electron-electron
       repulsion *vanishes* as the vacuum grows, while the wire four-center is
       exactly D-invariant (1.895512 at every D). At SCF level, H₂ chain/sto-3g/
       (2,1,1): this route gave -6.917 -> -6.568 Ha for D = 12 -> 30 bohr, against
       the wire kernel's -1.32206067 Ha at every D.

    Supplying the bare channels with the collapsed kernel's gauge would make the
    total cutoff-independent while converging to a physically meaningless energy --
    the CLAUDE.md §7 paper-over. The only sound fix is a kernel whose three
    channels are one mixed-boundary Poisson problem.

    Supplying the bare channels with the collapsed kernel's gauge would make the
    total cutoff-independent while converging to a physically meaningless energy --
    the CLAUDE.md §7 paper-over. The only sound fix is a kernel whose three
    channels are one mixed-boundary Poisson problem, and this route has none.

    Why this fails closed rather than dispatching to the wire kernel
    ---------------------------------------------------------------
    ``dim == 1`` does have a gauge-consistent mixed-boundary Hamiltonian --
    :func:`~vibeqc.periodic.ccm.lowd_scf.run_ccm_rhf_wire`, whose
    ``V_ne``/``E_nn``/``J``/``K`` ride one wire quadrature and whose neutral-cell
    Hartree total is gauge-free (``tests/test_ccm_lowd_four_center.py``
    ``test_wire_hartree_total_gauge_invariant``). It is **not** silently substituted
    here, for two reasons:

    * It is a **different operator**, not a repaired version of this one. The wire
      is the neutralised mixed-boundary Hamiltonian; the four-center route is the
      bare-``1/r`` minimum-image finite torus. They coincide only in the
      non-interacting limit, so neither is a drop-in for the other and neither
      gates the other (``192bc645``).
    * Its **correctness at scale is unverified** (``192bc645``): no converged
      core-bearing wire number exists -- a polyethylene ``g_max`` sweep gave
      ``-11.681 -> -13.141 -> -13.330`` Ha/atom with ``conv=False`` throughout.
      Separately, the wire carries ``S``/``T`` on the *plain* supercell
      (periodicity lives entirely in its Coulomb kernel), so the chain has open
      ends in the orbital space and ``energy / n_cells`` carries a ``1/N`` surface
      term: measured on the H₂ chain (sto-3g, ``exxdiv="strict"``), ``E/cell`` at
      ``nrep = 1,2,3,4,6`` runs ``-0.442, -0.661, -0.767, -0.831, -0.903`` Ha, still
      moving 0.07 Ha between ``nrep = 4`` and ``6``.

    Putting that behind a production entry point would trade a loud wrong number
    for a quiet unvalidated one. Callers who want the wire Hamiltonian ask for it
    by name.

    ``dim == 2`` has no mixed-boundary kernel at all: the slab four-center needs the
    §9 partial in-plane FT (the naive quadrature was built, measured wrong -- ``J``
    off 24 % -- and reverted, 2026-07-05).
    """
    if dim == 1:
        raise NotImplementedError(
            f"{who}: dim == 1 has no gauge-consistent Coulomb kernel on "
            "the 3-D-torus route. Its bare V_ne/E_nn lattice sums are conditionally "
            "convergent (the total diverges with nuclear_cutoff_bohr) and its neutral "
            "cderi collapses to a transverse-uniform sheet term that vanishes as 1/V "
            "with the vacuum padding, so any energy returned here would be a cutoff "
            "and vacuum artifact. Call run_ccm_rhf_wire(ccm) for the mixed-boundary "
            "wire Hamiltonian -- a DIFFERENT operator (neutralised mixed-boundary vs "
            "this route's bare-1/r finite torus), whose correctness at scale is still "
            "unverified and whose energy/n_cells carries a 1/N surface term -- or use "
            "the four-center route, where bare 1/r minimum image puts all channels in "
            "one gauge."
        )
    if dim == 2:
        raise NotImplementedError(
            f"{who}: dim == 2 has no gauge-consistent Coulomb kernel. "
            "The 3-D-torus route's V_ne/E_nn lattice sums are conditionally "
            "convergent in 2-D and its neutral cderi collapses to a transverse-"
            "uniform sheet term, so the returned energy would be a cutoff and "
            "vacuum-padding artifact rather than a physical total. Use the "
            "four-center route (bare 1/r minimum image puts all channels in one "
            "gauge); the slab mixed-boundary kernel is not implemented "
            "(docs/aiccm2026dev_a_lowd_greens.md section 9 partial in-plane FT)."
        )
    raise NotImplementedError(
        f"{who}: unsupported periodic dimension {dim}")


def _reject_vacuum_padded_direct(ccm, who="run_ccm_rhf_direct"):
    """Refuse a declared-3-D cell that is *measured* as sub-three-dimensional.

    IID 291: the declared-dimension check in
    :func:`_reject_low_dimensional_direct` was evaded by padding a 1-D chain
    with transverse vacuum and declaring the cell 3-D. The 3-D neutral
    construction then applies its BvK exchange-``q=0`` seam for transverse
    images that do not exist, and the Paper-1 SI chain run produced a
    positive ``N1 = 1`` energy and an energy ladder that diverges with
    cluster size. The measurement is :func:`ccm_vacuum_directions`
    (largest empty inter-atom slab per axis, see
    :mod:`vibeqc.periodic.ccm.system`); the same physics rationale as the
    declared-dim refusal applies unchanged.
    """
    from .system import ccm_vacuum_directions

    vacuum = ccm_vacuum_directions(ccm)
    if not vacuum:
        return
    axes = ", ".join(str(i + 1) for i in vacuum)
    raise NotImplementedError(
        f"{who}: the cell is declared 3-D but lattice direction(s) "
        f"{axes} carry only vacuum padding (measured from the atom "
        "positions, not the declaration -- IID 291). The 3-D neutral "
        "construction has no transverse periodic images to correct there, "
        "so its BvK exchange-q0 seam corrects for images that do not exist "
        "and the energy is a vacuum-padding artifact. Declare the system "
        "with its physical dimensionality (the route then fails closed with "
        "the dim < 3 message), or use the four-center route (bare 1/r "
        "minimum image puts all channels in one gauge)."
    )


def run_ccm_rhf_direct(
    ccm, *, initial_guess: object = "AUTO", cderi=None, cderi_build="fold", ke_cutoff=200.0,
    tail_ke_cutoff=None,
    aux_basis=None, cderi_symmetry=None, exxdiv="ewald", lat_opts=None,
    max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6, diis_dim=8, lindep_tol=1e-7,
):
    """Closed-shell HF for the neutral fitted-torus **real-Γ control**.

    This is one real Γ-supercell SCF (no k-mesh) that reproduces
    :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` to numerical implementation
    tolerance because both evaluate the same specified block-circulant
    Hamiltonian. Any RI fitting error is common to both representations. This
    is not the union-and-weight Γ-CCM construction.

    Assembles the neutral torus Hamiltonian in pure real algebra -- Ewald-gauge
    ``S``/``h``/``E_nn`` (:func:`ccm_direct_oneelectron`), RI-J/RI-K from the
    neutral Γ-supercell cderi ``L``, and the derived exchange-``q=0`` seam

        F = h + J(D) - 1/2 [ K(D) + ξ_N S D S ]        (exxdiv="ewald")

    with ``ξ_N`` the supercell Madelung constant
    (:func:`ccm_exchange_q0_madelung`; see the module docstring for the
    derivation and why ``F = ∂E/∂D`` holds). One real generalized eigenproblem
    per iteration over the ``N_c·n_μ`` supercell AOs.

    Parameters
    ----------
    cderi : ndarray, optional
        Reuse an already-built neutral cderi ``L`` (else built per
        ``cderi_build`` at ``ke_cutoff`` / ``aux_basis``). Passing
        ``aux_basis`` together with ``cderi`` raises :class:`ValueError`:
        the cderi already fixes the fitting basis, so the aux basis could
        not take effect.
    cderi_build : {"fold", "supercell"}
        How to build ``L`` when ``cderi`` is not supplied. ``"fold"``
        (default): per-q **unit-cell** fits folded onto the torus
        (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_fold`) -- the
        multi-k GDF's own fit objects, so GDF parity holds by construction
        (LiH (2,1,1): 4e-14 where the supercell fit misses by 1.35e-4), at
        the GDF's cost and MB-scale memory. ``"supercell"``: one Γ-supercell
        FFT fit (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`) --
        memory grows with supercell volume (49 GB measured at a 54-AO
        18³-bohr torus) and carries the open ultra-diffuse asymmetry
        finding; kept for validation/back-compat.
    cderi_symmetry : None | bool | "auto" | CCMSymmetry
        Forwarded to the cderi builder's ``symmetry``: the space-group
        k-pair star reduction of the fold's builder calls
        (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_fold`) /
        the pair reduction of the supercell fit
        (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`).
        Ignored when ``cderi`` is supplied.
    tail_ke_cutoff : float, optional
        RSGDF high-``|G|`` tail completion cutoff (Ha) for the cderi build.
        ``None`` (default) resolves through
        :func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_tail_ke_cutoff`, which
        mirrors what the multi-k GDF sibling would do at this ``nrep`` -- the
        two routes must run the same Hamiltonian, and until 2026-08-27 they
        did not (GitLab IID 307: -4.99e-01 Ha/cell on MgO/STO-3G at
        ``nrep=(1,1,1)``, where the Γ fast path auto-tailed and this route had
        no way to). ``0`` -- or any value at or below ``ke_cutoff``, which the
        builders would not sweep anyway -- opts out, deliberately keeping the
        dense-core parity hold for fast diagnostic runs, and resolves to
        ``None`` so the call is the historical one exactly. Raises when
        combined with a prebuilt ``cderi``, whose tail is already fixed. The
        applied value is recorded on the result as
        ``.rsgdf_tail_ke_cutoff``.
    exxdiv : {"ewald", None}
        Exchange-``q=0`` convention. ``"ewald"`` (matches the GDF /
        external KRHF) applies the seam; ``None`` deliberately omits it (the
        strict-zero-mode Hamiltonian -- a *distinct* finite-``N`` Hamiltonian,
        offset ``+ξ_N·N_e/(2 N_c)`` per unit cell -- kept for the pinned
        seam-demonstration control).
    lat_opts : LatticeSumOptions, optional
        Base lattice-sum options for the one-electron/nuclear sums (defaults
        to the GDF driver's defaults).
    conv_tol_grad : float
        DIIS-commutator residual bound (default ``1e-6``, the historical gate).
        ``conv_tol`` gates the energy criterion only; tighten both when
        converged densities matter.

    Returns
    -------
    CCMSCFResult
        Energies are **per supercell** (as all supercell-Γ CCM drivers);
        ``.exchange_q0`` records the convention label
        (:func:`~vibeqc.periodic.exchange_convention.exchange_q0_label`) --
        assert it matches in every cross-route comparison
        (``assert_matched_exchange_q0``).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf_direct',
    )
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_direct"
    )
    _warn_experimental()
    from vibeqc.periodic.exchange_convention import exchange_q0_label

    if exxdiv not in ("ewald", None):
        raise ValueError(
            f"run_ccm_rhf_direct: exxdiv must be 'ewald' or None, got {exxdiv!r}"
        )
    dim = int(ccm.unit_system.dim)
    if dim != 3:
        _reject_low_dimensional_direct(dim)
    _reject_vacuum_padded_direct(ccm)
    if cderi_build not in ("supercell", "fold"):
        raise ValueError(
            "run_ccm_rhf_direct: cderi_build must be 'supercell' or 'fold', "
            f"got {cderi_build!r}"
        )
    L = _neutral_cderi_for(ccm, cderi, cderi_build, ke_cutoff, aux_basis,
                           cderi_symmetry,
                           tail_ke_cutoff=tail_ke_cutoff)
    S, h, e_nn = ccm_direct_oneelectron(ccm, lat_opts=lat_opts)

    if exxdiv == "ewald":
        xi = ccm_exchange_q0_madelung(ccm)
        # The seam as a Fock operator: K -> K + ξ_N S D S (linear symmetric in
        # D, so F = ∂E/∂D holds through the standard 1/2 Tr[D(h+F)] functional).
        k_of_D = lambda D: _k_neutral_fast(L, D, S, xi)
    else:
        k_of_D = lambda D: _k_neutral_fast(L, D)

    res = _rhf_loop(
        ccm, S, h, e_nn,
        lambda D: ccm_ri_j_neutral(L, D), k_of_D,
        max_iter=max_iter, conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis_dim=diis_dim, lindep_tol=lindep_tol)
    res.backend = "ccm-neutral-direct-rhf"  # result-backend identity (IID 344)
    res.rsgdf_tail_ke_cutoff = (
        None if cderi is not None
        else ccm_neutral_tail_ke_cutoff(
            ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff))
    res.exchange_q0 = exchange_q0_label(exxdiv)
    return _with_ccm_guess(res, guess_selection)


def run_ccm_uhf_direct(
    ccm, *, initial_guess: object = "AUTO", cderi=None, cderi_build="fold", ke_cutoff=200.0,
    tail_ke_cutoff=None,
    aux_basis=None, cderi_symmetry=None, exxdiv="ewald", lat_opts=None,
    max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6, diis_dim=8,
    lindep_tol=1e-7,
):
    """Open-shell UHF for the neutral fitted-torus **real-Γ control**.

    This is the open-shell sibling of :func:`run_ccm_rhf_direct` and the
    BvK-ewald reference the open-shell RI correlation pairs with
    (``run_ccm_ump2(ccm, uhf, cderi=L)``).

    Same Hamiltonian as the closed-shell route (Ewald-gauge ``S``/``h``/``E_nn``
    from :func:`ccm_direct_oneelectron`, RI-J/RI-K from the neutral cderi ``L``),
    solved with two independent spin densities, and the exchange-``q=0`` seam
    applied **per spin**:

        F_σ = h + J(D_a + D_b) - [ K(D_σ) + ξ_N S D_σ S ]      (exxdiv="ewald")

    with ``ξ_N`` the supercell Madelung constant (:func:`ccm_exchange_q0_madelung`,
    shared verbatim with the multi-k ``exxdiv='ewald'`` shift and the closed-shell
    route). The per-spin seam ``ξ_N S D_σ S`` mirrors ``K(D_σ)``'s per-spin
    structure, so it is linear+symmetric in ``D_σ`` (``F_σ = ∂E/∂D_σ`` holds) and
    **collapses to the RHF seam for a closed shell**: with ``D_a = D_b = D/2`` and
    ``K`` linear, ``F_σ = h + J(D) - ½[K(D) + ξ_N S D S] = F_RHF``. The converged
    seam energy is ``-½ ξ_N (N_a + N_b) = -ξ_N N_e/2`` per supercell -- the same
    total shift as the closed-shell route (each ``Tr[D_σ S D_σ S] = N_σ`` at an
    idempotent ``D_σ``), so the ``exxdiv=None`` control sits above ``"ewald"`` by
    exactly ``ξ_N N_e/(2 N_c)`` per unit cell with the same converged densities.

    Parameters mirror :func:`run_ccm_rhf_direct` (``cderi`` / ``cderi_build`` /
    ``cderi_symmetry`` / ``exxdiv`` / ``lat_opts`` / SCF controls), plus
    ``conv_tol_grad`` -- the DIIS-commutator residual bound (default ``1e-6``,
    the historical gate; molecular ``UHFOptions.conv_tol_grad`` convention).
    Tighten it, not just ``conv_tol``, when converged *densities* matter: the
    energy is stationary, so ``|dE|`` is quadratic in the density error and an
    energy-only criterion can stop with a ~1e-5-loose minority-spin density.
    Spin counts come from the cluster supercell's charge + multiplicity.
    ``dim < 3`` fails closed (:func:`_reject_low_dimensional_direct`). Returns
    a :class:`~vibeqc.periodic.ccm.uhf.CCMUHFResult` (energies **per
    supercell**, ``.exchange_q0`` recorded).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_uhf_direct',
    )
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_uhf_direct"
    )
    _warn_experimental()
    from vibeqc.periodic.ccm.scf import (
        _diis_extrapolate,
        _orthonormaliser,
        _require_retained_occ,
    )
    from vibeqc.periodic.ccm.uhf import CCMUHFResult
    from vibeqc.periodic.exchange_convention import exchange_q0_label

    if exxdiv not in ("ewald", None):
        raise ValueError(
            f"run_ccm_uhf_direct: exxdiv must be 'ewald' or None, got {exxdiv!r}"
        )
    dim = int(ccm.unit_system.dim)
    if dim != 3:
        _reject_low_dimensional_direct(dim, who="run_ccm_uhf_direct")
    _reject_vacuum_padded_direct(ccm, who="run_ccm_uhf_direct")
    if cderi_build not in ("supercell", "fold"):
        raise ValueError(
            "run_ccm_uhf_direct: cderi_build must be 'supercell' or 'fold', "
            f"got {cderi_build!r}"
        )
    L = _neutral_cderi_for(ccm, cderi, cderi_build, ke_cutoff, aux_basis,
                           cderi_symmetry,
                           tail_ke_cutoff=tail_ke_cutoff)
    S, h, e_nn = ccm_direct_oneelectron(ccm, lat_opts=lat_opts)

    xi = ccm_exchange_q0_madelung(ccm) if exxdiv == "ewald" else 0.0

    def k_seam(D_sigma):
        # Per-spin exchange + the exchange-q=0 seam (mirrors K(D_σ); linear
        # symmetric in D_σ, so F_σ = ∂E/∂D_σ holds and it reduces to the RHF
        # ½(K + ξ_N S D S) for D_a = D_b = D/2). Occupied-rank factored
        # contraction (D_σ = C_σ C_σ^T is aufbau-PSD per spin).
        return _k_neutral_fast(L, D_sigma, S, xi)

    n_elec = ccm.supercell.n_electrons()
    mult = int(ccm.supercell.multiplicity)
    n_unpaired = mult - 1
    if (n_elec - n_unpaired) % 2 != 0 or n_unpaired > n_elec:
        raise ValueError(
            f"run_ccm_uhf_direct: cluster electron count {n_elec} and "
            f"multiplicity {mult} are incompatible (n_alpha/n_beta non-integer)."
        )
    n_alpha = (n_elec + n_unpaired) // 2
    n_beta = n_elec - n_alpha

    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(
        X, max(n_alpha, n_beta), who="run_ccm_uhf_direct", lindep_tol=lindep_tol)

    def diag(F):
        eps, Cp = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ Cp

    def density(C, n_occ):
        Co = C[:, :n_occ]
        return Co @ Co.T                       # UHF spin density (no factor 2)

    eps_a, Ca = diag(h)
    eps_b, Cb = diag(h)
    Da = density(Ca, n_alpha)
    Db = density(Cb, n_beta)

    diis_Fa, diis_Fb, diis_e = [], [], []
    e_last, e_elec, e_tot, conv, it = 0.0, 0.0, 0.0, False, 0
    for it in range(1, max_iter + 1):
        J = ccm_ri_j_neutral(L, Da + Db)
        Fa = h + J - k_seam(Da)
        Fb = h + J - k_seam(Db)
        Fa = 0.5 * (Fa + Fa.T)
        Fb = 0.5 * (Fb + Fb.T)

        e_elec = 0.5 * np.sum((Da + Db) * h + Da * Fa + Db * Fb)
        e_tot = e_elec + e_nn

        err_a = X.T @ (Fa @ Da @ S - S @ Da @ Fa) @ X
        err_b = X.T @ (Fb @ Db @ S - S @ Db @ Fb) @ X
        err = np.stack([err_a, err_b])
        if len(diis_Fa) == diis_dim:
            diis_Fa.pop(0); diis_Fb.pop(0); diis_e.pop(0)
        diis_Fa.append(Fa); diis_Fb.append(Fb); diis_e.append(err)
        if len(diis_Fa) >= 2:
            Fa = _diis_extrapolate(diis_Fa, diis_e)
            Fb = _diis_extrapolate(diis_Fb, diis_e)

        eps_a, Ca = diag(Fa)
        eps_b, Cb = diag(Fb)
        Da = density(Ca, n_alpha)
        Db = density(Cb, n_beta)

        # The energy is stationary at convergence, so |dE| is QUADRATIC in the
        # density error -- an energy-only gate can report converged=True with a
        # ~1e-5-loose minority-spin density. The DIIS-residual bound is
        # therefore an explicit, user-tightenable criterion (conv_tol_grad;
        # default 1e-6 = the historical gate), the molecular
        # UHFOptions/UKSOptions.conv_tol_grad convention.
        if it > 1 and abs(e_tot - e_last) < conv_tol \
                and np.max(np.abs(err)) < conv_tol_grad:
            conv = True; e_last = e_tot; break
        e_last = e_tot

    sz = 0.5 * (n_alpha - n_beta)
    ovlp_ab = Ca[:, :n_alpha].T @ S @ Cb[:, :n_beta]
    s_squared = float(sz * (sz + 1.0) + n_beta - np.sum(ovlp_ab ** 2))
    idem = max(np.linalg.norm(Da @ S @ Da - Da), np.linalg.norm(Db @ S @ Db - Db))
    res = CCMUHFResult(
        converged=conv, n_iter=it, energy=e_tot,
        energy_per_atom=e_tot / ccm.n_atoms, e_electronic=e_elec, e_nuclear=e_nn,
        n_alpha=n_alpha, n_beta=n_beta, s_squared=s_squared,
        mo_energies_alpha=eps_a, mo_energies_beta=eps_b,
        mo_coeffs_alpha=Ca, mo_coeffs_beta=Cb,
        density_alpha=Da, density_beta=Db, overlap=S,
        idempotency_error=float(idem))
    res.backend = "ccm-neutral-direct-uhf"  # result-backend identity (IID 344)
    res.rsgdf_tail_ke_cutoff = (
        None if cderi is not None
        else ccm_neutral_tail_ke_cutoff(
            ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff))
    res.exchange_q0 = exchange_q0_label(exxdiv)
    return _with_ccm_guess(res, guess_selection)


def run_ccm_uks_direct(
    ccm, functional="pbe", *, initial_guess: object = "AUTO", cderi=None, cderi_sr=None, cderi_build="fold",
    ke_cutoff=200.0, tail_ke_cutoff=None,
    aux_basis=None, cderi_symmetry=None, exxdiv="ewald", lat_opts=None,
    grid_options=None, becke_image_radius_bohr=10.0,
    max_iter=128, conv_tol=1e-8, conv_tol_grad=1e-6, diis_dim=8,
    lindep_tol=1e-7,
):
    """Open-shell KS for the neutral fitted-torus real-Γ control.

    This is the spin-polarized sibling of :func:`run_ccm_rks_direct`, completing
    the RHF/UHF/RKS/UKS matrix for one already specified neutral-torus
    Hamiltonian. It is not the union-and-weight Γ-CCM construction.

    Spin-polarized Kohn-Sham on the torus:

        F_sigma = h + J(D_a + D_b) + V_xc^sigma
                    - a_x [ K(D_sigma) + xi_N S D_sigma S ]

    with ``a_x`` the functional's exact-exchange fraction (0 for pure
    LDA/GGA/meta-GGA), ``J``/``K`` the neutral-cderi RI contractions, and the
    exchange-``q=0`` seam applied **per spin on the exact-exchange channel
    only**, scaled by ``a_x`` -- the composition of :func:`run_ccm_uhf_direct`'s
    per-spin seam (coefficient 1 on full exchange) with
    :func:`run_ccm_rks_direct`'s ``a_x`` scaling. Each per-spin seam term is
    linear+symmetric in ``D_sigma`` (``F_sigma = dE/dD_sigma`` holds), and for a
    closed shell (``D_a = D_b = D/2``) the whole Fock collapses to
    :func:`run_ccm_rks_direct`'s ``h + J + V_xc - (a_x/2)[K + xi_N S D S]``
    exactly, so the open-shell route inherits the closed-shell route's
    GDF / external-KRKS parity transitively.

    The XC term is built **periodically per spin, by the same construction as
    the closed-shell route** (the b3f74aa9-validated density/grid pairing):
    each spin density is cell-averaged to unit-cell lattice blocks
    (:func:`_supercell_density_to_lattice_blocks` is linear, so it applies per
    spin), fed to the spin-polarized C++ kernel ``build_xc_periodic_uks``
    (which requires ``Functional(name, 2)`` -- the libxc ``eval_polarised``
    path; per-spin one-particle densities, no factor 2) on the **periodic
    Becke grid** of the unit cell, and the returned ``V_xc^alpha(g)`` /
    ``V_xc^beta(g)`` lattice sets are folded onto the torus. ``e_xc`` comes
    back as the total per unit cell and is scaled to the supercell.

    Spin counts come from the cluster supercell's charge + multiplicity
    (exactly :func:`run_ccm_uhf_direct`), and ``conv_tol_grad`` is that
    route's explicit DIIS-residual bound (default ``1e-6``; tighten it when
    converged densities matter -- see :func:`run_ccm_uhf_direct`).
    Screened (HSE-class) hybrids run per the closed-shell route's
    EXPERIMENTAL screened-exchange contract (see :func:`run_ccm_rks_direct`),
    applied per spin: ``K_sr(D_sigma) = K[L_sr](D_sigma) + sigma_0 S
    D_sigma S`` with coefficient ``c_sr`` and **no** exchange-q=0 seam
    (finite kernel, eta = 0; ``exchange_q0_applicability = "inactive"``).
    Range-separated functionals with a full-range arm, and double hybrids,
    fail closed with pointers, like the closed-shell route. ``dim < 3``
    fails closed (:func:`_reject_low_dimensional_direct`).

    Returns a :class:`~vibeqc.periodic.ccm.dft.CCMKSResult` with
    ``open_shell=True`` (energies **per supercell**; ``mo_energies`` /
    ``mo_coeffs`` hold the alpha channel, ``density`` the total
    ``P_a + P_b``, per-spin data in the ``*_beta`` / ``density_*`` fields)
    and ``.exchange_q0`` recorded.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_uks_direct',
    )
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_uks_direct"
    )
    _warn_experimental()
    from vibeqc._vibeqc_core import Functional
    from vibeqc.periodic.ccm.dft import CCMKSResult
    from vibeqc.periodic.ccm.scf import (
        _diis_extrapolate,
        _orthonormaliser,
        _require_retained_occ,
    )
    from vibeqc.periodic.exchange_convention import exchange_q0_label

    if exxdiv not in ("ewald", None):
        raise ValueError(
            f"run_ccm_uks_direct: exxdiv must be 'ewald' or None, got {exxdiv!r}"
        )
    dim = int(ccm.unit_system.dim)
    if dim != 3:
        _reject_low_dimensional_direct(dim, who="run_ccm_uks_direct")
    _reject_vacuum_padded_direct(ccm, who="run_ccm_uks_direct")
    if cderi_build not in ("supercell", "fold"):
        raise ValueError(
            "run_ccm_uks_direct: cderi_build must be 'supercell' or 'fold', "
            f"got {cderi_build!r}"
        )
    # Spin-polarized functional object: build_xc_periodic_uks requires the
    # libxc eval_polarised path (spin=1 throws inside the kernel).
    func = Functional(functional, 2)
    _reject_external_xc_neutral_hamiltonian(
        ccm, func, who="run_ccm_uks_direct"
    )
    if (
        bool(getattr(func, "is_external", False))
        and float(becke_image_radius_bohr) <= 0.0
    ):
        raise ValueError(
            "run_ccm_uks_direct: external XC requires "
            "becke_image_radius_bohr > 0"
        )
    grid_options = _external_xc_grid_options(
        func, grid_options, who="run_ccm_uks_direct")
    exx_screened = None
    if bool(func.is_range_separated):
        from vibeqc.periodic_screened_exchange import resolve_periodic_exchange

        # Shared periodic RSH policy (see run_ccm_rks_direct): HSE-class
        # screened hybrids resolve; full-range-arm RSH raises here.
        exx_screened = resolve_periodic_exchange(
            func, where="run_ccm_uks_direct")
    if bool(getattr(func, "is_double_hybrid", False)):
        raise NotImplementedError(
            "run_ccm_uks_direct: open-shell double hybrids are not supported "
            "on this route (the closed-shell scaled-MP2 composition is "
            "run_ccm_double_hybrid_direct; the open-shell sibling needs a "
            "spin-pure ROKS-CCM half that does not exist yet); use a plain "
            "global hybrid."
        )
    alpha = float(func.hf_exchange_fraction)

    L = _neutral_cderi_for(ccm, cderi, cderi_build, ke_cutoff, aux_basis,
                           cderi_symmetry,
                           tail_ke_cutoff=tail_ke_cutoff)
    S, h, e_nn = ccm_direct_oneelectron(ccm, lat_opts=lat_opts)

    n_elec = ccm.supercell.n_electrons()
    mult = int(ccm.supercell.multiplicity)
    n_unpaired = mult - 1
    if (n_elec - n_unpaired) % 2 != 0 or n_unpaired > n_elec:
        raise ValueError(
            f"run_ccm_uks_direct: cluster electron count {n_elec} and "
            f"multiplicity {mult} are incompatible (n_alpha/n_beta non-integer)."
        )
    n_alpha = (n_elec + n_unpaired) // 2
    n_beta = n_elec - n_alpha

    # Periodic spin-polarized XC on the unit cell, folded onto the torus --
    # identical grid/cells/density pairing to run_ccm_rks_direct, per spin.
    xc_of_spins = _real_gamma_uks_xc_builder(
        ccm,
        func,
        grid_options=grid_options,
        becke_image_radius_bohr=becke_image_radius_bohr,
        lat_opts=lat_opts,
    )

    if exx_screened is not None and exx_screened.is_screened:
        # HSE-class screened hybrid (EXPERIMENTAL), per spin: the resolver
        # guarantees c_full == 0, so the whole exact exchange is the fitted
        # erfc-SR kernel + its finite zero mode -- no exchange-q=0 seam
        # (finite kernel; the aiccm2026dev-b contract's eta = 0). c_sr is
        # folded into k_seam with alpha = 1, so the per-spin assembly below
        # realises F_sigma -= c_sr K_sr(D_sigma) and
        # E_x = -(c_sr/2) sum_sigma Tr[D_sigma K_sr(D_sigma)].
        from .neutral import ccm_sr_exchange_zero_mode_constant

        L_sr = _neutral_cderi_for(
            ccm, cderi_sr, cderi_build, ke_cutoff, aux_basis, cderi_symmetry,
            omega_screen=exx_screened.omega_screen,
            tail_ke_cutoff=(None if cderi_sr is not None
                            else tail_ke_cutoff))
        sigma0 = ccm_sr_exchange_zero_mode_constant(
            ccm, exx_screened.omega_screen)
        c_sr = float(exx_screened.c_sr)
        alpha = 1.0

        def k_seam(D_sigma):
            return c_sr * _k_neutral_fast(L_sr, D_sigma, S, sigma0)
    else:
        xi = ccm_exchange_q0_madelung(ccm) \
            if (alpha != 0.0 and exxdiv == "ewald") else 0.0

        def k_seam(D_sigma):
            # Per-spin exact exchange + the exchange-q=0 seam, exactly
            # run_ccm_uhf_direct's operator; the a_x scaling is applied at
            # the Fock/energy assembly below (so F_sigma = dE/dD_sigma
            # holds through the a_x-scaled channel, the
            # run_ccm_rks_direct convention). Occupied-rank factored
            # contraction (per-spin aufbau-PSD density).
            return _k_neutral_fast(L, D_sigma, S, xi)

    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(
        X, max(n_alpha, n_beta), who="run_ccm_uks_direct", lindep_tol=lindep_tol)

    def diag(F):
        eps, Cp = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ Cp

    def density(C, n_occ):
        Co = C[:, :n_occ]
        return Co @ Co.T                       # per-spin density (no factor 2)

    eps_a, Ca = diag(h)
    eps_b, Cb = diag(h)
    Da = density(Ca, n_alpha)
    Db = density(Cb, n_beta)

    diis_Fa, diis_Fb, diis_e = [], [], []
    e_last, e_tot, e_xc, e_coul, e_hf_x, conv, it = 0.0, 0.0, 0.0, 0.0, 0.0, False, 0
    Fa = Fb = h
    for it in range(1, max_iter + 1):
        J = ccm_ri_j_neutral(L, Da + Db)
        e_xc, Va, Vb = xc_of_spins(Da, Db)
        Fa = h + J + Va
        Fb = h + J + Vb
        e_hf_x = 0.0
        if alpha != 0.0:
            Ka = k_seam(Da)
            Kb = k_seam(Db)
            Fa = Fa - alpha * Ka
            Fb = Fb - alpha * Kb
            # E_x^exact = -(a_x/2) sum_sigma Tr[D_sigma K_sigma] (the UHF
            # exchange energy scaled by a_x; K_sigma includes the seam, so the
            # converged seam energy is -a_x xi_N N_e/2 per supercell).
            e_hf_x = -0.5 * alpha * (np.sum(Da * Ka) + np.sum(Db * Kb))
        Fa = 0.5 * (Fa + Fa.T)
        Fb = 0.5 * (Fb + Fb.T)

        # The RHF-style 1/2 Tr[D(h+F)] identity does not hold with V_xc in F;
        # assemble the spin-polarized KS energy explicitly (the _rks_loop
        # convention, per spin).
        e_coul = 0.5 * np.sum((Da + Db) * J)
        e_tot = np.sum((Da + Db) * h) + e_coul + e_hf_x + e_xc + e_nn

        err_a = X.T @ (Fa @ Da @ S - S @ Da @ Fa) @ X
        err_b = X.T @ (Fb @ Db @ S - S @ Db @ Fb) @ X
        err = np.stack([err_a, err_b])
        # The energy is stationary at convergence, so |dE| is quadratic in
        # the density error.  Keep the explicit, user-tightenable commutator
        # bound (conv_tol_grad) alongside the energy criterion.
        # Fa/Fb, all energy components, and the residual above describe the
        # accepted spin densities Da/Db.  Terminate on that physical state;
        # Pulay extrapolation is used only to generate the next trial.
        if it > 1 and abs(e_tot - e_last) < conv_tol \
                and np.max(np.abs(err)) < conv_tol_grad:
            conv = True
            e_last = e_tot
            break
        if it == max_iter:
            e_last = e_tot
            break
        if len(diis_Fa) == diis_dim:
            diis_Fa.pop(0); diis_Fb.pop(0); diis_e.pop(0)
        diis_Fa.append(Fa); diis_Fb.append(Fb); diis_e.append(err)
        trial_Fa, trial_Fb = Fa, Fb
        if len(diis_Fa) >= 2:
            trial_Fa = _diis_extrapolate(diis_Fa, diis_e)
            trial_Fb = _diis_extrapolate(diis_Fb, diis_e)

        eps_a, Ca = diag(trial_Fa)
        eps_b, Cb = diag(trial_Fb)
        Da = density(Ca, n_alpha)
        Db = density(Cb, n_beta)
        e_last = e_tot

    # Report canonical orbitals of the physical terminal Focks, never the
    # extrapolated Pulay trial used to reach the accepted spin densities.
    eps_a, Ca = diag(Fa)
    eps_b, Cb = diag(Fb)
    res = CCMKSResult(
        converged=conv, n_iter=it, energy=float(e_tot),
        energy_per_atom=float(e_tot) / ccm.n_atoms,
        e_xc=float(e_xc), e_coulomb=float(e_coul),
        e_hf_exchange=float(e_hf_x), functional=functional,
        mo_energies=eps_a, mo_coeffs=Ca, density=Da + Db, fock=Fa, overlap=S,
        open_shell=True,
        density_alpha=Da, density_beta=Db,
        mo_energies_beta=eps_b, mo_coeffs_beta=Cb, fock_beta=Fb)
    # Preserve the exact supercell one-electron operator for the real-Gamma
    # runner adapter.  Like the physical spin Focks above it is an AO matrix,
    # not a per-cell-normalized scalar.
    res.hcore = np.asarray(h)
    res.backend = "ccm-neutral-direct-uks"  # result-backend identity (IID 344)
    res.rsgdf_tail_ke_cutoff = (
        None if cderi is not None
        else ccm_neutral_tail_ke_cutoff(
            ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff))
    res.exchange_q0 = exchange_q0_label(exxdiv)
    # Same applicability semantics as run_ccm_rks_direct: "active" iff a
    # full-range exact-exchange arm is present (seam-carrying); pure and
    # screened-only (HSE-class) runs carry no seam (eta = 0).
    res.exchange_q0_applicability = (
        "active"
        if (exx_screened is None and float(func.hf_exchange_fraction) != 0.0)
        else "inactive")
    return _with_ccm_guess(res, guess_selection)


from dataclasses import dataclass as _dataclass


def _run_gdf_control_with_gradient(ccm, functional, closed, aux_basis,
                                   gdf_kwargs, *, who):
    """Run the multi-k GDF control with ``compute_gradient=True``.

    Shared by :func:`run_ccm_direct_gradient` (single point) and
    :func:`run_ccm_direct_optimize` (per relaxation step). Returns
    ``(control, gradient)`` with ``gradient`` verified present.
    """
    from .ri import (
        run_ccm_rhf_gdf,
        run_ccm_rks_gdf,
        run_ccm_uhf_gdf,
        run_ccm_uks_gdf,
    )

    kw = dict(gdf_kwargs)
    kw.setdefault("aux_basis", aux_basis)
    # The analytic gradient differentiates the cached-Lpq rsgdf fit; the
    # closed-shell KS driver's Ewald-3D J/K fallback has no fit to
    # differentiate (its driver raises without use_compcell=True). The
    # open-shell drivers take no such kwarg. Whatever path runs, the
    # parity gate verifies the energies coincide before any force is
    # attributed.
    if closed:
        kw.setdefault("use_compcell", True)
    kw["compute_gradient"] = True
    if functional is None:
        control = (run_ccm_rhf_gdf(ccm, **kw) if closed
                   else run_ccm_uhf_gdf(ccm, **kw))
    else:
        control = (run_ccm_rks_gdf(ccm, functional, **kw) if closed
                   else run_ccm_uks_gdf(ccm, functional, **kw))
    if not control.converged:
        raise ValueError(
            f"{who}: the multi-k GDF control SCF did not converge; no "
            "gradient premise to verify."
        )
    gradient = getattr(control.raw, "gradient", None)
    if gradient is None:
        raise RuntimeError(
            f"{who}: the multi-k driver returned no gradient "
            "(compute_gradient=True was requested); its fail-closed "
            "envelope guards should have raised instead."
        )
    return control, gradient


@_dataclass
class CCMDirectGradientResult:
    """Analytic nuclear gradient of the direct-torus energy (EXPERIMENTAL).

    Obtained through the **Fourier-representation identity** this route is
    built on: the real-Γ supercell state and the multi-k GDF state are two
    evaluations of the SAME block-circulant neutral fitted-torus
    Hamiltonian (the fold identity, verified per run by the parity gate
    below), so the production multi-k rsgdf analytic gradient — which
    differentiates exactly the shared-q fitted objective, exchange-q=0
    seam included (`periodic_gdf_gradient.py`, G-PBC-002 Item-4 rung 6) —
    IS the direct-route gradient wherever the two energies coincide.

    ``gradient`` is dE_cell/dR per unit-cell atom (Ha/bohr) under the
    cyclic constraint (displacing an atom moves all its periodic images);
    the per-supercell direct energy differentiates to
    ``N_c * gradient``. ``parity_residual_ha_per_cell`` is the measured
    |E_direct/N_c − E_gdf| the same-Hamiltonian premise was verified at —
    runs violating ``parity_tol`` raise instead of returning forces
    (the documented supercell-fit residual class, ~1e-4 Ha/cell on
    ultra-diffuse/covalent multi-cell fixtures, would otherwise leak an
    unquantified d(residual)/dR into the forces).
    """

    gradient: np.ndarray              # (n_unit_atoms, 3), dE_cell/dR
    e_direct_per_cell: float
    e_gdf_per_cell: float
    parity_residual_ha_per_cell: float
    parity_tol: float
    exchange_q0: str
    exchange_q0_applicability: str
    scf: object                       # the direct-route SCF result
    gdf: object                       # the multi-k control (CCMGDFResult)

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.scf, "guess_selection", None)


def run_ccm_direct_gradient(
    ccm, scf_result=None, *, initial_guess: object = "AUTO", functional=None, exxdiv="ewald",
    parity_tol=1e-6, cderi=None, cderi_build="fold", ke_cutoff=200.0,
    tail_ke_cutoff=None,
    aux_basis=None, cderi_symmetry=None, **gdf_kwargs,
):
    """Analytic forces for the real-Γ direct-torus route (EXPERIMENTAL).

    Milestone 2c composition (2026-08-19): rather than duplicating the
    fit-derivative machinery (CLAUDE.md §9), the gradient rides the
    landed production multi-k rsgdf analytic gradient
    (``run_k{r,u}h{f,s}_periodic_gdf(compute_gradient=True)``, public
    since 2026-07-30; full fit derivative with the metric-pseudoinverse
    Fréchet response, Ewald-gauge V_ne/E_nn, and the ``ξ_N·S·D·S`` seam
    Pulay term) through the route's own representation identity:

    1. Converge (or receive) the direct-torus SCF.
    2. Run the multi-k GDF control on the same cell/mesh with
       ``compute_gradient=True`` — the same Hamiltonian in its Bloch
       representation.
    3. **Verify the premise numerically**: the two energies must agree to
       ``parity_tol`` per cell (the H₂/LiH-class gates sit at
       1e-11–1e-12; the documented ~1e-4 supercell-fit residual class
       fails closed with a pointer instead of silently returning forces
       of a slightly different Hamiltonian).

    Scope: 3-D cells, ``exxdiv="ewald"`` only (the multi-k control has no
    strict-zero gradient), pure functionals + global hybrids, closed and
    open shell. Screened (HSE-class) hybrids fail closed — the fitted-K
    derivative is full-range only upstream
    (``periodic_k_gdf._reject_unsupported_multik_gradient``) and the
    ``σ₀``/erfc derivative is not built; double hybrids fail closed (the
    PT2 gradient needs relaxed densities this route does not have).
    ``gdf_kwargs`` reach the multi-k driver unchanged.

    Returns :class:`CCMDirectGradientResult` (per-cell gradient
    convention; multiply by ``N_c`` for the per-supercell energy).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_direct_gradient', reference=scf_result,
    )
    if gdf_kwargs.get("atomic_spins") is not None and len(gdf_kwargs["atomic_spins"]):
        raise ValueError("CCM direct reference supports HCORE only; atomic_spins requires SAD")
    # The matched control inherits the direct reference construction.
    control_kwargs = dict(gdf_kwargs, initial_guess=(
        guess_selection.effective if guess_selection is not None else "HCORE"
    ))
    from vibeqc._vibeqc_core import Functional

    from ..exchange_convention import assert_matched_exchange_q0
    from .ri import (
        run_ccm_rhf_gdf,
        run_ccm_rks_gdf,
        run_ccm_uhf_gdf,
        run_ccm_uks_gdf,
    )

    if exxdiv != "ewald":
        raise NotImplementedError(
            "run_ccm_direct_gradient: only exxdiv='ewald' has a matched "
            "multi-k analytic gradient (the strict-zero-mode control has "
            "no production gradient); got exxdiv=" + repr(exxdiv)
        )
    if functional is not None:
        func = Functional(functional, 1)
        if bool(getattr(func, "is_range_separated", False)):
            raise NotImplementedError(
                "run_ccm_direct_gradient: screened/range-separated hybrids "
                "have no fitted-K derivative upstream (multi-k "
                "compute_gradient rejects omega_screen != 0) and the "
                "direct route's sigma0/erfc derivative is not built; use "
                "a global hybrid or a pure functional."
            )
        if bool(getattr(func, "is_double_hybrid", False)):
            raise NotImplementedError(
                "run_ccm_direct_gradient: double hybrids need the relaxed-"
                "density PT2 gradient, which this route does not have."
            )

    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1

    if scf_result is None:
        kw = dict(cderi=cderi, cderi_build=cderi_build,
                  ke_cutoff=ke_cutoff, aux_basis=aux_basis,
                  cderi_symmetry=cderi_symmetry, exxdiv=exxdiv)
        if functional is None:
            scf_result = (run_ccm_rhf_direct(ccm, **kw, initial_guess=initial_guess) if closed
                          else run_ccm_uhf_direct(ccm, **kw, initial_guess=initial_guess))
        else:
            scf_result = (run_ccm_rks_direct(ccm, functional, **kw, initial_guess=initial_guess) if closed
                          else run_ccm_uks_direct(ccm, functional, **kw, initial_guess=initial_guess))
    if not scf_result.converged:
        raise ValueError(
            "run_ccm_direct_gradient: the direct-torus SCF reference is "
            "not converged."
        )

    control, gradient = _run_gdf_control_with_gradient(
        ccm, functional, closed, aux_basis, control_kwargs,
        who="run_ccm_direct_gradient")

    # The load-bearing check: the same-Hamiltonian premise, verified at
    # this run's actual numbers (never assumed from the fixture class).
    # The q=0 convention is only material when a full-range exact-exchange
    # arm is present (pure functionals record applicability="inactive" and
    # an empty label on the GDF control -- nothing to match there).
    scf_appl = (getattr(scf_result, "exchange_q0_applicability", None)
                or "active")
    if scf_appl == "active":
        assert_matched_exchange_q0(
            [str(getattr(scf_result, "exchange_q0", "") or "<unknown>"),
             str(getattr(control, "exchange_q0", "") or "<unknown>")],
            context="run_ccm_direct_gradient")
    n_c = int(ccm.n_cells)
    e_direct_cell = float(scf_result.energy) / n_c
    e_gdf_cell = float(control.energy)
    residual = abs(e_direct_cell - e_gdf_cell)
    if residual > float(parity_tol):
        raise ValueError(
            "run_ccm_direct_gradient: the direct-vs-GDF energy parity is "
            f"{residual:.3e} Ha/cell > parity_tol={float(parity_tol):.1e}. "
            "The two representations are not evaluating the same fitted "
            "Hamiltonian at this run's numerical settings (the documented "
            "~1e-4 supercell-fit residual class -- see "
            "HANDOVER_AICCM_DIRECT_TORUS.md Findings §2), so the multi-k "
            "gradient cannot be attributed to the direct energy. No "
            "forces returned."
        )

    return CCMDirectGradientResult(
        gradient=np.asarray(gradient, dtype=float),
        e_direct_per_cell=e_direct_cell,
        e_gdf_per_cell=e_gdf_cell,
        parity_residual_ha_per_cell=residual,
        parity_tol=float(parity_tol),
        exchange_q0=str(getattr(scf_result, "exchange_q0", "") or ""),
        exchange_q0_applicability=str(
            getattr(scf_result, "exchange_q0_applicability", None)
            or "active"),
        scf=scf_result,
        gdf=control,
    )


@_dataclass
class CCMDirectOptimizeResult:
    """Geometry relaxation on the direct-torus surface (EXPERIMENTAL).

    ``system_opt`` is the relaxed **unit cell** (all periodic images move
    with each atom — the cyclic coordinate); ``gradient`` is the final
    per-cell dE/dR (Ha/bohr, cartesian). ``parity_initial`` /
    ``parity_final`` are the :class:`CCMDirectGradientResult` endpoint
    verifications: the relaxation steps ride the multi-k GDF control
    objective (``parity_check="endpoints"``, the default) or are verified
    every step (``"every-step"``), and in BOTH modes the initial and
    final geometries carry a full direct-vs-control parity check, so the
    relaxed stationary point is attributable to the direct energy at the
    measured residuals.
    """

    converged: bool
    n_steps: int
    system_opt: object               # relaxed PeriodicSystem (unit cell)
    energy_per_cell: float           # control energy at the optimum
    gradient: np.ndarray             # (n_unit_atoms, 3) cartesian, per cell
    grad_max_frac: float             # the optimizer's fractional-grad gate
    parity_initial: object           # CCMDirectGradientResult at x0
    parity_final: object             # CCMDirectGradientResult at x_opt
    parity_check: str                # "endpoints" | "every-step"

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.parity_final, "guess_selection", None)


def run_ccm_direct_optimize(
    ccm, *, initial_guess: object = "AUTO", functional=None, exxdiv="ewald", parity_check="endpoints",
    parity_tol=1e-6, conv_tol_grad=4.5e-4, max_steps=50,
    aux_basis=None, **gdf_kwargs,
):
    """Relax unit-cell atomic positions on the direct-torus surface
    (EXPERIMENTAL, milestone 2c follow-up 2026-08-20; fixed lattice).

    The relaxation objective is the multi-k GDF control energy with its
    analytic gradient — by the per-run-verified representation identity
    (see :func:`run_ccm_direct_gradient`) the same surface as the direct
    energy wherever parity holds. ``parity_check`` picks where that
    premise is verified:

    * ``"endpoints"`` (default): every optimizer step runs ONE control
      SCF+gradient (production cost — no fold build per step); the full
      direct-vs-control parity check runs at the initial and final
      geometries. If the endpoint parity holds, the relaxed stationary
      point is attributable to the direct energy at the measured
      residuals; a mid-path parity excursion would not be seen (it also
      would not move the answer: the steps genuinely relax the control
      surface, and the endpoint checks pin what the answer means).
    * ``"every-step"``: :func:`run_ccm_direct_gradient` per step — a
      direct SCF (fold build included) + a control SCF per step, parity
      verified everywhere. 2-4x the cost; use for validation runs.

    Mirrors the runner's GDF relaxer (scipy L-BFGS-B on fractional
    coordinates, ``dE/dfrac = dE/dcart @ lattice``, non-convergence
    penalty barrier, independent gradient gate) — the G-PBC-002 optimizer
    wiring pattern. Scope inherits :func:`run_ccm_direct_gradient`:
    3-D, ``exxdiv="ewald"``, pure + global hybrids, closed and open shell;
    screened/double hybrids fail closed at the initial parity check.
    ``max_steps`` must be an integer greater than or equal to one. Returns
    :class:`CCMDirectOptimizeResult`.
    """
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or int(max_steps) < 1
    ):
        raise ValueError(
            "run_ccm_direct_optimize: max_steps must be an integer >= 1; "
            f"got {max_steps!r}."
        )
    max_steps = int(max_steps)
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_direct_optimize"
    )
    from scipy.optimize import minimize

    from vibeqc.bipole_optimize import (
        SCFNonConvergence,
        _atoms_to_flat,
        _flat_to_system,
    )
    from vibeqc.molecular_optimize import _gradient_converged
    from vibeqc.periodic.ccm import CCMSystem

    if parity_check not in ("endpoints", "every-step"):
        raise ValueError(
            "run_ccm_direct_optimize: parity_check must be 'endpoints' or "
            f"'every-step', got {parity_check!r}")

    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_direct_optimize',
    )
    if gdf_kwargs.get("atomic_spins") is not None and len(gdf_kwargs["atomic_spins"]):
        raise ValueError("CCM direct reference supports HCORE only; atomic_spins requires SAD")
    # The matched control inherits the direct reference construction.
    control_kwargs = dict(gdf_kwargs, initial_guess=(
        guess_selection.effective if guess_selection is not None else "HCORE"
    ))
    unit0 = ccm.unit_system
    nrep = tuple(int(n) for n in ccm.nrep)
    basis_name = ccm.basis_name
    n_atoms = len(unit0.unit_cell)
    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1

    def ccm_at(x):
        sys2 = _flat_to_system(unit0, x)
        return CCMSystem(sys2, nrep, basis_name), sys2

    # Initial full parity verification (also enforces the fail-closed
    # surface: screened/DH/open-shell-KS raise here, before any step).
    parity_initial = run_ccm_direct_gradient(
        ccm, functional=functional, exxdiv=exxdiv, parity_tol=parity_tol,
        aux_basis=aux_basis, **gdf_kwargs, initial_guess=initial_guess)

    _PENALTY_HA = 1.0e6
    _had_good_eval = {"ok": False}

    def _energy_and_gradient(x):
        ccm2, sys2 = ccm_at(x)
        if parity_check == "every-step":
            try:
                g = run_ccm_direct_gradient(
                    ccm2, functional=functional, exxdiv=exxdiv,
                    parity_tol=parity_tol, aux_basis=aux_basis,
                    **gdf_kwargs, initial_guess=initial_guess)
            except ValueError:
                if not _had_good_eval["ok"]:
                    raise
                return _PENALTY_HA, np.zeros(3 * n_atoms)
            e_obj, grad_cart = g.e_gdf_per_cell, g.gradient
        else:
            try:
                control, grad_cart = _run_gdf_control_with_gradient(
                    ccm2, functional, closed, aux_basis, dict(control_kwargs),
                    who="run_ccm_direct_optimize")
            except ValueError:
                # Non-converged probe geometry: recoverable line-search
                # barrier, exactly the runner relaxer's pattern; only a
                # failure at the initial geometry aborts.
                if not _had_good_eval["ok"]:
                    raise SCFNonConvergence(
                        "run_ccm_direct_optimize: the control SCF did not "
                        "converge at the initial geometry; refusing to "
                        "relax against a non-converged objective.")
                return _PENALTY_HA, np.zeros(3 * n_atoms)
            e_obj = float(control.energy)
        _had_good_eval["ok"] = True
        grad_cart = np.asarray(grad_cart, dtype=float)
        # dE/d(frac) = dE/d(cart) @ lattice, row per atom -- the runner
        # relaxer's conversion (lattice columns = lattice vectors).
        grad_frac = grad_cart @ np.asarray(sys2.lattice, dtype=float)
        return float(e_obj), grad_frac.ravel()

    x0 = _atoms_to_flat(unit0)
    res = minimize(
        _energy_and_gradient, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": max_steps, "gtol": float(conv_tol_grad)})
    sys_opt = _flat_to_system(unit0, res.x)
    # Independent gradient gate: scipy's success flag can trip on ftol at
    # a non-stationary geometry (shared _gradient_converged semantics).
    converged, grad_max = _gradient_converged(
        bool(res.success), res.jac, float(conv_tol_grad))

    # Final full parity verification at the relaxed geometry -- this is
    # what makes the stationary point attributable to the direct energy.
    ccm_opt = CCMSystem(sys_opt, nrep, basis_name)
    parity_final = run_ccm_direct_gradient(
        ccm_opt, functional=functional, exxdiv=exxdiv,
        parity_tol=parity_tol, aux_basis=aux_basis, **gdf_kwargs, initial_guess=initial_guess)

    return CCMDirectOptimizeResult(
        converged=bool(converged), n_steps=int(res.nit),
        system_opt=sys_opt,
        energy_per_cell=float(parity_final.e_gdf_per_cell),
        gradient=np.asarray(parity_final.gradient, dtype=float),
        grad_max_frac=float(grad_max),
        parity_initial=parity_initial, parity_final=parity_final,
        parity_check=str(parity_check))


@_dataclass
class CCMDoubleHybridDirectResult:
    """Double-hybrid composition on the real-Γ direct-torus route
    (EXPERIMENTAL). Energies **per supercell** like every direct driver."""
    converged: bool
    energy: float                    # e_scf + e_pt2 (per supercell, Ha)
    energy_per_atom: float
    e_scf: float                     # hybrid-KS SCF half (seam included)
    e_pt2: float                     # c_os·e_os + c_ss·e_ss
    e_pt2_os: float                  # unscaled opposite-spin PT2
    e_pt2_ss: float                  # unscaled same-spin PT2
    mp2_c_os: float
    mp2_c_ss: float
    functional: str
    ks: object                       # the CCMKSResult of the SCF half
    mp2: object                      # the CCMMP2Result of the PT2 step
    exchange_q0: str = ""
    exchange_q0_applicability: str = "inactive"

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.ks, "guess_selection", None)


def run_ccm_double_hybrid_direct(
    ccm, functional="b2plyp", *, initial_guess: object = "AUTO", cderi=None, cderi_build="fold",
    ke_cutoff=200.0, tail_ke_cutoff=None,
    aux_basis=None, cderi_symmetry=None, exxdiv="ewald",
    lat_opts=None, grid_options=None, becke_image_radius_bohr=10.0,
    max_iter=128, conv_tol=1e-8, diis_dim=8, lindep_tol=1e-7,
):
    """Closed-shell double hybrid on the real-Γ direct-torus route
    (EXPERIMENTAL, 2026-07-18).

    The standard two-step double-hybrid composition (Grimme, J. Chem. Phys.
    124, 034108 (2006), Eqs. 1-3: hybrid-KS SCF, then a spin-component-
    scaled MP2 correction evaluated on the converged KS orbitals), assembled
    entirely on the direct-torus machinery:

    1. **SCF half**: :func:`run_ccm_rks_direct` with the double-hybrid
       functional (its DFT part + ``a_x``-scaled exact exchange with the
       exchange-q=0 seam, e.g. ``a_x = 0.53`` for b2plyp) — one real
       Γ-supercell KS-SCF on the neutral cderi ``L``.
    2. **PT2 half**: :func:`~vibeqc.periodic.ccm.mp2.run_ccm_mp2` on the
       converged KS orbitals/eigenvalues, **on the same** ``L`` (the
       BvK-ewald reference composition of ``run_ccm_ri_mp2(reference=
       "direct")``, here with KS orbitals), scaled by the functional's
       ``mp2_c_os`` / ``mp2_c_ss`` (b2plyp: 0.27/0.27):

           E = E_SCF[KS] + c_os·E_os^PT2 + c_ss·E_ss^PT2

    The KS eigenvalues entering the PT2 denominators carry the declared
    ``exxdiv`` convention through the SCF (each occupied shifted by
    ``-a_x·ξ_N`` under ``"ewald"``) — the same mechanism as PySCF's exxdiv
    on KMP2 denominators, measured route-side in ``HANDOVER_D2_EXXDIV.md``
    § "Part 2". This driver composes the direct (BvK-ewald) reference by
    construction; it does not touch the ``run_ccm_ri_mp2`` default
    (``reference="neutral"``), whose flip remains the pending maintainer
    decision recorded there.

    Closed shell only (the open-shell sibling needs a spin-pure ROKS-CCM
    half that does not exist yet — the molecular precedent is
    ``run_double_hybrid``'s ROKS + semicanonical ROHF-MP2 path). No
    dispersion composition here (molecular ``dispersion=`` semantics need
    the periodic D3/D4 lattice story, out of scope for this control route).
    Range-separated double hybrids fail closed through the SCF half's
    resolver. Keyword contract otherwise identical to
    :func:`run_ccm_rks_direct`.

    Returns :class:`CCMDoubleHybridDirectResult` (per supercell;
    ``.exchange_q0`` / ``.exchange_q0_applicability`` mirrored from the
    SCF half — the full-range arm makes the convention "active").
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_double_hybrid_direct',
    )
    from vibeqc._vibeqc_core import Functional

    from .mp2 import run_ccm_mp2

    func = Functional(functional, 1)
    if not bool(getattr(func, "is_double_hybrid", False)):
        raise ValueError(
            f"run_ccm_double_hybrid_direct: {functional!r} is not a "
            f"double-hybrid functional (mp2_c_os = "
            f"{float(getattr(func, 'mp2_c_os', 0.0))}, mp2_c_ss = "
            f"{float(getattr(func, 'mp2_c_ss', 0.0))}); use "
            "run_ccm_rks_direct."
        )
    n_elec = int(ccm.supercell.n_electrons())
    if n_elec % 2 != 0 or int(ccm.supercell.multiplicity) != 1:
        raise NotImplementedError(
            "run_ccm_double_hybrid_direct: closed-shell clusters only (the "
            "open-shell sibling needs a spin-pure ROKS-CCM half that does "
            "not exist yet)."
        )

    # One L for both halves -- the same-kernel consistency contract of
    # run_ccm_ri_mp2 (a mismatched reference/correlation kernel is silently
    # wrong).
    L = _neutral_cderi_for(ccm, cderi, cderi_build, ke_cutoff, aux_basis,
                           cderi_symmetry,
                           tail_ke_cutoff=tail_ke_cutoff)
    ks = run_ccm_rks_direct(
        ccm, functional, cderi=L, exxdiv=exxdiv, lat_opts=lat_opts,
        grid_options=grid_options,
        becke_image_radius_bohr=becke_image_radius_bohr,
        max_iter=max_iter, conv_tol=conv_tol, diis_dim=diis_dim,
        lindep_tol=lindep_tol, _allow_double_hybrid_scf=True, initial_guess=initial_guess)
    if not ks.converged:
        raise RuntimeError(
            f"run_ccm_double_hybrid_direct({functional!r}): the KS SCF half "
            f"did not converge (n_iter = {ks.n_iter}); tighten max_iter/"
            "conv_tol and retry."
        )

    mp2 = run_ccm_mp2(ccm, ks, cderi=L, initial_guess=initial_guess)
    c_os = float(func.mp2_c_os)
    c_ss = float(func.mp2_c_ss)
    e_pt2 = c_os * mp2.e_os + c_ss * mp2.e_ss
    e_tot = float(ks.energy) + e_pt2
    return CCMDoubleHybridDirectResult(
        converged=bool(ks.converged), energy=e_tot,
        energy_per_atom=e_tot / ccm.n_atoms,
        e_scf=float(ks.energy), e_pt2=float(e_pt2),
        e_pt2_os=float(mp2.e_os), e_pt2_ss=float(mp2.e_ss),
        mp2_c_os=c_os, mp2_c_ss=c_ss, functional=functional,
        ks=ks, mp2=mp2,
        exchange_q0=str(ks.exchange_q0 or ""),
        exchange_q0_applicability=str(ks.exchange_q0_applicability or ""))


def _direct_reference_and_cderi(
    ccm, *, initial_guess: object = "AUTO", reference, ke_cutoff, aux_basis, cderi_symmetry, exxdiv,
    fold_threads, open_shell, who,
):
    """One cderi ``L`` plus the SCF reference built on that same ``L``.

    The consistency guarantee the one-call DLPNO entry points exist to
    provide: ``ccm_dlpno_*(ccm, scf, cderi=L)`` with a ``scf`` converged on
    a different kernel is silently wrong, which is exactly the hazard
    :func:`~vibeqc.periodic.ccm.mp2.run_ccm_ri_mp2` was created to remove
    for canonical MP2.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='_direct_reference_and_cderi',
    )
    from .neutral import ccm_neutral_cderi_fold
    from .ri import run_ccm_rhf_ri_neutral

    if reference not in ("direct", "neutral"):
        raise ValueError(
            f"{who}: reference must be 'direct' or 'neutral', "
            f"got {reference!r}")
    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1
    if open_shell and closed:
        raise ValueError(
            f"{who}: this is the open-shell entry point but the cluster is "
            "closed-shell; use the closed-shell sibling.")
    if not open_shell and not closed:
        raise NotImplementedError(
            f"{who}: closed-shell clusters only; use "
            "run_ccm_dlpno_ump2_direct for open shells.")

    L = ccm_neutral_cderi_fold(
        ccm, ke_cutoff=float(ke_cutoff), aux_basis=aux_basis,
        symmetry=cderi_symmetry, fold_threads=fold_threads)
    if open_shell:
        if reference == "direct":
            scf = run_ccm_uhf_direct(ccm, cderi=L, exxdiv=exxdiv, initial_guess=initial_guess)
        else:
            from .uhf import run_ccm_uhf

            scf = run_ccm_uhf(ccm, cderi=L, initial_guess=initial_guess)
    elif reference == "direct":
        scf = run_ccm_rhf_direct(ccm, cderi=L, exxdiv=exxdiv, initial_guess=initial_guess)
    else:
        scf = run_ccm_rhf_ri_neutral(ccm, cderi=L, initial_guess=initial_guess)
    if not scf.converged:
        raise ValueError(
            f"{who}: the {reference!r} SCF reference did not converge; "
            "refusing to correlate a non-converged state.")
    return scf, L


def run_ccm_dlpno_mp2_direct(
    ccm, *, initial_guess: object = "AUTO", reference="direct", ke_cutoff=200.0, aux_basis=None,
    cderi_symmetry=None, exxdiv="ewald", fold_threads=None, **dlpno_kwargs,
):
    """One-call closed-shell DLPNO-MP2 on the BvK-ewald direct-torus
    reference (EXPERIMENTAL, 2026-08-21).

    DLPNO is the natural correlation method for a cyclic cluster: the
    Coulomb interaction decays with distance, so pair energies screen on
    the **minimum-image (BvK-torus) centroid distance** and the PNO/PAO
    domains truncate per pair. That machinery already exists
    (:func:`~vibeqc.periodic.ccm.dlpno.ccm_dlpno_mp2`, A-line follow-on
    chat) -- what this adds is a one-call entry that GUARANTEES the SCF
    reference and the cderi are the same kernel, and the option to ride
    this route's reference.

    ``reference`` selects the complete SCF route, exactly as on
    ``run_ccm_ri_mp2``:

    * ``"direct"`` (the default here) -- the BvK-ewald direct-torus SCF on
      the shared ``L``. The exchange-q=0 seam shifts every occupied
      eigenvalue by ``-xi_N``, i.e. every correlation denominator, which is
      why the choice matters: on the strict-zero neutral reference the
      correlation carries a distortion that GROWS with cluster size
      (``HANDOVER_D2_EXXDIV.md``: diamond RI-MP2 -59.0 vs KMP2 -29.9/-41.1
      mHa/atom at (2,2,2)).
    * ``"neutral"`` -- the lean strict-zero SCF, the historical default of
      the non-DLPNO one-call wrappers. Kept for continuity and for
      neutral-vs-neutral comparisons; not recommended for absolute
      correlation at multi-cell clusters.

    ``dlpno_kwargs`` reach ``ccm_dlpno_mp2`` unchanged (``localize``,
    ``tcut_pno``, ``tcut_mkn``, ``n_frozen``, ...); at the default zero
    truncations the result is the canonical RI-MP2 correlation to machine
    precision on either reference (``tests/test_ccm_direct_dlpno.py``).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_dlpno_mp2_direct',
    )
    from .dlpno import ccm_dlpno_mp2

    scf, L = _direct_reference_and_cderi(
        ccm, reference=reference, ke_cutoff=ke_cutoff, aux_basis=aux_basis,
        cderi_symmetry=cderi_symmetry, exxdiv=exxdiv,
        fold_threads=fold_threads, open_shell=False,
        who="run_ccm_dlpno_mp2_direct", initial_guess=initial_guess)
    return ccm_dlpno_mp2(ccm, scf, cderi=L, ke_cutoff=float(ke_cutoff),
                         **dlpno_kwargs)


def run_ccm_dlpno_ump2_direct(
    ccm, *, initial_guess: object = "AUTO", reference="direct", ke_cutoff=200.0, aux_basis=None,
    cderi_symmetry=None, exxdiv="ewald", fold_threads=None, **dlpno_kwargs,
):
    """One-call **open-shell** DLPNO-UMP2 on the direct-torus reference
    (EXPERIMENTAL, 2026-08-22).

    The spin-polarized sibling of :func:`run_ccm_dlpno_mp2_direct`,
    composing :func:`~vibeqc.periodic.ccm.dlpno_ump2.ccm_dlpno_ump2` on
    :func:`run_ccm_uhf_direct` (per-spin exchange-q=0 seam) with one shared
    ``L``. Open-shell ionics are exactly where the reference choice bites
    hardest, since the seam enters each spin channel's denominators.

    ``reference`` selects the complete SCF route, exactly as on
    ``run_ccm_ri_mp2``:

    * ``"direct"`` (the default here) -- the BvK-ewald direct-torus SCF on
      the shared ``L``. The exchange-q=0 seam shifts every occupied
      eigenvalue by ``-xi_N``, i.e. every correlation denominator, which is
      why the choice matters: on the strict-zero neutral reference the
      correlation carries a distortion that GROWS with cluster size
      (``HANDOVER_D2_EXXDIV.md``: diamond RI-MP2 -59.0 vs KMP2 -29.9/-41.1
      mHa/atom at (2,2,2)).
    * ``"neutral"`` -- the lean strict-zero SCF, the historical default of
      the non-DLPNO one-call wrappers. Kept for continuity and for
      neutral-vs-neutral comparisons; not recommended for absolute
      correlation at multi-cell clusters.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_dlpno_ump2_direct',
    )
    from .dlpno_ump2 import ccm_dlpno_ump2

    scf, L = _direct_reference_and_cderi(
        ccm, reference=reference, ke_cutoff=ke_cutoff, aux_basis=aux_basis,
        cderi_symmetry=cderi_symmetry, exxdiv=exxdiv,
        fold_threads=fold_threads, open_shell=True,
        who="run_ccm_dlpno_ump2_direct", initial_guess=initial_guess)
    return ccm_dlpno_ump2(ccm, scf, cderi=L, ke_cutoff=float(ke_cutoff),
                          **dlpno_kwargs, initial_guess=initial_guess)


def run_ccm_dlpno_ccsd_direct(
    ccm, *, initial_guess: object = "AUTO", reference="direct", ke_cutoff=200.0, aux_basis=None,
    cderi_symmetry=None, exxdiv="ewald", fold_threads=None, **dlpno_kwargs,
):
    """One-call closed-shell DLPNO-CCSD(T) on the direct-torus reference
    (EXPERIMENTAL, 2026-08-22).

    Composes :func:`~vibeqc.periodic.ccm.dlpno_ccsd.ccm_dlpno_ccsd` on the
    same one-cderi/one-reference contract as the MP2 siblings. Triples are
    on by default there (``compute_triples=True``); pass
    ``compute_triples=False`` through ``dlpno_kwargs`` for CCSD only.

    ``reference`` selects the complete SCF route, exactly as on
    ``run_ccm_ri_mp2``:

    * ``"direct"`` (the default here) -- the BvK-ewald direct-torus SCF on
      the shared ``L``. The exchange-q=0 seam shifts every occupied
      eigenvalue by ``-xi_N``, i.e. every correlation denominator, which is
      why the choice matters: on the strict-zero neutral reference the
      correlation carries a distortion that GROWS with cluster size
      (``HANDOVER_D2_EXXDIV.md``: diamond RI-MP2 -59.0 vs KMP2 -29.9/-41.1
      mHa/atom at (2,2,2)).
    * ``"neutral"`` -- the lean strict-zero SCF, the historical default of
      the non-DLPNO one-call wrappers. Kept for continuity and for
      neutral-vs-neutral comparisons; not recommended for absolute
      correlation at multi-cell clusters.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_dlpno_ccsd_direct',
    )
    from .dlpno_ccsd import ccm_dlpno_ccsd

    scf, L = _direct_reference_and_cderi(
        ccm, reference=reference, ke_cutoff=ke_cutoff, aux_basis=aux_basis,
        cderi_symmetry=cderi_symmetry, exxdiv=exxdiv,
        fold_threads=fold_threads, open_shell=False,
        who="run_ccm_dlpno_ccsd_direct", initial_guess=initial_guess)
    return ccm_dlpno_ccsd(ccm, scf, cderi=L, ke_cutoff=float(ke_cutoff),
                          **dlpno_kwargs)


def run_ccm_rhf_direct_rijcosx(
    ccm, *, initial_guess: object = "AUTO", cderi=None, ke_cutoff=200.0, tail_ke_cutoff=None,
    aux_basis=None, exxdiv="ewald",
    omega=None, grid_options=None, lr_ke_cutoff=None, lat_opts=None,
    max_iter=128, conv_tol=1e-9, conv_tol_grad=1e-6, diis_dim=8, lindep_tol=1e-7,
):
    """Real-Γ RIJCOSX control in the **neutral fitted-torus gauge**.

    This combines neutral RI-J, periodic chain-of-spheres exchange on the
    supercell grid, and the derived exchange-``q=0`` seam. It is a same-H
    representation control, not Γ-CCM construction evidence.

    The accelerated sibling of :func:`run_ccm_rhf_direct`: ``J`` stays the
    neutral-cderi contraction, while ``K`` comes from the validated periodic
    COSX engine (:class:`vibeqc.periodic_cosx_k.KPointCosxK`, the M3b
    composed SR+LR exchange) run at Γ on the cluster supercell -- real-space
    erfc-SR seminumerical exchange on the supercell Becke grid plus the
    reciprocal-space erf-LR complement, **in the same G=0-dropped gauge as
    the neutral cderi**, so the identical seam ``K -> K + ξ_N S D S``
    (``exxdiv="ewald"``) applies on top with no re-derivation. Exchange is
    short-ranged in insulators (decays within the WS cell), which is what
    makes the seminumerical SR part the natural fast exchange for the torus.

    Unlike the research route :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_rijcosx`
    (bare-1/r WSSC RI-J + *molecular* COSX on the supercell -- no torus wrap,
    no gauge control; ~few-% periodically), this route is gauge-complete.
    **Measured accuracy vs** :func:`run_ccm_rhf_direct` (2026-07-02): 22-30
    µHa/cell on the genuine-3-D controls (compact H₂ (2,2,2) torus and the
    vacuum-padded dim=3 H₂ anchor), invariant under ω / LR-cutoff / grid-tier
      changes -- i.e. exactly the underlying M3b COSX engine's validated
    composition floor (0.024-0.027 mHa backend parity, M3b-4c). Pushing the
    route below ~10 µHa/cell therefore awaits a floor improvement in the
    shared engine, not more CCM-side tuning.

    Parameters
    ----------
    omega : float, optional
        SR/LR range-separation parameter (bohr⁻¹). Default: the multi-k
        driver's rule at the torus, ``5 / min(half-super-period,
        cutoff_bohr)`` -- the erfc-SR part is dead (erfc(5) ≈ 1.5e-12) at
        both the alias boundary and the cell-list cutoff.
    grid_options : GridOptions, optional
        COSX quadrature tier (default: the molecular RIJCOSX standard tier,
        35 radial / 9x18 angular).
    lr_ke_cutoff : float, optional
        Reciprocal mesh cutoff for the LR complement (defaults to the
        engine's dense-mesh rule).

    Scope (two exclusions, both loud):

    * **Genuine 3-D cells only** (``dim == 3``). For ``dim < 3`` the LR erf
      complement is built on the dimension-collapsed reciprocal mesh (vacuum
      axes carry only ``G_perp = 0``) while the real-space SR half is exact in
      all directions -- the two halves then do not compose to the collapsed-
      convention kernel the neutral cderi represents. Measured on a dim=1
      H-chain torus: K-matrix error 0.08-0.6, *growing* with the SR share
      (2026-07-02). This route raises for ``dim < 3``; no neutral fitted-torus
      control is defined there. The separately derived D3 mixed-boundary
      wire/slab kernel work is required for a low-D neutral control.
    * The real-space SR envelope criterion (shell extents ≲ 25 bohr at the
      1e-4 level; ultra-diffuse bases such as Li/sto-3g 2sp warn and belong
      on :func:`run_ccm_rhf_direct` / the GDF route --
      ``handovers/HANDOVER_RIJCOSX_M3A.md`` § M3b-6).

    Returns a ``CCMSCFResult`` with ``.exchange_q0`` recorded.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rhf_direct_rijcosx',
    )
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_direct_rijcosx"
    )
    _warn_experimental()
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.periodic.exchange_convention import exchange_q0_label
    from vibeqc.periodic_cosx_k import KPointCosxK

    if exxdiv not in ("ewald", None):
        raise ValueError(
            "run_ccm_rhf_direct_rijcosx: exxdiv must be 'ewald' or None, "
            f"got {exxdiv!r}"
        )
    sysc = ccm.cluster_system
    if int(sysc.dim) != 3:
        raise NotImplementedError(
            "run_ccm_rhf_direct_rijcosx: dim < 3 is not supported -- the LR "
            "erf complement lives on the dimension-collapsed reciprocal mesh "
            "(vacuum axes carry only G_perp = 0) and cannot compose with the "
            "exact real-space SR exchange (measured K error 0.08-0.6 on a "
            "dim=1 chain torus, 2026-07-02). No neutral fitted-torus HF or "
            "RIJCOSX control is defined below 3-D. For a dim=1 union-and-weight "
            "mixed-boundary study use run_ccm_rhf_wire within its stated "
            "light-element scope; the slab counterpart remains unavailable."
        )
    base = lat_opts if lat_opts is not None else LatticeSumOptions()
    if omega is None:
        # The multi-k driver's SR/LR split rule (M3b-4), evaluated for the
        # torus: the erfc-SR reach must fit inside both the BvK half-super-
        # period (alias boundary) and the cell-list cutoff.
        a_sc = np.asarray(sysc.lattice, dtype=float)
        half_supers = [
            0.5 * float(np.linalg.norm(a_sc[:, i]))
            for i in range(int(sysc.dim))
        ]
        sr_reach = min(min(half_supers), float(base.cutoff_bohr))
        omega = 5.0 / sr_reach

    L = np.asarray(
        cderi if cderi is not None
        else ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff,
                               tail_ke_cutoff=tail_ke_cutoff,
                               aux_basis=aux_basis),
        dtype=float)
    S, h, e_nn = ccm_direct_oneelectron(ccm, lat_opts=lat_opts)

    bridge = KPointCosxK(
        ccm.basis, sysc, lat_opts=base, omega=float(omega),
        grid_options=grid_options,
    )
    gamma = [np.zeros(3)]

    def k_cosx(D):
        # Γ-only fold: one real-space K(g) build + the LR erf complement in
        # the G=0-dropped gauge (real at Γ).
        K = bridge.k_matrices(
            [D], gamma, lr_complement=True, lr_ke_cutoff=lr_ke_cutoff)[0]
        return np.real(K)

    if exxdiv == "ewald":
        xi = ccm_exchange_q0_madelung(ccm)
        k_of_D = lambda D: k_cosx(D) + xi * (S @ D @ S)
    else:
        k_of_D = k_cosx

    res = _rhf_loop(
        ccm, S, h, e_nn,
        lambda D: ccm_ri_j_neutral(L, D), k_of_D,
        max_iter=max_iter, conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis_dim=diis_dim, lindep_tol=lindep_tol)
    # result-backend identity (IID 344)
    res.backend = "ccm-neutral-direct-rijcosx-rhf"
    res.rsgdf_tail_ke_cutoff = (
        None if cderi is not None
        else ccm_neutral_tail_ke_cutoff(
            ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff))
    res.exchange_q0 = exchange_q0_label(exxdiv)
    return _with_ccm_guess(res, guess_selection)
