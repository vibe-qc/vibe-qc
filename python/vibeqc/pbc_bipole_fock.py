"""Shared BIPOLE two-electron Fock builders (M2 of the driver unification).

The four BIPOLE periodic-SCF drivers (``pbc_bipole`` RHF, ``pbc_bipole_rks``,
``pbc_bipole_uhf``, ``pbc_bipole_uks``) all assemble the same CRYSTAL-gauge
two-electron Fock operator: a direct erfc-screened short-range ``J_SR`` and
exchange ``K``, an analytic reciprocal-space long-range ``J_LR`` with a
neutralising background, and -- under the Ewald exchange split -- the k-space
``K_LR`` + G=0/Madelung correction. This module hosts that assembly once.

Design:

* :func:`build_bipole_restricted_fock` serves RHF (``alpha_hf=1.0``) and RKS
  (``alpha_hf`` = the functional's HF-exchange fraction). It reproduces the
  RHF driver's arithmetic bit-for-bit at ``alpha_hf=1.0``; the DFT variants
  differ only by the (powers-of-two-exact) ``alpha_hf`` scaling and the
  f2e block-assembly association, which is < 1e-12 on the SCF energy.
* XC is intentionally NOT built here. The DFT drivers add ``V_xc`` to F(k)
  in their own per-k assembly so RHF/UHF stay exact references for the
  shared HF pieces.
* The per-k Fock assembly (Bloch sum of ``f2e_real`` + Hcore + optional
  ``V_xc`` + ``K_corr``, then Hermitisation) stays in each driver: the
  builder returns the real-space ``f2e_real`` and the k-space ``K_corr``
  pieces, plus the component energies.

The gauge invariants the builder must preserve are documented inline at the
RHF driver: one shared Ewald (alpha, K_max) envelope for
``V_ne``/``E_nn``/``J_LR``; Gamma-only density locality applied by the driver
before each rebuild; the energy-component closure
``e_2e = e_j_short_range + e_j_long_range + e_exchange``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    LatticeMatrixSet,
    build_fock_2e_real_space,
    build_jk_2e_real_space,
    make_lattice_matrix_set,
)
from .pbc_bipole_common import (
    _bloch_sum_blocks,
    _bloch_sum_blocks_multi_k,
    _cell_key,
    _lattice_contract,
    _lattice_contract_blocks,
    _copy_lattice_with_blocks,
    bvk_torus_density_matrices,
    home_cell_block,
)


@dataclass
class BipoleFockContext:
    """Per-run invariants shared by every Fock build in one BIPOLE SCF.

    Built once before the SCF loop and passed to the builders on each
    iteration. Bundles exactly the enclosing-scope state the driver's
    former ``_build_fock_for_density`` closure captured, so the body can
    live at module scope unchanged.
    """

    basis: Any
    system: Any
    lat_opts_2e: Any
    use_ewald_j_split: bool
    exchange_split_active: bool
    n_k: int
    omega_used: Optional[float]
    ewald_precision: float
    ewald_cell_volume: Optional[float]
    n_elec: int
    xi_madelung: float
    j_lr_cache: Any
    x_lr_cache: Any
    incremental_jk: Any
    rep_cell_indices: Any
    fock_sym_map: Any
    mp_config: Any
    s_lat: LatticeMatrixSet
    s_k_list: List[np.ndarray]
    k_points: List[Any]
    weights: np.ndarray
    k_points_full: List[Any]
    weights_full: np.ndarray
    ir_mapping: np.ndarray
    bvk_mesh: Any
    n_occ: int
    plog: Any
    penetration_dispatch_j: Any = None  # QuartetBipolarDispatch, built by the driver before SCF
    penetration_dispatch_k: Any = None
    spherical_moment_buffer: Any = None  # SphericalMomentBuffer, built once per geometry
    quartet_tensor_cache: Any = None  # QuartetTensorCache, pre-computed interaction tensors
    far_field_fock_kernel: Any = None  # FarFieldFockKernel, pre-computed Fock kernel
    apply_bipolar_far_field_to_fock: bool = False  # research: replace exact with far-field
    exact_j_for_pure_rks: bool = False
    exact_j_ke_cutoff: float = 200.0
    exact_j_chunk_size: int = 512
    sr_image_extent: Optional[float] = None
    exact_zone_bohr: Optional[float] = None
    symmetry_reconstruction_map: Any = None  # SymmetryFockReconstructionMap for far-field
    output_cell_farming_task_kind: Optional[str] = None
    output_cell_farming_strategy: str = "cyclic"
    resolved_domain: Any = None
    sr_workspace: Any = None


def _sr_workspace_for(ctx):
    """Own geometry-only native preparation for the lifetime of this Fock context."""
    if getattr(ctx, "sr_workspace", None) is None:
        from . import _vibeqc_core

        factory = getattr(_vibeqc_core, "PeriodicJKWorkspace", None)
        if factory is not None:
            ctx.sr_workspace = factory()
    return getattr(ctx, "sr_workspace", None)


@dataclass(frozen=True)
class BipoleOutputCellFarmingExecution:
    """Executed direct-ERI output-cell farming contract.

    One task owns one complete caller-declared real-space AO output task,
    possibly restricted by a symmetry shell mask, while retaining the full
    internal ``(c_lambda, c_sigma)`` translation sum. This is an execution
    label, not a claim that the radial output cells are Gamma-CCM WSCs or
    chi-CCM finite-group residue classes.
    """

    task_kind: str
    strategy: str
    world_size: int
    rank: int
    global_task_count: int
    local_task_count: int
    local_task_counts: Tuple[int, ...]
    ordered_cell_fingerprint: str
    schema: str = field(
        init=False,
        default="vibeqc.pbc-bipole.output-cell-farming-execution/v1",
    )
    complete_internal_translation_sum: bool = field(init=False, default=True)
    result_distribution: str = field(
        init=False,
        default="allgather-complete-blocks",
    )

    @property
    def active(self) -> bool:
        """Whether more than one MPI rank executed the declared schedule."""
        return self.world_size > 1


@dataclass(frozen=True)
class BipoleScreenedExchangeExecution:
    """Attestation emitted only after the direct screened-K fold executes."""

    c_sr: float
    omega_screen_bohr_inv: float
    schema: str = field(
        init=False,
        default="vibeqc.pbc-bipole.screened-exchange-execution/v1",
    )
    assembly: str = field(init=False, default="short-range-direct")

    def __post_init__(self) -> None:
        if isinstance(self.c_sr, bool) or isinstance(
            self.omega_screen_bohr_inv,
            bool,
        ):
            raise ValueError(
                "BipoleScreenedExchangeExecution values must be finite "
                "and positive"
            )
        try:
            c_sr = float(self.c_sr)
            omega = float(self.omega_screen_bohr_inv)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "BipoleScreenedExchangeExecution values must be finite "
                "and positive"
            ) from exc
        if not np.isfinite(c_sr) or c_sr <= 0.0:
            raise ValueError(
                "BipoleScreenedExchangeExecution.c_sr must be finite "
                "and positive"
            )
        if not np.isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "BipoleScreenedExchangeExecution.omega_screen_bohr_inv "
                "must be finite and positive"
            )
        object.__setattr__(self, "c_sr", c_sr)
        object.__setattr__(self, "omega_screen_bohr_inv", omega)


@dataclass(frozen=True)
class BipoleChiScreenedExchangeExecution(BipoleScreenedExchangeExecution):
    """Executed signed erfc arm, available only on the explicit chi route.

    CAM/LRC exchange uses a negative erfc coefficient in the canonical
    full-plus-erfc representation. This attestation deliberately leaves the
    legacy HSE-only execution contract unchanged.
    """

    schema: str = field(
        init=False, default="vibeqc.chi.screened-exchange-execution/v1",
    )

    def __post_init__(self) -> None:
        if isinstance(self.c_sr, bool) or isinstance(self.omega_screen_bohr_inv, bool):
            raise ValueError("chi screened exchange requires finite nonzero c_sr and positive omega")
        try:
            c_sr = float(self.c_sr)
            omega = float(self.omega_screen_bohr_inv)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("chi screened exchange requires finite nonzero c_sr and positive omega") from exc
        if not np.isfinite(c_sr) or c_sr == 0.0 or not np.isfinite(omega) or omega <= 0.0:
            raise ValueError("chi screened exchange requires finite nonzero c_sr and positive omega")
        object.__setattr__(self, "c_sr", c_sr)
        object.__setattr__(self, "omega_screen_bohr_inv", omega)


@dataclass
class BipoleRestrictedFockBuild:
    """Restricted (closed-shell) BIPOLE two-electron Fock build.

    Carries the real-space ``f2e_real`` and the k-space exchange-split
    pieces (``k_corr_per_k``) the driver folds into F(k), plus the
    component energies. ``k_corr_per_k`` is ``None`` outside the Ewald
    exchange split (and for pure functionals).
    """

    f2e_real: LatticeMatrixSet
    k_corr_per_k: Optional[List[np.ndarray]]
    e_j_short_range: Optional[float] = None
    e_j_long_range: Optional[float] = None
    e_exchange: Optional[float] = None
    e_j_multipole: Optional[float] = None
    e_2e_k_correction: float = 0.0
    # The probe-charge-Madelung (exxdiv='ewald') share of the exchange
    # energy on its own -- exactly what exchange_exxdiv='none' removes, so
    # exactly what a code treating the q -> 0 singularity differently will
    # disagree by at the same mesh. None when no corrected exchange split
    # ran; exactly 0.0 under exchange_exxdiv='none'. See #82.
    e_exchange_finite_size: Optional[float] = None
    # Dormant PDR 1988 Ch. II.4c quartet-prototype diagnostic.
    e_j_bipolar_quartet: Optional[float] = None
    screened_exchange_execution: Optional[
        BipoleScreenedExchangeExecution
    ] = None
    output_cell_farming_execution: Optional[
        BipoleOutputCellFarmingExecution
    ] = None

    # Executed physical-kernel energies; erfc includes its finite zero mode.
    e_hf_exchange_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class BipoleUnrestrictedFockBuild:
    """Unrestricted (open-shell) BIPOLE two-electron Fock build.

    ``f_alpha_k_list``/``f_beta_k_list`` are assembled (F^2e + Hcore −
    K_corr, hermitised) only when the caller passes ``hcore_k_list``;
    a driver that folds its own per-k terms in between (UKS adds the
    per-spin V_xc before hermitising) passes ``hcore_k_list=None`` and
    assembles from ``f2e_*_real`` + ``k_corr_*_per_k`` itself.
    """

    f2e_alpha_real: LatticeMatrixSet
    f2e_beta_real: LatticeMatrixSet
    f_alpha_k_list: List[np.ndarray]
    f_beta_k_list: List[np.ndarray]
    e_j_short_range: Optional[float] = None
    e_j_long_range: Optional[float] = None
    e_exchange: Optional[float] = None
    e_j_multipole: Optional[float] = None
    e_2e_k_correction: float = 0.0
    # The probe-charge-Madelung (exxdiv='ewald') share of the exchange
    # energy on its own -- exactly what exchange_exxdiv='none' removes, so
    # exactly what a code treating the q -> 0 singularity differently will
    # disagree by at the same mesh. None when no corrected exchange split
    # ran; exactly 0.0 under exchange_exxdiv='none'. See #82.
    e_exchange_finite_size: Optional[float] = None
    k_corr_alpha_per_k: Optional[List[np.ndarray]] = None
    k_corr_beta_per_k: Optional[List[np.ndarray]] = None
    screened_exchange_execution: Optional[
        BipoleScreenedExchangeExecution
    ] = None

    # Executed physical-kernel energies; erfc includes its finite zero mode.
    e_hf_exchange_components: Dict[str, float] = field(default_factory=dict)


def split_k_density_list(
    ctx: BipoleFockContext, density: LatticeMatrixSet, *, retained_density_k=None,
) -> List[np.ndarray]:
    """Per-k density matrices for the Ewald-exchange-split paths.

    Exact for every density representation the SCF loop produces (orbital
    rebuilds, SAD/PATOM local guesses, caller warm-starts, damped and
    ODA-mixed densities): at Γ the BvK representative is the home-cell
    block; at multi-k the BvK-torus fold inverts the Bloch transform
    exactly (see ``bvk_torus_density_matrices``).
    """
    if retained_density_k is not None:
        # The chi route retains the physical density through damping and
        # terminal rebuilds. Recovering it from a selected real-space support
        # can alias finite-group characters when that support omits residues.
        matrices = np.asarray(retained_density_k, dtype=complex)
        expected = (ctx.n_k, int(ctx.basis.nbasis), int(ctx.basis.nbasis))
        if matrices.shape != expected or not np.all(np.isfinite(matrices)):
            raise ValueError("chi exchange requires finite per-k density matrices")
        if not np.allclose(matrices, matrices.conj().transpose(0, 2, 1), atol=1e-10, rtol=0.0):
            raise ValueError("chi exchange requires Hermitian per-k density matrices")
        fractional = (np.asarray(ctx.k_points) @ np.asarray(ctx.system.lattice)
                      / (2.0 * np.pi))
        weights = np.asarray(ctx.weights)
        for i, point in enumerate(fractional):
            sums = fractional + point
            partners = np.flatnonzero(
                np.max(np.abs(sums - np.rint(sums)), axis=1) < 1e-8
            )
            if (len(partners) != 1
                    or not np.isclose(weights[i], weights[partners[0]], atol=1e-12, rtol=0.0)
                    or not np.allclose(matrices[i], matrices[partners[0]].conj(),
                                       atol=1e-8, rtol=1e-10)):
                raise NotImplementedError(
                    "direct chi exchange requires a complete "
                    "time-reversal-compatible Bloch density"
                )
        return [matrix for matrix in matrices]
    if ctx.n_k == 1:
        return [home_cell_block(density).astype(complex)]
    assert ctx.bvk_mesh is not None
    return bvk_torus_density_matrices(density, ctx.k_points, ctx.bvk_mesh)


def restricted_density_from_coeffs(
    coeffs_per_k: Sequence[np.ndarray], n_occ: int
) -> List[np.ndarray]:
    """Closed-shell per-k density matrices ``2.C_occ C_occ+`` from orbitals."""
    out: List[np.ndarray] = []
    for C_k_raw in coeffs_per_k:
        C_k_arr = np.asarray(C_k_raw)
        C_occ_k = C_k_arr[:, :n_occ]
        out.append(2.0 * (C_occ_k @ C_occ_k.conj().T))
    return out


def _pair_resolved_density_for_sr(
    basis: Any,
    mapping: Any,
    density: LatticeMatrixSet,
) -> LatticeMatrixSet:
    """Return the pair-masked density used only by direct SR J/K.

    The SCF density is the physical Bloch density and must remain unmasked
    for one-electron, XC, reciprocal-J, and energy contractions.  SYM3b's
    pair masks define only the finite direct-SR tensor domain, so apply them
    to a private copy at that builder boundary.
    """
    from .pair_resolved_truncation import mask_lattice_blocks_to_domain

    masked = make_lattice_matrix_set(
        int(basis.nbasis),
        list(density.cells),
        [np.asarray(block, dtype=float).copy() for block in density.blocks],
    )
    mask_lattice_blocks_to_domain(basis, mapping.density_domain, masked)
    return masked


def unrestricted_density_from_coeffs(
    coeffs_per_k: Sequence[np.ndarray],
    n_occ: int,
    occ_per_k: Optional[Sequence[np.ndarray]] = None,
) -> List[np.ndarray]:
    """Open-shell per-k density matrices from orbitals or occupations."""
    out: List[np.ndarray] = []
    for idx, C_k_raw in enumerate(coeffs_per_k):
        C_k = np.asarray(C_k_raw)
        if occ_per_k is None:
            C_occ = C_k[:, :n_occ] if n_occ > 0 else C_k[:, :0]
            out.append(C_occ @ C_occ.conj().T)
        else:
            occ = np.asarray(occ_per_k[idx], dtype=float)
            out.append((C_k * occ[None, :]) @ C_k.conj().T)
    return out


def _can_pair_sr(ctx) -> bool:
    from . import _vibeqc_core as core

    return (hasattr(core, "build_jk_2e_real_space_domains_pair")
            and getattr(ctx, "rep_cell_indices", None) is None
            and not getattr(getattr(ctx, "fock_sym_map", None), "pair_resolved", False)
            and getattr(ctx, "output_cell_farming_task_kind", None) is None)


def _screened_exchange_lattice(
    ctx: BipoleFockContext,
    density: LatticeMatrixSet,
    omega_screen: float,
    *,
    _chi_exchange: bool = False,
    companion_density=None,
):
    """Direct real-space erfc(omega_screen*r)/r exchange lattice matrix.

    The screened-hybrid K (HSE-type). The output blocks stay on the
    ``lat_opts_2e`` direct cell list, while the internal ket-image sum uses
    the same resolved padded domain as the erfc J build. Keeping those
    domains aligned is required because a translated ket pair can remain
    inside the smeared erfc range even when its output block is inside the
    smaller electronic cutoff. Under physical pair support
    (``pair_complete_1e``, the ab initio default since #316) that padded
    radius is the pair-interaction reach, which bounds the exchange
    pair-pair distance as well: a Gamma-only (Born-von-Karman-periodic)
    density then gains its image-shell screened exchange out to the
    resolved precision, where the historical index ball kept K on the
    electronic-cutoff cell list. The multi-k Ewald drivers sum the same
    K_erfc only to ``cutoff_bohr`` (#501).
    Deliberately NOT routed through the multipole/bipolar far-field
    (CRYSTAL disables the bipolar expansion for RSH exchange) and
    carries NO K_LR / Madelung / exxdiv seam: the erfc kernel's G -> 0
    limit is the finite pi/omega_screen^2, which the absolutely-
    convergent direct sum already contains correctly.
    """
    from . import _vibeqc_core as core

    # Older native runtimes retain the combined J/K traversal. Selection is
    # an optimization only; never infer support from a caught build error.
    k_only = bool(getattr(core, "periodic_jk_coulomb_selection", False))
    selection = {"compute_coulomb": False} if k_only else {}
    if companion_density is not None:
        if not _can_pair_sr(ctx):
            raise ValueError("paired screened exchange requires the unreduced serial route")
        selection["companion_density"] = companion_density
    if (_chi_exchange and ctx.fock_sym_map is not None
            and getattr(ctx.fock_sym_map, "pair_resolved", False)):
        density = _pair_resolved_density_for_sr(ctx.basis, ctx.fock_sym_map, density)
    if ctx.rep_cell_indices is not None:
        from .bipole_symmetry_fock import build_jk_reduced_symmetrized

        jk = build_jk_reduced_symmetrized(
            ctx.basis,
            ctx.system,
            ctx.lat_opts_2e,
            density,
            float(omega_screen),
            ctx.fock_sym_map,
            ctx.rep_cell_indices,
            internal_extent_bohr=ctx.sr_image_extent,
            output_cell_farming_task_kind=ctx.output_cell_farming_task_kind,
            output_cell_farming_strategy=ctx.output_cell_farming_strategy,
            workspace=_sr_workspace_for(ctx),
            **selection,
        )
    elif ctx.sr_image_extent is not None:
        if ctx.exact_zone_bohr is not None:
            raise NotImplementedError(
                "exact_zone_bohr is not wired for the dedicated "
                "screened-exchange (HSE-type) K traversal; omit the zone "
                "restriction for screened hybrids"
            )
        jk = _sr_image_padded_jk(
            ctx.basis,
            ctx.system,
            ctx.lat_opts_2e,
            density,
            float(omega_screen),
            float(ctx.sr_image_extent),
            output_cell_farming_task_kind=ctx.output_cell_farming_task_kind,
            output_cell_farming_strategy=ctx.output_cell_farming_strategy,
            workspace=_sr_workspace_for(ctx),
            **selection,
        )
    elif k_only or companion_density is not None:
        cells = _sr_internal_cells(
            ctx.basis, ctx.system, ctx.lat_opts_2e,
            None if ctx.lat_opts_2e.pair_complete_1e else ctx.lat_opts_2e.cutoff_bohr,
        )
        jk = _build_jk_domains_output_cells_mpi(
            ctx.basis, ctx.system, ctx.lat_opts_2e, density, cells,
            list(range(len(cells))), omega=float(omega_screen),
            workspace=_sr_workspace_for(ctx), compute_coulomb=False,
            **({"companion_density": companion_density} if companion_density is not None else {}),
        )
    else:
        jk = build_jk_2e_real_space(
            ctx.basis, ctx.system, ctx.lat_opts_2e, density, float(omega_screen)
        )
    return jk.K if companion_density is None else tuple(part.K for part in jk)


def _screened_k_blocks_for_cells(K_lat, cells, c_sr: float):
    """Per-cell ``c_sr * K_erfc`` blocks aligned to ``cells`` by cell key.

    The J-only base build and the dedicated K traversal both run on the
    ``lat_opts_2e`` operator cell list, but alignment by index is not
    guaranteed across builders -- match by cell key and fail loudly on a
    genuine mismatch.
    """
    k_by_key = {
        _cell_key(cell): np.asarray(block, dtype=float)
        for cell, block in zip(K_lat.cells, K_lat.blocks)
    }
    blocks = []
    for cell in cells:
        key = _cell_key(cell)
        if key not in k_by_key:
            raise RuntimeError(
                "screened exchange: the J-only Fock build and the "
                f"K_erfc traversal disagree on the cell list (cell {key} "
                "has no K block). This is an internal invariant "
                "violation, not a user error."
            )
        blocks.append(float(c_sr) * k_by_key[key])
    return blocks


def _add_screened_exchange_restricted(
    ctx: BipoleFockContext,
    density: LatticeMatrixSet,
    base: BipoleRestrictedFockBuild,
    exx,
    *,
    _chi_exchange: bool = False,
) -> BipoleRestrictedFockBuild:
    """Add ``-c_sr/2 K_erfc`` to the base J plus full-exchange build."""
    screened_kwargs = {"_chi_exchange": True} if _chi_exchange else {}
    K_lat = _screened_exchange_lattice(ctx, density, exx.omega_screen, **screened_kwargs)
    f2e = base.f2e_real
    n_cells = len(f2e.cells)
    K_blocks = _screened_k_blocks_for_cells(K_lat, f2e.cells, exx.c_sr)
    # Native .blocks converts the complete vector on every access.
    source_blocks = f2e.blocks
    for c in range(n_cells):
        f2e.set_block(
            c, np.asarray(source_blocks[c], dtype=float) - 0.5 * K_blocks[c]
        )
    # Re-apply the SYM3b Fock symmetrisation after the K fold (the
    # base build's blocks were symmetrised before we touched them).
    if ctx.fock_sym_map is not None:
        from .bipole_symmetry_fock import symmetrize_fock_blocks

        f2e_blocks = [
            np.asarray(block, dtype=float) for block in f2e.blocks
        ]
        symmetrize_fock_blocks(
            f2e_blocks, ctx.fock_sym_map, cells=list(f2e.cells)
        )
        for c in range(n_cells):
            f2e.set_block(c, f2e_blocks[c])
    erfc_energy = -0.25 * _lattice_contract_blocks(
        density,
        f2e.cells,
        K_blocks,
        operator_name="K_erfc",
    )
    # The base is the executed full-only arm. Record its physical zero
    # term before adding the independently contracted erfc contribution.
    base.e_hf_exchange_components = {
        "full_nonzero": float(base.e_exchange or 0.0) - float(base.e_exchange_finite_size or 0.0),
        "full_zero": float(base.e_exchange_finite_size or 0.0),
        "erfc": float(erfc_energy),
    }
    base.e_exchange = (sum(base.e_hf_exchange_components.values()) if _chi_exchange
                       else float(base.e_exchange or 0.0) + erfc_energy)
    execution_type = BipoleChiScreenedExchangeExecution if _chi_exchange else BipoleScreenedExchangeExecution
    base.screened_exchange_execution = execution_type(
        c_sr=exx.c_sr,
        omega_screen_bohr_inv=exx.omega_screen,
    )
    return base


def _sr_padded_lat_opts(lat_opts_2e, image_extent: float):
    """Preserve physical AO-pair support and set the internal interaction reach."""
    from .pbc_bipole_common import _lattice_options_passthrough

    lp = _lattice_options_passthrough(lat_opts_2e)
    physical_pairs = getattr(lat_opts_2e, "pair_complete_1e", False)
    lp.cutoff_bohr = lat_opts_2e.cutoff_bohr if physical_pairs else float(image_extent)
    if physical_pairs:
        lp.eri_interaction_cutoff_bohr = float(image_extent)
    lp.coulomb_method = lat_opts_2e.coulomb_method
    return lp


def _sr_density_cells(basis, system, lattice_opts, image_extent=None):
    """Density image support for the selected real-space operator."""
    from ._vibeqc_core import direct_lattice_cells

    if lattice_opts.pair_complete_1e:
        return _sr_internal_cells(basis, system, lattice_opts, image_extent)
    return list(direct_lattice_cells(system, 2.0 * float(lattice_opts.cutoff_bohr)))


def _sr_internal_cells(basis, system, lattice_opts, image_extent):
    """Enclose physical product pairs while preserving the historical ball."""
    from ._vibeqc_core import direct_lattice_cells

    if getattr(lattice_opts, "pair_complete_1e", False):
        from .lattice_screening import physical_eri_cells

        return physical_eri_cells(basis, system, lattice_opts, image_extent)
    extent = float(image_extent if image_extent is not None else (
        lattice_opts.eri_interaction_cutoff_bohr or lattice_opts.cutoff_bohr
    ))
    return list(direct_lattice_cells(system, extent))


def _density_on_cache_cells(density, cache, system):
    """The density on the Ewald long-range J cache's own cell list (#503).

    Without a precomputed ``rho_hat`` the long-range J sums the per-cell
    density over the cache's cells and refuses a different list. The KS
    drivers widen their density list by the exact XC difference cells
    (#434, 60de9754) while the cache stays on the operator list, so on the
    legacy gauge every first-cycle, damped or mixed density was refused.
    Hand the J kernel exactly the operator cells, which is the historical
    operator before that widening. Returns ``density`` itself when the two
    lists already coincide.
    """
    from ._vibeqc_core import make_lattice_matrix_set as _make_lms

    indices = np.rint(np.linalg.solve(
        np.asarray(system.lattice, dtype=float),
        np.asarray(cache.cells_r_cart, dtype=float).T,
    )).astype(int).T
    keys = [tuple(int(v) for v in index) for index in indices]
    density_keys = [_cell_key(cell) for cell in density.cells]
    if density_keys == keys:
        return density
    by_key = dict(zip(density_keys, zip(density.cells, density.blocks)))
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(
            "long-range J: the density list lacks cache cell(s) "
            f"{missing[:4]}; it must cover the operator cell list"
        )
    return _make_lms(
        int(density.nbf),
        [by_key[key][0] for key in keys],
        [np.asarray(by_key[key][1], dtype=float) for key in keys],
    )


def _reciprocal_blocks_on_output(blocks, cache, system, output_cells):
    """Align reciprocal pair blocks with an exchange output enclosure."""
    indices = np.rint(np.linalg.solve(
        np.asarray(system.lattice, dtype=float),
        np.asarray(cache.cells_r_cart, dtype=float).T,
    )).astype(int).T
    by_cell = {tuple(index): np.asarray(block, dtype=float)
               for index, block in zip(indices, blocks)}
    zero = np.zeros(cache.ft_per_cell.shape[1:3], dtype=float)
    return [by_cell.get(_cell_key(cell), zero).copy() for cell in output_cells]


def _build_jk_domains_output_cells_mpi(
    basis,
    system,
    opts,
    density: LatticeMatrixSet,
    cells,
    output_indices: Sequence[int],
    output_shell_masks: Sequence = (),
    *,
    omega: float,
    compute_exchange: bool = True,
    compute_coulomb: bool = True,
    task_kind: Optional[str] = None,
    strategy: str = "cyclic",
    workspace=None,
    companion_density=None,
):
    """Build complete direct-ERI output blocks, optionally farmed by MPI.

    ``output_indices`` is an explicit logical worklist.  Each owned output
    block is evaluated by the native domains kernel with the unchanged full
    ``cells`` list as its internal translation domain, then complete blocks
    are allgathered in the caller's declared order.  There is no floating
    reduction and no electrostatic distance cutoff introduced here.

    The native domains API historically interprets an empty output list as
    "all cells".  This wrapper deliberately gives an empty logical worklist
    its ordinary meaning and, critically, never enters the native kernel on
    an oversubscribed rank with no work.

    ``task_kind=None`` preserves the serial native call exactly.  A non-empty
    task kind opts into the construction-neutral scheduler; chi-CCM uses the
    precise execution label ``"chi-direct-output-cell"``.  These radial AO
    output blocks must not be called Gamma-CCM WSCs or individual tied
    Wigner representatives.
    """
    from types import SimpleNamespace

    from ._vibeqc_core import build_jk_2e_real_space_domains

    if workspace is not None or not compute_coulomb:
        from functools import partial

        native_options = {"workspace": workspace} if workspace is not None else {}
        if not compute_coulomb:
            native_options["compute_coulomb"] = False
        build_jk_2e_real_space_domains = partial(
            build_jk_2e_real_space_domains, **native_options,
        )

    cells = list(cells)
    requested = tuple(int(index) for index in output_indices)
    if any(index < 0 or index >= len(cells) for index in requested):
        raise ValueError(
            "_build_jk_domains_output_cells_mpi: output index outside "
            f"the {len(cells)}-cell internal domain"
        )
    if len(set(requested)) != len(requested):
        raise ValueError(
            "_build_jk_domains_output_cells_mpi: output indices must be "
            "unique"
        )
    masks = list(output_shell_masks)
    if masks and len(masks) != len(requested):
        raise ValueError(
            "_build_jk_domains_output_cells_mpi: output_shell_masks must "
            "be empty or parallel to output_indices"
        )

    if companion_density is not None:
        if task_kind is not None:
            raise ValueError("paired density traversal requires serial output-cell ownership")
        from ._vibeqc_core import build_jk_2e_real_space_domains_pair

        if requested:
            return build_jk_2e_real_space_domains_pair(
                basis, system, opts, density, companion_density, cells,
                list(requested), masks, float(omega), bool(compute_exchange),
                workspace=workspace, compute_coulomb=compute_coulomb,
            )
        def empty_result():
            zeros = [np.zeros((int(basis.nbasis), int(basis.nbasis))) for _ in cells]
            return SimpleNamespace(
                J=make_lattice_matrix_set(int(basis.nbasis), cells, zeros),
                K=make_lattice_matrix_set(int(basis.nbasis), cells, zeros) if compute_exchange else None,
            )
        return empty_result(), empty_result()

    if task_kind is None and requested:
        return build_jk_2e_real_space_domains(
            basis,
            system,
            opts,
            density,
            cells,
            list(requested),
            masks,
            float(omega),
            bool(compute_exchange),
        )
    if task_kind is not None and (
        not isinstance(task_kind, str) or not task_kind.strip()
    ):
        raise ValueError(
            "_build_jk_domains_output_cells_mpi: task_kind must be a "
            "non-empty string or None"
        )

    from .mpi import LatticeOutputPartition

    keys = [_cell_key(cells[index]) for index in requested]
    partition = LatticeOutputPartition.create(keys, strategy=strategy)
    execution = (
        None
        if task_kind is None
        else BipoleOutputCellFarmingExecution(
            task_kind=task_kind,
            strategy=partition.strategy,
            world_size=partition.size,
            rank=partition.rank,
            global_task_count=partition.n_tasks,
            local_task_count=partition.n_local,
            local_task_counts=partition.task_counts,
            ordered_cell_fingerprint=partition.task_fingerprint,
        )
    )
    if partition.size == 1 and requested:
        jk = build_jk_2e_real_space_domains(
            basis,
            system,
            opts,
            density,
            cells,
            list(requested),
            masks,
            float(omega),
            bool(compute_exchange),
        )
        return SimpleNamespace(
            J=jk.J,
            K=(jk.K if compute_exchange else None),
            output_cell_farming_execution=execution,
        )

    local_values = []
    if partition.n_local:
        local_output_indices = [
            requested[pos] for pos in partition.local_indices
        ]
        local_masks = (
            [masks[pos] for pos in partition.local_indices] if masks else []
        )
        local_jk = build_jk_2e_real_space_domains(
            basis,
            system,
            opts,
            density,
            cells,
            local_output_indices,
            local_masks,
            float(omega),
            bool(compute_exchange),
        )
        # Export each native vector once, not once per owned output cell.
        local_j_blocks = local_jk.J.blocks
        local_k_blocks = local_jk.K.blocks if compute_exchange else None
        for output_index in local_output_indices:
            j_block = np.array(
                local_j_blocks[output_index], dtype=float, copy=True
            )
            k_block = None
            if compute_exchange:
                k_block = np.array(
                    local_k_blocks[output_index], dtype=float, copy=True
                )
            local_values.append((j_block, k_block))

    gathered = partition.allgather_ordered(local_values)
    nbf = int(basis.nbasis)
    j_blocks = [np.zeros((nbf, nbf), dtype=float) for _ in cells]
    k_blocks = (
        [np.zeros((nbf, nbf), dtype=float) for _ in cells]
        if compute_exchange
        else None
    )
    for output_index, (j_block, k_block) in zip(requested, gathered):
        j_blocks[output_index] = j_block
        if k_blocks is not None:
            if k_block is None:
                raise RuntimeError(
                    "_build_jk_domains_output_cells_mpi: exchange task "
                    "returned no K block"
                )
            k_blocks[output_index] = k_block
    return SimpleNamespace(
        J=make_lattice_matrix_set(nbf, cells, j_blocks),
        K=(
            None
            if k_blocks is None
            else make_lattice_matrix_set(nbf, cells, k_blocks)
        ),
        output_cell_farming_execution=execution,
    )


def _sr_image_padded_jk(
    basis,
    system,
    lat_opts_2e,
    density: LatticeMatrixSet,
    omega: float,
    image_extent: float,
    *,
    compute_exchange: bool = True,
    compute_coulomb: bool = True,
    exact_zone_bohr: Optional[float] = None,
    output_cell_farming_task_kind: Optional[str] = None,
    output_cell_farming_strategy: str = "cyclic",
    workspace=None,
    companion_density=None,
):
    """SR erfc J/K with the ket-image sum extended past the cutoff (M4a).

    The 2026-07-12 split-gap triage (the original investigation
    Sec. 0a) proved the (c_lam, c_sig) ket-image sum in
    ``build_jk_2e_real_space_impl`` must reach the *smeared* erfc kernel
    range (``erf(sqrt(m) r) - erf(sqrt(m_w) r)`` decay,
    ``1/m_w = 1/g_bra + 1/g_ket + 1/w^2``), not just the output cutoff:
    at an 8-bohr ball MgO/STO-3G loses -515 mHa of J_SR
    (home J[Mg3s,Mg3s] 1.6055 vs the Poisson-anchored 1.8529059).

    Implementation (M4b: the M1 full-domain binding): build on the
    *padded* cell list with ``output_indices`` = the caller's cutoff
    cells (matched by cell key -- sort tie order is not guaranteed
    identical between the two lists), then rebuild J/K on the caller's
    original template so every downstream consumer (Bloch sums, energy
    contractions, incremental deltas) sees the unchanged cell set. The
    density is passed as-is: the dominant recovered mass is a ket PAIR
    translated far out (small ``h = c_sig - c_lam``, which the existing
    density support covers); images whose ``h`` falls outside the
    support are skipped by the builder exactly as before.
    ``compute_exchange=False`` skips the K contraction entirely
    (pure-functional padded J; ``K`` comes back ``None`` -- pre-M4b
    this path computed and discarded a full padded K).
    ``compute_coulomb=False`` skips Coulomb work for dedicated screened
    exchange, returning zero J blocks on the unchanged output template. The padded
    traversal honors ``lat_opts_2e.sr_range_screening`` (charge-pair
    Schwarz screening with angular and contraction envelopes). The native
    flag is OFF by default; the production precision policy enables it.
    The explicit M4a oracle can therefore retain an unscreened comparison.

    ``exact_zone_bohr`` (BIPOLE-EXACT-ZONE increment 1) restricts the
    *output* zone of the exact erfc traversal to cells with
    ``|r_cell| <= exact_zone_bohr`` while every downstream consumer
    keeps the full cutoff cell template: far output cells come back as
    exact zeros, so their F^2e content is the reciprocal J^LR channel
    plus the neutralising background alone. This is the CRYSTAL
    bielectronic-zone / monoelectronic-zone split expressed in the
    Ewald-split gauge -- the exact bielectronic zone is bounded
    independently of the (cheap, one-electron) fold-convergence range,
    and the far zone is carried by the Ewald model of the density:
    Dovesi, Pisani, Roetti & Saunders, Phys. Rev. B 28, 5781 (1983),
    Eq. (24a) rearrangement (exact zone + infinite multipolar/Ewald
    series); Pisani, Dovesi & Roetti, Lecture Notes in Chemistry 48
    (1988), Sec. II.4b/II.4d (zone partition, model far field). The
    omitted SR far-tail is second-order in the AO pair-overlap tail
    (bra pair must exist at the far cell AND couple through the
    screened kernel), unlike the S(k)-fold error which is first-order
    -- measured on LiH/STO-3G Gamma (the fold-range worst case):
    see tests/test_bipole_exact_zone.py.
    """
    from types import SimpleNamespace

    from ._vibeqc_core import direct_lattice_cells

    if float(image_extent) <= float(lat_opts_2e.cutoff_bohr):
        raise ValueError(
            "sr_image_extent must exceed the electronic cutoff_bohr "
            f"({image_extent!r} vs {lat_opts_2e.cutoff_bohr!r})"
        )
    if exact_zone_bohr is not None:
        if not float(exact_zone_bohr) < float(lat_opts_2e.cutoff_bohr):
            raise ValueError(
                "exact_zone_bohr must be strictly below the electronic "
                f"cutoff_bohr ({exact_zone_bohr!r} vs "
                f"{lat_opts_2e.cutoff_bohr!r}); at or above the cutoff "
                "the restriction is a no-op -- omit it instead"
            )
    lat_pad = _sr_padded_lat_opts(lat_opts_2e, image_extent)
    cells_pad = _sr_internal_cells(basis, system, lat_opts_2e, image_extent)
    if lat_opts_2e.pair_complete_1e:
        orig_cells = cells_pad
    else:
        orig_cells = list(direct_lattice_cells(system, float(lat_opts_2e.cutoff_bohr)))
    pad_index = {_cell_key(c): i for i, c in enumerate(cells_pad)}
    output_masks = []
    if exact_zone_bohr is None:
        zone_positions = list(range(len(orig_cells)))
    elif lat_opts_2e.pair_complete_1e:
        centers = np.asarray([shell.origin for shell in basis.shells()], dtype=float)
        zone_positions = []
        for pos, cell in enumerate(orig_cells):
            delta = centers[:, None] - centers[None, :] - np.asarray(cell.r_cart)
            mask = (np.sum(delta * delta, axis=2) <= float(exact_zone_bohr)**2).astype(np.uint8)
            if np.any(mask):
                zone_positions.append(pos)
                output_masks.append(mask.ravel())
    else:
        zone_positions = [
            i
            for i, c in enumerate(orig_cells)
            if float(np.linalg.norm(np.asarray(c.r_cart, dtype=float)))
            <= float(exact_zone_bohr)
        ]
    output_indices = [
        pad_index[_cell_key(orig_cells[i])] for i in zone_positions
    ]
    jk = _build_jk_domains_output_cells_mpi(
        basis,
        system,
        lat_pad,
        density,
        cells_pad,
        output_indices,
        output_masks,
        omega=float(omega),
        compute_exchange=compute_exchange,
        compute_coulomb=compute_coulomb,
        task_kind=output_cell_farming_task_kind,
        strategy=output_cell_farming_strategy,
        workspace=workspace,
        **({"companion_density": companion_density} if companion_density is not None else {}),
    )

    nbf = int(basis.nbasis)

    def _blocks_on_template(src) -> List[np.ndarray]:
        blocks = [
            np.zeros((nbf, nbf), dtype=float) for _ in range(len(orig_cells))
        ]
        # Each pybind .blocks read exports the whole padded domain.
        source_blocks = src.blocks if output_indices else ()
        for pos, pad_i in zip(zone_positions, output_indices):
            blocks[pos] = np.asarray(source_blocks[pad_i], dtype=float).copy()
        return blocks

    def project(jk):
        if lat_opts_2e.pair_complete_1e:
            from ._vibeqc_core import make_lattice_matrix_set

            return SimpleNamespace(
                J=make_lattice_matrix_set(nbf, orig_cells, _blocks_on_template(jk.J)),
                K=(make_lattice_matrix_set(nbf, orig_cells, _blocks_on_template(jk.K))
                   if compute_exchange else None),
                output_cell_farming_execution=getattr(jk, "output_cell_farming_execution", None),
            )

        # The template is compute_overlap_lattice's cell list, which under
        # pair_complete_1e (#429) extends past the plain |g| ball this
        # two-electron build enumerates (cpp/include/vibeqc/lattice_pair_cells.hpp);
        # J/K are zero out there by their own truncation. Identical lists with
        # the switch off, where fill_missing never fires.
        J = _copy_lattice_with_blocks(
            basis,
            system,
            lat_opts_2e,
            orig_cells,
            _blocks_on_template(jk.J),
            fill_missing=True,
        )
        K = None
        if compute_exchange:
            K = _copy_lattice_with_blocks(
                basis,
                system,
                lat_opts_2e,
                orig_cells,
                _blocks_on_template(jk.K),
                fill_missing=True,
            )
        return SimpleNamespace(
            J=J,
            K=K,
            output_cell_farming_execution=getattr(
                jk, "output_cell_farming_execution", None
            ),
        )

    return project(jk) if companion_density is None else tuple(project(part) for part in jk)


def build_bipole_restricted_fock(
    ctx: BipoleFockContext,
    density: LatticeMatrixSet,
    *,
    coeffs_for_rho: Optional[Sequence[np.ndarray]],
    alpha_hf: float = 1.0,
    exchange_assembly=None,
    _chi_exchange: bool = False,
    _chi_density_k=None,
    use_incremental: bool = True,
    reseed_incremental: bool = False,
) -> BipoleRestrictedFockBuild:
    """Build the restricted BIPOLE F^2e(g) and the k-space K_corr pieces.

    ``alpha_hf`` scales HF exchange: 1.0 for RHF (bit-identical to the RHF
    driver's former inline build), the functional's HF-exchange fraction
    for RKS. ``coeffs_for_rho`` is supplied only when the density is exactly
    represented by the current per-k orbitals; damped / ODA-mixed densities
    pass ``None`` so the J^LR density transform uses the real-space blocks.
    ``use_incremental=False`` forces a full J_SR/K_SR build even when the
    incremental accumulator is active (ODA's extra naive build and the
    post-convergence rebuild, both off the per-iter ΔD chain).
    ``reseed_incremental=True`` (only meaningful with ``use_incremental=False``)
    additionally re-syncs the active accumulator to that full build through
    :meth:`IncrementalJK.seed`, so the ΔD chain continues from the exact
    operator: the drivers' terminal confirmation rebuild uses it (#116).

    ``exchange_assembly`` (a
    :class:`vibeqc.periodic_screened_exchange.PeriodicExchangeAssembly`)
    adds the physical erfc kernel to the full-range exchange build.
    The private ``_chi_exchange`` opt-in permits mixed CAM/LRC assemblies;
    ordinary periodic callers retain the HSE-only screened contract.
    Only ``c_full`` scales the Ewald split and Madelung correction.
    The physical erfc arm contains its finite zero mode in the direct sum,
    adds no Madelung term, and is rebuilt on every call. The chi caller
    supplies its authoritative per-k density through ``_chi_density_k``.

    The per-k Fock assembly (Bloch sum + Hcore + V_xc + K_corr) stays in
    the caller; this returns the real-space ``f2e_real`` and ``k_corr_per_k``.
    """
    if ctx.resolved_domain is not None:
        ctx.resolved_domain.require_fock_context(ctx)
    if exchange_assembly is not None and exchange_assembly.is_screened:
        if not _chi_exchange and (
            float(exchange_assembly.c_full) != 0.0 or float(alpha_hf) != 0.0
        ):
            # Mixed CAM/LRC assemblies belong only to the explicit chi
            # route. The established periodic resolver stays fail-closed.
            raise NotImplementedError(
                "build_bipole_restricted_fock: screened exchange with a "
                "nonzero full-range arm is not supported (got c_full = "
                f"{exchange_assembly.c_full}, alpha_hf = {alpha_hf})."
            )
        if (
            ctx.fock_sym_map is not None
            and getattr(ctx.fock_sym_map, "pair_resolved", False)
            and ctx.rep_cell_indices is None
        ):
            # The dedicated K_erfc traversal
            # (_screened_exchange_lattice, HSE-chat-owned) runs on the
            # radial cell list, which cannot be key-matched against the
            # pair-resolved J template. The reduced build routes K
            # through the pair-resolved-aware builder, so
            # use_fock_symmetry_reduce composes; enforcement-only does
            # not (yet).
            raise NotImplementedError(
                "build_bipole_restricted_fock: pair-resolved SYM3b "
                "enforcement + screened exchange needs "
                "use_fock_symmetry_reduce=True (the enforcement-only "
                "composition is not wired)."
            )
        # Disable the pure-RKS exact-J shortcut for the base build: its
        # f2e spans the density cell list, while the dedicated K
        # traversal spans the lat_opts_2e operator list -- the fold
        # below needs the J and K builds on the same cells.
        from dataclasses import replace as _dc_replace

        base_ctx = (
            _dc_replace(ctx, exact_j_for_pure_rks=False)
            if ctx.exact_j_for_pure_rks
            else ctx
        )
        base = build_bipole_restricted_fock(
            base_ctx,
            density,
            coeffs_for_rho=coeffs_for_rho,
            alpha_hf=float(exchange_assembly.c_full),
            _chi_exchange=_chi_exchange,
            _chi_density_k=_chi_density_k,
            use_incremental=use_incremental,
            reseed_incremental=reseed_incremental,
        )
        return _add_screened_exchange_restricted(
            base_ctx, density, base, exchange_assembly,
            _chi_exchange=_chi_exchange,
        )
    if exchange_assembly is not None:
        alpha_hf = float(exchange_assembly.c_full)
    # ---- unpack the shared context into the names the body uses ----------
    basis = ctx.basis
    system = ctx.system
    lat_opts_2e = ctx.lat_opts_2e
    use_ewald_j_split = ctx.use_ewald_j_split
    exchange_split_active = ctx.exchange_split_active
    n_k = ctx.n_k
    omega_used = ctx.omega_used
    ewald_precision = ctx.ewald_precision
    ewald_cell_volume = ctx.ewald_cell_volume
    n_elec = ctx.n_elec
    _xi_madelung = ctx.xi_madelung
    j_lr_cache = ctx.j_lr_cache
    x_lr_cache = ctx.x_lr_cache
    incremental_jk = ctx.incremental_jk
    _rep_cell_indices = ctx.rep_cell_indices
    _fock_sym_map = ctx.fock_sym_map
    _mp_config = ctx.mp_config
    # M3: pair-resolved SYM3b mode -- the mapping carries the
    # group-invariant output/internal/density domains and the build
    # routes through the M1 full-domain binding. M4b: sr_image_extent
    # composes with both symmetry modes (padded internal ball through
    # the same binding).
    _pair_mode = _fock_sym_map is not None and getattr(
        _fock_sym_map, "pair_resolved", False
    )
    if _pair_mode and _mp_config.enabled:
        raise NotImplementedError(
            "pair-resolved SYM3b + the multipole far-field (G1, gated "
            "research artifact) is not wired; disable one of the two."
        )
    S_lat = ctx.s_lat
    S_k_list = ctx.s_k_list
    k_points = ctx.k_points
    weights = ctx.weights
    k_points_full = ctx.k_points_full
    weights_full = ctx.weights_full
    _ir_mapping = ctx.ir_mapping
    plog = ctx.plog
    exact_j_for_pure_rks = ctx.exact_j_for_pure_rks
    exact_j_ke_cutoff = ctx.exact_j_ke_cutoff
    exact_j_chunk_size = ctx.exact_j_chunk_size
    output_cell_farming_execution = None

    def _record_output_cell_farming(jk):
        nonlocal output_cell_farming_execution
        executed = getattr(jk, "output_cell_farming_execution", None)
        if executed is not None:
            if (
                output_cell_farming_execution is not None
                and executed != output_cell_farming_execution
            ):
                raise RuntimeError(
                    "build_bipole_restricted_fock: direct-ERI output-cell "
                    "farming changed its execution contract within one "
                    "Fock build"
                )
            output_cell_farming_execution = executed
        return jk

    def _split_k_density_list(d: LatticeMatrixSet) -> List[np.ndarray]:
        return split_k_density_list(ctx, d, retained_density_k=_chi_density_k)

    def _density_matrices_from_coeffs(
        coeffs_per_k: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        return restricted_density_from_coeffs(coeffs_per_k, ctx.n_occ)

    # ---- body (transcribed from the RHF driver; alpha_hf-generalised) ----
    if use_ewald_j_split and exact_j_for_pure_rks and alpha_hf == 0.0:
        from .ewald_composed import compute_j_ewald_3d_ft_k_density_to_cells

        assert omega_used is not None
        D_k_exact = _split_k_density_list(density)
        exact_cells = list(density.cells)
        J_exact_blocks = compute_j_ewald_3d_ft_k_density_to_cells(
            basis,
            system,
            D_k_exact,
            k_points,
            weights,
            exact_cells,
            float(omega_used),
            lattice_opts=lat_opts_2e,
            ke_cutoff=exact_j_ke_cutoff,
            chunk_size=exact_j_chunk_size,
        )
        f2e_real = make_lattice_matrix_set(
            basis.nbasis,
            exact_cells,
            [np.asarray(block, dtype=float) for block in J_exact_blocks],
        )
        e_j_short_range = 0.5 * _lattice_contract_blocks(
            density,
            exact_cells,
            J_exact_blocks,
            operator_name="J_exact",
        )
        e_j_long_range = 0.0
        e_exchange = None
        e_j_multipole = None
        e_j_bipolar_quartet = None
        K_corr_per_k = None
        e_2e_k_correction = 0.0
        e_exchange_finite_size = None
    elif use_ewald_j_split:
        need_k = alpha_hf > 0.0 or _mp_config.enabled
        if exchange_split_active:
            # Ewald exchange split: ONE fused traversal gives both
            # J_SR and K_SR, erfc(w)-screened. The erfc kernel makes
            # the direct K sum absolutely convergent regardless of
            # the density's cell decay (the full Bloch fold at Γ is
            # constant across cells); the LR + G=0 exchange pieces
            # are added in k-space below.
            # SYM3b: when point-group reduction is active, build J_SR/K_SR
            # only at the orbit-representative cells (full internal sum) and
            # reconstruct the rest by rotation -- bit-identical to
            # symmetrize_fock_blocks(full build). Composes with the
            # incremental ΔD path because reconstruction is linear.
            def _jk_sr_build(d):
                if _pair_mode:
                    d = _pair_resolved_density_for_sr(
                        basis, _fock_sym_map, d
                    )
                if _rep_cell_indices is not None:
                    from .bipole_symmetry_fock import (
                        build_jk_reduced_symmetrized,
                    )

                    jk = build_jk_reduced_symmetrized(
                        basis, system, lat_opts_2e, d,
                        float(omega_used), _fock_sym_map, _rep_cell_indices,
                        compute_exchange=need_k,
                        internal_extent_bohr=ctx.sr_image_extent,
                        output_cell_farming_task_kind=(
                            ctx.output_cell_farming_task_kind
                        ),
                        output_cell_farming_strategy=(
                            ctx.output_cell_farming_strategy
                        ),
                        workspace=_sr_workspace_for(ctx),
                    )
                    return _record_output_cell_farming(jk)
                if _pair_mode:
                    from .bipole_symmetry_fock import build_jk_pair_resolved

                    jk = build_jk_pair_resolved(
                        basis, system, lat_opts_2e, d,
                        float(omega_used), _fock_sym_map,
                        internal_extent_bohr=ctx.sr_image_extent,
                        output_cell_farming_task_kind=(
                            ctx.output_cell_farming_task_kind
                        ),
                        output_cell_farming_strategy=(
                            ctx.output_cell_farming_strategy
                        ),
                        workspace=_sr_workspace_for(ctx),
                    )
                    return _record_output_cell_farming(jk)
                if ctx.sr_image_extent is not None:
                    jk = _sr_image_padded_jk(
                        basis, system, lat_opts_2e, d,
                        float(omega_used), float(ctx.sr_image_extent),
                        exact_zone_bohr=ctx.exact_zone_bohr,
                        output_cell_farming_task_kind=(
                            ctx.output_cell_farming_task_kind
                        ),
                        output_cell_farming_strategy=(
                            ctx.output_cell_farming_strategy
                        ),
                        workspace=_sr_workspace_for(ctx),
                    )
                    return _record_output_cell_farming(jk)
                return build_jk_2e_real_space(
                    basis, system, lat_opts_2e, d, float(omega_used)
                )

            def _j_sr_only_build(d):
                from types import SimpleNamespace

                # ---- BIPOLE far-field: use skip-mask build when available.
                if (
                    ctx.spherical_moment_buffer is not None
                    and ctx.penetration_dispatch_j is not None
                ):
                    from .bipole_bipolar_skip_mask import (
                        dispatch_to_bipolar_skip_mask,
                    )
                    from ._vibeqc_core import (
                        build_jk_2e_real_space_bipolar_dispatch,
                    )
                    n_sh = len(list(basis.shells()))
                    from ._vibeqc_core import direct_lattice_cells as _dlc
                    _cells_for_mask = _dlc(system, lat_opts_2e.cutoff_bohr)
                    n_c = len(_cells_for_mask)
                    skip_mask = dispatch_to_bipolar_skip_mask(
                        ctx.penetration_dispatch_j, n_c, n_sh,
                    )
                    jk = build_jk_2e_real_space_bipolar_dispatch(
                        basis, system, lat_opts_2e, d, skip_mask,
                        float(omega_used) if use_ewald_j_split else 0.0,
                        False,  # compute_exchange=False → J-only
                    )
                    return SimpleNamespace(J=jk.J, K=None)

                if _pair_mode:
                    d = _pair_resolved_density_for_sr(
                        basis, _fock_sym_map, d
                    )
                if _pair_mode:
                    from .bipole_symmetry_fock import build_jk_pair_resolved

                    jk = build_jk_pair_resolved(
                        basis, system, lat_opts_2e, d,
                        float(omega_used), _fock_sym_map,
                        compute_exchange=False,
                        internal_extent_bohr=ctx.sr_image_extent,
                        output_cell_farming_task_kind=(
                            ctx.output_cell_farming_task_kind
                        ),
                        output_cell_farming_strategy=(
                            ctx.output_cell_farming_strategy
                        ),
                        workspace=_sr_workspace_for(ctx),
                    )
                    return _record_output_cell_farming(jk)
                if ctx.sr_image_extent is not None:
                    # M4b: padded J-only build (no wasted K).
                    jk = _sr_image_padded_jk(
                        basis, system, lat_opts_2e, d,
                        float(omega_used), float(ctx.sr_image_extent),
                        compute_exchange=False,
                        exact_zone_bohr=ctx.exact_zone_bohr,
                        output_cell_farming_task_kind=(
                            ctx.output_cell_farming_task_kind
                        ),
                        output_cell_farming_strategy=(
                            ctx.output_cell_farming_strategy
                        ),
                        workspace=_sr_workspace_for(ctx),
                    )
                    return _record_output_cell_farming(jk)
                return SimpleNamespace(
                    J=build_fock_2e_real_space(
                        basis,
                        system,
                        lat_opts_2e,
                        d,
                        0.0,  # exchange_scale = 0 -> pure J
                        float(omega_used),  # omega > 0 -> erfc kernel
                    ),
                    K=None,
                )

            if incremental_jk is not None and use_incremental:
                # Differential build: J_SR/K_SR from ΔD, accumulated;
                # the C++ density-envelope screening cheapens it as
                # the SCF converges. Bit-identical to the full build
                # up to the (reset-bounded) screening threshold.
                if alpha_hf > 0.0 or _rep_cell_indices is not None:
                    F_J_SR_lat, F_K_full_lat = incremental_jk.build(
                        density, _jk_sr_build
                    )
                else:
                    F_J_SR_lat, F_K_full_lat = incremental_jk.build(
                        density, _j_sr_only_build
                    )
            else:
                if alpha_hf > 0.0 or _rep_cell_indices is not None:
                    _jk_sr_lat = _jk_sr_build(density)
                    F_J_SR_lat = _jk_sr_lat.J
                    F_K_full_lat = _jk_sr_lat.K  # K_SR(erfc w)
                else:
                    F_J_SR_lat = _j_sr_only_build(density).J
                    F_K_full_lat = None
                if incremental_jk is not None and reseed_incremental:
                    # #116: the same closure the accumulator's own full
                    # build would call, so the seeded state is exactly a
                    # reset at this density (blocks copied by seed()).
                    incremental_jk.seed(density, F_J_SR_lat, F_K_full_lat)
        else:
            # Legacy convention: erfc-screened J + full-Coulomb
            # direct K (separate traversals). Known to mis-state
            # absolute energies (formally divergent K series with
            # the multi-k folded density); reached only via explicit
            # use_exchange_ewald_split=False now that the corrected
            # exchange split is the default at Γ and multi-k.
            if ctx.sr_image_extent is not None:
                F_J_SR_lat = _sr_image_padded_jk(
                    basis,
                    system,
                    lat_opts_2e,
                    density,
                    float(omega_used),
                    float(ctx.sr_image_extent),
                    compute_exchange=False,
                    output_cell_farming_task_kind=(
                        ctx.output_cell_farming_task_kind
                    ),
                    output_cell_farming_strategy=(
                        ctx.output_cell_farming_strategy
                    ),
                    workspace=_sr_workspace_for(ctx),
                ).J
            else:
                F_J_SR_lat = build_fock_2e_real_space(
                    basis,
                    system,
                    lat_opts_2e,
                    density,
                    0.0,  # exchange_scale = 0 -> pure J
                    float(omega_used),  # omega > 0 -> erfc kernel
                )
            F_K_full_lat = None
            if need_k:
                jk_full_lat = build_jk_2e_real_space(
                    basis,
                    system,
                    lat_opts_2e,
                    density,
                    0.0,
                )
                F_K_full_lat = jk_full_lat.K
        cells_lr = F_J_SR_lat.cells
        F2e_direct_blocks = []
        K_full_blocks = []
        # Convert each native block vector once, outside the cell loop.
        j_sr_blocks = F_J_SR_lat.blocks
        k_full_blocks = F_K_full_lat.blocks if F_K_full_lat is not None else None
        for c in range(len(cells_lr)):
            j_sr = np.asarray(j_sr_blocks[c], dtype=float)
            if k_full_blocks is not None:
                k_full = np.asarray(k_full_blocks[c], dtype=float)
            else:
                k_full = np.zeros_like(j_sr)
            K_full_blocks.append(k_full)
            F2e_direct_blocks.append(j_sr - 0.5 * alpha_hf * k_full)

        rho_hat_for_LR = None
        D_k_split: Optional[List[np.ndarray]] = None
        if exchange_split_active and n_k == 1:
            # Γ + wide density list: r̂(K) computed exactly from the
            # BvK representative D(Γ) (= home-cell block for every
            # density representation at n_k = 1 -- Bloch-constant,
            # SAD-local, damped, ODA-mixed). This also keeps the
            # J^LR block build on the cache's (operator) cell list:
            # with rho_hat given, the wide density set is not
            # consulted (see compute_J_long_range_real_space_blocks).
            from .bipole_fock_ewald import compute_rho_hat_from_k_density

            rho_hat_for_LR = compute_rho_hat_from_k_density(
                _split_k_density_list(density),
                [np.zeros(3)],
                [1.0],
                j_lr_cache,
            )
        elif exchange_split_active:
            # Multi-k split: per-k D(k) via the exact BvK-torus fold
            # -- valid for every representation (orbital rebuilds,
            # SAD-local, damped, ODA-mixed, warm-starts), and
            # consistent with the real-space density the direct
            # erfc builders contract. Used for r̂(K), the LR-exchange
            # channels, and the G=0 correction below.
            from .bipole_fock_ewald import compute_rho_hat_from_k_density

            D_k_split = _split_k_density_list(density)
            rho_hat_for_LR = compute_rho_hat_from_k_density(
                D_k_split,
                k_points,
                weights,
                j_lr_cache,
            )
        elif coeffs_for_rho is not None:
            from .bipole_fock_ewald import compute_rho_hat_from_k_density

            D_k_for_rho = _density_matrices_from_coeffs(coeffs_for_rho)
            if _ir_mapping.size > 0:
                D_k_full = [D_k_for_rho[int(idx)] for idx in _ir_mapping]
                rho_hat_for_LR = compute_rho_hat_from_k_density(
                    D_k_full,
                    k_points_full,
                    weights_full,
                    j_lr_cache,
                )
            else:
                rho_hat_for_LR = compute_rho_hat_from_k_density(
                    D_k_for_rho,
                    k_points,
                    weights,
                    j_lr_cache,
                )
        from .bipole_fock_ewald import compute_J_long_range_real_space_blocks

        lr_density = (
            density if rho_hat_for_LR is not None
            else _density_on_cache_cells(density, j_lr_cache, system)
        )
        F_LR_blocks = compute_J_long_range_real_space_blocks(
            lr_density,
            basis,
            system,
            omega_used,
            precision=ewald_precision,
            cache=j_lr_cache,
            rho_hat=rho_hat_for_LR,
        )
        if lat_opts_2e.pair_complete_1e or lr_density is not density:
            F_LR_blocks = _reciprocal_blocks_on_output(
                F_LR_blocks, j_lr_cache, system, F_J_SR_lat.cells,
            )
        assert ewald_cell_volume is not None
        j_background_potential = (
            -np.pi
            * float(n_elec)
            / (float(omega_used) * float(omega_used) * float(ewald_cell_volume))
        )
        if _pair_mode:
            # J^LR lives on the wider pair template, but the uniform
            # background is the scalar shift v_bg.S(k) of the actual
            # generalised eigenproblem.  Keep it on S_lat and emit zeros
            # on pair-only extension cells; using S_ext there changes the
            # Fock without changing the SCF overlap and breaks Pulay
            # stationarity.
            zero_s = np.zeros((basis.nbasis, basis.nbasis), dtype=float)
            s_radial = {
                _cell_key(cell): np.asarray(block, dtype=float)
                for cell, block in zip(S_lat.cells, S_lat.blocks)
            }
            s_blocks = {
                _cell_key(cell): s_radial.get(_cell_key(cell), zero_s)
                for cell in cells_lr
            }
        else:
            s_blocks = {
                _cell_key(cell): np.asarray(block, dtype=float)
                for cell, block in zip(S_lat.cells, S_lat.blocks)
            }
        for c, cell in enumerate(cells_lr):
            key = _cell_key(cell)
            F_LR_blocks[c] = F_LR_blocks[c] + j_background_potential * s_blocks.get(key, np.zeros_like(F_LR_blocks[c]))
        e_j_short_range = 0.5 * _lattice_contract(
            density,
            F_J_SR_lat,
            operator_name="J_SR",
        )
        e_j_long_range = 0.5 * _lattice_contract_blocks(
            density,
            cells_lr,
            F_LR_blocks,
            operator_name="J_LR",
        )
        e_exchange = None
        if alpha_hf > 0.0:
            e_exchange = -0.25 * alpha_hf * _lattice_contract_blocks(
                density,
                cells_lr,
                K_full_blocks,
                operator_name="K_full",
            )

        # ---- Ewald exchange split: K_LR + G=0/Madelung correction.
        # These are k-space objects (dense at each k), added to F(k)
        # after the Bloch sum rather than to the real-space blocks:
        #   K_corr(Γ) = K_LR(Γ) + (ξ_M - pi/(Vw^2)).S(Γ).D(Γ).S(Γ)
        # The -pi/(Vw^2).SDS piece removes the K=0 component the
        # absolutely-convergent direct K_SR sum implicitly contains
        # (r̃_pair(K=0) = S(Γ)); +ξ_M.SDS is the probe-charge Ewald
        # finite-size correction (exxdiv='ewald'; equals PySCF's
        # _ewald_exxdiv_for_G0). q->0 exchange singularity treatment
        # per Gygi & Baldereschi, Phys. Rev. B 34, 4405(R) (1986),
        # doi:10.1103/PhysRevB.34.4405; probe-charge Ewald form per
        # Sundararaman & Arias, Phys. Rev. B 87, 165122 (2013),
        # doi:10.1103/PhysRevB.87.165122.
        K_corr_per_k: Optional[List[np.ndarray]] = None
        e_2e_k_correction = 0.0
        # The q+G=0 gauge's OWN energy share, split out of e_exchange so a
        # cross-code comparison can see it. #82 spent fourteen days on a
        # +0.64 Ha MgO/STO-3G (2,2,2) offset against CRYSTAL23, which applies
        # no such correction at all; the term itself is -2.88 Ha there.
        e_exchange_finite_size: Optional[float] = None
        if exchange_split_active and alpha_hf > 0.0 and n_k == 1:
            from .bipole_fock_ewald import (
                compute_K_long_range_gamma,
                exchange_q0_gauge_constant,
            )

            D_gamma = np.real(_split_k_density_list(density)[0])
            K_LR_gamma = compute_K_long_range_gamma(j_lr_cache, D_gamma)
            S_gamma = np.real(np.asarray(S_k_list[0]))
            c_g0 = exchange_q0_gauge_constant(
                _xi_madelung, omega_used, ewald_cell_volume, 1
            )
            K_corr_per_k = [
                alpha_hf * (K_LR_gamma + c_g0 * (S_gamma @ D_gamma @ S_gamma))
            ]
            # ΔF(Γ) = -1/2.K_corr => ΔE_2e = 1/2.Tr[D.ΔF] = -1/4.Tr[D.K_corr]
            e_2e_k_correction = -0.25 * float(
                np.einsum("ij,ji->", D_gamma, K_corr_per_k[0]).real
            )
            # The xi_M share alone -- the part that vanishes under
            # exchange_exxdiv='none' -- at the same -1/4 a_HF prefactor.
            # NOT the c_g0 share: the -pi/(alpha^2 V n_k) half of c_g0 is an
            # internal transfer between this split's real-space K_SR arm and
            # its k-space arm, not a convention a comparing code could have
            # chosen differently.
            e_exchange_finite_size = -0.25 * alpha_hf * float(
                _xi_madelung
            ) * float(
                np.einsum(
                    "ij,ji->", D_gamma, S_gamma @ D_gamma @ S_gamma
                ).real
            )
            e_exchange = e_exchange + e_2e_k_correction
        elif exchange_split_active and alpha_hf > 0.0:
            # Multi-k (option (b) Phase 3): q = k-k′ LR-exchange
            # channels + the supercell G=0/Madelung correction:
            #   K_corr(k) = K_LR(k) + (ξ_M(sc) - pi/(V_sc.w^2)).S(k)D(k)S(k)
            # with V_sc = n_k.V. The -pi/(V_sc.w^2) piece removes the
            # q=0, G=0 component the absolutely-convergent direct
            # K_SR(erfc) sum implicitly contains at each k -- its
            # Bloch fold carries w_k.ṽ_erfc(K->0)/V = pi/(n_k.V.w^2)
            # against S(k)D(k)S(k) (lim_{K->0} (4pi/K^2)(1-e^{-K^2/4w^2})
            # = pi/w^2) -- and +ξ_M(supercell).SDS is the probe-charge
            # Ewald finite-size correction, PySCF's
            # _ewald_exxdiv_for_G0 with the same single madelung
            # constant at every k. References as in the Γ branch
            # (Gygi-Baldereschi 1986; Sundararaman-Arias 2013).
            from .bipole_fock_ewald import (
                compute_K_long_range_mesh,
                exchange_q0_gauge_constant,
            )

            assert x_lr_cache is not None and D_k_split is not None
            c_g0 = exchange_q0_gauge_constant(
                _xi_madelung, omega_used, ewald_cell_volume, n_k
            )
            K_LR_per_k = compute_K_long_range_mesh(
                x_lr_cache, k_points, weights, [D_k_split],
            )[0]
            K_corr_per_k = []
            e_exchange_finite_size = 0.0
            for k_idx in range(n_k):
                K_LR_k = K_LR_per_k[k_idx]
                S_k_c = np.asarray(S_k_list[k_idx], dtype=complex)
                D_k_c = D_k_split[k_idx]
                sds_k = S_k_c @ D_k_c @ S_k_c
                K_corr_k = alpha_hf * (K_LR_k + c_g0 * sds_k)
                K_corr_per_k.append(K_corr_k)
                # ΔF(k) = -1/2.K_corr(k) =>
                # ΔE_2e = 1/2.S_k w_k.Tr[D(k).ΔF(k)]
                e_2e_k_correction += -0.25 * float(
                    weights[k_idx]
                ) * float(np.einsum("ij,ji->", D_k_c, K_corr_k).real)
                e_exchange_finite_size += -0.25 * float(
                    weights[k_idx]
                ) * alpha_hf * float(_xi_madelung) * float(
                    np.einsum("ij,ji->", D_k_c, sds_k).real
                )
            e_exchange = e_exchange + e_2e_k_correction

        f2e_real = F_J_SR_lat
        for c in range(len(cells_lr)):
            f2e_real.set_block(
                c,
                F2e_direct_blocks[c] + F_LR_blocks[c],
            )

        # ---- Dormant PDR 1988 Ch. II.4c quartet-expansion prototype -----
        # This research-only branch is unreachable from the four public
        # drivers. If supplied by a low-level diagnostic context, it uses the
        # prototype dispatch/contractor in place of selected exact quartets;
        # that substitution is not a supported production Hamiltonian.
        e_j_multipole: Optional[float] = None
        e_j_bipolar_quartet: Optional[float] = None
        # Track whether _j_sr_only_build already built J_SR with the skip
        # mask, so the far-field section can avoid rebuilding redundantly.
        _jsr_already_skip_masked = False
        if (
            ctx.spherical_moment_buffer is not None
            and ctx.penetration_dispatch_j is not None
        ):
            from .bipole_bipolar_skip_mask import (
                dispatch_to_bipolar_skip_mask,
            )

            # Build C++ skip mask from the penetration dispatch.
            n_sh = len(list(basis.shells()))
            n_c = len(cells_lr)
            skip_mask = dispatch_to_bipolar_skip_mask(
                ctx.penetration_dispatch_j, n_c, n_sh,
            )

            # For pure DFT (alpha_hf == 0), _j_sr_only_build already built
            # J_SR with the far-field skip mask (_j_sr_only_build lines
            # 767-788).  Reuse that result to avoid a redundant C++ traversal.
            # For hybrids (alpha_hf > 0), _jk_sr_build did NOT use the skip
            # mask, so we MUST rebuild here to get K with far-field skipped.
            if alpha_hf > 0.0:
                from ._vibeqc_core import (
                    build_jk_2e_real_space_bipolar_dispatch,
                )

                jk_near = build_jk_2e_real_space_bipolar_dispatch(
                    basis,
                    system,
                    lat_opts_2e,
                    density,
                    skip_mask,
                    float(omega_used) if use_ewald_j_split else 0.0,
                    True,  # compute_exchange=True → need K for hybrids
                )
            else:
                # Pure DFT: reuse J_SR from _j_sr_only_build (already
                # computed with skip mask at lines 842-844).
                from types import SimpleNamespace
                jk_near = SimpleNamespace(
                    J=F_J_SR_lat, K=None,
                )
                _jsr_already_skip_masked = True

            # Compute far-field J: use pre-computed kernel if available.
            if ctx.far_field_fock_kernel is not None:
                from .bipole_far_field_kernel import (
                    apply_far_field_fock_kernel,
                    apply_far_field_fock_kernel_with_symmetry,
                )
                density_dict: Dict[Tuple[int, int, int], np.ndarray] = {
                    (cell.index[0], cell.index[1], cell.index[2]): (
                        np.asarray(density.blocks[c], dtype=float)
                    )
                    for c, cell in enumerate(density.cells)
                }
                if ctx.symmetry_reconstruction_map is not None:
                    ff_result = apply_far_field_fock_kernel_with_symmetry(
                        ctx.far_field_fock_kernel,
                        ctx.symmetry_reconstruction_map,
                        density_dict,
                    )
                else:
                    ff_result = apply_far_field_fock_kernel(
                        ctx.far_field_fock_kernel, density_dict,
                    )
            else:
                # Fallback: C++ direct contractor (no pre-computed kernel).
                from .bipole_contractor_native import (
                    compute_bipolar_coulomb_far_field_native,
                )
                density_dict = {
                    (cell.index[0], cell.index[1], cell.index[2]): (
                        np.asarray(density.blocks[c], dtype=float)
                    )
                    for c, cell in enumerate(density.cells)
                }
                ff_result = compute_bipolar_coulomb_far_field_native(
                    ctx.spherical_moment_buffer,
                    ctx.penetration_dispatch_j,
                    density_dict,
                    ewald_omega=omega_used if use_ewald_j_split else 0.0,
                    nbf=basis.nbasis,
                )
            e_j_bipolar_quartet = ff_result.e_coulomb_far

            # Restricted hybrid consistency fix: the Fock K must be the
            # SAME skip-masked near-field K the unrestricted path uses
            # (near-field exact, far-K quartets skipped -- see
            # build_bipole_unrestricted_fock's G6 branch), not the earlier
            # full K_full_blocks build.  Before this fix the hybrid path
            # paid for a skip-masked JK rebuild and then combined its J
            # with the FULL K, silently diverging from the documented
            # approximation and from the unrestricted route.  The exchange
            # energy must contract the same near-K operator so energy and
            # Fock stay variational partners (the Ewald LR exchange
            # correction computed above is re-added on top).
            _k_near_blocks: Optional[List[np.ndarray]] = None
            if jk_near.K is not None:
                _k_near_blocks = [
                    np.asarray(jk_near.K.blocks[c], dtype=float)
                    for c in range(n_c)
                ]
                e_exchange = -0.25 * alpha_hf * _lattice_contract_blocks(
                    density,
                    cells_lr,
                    _k_near_blocks,
                    operator_name="K_SR_near",
                )
                if e_2e_k_correction:
                    e_exchange = e_exchange + e_2e_k_correction

            # Combine: C++ near-field exact + Python far-field multipole.
            for c in range(n_c):
                key = (
                    cells_lr[c].index[0],
                    cells_lr[c].index[1],
                    cells_lr[c].index[2],
                )
                j_near = np.asarray(
                    jk_near.J.blocks[c], dtype=float
                )
                j_ff = ff_result.fock_blocks.get(
                    key, np.zeros_like(j_near),
                )
                if _k_near_blocks is not None:
                    k_block = _k_near_blocks[c]
                else:
                    k_block = K_full_blocks[c]
                F2e_direct_blocks[c] = (
                    j_near + j_ff - 0.5 * alpha_hf * k_block
                )
                f2e_real.set_block(
                    c, F2e_direct_blocks[c] + F_LR_blocks[c],
                )

            e_j_short_range = 0.5 * _lattice_contract_blocks(
                density,
                cells_lr,
                [
                    np.asarray(jk_near.J.blocks[c], dtype=float)
                    + ff_result.fock_blocks.get(
                        (
                            cells_lr[c].index[0],
                            cells_lr[c].index[1],
                            cells_lr[c].index[2],
                        ),
                        np.zeros((basis.nbasis, basis.nbasis), dtype=float),
                    )
                    for c in range(n_c)
                ],
                operator_name="J_SR_bipolar",
            )

            if ff_result.n_quartets > 0:
                plog.info(
                    f"  Dormant PDR 1988 quartet prototype (L_max="
                    f"{ctx.spherical_moment_buffer.L_max}): "
                    f"{ff_result.n_quartets} far quartets, "
                    f"E_J_bipolar = {e_j_bipolar_quartet:+.6f} Ha"
                )

        elif _mp_config.enabled:
            # Legacy G1 path: retired cell-level far-field replacement.
            # Known broken under the corrected Ewald exchange split.
            from .bipole_fock_multipole import apply_multipole_far_field

            mp_result = apply_multipole_far_field(
                density,
                basis,
                system,
                lat_opts_2e,
                _mp_config,
                J_SR_blocks=[
                    np.asarray(F_J_SR_lat.blocks[c], dtype=float)
                    for c in range(len(cells_lr))
                ],
                K_blocks=K_full_blocks,
                F_LR_blocks=F_LR_blocks,
                exchange_scale=0.5 * alpha_hf,
            )
            for c in range(len(cells_lr)):
                f2e_real.set_block(c, mp_result.f2e_blocks[c])
            e_j_multipole = mp_result.e_j_multipole
    else:
        if alpha_hf > 0.0:
            jk_full = build_jk_2e_real_space(
                basis,
                system,
                lat_opts_2e,
                density,
                0.0,
            )
            e_j_short_range = 0.5 * _lattice_contract(
                density,
                jk_full.J,
                operator_name="J",
            )
            e_exchange = -0.25 * alpha_hf * _lattice_contract(
                density,
                jk_full.K,
                operator_name="K",
            )
            f2e_real = jk_full.J
            for c in range(len(f2e_real.cells)):
                f2e_real.set_block(
                    c,
                    np.asarray(jk_full.J.blocks[c], dtype=float)
                    - 0.5
                    * alpha_hf
                    * np.asarray(jk_full.K.blocks[c], dtype=float),
                )
        else:
            f2e_real = build_fock_2e_real_space(
                basis,
                system,
                lat_opts_2e,
                density,
                0.0,
                0.0,
            )
            e_j_short_range = 0.5 * _lattice_contract(
                density,
                f2e_real,
                operator_name="J",
            )
            e_exchange = None
        e_j_long_range = None
        e_j_multipole = None
        K_corr_per_k = None
        e_2e_k_correction = 0.0
        e_exchange_finite_size = None
        e_j_bipolar_quartet = None

    # SYM3b: enforce Fock symmetry on real-space blocks. The template is
    # passed explicitly: the exact-J pure-RKS path emits f2e on the
    # DENSITY cell list (wider than the mapping's), and positional
    # association across different radial lists silently scrambles
    # blocks (direct_lattice_cells tie order is not stable across
    # radii) -- the key-based association is exact for both templates.
    if _fock_sym_map is not None:
        from .bipole_symmetry_fock import symmetrize_fock_blocks

        f2e_blocks = [np.asarray(block, dtype=float) for block in f2e_real.blocks]
        symmetrize_fock_blocks(
            f2e_blocks, _fock_sym_map, cells=list(f2e_real.cells)
        )
        for c in range(len(f2e_real.cells)):
            f2e_real.set_block(c, f2e_blocks[c])

    exchange_components = {
        "full_nonzero": float(e_exchange or 0.0) - float(e_exchange_finite_size or 0.0),
        "full_zero": float(e_exchange_finite_size or 0.0),
        "erfc": 0.0,
    }
    if _chi_exchange:
        e_exchange = sum(exchange_components.values())
    return BipoleRestrictedFockBuild(
        e_hf_exchange_components=exchange_components,
        f2e_real=f2e_real,
        k_corr_per_k=K_corr_per_k,
        e_j_short_range=e_j_short_range,
        e_j_long_range=e_j_long_range,
        e_exchange=e_exchange,
        e_j_multipole=e_j_multipole,
        e_2e_k_correction=e_2e_k_correction,
        e_exchange_finite_size=e_exchange_finite_size,
        e_j_bipolar_quartet=e_j_bipolar_quartet,
        output_cell_farming_execution=output_cell_farming_execution,
    )


def _same_lattice_density(left: LatticeMatrixSet, right: LatticeMatrixSet) -> bool:
    """Exact ordered inputs, including cell geometry, for within-build reuse."""
    if left.nbf != right.nbf:
        return False
    left_cells, right_cells = left.cells, right.cells
    left_blocks, right_blocks = left.blocks, right.blocks
    if not (len(left_cells) == len(right_cells) == len(left_blocks) == len(right_blocks)):
        return False
    return all(
        np.array_equal(a.index, b.index)
        and np.array_equal(a.r_cart, b.r_cart)
        and np.array_equal(da, db)
        for a, b, da, db in zip(left_cells, right_cells, left_blocks, right_blocks)
    )


def _reuse_paired_spin_jk(build_jk):
    """Share identical inputs in one sequential pair of spin builds.

    Compare the actual callback inputs, including incremental delta densities,
    rather than assuming equal absolute spin densities imply equal histories.
    Both calls must finish before their results are mutated. The second result
    owns independent matrices because unrestricted assembly modifies J in place.
    This closure must be recreated for every Fock build, never retained in a
    context or accumulator. The density contraction is unchanged (Izmaylov et
    al., JCP 125, 104103 (2006), Eq. (7), doi:10.1063/1.2347713).
    """
    from types import SimpleNamespace

    previous_density = previous_jk = None

    def build(density):
        nonlocal previous_density, previous_jk
        if previous_density is not None and _same_lattice_density(previous_density, density):
            def copy_lattice(lattice):
                if lattice is None:
                    return None
                return make_lattice_matrix_set(
                    int(lattice.nbf), list(lattice.cells),
                    [np.asarray(block, dtype=float).copy() for block in lattice.blocks],
                )

            return SimpleNamespace(
                J=copy_lattice(previous_jk.J), K=copy_lattice(previous_jk.K),
                output_cell_farming_execution=getattr(
                    previous_jk, "output_cell_farming_execution", None),
            )
        previous_density = density
        previous_jk = build_jk(density)
        return previous_jk

    return build


def _add_screened_exchange_unrestricted(
    ctx: BipoleFockContext,
    density_alpha: LatticeMatrixSet,
    density_beta: LatticeMatrixSet,
    base: BipoleUnrestrictedFockBuild,
    exx,
    *,
    _chi_exchange: bool = False,
) -> BipoleUnrestrictedFockBuild:
    """Add per-spin ``-c_sr K_erfc(D_s)`` to a J plus full-exchange build.

    Per-spin convention: ``F_s -= c_sr * K_s(D_s)`` (K built from the
    per-spin density; no 1/2), so
    ``e_exchange = -c_sr/2 * sum_s Tr[D_s K_s]`` -- same closure as the
    full-range per-spin fold.
    """
    screened_kwargs = {"_chi_exchange": True} if _chi_exchange else {}
    # K is the density contraction of the same spin-independent erfc ERIs:
    # Heyd et al. (2003), Sec. II.A, Eq. (6), doi:10.1063/1.1564060.
    # See also Izmaylov et al. (2006), Eq. (7), doi:10.1063/1.2347713.
    # Identical spin densities therefore need only one traversal. Require
    # exact input equality, not a closed-shell label or a numerical tolerance;
    # keep both spin folds/energy contractions and never retain K across builds.
    same_density = _same_lattice_density(density_alpha, density_beta)
    if not same_density and _can_pair_sr(ctx):
        K_a_lat, K_b_lat = _screened_exchange_lattice(
            ctx, density_alpha, exx.omega_screen,
            companion_density=density_beta, **screened_kwargs,
        )
    else:
        K_a_lat = _screened_exchange_lattice(ctx, density_alpha, exx.omega_screen, **screened_kwargs)
        K_b_lat = (K_a_lat if same_density else _screened_exchange_lattice(
            ctx, density_beta, exx.omega_screen, **screened_kwargs))
    f2e_a = base.f2e_alpha_real
    f2e_b = base.f2e_beta_real
    n_cells = len(f2e_a.cells)
    K_a_blocks = _screened_k_blocks_for_cells(K_a_lat, f2e_a.cells, exx.c_sr)
    K_b_blocks = _screened_k_blocks_for_cells(K_b_lat, f2e_b.cells, exx.c_sr)
    source_a_blocks, source_b_blocks = f2e_a.blocks, f2e_b.blocks
    for c in range(n_cells):
        f2e_a.set_block(
            c, np.asarray(source_a_blocks[c], dtype=float) - K_a_blocks[c]
        )
        f2e_b.set_block(
            c, np.asarray(source_b_blocks[c], dtype=float) - K_b_blocks[c]
        )
    if ctx.fock_sym_map is not None:
        from .bipole_symmetry_fock import symmetrize_fock_blocks

        a_blocks = [
            np.asarray(block, dtype=float) for block in f2e_a.blocks
        ]
        b_blocks = [
            np.asarray(block, dtype=float) for block in f2e_b.blocks
        ]
        symmetrize_fock_blocks(a_blocks, ctx.fock_sym_map, cells=list(f2e_a.cells))
        symmetrize_fock_blocks(b_blocks, ctx.fock_sym_map, cells=list(f2e_b.cells))
        for c in range(n_cells):
            f2e_a.set_block(c, a_blocks[c])
            f2e_b.set_block(c, b_blocks[c])
    erfc_energy = -0.5 * (
        _lattice_contract_blocks(
            density_alpha,
            f2e_a.cells,
            K_a_blocks,
            operator_name="K_erfc_alpha",
        )
        + _lattice_contract_blocks(
            density_beta,
            f2e_b.cells,
            K_b_blocks,
            operator_name="K_erfc_beta",
        )
    )
    # The base is the executed full-only arm. Record its physical zero
    # term before adding the independently contracted erfc contribution.
    base.e_hf_exchange_components = {
        "full_nonzero": float(base.e_exchange or 0.0) - float(base.e_exchange_finite_size or 0.0),
        "full_zero": float(base.e_exchange_finite_size or 0.0),
        "erfc": float(erfc_energy),
    }
    base.e_exchange = (sum(base.e_hf_exchange_components.values()) if _chi_exchange
                       else float(base.e_exchange or 0.0) + erfc_energy)
    execution_type = BipoleChiScreenedExchangeExecution if _chi_exchange else BipoleScreenedExchangeExecution
    base.screened_exchange_execution = execution_type(
        c_sr=exx.c_sr,
        omega_screen_bohr_inv=exx.omega_screen,
    )
    return base


def build_bipole_unrestricted_fock(
    ctx: BipoleFockContext,
    density_alpha: LatticeMatrixSet,
    density_beta: LatticeMatrixSet,
    *,
    n_alpha: int,
    n_beta: int,
    coeffs_alpha_for_rho: Optional[Sequence[np.ndarray]],
    coeffs_beta_for_rho: Optional[Sequence[np.ndarray]],
    hcore_k_list: Optional[Sequence[np.ndarray]] = None,
    occupations_alpha_per_k: Optional[Sequence[np.ndarray]] = None,
    occupations_beta_per_k: Optional[Sequence[np.ndarray]] = None,
    smearing_temperature: float = 0.0,
    incremental_jk_alpha: Any = None,
    incremental_jk_beta: Any = None,
    incremental_jk_total: Any = None,
    use_incremental: bool = True,
    reseed_incremental: bool = False,
    alpha_hf: float = 1.0,
    exchange_assembly=None,
    _chi_exchange: bool = False,
    _chi_density_k=None,
    patom_hf_like: bool = False,
    use_reduced_j_only: bool = False,
) -> BipoleUnrestrictedFockBuild:
    """Build the unrestricted BIPOLE F^2e blocks and per-spin F(k).

    UHF is the ``alpha_hf=1.0`` reference case (bit-identical to the
    pre-M2c inline build). UKS (M2d) parameterises its deltas:

    - ``alpha_hf`` scales the exchange (``exchange_scale``); a pure
      functional (``alpha_hf=0``) takes the J-only builds and reports
      ``e_exchange=None``.
    - ``patom_hf_like`` forces full HF-like exchange for the one PATOM
      in-field seed build, independent of the target functional.
    - ``incremental_jk_total`` is the pure-functional single ΔD
      accumulator on the total density (hybrids use the per-spin pair).
    - ``use_reduced_j_only`` routes the J-only builds through the
      SYM3b-reduced builder when the mask is active (UKS convention;
      UHF's legacy-gauge J build stays unreduced).
    - ``hcore_k_list=None`` skips the per-k assembly; the caller
      assembles F(k) itself from ``f2e_*_real`` + ``k_corr_*_per_k``
      (UKS inserts per-spin V_xc before hermitising).
    - ``exchange_assembly`` generalises the exchange to the CAM form
      for screened hybrids exactly as in
      :func:`build_bipole_restricted_fock`: the base J plus full-range
      exchange build runs first and the per-spin physical erfc K is
      added. Mixed assemblies require the private ``_chi_exchange`` opt-in.
      Requires ``hcore_k_list=None`` (the UKS calling
      convention; UHF carries no functional and never screens).
    """
    if ctx.resolved_domain is not None:
        ctx.resolved_domain.require_fock_context(ctx)
    # The PATOM in-field seed is one full-HF-like build independent of
    # the target functional (patom_hf_like forces exchange_scale=1.0),
    # so it bypasses the screened branch by design.
    if (
        exchange_assembly is not None
        and exchange_assembly.is_screened
        and not patom_hf_like
    ):
        if not _chi_exchange and (
            float(exchange_assembly.c_full) != 0.0 or float(alpha_hf) != 0.0
        ):
            raise NotImplementedError(
                "build_bipole_unrestricted_fock: screened exchange with a "
                "nonzero full-range arm is not supported "
                f"(c_full = {exchange_assembly.c_full}, alpha_hf = "
                f"{alpha_hf})."
            )
        if hcore_k_list is not None:
            raise NotImplementedError(
                "build_bipole_unrestricted_fock: the screened-exchange "
                "branch folds K into the real-space blocks after the "
                "base build; pass hcore_k_list=None and assemble F(k) "
                "in the caller (the UKS convention)."
            )
        if (
            ctx.fock_sym_map is not None
            and getattr(ctx.fock_sym_map, "pair_resolved", False)
            and ctx.rep_cell_indices is None
        ):
            # See build_bipole_restricted_fock: the dedicated K_erfc
            # traversal runs on the radial list; only the reduced build
            # composes with pair-resolved SYM3b.
            raise NotImplementedError(
                "build_bipole_unrestricted_fock: pair-resolved SYM3b "
                "enforcement + screened exchange needs "
                "use_fock_symmetry_reduce=True (the enforcement-only "
                "composition is not wired)."
            )
        base = build_bipole_unrestricted_fock(
            ctx,
            density_alpha,
            density_beta,
            n_alpha=n_alpha,
            n_beta=n_beta,
            coeffs_alpha_for_rho=coeffs_alpha_for_rho,
            coeffs_beta_for_rho=coeffs_beta_for_rho,
            hcore_k_list=None,
            occupations_alpha_per_k=occupations_alpha_per_k,
            occupations_beta_per_k=occupations_beta_per_k,
            smearing_temperature=smearing_temperature,
            incremental_jk_alpha=incremental_jk_alpha,
            incremental_jk_beta=incremental_jk_beta,
            incremental_jk_total=incremental_jk_total,
            use_incremental=use_incremental,
            reseed_incremental=reseed_incremental,
            alpha_hf=float(exchange_assembly.c_full),
            _chi_exchange=_chi_exchange,
            _chi_density_k=_chi_density_k,
            patom_hf_like=False,
            use_reduced_j_only=use_reduced_j_only,
        )
        return _add_screened_exchange_unrestricted(
            ctx, density_alpha, density_beta, base, exchange_assembly,
            _chi_exchange=_chi_exchange,
        )
    if exchange_assembly is not None:
        alpha_hf = float(exchange_assembly.c_full)

    basis = ctx.basis
    system = ctx.system
    lat_opts_2e = ctx.lat_opts_2e
    use_ewald_j_split = ctx.use_ewald_j_split
    exchange_split_active = ctx.exchange_split_active
    n_k = ctx.n_k
    omega_used = ctx.omega_used
    ewald_precision = ctx.ewald_precision
    ewald_cell_volume = ctx.ewald_cell_volume
    n_elec = ctx.n_elec
    _xi_madelung = ctx.xi_madelung
    j_lr_cache = ctx.j_lr_cache
    x_lr_cache = ctx.x_lr_cache
    _rep_cell_indices = ctx.rep_cell_indices
    _fock_sym_map = ctx.fock_sym_map
    _mp_config = ctx.mp_config
    # M3: pair-resolved SYM3b mode (see build_bipole_restricted_fock).
    # M4b: sr_image_extent composes with both symmetry modes.
    _pair_mode = _fock_sym_map is not None and getattr(
        _fock_sym_map, "pair_resolved", False
    )
    if _pair_mode and _mp_config.enabled:
        raise NotImplementedError(
            "pair-resolved SYM3b + the multipole far-field (G1, gated "
            "research artifact) is not wired; disable one of the two."
        )
    S_lat = ctx.s_lat
    S_k_list = ctx.s_k_list
    k_points = ctx.k_points
    weights = ctx.weights
    k_points_full = ctx.k_points_full
    weights_full = ctx.weights_full
    _ir_mapping = ctx.ir_mapping
    plog = ctx.plog

    exchange_scale = 1.0 if patom_hf_like else float(alpha_hf)
    need_k = exchange_scale > 0.0

    def _spin_density_k(density, spin):
        retained = None if _chi_density_k is None else _chi_density_k[spin]
        return split_k_density_list(ctx, density, retained_density_k=retained)

    # Combine on the spin densities' OWN template, never the radial
    # overlap list. Under the Ewald exchange split the spin densities
    # live on the wide (2x-cutoff) cell list so the C++ builder can
    # resolve every P(b-a) lookup; under pair-resolved SYM3b they live
    # on the masked pair domain. Collapsing the total onto the radial
    # cutoff list (the old _combine_density_sets template) silently
    # zeroed P(h) for |h| in (cutoff, 2x cutoff] in the pure-functional
    # J-only contraction below. With the plain builder that loss is
    # masked -- the cell-level Schwarz guard confines the J traversal's
    # ket displacements to the internal 1x list anyway -- but it goes
    # live whenever the internal ball is wider than the radial list:
    # the M4a sr_image_extent padded build (He/STO-3G a=4 pad-8 loses
    # 2.5e-3 Ha of J_SR vs the RHF build on the same density) and
    # Schwarz-off builds. RHF passes its wide density straight through;
    # the unrestricted total must carry the same support. Where the
    # templates coincide (legacy gauge, non-split) this is
    # block-identical to the old combine.
    if [_cell_key(c) for c in density_alpha.cells] != [
        _cell_key(c) for c in density_beta.cells
    ]:
        raise ValueError(
            "build_bipole_unrestricted_fock: spin density templates "
            "disagree."
        )
    density_total = make_lattice_matrix_set(
        int(basis.nbasis),
        list(density_alpha.cells),
        [
            np.asarray(a, dtype=float) + np.asarray(b, dtype=float)
            for a, b in zip(density_alpha.blocks, density_beta.blocks)
        ],
    )
    if use_ewald_j_split:
        def _jk_sr_build(d):
            if _pair_mode:
                d = _pair_resolved_density_for_sr(
                    basis, _fock_sym_map, d
                )
            if _rep_cell_indices is not None:
                from .bipole_symmetry_fock import build_jk_reduced_symmetrized

                return build_jk_reduced_symmetrized(
                    basis,
                    system,
                    lat_opts_2e,
                    d,
                    float(omega_used),
                    _fock_sym_map,
                    _rep_cell_indices,
                    compute_exchange=need_k,
                    internal_extent_bohr=ctx.sr_image_extent,
                    workspace=_sr_workspace_for(ctx),
                )
            if _pair_mode:
                from .bipole_symmetry_fock import build_jk_pair_resolved

                return build_jk_pair_resolved(
                    basis, system, lat_opts_2e, d,
                    float(omega_used), _fock_sym_map,
                    internal_extent_bohr=ctx.sr_image_extent,
                    workspace=_sr_workspace_for(ctx),
                )
            if ctx.sr_image_extent is not None:
                return _sr_image_padded_jk(
                    basis, system, lat_opts_2e, d,
                    float(omega_used), float(ctx.sr_image_extent),
                    exact_zone_bohr=ctx.exact_zone_bohr,
                    workspace=_sr_workspace_for(ctx),
                )
            return build_jk_2e_real_space(
                basis, system, lat_opts_2e, d, float(omega_used)
            )

        def _j_sr_only_build(d):
            from types import SimpleNamespace

            if _pair_mode:
                d = _pair_resolved_density_for_sr(
                    basis, _fock_sym_map, d
                )
            if _pair_mode:
                from .bipole_symmetry_fock import build_jk_pair_resolved

                jk_pr = build_jk_pair_resolved(
                    basis, system, lat_opts_2e, d,
                    float(omega_used), _fock_sym_map,
                    compute_exchange=False,
                    internal_extent_bohr=ctx.sr_image_extent,
                    workspace=_sr_workspace_for(ctx),
                )
                return SimpleNamespace(J=jk_pr.J, K=None)
            if ctx.sr_image_extent is not None:
                # M4b: padded J-only build (no wasted K).
                return _sr_image_padded_jk(
                    basis, system, lat_opts_2e, d,
                    float(omega_used), float(ctx.sr_image_extent),
                    compute_exchange=False,
                    exact_zone_bohr=ctx.exact_zone_bohr,
                    workspace=_sr_workspace_for(ctx),
                )
            return SimpleNamespace(
                J=build_fock_2e_real_space(
                    basis,
                    system,
                    lat_opts_2e,
                    d,
                    0.0,
                    float(omega_used),
                ),
                K=None,
            )

        if exchange_split_active and need_k:
            # Keep each spin accumulator independent; only share the actual
            # full-density or delta-density contraction within this build.
            _jk_sr_spin_build = _reuse_paired_spin_jk(_jk_sr_build)

            def _jk_sr_pair_build(first, second):
                if (_can_pair_sr(ctx) and ctx.sr_image_extent is not None
                        and not _same_lattice_density(first, second)):
                    return _sr_image_padded_jk(
                        basis, system, lat_opts_2e, first,
                        float(omega_used), float(ctx.sr_image_extent),
                        exact_zone_bohr=ctx.exact_zone_bohr,
                        workspace=_sr_workspace_for(ctx), companion_density=second,
                    )
                return _jk_sr_spin_build(first), _jk_sr_spin_build(second)

            if incremental_jk_alpha is not None and use_incremental:
                from types import SimpleNamespace
                from .bipole_fock_ewald import IncrementalJK

                if type(incremental_jk_alpha) is IncrementalJK and type(incremental_jk_beta) is IncrementalJK:
                    (_Ja, _Ka), (_Jb, _Kb) = IncrementalJK.build_pair(
                        incremental_jk_alpha, incremental_jk_beta,
                        density_alpha, density_beta, _jk_sr_pair_build,
                    )
                else:
                    # Keep the existing build callback contract for custom
                    # accumulators; their preparation/commit rules may differ.
                    _Ja, _Ka = incremental_jk_alpha.build(density_alpha, _jk_sr_spin_build)
                    _Jb, _Kb = incremental_jk_beta.build(density_beta, _jk_sr_spin_build)
                jk_alpha = SimpleNamespace(J=_Ja, K=_Ka)
                jk_beta = SimpleNamespace(J=_Jb, K=_Kb)
            else:
                jk_alpha, jk_beta = _jk_sr_pair_build(density_alpha, density_beta)
                if incremental_jk_alpha is not None and reseed_incremental:
                    # #116: re-sync both spin chains to this exact build
                    # (seed() copies, before the in-place J summation below).
                    incremental_jk_alpha.seed(density_alpha, jk_alpha.J, jk_alpha.K)
                    incremental_jk_beta.seed(density_beta, jk_beta.J, jk_beta.K)
            F_J_SR_lat = jk_alpha.J
            j_alpha_blocks = F_J_SR_lat.blocks
            j_beta_blocks = jk_beta.J.blocks
            for c in range(len(j_alpha_blocks)):
                F_J_SR_lat.set_block(
                    c,
                    np.asarray(j_alpha_blocks[c], dtype=float)
                    + np.asarray(j_beta_blocks[c], dtype=float),
                )
            K_alpha_blocks = [
                np.asarray(block, dtype=float).copy() for block in jk_alpha.K.blocks
            ]
            K_beta_blocks = [
                np.asarray(block, dtype=float).copy() for block in jk_beta.K.blocks
            ]
        else:
            # J-only builds: pure functionals in the corrected gauge
            # (incremental/reduced variants) and the legacy Ewald-J
            # gauge (per-spin full-range K added below when needed).
            if incremental_jk_total is not None and use_incremental:
                F_J_SR_lat, _ = incremental_jk_total.build(
                    density_total,
                    # SYM3b: reduced J-only contraction when active;
                    # else the J-only build (image-padded when
                    # sr_image_extent is set).
                    (lambda d: _jk_sr_build(d))
                    if (use_reduced_j_only and _rep_cell_indices is not None)
                    else _j_sr_only_build,
                )
            elif use_reduced_j_only and _rep_cell_indices is not None:
                _jk_total = _jk_sr_build(density_total)
                F_J_SR_lat = _jk_total.J
                if incremental_jk_total is not None and reseed_incremental:
                    incremental_jk_total.seed(
                        density_total, _jk_total.J, _jk_total.K
                    )
            else:
                _j_total = _j_sr_only_build(density_total)
                F_J_SR_lat = _j_total.J
                if incremental_jk_total is not None and reseed_incremental:
                    incremental_jk_total.seed(density_total, _j_total.J, None)
            if need_k or _mp_config.enabled:
                jk_alpha = build_jk_2e_real_space(
                    basis, system, lat_opts_2e, density_alpha, 0.0
                )
                jk_beta = build_jk_2e_real_space(
                    basis, system, lat_opts_2e, density_beta, 0.0
                )
                K_alpha_blocks = [
                    np.asarray(block, dtype=float).copy()
                    for block in jk_alpha.K.blocks
                ]
                K_beta_blocks = [
                    np.asarray(block, dtype=float).copy()
                    for block in jk_beta.K.blocks
                ]
            else:
                # Pure functional: skip the per-spin K traversals.
                K_alpha_blocks = [
                    np.zeros((basis.nbasis, basis.nbasis))
                    for _ in range(len(F_J_SR_lat.cells))
                ]
                K_beta_blocks = list(K_alpha_blocks)

        rho_hat_for_LR = None
        D_a_k_split: Optional[List[np.ndarray]] = None
        D_b_k_split: Optional[List[np.ndarray]] = None
        if exchange_split_active and n_k == 1:
            from .bipole_fock_ewald import compute_rho_hat_from_k_density

            D_gamma_total = _spin_density_k(density_alpha, 0)[0] + _spin_density_k(
                density_beta, 1
            )[0]
            rho_hat_for_LR = compute_rho_hat_from_k_density(
                [D_gamma_total.astype(complex)],
                [np.zeros(3)],
                [1.0],
                j_lr_cache,
            )
        elif exchange_split_active:
            from .bipole_fock_ewald import compute_rho_hat_from_k_density

            D_a_k_split = _spin_density_k(density_alpha, 0)
            D_b_k_split = _spin_density_k(density_beta, 1)
            rho_hat_for_LR = compute_rho_hat_from_k_density(
                [Da + Db for Da, Db in zip(D_a_k_split, D_b_k_split)],
                k_points,
                weights,
                j_lr_cache,
            )
        elif coeffs_alpha_for_rho is not None and coeffs_beta_for_rho is not None:
            from .bipole_fock_ewald import compute_rho_hat_from_k_density

            occ_a = occupations_alpha_per_k if smearing_temperature > 0.0 else None
            occ_b = occupations_beta_per_k if smearing_temperature > 0.0 else None
            D_a_k = unrestricted_density_from_coeffs(
                coeffs_alpha_for_rho, n_alpha, occ_a
            )
            D_b_k = unrestricted_density_from_coeffs(
                coeffs_beta_for_rho, n_beta, occ_b
            )
            D_total_k = [Da + Db for Da, Db in zip(D_a_k, D_b_k)]
            if _ir_mapping.size > 0:
                D_total_k_full = [D_total_k[int(idx)] for idx in _ir_mapping]
                rho_hat_for_LR = compute_rho_hat_from_k_density(
                    D_total_k_full,
                    k_points_full,
                    weights_full,
                    j_lr_cache,
                )
            else:
                rho_hat_for_LR = compute_rho_hat_from_k_density(
                    D_total_k,
                    k_points,
                    weights,
                    j_lr_cache,
                )

        from .bipole_fock_ewald import compute_J_long_range_real_space_blocks

        lr_density = (
            density_total if rho_hat_for_LR is not None
            else _density_on_cache_cells(density_total, j_lr_cache, system)
        )
        F_LR_blocks = compute_J_long_range_real_space_blocks(
            lr_density,
            basis,
            system,
            omega_used,
            precision=ewald_precision,
            cache=j_lr_cache,
            rho_hat=rho_hat_for_LR,
        )
        if lat_opts_2e.pair_complete_1e or lr_density is not density_total:
            F_LR_blocks = _reciprocal_blocks_on_output(
                F_LR_blocks, j_lr_cache, system, F_J_SR_lat.cells,
            )
        assert ewald_cell_volume is not None
        j_background_potential = (
            -np.pi
            * float(n_elec)
            / (float(omega_used) * float(omega_used) * float(ewald_cell_volume))
        )
        if _pair_mode:
            # The background is v_bg.S(k) on the SCF overlap template,
            # not on the wider pair-only J^LR extension (restricted path
            # above carries the full rationale).
            zero_s = np.zeros((basis.nbasis, basis.nbasis), dtype=float)
            s_radial = {
                _cell_key(cell): np.asarray(block, dtype=float)
                for cell, block in zip(S_lat.cells, S_lat.blocks)
            }
            s_blocks = {
                _cell_key(cell): s_radial.get(_cell_key(cell), zero_s)
                for cell in F_J_SR_lat.cells
            }
        else:
            s_blocks = {
                _cell_key(cell): np.asarray(block, dtype=float)
                for cell, block in zip(S_lat.cells, S_lat.blocks)
            }
        for c, cell in enumerate(F_J_SR_lat.cells):
            key = _cell_key(cell)
            F_LR_blocks[c] = F_LR_blocks[c] + j_background_potential * s_blocks.get(key, np.zeros_like(F_LR_blocks[c]))
        j_sr_blocks = [
            np.asarray(block, dtype=float).copy() for block in F_J_SR_lat.blocks
        ]
        if need_k:
            alpha_blocks = [
                j_sr + j_lr - exchange_scale * k_a
                for j_sr, j_lr, k_a in zip(j_sr_blocks, F_LR_blocks, K_alpha_blocks)
            ]
            beta_blocks = [
                j_sr + j_lr - exchange_scale * k_b
                for j_sr, j_lr, k_b in zip(j_sr_blocks, F_LR_blocks, K_beta_blocks)
            ]
        else:
            alpha_blocks = [
                j_sr + j_lr for j_sr, j_lr in zip(j_sr_blocks, F_LR_blocks)
            ]
            beta_blocks = [
                j_sr + j_lr for j_sr, j_lr in zip(j_sr_blocks, F_LR_blocks)
            ]
        f2e_alpha_real = F_J_SR_lat
        for c, block in enumerate(alpha_blocks):
            f2e_alpha_real.set_block(c, block)
        # On the J build's OWN template: _copy_lattice_with_blocks would
        # re-template onto the radial overlap list, which drops the
        # pair-resolved output cells (identical templates in legacy
        # mode, so this is a pure generalisation).
        from ._vibeqc_core import make_lattice_matrix_set as _make_lms_beta

        f2e_beta_real = _make_lms_beta(
            int(basis.nbasis),
            list(F_J_SR_lat.cells),
            [np.asarray(b, dtype=float) for b in beta_blocks],
        )

        e_j_multipole: Optional[float] = None
        e_j_bipolar_quartet: Optional[float] = None

        # ---- Dormant PDR 1988 Ch. II.4c quartet prototype (J + K) ----
        # Research-only, driver-unreachable branch. It rebuilds selected
        # short-range terms with prototype skip masks and adds a prototype
        # far-field J contribution. Its exchange treatment and periodic
        # quartet domain are not certified for production use.
        if (
            ctx.spherical_moment_buffer is not None
            and ctx.penetration_dispatch_j is not None
        ):
            from .bipole_bipolar_skip_mask import (
                dispatch_to_bipolar_skip_mask,
            )
            from ._vibeqc_core import (
                build_jk_2e_real_space_bipolar_dispatch,
            )

            n_sh = len(list(basis.shells()))
            n_c = len(F_J_SR_lat.cells)
            skip_mask = dispatch_to_bipolar_skip_mask(
                ctx.penetration_dispatch_j, n_c, n_sh,
            )

            # Rebuild J_SR with far-field quartets skipped (J only, no K).
            jk_near = build_jk_2e_real_space_bipolar_dispatch(
                basis,
                system,
                lat_opts_2e,
                density_total,
                skip_mask,
                float(omega_used) if use_ewald_j_split else 0.0,
                False,  # compute_exchange=False, K rebuilt below
            )

            # Compute far-field J from total density.
            if ctx.far_field_fock_kernel is not None:
                from .bipole_far_field_kernel import (
                    apply_far_field_fock_kernel,
                    apply_far_field_fock_kernel_with_symmetry,
                )
                density_dict: Dict[Tuple[int, int, int], np.ndarray] = {
                    (cell.index[0], cell.index[1], cell.index[2]): (
                        np.asarray(density_total.blocks[c], dtype=float)
                    )
                    for c, cell in enumerate(density_total.cells)
                }
                if ctx.symmetry_reconstruction_map is not None:
                    ff_result = apply_far_field_fock_kernel_with_symmetry(
                        ctx.far_field_fock_kernel,
                        ctx.symmetry_reconstruction_map,
                        density_dict,
                    )
                else:
                    ff_result = apply_far_field_fock_kernel(
                        ctx.far_field_fock_kernel, density_dict,
                    )
            else:
                from .bipole_contractor_native import (
                    compute_bipolar_coulomb_far_field_native,
                )
                density_dict = {
                    (cell.index[0], cell.index[1], cell.index[2]): (
                        np.asarray(density_total.blocks[c], dtype=float)
                    )
                    for c, cell in enumerate(density_total.cells)
                }
                ff_result = compute_bipolar_coulomb_far_field_native(
                    ctx.spherical_moment_buffer,
                    ctx.penetration_dispatch_j,
                    density_dict,
                    ewald_omega=omega_used if use_ewald_j_split else 0.0,
                    nbf=basis.nbasis,
                )
            e_j_bipolar_quartet = ff_result.e_coulomb_far

            # Combine J_SR = near-field exact + far-field multipole.
            j_sr_blocks = []
            for c in range(n_c):
                key = (
                    F_J_SR_lat.cells[c].index[0],
                    F_J_SR_lat.cells[c].index[1],
                    F_J_SR_lat.cells[c].index[2],
                )
                j_near = np.asarray(jk_near.J.blocks[c], dtype=float)
                j_ff = ff_result.fock_blocks.get(
                    key, np.zeros_like(j_near),
                )
                j_sr_blocks.append(j_near + j_ff)

            if ff_result.n_quartets > 0:
                plog.info(
                    f"  Dormant PDR 1988 quartet prototype (L_max="
                    f"{ctx.spherical_moment_buffer.L_max}): "
                    f"{ff_result.n_quartets} far quartets, "
                    f"E_J_bipolar = {e_j_bipolar_quartet:+.6f} Ha"
                )

            # Prototype exchange skip path retained only for low-level
            # diagnostics; the public drivers cannot activate it.
            if need_k:
                jk_alpha_ff = build_jk_2e_real_space_bipolar_dispatch(
                    basis,
                    system,
                    lat_opts_2e,
                    density_alpha,
                    skip_mask,
                    float(omega_used) if use_ewald_j_split else 0.0,
                    True,  # compute_exchange=True
                )
                jk_beta_ff = build_jk_2e_real_space_bipolar_dispatch(
                    basis,
                    system,
                    lat_opts_2e,
                    density_beta,
                    skip_mask,
                    float(omega_used) if use_ewald_j_split else 0.0,
                    True,
                )
                # Replace K blocks with near-field (far-field skipped).
                # Align cell lists: the bipolar dispatch returns blocks
                # on the J_SR cell list.
                K_alpha_blocks = [
                    np.asarray(block, dtype=float).copy()
                    for block in jk_alpha_ff.K.blocks
                ]
                K_beta_blocks = [
                    np.asarray(block, dtype=float).copy()
                    for block in jk_beta_ff.K.blocks
                ]

        # ---- Legacy G1 cell-level far-field (retired, compatibility) ------
        elif _mp_config.enabled:
            from .bipole_fock_multipole import apply_multipole_far_field

            mp_result = apply_multipole_far_field(
                density_total,
                basis,
                system,
                lat_opts_2e,
                _mp_config,
                J_SR_blocks=j_sr_blocks,
                K_blocks=None,
                F_LR_blocks=F_LR_blocks,
                exchange_scale=0.0,
            )
            for c in range(len(F_J_SR_lat.cells)):
                f2e_alpha_real.set_block(
                    c,
                    mp_result.f2e_blocks[c] - K_alpha_blocks[c],
                )
                f2e_beta_real.set_block(
                    c,
                    mp_result.f2e_blocks[c] - K_beta_blocks[c],
                )
            e_j_multipole = mp_result.e_j_multipole
            if mp_result.n_far_cells > 0:
                plog.info(
                    f"  BIPOLE multipole far-field (L_max={_mp_config.L_max}, "
                    f"R={_mp_config.R_bipole:.1f} bohr): "
                    f"{mp_result.n_far_cells}/{mp_result.n_total_cells} cells replaced, "
                    f"E_J_far = {e_j_multipole:+.6f} Ha"
                )

        e_j_short_range = 0.5 * _lattice_contract_blocks(
            density_total,
            F_J_SR_lat.cells,
            j_sr_blocks,
            operator_name="J_SR",
        )
        e_j_long_range = 0.5 * _lattice_contract_blocks(
            density_total,
            F_J_SR_lat.cells,
            F_LR_blocks,
            operator_name="J_LR",
        )
        # Exchange factor: F_s -= a_HF.K_s(D_s) (PER-SPIN densities; no
        # 1/2 -- that belongs to the closed-shell convention where K is
        # built from the doubled TOTAL density). The variational
        # E_2e = 1/2.S Tr[D_s F_s] then carries -1/2.a.S Tr[D_s K_s],
        # consistent with e_exchange below.
        e_exchange: Optional[float]
        if need_k:
            e_exchange = -0.5 * exchange_scale * (
                _lattice_contract_blocks(
                    density_alpha,
                    F_J_SR_lat.cells,
                    K_alpha_blocks,
                    operator_name="K_alpha",
                )
                + _lattice_contract_blocks(
                    density_beta,
                    F_J_SR_lat.cells,
                    K_beta_blocks,
                    operator_name="K_beta",
                )
            )
        else:
            e_exchange = None

        K_corr_alpha_per_k: Optional[List[np.ndarray]] = None
        K_corr_beta_per_k: Optional[List[np.ndarray]] = None
        e_2e_k_correction = 0.0
        # q+G=0 gauge share on its own -- see the restricted builder (#82).
        e_exchange_finite_size: Optional[float] = None
        if exchange_split_active and need_k and n_k == 1:
            from .bipole_fock_ewald import (
                compute_K_long_range_gamma,
                exchange_q0_gauge_constant,
            )

            S_gamma = np.real(np.asarray(S_k_list[0]))
            c_g0 = exchange_q0_gauge_constant(
                _xi_madelung, omega_used, ewald_cell_volume, 1
            )
            D_a_gamma = np.real(_spin_density_k(density_alpha, 0)[0])
            D_b_gamma = np.real(_spin_density_k(density_beta, 1)[0])
            K_corr_alpha_per_k = [
                exchange_scale
                * (
                    compute_K_long_range_gamma(j_lr_cache, D_a_gamma)
                    + c_g0 * (S_gamma @ D_a_gamma @ S_gamma)
                )
            ]
            K_corr_beta_per_k = [
                exchange_scale
                * (
                    compute_K_long_range_gamma(j_lr_cache, D_b_gamma)
                    + c_g0 * (S_gamma @ D_b_gamma @ S_gamma)
                )
            ]
            e_2e_k_correction = -0.5 * float(
                np.einsum("ij,ji->", D_a_gamma, K_corr_alpha_per_k[0]).real
                + np.einsum("ij,ji->", D_b_gamma, K_corr_beta_per_k[0]).real
            )
            e_exchange_finite_size = -0.5 * exchange_scale * float(
                _xi_madelung
            ) * float(
                np.einsum(
                    "ij,ji->", D_a_gamma, S_gamma @ D_a_gamma @ S_gamma
                ).real
                + np.einsum(
                    "ij,ji->", D_b_gamma, S_gamma @ D_b_gamma @ S_gamma
                ).real
            )
            e_exchange = (e_exchange or 0.0) + e_2e_k_correction
        elif exchange_split_active and need_k:
            from .bipole_fock_ewald import (
                compute_K_long_range_mesh,
                exchange_q0_gauge_constant,
            )

            assert x_lr_cache is not None
            assert D_a_k_split is not None and D_b_k_split is not None
            c_g0 = exchange_q0_gauge_constant(
                _xi_madelung, omega_used, ewald_cell_volume, n_k
            )
            K_LR_alpha, K_LR_beta = compute_K_long_range_mesh(
                x_lr_cache, k_points, weights, [D_a_k_split, D_b_k_split],
            )
            K_corr_alpha_per_k = []
            K_corr_beta_per_k = []
            e_exchange_finite_size = 0.0
            for k_idx in range(n_k):
                S_k_c = np.asarray(S_k_list[k_idx], dtype=complex)
                K_a = exchange_scale * (
                    K_LR_alpha[k_idx]
                    + c_g0 * (S_k_c @ D_a_k_split[k_idx] @ S_k_c)
                )
                K_b = exchange_scale * (
                    K_LR_beta[k_idx]
                    + c_g0 * (S_k_c @ D_b_k_split[k_idx] @ S_k_c)
                )
                K_corr_alpha_per_k.append(K_a)
                K_corr_beta_per_k.append(K_b)
                e_2e_k_correction += -0.5 * float(weights[k_idx]) * float(
                    np.einsum("ij,ji->", D_a_k_split[k_idx], K_a).real
                    + np.einsum("ij,ji->", D_b_k_split[k_idx], K_b).real
                )
                e_exchange_finite_size += -0.5 * float(
                    weights[k_idx]
                ) * exchange_scale * float(_xi_madelung) * float(
                    np.einsum(
                        "ij,ji->",
                        D_a_k_split[k_idx],
                        S_k_c @ D_a_k_split[k_idx] @ S_k_c,
                    ).real
                    + np.einsum(
                        "ij,ji->",
                        D_b_k_split[k_idx],
                        S_k_c @ D_b_k_split[k_idx] @ S_k_c,
                    ).real
                )
            e_exchange = (e_exchange or 0.0) + e_2e_k_correction
    else:
        F_J_lat = build_fock_2e_real_space(
            basis,
            system,
            lat_opts_2e,
            density_total,
            0.0,
            0.0,
        )
        jk_alpha = build_jk_2e_real_space(
            basis, system, lat_opts_2e, density_alpha, 0.0
        )
        jk_beta = build_jk_2e_real_space(
            basis, system, lat_opts_2e, density_beta, 0.0
        )
        j_blocks = [
            np.asarray(block, dtype=float).copy() for block in F_J_lat.blocks
        ]
        K_alpha_blocks = [
            np.asarray(block, dtype=float).copy() for block in jk_alpha.K.blocks
        ]
        K_beta_blocks = [
            np.asarray(block, dtype=float).copy() for block in jk_beta.K.blocks
        ]
        if need_k:
            # K_s is built from the per-spin density, so the unrestricted
            # Fock operator carries the full functional coefficient:
            # F_s = J[D_alpha + D_beta] - alpha_hf K[D_s].
            alpha_blocks = [
                j - exchange_scale * k
                for j, k in zip(j_blocks, K_alpha_blocks)
            ]
            beta_blocks = [
                j - exchange_scale * k
                for j, k in zip(j_blocks, K_beta_blocks)
            ]
        else:
            alpha_blocks = list(j_blocks)
            beta_blocks = list(j_blocks)
        f2e_alpha_real = F_J_lat
        for c, block in enumerate(alpha_blocks):
            f2e_alpha_real.set_block(c, block)
        # On the J build's OWN template, as in the Ewald-split branch
        # above: _copy_lattice_with_blocks re-templates onto the radial
        # overlap list, which under physical pair support is narrower
        # than the pair-complete two-electron list and silently dropped
        # the far exchange blocks from the beta operator only (#503:
        # F_beta differed from F_alpha by 2e-6 on a closed-shell 1-D
        # H2 chain, F_alpha matching the RKS Fock).
        f2e_beta_real = make_lattice_matrix_set(
            int(basis.nbasis), list(F_J_lat.cells), beta_blocks
        )
        e_j_short_range = 0.5 * _lattice_contract_blocks(
            density_total,
            F_J_lat.cells,
            j_blocks,
            operator_name="J",
        )
        e_j_long_range = None
        if need_k:
            e_exchange = -0.5 * exchange_scale * (
                _lattice_contract_blocks(
                    density_alpha,
                    F_J_lat.cells,
                    K_alpha_blocks,
                    operator_name="K_alpha",
                )
                + _lattice_contract_blocks(
                    density_beta,
                    F_J_lat.cells,
                    K_beta_blocks,
                    operator_name="K_beta",
                )
            )
        else:
            e_exchange = None
        e_j_multipole = None
        K_corr_alpha_per_k = None
        K_corr_beta_per_k = None
        e_2e_k_correction = 0.0
        e_exchange_finite_size = None

    if _fock_sym_map is not None:
        from .bipole_symmetry_fock import symmetrize_fock_blocks

        a_blocks = [np.asarray(block, dtype=float) for block in f2e_alpha_real.blocks]
        b_blocks = [np.asarray(block, dtype=float) for block in f2e_beta_real.blocks]
        symmetrize_fock_blocks(
            a_blocks, _fock_sym_map, cells=list(f2e_alpha_real.cells)
        )
        symmetrize_fock_blocks(
            b_blocks, _fock_sym_map, cells=list(f2e_beta_real.cells)
        )
        for c in range(len(f2e_alpha_real.cells)):
            f2e_alpha_real.set_block(c, a_blocks[c])
            f2e_beta_real.set_block(c, b_blocks[c])

    f_alpha_k_list: List[np.ndarray] = []
    f_beta_k_list: List[np.ndarray] = []
    if hcore_k_list is not None:
        F_a_2e_all = _bloch_sum_blocks_multi_k(
            f2e_alpha_real.blocks,
            f2e_alpha_real.cells,
            k_points,
        )
        F_b_2e_all = _bloch_sum_blocks_multi_k(
            f2e_beta_real.blocks,
            f2e_beta_real.cells,
            k_points,
        )
        for k_idx, k in enumerate(k_points):
            F_a = F_a_2e_all[k_idx] + np.asarray(hcore_k_list[k_idx], dtype=complex)
            F_b = F_b_2e_all[k_idx] + np.asarray(hcore_k_list[k_idx], dtype=complex)
            if K_corr_alpha_per_k is not None:
                F_a = F_a - K_corr_alpha_per_k[k_idx]
                F_b = F_b - K_corr_beta_per_k[k_idx]
            f_alpha_k_list.append(0.5 * (F_a + F_a.conj().T))
            f_beta_k_list.append(0.5 * (F_b + F_b.conj().T))

    exchange_components = {
        "full_nonzero": float(e_exchange or 0.0) - float(e_exchange_finite_size or 0.0),
        "full_zero": float(e_exchange_finite_size or 0.0),
        "erfc": 0.0,
    }
    if _chi_exchange:
        e_exchange = sum(exchange_components.values())
    return BipoleUnrestrictedFockBuild(
        e_hf_exchange_components=exchange_components,
        f2e_alpha_real=f2e_alpha_real,
        f2e_beta_real=f2e_beta_real,
        f_alpha_k_list=f_alpha_k_list,
        f_beta_k_list=f_beta_k_list,
        e_j_short_range=e_j_short_range,
        e_j_long_range=e_j_long_range,
        e_exchange=e_exchange,
        e_j_multipole=e_j_multipole,
        e_2e_k_correction=e_2e_k_correction,
        e_exchange_finite_size=e_exchange_finite_size,
        k_corr_alpha_per_k=K_corr_alpha_per_k,
        k_corr_beta_per_k=K_corr_beta_per_k,
    )
