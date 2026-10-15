"""Phase 15b: multi-k periodic UHF SCF driver using the composed
EWALD_3D Coulomb dispatch.

Open-shell counterpart of :mod:`vibeqc.periodic_rhf_multi_k_ewald`.
Carries two density matrices ``D_a(g), D_b(g)`` per real-space cell
and constructs the per-spin 2e Fock at every k as

    F^{2e,s}(k)  =  J(D_total)(k)  -  K(D_s)(k)

with ``D_total = D_a + D_b`` (one-particle convention, no factor 2)
and full-range exchange. The Hartree J comes from the shared
analytic-FT helper
(:func:`vibeqc.periodic_fock_multi_k.ewald_3d_j_blocks`) -- the same
w-invariant reciprocal-space J the closed-shell multi-k Fock builder
uses.

The exact exchange takes the same ``q -> 0`` gauge as
:func:`vibeqc.run_rhf_periodic_multi_k_ewald3d`, so a closed-shell
singlet reproduces it (GitLab #399):

  - On every mesh by default the full-range arm is the corrected
    Ewald split of :mod:`vibeqc.periodic_corrected_exchange`: erfc-screened
    K in real space plus the reciprocal and ``q + G = 0`` arms per spin.
    The SCF density on a finite mesh is Born-von-Karman-torus periodic,
    so D(g) does not decay and a bare ``1/r`` exchange sum over a radial
    image ball does not converge. Until #399 this driver summed exactly
    that: H2/STO-3G in a 6-bohr box at (2,1,1) fell from -2.26 to
    -6.22 Ha as the lattice cutoff went from 10 to 18 bohr, while RHF
    held -1.19450 Ha.
  - At a single Gamma point the corrected split is the default too
    (GitLab #23); ``exchange_exxdiv='none'`` zeroes the image densities and
    takes the molecular-limit kernel of
    :func:`run_uhf_periodic_gamma_ewald3d`.

Bloch-sums each per-spin F^{2e,s}(g) to F^{2e,s}(k), adds Hcore(k),
diagonalises per spin per k. DIIS is per-spin (separate Pulay history
a vs b). Inverse-Bloch fold rebuilds D_a_real and D_b_real from the
per-k MOs each iteration.

Scope
-----

  - Multi-k periodic UHF with EWALD_3D. A single [1,1,1] mesh takes the
    corrected split by default and the molecular-limit exchange of
    :func:`run_uhf_periodic_gamma_ewald3d` with ``exchange_exxdiv='none'``.
  - Closed-shell case (multiplicity = 1) reproduces the multi-k
    Ewald RHF energy on the same options and mesh.
  - Standard HF (no DFT). Finite-temperature smearing uses separate
    per-spin Fermi levels at fixed ``n_alpha`` / ``n_beta`` and
    converges the Mermin free energy. Gilat-Raubenheimer integration is
    available at T=0 on full or symmetry-reduced regular meshes.
"""

from __future__ import annotations


from .guess import periodic_result_selection

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
    level_shift_at_iter,
    build_jk_gamma_molecular_limit,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints_fractional,
)
from .ewald_j import auto_grid
from .guess import (
    initial_densities_open_shell,
    periodic_fock_guess_k,
)
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .periodic_fock_multi_k import (
    ewald_3d_j_blocks,
    make_ewald_3d_lattice_j_cache,
)
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
    _g0_block,
    _hermitize_k_matrices,
)
from .periodic.exchange_convention import (
    BARE_IMAGE_SUM,
    BVK_EWALD,
    MOLECULAR_LIMIT,
)
from .periodic_corrected_exchange import (
    CorrectedEwaldExchange,
    corrected_exchange_unavailable_reason,
    is_single_gamma_mesh,
    is_unsupported_gauge,
    resolve_exchange_gauge,
)
from .periodic_screened_exchange import (
    PeriodicExchangeAssembly,
    build_exchange_blocks,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicUHFAccelerator,
)
from .periodic_k_density import density_matrices_per_k as _density_matrices_per_k
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing import spin_channel_occupations as _spin_channel_occupations

__all__ = [
    "PeriodicUHFMultiKEwaldResult",
    "run_uhf_periodic_multi_k_ewald3d",
]


@dataclass
class PeriodicUHFMultiKEwaldResult:
    """Result of :func:`run_uhf_periodic_multi_k_ewald3d`.

    Per-cell quantities (``energy``, ``e_electronic``, ``e_nuclear``)
    plus per-k matrices for each spin (``mo_energies_alpha``,
    ``mo_coeffs_alpha``, ``fock_alpha``, ``mo_energies_beta``, ...)
    and the converged real-space densities ``density_alpha``,
    ``density_beta`` (each a :class:`LatticeMatrixSet`). Reports
    ``s_squared`` evaluated at the home-cell (Γ-block) MOs.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float

    # a spin
    mo_energies_alpha: List[np.ndarray]
    mo_coeffs_alpha: List[np.ndarray]
    fock_alpha: List[np.ndarray]
    density_alpha: LatticeMatrixSet

    # b spin
    mo_energies_beta: List[np.ndarray]
    mo_coeffs_beta: List[np.ndarray]
    fock_beta: List[np.ndarray]
    density_beta: LatticeMatrixSet

    # Per-k overlap + Hcore (shared between a/b).
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]

    scf_trace: List[SCFIteration] = field(default_factory=list)
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    smearing_temperature: float = 0.0
    fermi_level_alpha: float = 0.0
    fermi_level_beta: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations_alpha: List[np.ndarray] = field(default_factory=list)
    occupations_beta: List[np.ndarray] = field(default_factory=list)

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None

    # Preserve the physical per-k state independently of lattice cutoff.
    density_alpha_k: Optional[List[np.ndarray]] = None
    density_beta_k: Optional[List[np.ndarray]] = None

    # The exchange q=0 gauge that ran, as on the RHF sibling (GitLab #23).
    exchange_q0: str = BVK_EWALD


def _build_uhf_fock_blocks_ewald3d(
    basis: BasisSet,
    system: PeriodicSystem,
    D_alpha_real: LatticeMatrixSet,
    D_beta_real: LatticeMatrixSet,
    omega: float,
    lat_opts: LatticeSumOptions,
    grid_shape_t: Tuple[int, int, int],
    origin: Optional[Sequence[float]],
    spacing_bohr: float,
    j_cache=None,
    d_total_buffer: Optional[LatticeMatrixSet] = None,
    full_range_alpha: Optional[float] = None,
    symmetry_reduction=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute per-cell ``(F^{2e,a}(g), F^{2e,b}(g))`` blocks.

    One analytic-FT Hartree contraction + one lattice-ERI exchange build
    per spin:

      - J(D_total): analytic-FT Hartree per cell via
        :func:`vibeqc.periodic_fock_multi_k.ewald_3d_j_blocks`
        (w-invariant; Bloch-sums to the Γ J at the (1,1,1) mesh).
      - K(D_a), K(D_b): per-spin full-range exchange through the shared
        :func:`vibeqc.periodic_screened_exchange.build_exchange_blocks`.
        ``full_range_alpha`` puts it on the short-range ``erfc(alpha r)/r``
        arm of the corrected Ewald split, whose reciprocal and ``q + G = 0``
        arms the caller subtracts per k (#399); ``None`` keeps the bare
        ``1/r`` kernel, valid only where the density decays with image
        distance.

    Then ``F^{2e,s}(g) = J_total(g) - K(D_s)(g)``.

    ``d_total_buffer`` is an optional reusable ``LatticeMatrixSet``
    container (same cells / nbf) whose blocks are overwritten with
    ``D_a + D_b``; ``None`` builds a fresh container from the overlap
    template (recomputing the overlap lattice integrals just to obtain
    a container -- the one-shot behaviour).
    """
    from ._vibeqc_core import make_lattice_matrix_set

    n_cells = len(D_alpha_real.cells)

    # Total density D_total = D_a + D_b. Build via the overlap template
    # and mutate the C++ storage in place via ``set_block``.
    D_total_real = (
        d_total_buffer
        if d_total_buffer is not None
        else make_lattice_matrix_set(
            basis.nbasis, D_alpha_real.cells,
            [np.zeros((basis.nbasis, basis.nbasis)) for _ in range(n_cells)],
        )
    )
    for g in range(n_cells):
        D_total_real.set_block(
            g,
            np.asarray(D_alpha_real.blocks[g], dtype=float)
            + np.asarray(D_beta_real.blocks[g], dtype=float),
        )

    # J(D_total) per cell via the shared EWALD_3D J-blocks helper
    # (analytic-FT default -- w-invariant, Bloch-sums to the Γ J at the
    # (1,1,1) mesh; FFT-Poisson split behind VIBEQC_J_EWALD3D_BACKEND=grid).
    J_total_blocks = ewald_3d_j_blocks(
        basis,
        system,
        D_total_real,
        float(omega),
        lattice_opts=lat_opts,
        grid_shape=grid_shape_t,
        origin=origin,
        spacing_bohr=spacing_bohr,
        j_cache=j_cache,
    )

    # Per-spin exchange: one lattice-ERI traversal per spin through the
    # shared builder, which the UKS sibling and the closed-shell route
    # use too. tests/test_periodic_uhf_multi_k_ewald.py pins K-equality
    # with the two-traversal J_full - F_full reconstruction.
    K_alpha_blocks = build_exchange_blocks(
        basis, system, lat_opts, D_alpha_real, _HF_EXCHANGE,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
    )
    K_beta_blocks = build_exchange_blocks(
        basis, system, lat_opts, D_beta_real, _HF_EXCHANGE,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
    )

    F_alpha_blocks: List[np.ndarray] = []
    F_beta_blocks: List[np.ndarray] = []
    for g in range(n_cells):
        J_total = np.asarray(J_total_blocks[g], dtype=float)
        F_alpha_blocks.append(J_total - K_alpha_blocks[g])
        F_beta_blocks.append(J_total - K_beta_blocks[g])

    return F_alpha_blocks, F_beta_blocks


#: Pure Hartree-Fock exchange: the full-range arm only.
_HF_EXCHANGE = PeriodicExchangeAssembly(1.0, 0.0, 0.0)


def _single_gamma_molecular_limit_uhf_focks(
    basis: BasisSet,
    system: PeriodicSystem,
    D_alpha_real: LatticeMatrixSet,
    D_beta_real: LatticeMatrixSet,
    Hcore_gamma: np.ndarray,
    *,
    lattice_opts: LatticeSumOptions,
    j_build,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Single-Γ per-spin Fock matching :func:`run_uhf_periodic_gamma_ewald3d`.

    The open-shell twin of the RHF driver's
    ``_build_single_gamma_molecular_limit_fock``: the image-density blocks
    are zero (the driver localises every density it builds), so the bare
    exchange sum has exactly one term and IS the molecular-limit K. Same
    arithmetic as the Γ UHF driver, ``K(D_s) = K(2 D_s) / 2``, so the two
    agree.
    """
    D_a = np.asarray(_g0_block(D_alpha_real), dtype=float)
    D_b = np.asarray(_g0_block(D_beta_real), dtype=float)
    J = np.asarray(j_build(D_a + D_b))
    focks = []
    for D_s in (D_a, D_b):
        K_s = 0.5 * np.asarray(
            build_jk_gamma_molecular_limit(
                basis, system, lattice_opts, 2.0 * D_s, 0.0
            ).K
        )
        F = np.asarray(Hcore_gamma, dtype=complex) + J - K_s
        focks.append([0.5 * (F + F.conj().T)])
    return focks[0], focks[1]


def _localize_home_cell(D_real: LatticeMatrixSet, D_home: np.ndarray) -> LatticeMatrixSet:
    """Set ``D(g=0) = D_home`` and every image block to zero, in place."""
    home = np.asarray(np.real(D_home), dtype=float)
    home = 0.5 * (home + home.T)
    zero = np.zeros_like(home)
    for g_idx, cell in enumerate(D_real.cells):
        is_home = (np.asarray(cell.index, dtype=int) == 0).all()
        D_real.set_block(g_idx, home if is_home else zero)
    return D_real


def _bloch_sum_blocks(
    blocks: Sequence[np.ndarray],
    cells,
    k_cart: np.ndarray,
) -> np.ndarray:
    """F(k) = S_g e^{i k.R_g} F(g)."""
    k = np.asarray(k_cart, dtype=float).reshape(3)
    F_k = np.zeros_like(blocks[0], dtype=complex)
    for g_idx, block in enumerate(blocks):
        R_g = np.asarray(cells[g_idx].r_cart, dtype=float)
        phase = np.exp(1j * float(np.dot(k, R_g)))
        F_k = F_k + phase * block
    return F_k


def run_uhf_periodic_multi_k_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options=None,
    *,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    auto_optimize_truncation: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    initial_density_k: Optional[Sequence] = None,
    verbose: Optional[int] = None,
    bz_integration: Optional[str] = None,
    exchange_exxdiv: Optional[str] = None,
) -> PeriodicUHFMultiKEwaldResult:
    """Multi-k open-shell UHF SCF with EWALD_3D Coulomb.

    Mirror of :func:`run_rhf_periodic_multi_k_ewald3d` for open-shell
    systems. Same DIIS / damping / orthogonaliser conventions, but
    everything runs per-spin: two Pulay-DIIS instances, two
    diagonalisations per k, two density matrices folded back via
    inverse Bloch.

    Integer occupations follow the molecule's multiplicity (assumed
    equal at every k). Positive ``options.smearing_temperature`` enables
    per-spin Fermi-Dirac occupations with fixed ``n_alpha`` / ``n_beta``.
    ``bz_integration="gilat"`` instead applies sharp T=0
    Gilat-Raubenheimer occupations independently to each spin channel on a
    full or symmetry-reduced regular mesh.

    ``exchange_exxdiv`` selects the exact-exchange ``q -> 0`` gauge exactly
    as on :func:`vibeqc.run_rhf_periodic_multi_k_ewald3d`: ``None``
    (default) takes the corrected Ewald split at every mesh, a single Gamma
    point included (GitLab #23); ``'ewald'`` does the same and fails closed
    where the split is unavailable; ``'none'`` takes the molecular-limit
    kernel of :func:`run_uhf_periodic_gamma_ewald3d` and is accepted only at
    a single Gamma point. A closed-shell singlet therefore reproduces the
    RHF energy on the same mesh and options (GitLab #399).
    """
    from ._vibeqc_core import PeriodicRHFOptions as _Opts

    opts = options if options is not None else _Opts()
    from .guess import select_periodic_driver_guess
    guess = select_periodic_driver_guess(
        system, opts, route="ewald", method="UHF", driver="run_uhf_periodic_multi_k_ewald3d",
        multi_k=True, restart_supplied=initial_density_k is not None,
    ).transport
    if initial_density_k is not None:
        if len(kmesh.kpoints) != int(np.prod(kmesh.mesh)):
            raise NotImplementedError("READ requires a complete full k mesh on the Ewald lattice adapter")
        if getattr(opts, "atomic_spins", None):
            raise ValueError("atomic_spins requires SAD without a restart density")
        if len(initial_density_k) != 2:
            raise ValueError("open-shell per-k READ requires alpha and beta block lists")
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError(
            "run_uhf_periodic_multi_k_ewald3d: smearing_temperature must be >= 0"
        )
    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "run_uhf_periodic_multi_k_ewald3d: bz_integration must be None, "
            f"'smearing', or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "run_uhf_periodic_multi_k_ewald3d: bz_integration='gilat' is a "
            "T=0 integrator; do not combine it with smearing_temperature > 0"
        )
    if (getattr(opts, "initial_guess", None) == InitialGuess.READ) and initial_density_k is None:
        raise NotImplementedError(
            "periodic READ requires initial_density_k with a complete per-k "
            "density payload. Use run_periodic_job(read_from=...) to load "
            "a compatible result or archive; see docs/user_guide/initial_guess.md."
        )
    # SPINLOCK phases/MOM holding are not wired on the multi-k UHF Ewald
    # driver. The initial SAD spin seed is supported. Use multi-k UKS (PATTERN_HOLD)
    # or the Γ UHF/UKS Ewald drivers (both modes).
    from .spinlock_periodic import check_spinlock_support
    check_spinlock_support(opts, set(), "the multi-k UHF Ewald driver")
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)

    # ---- Force EWALD_3D gauge (gauge consistency; historical design record F4 2026-06-01) ----
    # This driver hard-codes the Hartree J to the Ewald-3D builder, so V_ne
    # (compute_nuclear_lattice_dispatch) and e_nuc (nuclear_repulsion_per_cell)
    # MUST share that gauge. Without the force, a default options object
    # (coulomb_method=DIRECT_TRUNCATED) makes nuclear_repulsion_per_cell return
    # the molecular 1/d sum and madelung_energy_correction_for_lat the bare-gauge
    # +a_M.Q_e^2/2L term; those only partially cancel (~0.74 mHa on H2/30-bohr),
    # so the SCF converged to a non-physical energy with no warning (CLAUDE.md
    # Sec.7). The RHF multi-k sibling already forces this (audit F1); extending it
    # here aligns e_nuclear with run_rhf_periodic_multi_k_ewald3d and zeroes the
    # now-redundant Madelung term (madelung_energy_correction_for_lat returns
    # 0.0 for EWALD_3D).
    # EWALD_3D V_ne (compute_nuclear_lattice_dispatch) is implemented only for
    # dim == 3 -- the 1D/2D Ewald variants raise (periodic_v_ne.py). So gate the
    # force on dim == 3; low-dim cells keep their DIRECT_TRUNCATED gauge (the
    # historical behaviour for these drivers on 1D/2D chains).
    if system.dim == 3 and lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        plog.info(
            "coulomb_method forced to EWALD_3D for gauge consistency "
            f"(was {lat_opts.coulomb_method!r}); this driver's Hartree J "
            "is Ewald-3D and V_ne / e_nuc must match"
        )
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    # w must match the nuclear Ewald a (auto-selected from
    # nuclear_cutoff_bohr in the C++ ewald engine) so that the
    # jellium background terms cancel exactly -- see the matching
    # block in run_rhf_periodic_gamma_ewald3d. User can override via
    # opts.ewald_omega (rarely needed). The driver kwarg ``omega`` is
    # retained for signature parity but is overridden here.
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if omega <= 0.0:
        _user_omega = getattr(opts, "ewald_omega", None)
        if _user_omega is not None and float(_user_omega) > 0.0:
            omega = float(_user_omega)
        else:
            from .pbc_bipole_common import default_ewald_alpha

            V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
            # GitLab #651: CRYSTAL's 2.8/V^(1/3), bounded below so that
            # erfc(omega r)/r has decayed to opts.ewald_tolerance at this
            # route's fixed real-space image cutoff (lat_opts.cutoff_bohr).
            omega = default_ewald_alpha(
                V_cell,
                real_cutoff_bohr=float(lat_opts.cutoff_bohr),
                tolerance=float(getattr(opts, "ewald_tolerance", 1e-12)),
            )

    lat = np.asarray(system.lattice, dtype=float)

    if grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)
    plog.info(
        f"UHF multi-k EWALD_3D / omega = {float(omega):.3f}, "
        f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
    )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    from .options_dump import dump_active_settings

    dump_active_settings(
        plog,
        [
            ("PeriodicRHFOptions", opts),
            ("LatticeSumOptions", lat_opts),
            (
                "Driver kwargs",
                {
                    "omega": float(omega),
                    "grid_shape": grid_shape_t,
                    "origin": origin,
                    "spacing_bohr": float(spacing_bohr),
                    "linear_dep_threshold": float(linear_dep_threshold),
                    "canonical_orth_normalize_diag_first": canonical_orth_normalize_diag_first,
                    "auto_optimize_truncation": auto_optimize_truncation,
                    "bz_integration": bz_integration,
                },
            ),
        ],
    )
    if plog.level >= 5:
        from .scf_log import format_basis_summary

        plog.write_raw(format_basis_summary(basis))

    # Open-shell occupations.
    n_elec = int(system.n_electrons())
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(
            f"run_uhf_periodic_multi_k_ewald3d: multiplicity must be >= 1, got {mult}"
        )
    if (n_elec + mult - 1) % 2 != 0 or (n_elec - mult + 1) % 2 != 0:
        raise ValueError(
            f"run_uhf_periodic_multi_k_ewald3d: (n_electrons={n_elec}, "
            f"multiplicity={mult}) cannot be split into integer a/b."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2

    k_points = list(kmesh.kpoints)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = len(k_points)
    if n_k == 0:
        raise ValueError("kmesh has no k-points")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"kmesh.weights must sum to 1; got {weights.sum():.6f}")
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}; "
        f"n_alpha = {n_alpha}, n_beta = {n_beta}"
    )
    single_gamma_mesh = is_single_gamma_mesh(kmesh)
    # ---- Exact-exchange q -> 0 gauge (#399, #23) --------------------------
    # The same rule as run_rhf_periodic_multi_k_ewald3d, shared through
    # resolve_exchange_gauge, so a closed-shell singlet reproduces it: the
    # corrected Ewald split at every mesh by default, a single Gamma point
    # included; exchange_exxdiv='none' names the molecular-limit kernel of
    # run_uhf_periodic_gamma_ewald3d at a single Gamma point.
    exchange_gauge = resolve_exchange_gauge(
        system,
        kmesh,
        exchange_exxdiv=exchange_exxdiv,
        slab_mode=False,
        pair_complete_1e=bool(lat_opts.pair_complete_1e),
        where="run_uhf_periodic_multi_k_ewald3d",
    )
    # This flag is the gauge, not the mesh: the density localisation, the
    # Gamma J closure and the molecular-limit Fock select together.
    single_gamma_kmesh = exchange_gauge.molecular_limit
    if single_gamma_mesh:
        plog.info(exchange_gauge.log_line)

    # ---- Auto-optimise lattice truncation (default ON) -------------------
    if auto_optimize_truncation and lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        from .eigs_preflight import (
            format_truncation_optimization_report,
            optimize_truncation,
        )

        k_arr = [np.asarray(k, dtype=float) for k in k_points]
        opt_rep = optimize_truncation(
            system,
            basis,
            lattice_opts=lat_opts,
            k_points_cart=k_arr,
        )
        if (
            opt_rep.n_evaluations > 1
            or opt_rep.optimized_lattice_opts.cutoff_bohr != lat_opts.cutoff_bohr
        ):
            plog.write_raw(format_truncation_optimization_report(opt_rep))
            if not opt_rep.converged:
                plog.warn("auto_optimize_truncation did not converge.")
            lat_opts = opt_rep.optimized_lattice_opts

    # ---- Real-space one-electron integrals -------------------------------
    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    from .lattice_screening import on_physical_eri_cells

    S_lat = on_physical_eri_cells(S_lat, basis, system, lat_opts)
    T_lat = on_physical_eri_cells(T_lat, basis, system, lat_opts)
    V_lat = on_physical_eri_cells(V_lat, basis, system, lat_opts)
    cells = list(S_lat.cells)

    # Fused C++ multi-k Bloch transforms (OpenMP-parallel over k) when
    # available; the per-k Python fold is the fallback. Consumed by the
    # per-iteration Fock and accelerator-density transforms in the SCF
    # loop below (same per-block accumulation order as
    # ``_bloch_sum_blocks``).
    cell_r_list = [np.asarray(c.r_cart, dtype=float) for c in cells]
    k_arr_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points]
    try:
        from ._vibeqc_core import (
            assemble_fock_multi_k as _asm_mk,
            bloch_sum_multi_k as _bloch_mk,
        )
    except ImportError:
        _asm_mk = None
        _bloch_mk = None

    # Per-k S(k), Hcore(k), orthogonaliser X(k).
    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    # Per-k linear-dependence preflight; see periodic_rhf_multi_k_ewald
    # for the rationale (Searle et al., ARCHER eCSE04-16, 2017).
    from .linear_dependence import scf_preflight_overlap_check

    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        scf_preflight_overlap_check(
            S_k,
            plog=plog,
            label=f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})",
            basis=basis,
        )
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k,
            linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if max(n_alpha, n_beta) > n_kept:
            raise RuntimeError(
                f"run_uhf_periodic_multi_k_ewald3d: orth dropped too many "
                f"directions (n_a={n_alpha}, n_b={n_beta}, "
                f"n_kept={n_kept}) at k = {k_arr}"
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    if (
        guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
        and getattr(opts, "atomic_spins", None)
    ):
        raise RuntimeError(
            "GuessEngine: atomic_spins (ATOMSPIN broken-symmetry seed) "
            f"requires the SAD guess; resolved guess is {guess.name}"
        )
    fock_guess_per_k = guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
    guess_fock_k = Hcore_k_list
    if fock_guess_per_k:
        guess_fock_k = list(
            periodic_fock_guess_k(
                system,
                basis,
                k_points,
                guess,
                lattice_opts=lat_opts,
                kinetic_lattice=T_lat,
                overlap_lattice=S_lat,
            )
        )

    # ---- Initial guess: diagonalize Hcore(k) or F_SAP(k) for both spins.
    C_alpha_per_k: List[np.ndarray] = []
    eps_alpha_per_k: List[np.ndarray] = []
    C_beta_per_k: List[np.ndarray] = []
    eps_beta_per_k: List[np.ndarray] = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        C_a, eps_a = _diag_in_orth_basis(F_guess_k, X_k)
        C_b, eps_b = _diag_in_orth_basis(F_guess_k, X_k)
        C_alpha_per_k.append(C_a.astype(complex))
        eps_alpha_per_k.append(eps_a)
        C_beta_per_k.append(C_b.astype(complex))
        eps_beta_per_k.append(eps_b)

    # Build real-space spin densities. UHF densities are per-spin
    # one-particle (no factor 2). The C++ fractional-occupation builder
    # handles both hard Aufbau occupations and finite-T occupations.
    def _occupations_per_spin(
        eps_spin_per_k: Sequence[np.ndarray],
        n_spin: int,
    ) -> Tuple[List[np.ndarray], float, float]:
        if use_gilat and n_spin > 0:
            from .bz_integration import gilat_occupations_for_kmesh

            occ, e_fermi = gilat_occupations_for_kmesh(
                system,
                kmesh,
                eps_spin_per_k,
                float(n_spin),
                spin_degeneracy=1.0,
            )
            return occ, float(e_fermi), 0.0
        # The shared open-shell fill: one chemical potential per spin over
        # the whole mesh, also at T = 0 (#176). The per-k integer Aufbau this
        # replaced filled the lowest n_spin bands at every k and reported
        # mu = 0.
        result = _spin_channel_occupations(
            eps_spin_per_k,
            weights=weights,
            n_electrons=float(n_spin),
            smearing=smearing_T,
        )
        return (
            [np.asarray(o, dtype=float) for o in result.occupations_per_k],
            float(result.mu),
            float(result.entropy),
        )

    def _spin_density(C_per_k_local, occ_per_k):
        """Build a one-particle (no factor 2) real-space spin density.

        At a single Gamma point in the molecular-limit gauge the inverse
        fold cannot recover any cell decay and would fill every image block
        with P(Gamma); the Gamma convention is P(g != 0) = 0 instead, as on
        the RHF driver (``_localize_single_gamma_density``).
        """
        D = real_space_density_from_kpoints_fractional(
            C_per_k_local,
            occ_per_k,
            kmesh,
            cells,
        )
        if single_gamma_kmesh:
            C0 = np.asarray(C_per_k_local[0], dtype=complex)
            occ0 = np.asarray(occ_per_k[0], dtype=float).reshape(-1)
            _localize_home_cell(D, (C0 * occ0[None, :]) @ C0.conj().T)
        return D

    occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
        eps_alpha_per_k, n_alpha
    )
    occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
        eps_beta_per_k, n_beta
    )
    entropy = entropy_alpha + entropy_beta
    D_alpha_real = _spin_density(C_alpha_per_k, occ_alpha_per_k)
    D_beta_real = _spin_density(C_beta_per_k, occ_beta_per_k)

    initial_alpha_k = None
    initial_beta_k = None

    # Density-mode guesses: overwrite the g=0 spin-density blocks with
    # GuessEngine output. This mirrors the multi-k UKS path and lets
    # ATOMSPIN build a broken-symmetry SAD density before the first Fock
    # build; HCORE still falls through to the per-k Hcore diagonalisation.
    if initial_density_k is not None:
        plog.info("initial guess: READ (complete per-k spin density)")
    elif fock_guess_per_k:
        # Common one-particle orbitals are occupied through the route's
        # existing per-spin global-k Aufbau/smearing policy.
        plog.info(
            f"initial guess: {guess.name} "
            "(one-particle Hamiltonian diagonalise at each k)"
        )
    else:
        seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
        split = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha,
            n_beta,
            seed_guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts,
            atomic_spins=getattr(opts, "atomic_spins", None) or None,
            overlap=S_k_list, weights=weights,
        )
        if split is not None:
            plog.info(
                f"initial guess: {guess.name} "
                "(spin-split density via GuessEngine)"
            )
            D_a_guess, D_b_guess = split
            initial_alpha_k = [np.asarray(D_a_guess, dtype=complex).copy() for _ in k_points]
            initial_beta_k = [np.asarray(D_b_guess, dtype=complex).copy() for _ in k_points]
            zero_block = np.zeros_like(D_a_guess, dtype=float)
            for g_idx in range(len(D_alpha_real.cells)):
                is_g0 = (
                    np.asarray(D_alpha_real.cells[g_idx].index, dtype=int)
                    == np.array([0, 0, 0], dtype=int)
                ).all()
                D_alpha_real.set_block(
                    g_idx, D_a_guess if is_g0 else zero_block
                )
                D_beta_real.set_block(
                    g_idx, D_b_guess if is_g0 else zero_block
                )
        else:
            plog.info(
                f"initial guess: {guess.name} "
                "(Hcore-diagonalise at each k, spin-degenerate)"
            )

    # Cache the iteration-invariant analytic-FT Hartree-J machinery once
    # (the 8b148bb5 E2 pattern). PATOM consumes it for the one in-field
    # re-polarisation step, and the SCF loop reuses it for every iteration.
    if single_gamma_kmesh:
        gamma_j_build = make_ewald_3d_gamma_j_builder(
            basis, system, omega=float(omega), lattice_opts=lat_opts,
            grid_shape=grid_shape_t, origin=origin, spacing_bohr=spacing_bohr,
        )
        j_cache = None
    else:
        gamma_j_build = None
        j_cache = make_ewald_3d_lattice_j_cache(
            basis, system, D_alpha_real.cells, lattice_opts=lat_opts,
        )
    # Reusable D_total container: the LatticeMatrixSet structure (cells,
    # nbf) is fixed for the whole SCF, so build the container once from
    # the overlap template and overwrite every block per Fock build.
    # Rebuilding it per iteration recomputed the overlap lattice
    # integrals just to obtain a container.
    from ._vibeqc_core import make_lattice_matrix_set

    d_total_buffer = make_lattice_matrix_set(
        basis.nbasis, cells,
        [np.zeros((basis.nbasis, basis.nbasis)) for _ in cells],
    )

    # ---- Corrected-Ewald full-range exchange (#399) -----------------------
    # Summed as a bare 1/r kernel over a radial image ball the exchange does
    # not converge, because the SCF density on a finite k mesh is BvK-torus
    # periodic and D(g) does not decay. Use the convergent Ewald split
    # instead (erfc K_SR in real space, reciprocal K_LR per q = k - k',
    # probe-charge q + G = 0 term), per spin, exactly as the RHF and UKS
    # siblings do; see vibeqc.periodic_corrected_exchange. Not applied at a
    # single Gamma point in the molecular-limit gauge, where every image
    # block is zero and the bare sum has exactly one term.
    corrected_exchange = None
    if not single_gamma_kmesh:
        _unavailable = corrected_exchange_unavailable_reason(
            system, kmesh, slab_mode=False
        )
        if _unavailable is not None and exchange_exxdiv == "ewald":
            raise NotImplementedError(
                "run_uhf_periodic_multi_k_ewald3d: exchange_exxdiv='ewald' "
                f"was requested, but {_unavailable}."
            )
        if _unavailable is None:
            corrected_exchange = CorrectedEwaldExchange.build(
                basis,
                system,
                np.asarray(
                    [np.asarray(c.r_cart, dtype=float) for c in cells]
                ),
                kmesh,
                float(omega),
                where="run_uhf_periodic_multi_k_ewald3d",
                lattice_opts=lat_opts,
            )
        elif is_unsupported_gauge(system, slab_mode=False):
            # dim != 3: the reciprocal cache is 3D-only. These cells keep
            # the historical kernel and say so, as on the RHF sibling.
            plog.info(
                "exact exchange stays on the uncorrected real-space image "
                f"sum: {_unavailable}. Energies on this route are not "
                "lattice-cutoff converged."
            )
        else:
            # An explicit k list with no star metadata: fail closed rather
            # than leave it on the bare kernel while a full mesh takes the
            # corrected split, as on the RHF sibling.
            raise NotImplementedError(
                "run_uhf_periodic_multi_k_ewald3d: multi-k Hartree-Fock "
                f"needs full-range exact exchange, but {_unavailable}. "
                "Use a Monkhorst-Pack mesh (full, or symmetry-reduced "
                "with vq.attach_symmetry + use_symmetry=True)."
            )

    # SYM3b orbit reduction for the real-space exchange arm, wedge path
    # only (the RHF driver's comment has the profile and the
    # symmetrized-answer rationale); here it applies per spin.
    exchange_symmetry_reduction = None
    if (
        corrected_exchange is not None
        and corrected_exchange.unfolding is not None
        and not lat_opts.pair_complete_1e
    ):
        from ._vibeqc_core import direct_lattice_cells as _radial_cells
        from .bipole_symmetry_fock import (
            cell_orbit_mapping,
            representative_cell_indices,
        )

        _sym_mapping = cell_orbit_mapping(
            system, basis, list(_radial_cells(system, lat_opts.cutoff_bohr))
        )
        if _sym_mapping is not None:
            exchange_symmetry_reduction = (
                _sym_mapping,
                representative_cell_indices(_sym_mapping),
            )

    def _uhf_focks(D_a_real, D_b_real, D_a_k, D_b_k):
        """Per-k ``(F_a(k), F_b(k))`` including Hcore(k), in the run's gauge.

        ``D_a_k`` / ``D_b_k`` are the per-k spin densities behind the
        real-space sets; the corrected split reads them directly for its
        reciprocal arms (and, on a symmetry-reduced mesh, to rebuild the
        real-space sets exactly from the unfolded wedge).
        """
        if single_gamma_kmesh:
            return _single_gamma_molecular_limit_uhf_focks(
                basis,
                system,
                D_a_real,
                D_b_real,
                Hcore_k_list[0],
                lattice_opts=lat_opts,
                j_build=gamma_j_build,
            )
        if (
            corrected_exchange is not None
            and corrected_exchange.unfolding is not None
        ):
            # Symmetry-reduced wedge: fold the unfolded wedge densities over
            # the full mesh (Pisani, Dovesi & Roetti 1988, Eqs. II.7.21 and
            # II.7.48) instead of the plain weighted wedge fold.
            D_a_real = corrected_exchange.fold_density(D_a_k, lat_opts)
            D_b_real = corrected_exchange.fold_density(D_b_k, lat_opts)
        F_a_blocks, F_b_blocks = _build_uhf_fock_blocks_ewald3d(
            basis,
            system,
            D_a_real,
            D_b_real,
            omega,
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
            full_range_alpha=(
                None if corrected_exchange is None else float(omega)
            ),
            symmetry_reduction=exchange_symmetry_reduction,
        )
        # F_s(k) = S_g e^{i k.R_g} F^{2e,s}(g) + Hcore(k) at every k in one
        # fused call (OpenMP-parallel over k) when the C++ kernel is
        # available; per-k Python fold otherwise.
        if _asm_mk is not None:
            F_a = [np.asarray(m) for m in _asm_mk(
                F_a_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
            F_b = [np.asarray(m) for m in _asm_mk(
                F_b_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
        else:
            F_a = [
                _bloch_sum_blocks(F_a_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]
            F_b = [
                _bloch_sum_blocks(F_b_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]
        if corrected_exchange is None:
            return F_a, F_b
        # The reciprocal and q + G = 0 arms of the split, per spin. The
        # coefficient is 1 on the one-particle spin density (the
        # closed-shell driver's 0.5 acts on D_total = 2 D_s).
        t_a = corrected_exchange.k_space_terms_all_k(S_k_list, list(D_a_k))
        t_b = corrected_exchange.k_space_terms_all_k(S_k_list, list(D_b_k))
        return (
            _hermitize_k_matrices([f - t for f, t in zip(F_a, t_a)]),
            _hermitize_k_matrices([f - t for f, t in zip(F_b, t_b)]),
        )

    if guess == InitialGuess.PATOM:
        plog.info("initial guess: PATOM (SAD + one periodic in-field step)")
        F_a_seed, F_b_seed = _uhf_focks(
            D_alpha_real,
            D_beta_real,
            initial_alpha_k
            if initial_alpha_k is not None
            else _density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k),
            initial_beta_k
            if initial_beta_k is not None
            else _density_matrices_per_k(C_beta_per_k, occ_beta_per_k),
        )
        C_alpha_per_k = []
        eps_alpha_per_k = []
        C_beta_per_k = []
        eps_beta_per_k = []
        for k_idx, k_cart in enumerate(k_points):
            C_a, eps_a = _diag_in_orth_basis(
                F_a_seed[k_idx],
                X_k_list[k_idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                F_b_seed[k_idx],
                X_k_list[k_idx],
            )
            C_alpha_per_k.append(C_a)
            eps_alpha_per_k.append(eps_a)
            C_beta_per_k.append(C_b)
            eps_beta_per_k.append(eps_b)
        occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
            eps_alpha_per_k, n_alpha
        )
        occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
            eps_beta_per_k, n_beta
        )
        entropy = entropy_alpha + entropy_beta
        D_alpha_real = _spin_density(C_alpha_per_k, occ_alpha_per_k)
        D_beta_real = _spin_density(C_beta_per_k, occ_beta_per_k)
    if guess == InitialGuess.PATOM:
        initial_alpha_k = None
        initial_beta_k = None

    if initial_density_k is not None:
        from .guess import periodic_restart_lattice_density, normalize_spin_density_k_guess
        initial_alpha_k, initial_beta_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k_list, weights, n_alpha, n_beta,
        )
        D_alpha_real = periodic_restart_lattice_density(
            initial_alpha_k, S_k_list, weights, n_alpha, kmesh, D_alpha_real.cells, retained_k_system=system,
        )
        D_beta_real = periodic_restart_lattice_density(
            initial_beta_k, S_k_list, weights, n_beta, kmesh, D_beta_real.cells, retained_k_system=system,
        )
        if single_gamma_kmesh:
            _localize_home_cell(D_alpha_real, initial_alpha_k[0])
            _localize_home_cell(D_beta_real, initial_beta_k[0])
    D_alpha_prev: Optional[LatticeMatrixSet] = None
    D_beta_prev: Optional[LatticeMatrixSet] = None
    D_a_used_per_k_prev = None
    D_b_used_per_k_prev = None

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_uhf_periodic_multi_k_ewald3d: damping must be in "
            f"[0, 1); got {damping}"
        )

    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[MultiKPeriodicUHFAccelerator] = (
        MultiKPeriodicUHFAccelerator(opts) if use_diis else None
    )

    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )

    level_shift = float(getattr(opts, "level_shift", 0.0))
    # Explicit per-iteration schedule (unified with the molecular
    # drivers). Empty ⇒ the constant `level_shift`; non-empty ⇒ resolved
    # per iteration by the shared C++ helper, applied per spin per k.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    _ls_max_iter = int(opts.max_iter)

    # Phase C1c -- quadratic SCF fallback (per-spin per-k Newton step).
    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # ---- SCF loop --------------------------------------------------------
    plog.banner("SCF (UHF multi-k, EWALD_3D)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    E_prev = 0.0
    F_alpha_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    F_beta_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    converged = False
    terminal_recheck_at_cap = False
    iter_idx = 0
    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter

        D_a_cur_per_k = (
            initial_alpha_k if iter_idx == 1 and initial_alpha_k is not None
            else _density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k)
        )
        D_b_cur_per_k = (
            initial_beta_k if iter_idx == 1 and initial_beta_k is not None
            else _density_matrices_per_k(C_beta_per_k, occ_beta_per_k)
        )

        # Density damping (per-spin) when DIIS not yet active.
        if iter_idx > 1 and damping > 0.0 and not diis_active:
            D_alpha_used = _damp_lattice_matrix(
                D_alpha_real,
                D_alpha_prev,
                damping,
            )
            D_beta_used = _damp_lattice_matrix(
                D_beta_real,
                D_beta_prev,
                damping,
            )
            D_a_used_per_k = [
                damping * prev + (1.0 - damping) * cur
                for cur, prev in zip(D_a_cur_per_k, D_a_used_per_k_prev)
            ]
            D_b_used_per_k = [
                damping * prev + (1.0 - damping) * cur
                for cur, prev in zip(D_b_cur_per_k, D_b_used_per_k_prev)
            ]
        else:
            D_alpha_used = D_alpha_real
            D_beta_used = D_beta_real
            D_a_used_per_k = D_a_cur_per_k
            D_b_used_per_k = D_b_cur_per_k

        # Per-spin Fock at every k, in the run's exchange gauge.
        F_alpha_k_list, F_beta_k_list = _uhf_focks(
            D_alpha_used, D_beta_used, D_a_used_per_k, D_b_used_per_k
        )

        # Per-cell electronic energy + per-k commutator errors.
        # E_elec = 1/2 tr[D_total . Hcore] + 1/2 tr[D_a . F_a] + 1/2 tr[D_b . F_b]
        E_elec = 0.0
        grad_norm_sum = 0.0
        error_alpha_k_list: List[np.ndarray] = []
        error_beta_k_list: List[np.ndarray] = []
        for idx in range(n_k):
            D_a_k = D_a_used_per_k[idx]
            D_b_k = D_b_used_per_k[idx]
            H_k = Hcore_k_list[idx]
            F_a_k = F_alpha_k_list[idx]
            F_b_k = F_beta_k_list[idx]
            w = float(weights[idx])
            E_elec += w * (
                0.5 * np.real(np.trace((D_a_k + D_b_k) @ H_k))
                + 0.5 * np.real(np.trace(D_a_k @ F_a_k))
                + 0.5 * np.real(np.trace(D_b_k @ F_b_k))
            )
            S_k = S_k_list[idx]
            FDS_a = F_a_k @ D_a_k @ S_k
            FDS_b = F_b_k @ D_b_k @ S_k
            err_a = FDS_a - FDS_a.conj().T
            err_b = FDS_b - FDS_b.conj().T
            error_alpha_k_list.append(err_a)
            error_beta_k_list.append(err_b)
            grad_norm_sum += w * float(
                np.sqrt(np.linalg.norm(err_a) ** 2 + np.linalg.norm(err_b) ** 2)
            )
        # Madelung-leak correction (v0.6.1).
        _D_g0 = np.asarray(_g0_block(D_alpha_used)) + np.asarray(_g0_block(D_beta_used))
        _S_g0 = np.asarray(_g0_block(S_lat))
        E_madelung_fix = _madelung_energy_correction_for_lat(
            _D_g0, _S_g0, system, lat_opts
        )
        E_total = float(E_elec) + e_nuc + E_madelung_fix

        free_energy = E_total - smearing_T * entropy
        dE = free_energy - E_prev
        # Divergence detection (v0.6.2).
        check_scf_divergence(
            "run_uhf_periodic_multi_k_ewald3d",
            iter_idx,
            E_total,
            grad_norm_sum,
            dE,
        )
        diis_sub = accel.subspace_size if accel is not None else 0
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum),
                diis_subspace=diis_sub,
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm_sum),
            diis=diis_sub,
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )

        # Phase C1c gate. When the quadratic fallback is active, take
        # per-spin per-k Newton steps in MO space -- bypass DIIS and
        # level shift since the Newton step is its own update.
        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )

        new_C_alpha: List[np.ndarray] = []
        new_eps_alpha: List[np.ndarray] = []
        new_C_beta: List[np.ndarray] = []
        new_eps_beta: List[np.ndarray] = []

        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            for idx in range(n_k):
                C_a, eps_a = quadratic_step(
                    F_alpha_k_list[idx],
                    C_alpha_per_k[idx],
                    eps_alpha_per_k[idx],
                    n_alpha,
                    shift=quadratic_fallback_shift,
                    max_step=quadratic_fallback_max_step,
                )
                C_b, eps_b = quadratic_step(
                    F_beta_k_list[idx],
                    C_beta_per_k[idx],
                    eps_beta_per_k[idx],
                    n_beta,
                    shift=quadratic_fallback_shift,
                    max_step=quadratic_fallback_max_step,
                )
                new_C_alpha.append(C_a)
                new_eps_alpha.append(eps_a)
                new_C_beta.append(C_b)
                new_eps_beta.append(eps_b)
        else:
            # SCF-accelerator extrapolation. DIIS runs per-spin
            # per-k Pulay internally; KDIIS / EDIIS / ADIIS use a
            # single spin-coupled history (one coefficient set
            # applied to both spins). For the bridged paths
            # (EDIIS / ADIIS / EDIIS_DIIS) we Bloch-sum the per-cell
            # D_a / D_b densities to per-k for input.
            if accel is not None:
                # Use the same per-k density as the Fock and energy; a
                # Bloch sum over every cutoff cell overcounts BvK residues.
                density_alpha_k_list = D_a_used_per_k
                density_beta_k_list = D_b_used_per_k
                F_a_ex, F_b_ex = accel.extrapolate_uhf(
                    F_alpha_k_list,
                    F_beta_k_list,
                    error_alpha_k_list=error_alpha_k_list,
                    error_beta_k_list=error_beta_k_list,
                    density_alpha_k_list=density_alpha_k_list,
                    density_beta_k_list=density_beta_k_list,
                    energy=free_energy,
                    mo_coeffs_alpha_k_list=C_alpha_per_k,
                    mo_coeffs_beta_k_list=C_beta_per_k,
                    n_alpha=n_alpha,
                    n_beta=n_beta,
                    weights=list(weights),
                    cells=cells,
                    kpoints=list(k_points),
                )
                if diis_active:
                    F_alpha_k_list = F_a_ex
                    F_beta_k_list = F_b_ex

            # Optional Saunders-Hillier level shift per spin per k.
            b = (
                level_shift_at_iter(
                    level_shift, 0, _ls_schedule, _ls_max_iter, iter_idx
                )
                if _ls_schedule
                else level_shift
            )
            if b != 0.0:
                for idx in range(n_k):
                    S_k = S_k_list[idx]
                    D_a_k = D_a_used_per_k[idx]
                    D_b_k = D_b_used_per_k[idx]
                    F_alpha_k_list[idx] = (
                        F_alpha_k_list[idx]
                        + b * S_k
                        - b * (S_k @ D_a_k @ S_k)
                    )
                    F_beta_k_list[idx] = (
                        F_beta_k_list[idx]
                        + b * S_k
                        - b * (S_k @ D_b_k @ S_k)
                    )
                    F_alpha_k_list[idx] = 0.5 * (
                        F_alpha_k_list[idx] + F_alpha_k_list[idx].conj().T
                    )
                    F_beta_k_list[idx] = 0.5 * (
                        F_beta_k_list[idx] + F_beta_k_list[idx].conj().T
                    )

            for idx in range(n_k):
                C_a, eps_a = _diag_in_orth_basis(
                    F_alpha_k_list[idx],
                    X_k_list[idx],
                )
                C_b, eps_b = _diag_in_orth_basis(
                    F_beta_k_list[idx],
                    X_k_list[idx],
                )
                new_C_alpha.append(C_a)
                new_eps_alpha.append(eps_a)
                new_C_beta.append(C_b)
                new_eps_beta.append(eps_b)

        C_alpha_per_k = new_C_alpha
        eps_alpha_per_k = new_eps_alpha
        C_beta_per_k = new_C_beta
        eps_beta_per_k = new_eps_beta
        occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
            eps_alpha_per_k, n_alpha
        )
        occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
            eps_beta_per_k, n_beta
        )
        entropy = entropy_alpha + entropy_beta

        # Rebuild D_a, D_b real-space densities (one-particle conv).
        D_alpha_new = _spin_density(C_alpha_per_k, occ_alpha_per_k)
        D_beta_new = _spin_density(C_beta_per_k, occ_beta_per_k)
        D_alpha_prev = D_alpha_used
        D_beta_prev = D_beta_used
        D_a_used_per_k_prev = D_a_used_per_k
        D_b_used_per_k_prev = D_b_used_per_k
        D_alpha_real = D_alpha_new
        D_beta_real = D_beta_new

        if converged:
            # The provisional criteria describe D_*_used.  Build the
            # physical, unshifted Fock on the freshly diagonalised density
            # and take one look-ahead step before certifying a fixed point.
            # This catches damped energy plateaus and occupation-only motion
            # that the old commutator could not see.
            F_alpha_check, F_beta_check = _uhf_focks(
                D_alpha_real,
                D_beta_real,
                _density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k),
                _density_matrices_per_k(C_beta_per_k, occ_beta_per_k),
            )

            C_alpha_check: List[np.ndarray] = []
            eps_alpha_check: List[np.ndarray] = []
            C_beta_check: List[np.ndarray] = []
            eps_beta_check: List[np.ndarray] = []
            for idx in range(n_k):
                C_a, eps_a = _diag_in_orth_basis(
                    F_alpha_check[idx], X_k_list[idx]
                )
                C_b, eps_b = _diag_in_orth_basis(
                    F_beta_check[idx], X_k_list[idx]
                )
                C_alpha_check.append(C_a)
                eps_alpha_check.append(eps_a)
                C_beta_check.append(C_b)
                eps_beta_check.append(eps_b)
            occ_alpha_check, _, entropy_alpha_check = _occupations_per_spin(
                eps_alpha_check, n_alpha
            )
            occ_beta_check, _, entropy_beta_check = _occupations_per_spin(
                eps_beta_check, n_beta
            )
            D_alpha_input_per_k = [
                (C * occ) @ C.conj().T
                for C, occ in zip(C_alpha_per_k, occ_alpha_per_k)
            ]
            D_beta_input_per_k = [
                (C * occ) @ C.conj().T
                for C, occ in zip(C_beta_per_k, occ_beta_per_k)
            ]
            D_alpha_check_per_k = [
                (C * occ) @ C.conj().T
                for C, occ in zip(C_alpha_check, occ_alpha_check)
            ]
            D_beta_check_per_k = [
                (C * occ) @ C.conj().T
                for C, occ in zip(C_beta_check, occ_beta_check)
            ]
            density_fixed_point_residual = max(
                [
                    float(np.max(np.abs(new - old)))
                    for new, old in zip(
                        D_alpha_check_per_k, D_alpha_input_per_k
                    )
                ]
                + [
                    float(np.max(np.abs(new - old)))
                    for new, old in zip(
                        D_beta_check_per_k, D_beta_input_per_k
                    )
                ],
                default=0.0,
            )
            E_elec_check = 0.0
            for idx in range(n_k):
                D_a = D_alpha_check_per_k[idx]
                D_b = D_beta_check_per_k[idx]
                w = float(weights[idx])
                E_elec_check += w * (
                    0.5
                    * np.real(
                        np.trace((D_a + D_b) @ Hcore_k_list[idx])
                    )
                    + 0.5 * np.real(np.trace(D_a @ F_alpha_check[idx]))
                    + 0.5 * np.real(np.trace(D_b @ F_beta_check[idx]))
                )
            D_alpha_check_real = _spin_density(
                C_alpha_check, occ_alpha_check
            )
            D_beta_check_real = _spin_density(
                C_beta_check, occ_beta_check
            )
            D_g0_check = np.asarray(_g0_block(D_alpha_check_real)) + np.asarray(
                _g0_block(D_beta_check_real)
            )
            E_madelung_check = _madelung_energy_correction_for_lat(
                D_g0_check,
                _S_g0,
                system,
                lat_opts,
            )
            free_energy_check = (
                float(E_elec_check)
                + e_nuc
                + E_madelung_check
                - smearing_T * (entropy_alpha_check + entropy_beta_check)
            )
            converged = (
                abs(free_energy_check - free_energy)
                < float(opts.conv_tol_energy)
                and density_fixed_point_residual < float(opts.conv_tol_grad)
            )
            terminal_recheck_at_cap = (
                not converged and iter_idx >= int(opts.max_iter)
            )

        if damper is not None:
            damper.update(E_total)
        E_prev = free_energy
        if converged:
            break

    # ---- Final pass on converged D's --------------------------------------
    if converged or terminal_recheck_at_cap:
        F_alpha_k_list, F_beta_k_list = _uhf_focks(
            D_alpha_real,
            D_beta_real,
            _density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k),
            _density_matrices_per_k(C_beta_per_k, occ_beta_per_k),
        )
        E_elec = 0.0
        final_C_alpha: List[np.ndarray] = []
        final_C_beta: List[np.ndarray] = []
        final_eps_alpha: List[np.ndarray] = []
        final_eps_beta: List[np.ndarray] = []
        for idx in range(n_k):
            C_a, eps_a = _diag_in_orth_basis(
                F_alpha_k_list[idx],
                X_k_list[idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                F_beta_k_list[idx],
                X_k_list[idx],
            )
            final_C_alpha.append(C_a)
            final_C_beta.append(C_b)
            final_eps_alpha.append(eps_a)
            final_eps_beta.append(eps_b)
        occ_alpha_final, mu_alpha, entropy_alpha = _occupations_per_spin(
            final_eps_alpha, n_alpha
        )
        occ_beta_final, mu_beta, entropy_beta = _occupations_per_spin(
            final_eps_beta, n_beta
        )
        entropy = entropy_alpha + entropy_beta
        for idx in range(n_k):
            C_a = final_C_alpha[idx]
            C_b = final_C_beta[idx]
            D_a_k = (C_a * occ_alpha_final[idx]) @ C_a.conj().T
            D_b_k = (C_b * occ_beta_final[idx]) @ C_b.conj().T
            w = float(weights[idx])
            E_elec += w * (
                0.5 * np.real(np.trace((D_a_k + D_b_k) @ Hcore_k_list[idx]))
                + 0.5 * np.real(np.trace(D_a_k @ F_alpha_k_list[idx]))
                + 0.5 * np.real(np.trace(D_b_k @ F_beta_k_list[idx]))
            )
        C_alpha_per_k = final_C_alpha
        C_beta_per_k = final_C_beta
        eps_alpha_per_k = final_eps_alpha
        eps_beta_per_k = final_eps_beta
        occ_alpha_per_k = occ_alpha_final
        occ_beta_per_k = occ_beta_final
        # Keep the reported real-space densities on the same terminal state
        # as the canonical orbitals and occupations.  The final
        # diagonalisation generally changes both, especially for finite-T and
        # Gilat occupations, so the pre-terminal SCF density is not a valid
        # result payload here.
        D_alpha_real = _spin_density(C_alpha_per_k, occ_alpha_per_k)
        D_beta_real = _spin_density(C_beta_per_k, occ_beta_per_k)
        # Madelung-leak correction (v0.6.1).
        _D_g0_f = np.asarray(_g0_block(D_alpha_real)) + np.asarray(
            _g0_block(D_beta_real)
        )
        _S_g0_f = np.asarray(_g0_block(S_lat))
        E_madelung_fix_f = _madelung_energy_correction_for_lat(
            _D_g0_f, _S_g0_f, system, lat_opts
        )
        E_total = float(E_elec) + e_nuc + E_madelung_fix_f
        free_energy = E_total - smearing_T * entropy

    # <S^2> of the multi-k determinant: Sz(Sz+1) + n_b minus the BZ-weighted,
    # occupation-weighted a/b overlap, the formula every other multi-k
    # open-shell driver uses. The Gamma-block diagnostic this replaced took
    # np.real(C) of the k = 0 orbitals, which is only right when F(Gamma) is
    # real: the corrected exchange (#399) makes it complex-Hermitian, the
    # eigenvectors carry an arbitrary complex phase, and a closed-shell
    # singlet with identical a/b densities read <S^2> = 4.9e-6.
    from .periodic_k_gdf import _multi_k_s_squared

    s2 = _multi_k_s_squared(
        n_alpha,
        n_beta,
        C_alpha_per_k,
        C_beta_per_k,
        S_k_list,
        weights,
        occ_alpha_k=occ_alpha_per_k,
        occ_beta_k=occ_beta_per_k,
    )

    plog.converged(n_iter=iter_idx, energy=E_total, converged=converged)
    return PeriodicUHFMultiKEwaldResult(
        exchange_q0=(
            MOLECULAR_LIMIT if single_gamma_kmesh
            else BVK_EWALD if corrected_exchange is not None
            else BARE_IMAGE_SUM
        ),
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=E_total,
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        n_iter=iter_idx,
        converged=converged,
        s_squared=float(s2),
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        mo_energies_alpha=eps_alpha_per_k,
        mo_coeffs_alpha=C_alpha_per_k,
        fock_alpha=F_alpha_k_list,
        density_alpha=D_alpha_real,
        density_alpha_k=_density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k),
        density_beta_k=_density_matrices_per_k(C_beta_per_k, occ_beta_per_k),
        mo_energies_beta=eps_beta_per_k,
        mo_coeffs_beta=C_beta_per_k,
        fock_beta=F_beta_k_list,
        density_beta=D_beta_real,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
        smearing_temperature=float(smearing_T),
        fermi_level_alpha=float(mu_alpha),
        fermi_level_beta=float(mu_beta),
        entropy=float(entropy),
        free_energy=float(free_energy),
        occupations_alpha=[
            np.asarray(occ, dtype=float).copy() for occ in occ_alpha_per_k
        ],
        occupations_beta=[
            np.asarray(occ, dtype=float).copy() for occ in occ_beta_per_k
        ],
    )
