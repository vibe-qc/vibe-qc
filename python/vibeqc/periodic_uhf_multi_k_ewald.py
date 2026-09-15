"""Phase 15b: multi-k periodic UHF SCF driver using the composed
EWALD_3D Coulomb dispatch.

Open-shell counterpart of :mod:`vibeqc.periodic_rhf_multi_k_ewald`.
Carries two density matrices ``D_a(g), D_b(g)`` per real-space cell
and constructs the per-spin 2e Fock at every k as

    F^{2e,s}(g)  =  J(D_total)(g)  -  K(D_s)(g)

with ``D_total = D_a + D_b`` (one-particle convention, no factor 2)
and full-range exchange. The Hartree J comes from the shared
analytic-FT helper
(:func:`vibeqc.periodic_fock_multi_k.ewald_3d_j_blocks`) -- the same
w-invariant reciprocal-space J the closed-shell multi-k Fock builder
uses, whose per-cell blocks Bloch-sum to the Γ ``build_j_ewald_3d`` J
at the (1,1,1) mesh. Per-spin K is extracted from the closed-shell
builder via

    K(D_s)  =  2 . (J_full(D_s)  -  F_full(D_s))

where ``J_full(D)`` and ``F_full(D) = J_full(D) - 1/2 K(D)`` are two
separate ``build_fock_2e_real_space`` calls at w = 0. Cost per
iteration: one analytic-FT J(D_total) contraction + 4 lattice-ERI
builds (2 J_full(D_s) + 2 F_full(D_s)), no FFT Poisson on the
default backend.

Bloch-sums each per-spin F^{2e,s}(g) to F^{2e,s}(k), adds Hcore(k),
diagonalises per spin per k. DIIS is per-spin (separate Pulay history
a vs b). Inverse-Bloch fold rebuilds D_a_real and D_b_real from the
per-k MOs each iteration.

Scope
-----

  - Multi-k periodic UHF with EWALD_3D. The Γ-only path (single
    [1,1,1] mesh) reproduces :func:`run_uhf_periodic_gamma_ewald3d`
    by construction (the multi-k builder reduces to the molecular-
    limit J convention at [1,1,1] mesh -- same caveat as the RHF
    case).
  - Closed-shell case (multiplicity = 1) reproduces the multi-k
    Ewald RHF energy to ~µHa.
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
    build_jk_2e_real_space,
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
from .periodic_fock_multi_k import (
    ewald_3d_j_blocks,
    make_ewald_3d_lattice_j_cache,
)
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
    _g0_block,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicUHFAccelerator,
)
from .periodic_k_density import density_matrices_per_k as _density_matrices_per_k
from .periodic_uhf_ewald import _spin_squared
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing.fermi_dirac import fermi_dirac_occupations_per_k

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
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute per-cell ``(F^{2e,a}(g), F^{2e,b}(g))`` blocks.

    One analytic-FT Hartree contraction + two fused lattice-ERI calls
    (one per spin):

      - J(D_total): analytic-FT Hartree per cell via
        :func:`vibeqc.periodic_fock_multi_k.ewald_3d_j_blocks`
        (w-invariant; Bloch-sums to the Γ J at the (1,1,1) mesh).
      - K(D_a), K(D_b): per-spin full-range exchange via the fused
        ``build_jk_2e_real_space`` (one lattice-ERI traversal per spin
        returns both J and K; only K is consumed here -- J comes from
        the analytic FT above).

    Then ``F^{2e,s}(g) = J_total(g) - K(D_s)(g)``.

    ``d_total_buffer`` is an optional reusable ``LatticeMatrixSet``
    container (same cells / nbf) whose blocks are overwritten with
    ``D_a + D_b``; ``None`` builds a fresh container from the overlap
    template (recomputing the overlap lattice integrals just to obtain
    a container -- the one-shot behaviour).
    """
    n_cells = len(D_alpha_real.cells)

    # Total density D_total = D_a + D_b. Build via the overlap template
    # and mutate the C++ storage in place via ``set_block``.
    D_total_real = (
        d_total_buffer
        if d_total_buffer is not None
        else compute_overlap_lattice(basis, system, lat_opts)
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

    # Per-spin exchange via the fused J+K builder: ONE lattice-ERI
    # traversal per spin returns both matrices. The previous
    # K(D_s) = 2.(J_full(D_s) - F_full(D_s)) reconstruction spent TWO
    # full traversals per spin on the same quartets (4 of the 6
    # lattice-ERI calls per Fock build -- the dominant cost of the
    # multi-k iteration per the 2026-05-25 profiling).
    # tests/test_periodic_uhf_multi_k_ewald.py pins K-equality of the
    # two routes at machine precision.
    jk_alpha = build_jk_2e_real_space(
        basis, system, lat_opts, D_alpha_real, 0.0,
    )
    jk_beta = build_jk_2e_real_space(
        basis, system, lat_opts, D_beta_real, 0.0,
    )

    F_alpha_blocks: List[np.ndarray] = []
    F_beta_blocks: List[np.ndarray] = []
    for g in range(n_cells):
        J_total = np.asarray(J_total_blocks[g], dtype=float)
        K_a = np.asarray(jk_alpha.K.blocks[g], dtype=float)
        K_b = np.asarray(jk_beta.K.blocks[g], dtype=float)
        F_alpha_blocks.append(J_total - K_a)
        F_beta_blocks.append(J_total - K_b)

    return F_alpha_blocks, F_beta_blocks


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

    # ---- Force EWALD_3D gauge (gauge consistency; handover F4 2026-06-01) ----
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
    def _integer_occupations(nbf: int, n_occ_each: int) -> List[np.ndarray]:
        occ_per_k = []
        for _ in range(n_k):
            occ = np.zeros(nbf, dtype=float)
            occ[:n_occ_each] = 1.0
            occ_per_k.append(occ)
        return occ_per_k

    def _occupations_per_spin(
        eps_spin_per_k: Sequence[np.ndarray],
        n_spin: int,
    ) -> Tuple[List[np.ndarray], float, float]:
        nbf = eps_spin_per_k[0].shape[0] if eps_spin_per_k else 0
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
        if smearing_T <= 0.0 or n_spin == 0:
            return _integer_occupations(nbf, n_spin), 0.0, 0.0
        occ, mu, entropy = fermi_dirac_occupations_per_k(
            eps_spin_per_k,
            weights,
            float(n_spin),
            smearing_T,
            spin_degeneracy=1.0,
        )
        return occ, float(mu), float(entropy)

    def _spin_density(C_per_k_local, occ_per_k):
        """Build a one-particle (no factor 2) real-space spin density."""
        return real_space_density_from_kpoints_fractional(
            C_per_k_local,
            occ_per_k,
            kmesh,
            cells,
        )

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
    j_cache = make_ewald_3d_lattice_j_cache(
        basis, system, D_alpha_real.cells, lattice_opts=lat_opts,
    )
    # Reusable D_total container: the LatticeMatrixSet structure (cells,
    # nbf) is fixed for the whole SCF, so build the container once from
    # the overlap template and overwrite every block per Fock build.
    # Rebuilding it per iteration recomputed the overlap lattice
    # integrals just to obtain a container.
    d_total_buffer = compute_overlap_lattice(basis, system, lat_opts)
    if guess == InitialGuess.PATOM:
        plog.info("initial guess: PATOM (SAD + one periodic in-field step)")
        F_alpha_blocks, F_beta_blocks = _build_uhf_fock_blocks_ewald3d(
            basis,
            system,
            D_alpha_real,
            D_beta_real,
            omega,
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
        )
        C_alpha_per_k = []
        eps_alpha_per_k = []
        C_beta_per_k = []
        eps_beta_per_k = []
        for k_idx, k_cart in enumerate(k_points):
            F_a = _bloch_sum_blocks(F_alpha_blocks, cells, np.asarray(k_cart))
            F_b = _bloch_sum_blocks(F_beta_blocks, cells, np.asarray(k_cart))
            C_a, eps_a = _diag_in_orth_basis(
                F_a + Hcore_k_list[k_idx],
                X_k_list[k_idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                F_b + Hcore_k_list[k_idx],
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

        # 2e Fock blocks for both spins.
        F_alpha_blocks, F_beta_blocks = _build_uhf_fock_blocks_ewald3d(
            basis,
            system,
            D_alpha_used,
            D_beta_used,
            omega,
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
        )

        # F_s(k) = S_g e^{i k.R_g} F^{2e,s}(g) + Hcore(k) at every k in
        # one fused call (OpenMP-parallel over k) when the C++ kernel is
        # available; per-k Python fold otherwise.
        if _asm_mk is not None:
            F_alpha_k_list = [np.asarray(m) for m in _asm_mk(
                F_alpha_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
            F_beta_k_list = [np.asarray(m) for m in _asm_mk(
                F_beta_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
        else:
            F_alpha_k_list = [
                _bloch_sum_blocks(F_alpha_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]
            F_beta_k_list = [
                _bloch_sum_blocks(F_beta_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]

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
            F_alpha_check_blocks, F_beta_check_blocks = (
                _build_uhf_fock_blocks_ewald3d(
                    basis,
                    system,
                    D_alpha_real,
                    D_beta_real,
                    omega,
                    lat_opts,
                    grid_shape_t,
                    origin,
                    spacing_bohr,
                    j_cache=j_cache,
                    d_total_buffer=d_total_buffer,
                )
            )
            if _asm_mk is not None:
                F_alpha_check = [
                    np.asarray(m)
                    for m in _asm_mk(
                        F_alpha_check_blocks,
                        cell_r_list,
                        k_arr_list,
                        Hcore_k_list,
                    )
                ]
                F_beta_check = [
                    np.asarray(m)
                    for m in _asm_mk(
                        F_beta_check_blocks,
                        cell_r_list,
                        k_arr_list,
                        Hcore_k_list,
                    )
                ]
            else:
                F_alpha_check = [
                    _bloch_sum_blocks(F_alpha_check_blocks, cells, k) + H
                    for k, H in zip(k_arr_list, Hcore_k_list)
                ]
                F_beta_check = [
                    _bloch_sum_blocks(F_beta_check_blocks, cells, k) + H
                    for k, H in zip(k_arr_list, Hcore_k_list)
                ]

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
        F_alpha_blocks, F_beta_blocks = _build_uhf_fock_blocks_ewald3d(
            basis,
            system,
            D_alpha_real,
            D_beta_real,
            omega,
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
        )
        E_elec = 0.0
        if _asm_mk is not None:
            F_alpha_k_list = [np.asarray(m) for m in _asm_mk(
                F_alpha_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
            F_beta_k_list = [np.asarray(m) for m in _asm_mk(
                F_beta_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
        else:
            F_alpha_k_list = [
                _bloch_sum_blocks(F_alpha_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]
            F_beta_k_list = [
                _bloch_sum_blocks(F_beta_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]
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

    # <S^2> from the home-cell (Γ-block) MOs as a representative
    # diagnostic. Multi-k <S^2> is more involved; this is the standard
    # quick check used by PySCF.pbc / CRYSTAL.
    if n_alpha == 0 or n_beta == 0:
        s2 = 0.25 * (n_alpha - n_beta) * (n_alpha - n_beta + 2) + n_beta
    else:
        # Use k=0 if present, else first k.
        k0_idx = 0
        for i, k in enumerate(k_points):
            if np.allclose(np.asarray(k), 0.0):
                k0_idx = i
                break
        S_real = np.real(S_k_list[k0_idx])
        s2 = _spin_squared(
            n_alpha,
            n_beta,
            np.real(C_alpha_per_k[k0_idx]),
            np.real(C_beta_per_k[k0_idx]),
            S_real,
        )

    plog.converged(n_iter=iter_idx, energy=E_total, converged=converged)
    return PeriodicUHFMultiKEwaldResult(
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
