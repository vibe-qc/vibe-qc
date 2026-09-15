"""Multi-k restricted open-shell HF for periodic systems --- EWALD_3D.

Full Brillouin-zone k-point sampling for spin-pure restricted-open-shell
HF, the production sibling of :func:`run_rohf_periodic_gamma_ewald3d`.

The Hartree term uses the analytic-FT Ewald builder. Exact exchange uses
the finite-mesh Born-von-Karman Ewald split: direct erfc short range,
reciprocal ``q = k-k'`` long range, and the supercell probe-charge
``q -> 0`` correction. At each k the resulting per-spin Fock matrices are
combined into Roothaan's single effective Fock
(:func:`vibeqc.rohf.roothaan_effective_fock`, complex-Hermitian at non-Γ
k) and diagonalised once, with the standard open-shell occupation
(:func:`vibeqc.rohf._roothaan_occupations`).

Occupations follow the cell multiplicity and are equal at every k
(gapped-insulator setup, matching the multi-k UHF / RHF drivers --- no
smearing; metals need fractional spin occupations, a follow-up).

Verification status: the complex-Hermitian Roothaan coupling and occupation
are unit-checked (``tests/test_rohf.py`` plus the periodic tests). The
LiH+/STO-3G ``(3,1,1)`` production anchor agrees with an out-of-process
PySCF 2.13.1 KROHF/GDF reference to 0.063 mHa; see
``tests/test_periodic_rohf_multi_k_ewald.py``.
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
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
    build_jk_2e_real_space,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints_fractional,
)
from .ewald_j import auto_grid
from .guess import periodic_fock_guess_k
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
from .periodic_uhf_multi_k_ewald import _bloch_sum_blocks
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .rohf import (
    _commutator_error,
    _densities_from_occupations,
    _diis_extrapolate,
    _roothaan_occupations,
    roothaan_effective_fock,
)

__all__ = [
    "PeriodicROHFMultiKEwaldResult",
    "run_rohf_periodic_multi_k_ewald3d",
]


@dataclass
class PeriodicROHFMultiKEwaldResult:
    """Result of :func:`run_rohf_periodic_multi_k_ewald3d`.

    One spin-restricted orbital set per k (``mo_energies`` / ``mo_coeffs``
    are per-k lists), closed/open/virtual ``mo_occupations`` per k, and the
    converged per-spin real-space densities. Exposes the UHF-style
    ``*_alpha`` / ``*_beta`` per-k aliases (identical spatial orbitals).
    ``s_squared`` is exact for a restricted-open determinant.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float
    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    mo_occupations: List[np.ndarray]
    density_alpha: LatticeMatrixSet
    density_beta: LatticeMatrixSet
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    n_alpha: int
    n_beta: int
    scf_trace: List[SCFIteration] = field(default_factory=list)
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    sr_image_extent_bohr: Optional[float] = None
    method: str = "rohf"
    # Engine provenance for the public runner dispatch
    # (run_periodic_job(method="ROHF", jk_method="bipole")).
    runtime_backend: str = "bipole-rohf-multi-k-ewald"
    # Final total real-space density (D_alpha + D_beta) and the Cartesian
    # k-points/weights this result spans -- consumed by the runner's
    # Gamma-locating sidecar writers and the multi-k QVF payload.
    density: Optional[LatticeMatrixSet] = None
    kpoints_cart: Optional[np.ndarray] = None
    kpoint_weights: Optional[np.ndarray] = None

    @property
    def mo_energies_alpha(self) -> List[np.ndarray]:
        return self.mo_energies

    @property
    def mo_energies_beta(self) -> List[np.ndarray]:
        return self.mo_energies

    @property
    def mo_coeffs_alpha(self) -> List[np.ndarray]:
        return self.mo_coeffs

    @property
    def mo_coeffs_beta(self) -> List[np.ndarray]:
        return self.mo_coeffs

    @property
    def occupations(self) -> List[np.ndarray]:
        """Runner-facing alias: per-k 2/1/0 restricted occupations."""
        return self.mo_occupations

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None

    # Preserve the physical per-k state independently of lattice cutoff.
    density_alpha_k: Optional[List[np.ndarray]] = None
    density_beta_k: Optional[List[np.ndarray]] = None


def run_rohf_periodic_multi_k_ewald3d(
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
    sr_image_precision: Optional[float] = 1.0e-6,
    sr_image_extent_bohr: Optional[float] = None,
    progress: Union[bool, ProgressLogger, None] = None,
    initial_density_k: Optional[Sequence] = None,
    verbose: Optional[int] = None,
) -> PeriodicROHFMultiKEwaldResult:
    """Multi-k restricted-open-shell HF SCF with EWALD_3D Coulomb.

    Mirror of :func:`vibeqc.run_uhf_periodic_multi_k_ewald3d`; a/b counts
    come from ``system.multiplicity`` and are equal at every k. Returns a
    spin-pure determinant (``s_squared = S(S+1)`` exactly).
    """
    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_periodic_driver_guess
    guess = select_periodic_driver_guess(
        system, opts, route="ewald", method="ROHF",
        driver="run_rohf_periodic_multi_k_ewald3d", multi_k=True,
        restart_supplied=initial_density_k is not None,
    ).transport
    if initial_density_k is not None:
        if len(kmesh.kpoints) != int(np.prod(kmesh.mesh)):
            raise NotImplementedError("READ requires a complete full k mesh on the Ewald lattice adapter")
        if getattr(opts, "atomic_spins", None):
            raise ValueError("atomic_spins requires SAD without a restart density")
        if len(initial_density_k) != 2:
            raise ValueError("open-shell per-k READ requires alpha and beta block lists")
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    if float(getattr(opts, "fock_mixing", 0.0) or 0.0) != 0.0:
        raise NotImplementedError(
            "run_rohf_periodic_multi_k_ewald3d does not implement "
            "fock_mixing; use DIIS/damping or set it to zero"
        )
    if float(getattr(opts, "smearing_temperature", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            "run_rohf_periodic_multi_k_ewald3d does not implement "
            "finite-temperature occupations"
        )

    from .pbc_bipole_common import _expand_ibz_kmesh_for_ewald_j

    if np.asarray(getattr(kmesh, "ir_mapping", []), dtype=int).size > 0:
        kmesh = _expand_ibz_kmesh_for_ewald_j(system, kmesh, plog)

    if system.dim == 3 and lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        plog.info(
            "coulomb_method forced to EWALD_3D for gauge consistency "
            f"(was {lat_opts.coulomb_method!r})"
        )
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    # The Ewald split must use one alpha across J/K and the nuclear gauge.
    # Keep the historical keyword for API compatibility, but resolve the
    # state from PeriodicRHFOptions (explicit override) or the CRYSTAL
    # cell-volume default.  A caller-supplied keyword used to affect only J,
    # while K remained a divergent bare-Coulomb image sum.
    _user_omega = getattr(opts, "ewald_omega", None)
    if _user_omega is not None and float(_user_omega) > 0.0:
        omega = float(_user_omega)
    else:
        from .pbc_bipole_common import default_ewald_alpha

        v_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        # GitLab #651: CRYSTAL's 2.8/V^(1/3), bounded below so that
        # erfc(omega r)/r has decayed to opts.ewald_tolerance at this
        # route's fixed real-space image cutoff (lat_opts.cutoff_bohr).
        omega = default_ewald_alpha(
            v_cell,
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

    # ---- Occupations + k-points -----------------------------------------
    n_elec = int(system.n_electrons())
    mult = int(system.multiplicity)
    if mult < 1 or (n_elec + mult - 1) % 2 != 0:
        raise ValueError(
            f"run_rohf_periodic_multi_k_ewald3d: (n_electrons={n_elec}, "
            f"multiplicity={mult}) cannot be split into integer a/b counts."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2
    k_points = list(kmesh.kpoints)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = len(k_points)
    if n_k == 0:
        raise ValueError("kmesh has no k-points")
    if weights.shape != (n_k,) or not np.all(np.isfinite(weights)):
        raise ValueError(
            "kmesh.weights must be one finite value per stored k-point; "
            f"got shape={weights.shape} for n_k={n_k}"
        )
    if np.any(weights < 0.0):
        raise ValueError("kmesh.weights must be non-negative")
    mesh = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
    if int(np.prod(mesh)) != n_k:
        raise ValueError(
            "run_rohf_periodic_multi_k_ewald3d requires a complete "
            "Monkhorst-Pack mesh carrying its mesh dimensions; "
            f"got mesh={mesh} for {n_k} k-points"
        )
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError(f"kmesh.weights must sum to 1; got {weights.sum():.6f}")
    if n_k > 1 and not np.allclose(weights, 1.0 / float(n_k), atol=1.0e-9):
        raise ValueError(
            "run_rohf_periodic_multi_k_ewald3d requires uniform weights on "
            "the expanded full Monkhorst-Pack mesh"
        )
    if n_k == 1 and not np.allclose(
        np.asarray(k_points[0], dtype=float), 0.0, atol=1.0e-10
    ):
        raise NotImplementedError(
            "run_rohf_periodic_multi_k_ewald3d does not support a lone "
            "non-Gamma twist; use Gamma or a complete Monkhorst-Pack mesh"
        )


    if (n_k > 1 and guess == InitialGuess.READ) and initial_density_k is None:
        raise NotImplementedError(
            "run_rohf_periodic_multi_k_ewald3d: multi-k READ needs a full "
            "per-k spin-resolved restart, which this direct entry does not "
            "accept; use a non-READ guess"
        )
    plog.info(
        f"ROHF multi-k EWALD_3D / omega = {float(omega):.3f}, {n_k} k-points; "
        f"closed={n_beta}, open={n_alpha - n_beta}"
    )

    if auto_optimize_truncation and lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        from .eigs_preflight import optimize_truncation

        k_arr = [np.asarray(k, dtype=float) for k in k_points]
        opt_rep = optimize_truncation(
            system, basis, lattice_opts=lat_opts, k_points_cart=k_arr
        )
        lat_opts = opt_rep.optimized_lattice_opts

    # Resolve the padded erfc exchange domain only after automatic cutoff
    # optimization. The extent is defined relative to the final operator
    # cutoff; resolving it earlier can under-pad a grown domain.
    from .pbc_bipole_common import resolve_bipole_sr_image_extent

    sr_image_extent = resolve_bipole_sr_image_extent(
        basis,
        system,
        lat_opts,
        lat_opts,
        omega=float(omega),
        use_ewald_j_split=True,
        sr_image_precision=sr_image_precision,
        sr_image_extent_bohr=sr_image_extent_bohr,
        erfc_sr_build_active=True,
        plog=plog,
    )

    # ---- Real-space one-electron integrals + per-k S/Hcore/X ------------
    with plog.stage("integrals_lattice", detail=f"cutoff {lat_opts.cutoff_bohr:.2f}"):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    cells = list(S_lat.cells)

    from .linear_dependence import scf_preflight_overlap_check

    # Fused C++ multi-k Bloch transforms (OpenMP-parallel over k) when
    # available; the per-k Python fold is the fallback. Used for the
    # one-time S/T/V setup here and for the per-iteration Fock
    # transforms in the SCF loop below.
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

    if _bloch_mk is not None:
        S_k_all = [np.asarray(m) for m in
                   _bloch_mk(list(S_lat.blocks), cell_r_list, k_arr_list)]
        T_k_all = [np.asarray(m) for m in
                   _bloch_mk(list(T_lat.blocks), cell_r_list, k_arr_list)]
        V_k_all = [np.asarray(m) for m in
                   _bloch_mk(list(V_lat.blocks), cell_r_list, k_arr_list)]
    else:
        S_k_all = [_bloch_sum_blocks(list(S_lat.blocks), cells, k)
                   for k in k_arr_list]
        T_k_all = [_bloch_sum_blocks(list(T_lat.blocks), cells, k)
                   for k in k_arr_list]
        V_k_all = [_bloch_sum_blocks(list(V_lat.blocks), cells, k)
                   for k in k_arr_list]

    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    for k_idx, k in enumerate(k_points):
        S_k = S_k_all[k_idx]
        T_k = T_k_all[k_idx]
        V_k = V_k_all[k_idx]
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * ((T_k + V_k) + (T_k + V_k).conj().T)
        scf_preflight_overlap_check(S_k, plog=plog, label=f"S(k={k_idx})", basis=basis)
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k, linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if n_alpha > n_kept:
            raise RuntimeError(
                f"run_rohf_periodic_multi_k_ewald3d: orth kept {n_kept} "
                f"directions; need >= {n_alpha} at k={k_idx}."
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    # ---- Initial guess: Fock-mode diagonalisation per k ------------------
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
    C_per_k: List[np.ndarray] = []
    occ_per_k: List[np.ndarray] = []
    mo_energies_k: List[np.ndarray] = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        c, eps = _diag_in_orth_basis(F_guess_k, X_k)
        c = np.asarray(c, dtype=complex)
        eps_alpha = np.einsum(
            "pi,pq,qi->i", c.conj(), F_guess_k, c
        ).real
        occ = _roothaan_occupations(eps, eps_alpha, n_alpha, n_beta)
        C_per_k.append(c)
        occ_per_k.append(occ)
        mo_energies_k.append(np.asarray(eps, dtype=float))

    damping = float(opts.damping)
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    level_shift = float(getattr(opts, "level_shift", 0.0))
    j_cache = make_ewald_3d_lattice_j_cache(
        basis, system, cells, lattice_opts=lat_opts
    )

    # Exact exchange on a finite k mesh is a BvK-torus quantity.  The old
    # full-Coulomb real-space K sum contracted a mesh-periodic density over an
    # arbitrary radial image ball, so increasing cutoff repeated non-decaying
    # density blocks and drove the energy downward.  Use the convergent Ewald
    # split already validated by the BIPOLE drivers: K_SR(erfc), reciprocal
    # q=k-k' K_LR channels, and the supercell probe-charge correction.
    from .periodic_corrected_exchange import corrected_ewald_reciprocal_cutoff
    from .bipole_fock_ewald import (
        _build_j_long_range_cache,
        build_k_exchange_long_range_cache,
        compute_K_long_range_at_k,
        probe_charge_madelung_supercell,
        exchange_q0_gauge_constant,
    )

    volume = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    # GitLab #651: the reciprocal envelope must resolve exp(-K^2 / 4 omega^2)
    # for the omega actually used. CRYSTAL's volume-only K_max is calibrated
    # to CRYSTAL's volume-matched alpha and truncated this K_LR arm as soon as
    # omega was bounded above it (or set explicitly): H2 in a 30-bohr box at
    # omega 0.438 read -1.0790 Ha against the converged -1.1171 Ha.
    k_max = corrected_ewald_reciprocal_cutoff(volume, float(omega))
    cells_r_cart = np.asarray(
        [np.asarray(cell.r_cart, dtype=float) for cell in cells], dtype=float
    )
    exchange_j_cache = _build_j_long_range_cache(
        basis, system, cells_r_cart, omega, 1.0e-8, K_max=k_max
    )
    exchange_k_cache = build_k_exchange_long_range_cache(
        basis, system, exchange_j_cache, K_max=k_max
    )
    xi_madelung = probe_charge_madelung_supercell(system, mesh)
    exchange_g0 = exchange_q0_gauge_constant(xi_madelung, omega, volume, n_k)

    from .pbc_bipole_common import bvk_torus_density_matrices

    def _fold(occ_select) -> LatticeMatrixSet:
        occ_per_k_sel = [occ_select(o) for o in occ_per_k]
        return real_space_density_from_kpoints_fractional(
            C_per_k, occ_per_k_sel, kmesh, cells
        )

    D_alpha_real = _fold(lambda o: (o > 0.0).astype(float))
    D_beta_real = _fold(lambda o: (o == 2.0).astype(float))
    from .guess import initial_densities_open_shell

    guess_densities = None
    if not fock_guess_per_k and initial_density_k is None:
        guess_densities = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha,
            n_beta,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts,
            read_density_alpha=getattr(opts, "read_density_alpha", None),
            read_density_beta=getattr(opts, "read_density_beta", None),
            read_path=getattr(opts, "read_path", ""),
            overlap=S_k_list, weights=weights,
            atomic_spins=getattr(opts, "atomic_spins", None),
        )
    if fock_guess_per_k:
        plog.info(
            f"initial guess: {guess.name} "
            "(per-k one-particle Hamiltonian diagonalisation; "
            "restricted open-shell occupations)"
        )
        density_from_c_per_k = True
    elif guess_densities is not None:
        guess_alpha, guess_beta = guess_densities
        for g_idx, cell in enumerate(D_alpha_real.cells):
            is_home = bool(
                np.array_equal(np.asarray(cell.index, dtype=int), np.zeros(3, dtype=int))
            )
            D_alpha_real.set_block(
                g_idx,
                np.asarray(guess_alpha if is_home else np.zeros_like(guess_alpha)),
            )
            D_beta_real.set_block(
                g_idx,
                np.asarray(guess_beta if is_home else np.zeros_like(guess_beta)),
            )
        plog.info(f"initial guess: {guess.name} (g=0 density via GuessEngine)")
        density_from_c_per_k = False
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise at each k)")
        density_from_c_per_k = True
    if initial_density_k is not None:
        from .guess import periodic_restart_lattice_density, normalize_spin_density_k_guess
        initial_alpha_k, initial_beta_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k_list, weights, n_alpha, n_beta,
        )
        D_alpha_real = periodic_restart_lattice_density(
            initial_alpha_k, S_k_list, weights, n_alpha, kmesh, D_alpha_real.cells,
        )
        D_beta_real = periodic_restart_lattice_density(
            initial_beta_k, S_k_list, weights, n_beta, kmesh, D_beta_real.cells,
        )
        density_from_c_per_k = False
    D_alpha_prev: Optional[LatticeMatrixSet] = None
    D_beta_prev: Optional[LatticeMatrixSet] = None

    if guess == InitialGuess.PATOM:
        from .guess import patom_ewald_spin_step
        D_alpha_real, D_beta_real = patom_ewald_spin_step(
            basis, system, D_alpha_real, D_beta_real, Hcore_k_list, S_k_list,
            kmesh, n_alpha, n_beta, lat_opts, omega,
            lambda density: ewald_3d_j_blocks(
                basis, system, density, omega, lattice_opts=lat_opts,
                grid_shape=grid_shape_t, origin=origin,
                spacing_bohr=spacing_bohr, j_cache=j_cache,
            ),
            exchange_k_cache, exchange_g0, sr_image_extent=sr_image_extent,
        )
        D_alpha_prev = D_beta_prev = None
        density_from_c_per_k = False

    # Per-k DIIS histories for the effective Fock.
    fock_hist: List[List[np.ndarray]] = [[] for _ in range(n_k)]
    err_hist: List[List[np.ndarray]] = [[] for _ in range(n_k)]

    scf_trace: List[SCFIteration] = []
    E_prev = 0.0
    converged = False
    fock_eff_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    iter_idx = 0

    # Reusable D_total container: the LatticeMatrixSet structure (cells,
    # nbf) is fixed for the whole SCF, so build the container once from
    # the overlap template and overwrite every block each iteration.
    # Rebuilding it per iteration recomputed the overlap lattice
    # integrals just to obtain a container.
    D_total_used = compute_overlap_lattice(basis, system, lat_opts)

    plog.banner("SCF (ROHF multi-k, EWALD_3D)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    max_scf_iter = int(opts.max_iter)
    recheck_undamped = False
    terminal_consistency_rebuild = False
    for evaluation_idx in range(1, max_scf_iter + 2):
        # A density-mode Gamma guess cannot be represented by the Hcore MOs
        # returned before the first diagonalisation.  If max_iter == 1, take
        # one uncounted terminal evaluation of that first orbital state so
        # every result field still describes one evaluated density.
        consistency_rebuild = terminal_consistency_rebuild
        iter_idx = min(evaluation_idx, max_scf_iter)
        diis_active = (
            not consistency_rebuild
            and use_diis
            and iter_idx >= diis_start_iter
        )
        force_undamped = recheck_undamped
        recheck_undamped = False
        damping_applied = bool(
            iter_idx > 1
            and iter_idx < max_scf_iter
            and damping > 0.0
            and not diis_active
            and D_alpha_prev is not None
            and not force_undamped
        )
        if damping_applied:
            D_alpha_used = _damp_lattice_matrix(D_alpha_real, D_alpha_prev, damping)
            D_beta_used = _damp_lattice_matrix(D_beta_real, D_beta_prev, damping)
        else:
            D_alpha_used = D_alpha_real
            D_beta_used = D_beta_real
        density_used_from_c = density_from_c_per_k and not damping_applied

        for g_idx in range(len(cells)):
            D_total_used.set_block(
                g_idx,
                np.asarray(D_alpha_used.blocks[g_idx], dtype=float)
                + np.asarray(D_beta_used.blocks[g_idx], dtype=float),
            )
        J_blocks = ewald_3d_j_blocks(
            basis,
            system,
            D_total_used,
            omega,
            lattice_opts=lat_opts,
            grid_shape=grid_shape_t,
            origin=origin,
            spacing_bohr=spacing_bohr,
            j_cache=j_cache,
        )
        if sr_image_extent is not None:
            from .pbc_bipole_fock import _sr_image_padded_jk

            K_alpha_sr = _sr_image_padded_jk(
                basis,
                system,
                lat_opts,
                D_alpha_used,
                float(omega),
                float(sr_image_extent),
            ).K
            K_beta_sr = _sr_image_padded_jk(
                basis,
                system,
                lat_opts,
                D_beta_used,
                float(omega),
                float(sr_image_extent),
            ).K
        else:
            K_alpha_sr = build_jk_2e_real_space(
                basis, system, lat_opts, D_alpha_used, omega
            ).K
            K_beta_sr = build_jk_2e_real_space(
                basis, system, lat_opts, D_beta_used, omega
            ).K
        F_alpha_blocks = [
            np.asarray(j, dtype=float) - np.asarray(k, dtype=float)
            for j, k in zip(J_blocks, K_alpha_sr.blocks)
        ]
        F_beta_blocks = [
            np.asarray(j, dtype=float) - np.asarray(k, dtype=float)
            for j, k in zip(J_blocks, K_beta_sr.blocks)
        ]
        D_alpha_used_k = bvk_torus_density_matrices(
            D_alpha_used, k_points, mesh
        )
        D_beta_used_k = bvk_torus_density_matrices(
            D_beta_used, k_points, mesh
        )

        # F_s(k) = Σ_g e^{i k·R_g} F^{2e,s}(g) + Hcore(k) at every k in
        # one fused call (OpenMP-parallel over k) when the C++ kernel is
        # available; per-k Python fold otherwise.
        if _asm_mk is not None:
            F_a_k_all = [np.asarray(m) for m in _asm_mk(
                F_alpha_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
            F_b_k_all = [np.asarray(m) for m in _asm_mk(
                F_beta_blocks, cell_r_list, k_arr_list, Hcore_k_list)]
        else:
            F_a_k_all = [
                _bloch_sum_blocks(F_alpha_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]
            F_b_k_all = [
                _bloch_sum_blocks(F_beta_blocks, cells, k) + H
                for k, H in zip(k_arr_list, Hcore_k_list)
            ]

        E_elec = 0.0
        grad_norm_sum = 0.0
        density_fixed_point_residual = 0.0
        new_C: List[np.ndarray] = []
        new_occ: List[np.ndarray] = []
        new_eps: List[np.ndarray] = []
        for idx in range(n_k):
            k_arr = np.asarray(k_points[idx])
            F_a = F_a_k_all[idx]
            F_b = F_b_k_all[idx]
            S_k = S_k_list[idx]
            X_k = X_k_list[idx]
            K_alpha_lr = compute_K_long_range_at_k(
                exchange_k_cache,
                k_arr,
                k_points,
                weights,
                D_alpha_used_k,
            )
            K_beta_lr = compute_K_long_range_at_k(
                exchange_k_cache,
                k_arr,
                k_points,
                weights,
                D_beta_used_k,
            )
            F_a = F_a - K_alpha_lr - exchange_g0 * (
                S_k @ D_alpha_used_k[idx] @ S_k
            )
            F_b = F_b - K_beta_lr - exchange_g0 * (
                S_k @ D_beta_used_k[idx] @ S_k
            )
            # Contract the same density used to build this Fock. This matters
            # for density-mode guesses and pre-DIIS damping, where rebuilding
            # from the last orbital coefficients would mix two SCF states.
            d_a = np.asarray(D_alpha_used_k[idx], dtype=complex)
            d_b = np.asarray(D_beta_used_k[idx], dtype=complex)
            d_t = d_a + d_b
            w = float(weights[idx])
            E_elec += w * (
                0.5 * np.real(np.trace(d_t @ Hcore_k_list[idx]))
                + 0.5 * np.real(np.trace(d_a @ F_a))
                + 0.5 * np.real(np.trace(d_b @ F_b))
            )
            f_eff = roothaan_effective_fock(F_a, F_b, d_a, d_b, S_k)
            err = _commutator_error(f_eff, d_t, S_k, X_k)
            grad_norm_sum += w * float(np.linalg.norm(err))
            fock_eff_k_list[idx] = f_eff

            f_diag = f_eff
            if diis_active:
                fock_hist[idx].append(f_eff.copy())
                err_hist[idx].append(err.copy())
                if len(fock_hist[idx]) > int(opts.diis_subspace_size):
                    fock_hist[idx].pop(0)
                    err_hist[idx].pop(0)
                if len(fock_hist[idx]) > 1:
                    f_diag = _diis_extrapolate(fock_hist[idx], err_hist[idx])
            if level_shift > 0.0:
                f_diag = f_diag + level_shift * (S_k - S_k @ d_a @ S_k)

            c, eps = _diag_in_orth_basis(f_diag, X_k)
            c = np.asarray(c, dtype=complex)
            eps_alpha = np.einsum("pi,pq,qi->i", c.conj(), F_a, c).real
            occ = _roothaan_occupations(eps, eps_alpha, n_alpha, n_beta)
            new_C.append(c)
            new_eps.append(eps)
            new_occ.append(occ)

            d_a_next, d_b_next = _densities_from_occupations(c, occ)
            density_fixed_point_residual = max(
                density_fixed_point_residual,
                float(np.max(np.abs(d_a_next - d_a))),
                float(np.max(np.abs(d_b_next - d_b))),
            )

        _D_g0 = np.asarray(_g0_block(D_alpha_used)) + np.asarray(
            _g0_block(D_beta_used)
        )
        E_madelung_fix = _madelung_energy_correction_for_lat(
            _D_g0, np.asarray(_g0_block(S_lat)), system, lat_opts
        )
        E_total = float(E_elec) + e_nuc + E_madelung_fix

        if consistency_rebuild:
            # The visible max-iteration cycle consumed a density-mode guess;
            # replace its trace payload with the evaluated orbital state.
            # This rebuild is result finalisation, not a second SCF update.
            previous = scf_trace[-1]
            scf_trace[-1] = SCFIteration(
                iter=previous.iter,
                energy=float(E_total),
                delta_e=float(E_total - previous.energy),
                grad_norm=float(grad_norm_sum),
                diis_subspace=previous.diis_subspace,
            )
            mo_energies_k = [
                np.real(np.diag(c.conj().T @ f @ c))
                for c, f in zip(C_per_k, fock_eff_k_list)
            ]
            break

        dE = E_total - E_prev
        check_scf_divergence(
            "run_rohf_periodic_multi_k_ewald3d", iter_idx, E_total, grad_norm_sum, dE
        )
        diis_sub = max((len(h) for h in fock_hist), default=0) if diis_active else 0
        scf_trace.append(
            SCFIteration(
                iter=iter_idx, energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum), diis_subspace=diis_sub,
            )
        )
        plog.iteration(
            iter_idx, energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0), grad=float(grad_norm_sum),
            diis=diis_sub,
        )

        provisional_converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )
        converged = (
            provisional_converged
            and density_used_from_c
            and density_fixed_point_residual < float(opts.conv_tol_grad)
        )

        # A damped density is a line-search state, not the density carried by
        # the orbital/occupation state. Recheck the undamped density before
        # certifying convergence; the final allowed cycle is always undamped
        # so non-converged results also return the state that was evaluated.
        if provisional_converged and damping_applied:
            converged = False
            recheck_undamped = True

        # The energy, Fock, and commutator above were evaluated on the
        # current orbital density. Return that state, not the unevaluated
        # density produced by the look-ahead diagonalisation. At a certified
        # fixed point the look-ahead density is also explicitly within the
        # convergence tolerance.
        E_prev = E_total
        if converged or iter_idx == max_scf_iter:
            if not density_used_from_c:
                C_per_k = new_C
                occ_per_k = new_occ
                D_alpha_real = _fold(lambda o: (o > 0.0).astype(float))
                D_beta_real = _fold(lambda o: (o == 2.0).astype(float))
                density_from_c_per_k = True
                terminal_consistency_rebuild = True
                continue
            mo_energies_k = [
                np.real(np.diag(c.conj().T @ f @ c))
                for c, f in zip(C_per_k, fock_eff_k_list)
            ]
            break

        if recheck_undamped:
            continue

        # Commit the new MOs + occupations and rebuild the next density.
        C_per_k = new_C
        occ_per_k = new_occ
        mo_energies_k = new_eps
        D_alpha_prev, D_beta_prev = D_alpha_real, D_beta_real
        D_alpha_real = _fold(lambda o: (o > 0.0).astype(float))
        D_beta_real = _fold(lambda o: (o == 2.0).astype(float))
        density_from_c_per_k = True

    spin = 0.5 * (n_alpha - n_beta)
    s_squared = spin * (spin + 1.0)

    # Final total density for the output writers: overwrite the reusable
    # container with the converged (undamped) per-spin sum.
    for g_idx in range(len(cells)):
        D_total_used.set_block(
            g_idx,
            np.asarray(D_alpha_real.blocks[g_idx], dtype=float)
            + np.asarray(D_beta_real.blocks[g_idx], dtype=float),
        )

    from .periodic_k_density import density_matrices_per_k
    return PeriodicROHFMultiKEwaldResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=float(E_total),
        e_electronic=float(E_elec),
        e_nuclear=float(e_nuc),
        n_iter=iter_idx,
        converged=converged,
        s_squared=s_squared,
        s_squared_ideal=s_squared,
        mo_energies=mo_energies_k,
        mo_coeffs=C_per_k,
        mo_occupations=occ_per_k,
        density_alpha=D_alpha_real,
        density_alpha_k=density_matrices_per_k(C_per_k, [(o > 0.0).astype(float) for o in occ_per_k]),
        density_beta_k=density_matrices_per_k(C_per_k, [(o == 2.0).astype(float) for o in occ_per_k]),
        density_beta=D_beta_real,
        fock=fock_eff_k_list,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        n_alpha=n_alpha,
        n_beta=n_beta,
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
        sr_image_extent_bohr=sr_image_extent,
        density=D_total_used,
        kpoints_cart=np.asarray(
            [np.asarray(k, dtype=float).reshape(3) for k in k_points],
            dtype=float,
        ),
        kpoint_weights=np.asarray(weights, dtype=float).copy(),
    )
