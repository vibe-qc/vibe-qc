"""Phase 15c-2: multi-k closed-shell RKS SCF driver with composed
EWALD_3D Coulomb dispatch.

Multi-k DFT counterpart of :func:`run_rks_periodic_gamma_ewald3d`
(Phase 15c-1) and structural sibling of
:func:`run_rhf_periodic_multi_k_ewald3d`. At every SCF iteration:

    F(k)  =  H_core(k)  +  Bloch_k[J_SR(g) + J_LR(g) - K_HF(g)/2]
                       +  Bloch_k[V_xc(g)]

where ``K_HF = c_full*K_full + c_sr*K_erfc(omega_screen)`` per the CAM
assembly of :func:`vibeqc.periodic_screened_exchange.resolve_periodic_exchange`
(pure DFT skips K; global hybrids keep the full-range fraction;
screened hybrids like hse06 ride the erfc arm; LR-heavy RSH fails
closed). The 2e Fock blocks come from
:func:`build_periodic_fock_ewald3d_k` (assembly-aware), and the XC
contribution comes from :func:`build_xc_periodic` on the periodic
Becke grid using the SCF's *real-space density* (the proper
LatticeMatrixSet built from k-resolved MOs via
:func:`real_space_density_from_kpoints`).

Density flow.  Multi-k SCF carries the density as a
:class:`LatticeMatrixSet` ``D_real`` whose blocks ``D(g)`` Bloch-
sum to the per-k density matrices. ``build_xc_periodic`` consumes
this directly -- no degenerate-LMS hack needed (that was only the
Γ-only driver's molecular-limit trick).

Energy formula. Same shape as the C++ RKS DIRECT_TRUNCATED driver:

    E_elec  =  E_xc  +  S_k w_k . Re tr(D(k).H_core(k))
                     +  1/2 S_k w_k . Re tr(D(k).F^{HF-part}(k))

where ``F^{HF-part}(k) = Bloch_k[J - (a/2).K]`` (the part of F that's
linear in D and contributes 1/2.tr to the energy; V_xc is reported
through E_xc rather than a trace).

Scope.

  * Multi-k closed-shell RKS (multiplicity = 1, even electrons).
  * Pulay DIIS on per-k Fock with k-weighted Frobenius inner
    product (mirrors the multi-k RHF Ewald driver).
  * Saunders-Hillier level shift via ``options.level_shift``.
  * Fermi-Dirac smearing via ``options.smearing_temperature``
    (matches the multi-k RHF Ewald driver's free-energy convergence
    formulation: A = E - T.S).
  * Periodic Becke partition selectable via
    ``options.use_periodic_becke``.

The UKS multi-k Ewald driver (Phase 15c-3) reuses the same Pulay
DIIS / smearing / level-shift machinery but with per-spin densities.
"""

from __future__ import annotations


from .guess import periodic_result_selection

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    Functional,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicSystem,
    SCFIteration,
    XCKind,
    bloch_sum,
    build_grid,
    build_xc_periodic,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    get_num_threads,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints,
    real_space_density_from_kpoints_fractional,
)
from .ewald_j import auto_grid
from .guess import (
    initial_density_closed_shell,
    periodic_fock_guess_k,
)
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .periodic_fock_multi_k import (
    build_periodic_fock_ewald3d_k,
    build_periodic_fock_slab_ewald2d_k,
    make_ewald_3d_lattice_j_cache,
    make_slab_ewald_2d_lattice_j_cache,
)
from .periodic_corrected_exchange import (
    CorrectedEwaldExchange,
    corrected_exchange_unavailable_reason,
)
from .periodic_screened_exchange import resolve_periodic_exchange
from .periodic_density_mixing import (
    AndersonMixer,
    BroydenMixer,
    KerkerPreconditioner,
    per_k_density_to_vector,
    vector_to_per_k_density,
)
from .periodic_grid import build_periodic_becke_grid
from .periodic_k_density import density_matrices_per_k as _density_matrices_per_k
from .periodic_k_density import real_space_density_from_per_k_density
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
    _g0_block,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicSCFAccelerator,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing import (
    closed_shell_periodic_occupations as _closed_shell_periodic_occupations,
    occupations_are_per_k_integer_aufbau as _occupations_are_per_k_integer_aufbau,
)

__all__ = [
    "PeriodicRKSMultiKEwaldResult",
    "run_rks_periodic_multi_k_ewald3d",
]


_BYTES_PER_DOUBLE = 8
_DEFAULT_LEGACY_XC_LIMIT_GIB = 16.0
_LEGACY_XC_MEMORY_FRACTION = 0.60
_TRUE_ENV = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV


def _gib(n_bytes: int) -> float:
    return float(n_bytes) / float(1024**3)


def _detected_memory_limit_bytes() -> Optional[int]:
    """Best-effort physical/cgroup memory limit for fail-fast guards."""
    candidates: List[int] = []

    # Linux cgroup v2 and v1. These are absent on macOS and many local
    # shells; silently ignore them there.
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
            if raw and raw != "max":
                value = int(raw)
                # cgroup v1 may report a sentinel near LONG_MAX when no
                # memory limit is active.
                if 0 < value < 1 << 60:
                    candidates.append(value)
        except (OSError, ValueError):
            pass

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            candidates.append(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass

    return min(candidates) if candidates else None


def _legacy_periodic_xc_limit_bytes() -> int:
    override = os.environ.get("VIBEQC_PERIODIC_XC_MAX_ESTIMATED_GB")
    if override:
        try:
            return int(float(override) * (1024**3))
        except ValueError:
            pass

    detected = _detected_memory_limit_bytes()
    default_limit = int(_DEFAULT_LEGACY_XC_LIMIT_GIB * (1024**3))
    if detected is None:
        return default_limit
    return int(_LEGACY_XC_MEMORY_FRACTION * float(detected))


def _legacy_periodic_xc_thread_count() -> int:
    try:
        return max(1, int(get_num_threads()))
    except Exception:
        pass

    raw = os.environ.get("OMP_NUM_THREADS", "").split(",", 1)[0].strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, int(os.cpu_count() or 1))


def _estimate_legacy_periodic_xc_cache_bytes(
    *,
    n_grid: int,
    nbf: int,
    n_cells: int,
    is_gga: bool,
    n_threads: Optional[int] = None,
) -> int:
    """Rough peak estimate for the current C++ periodic-XC footprint.

    ``build_xc_periodic`` currently stores AO values for every density
    cell. GGA stores value + three gradients per cell, then each OpenMP
    worker allocates several ``n_grid x nbf`` temporaries during the
    gradient-density contraction.
    """
    persistent_matrices_per_cell = 4 if is_gga else 1
    scratch_matrices_per_thread = 7 if is_gga else 1
    active_threads = min(
        max(
            1,
            int(
                n_threads
                if n_threads is not None
                else _legacy_periodic_xc_thread_count()
            ),
        ),
        max(1, int(n_cells)),
    )
    matrix_bytes = int(n_grid) * int(nbf) * _BYTES_PER_DOUBLE
    persistent = persistent_matrices_per_cell * (int(n_cells) + 1)
    scratch = scratch_matrices_per_thread * active_threads
    return matrix_bytes * (persistent + scratch)


def _guard_legacy_periodic_xc_memory(
    *,
    n_grid: int,
    nbf: int,
    n_cells: int,
    is_gga: bool,
    functional: str,
) -> None:
    if _env_flag("VIBEQC_ALLOW_LEGACY_PERIODIC_XC_OOM"):
        return
    estimate = _estimate_legacy_periodic_xc_cache_bytes(
        n_grid=n_grid,
        nbf=nbf,
        n_cells=n_cells,
        is_gga=is_gga,
    )
    n_threads = _legacy_periodic_xc_thread_count()
    limit = _legacy_periodic_xc_limit_bytes()
    if estimate <= limit:
        return
    raise MemoryError(
        "run_rks_periodic_multi_k_ewald3d: refusing legacy periodic-XC "
        "build before the OS kills the process. Estimated rough peak "
        f"AO/XC memory = {_gib(estimate):.1f} GiB "
        f"(functional={functional!r}, n_grid={n_grid}, nbf={nbf}, "
        f"n_cells={n_cells}, threads={n_threads}, GGA={is_gga}); "
        f"guard limit = "
        f"{_gib(limit):.1f} GiB. The legacy EWALD_3D/FFT-Poisson RKS "
        "path is not suitable for large tight ionic crystals. Use the "
        "GDF periodic route where available, coarsen opts.grid for a "
        "debug run, or set VIBEQC_PERIODIC_XC_MAX_ESTIMATED_GB / "
        "VIBEQC_ALLOW_LEGACY_PERIODIC_XC_OOM=1 to override this guard."
    )


@dataclass
class PeriodicRKSMultiKEwaldResult:
    """Result of :func:`run_rks_periodic_multi_k_ewald3d`.

    Per-cell scalars (energy, e_electronic, e_xc, e_coulomb,
    e_hf_exchange, e_nuclear) and per-k matrices (mo_energies,
    mo_coeffs, fock, overlap, hcore) alongside the converged real-
    space density ``D_real``. Layout mirrors the multi-k RHF Ewald
    result with KS-specific energy decomposition fields added.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_xc: float
    e_coulomb: float
    e_hf_exchange: float
    n_iter: int
    converged: bool

    # Per-k arrays -- lists of length ``len(kmesh.kpoints)``.
    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]

    # Real-space converged density.
    density: LatticeMatrixSet

    # Diagnostic trace.
    scf_trace: List[SCFIteration] = field(default_factory=list)

    functional: str = ""
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)

    # Smearing diagnostics (mirror multi-k RHF Ewald).
    smearing_temperature: float = 0.0
    fermi_level: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations: List[np.ndarray] = field(default_factory=list)
    # True when hybrid exact exchange ran on a symmetry-reduced
    # Monkhorst-Pack mesh via the Pisani/Dovesi star unfolding; the
    # runner turns this into the kpoint_symmetry_unfolding citation.
    used_kpoint_symmetry_unfolding: bool = False
    # Snapshot of the lattice support actually used after truncation
    # optimization. Analytic derivatives must use this effective support.
    effective_lattice_opts: Optional[LatticeSumOptions] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None

    # Preserve the physical per-k state independently of lattice cutoff.
    density_k: Optional[List[np.ndarray]] = None


def _bloch_sum_lms_at_k(
    lms: LatticeMatrixSet,
    k_cart: np.ndarray,
) -> np.ndarray:
    """Bloch-sum a LatticeMatrixSet at a given k-point: returns the
    complex (n_bf, n_bf) matrix S_g e^{i k.R_g} M(g). Used to fold
    the V_xc real-space LatticeMatrixSet (output of
    build_xc_periodic) into per-k Fock contributions."""
    return np.asarray(bloch_sum(lms, np.asarray(k_cart, dtype=float).reshape(3)))


# Density-space mixers selectable via the ``density_mixer`` kwarg. These are
# real-space density-matrix fixed-point mixers (the v0.10.x D4 metal-mixing
# program); when one is active it *replaces* Fock-space DIIS and linear density
# damping (they are alternative convergence strategies -- stacking them would
# hide, not cure, an oscillation, CLAUDE.md Sec.7).
_DENSITY_MIXERS = {"anderson": AndersonMixer, "broyden": BroydenMixer}


def _make_density_mixer(name, depth, beta):
    """Resolve a ``density_mixer`` selector to a mixer instance, or ``None``.

    ``None`` / ``"diis"`` keep the legacy Fock-DIIS + linear-damping path
    (returns ``None``); ``"anderson"`` / ``"broyden"`` return a fresh
    density-space mixer.
    """
    if name is None:
        return None
    key = str(name).strip().lower()
    if key == "diis":
        return None
    if key not in _DENSITY_MIXERS:
        raise ValueError(
            f"run_rks_periodic_multi_k_ewald3d: density_mixer={name!r} is not "
            f"recognised; expected one of None, 'diis', "
            f"{', '.join(repr(k) for k in sorted(_DENSITY_MIXERS))}."
        )
    return _DENSITY_MIXERS[key](depth=int(depth), beta=float(beta))


def run_rks_periodic_multi_k_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options: Optional[PeriodicKSOptions] = None,
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
    density_mixer: Optional[str] = None,
    density_mixer_depth: int = 8,
    density_mixer_beta: float = 0.5,
    density_mixer_kerker: bool = False,
    kerker_k0: float = 1.5,
    kerker_strength: float = 1.0,
    kerker_cutoff_ha: float = 120.0,
) -> PeriodicRKSMultiKEwaldResult:
    """Multi-k closed-shell periodic RKS SCF with EWALD_3D Coulomb.

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    kmesh
        :class:`BlochKMesh` (e.g. from :func:`vibeqc.monkhorst_pack`).
    options
        Optional :class:`PeriodicKSOptions`. Defaults: PBE, DIIS on,
        no level shift, molecular Becke partition. Honours
        ``functional``, ``grid``, ``use_periodic_becke``,
        ``becke_image_radius_bohr``, ``level_shift``,
        ``smearing_temperature``, ``damping``, ``max_iter``,
        ``conv_tol_*``, ``diis_*``, ``initial_guess``, ``lattice_opts``.
    omega, grid_shape, origin, spacing_bohr
        Ewald splitting + FFT-Poisson grid controls.
    linear_dep_threshold
        Per-k S(k) eigenvalue floor for canonical orthogonalisation.
    density_mixer
        Real-space density-matrix mixer (v0.10.x D4 metal-mixing program).
        ``None`` / ``"diis"`` (default) keep the Fock-space DIIS + linear
        density-damping path. ``"anderson"`` selects Anderson/Pulay density
        mixing and ``"broyden"`` selects limited-memory Broyden -- both
        extrapolate over a history of density residuals and can converge
        near-degenerate / metallic cells where Fock-DIIS limit-cycles. When a
        density mixer is active it *replaces* Fock-DIIS, linear damping and the
        quadratic-SCF fallback (alternative convergence strategies are not
        stacked -- CLAUDE.md Sec.7); Fermi-Dirac / Gilat smearing and the level
        shift still apply (they are physics / virtual-space controls, not
        anti-oscillation aids).
    density_mixer_depth, density_mixer_beta
        Residual-history length and linear-mixing parameter for the density
        mixer (ignored unless ``density_mixer`` selects one).
    density_mixer_kerker
        Apply the Kerker preconditioner (D4b) to the density mixer's residual --
        damps long-wavelength (small-|G|) charge sloshing on metals. Requires a
        ``density_mixer``; raises otherwise. It filters the *residual*, so the
        converged energy is unchanged (exact SCF solution -- CLAUDE.md Sec.7); only
        the convergence path changes.
    kerker_k0, kerker_strength, kerker_cutoff_ha
        Kerker screening wave-vector ``G₀`` (1/bohr; larger damps more
        long-wavelength modes), damping fraction ``g in (0, 1]``, and the
        plane-wave cutoff sizing the collocation grid. Ignored unless
        ``density_mixer_kerker`` is set.

    Returns
    -------
    :class:`PeriodicRKSMultiKEwaldResult`.
    """
    opts = options if options is not None else PeriodicKSOptions()
    from .guess import select_periodic_driver_guess
    guess = select_periodic_driver_guess(
        system, opts, route="ewald", method="RKS", driver="run_rks_periodic_multi_k_ewald3d",
        multi_k=True, restart_supplied=initial_density_k is not None,
    ).transport
    if initial_density_k is not None:
        if len(kmesh.kpoints) != int(np.prod(kmesh.mesh)):
            raise NotImplementedError("READ requires a complete full k mesh on the Ewald lattice adapter")
        if getattr(opts, "atomic_spins", None):
            raise ValueError("atomic_spins requires SAD without a restart density")
    if (getattr(opts, "initial_guess", None) == InitialGuess.READ) and initial_density_k is None:
        raise NotImplementedError(
            "periodic READ requires initial_density_k with a complete per-k "
            "density payload. Use run_periodic_job(read_from=...) to load "
            "a compatible result or archive; see docs/user_guide/initial_guess.md."
        )
    lat_opts: LatticeSumOptions = opts.lattice_opts
    slab_mode = lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D
    if slab_mode and system.dim != 2:
        raise ValueError(
            "run_rks_periodic_multi_k_ewald3d: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}"
        )
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
    # here aligns e_nuclear with run_rks_periodic_gamma_ewald3d and zeroes the
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

    # w must match the nuclear Ewald a so the jellium background
    # terms cancel exactly.  When not specified explicitly, use
    # CRYSTAL's default a = 2.8/V^{1/3} (matching the BIPOLE driver).
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if slab_mode:
        if omega <= 0.0:
            omega = float(getattr(lat_opts, "slab_ewald_alpha", 0.4))
        if omega <= 0.0:
            omega = 0.4
        lat_opts.slab_ewald_alpha = float(omega)
    elif omega <= 0.0:
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

    if slab_mode:
        grid_shape_t = (0, 0, 0)
    elif grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)
    if slab_mode:
        plog.info(
            f"RKS multi-k SLAB_EWALD_2D / functional={opts.functional!r}, "
            f"alpha = {float(omega):.3f}"
        )
    else:
        plog.info(
            f"RKS multi-k EWALD_3D / functional={opts.functional!r}, "
            f"omega = {float(omega):.3f}, "
            f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
        )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    from .options_dump import dump_active_settings

    dump_active_settings(
        plog,
        [
            ("PeriodicKSOptions", opts),
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
                },
            ),
        ],
    )
    if plog.level >= 5:
        from .scf_log import format_basis_summary

        plog.write_raw(format_basis_summary(basis))

    # Closed-shell sanity.
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_rks_periodic_multi_k_ewald3d: closed-shell RKS requires "
            f"even electron count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_rks_periodic_multi_k_ewald3d: closed-shell RKS requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError(
            "run_rks_periodic_multi_k_ewald3d: smearing_temperature must be >= 0"
        )

    # BZ-integration backend (opt-in). None / "smearing" keeps the temperature
    # path; "gilat" selects the Gilat-Raubenheimer net (CRYSTAL SHRINK-ISP
    # analogue) on the full regular mesh. GR is T=0, so no finite smearing.
    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "run_rks_periodic_multi_k_ewald3d: bz_integration must be None, "
            f"'smearing', or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "run_rks_periodic_multi_k_ewald3d: bz_integration='gilat' is a "
            "T=0 integrator; do not combine it with smearing_temperature > 0"
        )
    # Both finite-T smearing and GR give fractional occupations -> shared
    # fractional-density build path (vs the integer Aufbau path).
    use_fractional_density = (smearing_T > 0.0) or use_gilat

    # ---- Functional + DFT grid ------------------------------------------
    func = Functional(opts.functional, 1)  # spin-unpolarised RKS
    # CAM exchange assembly (global hybrids: full-range fraction;
    # screened hybrids like hse06: erfc short-range kernel; LR-heavy
    # RSH fails closed). omega_screen is unrelated to this driver's
    # ``omega`` (the Ewald split alpha).
    exx = resolve_periodic_exchange(
        func, where="run_rks_periodic_multi_k_ewald3d"
    )

    if opts.use_periodic_becke:
        grid = build_periodic_becke_grid(
            system,
            grid_options=opts.grid,
            image_radius_bohr=float(opts.becke_image_radius_bohr),
        )
    else:
        grid = build_grid(system.unit_cell_molecule(), opts.grid)

    k_points = list(kmesh.kpoints)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = len(k_points)
    if n_k == 0:
        raise ValueError("kmesh has no k-points")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"kmesh.weights must sum to 1; got {weights.sum():.6f}")
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}"
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

    # The current C++ periodic XC builder materialises AO values (and,
    # for GGA, three AO-gradient matrices) for every real-space density
    # cell. For MgO/pob-TZVP-sized inputs this reaches tens of GiB and
    # the kernel is often SIGKILLed before Python can report anything.
    # Fail fast with an actionable message instead.
    n_xc_cells = len(direct_lattice_cells(system, float(lat_opts.cutoff_bohr)))
    n_xc_grid = int(np.asarray(grid.points).shape[0])
    _guard_legacy_periodic_xc_memory(
        n_grid=n_xc_grid,
        nbf=int(basis.nbasis),
        n_cells=n_xc_cells,
        is_gga=(func.kind == XCKind.GGA),
        functional=str(opts.functional),
    )

    # ---- Real-space one-electron integrals ------------------------------
    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        if slab_mode:
            from .periodic_v_ne_slab import build_v_ne_slab_ewald_2d_k_cache

            slab_vne_cache = build_v_ne_slab_ewald_2d_k_cache(
                basis,
                system,
                lat_opts,
                alpha=float(omega),
            )
            V_lat = None
        else:
            from .periodic_v_ne import compute_nuclear_lattice_dispatch

            V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    cells = list(S_lat.cells)

    # Per-k S(k), Hcore(k), orthogonaliser X(k).
    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    # Per-k linear-dependence preflight; see periodic_rhf_multi_k_ewald
    # for the rationale (Searle et al., ARCHER eCSE04-16, 2017).
    from .linear_dependence import (
        PERIODIC_OVERLAP_HINT,
        scf_preflight_overlap_check,
    )

    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        if slab_mode:
            from .periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_k_matrix

            V_k = compute_v_ne_slab_ewald_2d_k_matrix(
                basis,
                system,
                lat_opts,
                k_arr,
                alpha=float(omega),
                cache=slab_vne_cache,
            )
        else:
            V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        scf_preflight_overlap_check(
            S_k,
            plog=plog,
            label=f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})",
            basis=basis,
            remediation_hint=PERIODIC_OVERLAP_HINT,
        )
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k,
            linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if n_occ > n_kept:
            raise RuntimeError(
                "run_rks_periodic_multi_k_ewald3d: canonical "
                f"orthogonalisation at k = {k_arr} dropped too many "
                f"directions (n_occ = {n_occ}, n_kept = {n_kept}); "
                "loosen linear_dep_threshold or pick a less redundant basis."
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

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

    # T_lat / V_lat are folded into Hcore(k), and SAP has consumed T_lat when
    # requested. Free the per-cell one-electron lattice integrals before the
    # SCF iterations (S_lat is still needed for its g0 block).
    del T_lat, V_lat

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    # ---- Initial guess: diagonalize Hcore(k) or F_SAP(k) ----------------
    C_per_k: List[np.ndarray] = []
    eps_per_k: List[np.ndarray] = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        C_k, eps_k = _diag_in_orth_basis(F_guess_k, X_k)
        C_per_k.append(C_k.astype(complex))
        eps_per_k.append(eps_k)

    n_electrons_per_cell = float(n_elec)

    def _occupations_from_eps(
        eps_per_k_local: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], float, float]:
        if use_gilat:
            # Gilat-Raubenheimer net occupations + E_F (sharp Fermi surface ->
            # entropy 0). Handles a full Monkhorst-Pack mesh or a symmetry-
            # reduced (IBZ) mesh -- IBZ is expanded to the full BZ internally
            # (eps_n(R k) = eps_n(k)) and the occupations gathered back onto the
            # IBZ k-points.
            from .bz_integration import gilat_occupations_for_kmesh

            occ_gr, ef_gr = gilat_occupations_for_kmesh(
                system,
                kmesh,
                eps_per_k_local,
                n_electrons_per_cell,
                spin_degeneracy=2.0,
            )
            return occ_gr, float(ef_gr), 0.0
        return _closed_shell_periodic_occupations(
            eps_per_k_local,
            weights,
            n_electrons_per_cell,
            n_occ,
            smearing_T,
        )

    occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)

    def _fractional_density_needed(
        occ_per_k_local: Sequence[np.ndarray],
    ) -> bool:
        """Whether these occupations need the occupation-driven builder.

        The fixed ``C[:, :n_occ]`` slice builder is exact only for the legacy
        per-k integer Aufbau pattern. The T = 0 fill is one global Fermi level
        over the mesh (#85), so a band-overlap mesh can occupy a different
        number of bands per k (or a shared fractional group at mu), which no
        fixed slice reproduces; those occupations must be built from
        themselves, as the native multi-k GDF route does (#509).
        """
        return not _occupations_are_per_k_integer_aufbau(occ_per_k_local, n_occ)

    if _fractional_density_needed(occ_per_k):
        D_real = real_space_density_from_kpoints_fractional(
            C_per_k,
            occ_per_k,
            kmesh,
            cells,
        )
    else:
        D_real = real_space_density_from_kpoints(
            C_per_k,
            [n_occ] * n_k,
            kmesh,
            cells,
        )

    # Density-mode guesses: overwrite D_real at g=0 with the engine
    # density and zero the other cells. See run_rhf_periodic_multi_k_ewald3d
    # for the rationale.
    if initial_density_k is not None:
        D_engine = None
        plog.info("initial guess: READ (complete per-k density)")
    elif fock_guess_per_k:
        # Keep this route's global-k Aufbau/smearing occupations on the
        # one-particle guess orbitals instead of replacing them with a
        # repeated Gamma density.
        D_engine = None
        plog.info(
            f"initial guess: {guess.name} "
            "(one-particle Hamiltonian diagonalise at each k)"
        )
    else:
        D_engine = initial_density_closed_shell(
            system.unit_cell_molecule(),
            basis,
            n_occ,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts,
            overlap=S_k_list, weights=weights,
        )
    if D_engine is not None:
        plog.info(f"initial guess: {guess.name} (density via GuessEngine)")
        for g_idx in range(len(D_real.cells)):
            if (D_real.cells[g_idx].index == np.array([0, 0, 0])).all():
                D_real.set_block(g_idx, D_engine)
            else:
                D_real.set_block(g_idx, np.zeros_like(D_engine, dtype=float))
    elif not fock_guess_per_k and initial_density_k is None:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise at each k)")

    D_per_k = _density_matrices_per_k(C_per_k, occ_per_k)
    if D_engine is not None:
        D_engine_k = np.asarray(D_engine, dtype=complex)
        D_engine_k = 0.5 * (D_engine_k + D_engine_k.conj().T)
        D_per_k = [D_engine_k.copy() for _ in range(n_k)]

    if initial_density_k is not None:
        from .guess import periodic_restart_lattice_density, normalize_density_k_guess
        D_per_k = normalize_density_k_guess(initial_density_k, S_k_list, weights, n_elec)
        D_real = periodic_restart_lattice_density(
            D_per_k, S_k_list, weights, n_elec, kmesh, D_real.cells, retained_k_system=system,
        )
    D_real_prev: Optional[LatticeMatrixSet] = None
    D_per_k_prev: Optional[List[np.ndarray]] = None

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_rks_periodic_multi_k_ewald3d: damping must be in "
            f"[0, 1); got {damping}"
        )

    # ---- Density-space mixer (D4 metal-mixing program) ------------------
    # When selected it replaces Fock-DIIS / linear damping / quadratic
    # fallback (see the density_mixer docstring + CLAUDE.md Sec.7).
    mixer = _make_density_mixer(
        density_mixer, density_mixer_depth, density_mixer_beta
    )
    if mixer is not None:
        plog.info(
            f"density mixer: {density_mixer!r} "
            f"(depth={int(density_mixer_depth)}, beta={float(density_mixer_beta)}) "
            "-- Fock-DIIS, linear damping and quadratic fallback disabled"
        )

    # ---- Kerker preconditioner (D4b) -----------------------------------
    # Damps long-wavelength charge sloshing on metals by filtering the density
    # mixer's *residual* (fixed point invariant -> CLAUDE.md Sec.7-safe). It is a
    # preconditioner FOR the density mixer, so it requires one; fail closed
    # rather than silently no-op if requested without a mixer.
    kerker: Optional[KerkerPreconditioner] = None
    if density_mixer_kerker:
        if mixer is None:
            raise ValueError(
                "run_rks_periodic_multi_k_ewald3d: density_mixer_kerker=True "
                "requires a density_mixer ('anderson' or 'broyden') -- Kerker "
                "preconditions the density mixer's residual, it is not a "
                "standalone accelerator."
            )
        kerker = KerkerPreconditioner(
            basis,
            system,
            _g0_block(S_lat),
            k0=float(kerker_k0),
            strength=float(kerker_strength),
            cutoff_ha=float(kerker_cutoff_ha),
        )
        plog.info(
            f"Kerker preconditioner: k0={float(kerker_k0)}, "
            f"strength={float(kerker_strength)}, cutoff={float(kerker_cutoff_ha)} Ha "
            "(long-wavelength residual damping)"
        )

    use_diis = bool(opts.use_diis) and mixer is None
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[MultiKPeriodicSCFAccelerator] = (
        MultiKPeriodicSCFAccelerator(opts) if use_diis else None
    )

    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )

    level_shift = float(getattr(opts, "level_shift", 0.0))

    # Phase C1c -- quadratic SCF fallback (per-k Newton step). Disabled when a
    # density mixer is active (it owns the convergence strategy; CLAUDE.md Sec.7).
    quadratic_fallback_iter = (
        0 if mixer is not None else int(getattr(opts, "quadratic_fallback_iter", 0))
    )
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # ---- SCF loop -------------------------------------------------------
    scf_trace: List[SCFIteration] = []
    E_prev = 0.0
    F_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]

    # XC bookkeeping -- populated each iteration.
    E_xc = 0.0
    E_coulomb_per_cell = 0.0
    E_hf_K_per_cell = 0.0
    E_total = 0.0
    E_elec = 0.0

    scf_label = "SLAB_EWALD_2D" if slab_mode else "EWALD_3D"
    plog.banner(f"SCF (RKS multi-k {opts.functional!r}, {scf_label})")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    converged = False
    iter_idx = 0

    # Cache the iteration-invariant per-cell analytic-FT Hartree-J machinery
    # once (E2, docs/pbc_audit_2026-06.md), keyed on the fixed D_real.cells,
    # so the SCF loop redoes only the per-cell density contraction.
    # Bit-identical to the per-iteration rebuild; grid backend -> None.
    if slab_mode:
        j_cache = make_slab_ewald_2d_lattice_j_cache(
            basis,
            system,
            D_real.cells,
            lattice_opts=lat_opts,
            alpha=float(omega),
        )
    else:
        j_cache = make_ewald_3d_lattice_j_cache(
            basis, system, D_real.cells, lattice_opts=lat_opts,
        )

    # ---- Corrected-Ewald full-range exchange -----------------------------
    # The full-Coulomb (c_full) arm cannot be summed in real space on a
    # finite k mesh: the SCF density is BvK-torus periodic, so D(g) does
    # not decay and a bare 1/r image sum diverges (energy fell 177 mHa
    # going from a 12- to a 20-bohr cutoff on H2/PBE0 before this).
    # Route it through the convergent Ewald split instead -- erfc K_SR in
    # real space, reciprocal K_LR per q = k-k', plus the q+G=0
    # probe-charge term -- the same construction the ROHF/ROKS/BIPOLE
    # routes use and what PySCF's exxdiv='ewald' implements. The c_sr
    # erfc arm of a screened hybrid is already short-ranged and stays on
    # the real-space path untouched (hse06 has c_full = 0).
    corrected_exchange = None
    if exx.c_full != 0.0 or guess == InitialGuess.PATOM:
        _unavailable = corrected_exchange_unavailable_reason(
            system, kmesh, slab_mode=slab_mode
        )
        if _unavailable is not None:
            raise NotImplementedError(
                "run_rks_periodic_multi_k_ewald3d: this job needs "
                f"full-range exact exchange (c_full = {exx.c_full:g}) but "
                f"{_unavailable}. Falling back to the bare real-space image "
                "sum is not an option -- it does not converge on a finite k "
                "mesh. Use a Monkhorst-Pack mesh (full, or symmetry-reduced "
                "with vq.attach_symmetry + use_symmetry=True), or "
                "jk_method='bipole' / run_pbc_bipole_rks, or a pure "
                "functional, or a screened hybrid (hse06)."
            )
        corrected_exchange = CorrectedEwaldExchange.build(
            basis,
            system,
            np.asarray([np.asarray(c.r_cart, dtype=float) for c in cells]),
            kmesh,
            float(omega),
            where="run_rks_periodic_multi_k_ewald3d",
            slab_mode=slab_mode,
        )



    # Wedge handle: set either by the corrected split (c_full path, which
    # carries its own unfolding) or standalone for a screened hybrid
    # (c_sr-only, e.g. hse06) on a supported symmetry-reduced MP mesh.
    # The Pisani/Dovesi density transformation applies to 2D slabs and
    # 3D crystals alike. The c_sr arm
    # has no k-space pieces, but its real-space K_erfc is FIRST order in
    # the density, and the plain weighted wedge fold uses each
    # representative in place of its orbit average -- a truncation-
    # asymmetry-level error on shift-bearing cells that pure DFT (second
    # order) never showed. The unfolded fold fixes that, and it also
    # unlocks the SYM3b orbit reduction for the 3D K_erfc build. Slabs
    # retain the raw truncated-operator convention used by their full mesh.
    wedge_unfolding = (
        corrected_exchange.unfolding
        if corrected_exchange is not None
        else None
    )
    if (
        wedge_unfolding is None
        and exx.c_sr != 0.0
        and (
            (slab_mode and int(system.dim) == 2)
            or (not slab_mode and int(system.dim) == 3)
        )
    ):
        _n_input = len(list(kmesh.kpoints))
        _mesh_dims = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
        _ir = np.asarray(
            getattr(kmesh, "ir_mapping", []), dtype=int
        ).reshape(-1)
        _n_full = int(np.prod(_mesh_dims))
        if _n_full > _n_input:
            _operations = getattr(
                getattr(system, "symmetry", None), "operations", None
            )
            if _ir.size != _n_full or not _operations:
                raise NotImplementedError(
                    "run_rks_periodic_multi_k_ewald3d: cannot reconstruct "
                    "the full-star density for this symmetry-reduced "
                    "Monkhorst-Pack mesh. Reattach current symmetry data "
                    "or run the corresponding full mesh."
                )
            from .periodic_k_symmetry import KMeshUnfolding

            try:
                wedge_unfolding = KMeshUnfolding.build(system, basis, kmesh)
            except ValueError as exc:
                # A representative-only fold is not a physical fallback:
                # it replaces each star member by the IBZ representative.
                raise NotImplementedError(
                    "run_rks_periodic_multi_k_ewald3d: cannot reconstruct "
                    "the full-star density for this symmetry-reduced "
                    "Monkhorst-Pack mesh. Reattach current symmetry data "
                    "or run the corresponding full mesh."
                ) from exc

    # SYM3b orbit reduction for the real-space exchange arms, wedge path
    # only. Profiled at 94 % of a wedge SCF iteration (LiH rock-salt,
    # cutoff 12: 45 s of 45.2 s), and the k-mesh-independent cost is why a
    # wedge run could be SLOWER than the full mesh. The reduced build
    # equals the *symmetrized* full build, which differs from the raw one
    # by the operator truncation asymmetry on shift-bearing cells -- the
    # wedge is already the symmetrized-answer gauge (its density is the
    # orbit-symmetrized object), so the reduction is only enabled there;
    # full-mesh numbers stay byte-identical. Mapping build is one-time
    # pure Python on the SAME radial cell list the K builder uses, so
    # positional block indexing is unchanged.
    exchange_symmetry_reduction = None
    if wedge_unfolding is not None and not slab_mode:
        from ._vibeqc_core import direct_lattice_cells as _radial_cells
        from .bipole_symmetry_fock import (
            cell_orbit_mapping,
            representative_cell_indices,
        )

        # The mapping must index the K builder's OWN radial cell list
        # (direct_lattice_cells at the 2e cutoff), not the overlap
        # template ``cells``: under LatticeSumOptions.pair_complete_1e
        # (#429) the template is longer. Identical lists with the switch
        # off, so byte-identical there.
        _sym_mapping = cell_orbit_mapping(
            system, basis, list(_radial_cells(system, lat_opts.cutoff_bohr))
        )
        if _sym_mapping is not None:
            exchange_symmetry_reduction = (
                _sym_mapping,
                representative_cell_indices(_sym_mapping),
            )

    def _corrected_k_space_terms(
        D_input: LatticeMatrixSet,
        exchange_assembly,
        D_input_per_k: Optional[List[np.ndarray]] = None,
    ) -> Optional[List[np.ndarray]]:
        """``0.5 * c_full * (K_LR(k) + xi_G0 S D S)`` per k, or ``None``.

        The 0.5 matches the ``J - 0.5 K`` closed-shell fold in
        :func:`build_fock_2e_ewald3d_blocks`: ``D_input`` is the total
        (doubly-occupied) density, and K is linear in it.

        ``D_input_per_k`` is the driver's own ``D(k)``, kept in lock-step
        with ``D_input`` (damping included). Passing it skips the
        real-space -> k round trip, which is not merely cheaper: the
        inverse fold requires the lattice cell list to cover the BvK
        torus, and standard fixtures do not (a 30-bohr cell at a 12-bohr
        cutoff carries one cell and covers a ``(2,1,1)`` torus not at
        all). The per-k density is the primary object here -- the
        real-space one is folded *from* it -- so using it directly is
        also the more faithful choice.
        """
        if corrected_exchange is None or exchange_assembly is None:
            return None
        c_full = float(exchange_assembly.c_full)
        if c_full == 0.0:
            return None
        D_k_list = (
            list(D_input_per_k)
            if D_input_per_k is not None
            else corrected_exchange.torus_densities(D_input)
        )
        terms = corrected_exchange.k_space_terms_all_k(S_k_list, D_k_list)
        return [0.5 * c_full * t for t in terms]

    def _build_hf_k(
        D_input: LatticeMatrixSet,
        exchange_assembly,
        D_input_per_k: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        # exchange_assembly None -> J-only blocks (reporting reference).
        if slab_mode:
            return build_periodic_fock_slab_ewald2d_k(
                basis,
                system,
                D_input,
                alpha=float(omega),
                k_points_cart=[np.asarray(k) for k in k_points],
                Hcore_k=None,
                lattice_opts=lat_opts,
                exchange_scale=0.0,
                exchange_assembly=exchange_assembly,
                j_cache=j_cache,
            )
        F_HF_k = build_periodic_fock_ewald3d_k(
            basis,
            system,
            D_input,
            omega=float(omega),
            k_points_cart=[np.asarray(k) for k in k_points],
            Hcore_k=None,
            lattice_opts=lat_opts,
            grid_shape=grid_shape_t,
            origin=origin,
            spacing_bohr=spacing_bohr,
            exchange_scale=0.0,
            exchange_assembly=exchange_assembly,
            # Puts the c_full arm on erfc(alpha r)/r; the reciprocal and
            # q+G=0 arms of the same split are subtracted just below.
            full_range_alpha=(
                None if corrected_exchange is None else float(omega)
            ),
            symmetry_reduction=exchange_symmetry_reduction,
            j_cache=j_cache,
        )
        k_space = _corrected_k_space_terms(
            D_input, exchange_assembly, D_input_per_k
        )
        if k_space is not None:
            F_HF_k = [f - t for f, t in zip(F_HF_k, k_space)]
        return F_HF_k

    if guess == InitialGuess.PATOM:
        seed_exx = resolve_periodic_exchange(
            None, where="run_rks_periodic_multi_k_ewald3d PATOM",
        )
        seed_density = (wedge_unfolding.fold_density(D_per_k, lat_opts)
                        if wedge_unfolding is not None else D_real)
        seed_focks = [h + f for h, f in zip(
            Hcore_k_list, _build_hf_k(seed_density, seed_exx, D_per_k),
        )]
        C_per_k, eps_per_k = [], []
        for fock, x in zip(seed_focks, X_k_list):
            c, e = _diag_in_orth_basis(fock, x)
            C_per_k.append(c.astype(complex))
            eps_per_k.append(e)
        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)
        D_per_k = _density_matrices_per_k(C_per_k, occ_per_k)
        D_real = real_space_density_from_kpoints_fractional(
            C_per_k, occ_per_k, kmesh, cells,
        )
        D_real_prev = D_per_k_prev = None

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        D_used = D_real
        D_used_per_k = [D.copy() for D in D_per_k]
        if iter_idx > 1 and damping > 0.0 and not diis_active and mixer is None:
            D_used = _damp_lattice_matrix(D_real, D_real_prev, damping)
            if D_per_k_prev is not None:
                D_used_per_k = [
                    damping * D_prev + (1.0 - damping) * D_cur
                    for D_cur, D_prev in zip(D_per_k, D_per_k_prev)
                ]
        if wedge_unfolding is not None:
            # Symmetry-reduced wedge on the hybrid path (full-range OR
            # screened): the weighted wedge fold above uses each
            # representative in place of its orbit average and is wrong
            # at multi-cell cutoffs. J, K and XC are all fed from
            # D_used, so rebuild it exactly by unfolding the wedge per-k
            # densities to the full BZ and folding over the full mesh
            # (Pisani, Dovesi & Roetti 1988, Eqs. II.7.21 / II.7.48).
            D_used = wedge_unfolding.fold_density(D_used_per_k, lat_opts)

        # --- HF part of the Fock at every k via the Ewald composed
        #     dispatcher with the CAM exchange assembly. For pure DFT
        #     the K build is internally skipped.
        F_HF_k_list = _build_hf_k(D_used, exx, D_used_per_k)

        # --- XC contribution. build_xc_periodic returns a LatticeMatrixSet
        #     V_xc(g) and a scalar E_xc per unit cell. Bloch-fold to
        #     per-k V_xc(k) so it can be added to each F(k).
        xc_contrib = build_xc_periodic(
            basis,
            system,
            grid,
            func,
            D_used,
            lat_opts,
        )
        E_xc = float(xc_contrib.e_xc)
        V_xc_set = xc_contrib.V_xc
        V_xc_k_list: List[np.ndarray] = [
            _bloch_sum_lms_at_k(V_xc_set, np.asarray(k)) for k in k_points
        ]

        # --- Total per-k Fock.
        F_k_list = []
        for idx in range(n_k):
            F_k = Hcore_k_list[idx] + F_HF_k_list[idx] + V_xc_k_list[idx]
            F_k = 0.5 * (F_k + F_k.conj().T)
            F_k_list.append(F_k)

        # --- Per-cell electronic energy.
        # E_elec = E_xc + S_k w_k tr(D(k).H(k)) + 1/2 S_k w_k tr(D(k).F_HF(k))
        # where F_HF(k) = J(k) - (a/2).K(k) (the Ewald 2e contribution
        # already in F_HF_k_list, *not* including Hcore or V_xc).
        E_core_trace = 0.0
        E_HF_trace = 0.0
        grad_norm_sum = 0.0
        error_k_list: List[np.ndarray] = []
        for idx in range(n_k):
            D_k = D_used_per_k[idx]
            w = float(weights[idx])
            E_core_trace += w * np.real(np.trace(D_k @ Hcore_k_list[idx]))
            E_HF_trace += 0.5 * w * np.real(np.trace(D_k @ F_HF_k_list[idx]))
            S_k = S_k_list[idx]
            FDS = F_k_list[idx] @ D_k @ S_k
            grad = FDS - FDS.conj().T
            error_k_list.append(grad)
            grad_norm_sum += w * float(np.linalg.norm(grad))
        E_elec = E_xc + float(E_core_trace) + float(E_HF_trace)
        # Madelung-leak correction (v0.6.1).
        _D_g0 = _g0_block(D_real)
        _S_g0 = _g0_block(S_lat)
        E_madelung_fix = (
            0.0
            if slab_mode
            else _madelung_energy_correction_for_lat(_D_g0, _S_g0, system, lat_opts)
        )
        E_total = E_elec + e_nuc + E_madelung_fix

        # Diagnostic energy breakdown -- split E_HF_trace into J + K
        # without rebuilding (use a J-only Ewald build per cell).
        # Cheap for reporting: only at convergence we need to be exact.
        # Per-iter we leave E_coulomb / E_hf_exchange at last-converged
        # values; a single re-build at the end gives precise numbers.

        free_energy = E_total - smearing_T * entropy
        dE = free_energy - E_prev
        # Divergence detection (v0.6.2).
        check_scf_divergence(
            "run_rks_periodic_multi_k_ewald3d",
            iter_idx,
            free_energy,
            grad_norm_sum,
            dE,
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(free_energy),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(free_energy),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm_sum),
            diis=(accel.subspace_size if accel is not None else 0),
        )

        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )

        # Phase C1c gate -- bypass DIIS + level shift when active.
        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )

        new_C_per_k: List[np.ndarray] = []
        new_eps_per_k: List[np.ndarray] = []

        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            for idx in range(n_k):
                C_k, eps_k = quadratic_step(
                    F_k_list[idx],
                    C_per_k[idx],
                    eps_per_k[idx],
                    n_occ,
                    shift=quadratic_fallback_shift,
                    max_step=quadratic_fallback_max_step,
                )
                new_C_per_k.append(C_k)
                new_eps_per_k.append(eps_k)
        else:
            # --- SCF-accelerator extrapolation. DIIS / KDIIS run
            #     natively on the per-k complex Hermitian matrices;
            #     EDIIS / ADIIS / EDIIS_DIIS bridge through per-cell
            #     blocks. ``density_k_list`` is the per-k Bloch sum of
            #     the per-cell D_used (the RKS driver keeps density
            #     per-cell because the XC build operates there).
            if accel is not None:
                density_k_list = [
                    _bloch_sum_lms_at_k(D_used, np.asarray(k)) for k in k_points
                ]
                F_ex_list = accel.extrapolate_rhf(
                    F_k_list,
                    error_k_list=error_k_list,
                    density_k_list=density_k_list,
                    energy=free_energy,
                    mo_coeffs_k_list=C_per_k,
                    n_occ=n_occ,
                    weights=list(weights),
                    cells=cells,
                    kpoints=list(k_points),
                )
                if diis_active:
                    F_k_list = F_ex_list

            # --- Saunders-Hillier level shift per k.
            if level_shift != 0.0:
                F_for_diag: List[np.ndarray] = []
                for idx in range(n_k):
                    D_k = D_used_per_k[idx]
                    S_k = S_k_list[idx]
                    F_shift = (
                        F_k_list[idx]
                        + level_shift * S_k
                        - (level_shift / 2.0) * (S_k @ D_k @ S_k)
                    )
                    F_shift = 0.5 * (F_shift + F_shift.conj().T)
                    F_for_diag.append(F_shift)
            else:
                F_for_diag = F_k_list

            # --- Diagonalize.
            for idx in range(n_k):
                C_k, eps_k = _diag_in_orth_basis(F_for_diag[idx], X_k_list[idx])
                new_C_per_k.append(C_k)
                new_eps_per_k.append(eps_k)
        C_per_k = new_C_per_k
        eps_per_k = new_eps_per_k

        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)
        D_new_per_k = _density_matrices_per_k(C_per_k, occ_per_k)

        if _fractional_density_needed(occ_per_k):
            D_real_new = real_space_density_from_kpoints_fractional(
                C_per_k,
                occ_per_k,
                kmesh,
                cells,
            )
        else:
            D_real_new = real_space_density_from_kpoints(
                C_per_k,
                [n_occ] * n_k,
                kmesh,
                cells,
            )

        # --- Density-space mixing (Anderson / Broyden). The SCF map is
        #     D_in -> F[D_in] -> diagonalise -> D_out; the mixer extrapolates
        #     the next input from the (D_in, D_out) residual history. We mix the
        #     per-k density *matrices* (the electron-count-exact SCF variable)
        #     and rebuild D_real from the mixed D(k) via the canonical fold --
        #     NOT mix D_real then Bloch-sum back (the cutoff cell list aliases,
        #     blowing up the per-k electron count; see periodic_density_mixing).
        if mixer is not None:
            # Optional Kerker preconditioning of the density residual: feed the
            # mixer a residual whose long-wavelength (small-|G|) content is
            # damped. Implemented as D_out' = D_in + Kerker(D_out - D_in), so
            # the mixer's residual becomes the preconditioned one; at the fixed
            # point (D_out = D_in) it is the unchanged true SCF solution (Sec.7).
            if kerker is not None:
                resid = [
                    np.asarray(Dn) - np.asarray(Du)
                    for Dn, Du in zip(D_new_per_k, D_used_per_k)
                ]
                resid = kerker.precondition(resid, weights)
                D_out_eff = [
                    np.asarray(Du) + r for Du, r in zip(D_used_per_k, resid)
                ]
            else:
                D_out_eff = D_new_per_k
            x_in = per_k_density_to_vector(D_used_per_k)
            x_out = per_k_density_to_vector(D_out_eff)
            x_next = mixer.update(x_in, x_out)
            D_new_per_k = vector_to_per_k_density(x_next, D_new_per_k)
            D_real_new = real_space_density_from_per_k_density(
                D_new_per_k, kmesh, cells
            )

        D_real_prev = D_used
        D_per_k_prev = [D.copy() for D in D_used_per_k]
        D_real = D_real_new
        D_per_k = D_new_per_k

        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy
        if converged:
            break

    # ---- Final consistency pass on the converged D ----------------------
    if converged:
        if wedge_unfolding is not None:
            # Same exact fold for the reported density and both final
            # builds (F_HF and the J-only reference), so the energy
            # decomposition comes from one consistent density.
            D_real = wedge_unfolding.fold_density(D_per_k, lat_opts)
        F_HF_k_list = _build_hf_k(D_real, exx, D_per_k)
        # J-only build for the Coulomb / HF-exchange decomposition
        # (used for reporting only, not the Fock).
        J_only_k_list = _build_hf_k(D_real, None)
        xc_contrib = build_xc_periodic(
            basis,
            system,
            grid,
            func,
            D_real,
            lat_opts,
        )
        E_xc = float(xc_contrib.e_xc)
        V_xc_set = xc_contrib.V_xc
        V_xc_k_list = [_bloch_sum_lms_at_k(V_xc_set, np.asarray(k)) for k in k_points]
        F_k_list = []
        for idx in range(n_k):
            F_k = Hcore_k_list[idx] + F_HF_k_list[idx] + V_xc_k_list[idx]
            F_k = 0.5 * (F_k + F_k.conj().T)
            F_k_list.append(F_k)

        final_C_per_k: List[np.ndarray] = []
        final_eps_per_k: List[np.ndarray] = []
        for idx in range(n_k):
            C_k, eps_k = _diag_in_orth_basis(F_k_list[idx], X_k_list[idx])
            final_C_per_k.append(C_k)
            final_eps_per_k.append(eps_k)
        C_per_k = final_C_per_k
        eps_per_k = final_eps_per_k
        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)
        _fractional_final = _fractional_density_needed(occ_per_k)
        if _fractional_final:
            D_real = real_space_density_from_kpoints_fractional(
                C_per_k,
                occ_per_k,
                kmesh,
                cells,
            )
        else:
            D_real = real_space_density_from_kpoints(
                C_per_k,
                [n_occ] * n_k,
                kmesh,
                cells,
            )

        E_core_trace = 0.0
        E_HF_trace = 0.0
        E_J_trace = 0.0
        for idx in range(n_k):
            C_k = C_per_k[idx]
            if _fractional_final:
                D_k = (C_k * occ_per_k[idx][None, :].astype(complex)) @ C_k.conj().T
            else:
                C_occ = C_k[:, :n_occ]
                D_k = 2.0 * (C_occ @ C_occ.conj().T)
            w = float(weights[idx])
            E_core_trace += w * np.real(np.trace(D_k @ Hcore_k_list[idx]))
            E_HF_trace += 0.5 * w * np.real(np.trace(D_k @ F_HF_k_list[idx]))
            E_J_trace += 0.5 * w * np.real(np.trace(D_k @ J_only_k_list[idx]))
        E_elec = E_xc + float(E_core_trace) + float(E_HF_trace)
        # Madelung-leak correction (v0.6.1).
        _D_g0 = _g0_block(D_real)
        _S_g0 = _g0_block(S_lat)
        E_madelung_fix = (
            0.0
            if slab_mode
            else _madelung_energy_correction_for_lat(_D_g0, _S_g0, system, lat_opts)
        )
        E_total = E_elec + e_nuc + E_madelung_fix
        E_coulomb_per_cell = float(E_J_trace)
        # tr(D.F_HF) = tr(D.J) - (a/2).tr(D.K)  ->
        # E_HF_trace - E_J_trace = -(a/4).tr(D.K).factor depending on
        # the 1/2 already included in E_HF_trace. Match the Γ-only KS
        # reporting convention: E_hf_exchange = -(a/2).1/2.tr(D.K) =
        # E_HF_trace - E_J_trace.
        E_hf_K_per_cell = float(E_HF_trace - E_J_trace)

    free_energy = E_total - smearing_T * entropy
    plog.converged(n_iter=iter_idx, energy=E_total, converged=converged)

    from .eigs_preflight import _copy_lattice_opts

    return PeriodicRKSMultiKEwaldResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        used_kpoint_symmetry_unfolding=wedge_unfolding is not None,
        energy=E_total,
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        e_xc=float(E_xc),
        e_coulomb=float(E_coulomb_per_cell),
        e_hf_exchange=float(E_hf_K_per_cell),
        n_iter=iter_idx,
        converged=converged,
        mo_energies=eps_per_k,
        mo_coeffs=C_per_k,
        fock=F_k_list,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        density=D_real,
        density_k=(_density_matrices_per_k(C_per_k, occ_per_k) if converged
                   else [d.copy() for d in D_per_k]),
        scf_trace=scf_trace,
        functional=str(opts.functional),
        omega=float(omega),
        grid_shape=grid_shape_t,
        smearing_temperature=smearing_T,
        fermi_level=float(fermi_level),
        entropy=float(entropy),
        free_energy=float(free_energy),
        occupations=[np.asarray(o, dtype=float) for o in occ_per_k],
        effective_lattice_opts=_copy_lattice_opts(lat_opts),
    )
