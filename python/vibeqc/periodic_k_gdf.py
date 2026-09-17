"""Native multi-k periodic RHF + RKS via Gaussian density fitting.

This module implements closed-shell multi-k SCF on arbitrary Bravais
lattices and Monkhorst-Pack k-meshes, pairing the cell-resolved C++
DF kernels (``compute_{2c,3c}_eri_lattice_blocks``) with Bloch
phase assembly (``build_lpq_bloch_native(q_cart)``) and a per-k
SCF loop reusing the multi-k DIIS machinery from the legacy Ewald
driver.

Coulomb is k-diagonal (built once from ``Lpq(q=0)`` against the
k-weighted density). Exchange is per-pair: ``K(k_i)`` accumulates
contributions from every ``k_j`` weighted by the mesh, contracted
through ``Lpq(k_i,k_j)`` and its Hermitian conjugate. The q-dependent
Coulomb metric and inverse square root are pre-computed once per unique
momentum transfer, while the three-centre tensor retains the required ket
Bloch phase and is built per pair. Storage is currently dense in memory, with
streaming queued for paper-grade systems where the
``nkpts^2 x naux x nao^2`` tensor exceeds host RAM.

Convention notes
----------------
* HF / hybrid exchange carries the ``exxdiv='ewald'`` Madelung
  correction: the multi-k branch applies the finite-k-mesh G=0
  divergence shift via the BvK-supercell constant
  (:func:`_madelung_for_kmesh`), and the Γ HF fast-path delegates to
  :func:`vibeqc.run_pbc_gdf_rhf` (primitive-cell shift). PySCF parity
  therefore uses ``exxdiv='ewald'`` (PySCF's default). The legacy
  molecular-limit driver :func:`vibeqc.run_rhf_periodic_gamma_gdf`
  (exxdiv=None semantics) remains the Γ fallback for RKS / dim<3 /
  charged / smeared / use_compcell runs.
* Generic Bravais: the driver uses arbitrary lattice vectors via
  the existing ``bloch_sum`` and Bloch-block helpers, with no
  cubic-only assumption. 1D / 2D / 3D periodicity all supported.
* Closed-shell RHF and RKS only in this module. UHF / UKS multi-k
  is a separate module (pending).
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

import os
import warnings
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    CoulombMethod,
    Functional,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    PeriodicXCDensityDomain,
    SCFAccelerator,
    SCFIteration,
    bloch_sum,
    build_grid,
    build_xc_periodic,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    ewald_nuclear_repulsion,
    level_shift_at_iter,
    nuclear_repulsion_per_cell,
)
from ._vibeqc_core import (
    direct_lattice_cells as _direct_cells,
)
from ._vibeqc_core import (
    monkhorst_pack as _mp_native,
)
from .aux_basis import (
    _accumulate_exchange_from_factors,
    _exchange_auxiliary_panel_rank,
    build_lpq_bloch_compcell,
    build_lpq_bloch_mdf,
    build_lpq_bloch_native_fft,
    default_aux_for,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from .guess import (
    _coerce_periodic_driver_guess,
    initial_densities_open_shell,
    initial_density_closed_shell,
    periodic_fock_guess_k,
    resolve_initial_guess,
)
from .kpoints import KPoints, _integer_counts
from .lattice_screening import (
    RcutStrategy,
    make_lattice_opts,
)
from .linear_dependence import (
    LinearDependenceError,
    PeriodicLinearDependenceSummary,
    raise_if_severe,
    scf_preflight_overlap_check,
)
from .madelung import apply_exxdiv_ewald_to_K
from .occupations import (
    hartree_to_kelvin_temperature as _hartree_to_kelvin_temperature,
)
from .periodic_k_density import (
    density_matrices_per_k as _density_from_orbitals,
)
from .periodic_k_density import (
    real_space_density_from_per_k_density as _real_space_density_from_per_k_density,
)
from .smearing import (
    SmearingOptions as _SmearingOptions,
    apply_smearing_open_shell as _apply_smearing_open_shell,
    closed_shell_periodic_occupations as _closed_shell_periodic_occupations,
    smeared_occupation_selfconsistency_tolerance as _smeared_occ_tol_fn,
)
from .smearing.apply import _global_aufbau_with_mu
from .options_dump import dump_active_settings
from .periodic_fock_multi_k import (
    _ewald_3d_lattice_j_cache_fits_memory_target,
    build_periodic_fock_ewald3d_k,
    build_periodic_j_ewald3d_k_from_k_density,
    make_ewald_3d_lattice_j_cache,
)
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_gdf import (
    PeriodicRHFGDFResult,
    _resolve_fock_mixing,
    _resolve_level_shift_warmup_cycles,
    run_rhf_periodic_gamma_gdf,
)
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _diag_in_orth_basis,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicSCFAccelerator,
)
from .periodic_screened_exchange import (
    reject_periodic_gdf_unsupported_functional,
    reject_unscreened_range_separated,
)
from .progress import ProgressLogger, resolve_progress

__all__ = [
    "PeriodicKRHFGDFResult",
    "PeriodicKRKSGDFResult",
    "run_krhf_periodic_gdf",
    "run_krks_periodic_gdf",
]


# =====================================================================
#              Basis-aware one-electron lattice-sum cutoff
# =====================================================================


def _is_pyscf_auto(strategy: object) -> bool:
    """True when ``strategy`` selects PySCF-style per-shell rcut tuning.

    Accepts the :class:`RcutStrategy` enum member, the lowercase string
    ``"pyscf_auto"`` (the driver default), or its ``repr`` form.
    """
    if strategy is None:
        return False
    if isinstance(strategy, RcutStrategy):
        return strategy is RcutStrategy.PYSCF_AUTO
    return str(strategy).strip().lower() in (
        "pyscf_auto",
        "rcutstrategy.pyscf_auto",
    )


def _gdf_oneel_cutoff_bound(system, basis, *, error=1e-12, pair_complete=False):
    """Bound the omitted 3D overlap and kinetic lattice sums per AO element.

    Absolute Gaussian envelopes retain contraction and angular factors.
    Their convolution is C exp(-beta*d**2). Split its decay between the
    outside-ball tail and a uniform lattice sum; no overlap eigenvalue is
    used as evidence of image convergence. Lippert et al. (1997), Sec. 4,
    motivates overlap-based image screening; the Gaussian product rule
    and kinetic Laplacian give the bounds used here.
    """
    from .aux_basis import _rsgdf_shell_envelopes, _rsgdf_log_lattice_gaussian_bound

    lattice = np.asarray(system.lattice, dtype=float)
    if (int(system.dim) != 3 or not np.isfinite(lattice).all()
            or not np.isfinite(error) or error <= 0):
        raise ValueError('GDF one-electron bound requires a finite 3D lattice and positive error')
    sigma = float(np.linalg.svd(lattice, compute_uv=False)[-1])
    if sigma <= 0:
        raise ValueError('GDF one-electron bound requires a nonsingular lattice')
    overlap = _rsgdf_shell_envelopes(basis)
    kinetic = []
    origins = []
    for shell in basis.shells():
        alphas = np.asarray(shell.exponents, dtype=float)
        coefficients = np.asarray(shell.coefficients, dtype=float)
        angular = int(shell.l)
        decay = float(alphas.min()) / 2
        terms = []
        origins.append(np.asarray(shell.origin, dtype=float))
        for alpha, coefficient in zip(alphas, coefficients):
            if coefficient == 0:
                continue
            # -Laplacian/2 of a solid-harmonic Gaussian. A Cartesian
            # monomial also has the bounded second derivative of its
            # polynomial, which vanishes for a harmonic polynomial.
            radial_terms = [(angular, alpha*(2*angular+3)),
                            (angular+2, 2*alpha**2)]
            if not shell.pure and angular >= 2:
                radial_terms.append((angular-2, .5*angular*(angular-1)))
            for degree, multiplier in radial_terms:
                power = degree / 2
                log_maximum = (0. if degree == 0 else
                               power*(np.log(power/(alpha-decay))-1))
                terms.append(np.log(abs(coefficient)) + np.log(multiplier) + log_maximum)
        kinetic.append((decay, float(np.logaddexp.reduce(terms, initial=-np.inf))))
    radius_squared = 0.
    for right in (overlap, kinetic):
        for a, ca in overlap:
            for b, cb in right:
                beta = a*b/(a+b)
                log_prefactor = ca + cb + 1.5*np.log(np.pi/(a+b))
                log_sum = _rsgdf_log_lattice_gaussian_bound(.2*beta, sigma)
                radius_squared = max(radius_squared,
                    (log_prefactor+log_sum-np.log(error))/(.8*beta))
    radius = np.sqrt(radius_squared)
    if not pair_complete:
        # A lattice-vector ball must cover the largest in-cell shell
        # displacement; a physical shell-pair ball already includes it.
        span = max(float(np.linalg.norm(a-b)) for a in origins for b in origins)
        radius += span
    if not np.isfinite(radius):
        raise ValueError('GDF one-electron cutoff is not representable')
    return float(np.nextafter(radius, np.inf))


def _gdf_oneel_memory_estimate(system, basis, lattice_opts, *, n_kpoints=1,
                              ecp_active=False, n_grid_points=0):
    """Conservative dense-set reservation before enumerating one-electron cells."""
    import math
    from .memory import MemoryEstimate

    lattice = np.asarray(system.lattice, dtype=float)
    inverse = np.linalg.inv(lattice)
    radius = float(lattice_opts.cutoff_bohr)
    if bool(lattice_opts.pair_complete_1e):
        origins = [np.asarray(sh.origin) for sh in basis.shells()]
        radius += max(float(np.linalg.norm(a-b)) for a in origins for b in origins)
    def box_count(reach):
        if not np.isfinite(reach) or reach < 0:
            raise ValueError('GDF one-electron image radius must be finite and nonnegative')
        return math.prod(2*math.ceil(float(np.linalg.norm(row))*reach)+3 for row in inverse)
    cell_capacity = box_count(radius)
    cells = cell_capacity
    if int(system.dim) == 3:
        # A matrix exists only for a retained point of the lattice sphere,
        # whereas direct_lattice_cells reserves its metadata for the index
        # box. Bound these separately without allocating either enumeration.
        # Centered fundamental parallelepipeds tile space without overlap;
        # every tile centered in the sphere lies inside radius R+sum(|a_i|)/2.
        # Hence N*volume <= 4*pi/3*(R+sum(|a_i|)/2)^3. Keep the box bound for
        # ill-conditioned/overflowing geometries where rounding margins fail.
        lengths = np.linalg.norm(lattice, axis=0)
        rounding = 64 * np.finfo(float).eps
        volume_lower = abs(float(np.linalg.det(lattice))) - rounding * math.prod(lengths)
        expanded = radius + .5 * float(sum(lengths))
        expanded = float(np.nextafter(expanded * (1 + rounding), np.inf))
        if volume_lower > 0 and np.isfinite(expanded):
            with np.errstate(over='ignore', invalid='ignore'):
                sphere_bound = float((4*np.pi/3) * np.float64(expanded)**3
                                     / volume_lower * (1 + rounding))
            if np.isfinite(sphere_bound):
                cells = min(cells, math.ceil(sphere_bound))
    nuclear_images = box_count(float(lattice_opts.nuclear_cutoff_bohr))
    nbf = int(basis.nbasis)
    chunk_mib = float(os.environ.get('VIBEQC_VNE_EWALD3D_FT_CHUNK_MIB', '256.0'))
    if not np.isfinite(chunk_mib) or chunk_mib <= 0:
        raise ValueError('GDF nuclear Fourier workspace must be finite and positive')
    # S/T/V and nuclear temporary/copy sets can overlap during setup.
    # ECP construction adds one retained set and streams projector images.
    return MemoryEstimate(by_category={
        'GDF one-electron lattice matrices and metadata':
            (8+int(ecp_active))*(cells*8*nbf**2 + cell_capacity*256),
        'GDF one-electron Bloch matrices': 16*8*int(n_kpoints)*nbf**2,
        'GDF nuclear images': 256*nuclear_images,
        'GDF nuclear Fourier workspace': int(chunk_mib*1024**2),
        'GDF retained quadrature': 40*int(n_grid_points),
    })


def _preflight_gdf_oneel_memory(system, basis, lattice_opts, *, n_kpoints=1,
                               ecp_active=False, grid=None, plog=None):
    from .memory import available_memory_bytes

    if int(system.dim) != 3:
        return
    estimate = _gdf_oneel_memory_estimate(
        system, basis, lattice_opts, n_kpoints=n_kpoints, ecp_active=ecp_active,
        n_grid_points=0 if grid is None else int(grid.n_points),
    )
    available = available_memory_bytes()
    if plog is not None:
        budget = str(available) if available > 0 else "unknown"
        plog.info(
            f"GDF one-electron setup reservation: {estimate.total_bytes} bytes "
            f"(including headroom); available budget={budget} bytes; "
            f"AO cutoff={lattice_opts.cutoff_bohr:.6g} bohr, "
            f"nuclear cutoff={lattice_opts.nuclear_cutoff_bohr:.6g} bohr"
        )
        for category, byte_count in estimate.by_category.items():
            plog.info(f"  {category}: {byte_count} bytes before headroom")
    if available > 0 and estimate.total_bytes > available:
        raise MemoryError(
            f'GDF one-electron setup requires an estimated {estimate.total_bytes} bytes; '
            f'available budget is {available}. AO images were not allocated.'
        )


def _oneel_lattice_opts(
    system, basis, base_opts, *, rcut_strategy, k_points_cart, plog=None,
):
    """Converge the bare 3D S/T image sums independently of metric signature.

    A truncated overlap can be positive and still be inaccurate: Gamma
    NaCl/LANL2DZ at 15 bohr has a 0.315 matrix error. The same resolved AO
    domain must reach the nuclear/ECP Hamiltonian and its derivatives.
    This bounds raw S/T tails, not an SCF energy or conditioning error.
    An explicit flat strategy retains the caller's finite-domain choice.
    """
    if not _is_pyscf_auto(rcut_strategy):
        return base_opts
    if int(system.dim) != 3:
        return _low_dim_oneel_lattice_opts(
            system, basis, base_opts, rcut_strategy=rcut_strategy,
            k_points_cart=k_points_cart, plog=plog,
        )
    grown = max(float(base_opts.cutoff_bohr), _gdf_oneel_cutoff_bound(
        system, basis, pair_complete=bool(base_opts.pair_complete_1e),
    ))
    if grown <= float(base_opts.cutoff_bohr):
        return base_opts
    widened = make_lattice_opts(basis, strategy=RcutStrategy.FLAT,
                               base_opts=base_opts, cutoff_bohr=grown)
    if plog is not None:
        plog.info(f'One-electron AO cutoff {base_opts.cutoff_bohr:.2f} -> '
                  f'{grown:.2f} bohr (absolute S/T image-tail bound 1e-12)')
    return widened


def _low_dim_oneel_lattice_opts(
    system: PeriodicSystem,
    basis: BasisSet,
    base_opts: LatticeSumOptions,
    *,
    rcut_strategy: object,
    k_points_cart: "np.ndarray",
    plog=None,
) -> LatticeSumOptions:
    """Historical non-PSD guard for the unchanged low-dimensional routes."""
    if not _is_pyscf_auto(rcut_strategy):
        return base_opts

    # Lazy import: eigs_preflight does `import vibeqc` at module load,
    # which re-enters the partially-initialised package while this module
    # is itself being imported by vibeqc/__init__. Deferring to call time
    # sidesteps that ordering hazard.
    from .eigs_preflight import optimize_truncation

    rep = optimize_truncation(
        system,
        basis,
        lattice_opts=base_opts,
        k_points_cart=k_points_cart,
        target_severity="error",
        cutoff_growth_factor=1.5,
        joint_growth=False,
    )
    if not rep.converged:
        # Still has critical negative directions: no cutoff below the cap fixed
        # the corrupted metric. Leave the caller's opts so the per-k preflight
        # aborts with the hint.
        return base_opts
    grown = float(rep.optimized_lattice_opts.cutoff_bohr)
    if grown <= float(base_opts.cutoff_bohr) + 1e-9:
        return base_opts  # already PSD at caller's cutoff (tight basis)
    widened = make_lattice_opts(
        basis,
        strategy=RcutStrategy.FLAT,
        base_opts=base_opts,
        cutoff_bohr=grown,
    )
    if plog is not None:
        plog.info(
            "S/T lattice cutoff grown "
            f"{float(base_opts.cutoff_bohr):.2f} -> {grown:.2f} bohr "
            f"(overlap non-critical at all k, severity={rep.final_severity}) "
            f"for diffuse basis '{basis.name}'"
        )
    return widened



# Appended to the LinearDependenceError message when a periodic S(k)
# preflight aborts, replacing the generic "upstream bug" framing with
# actionable remedies. Passed to scf_preflight_overlap_check by the
# multi-k GDF drivers below.
# Shared with every other periodic driver; see linear_dependence for the
# truncated-Bloch-sum explanation and the measured convergence data.
from .linear_dependence import PERIODIC_OVERLAP_HINT as _PERIODIC_OVERLAP_HINT


def _gdf_overlap_preflight(
    S: np.ndarray,
    *,
    plog,
    label: str,
    basis: BasisSet,
):
    """GDF-specific overlap policy after cutoff auto-growth.

    ``critical`` still aborts: the Bloch overlap has negative eigenvalues and
    the metric signature is wrong. ``error`` is allowed here because it is a
    PSD near-linear-dependence case, and the multi-k GDF driver immediately
    applies canonical orthogonalisation with ``linear_dep_threshold``.
    """
    rep = scf_preflight_overlap_check(
        S,
        plog=plog,
        label=label,
        basis=basis,
        remediation_hint=_PERIODIC_OVERLAP_HINT,
        raise_on_severe=False,
    )
    if rep.severity == "critical":
        try:
            raise_if_severe(rep, allow_warn=True, allow_critical=False)
        except LinearDependenceError as exc:
            raise LinearDependenceError(
                f"{exc.args[0]} {_PERIODIC_OVERLAP_HINT}",
                rep,
            ) from None
    if rep.severity == "error" and plog is not None:
        plog.info(
            f"[WARN] {label}: near-linear-dependent periodic overlap "
            "will be handled by canonical orthogonalisation"
        )
    return rep


# =====================================================================
#                              Result types
# =====================================================================


@dataclass
class PeriodicKRHFGDFResult:
    """Result of :func:`run_krhf_periodic_gdf`.

    All per-k arrays are length-``nkpts`` Python lists holding
    complex Hermitian (or real, for Γ-only) matrices in AO basis.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool

    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    density: List[np.ndarray]

    kpoints_cart: np.ndarray
    kpoint_weights: np.ndarray
    scf_trace: List[SCFIteration] = field(default_factory=list)

    functional: Optional[str] = None
    e_xc: float = 0.0
    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    e_dft_plus_u: float = 0.0

    fock_mixing: float = 0.0
    level_shift: float = 0.0
    level_shift_warmup_cycles: int = 0
    smearing_temperature: float = 0.0
    fermi_level: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations: List[np.ndarray] = field(default_factory=list)

    aux_basis_name: str = ""
    # What canonical orthogonalisation and the density fit actually
    # discarded. Populated by the drivers; rendered into the .out by
    # periodic_runner. None on routes that do not build it.
    linear_dependence: Optional["PeriodicLinearDependenceSummary"] = None
    n_aux: int = 0
    backend: str = "native-multi-k-gdf"
    #: RSGDF high-``|G|`` tail cutoff (Ha) actually APPLIED, or ``None``
    #: for base-mesh-only (GitLab IID 307). The Γ fast path auto-sizes one
    #: on the tight-core class and reports it here; the multi-k loop does
    #: not consume a tail at all (IID 146), so this stays ``None`` there
    #: even when ``rsgdf_tail_ke_cutoff`` was requested -- a requested but
    #: unapplied tail must not read back as applied.
    rsgdf_tail_ke_cutoff: Optional[float] = None
    #: The BASE rsgdf reciprocal mesh (Ha) actually used, so a consumer's
    #: ``*_executed`` field can be a real read rather than a copy of its own
    #: request (GitLab IID 307; see studies/aiccm-2026/run_case_cmp.py).
    rsgdf_ke_cutoff: float = 200.0

    # (n_atoms, 3) real analytic nuclear gradient in Ha/bohr, populated
    # when the driver is called with compute_gradient=True; None otherwise.
    gradient: Optional[np.ndarray] = None

    density_lattice_mesh: Optional[Tuple[int, int, int]] = None
    density_lattice_reserved_peak_bytes: int = 0
    density_lattice: Optional[LatticeMatrixSet] = None

    @property
    def energy_per_cell_ha(self) -> float:
        return float(self.energy)

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


@dataclass
class PeriodicKRKSGDFResult(PeriodicKRHFGDFResult):
    """Result of :func:`run_krks_periodic_gdf`."""


# =====================================================================
#                              Setup helpers
# =====================================================================


def _options_or_default(options, *, is_ks: bool):
    if options is not None:
        return options
    return PeriodicKSOptions() if is_ks else PeriodicRHFOptions()


@contextmanager
def _gamma_restart_options(options, initial_density_k, n_basis):
    """Transport a one-block restart without changing the Gamma Hamiltonian.

    Gamma drivers consume the prepared density on options. Restore caller
    fields even when SCF fails; selecting READ must not bypass these drivers.
    """
    if initial_density_k is None:
        yield options
        return
    from .guess_read import _real_density
    blocks = list(initial_density_k)
    if len(blocks) != 1:
        raise ValueError("READ: Gamma restart requires exactly one density block")
    density = _real_density(blocks[0])
    if density.shape != (n_basis, n_basis):
        raise ValueError("READ: Gamma density must match the basis")
    original_guess = options.initial_guess
    # The native matrix getter is a view; preserve its storage before assignment.
    original_density = np.array(options.read_density, copy=True)
    try:
        options.initial_guess = InitialGuess.READ
        options.read_density = density
        yield options
    finally:
        options.initial_guess = original_guess
        options.read_density = original_density


def _canonical_gdf_density_mixer(
    density_mixer: Optional[str],
) -> Optional[str]:
    """Collapse every spelling of the default Fock-DIIS route to None.

    The public mixer contract treats ``None``, ``""``, ``"none"``, and
    ``"diis"`` as aliases. Gamma fast-path selection must therefore see the
    same canonical value as the later SCF-accelerator setup; otherwise an
    alias changes the Hamiltonian route despite selecting no density mixer.
    Active Anderson/Broyden requests are returned unchanged.
    """
    key = None if density_mixer is None else str(density_mixer).strip().lower()
    if key in (None, "", "none", "diis"):
        return None
    return density_mixer


@dataclass(frozen=True)
class _GammaKMeshInfo:
    """Single-Γ k-mesh metadata for the native Γ fast path."""

    kpoints_cart: np.ndarray
    weights: np.ndarray
    input_n_kpoints: int


def _gamma_kmesh_info(
    system: PeriodicSystem,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh],
) -> Optional[_GammaKMeshInfo]:
    """Return metadata when ``kmesh`` is exactly a single Γ point.

    The KRHF/KRKS public entry points use multi-k-shaped APIs, but for
    a single Γ point the answer is delegated to the (fully native and
    well-tested) Γ-GDF driver. Anything else falls through into the
    real multi-k SCF.
    """
    if isinstance(kmesh, KPoints):
        kpoints = np.asarray(kmesh.kpoints_cart, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(kmesh.weights, dtype=np.float64).reshape(-1)
    elif isinstance(kmesh, BlochKMesh):
        kpoints = np.asarray(kmesh.kpoints, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(kmesh.weights, dtype=np.float64).reshape(-1)
    else:
        mesh = _mesh_tuple_for_system(system, kmesh)
        if mesh != (1, 1, 1):
            return None
        bm = _mp_native(system, [1, 1, 1], [0, 0, 0], False)
        kpoints = np.asarray(bm.kpoints, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(bm.weights, dtype=np.float64).reshape(-1)

    if kpoints.shape != (1, 3) or weights.shape != (1,):
        return None
    if not np.allclose(kpoints[0], 0.0, atol=1e-12, rtol=0.0):
        return None
    if not np.isclose(float(weights[0]), 1.0, atol=1e-12, rtol=0.0):
        return None
    return _GammaKMeshInfo(
        kpoints_cart=kpoints.copy(),
        weights=weights.copy(),
        input_n_kpoints=1,
    )


# Relative cut for the signed-Gram factorisation of a density. Modes
# below this fraction of the largest |eigenvalue| are round-off (an
# aufbau density's null space sits at ~1e-16 x n_occ), so dropping them
# changes K at machine precision while collapsing the factor rank from
# nbf to the occupied count -- which is the whole point. A fractional
# occupation of 1e-10 is still four orders above the cut and is kept.
_GRAM_FACTOR_RTOL = 1.0e-14

# Above this relative asymmetry a "density" is not Hermitian and the
# eigendecomposition below would silently read one triangle only. The
# dense contraction is then used verbatim.
_GRAM_HERMITICITY_RTOL = 1.0e-12


def _signed_gram_factors(
    density: np.ndarray,
) -> Optional[List[Tuple[float, np.ndarray]]]:
    """Factor a Hermitian ``D`` as ``S_t s_t W_t W_t^H``, ``s_t = +-1``.

    Returns ``None`` when ``D`` is not Hermitian to
    :data:`_GRAM_HERMITICITY_RTOL`, which tells the caller to fall back
    to the dense contraction rather than trust ``eigh`` (it reads a
    single triangle and would silently symmetrise the input).

    The **signed** split is what makes this exact for any Hermitian
    input rather than only for a physical density: negative eigenvalues
    become a second factor carried with ``s = -1``. A converged aufbau
    density is positive semi-definite so the negative block is empty,
    but a caller may legitimately pass a density *difference* or a
    slightly indefinite restart density, and those must not be silently
    projected onto their positive part.
    """
    D = np.asarray(density)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        return None
    scale = float(np.max(np.abs(D))) if D.size else 0.0
    if scale == 0.0:
        return []
    if float(np.max(np.abs(D - D.conj().T))) > _GRAM_HERMITICITY_RTOL * scale:
        return None
    lam, vec = np.linalg.eigh(D)
    cut = _GRAM_FACTOR_RTOL * float(np.max(np.abs(lam)))
    out: List[Tuple[float, np.ndarray]] = []
    for sign in (1.0, -1.0):
        keep = (lam * sign) > cut
        if not np.any(keep):
            continue
        out.append(
            (sign, vec[:, keep] * np.sqrt(lam[keep] * sign)[None, :])
        )
    return out


def _k_from_signed_factors(
    lpq_cache: Dict[Tuple[int, int], np.ndarray],
    factor_sets: Sequence[Sequence[Tuple[float, np.ndarray]]],
    k_weights: np.ndarray,
    rows: Sequence[int],
    *,
    nbasis: int,
    workspace_byte_cap: int = 32 * 1024**2,
) -> List[np.ndarray]:
    """The exchange contraction, evaluated on factored densities.

    ``K(k_i)_pq = S_j w_j S_t s_t S_P S_m A_Ppm conj(A_Pqm)`` with
    ``A_Ppm = S_r L(k_i,k_j)_Ppr W_t(k_j)_rm``. Each ``(k_i, k_j, t)``
    is two GEMM calls costing ``2 naux nbf^2 m`` -- against the density
    form's ``2 naux nbf^3`` -- and no ``L.conj()`` copy is materialised,
    because the conjugate rides the second GEMM's operand.

    Auxiliary panels bound the three transformed-factor copies, a
    possible contiguous copy of the input panel and two AO-matrix
    temporaries. Each panel uses BLAS for both contractions. This
    numerical workspace cap excludes retained factors, densities and
    returned matrices, and is not a whole-process RSS limit.

    **MPI seam (HANDOVER_MPI.md item 4c).** The ``j`` sum is a plain
    reduction and the bra loop over ``rows`` is independent, so the bra
    index is the distribution axis: each rank computes *complete*
    ``K(k_i)`` matrices for its own share and the results are reassembled
    in global order with :meth:`~vibeqc.mpi.KPointPartition.
    allgather_ordered`. Nothing is partially summed across ranks, so the
    farmed result is exact rather than merely close, and a rank needs
    only its own ``i`` against all ``j`` -- which is precisely the
    replicated cderi cache item 4 already leaves on every rank.

    Under a single rank the partition is the identity and the gather a
    no-op. Panels accumulate in ascending auxiliary-index order.
    """
    n_k = len(factor_sets)
    row_list = [int(i) for i in rows]

    from .mpi import KPointPartition
    from ._vibeqc_core import get_num_threads

    partition = KPointPartition.create(len(row_list), strategy="block")
    local_rows = [row_list[pos] for pos in partition.local_indices]
    max_rank = max((np.shape(w)[1] for per_k in factor_sets for _, w in per_k), default=0)
    # A single process-wide cap covers every simultaneous bra worker. This
    # includes panel copies, matrix temporaries and a small executor allowance;
    # retained input factors/densities and returned K matrices remain separate.
    minimum = 48 * nbasis**2 + 48 * nbasis * max_rank
    worker_overhead = 64 * 1024
    workers = min(len(local_rows), max(1, int(get_num_threads())),
                  max(1, int(workspace_byte_cap) // (minimum + worker_overhead)))
    per_worker_cap = int(workspace_byte_cap)
    if workers > 1:
        per_worker_cap = int(workspace_byte_cap) // workers - worker_overhead

    def _k_for_bra(i: int) -> np.ndarray:
        K_i = np.zeros((nbasis, nbasis), dtype=complex)
        for j in range(n_k):
            w_j = float(k_weights[j])
            facs = factor_sets[j]
            if w_j == 0.0 or not facs:
                continue
            Lpq = np.asarray(lpq_cache[(i, j)])
            naux = int(Lpq.shape[0])
            for sign, W in facs:
                W = np.asarray(W)
                if W.size == 0:
                    continue
                m = int(W.shape[1])
                panel_rank = _exchange_auxiliary_panel_rank(
                    naux, nbasis, 48 * nbasis * m + 16 * nbasis**2,
                    per_worker_cap,
                )
                for first in range(0, naux, panel_rank):
                    panel = Lpq[first:first + panel_rank]
                    count = len(panel)
                    flat = panel.reshape(count * nbasis, nbasis)
                    a = (flat @ W).reshape(count, nbasis, m)
                    a_pm = a.transpose(1, 0, 2).reshape(nbasis, count * m)
                    K_i += (w_j * float(sign)) * (a_pm @ a_pm.conj().T)
                    # Release before constructing the next panel; Python
                    # assignment otherwise keeps the old RHS alive too.
                    del flat, a, a_pm
        return 0.5 * (K_i + K_i.conj().T)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        # Pure numerical workers: no output channel or native OpenMP calls.
        # NumPy releases the GIL for GEMM; BLAS defaults to one thread at
        # package initialization, preserving k-point parallelism on a node.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            local = list(pool.map(_k_for_bra, local_rows))
    else:
        local = [_k_for_bra(i) for i in local_rows]
    K_out: List[np.ndarray] = partition.allgather_ordered(local)
    return K_out


def _k_from_densities_dense(
    lpq_cache: Dict[Tuple[int, int], np.ndarray],
    densities: Sequence[np.ndarray],
    k_weights: np.ndarray,
    rows: Sequence[int],
    *,
    nbasis: int,
    workspace_byte_cap: int = 32 * 1024**2,
) -> List[np.ndarray]:
    """The historical density-form contraction, kept as the reference.

    ``2 naux nbf^3`` per pair with bounded auxiliary panels. Retained
    as the fallback for a non-Hermitian "density" and as the fixed point
    the factored kernel is pinned against
    (``tests/test_periodic_k_gdf.py``): the two are the same operator
    reassociated, so they must agree to round-off forever.
    """
    K_out: List[np.ndarray] = []
    for i in rows:
        K_i = np.zeros((nbasis, nbasis), dtype=complex)
        for j in range(len(densities)):
            _accumulate_exchange_from_factors(
                K_i, lpq_cache[(i, j)], densities[j], k_weights[j],
                workspace_byte_cap,
            )
        K_out.append(0.5 * (K_i + K_i.conj().T))
    return K_out


def _build_k_from_densities(
    lpq_cache: Dict[Tuple[int, int], np.ndarray],
    densities: Sequence[np.ndarray],
    k_weights: np.ndarray,
    rows: Sequence[int],
    *,
    nbasis: int,
) -> List[np.ndarray]:
    """Factored exchange when every density factors, dense otherwise.

    The fallback is all-or-nothing on purpose: mixing the two
    contractions within one build would make the result depend on which
    k-point happened to be indefinite, which is a worse property than
    being uniformly slower.
    """
    factor_sets = [_signed_gram_factors(D) for D in densities]
    if any(f is None for f in factor_sets):
        return _k_from_densities_dense(
            lpq_cache, densities, k_weights, rows, nbasis=nbasis
        )
    return _k_from_signed_factors(
        lpq_cache, factor_sets, k_weights, rows, nbasis=nbasis
    )


def _build_k_from_lpq_cache(
    lpq_cache: Dict[Tuple[int, int], np.ndarray],
    densities: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    nbasis: int,
) -> List[np.ndarray]:
    """Exchange matrices from cached per-(k_i, k_j) density-fit tensors.

    The full Monkhorst-Pack grids built by the driver have uniform
    weights, but public ``KPoints`` inputs may be symmetry-reduced. The
    exchange BZ average therefore uses the supplied weight of the source
    density point ``k_j`` rather than assuming ``1 / n_k``.

    Since 2026-08-02 the contraction runs on the **occupied-index**
    (factored) form: each density is split into signed Gram factors
    (:func:`_signed_gram_factors`) whose rank is the number of occupied
    orbitals, so the cost per pair falls from ``2 naux nbf^3`` to
    ``2 naux nbf^2 n_occ``. This is the same operator reassociated, not
    an approximation -- it moves converged energies only at the
    reassociation level (~1e-13 Ha, measured). A non-Hermitian input
    falls back to :func:`_k_from_densities_dense` verbatim.
    """
    n_k = len(densities)
    k_weights = np.asarray(weights, dtype=float).reshape(-1)
    if k_weights.shape != (n_k,):
        raise ValueError(
            "_build_k_from_lpq_cache: weights length must match densities "
            f"({k_weights.shape[0]} != {n_k})"
        )
    return _build_k_from_densities(
        lpq_cache, densities, k_weights, range(n_k), nbasis=nbasis
    )


def _build_k_from_lpq_factors(
    lpq_cache: Dict[Tuple[int, int], np.ndarray],
    factors: Sequence[Sequence[np.ndarray]],
    weights: Sequence[float],
    *,
    nbasis: int,
) -> List[np.ndarray]:
    """Exchange from *factored* densities ``D(k_j) = S_t W_t W_t^H``.

    The entry point for a caller that already **has** the factors -- a
    restricted-open-shell SCF, whose occupied MO blocks are exactly the
    ``W`` this wants, so it never forms the density for the contraction
    at all. Callers holding only a density go through
    :func:`_build_k_from_lpq_cache`, which recovers equivalent factors
    from an eigendecomposition; the contraction itself is the same
    :func:`_k_from_signed_factors` either way. See that function for the
    algebra and the cost.

    ``K`` is a Gram sum of positive terms here, hence exactly Hermitian
    and positive semi-definite; the symmetrisation only removes
    round-off, matching the sibling builder's post-condition.

    Parameters
    ----------
    factors
        One list per k-point. Each entry is an ``(nbf, m)`` AO factor;
        their Gram sum is the density at that k. Several factors per k
        are allowed, so a convex mix of densities can be passed directly
        as ``sqrt(a) W`` blocks -- though a caller that mixes iterate
        after iterate should instead exploit the linearity of ``K`` and
        combine the *results*, or the factor rank grows every step.
    """
    n_k = len(factors)
    k_weights = np.asarray(weights, dtype=float).reshape(-1)
    if k_weights.shape != (n_k,):
        raise ValueError(
            "_build_k_from_lpq_factors: weights length must match factors "
            f"({k_weights.shape[0]} != {n_k})"
        )
    signed = [[(1.0, np.asarray(W)) for W in per_k] for per_k in factors]
    return _k_from_signed_factors(
        lpq_cache, signed, k_weights, range(n_k), nbasis=nbasis
    )


def _build_k_ibz_native(
    lpq_cache: Dict[Tuple[int, int], np.ndarray],
    densities_full: Sequence[np.ndarray],
    weights_full: Sequence[float],
    ibz_rows: Sequence[int],
    *,
    nbasis: int,
) -> List[np.ndarray]:
    """Exchange at the IBZ bras only, from a ``(IBZ x full)`` cderi cache.

    Exchange is the one ``n_k^2`` term in the multi-k GDF SCF: `K(k_i)`
    needs a sum over the WHOLE Brillouin zone for every bra ``k_i``. The
    symmetry saving is therefore on the *bra* index alone -- the wedge
    determines which ``K(k_i)`` must be built, never which ``k_j`` are
    summed. Reducing the ket sum as well is the mistake that measured
    +1.389 Ha on MgO (weighting representatives is not orbit summation).

    So this is the same contraction as
    :func:`_build_k_from_lpq_cache` -- literally the same kernel, given
    the wedge's bra rows instead of all of them -- evaluated on the
    ``len(ibz_rows) x n_k`` sub-block instead of the full ``n_k x n_k``.
    Given identical inputs it returns exactly the IBZ rows of the
    full-BZ result, which is what the parity regression pins.

    Parameters
    ----------
    lpq_cache
        Keyed ``(i, j)`` with ``i`` a **full-mesh** index that appears in
        ``ibz_rows`` and ``j`` running over the full mesh. Full-mesh
        indexing throughout so the cache is interchangeable with the
        full-BZ one.
    densities_full
        Densities at every full-mesh point -- the wedge densities
        unfolded through
        :func:`vibeqc.periodic_k_symmetry.expand_k_matrices_to_full`.
    weights_full
        Full-mesh BZ weights for the ket sum (uniform ``1/n_k`` on a
        regular mesh). NOT the IBZ weights.
    ibz_rows
        Full-mesh indices of the wedge representatives, in the order the
        caller wants the results.

    Returns
    -------
    list of ``(nbasis, nbasis)`` complex, one per entry of ``ibz_rows``.
    """
    n_k = len(densities_full)
    k_weights = np.asarray(weights_full, dtype=float).reshape(-1)
    if k_weights.shape != (n_k,):
        raise ValueError(
            "_build_k_ibz_native: weights_full length must match "
            f"densities_full ({k_weights.shape[0]} != {n_k})"
        )
    rows = [int(i) for i in ibz_rows]
    if any(not 0 <= i < n_k for i in rows):
        raise IndexError(
            f"_build_k_ibz_native: ibz_rows {rows} out of range for "
            f"n_k={n_k}"
        )

    return _build_k_from_densities(
        lpq_cache, densities_full, k_weights, rows, nbasis=nbasis
    )




def _ibz_symmetry_consistency(
    matrices_k, S_k, rows, star_map, system, basis, kmesh_full
):
    """Is this state actually invariant under the wedge's symmetry?

    IBZ-native exchange rebuilds ``K`` at the wedge and transports it, so
    it *imposes* the crystal symmetry on the solution. That is exactly
    right when the state is symmetric and silently wrong when it is not
    -- a variationally symmetry-broken UHF solution is a different state,
    not a less accurate one.

    Measuring the transport residual against zero would be useless,
    because a truncated Bloch sum is not exactly symmetric either: the
    per-atom lattice shifts of shift-bearing operations clip differently
    at the cell-list boundary. So this **self-calibrates** against the
    overlap, which is a pure one-electron object carrying the same
    truncation asymmetry and no state information at all. The overlap's
    residual is the floor; the density is compared to it.

    Returns ``(observed, floor)``, both relative.
    """
    from .periodic_k_symmetry import expand_k_matrices_to_full

    def _rel_residual(mats):
        unfolded = expand_k_matrices_to_full(
            [np.asarray(mats[i]) for i in rows],
            star_map,
            system,
            basis,
            kmesh_full,
        )
        worst = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(unfolded, mats)
        )
        scale = max(
            float(np.max(np.abs(np.asarray(b)))) for b in mats
        ) or 1.0
        return worst / scale

    return _rel_residual(matrices_k), _rel_residual(S_k)


def _require_ibz_symmetric_state(
    matrices_k, S_k, rows, star_map, system, basis, kmesh_full, *, label
):
    """Refuse IBZ-native when the state breaks the wedge's symmetry.

    Threshold. The overlap floor alone is too clean to calibrate
    against: ``S(k)`` is evaluated analytically once, while a converged
    density also carries SCF and DIIS noise, so on LiH FCC (2,2,2) at a
    26-bohr cutoff the symmetric singlet density sits at 3.1e-7 relative
    against an overlap floor of 2.4e-9 -- more than a hundred times the
    floor while being perfectly symmetric. The gate therefore uses an
    absolute relative tolerance as well, placed in the wide gap the
    measurements leave:

    * symmetric singlet density: **3.1e-7**
    * symmetry-broken triplet density: **1.8e-3**

    Nearly four orders apart at a converged cutoff, so ``1e-5`` sits far
    from both edges. The ``1000 x floor`` term relaxes the gate at loose
    cell lists, where truncation asymmetry alone would trip it: a
    *symmetric* density measures ~100x the overlap floor even when
    perfectly symmetric, because it also carries SCF and DIIS noise that
    the analytic overlap does not.

    **Known limitation, stated rather than hidden.** The diagnostic only
    separates cleanly once the cell list is converged. At 26 bohr the
    symmetric and broken densities differ by ~6000x (3.1e-7 vs 1.8e-3);
    at 20 bohr they are only ~3x apart (5.6e-4 vs 1.8e-3) and a broken
    state can slip through. That is the same cutoff regime the driver
    already warns about, so the gate is sharp exactly where the flag is
    trustworthy -- but do not read a pass at a loose cutoff as proof of
    a symmetric state.
    """
    observed, floor = _ibz_symmetry_consistency(
        matrices_k, S_k, rows, star_map, system, basis, kmesh_full
    )
    limit = max(1.0e-5, 1000.0 * floor)
    if observed > limit:
        raise NotImplementedError(
            f"ibz_native=True: the {label} does not respect the "
            "irreducible wedge's symmetry (relative transport residual "
            f"{observed:.2e} against an overlap-calibrated floor of "
            f"{floor:.2e})."
            + _ibz_residual_diagnosis(S_k)
            + " IBZ-native exchange IMPOSES that symmetry, so "
            "it would converge to a different -- symmetry-adapted -- "
            "state rather than the one this calculation is finding. That "
            "is a variational symmetry break, not an accuracy knob: on "
            "LiH FCC (2,2,2) the broken UHF triplet is 7e-4 Ha BELOW the "
            "symmetric solution and the gap is unchanged from conv_tol "
            "1e-8 to 1e-12. Run without ibz_native, or converge a "
            "symmetry-adapted state deliberately."
        )


def _ibz_residual_diagnosis(S_k, *, lindep_thr: float = 1e-9) -> str:
    """Name the mechanism when the wedge residual is not a real symmetry break.

    A large transport residual has (at least) two very different causes,
    and the message must not assert the wrong one. Besides a genuine
    variational symmetry break, the overlap itself can be the problem: an
    under-converged Bloch sum makes ``S(k)`` non-positive-definite, and
    canonical orthogonalisation then drops a **k-dependent** number of
    vectors. Star-related k-points whose retained counts differ live in
    different-dimensional subspaces, so NO symmetry transport can match
    them and the residual is large no matter how symmetric the state is.

    Measured on MgO rocksalt (3,3,3) -- same geometry, same mesh,
    symmorphic, only the basis changed::

        sto-3g   min eig(S(k)) in [3.5e-01, 4.1e-01]   retained [14] uniform
        6-31g    min eig(S(k)) in [-6.4e-07, 2.3e-05]  retained [20, 22] SPLIT

    The 6-31g run failed this gate at 2.95e-01 while its density was fine.
    That is the same defect family the Bloch-overlap PSD work tracks (see
    ``handovers/HANDOVER_GPW_MULTIK_OVERLAP_PSD.md`` and
    ``bloch_overlap_cutoff_bohr``), not an IBZ defect.

    Returns a sentence to splice into the refusal, or "" when the overlap
    looks healthy and a genuine symmetry break is the better explanation.
    """
    try:
        counts, worst = [], 0.0
        for S in S_k:
            M = np.asarray(S)
            ev = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
            counts.append(int(np.sum(ev > lindep_thr)))
            worst = min(worst, float(ev.min()))
        if len(set(counts)) <= 1:
            return ""
        return (
            f" NOTE: this is very likely NOT a symmetry break in your state."
            f" The Bloch overlap S(k) retains a k-DEPENDENT number of"
            f" vectors here ({sorted(set(counts))} across the mesh, worst"
            f" eigenvalue {worst:.2e}), so star-related k-points sit in"
            f" different-dimensional subspaces and no transport can match"
            f" them. That is an under-converged overlap lattice sum, not an"
            f" IBZ limitation: raise the one-electron lattice cutoff (or"
            f" use a less linearly-dependent basis) until the retained"
            f" count is uniform, then retry."
        )
    except Exception:
        return ""

def _resolve_ibz_native_state(system, kpoints_cart, kmesh, weights, plog):
    """Resolve the wedge, star map and full mesh for IBZ-native exchange.

    Returns ``(rows, star_map, kmesh_full)`` or ``None`` when the request
    cannot be honoured, having said why. Fails closed rather than
    silently running full-BZ, because a silent fallback would make a
    performance flag look like it worked while changing nothing.

    Preconditions, each of which would otherwise give a wrong or
    meaningless answer:

    * symmetry attached (`attach_symmetry`) -- there is no wedge without it;
    * a tuple/BlochKMesh whose full mesh the driver is actually running,
      so bra indices line up with `kpoints_cart`;
    * uniform weights -- a custom quadrature is not a Monkhorst-Pack mesh
      and its stars are not the spglib ones.

    **Accuracy precondition, deliberately a warning and not a hard gate:**
    the symmetry transport is only as good as the cell list. Measured on
    LiH FCC (2,2,2), the transport residual on K is 1.5e-3 at
    ``cutoff_bohr = 15`` (comparable to the energy's own drift to the
    next cutoff), 8.2e-6 at 20, and 1.1e-8 at 26. Below ~20 bohr the
    transport is no longer negligible against the truncation error, so
    the flag says so rather than pretending otherwise.
    """
    from ._vibeqc_core import monkhorst_pack as _mp_sym
    from .periodic_k_symmetry import star_operations

    if getattr(system, "symmetry", None) is None:
        raise NotImplementedError(
            "ibz_native=True requires crystal symmetry: call "
            "vibeqc.attach_symmetry(system) first. There is no "
            "irreducible wedge without it."
        )
    # Non-symmorphic groups are refused HERE, up front, so the failure
    # names the real cause. The symmetry transport applies a per-atom
    # lattice-shift Bloch phase, which covers a point operation followed
    # by a *lattice* translation. A non-symmorphic operation carries a
    # FRACTIONAL translation tau (a glide or screw), contributing a
    # further exp(-i k . tau) that this representation does not build.
    #
    # Without this check the mismatch surfaces later as a large density
    # transport residual, and `_require_ibz_symmetric_state` then reports
    # a "variational symmetry break" -- blaming the user's density for an
    # incompleteness in our own transport, and sending them to look for a
    # broken-symmetry SCF solution that is not there. Measured on Si
    # diamond, Fd-3m (No. 227): 24 of 48 operations carry a fractional
    # translation and the residual is 5.8e-01 against a 8.5e-05 floor,
    # while LiH rocksalt, Fm-3m (No. 225), has 0 of 48 and works.
    _ops = getattr(getattr(system, "symmetry", None), "operations", None) or ()
    _n_frac = 0
    for _op in _ops:
        _tau = getattr(_op, "translation", None)
        if _tau is None:
            continue
        _t = np.asarray(_tau, dtype=float).reshape(-1)
        if float(np.linalg.norm((_t + 0.5) % 1.0 - 0.5)) > 1.0e-8:
            _n_frac += 1
    if _n_frac:
        _sym = system.symmetry
        raise NotImplementedError(
            f"ibz_native=True does not support non-symmorphic space groups. "
            f"{getattr(_sym, 'international_symbol', '?')} "
            f"(No. {getattr(_sym, 'number', '?')}) has {_n_frac} of "
            f"{len(_ops)} operations carrying a fractional translation "
            f"(glide/screw). The wedge transport applies a per-atom "
            f"lattice-shift Bloch phase, which represents a point operation "
            f"plus a LATTICE translation; a fractional translation tau adds "
            f"an exp(-i k.tau) factor this implementation does not build, so "
            f"the transported density would be wrong rather than merely "
            f"imprecise. This is a limitation of ibz_native, NOT a problem "
            f"with your density or your convergence. Run without "
            f"ibz_native=True (the full-BZ path has no such restriction and "
            f"is correct here); symmorphic cells such as rocksalt Fm-3m are "
            f"unaffected."
        )

    mesh = getattr(kmesh, "mesh", None)
    if mesh is None and isinstance(kmesh, (list, tuple)):
        mesh = kmesh
    if mesh is None or len(tuple(mesh)) != 3:
        raise NotImplementedError(
            "ibz_native=True needs a Monkhorst-Pack mesh it can reduce "
            f"(tuple or BlochKMesh carrying `mesh`); got {type(kmesh).__name__}."
        )
    mesh = _integer_counts(mesh, name="IBZ native mesh")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if not np.allclose(w, w[0], atol=1e-12):
        raise NotImplementedError(
            "ibz_native=True expects the uniform weights of a full "
            "Monkhorst-Pack mesh; this mesh carries a custom quadrature, "
            "whose stars are not the spglib ones."
        )

    kfull = _mp_sym(system, mesh, [0, 0, 0], False)
    kibz = _mp_sym(system, mesh, [0, 0, 0], True)
    kf = [np.asarray(k, dtype=float).reshape(3) for k in kfull.kpoints]
    kpts = np.asarray(kpoints_cart, dtype=float).reshape(-1, 3)
    if len(kf) != kpts.shape[0]:
        raise NotImplementedError(
            f"ibz_native=True: the driver is running {kpts.shape[0]} "
            f"k-points but mesh {tuple(mesh)} has {len(kf)}; the bra "
            "indices would not line up."
        )
    # Match the driver's k-list to the reduced representatives MODULO a
    # reciprocal-lattice vector: a Bloch momentum is only defined up to
    # G, and spglib's representatives need not sit in the same BZ image
    # the driver chose. Comparing raw Cartesian distance rejects
    # perfectly good meshes -- (3,3,3) was refused that way.
    lattice = np.asarray(system.lattice, dtype=float)

    def _frac(k):
        # k = B . f with B the reciprocal lattice (columns b_i), so
        # f = B^-1 k = (A^T k) / 2pi for A the direct lattice.
        return (lattice.T @ np.asarray(k, dtype=float).reshape(3)) / (
            2.0 * np.pi
        )

    kpts_frac = [_frac(kpts[i]) for i in range(kpts.shape[0])]
    rows = []
    for kk in kibz.kpoints:
        f_ref = _frac(kk)
        d = [
            float(np.max(np.abs(((f - f_ref) + 0.5) % 1.0 - 0.5)))
            for f in kpts_frac
        ]
        i = int(np.argmin(d))
        if d[i] > 1e-8:
            raise NotImplementedError(
                "ibz_native=True: an irreducible representative is not "
                "present in the driver's k-list even modulo a "
                "reciprocal-lattice vector; refusing rather than "
                "guessing a correspondence."
            )
        rows.append(i)
    star_map = star_operations(system, kibz, kfull)
    # ORDER IS LOAD-BEARING: `star_map.entries` carries ``i_rep`` as an
    # index into ``kmesh_ibz.kpoints``, and `expand_k_matrices_to_full`
    # looks up ``M_k_ibz[i_rep]``. So the returned rows must stay in
    # kibz order -- sorting them silently matches each wedge matrix to
    # the wrong star. That bug is invisible on small meshes because
    # their kibz order is already ascending ((2,2,2) -> [0, 4, 6],
    # (3,3,3) -> [0, 9, 12, 21]) and only appears once it is not:
    # (4,4,4) is [0, 16, 32, 20, 36, 52, 40, 57], which sorting
    # permutes, and it cost 0.44 Ha before this was caught. Dedupe
    # order-preservingly rather than via set().
    seen = set()
    ordered_rows = []
    for i in rows:
        if i not in seen:
            seen.add(i)
            ordered_rows.append(i)
    return ordered_rows, star_map, kfull


@dataclass(frozen=True)
class _RangeSeparatedGdfCache(Mapping):
    factors: Dict[Tuple[int, int], np.ndarray]
    reserved_peak_bytes: int
    retained_factor_bytes: int
    reciprocal_vector_counts: Tuple[int, ...]
    q_batch_sizes: Tuple[Tuple[int, ...], ...] = ()
    source_parameters: tuple = ()
    source_signature: str = ""
    kpoints_cart: tuple = ()
    aux_basis: object = None
    memory_byte_cap: int = 0
    native_workspace_byte_cap: int = 0
    factor_frame: str = "eigenmode"

    # The SCF consumes the pair mapping; derivatives also need its source.
    # Keeping one object prevents dropping that provenance at the builder
    # boundary and reconstructing a different fit after convergence.
    def __getitem__(self, pair):
        return self.factors[pair]

    def __iter__(self):
        return iter(self.factors)

    def __len__(self):
        return len(self.factors)

    @property
    def n_fit(self):
        return int(self.factors[(0, 0)].shape[0])

    @property
    def k_cart_list(self):
        return self.kpoints_cart

    @property
    def retained_cache_bytes(self):
        """Factors and mapping metadata that coexist with a force rebuild."""
        return (self.retained_factor_bytes + 4096 + 512 * len(self.factors)
                + 256 * len(self.kpoints_cart))

    @property
    def lr_ke_cutoff(self):
        return float(self.source_parameters[3])


def _build_range_separated_lpq_cache(
    system,
    basis,
    auxiliary,
    kpoints_cart: np.ndarray,
    need_k_pairs: bool,
    *,
    omega: float,
    pair_cutoff: Optional[float] = None,
    auxiliary_cutoff: Optional[float] = None,
    ke_cutoff: Optional[float] = None,
    linear_dep_thr: float,
    memory_byte_cap: int,
    native_workspace_byte_cap: int,
    image_candidate_cap: int,
    reciprocal_candidate_cap: int,
    bra_rows: Optional[Sequence[int]] = None,
    integral_screen_error: float = 0.0,
    raw_integral_error: Optional[float] = None,
    pair_cache_builder=None,
) -> Mapping:
    """Build shared-q fits while reserving the complete retained cache.

    The cap is per process and includes the full-rank factor-cache bound,
    a conservative allowance for pair/group metadata and one live source
    build with whitening. Each q metric is built and diagonalized once.
    Supplying ``raw_integral_error`` plans finite SR and LR domains for
    the whole cache. Explicit cutoffs then serve as lower bounds, so a
    caller cannot silently undercut the requested raw-integral accuracy.
    Without an error target all three finite cutoffs must be explicit.
    This leaf does not distribute storage across MPI.

    A private pair-cache adapter may select and transport representatives.
    It receives this same admitted SR/LR source, with an optional canonical
    auxiliary frame, never the legacy reciprocal-only pair builder.
    A typed private MDF injection instead builds its own PW-aware cache on
    the resolved finite domains, preserving that cache's derivative identity.
    """
    from .aux_basis import (
        _RangeSeparatedGdfAdmissionError,
        _build_lpq_range_separated_shared_q,
        _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature,
        _plan_range_separated_gdf_cutoffs,
    )

    kpoints = np.asarray(kpoints_cart)
    if (kpoints.ndim != 2 or kpoints.shape[1] != 3 or len(kpoints) == 0
            or np.iscomplexobj(kpoints) or not np.isfinite(kpoints).all()):
        raise ValueError("range-separated GDF cache requires finite real k points of shape (nk,3)")
    n_k = len(kpoints)
    bras = range(n_k) if bra_rows is None else tuple(bra_rows)
    if (not bras or any(int(i) != i or i < 0 or i >= n_k for i in bras)
            or len(set(bras)) != len(bras)):
        raise ValueError("range-separated GDF bra rows must be distinct indices in the k mesh")
    # Hartree needs every diagonal, including bras outside the exchange
    # wedge. Count before materializing the pair/group bookkeeping.
    n_pairs = len(bras) * n_k + n_k - len(bras) if need_k_pairs else n_k
    factor_bound = 16 * n_pairs * int(auxiliary.nbasis) * int(basis.nbasis)**2
    metadata_bound = 4096 + 512 * n_pairs + 256 * n_k
    cache_reservation = factor_bound + metadata_bound
    if cache_reservation >= int(memory_byte_cap):
        raise MemoryError("range-separated GDF retained factor cache exceeds memory cap")
    build_cap = int(memory_byte_cap) - cache_reservation
    if need_k_pairs:
        pairs = [(int(i), j) for i in bras for j in range(n_k)]
        bra_set = set(bras)
        pairs.extend((i, i) for i in range(n_k) if i not in bra_set)
    else:
        pairs = [(i, i) for i in range(n_k)]
    groups = {}
    for i, j in pairs:
        q = _canonical_reciprocal_transfer(system, kpoints[j] - kpoints[i])
        key = tuple(float(value) for value in np.round(q, 14))
        groups.setdefault(key, (q, []))[1].append((i, j))
    if raw_integral_error is not None:
        # In the planner only the LR pair-image bound depends on q,
        # through the smallest fractional nonzero momentum. The SR and
        # reciprocal tails are uniform in q. Evaluate the zero-transfer
        # convention and the worst nonzero transfer, rather than repeating
        # the shell-triple bounds for every pair in a dense k mesh.
        plans = [_plan_range_separated_gdf_cutoffs(
            system, basis, auxiliary, np.zeros(3), omega=omega,
            raw_integral_error=raw_integral_error,
        )]
        lattice = np.asarray(system.lattice, dtype=float)
        closest_distance, closest_q = np.inf, None
        for q, _ in groups.values():
            fractional = lattice.T @ q / (2 * np.pi)
            distance = float(np.linalg.norm(fractional - np.rint(fractional)))
            if 1e-12 <= distance < closest_distance:
                closest_distance, closest_q = distance, q
        if closest_q is not None:
            plans.append(_plan_range_separated_gdf_cutoffs(
                system, basis, auxiliary, closest_q, omega=omega,
                raw_integral_error=raw_integral_error,
            ))
        def cutoff_floor(explicit, field):
            if explicit is not None and (not np.isfinite(explicit) or explicit <= 0):
                raise ValueError("range-separated GDF cutoff floors must be finite and positive")
            return max(float(explicit or 0), *(getattr(plan, field) for plan in plans))
        pair_cutoff = cutoff_floor(pair_cutoff, 'pair_cutoff')
        auxiliary_cutoff = cutoff_floor(auxiliary_cutoff, 'auxiliary_cutoff')
        ke_cutoff = cutoff_floor(ke_cutoff, 'ke_cutoff')
        screen_budget = min(plan.integral_screen_error for plan in plans)
        if not np.isfinite(integral_screen_error) or integral_screen_error < 0:
            raise ValueError("range-separated GDF screening error must be finite and nonnegative")
        integral_screen_error = min(integral_screen_error, screen_budget) if integral_screen_error else screen_budget
    elif any(cutoff is None for cutoff in (pair_cutoff, auxiliary_cutoff, ke_cutoff)):
        raise ValueError("range-separated GDF requires a raw error target or all three explicit cutoffs")
    from .periodic_mdf import _MdfScfSource, _build_mdf_cache

    if isinstance(pair_cache_builder, _MdfScfSource):
        if int(system.dim) != 3 or set(bras) != set(range(n_k)):
            raise NotImplementedError('private MDF SCF source requires a 3D full bra mesh')
        # The MDF builder admits the complete Gaussian/PW cache itself.
        # Drop planning bookkeeping before it constructs its own groups.
        del groups, pairs
        return _build_mdf_cache(
            system, basis, auxiliary, kpoints, need_k_pairs,
            plane_wave_cutoff=float(pair_cache_builder.plane_wave_cutoff),
            omega=omega, pair_cutoff=pair_cutoff, auxiliary_cutoff=auxiliary_cutoff,
            ke_cutoff=ke_cutoff, linear_dep_thr=linear_dep_thr,
            memory_byte_cap=memory_byte_cap, native_workspace_byte_cap=native_workspace_byte_cap,
            image_candidate_cap=image_candidate_cap, reciprocal_candidate_cap=reciprocal_candidate_cap,
            integral_screen_error=integral_screen_error, bra_rows=bras,
        )
    if pair_cache_builder is not None:
        # Reserve the full cache even when an adapter reconstructs some
        # members. Its AO/auxiliary transport needs live pair tensors and
        # transformation matrices alongside the current q eigensystem.
        transport_bytes = 64 * (
            int(auxiliary.nbasis) * int(basis.nbasis)**2 + int(auxiliary.nbasis)**2
        ) + 64 * 1024
        if transport_bytes >= build_cap:
            raise MemoryError('range-separated GDF pair-transport workspace exceeds memory cap')
        metric_state, current_frame = {}, None
        counts, sizes, frames = {}, {}, set()
        peak = cache_reservation + transport_bytes

        def build_pair(bra, ket, *, canonical_auxiliary_basis=False):
            nonlocal current_frame, peak
            frames.add(bool(canonical_auxiliary_basis))
            if len(frames) != 1:
                raise ValueError('range-separated GDF adapter mixed auxiliary coordinate frames')
            q = _canonical_reciprocal_transfer(system, np.asarray(ket)-np.asarray(bra))
            key = tuple(float(value) for value in np.round(q, 14))
            if key not in groups:
                raise ValueError('range-separated GDF adapter requested a transfer outside its mesh')
            # Keep only one metric state. An adapter may revisit a q,
            # which rebuilds that metric rather than retaining every q's
            # eigensystem outside the admitted storage model.
            frame = (key, bool(canonical_auxiliary_basis))
            if frame != current_frame:
                metric_state.clear()
                current_frame = frame
            batch = _build_lpq_range_separated_shared_q(
                system, basis, auxiliary, np.asarray(bra)[None, :], groups[key][0],
                omega=omega, pair_cutoff=pair_cutoff, auxiliary_cutoff=auxiliary_cutoff,
                ke_cutoff=ke_cutoff, linear_dep_thr=linear_dep_thr,
                memory_byte_cap=build_cap - transport_bytes,
                native_workspace_byte_cap=native_workspace_byte_cap,
                image_candidate_cap=image_candidate_cap,
                reciprocal_candidate_cap=reciprocal_candidate_cap,
                integral_screen_error=integral_screen_error, _metric_state=metric_state,
                canonical_auxiliary_basis=canonical_auxiliary_basis,
            )
            count = batch.reciprocal_vector_count
            if key in counts and counts[key] != count:
                raise RuntimeError('range-separated GDF adapter changed its reciprocal domain')
            counts[key] = count
            sizes.setdefault(key, []).append(1)
            peak = max(peak, cache_reservation + transport_bytes + batch.reserved_peak_bytes)
            return batch.factors[0]

        cache = pair_cache_builder(build_pair, kpoints, basis, auxiliary, need_k_pairs)
        # Numerical injection tests may provide an independently built fit
        # with its own finite source and response metadata. Preserve that
        # explicit identity instead of relabeling it with this policy.
        if isinstance(cache, _RangeSeparatedGdfCache):
            return cache
        if set(cache) != set(pairs):
            raise ValueError('range-separated GDF adapter returned a different pair mesh')
        for factors in cache.values():
            if (np.ndim(factors) != 3 or np.shape(factors)[1:] != (basis.nbasis, basis.nbasis)
                    or not 0 < np.shape(factors)[0] <= auxiliary.nbasis):
                raise ValueError('range-separated GDF adapter returned incompatible factor dimensions')
            flat = np.asarray(factors).reshape(-1)
            for first in range(0, flat.size, 4096):
                if not np.isfinite(flat[first:first+4096]).all():
                    raise ValueError('range-separated GDF adapter returned nonfinite factors')
        if not counts:
            raise ValueError('range-separated GDF adapter did not evaluate an admitted source')
        del metric_state
        return _RangeSeparatedGdfCache(
            cache, peak, sum(factor.nbytes for factor in cache.values()),
            tuple(counts[key] for key in sorted(counts)),
            tuple(tuple(sizes[key]) for key in sorted(sizes)),
            (float(omega), float(pair_cutoff), float(auxiliary_cutoff),
             float(ke_cutoff), float(linear_dep_thr), int(image_candidate_cap),
             int(reciprocal_candidate_cap), float(integral_screen_error)),
            _range_separated_gdf_source_signature(system, basis, auxiliary),
            tuple(tuple(float(v) for v in k) for k in kpoints),
            auxiliary, int(memory_byte_cap), int(native_workspace_byte_cap),
            'canonical' if True in frames else 'eigenmode',
        )
    cache = {}
    peak, retained, counts, batch_sizes = cache_reservation, 0, [], []
    for key in sorted(groups):
        q, group_pairs = groups[key]
        metric_state, sizes = {}, []
        first, batch_size, vector_count = 0, len(group_pairs), None
        while first < len(group_pairs):
            selected = group_pairs[first:first + batch_size]
            try:
                batch = _build_lpq_range_separated_shared_q(
                    system, basis, auxiliary,
                    np.asarray([kpoints[i] for i, _ in selected]), q,
                    omega=omega, pair_cutoff=pair_cutoff,
                    auxiliary_cutoff=auxiliary_cutoff, ke_cutoff=ke_cutoff,
                    linear_dep_thr=linear_dep_thr, memory_byte_cap=build_cap,
                    native_workspace_byte_cap=native_workspace_byte_cap,
                    image_candidate_cap=image_candidate_cap,
                    reciprocal_candidate_cap=reciprocal_candidate_cap,
                    integral_screen_error=integral_screen_error,
                    _metric_state=metric_state,
                )
            except _RangeSeparatedGdfAdmissionError as exc:
                # Retry only reservations that shrink with the k batch.
                # Fixed-q domains and actual allocator errors propagate.
                if not exc.retry_with_fewer_kpoints or len(selected) == 1:
                    raise
                batch_size = max(1, len(selected) // 2)
                continue
            if vector_count is not None and vector_count != batch.reciprocal_vector_count:
                raise RuntimeError("range-separated GDF q batches changed reciprocal domain")
            vector_count = batch.reciprocal_vector_count
            for pair, factor in zip(selected, batch.factors):
                cache[pair] = factor
            retained += batch.factors.nbytes
            peak = max(peak, cache_reservation + batch.reserved_peak_bytes)
            sizes.append(len(selected))
            first += len(selected)
            # Cache entries own views of this batch, without a second copy.
            # Only the current q's eigensystem/whitener survives the batch.
            del batch
        counts.append(vector_count)
        batch_sizes.append(tuple(sizes))
        del metric_state
    return _RangeSeparatedGdfCache(
        cache, peak, retained, tuple(counts), tuple(batch_sizes),
        (float(omega), float(pair_cutoff), float(auxiliary_cutoff),
         float(ke_cutoff), float(linear_dep_thr), int(image_candidate_cap),
         int(reciprocal_candidate_cap), float(integral_screen_error)),
        _range_separated_gdf_source_signature(system, basis, auxiliary),
        tuple(tuple(float(v) for v in k) for k in kpoints),
        auxiliary, int(memory_byte_cap), int(native_workspace_byte_cap),
    )


def _build_scf_range_separated_lpq_cache(
    system, basis, auxiliary, kpoints_cart, need_k_pairs, *,
    omega, raw_integral_error, ke_cutoff, linear_dep_thr, lat_opts,
    fit_screen_threshold, options, open_shell, progress, bra_rows=None,
    oneel_lattices=(), xc_grid=None, xc_cells=(), functional=None,
    xc_density_domain=PeriodicXCDensityDomain.AUTO,
    compute_gradient=False, density_return_bytes=0,
    pair_cache_builder=None,
):
    """Shared bulk SCF fit policy for Gamma and general k-point drivers.

    The raw source cap reserves the SCF matrices and accelerator history
    separately, with the normal process-memory headroom. Native scratch is
    bounded across all workers. The source never uses an exponent-scaled
    reciprocal tail or a dense AO-pair Fourier bundle.
    """
    from .memory import (
        available_memory_bytes, estimate_periodic_multik_gdf,
        estimate_periodic_xc_value, estimate_periodic_xc_gradient,
    )

    estimate = estimate_periodic_multik_gdf(
        n_basis=int(basis.nbasis), n_aux=int(auxiliary.nbasis),
        n_kpoints=len(kpoints_cart), need_k_pairs=need_k_pairs,
        open_shell=open_shell,
        n_ibz_kpoints=len(bra_rows) if bra_rows is not None else None,
        diis_subspace_size=(int(getattr(options, 'diis_subspace_size', 8))
                            if bool(getattr(options, 'use_diis', True)) else 0),
    )
    scf_bytes = sum(value for name, value in estimate.by_category.items()
                    if name != 'GDF Lpq factor cache')
    # These sets were constructed before the fit and remain live for initial
    # guesses and later consumers. Reading .blocks here would copy every
    # matrix through pybind11 just to measure it.
    retained_oneel = sum(
        len(lattice.cells) * (8 * int(lattice.nbf)**2 + 256)
        for lattice in oneel_lattices if lattice is not None
    )
    xc_bytes = 0
    if xc_grid is not None:
        from ._vibeqc_core import periodic_xc_domain_counts

        active_count, bra_count = periodic_xc_domain_counts(
            system, xc_cells, lat_opts, xc_density_domain,
        )
        xc_estimate = estimate_periodic_xc_value(
            n_basis=int(basis.nbasis), n_atoms=len(system.unit_cell),
            n_grid_points=int(xc_grid.n_points), n_cells=len(xc_cells),
            functional_kind=str(functional.kind), open_shell=open_shell,
            n_active_cells=active_count, n_bra_cells=bra_count,
        )
        xc_bytes = sum(value for name, value in xc_estimate.by_category.items()
                       if name != 'Python runtime + NumPy overhead')
        if bool(getattr(functional, 'is_external', False)):
            from .memory import (
                _external_xc_bridge_peak_bytes, _skala_xc_peak_bytes,
                _SKALA_FUNCTIONAL_NAMES,
            )

            if str(functional.name).lower() in _SKALA_FUNCTIONAL_NAMES:
                owners = np.asarray(xc_grid.atom_of_point, dtype=int)
                sizes = np.bincount(owners, minlength=len(system.unit_cell))
                xc_bytes += _skala_xc_peak_bytes(tuple(int(n) for n in sizes))
            else:
                xc_bytes += _external_xc_bridge_peak_bytes(int(xc_grid.n_points))
        elif compute_gradient:
            gradient_estimate = estimate_periodic_xc_gradient(
                n_basis=int(basis.nbasis), n_atoms=len(system.unit_cell),
                n_grid_points=int(xc_grid.n_points), n_cells=len(xc_cells),
                n_active_cells=active_count, n_bra_cells=bra_count,
                functional_kind=str(functional.kind), open_shell=open_shell,
            )
            gradient_bytes = sum(v for k, v in gradient_estimate.by_category.items()
                                 if k != 'Python runtime + NumPy overhead')
            xc_bytes = max(xc_bytes, gradient_bytes)
    scf_bytes += retained_oneel + xc_bytes + int(density_return_bytes)
    cap = int(available_memory_bytes() / estimate.headroom_factor) - scf_bytes
    if cap <= 0:
        raise MemoryError('periodic GDF SCF state exceeds the available memory budget')
    from ._vibeqc_core import get_num_threads, gdf_short_range_workspace_bytes

    requested_workspace = max(64 * 1024**2, gdf_short_range_workspace_bytes(
        basis, auxiliary, get_num_threads(),
    ))
    if compute_gradient:
        requested_workspace = max(requested_workspace, gdf_short_range_workspace_bytes(
            basis, auxiliary, get_num_threads(), 1, len(system.unit_cell),
        ))
    # The complete source also needs q tensors and whitening. If memory
    # constrains this reservation, native admission reduces its team using
    # the same linked-engine estimate, never a fixed thread ceiling.
    workspace = min(requested_workspace, cap // 8)
    result = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints_cart, need_k_pairs,
        omega=float(omega), raw_integral_error=float(raw_integral_error),
        pair_cutoff=float(lat_opts.cutoff_bohr),
        ke_cutoff=float(ke_cutoff) if ke_cutoff is not None else None,
        linear_dep_thr=float(linear_dep_thr), memory_byte_cap=cap,
        native_workspace_byte_cap=workspace,
        image_candidate_cap=10_000_000, reciprocal_candidate_cap=10_000_000,
        integral_screen_error=float(fit_screen_threshold), bra_rows=bra_rows,
        pair_cache_builder=pair_cache_builder,
    )
    if progress is not None:
        from .periodic_mdf import _MdfCache

        if isinstance(result, _MdfCache):
            source = dict(result.source_parameters[2])
            omega, pair, auxiliary_cutoff, ke = (
                source[key] for key in ('omega', 'pair_cutoff', 'auxiliary_cutoff', 'ke_cutoff'))
            label = f'private MDF (PW cutoff={result.source_parameters[1]:g} Ha; numerical acceptance pending)'
        else:
            omega, pair, auxiliary_cutoff, ke, *_ = result.source_parameters
            label = 'SR/LR GDF'
        progress.info(
            f"{label}: omega={omega:g}, AO images={pair:.2f} bohr, "
            f"auxiliary images={auxiliary_cutoff:.2f} bohr, LR cutoff={ke:.2f} Ha; "
            f"reserved source/cache peak={result.reserved_peak_bytes / 1024**2:.1f} MiB"
        )
    return result


def _build_rsgdf_lpq_cache_shared_q(
    system,
    basis,
    aux_modrho,
    kpoints_cart: np.ndarray,
    need_k_pairs: bool,
    *,
    ke_cutoff: float,
    tail_ke_cutoff: Optional[float] = None,
    lat_opts,
    linear_dep_thr: float,
    fit_screen_threshold: float,
    progress,
    q_metric_cache: Optional[dict],
    bra_rows: Optional[Sequence[int]] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """RSGDF per-pair Lpq cache with shared-q batched pair-FT passes.

    Groups the required ``(k_i, k_j)`` cderi pairs by their canonical
    momentum transfer ``q = k_j - k_i`` (the same 14-decimal key the
    q-metric cache shares state by) and builds each group through ONE
    :func:`vibeqc.aux_basis.build_lpq_bloch_native_fft_shared_q` call,
    so the dominant ket-Bloch pair-FT pass runs once per unique q
    instead of once per pair. On a regular full mesh that is ``n_k``
    passes for hybrid exchange (previously ``n_k^2``) and ONE pass for
    the diagonal-only J/COSX builds (previously ``n_k``). Exact work
    sharing — energies are invariant to floating-point rounding
    (regression: ``tests/test_rsgdf_shared_q_batch.py``).

    The per-q group loop below IS the MPI seam (handovers/HANDOVER_MPI.md
    step 4, "first production periodic k-point farming path"): whole
    q-groups are farmed across ranks via :class:`~vibeqc.mpi.
    KPointPartition` and reassembled with one ordered gather. Under a
    single rank the partition is the identity and the gather a no-op, so
    the serial path is byte-unchanged.
    """
    from .aux_basis import (
        _canonical_reciprocal_transfer,
        build_lpq_bloch_native_fft_shared_q,
    )
    from .mpi import KPointPartition

    n_k = int(np.asarray(kpoints_cart).shape[0])
    if need_k_pairs:
        # ``bra_rows`` restricts the BRA index only (IBZ-native
        # exchange): K(k_i) is needed at the wedge, but each one still
        # sums its ket over the whole zone, so the ket range is never
        # reduced. The diagonal pairs are always built for every k
        # because the Hartree fitted density consumes them full-mesh.
        bras = range(n_k) if bra_rows is None else [int(i) for i in bra_rows]
        pairs = [(i, j) for i in bras for j in range(n_k)]
        if bra_rows is not None:
            pairs += [(i, i) for i in range(n_k) if (i, i) not in set(pairs)]
    else:
        pairs = [(i, i) for i in range(n_k)]

    groups: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    for i, j in pairs:
        q = _canonical_reciprocal_transfer(
            system, np.asarray(kpoints_cart[j]) - np.asarray(kpoints_cart[i])
        )
        q_key = tuple(float(v) for v in np.round(q, 14))
        group = groups.setdefault(q_key, {"q": q, "pairs": []})
        group["pairs"].append((i, j))

    lpq_cache: Dict[Tuple[int, int], np.ndarray] = {}
    # Sorted so every MPI rank walks the groups in the same order. The
    # insertion order above is already deterministic given identical
    # inputs, but the farming below indexes groups by position, and an
    # explicit key is cheaper than trusting that invariant.
    group_list = [groups[key] for key in sorted(groups)]

    # ---- k-point farming over momentum-transfer groups ---------------
    # The groups are independent: each owns one q, builds its own metric
    # and pair-FT, and touches no shared state (the q-metric cache is
    # keyed by q, so distinct groups never collide). That makes this the
    # coarse-grained MPI seam named in handovers/HANDOVER_MPI.md -- one
    # partition, one gather per SCF *setup*, nothing per iteration, which
    # is what keeps it viable on Ethernet.
    #
    # Replicated-data model (the handover's phase-1 decision): every rank
    # ends up with the complete cderi cache, so the SCF loop that follows
    # is untouched. That trades memory for simplicity -- it parallelises
    # the build time, not the footprint; distributing the cache itself is
    # separate, later work.
    #
    # Serially (and whenever MPI is inactive) the partition is the
    # identity and the gather is a no-op, so this is exactly the previous
    # loop.
    partition = KPointPartition.create(len(group_list), strategy="block")
    local_built = []
    for group_index in partition.local_indices:
        group = group_list[group_index]
        group_pairs = group["pairs"]
        k_bras = np.asarray(
            [kpoints_cart[i] for i, _ in group_pairs], dtype=float
        )
        lpq_list = build_lpq_bloch_native_fft_shared_q(
            system,
            basis,
            aux_modrho,
            k_bras,
            group["q"],
            ke_cutoff=float(ke_cutoff),
            tail_ke_cutoff=(
                float(tail_ke_cutoff) if tail_ke_cutoff is not None else None
            ),
            lat_opts=lat_opts,
            linear_dep_thr=float(linear_dep_thr),
            fit_screen_threshold=float(fit_screen_threshold),
            progress=progress,
            _q_metric_cache=q_metric_cache,
        )
        local_built.append((group_pairs, lpq_list))

    for group_pairs, lpq_list in partition.allgather_ordered(local_built):
        for pair, lpq in zip(group_pairs, lpq_list):
            lpq_cache[pair] = lpq
    return lpq_cache


def _wrap_gamma_gdf_result(
    gamma: PeriodicRHFGDFResult,
    info: _GammaKMeshInfo,
    *,
    functional: Optional[str],
    result_cls,
):
    """Adapt the native Γ-GDF result to the KRHF/KRKS result shape.

    Propagates the inner driver's ``+PARITY_HELD`` backend marker instead
    of overwriting it with the bare adapter literal: a dense-core hold
    computed by ``run_pbc_gdf_rhf`` (or marked by the legacy-fallback
    caller) must survive into the public result, or a pipeline reading
    ``result.backend`` silently sees an un-held absolute energy.
    """
    from .pbc_gdf import _gdf_backend_with_parity_hold

    _inner_backend = str(getattr(gamma, "backend", "") or "")
    _backend = _gdf_backend_with_parity_hold(
        "native-gamma-gdf-via-k-gdf",
        "+PARITY_HELD" in _inner_backend,
    )
    return result_cls(
        guess_selection=getattr(gamma, "guess_selection", None),
        restart_basis=getattr(gamma, "restart_basis", None),
        restart_lattice=getattr(gamma, "restart_lattice", None),
        restart_kpoints=np.asarray(info.kpoints_cart).copy(),
        restart_weights=np.asarray(info.weights).copy(),
        energy=float(gamma.energy),
        e_electronic=float(gamma.e_electronic),
        e_nuclear=float(gamma.e_nuclear),
        n_iter=int(gamma.n_iter),
        converged=bool(gamma.converged),
        mo_energies=[np.asarray(gamma.mo_energies)],
        mo_coeffs=[np.asarray(gamma.mo_coeffs)],
        fock=[np.asarray(gamma.fock)],
        overlap=[np.asarray(gamma.overlap)],
        hcore=[np.asarray(gamma.hcore)],
        density=[np.asarray(gamma.density)],
        kpoints_cart=np.asarray(info.kpoints_cart, dtype=np.float64),
        kpoint_weights=np.asarray(info.weights, dtype=np.float64),
        scf_trace=list(gamma.scf_trace),
        functional=functional or str(getattr(gamma, "functional", "") or "") or None,
        e_xc=float(getattr(gamma, "e_xc", 0.0)),
        e_coulomb=float(getattr(gamma, "e_coulomb", 0.0)),
        e_hf_exchange=float(getattr(gamma, "e_hf_exchange", 0.0)),
        e_dft_plus_u=float(getattr(gamma, "e_dft_plus_u", 0.0)),
        fock_mixing=float(getattr(gamma, "fock_mixing", 0.0)),
        level_shift=float(getattr(gamma, "level_shift", 0.0)),
        level_shift_warmup_cycles=int(getattr(gamma, "level_shift_warmup_cycles", 0)),
        smearing_temperature=float(getattr(gamma, "smearing_temperature", 0.0)),
        fermi_level=float(getattr(gamma, "fermi_level", 0.0)),
        entropy=float(getattr(gamma, "entropy", 0.0)),
        free_energy=float(getattr(gamma, "free_energy", gamma.energy)),
        occupations=[np.asarray(getattr(gamma, "occupations", np.empty(0)))],
        aux_basis_name=str(getattr(gamma, "aux_basis_name", "") or ""),
        n_aux=int(getattr(gamma, "n_aux", 0)),
        backend=_backend,
        rsgdf_tail_ke_cutoff=getattr(gamma, "rsgdf_tail_ke_cutoff", None),
        rsgdf_ke_cutoff=float(getattr(gamma, "rsgdf_ke_cutoff", 200.0)),
        # The pure PBC-GDF Γ fast path computes the analytic gradient
        # itself (compute_gradient passthrough); the legacy gamma driver
        # has no gradient attribute and leaves this None.
        gradient=getattr(gamma, "gradient", None),
    )


def _clone_lattice_options(source: LatticeSumOptions) -> LatticeSumOptions:
    """Copy the pybind lattice-options value object field by field."""
    target = LatticeSumOptions()
    for name in (
        "becke_image_radius_bohr",
        "cutoff_bohr",
        "nuclear_cutoff_bohr",
        "schwarz_threshold",
        "schwarz_threshold_forces",
        "screening_exchange_threshold",
        "screening_overlap_threshold",
        "slab_ewald_alpha",
        "sr_range_screening",
        "sr_sparse_traversal",
        "pair_complete_1e",
        "eri_interaction_cutoff_bohr",
    ):
        setattr(target, name, getattr(source, name))
    target.coulomb_method = source.coulomb_method
    return target


def _clone_slab_lattice_options(source: LatticeSumOptions) -> LatticeSumOptions:
    """Copy lattice controls while selecting the rigorous 2D Coulomb gauge."""
    target = _clone_lattice_options(source)
    target.coulomb_method = CoulombMethod.SLAB_EWALD_2D
    return target


def _run_closed_shell_slab_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh],
    opts: Union[PeriodicRHFOptions, PeriodicKSOptions],
    *,
    functional: Optional[str],
    aux_basis: Optional[str],
    aux_drop_eta: float,
    linear_dep_threshold: float,
    gdf_linear_dep_threshold: float,
    apply_modrho: bool,
    fock_mixing_value: float,
    level_shift_warmup_cycles: Optional[int],
    use_compcell: bool,
    apply_aft_correction: bool,
    aft_ft_convention: str,
    aft_precision: float,
    rcut_strategy: Optional[object],
    rcut_precision: float,
    k_exchange: str,
    gdf_method: str,
    rsgdf_ke_cutoff: float,
    rsgdf_tail_ke_cutoff: Optional[float],
    fit_screen_threshold: float,
    bz_integration: Optional[str],
    density_mixer: Optional[str],
    dft_plus_u_sites: Optional[Sequence[object]],
    initial_density_k: Optional[Sequence[np.ndarray]],
    check_energy_sanity: bool,
    progress: Union[bool, ProgressLogger, None],
    verbose: Optional[int],
    compute_gradient: bool = False,
) -> PeriodicKRHFGDFResult:
    """Public adapter for the bounded closed-shell slab-GDF envelope."""
    if isinstance(kmesh, (KPoints, BlochKMesh)):
        raise NotImplementedError(
            "slab GDF currently requires a full Gamma-centered tuple mesh; "
            "custom, shifted, weighted, and symmetry-reduced meshes remain "
            "fail-closed"
        )
    mesh_raw = _integer_counts(kmesh, name="slab GDF mesh")
    if len(mesh_raw) == 2:
        mesh_raw.append(1)
    if len(mesh_raw) != 3 or any(value < 1 for value in mesh_raw):
        raise ValueError(
            "slab GDF requires a positive integer (n1,n2,1) mesh"
        )
    mesh = tuple(mesh_raw)
    if mesh[2] != 1:
        raise ValueError("slab GDF requires kmesh=(n1,n2,1)")
    if float(aux_drop_eta) != 0.0:
        raise NotImplementedError(
            "slab GDF does not yet implement auxiliary primitive culling"
        )
    if not apply_modrho:
        raise NotImplementedError("slab GDF requires apply_modrho=True")
    if fock_mixing_value != 0.0:
        raise NotImplementedError("slab GDF does not yet implement fock_mixing")
    if float(getattr(opts, "level_shift", 0.0) or 0.0) != 0.0:
        raise NotImplementedError("slab GDF does not yet implement level_shift")
    if level_shift_warmup_cycles not in (None, 0):
        raise NotImplementedError(
            "slab GDF does not yet implement level-shift warmup cycles"
        )
    if use_compcell:
        raise NotImplementedError(
            "slab GDF uses its dedicated signed truncated metric, not compcell"
        )
    if not apply_aft_correction:
        raise NotImplementedError(
            "slab GDF fixes its finite 2D reciprocal metric convention; "
            "apply_aft_correction=False is not defined for this route"
        )
    if str(aft_ft_convention).strip().lower() != "libint":
        raise NotImplementedError(
            "slab GDF currently fixes aft_ft_convention='libint'"
        )
    if float(aft_precision) != 1e-10:
        raise NotImplementedError(
            "slab GDF does not expose the bulk aft_precision control"
        )
    if not _is_pyscf_auto(rcut_strategy) or float(rcut_precision) != 1e-8:
        raise NotImplementedError(
            "slab GDF does not yet expose bulk rcut_strategy/rcut_precision "
            "controls; set lattice_opts cutoffs explicitly"
        )
    if k_exchange != "gdf":
        raise NotImplementedError(
            "slab GDF currently implements fitted full-range exchange only; "
            "COSX remains fail-closed"
        )
    if gdf_method != "rsgdf":
        raise NotImplementedError(
            "slab GDF uses the dedicated signed truncated metric; the public "
            "adapter currently accepts gdf_method='rsgdf' only"
        )
    if rsgdf_tail_ke_cutoff is not None:
        raise NotImplementedError(
            "slab GDF does not yet implement the bulk high-|G| tail correction"
        )
    if float(fit_screen_threshold) != 0.0:
        raise NotImplementedError(
            "slab GDF fit_screen_threshold is not yet implemented"
        )
    if bz_integration is not None:
        raise NotImplementedError(
            "slab GDF currently supports integer zero-temperature "
            "occupations only"
        )
    mixer_key = (
        None if density_mixer is None else str(density_mixer).strip().lower()
    )
    if mixer_key not in (None, "", "none", "diis"):
        raise NotImplementedError(
            "slab GDF currently supports Fock DIIS only; Anderson, Broyden, "
            "and Kerker density mixing remain fail-closed"
        )
    if dft_plus_u_sites:
        raise NotImplementedError("slab GDF does not yet implement DFT+U")
    if initial_density_k is not None or getattr(
        opts, "initial_guess", InitialGuess.HCORE
    ) == InitialGuess.READ:
        raise NotImplementedError(
            "slab GDF does not yet implement per-k density restart"
        )

    from .periodic_rhf_gdf import _run_krhf_periodic_slab_gdf

    original_lattice_opts = _clone_lattice_options(opts.lattice_opts)
    slab_lat_opts = _clone_slab_lattice_options(original_lattice_opts)
    opts.lattice_opts = slab_lat_opts
    try:
        slab = _run_krhf_periodic_slab_gdf(
            system,
            basis,
            mesh,
            opts,
            functional=functional,
            aux_basis=aux_basis,
            ke_cutoff=float(rsgdf_ke_cutoff),
            linear_dep_threshold=linear_dep_threshold,
            gdf_linear_dep_threshold=gdf_linear_dep_threshold,
            progress=progress,
            verbose=verbose,
        )
    finally:
        opts.lattice_opts = original_lattice_opts

    # One global Fermi level across the whole 2D mesh -- the same contract
    # 60104fc01 established for the 3D native multi-k GDF route, which did
    # not cover this slab adapter. Writing ``2.0`` into the first ``n_occ``
    # slots at *every* k independently is a per-k Aufbau, and it
    # mis-classifies any system whose bands cross between k points: the hBN
    # sto-3g slab reported a VBM at k_idx 0 lying 3.270 eV *above* its
    # "CBM" at k_idx 13 purely because both edges were read off a per-k
    # occupation index rather than off one chemical potential.
    occupations, fermi_level = _global_aufbau_with_mu(
        [np.asarray(np.real(eps), dtype=float) for eps in slab.mo_energies],
        np.asarray(slab.kpoint_weights, dtype=float),
        float(system.n_electrons()),
        occ_value=2.0,
    )
    result_cls = PeriodicKRKSGDFResult if functional else PeriodicKRHFGDFResult
    result = result_cls(
        guess_selection=getattr(slab, "guess_selection", None),
        restart_basis=getattr(slab, "restart_basis", None),
        restart_lattice=getattr(slab, "restart_lattice", None),
        restart_kpoints=np.asarray(slab.kpoints_cart).copy(),
        restart_weights=np.asarray(slab.kpoint_weights).copy(),
        energy=float(slab.energy),
        e_electronic=float(slab.e_electronic),
        e_nuclear=float(slab.e_nuclear),
        n_iter=int(slab.n_iter),
        converged=bool(slab.converged),
        mo_energies=list(slab.mo_energies),
        mo_coeffs=list(slab.mo_coeffs),
        fock=list(slab.fock),
        overlap=list(slab.overlap),
        hcore=list(slab.hcore),
        density=list(slab.density),
        kpoints_cart=np.asarray(slab.kpoints_cart, dtype=float),
        kpoint_weights=np.asarray(slab.kpoint_weights, dtype=float),
        scf_trace=list(slab.scf_trace),
        functional=functional,
        e_xc=float(slab.e_xc),
        e_coulomb=float(slab.e_coulomb),
        e_hf_exchange=float(slab.e_hf_exchange),
        fock_mixing=0.0,
        level_shift=0.0,
        smearing_temperature=0.0,
        fermi_level=fermi_level,
        entropy=0.0,
        free_energy=float(slab.energy),
        occupations=occupations,
        aux_basis_name=str(slab.aux_basis_name),
        n_aux=int(slab.n_aux),
        backend=(
            "native-multik-slab-truncated-gdf-rks"
            if functional
            else "native-multik-slab-truncated-gdf-rhf"
        ),
    )
    if compute_gradient and result.converged:
        # G-PBC-002 § 6 rung 5: the slab analytic gradient on the
        # converged D(k)/C(k)/eps(k). The adapter (not the SCF driver)
        # assembles it, handing the SCF's exact provenance -- the slab
        # lattice options actually run under, the fit's
        # ke_cutoff/threshold, the identically rebuilt modrho aux basis,
        # and (for KS) the identically rebuilt quadrature -- to the
        # FD-gated assemblers (the e_nuc gauge lesson: differentiate
        # the energy the SCF actually converged).
        from .aux_basis import _slab_probe_charge_madelung_for_kmesh
        from .periodic_gdf_gradient import (
            _build_slab_gdf_gradient_cache,
            _compute_krhf_gradient_slab_gdf,
            _compute_krks_gradient_slab_gdf,
        )

        grad_plog = resolve_progress(progress, verbose=verbose)
        kpts = np.asarray(slab.kpoints_cart, dtype=float)
        k_weights = np.asarray(slab.kpoint_weights, dtype=float)
        molecule = system.unit_cell_molecule()
        aux_name = aux_basis or default_aux_for(basis.name)
        raw_aux = make_aux_basis_set(molecule, aux_name=aux_name)
        aux_modrho = make_modrho_aux_basis(raw_aux, molecule)
        grad_cache = _build_slab_gdf_gradient_cache(
            system,
            basis,
            aux_modrho,
            kpts,
            ke_cutoff=float(rsgdf_ke_cutoff),
            lat_opts=slab_lat_opts,
            linear_dep_thr=float(gdf_linear_dep_threshold),
        )
        grad_plog.info(
            "analytic gradient: slab signed-fit rebuilt "
            f"({len(grad_cache.groups)} q group(s))"
        )
        D_g = [np.asarray(D) for D in slab.density]
        S_g = [np.asarray(S) for S in slab.overlap]
        # Closed-shell zero-temperature energy-weighted density. The
        # occupations are the SAME global single-Fermi-level fill that built
        # D above (the slab route supports no finite-T smearing), so W and D
        # stay consistent: an energy-weighted density built from a per-k
        # ``[:n_occ]`` slice while D came from a global fill would
        # differentiate a different function than the SCF minimised.
        W_g = []
        for ik in range(len(D_g)):
            C_k = np.asarray(slab.mo_coeffs[ik])
            eps_k = np.asarray(np.real(slab.mo_energies[ik]), dtype=float)
            occ_k = np.asarray(occupations[ik], dtype=float)
            W_g.append((C_k * (occ_k * eps_k)[None, :]) @ C_k.conj().T)
        # BvK probe-charge constant exactly as the SCF's exxdiv K-shift
        # resolves it (pure lattice functional, d xi/dR = 0; unused at
        # alpha_hf = 0, where the SCF never computes it either).
        alpha_hf = (
            float(Functional(functional, 1).hf_exchange_fraction)
            if functional
            else 1.0
        )
        xi_bvk = (
            _slab_probe_charge_madelung_for_kmesh(system, mesh)
            if alpha_hf != 0.0
            else 0.0
        )
        if functional:
            # Rebuild the driver's exact quadrature + kmesh objects
            # (deterministic builders on identical inputs).
            grid_options = getattr(opts, "grid", None) or GridOptions()
            if bool(getattr(opts, "use_periodic_becke", False)):
                xc_grid = build_periodic_becke_grid(
                    system,
                    grid_options=grid_options,
                    image_radius_bohr=float(
                        getattr(opts, "becke_image_radius_bohr", 0.0)
                    ),
                )
            else:
                xc_grid = build_grid(molecule, grid_options)
            kmesh_bloch = _mp_native(system, list(mesh), [0, 0, 0], False)
            result.gradient = _compute_krks_gradient_slab_gdf(
                system,
                basis,
                D_g,
                W_g,
                S_g,
                k_weights,
                kpts,
                grad_cache,
                functional=functional,
                xc_grid=xc_grid,
                kmesh_bloch=kmesh_bloch,
                bvk_probe_charge_madelung=xi_bvk,
                lat_opts=slab_lat_opts,
            )
        else:
            result.gradient = _compute_krhf_gradient_slab_gdf(
                system,
                basis,
                D_g,
                W_g,
                S_g,
                k_weights,
                kpts,
                grad_cache,
                alpha_hf=alpha_hf,
                bvk_probe_charge_madelung=xi_bvk,
                lat_opts=slab_lat_opts,
            )
        grad_plog.info(
            "analytic gradient: slab GDF assembly done "
            f"({'KRKS ' + functional if functional else 'KRHF'}, "
            f"{len(D_g)} k-point(s))"
        )
    if check_energy_sanity:
        result_plog = resolve_progress(progress, verbose=verbose)
        _check_energy_sanity(result, system, result_plog)
    return result


def _mesh_tuple_for_system(
    system: PeriodicSystem,
    mesh: Union[Sequence[int], KPoints, BlochKMesh],
) -> Tuple[int, int, int]:
    dim = int(system.dim)
    if dim not in (1, 2, 3):
        raise ValueError(f"PeriodicSystem.dim must be 1, 2, or 3; got {dim}")
    if isinstance(mesh, (KPoints, BlochKMesh)):
        mesh_metadata = getattr(mesh, "mesh", None)
        if mesh_metadata is None:
            raise ValueError(
                "periodic GDF: this operation requires structured k-mesh "
                "metadata; explicit unstructured k-points are insufficient"
            )
        arr = _integer_counts(mesh_metadata, name="periodic GDF kmesh")
    else:
        arr = _integer_counts(mesh, name="periodic GDF kmesh")
    if len(arr) == dim:
        arr = arr + [1] * (3 - dim)
    elif len(arr) != 3:
        raise ValueError(
            f"periodic GDF: kmesh tuple must have length {dim} for "
            f"dim={dim} systems or length 3; got {arr!r}"
        )
    out = tuple(arr)
    if any(x < 1 for x in out):
        raise ValueError(f"periodic GDF: kmesh entries must be >= 1; got {arr!r}")
    return tuple(out[i] if i < dim else 1 for i in range(3))


def _expand_ibz_kmesh_to_full_bz(
    system: PeriodicSystem,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh],
) -> Optional[BlochKMesh]:
    """Expand a symmetry-reduced (IBZ) k-mesh to its full parent BZ mesh.

    Returns the full Monkhorst-Pack :class:`BlochKMesh` when ``kmesh`` is
    an API-blessed IBZ reduction (non-empty ``ir_mapping``), and ``None``
    when ``kmesh`` is already a full-BZ sampling (or a plain mesh tuple,
    which never carries a reduction).

    A reduced mesh cannot simply be *weighted* for HF exchange. K(k_i)
    needs the k_j sum over the FULL BZ, and the omitted orbit members
    are not equivalent to their representative: the exchange integrand
    is not per-k invariant unless L and D are symmetry-unfolded onto
    each star member (D(R.k) = P(R) D(k) P(R)^T, plus the per-atom
    lattice-shift Bloch phase -- see periodic_k_symmetry). Weighting
    the representatives instead converged 1.389 Ha away from the full
    mesh on MgO primitive FCC / STO-3G (2,2,2) (2026-07-17
    production-k-sampling audit).

    So reconstruct the full Monkhorst-Pack mesh from the reduction's
    own ``mesh``/shift metadata and run on that -- correct by
    construction, and the same thing the BIPOLE drivers already do
    (_expand_ibz_kmesh_for_ewald_j). This makes IBZ input WORK; it does
    NOT make it cheaper, because every full-mesh point is still built
    and diagonalised. True IBZ-native reduction -- diagonalising only
    at the wedge and unfolding for the exchange sum -- is the separate
    open item, and the exact transport it needs already exists in
    periodic_k_symmetry.

    Expansion is only safe when the metadata genuinely describes the
    parent mesh. An explicit ``KPoints`` list converted through
    ``to_bloch_kmesh`` reports the C++ default mesh (1, 1, 1) -- the
    same metadata gap that produced the 78 mHa exxdiv bug -- so a mesh
    that cannot account for its own reduction still fails closed rather
    than being expanded against a fabricated parent.

    The caller must adopt the returned mesh for EVERYTHING downstream --
    k-point list, weights, and the ``BlochKMesh`` handed to occupation
    and real-space-density helpers. Expanding the k list while keeping
    the wedge ``BlochKMesh`` crashed every KRKS/KUKS IBZ job ("size
    mismatch across k inputs" in real_space_density_from_kpoints_
    fractional: 8 densities against a 6-point wedge on H2 (2,2,2)).
    """
    _ir_raw = getattr(kmesh, "ir_mapping", None)
    ir_mapping = (
        np.zeros(0, dtype=np.int64)
        if _ir_raw is None
        else np.asarray(_ir_raw, dtype=np.int64).reshape(-1)
    )
    if ir_mapping.size == 0:
        return None
    if isinstance(kmesh, KPoints):
        n_reduced = int(np.asarray(kmesh.kpoints_cart).reshape(-1, 3).shape[0])
    else:
        n_reduced = int(np.asarray(kmesh.kpoints).reshape(-1, 3).shape[0])
    _mesh_raw = getattr(kmesh, "mesh", None)
    mesh_meta = tuple(_integer_counts(
        (1, 1, 1) if _mesh_raw is None else _mesh_raw,
        name="IBZ parent mesh",
    ))
    # The native BlochKMesh calls the half-step shift flags ``is_shift``;
    # the Python KPoints dataclass calls them ``shift``. Reading only
    # ``is_shift`` silently expanded a shifted KPoints reduction against
    # the Γ-centered parent -- a different BZ sampling (~44 mHa on MgO
    # (2,2,2), see KPoints.monkhorst_pack's GDF-driver caveat).
    _shift_raw = getattr(kmesh, "is_shift", None)
    if _shift_raw is None:
        _shift_raw = getattr(kmesh, "shift", None)
    shift_meta = tuple(
        int(x) for x in ((0, 0, 0) if _shift_raw is None else _shift_raw)
    )
    full_n = int(np.prod(mesh_meta)) if len(mesh_meta) == 3 else 0
    # use_symmetry=True on a mesh whose points are all in distinct orbits
    # (identity ir_mapping) reduces nothing: the "wedge" already IS the
    # full-BZ quadrature, e.g. (2,1,1) on a cubic vacuum box, where Γ and
    # X are unrelated. Nothing to expand -- and refusing it (as the first
    # expansion cut did) fails a mesh that needs no help.
    if (
        n_reduced == full_n
        and ir_mapping.size == full_n
        and np.array_equal(ir_mapping, np.arange(full_n, dtype=np.int64))
    ):
        return None
    # ir_mapping is defined as one entry per FULL-mesh point, so it is
    # the independent witness that mesh_meta is the real parent.
    if full_n <= n_reduced or ir_mapping.size != full_n:
        raise NotImplementedError(
            "periodic k-GDF: this symmetry-reduced (IBZ) k-mesh cannot "
            "be expanded to its full BZ -- it carries "
            f"{n_reduced} points and ir_mapping of length "
            f"{ir_mapping.size}, but declares mesh={mesh_meta} "
            f"({full_n} full points), so its parent mesh is unknown. "
            "HF exchange needs the full-BZ k_j sum; weighting the "
            "irreducible representatives converges to a wrong energy "
            "(measured +1.389 Ha on MgO/STO-3G (2,2,2)). Pass the "
            "full mesh explicitly: kmesh=(n1, n2, n3) or "
            "KPoints.monkhorst_pack(system, mesh, symmetry=False)."
        )
    expanded = _mp_native(system, list(mesh_meta), list(shift_meta), False)
    n_expanded = int(np.asarray(expanded.kpoints).reshape(-1, 3).shape[0])
    if n_expanded != full_n:
        raise RuntimeError(
            "periodic k-GDF: IBZ expansion produced "
            f"{n_expanded} points for declared mesh={mesh_meta} "
            f"({full_n} expected); refusing to continue on a mesh "
            "that does not match its own metadata."
        )
    w_sum = float(np.asarray(expanded.weights, dtype=np.float64).sum())
    if not np.isclose(w_sum, 1.0, atol=1e-9):
        raise RuntimeError(
            "periodic k-GDF: IBZ expansion produced weights summing "
            f"to {w_sum:.6f}, not 1."
        )
    return expanded


def _kmesh_to_kpoints_weights(
    system: PeriodicSystem,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh],
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise ``kmesh`` to ``(kpoints_cart, weights)`` arrays.

    ``kpoints_cart`` is shape ``(n_k, 3)`` in bohr⁻¹.
    ``weights`` is shape ``(n_k,)`` summing to 1.

    Symmetry-reduced (IBZ) input is expanded to its full parent BZ via
    :func:`_expand_ibz_kmesh_to_full_bz` -- but drivers must do that
    expansion THEMSELVES, before this call, so their ``kmesh_bloch``
    matches the returned arrays; this in-place expansion is only the
    backstop that keeps the returned arrays correct for any caller.
    """
    if isinstance(kmesh, KPoints):
        kpts = np.asarray(kmesh.kpoints_cart, dtype=np.float64).reshape(-1, 3)
        w = np.asarray(kmesh.weights, dtype=np.float64).reshape(-1)
    elif isinstance(kmesh, BlochKMesh):
        kpts = np.asarray(kmesh.kpoints, dtype=np.float64).reshape(-1, 3)
        w = np.asarray(kmesh.weights, dtype=np.float64).reshape(-1)
    else:
        mesh = _mesh_tuple_for_system(system, kmesh)
        bm = _mp_native(system, list(mesh), [0, 0, 0], False)
        kpts = np.asarray(bm.kpoints, dtype=np.float64).reshape(-1, 3)
        w = np.asarray(bm.weights, dtype=np.float64).reshape(-1)
    if kpts.shape[0] == 0:
        raise ValueError("periodic k-GDF: kmesh has zero k-points")
    if not np.isclose(float(w.sum()), 1.0, atol=1e-9):
        raise ValueError(
            f"periodic k-GDF: kpoint weights must sum to 1; got {float(w.sum()):.6f}"
        )
    expanded = _expand_ibz_kmesh_to_full_bz(system, kmesh)
    if expanded is not None:
        kpts = np.asarray(expanded.kpoints, dtype=np.float64).reshape(-1, 3)
        w = np.asarray(expanded.weights, dtype=np.float64).reshape(-1)
    return kpts, w


# =====================================================================


def _gdf_density_return_finalizer(
    system, basis, kmesh, requested, n_channels, memory_byte_cap,
):
    """Preflight an optional producer-owned density representation."""
    if not requested:
        return lambda result: result
    from .periodic_k_density import (
        _lattice_density_return_reservation, _plan_lattice_density_return,
    )

    mesh = _mesh_tuple_for_system(system, kmesh)
    _lattice_density_return_reservation(
        mesh, int(basis.nbasis), n_channels, memory_byte_cap,
    )
    points, weights = _kmesh_to_kpoints_weights(system, kmesh)
    plan = _plan_lattice_density_return(
        system, points, weights, mesh,
        nbf=int(basis.nbasis), n_channels=n_channels,
        memory_byte_cap=memory_byte_cap,
    )
    return plan.attach


def _occupations_per_k(
    eps_per_k: Sequence[np.ndarray],
    weights: np.ndarray,
    n_elec_per_cell: int,
    smearing_T: float,
    n_occ_each: int,
    *,
    bz_integration: Optional[str] = None,
    system: Optional[PeriodicSystem] = None,
    kmesh: Optional[BlochKMesh] = None,
) -> Tuple[List[np.ndarray], float, float]:
    """Fermi-Dirac or hard-Aufbau occupations across the k-mesh.

    Returns ``(occ_per_k, fermi_level, entropy_per_cell)``.
    """
    if bz_integration is not None:
        bz_integration = str(bz_integration).strip().lower()
        if bz_integration not in ("smearing", "gilat"):
            raise ValueError(
                "run_krhf_periodic_gdf: bz_integration must be None, "
                f"'smearing', or 'gilat'; got {bz_integration!r}"
            )
    if bz_integration == "gilat":
        if smearing_T > 0.0:
            raise NotImplementedError(
                "run_krhf_periodic_gdf: bz_integration='gilat' is a "
                "sharp-Fermi-surface occupation backend and cannot be "
                "combined with finite-temperature smearing."
            )
        if system is None or kmesh is None:
            raise ValueError(
                "run_krhf_periodic_gdf: Gilat occupations require the "
                "PeriodicSystem and BlochKMesh metadata."
            )
        from .bz_integration import gilat_occupations_for_kmesh

        occ_gr, ef_gr = gilat_occupations_for_kmesh(
            system,
            kmesh,
            eps_per_k,
            float(n_elec_per_cell),
            spin_degeneracy=2.0,
        )
        return occ_gr, float(ef_gr), 0.0
    if smearing_T <= 0.0:
        occ, mu = _global_aufbau_with_mu(
            eps_per_k,
            weights,
            float(n_elec_per_cell),
        )
        return occ, float(mu), 0.0
    # Global BZ filling (one Fermi level across all k; PySCF KSCF
    # get_occ convention) -- per-k-independent Aufbau silently
    # mis-occupies band-overlap systems (the b4a6faba-regressed
    # rebuild filled exactly n_occ at every k; on LiH FCC (2,2,2)
    # that lands at -2.96 Ha instead of the documented -7.92 Ha).
    return _closed_shell_periodic_occupations(
        eps_per_k,
        weights,
        float(n_elec_per_cell),
        int(n_occ_each),
        float(smearing_T),
    )


def _is_per_k_integer_aufbau(
    occupations: Sequence[np.ndarray], n_occ_each: int
) -> bool:
    """Whether every k point has the legacy ``2[:n_occ], 0`` pattern."""
    n_occ = int(n_occ_each)
    for occ in occupations:
        actual = np.asarray(occ, dtype=float)
        expected = np.zeros_like(actual)
        expected[:n_occ] = 2.0
        if not np.array_equal(actual, expected):
            return False
    return True


def _normalise_initial_density_k(
    initial_density_k: Sequence[np.ndarray],
    *,
    n_k: int,
    n_basis: int,
    label: str,
) -> List[np.ndarray]:
    """Validate caller-supplied per-k density blocks."""
    blocks = list(initial_density_k)
    if len(blocks) != int(n_k):
        raise ValueError(
            f"{label}: initial_density_k has {len(blocks)} blocks; "
            f"expected {int(n_k)} for the target k-mesh."
        )
    out: List[np.ndarray] = []
    for ik, block in enumerate(blocks):
        D = np.asarray(block, dtype=complex)
        if D.shape != (int(n_basis), int(n_basis)):
            raise ValueError(
                f"{label}: initial_density_k[{ik}] has shape {D.shape}; "
                f"expected {(int(n_basis), int(n_basis))}."
            )
        out.append(0.5 * (D + D.conj().T))
    return out


def _madelung_for_kmesh(system: PeriodicSystem, mesh: Sequence[int]) -> float:
    """k-mesh-aware Ewald-Madelung constant ``ξ`` for the exxdiv shift.

    The finite-k-mesh HF-exchange divergence correction (``exxdiv='ewald'``)
    uses the Madelung constant of the **Born-von-Kármán supercell** implied
    by the k-mesh -- the cell whose lattice vectors are the primitive ones
    scaled by the per-direction mesh count -- NOT the primitive cell. For an
    ``(n1, n2, n3)`` Monkhorst-Pack mesh the supercell lattice is
    ``A . diag(n1, n2, n3)``.

    Matches PySCF ``pyscf.pbc.tools.pbc.madelung(cell, kpts)`` (verified
    out-of-process: LiH primitive FCC at (2,2,2) -> ξ = 0.297038, exactly
    half the primitive-cell ξ = 0.594076). Using the primitive-cell value
    over-counts the exxdiv K-shift by ``Nk^(1/3)`` and over-binds the
    multi-k total energy -- LiH (2,2,2): -592 mHa (the bug behind the prior
    -2495 Ha; the cderi-gauge fix exposes it).
    """
    from .madelung import madelung_constant_for_cell

    A = np.asarray(system.lattice, dtype=float)
    A_super = A @ np.diag([float(n) for n in mesh])
    # ξ depends only on the lattice geometry + volume; reuse one atom as a
    # placeholder (madelung_constant_for_cell ignores atom Z / positions).
    super_sys = PeriodicSystem(3, A_super, [system.unit_cell[0]])
    return float(madelung_constant_for_cell(super_sys))


# Positive-energy slack for the multi-k energy-sanity guard (Ha). A bound
# neutral closed-shell cell has E_total < 0, but cramped/artificial Bravais
# smoke-test cells can converge just above zero (the 8-bohr hexagonal H₂
# coverage cell lands at +0.067 Ha); a positive energy beyond this slack is
# unphysical (the broken multi-k compcell+AFT LiH lands at +8.485 Ha).
POSITIVE_E_SLACK_HA = 1.0


def _check_energy_sanity(
    result: PeriodicKRHFGDFResult,
    system: PeriodicSystem,
    plog: ProgressLogger,
    *,
    entry: str = "run_krhf_periodic_gdf",
) -> None:
    """Post-condition: the multi-k SCF total energy is physically sane.

    A bound, neutral, closed-shell unit cell has a **negative** total
    energy whose magnitude is of order ``S_atoms Z^2/2`` (the loose
    hydrogenic bound on absolute binding). Two failure signatures are
    rejected, both observed on the multi-k GDF path for tight ionic
    crystals (LiH primitive FCC, kmesh=(2,2,2), def2-svp-jk aux):

    * **Runaway** -- ``|E_total|`` orders of magnitude beyond
      ``max(10.SZ^2, 100)`` Ha. ``use_compcell=True``/``exxdiv='ewald'``
      lands at ``E ≈ -2495 Ha`` vs PySCF ``-7.92 Ha``: the per-q
      compcell ``Lpq`` fit is internally inconsistent and the SCF
      "converges" to a numerical fixed point of a broken Fock.
    * **Unbound** -- ``E_total`` positive beyond ``POSITIVE_E_SLACK_HA``.
      A bound neutral cell has ``E_total < 0``; the
      ``apply_aft_correction=True`` variant lands at ``+8.485 Ha``, which
      the runaway bound alone would miss (its magnitude is comparable to
      the true ``-7.92``). A *small* positive energy is tolerated:
      cramped/artificial Bravais smoke-test cells can converge just above
      zero without being the catastrophic-garbage pattern (e.g. the
      8-bohr hexagonal H₂ "coverage" cell at ``+0.067 Ha``). A fixed
      slack (rather than a ``SZ^2``-scaled one) keeps the check strict for
      heavy cells, where any sizeable positive energy is unphysical.

    When a non-physical energy is reported as **converged**, this is the
    silent-corruption pattern CLAUDE.md Sec.7 warns about -- a user gets
    ``converged=True`` with a meaningless number and may use it
    downstream. We RAISE rather than return it (do NOT paper over with
    damping/thresholds -- Sec.7). For a non-converged run (``converged=False``
    already signals failure) we warn + tag the backend so partial state
    stays inspectable.

    Multi-k GDF parity landed 2026-06-02 (the per-(k_i,k_j)-resolved
    Lpq cache + the BvK-supercell exxdiv Madelung; see
    ``handovers/HANDOVER_GDF_OUTSTANDING.md`` Sec. 1), so this guard firing means a
    regression in that machinery -- a gauge mismatch, a wrong Madelung
    convention, or an inconsistent cderi cache. H₂-style vacuum-box
    cells are fine at multi-k -- the guard only trips on the genuinely
    broken numbers, so it does not fire on a correct multi-k energy
    (e.g. LiH ``-7.92 Ha``).
    """
    E = float(result.energy)
    z_sum_sq = sum(atom.Z**2 for atom in system.unit_cell)
    sane_bound = max(10.0 * z_sum_sq, 100.0)  # loose hydrogenic + floor
    runaway = abs(E) > sane_bound
    unbound = E > POSITIVE_E_SLACK_HA
    if not (runaway or unbound):
        return

    reason = "runaway divergence" if runaway else "positive (unbound) total energy"
    # Give the reader the numbers needed to discriminate among the causes
    # listed below. Without them "inconsistent Lpq cache" and "the fit
    # discarded half its directions" look identical from the .out.
    lindep = getattr(result, "linear_dependence", None)
    lindep_note = (
        f" Linear-dependence state at this geometry -- {lindep.one_line()}."
        if lindep is not None
        else ""
    )
    msg = (
        f"{entry}: SCF total energy {E:.6e} Ha is non-physical "
        f"({reason}). A bound, neutral, closed-shell cell has E_total < 0 "
        f"(a positive energy beyond {POSITIVE_E_SLACK_HA:g} Ha slack is "
        f"unbound) and |E_total| < {sane_bound:.2e} Ha (loose hydrogenic "
        "bound on S_atoms Z^2/2). The SCF has converged to a numerical fixed "
        "point of a broken Fock -- a gauge mismatch, a wrong exxdiv Madelung "
        "convention, or an internally-inconsistent Lpq cache (CLAUDE.md Sec.7 -- "
        "not a convergence-aid problem). Multi-k GDF parity landed 2026-06-02 "
        "(handovers/HANDOVER_GDF_OUTSTANDING.md Sec. 1), so this firing indicates a "
        "regression in that machinery. Pass check_energy_sanity=False to "
        "bypass this guard (diagnostics only)."
        + lindep_note
    )
    try:
        result.backend = result.backend + "+SANITY_FAILED"
    except Exception:
        pass
    if result.converged:
        raise RuntimeError(msg)
    plog.info("  WARNING: " + msg)


def _build_xc_k_from_density(
    *,
    basis: BasisSet,
    system: PeriodicSystem,
    grid: object,
    func: Functional,
    density_k: Sequence[np.ndarray],
    kmesh_bloch: BlochKMesh,
    cells: Sequence[object],
    kpoints_cart: Sequence[np.ndarray],
    lat_opts: LatticeSumOptions,
    density_domain: PeriodicXCDensityDomain = PeriodicXCDensityDomain.AUTO,
) -> tuple[float, List[np.ndarray]]:
    """Build periodic XC from the full finite-torus density.

    Local and semilocal XC are primitive-cell functionals of the density, but
    the AO matrix returned by :func:`build_xc_periodic` is a lattice matrix.
    A multi-k finite-torus RKS Fock must therefore reconstruct the real-space
    density blocks from all k-point density matrices and Bloch-fold the
    resulting ``V_xc(R)`` to each k.  Feeding only the home-cell density and
    adding one Γ matrix to every k is exact only in the vacuum/molecular
    limit; on tight crystals it can make the KS map non-variational.
    """

    density_real = _real_space_density_from_per_k_density(
        density_k,
        kmesh_bloch,
        cells,
    )
    xc_contrib = build_xc_periodic(
        basis,
        system,
        grid,
        func,
        density_real,
        lat_opts,
        density_domain,
    )
    vxc_k: List[np.ndarray] = []
    for k_cart in kpoints_cart:
        vk = np.asarray(
            bloch_sum(xc_contrib.V_xc, np.asarray(k_cart, dtype=float).reshape(3)),
            dtype=complex,
        )
        vxc_k.append(0.5 * (vk + vk.conj().T))
    return float(xc_contrib.e_xc), vxc_k


def _xc_density_cells_for_domain(
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    cells: Sequence[object],
    density_domain: PeriodicXCDensityDomain,
) -> Sequence[object]:
    """Return a difference-complete lattice domain for periodic XC."""

    if density_domain != PeriodicXCDensityDomain.PERIODIC_LATTICE:
        return cells
    from .periodic_external_xc import periodic_xc_difference_closed_cells

    image_radius = float(
        getattr(lat_opts, "becke_image_radius_bohr", 0.0)
    )
    if image_radius <= 0.0:
        image_radius = 10.0
    return periodic_xc_difference_closed_cells(
        system,
        float(lat_opts.cutoff_bohr),
        image_radius,
    )


def _build_xc_k_from_density_uks(
    *,
    basis: BasisSet,
    system: PeriodicSystem,
    grid: object,
    func: Functional,
    density_alpha_k: Sequence[np.ndarray],
    density_beta_k: Sequence[np.ndarray],
    kmesh_bloch: BlochKMesh,
    cells: Sequence[object],
    kpoints_cart: Sequence[np.ndarray],
    lat_opts: LatticeSumOptions,
    density_domain: PeriodicXCDensityDomain = PeriodicXCDensityDomain.AUTO,
) -> tuple[float, List[np.ndarray], List[np.ndarray]]:
    """Open-shell sibling of :func:`_build_xc_k_from_density`.

    Folds each spin's per-k density to the full real-space finite-torus
    density set (the inverse Bloch sum, ``P_s(g) = S_k w_k e^{-ik.g}
    P_s(k)``), evaluates the spin-polarised periodic XC on the pair, and
    Bloch-folds ``V_a(g)`` / ``V_b(g)`` back to every k.  This is the
    same convention the closed-shell multi-k branch uses -- the
    BZ-averaged home-cell shortcut it replaces was exact only in the
    vacuum/molecular limit (the 2026-07-09 KRKS finding's defect class,
    commit b3f74aa9).  With ``P_a = P_b = P/2`` the folded spin densities
    reproduce the closed-shell density pointwise, so KUKS(mult=1) ==
    KRKS holds by construction on any grid.
    """
    from ._vibeqc_core import build_xc_periodic_uks

    density_alpha_real = _real_space_density_from_per_k_density(
        density_alpha_k,
        kmesh_bloch,
        cells,
    )
    density_beta_real = _real_space_density_from_per_k_density(
        density_beta_k,
        kmesh_bloch,
        cells,
    )
    xc_contrib = build_xc_periodic_uks(
        basis,
        system,
        grid,
        func,
        density_alpha_real,
        density_beta_real,
        lat_opts,
        density_domain,
    )
    va_k: List[np.ndarray] = []
    vb_k: List[np.ndarray] = []
    for k_cart in kpoints_cart:
        k_arr = np.asarray(k_cart, dtype=float).reshape(3)
        va = np.asarray(bloch_sum(xc_contrib.V_alpha, k_arr), dtype=complex)
        vb = np.asarray(bloch_sum(xc_contrib.V_beta, k_arr), dtype=complex)
        va_k.append(0.5 * (va + va.conj().T))
        vb_k.append(0.5 * (vb + vb.conj().T))
    return float(xc_contrib.e_xc), va_k, vb_k


def _preflight_gdf_lpq_memory(
    plog,
    *,
    n_basis: int,
    n_aux: int,
    n_kpoints: int,
    need_k_pairs: bool,
    open_shell: bool,
    route_label: str,
    options=None,
    n_ibz_kpoints: Optional[int] = None,
) -> None:
    """Estimate + gate the dense multi-k GDF ``Lpq`` cache before building it.

    The per-pair ``Lpq`` cache is the multi-k GDF memory bottleneck and is
    held dense in RAM (streaming is future work, see the module docstring), so
    a paper-grade cell can be OOM-killed (exit 137) before SCF iter 1 with no
    ``.out``/``.system``/``.qvf`` artifacts -- the prompt-75 NiO/def2-SVP KUKS
    ``(4,4,4)`` case. This logs the peak estimate (so a `progress`-on rerun
    localises the cost) and raises :class:`~vibeqc.memory.InsufficientMemoryError`
    early with route-specific remedies when it cannot fit. Override with
    ``VIBEQC_GDF_MEMORY_OVERRIDE=1`` (or a truthy ``options.memory_override``).

    ``n_ibz_kpoints`` is the resolved wedge size on an ``ibz_native`` run.
    It MUST be threaded through: the reduced build stores ``n_IBZ x n_k``
    exchange blocks plus the remaining Hartree diagonals. Gating it at
    ``n_k^2`` aborts runs that fit and contradicts the allocated cache.
    """
    import os as _os

    from .memory import check_periodic_gdf_memory, estimate_periodic_multik_gdf

    est = estimate_periodic_multik_gdf(
        n_basis=int(n_basis),
        n_aux=int(n_aux),
        n_kpoints=int(n_kpoints),
        need_k_pairs=bool(need_k_pairs),
        open_shell=bool(open_shell),
        n_ibz_kpoints=(None if n_ibz_kpoints is None else int(n_ibz_kpoints)),
        diis_subspace_size=(
            int(getattr(options, "diis_subspace_size", 8))
            if bool(getattr(options, "use_diis", True)) else 0
        ),
    )
    if not need_k_pairs:
        pair_kind = "diagonal pairs"
    elif n_ibz_kpoints is None:
        pair_kind = "k^2 exchange pairs"
    else:
        pair_kind = (
            f"{int(n_ibz_kpoints)} wedge bras plus all Hartree diagonals"
        )
    plog.info(
        f"GDF Lpq cache peak estimate ~{est.total_gb:.1f} GB "
        f"({int(n_kpoints)} k-points, {pair_kind}, naux={int(n_aux)}, "
        f"nao={int(n_basis)})"
    )
    allow = bool(getattr(options, "memory_override", False)) or bool(
        _os.environ.get("VIBEQC_GDF_MEMORY_OVERRIDE")
    )
    check_periodic_gdf_memory(
        est,
        n_kpoints=int(n_kpoints),
        route_label=route_label,
        allow_exceed=allow,
    )


def _reject_slab_dim(system: PeriodicSystem, entry: str) -> None:
    """Fail closed when a non-slab GDF driver is handed a ``dim=2`` slab.

    The legacy bulk Bloch / AFT machinery assumes 3-D periodicity. Handed a
    ``dim=2`` slab it silently treats the layer as a 3-D crystal of sheets stacked
    ``a3`` apart (the normal-axis k-mesh pinned to a single point), so the
    total energy depends on the bookkeeping ``|a3|`` *and* on the k-mesh --
    the CLAUDE.md Sec. 7 "impossible / mesh-dependent energy" symptom (a
    purely geometric Madelung sum cannot depend on the k-mesh). Slabs must use
    the vacuum-free 2D gauge instead. The Gamma-only ``run_pbc_gdf_*`` drivers
    already guard this; the multi-k entries did not, which is how a slab
    reached the 3-D crystal path. Closed-shell RHF/RKS now delegates before
    this guard to the dedicated signed slab-truncated metric; open-shell GDF
    continues to stop here.

    ``dim == 1`` is deliberately NOT rejected. There is no rigorous 1-D Coulomb
    gauge yet (``CoulombMethod.NEUTRALIZED_1D`` still raises), so the
    cached-Lpq GDF Hartree is the *supported* polymer/wire route -- see
    ``5fe4d021`` ("Lpq Hartree for multi-k dim<3 KS"), which fixed the dim=1
    H2-chain from -3.0897 to the variational -3.89 Ha/cell. Guarding ``dim != 3``
    here would silently revert that fix. Only the slab has a rigorous vacuum-free
    alternative to redirect users to, so only the slab fails closed.
    """
    dim = int(system.dim)
    if dim == 2:
        raise NotImplementedError(
            f"{entry}: GDF is a bulk (dim=3) Coulomb builder; got dim={dim} "
            "(a slab). Running a slab through it would treat it as a 3-D "
            "crystal of sheets a3 apart, giving a3- and k-mesh-dependent "
            "energies. Use jk_method='auto' (or 'slab_ewald_2d') -- the "
            "rigorous vacuum-free 2D Coulomb (SLAB_EWALD_2D). For a "
            "3-D-with-vacuum reference cell instead, build it with "
            "vibeqc.build.slab(..., periodic_z=True) so dim=3."
        )


def _warn_multik_dense_core_gdf_parity_hold(
    system: PeriodicSystem,
    gdf_method: str,
    basis,
    tail_ke_cutoff,
    rsgdf_ke_cutoff: float,
    entry: str,
) -> bool:
    """Warn + report the dense-core absolute-energy hold for MULTI-K GDF.

    Reuses the Γ classifier ``pbc_gdf._gamma_dense_core_gdf_parity_held``
    (tight-core AO products unresolved on the default rsgdf mesh; the class
    is basis/cell-driven, not Γ-specific). Measured at multi-k
    (production-k-sampling audit, 2026-07-17, MgO primitive FCC / STO-3G vs
    live PySCF 2.13.1 on identical meshes): the untailed default is
    -0.503 Ha ((1,1,2)) / -0.505 Ha ((2,2,2), both Γ-centered and shifted)
    -- mesh-independent, the P01 class -- while
    ``rsgdf_tail_ke_cutoff = 1.1*10*zeta_max`` closes (1,1,2) to +0.16 mHa
    (the same shared-thresholds floor as the Γ ladder) at ~40x the untailed
    build cost per (k_i,k_j) pair.

    **Which multi-k routes auto-size the tail (IID 518).** Pure DFT
    (``alpha == 0``) on ``dim=3`` DOES, from ``run_krhf_periodic_gdf``,
    which resolves ``pbc_gdf._auto_rsgdf_tail_ke_cutoff`` before calling
    this helper -- so a dense-core KRKS run arrives here already tailed
    and is NOT tagged. That branch is the one case where the 40x cost
    argument is inverted by measurement: it does not add a tail to an
    existing GDF J, it replaces the real-space EWALD_3D J outright, and
    the tailed run is 8x FASTER (593 s vs 4848 s on MgO/STO-3G (2,2,2),
    IID 146) while closing -1032.5 mHa of error vs PySCF and CRYSTAL 23.

    Every OTHER multi-k route -- KRHF, hybrids, and both open-shell
    drivers (KUHF / KUKS, which have no ``use_compcell`` and always ride
    the cached-Lpq GDF J) -- still does NOT auto-size: for them the tail
    is genuine added cderi cost at n_k^2 pair builds, a production-scale
    cost the caller must opt into. Those warn + tag instead of silently
    returning the un-tailed value, and whether THEY should auto-tail by
    default remains an open maintainer call (see HANDOVER_OPEN_BUGS_V015
    production-k-sampling section). One visible consequence: on a
    dense-core cell a KUKS(M=1) run does not reproduce its KRKS twin,
    and is tagged ``+PARITY_HELD`` where the KRKS run is not.
    """
    from .pbc_gdf import (
        _RSGDF_PARITY_TAIL_RATIO,
        _gamma_dense_core_gdf_parity_held,
        _reject_dense_core_mdf,
    )

    # mdf on the compact-dense-core class fails closed (measured
    # non-convergent at -11741 Ha on MgO/STO-3G (1,1,2); the rsgdf tail
    # remediation in the warning below does not exist for mdf). Both
    # multi-k drivers pass through this helper before the cderi build.
    _reject_dense_core_mdf(system, gdf_method, basis, entry)
    held = _gamma_dense_core_gdf_parity_held(
        system,
        gdf_method,
        ao_basis=basis,
        tail_ke_cutoff=tail_ke_cutoff,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    )
    if held:
        warnings.warn(
            f"{entry}: dense-core cell on the multi-k GDF route -- ABSOLUTE "
            "energies are parity-held (P01 class; measured -0.50 Ha vs "
            "PySCF on MgO/STO-3G at the untailed default). Pass "
            f"rsgdf_tail_ke_cutoff >= {_RSGDF_PARITY_TAIL_RATIO:.0f} x the "
            "steepest AO primitive exponent to close it (~uHa-to-0.2-mHa "
            "class, measured ~40x the untailed cderi build cost per k-pair), "
            "or use a pseudopotential-class basis without tight-core AO "
            "products. Result backend is tagged +PARITY_HELD.",
            stacklevel=3,
        )
    return held


def _reject_unsupported_multik_gradient(
    entry: str,
    *,
    dim: int,
    gdf_method: str,
    smearing_temperature: float,
    k_exchange: str,
    screened_omega: Optional[float],
    functional_is_range_separated: bool,
    weights: np.ndarray,
    use_compcell_effective: bool = True,
    bz_integration: Optional[str] = None,
    dft_plus_u_sites: Optional[Sequence[object]] = None,
    smearing_flavor: str = "fermi-dirac",
    ibz_native: bool = False,
) -> None:
    """Fail-closed envelope guards for ``compute_gradient=True`` on the
    multi-k GDF drivers (G-PBC-002 Item-4 rung 6).

    The multi-k analytic gradient differentiates exactly the objective
    the rsgdf shared-q Lpq SCF converged: a full-range fit
    (Schwarz-screened or not — the gradient cache mirrors the SCF's
    pair mask since 2026-07-30, differentiating the screened objective
    at fixed mask) on a uniform full-BZ mesh, at T = 0 Aufbau
    occupations or (since 2026-07-30) finite-temperature Fermi-Dirac
    occupations, where the differentiated objective is the Mermin free
    energy ``A = E - T S`` the driver reports as ``free_energy``
    (Mermin, Phys. Rev. 137, A1441 (1965): A is stationary at the
    self-consistent finite-T solution, so the occupation- and
    mu-response terms in dA/dR vanish and the T = 0 assemblers apply
    at the fractional-occupation D(k)/W(k); Marzari-Vanderbilt 1999's
    smeared-force statement). Every configuration whose objective
    contains a term the assemblers do not differentiate raises here,
    loud and named, BEFORE the SCF runs (the Γ drivers' fail-closed
    pattern, ``run_pbc_gdf_rhf``).
    """
    if ibz_native:
        # The gradient assemblers differentiate the FULL-BZ exchange
        # objective, but an ibz_native SCF converges the K-TRANSPORTED
        # objective (exchange built at the irreducible wedge and
        # symmetry-transported, 2026-08-01 increment 2). The two agree
        # only to the transport residual (1e-7-class at converged
        # cutoffs, 1e-3-class at loose ones), so differentiating one
        # against the other is a silent objective mismatch. Lifting
        # this needs either a transported-K derivative or an FD gate
        # pinning the mismatch below the gradient tolerance per cutoff.
        raise NotImplementedError(
            f"{entry}: compute_gradient=True with ibz_native=True is "
            "not supported: the analytic gradient differentiates the "
            "full-BZ exchange objective while the ibz-native SCF "
            "converges the symmetry-transported one. Run the gradient "
            "with ibz_native=False (same physics, full-BZ exchange "
            "build)."
        )
    if int(dim) != 3:
        # dim=2 closed-shell slabs never reach this guard: the KRHF/KRKS
        # entries divert them to the dedicated slab route, whose
        # analytic gradient landed 2026-07-30 (G-PBC-002 § 6 rung 5).
        # What remains here is dim=1 wires (no wire gradient exists) and
        # dim=2 on the open-shell entries (no open-shell slab SCF
        # exists, let alone its gradient).
        raise NotImplementedError(
            f"{entry}: compute_gradient currently supports 3D periodic "
            f"systems only on this entry (got dim={int(dim)}); wire "
            "(dim=1) and open-shell slab multi-k GDF gradients are "
            "future items. Closed-shell dim=2 slabs get the analytic "
            "gradient via run_krhf_periodic_gdf / run_krks_periodic_gdf "
            "(the dedicated slab route)."
        )
    if str(gdf_method) != "rsgdf":
        raise NotImplementedError(
            f"{entry}: compute_gradient currently supports "
            "gdf_method='rsgdf' only on multi-k meshes (got "
            f"{gdf_method!r}); the compcell/MDF multi-k fit derivatives "
            "are not implemented."
        )
    if (
        float(smearing_temperature) > 0.0
        and str(smearing_flavor) not in ("fermi-dirac", "mermin")
    ):
        # Envelope restriction: non-Fermi-Dirac/non-Mermin analytic-gradient
        # paths lack dedicated full-SCF force validation, and MP occupations
        # can be negative while the implementation forms sqrt(f) blocks. Their
        # reported generalized free energies are nevertheless variational.
        raise NotImplementedError(
            f"{entry}: compute_gradient supports Fermi-Dirac / Mermin "
            "smearing only -- non-Fermi-Dirac force paths are not yet "
            f"validated; run with flavor='fermi-dirac' or 'mermin', "
            f"or smearing_temperature=0 instead of {smearing_flavor!r}."
        )
    if bz_integration is not None and str(
        bz_integration
    ).strip().lower() != "smearing":
        # bz_integration='smearing' is literally the default
        # occupation path (the string only labels it; see
        # _occupations_per_k), so it inherits the smearing envelope
        # above. Every other quadrature (e.g. the Gilat-Raubenheimer
        # sharp-Fermi net) has no differentiated occupation response.
        raise NotImplementedError(
            f"{entry}: compute_gradient supports the default Aufbau / "
            "Fermi-Dirac-smearing occupations only; "
            f"bz_integration={bz_integration!r} has no gradient "
            "response."
        )
    if functional_is_range_separated or screened_omega is not None:
        raise NotImplementedError(
            f"{entry}: compute_gradient does not support range-separated "
            "/ screened hybrids (omega_screen != 0): the fitted-K "
            "derivative is full-range only and the screened-COSX "
            "exchange path has no fit derivative. Use a global hybrid "
            "or a pure functional."
        )
    if str(k_exchange) != "gdf":
        raise NotImplementedError(
            f"{entry}: compute_gradient requires the fitted GDF exchange "
            f"(k_exchange='gdf'); the {k_exchange!r} exchange backend is "
            "not the differentiated Lpq contraction."
        )
    if not use_compcell_effective:
        raise NotImplementedError(
            f"{entry}: compute_gradient requires the cached-Lpq GDF SCF "
            "(use_compcell=True): the use_compcell=False Ewald-3D J/K "
            "path has no rsgdf fit to differentiate."
        )
    if dft_plus_u_sites:
        raise NotImplementedError(
            f"{entry}: compute_gradient does not differentiate the DFT+U "
            "energy term; drop dft_plus_u_sites for gradients."
        )
    w = np.asarray(weights, dtype=float).reshape(-1)
    n_k = int(w.shape[0])
    if n_k == 0 or not np.allclose(w, 1.0 / n_k, rtol=0.0, atol=1e-12):
        raise NotImplementedError(
            f"{entry}: compute_gradient requires a uniform full-BZ mesh "
            "(every k weight = 1/n_k). Symmetry-reduced (IBZ) or "
            "custom-weight k-point sets need the orbit-unfolded exchange "
            "derivative; pass the full mesh, e.g. kmesh=(n1, n2, n3)."
        )


def _build_multik_gradient_cache_checked(
    system: PeriodicSystem,
    basis: BasisSet,
    aux_modrho: BasisSet,
    kpoints_cart: np.ndarray,
    *,
    rsgdf_ke_cutoff: float,
    rsgdf_tail_ke_cutoff: Optional[float],
    lat_opts: LatticeSumOptions,
    gdf_linear_dep_threshold: float,
    n_fit_scf: int,
    entry: str,
    plog: ProgressLogger,
    fit_screen_threshold: float = 0.0,
    source_cache=None,
):
    """Rebuild the shared-q rsgdf fit for the gradient and cross-check
    its q=0 rank against the converged SCF fit (the Γ drivers'
    fit-consistency pattern -- any mismatch is Fréchet-amplified near
    the linear-dep threshold, M6 rung 9). ``fit_screen_threshold`` is
    the SCF's own Schwarz screen; the cache re-derives the identical
    per-q pair mask from the same mask builder."""
    from .periodic_gdf_gradient import (
        _build_multik_rsgdf_gradient_cache,
        _multik_rsgdf_q0_group,
    )

    from .periodic_mdf import _MdfCache

    if isinstance(source_cache, (_RangeSeparatedGdfCache, _MdfCache)):
        from .aux_basis import _range_separated_gdf_source_signature

        label = 'private MDF' if isinstance(source_cache, _MdfCache) else 'SR/LR'
        if (source_cache.aux_basis is None
                or source_cache.source_signature != _range_separated_gdf_source_signature(
                system, basis, source_cache.aux_basis)
                or not np.array_equal(np.asarray(source_cache.kpoints_cart), kpoints_cart)):
            raise ValueError(f"{entry}: {label} gradient source differs from the SCF geometry or mesh")
        if source_cache[(0, 0)].shape[0] != int(n_fit_scf):
            raise RuntimeError(f"{entry}: {label} gradient source differs from the SCF fit rank")
        plog.info(f"analytic gradient: retained {label} source parameters; bounded raw-source rebuild")
        return source_cache

    cache = _build_multik_rsgdf_gradient_cache(
        system,
        basis,
        aux_modrho,
        np.asarray(kpoints_cart, dtype=float),
        ke_cutoff=float(rsgdf_ke_cutoff),
        tail_ke_cutoff=(
            float(rsgdf_tail_ke_cutoff)
            if rsgdf_tail_ke_cutoff is not None
            else None
        ),
        lat_opts=lat_opts,
        linear_dep_thr=float(gdf_linear_dep_threshold),
        fit_screen_threshold=float(fit_screen_threshold),
    )
    n_fit_cache = int(_multik_rsgdf_q0_group(cache).n_fit)
    if n_fit_cache != int(n_fit_scf):
        raise RuntimeError(
            f"{entry}: multi-k gradient-cache q=0 fit rank "
            f"({n_fit_cache}) does not match the converged SCF fit "
            f"({int(n_fit_scf)})."
        )
    plog.info(
        f"analytic gradient: multi-k rsgdf fit rebuilt "
        f"({len(cache.groups)} q group(s), {n_fit_cache} fit vectors "
        "at q=0)"
    )
    return cache



# ---------------------------------------------------------------------------
# Effective core potentials on the GDF drivers (2026-09)
# ---------------------------------------------------------------------------
#
# The inline ECP fields on the options (blocks + home centres from the pob
# CRYSTAL records or a basis sidecar, per-atom Z_eff, aggregate core count)
# enter the one-electron Hamiltonian in three places: V_ne is built with
# Z_eff, the lattice-summed V_ECP is Bloch-summed into every Hcore(k), and
# the ionic repulsion uses Z_eff; the driver then fills the valence count.
# V_ne and the Ewald repulsion are linear in the point charges and read
# them from the PeriodicSystem's atoms, so a copy of the system carrying
# Z_eff as its atomic numbers is the charged nuclear frame; the basis, the
# lattice and the positions are untouched.


def _periodic_ecp_context(opts, system, driver: str):
    """Return ``(blocks, home_centers, z_eff, ncore, system_v)`` or ``None``."""
    blocks = list(getattr(opts, "ecp_primitive_blocks", None) or [])
    if not blocks:
        stray = int(getattr(opts, "ecp_total_ncore", 0) or 0) != 0 or bool(
            getattr(opts, "ecp_home_centers", None)
        )
        if stray:
            raise ValueError(
                f"{driver}: ECP metadata is set without ecp_primitive_blocks; "
                "supply the operator or clear ecp_total_ncore / ecp_home_centers"
            )
        return None
    if int(system.dim) != 3:
        raise NotImplementedError(
            f"{driver}: periodic ECPs are implemented for 3D cells only"
        )
    atoms = list(system.unit_cell)
    z_eff = [float(q) for q in (getattr(opts, "ecp_effective_charges", None) or [])]
    if len(z_eff) != len(atoms):
        raise ValueError(
            f"{driver}: ecp_effective_charges has {len(z_eff)} entries for "
            f"{len(atoms)} atoms"
        )
    centers = [list(c) for c in (getattr(opts, "ecp_home_centers", None) or [])]
    if len(centers) != len(blocks):
        raise ValueError(
            f"{driver}: {len(blocks)} ECP blocks but {len(centers)} home centres"
        )
    ncore = int(getattr(opts, "ecp_total_ncore", 0) or 0)
    from_charges = round(sum(float(a.Z) for a in atoms) - sum(z_eff))
    if ncore != from_charges:
        raise ValueError(
            f"{driver}: ecp_total_ncore={ncore} disagrees with the effective "
            f"charges ({from_charges} core electrons)"
        )
    for q in z_eff:
        if abs(q - round(q)) > 1e-9 or q <= 0:
            raise ValueError(f"{driver}: effective charge {q} is not a positive integer")
    from ._vibeqc_core import Atom, PeriodicSystem

    system_v = PeriodicSystem(
        int(system.dim),
        np.asarray(system.lattice, dtype=float),
        [Atom(round(q), list(a.xyz)) for a, q in zip(atoms, z_eff)],
        int(system.charge),
        int(system.multiplicity),
    )
    return blocks, centers, z_eff, ncore, system_v


def _ecp_lattice_blocks(basis, system, lat_opts, ecp_ctx):
    """Return home-bra/translated-ket ECP blocks on their native cell list.

    AO cells honor ``pair_complete_1e``. Projector images use the finite
    ``nuclear_cutoff_bohr`` domain; its convergence must be checked along
    with the AO domain before trusting dense-cell energies.
    """
    from ._vibeqc_core import compute_ecp_lattice_from_primitives

    blocks, centers, _z_eff, _ncore, _system_v = ecp_ctx
    return compute_ecp_lattice_from_primitives(
        basis, system, lat_opts, centers, blocks
    )


def run_krhf_periodic_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh] = (1, 1, 1),
    options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    *,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    apply_modrho: bool = True,
    fock_mixing: Optional[float] = None,
    level_shift_warmup_cycles: Optional[int] = None,
    use_compcell: bool = False,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_ft_convention: str = "libint",
    aft_precision: float = 1e-10,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    k_exchange: str = "gdf",
    gdf_method: str = "rsgdf",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    fit_screen_threshold: float = 0.0,
    mdf_ke_cutoff: float = 40.0,
    ibz_native: bool = False,
    bz_integration: Optional[str] = None,
    density_mixer: Optional[str] = None,
    density_mixer_depth: int = 8,
    density_mixer_beta: float = 0.5,
    density_mixer_kerker: bool = False,
    kerker_k0: float = 1.5,
    kerker_strength: float = 1.0,
    kerker_cutoff_ha: float = 120.0,
    dft_plus_u_sites: Optional[Sequence[object]] = None,
    initial_density_k: Optional[Sequence[np.ndarray]] = None,
    compute_gradient: bool = False,
    return_lattice_density: bool = False,
    lattice_density_memory_bytes: int = 128 * 1024**2,
    check_energy_sanity: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    _lpq_cache_builder=None,
    xc_density_domain: PeriodicXCDensityDomain = (
        PeriodicXCDensityDomain.AUTO
    ),
) -> PeriodicKRHFGDFResult:
    """Run closed-shell periodic HF / KS multi-k SCF via native GDF.

    For a single Γ point this delegates to
    :func:`vibeqc.run_rhf_periodic_gamma_gdf` (kept in lock-step with
    the multi-k path); for any other ``kmesh`` it runs the full
    multi-k loop here.

    For ``system.dim == 2``, this entry selects the dedicated signed
    slab-truncated fit on a full Gamma-centered ``(n1,n2,1)`` tuple mesh.
    That bounded route supports zero-temperature closed-shell RHF/RKS and
    full-range hybrids only, including ``compute_gradient=True`` (the
    analytic slab gradient, G-PBC-002 § 6). Custom/IBZ meshes, range
    separation, DFT+U, and density restart remain fail-closed;
    ``jk_method='auto'`` continues to select the independent direct
    ``SLAB_EWALD_2D`` route.

    Parameters
    ----------
    system, basis
        Periodic system and AO basis.
    kmesh
        ``(n1, n2, n3)`` Monkhorst-Pack mesh, a :class:`KPoints`
        instance, or a :class:`BlochKMesh`. Defaults to Γ-only.

        **Mesh-convention caveat**: the tuple form samples the
        **Γ-centered** (Γ-inclusive) mesh — the same convention as PySCF
        ``cell.make_kpts([n1, n2, n3])`` and the convention every
        PySCF-parity pin in the test suite uses. A
        :class:`KPoints` built via ``KPoints.monkhorst_pack(system,
        mesh, symmetry=False)`` instead samples the **half-step-shifted
        original Monkhorst-Pack mesh (no Γ point)**. Both are legitimate
        BZ quadratures, but they are *different samplings*: on MgO
        primitive FCC / STO-3G at (2,2,2) they differ by ~44 mHa (both
        vibe-qc and PySCF agree on the size of that shift). Pass the
        tuple (or a Γ-centered ``KPoints``) when comparing against
        PySCF defaults or the pinned references. Symmetry-reduced
        (IBZ) meshes are expanded up front to their full parent BZ
        mesh (same shift, same dims) — correct, not cheaper; see
        ``_expand_ibz_kmesh_to_full_bz``.
    options
        :class:`PeriodicRHFOptions` (HF) or :class:`PeriodicKSOptions`
        (KS).
    functional
        libxc functional name when running KS; ``None`` means HF.
    aux_basis
        Auxiliary basis name. Defaults to ``default_aux_for(basis.name)``.
    aux_drop_eta
        Auxiliary primitive cull threshold passed to
        :func:`make_aux_basis_set`.
    linear_dep_threshold
        Per-k overlap eigenvalue floor for canonical orthogonalisation.
    gdf_linear_dep_threshold
        Auxiliary metric eigenvalue floor for ``Lpq`` Cholesky-style
        fitting. RSGDF FFT builders interpret this as an absolute
        eigenvalue threshold, matching PySCF's convention; legacy
        compcell/MDF/bare builders interpret it relative to their
        largest metric eigenvalue. Keep the default when comparing
        builder families unless you intentionally want builder-specific
        truncation behavior.
    apply_modrho
        Whether the auxiliary basis is renormalised via
        :func:`aux_basis.modrho_renormalise` before fitting (default
        on; matches the Γ-only driver).
    fock_mixing
        Override the resolver-resolved CRYSTAL FMIXING fraction.
    level_shift_warmup_cycles
        Override the resolver-resolved level-shift warm-up length.
    k_exchange
        Exchange backend on the ``use_compcell=True`` path:
        ``'gdf'`` (default) contracts the cached per-(k_i, k_j) Lpq
        tensors -- O(N_k^2) pair contractions per iteration; ``'cosx'``
        (**EXPERIMENTAL**) builds K via the real-space multi-k COSX
        engine (:class:`vibeqc.periodic_cosx_k.KPointCosxK`, M3b-3):
        one K(g) block build per iteration (mesh-size independent) +
        Bloch folds, and the off-diagonal Lpq tensors are skipped at
        setup (only the diagonal J pairs are built). The Coulomb J
        stays on GDF either way; the exxdiv='ewald' correction
        applies identically to both backends. Requires
        ``use_compcell=True`` and a multi-k mesh (the Γ-only fast
        path ignores it, like ``use_compcell``).

        Since M3b-4b the COSX K is the composed SR+LR exchange --
        matrix-level validated at ~1e-4 against the independent RSGDF
        route. Pair it with ``gdf_method='rsgdf'`` for a single-gauge
        Fock (see below); a runtime warning documents the remaining
        experimental status (dense-mesh SCF convergence -- M3b-4c).
    gdf_method
        Lpq builder for the ``use_compcell=True`` cache: ``'rsgdf'``
        (default -- the (k_i,k_j)-ket-resolved all-FT Bloch-pair route,
        :func:`vibeqc.aux_basis.build_lpq_bloch_native_fft`; validated
        at µHa parity vs PySCF on LiH FCC (2,2,2),
        handovers/HANDOVER_GDF_OUTSTANDING.md Sec. 1) or ``'compcell'`` (Sun-2017
        compensated charges + AFT correction -- q-only cderi, exact
        only in the vacuum-box limit; catastrophically wrong on tight
        ionic cells, the M3b-5 finding #1 / the -2495 Ha class -- keep
        it off the default until the compcell builder is
        pair-resolved). The RSGDF route is also the consistent partner
        for ``k_exchange='cosx'`` (single-gauge pairing, M3b-4c).
    rsgdf_ke_cutoff
        Dense-FFT-mesh kinetic-energy cutoff (Ha) for
        ``gdf_method='rsgdf'`` (default 200).
    rsgdf_tail_ke_cutoff
        Optional high-|G| tail completion (Ha) for the rsgdf cderi:
        extends the fit's 2c-metric and 3c-tensor G-sums over the exact
        complementary reciprocal shell, needed on dense-core cells
        (P01/MgO class) whose tight AO products are unresolved at any
        affordable base mesh; a parity-sized tail
        (``>= _RSGDF_PARITY_TAIL_RATIO x zeta_max``) lifts the
        ``+PARITY_HELD`` tag. rsgdf-only (any other ``gdf_method``
        fails closed). Setting it on a pure-DFT dim=3 multi-k run
        routes the Hartree through the cached-Lpq GDF J (the same
        tail-consuming machinery HF/hybrids ride) instead of the
        default EWALD_3D J, which builds no cderi and cannot consume a
        tail (IID 146); ``None`` (default) keeps the EWALD_3D routing.
    fit_screen_threshold
        Cauchy-Schwarz pre-screen of the three-centre GDF fit on the
        ``gdf_method='rsgdf'`` path (default ``0.0`` = off, exact). AO
        pairs whose Schwarz bound ``max_P sqrt((P|P)).sqrt((muν|muν))``
        stays below the threshold are dropped from every per-(k_i,k_j)
        fit tensor; kept/dropped pair counts are logged (no silent
        truncation). ``1e-10``-class values have reproduced the
        unscreened energy to well below SCF accuracy on the current
        s/p-heavy regression set; for higher angular momentum the
        polynomial factor is a conservative screening estimate rather
        than a formally proven bound. Forwarded to
        :func:`vibeqc.aux_basis.build_lpq_bloch_native_fft` -- every
        consumer of this driver (CCM ``run_ccm_rhf_gdf`` /
        ``run_ccm_rks_gdf``, RIJCOSX diagonal-J, KS) inherits it.
    bz_integration
        ``None`` / ``"smearing"`` use the existing global Aufbau or
        Fermi-Dirac occupation path. ``"gilat"`` selects the
        parameter-free Gilat-Raubenheimer net at T = 0; it cannot be
        combined with finite-temperature smearing.
    dft_plus_u_sites
        Optional Dudarev +U sites. For true multi-k meshes, the driver
        builds the k-averaged per-spin occupation matrix and adds the
        resulting ``S(k) V_U S(k)`` shift to each closed-shell Fock block.
    compute_gradient
        When ``True``, compute the analytic nuclear gradient of the
        converged total energy and store it on ``result.gradient``
        (``(n_atoms, 3)`` Ha/bohr). G-PBC-002 Item 4: the multi-k
        assembly rebuilds the SCF's exact shared-q rsgdf fit and
        differentiates every energy term (one-electron/W/nn, DF-J,
        DF-K over all q groups, exxdiv W-shift, XC Pulay for KS) --
        full-SCF FD gates on H2 (2,1,1) sit at 1.1e-8 (KRHF) /
        4.1e-10 (KRKS lda) / 1.5e-8 (KRKS pbe0) Ha/bohr. Supported
        envelope: 3D cells, ``gdf_method='rsgdf'`` with the cached-Lpq
        SCF (``use_compcell=True``; HF/hybrids promote automatically),
        uniform full-BZ meshes, T = 0 Aufbau occupations OR
        finite-temperature Fermi-Dirac smearing (the gradient is then
        dA/dR of the reported Mermin ``free_energy = E - T.S`` at the
        fractional-occupation D(k)/W(k) -- the occupation and mu
        responses vanish at self-consistency), pure and global-hybrid
        functionals; Schwarz-screened fits are differentiated at the
        SCF's fixed pair mask, and ``bz_integration='smearing'`` is
        accepted (it is the same occupation path). Non-Fermi-Dirac
        smearing flavors, ``bz_integration='gilat'``,
        IBZ/custom-weight k-points, range-separated functionals,
        ``k_exchange='cosx'``, and DFT+U raise
        ``NotImplementedError`` naming the reason
        (:func:`_reject_unsupported_multik_gradient`). Γ-only tuple
        meshes delegate the gradient to
        :func:`vibeqc.run_pbc_gdf_rhf` on the pure Γ fast path and
        fail closed on the legacy molecular-limit fallback. On a
        ``dim=2`` slab the dedicated route assembles the slab analytic
        gradient instead (G-PBC-002 § 6: bare 2D-Ewald nn + slab V_ne
        + S/T/W folds + signed-fit DF-J/K + BvK probe-charge W-shift,
        plus the XC Pulay for KS) inside the slab route's own
        envelope; full-SCF FD gates on the compact H2 (2,2,1) slab
        fixture sit in ``tests/test_slab_2d_routing.py``.
    check_energy_sanity
        When ``True`` (default) a post-SCF guard rejects a non-physical
        total energy: a converged run that lands at a runaway
        (``|E| > max(10.SZ^2, 100)`` Ha) or positive (unbound) energy
        RAISES ``RuntimeError`` instead of returning a converged garbage
        number (CLAUDE.md Sec.7 silent-corruption). Set ``False`` to bypass
        the guard for parity/debug scripts that want the raw value back;
        see :func:`_check_energy_sanity`.
    progress, verbose
        Live progress logging passthrough.
    """
    from .periodic_mdf import _MdfScfSource

    private_mdf = isinstance(_lpq_cache_builder, _MdfScfSource)
    if private_mdf:
        _lpq_cache_builder.require_driver(
            system, gdf_method=gdf_method, k_exchange=k_exchange, ibz_native=ibz_native)
    _finish_density_return = _gdf_density_return_finalizer(
        system, basis, kmesh, return_lattice_density, 1,
        lattice_density_memory_bytes,
    )
    caller_supplied_options = options is not None
    opts = _options_or_default(options, is_ks=functional is not None)
    lat_opts: LatticeSumOptions = opts.lattice_opts
    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", InitialGuess.AUTO),
        driver="run_krhf_periodic_gdf",
        supported=periodic_guess_capabilities('gdf', 'RHF', dim=getattr(system, "dim", 3), multi_k=True, transport='k'),
        restart_supplied=initial_density_k is not None,
    )
    from .guess import guess_ecp_context, validate_guess_ecp
    guess_mol = system.unit_cell_molecule()
    _guess_ecp = validate_guess_ecp(
        guess, guess_ecp_context(opts, molecule=guess_mol), guess_mol)
    if guess == InitialGuess.READ and initial_density_k is None:
        raise ValueError(
            "run_krhf_periodic_gdf: initial_guess=READ requires "
            "initial_density_k with one compatible density per k-point"
        )
    if (
        int(system.dim) != 3
        and guess == InitialGuess.SAP
    ):
        raise NotImplementedError(
            "run_krhf_periodic_gdf: initial_guess='SAP' requires a 3-D "
            "periodic system. The lattice SAP potential uses the 3-D Ewald "
            "split, so a lower-dimensional request cannot be replaced by "
            "Hcore."
        )
    requested_functional_name = (
        functional or str(getattr(opts, "functional", "") or "")
    )
    requested_functional = (
        Functional(requested_functional_name, 1)
        if requested_functional_name
        else None
    )
    requested_external_xc = bool(
        requested_functional is not None
        and getattr(requested_functional, "is_external", False)
    )
    if requested_external_xc:
        from .pbc_bipole_common import reject_bipole_ecp_options
        from .periodic_external_xc import (
            _grid_options_for_external,
            _require_zero_temperature_external_xc,
        )

        _require_zero_temperature_external_xc(
            getattr(opts, "smearing_temperature", 0.0),
            where="run_krhf_periodic_gdf",
        )

        try:
            reject_bipole_ecp_options(
                opts,
                driver="run_krhf_periodic_gdf external XC",
                basis=basis,
                system=system,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "run_krhf_periodic_gdf: ECP-bearing bases are not "
                "implemented with external XC; use an all-electron basis"
            ) from exc
        opts.grid = _grid_options_for_external(
            requested_functional,
            opts.grid if caller_supplied_options else None,
            where="run_krhf_periodic_gdf",
        )
    density_mixer = _canonical_gdf_density_mixer(density_mixer)
    fock_mixing_value = _resolve_fock_mixing(opts, fock_mixing)
    if int(system.dim) == 2:
        if requested_external_xc:
            raise NotImplementedError(
                "run_krhf_periodic_gdf: full-grid external XC is not yet "
                "wired to the signed 2D slab-GDF density domain. Use a "
                "1D/3D GDF calculation; 2D external XC remains gated."
            )
        slab_functional = (
            functional or str(getattr(opts, "functional", "") or "") or None
        )
        slab_result = _run_closed_shell_slab_gdf(
            system,
            basis,
            kmesh,
            opts,
            functional=slab_functional,
            aux_basis=aux_basis,
            aux_drop_eta=aux_drop_eta,
            linear_dep_threshold=linear_dep_threshold,
            gdf_linear_dep_threshold=gdf_linear_dep_threshold,
            apply_modrho=apply_modrho,
            fock_mixing_value=fock_mixing_value,
            level_shift_warmup_cycles=level_shift_warmup_cycles,
            use_compcell=use_compcell,
            apply_aft_correction=apply_aft_correction,
            aft_ft_convention=aft_ft_convention,
            aft_precision=aft_precision,
            rcut_strategy=rcut_strategy,
            rcut_precision=rcut_precision,
            k_exchange=k_exchange,
            gdf_method=gdf_method,
            rsgdf_ke_cutoff=rsgdf_ke_cutoff,
            rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
            fit_screen_threshold=fit_screen_threshold,
            bz_integration=bz_integration,
            density_mixer=density_mixer,
            dft_plus_u_sites=dft_plus_u_sites,
            initial_density_k=initial_density_k,
            check_energy_sanity=check_energy_sanity,
            progress=progress,
            verbose=verbose,
            compute_gradient=compute_gradient,
        )
        return _finish_density_return(slab_result)
    _reject_slab_dim(system, "run_krhf_periodic_gdf")
    # ECP context first: the Gamma fast path below must know about it.
    _ecp_ctx = _periodic_ecp_context(opts, system, "run_krhf_periodic_gdf")
    _ecp_ncore = _ecp_ctx[3] if _ecp_ctx is not None else 0
    _system_v = _ecp_ctx[4] if _ecp_ctx is not None else system
    plog = resolve_progress(progress, verbose=verbose)

    # ---------------- Γ fast path ----------------------------------
    # The Γ-fast-path delegates to run_rhf_periodic_gamma_gdf which
    # short-circuits J/K to Ewald-3D + molecular-limit-K on dim=3
    # (it doesn't have a compcell option). For Γ-only compcell SCF,
    # users should call ``vibeqc.run_pbc_gdf_rhf`` directly -- the
    # Γ-only driver that uses Lpq for both J and K. We attempted to
    # skip this fast path when use_compcell=True and fall through to
    # the multi-k branch with n_k=1, but the multi-k SCF loop's
    # per-k weighting doesn't degenerate cleanly to Γ-only
    # (gives ~5x incorrect energy on H2). Keeping the fast path means
    # ``use_compcell=True`` at ``kmesh=(1,1,1)`` is silently ignored
    # -- surfaced via a warning so users know to switch drivers.
    if k_exchange not in ("gdf", "cosx"):
        raise ValueError(
            f"run_krhf_periodic_gdf: k_exchange must be 'gdf' or "
            f"'cosx'; got {k_exchange!r}"
        )
    if k_exchange == "cosx" and not use_compcell:
        raise ValueError(
            "run_krhf_periodic_gdf: k_exchange='cosx' requires "
            "use_compcell=True (the COSX K rides the cached-Lpq GDF-J "
            "path; the legacy Ewald-3D path has its own exchange)"
        )
    if gdf_method not in ("compcell", "rsgdf", "mdf"):
        raise ValueError(
            f"run_krhf_periodic_gdf: gdf_method must be 'compcell', "
            f"'rsgdf', or 'mdf'; got {gdf_method!r}"
        )
    bulk_sr = gdf_method == "rsgdf" and int(system.dim) == 3
    if bulk_sr and rsgdf_tail_ke_cutoff is not None:
        warnings.warn(
            "rsgdf_tail_ke_cutoff is obsolete for the SR/LR fit; "
            "use rsgdf_g_precision to control the raw integral error",
            DeprecationWarning, stacklevel=2,
        )
        rsgdf_tail_ke_cutoff = None
    if float(fit_screen_threshold) < 0.0:
        raise ValueError(
            "run_krhf_periodic_gdf: fit_screen_threshold must be >= 0; "
            f"got {fit_screen_threshold}"
        )
    if float(fit_screen_threshold) > 0.0 and gdf_method != "rsgdf":
        # Loud, not silent: the Schwarz fit screen lives in the rsgdf
        # builder (build_lpq_bloch_native_fft); a threshold on the
        # compcell/mdf routes would be silently ignored otherwise.
        raise NotImplementedError(
            "run_krhf_periodic_gdf: fit_screen_threshold is implemented "
            f"for gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    if requested_external_xc:
        if xc_density_domain == PeriodicXCDensityDomain.MOLECULAR_HOME:
            raise ValueError(
                "run_krhf_periodic_gdf: a full-grid external functional "
                "requires the periodic-lattice XC density domain"
            )
        if not bool(getattr(opts, "use_periodic_becke", False)):
            raise ValueError(
                "run_krhf_periodic_gdf: full-grid external XC requires "
                "use_periodic_becke=True; a molecular atom partition is "
                "not valid for a compact periodic density"
            )
        external_image_radius = float(
            getattr(opts, "becke_image_radius_bohr", 0.0)
        )
        if not np.isfinite(external_image_radius) or external_image_radius <= 0.0:
            raise ValueError(
                "run_krhf_periodic_gdf: external XC requires "
                "becke_image_radius_bohr to be finite and > 0"
            )
        if float(lat_opts.cutoff_bohr) < external_image_radius:
            raise ValueError(
                "run_krhf_periodic_gdf: external XC cutoff_bohr must be at "
                "least becke_image_radius_bohr"
            )
        lat_opts.becke_image_radius_bohr = external_image_radius
        xc_density_domain = PeriodicXCDensityDomain.PERIODIC_LATTICE

    # The general engine also owns the Gamma limit of the SR/LR route.
    # Convergence controls must never select a different Coulomb gauge.
    gamma_info = None if bulk_sr else _gamma_kmesh_info(system, kmesh)
    explicit_periodic_gamma_xc = bool(
        gamma_info is not None
        and xc_density_domain == PeriodicXCDensityDomain.PERIODIC_LATTICE
    )
    # An explicit periodic XC domain uses the generic finite-torus engine even
    # at Gamma.  Its exact difference-closed cell set is the N_k=1 limit of
    # the multi-k fold; the historical Gamma fast paths below own smaller
    # home/overlap domains and cannot honour this contract.
    if (
        gamma_info is not None
        and not explicit_periodic_gamma_xc
        and dft_plus_u_sites
    ):
        raise NotImplementedError(
            "run_krhf_periodic_gdf: dft_plus_u_sites is wired on the true "
            "multi-k GDF loop only. Use a non-Gamma k-mesh such as "
            "(1,1,2), or omit dft_plus_u_sites for the Gamma GDF fast path."
        )
    # Closed-shell Γ HF *and* KS in run_pbc_gdf_rhf's supported domain
    # delegate to that PySCF-µHa-validated driver (exxdiv='ewald'), so
    # kmesh=(1,1,1) is the Nk=1 limit of the multi-k exxdiv='ewald' path --
    # no convention discontinuity vs (2,1,1)+, and consistent with
    # run_periodic_job's default-Γ routing. KS delegation landed 2026-07-09
    # (the Finding-§4 residual): the legacy molecular-limit gamma driver's
    # exchange channel carries NO exxdiv convention at all -- its full-range
    # real-space K left Γ-path HYBRID KS +8.32e-2 Ha off the exxdiv-matched
    # real-Γ direct route / external PySCF KRKS on rocksalt LiH/STO-3G PBE0
    # (not the strict-zero-mode offset a_x·ξ·N_e/2 = 0.297 Ha, a distinct
    # truncated-lattice-sum gauge; see tests/test_ccm_rks_direct.py).
    # The legacy gamma driver (below) stays the fallback for charged /
    # open-shell / dim<3 cells, finite-T smearing, fock-mixing / level-shift
    # convergence aids, and explicit use_compcell (kept on the historical
    # warning path). The gate mirrors run_pbc_gdf_rhf's preconditions + the
    # knobs it honours (DIIS + damping, not fock_mixing / level_shift /
    # smearing).
    if (
        gamma_info is not None
        and not explicit_periodic_gamma_xc
        and not use_compcell
        and density_mixer is None
    ):
        # Resolve KS-via-options too, so the result class + wrapper agree
        # with what run_pbc_gdf_rhf will actually run.
        _gamma_func = (
            functional or str(getattr(opts, "functional", "") or "") or None
        )
        _gamma_q_nuc = float(sum(atom.Z for atom in system.unit_cell))
        _gamma_n_elec = int(system.n_electrons())
        _pure_gdf_gamma_ok = (
            int(system.dim) == 3
            and _ecp_ctx is None
            and _gamma_n_elec % 2 == 0
            and int(system.multiplicity) == 1
            and abs(_gamma_q_nuc - _gamma_n_elec) <= 0.5
            and float(getattr(opts, "smearing_temperature", 0.0) or 0.0) <= 0.0
            and fock_mixing_value == 0.0
            and float(getattr(opts, "level_shift", 0.0) or 0.0) == 0.0
        )
        if _pure_gdf_gamma_ok:
            from .pbc_gdf import run_pbc_gdf_rhf

            with _gamma_restart_options(opts, initial_density_k, basis.nbasis):
                gamma = run_pbc_gdf_rhf(
                    system,
                    basis,
                    opts,
                    functional=_gamma_func,
                    aux_basis=aux_basis,
                    aux_drop_eta=aux_drop_eta,
                    exxdiv="ewald",
                    gdf_method=gdf_method,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    linear_dep_threshold=linear_dep_threshold,
                    gdf_linear_dep_threshold=gdf_linear_dep_threshold,
                    fit_screen_threshold=fit_screen_threshold,
                    compute_gradient=compute_gradient,
                    progress=plog,
                    verbose=verbose,
                )
            if initial_density_k is not None:
                gamma.guess_selection = periodic_result_selection(
                    system, opts.initial_guess, restarted=True)
            wrapped = _wrap_gamma_gdf_result(
                gamma,
                gamma_info,
                functional=_gamma_func,
                result_cls=(
                    PeriodicKRKSGDFResult
                    if _gamma_func is not None
                    else PeriodicKRHFGDFResult
                ),
            )
            return _finish_density_return(wrapped)
    if (
        gamma_info is not None
        and not explicit_periodic_gamma_xc
        and rsgdf_tail_ke_cutoff is not None
    ):
        raise NotImplementedError(
            "run_krhf_periodic_gdf: rsgdf_tail_ke_cutoff requires the pure "
            "PBC-GDF Gamma fast path or a true multi-k mesh. This Gamma "
            "k-mesh would fall back to the legacy molecular-limit GDF driver, "
            "which has no high-|G| tail correction."
        )
    if (
        gamma_info is not None
        and not explicit_periodic_gamma_xc
        and use_compcell
        and density_mixer is None
    ):
        plog.info(
            "  WARNING: use_compcell=True at kmesh=(1,1,1) is currently "
            "ignored (the Γ-fastpath delegates to run_rhf_periodic_gamma_gdf "
            "which doesn't support compcell). For Γ-only compcell SCF, "
            "call vibeqc.run_pbc_gdf_rhf(...) directly. Continuing with "
            "the legacy Ewald-3D + molecular-limit-K path. "
            "(k_exchange is ignored on this path too.)"
        )
    if (
        gamma_info is not None
        and not explicit_periodic_gamma_xc
        and float(fit_screen_threshold) > 0.0
    ):
        # Reaching here means the pure PBC-GDF Γ fast path (which
        # supports the screen via run_pbc_gdf_rhf) was NOT taken --
        # this Γ k-mesh falls back to the legacy molecular-limit GDF
        # driver, which has no screened fit. Loud, not silent.
        raise NotImplementedError(
            "run_krhf_periodic_gdf: fit_screen_threshold at a Γ k-mesh "
            "requires the pure PBC-GDF Γ fast path (rsgdf, closed-shell, "
            "no density_mixer / fock_mixing / level_shift / smearing / "
            "compcell); this run would fall back to the legacy "
            "molecular-limit GDF driver, which has no screened fit. "
            "Call vibeqc.run_pbc_gdf_rhf(..., fit_screen_threshold=...) "
            "directly, adjust the conflicting options, or drop the "
            "threshold."
        )
    if (
        gamma_info is not None
        and not explicit_periodic_gamma_xc
        and compute_gradient
        and density_mixer is None
    ):
        # Reaching here means the pure PBC-GDF Γ fast path (which
        # computes the gradient via run_pbc_gdf_rhf) was NOT taken --
        # this Γ k-mesh falls back to the legacy molecular-limit GDF
        # driver, which has no analytic gradient. Loud, not silent
        # (the rsgdf_tail_ke_cutoff / fit_screen_threshold precedent).
        raise NotImplementedError(
            "run_krhf_periodic_gdf: compute_gradient at a Γ k-mesh "
            "requires the pure PBC-GDF Γ fast path (rsgdf, closed-shell, "
            "no use_compcell / density_mixer / fock_mixing / level_shift "
            "/ smearing); this run would fall back to the legacy "
            "molecular-limit GDF driver, which has no analytic gradient. "
            "Call vibeqc.run_pbc_gdf_rhf(..., compute_gradient=True) "
            "directly or adjust the conflicting options."
        )
    if (
        gamma_info is not None
        and _ecp_ctx is None
        and not explicit_periodic_gamma_xc
        and density_mixer is None
    ):
        from .pbc_gdf import _reject_legacy_gamma_gdf

        _reject_legacy_gamma_gdf(system, basis, "run_krhf_periodic_gdf")
        with _gamma_restart_options(opts, initial_density_k, basis.nbasis):
            gamma = run_rhf_periodic_gamma_gdf(
                system,
                basis,
                opts,
                functional=functional,
                aux_basis=aux_basis,
                aux_drop_eta=aux_drop_eta,
                linear_dep_threshold=linear_dep_threshold,
                gdf_linear_dep_threshold=gdf_linear_dep_threshold,
                apply_modrho=apply_modrho,
                fock_mixing=fock_mixing_value,
                level_shift_warmup_cycles=level_shift_warmup_cycles,
                progress=plog,
                verbose=verbose,
            )
        if initial_density_k is not None:
            gamma.guess_selection = periodic_result_selection(
                system, opts.initial_guess, restarted=True)
        result_cls = (
            PeriodicKRKSGDFResult if functional is not None else PeriodicKRHFGDFResult
        )
        wrapped = _wrap_gamma_gdf_result(
            gamma,
            gamma_info,
            functional=functional,
            result_cls=result_cls,
        )
        return _finish_density_return(wrapped)

    # ---------------- Multi-k branch ------------------------------
    func_name = functional or str(getattr(opts, "functional", "") or "")
    is_ks = bool(func_name)
    func = Functional(func_name, 1) if is_ks else None
    if requested_external_xc and compute_gradient:
        raise NotImplementedError(
            "run_krhf_periodic_gdf: analytic gradients are not implemented "
            "for full-grid external XC functionals; use whole-SCF finite "
            "differences"
        )
    reject_periodic_gdf_unsupported_functional(
        func, where="run_krhf_periodic_gdf"
    )
    # Range-separated policy (shared periodic_screened_exchange):
    # HSE-type screened hybrids (c_full = 0, exchange = c_sr *
    # K_erfc(omega_screen)) are supported on the k_exchange='cosx'
    # backend via the SR(split) + reciprocal band composition -- the
    # erfc kernel has no G -> 0 divergence, so NO exxdiv Madelung
    # shift applies to the screened exchange. The Lpq-contracted GDF K
    # remains full-range only and fails closed on any range-separated
    # functional; c_full > 0 functionals (wb97x, cam-b3lyp, ...) fail
    # closed on every backend.
    screened_omega = None
    if (
        k_exchange == "cosx"
        and func is not None
        and bool(getattr(func, "is_range_separated", False))
    ):
        from .periodic_screened_exchange import resolve_periodic_exchange

        _exx = resolve_periodic_exchange(
            func, where="run_krhf_periodic_gdf(k_exchange='cosx')"
        )
        alpha = float(_exx.c_sr)
        screened_omega = float(_exx.omega_screen)
    else:
        # The Lpq-contracted K is full-range only; screened hybrids
        # must not silently run as their full-range twins.
        reject_unscreened_range_separated(func, where="run_krhf_periodic_gdf")
        alpha = float(func.hf_exchange_fraction) if func is not None else 1.0
    level_shift = float(getattr(opts, "level_shift", 0.0))
    max_iter = int(opts.max_iter)
    warmup_cycles = _resolve_level_shift_warmup_cycles(
        opts,
        level_shift=level_shift,
        max_iter=max_iter,
        override=level_shift_warmup_cycles,
    )
    # Explicit per-iteration schedule (unified with the molecular
    # drivers). Empty ⇒ the warm-up step function; non-empty ⇒ resolved
    # per iteration by the shared C++ helper.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError("run_krhf_periodic_gdf: smearing_temperature must be >= 0")
    if bz_integration is not None:
        bz_integration = str(bz_integration).strip().lower()
        if bz_integration not in ("smearing", "gilat"):
            raise ValueError(
                "run_krhf_periodic_gdf: bz_integration must be None, "
                f"'smearing', or 'gilat'; got {bz_integration!r}"
            )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise NotImplementedError(
            "run_krhf_periodic_gdf: bz_integration='gilat' is a "
            "sharp-Fermi-surface occupation backend and cannot be combined "
            "with finite-temperature smearing."
        )
    label = f"KRKS {func_name}" if is_ks else "KRHF"

    n_elec = system.n_electrons() - _ecp_ncore
    if n_elec % 2 != 0:
        raise ValueError(
            "run_krhf_periodic_gdf: closed-shell RHF/RKS requires "
            f"even electron count; got {n_elec}"
            + (f" ({_ecp_ncore} ECP core electrons removed)" if _ecp_ncore else "")
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_krhf_periodic_gdf: closed-shell RHF/RKS requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2

    if requested_external_xc and isinstance(kmesh, (KPoints, BlochKMesh)):
        from .periodic_external_xc import _reject_reduced_kmesh

        input_kmesh_bloch = (
            kmesh.to_bloch_kmesh() if isinstance(kmesh, KPoints) else kmesh
        )
        _reject_reduced_kmesh(input_kmesh_bloch, system)

    # Symmetry-reduced (IBZ) input: adopt the expanded full-BZ mesh as
    # THE kmesh before anything is derived from it, so the k-point
    # arrays and ``kmesh_bloch`` below describe the same k list. The
    # first expansion (375b6363d) expanded only the arrays inside
    # _kmesh_to_kpoints_weights and left ``kmesh_bloch`` on the wedge,
    # so every KRKS/KUKS IBZ job crashed folding n_full densities onto
    # the wedge mesh ("size mismatch across k inputs" on H2 (2,2,2):
    # 8 densities vs the 6-point wedge).
    kmesh_full = _expand_ibz_kmesh_to_full_bz(system, kmesh)
    if kmesh_full is not None:
        kmesh = kmesh_full
    kpoints_cart, weights = _kmesh_to_kpoints_weights(system, kmesh)
    n_k = kpoints_cart.shape[0]

    # Multi-k analytic-gradient envelope (G-PBC-002 Item-4 rung 6):
    # fail closed BEFORE the SCF on anything the gradient assembly
    # does not differentiate. use_compcell is effective-after-promotion
    # (HF/hybrids flip it on below); pure-DFT Ewald-3D J stays rejected.
    if compute_gradient:
        _reject_unsupported_multik_gradient(
            "run_krhf_periodic_gdf",
            dim=int(system.dim),
            gdf_method=gdf_method,
            smearing_temperature=smearing_T,
            k_exchange=k_exchange,
            screened_omega=screened_omega,
            functional_is_range_separated=bool(
                getattr(func, "is_range_separated", False)
            ),
            weights=weights,
            use_compcell_effective=bool(bulk_sr or use_compcell or alpha > 0.0),
            bz_integration=bz_integration,
            dft_plus_u_sites=dft_plus_u_sites,
            ibz_native=bool(ibz_native),
            # The KRHF surface has no flavor control: smearing_T routes
            # through SmearingOptions.from_legacy_kwarg -> Fermi-Dirac.
            smearing_flavor="fermi-dirac",
        )

    # Resolve kmesh to BlochKMesh + lattice cells for Ewald Fock builder.
    if isinstance(kmesh, BlochKMesh):
        kmesh_bloch = kmesh
    elif isinstance(kmesh, KPoints):
        kmesh_bloch = kmesh.to_bloch_kmesh()
    else:
        mesh = _mesh_tuple_for_system(system, kmesh)
        kmesh_bloch = _mp_native(system, list(mesh), [0, 0, 0], False)
    # Admit the bulk AO domain before any lattice/XC cell enumeration or
    # quadrature allocation (#99). The later integral-only guard was too
    # late even for the direct cell lists printed in the setup log.
    # Preserve the historical low-dimensional cutoff-resolution order.
    oneel_lat_opts = None
    if int(system.dim) == 3:
        oneel_lat_opts = _oneel_lattice_opts(
            system, basis, lat_opts,
            rcut_strategy=rcut_strategy, k_points_cart=kpoints_cart, plog=plog,
        )
        _preflight_gdf_oneel_memory(
            system, basis, oneel_lat_opts, n_kpoints=n_k,
            ecp_active=_ecp_ctx is not None, plog=plog,
        )
    cells = _direct_cells(system, lat_opts.cutoff_bohr)
    xc_cells = _xc_density_cells_for_domain(
        system,
        lat_opts,
        cells,
        xc_density_domain,
    )

    aux_name = aux_basis or default_aux_for(basis.name)

    plog.banner(f"run_krhf_periodic_gdf  {label}  kmesh={n_k} k-points")

    dense_core_parity_held = not bulk_sr and _warn_multik_dense_core_gdf_parity_hold(
        system,
        gdf_method,
        basis,
        rsgdf_tail_ke_cutoff,
        rsgdf_ke_cutoff,
        "run_krhf_periodic_gdf",
    )
    plog.info(
        f"{label} multi-k native GDF / aux={aux_name}, "
        f"cutoff={lat_opts.cutoff_bohr:.2f} bohr"
    )
    if use_gilat:
        plog.info("bz integration: Gilat-Raubenheimer sharp-Fermi net")
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    dim = int(system.dim)
    active_lengths = [
        float(np.linalg.norm(np.asarray(system.lattice, dtype=float)[:, i]))
        for i in range(dim)
    ]
    plog.info(
        f"periodicity: dim={dim}D, active lengths="
        + ", ".join(f"{x:.3f}" for x in active_lengths)
        + " bohr"
    )
    n_int_cells = len(direct_lattice_cells(system, lat_opts.cutoff_bohr))
    n_nuc_cells = len(direct_lattice_cells(system, lat_opts.nuclear_cutoff_bohr))
    plog.info(
        "lattice cells: "
        f"one-electron/GDF cutoff -> {n_int_cells}, "
        f"nuclear cutoff -> {n_nuc_cells}"
    )
    dump_active_settings(
        plog,
        [
            ("PeriodicKSOptions" if is_ks else "PeriodicRHFOptions", opts),
            ("LatticeSumOptions", lat_opts),
            (
                "k-GDF kwargs",
                {
                    "functional": func_name or None,
                    "hf_exchange_fraction": alpha,
                    "fock_mixing": fock_mixing_value,
                    "fmixing_percent": 100.0 * fock_mixing_value,
                    "level_shift": level_shift,
                    "level_shift_warmup_cycles": warmup_cycles,
                    "smearing_temperature": smearing_T,
                    "aux_basis": aux_name,
                    "aux_drop_eta": float(aux_drop_eta),
                    "linear_dep_threshold": float(linear_dep_threshold),
                    "gdf_linear_dep_threshold": float(gdf_linear_dep_threshold),
                    "apply_modrho": bool(apply_modrho),
                    "n_kpoints": n_k,
                },
            ),
        ],
    )

    # ---- Functional + grid ----------------------------------------
    grid = None
    if is_ks:
        grid_options = getattr(opts, "grid", None)
        if grid_options is None:
            grid_options = GridOptions()
        if requested_external_xc or bool(
            getattr(opts, "use_periodic_becke", False)
        ):
            grid = build_periodic_becke_grid(
                system,
                grid_options=grid_options,
                image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
            )
        else:
            grid = build_grid(system.unit_cell_molecule(), grid_options)

    # ---- Real-space one-electron integrals ------------------------
    # Resolve one AO image domain for S/T, nuclear attraction and ECPs.
    # The independent projector/nuclear-image cutoff stays explicit.
    if oneel_lat_opts is None:
        oneel_lat_opts = _oneel_lattice_opts(
            system, basis, lat_opts,
            rcut_strategy=rcut_strategy, k_points_cart=kpoints_cart, plog=plog,
        )
    # Recheck after setup allocations, now including retained quadrature
    # and the current available budget. This does not change the AO domain.
    _preflight_gdf_oneel_memory(
        system, basis, oneel_lat_opts, n_kpoints=len(kpoints_cart),
        ecp_active=_ecp_ctx is not None, grid=grid, plog=plog,
    )
    with plog.stage(
        "integrals_lattice",
        detail=f"S/T at cutoff {oneel_lat_opts.cutoff_bohr:.2f} bohr, "
        "V on the screened AO-pair domain",
    ):
        S_lat = compute_overlap_lattice(basis, system, oneel_lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, oneel_lat_opts)
        from .periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        # Always use Ewald-3D gauge for V_ne and e_nuc -- J/K are
        # built via build_periodic_fock_ewald3d_k (Ewald gauge). With an ECP
        # the nuclear frame carries Z_eff and the lattice-summed V_ECP joins
        # the one-electron set.
        gauge_lat_opts = _gauge_lat_opts_for_v_ne_and_e_nuc(oneel_lat_opts, system)
        V_lat = compute_nuclear_lattice_dispatch(basis, _system_v, gauge_lat_opts)
        V_ecp_lat = (
            _ecp_lattice_blocks(basis, system, oneel_lat_opts, _ecp_ctx)
            if _ecp_ctx is not None
            else None
        )

    # ---- Per-k S(k), Hcore(k), canonical orthog X(k) -------------
    S_k: List[np.ndarray] = []
    Hcore_k: List[np.ndarray] = []
    X_k: List[np.ndarray] = []
    n_kept_k: List[int] = []
    # Overlap spectrum extremes over the mesh, for the linear-dependence
    # report: these were previously computed inside the preflight and
    # discarded, leaving the .out with no way to say how conditioned the
    # periodic basis actually was.
    _s_lo, _s_hi = float("inf"), float("-inf")
    for k_idx in range(n_k):
        k_arr = kpoints_cart[k_idx]
        Sk = np.asarray(bloch_sum(S_lat, k_arr))
        Tk = np.asarray(bloch_sum(T_lat, k_arr))
        Vk = np.asarray(bloch_sum(V_lat, k_arr))
        if V_ecp_lat is not None:
            Vk = Vk + np.asarray(bloch_sum(V_ecp_lat, k_arr))
        Sk = 0.5 * (Sk + Sk.conj().T)
        Hk = 0.5 * ((Tk + Vk) + (Tk + Vk).conj().T)
        _gdf_overlap_preflight(
            Sk,
            plog=plog,
            label=f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})",
            basis=basis,
        )
        _ev = np.linalg.eigvalsh(Sk)
        _s_lo = min(_s_lo, float(_ev[0]))
        _s_hi = max(_s_hi, float(_ev[-1]))
        Xk, n_kept = _canonical_orthogonalizer_complex(
            Sk,
            linear_dep_threshold,
            normalize_diag_first=True,
        )
        if n_occ > n_kept:
            raise RuntimeError(
                "run_krhf_periodic_gdf: canonical orthogonalisation at "
                f"k = {k_arr} dropped too many directions "
                f"(n_occ={n_occ}, n_kept={n_kept}); loosen "
                "linear_dep_threshold or pick a less redundant basis."
            )
        S_k.append(Sk)
        Hcore_k.append(Hk)
        X_k.append(Xk)
        n_kept_k.append(n_kept)

    dftu_sites = list(dft_plus_u_sites or ())
    dftu_sites_cxx = []
    dftu_ao_groups: List[List[int]] = []
    if dftu_sites:
        from ._vibeqc_core import _HubbardSiteCxx
        from .dft_plus_u import ao_group_indices

        ao_groups_map = ao_group_indices(basis)
        for site in dftu_sites:
            key = (int(site.atom_index), int(site.l))
            if key not in ao_groups_map:
                raise ValueError(
                    f"HubbardSite(atom_index={site.atom_index}, l={site.l}) "
                    "has no AOs in the basis."
                )
            dftu_sites_cxx.append(
                _HubbardSiteCxx(
                    int(site.atom_index),
                    int(site.l),
                    float(site.U_eff_hartree),
                )
            )
            dftu_ao_groups.append(list(ao_groups_map[key]))
        plog.info(
            "DFT+U: closed-shell multi-k GDF projector active "
            f"({len(dftu_sites_cxx)} site(s))"
        )

    if int(system.dim) == 3:
        # Converged Ewald nuclear energy. nuclear_repulsion_per_cell
        # (EWALD_3D) truncates its real-space sum at nuclear_cutoff_bohr,
        # which on dense ionic cells is unconverged at the 1e-5 Ha level
        # with a spurious geometry dependence (see pbc_gdf._pbc_gdf_gamma_setup).
        e_nuc = float(ewald_nuclear_repulsion(_system_v))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(_system_v, gauge_lat_opts))

    if ibz_native and k_exchange != "gdf":
        raise NotImplementedError(
            "ibz_native=True applies to the fitted GDF exchange "
            f"(k_exchange='gdf'); got k_exchange={k_exchange!r}. The COSX "
            "backend builds K in real space from its own bridge and never "
            "touches the per-(k_i,k_j) cderi cache this flag reduces."
        )
    # ---- IBZ-native exchange (opt-in) --------------------------------
    # Exchange is the only n_k^2 term. Under this flag it is built at the
    # irreducible wedge and symmetry-transported to the rest of the mesh,
    # which reduces BOTH the cderi build (n_k^2 -> n_IBZ x n_k pairs) and
    # the per-iteration contraction. Everything else -- diagonalisation
    # at every k, occupations, the energy expression, the result shape --
    # is untouched, so bands/DOS/COOP/QVF consumers see no change.
    _ibz_native_state = None
    _ibz_symmetry_checked: List[bool] = []
    if ibz_native:
        if alpha == 0.0:
            raise NotImplementedError(
                "ibz_native=True has no effect without exact exchange "
                "(alpha == 0): the n_k^2 term it reduces is the HF "
                "exchange. Drop the flag for a pure functional."
            )
        _ibz_native_state = _resolve_ibz_native_state(
            system, kpoints_cart, kmesh, weights, plog
        )
        _rows_n = len(_ibz_native_state[0])
        plog.info(
            f"IBZ-native exchange: {_rows_n} of {n_k} k-points carry the "
            f"exchange build ({n_k / max(_rows_n, 1):.1f}x fewer cderi "
            f"pairs); the rest are symmetry-transported."
        )
        if float(getattr(lat_opts, "cutoff_bohr", 0.0)) < 20.0:
            warnings.warn(
                "ibz_native=True at lattice cutoff "
                f"{float(getattr(lat_opts, 'cutoff_bohr', 0.0)):.1f} bohr: "
                "the symmetry transport is only as accurate as the cell "
                "list. Measured on LiH FCC (2,2,2) the K transport "
                "residual is 1.5e-3 at 15 bohr (comparable to the "
                "energy's own drift to the next cutoff), 8.2e-6 at 20 "
                "and 1.1e-8 at 26. Use cutoff_bohr >= 20 for the "
                "transport to stay below the truncation floor.",
                stacklevel=2,
            )


    if bulk_sr:
        # Pure DFT uses the same fitted Hartree as hybrids; no unfitted
        # Ewald-J fallback is selected when exact exchange is zero.
        #
        # This promotion overrides an explicit use_compcell=False, and it
        # fires on the shipped default (gdf_method='rsgdf', dim == 3), so it
        # is the usual case rather than a corner. It cannot refuse instead:
        # ``use_compcell`` defaults to False, so a caller who passed False is
        # indistinguishable from one who said nothing. Say so in the log, like
        # the three promotions below -- a silent override here is what let a
        # regression test be written against a premise the route never had
        # (#275).
        if not use_compcell:
            plog.info(
                "multi-k bulk short-range GDF (gdf_method='rsgdf', dim=3): "
                "routing to the cached-Lpq GDF Hartree, so J is the fitted "
                "one, not the analytic EWALD_3D J. use_compcell=False is not "
                "available on this route."
            )
        use_compcell = True

    # Route HF/hybrid multi-k to the exxdiv-corrected compcell GDF path.
    #
    # The exxdiv Madelung exchange-divergence K-shift lives ENTIRELY inside
    # the use_compcell=True branch below; the use_compcell=False Ewald-3D-K
    # Fock builder does not apply it. For HF/hybrid (alpha > 0) that leaves
    # the finite-k-mesh exchange divergence uncorrected, so the energy is
    # wrong by the Madelung shift (catastrophically -- +17 Ha -- on sparse
    # meshes). The compcell + exxdiv='ewald' path is PySCF-µHa-correct, so we
    # route HF/hybrid there automatically rather than return wrong energies.
    # Pure DFT (alpha == 0, no HF exchange) would keep the cheaper Ewald-3D
    # path, but on the shipped dim=3 rsgdf route ``bulk_sr`` above has already
    # promoted it, so only non-rsgdf methods reach the Ewald-3D J here.
    # [Maintainer decision 2026-06-04: route to correct path.]
    if not use_compcell and alpha > 0.0:
        use_compcell = True
        plog.info(
            "multi-k HF/hybrid exchange: routing to the compcell GDF path "
            "with exxdiv='ewald' (the use_compcell=False Ewald-3D-K path "
            "omits the exxdiv Madelung exchange-divergence correction)."
        )
    # Route pure-DFT multi-k on dim<3 (vacuum-padded wire/slab) to the
    # cached-Lpq GDF Hartree as well. The EWALD_3D J is 3D-only: its
    # analytic-FT kernel raises on dim<3, so ewald_3d_j_blocks silently
    # degrades to the diagnostic FFT-Poisson grid backend there -- wrong
    # on vacuum-padded low-D cells (the 2026-07-09 Finding-§4 residual:
    # dim=1 H2-chain (3,1,1) PBE reported -3.0897 Ha/cell where its own
    # stored density evaluates to -3.7814 under the same functional and
    # the variational minimum sits at -3.8947; the Lpq J lands at the
    # real-Γ direct route's minimum to ~1e-10). HF/hybrids already ride
    # the cached-Lpq path via the exxdiv routing above; this closes the
    # same gap for the alpha == 0 KS branch. Pure DFT on dim=3 keeps the
    # cheaper EWALD_3D path (validated: 29 uHa vs external PySCF at a
    # non-TRIM mesh, tests/test_krks_gdf_xc_density.py).
    if not use_compcell and is_ks and int(system.dim) != 3:
        use_compcell = True
        plog.info(
            "multi-k pure-KS on dim<3: routing to the cached-Lpq GDF "
            "Hartree (the EWALD_3D J falls back to the diagnostic "
            "FFT-Poisson grid backend on dim<3, which mis-sums the "
            "vacuum-padded Coulomb)."
        )

    # Multi-k rsgdf tail consumption (IID 146). The high-|G| tail
    # completion exists only on the rsgdf cderi builders, so a tail on
    # any other multi-k gdf_method would be silently ignored -- fail
    # closed instead (the run_kuhf_periodic_gdf precedent).
    if rsgdf_tail_ke_cutoff is not None and gdf_method != "rsgdf":
        raise NotImplementedError(
            "run_krhf_periodic_gdf: rsgdf_tail_ke_cutoff is implemented "
            f"for gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    # Pure-DFT (alpha == 0) multi-k on dim=3 keeps the EWALD_3D
    # real-space J by default -- a machinery with no fitted cderi and
    # therefore no high-|G| tail to extend. (On gdf_method='rsgdf' that
    # default no longer survives ``bulk_sr`` above, so this promotion is
    # reachable only for the other dim=3 methods.) An explicit
    # rsgdf_tail_ke_cutoff there was accepted, echoed in the .out, and
    # silently IGNORED (IID 146: MgO/STO-3G (2,2,2) KRKS-LDA tailed vs
    # untailed byte-identical at -271.531535367637, while the same knob
    # moves multi-k KRHF on the same cell by +0.505030889141 Ha). Route
    # the run to the cached-Lpq GDF Hartree -- the same
    # method-independent, tail-consuming J machinery HF/hybrids
    # (alpha > 0 promotion above) and dim<3 pure-KS already ride -- so
    # the knob is consumed. tail=None keeps the default EWALD_3D
    # routing bit-identical.
    # (The IID 518 dense-core auto-tail is resolved earlier, before the
    # parity-hold classifier, so the resolved tail feeds the tag; by the
    # time control reaches here rsgdf_tail_ke_cutoff already carries it
    # and the promotion below fires on it exactly like an explicit knob.)
    if not use_compcell and rsgdf_tail_ke_cutoff is not None:
        use_compcell = True
        plog.info(
            "multi-k pure-DFT with rsgdf_tail_ke_cutoff: routing to the "
            "cached-Lpq GDF Hartree (the default EWALD_3D J builds no "
            "cderi and has no high-|G| tail; the knob would otherwise "
            "be silently ignored)."
        )

    # ---- Aux basis + Lpq(q) cache (GDF density-fitting path) -------
    # When use_compcell=True, build per-(k_i,k_j) Lpq once and contract
    # J/K from the cached cderi each iteration (like PySCF's GDF).
    # When use_compcell=False, the aux is built but not used -- the
    # per-iteration Fock goes through EWALD_3D real-space J/K instead.
    mol = system.unit_cell_molecule()
    with plog.stage("aux_basis", detail=aux_name):
        aux = make_aux_basis_set(
            mol,
            aux_name=aux_name,
            drop_eta=float(aux_drop_eta),
        )
    plog.info(f"aux basis: {aux_name}  ({aux.nbasis} BFs / {aux.nshells} shells)")

    # Per-pair Lpq cache: lpq_cache[(ki, kj)] = Lpq(k_i,k_j) tensor.
    lpq_cache: Dict[Tuple[int, int], np.ndarray] = {}
    # Retained fit dimension; stays 0 on the routes that build no cderi
    # (EWALD_3D real-space J/K), which is the honest value to report.
    n_fit = 0
    if use_compcell:
        # The off-diagonal (k_i != k_j) Lpq tensors exist solely for
        # the GDF exchange contraction; the COSX exchange backend
        # works in real space and never touches them -- only the
        # diagonal J pairs are built then (the structural setup +
        # memory saving of the COSX route).
        need_k_pairs = (alpha != 0.0 and k_exchange == "gdf") or guess == InitialGuess.PATOM

        # Bulk RSGDF uses the admitted real-space SR/reciprocal LR cache
        # below, including private symmetry adapters. The legacy pair
        # builder remains the lower-dimensional source. MDF and compensated
        # cell fits retain their separate source dispatch here.
        q_metric_cache = None
        if gdf_method == "rsgdf":
            aux_modrho = aux if bulk_sr else make_modrho_aux_basis(aux, mol)
            q_metric_cache = {}

            def _build_pair_lpq(
                ki: np.ndarray,
                kj: np.ndarray,
                *,
                canonical_auxiliary_basis: bool = False,
            ):
                return build_lpq_bloch_native_fft(
                    system,
                    basis,
                    aux_modrho,
                    ki,
                    kj,
                    ke_cutoff=float(rsgdf_ke_cutoff),
                    tail_ke_cutoff=(
                        float(rsgdf_tail_ke_cutoff)
                        if rsgdf_tail_ke_cutoff is not None
                        else None
                    ),
                    lat_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    fit_screen_threshold=float(fit_screen_threshold),
                    progress=plog,
                    _q_metric_cache=q_metric_cache,
                    canonical_auxiliary_basis=canonical_auxiliary_basis,
                )
        elif gdf_method == "mdf":

            def _build_pair_lpq(ki: np.ndarray, kj: np.ndarray):
                # Mixed Density Fitting: ket-resolved combined cderi
                # [L_gauss; cderi_pw]; the multi-k J/K take the conjugate.
                return build_lpq_bloch_mdf(
                    system,
                    basis,
                    aux,
                    ki,
                    kj,
                    molecule=mol,
                    lat_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    compcell_eta=float(compcell_eta),
                    mdf_ke_cutoff=float(mdf_ke_cutoff),
                    rcut_strategy=rcut_strategy,
                    rcut_precision=float(rcut_precision),
                )
        else:

            def _build_pair_lpq(ki: np.ndarray, kj: np.ndarray):
                return build_lpq_bloch_compcell(
                    system,
                    basis,
                    aux,
                    kj - ki,  # momentum transfer q = k_j - k_i
                    molecule=mol,
                    lat_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    compcell_eta=float(compcell_eta),
                    apply_aft_correction=bool(apply_aft_correction),
                    aft_ft_convention=str(aft_ft_convention),
                    aft_precision=float(aft_precision),
                    rcut_strategy=rcut_strategy,
                    rcut_precision=float(rcut_precision),
                )

        # Fail early rather than OOM-killing a doomed run: gate the dense
        # per-pair Lpq cache peak against available RAM (prompt-75 pattern).
        _preflight_gdf_lpq_memory(
            plog,
            n_basis=basis.nbasis,
            n_aux=aux.nbasis,
            n_kpoints=n_k,
            need_k_pairs=need_k_pairs,
            open_shell=False,
            route_label=label,
            options=opts,
            n_ibz_kpoints=(
                len(_ibz_native_state[0])
                if (_ibz_native_state is not None and need_k_pairs)
                else None
            ),
        )
        with plog.stage(
            "gdf_cderi",
            detail=f"per-pair Lpq for {n_k} k-points ({gdf_method})",
        ):
            if not need_k_pairs:
                n_pairs = n_k
            elif _ibz_native_state is not None:
                n_pairs = len(_ibz_native_state[0]) * n_k + n_k - len(_ibz_native_state[0])
            else:
                n_pairs = n_k * n_k
            plog.info(
                f"Building per-pair Lpq cache ({n_pairs} pairs, "
                f"{gdf_method})..."
            )
            if _lpq_cache_builder is not None and not bulk_sr:
                if gdf_method != "rsgdf":
                    raise ValueError(
                        "_lpq_cache_builder requires gdf_method='rsgdf'"
                    )
                lpq_cache = _lpq_cache_builder(
                    _build_pair_lpq,
                    kpoints_cart,
                    basis,
                    aux_modrho,
                    need_k_pairs,
                )
            elif gdf_method == "rsgdf":
                # Shared-q batch: one ket-Bloch pair-FT pass per unique
                # momentum transfer, folded across the k-points that
                # share it (n_k passes for hybrid exchange instead of
                # n_k^2; one pass for diagonal-only builds).
                if bulk_sr:
                    lpq_cache = _build_scf_range_separated_lpq_cache(
                        system, basis, aux, kpoints_cart, need_k_pairs,
                        omega=rsgdf_omega, raw_integral_error=rsgdf_g_precision,
                        ke_cutoff=rsgdf_ke_cutoff, linear_dep_thr=gdf_linear_dep_threshold,
                        lat_opts=lat_opts, fit_screen_threshold=fit_screen_threshold,
                        options=opts, open_shell=False, progress=plog,
                        oneel_lattices=(S_lat, T_lat, V_lat, V_ecp_lat),
                        xc_grid=grid, xc_cells=xc_cells, functional=func,
                        xc_density_domain=xc_density_domain,
                        compute_gradient=compute_gradient,
                        density_return_bytes=getattr(
                            getattr(_finish_density_return, '__self__', None),
                            'reserved_peak_bytes', 0,
                        ),
                        bra_rows=(_ibz_native_state[0] if (_ibz_native_state is not None and need_k_pairs) else None),
                        pair_cache_builder=_lpq_cache_builder,
                    )
                else:
                    lpq_cache = _build_rsgdf_lpq_cache_shared_q(
                        system,
                        basis,
                        aux_modrho,
                        kpoints_cart,
                        need_k_pairs,
                        ke_cutoff=float(rsgdf_ke_cutoff),
                        tail_ke_cutoff=(
                            float(rsgdf_tail_ke_cutoff)
                            if rsgdf_tail_ke_cutoff is not None
                            else None
                        ),
                        lat_opts=lat_opts,
                        linear_dep_thr=float(gdf_linear_dep_threshold),
                        fit_screen_threshold=float(fit_screen_threshold),
                        progress=plog,
                        q_metric_cache=q_metric_cache,
                        bra_rows=(
                            _ibz_native_state[0]
                            if (_ibz_native_state is not None and need_k_pairs)
                            else None
                        ),
                    )
            else:
                for i in range(n_k):
                    ki = kpoints_cart[i]
                    if need_k_pairs:
                        # Hybrid: build all (k_i, k_j) pairs for exchange.
                        for j in range(n_k):
                            lpq_cache[(i, j)] = _build_pair_lpq(
                                ki, kpoints_cart[j]
                            )
                    else:
                        # Pure DFT / J-only, or COSX exchange backend:
                        # only diagonal pairs needed.
                        lpq_cache[(i, i)] = _build_pair_lpq(ki, ki)
            n_fit = lpq_cache[(0, 0)].shape[0] if lpq_cache else 0
            plog.info(
                f"Lpq cache built: {len(lpq_cache)} pairs, "
                f"{n_fit} fit vectors, "
                f"shape=({n_fit}, {basis.nbasis}, {basis.nbasis})"
            )
            if q_metric_cache is not None:
                q_states = (lpq_cache.q_batch_sizes if private_mdf else
                            getattr(lpq_cache, 'reciprocal_vector_counts', q_metric_cache))
                plog.info(
                    f"{'Private MDF' if private_mdf else 'RSGDF'} q-state shared across pairs: "
                    f"{len(q_states)} unique momentum transfers"
                )

    # ---- Multi-k COSX exchange bridge (k_exchange='cosx') ----------
    # SCF-invariant setup: truncated cell list, per-relative-shift
    # analytic-integral caches, periodic Becke grid, Q-junction. Per
    # iteration the bridge folds D(k) -> D(g), runs one real-space
    # K(g) build (mesh-size independent), and Bloch-folds to K(k).
    cosx_bridge = None
    if k_exchange == "cosx" and use_compcell and alpha != 0.0:
        from .periodic_cosx_k import KPointCosxK

        # Range-separation parameter for the SR/LR exchange split
        # (M3b-4): the real-space erfc-SR part must be dead at both
        # the cell-list cutoff and the BvK half-super-period (alias
        # boundary); the smooth LR-erf complement is built in
        # reciprocal space and restores the full kernel exactly, so
        # w only tunes the split -- erfc(5) ≈ 1.5e-12 sets the reach.
        lat_np = np.asarray(system.lattice, dtype=float)
        mesh_dims = _mesh_tuple_for_system(system, kmesh)
        half_supers = [
            0.5 * mesh_dims[i] * float(np.linalg.norm(lat_np[:, i]))
            for i in range(int(system.dim))
        ]
        sr_reach = min(min(half_supers), float(lat_opts.cutoff_bohr))
        cosx_omega = 5.0 / sr_reach
        # Screened (HSE-type) exchange composition policy. With
        # w_s >= the alias-safe split, erfc(w_s r)/r is evaluated
        # entirely in real space (band == 0, the CRYSTAL-style direct
        # SR assembly). With HSE-class small w_s, the SR part stays at
        # the alias-safe split and the cusp-free band kernel completes
        # the physical kernel in reciprocal space -- measured
        # equivalent to the pure real-space evaluation at 3e-6 Ha on
        # the dimerized-chain (1,1,2) hse06 anchor, and both agree
        # with the BIPOLE screened backend at 0.20 mHa. The G = 0
        # convention includes the erfc kernel's FINITE pi/w_s^2 zero
        # mode (VASP/CRYSTAL convention, shared with BIPOLE); PySCF's
        # exxdiv=None drops it -- add the analytic
        # (pi/w_s^2/(V.N_k)).S D S term to PySCF totals before
        # comparing (verified to 7.3e-5 Ha on the anchor).
        if screened_omega is not None and screened_omega >= cosx_omega:
            cosx_omega = float(screened_omega)

        with plog.stage("cosx_caches"):
            cosx_bridge = KPointCosxK(
                basis, system, lat_opts=lat_opts, omega=cosx_omega
            )
        if screened_omega is not None:
            plog.info(
                "K backend: multi-k COSX, HSE-type SCREENED exchange "
                f"(physical erfc kernel w_s = {screened_omega:.4f} "
                f"bohr^-1, c_sr = {alpha:.4f}; numerical SR split "
                f"w_p = {cosx_omega:.3f} bohr^-1"
                + (
                    ", pure real-space"
                    if abs(cosx_omega - screened_omega) < 1e-12
                    else " + reciprocal band complement"
                )
                + "; no exxdiv shift -- the erfc kernel has no G->0 "
                "divergence; "
                f"cells={len(cosx_bridge.cells)}, "
                f"deltas={cosx_bridge.caches.n_deltas})"
            )
        else:
            plog.info(
                "K backend: multi-k COSX, range-separated "
                f"(SR: real-space erfc, w = {cosx_omega:.3f} bohr⁻¹; "
                "LR: reciprocal-space erf complement; "
                f"cells={len(cosx_bridge.cells)}, "
                f"deltas={cosx_bridge.caches.n_deltas})"
            )
        if gdf_method == "rsgdf":
            plog.info(
                "  K/J pairing: single-gauge (RSGDF J + COSX K share "
                "the Bloch pair-FT conventions). Validated at "
                "sub-mHa backend parity (0.024-0.027 mHa vs "
                "k_exchange='gdf' on the chain anchor across "
                "(1,1,2)-(1,1,6)) with clean SCF convergence through "
                "(1,1,8) -- M3b-4c."
            )
            plog.info(
                "  Scope criterion: the SR exchange range (≈5/w) "
                "PLUS the basis pair extent must fit inside the BvK "
                "half-super-period. Diffuse-basis tight cells at "
                "small meshes violate it (measured: LiH/sto-3g "
                "(2,2,2): w-invariance broken at 2e-2, SCF stall, "
                "534 mHa off -- Li 2sp extent ~15 bohr vs 5.46-bohr "
                "half-super-period); the BvK-consistent SR cell "
                "summation is M3b-6 (handovers/HANDOVER_RIJCOSX_M3A.md "
                "Sec. M3b-5)."
            )
        else:
            plog.info(
                "  WARNING: k_exchange='cosx' with "
                "gdf_method='compcell' is a MIXED-GAUGE Fock: the "
                "COSX K is matrix-level exact (~1e-4 vs the "
                "independent RSGDF exchange), but on vacuum-padded "
                "systems the compcell-J tensors carry a flagged "
                "zone-edge deviation (handovers/HANDOVER_RIJCOSX_M3A.md "
                "Sec. M3b-4b, escalated to the GDF route), and the "
                "inconsistency degrades dense-k-mesh SCF convergence "
                "(100-iter stalls at (1,1,6)+ on the chain anchor). "
                "Pair with gdf_method='rsgdf' for the validated "
                "single-gauge combination."
            )

    # ---- Initial guess: shared Fock/density artifacts per k -------
    guess_fock_k = Hcore_k
    if initial_density_k is None and guess in (
        InitialGuess.SAP,
        InitialGuess.HUECKEL,
    ):
        fock_guess = periodic_fock_guess_k(
            system,
            basis,
            kpoints_cart,
            guess,
            lattice_opts=oneel_lat_opts,
            kinetic_lattice=T_lat,
            overlap_lattice=S_lat,
            ecp_context=_guess_ecp,
        )
        assert fock_guess is not None
        guess_fock_k = fock_guess
    C_k: List[np.ndarray] = []
    eps_k: List[np.ndarray] = []
    for i in range(n_k):
        Ci, ei = _diag_in_orth_basis(guess_fock_k[i], X_k[i])
        C_k.append(Ci.astype(complex))
        eps_k.append(ei)
    occ_k, fermi_level, entropy = _occupations_per_k(
        eps_k,
        weights,
        n_elec,
        smearing_T,
        n_occ,
        bz_integration=bz_integration,
        system=system,
        kmesh=kmesh_bloch,
    )
    D_k = _density_from_orbitals(C_k, occ_k)
    if initial_density_k is not None:
        from .guess import normalize_density_k_guess
        D_k = normalize_density_k_guess(initial_density_k, S_k, weights, n_elec)
        plog.info("initial guess: READ (caller-supplied per-k density)")
    elif guess in (InitialGuess.SAD, InitialGuess.MINAO, InitialGuess.PATOM):
        density_guess = initial_density_closed_shell(
            system.unit_cell_molecule(),
            basis,
            n_occ,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=oneel_lat_opts,
            overlap=S_k, weights=weights,
            ecp_context=_guess_ecp,
        )
        assert density_guess is not None
        density_guess = np.asarray(density_guess, dtype=complex)
        density_guess = 0.5 * (
            density_guess + density_guess.conj().T
        )
        D_k = [density_guess.copy() for _ in range(n_k)]
        plog.info(
            f"initial guess: {guess.name} "
            "(shared g=0 density injected at every k-point)"
        )
    elif guess in (InitialGuess.SAP, InitialGuess.HUECKEL):
        plog.info(
            f"initial guess: {guess.name} "
            "(shared per-k Fock diagonalisation)"
        )
    else:
        plog.info("initial guess: HCORE (per-k Hcore diagonalisation)")
    if smearing_T > 0.0:
        plog.info(
            "smearing: Fermi-Dirac kBT = "
            f"{smearing_T:.6g} Ha "
            f"({_hartree_to_kelvin_temperature(smearing_T):.1f} K)"
        )

    # ---- SCF setup -----------------------------------------------
    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_krhf_periodic_gdf: damping must be in [0, 1); got {damping}"
        )
    if fock_mixing_value != 0.0:
        plog.info(
            "fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock/KS matrix weight, applied per k)"
        )
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[MultiKPeriodicSCFAccelerator] = (
        MultiKPeriodicSCFAccelerator(opts) if use_diis else None
    )

    # ---- Density-space mixer (Anderson / Broyden [+ Kerker]) ----------
    # Ported from the multi-k EWALD_3D RKS driver (same machinery in
    # periodic_density_mixing): mix the per-k density MATRICES, with the
    # Kerker filter optionally preconditioning the residual on a
    # plane-wave grid. When selected it REPLACES Fock-DIIS, linear
    # damping, dynamic damping, and fock mixing (one accelerator owns
    # the update; see the Ewald driver + CLAUDE.md Sec.7).
    from .periodic_density_mixing import (
        AndersonMixer as _AndersonMixer,
        BroydenMixer as _BroydenMixer,
        KerkerPreconditioner as _KerkerPreconditioner,
        per_k_density_to_vector as _per_k_density_to_vector,
        vector_to_per_k_density as _vector_to_per_k_density,
    )

    _mixer_key = (
        None if density_mixer is None else str(density_mixer).strip().lower()
    )
    if _mixer_key in (None, "", "none", "diis"):
        density_space_mixer = None
    elif _mixer_key == "anderson":
        density_space_mixer = _AndersonMixer(
            depth=int(density_mixer_depth), beta=float(density_mixer_beta)
        )
    elif _mixer_key == "broyden":
        density_space_mixer = _BroydenMixer(
            depth=int(density_mixer_depth), beta=float(density_mixer_beta)
        )
    else:
        raise ValueError(
            f"run_krhf_periodic_gdf: density_mixer={density_mixer!r} is not "
            f"recognised; expected one of None, 'diis', 'anderson', 'broyden'."
        )
    kerker_precond: Optional[_KerkerPreconditioner] = None
    if density_space_mixer is not None:
        if density_mixer_kerker:
            from .periodic_rhf_multi_k_ewald import _g0_block

            kerker_precond = _KerkerPreconditioner(
                basis,
                system,
                _g0_block(S_lat),
                k0=float(kerker_k0),
                strength=float(kerker_strength),
                cutoff_ha=float(kerker_cutoff_ha),
            )
        use_diis = False
        accel = None
        damper = None
        damping = 0.0
        fock_mixing_value = 0.0
        plog.info(
            f"density mixer: {density_mixer!r} "
            f"(depth={int(density_mixer_depth)}, "
            f"beta={float(density_mixer_beta)}"
            + (
                f"; Kerker k0={float(kerker_k0)}, "
                f"strength={float(kerker_strength)}, "
                f"cutoff={float(kerker_cutoff_ha)} Ha"
                if kerker_precond is not None
                else ""
            )
            + ") -- Fock-DIIS, damping and fock mixing disabled"
        )
    elif density_mixer_kerker:
        raise ValueError(
            "run_krhf_periodic_gdf: density_mixer_kerker=True requires "
            "density_mixer='anderson' or 'broyden' -- Kerker preconditions "
            "the density mixer's residual, it is not a standalone "
            "accelerator."
        )
    if level_shift != 0.0:
        if warmup_cycles > 0:
            cycle_word = "cycle" if warmup_cycles == 1 else "cycles"
            plog.info(
                f"level-shift warm-up: {warmup_cycles} {cycle_word} at "
                f"{level_shift:.3f} Ha (per k); restart unshifted afterwards"
            )
        else:
            plog.info(
                f"level shift: {level_shift:.3f} Ha (per k) "
                "applied at each diagonalization"
            )

    plog.banner(f"SCF ({label} multi-k, native GDF)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    _lindep = PeriodicLinearDependenceSummary(
        n_basis=int(basis.nbasis),
        n_kept_per_k=list(n_kept_k),
        threshold=float(linear_dep_threshold),
        min_overlap_eigenvalue=_s_lo,
        max_overlap_eigenvalue=_s_hi,
        n_aux=int(aux.nbasis),
        n_fit_kept=int(n_fit),
        aux_threshold=float(gdf_linear_dep_threshold),
    )
    result = PeriodicKRHFGDFResult(
                 restart_kpoints=np.asarray(kmesh_bloch.kpoints).copy(), restart_weights=np.asarray(kmesh_bloch.weights).copy(),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        n_iter=0,
        converged=False,
        mo_energies=[e.copy() for e in eps_k],
        mo_coeffs=[C.copy() for C in C_k],
        fock=[np.empty((0, 0), dtype=complex) for _ in range(n_k)],
        overlap=[S.copy() for S in S_k],
        hcore=[H.copy() for H in Hcore_k],
        density=[D.copy() for D in D_k],
        kpoints_cart=kpoints_cart.copy(),
        kpoint_weights=weights.copy(),
        scf_trace=scf_trace,
        functional=func_name or None,
        fock_mixing=fock_mixing_value,
        level_shift=level_shift,
        level_shift_warmup_cycles=warmup_cycles,
        smearing_temperature=smearing_T,
        fermi_level=float(fermi_level),
        entropy=float(entropy),
        occupations=[np.asarray(o, dtype=float) for o in occ_k],
        aux_basis_name=aux_name,
        linear_dependence=_lindep,
        n_aux=int(aux.nbasis),
        rsgdf_ke_cutoff=float(getattr(lpq_cache, 'lr_ke_cutoff', rsgdf_ke_cutoff)),
        backend=(
            f"native-multi-k-gdf-{k_exchange}-"
            f"{'rks' if is_ks else 'rhf'}"
        ),
    )
    if private_mdf:
        result.backend += '+private-mdf-unvalidated'
    if dense_core_parity_held:
        result.backend = result.backend + "+PARITY_HELD"

    F_prev_k: Optional[List[np.ndarray]] = None
    D_prev_k: List[np.ndarray] = [D.copy() for D in D_k]
    E_prev = 0.0

    # ---- GDF J/K builders (cached Lpq contraction) -----------------
    def _build_j_from_lpq(D_k_in: List[np.ndarray]) -> List[np.ndarray]:
        from .aux_basis import _build_coulomb_from_diagonal_factors

        return _build_coulomb_from_diagonal_factors(
            [lpq_cache[(i, i)] for i in range(n_k)], D_k_in, weights,
        )

    def _build_k_from_lpq(D_k_in: List[np.ndarray]) -> List[np.ndarray]:
        """Build K matrices from cached Lpq.

        Full-BZ by default: all ``(k_i, k_j)`` pairs. Under
        ``ibz_native`` the exchange is built at the wedge bras only and
        the remaining ``K(k)`` are obtained by symmetry transport --
        which is why the ket sum, the density list and the returned
        shape are all still full-mesh.
        """
        if _ibz_native_state is None:
            return _build_k_from_lpq_cache(
                lpq_cache,
                D_k_in,
                weights,
                nbasis=basis.nbasis,
            )
        rows, star_map, kmesh_full_native = _ibz_native_state
            # Checked EVERY iteration, not once: the initial guess is
            # symmetric even when the converged state is not, so a
            # one-shot check on the guess passes and the break appears
            # later. Measured on the LiH FCC triplet -- guess symmetric,
            # converged density 1.7e-3 asymmetric. The cost is one
            # transport of n_IBZ matrices, negligible against the K build.
        if True:
            _require_ibz_symmetric_state(
                [np.asarray(D) for D in D_k_in],
                S_k,
                rows,
                star_map,
                system,
                basis,
                kmesh_full_native,
                label="density at this SCF iteration",
            )
        K_rows = _build_k_ibz_native(
            lpq_cache,
            D_k_in,
            weights,
            rows,
            nbasis=basis.nbasis,
        )
        from .periodic_k_symmetry import expand_k_matrices_to_full

        return expand_k_matrices_to_full(
            K_rows, star_map, system, basis, kmesh_full_native
        )

    # Iteration-invariant EWALD_3D Hartree-J cache: the per-cell AO-pair
    # FT dominates every build_periodic_fock_ewald3d_k call (~93 % of a
    # multi-k GDF SCF iteration profiled on LiH (1,1,2): 557 s of 601 s
    # total were pair-FT recomputation), and it depends only on
    # (basis, cells, mesh) -- never on the density. Same mechanism the
    # multi-k EWALD_3D RHF/RKS drivers already hoist
    # (make_ewald_3d_lattice_j_cache; bit-identical contraction).
    #
    # Built ONLY when the EWALD_3D J/K branch below will actually run:
    # the cached-Lpq GDF branch (use_compcell=True, every HF/hybrid run
    # after the auto-routing above) never touches it, and the cache is
    # the dominant memory of a small-cell multi-k GDF run -- the per-cell
    # AO-pair FT is (n_cells, nbf, nbf, n_G) on its own VIBEQC_J_EWALD3D_KE
    # mesh (c-diamond primitive sto-3g, 177 cells x 10^2 AO-pairs x 10417
    # G-points = 2.95 GB, ~5.9 GB transient with the scale copy; measured
    # 6.13 GB peak RSS -> 2026-07-09, HANDOVER_GDF_FIT_SCREENING.md). It
    # is also ke-independent of rsgdf_ke_cutoff, which is why it was
    # misattributed to the GDF fit tensors in the original finding.
    #
    # Pure DFT is the remaining live 3D branch here. Small AO-pair FT caches
    # retain their fast reuse path, but only when the full tensor plus one
    # construction temporary fits the dense-cache target. Larger cases
    # contract J from D(k) under the smaller reciprocal/cell batch target:
    # retaining the old cache would require 38.9 GiB for Si/def2-SVP (P15,
    # 2x2x2 / 4x4x4) and 296.8 GiB for conventional NaCl/def2-SVP (P10,
    # 4x4x4). The diagnostic grid backend keeps the legacy density route.
    _ewald_j_backend = os.environ.get(
        "VIBEQC_J_EWALD3D_BACKEND", "analytic_ft"
    ).lower()
    _pure_dft_analytic_ewald_j = (
        is_ks
        and alpha == 0.0
        and int(system.dim) == 3
        and _ewald_j_backend == "analytic_ft"
        and not (use_compcell and lpq_cache)
    )
    _ewald_j_cache_fits_target = (
        _ewald_3d_lattice_j_cache_fits_memory_target(
            basis,
            system,
            cells,
        )
        if _pure_dft_analytic_ewald_j
        else False
    )
    _stream_ewald_j_from_k = (
        _pure_dft_analytic_ewald_j and not _ewald_j_cache_fits_target
    )
    _ewald_j_cache = (
        make_ewald_3d_lattice_j_cache(
            basis, system, cells, lattice_opts=lat_opts
        )
        if not _stream_ewald_j_from_k and not (use_compcell and lpq_cache)
        else None
    )

    if guess == InitialGuess.PATOM:
        if use_compcell:
            J_seed = _build_j_from_lpq(D_k)
            madelung = _madelung_for_kmesh(system, _mesh_tuple_for_system(system, kmesh))
            K_seed = list(apply_exxdiv_ewald_to_K(
                _build_k_from_lpq(D_k), S_k, D_k, madelung,
            ))
            F_seed = [h + j - 0.5 * k for h, j, k in zip(Hcore_k, J_seed, K_seed)]
        else:
            from .periodic_corrected_exchange import CorrectedEwaldExchange
            from . import _vibeqc_core as patom_core
            from types import SimpleNamespace as SeedMesh
            seed_mesh = SeedMesh(
                kpoints=list(kmesh_bloch.kpoints), weights=list(kmesh_bloch.weights),
                mesh=_mesh_tuple_for_system(system, kmesh), ir_mapping=[],
            )
            seed_exchange = CorrectedEwaldExchange.build(
                basis, system, np.asarray([c.r_cart for c in cells]),
                seed_mesh, 0.5, where="run_krhf_periodic_gdf PATOM",
            )
            seed_real = _real_space_density_from_per_k_density(D_k, kmesh_bloch, cells)
            corrections = seed_exchange.k_space_terms_all_k(S_k, D_k)
            if _stream_ewald_j_from_k:
                J_seed = build_periodic_j_ewald3d_k_from_k_density(
                    basis, system, D_k, list(kpoints_cart), weights, cells, omega=0.5,
                )
                K_sr = patom_core.build_jk_2e_real_space(basis, system, lat_opts, seed_real, 0.5).K
                F_seed = [h + j - 0.5 * (np.asarray(patom_core.bloch_sum(K_sr, k)) + correction)
                          for h, j, k, correction in
                          zip(Hcore_k, J_seed, kpoints_cart, corrections)]
            else:
                F_seed = build_periodic_fock_ewald3d_k(
                    basis, system, seed_real, omega=0.5,
                    k_points_cart=list(kpoints_cart), Hcore_k=Hcore_k,
                    lattice_opts=lat_opts, exchange_scale=1.0,
                    full_range_alpha=0.5, j_cache=_ewald_j_cache,
                )
                F_seed = [f - 0.5 * c for f, c in zip(F_seed, corrections)]
        C_k, eps_k = [], []
        for fock, x in zip(F_seed, X_k):
            c, e = _diag_in_orth_basis(fock, x)
            C_k.append(c.astype(complex))
            eps_k.append(e)
        occ_k, fermi_level, entropy = _occupations_per_k(
            eps_k, weights, n_elec, smearing_T, n_occ,
            bz_integration=bz_integration, system=system, kmesh=kmesh_bloch,
        )
        D_k = _density_from_orbitals(C_k, occ_k)
        D_prev_k = [d.copy() for d in D_k]
        result.mo_coeffs = [c.copy() for c in C_k]
        result.mo_energies = [e.copy() for e in eps_k]
        result.density = [d.copy() for d in D_k]
        result.occupations = [np.asarray(o).copy() for o in occ_k]
        result.fermi_level = float(fermi_level)
        result.entropy = float(entropy)

    from .periodic_k_density import _fermi_density_entropy, _needs_fermi_seed_step
    seed_step = max_iter > 0 and smearing_T > 0.0 and _needs_fermi_seed_step(
        D_k, S_k, X_k, weights, 2.,
    )
    # SAD/MINAO/READ seeds need not be fermionically representable. They
    # still select the first physical Fock; refill that Fock before evaluating
    # Mermin's functional (1965, Eqs. 1-4), rather than clipping a seed entropy.
    for it in range(0 if seed_step else 1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        if warmup_cycles > 0 and it == warmup_cycles + 1:
            if accel is not None:
                accel = MultiKPeriodicSCFAccelerator(opts)
            F_prev_k = None
            plog.info("restart: unshifted Fock with fresh DIIS history (per k)")

        if _ls_schedule:
            active_level_shift = level_shift_at_iter(
                level_shift, warmup_cycles, _ls_schedule, max_iter, it
            )
        else:
            active_level_shift = (
                level_shift
                if (level_shift != 0.0 and (warmup_cycles == 0 or it <= warmup_cycles))
                else 0.0
            )
        diis_active = use_diis and it >= diis_start_iter

        # Density damping (per k, in AO basis).
        if it <= 1 or damping == 0.0 or diis_active:
            D_used = [D.copy() for D in D_k]
        else:
            D_used = [
                damping * Dp + (1.0 - damping) * Dn for Dp, Dn in zip(D_prev_k, D_k)
            ]

        # ---- J + K build -------------------------------------------
        if use_compcell and lpq_cache:
            # True GDF: contract cached Lpq per iteration (J always;
            # K per the selected exchange backend).
            J_k = _build_j_from_lpq(D_used)
            if alpha != 0.0:
                if cosx_bridge is not None:
                    K_k = cosx_bridge.k_matrices(
                        D_used, list(kpoints_cart),
                        lr_complement=True,
                        weights=list(weights),
                        screened_omega=screened_omega,
                    )
                else:
                    K_k = _build_k_from_lpq(D_used)
            else:
                K_k = [np.zeros_like(J_k[0]) for _ in range(n_k)]
            # Apply exxdiv='ewald' Madelung correction to K.
            # k-mesh-aware (Born-von-Kármán supercell) Madelung -- NOT the
            # primitive-cell ξ, which over-counts the exxdiv K-shift by
            # Nk^(1/3) and over-binds the multi-k energy (LiH (2,2,2):
            # -592 mHa). See _madelung_for_kmesh.
            #
            # The shift's energy contribution enters ONCE, through the
            # shifted K inside ``E_elec = Tr[D.Hcore] + 1/2.Tr[D.F_2e]``
            # below -- do NOT also add ``exxdiv_ewald_energy_shift`` to
            # E_total (that double-counts the Madelung correction; on
            # H₂/12-bohr (2,1,1) the double-count over-binds by
            # ~150 mHa vs the published -1.12013988 Ha).
            if alpha != 0.0 and screened_omega is None:
                # BvK mesh from the CALLER's kmesh argument, not
                # ``kmesh_bloch.mesh``: the explicit-KPoints conversion
                # (``to_bloch_kmesh``) reports the C++ default (1,1,1),
                # which silently swapped the BvK-supercell Madelung for
                # the primitive-cell one (measured 78 mHa on the H2
                # dimerized chain, uniform explicit [Gamma, X] vs the
                # identical tuple (1,1,2) mesh). Explicit KPoints must
                # declare ``mesh`` metadata; without it this raises
                # (fail-closed) rather than converging a wrong exxdiv.
                # Screened (HSE-type) exchange skips the shift entirely:
                # the erfc kernel has no G -> 0 divergence.
                madelung = _madelung_for_kmesh(
                    system, _mesh_tuple_for_system(system, kmesh)
                )
                K_k = list(apply_exxdiv_ewald_to_K(K_k, S_k, D_used, madelung))
            F_k = [Jk - 0.5 * alpha * Kk for Jk, Kk in zip(J_k, K_k)]
            F_2e_k = [np.asarray(f).copy() for f in F_k]

            # XC on the full real-space finite-torus density, Bloch-folded
            # to each k. A single Γ AO matrix is only a vacuum-limit shortcut.
            V_xc_k = None
            E_xc = 0.0
            if is_ks:
                E_xc, V_xc_k = _build_xc_k_from_density(
                    basis=basis,
                    system=system,
                    grid=grid,
                    func=func,
                    density_k=D_used,
                    kmesh_bloch=kmesh_bloch,
                    cells=xc_cells,
                    kpoints_cart=kpoints_cart,
                    lat_opts=lat_opts,
                    density_domain=xc_density_domain,
                )
            E_coulomb = 0.5 * sum(
                float(weights[i]) * float(np.real(np.trace(D_used[i] @ J_k[i])))
                for i in range(n_k)
            )
            E_hf_K = (
                -0.25
                * alpha
                * sum(
                    float(weights[i]) * float(np.real(np.trace(D_used[i] @ K_k[i])))
                    for i in range(n_k)
                )
                if alpha != 0.0
                else 0.0
            )
            plog.info(
                f"J backend: native multi-k GDF (cached Lpq, "
                f"{len(lpq_cache)} pairs); K backend: "
                + ("multi-k COSX (real-space K(g) + Bloch fold)"
                   if cosx_bridge is not None
                   else "multi-k GDF (k-pair Lpq contraction)")
            )
            # Fold Hcore (+ V_xc for RKS) into the Fock that gets
            # extrapolated + diagonalised. The M3b-4b/4c refactor moved
            # XC into the branches and left this fold in the EWALD_3D
            # branch only -- every compcell SCF then diagonalised the
            # bare 2e Fock (J - 1/2aK), walking to a spurious fixed point
            # (LiH (2,2,2): -2.96 Ha instead of -7.92, converged=True).
            for i in range(n_k):
                Fi = F_k[i] + Hcore_k[i]
                if V_xc_k is not None:
                    Fi = Fi + V_xc_k[i]
                Fi = 0.5 * (Fi + Fi.conj().T)
                F_k[i] = Fi
        else:
            # ---- J + K via EWALD_3D gauge -----------------------------
            if _stream_ewald_j_from_k:
                # Pure DFT: contract the exact density used this iteration
                # directly from D(k). This preserves fractional occupations
                # under smearing/damping without materialising D(g) or the
                # persistent all-cell/all-G AO-pair FT cache.
                F_k = build_periodic_j_ewald3d_k_from_k_density(
                    basis,
                    system,
                    D_used,
                    [np.asarray(k) for k in kpoints_cart],
                    weights,
                    cells,
                    omega=0.5,
                )
            else:
                # Exchange or the diagnostic grid backend still needs the
                # real-space density. Rebuilding it from C(k) and hard n_occ
                # would silently replace a smeared/damped density with an
                # integer-Aufbau one.
                D_real = _real_space_density_from_per_k_density(
                    D_used,
                    kmesh_bloch,
                    cells,
                )
                F_k = build_periodic_fock_ewald3d_k(
                    basis,
                    system,
                    D_real,
                    omega=0.5,
                    k_points_cart=[np.asarray(k) for k in kpoints_cart],
                    Hcore_k=None,
                    lattice_opts=lat_opts,
                    exchange_scale=alpha,
                    j_cache=_ewald_j_cache,
                )

            # XC on the full real-space finite-torus density, Bloch-folded
            # to each k.
            V_xc_k = None
            E_xc = 0.0
            if is_ks:
                E_xc, V_xc_k = _build_xc_k_from_density(
                    basis=basis,
                    system=system,
                    grid=grid,
                    func=func,
                    density_k=D_used,
                    kmesh_bloch=kmesh_bloch,
                    cells=xc_cells,
                    kpoints_cart=kpoints_cart,
                    lat_opts=lat_opts,
                    density_domain=xc_density_domain,
                )

            # Save F_2e before adding V_xc (for energy decomposition).
            F_2e_k = [np.asarray(f).copy() for f in F_k]
            for i in range(n_k):
                Fi = F_k[i] + Hcore_k[i]
                if V_xc_k is not None:
                    Fi = Fi + V_xc_k[i]
                Fi = 0.5 * (Fi + Fi.conj().T)
                F_k[i] = Fi

            # ---- Energy decomposition (J-only Fock for E_J, E_K). ------
            E_coulomb = 0.0
            E_hf_K = 0.0
            if alpha != 0.0:
                F_J_k = build_periodic_fock_ewald3d_k(
                    basis,
                    system,
                    D_real,
                    omega=0.5,
                    k_points_cart=[np.asarray(k) for k in kpoints_cart],
                    Hcore_k=None,
                    lattice_opts=lat_opts,
                    exchange_scale=0.0,
                    j_cache=_ewald_j_cache,
                )
                E_J_val = 0.0
                E_K_val = 0.0
                for i in range(n_k):
                    w = float(weights[i])
                    J_k_i = np.asarray(F_J_k[i])
                    K_k_i = 2.0 * (J_k_i - F_2e_k[i]) / alpha
                    K_k_i = 0.5 * (K_k_i + K_k_i.conj().T)
                    E_J_val += w * float(np.real(np.trace(D_used[i] @ J_k_i)))
                    E_K_val += w * float(np.real(np.trace(D_used[i] @ K_k_i)))
                E_coulomb = 0.5 * E_J_val
                E_hf_K = -0.25 * alpha * E_K_val
            else:
                # For a pure functional F_2e is the Hartree J itself.  Keep
                # the public component breakdown complete instead of leaving
                # e_coulomb at its initialization value while total energy
                # correctly includes 1/2 Tr[D J].
                E_coulomb = 0.5 * sum(
                    float(weights[i])
                    * float(
                        np.real(
                            np.trace(
                                np.asarray(D_used[i])
                                @ np.asarray(F_2e_k[i])
                            )
                        )
                    )
                    for i in range(n_k)
                )

        # ---- DFT+U closed-shell Fock contribution ------------------
        # D_used is the total closed-shell density that built this
        # iteration's Coulomb/XC Fock. The Dudarev kernel is per spin, so
        # pass P_sigma(k) = 1/2 P_total(k), then add the same k-independent
        # V_AO_s to alpha and beta through a single closed-shell Fock shift.
        e_dft_plus_u = 0.0
        if dftu_sites_cxx:
            from ._vibeqc_core import (
                _compute_dft_plus_u_multi_k_per_spin_cxx,
            )

            P_sigma_k = [
                0.5 * np.asarray(D, dtype=np.complex128) for D in D_used
            ]
            E_sigma, V_AO = _compute_dft_plus_u_multi_k_per_spin_cxx(
                dftu_sites_cxx,
                dftu_ao_groups,
                [np.asarray(S, dtype=np.complex128) for S in S_k],
                P_sigma_k,
                list(weights),
            )
            e_dft_plus_u = 2.0 * float(E_sigma)
            V_AO_c = np.asarray(V_AO, dtype=np.complex128)
            for i in range(n_k):
                F_u = S_k[i] @ V_AO_c @ S_k[i]
                F_k[i] = F_k[i] + F_u
                F_k[i] = 0.5 * (F_k[i] + F_k[i].conj().T)

        if it == 0:
            C_k, eps_k = [], []
            for fock, x in zip(F_k, X_k):
                c, e = _diag_in_orth_basis(fock, x)
                C_k.append(c.astype(complex))
                eps_k.append(e)
            occ_k, fermi_level, entropy = _occupations_per_k(
                eps_k, weights, n_elec, smearing_T, n_occ,
                bz_integration=bz_integration, system=system, kmesh=kmesh_bloch,
            )
            D_k = _density_from_orbitals(C_k, occ_k)
            D_prev_k = [d.copy() for d in D_k]
            plog.info("initial density: Fermi refill of the selected seed Fock")
            continue

        # ---- Energy (E_elec = Tr[D.Hcore] + 0.5 Tr[D.F_2e]). -------
        E_elec = 0.0
        for i in range(n_k):
            w = float(weights[i])
            Di = D_used[i]
            Hi = Hcore_k[i]
            Fi = F_2e_k[i]
            E_elec += w * float(
                np.real(np.trace(Di @ Hi)) + 0.5 * np.real(np.trace(Di @ Fi))
            )
        E_elec += E_xc
        E_total = E_elec + float(e_nuc) + e_dft_plus_u
        accepted_entropy = (
            _fermi_density_entropy(D_used, S_k, X_k, weights, 2.)
            if smearing_T > 0.0 else 0.0
        )
        free_energy = E_total - smearing_T * accepted_entropy

        # ---- Convergence -------------------------------------------
        grad_k: List[np.ndarray] = []
        grad_norm_sq = 0.0
        for i in range(n_k):
            FDS = F_k[i] @ D_used[i] @ S_k[i]
            err = FDS - FDS.conj().T
            grad_k.append(err)
            grad_norm_sq += float(weights[i]) * float(np.linalg.norm(err) ** 2)
        grad_norm = float(np.sqrt(grad_norm_sq))
        dE = free_energy - E_prev

        scf_trace.append(
            SCFIteration(
                iter=it,
                energy=float(free_energy),
                delta_e=float(dE if it > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            it,
            energy=float(free_energy),
            dE=float(dE if it > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )

        converged = (
            it > 1
            and (warmup_cycles == 0 or it > warmup_cycles)
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        if converged and not use_gilat:
            # Global-BZ convergence additionally requires the occupations
            # that built D_used (previous iteration's occ_k) to be the
            # Aufbau/Fermi filling of the current plain Fock's eigenvalues.
            # The energy + commutator tests cannot see a
            # frozen-occupation fixed point on zero-commutator fixtures
            # (H2-in-box class), where DIIS extrapolation over zero
            # error vectors used to freeze the eigenvalues -- see the
            # zero-commutator floor in periodic_scf_accelerators.py and
            # smeared_occupation_selfconsistency_tolerance. The same guard
            # is required for T=0 global Aufbau when states cross between
            # k points: a commutator only tests the occupied subspace at
            # each k, not whether the globally lowest states were selected.
            eps_chk = [
                _diag_in_orth_basis(F_k[i], X_k[i])[1] for i in range(n_k)
            ]
            occ_chk, _, _ = _occupations_per_k(
                eps_chk,
                weights,
                n_elec,
                smearing_T,
                n_occ,
                bz_integration=bz_integration,
                system=system,
                kmesh=kmesh_bloch,
            )
            occ_residual = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(oo, dtype=float)
                            - np.asarray(oc, dtype=float)
                        )
                    )
                )
                for oo, oc in zip(occ_k, occ_chk)
            )
            if occ_residual > _smeared_occ_tol_fn(
                float(opts.conv_tol_energy)
            ):
                converged = False

        # A terminating result owns the physical, unshifted Fock built from
        # the same accepted density as its energy.  Do not run that operator
        # through DIIS/Fock mixing/level shifting and then report it beside a
        # different next-generation density.  Canonical orbitals are obtained
        # from the physical operator; on an unconverged max-iteration return
        # their refill is diagnostic and the accepted mixed density remains
        # authoritative.
        terminating_state = converged or it == max_iter

        # ---- SCF-accelerator extrapolation, FMIXING, level shift ----
        # The full {DIIS, KDIIS, EDIIS, EDIIS_DIIS, ADIIS} family +
        # dynamic_damping is wired here (M4); GDF keeps density per-k
        # natively, so ``density_k_list`` is ``D_used`` with no Bloch
        # sum. See ``MultiKPeriodicSCFAccelerator`` in
        # ``periodic_scf_accelerators.py`` for the per-mode dispatch.
        if not terminating_state and accel is not None:
            if (
                smearing_T <= 0.0
                and getattr(opts, "scf_accelerator", None)
                == SCFAccelerator.KDIIS
                and not _is_per_k_integer_aufbau(occ_k, n_occ)
            ):
                raise NotImplementedError(
                    "run_krhf_periodic_gdf: KDIIS requires a fixed number "
                    "of occupied orbitals at every k point and cannot be "
                    "used after global T=0 Aufbau detects band overlap or "
                    "a fractionally occupied Fermi degeneracy. Use the "
                    "default EDIIS_DIIS/DIIS accelerator, or explicit "
                    "finite-temperature smearing for a metal."
                )
            F_ex_list = accel.extrapolate_rhf(
                F_k,
                error_k_list=grad_k,
                density_k_list=D_used,
                energy=E_total,
                mo_coeffs_k_list=C_k,
                n_occ=n_occ,
                weights=list(weights),
                cells=cells,
                kpoints=list(kpoints_cart),
            )
            if diis_active:
                F_k = F_ex_list
        if not terminating_state and fock_mixing_value != 0.0:
            if F_prev_k is not None:
                F_mixed: List[np.ndarray] = []
                for i in range(n_k):
                    Fmix = (1.0 - fock_mixing_value) * F_k[
                        i
                    ] + fock_mixing_value * F_prev_k[i]
                    F_mixed.append(0.5 * (Fmix + Fmix.conj().T))
                F_k = F_mixed
            F_prev_k = [F.copy() for F in F_k]

        # Per-k Saunders-Hillier level shift (only at diagonalization).
        F_diag = []
        for i in range(n_k):
            if not terminating_state and active_level_shift != 0.0:
                Fi_shifted = (
                    F_k[i]
                    + active_level_shift * S_k[i]
                    - (active_level_shift / 2.0) * (S_k[i] @ D_used[i] @ S_k[i])
                )
                Fi_shifted = 0.5 * (Fi_shifted + Fi_shifted.conj().T)
                F_diag.append(Fi_shifted)
            else:
                F_diag.append(F_k[i])

        # ---- Diagonalise per k + occupations + density -------------
        C_new: List[np.ndarray] = []
        eps_new: List[np.ndarray] = []
        for i in range(n_k):
            Ci, ei = _diag_in_orth_basis(F_diag[i], X_k[i])
            C_new.append(Ci.astype(complex))
            eps_new.append(ei)
        occ_k, fermi_level, entropy = _occupations_per_k(
            eps_new,
            weights,
            n_elec,
            smearing_T,
            n_occ,
            bz_integration=bz_integration,
            system=system,
            kmesh=kmesh_bloch,
        )
        D_new = _density_from_orbitals(C_new, occ_k)
        if terminating_state:
            # Energy, nonlinear XC, +U, and the physical Fock above all use
            # D_used. Preserve that accepted state rather than pairing them
            # with the canonical refill from one further trial generation.
            D_new = [np.asarray(D).copy() for D in D_used]

        # ---- Density-space mixing (Anderson / Broyden [+ Kerker]) ----
        # Same construction as the multi-k EWALD_3D driver: mix the per-k
        # density matrices; the Kerker filter preconditions the residual
        # (D_out' = D_in + K(D_out - D_in)), leaving the fixed point
        # unchanged.
        if not terminating_state and density_space_mixer is not None:
            if kerker_precond is not None:
                _resid = [
                    np.asarray(Dn) - np.asarray(Du)
                    for Dn, Du in zip(D_new, D_used)
                ]
                _resid = kerker_precond.precondition(_resid, weights)
                _D_out_eff = [
                    np.asarray(Du) + r for Du, r in zip(D_used, _resid)
                ]
            else:
                _D_out_eff = D_new
            _x_in = _per_k_density_to_vector(D_used)
            _x_out = _per_k_density_to_vector(_D_out_eff)
            _x_next = density_space_mixer.update(_x_in, _x_out)
            D_new = _vector_to_per_k_density(_x_next, D_new)

        D_prev_k = D_used
        D_k = D_new
        C_k = C_new
        eps_k = eps_new
        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy

        # Update the result placeholder so partial-run callers see
        # the last iter's state. We do this every iter (rather than
        # only on converge) so a max_iter abort still yields useful
        # numbers.
        result.energy = E_total
        result.e_electronic = E_elec
        result.e_xc = E_xc
        result.e_coulomb = E_coulomb
        result.e_hf_exchange = E_hf_K
        result.e_dft_plus_u = e_dft_plus_u
        result.n_iter = it
        result.mo_energies = [e.copy() for e in eps_new]
        result.mo_coeffs = [C.copy() for C in C_new]
        result.fock = [F.copy() for F in F_k]
        result.density = [D.copy() for D in D_k]
        result.fermi_level = float(fermi_level)
        result.entropy = float(accepted_entropy)
        result.free_energy = float(free_energy)
        result.occupations = [np.asarray(o, dtype=float) for o in occ_k]

        if converged:
            result.converged = True
            if cosx_bridge is not None:
                # Post-convergence COSX one-center replacement -- the
                # molecular-RIJCOSX / dedicated-Gamma-builder lifecycle
                # (commit 40dad042 + handovers/HANDOVER_RIJCOSX_PBC.md
                # 2026-07-15): the reported energy stays on the
                # uncorrected iterated K surface; only the returned
                # Fock and orbitals are upgraded. The correction is a
                # K(g=0) block, so it enters every K(k) identically;
                # K carries -0.5*alpha into the closed-shell Fock.
                C1 = cosx_bridge.one_center_correction(
                    D_used, weights=list(weights)
                )
                dF = -0.5 * alpha * C1
                fock_corr: List[np.ndarray] = []
                mo_e_corr: List[np.ndarray] = []
                mo_c_corr: List[np.ndarray] = []
                for i in range(n_k):
                    Fi = np.asarray(result.fock[i]) + dF
                    Fi = 0.5 * (Fi + Fi.conj().T)
                    Ci, ei = _diag_in_orth_basis(Fi, X_k[i])
                    fock_corr.append(Fi)
                    mo_c_corr.append(Ci.astype(complex))
                    mo_e_corr.append(ei)
                result.fock = fock_corr
                result.mo_energies = mo_e_corr
                result.mo_coeffs = mo_c_corr
                plog.info(
                    "COSX one-center correction applied to the "
                    "returned Fock/orbitals (post-convergence; "
                    "energy stays on the iterated surface)"
                )
            if compute_gradient:
                if (
                    smearing_T <= 0.0
                    and not _is_per_k_integer_aufbau(
                        result.occupations, n_occ
                    )
                ):
                    raise NotImplementedError(
                        "run_krhf_periodic_gdf: compute_gradient=True is "
                        "not validated for a band-overlap or degenerate "
                        "T=0 global-Aufbau ensemble. Use a gapped mesh "
                        "with integer occupations at every k point, or "
                        "positive Fermi-Dirac smearing (whose analytic "
                        "free-energy gradient is supported)."
                    )
                # G-PBC-002 Item-4 rung 6: the multi-k rsgdf analytic
                # gradient on the converged D(k)/C(k)/eps(k). The
                # entry guards pinned the supported envelope; here the
                # SCF's exact fit parameters, resolved lattice options
                # and quadrature are handed to the FD-gated assemblers
                # (the e_nuc gauge lesson: differentiate the energy the
                # SCF actually converged).
                from .periodic_gdf_gradient import (
                    _compute_krhf_gradient_multik,
                    _compute_krks_gradient_multik,
                )

                grad_cache = _build_multik_gradient_cache_checked(
                    system,
                    basis,
                    aux_modrho,
                    kpoints_cart,
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    lat_opts=lat_opts,
                    gdf_linear_dep_threshold=gdf_linear_dep_threshold,
                    n_fit_scf=n_fit,
                    entry="run_krhf_periodic_gdf",
                    plog=plog,
                    fit_screen_threshold=float(fit_screen_threshold),
                    source_cache=lpq_cache,
                )
                D_g = [np.asarray(D) for D in result.density]
                S_g = [np.asarray(S) for S in result.overlap]
                # Closed-shell energy-weighted density. T = 0:
                # W(k) = 2 C_occ(k) diag(eps_occ(k)) C_occ(k)^H
                # (integer slice, bit-identical to the pre-smearing
                # wiring). T > 0: the Mermin free-energy weight
                # W(k) = sum_i f_i(k) eps_i(k) c_i(k) c_i(k)^H with
                # the SCF's fractional occupations f_i(k) in [0, 2]
                # (result.occupations); D(k) is already the
                # fractional-occupation density, so the DF-J/K and
                # exxdiv D S D terms flow unchanged. The gradient is
                # dA/dR of result.free_energy = E - T S: at
                # self-consistency the occupation- and mu-response
                # terms vanish (Mermin 1965, Eqs. (1)-(4)/(9)-(10);
                # Marzari-Vanderbilt 1999's smeared-force statement)
                # because the Fermi-Dirac entropy is the exact
                # conjugate of the occupation function.
                if private_mdf:
                    from .periodic_mdf import _build_mdf_scf_energy_weighted_density

                    W_g = _build_mdf_scf_energy_weighted_density(
                        grad_cache, (D_g,), (result.fock,), S_g,
                        (result.mo_coeffs,), (result.mo_energies,), weights,
                        stationarity_tolerance=float(opts.conv_tol_grad),
                    )
                else:
                    W_g = []
                    for ik in range(n_k):
                        if smearing_T > 0.0:
                            occ_ik = np.asarray(
                                result.occupations[ik], dtype=float
                            )
                            keep = occ_ik > 1e-14
                            C_keep = np.asarray(result.mo_coeffs[ik])[:, keep]
                            w_i = occ_ik[keep] * np.real(
                                np.asarray(result.mo_energies[ik])[keep]
                            )
                            W_g.append(
                                (C_keep * w_i[None, :]) @ C_keep.conj().T
                            )
                        else:
                            C_occ = np.asarray(result.mo_coeffs[ik])[:, :n_occ]
                            eps_occ = np.asarray(result.mo_energies[ik])[:n_occ]
                            W_g.append(
                                2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T
                            )
                # BvK Madelung xi exactly as the SCF's exxdiv K-shift
                # resolves it (unused at alpha = 0, where the SCF never
                # computes it either).
                xi = (
                    _madelung_for_kmesh(
                        system, _mesh_tuple_for_system(system, kmesh)
                    )
                    if alpha != 0.0
                    else 0.0
                )
                if is_ks:
                    result.gradient = _compute_krks_gradient_multik(
                        system,
                        basis,
                        D_g,
                        W_g,
                        S_g,
                        weights,
                        kpoints_cart,
                        grad_cache,
                        functional=func_name,
                        xc_grid=grid,
                        kmesh_bloch=kmesh_bloch,
                        xc_lat_opts=lat_opts,
                        xc_density_cells=xc_cells,
                        xc_density_domain=xc_density_domain,
                        madelung=xi,
                        oneel_lat_opts=oneel_lat_opts,
                        gauge_lat_opts=gauge_lat_opts,
                        ecp_context=_ecp_ctx,
                        ecp_lat_opts=oneel_lat_opts,
                        xc_grid_options=grid_options,
                        xc_use_periodic_becke=bool(
                            getattr(opts, "use_periodic_becke", False)
                        ),
                        xc_becke_image_radius_bohr=float(
                            getattr(opts, "becke_image_radius_bohr", 0.0)
                        ),
                    )
                else:
                    result.gradient = _compute_krhf_gradient_multik(
                        system,
                        basis,
                        D_g,
                        W_g,
                        S_g,
                        weights,
                        kpoints_cart,
                        grad_cache,
                        alpha_hf=alpha,
                        madelung=xi,
                        oneel_lat_opts=oneel_lat_opts,
                        gauge_lat_opts=gauge_lat_opts,
                        ecp_context=_ecp_ctx,
                        ecp_lat_opts=oneel_lat_opts,
                    )
                plog.info(
                    "analytic gradient: multi-k GDF assembly done "
                    f"({'KRKS ' + func_name if is_ks else 'KRHF'}, "
                    f"{n_k} k-point(s))"
                )
            plog.converged(
                n_iter=result.n_iter,
                energy=result.energy,
                converged=True,
            )
            if check_energy_sanity:
                _check_energy_sanity(result, system, plog)
            return _finish_density_return(result)

    result.converged = False
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    if check_energy_sanity:
        _check_energy_sanity(result, system, plog)
    return _finish_density_return(result)


@dataclass
class PeriodicKUHFGDFResult:
    """Result of :func:`run_kuhf_periodic_gdf` / :func:`run_kuks_periodic_gdf`.

    Open-shell multi-k GDF: per-spin a/b per-k lists (length ``nkpts``,
    complex Hermitian in AO basis) + the ``<S^2>`` spin-contamination
    diagnostic. Mirrors :class:`PeriodicKRHFGDFResult` split by spin.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float

    mo_energies_alpha: List[np.ndarray]
    mo_coeffs_alpha: List[np.ndarray]
    density_alpha: List[np.ndarray]
    fock_alpha: List[np.ndarray]
    mo_energies_beta: List[np.ndarray]
    mo_coeffs_beta: List[np.ndarray]
    density_beta: List[np.ndarray]
    fock_beta: List[np.ndarray]

    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    kpoints_cart: np.ndarray
    kpoint_weights: np.ndarray
    scf_trace: List[SCFIteration] = field(default_factory=list)

    functional: Optional[str] = None
    e_xc: float = 0.0
    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    aux_basis_name: str = ""
    # What canonical orthogonalisation and the density fit actually
    # discarded. Populated by the drivers; rendered into the .out by
    # periodic_runner. None on routes that do not build it.
    linear_dependence: Optional["PeriodicLinearDependenceSummary"] = None
    n_aux: int = 0
    #: RSGDF high-``|G|`` tail cutoff (Ha) actually APPLIED, or ``None``
    #: for base-mesh-only -- mirrors :class:`PeriodicKRHFGDFResult`
    #: (GitLab IID 307); the multi-k loop does not consume a tail
    #: (IID 146), so this stays ``None`` here even when requested. The
    #: field exists so an open-shell cross-route comparison can assert
    #: matched reciprocal support (GitLab IID 490).
    rsgdf_tail_ke_cutoff: Optional[float] = None
    #: The BASE rsgdf reciprocal mesh (Ha) actually used (IID 307/490).
    rsgdf_ke_cutoff: float = 200.0
    backend: str = "native-multi-k-gdf-uhf"

    # Smearing (open-shell: independent per-spin chemical potentials).
    # All default to the no-smearing values so T = 0 runs are unchanged.
    smearing_temperature: float = 0.0
    fermi_level_alpha: float = 0.0
    fermi_level_beta: float = 0.0
    entropy: float = 0.0  # S/k_B per cell, summed over both spin channels
    free_energy: float = 0.0  # Mermin A = E - T.(S_a + S_b)
    occupations_alpha: List[np.ndarray] = field(default_factory=list)
    occupations_beta: List[np.ndarray] = field(default_factory=list)
    # The open-shell multi-k GDF loop does not implement previous-Fock mixing.
    fock_mixing: float = 0.0

    # (n_atoms, 3) real analytic nuclear gradient in Ha/bohr, populated
    # when the driver is called with compute_gradient=True; None otherwise.
    gradient: Optional[np.ndarray] = None

    density_lattice_mesh: Optional[Tuple[int, int, int]] = None
    density_lattice_reserved_peak_bytes: int = 0
    density_alpha_lattice: Optional[LatticeMatrixSet] = None
    density_beta_lattice: Optional[LatticeMatrixSet] = None

    @property
    def energy_per_cell_ha(self) -> float:
        return float(self.energy)

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _gdf_open_shell_occupations(
    eps_alpha, eps_beta, *, weights, n_alpha, n_beta, smearing,
):
    """One BZ-wide chemical potential per spin, including the T=0 limit.

    Use the same weighted, symmetry-preserving Aufbau as restricted GDF.
    Per-k integer slices select a different state at band crossings or a
    partially occupied degenerate frontier (#714).
    """
    if smearing.enabled:
        return _apply_smearing_open_shell(
            eps_alpha, eps_beta, weights=weights, n_alpha=n_alpha,
            n_beta=n_beta, smearing=smearing,
        )
    from .smearing import SmearingResult

    results = []
    for eps, count in ((eps_alpha, n_alpha), (eps_beta, n_beta)):
        occupations, mu = _global_aufbau_with_mu(
            eps, weights, float(count), occ_value=1.0,
        )
        results.append(SmearingResult(
            occupations_per_k=occupations, mu=mu, entropy=0.0,
            free_energy_correction=0.0, smearing=smearing,
        ))
    return tuple(results)


def _multi_k_s_squared(
    n_alpha: int,
    n_beta: int,
    C_alpha_k: List[np.ndarray],
    C_beta_k: List[np.ndarray],
    S_k: List[np.ndarray],
    weights: Sequence[float],
    *,
    occ_alpha_k: Optional[Sequence[np.ndarray]] = None,
    occ_beta_k: Optional[Sequence[np.ndarray]] = None,
) -> float:
    """<S^2> for a multi-k UHF/UKS determinant: the spin value
    ``Sz(Sz+1) + n_b`` minus the BZ-weighted a/b overlap
    ``S_k w_k S_ij n^a_i(k) n^b_j(k) |<a_i(k)|S(k)|b_j(k)>|^2``. At M=1 (a==b)
    this is 0; for an integer-filled doublet it is 0.75.

    ``occ_alpha_k`` / ``occ_beta_k`` are the per-k fractional occupations (in
    ``[0, 1]``) from the smearing path. When supplied and non-empty (T > 0)
    the a/b overlap is weighted by ``n^a_i n^b_j`` over **all** orbitals -- the
    fractional-occupation (ensemble-UHF) generalisation of Szabo & Ostlund
    Eq. 2.271. When absent or empty (T = 0 / integer filling) it falls back to
    the first-``n_s`` hard cutoff with unit weights, which is the exact
    integer-occupation value and stays **bit-identical** to the pre-smearing
    path. ``n_alpha`` / ``n_beta`` are the conserved per-spin totals
    (``S_k w_k S_i n^s_i = n_s`` exactly under the fixed-multiplicity
    constraint), so the ``Sz(Sz+1) + n_b`` part is unchanged by fractional
    occupation."""
    diff = n_alpha - n_beta
    s2 = 0.25 * diff * (diff + 2) + n_beta
    smeared = (
        occ_alpha_k is not None
        and occ_beta_k is not None
        and len(occ_alpha_k) > 0
    )
    if smeared:
        for i in range(len(S_k)):
            oa = np.asarray(occ_alpha_k[i], dtype=float)
            ob = np.asarray(occ_beta_k[i], dtype=float)
            # All-orbital a/b overlap weighted by the fractional per-spin
            # occupations n^a_i n^b_j (reduces to the first-n_s unit-weight
            # cutoff below when occ in {0, 1}).
            M = C_alpha_k[i].conj().T @ S_k[i] @ C_beta_k[i]
            s2 -= float(weights[i]) * float(
                np.sum(np.outer(oa, ob) * np.abs(M) ** 2)
            )
    elif n_alpha > 0 and n_beta > 0:
        for i in range(len(S_k)):
            Ca = C_alpha_k[i][:, :n_alpha]
            Cb = C_beta_k[i][:, :n_beta]
            M = Ca.conj().T @ S_k[i] @ Cb
            s2 -= float(weights[i]) * float(np.sum(np.abs(M) ** 2))
    return float(s2)


def run_kuhf_periodic_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh] = (1, 1, 1),
    options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    *,
    initial_density_k: Optional[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = None,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_ft_convention: str = "libint",
    aft_precision: float = 1e-10,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    gdf_method: str = "rsgdf",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    fit_screen_threshold: float = 0.0,
    mdf_ke_cutoff: float = 40.0,
    ibz_native: bool = False,
    k_exchange: str = "gdf",
    compute_gradient: bool = False,
    return_lattice_density: bool = False,
    lattice_density_memory_bytes: int = 128 * 1024**2,
    check_energy_sanity: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    _lpq_cache_builder=None,
    xc_density_domain: PeriodicXCDensityDomain = (
        PeriodicXCDensityDomain.AUTO
    ),
) -> PeriodicKUHFGDFResult:
    """Open-shell (UHF / UKS) periodic multi-k SCF via native GDF.

    The open-shell sibling of :func:`run_krhf_periodic_gdf`: spin-independent
    per-``(kᵢ,kⱼ)`` ``Lpq`` cderi cache, Hartree ``J`` from the BZ-summed
    **total** density ``S_j w_j (Da+Db)(k_j)``, per-spin exchange ``Ks``
    from the same cache, and the ``exxdiv='ewald'`` supercell-Madelung
    K-shift applied per spin. a/b occupations follow ``multiplicity`` via
    per-k Aufbau (``na=(n_e+mult-1)//2``); at M=1 on a gapped cell this
    reproduces :func:`run_krhf_periodic_gdf`. ``functional`` selects UKS
    (native spin-polarised XC on the per-spin real-space finite-torus
    density fold, Bloch-folded to every k -- the closed-shell multi-k
    convention, :func:`_build_xc_k_from_density_uks` -- plus the
    functional's HF-exchange fraction); ``None`` runs UHF. KS runs
    default ``options`` to :class:`PeriodicKSOptions` (periodic-Becke
    grid); UHF keeps :class:`PeriodicRHFOptions`.

    Returns a :class:`PeriodicKUHFGDFResult`. ``smearing_temperature > 0``
    enables per-spin Fermi-Dirac smearing with independent global chemical
    potentials mu_a, mu_b across the BZ (:func:`apply_smearing_open_shell`),
    Mermin free energy ``A = E - T(S_a + S_b)``, and fractional per-spin
    occupations (``occupations_alpha/beta``); ``T = 0`` keeps the exact per-k
    Aufbau, bit-identical to the pre-smearing driver.

    ``rsgdf_tail_ke_cutoff`` extends the rsgdf fit's M / T G-sums over the
    exact complementary reciprocal shell up to the given cutoff (Hartree),
    exactly like the closed-shell driver: needed on dense-core cells
    (P01/MgO class) whose tight AO products are unresolved at any
    affordable base mesh. ``E(ke_cutoff=a, tail_ke_cutoff=b)`` equals
    ``E(ke_cutoff=b)`` by construction, and a parity-sized tail
    (``>= _RSGDF_PARITY_TAIL_RATIO x zeta_max``) lifts the dense-core
    ``+PARITY_HELD`` tag. The shared-q batched cderi build amortises the
    tail's pair-FT pass across the k-points of each momentum transfer.

    ``compute_gradient=True`` computes the analytic nuclear gradient of
    the converged total energy on ``result.gradient`` (``(n_atoms, 3)``
    Ha/bohr; G-PBC-002 Item 4). Same supported envelope and fail-closed
    guards as :func:`run_krhf_periodic_gdf`: 3D, ``gdf_method='rsgdf'``,
    uniform full-BZ meshes, T = 0 Aufbau, ``k_exchange='gdf'``, pure or
    global-hybrid functionals; the per-spin HF assembly is FD-gated at
    1.6e-8 Ha/bohr (H2 triplet, (2,1,1)), while genuine open-shell KUKS
    includes the atom-centred XC grid-motion response and is FD-gated at
    about 7e-9 Ha/bohr (LiH triplet PBE). The KUKS(M=1) collapse onto
    KRKS remains pinned at machine precision.
    """
    from .periodic_mdf import _MdfScfSource

    private_mdf = isinstance(_lpq_cache_builder, _MdfScfSource)
    if private_mdf:
        _lpq_cache_builder.require_driver(
            system, gdf_method=gdf_method, k_exchange=k_exchange, ibz_native=ibz_native)
    if _lpq_cache_builder is not None and (int(system.dim) != 3 or gdf_method != 'rsgdf'):
        raise NotImplementedError('open-shell private cache injection requires the 3D rsgdf driver')
    _finish_density_return = _gdf_density_return_finalizer(
        system, basis, kmesh, return_lattice_density, 2,
        lattice_density_memory_bytes,
    )
    _reject_slab_dim(system, "run_kuhf_periodic_gdf")
    from .periodic_scf_accelerators import MultiKPeriodicUHFAccelerator
    from .periodic_uhf_ewald import _spin_squared  # noqa: F401  (parity ref)
    from .periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    plog = resolve_progress(progress, verbose=verbose)
    # KS runs must default to PeriodicKSOptions: PeriodicRHFOptions carries no
    # ``use_periodic_becke``, so defaulting KUKS to it silently selected the
    # molecular Becke grid while the closed-shell KRKS wrapper defaulted to the
    # periodic-Becke grid -- the grid/density-convention split behind the
    # -2.4 Ha KUKS(mult=1)-vs-KRKS failure of the first fold fix (2026-07-09).
    caller_supplied_options = options is not None
    opts = _options_or_default(options, is_ks=functional is not None)
    lat_opts: LatticeSumOptions = opts.lattice_opts
    guess = _coerce_periodic_driver_guess(
        resolve_initial_guess(
            system.unit_cell_molecule(),
            getattr(opts, "initial_guess", InitialGuess.AUTO),
            is_periodic=True,
            is_open_shell=True,
        ),
        driver="run_kuhf_periodic_gdf",
        supported=periodic_guess_capabilities('gdf', 'UHF', dim=getattr(system, "dim", 3), multi_k=True, transport='k'),
        restart_supplied=initial_density_k is not None,
    )
    from .guess import guess_ecp_context, validate_guess_ecp
    guess_mol = system.unit_cell_molecule()
    _guess_ecp = validate_guess_ecp(
        guess, guess_ecp_context(opts, molecule=guess_mol), guess_mol)
    if guess == InitialGuess.READ and initial_density_k is None:
        raise ValueError("run_kuhf_periodic_gdf: READ requires paired initial_density_k spin blocks")
    if getattr(opts, "atomic_spins", None) and guess != InitialGuess.SAD:
        raise RuntimeError(
            "GuessEngine: atomic_spins (ATOMSPIN broken-symmetry seed) "
            f"requires the SAD guess; resolved guess is {guess.name}"
        )

    func_name = functional or str(getattr(opts, "functional", "") or "")
    is_ks = bool(func_name)
    func = Functional(func_name, 2) if is_ks else None  # spin-polarized
    requested_external_xc = bool(
        func is not None and getattr(func, "is_external", False)
    )
    if requested_external_xc:
        from .pbc_bipole_common import reject_bipole_ecp_options
        from .periodic_external_xc import (
            _grid_options_for_external,
            _require_zero_temperature_external_xc,
        )

        _require_zero_temperature_external_xc(
            getattr(opts, "smearing_temperature", 0.0),
            where="run_kuhf_periodic_gdf",
        )

        try:
            reject_bipole_ecp_options(
                opts,
                driver="run_kuhf_periodic_gdf external XC",
                basis=basis,
                system=system,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "run_kuhf_periodic_gdf: ECP-bearing bases are not "
                "implemented with external XC; use an all-electron basis"
            ) from exc
        opts.grid = _grid_options_for_external(
            func,
            opts.grid if caller_supplied_options else None,
            where="run_kuhf_periodic_gdf",
        )
        if xc_density_domain == PeriodicXCDensityDomain.MOLECULAR_HOME:
            raise ValueError(
                "run_kuhf_periodic_gdf: a full-grid external functional "
                "requires the periodic-lattice XC density domain"
            )
        if not bool(getattr(opts, "use_periodic_becke", False)):
            raise ValueError(
                "run_kuhf_periodic_gdf: full-grid external XC requires "
                "use_periodic_becke=True; a molecular atom partition is "
                "not valid for a compact periodic density"
            )
        external_image_radius = float(
            getattr(opts, "becke_image_radius_bohr", 0.0)
        )
        if not np.isfinite(external_image_radius) or external_image_radius <= 0.0:
            raise ValueError(
                "run_kuhf_periodic_gdf: external XC requires "
                "becke_image_radius_bohr to be finite and > 0"
            )
        if float(lat_opts.cutoff_bohr) < external_image_radius:
            raise ValueError(
                "run_kuhf_periodic_gdf: external XC cutoff_bohr must be at "
                "least becke_image_radius_bohr"
            )
        lat_opts.becke_image_radius_bohr = external_image_radius
        if compute_gradient:
            raise NotImplementedError(
                "run_kuhf_periodic_gdf: analytic gradients are not "
                "implemented for full-grid external XC functionals; use "
                "whole-SCF finite differences"
            )
        xc_density_domain = PeriodicXCDensityDomain.PERIODIC_LATTICE
    reject_periodic_gdf_unsupported_functional(
        func, where="run_kuhf_periodic_gdf"
    )
    # Match the closed-shell multi-k policy: HSE-type functionals
    # (c_full = 0, c_sr > 0) use the physical erfc screened COSX
    # exchange. Fitted GDF K and range-separated functionals with a
    # full-range arm remain fail-closed.
    screened_omega = None
    if (
        k_exchange == "cosx"
        and func is not None
        and bool(getattr(func, "is_range_separated", False))
    ):
        from .periodic_screened_exchange import resolve_periodic_exchange

        _exx = resolve_periodic_exchange(
            func, where="run_kuhf_periodic_gdf(k_exchange='cosx')"
        )
        alpha = float(_exx.c_sr)
        screened_omega = float(_exx.omega_screen)
    else:
        reject_unscreened_range_separated(func, where="run_kuhf_periodic_gdf")
        alpha = float(func.hf_exchange_fraction) if func is not None else 1.0
    label = f"KUKS {func_name}" if is_ks else "KUHF"

    if gdf_method not in ("compcell", "rsgdf", "mdf"):
        raise ValueError(
            f"run_kuhf_periodic_gdf: gdf_method must be 'compcell', "
            f"'rsgdf', or 'mdf'; got {gdf_method!r}"
        )
    bulk_sr = gdf_method == "rsgdf" and int(system.dim) == 3
    if bulk_sr and rsgdf_tail_ke_cutoff is not None:
        warnings.warn(
            "rsgdf_tail_ke_cutoff is obsolete for the SR/LR fit; "
            "use rsgdf_g_precision to control the raw integral error",
            DeprecationWarning, stacklevel=2,
        )
        rsgdf_tail_ke_cutoff = None
    if float(fit_screen_threshold) < 0.0:
        raise ValueError(
            "run_kuhf_periodic_gdf: fit_screen_threshold must be >= 0; "
            f"got {fit_screen_threshold}"
        )
    if float(fit_screen_threshold) > 0.0 and gdf_method != "rsgdf":
        raise NotImplementedError(
            "run_kuhf_periodic_gdf: fit_screen_threshold is implemented "
            f"for gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    if rsgdf_tail_ke_cutoff is not None and gdf_method != "rsgdf":
        raise NotImplementedError(
            "run_kuhf_periodic_gdf: rsgdf_tail_ke_cutoff is implemented "
            f"for gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    if k_exchange not in ("gdf", "cosx"):
        raise ValueError(
            "run_kuhf_periodic_gdf: k_exchange must be 'gdf' or "
            f"'cosx'; got {k_exchange!r}"
        )
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0) or 0.0)
    if smearing_T < 0.0:
        raise ValueError(
            "run_kuhf_periodic_gdf: smearing_temperature must be >= 0"
        )
    # Open-shell Fermi-Dirac smearing (M3): independent per-spin chemical
    # potentials mu_a, mu_b via apply_smearing_open_shell. T = 0 keeps the
    # exact per-k Aufbau path (bit-identical to the pre-smearing driver).
    smear_opts = _SmearingOptions.from_legacy_kwarg(smearing_T)

    _ecp_ctx = _periodic_ecp_context(opts, system, "run_kuhf_periodic_gdf")
    _ecp_ncore = _ecp_ctx[3] if _ecp_ctx is not None else 0
    _system_v = _ecp_ctx[4] if _ecp_ctx is not None else system
    n_elec = system.n_electrons() - _ecp_ncore
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(f"run_kuhf_periodic_gdf: multiplicity must be >= 1, got {mult}")
    if (n_elec + mult - 1) % 2 != 0:
        raise ValueError(
            f"run_kuhf_periodic_gdf: n_electrons={n_elec} and multiplicity="
            f"{mult} cannot be split into integer a/b occupations."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2

    if requested_external_xc and isinstance(kmesh, (KPoints, BlochKMesh)):
        from .periodic_external_xc import _reject_reduced_kmesh

        input_kmesh_bloch = (
            kmesh.to_bloch_kmesh() if isinstance(kmesh, KPoints) else kmesh
        )
        _reject_reduced_kmesh(input_kmesh_bloch, system)

    # Symmetry-reduced (IBZ) input: adopt the expanded full-BZ mesh as
    # THE kmesh before anything is derived from it (see the matching
    # note in run_krhf_periodic_gdf -- expanding only the arrays while
    # ``kmesh_bloch`` stays on the wedge crashes every KS job here).
    kmesh_full = _expand_ibz_kmesh_to_full_bz(system, kmesh)
    if kmesh_full is not None:
        kmesh = kmesh_full
    kpoints_cart, weights = _kmesh_to_kpoints_weights(system, kmesh)
    n_k = kpoints_cart.shape[0]

    # Multi-k analytic-gradient envelope (G-PBC-002 Item-4 rung 6):
    # fail closed BEFORE the SCF on anything the per-spin gradient
    # assembly does not differentiate. The open-shell driver always
    # builds the cached Lpq fit, so use_compcell holds by construction.
    if compute_gradient:
        _reject_unsupported_multik_gradient(
            "run_kuhf_periodic_gdf",
            dim=int(system.dim),
            gdf_method=gdf_method,
            smearing_temperature=smearing_T,
            k_exchange=k_exchange,
            screened_omega=screened_omega,
            functional_is_range_separated=bool(
                getattr(func, "is_range_separated", False)
            ),
            weights=weights,
            # The KUHF surface has no flavor control: smearing_T routes
            # through SmearingOptions.from_legacy_kwarg -> Fermi-Dirac.
            smearing_flavor="fermi-dirac",
            ibz_native=bool(ibz_native),
        )
    if isinstance(kmesh, BlochKMesh):
        kmesh_bloch = kmesh
    elif isinstance(kmesh, KPoints):
        kmesh_bloch = kmesh.to_bloch_kmesh()
    else:
        mesh = _mesh_tuple_for_system(system, kmesh)
        kmesh_bloch = _mp_native(system, list(mesh), [0, 0, 0], False)
    # Admit the bulk AO domain before any lattice/XC cell enumeration or
    # quadrature allocation (#99). The later integral-only guard was too
    # late even for the direct cell lists printed in the setup log.
    # Preserve the historical low-dimensional cutoff-resolution order.
    oneel_lat_opts = None
    if int(system.dim) == 3:
        oneel_lat_opts = _oneel_lattice_opts(
            system, basis, lat_opts,
            rcut_strategy=rcut_strategy, k_points_cart=kpoints_cart, plog=plog,
        )
        _preflight_gdf_oneel_memory(
            system, basis, oneel_lat_opts, n_kpoints=n_k,
            ecp_active=_ecp_ctx is not None, plog=plog,
        )
    cells = _direct_cells(system, lat_opts.cutoff_bohr)
    xc_cells = _xc_density_cells_for_domain(
        system,
        lat_opts,
        cells,
        xc_density_domain,
    )

    aux_name = aux_basis or default_aux_for(basis.name)
    plog.banner(f"run_kuhf_periodic_gdf  {label}  kmesh={n_k} k-points")
    # A parity-sized rsgdf_tail_ke_cutoff lifts the dense-core hold here
    # exactly like the closed-shell driver (the classifier compares the
    # tail against _RSGDF_PARITY_TAIL_RATIO x zeta_max).
    dense_core_parity_held = not bulk_sr and _warn_multik_dense_core_gdf_parity_hold(
        system,
        gdf_method,
        basis,
        rsgdf_tail_ke_cutoff,
        rsgdf_ke_cutoff,
        "run_kuhf_periodic_gdf",
    )
    plog.info(
        f"{label} multi-k GDF / aux={aux_name}, n_alpha={n_alpha}, "
        f"n_beta={n_beta} (mult={mult}), alpha={alpha:g}"
    )

    # ---- Functional grid (UKS) ----------------------------------------
    grid = None
    if is_ks:
        grid_options = getattr(opts, "grid", None) or GridOptions()
        if requested_external_xc or bool(
            getattr(opts, "use_periodic_becke", False)
        ):
            grid = build_periodic_becke_grid(
                system, grid_options=grid_options,
                image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
            )
        else:
            grid = build_grid(system.unit_cell_molecule(), grid_options)

    # ---- One-electron integrals (Ewald-3D gauge for V_ne/e_nuc) -------
    # Resolve the same AO image domain as the restricted/Gamma paths.
    if oneel_lat_opts is None:
        oneel_lat_opts = _oneel_lattice_opts(
            system, basis, lat_opts,
            rcut_strategy=rcut_strategy, k_points_cart=kpoints_cart, plog=plog,
        )
    # Recheck after setup allocations, now including retained quadrature
    # and the current available budget. This does not change the AO domain.
    _preflight_gdf_oneel_memory(
        system, basis, oneel_lat_opts, n_kpoints=len(kpoints_cart),
        ecp_active=_ecp_ctx is not None, grid=grid, plog=plog,
    )
    with plog.stage(
        "integrals_lattice",
        detail=f"S/T cutoff {oneel_lat_opts.cutoff_bohr:.2f}, "
        "V on the screened AO-pair domain",
    ):
        S_lat = compute_overlap_lattice(basis, system, oneel_lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, oneel_lat_opts)
        gauge_lat_opts = _gauge_lat_opts_for_v_ne_and_e_nuc(oneel_lat_opts, system)
        V_lat = compute_nuclear_lattice_dispatch(basis, _system_v, gauge_lat_opts)
        V_ecp_lat = (
            _ecp_lattice_blocks(basis, system, oneel_lat_opts, _ecp_ctx)
            if _ecp_ctx is not None
            else None
        )

    S_k: List[np.ndarray] = []
    Hcore_k: List[np.ndarray] = []
    X_k: List[np.ndarray] = []
    n_kept_k: List[int] = []
    _s_lo, _s_hi = float("inf"), float("-inf")
    for k_idx in range(n_k):
        k_arr = kpoints_cart[k_idx]
        Sk = np.asarray(bloch_sum(S_lat, k_arr))
        Tk = np.asarray(bloch_sum(T_lat, k_arr))
        Vk = np.asarray(bloch_sum(V_lat, k_arr))
        if V_ecp_lat is not None:
            Vk = Vk + np.asarray(bloch_sum(V_ecp_lat, k_arr))
        Sk = 0.5 * (Sk + Sk.conj().T)
        Hk = 0.5 * ((Tk + Vk) + (Tk + Vk).conj().T)
        _gdf_overlap_preflight(
            Sk,
            plog=plog,
            label=f"S(k={k_idx})",
            basis=basis,
        )
        _ev = np.linalg.eigvalsh(Sk)
        _s_lo = min(_s_lo, float(_ev[0]))
        _s_hi = max(_s_hi, float(_ev[-1]))
        Xk, n_kept = _canonical_orthogonalizer_complex(
            Sk, linear_dep_threshold, normalize_diag_first=True
        )
        if max(n_alpha, n_beta) > n_kept:
            raise RuntimeError(
                f"run_kuhf_periodic_gdf: orthogonalisation at k={k_idx} kept "
                f"{n_kept} directions; need >= {max(n_alpha, n_beta)}."
            )
        S_k.append(Sk)
        Hcore_k.append(Hk)
        X_k.append(Xk)
        n_kept_k.append(n_kept)

    if int(system.dim) == 3:
        # Converged Ewald nuclear energy — see pbc_gdf._pbc_gdf_gamma_setup
        # for the truncation-artefact rationale.
        e_nuc = float(ewald_nuclear_repulsion(_system_v))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(_system_v, gauge_lat_opts))

    # ---- IBZ-native exchange (opt-in, open shell) --------------------
    # Same design as the closed-shell driver: exchange is built at the
    # irreducible wedge and symmetry-transported to the rest of the mesh.
    # Both spin channels transport independently -- K_alpha and K_beta
    # are each operators in the AO Bloch basis with the same
    # transformation law -- so the per-spin closure below needs no
    # special casing. Diagonalisation stays full-mesh, so the result
    # shape and every downstream consumer are unchanged.
    if ibz_native and k_exchange != "gdf":
        raise NotImplementedError(
            "ibz_native=True applies to the fitted GDF exchange "
            f"(k_exchange='gdf'); got k_exchange={k_exchange!r}. The COSX "
            "backend builds K in real space from its own bridge and never "
            "touches the per-(k_i,k_j) cderi cache this flag reduces."
        )
    _ibz_native_state = None
    _ibz_symmetry_checked: List[bool] = []
    if ibz_native:
        if alpha == 0.0:
            raise NotImplementedError(
                "ibz_native=True has no effect without exact exchange "
                "(alpha == 0): the n_k^2 term it reduces is the HF "
                "exchange. Drop the flag for a pure functional."
            )
        _ibz_native_state = _resolve_ibz_native_state(
            system, kpoints_cart, kmesh, weights, plog
        )
        _rows_n = len(_ibz_native_state[0])
        plog.info(
            f"IBZ-native exchange: {_rows_n} of {n_k} k-points carry the "
            f"exchange build ({n_k / max(_rows_n, 1):.1f}x fewer cderi "
            f"pairs); the rest are symmetry-transported (per spin)."
        )
        if float(getattr(lat_opts, "cutoff_bohr", 0.0)) < 20.0:
            warnings.warn(
                "ibz_native=True at lattice cutoff "
                f"{float(getattr(lat_opts, 'cutoff_bohr', 0.0)):.1f} bohr: "
                "the symmetry transport is only as accurate as the cell "
                "list. Use cutoff_bohr >= 20 for the transport to stay "
                "below the truncation floor.",
                stacklevel=2,
            )


    # ---- Per-(kᵢ,kⱼ) Lpq cderi cache (spin-independent) ---------------
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=float(aux_drop_eta))
    q_metric_cache = None
    if gdf_method == "rsgdf":
        # The rsgdf cache is built through the shared-q batch below
        # (_build_rsgdf_lpq_cache_shared_q); no per-pair closure needed.
        aux_modrho = aux if bulk_sr else make_modrho_aux_basis(aux, mol)
        q_metric_cache = {}
    elif gdf_method == "mdf":

        def _build_pair_lpq(ki: np.ndarray, kj: np.ndarray) -> np.ndarray:
            # MDF: ket-resolved combined cderi [L_gauss; cderi_pw].
            return build_lpq_bloch_mdf(
                system, basis, aux, ki, kj, molecule=mol, lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                compcell_eta=float(compcell_eta),
                mdf_ke_cutoff=float(mdf_ke_cutoff),
                rcut_strategy=rcut_strategy, rcut_precision=float(rcut_precision),
            )
    else:

        def _build_pair_lpq(ki: np.ndarray, kj: np.ndarray) -> np.ndarray:
            return build_lpq_bloch_compcell(
                system, basis, aux, kj - ki, molecule=mol, lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                compcell_eta=float(compcell_eta),
                apply_aft_correction=bool(apply_aft_correction),
                aft_ft_convention=str(aft_ft_convention),
                aft_precision=float(aft_precision),
                rcut_strategy=rcut_strategy, rcut_precision=float(rcut_precision),
            )

    lpq_cache: Dict[Tuple[int, int], np.ndarray] = {}
    need_k_pairs = (alpha != 0.0 and k_exchange == "gdf") or guess == InitialGuess.PATOM
    # Fail early rather than OOM-killing a doomed run: the dense per-pair Lpq
    # cache built below is the multi-k open-shell GDF memory bottleneck. Abort
    # with a route-specific diagnostic if it cannot fit (prompt-75 NiO KUKS).
    _preflight_gdf_lpq_memory(
        plog,
        n_basis=basis.nbasis,
        n_aux=aux.nbasis,
        n_kpoints=n_k,
        need_k_pairs=need_k_pairs,
        open_shell=True,
        route_label=label,
        options=opts,
        n_ibz_kpoints=(
            len(_ibz_native_state[0])
            if (_ibz_native_state is not None and need_k_pairs)
            else None
        ),
    )
    with plog.stage("gdf_cderi", detail=f"per-pair Lpq, {n_k} k ({gdf_method})"):
        if gdf_method == "rsgdf":
            # Shared-q batch: one ket-Bloch pair-FT pass per unique
            # momentum transfer (see _build_rsgdf_lpq_cache_shared_q).
            if bulk_sr:
                lpq_cache = _build_scf_range_separated_lpq_cache(
                    system, basis, aux, kpoints_cart, need_k_pairs,
                    omega=rsgdf_omega, raw_integral_error=rsgdf_g_precision,
                    ke_cutoff=rsgdf_ke_cutoff, linear_dep_thr=gdf_linear_dep_threshold,
                    lat_opts=lat_opts, fit_screen_threshold=fit_screen_threshold,
                    options=opts, open_shell=True, progress=plog,
                    pair_cache_builder=_lpq_cache_builder,
                        oneel_lattices=(S_lat, T_lat, V_lat, V_ecp_lat),
                        xc_grid=grid, xc_cells=xc_cells, functional=func,
                        xc_density_domain=xc_density_domain,
                        compute_gradient=compute_gradient,
                        density_return_bytes=getattr(
                            getattr(_finish_density_return, '__self__', None),
                            'reserved_peak_bytes', 0,
                        ),
                    bra_rows=(_ibz_native_state[0] if (_ibz_native_state is not None and need_k_pairs) else None),
                )
            else:
                lpq_cache = _build_rsgdf_lpq_cache_shared_q(
                    system,
                    basis,
                    aux_modrho,
                    kpoints_cart,
                    need_k_pairs,
                    ke_cutoff=float(rsgdf_ke_cutoff),
                    tail_ke_cutoff=(
                        float(rsgdf_tail_ke_cutoff)
                        if rsgdf_tail_ke_cutoff is not None
                        else None
                    ),
                    lat_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    fit_screen_threshold=float(fit_screen_threshold),
                    progress=plog,
                    q_metric_cache=q_metric_cache,
                    bra_rows=(
                        _ibz_native_state[0]
                        if (_ibz_native_state is not None and need_k_pairs)
                        else None
                    ),
                )
        else:
            for i in range(n_k):
                ki = kpoints_cart[i]
                if need_k_pairs:
                    for j in range(n_k):
                        lpq_cache[(i, j)] = _build_pair_lpq(
                            ki, kpoints_cart[j]
                        )
                else:
                    lpq_cache[(i, i)] = _build_pair_lpq(ki, ki)
    n_fit = lpq_cache[(0, 0)].shape[0]
    plog.info(f"Lpq cache: {len(lpq_cache)} pairs, {n_fit} fit vectors")
    if q_metric_cache is not None:
        q_states = (lpq_cache.q_batch_sizes if private_mdf else
                    getattr(lpq_cache, 'reciprocal_vector_counts', q_metric_cache))
        plog.info(
            f"{'Private MDF' if private_mdf else 'RSGDF'} q-state shared across pairs: "
            f"{len(q_states)} unique momentum transfers"
        )

    # COSX is a spin-independent exchange operator builder: the same
    # geometry/grid cache acts separately on D_alpha and D_beta.  The
    # returned matrices use the same G=0-dropped gauge as the GDF factors;
    # the existing per-spin Ewald Madelung correction below is therefore
    # applied exactly once for either exchange backend.
    cosx_bridge = None
    if k_exchange == "cosx" and alpha != 0.0:
        from .periodic_cosx_k import KPointCosxK

        lat_np = np.asarray(system.lattice, dtype=float)
        mesh_dims = _mesh_tuple_for_system(system, kmesh)
        half_supers = [
            0.5 * mesh_dims[axis] * float(np.linalg.norm(lat_np[axis]))
            for axis in range(int(system.dim))
        ]
        sr_reach = min(min(half_supers), float(lat_opts.cutoff_bohr))
        if sr_reach <= 0.0:
            raise ValueError("run_kuhf_periodic_gdf: invalid COSX super-period")
        cosx_omega = 5.0 / sr_reach
        if screened_omega is not None and screened_omega >= cosx_omega:
            cosx_omega = float(screened_omega)
        with plog.stage("cosx_caches"):
            cosx_bridge = KPointCosxK(
                basis, system, lat_opts=lat_opts, omega=cosx_omega
            )
        plog.info(
            "K backend: open-shell multi-k COSX, range-separated "
            f"split omega={cosx_omega:.3f} bohr^-1"
            + (
                f", physical erfc omega={screened_omega:.3f} bohr^-1"
                if screened_omega is not None
                else ""
            )
        )

    # ---- GDF J/K builders (copied from run_krhf_periodic_gdf) ---------
    def _build_j_from_lpq(D_k_in: List[np.ndarray]) -> List[np.ndarray]:
        from .aux_basis import _build_coulomb_from_diagonal_factors

        return _build_coulomb_from_diagonal_factors(
            [lpq_cache[(i, i)] for i in range(n_k)], D_k_in, weights,
        )

    def _build_k_from_lpq(D_k_in: List[np.ndarray]) -> List[np.ndarray]:
        if _ibz_native_state is None:
            return _build_k_from_lpq_cache(
                lpq_cache,
                D_k_in,
                weights,
                nbasis=basis.nbasis,
            )
        rows, star_map, kmesh_full_native = _ibz_native_state
            # Checked EVERY iteration, not once: the initial guess is
            # symmetric even when the converged state is not, so a
            # one-shot check on the guess passes and the break appears
            # later. Measured on the LiH FCC triplet -- guess symmetric,
            # converged density 1.7e-3 asymmetric. The cost is one
            # transport of n_IBZ matrices, negligible against the K build.
        if True:
            _require_ibz_symmetric_state(
                [np.asarray(D) for D in D_k_in],
                S_k,
                rows,
                star_map,
                system,
                basis,
                kmesh_full_native,
                label="density at this SCF iteration",
            )
        K_rows = _build_k_ibz_native(
            lpq_cache, D_k_in, weights, rows, nbasis=basis.nbasis
        )
        from .periodic_k_symmetry import expand_k_matrices_to_full

        return expand_k_matrices_to_full(
            K_rows, star_map, system, basis, kmesh_full_native
        )

    nbf = basis.nbasis

    def _occupy_and_density(
        C_alpha: List[np.ndarray], eps_alpha: List[np.ndarray],
        C_beta: List[np.ndarray], eps_beta: List[np.ndarray],
    ):
        """Occupations, densities and entropy at one global mu per spin."""
        a_res, b_res = _gdf_open_shell_occupations(
            eps_alpha, eps_beta, weights=list(weights),
            n_alpha=n_alpha, n_beta=n_beta, smearing=smear_opts,
        )
        return (
            _density_from_orbitals(C_alpha, a_res.occupations_per_k),
            _density_from_orbitals(C_beta, b_res.occupations_per_k),
            a_res.occupations_per_k, b_res.occupations_per_k,
            float(a_res.mu), float(b_res.mu),
            float(a_res.entropy + b_res.entropy),
        )

    # ---- Initial guess: shared per-k Fock/density artifacts -----------
    guess_fock_k = Hcore_k
    if guess in (InitialGuess.SAP, InitialGuess.HUECKEL):
        fock_guess = periodic_fock_guess_k(
            system,
            basis,
            kpoints_cart,
            guess,
            lattice_opts=oneel_lat_opts,
            kinetic_lattice=T_lat,
            overlap_lattice=S_lat,
            ecp_context=_guess_ecp,
        )
        assert fock_guess is not None
        guess_fock_k = fock_guess
    C_alpha_k: List[np.ndarray] = []
    eps_alpha_k: List[np.ndarray] = []
    for i in range(n_k):
        Ci, ei = _diag_in_orth_basis(guess_fock_k[i], X_k[i])
        C_alpha_k.append(Ci.astype(complex))
        eps_alpha_k.append(ei)
    C_beta_k = [C.copy() for C in C_alpha_k]
    eps_beta_k = [e.copy() for e in eps_alpha_k]
    (D_alpha_k, D_beta_k, occ_alpha_k, occ_beta_k,
     fermi_alpha, fermi_beta, entropy_total) = _occupy_and_density(
        C_alpha_k, eps_alpha_k, C_beta_k, eps_beta_k
    )
    if guess in (InitialGuess.SAD, InitialGuess.MINAO, InitialGuess.PATOM):
        density_guess = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha,
            n_beta,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=oneel_lat_opts,
            atomic_spins=getattr(opts, "atomic_spins", None) or None,
            overlap=S_k, weights=weights,
            ecp_context=_guess_ecp,
        )
        assert density_guess is not None
        density_alpha, density_beta = density_guess
        density_alpha = np.asarray(density_alpha, dtype=complex)
        density_beta = np.asarray(density_beta, dtype=complex)
        density_alpha = 0.5 * (
            density_alpha + density_alpha.conj().T
        )
        density_beta = 0.5 * (
            density_beta + density_beta.conj().T
        )
        D_alpha_k = [density_alpha.copy() for _ in range(n_k)]
        D_beta_k = [density_beta.copy() for _ in range(n_k)]
        plog.info(
            f"initial guess: {guess.name} "
            "(shared g=0 spin densities injected at every k-point)"
        )
    elif guess in (InitialGuess.SAP, InitialGuess.HUECKEL):
        plog.info(
            f"initial guess: {guess.name} "
            "(shared per-k Fock diagonalisation; spin-resolved occupations)"
        )
    elif guess == InitialGuess.HCORE:
        plog.info("initial guess: HCORE (per-k Hcore diagonalisation)")
    if smearing_T > 0.0:
        plog.info(
            "smearing: per-spin Fermi-Dirac kBT = "
            f"{smearing_T:.6g} Ha "
            f"({_hartree_to_kelvin_temperature(smearing_T):.1f} K)"
        )
    if initial_density_k is not None:
        from .guess import normalize_spin_density_k_guess
        if len(initial_density_k) != 2:
            raise ValueError("READ: initial_density_k must contain alpha and beta block lists")
        D_alpha_k, D_beta_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k, weights, n_alpha, n_beta,
        )
        plog.info("initial guess: READ (complete spin-resolved per-k density)")

    D_alpha_prev = [D.copy() for D in D_alpha_k]
    D_beta_prev = [D.copy() for D in D_beta_k]

    # ---- SCF setup ----------------------------------------------------
    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"run_kuhf_periodic_gdf: damping must be in [0, 1); got {damping}")
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    if guess == InitialGuess.PATOM:
        J_seed = _build_j_from_lpq([a + b for a, b in zip(D_alpha_k, D_beta_k)])
        madelung = _madelung_for_kmesh(system, _mesh_tuple_for_system(system, kmesh))
        spin_orbitals = []
        for densities in (D_alpha_k, D_beta_k):
            K_seed = list(apply_exxdiv_ewald_to_K(
                _build_k_from_lpq(densities), S_k, densities, madelung,
            ))
            coeffs, energies = [], []
            for h, j, k, x in zip(Hcore_k, J_seed, K_seed, X_k):
                c, e = _diag_in_orth_basis(h + j - k, x)
                coeffs.append(c.astype(complex))
                energies.append(e)
            spin_orbitals.append((coeffs, energies))
        (C_alpha_k, eps_alpha_k), (C_beta_k, eps_beta_k) = spin_orbitals
        (D_alpha_k, D_beta_k, occ_alpha_k, occ_beta_k,
         fermi_alpha, fermi_beta, entropy_total) = _occupy_and_density(
            C_alpha_k, eps_alpha_k, C_beta_k, eps_beta_k,
        )
        D_alpha_prev = [d.copy() for d in D_alpha_k]
        D_beta_prev = [d.copy() for d in D_beta_k]

    accel: Optional[MultiKPeriodicUHFAccelerator] = (
        MultiKPeriodicUHFAccelerator(opts) if use_diis else None
    )
    max_iter = int(opts.max_iter)

    scf_trace: List[SCFIteration] = []
    _lindep = PeriodicLinearDependenceSummary(
        n_basis=int(basis.nbasis),
        n_kept_per_k=list(n_kept_k),
        threshold=float(linear_dep_threshold),
        min_overlap_eigenvalue=_s_lo,
        max_overlap_eigenvalue=_s_hi,
        n_aux=int(aux.nbasis),
        n_fit_kept=int(n_fit),
        aux_threshold=float(gdf_linear_dep_threshold),
    )
    result = PeriodicKUHFGDFResult(
                 restart_kpoints=np.asarray(kmesh_bloch.kpoints).copy(), restart_weights=np.asarray(kmesh_bloch.weights).copy(),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0, e_electronic=0.0, e_nuclear=float(e_nuc), n_iter=0,
        converged=False, s_squared=0.0,
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        mo_energies_alpha=[e.copy() for e in eps_alpha_k],
        mo_coeffs_alpha=[C.copy() for C in C_alpha_k],
        density_alpha=[D.copy() for D in D_alpha_k],
        fock_alpha=[np.empty((0, 0), dtype=complex) for _ in range(n_k)],
        mo_energies_beta=[e.copy() for e in eps_beta_k],
        mo_coeffs_beta=[C.copy() for C in C_beta_k],
        density_beta=[D.copy() for D in D_beta_k],
        fock_beta=[np.empty((0, 0), dtype=complex) for _ in range(n_k)],
        overlap=[S.copy() for S in S_k], hcore=[H.copy() for H in Hcore_k],
        kpoints_cart=kpoints_cart.copy(), kpoint_weights=weights.copy(),
        scf_trace=scf_trace, functional=func_name or None,
        aux_basis_name=aux_name, n_aux=int(aux.nbasis),
        rsgdf_ke_cutoff=float(getattr(lpq_cache, 'lr_ke_cutoff', rsgdf_ke_cutoff)),
        linear_dependence=_lindep,
        backend=(
            f"native-multi-k-gdf-{k_exchange}-"
            f"{'uks' if is_ks else 'uhf'}"
        ),
        smearing_temperature=smearing_T,
        fermi_level_alpha=float(fermi_alpha),
        fermi_level_beta=float(fermi_beta),
        entropy=float(entropy_total),
        occupations_alpha=[np.asarray(o, dtype=float) for o in occ_alpha_k],
        occupations_beta=[np.asarray(o, dtype=float) for o in occ_beta_k],
    )
    if private_mdf:
        result.backend += '+private-mdf-unvalidated'
    if dense_core_parity_held:
        result.backend = result.backend + "+PARITY_HELD"
    plog.banner(f"SCF ({label} multi-k, native GDF)")

    E_prev = 0.0
    from .periodic_k_density import _fermi_density_entropy, _needs_fermi_seed_step
    seed_step = max_iter > 0 and smearing_T > 0.0 and (
        _needs_fermi_seed_step(D_alpha_k, S_k, X_k, weights, 1.)
        or _needs_fermi_seed_step(D_beta_k, S_k, X_k, weights, 1.)
    )
    # Reuse the selected seed's full spin Fock once if its atomic/restart
    # density has no Fermi entropy. Iteration 1 then starts from a physical
    # Fermi density; neither the seed nor a different state's entropy is
    # reported as an accepted free-energy iteration.
    for it in range(0 if seed_step else 1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and it >= diis_start_iter
        if it <= 1 or damping == 0.0 or diis_active:
            Da_used = [D.copy() for D in D_alpha_k]
            Db_used = [D.copy() for D in D_beta_k]
        else:
            Da_used = [damping * Dp + (1.0 - damping) * Dn
                       for Dp, Dn in zip(D_alpha_prev, D_alpha_k)]
            Db_used = [damping * Dp + (1.0 - damping) * Dn
                       for Dp, Dn in zip(D_beta_prev, D_beta_k)]
        D_total = [Da_used[i] + Db_used[i] for i in range(n_k)]

        J_k = _build_j_from_lpq(D_total)
        if alpha != 0.0:
            if cosx_bridge is None:
                Ka_k = _build_k_from_lpq(Da_used)
                Kb_k = _build_k_from_lpq(Db_used)
            else:
                Ka_k = cosx_bridge.k_matrices(
                    Da_used, list(kpoints_cart), lr_complement=True,
                    weights=list(weights),
                    screened_omega=screened_omega,
                )
                Kb_k = cosx_bridge.k_matrices(
                    Db_used, list(kpoints_cart), lr_complement=True,
                    weights=list(weights),
                    screened_omega=screened_omega,
                )
            # BvK mesh from the caller's kmesh argument (see the matching
            # note in run_krhf_periodic_gdf: kmesh_bloch.mesh is (1,1,1)
            # for explicit KPoints and silently mis-scales the exxdiv
            # Madelung).
            if screened_omega is None:
                madelung = _madelung_for_kmesh(
                    system, _mesh_tuple_for_system(system, kmesh)
                )
                Ka_k = list(apply_exxdiv_ewald_to_K(Ka_k, S_k, Da_used, madelung))
                Kb_k = list(apply_exxdiv_ewald_to_K(Kb_k, S_k, Db_used, madelung))
        else:
            Ka_k = [np.zeros_like(J_k[0]) for _ in range(n_k)]
            Kb_k = [np.zeros_like(J_k[0]) for _ in range(n_k)]

        Fa_2e = [J_k[i] - alpha * Ka_k[i] for i in range(n_k)]
        Fb_2e = [J_k[i] - alpha * Kb_k[i] for i in range(n_k)]

        E_xc = 0.0
        Va_xc_k = Vb_xc_k = None
        if is_ks:
            # Spin-polarised XC on the full real-space finite-torus density
            # (per-spin inverse Bloch fold), Bloch-folded back to every k --
            # the same convention as the closed-shell multi-k branch
            # (_build_xc_k_from_density). The historical BZ-averaged
            # home-cell shortcut (one k-independent V_xc(Γ) added to every
            # k) was exact only in the vacuum/molecular limit -- the
            # 2026-07-09 KRKS finding's defect class (commit b3f74aa9).
            E_xc, Va_xc_k, Vb_xc_k = _build_xc_k_from_density_uks(
                basis=basis,
                system=system,
                grid=grid,
                func=func,
                density_alpha_k=Da_used,
                density_beta_k=Db_used,
                kmesh_bloch=kmesh_bloch,
                cells=xc_cells,
                kpoints_cart=kpoints_cart,
                lat_opts=lat_opts,
                density_domain=xc_density_domain,
            )

        Fa_k: List[np.ndarray] = []
        Fb_k: List[np.ndarray] = []
        for i in range(n_k):
            Fa = Fa_2e[i] + Hcore_k[i]
            Fb = Fb_2e[i] + Hcore_k[i]
            if Va_xc_k is not None:
                Fa = Fa + Va_xc_k[i]
                Fb = Fb + Vb_xc_k[i]
            Fa_k.append(0.5 * (Fa + Fa.conj().T))
            Fb_k.append(0.5 * (Fb + Fb.conj().T))

        if it == 0:
            C_alpha_k, eps_alpha_k = [], []
            C_beta_k, eps_beta_k = [], []
            for fa, fb, x in zip(Fa_k, Fb_k, X_k):
                ca, ea = _diag_in_orth_basis(fa, x)
                cb, eb = _diag_in_orth_basis(fb, x)
                C_alpha_k.append(ca.astype(complex))
                eps_alpha_k.append(ea)
                C_beta_k.append(cb.astype(complex))
                eps_beta_k.append(eb)
            (D_alpha_k, D_beta_k, occ_alpha_k, occ_beta_k,
             fermi_alpha, fermi_beta, entropy_total) = _occupy_and_density(
                C_alpha_k, eps_alpha_k, C_beta_k, eps_beta_k,
            )
            D_alpha_prev = [d.copy() for d in D_alpha_k]
            D_beta_prev = [d.copy() for d in D_beta_k]
            plog.info("initial density: Fermi refill of the selected seed Fock")
            continue

        E_coulomb = 0.5 * sum(
            float(weights[i]) * float(np.real(np.trace(D_total[i] @ J_k[i])))
            for i in range(n_k)
        )
        E_hf_K = (
            -0.5 * alpha * sum(
                float(weights[i]) * (
                    float(np.real(np.trace(Da_used[i] @ Ka_k[i])))
                    + float(np.real(np.trace(Db_used[i] @ Kb_k[i])))
                )
                for i in range(n_k)
            )
            if alpha != 0.0 else 0.0
        )
        E_elec = 0.0
        for i in range(n_k):
            w = float(weights[i])
            E_elec += w * (
                float(np.real(np.trace(Da_used[i] @ Hcore_k[i])))
                + float(np.real(np.trace(Db_used[i] @ Hcore_k[i])))
                + 0.5 * float(np.real(np.trace(Da_used[i] @ Fa_2e[i])))
                + 0.5 * float(np.real(np.trace(Db_used[i] @ Fb_2e[i])))
            )
        E_elec += E_xc
        E_total = E_elec + float(e_nuc)
        # Mermin free energy A = E - T.(S_a + S_b); SCF converges on A and
        # the trace reports it. At T = 0 entropy_total = 0 => A = E_total,
        # so the no-smearing run is bit-identical to the pre-smearing driver.
        accepted_entropy = (
            _fermi_density_entropy(Da_used, S_k, X_k, weights, 1.)
            + _fermi_density_entropy(Db_used, S_k, X_k, weights, 1.)
            if smearing_T > 0.0 else 0.0
        )
        free_energy = E_total - smearing_T * accepted_entropy

        grad_a: List[np.ndarray] = []
        grad_b: List[np.ndarray] = []
        gnorm2 = 0.0
        for i in range(n_k):
            FDSa = Fa_k[i] @ Da_used[i] @ S_k[i]
            FDSb = Fb_k[i] @ Db_used[i] @ S_k[i]
            ea = FDSa - FDSa.conj().T
            eb = FDSb - FDSb.conj().T
            grad_a.append(ea)
            grad_b.append(eb)
            gnorm2 += float(weights[i]) * (
                float(np.linalg.norm(ea) ** 2) + float(np.linalg.norm(eb) ** 2)
            )
        grad_norm = float(np.sqrt(gnorm2))
        dE = free_energy - E_prev
        converged = (
            it > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        if converged:
            # At either temperature, convergence requires the per-spin
            # occupations that built Da_used/Db_used to be the Fermi
            # filling of the current plain Fock pair's own eigenvalues
            # (frozen-occupation trap on zero-commutator fixtures; see
            # the closed-shell driver's guard above and the
            # zero-commutator floor in periodic_scf_accelerators.py).
            eps_a_chk = [
                _diag_in_orth_basis(Fa_k[i], X_k[i])[1] for i in range(n_k)
            ]
            eps_b_chk = [
                _diag_in_orth_basis(Fb_k[i], X_k[i])[1] for i in range(n_k)
            ]
            a_chk, b_chk = _gdf_open_shell_occupations(
                eps_a_chk,
                eps_b_chk,
                weights=list(weights),
                n_alpha=n_alpha,
                n_beta=n_beta,
                smearing=smear_opts,
            )
            occ_residual = 0.0
            for stored_k, own_k in (
                (occ_alpha_k, a_chk.occupations_per_k),
                (occ_beta_k, b_chk.occupations_per_k),
            ):
                for oo, oc in zip(stored_k, own_k):
                    oo = np.asarray(oo, dtype=float)
                    oc = np.asarray(oc, dtype=float)
                    if oo.size == 0 or oc.size == 0:
                        continue
                    occ_residual = max(
                        occ_residual, float(np.max(np.abs(oo - oc)))
                    )
            if occ_residual > _smeared_occ_tol_fn(
                float(opts.conv_tol_energy)
            ):
                converged = False
        scf_trace.append(SCFIteration(
            iter=it, energy=float(free_energy),
            delta_e=float(dE if it > 1 else 0.0), grad_norm=float(grad_norm),
            diis_subspace=(accel.subspace_size if accel is not None else 0),
        ))
        plog.iteration(
            it, energy=float(free_energy), dE=float(dE if it > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )

        terminating_state = converged or it == max_iter
        if terminating_state:
            # Report canonical eigenpairs of the physical Fa/Fb operators
            # built from the accepted Da_used/Db_used state.  The prior
            # generation's orbitals do not in general diagonalize these
            # matrices, especially on a finite-iteration return.
            C_alpha_k = []
            eps_alpha_k = []
            C_beta_k = []
            eps_beta_k = []
            for i in range(n_k):
                Ca, eai = _diag_in_orth_basis(Fa_k[i], X_k[i])
                Cb, ebi = _diag_in_orth_basis(Fb_k[i], X_k[i])
                C_alpha_k.append(Ca.astype(complex))
                eps_alpha_k.append(eai)
                C_beta_k.append(Cb.astype(complex))
                eps_beta_k.append(ebi)
            (
                _Da_canonical,
                _Db_canonical,
                occ_alpha_k,
                occ_beta_k,
                fermi_alpha,
                fermi_beta,
                entropy_total,
            ) = _occupy_and_density(
                C_alpha_k,
                eps_alpha_k,
                C_beta_k,
                eps_beta_k,
            )

        result.energy = E_total
        result.e_electronic = E_elec
        result.e_xc = E_xc
        result.e_coulomb = E_coulomb
        result.e_hf_exchange = E_hf_K
        result.n_iter = it
        result.mo_energies_alpha = [e.copy() for e in eps_alpha_k]
        result.mo_coeffs_alpha = [C.copy() for C in C_alpha_k]
        result.density_alpha = [D.copy() for D in Da_used]
        result.fock_alpha = [F.copy() for F in Fa_k]
        result.mo_energies_beta = [e.copy() for e in eps_beta_k]
        result.mo_coeffs_beta = [C.copy() for C in C_beta_k]
        result.density_beta = [D.copy() for D in Db_used]
        result.fock_beta = [F.copy() for F in Fb_k]
        result.free_energy = float(free_energy)
        result.entropy = float(accepted_entropy)
        result.fermi_level_alpha = float(fermi_alpha)
        result.fermi_level_beta = float(fermi_beta)
        result.occupations_alpha = [np.asarray(o, dtype=float) for o in occ_alpha_k]
        result.occupations_beta = [np.asarray(o, dtype=float) for o in occ_beta_k]

        if converged:
            result.converged = True
            if cosx_bridge is not None:
                # Post-convergence COSX one-center replacement (same
                # lifecycle as the closed-shell driver): energy stays
                # on the uncorrected iterated surface; the returned
                # per-spin Fock/orbitals are upgraded. Per-spin K
                # carries -alpha into the open-shell Fock.
                dFa = -alpha * cosx_bridge.one_center_correction(
                    Da_used, weights=list(weights)
                )
                dFb = -alpha * cosx_bridge.one_center_correction(
                    Db_used, weights=list(weights)
                )
                fock_a: List[np.ndarray] = []
                fock_b: List[np.ndarray] = []
                mo_ea: List[np.ndarray] = []
                mo_eb: List[np.ndarray] = []
                mo_ca: List[np.ndarray] = []
                mo_cb: List[np.ndarray] = []
                for i in range(n_k):
                    Fa = np.asarray(result.fock_alpha[i]) + dFa
                    Fb = np.asarray(result.fock_beta[i]) + dFb
                    Fa = 0.5 * (Fa + Fa.conj().T)
                    Fb = 0.5 * (Fb + Fb.conj().T)
                    Ca, eai = _diag_in_orth_basis(Fa, X_k[i])
                    Cb, ebi = _diag_in_orth_basis(Fb, X_k[i])
                    fock_a.append(Fa)
                    fock_b.append(Fb)
                    mo_ca.append(Ca.astype(complex))
                    mo_cb.append(Cb.astype(complex))
                    mo_ea.append(eai)
                    mo_eb.append(ebi)
                result.fock_alpha = fock_a
                result.fock_beta = fock_b
                result.mo_energies_alpha = mo_ea
                result.mo_energies_beta = mo_eb
                result.mo_coeffs_alpha = mo_ca
                result.mo_coeffs_beta = mo_cb
                plog.info(
                    "COSX one-center correction applied to the "
                    "returned per-spin Fock/orbitals "
                    "(post-convergence; energy stays on the "
                    "iterated surface)"
                )
            result.s_squared = _multi_k_s_squared(
                n_alpha, n_beta, C_alpha_k, C_beta_k, S_k, weights,
                occ_alpha_k=occ_alpha_k, occ_beta_k=occ_beta_k,
            )
            if compute_gradient:
                # G-PBC-002 Item-4 rung 6: the open-shell multi-k rsgdf
                # analytic gradient on the converged per-spin
                # D_s(k)/C_s(k)/eps_s(k). Entry guards pinned the
                # envelope; the SCF's exact fit parameters and resolved
                # lattice options feed the FD-gated per-spin assemblers.
                from .periodic_gdf_gradient import (
                    _compute_kuhf_gradient_multik,
                    _compute_kuks_gradient_multik,
                )

                grad_cache = _build_multik_gradient_cache_checked(
                    system,
                    basis,
                    aux_modrho,
                    kpoints_cart,
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    lat_opts=lat_opts,
                    gdf_linear_dep_threshold=gdf_linear_dep_threshold,
                    n_fit_scf=n_fit,
                    entry="run_kuhf_periodic_gdf",
                    plog=plog,
                    fit_screen_threshold=float(fit_screen_threshold),
                    source_cache=lpq_cache,
                )
                D_a_g = [np.asarray(D) for D in result.density_alpha]
                D_b_g = [np.asarray(D) for D in result.density_beta]
                S_g = [np.asarray(S) for S in result.overlap]
                # Occupation-1 open-shell energy-weighted density (no
                # closed-shell factor 2 -- the Γ UHF convention). T = 0:
                # W(k) = sum_s C_s,occ(k) diag(eps_s,occ(k)) C_s,occ(k)^H
                # (integer slice, bit-identical). T > 0: the Mermin
                # free-energy weight
                # W(k) = sum_s sum_i f_s,i(k) eps_s,i(k) c c^H with the
                # SCF's per-spin fractional occupations in [0, 1]
                # (result.occupations_alpha/beta); the per-spin D_s(k)
                # are already fractional, so DF-J/K and the per-spin
                # exxdiv D_s S D_s terms flow unchanged. dA/dR of
                # result.free_energy: occupation/mu responses vanish at
                # self-consistency (Mermin 1965; Marzari-Vanderbilt
                # 1999) because the Fermi-Dirac entropy is the exact
                # conjugate of the occupation function, per spin
                # channel and chemical potential mu_s.
                if private_mdf:
                    from .periodic_mdf import _build_mdf_scf_energy_weighted_density

                    W_g = _build_mdf_scf_energy_weighted_density(
                        grad_cache, (D_a_g, D_b_g), (result.fock_alpha, result.fock_beta), S_g,
                        (result.mo_coeffs_alpha, result.mo_coeffs_beta),
                        (result.mo_energies_alpha, result.mo_energies_beta), weights,
                        stationarity_tolerance=float(opts.conv_tol_grad),
                    )
                else:
                    smeared_grad = bool(result.occupations_alpha)
                    W_g = []
                    for ik in range(n_k):
                        W = np.zeros_like(S_g[ik], dtype=np.complex128)
                        for C_s, e_s, n_s, occ_s in (
                            (
                                result.mo_coeffs_alpha[ik],
                                result.mo_energies_alpha[ik],
                                n_alpha,
                                (
                                    result.occupations_alpha[ik]
                                    if smeared_grad
                                    else None
                                ),
                            ),
                            (
                                result.mo_coeffs_beta[ik],
                                result.mo_energies_beta[ik],
                                n_beta,
                                (
                                    result.occupations_beta[ik]
                                    if smeared_grad
                                    else None
                                ),
                            ),
                        ):
                            if n_s == 0:
                                continue
                            if occ_s is not None:
                                occ_arr = np.asarray(occ_s, dtype=float)
                                keep = occ_arr > 1e-14
                                C_keep = np.asarray(C_s)[:, keep]
                                w_i = occ_arr[keep] * np.real(
                                    np.asarray(e_s)[keep]
                                )
                                W = W + (C_keep * w_i[None, :]) @ C_keep.conj().T
                            else:
                                C_occ = np.asarray(C_s)[:, :n_s]
                                eps_occ = np.asarray(e_s)[:n_s]
                                W = W + (
                                    (C_occ * eps_occ[None, :]) @ C_occ.conj().T
                                )
                        W_g.append(W)
                # BvK Madelung xi exactly as the SCF's per-spin exxdiv
                # K-shift resolves it (unused at alpha = 0).
                xi = (
                    _madelung_for_kmesh(
                        system, _mesh_tuple_for_system(system, kmesh)
                    )
                    if alpha != 0.0
                    else 0.0
                )
                if is_ks:
                    result.gradient = _compute_kuks_gradient_multik(
                        system,
                        basis,
                        D_a_g,
                        D_b_g,
                        W_g,
                        S_g,
                        weights,
                        kpoints_cart,
                        grad_cache,
                        functional=func_name,
                        xc_grid=grid,
                        kmesh_bloch=kmesh_bloch,
                        xc_lat_opts=lat_opts,
                        xc_density_cells=xc_cells,
                        xc_density_domain=xc_density_domain,
                        madelung=xi,
                        oneel_lat_opts=oneel_lat_opts,
                        gauge_lat_opts=gauge_lat_opts,
                        ecp_context=_ecp_ctx,
                        ecp_lat_opts=oneel_lat_opts,
                        xc_grid_options=grid_options,
                        xc_use_periodic_becke=bool(
                            getattr(opts, "use_periodic_becke", False)
                        ),
                        xc_becke_image_radius_bohr=float(
                            getattr(opts, "becke_image_radius_bohr", 0.0)
                        ),
                    )
                else:
                    result.gradient = _compute_kuhf_gradient_multik(
                        system,
                        basis,
                        D_a_g,
                        D_b_g,
                        W_g,
                        S_g,
                        weights,
                        kpoints_cart,
                        grad_cache,
                        alpha_hf=alpha,
                        madelung=xi,
                        oneel_lat_opts=oneel_lat_opts,
                        gauge_lat_opts=gauge_lat_opts,
                        ecp_context=_ecp_ctx,
                        ecp_lat_opts=oneel_lat_opts,
                    )
                plog.info(
                    "analytic gradient: multi-k GDF assembly done "
                    f"({'KUKS ' + func_name if is_ks else 'KUHF'}, "
                    f"{n_k} k-point(s))"
                )
            plog.converged(n_iter=it, energy=E_total, converged=True)
            return _finish_density_return(result)

        if terminating_state:
            break

        if accel is not None:
            Fa_ex, Fb_ex = accel.extrapolate_uhf(
                Fa_k, Fb_k,
                error_alpha_k_list=grad_a, error_beta_k_list=grad_b,
                density_alpha_k_list=Da_used, density_beta_k_list=Db_used,
                energy=E_total,
                mo_coeffs_alpha_k_list=C_alpha_k, mo_coeffs_beta_k_list=C_beta_k,
                n_alpha=n_alpha, n_beta=n_beta, weights=list(weights),
                cells=cells, kpoints=list(kpoints_cart),
            )
            if diis_active:
                Fa_k, Fb_k = Fa_ex, Fb_ex

        Ca_new: List[np.ndarray] = []
        ea_new: List[np.ndarray] = []
        Cb_new: List[np.ndarray] = []
        eb_new: List[np.ndarray] = []
        for i in range(n_k):
            Ca, eai = _diag_in_orth_basis(Fa_k[i], X_k[i])
            Cb, ebi = _diag_in_orth_basis(Fb_k[i], X_k[i])
            Ca_new.append(Ca.astype(complex))
            ea_new.append(eai)
            Cb_new.append(Cb.astype(complex))
            eb_new.append(ebi)
        D_alpha_prev = Da_used
        D_beta_prev = Db_used
        C_alpha_k, eps_alpha_k = Ca_new, ea_new
        C_beta_k, eps_beta_k = Cb_new, eb_new
        (D_alpha_k, D_beta_k, occ_alpha_k, occ_beta_k,
         fermi_alpha, fermi_beta, entropy_total) = _occupy_and_density(
            C_alpha_k, eps_alpha_k, C_beta_k, eps_beta_k
        )
        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy

    result.converged = False
    result.s_squared = _multi_k_s_squared(
        n_alpha, n_beta, C_alpha_k, C_beta_k, S_k, weights,
        occ_alpha_k=occ_alpha_k, occ_beta_k=occ_beta_k,
    )
    plog.converged(n_iter=result.n_iter, energy=result.energy, converged=False)
    return _finish_density_return(result)


def run_kuks_periodic_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh] = (1, 1, 1),
    options: Optional[PeriodicKSOptions] = None,
    *,
    functional: str,
    **kwargs,
) -> PeriodicKUHFGDFResult:
    """Open-shell periodic UKS multi-k SCF via native GDF -- thin wrapper
    over :func:`run_kuhf_periodic_gdf` with ``functional`` required
    (``compute_gradient=True`` and every other keyword ride through)."""
    if not functional:
        raise ValueError("run_kuks_periodic_gdf: a functional name is required.")
    return run_kuhf_periodic_gdf(
        system, basis, kmesh, options, functional=functional, **kwargs
    )


def run_krks_periodic_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh] = (1, 1, 1),
    options: Optional[PeriodicKSOptions] = None,
    *,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    gdf_linear_dep_threshold: float = 1e-9,
    apply_modrho: bool = True,
    fock_mixing: Optional[float] = None,
    level_shift_warmup_cycles: Optional[int] = None,
    linear_dep_threshold: float = 1e-7,
    use_compcell: bool = False,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_ft_convention: str = "libint",
    aft_precision: float = 1e-10,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    k_exchange: str = "gdf",
    gdf_method: str = "rsgdf",
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    fit_screen_threshold: float = 0.0,
    mdf_ke_cutoff: float = 40.0,
    ibz_native: bool = False,
    bz_integration: Optional[str] = None,
    density_mixer: Optional[str] = None,
    density_mixer_depth: int = 8,
    density_mixer_beta: float = 0.5,
    density_mixer_kerker: bool = False,
    kerker_k0: float = 1.5,
    kerker_strength: float = 1.0,
    kerker_cutoff_ha: float = 120.0,
    dft_plus_u_sites: Optional[Sequence[object]] = None,
    initial_density_k: Optional[Sequence[np.ndarray]] = None,
    compute_gradient: bool = False,
    return_lattice_density: bool = False,
    lattice_density_memory_bytes: int = 128 * 1024**2,
    check_energy_sanity: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    _lpq_cache_builder=None,
    xc_density_domain: PeriodicXCDensityDomain = (
        PeriodicXCDensityDomain.AUTO
    ),
) -> PeriodicKRKSGDFResult:
    """Run closed-shell periodic KS-DFT multi-k SCF via native GDF.

    Thin wrapper around :func:`run_krhf_periodic_gdf` that asserts a
    functional has been provided. Functional dispatch and exact-
    exchange mixing happen inside the shared SCF loop;
    ``compute_gradient`` follows the shared driver's supported
    envelope and fail-closed guards.
    """
    opts = _options_or_default(options, is_ks=True)
    func = functional or getattr(opts, "functional", None)
    if not func:
        raise ValueError("run_krks_periodic_gdf requires functional=...")
    return run_krhf_periodic_gdf(
        system,
        basis,
        kmesh,
        opts,
        functional=str(func),
        aux_basis=aux_basis,
        aux_drop_eta=aux_drop_eta,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        apply_modrho=apply_modrho,
        fock_mixing=fock_mixing,
        level_shift_warmup_cycles=level_shift_warmup_cycles,
        linear_dep_threshold=linear_dep_threshold,
        use_compcell=use_compcell,
        compcell_eta=compcell_eta,
        apply_aft_correction=apply_aft_correction,
        aft_ft_convention=aft_ft_convention,
        aft_precision=aft_precision,
        rcut_strategy=rcut_strategy,
        rcut_precision=rcut_precision,
        k_exchange=k_exchange,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_omega=rsgdf_omega,
        rsgdf_g_precision=rsgdf_g_precision,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        fit_screen_threshold=fit_screen_threshold,
        mdf_ke_cutoff=mdf_ke_cutoff,
        ibz_native=ibz_native,
        bz_integration=bz_integration,
        density_mixer=density_mixer,
        density_mixer_depth=density_mixer_depth,
        density_mixer_beta=density_mixer_beta,
        density_mixer_kerker=density_mixer_kerker,
        kerker_k0=kerker_k0,
        kerker_strength=kerker_strength,
        kerker_cutoff_ha=kerker_cutoff_ha,
        dft_plus_u_sites=dft_plus_u_sites,
        initial_density_k=initial_density_k,
        compute_gradient=compute_gradient,
        return_lattice_density=return_lattice_density,
        lattice_density_memory_bytes=lattice_density_memory_bytes,
        check_energy_sanity=check_energy_sanity,
        progress=progress,
        verbose=verbose,
        _lpq_cache_builder=_lpq_cache_builder,
        xc_density_domain=xc_density_domain,
    )
