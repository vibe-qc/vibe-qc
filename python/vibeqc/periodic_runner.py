"""High-level "run a periodic job" -- companion to :func:`run_job`.

Mirrors the molecular ``run_job`` API for periodic SCFs:

  - ``output.out``       text log (banner, system, basis, SCF trace,
                         energies, properties)
  - ``output.system``    runtime manifest (CPU, OS, libs, wall-time)
  - ``output.molden``    Γ-point MOs in MOLDEN format (when MOs exist)
  - ``output.xsf``       SCF density on a primitive-cell grid
                         (when ``write_density=True``)

Usage::

    from vibeqc import PeriodicSystem, BasisSet
    from vibeqc.periodic_runner import run_periodic_job

    sys_p = PeriodicSystem(...)
    basis = BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    run_periodic_job(
        sys_p, basis,
        method="RHF",
        output="output",
    )

Status: native GDF covers Gamma and multi-k RHF/ROHF/RKS/UHF/UKS. The
public RIJCOSX route keeps Gamma RHF on the dedicated periodic COSX
driver and routes true multi-k RHF/RKS/UHF/UKS through the native GDF
loop with ``k_exchange="cosx"``. The earlier PySCF-backed GDF spike is
retired because PySCF and CRYSTAL are external reference programs, not
in-process vibe-qc backends. Explicit ``jk_method="fft_poisson"`` is
retired as a public route; ``AUTO`` resolves to native GDF for
closed-shell jobs and BIPOLE for open-shell jobs.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import math
import os
import time
import warnings
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .spin_channels import is_open_shell_result, spin_densities

if TYPE_CHECKING:
    from ._vibeqc_core import BlochKMesh
    from .bands import BandStructure
    from .kpoints import KPoints

from .lattice_convention import cell_parameters, nearest_neighbour_distance
from ._initial_guess import coerce_initial_guess
from .guess import resolve_initial_guess

from ._vibeqc_core import (
    Atom,
    BasisSet,
    Crystal,
    D3BJParams,
    Functional,
    GridOptions,
    InitialGuess,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    PeriodicXCDensityDomain,
    SpaceGroup,
    SpinlockMode,
    attach_symmetry,
    direct_lattice_cells,
    get_num_threads,
    to_primitive,
)
from .banner import VIBEQC_VERSION, banner, enforce_runtime_pin_from_env, library_versions
from .occupations import (
    hartree_to_kelvin_temperature,
    resolve_smearing_temperature,
)
from .output import (
    Column,
    HeaderlessBlock,
    HeaderlessColumn,
    Level,
    ManifestUpdater,
    OutputChannel,
    OutputPlan,
    OutputWriter,
    Quantity,
    Table,
    active_policy,
    dry_run_manifest,
    flush,
    install_diagnostics_bridge,
    is_dry_run_estimate_requested,
    is_dry_run_requested,
    render_energy,
    render_energy_labeled,
    render_frequency,
    section_header,
    print_dry_run_summary,
    warn,
    write,
)
from .output._cpp_diagnostics import (
    _progress_handler as _CPP_PROGRESS_HANDLER_SLOT,
    install_progress_handler,
)
from .output.formats.scf_log import write_scf_trace
from .output._errors import (
    OutputFailureKind,
    warn_output_failure,
    warn_writer_failure,
)
from .output.citations import (
    citation_manifest_rows,
    format_bibtex,
    format_references_block,
    load_default_database,
    write_references_block,
)
from .output.formats.perf import _format_timing_summary
from .output.plan import _resolve_sidecar_request
from .structured_log import (
    StructuredLog,
    run_fingerprint,
)
from .structured_log import (
    structured_log as _structured_log_ctx,
)
from .pbc_gdf import (
    _reject_legacy_gamma_gdf,
    run_pbc_gdf_rhf,
    run_pbc_gdf_uhf,
    run_pbc_gdf_uks,
)
from .periodic.chi.scf import (
    _density_blocks_per_k,
    _require_supported_external_xc as _require_aiccm2026dev_b_external_xc,
    _resolve_backend as _resolve_aiccm2026dev_b_backend,
    _resolve_fitted_rsgdf_tail,
    _spin_density_blocks_per_k,
    cyclic_lattice_extension,
    inverse_bloch_transform,
    run_aiccm2026dev_b_rhf,
    run_aiccm2026dev_b_rks,
    run_aiccm2026dev_b_uhf,
    run_aiccm2026dev_b_uks,
)
from .periodic_convergence_auto import (
    ConvergenceStrategy,
    KnobResolution,
    classify_periodic_system,
    insulator_smearing_warning,
    resolve_convergence_strategy,
)
from .periodic_jk_method import (
    _AICCM_JK_METHODS,
    _warn_legacy_aiccm_spelling,
    AICCM_VARIANT_OF,
    AICCM_VARIANT_ROUTES,
    PeriodicJKMethod,
    describe_jk_method,
    jk_method_agrees_with_variant,
    pick_jk_method,
    resolve_aiccm_correlation,
    resolve_aiccm_variant,
    validate_jk_method,
)
from .periodic_k_gdf import (
    _canonical_gdf_density_mixer,
    _expand_ibz_kmesh_to_full_bz,
    _gamma_kmesh_info,
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
    run_kuhf_periodic_gdf,
    run_kuks_periodic_gdf,
)
from .periodic_rohf_gdf import run_krohf_periodic_gdf
from .periodic_screened_exchange import (
    reject_periodic_gdf_unsupported_functional,
)
from .output._stem_paths import stem_sibling


_PERIODIC_OUTPUT_WRITER: ContextVar[OutputWriter | None] = ContextVar(
    "vibeqc_periodic_output_writer",
    default=None,
)


def _periodic_output_lifecycle(func):
    """Finish or crash the job-scoped writer at the public API boundary."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        writer_token = _PERIODIC_OUTPUT_WRITER.set(None)
        progress_token = _CPP_PROGRESS_HANDLER_SLOT.set(None)
        try:
            result = func(*args, **kwargs)
        except Exception:
            writer = _PERIODIC_OUTPUT_WRITER.get()
            if writer is not None and writer.status == "running":
                writer.crash()
            raise
        else:
            writer = _PERIODIC_OUTPUT_WRITER.get()
            if writer is not None and writer.status == "running":
                writer.finish()
            return result
        finally:
            try:
                _CPP_PROGRESS_HANDLER_SLOT.reset(progress_token)
            finally:
                _PERIODIC_OUTPUT_WRITER.reset(writer_token)

    return wrapped


from .periodic_rhf_gdf import (
    PeriodicRHFGDFResult,
    run_rhf_periodic_gamma_gdf,
)
from .progress import ProgressLogger, resolve_progress
from .smearing import SmearingOptions

# Sentinel distinguishing "smearing_temperature not given" (auto strategy
# may fill it) from the documented explicit values 0.0 / None (both mean
# "smearing off, by user choice").
_SMEARING_UNSET = object()

# Default-on finite-temperature smearing for periodic GFN2-xTB (k_B T in
# Hartree, about 316 K). A Gamma frontier of a metallic cell can cross on a
# lattice sweep and flip the hard-Aufbau occupations between SCC branches;
# the small default width keeps the SCC on one branch. Explicit
# smearing_temperature=0 restores exact zero-temperature Aufbau. Mirrors
# kPeriodicGFN2DefaultElectronicTemperature in the C++ periodic drivers.
_GFN2_PERIODIC_DEFAULT_SMEARING_HA = 0.001

# Sentinel distinguishing "bz_integration not given" from explicit None
# (which means "use smearing").
_BZ_INTEGRATION_UNSET = object()

_BOHR_TO_ANGSTROM = 0.529177210903

__all__ = ["run_periodic_job"]


def _negotiate_external_xc_grid_profile(functional: Functional):
    """Resolve an external provider's required atom-grid profile.

    The high-level runner owns its :class:`PeriodicKSOptions`, so a provider
    capability is a selection request rather than a caller-option mismatch.
    Canonicalize it through :class:`GridOptions` here, before dry-run can
    certify the request and before any route constructs its atom grid.
    """

    if not bool(getattr(functional, "is_external", False)):
        return None
    capabilities = getattr(functional, "external_capabilities", None)
    if not isinstance(capabilities, dict):
        raise RuntimeError(
            "run_periodic_job: external-XC capability metadata is unavailable "
            f"for provider {functional.name!r}"
        )
    required = str(capabilities.get("required_grid_profile", "") or "").strip()
    if not required:
        return None
    required_options = GridOptions()
    required_options.atomic_grid_profile = required
    return required_options.atomic_grid_profile


def _require_external_xc_zero_temperature_request(
    system: PeriodicSystem,
    kpoints,
    *,
    smearing: Optional[SmearingOptions],
    smearing_temperature,
    smearing_unit: str,
    smearing_method: str,
    smearing_metallic: Optional[bool],
    smearing_band_gap_hartree: Optional[float],
) -> None:
    """Reject an explicitly requested positive temperature before dry-run.

    Automatic convergence can still propose smearing after this boundary; a
    capability filter below pins that automatic knob to zero for external XC.
    This helper covers every user and ``KPoints`` metadata request so a dry
    run cannot certify a calculation that the direct driver will refuse.
    """

    requested = smearing
    engaged = (
        requested is not None
        or smearing_temperature is not _SMEARING_UNSET
        or smearing_metallic is not None
        or smearing_band_gap_hartree is not None
    )
    if not engaged:
        requested = getattr(kpoints, "smearing", None)
        engaged = requested is not None
    temperature_arg = (
        0.0
        if smearing_temperature is _SMEARING_UNSET
        else smearing_temperature
    )
    if requested is not None:
        if temperature_arg not in (0.0, None):
            raise ValueError(
                "run_periodic_job: pass either smearing= or "
                "smearing_temperature=, not both"
            )
        if not isinstance(requested, SmearingOptions):
            raise TypeError(
                "run_periodic_job: smearing must be a SmearingOptions instance"
            )
        temperature = float(requested.temperature)
    elif engaged:
        resolution = resolve_smearing_temperature(
            temperature_arg,
            unit=smearing_unit,
            method=smearing_method,
            metallic=smearing_metallic,
            band_gap_hartree=smearing_band_gap_hartree,
            n_electrons=system.n_electrons(),
        )
        temperature = float(resolution.temperature)
    else:
        temperature = 0.0

    from .periodic_external_xc import _require_zero_temperature_external_xc

    _require_zero_temperature_external_xc(
        temperature,
        where="run_periodic_job",
    )


def _periodic_skala_memory_estimate(system: PeriodicSystem):
    """Return the parameter-only SKALA callback/model peak for one cell.

    The high-level periodic route always selects SKALA's element-specific
    atomic-grid profile, so this calculation asks the core for its exact raw
    atom-block sizes without building coordinates, AOs, importing PyTorch, or
    fetching the checkpoint.
    """

    from ._vibeqc_core import grid_atomic_point_counts
    from .memory import estimate_skala_xc_memory
    from .runner import _apply_grid_level

    grid = PeriodicKSOptions().grid
    _apply_grid_level(grid, "skala")
    atomic_grid_sizes = grid_atomic_point_counts(
        system.unit_cell_molecule(),
        grid,
    )
    return estimate_skala_xc_memory(
        atomic_grid_sizes=atomic_grid_sizes,
    )


def _merge_periodic_skala_memory(base, system: PeriodicSystem):
    """Add the SKALA phase to an existing route estimate, or stand alone.

    Most periodic route estimates are single-phase category sums today. Keep
    the merge correct for a phase-aware estimate too: ``raw_total_bytes``
    reads only ``phase_peaks`` when that mapping is populated, so the SKALA
    workspace must be reflected there as well as in the display categories.
    """

    skala = _periodic_skala_memory_estimate(system)
    if base is None:
        return skala
    label = "SKALA full-grid + model workspace"
    workspace = int(skala.by_category[label])
    previous = int(base.by_category.get(label, 0))
    base.by_category[label] = workspace
    delta = workspace - previous
    if delta and base.phase_peaks:
        for phase in tuple(base.phase_peaks):
            base.phase_peaks[phase] += delta
    return base


def _periodic_xc_gradient_dry_run_estimate_bytes(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    method_upper: str,
    functional: str | None,
    lattice_cutoff_bohr: float | None = None,
) -> int | None:
    """Best-effort dry-run estimate for periodic RKS/UKS gradients."""
    if method_upper not in ("RKS", "UKS"):
        return None

    from .memory import estimate_periodic_xc_gradient

    opts = PeriodicKSOptions()
    grid = opts.grid
    n_atoms = len(system.unit_cell)
    from .memory import _grid_points
    n_grid_points = _grid_points(system.unit_cell_molecule(), grid)
    # RKS/UKS Ewald paths may enlarge the density cutoff to 18 bohr before
    # calling the analytic gradient. Charging at least that many cells keeps
    # vq placement from seeing the cheaper default 15-bohr stencil.
    cutoff_base = (
        float(lattice_cutoff_bohr)
        if lattice_cutoff_bohr is not None
        else float(getattr(opts.lattice_opts, "cutoff_bohr", 15.0))
    )
    cutoff = max(cutoff_base, 18.0)
    n_cells = len(direct_lattice_cells(system, cutoff))
    spin = 2 if method_upper == "UKS" else 1
    func = Functional(str(functional), spin)
    kind = getattr(getattr(func, "kind", None), "name", None)
    if kind is None:
        kind = str(getattr(func, "kind", ""))

    return estimate_periodic_xc_gradient(
        n_basis=int(basis.nbasis),
        n_atoms=n_atoms,
        n_grid_points=n_grid_points,
        n_cells=n_cells,
        functional_kind=str(kind),
        open_shell=method_upper == "UKS",
    ).total_bytes


def _basis_primitive_count(basis: BasisSet) -> int | None:
    try:
        return int(sum(len(sh.exponents) for sh in basis.shells()))
    except Exception:
        return None


def _periodic_gpw_gapw_estimate(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    resolved_jk: PeriodicJKMethod,
    method_upper: str,
    functional: str | None,
    cutoff_ha: float,
    kpoints: object,
):
    """Best-effort estimate for GPW/GAPW FFT-grid SCFs."""
    if resolved_jk not in (PeriodicJKMethod.GPW, PeriodicJKMethod.GAPW):
        return None

    from .memory import estimate_periodic_gpw_gapw
    from .periodic_gapw_grid import nx_for_axis

    lattice = np.asarray(system.lattice, dtype=float)
    n_grid_points = 1
    for axis_length in np.linalg.norm(lattice, axis=0):
        n_grid_points *= nx_for_axis(float(axis_length), float(cutoff_ha))

    functional_kind: str | None = None
    if method_upper in ("ROKS", "RKS", "UKS") and functional is not None:
        func = Functional(
            str(functional),
            2 if method_upper in ("ROKS", "UKS") else 1,
        )
        kind = getattr(getattr(func, "kind", None), "name", None)
        functional_kind = str(kind if kind is not None else getattr(func, "kind", ""))

    try:
        n_kpoints = _bloch_kmesh_size(_runner_bloch_kmesh(system, kpoints))
    except Exception:
        n_kpoints = 1

    compact_multik = False
    if n_kpoints > 1:
        try:
            from .periodic_gapw_j import _multik_gpw_is_molecular_limit

            compact_multik = not _multik_gpw_is_molecular_limit(system)
        except Exception:
            compact_multik = False

    n_soft_basis: int | None = None
    augmentation_active: bool | None = False
    if resolved_jk == PeriodicJKMethod.GAPW:
        n_soft_basis = int(basis.nbasis)
        augmentation_active = True
        try:
            from .periodic_gapw_augment import softened_basis

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                soft_basis = softened_basis(basis, system)
            n_soft_basis = int(soft_basis.nbasis)
            full_prim = _basis_primitive_count(basis)
            soft_prim = _basis_primitive_count(soft_basis)
            if full_prim is not None and soft_prim is not None:
                augmentation_active = soft_prim != full_prim
        except Exception:
            augmentation_active = True

    return estimate_periodic_gpw_gapw(
        n_basis=int(basis.nbasis),
        n_grid_points=n_grid_points,
        route=resolved_jk.value,
        functional_kind=functional_kind,
        open_shell=method_upper in ("ROHF", "ROKS", "UHF", "UKS"),
        n_kpoints=n_kpoints,
        compact_multik=compact_multik,
        n_soft_basis=n_soft_basis,
        n_atoms=len(system.unit_cell),
        augmentation_active=augmentation_active,
        analytic_eri_one_centre=(
            resolved_jk == PeriodicJKMethod.GAPW
            and method_upper in ("RHF", "UHF")
        ),
    )


def _periodic_gpw_gapw_dry_run_estimate_bytes(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    resolved_jk: PeriodicJKMethod,
    method_upper: str,
    functional: str | None,
    cutoff_ha: float,
    kpoints: object,
) -> int | None:
    """Best-effort dry-run estimate for GPW/GAPW FFT-grid SCFs."""
    est = _periodic_gpw_gapw_estimate(
        system,
        basis,
        resolved_jk=resolved_jk,
        method_upper=method_upper,
        functional=functional,
        cutoff_ha=cutoff_ha,
        kpoints=kpoints,
    )
    return None if est is None else est.total_bytes


def _periodic_functional_needs_exchange(
    functional: str | None,
    *,
    open_shell: bool,
) -> bool:
    if functional is None:
        return False
    try:
        func = Functional(str(functional), 2 if open_shell else 1)
    except Exception:
        return False
    return bool(getattr(func, "is_hybrid", False)) or bool(
        getattr(func, "is_range_separated", False)
    )


def _periodic_gdf_aux_basis_size(
    system: PeriodicSystem,
    basis: BasisSet,
    aux_basis: str | None,
) -> int:
    """The auxiliary dimension the GDF drivers will actually allocate.

    ``aux_basis=None`` is the runner's ``<auto>``: every GDF driver resolves
    it as ``aux_basis or default_aux_for(basis.name)`` before building the
    cderi, so the preflight must size the same set. It used to fall back to
    ``3 x n_ao`` for ``None`` -- 396 against the 960 def2-svp-jkfit functions
    of the P05 NaCl conventional cell, which printed 452 GB for a 1096 GB
    dense Lpq cache (issue #92). The ``3 x n_ao`` floor remains only for a
    basis with no registered default (``default_aux_for`` raises there, and
    the driver would too) or an aux that cannot be built.
    """
    from .aux_basis import default_aux_for, make_aux_basis_set

    aux_name = aux_basis
    if not aux_name:
        try:
            aux_name = default_aux_for(basis.name)
        except Exception:
            aux_name = None
    if aux_name:
        try:
            return int(
                make_aux_basis_set(
                    system.unit_cell_molecule(),
                    aux_name=aux_name,
                ).nbasis
            )
        except Exception:
            pass
    return max(1, 3 * int(basis.nbasis))


def _periodic_gdf_estimate(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    resolved_jk: PeriodicJKMethod,
    method_upper: str,
    functional: str | None,
    kpoints: object,
    aux_basis: str | None,
    rsgdf_ke_cutoff: float = 200.0,
    n_ibz_kpoints: int | None = None,
    diis_subspace_size: int = 8,
    scf_options=None,
    compute_gradient: bool = False,
):
    if resolved_jk not in (PeriodicJKMethod.GDF, PeriodicJKMethod.RIJCOSX):
        return None

    from .memory import estimate_periodic_multik_gdf

    try:
        n_kpoints = _bloch_kmesh_size(_runner_bloch_kmesh(system, kpoints))
    except Exception:
        n_kpoints = 1
    n_aux = _periodic_gdf_aux_basis_size(system, basis, aux_basis)
    open_shell = method_upper in ("ROHF", "ROKS", "UHF", "UKS")
    need_k_pairs = method_upper in ("RHF", "ROHF", "UHF")
    if method_upper in ("RKS", "UKS"):
        need_k_pairs = _periodic_functional_needs_exchange(
            functional,
            open_shell=open_shell,
        )
    if resolved_jk == PeriodicJKMethod.RIJCOSX:
        need_k_pairs = True
    label_parts = [method_upper]
    if functional:
        label_parts.append(str(functional))
    label_parts.append(resolved_jk.value)
    route_label = " ".join(label_parts)
    # The space-group-reduced exchange build stores n_IBZ x n_k Lpq blocks,
    # so charging it n_k^2 would abort a run that fits (measured on NaCl
    # primitive / def2-SVP / def2-svp-jk at (4,4,4): the Lpq term is
    # 15.95 GB unreduced vs 1.99 GB at n_IBZ = 8) and would print an
    # estimate contradicting the cache actually allocated.
    # The wedge never reduces the diagonal-only cache.
    n_ibz = (
        int(n_ibz_kpoints)
        if (n_ibz_kpoints is not None and need_k_pairs)
        else None
    )
    estimate = estimate_periodic_multik_gdf(
        n_basis=int(basis.nbasis),
        n_aux=n_aux,
        n_kpoints=n_kpoints,
        need_k_pairs=need_k_pairs,
        open_shell=open_shell,
        n_ibz_kpoints=n_ibz,
        diis_subspace_size=diis_subspace_size,
    )
    # Count a reciprocal index box without allocating a mesh. In 3D the
    # source stores only G vectors and bounded native panels, never Nao^2*G.
    # The q=0 cutoff plan also bounds every q's reciprocal tail; q affects
    # only the real-space pair-image bound in this planner.
    import math
    from .aux_basis import (
        _plan_range_separated_gdf_cutoffs, _reciprocal_index_box_per_axis,
        default_aux_for, make_aux_basis_set,
    )
    ke = float(rsgdf_ke_cutoff)
    if not np.isfinite(ke) or ke <= 0:
        raise ValueError('periodic GDF reciprocal cutoff must be finite and positive')
    if int(system.dim) == 3:
        auxiliary = make_aux_basis_set(
            system.unit_cell_molecule(), aux_name=aux_basis or default_aux_for(basis.name),
        )
        plan = _plan_range_separated_gdf_cutoffs(
            system, basis, auxiliary, np.zeros(3), omega=.4, raw_integral_error=1e-10,
        )
        ke = max(ke, plan.ke_cutoff)
    reach = np.sqrt(2.0) * np.sqrt(ke)
    limits = _reciprocal_index_box_per_axis(system.lattice, reach, int(system.dim))
    # One additional reciprocal index covers canonical half-cell q shifts
    # and the enumerator's floating-point boundary slack.
    n_g_upper = math.prod(2*(int(n)+1)+1 if axis < int(system.dim) else 1
                          for axis, n in enumerate(limits))
    if int(system.dim) == 3:
        from .memory import estimate_periodic_gdf_source
        from ._vibeqc_core import get_num_threads, gdf_short_range_workspace_bytes
        source = estimate_periodic_gdf_source(
            n_basis=int(basis.nbasis), n_aux=n_aux, n_kpoints=n_kpoints,
            n_reciprocal_upper=n_g_upper,
            native_workspace_bytes=max(64 * 1024**2, gdf_short_range_workspace_bytes(
                basis, auxiliary, get_num_threads(),
            )),
        )
        estimate.by_category.update(source.by_category)
        # Charge the same AO-domain setup and Fourier workspace that the
        # runtime admits before constructing one-electron lattice blocks.
        from .periodic_k_gdf import _oneel_lattice_opts, _gdf_oneel_memory_estimate
        oneel_options = scf_options if scf_options is not None else PeriodicRHFOptions()
        oneel = _oneel_lattice_opts(
            system, basis, oneel_options.lattice_opts, rcut_strategy='pyscf_auto',
            k_points_cart=np.zeros((1, 3)),
        )
        oneel_estimate = _gdf_oneel_memory_estimate(
            system, basis, oneel, n_kpoints=n_kpoints,
            ecp_active=bool(getattr(oneel_options, 'ecp_primitive_blocks', [])),
        )
        estimate.by_category.update(oneel_estimate.by_category)
        if method_upper in ("RKS", "UKS"):
            from .memory import (
                _grid_points, estimate_periodic_xc_value, estimate_periodic_xc_gradient,
            )
            from ._vibeqc_core import periodic_xc_domain_counts
            from .periodic_k_gdf import _xc_density_cells_for_domain

            ks_options = scf_options if scf_options is not None else PeriodicKSOptions()
            xc_options = ks_options.lattice_opts
            xc_functional = Functional(str(functional), 2 if open_shell else 1)
            domain = (PeriodicXCDensityDomain.PERIODIC_LATTICE
                      if bool(getattr(xc_functional, 'is_external', False))
                      else PeriodicXCDensityDomain.AUTO)
            xc_cells = _xc_density_cells_for_domain(
                system, xc_options, direct_lattice_cells(system, xc_options.cutoff_bohr), domain,
            )
            active, bra = periodic_xc_domain_counts(system, xc_cells, xc_options, domain)
            xc_args = dict(
                n_basis=int(basis.nbasis), n_atoms=len(system.unit_cell),
                n_grid_points=_grid_points(system.unit_cell_molecule(), ks_options),
                n_cells=len(xc_cells), n_active_cells=active, n_bra_cells=bra,
                functional_kind=str(xc_functional.kind), open_shell=open_shell,
            )
            xc_estimate = estimate_periodic_xc_value(**xc_args)
            xc_bytes = sum(v for k, v in xc_estimate.by_category.items()
                           if k != 'Python runtime + NumPy overhead')
            if compute_gradient and not bool(getattr(xc_functional, 'is_external', False)):
                xc_gradient = estimate_periodic_xc_gradient(**xc_args)
                xc_bytes = max(xc_bytes, sum(v for k, v in xc_gradient.by_category.items()
                                           if k != 'Python runtime + NumPy overhead'))
            estimate.by_category['Periodic-XC retained data and phase workspace'] = xc_bytes
    else:
        estimate.by_category['GDF dense AO-pair FT bundle'] = int(basis.nbasis)**2*n_g_upper*16
    return SimpleNamespace(
        estimate=estimate,
        n_kpoints=n_kpoints,
        n_ibz_kpoints=n_ibz,
        route_label=route_label,
    )


# ============================================================
# Helpers -- symmetry reduction
# ============================================================


def _ibz_kpoint_count(system, kpoints) -> "int | None":
    """Size of the irreducible wedge of the resolved Monkhorst-Pack mesh.

    This is the number the space-group-reduced GDF exchange build
    (``ibz_native``) actually uses as its **bra** row count, so it is also
    what the memory preflight must charge: the reduced cache holds
    ``n_IBZ x n_k`` ``Lpq`` blocks, not ``n_k^2``
    (:func:`vibeqc.periodic_k_gdf._build_k_ibz_native`).

    Returns ``None`` -- never a guess -- when the wedge cannot be resolved
    (no attached symmetry, or a ``kpoints=`` spec that is not a
    reducible mesh). The caller decides whether that is a hard error
    (``symmetry_reduce_k=True`` was asked for) or simply "no reduction to
    report"; the driver's own :func:`_resolve_ibz_native_state` remains the
    authority on whether the wedge is usable at all.
    """
    if getattr(system, "symmetry", None) is None:
        return None
    try:
        from ._vibeqc_core import monkhorst_pack as _mp

        kmesh = _runner_bloch_kmesh(system, kpoints)
        mesh = getattr(kmesh, "mesh", None)
        if mesh is None and isinstance(kpoints, (list, tuple)):
            mesh = tuple(int(x) for x in kpoints)
        if mesh is None or len(tuple(mesh)) != 3:
            return None
        n_ibz = len(_mp(system, [int(x) for x in mesh], [0, 0, 0], True).kpoints)
    except Exception:
        return None
    return int(n_ibz) if n_ibz > 0 else None


# ============================================================
# Helpers -- ECP auto-attach
# ============================================================

_POB_SOURCE_DIR_BY_BASIS = {
    "pob-tzvp": "pob-TZVP",
    "pob-tzvp-rev2": "pob-TZVP-rev2",
    "pob-dzvp-rev2": "pob-DZVP-rev2",
}


#: Convention labels reported for a resolved Brillouin-zone mesh.
#: Which one applies is *measured* from the mesh, never assumed -- vibe-qc
#: has two, see :func:`kmesh_convention_of`.
KMESH_CONVENTION_GAMMA = "gamma-centred"
KMESH_CONVENTION_SHIFTED = "shifted (classical Monkhorst-Pack)"
KMESH_CONVENTION_UNKNOWN = "unresolved"

#: Backwards-compatible alias for the gamma-centred label.
KMESH_CONVENTION = KMESH_CONVENTION_GAMMA


def kmesh_convention_of(frac) -> str:
    """Name the convention of an already-resolved fractional k-list.

    Determined by whether Gamma is in the sampled set, because vibe-qc has
    **two** conventions reachable through the same ``kpoints=`` argument:

    * :func:`vibeqc.monkhorst_pack` (and the mesh-tuple form of
      ``run_periodic_job``) is always gamma-centred, ``is_shift=(0,0,0)``.
    * :meth:`vibeqc.KPoints.monkhorst_pack` applies the *classical*
      auto-shift, so it is gamma-centred for odd N and shifted by half a
      step for even N -- i.e. it matches ASE/GPAW where the tuple form
      does not.

    Measured 2026-08-02 on a cubic cell: for (2,2,2) and (4,4,4)
    ``vibeqc.monkhorst_pack`` contains Gamma and ``KPoints.monkhorst_pack``
    does not (``is_shift=(1,1,1)``); at (3,3,3) both contain it. So the
    convention cannot be inferred from the mesh tuple alone, and hardcoding
    one label would make the ``.out`` confidently wrong for the other API.
    """
    try:
        arr = np.asarray(frac, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return KMESH_CONVENTION_UNKNOWN
        if bool(np.any(np.all(np.abs(arr) < 1e-10, axis=1))):
            return KMESH_CONVENTION_GAMMA
        return KMESH_CONVENTION_SHIFTED
    except Exception:
        return KMESH_CONVENTION_UNKNOWN


def _kmesh_fractional(system, kpoints_cart) -> "np.ndarray | None":
    """Fractional coordinates of Cartesian k-points, wrapped to [-1/2, 1/2).

    ``k_cart = frac @ B`` with ``B = 2 pi (A^-1)^T`` (rows = reciprocal
    vectors), so ``frac = k_cart @ B^-1``. Returns None if the lattice is
    unusable, since this feeds provenance output that must never be the
    reason a job fails.
    """
    try:
        a_mat = np.asarray(system.lattice, dtype=float)
        b_mat = 2.0 * np.pi * np.linalg.inv(a_mat).T
        frac = np.asarray(kpoints_cart, dtype=float) @ np.linalg.inv(b_mat)
        return (frac + 0.5) % 1.0 - 0.5
    except Exception:
        return None


def write_kmesh_convention(convention: str = KMESH_CONVENTION_UNKNOWN) -> None:
    """Write the Brillouin-zone convention statement for a resolved mesh.

    Split out of :func:`write_kmesh_line` so routes that report a k-point
    *count* rather than a mesh tuple (the full-k semiempirical route) can
    state the same thing without duplicating the wording.

    ``convention`` comes from :func:`kmesh_convention_of`, i.e. it is
    measured from the k-list rather than assumed. When it could not be
    resolved, say so instead of guessing -- a confidently wrong convention
    line is worse than none, since the whole point is to make cross-code
    comparison trustworthy.
    """
    write(f"    k-mesh convention  = {convention}\n")
    if convention == KMESH_CONVENTION_GAMMA:
        write(
            "                         Gamma is sampled; ASE/GPAW's mesh of\n"
            "                         the same name is disjoint for even N\n"
        )
    elif convention == KMESH_CONVENTION_SHIFTED:
        write(
            "                         Gamma is NOT sampled; this matches\n"
            "                         ASE/GPAW, not the mesh-tuple form\n"
        )
    else:
        write(
            "                         k-list unavailable here; confirm the\n"
            "                         sampling before any cross-code claim\n"
        )


def write_kmesh_line(
    mesh,
    *,
    suffix: str = "",
    system=None,
    kpoints_cart=None,
    label: str = "kpoints",
) -> None:
    """Write the k-mesh line together with the convention that defines it.

    KPOINT-CONVENTION-UNPRINTED: ``kpoints=(N, N, N)`` does **not** name the
    same point set in every code, and printing the tuple alone lets a
    cross-code comparison silently compare different Brillouin-zone
    samplings. The mesh-tuple form builds the **Gamma-centred** mesh
    ``{0, 1/N, ..., (N-1)/N}`` along each axis; ASE/GPAW's
    ``monkhorst_pack`` builds the **classical shifted** mesh, offset by half
    a step.

    The convention printed here is **measured from the resolved k-list**
    (:func:`kmesh_convention_of`), not inferred from ``mesh``, because
    ``kpoints=`` also accepts a :class:`vibeqc.KPoints` built with the
    classical auto-shift -- which is *shifted* at even N and so lands on
    the opposite convention from the identical-looking tuple. Assuming the
    label would make the ``.out`` confidently wrong for that caller.

    Measured 2026-08-02, cubic cell, fractional coordinates wrapped to
    [-1/2, 1/2), vibe-qc ``monkhorst_pack`` vs ``ase.dft.kpoints
    .monkhorst_pack``::

        mesh       n    shared
        (2,2,2)    8      0
        (3,3,3)   27     27      <- identical
        (4,4,4)   64      0
        (5,5,5)  125    125      <- identical
        (6,6,6)  216      0

    The two conventions therefore diverge for **even** N and agree exactly
    for **odd** N: for odd N the half-step offset wraps onto the same
    lattice, for even N it lands exactly between vibe-qc's points. That is
    what makes this a trap rather than a nuisance -- a cross-code check
    validated at (3,3,3) passes and gives false confidence in a (4,4,4)
    production run.

    Emits the mesh and convention at STANDARD level, and the resolved
    k-point list at VERBOSE, so the sampling is reconstructable from the
    ``.out`` alone.
    """
    write(f"    {label}            = {mesh}{suffix}\n")

    frac = None
    if system is not None:
        if kpoints_cart is None:
            # Resolve the mesh spec ourselves so every route gets the
            # k-list, not only the ones already holding a built mesh.
            try:
                kpoints_cart = _runner_bloch_kmesh(system, mesh).kpoints
            except Exception:
                kpoints_cart = None
        if kpoints_cart is not None:
            frac = _kmesh_fractional(system, kpoints_cart)
            if frac is not None and frac.ndim != 2:
                frac = None

    write_kmesh_convention(kmesh_convention_of(frac))
    if frac is None:
        return
    write(f"    k-points resolved  = {frac.shape[0]}\n")
    write("    k-points (fractional, wrapped to [-1/2, 1/2)):\n", Level.VERBOSE)
    for i, row in enumerate(frac):
        write(
            f"      {i:5d}  {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}\n",
            Level.VERBOSE,
        )


def _system_atomic_numbers(system) -> set[int]:
    return {int(atom.Z) for atom in system.unit_cell}


def _format_atomic_numbers(zs: Sequence[int]) -> str:
    try:
        from .basis_crystal import _ELEMENT_SYMBOLS as _symbols
    except Exception:
        _symbols = []

    labels: list[str] = []
    for z in sorted(int(z) for z in zs):
        sym = _symbols[z] if 0 <= z < len(_symbols) else f"Z={z}"
        labels.append(f"{sym}(Z={z})")
    return ", ".join(labels)


def _bundled_pob_source_atoms(name: str) -> list:
    """Return bundled per-element POB source records, if available.

    The checked-in source files cover all-electron POB elements (for example
    Ni/O in P16) and let runtime ECP resolution avoid any network access. The
    heavier POB ECP records are not bundled as source files in this tree; those
    still require the legacy Bredow fetcher and therefore fail closed below if
    they cannot be resolved.
    """
    source_dir_name = _POB_SOURCE_DIR_BY_BASIS.get(name.lower())
    if source_dir_name is None:
        return []
    source_dir = (
        Path(__file__).resolve().parent
        / "basis_library"
        / "sources"
        / source_dir_name
    )
    if not source_dir.is_dir():
        return []

    from .basis_crystal import parse_crystal_atom_basis_file

    atoms = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.name.startswith(".") or "_" not in path.name:
            continue
        atoms.append(parse_crystal_atom_basis_file(path))
    return atoms


def _periodic_route_applies_ecp(resolved_jk, method_upper: str, system) -> bool:
    """Whether the driver ``run_periodic_job`` dispatches for this request
    consumes the inline ECP fields on the options struct.

    The k-point GDF drivers (``run_krhf_periodic_gdf`` and its RKS/UHF/UKS
    entries) build V_ne with Z_eff, Bloch-sum the lattice V_ECP into every
    Hcore(k), use the Z_eff ionic repulsion and fill the valence count. The
    Gamma fast paths, RIJCOSX, the Ewald open-shell drivers, ROHF/ROKS on
    GDF and BIPOLE do not, so an ECP-active cell is refused there rather
    than run all-electron in a valence basis.
    """
    return (
        resolved_jk == PeriodicJKMethod.GDF
        and method_upper in ("RHF", "RKS", "UHF", "UKS")
        and int(system.dim) == 3
    )


def _system_atomic_numbers_in_order(system) -> list:
    return [int(atom.Z) for atom in system.unit_cell]


def _resolve_ecp_data(system, basis) -> tuple:
    """Resolve the periodic ECP data for ``basis`` on ``system``.

    pob-TZVP-REV2 cells read the bundled CRYSTAL records; every other basis
    reads its ``.ecp`` sidecar per element (LANL2DZ, def2 beyond Kr, dhf, ...).

    When ``basis`` is a CRYSTAL-format basis (pob-TZVP-REV2 family)
    with heavy-element ECP blocks embedded in the basis files themselves,
    convert the parsed :class:`CrystalECP` data to the inline-primitive
    format the C++ periodic SCF drivers expect.

    Returns (ecp_primitive_blocks, ecp_home_centers, effective_charges,
    total_ncore). All-electron POB records retain the physical nuclear
    charges, with empty operator blocks/centers and zero core count.
    ECP-bearing POB atoms fail closed if the ECP
    records cannot be resolved; paper routes must not silently continue as
    all-electron calculations after missing runtime data.
    """
    name = getattr(basis, "name", "") or ""
    name_key = name.lower()
    if not name_key.startswith("pob"):
        # The sidecar is the ECP (handovers/HANDOVER_BASIS_ECP_UNIFICATION.md):
        # any bundled basis that replaces core electrons ships
        # basis/<name>.ecp, and the molecular wrappers attach it inline per
        # element. The periodic route reads the same file for the home-cell
        # atoms; the tuple shape (blocks, home centres, per-atom Z_eff, total
        # core count) is what _periodic_ecp_context consumes. Until 2026-09
        # only the pob CRYSTAL records were resolved here, so LANL2DZ or def2
        # beyond Kr on a periodic cell hit the ECP-paired-basis refusal below
        # instead of running with their ECP (#88).
        from .ecp_metadata import inline_ecp_data_for

        blocks, centers, eff_z, ncore = inline_ecp_data_for(
            system.unit_cell_molecule(), name
        )
        return (
            list(blocks),
            [list(map(float, c)) for c in centers],
            [float(q) for q in eff_z],
            int(ncore),
        )

    requested_zs = _system_atomic_numbers(system)
    local_atoms = _bundled_pob_source_atoms(name_key)
    local_by_z = {int(atom.Z): atom for atom in local_atoms}
    if requested_zs and requested_zs.issubset(local_by_z):
        from .basis_crystal import build_periodic_ecp_data

        return build_periodic_ecp_data(system, local_atoms)

    missing_local_zs = sorted(requested_zs.difference(local_by_z))
    try:
        from .basis_crystal import (
            build_periodic_ecp_data,
            fetch_bredow_basis_sets,
        )

        _, atoms_dict = fetch_bredow_basis_sets(
            names=[name_key], verbose=False, return_atoms=True
        )
        if not atoms_dict:
            raise RuntimeError("Bredow fetcher returned no parsed atom records")

        atom_list = atoms_dict.get(name_key, [])
        if not atom_list:
            raise RuntimeError(
                f"Bredow fetcher returned no records for basis {name!r}"
            )
        by_z = {int(atom.Z): atom for atom in atom_list}
        missing_fetched_zs = sorted(requested_zs.difference(by_z))
        if missing_fetched_zs:
            raise RuntimeError(
                "Bredow records do not include requested element(s): "
                f"{_format_atomic_numbers(missing_fetched_zs)}"
            )
        missing_ecp_zs = [
            z for z in missing_local_zs
            if not (by_z[z].has_ecp and by_z[z].ecp is not None)
        ]
        if missing_ecp_zs:
            raise RuntimeError(
                "requested POB heavy element record(s) did not include inline "
                f"ECP data: {_format_atomic_numbers(missing_ecp_zs)}"
            )

        return build_periodic_ecp_data(system, atom_list)
    except Exception as exc:
        raise RuntimeError(
            f"_resolve_ecp_data: basis {name!r} requires ECP source data for "
            f"{_format_atomic_numbers(missing_local_zs)}, but the data could "
            "not be resolved. The periodic SCF will not continue as an "
            f"all-electron calculation. Underlying error: {exc}"
        ) from exc


def _validate_smearing_dispatch(
    *,
    method: str,
    jk_method: PeriodicJKMethod,
    smearing_temperature: float,
    kpoints: object = None,
) -> None:
    """Fail fast when a selected backend cannot honour smearing.

    ``kpoints`` is the runner's k-mesh argument. The runner exposes
    open-shell smearing through the GDF drivers; the low-level multi-k Ewald
    UHF/UKS drivers also support per-spin smearing, but there is no current
    ``run_periodic_job`` Ewald dispatch route for them.
    """
    if float(smearing_temperature) <= 0.0:
        return
    if method in ("ROHF", "ROKS"):
        raise NotImplementedError(
            f"run_periodic_job: periodic {method} uses integer 2/1/0 "
            "occupations; electronic smearing is not implemented."
        )
    if jk_method == PeriodicJKMethod.BIPOLE:
        if method in ("RKS", "UHF", "UKS"):
            return
        raise NotImplementedError(
            "run_periodic_job: smearing_temperature > 0 is implemented "
            "for BIPOLE RKS/UHF/UKS, but BIPOLE RHF still requires integer "
            f"occupations. Got method={method!r}."
        )
    if method in ("UHF", "UKS"):
        # Open-shell smearing is wired on the runner's GDF drivers. The
        # exported multi-k Ewald UHF/UKS drivers also have per-spin smearing,
        # but run_periodic_job no longer exposes an Ewald dispatch branch.
        # BIPOLE UHF is accepted in the BIPOLE branch above; BIPOLE RHF
        # remains integer-occupation only.
        if jk_method in (PeriodicJKMethod.GDF, PeriodicJKMethod.RIJCOSX):
            return
        raise NotImplementedError(
            "run_periodic_job: open-shell (UHF/UKS) smearing_temperature > 0 "
            "is wired through run_periodic_job on the GDF drivers "
            "(jk_method='gdf', Gamma or multi-k) and on the multi-k "
            "RIJCOSX/COSX route. The exported multi-k "
            "Ewald UHF/UKS drivers support spin-resolved smearing directly; "
            "BIPOLE UHF supports spin-resolved smearing directly through "
            "this runner; BIPOLE RHF and the other runner routes do not. "
            f"Got method={method!r}, jk_method={jk_method.value!r}, "
            f"kpoints={'set' if kpoints is not None else 'None'}."
        )
    if jk_method == PeriodicJKMethod.RIJCOSX:
        if kpoints is not None:
            return
        raise NotImplementedError(
            "run_periodic_job: RIJCOSX smearing_temperature > 0 is wired "
            "only on the true multi-k COSX route. Pass kpoints= with at "
            "least two k-points, or omit smearing for the Gamma RIJCOSX "
            "RHF driver."
        )
    if jk_method not in (
        PeriodicJKMethod.GDF,
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
    ):
        raise NotImplementedError(
            "run_periodic_job: smearing_temperature > 0 is currently "
            "wired through the GDF, GPW, GAPW, and multi-k RIJCOSX "
            "closed-shell RHF/RKS "
            f"paths; selected J/K method is {jk_method.value!r}."
        )


def _qvf_periodic_property_payload_supported(
    jk_method: PeriodicJKMethod,
    *,
    uses_external_xc: bool = False,
    ecp_active: bool = False,
) -> bool:
    """Return whether the generic QVF band-property rebuild is valid.

    The generic rebuild assembles its own fixed-cutoff Ewald/HF-like
    operator. That is neither the converged BIPOLE operator nor the executed
    finite-character AICCM2026DEV-B operator. It also has no way to rebuild a
    full-grid external XC potential, so those finite but unrelated
    DOS/PDOS/COOP/COHP and Mayer arrays must be omitted.
    """
    if ecp_active:
        # The property Hamiltonian is rebuilt from bare nuclear charges
        # with no V_ECP, so DOS / PDOS / COOP / COHP for an ECP cell would
        # be an all-electron-in-valence-basis artefact (#88). It is also a
        # 4-index real-space build that dwarfs the SCF on dense-core ECP
        # bases (AgCl / pob-TZVP-rev2: 1 min SCF, >40 min payload). Skip
        # it until the payload consumes the ECP operator.
        return False
    if uses_external_xc:
        return False
    return jk_method not in (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.AICCM2026DEV_B,
    )


def citation_scf_accelerator(
    density_mixer: Optional[str],
    density_mixer_kerker: bool,
) -> str:
    """Name the accelerator this run actually used, for the citation plan.

    `CitationDatabase.assemble` defaults `scf_accelerator` to ``"diis"``
    and the periodic runner never passed it, so a run that mixed with
    Anderson (optionally Kerker-preconditioned) cited Pulay's DIIS papers
    and nothing else -- not merely a missing citation but a WRONG one, on
    the surface users copy into a paper's Methods section
    (PERIODIC-CITES-DIIS-IT-DID-NOT-USE).

    `assemble` takes a single accelerator key, while a periodic run
    applies a density mixer AND, optionally, the Kerker preconditioner.
    The combination is expressed as a composed route key in
    ``database.toml`` (``anderson_kerker``, ``broyden_kerker``) rather
    than by widening `assemble`'s signature, which is IO-chat-owned
    (CLAUDE.md § 16); routes are the implementing chat's (§ 8).

    Mirrors the accepted values of :func:`_validate_density_mixer_dispatch`:
    ``None`` / ``""`` / ``"none"`` / ``"diis"`` all run the default
    Fock-DIIS route, and Kerker is only reachable with Anderson/Broyden.
    """
    key = None if density_mixer is None else str(density_mixer).strip().lower()
    if key in (None, "", "none", "diis"):
        return "diis"
    if density_mixer_kerker:
        return f"{key}_kerker"
    return key


def _validate_density_mixer_dispatch(
    *,
    density_mixer: Optional[str],
    density_mixer_depth: int,
    density_mixer_beta: float,
    density_mixer_kerker: bool,
    kerker_k0: float,
    kerker_strength: float,
    kerker_cutoff_ha: float,
    jk_method: Optional["PeriodicJKMethod"] = None,
    method: Optional[str] = None,
    kpoints: Optional[object] = None,
) -> None:
    """Validate high-level density-mixer requests and gate unsupported routes.

    Anderson/Broyden/Kerker are wired in the lower-level EWALD_3D RKS driver
    (`run_rks_periodic_scf` / `run_rks_periodic_multi_k_ewald3d`) and -- since
    the prompt-22 exposure -- in the closed-shell multi-k GDF driver
    (`run_krhf_periodic_gdf`), which `run_periodic_job` reaches via
    `jk_method="gdf"` + `method="RHF"/"RKS"` or via the public multi-k
    RIJCOSX route (`jk_method="rijcosx"`, same GDF loop with
    `k_exchange="cosx"`). All other runner routes (GPW/GAPW, BIPOLE,
    AICCM, open-shell GDF/RIJCOSX) still fail closed: they cannot honour
    these knobs without silently changing the SCF update scheme.
    """
    non_default_parameter = (
        int(density_mixer_depth) != 8
        or float(density_mixer_beta) != 0.5
        or float(kerker_k0) != 1.5
        or float(kerker_strength) != 1.0
        or float(kerker_cutoff_ha) != 120.0
    )
    if (
        density_mixer is None
        and not density_mixer_kerker
        and not non_default_parameter
    ):
        return

    mixer_key = None if density_mixer is None else str(density_mixer).strip().lower()
    if mixer_key in (None, "", "none"):
        if density_mixer_kerker or non_default_parameter:
            raise ValueError(
                "run_periodic_job: density_mixer_* and kerker_* options require "
                "density_mixer='anderson' or 'broyden'."
            )
        return
    if mixer_key == "diis":
        if density_mixer_kerker or non_default_parameter:
            raise ValueError(
                "run_periodic_job: density_mixer='diis' is the default Fock-DIIS "
                "route and does not accept density_mixer_* or kerker_* options."
            )
        return
    if mixer_key not in {"anderson", "broyden"}:
        raise ValueError(
            "run_periodic_job: density_mixer must be None, 'diis', 'anderson', "
            "or 'broyden'. Use density_mixer_kerker=True to add Kerker "
            "preconditioning to Anderson/Broyden."
        )
    method_u = (method or "").upper()
    if (
        jk_method is not None
        and jk_method in (PeriodicJKMethod.GDF, PeriodicJKMethod.RIJCOSX)
        and method_u in ("RHF", "RKS")
        and kpoints is not None
    ):
        # Supported: the closed-shell multi-k GDF driver honours
        # Anderson/Broyden density mixing with optional Kerker
        # preconditioning (ported from the EWALD_3D driver). Gamma-only
        # requests must pass kpoints=(1,1,1) explicitly -- the runner's
        # default-Gamma GDF path calls run_pbc_gdf_rhf, which has no
        # mixer plumbing, and silently dropping the knob is worse than
        # asking for the Nk=1 mesh.
        return
    raise NotImplementedError(
        "run_periodic_job: density_mixer="
        f"{density_mixer!r} is supported on the closed-shell GDF route "
        "(jk_method='gdf' or multi-k 'rijcosx', method='RHF'/'RKS', "
        "explicit kpoints=) and the "
        "lower-level multi-k EWALD_3D RKS drivers "
        "(vibeqc.run_rks_periodic_scf / run_rks_periodic_multi_k_ewald3d). "
        f"Got method={method!r}, jk_method="
        f"{jk_method.value if jk_method is not None else None!r}, "
        f"kpoints={'set' if kpoints is not None else 'None'}. For a "
        "Gamma-only mixed run pass kpoints=(1,1,1) explicitly. Use "
        "DIIS/FMIXING controls for the GPW/GAPW/BIPOLE/AICCM and "
        "open-shell routes."
    )


_COMPACT_MGGA_DENSITY_MIXER_PROFILES = {
    "covalent-insulator",
    "ionic-insulator",
    "metallic-candidate",
}


def _functional_is_scan_family(functional: Optional[str]) -> bool:
    key = (
        str(functional or "")
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )
    return key == "scan" or key.startswith("r2scan")


def _runner_bloch_kmesh(
    system: PeriodicSystem,
    kpoints: Optional[
        Union[Tuple[int, int, int], List[int], int, "KPoints", "BlochKMesh"]
    ],
):
    """Materialize the high-level runner's k-point input as a BlochKMesh."""
    from ._vibeqc_core import monkhorst_pack as _mp

    if kpoints is None:
        return _mp(system, [1, 1, 1])
    if hasattr(kpoints, "to_bloch_kmesh") or (
        hasattr(kpoints, "kpoints") and hasattr(kpoints, "weights")
    ):
        from .kpoints import as_bloch_kmesh

        return as_bloch_kmesh(kpoints)
    if isinstance(kpoints, (list, tuple)):
        mesh = list(kpoints)
    else:
        mesh = [kpoints, kpoints, kpoints]
    return _mp(system, [int(n) for n in mesh])


def _remap_kpoints_after_primitive_reduction(
    original_system: PeriodicSystem,
    primitive_system: PeriodicSystem,
    kpoints,
):
    """Rebuild an object-form MP mesh on the reduced reciprocal lattice.

    Tuple/integer mesh specifications are materialized only after cell
    reduction and therefore need no adjustment.  ``KPoints`` and raw
    ``BlochKMesh`` objects already contain Cartesian vectors, however; reusing
    them would sample the primitive Hamiltonian on the conventional-cell
    reciprocal lattice.  Only a verifiable Monkhorst-Pack object has an
    unambiguous reduced-cell meaning.  Arbitrary explicit/path objects fail
    closed rather than silently changing their physical k vectors.
    """
    if kpoints is None or isinstance(kpoints, (int, np.integer, list, tuple)):
        return kpoints

    from .kpoints import KPoints

    if isinstance(kpoints, KPoints):
        if (
            kpoints.kind != "monkhorst-pack"
            or kpoints.mesh is None
            or kpoints.shift is None
        ):
            raise NotImplementedError(
                "run_periodic_job: reduce_to_primitive/symmetry reduction "
                "cannot remap an explicit, path, database, or generalized "
                "KPoints object. Pass a Monkhorst-Pack KPoints object or a "
                "mesh tuple so the reciprocal mesh can be rebuilt on the "
                "primitive lattice."
            )
        remapped = KPoints.monkhorst_pack(
            primitive_system,
            kpoints.mesh,
            shift=kpoints.shift,
            symmetry=bool(np.asarray(kpoints.ir_mapping).size),
        )
        # Occupation/BZ choices belong to the sampling request. A convergence
        # ladder and its prose rationale are cell-specific, so do not claim
        # that a conventional-cell verification certifies the remapped mesh.
        remapped.smearing = kpoints.smearing
        remapped.bz_integration = kpoints.bz_integration
        remapped.citation_numerics = tuple(kpoints.citation_numerics)
        return remapped

    if hasattr(kpoints, "kpoints") and hasattr(kpoints, "weights"):
        mesh = tuple(int(x) for x in getattr(kpoints, "mesh", ()))
        shift = tuple(int(x) for x in getattr(kpoints, "is_shift", ()))
        if len(mesh) == 3 and len(shift) == 3:
            from ._vibeqc_core import monkhorst_pack as _mp

            reduced = bool(
                np.asarray(getattr(kpoints, "ir_mapping", []), dtype=int).size
            )
            reference = _mp(
                original_system,
                list(mesh),
                list(shift),
                reduced,
            )
            reference_points = np.asarray(reference.kpoints, dtype=float)
            supplied_points = np.asarray(kpoints.kpoints, dtype=float)
            reference_weights = np.asarray(reference.weights, dtype=float)
            supplied_weights = np.asarray(kpoints.weights, dtype=float)
            if (
                reference_points.shape == supplied_points.shape
                and reference_weights.shape == supplied_weights.shape
                and np.allclose(
                    reference_points,
                    supplied_points,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    reference_weights,
                    supplied_weights,
                    rtol=0.0,
                    atol=1.0e-14,
                )
            ):
                return _mp(
                    primitive_system,
                    list(mesh),
                    list(shift),
                    reduced,
                )

    raise NotImplementedError(
        "run_periodic_job: reduce_to_primitive/symmetry reduction cannot "
        "prove that this object-form k-point list is a Monkhorst-Pack mesh. "
        "Pass a mesh tuple or KPoints.monkhorst_pack(...) so it can be "
        "rebuilt on the primitive reciprocal lattice."
    )


def _bloch_kmesh_size(kmesh) -> int:
    for attr in ("n_kpoints", "nkpts", "num_kpoints"):
        n = getattr(kmesh, attr, None)
        if isinstance(n, int):
            return int(n)
    for attr in ("kpoints_cart", "kpoints", "kpoints_frac"):
        pts = getattr(kmesh, attr, None)
        if pts is not None:
            return int(len(pts))
    return int(len(kmesh))


def _bloch_kmesh_full_size(kmesh) -> int:
    """Return the requested full-mesh size, including an IBZ parent mesh."""
    ir_mapping = np.asarray(
        getattr(kmesh, "ir_mapping", []), dtype=int
    ).reshape(-1)
    if ir_mapping.size > 0:
        return int(ir_mapping.size)
    stored_size = _bloch_kmesh_size(kmesh)
    mesh = getattr(kmesh, "mesh", None)
    if mesh is not None:
        try:
            full_size = int(np.prod(np.asarray(mesh, dtype=int)))
        except (TypeError, ValueError):
            full_size = 0
        # ``bloch_kmesh_from_lists`` assigns mesh=(1,1,1) to arbitrary
        # explicit lists because no MP parent is known. Do not let that
        # placeholder hide multiple stored k-points.
        if full_size > 0 and (stored_size <= 1 or full_size == stored_size):
            return full_size
    return stored_size


def _kmesh_contains_gamma(kmesh) -> bool:
    """Whether the requested Bloch mesh has an exact Γ point.

    Molden has no periodic representation: its ``[MO]`` block is one set of
    real coefficients over home-cell basis functions. k = 0 is the only
    k-point where the Bloch coefficients are real up to a global phase, so
    Γ membership -- not "the mesh is nothing but Γ" -- is what decides
    whether an orbital export can be produced at all. Every Γ-centred
    Monkhorst-Pack mesh qualifies regardless of subdivision; a shifted mesh
    does not.

    Uses the same |k| <= 1e-10 bohr^-1 test as
    :func:`_gamma_index_for_multi_k`, which performs the matching runtime
    selection against the converged result's own k list.
    """
    for attr in ("kpoints_cart", "kpoints"):
        pts = getattr(kmesh, attr, None)
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=float).reshape(-1, 3)
        if arr.shape[0] == 0:
            return False
        return bool(np.min(np.linalg.norm(arr, axis=1)) <= 1.0e-10)
    return False


def _is_multik_kpoints(kpoints) -> bool:
    """Best-effort: does this ``kpoints`` request more than the Γ-point?

    Non-Gamma one-point meshes need the same Bloch restart transport as
    larger meshes. A count alone cannot establish a Gamma-only request.
    """
    if kpoints is None:
        return False
    if isinstance(kpoints, bool):
        return False
    if isinstance(kpoints, int):
        return kpoints > 1
    if isinstance(kpoints, (list, tuple)):
        vals = list(kpoints)
        if len(vals) == 3 and all(isinstance(v, int) for v in vals):
            return int(vals[0]) * int(vals[1]) * int(vals[2]) > 1
        return len(vals) > 1
    ir_mapping = np.asarray(
        getattr(kpoints, "ir_mapping", []), dtype=int
    ).reshape(-1)
    if ir_mapping.size > 1:
        return True
    mesh = getattr(kpoints, "mesh", None)
    if mesh is not None:
        try:
            if int(np.prod(np.asarray(mesh, dtype=int))) > 1:
                return True
        except (TypeError, ValueError):
            pass
    kpts = getattr(kpoints, "kpoints", None)
    if kpts is None:
        kpts = getattr(kpoints, "kpoints_cart", None)
    if kpts is not None:
        try:
            points = np.asarray(kpts, dtype=float)
            if points.shape == (1, 3):
                return not np.allclose(points, 0, rtol=0, atol=1e-10)
            return True
        except (TypeError, ValueError):
            return True
    return True  # unknown explicit object -- assume multi-k (fail closed)


def _filter_bipole_restricted_open_convergence(
    strategy: ConvergenceStrategy,
    method: str,
) -> ConvergenceStrategy:
    """Remove AUTO aids the restricted-open corrected-Ewald engines ignore."""
    method_upper = str(method).upper()
    if method_upper not in ("ROHF", "ROKS"):
        return strategy

    knobs = dict(strategy.knobs)
    changed = False
    for name, reason in (
        (
            "fock_mixing",
            "the restricted-open corrected-Ewald engine does not implement "
            "Fock mixing",
        ),
        (
            "smearing_temperature",
            "the restricted-open corrected-Ewald engine uses fixed 2/1/0 "
            "occupations rather than finite-temperature smearing",
        ),
    ):
        resolution = knobs[name]
        if resolution.source == "auto" and float(resolution.value) != 0.0:
            knobs[name] = KnobResolution(
                0.0,
                "auto",
                f"capability-filtered to zero: {reason}",
            )
            changed = True
    if not changed:
        return strategy
    return ConvergenceStrategy(
        mode=strategy.mode,
        classification=strategy.classification,
        knobs=knobs,
    )


def _system_with_valid_default_multiplicity(
    system: PeriodicSystem,
) -> PeriodicSystem:
    """Return a copy with the lowest valid spin for an invalid default singlet.

    ``PeriodicSystem`` defaults to multiplicity 1, but an odd-electron unit
    cell needs an even multiplicity.  Basis construction already repairs its
    temporary ``Molecule``; this helper gives the SCF driver the same valid
    multiplicity without mutating the caller's system or overriding an
    explicit magnetic multiplicity.
    """
    n_electrons = int(system.n_electrons())
    multiplicity = int(system.multiplicity)
    if multiplicity != 1 or (n_electrons - multiplicity + 1) % 2 == 0:
        return system

    atoms = [
        Atom(int(atom.Z), [float(x) for x in atom.xyz])
        for atom in system.unit_cell
    ]
    normalized = PeriodicSystem(
        dim=int(system.dim),
        lattice=np.asarray(system.lattice, dtype=float),
        unit_cell=atoms,
        charge=int(system.charge),
        multiplicity=int(system.unit_cell_molecule().multiplicity),
    )
    normalized.symmetry = system.symmetry
    return normalized


def _infer_aiccm_scf_reference(
    system: PeriodicSystem,
    *,
    functional: Optional[str],
    scf_reference: Optional[str],
) -> str:
    """Return the ``method_upper`` spelling for ``method="aiccm"`` (D-3).

    ``functional`` decides HF versus KS. With ``scf_reference=None`` the
    shell is inferred by the parity rule the CCM library already applies
    (``run_ccm_scf``, periodic/ccm/route.py: closed iff the electron count is
    even AND the multiplicity is 1), the same rule the periodic semiempirical
    planner uses. ``"rohf"`` / ``"roks"`` are the two explicit references the
    parity rule cannot express.
    """
    is_ks = functional is not None
    if scf_reference is None:
        open_shell = int(getattr(system, "multiplicity", 1) or 1) != 1
        try:
            open_shell = open_shell or int(system.n_electrons()) % 2 == 1
        except Exception:  # pragma: no cover - mirrors periodic/ccm/route.py
            pass
        if open_shell:
            return "UKS" if is_ks else "UHF"
        return "RKS" if is_ks else "RHF"
    key = str(scf_reference).strip().lower()
    if key == "rohf":
        if is_ks:
            raise ValueError(
                "run_periodic_job: scf_reference='rohf' is a Hartree-Fock "
                f"reference but functional={functional!r} was given; drop "
                "functional= or use scf_reference='roks'."
            )
        return "ROHF"
    if key == "roks":
        if not is_ks:
            raise ValueError(
                "run_periodic_job: scf_reference='roks' is a Kohn-Sham "
                "reference and requires functional=..."
            )
        return "ROKS"
    raise ValueError(
        f"run_periodic_job: unknown scf_reference={scf_reference!r} for "
        "method='aiccm'. Valid: None (inferred from functional and the "
        "system's multiplicity/electron parity), 'rohf', 'roks'."
    )


def _validate_closed_shell_electron_count(
    system: PeriodicSystem,
    method_upper: str,
) -> None:
    """Reject an odd-electron cell selected for restricted RHF or KS."""
    if method_upper not in ("RHF", "RKS"):
        return
    try:
        n_electrons = int(system.n_electrons())
    except Exception:
        return
    if n_electrons % 2 == 0:
        return

    unrestricted_method = "UKS" if method_upper == "RKS" else "UHF"
    raise ValueError(
        f"method={method_upper!r} is a closed-shell (restricted) "
        f"method but the cell has an odd electron count ({n_electrons}); "
        "a restricted single determinant needs paired electrons. Use "
        f"method={unrestricted_method!r} (spin-polarised) — for an "
        "open-shell metal such as an odd-Z transition metal, add "
        "atomic_spins=... to seed the moments. If the cell should be "
        "neutral and even, check the charge / composition."
    )


def _reduce_system_to_primitive(
    system: PeriodicSystem,
    *,
    symprec: float = 1e-4,
) -> tuple[PeriodicSystem, SpaceGroup]:
    """Reduce a ``PeriodicSystem`` to its primitive cell via spglib.

    Returns ``(primitive_system, space_group)`` where ``space_group``
    is the symmetry analysis of the **original** cell. The primitive
    system carries its own symmetry analysis as well.

    Raises ``ValueError`` if the primitive cell is identical to the
    input (i.e. the input is already primitive), since no reduction
    is possible.
    """
    # Attach symmetry to the original cell first.
    attach_symmetry(system, symprec=symprec)
    sg_original = system.symmetry
    if sg_original is None:
        raise RuntimeError("attach_symmetry did not populate system.symmetry")

    # Build a Crystal from the PeriodicSystem so we can call to_primitive.
    L = np.asarray(system.lattice, dtype=float, order="F")
    n_atoms_in = len(system.unit_cell)

    # Fractional coordinates: r_frac = L^{-1} . r_cart
    inv_L = np.linalg.inv(L)
    frac_coords = np.empty((3, n_atoms_in), dtype=float, order="F")
    species = []
    for i, atom in enumerate(system.unit_cell):
        r_cart = np.array(atom.xyz, dtype=float)
        frac = inv_L @ r_cart
        frac_coords[:, i] = frac
        species.append(int(atom.Z))

    crystal_in = Crystal(L, frac_coords, species)
    crystal_prim = to_primitive(crystal_in, symprec=symprec)

    n_atoms_out = crystal_prim.n_atoms
    if n_atoms_out == n_atoms_in:
        raise ValueError(
            "reduce_to_primitive: input cell is already primitive "
            f"({n_atoms_in} atoms, space group "
            f"{sg_original.international_symbol} "
            f"(No. {sg_original.number})). "
            "Set reduce_to_primitive=False to skip reduction."
        )

    # Build a new PeriodicSystem from the primitive Crystal.
    prim_lattice = np.asarray(crystal_prim.lattice, dtype=float, order="F")
    prim_atoms = []
    for col in range(n_atoms_out):
        r_cart = prim_lattice @ np.asarray(
            crystal_prim.fractional_coords[:, col], dtype=float
        )
        prim_atoms.append(Atom(int(crystal_prim.species[col]), r_cart.tolist()))

    system_prim = PeriodicSystem(
        dim=system.dim,
        lattice=prim_lattice,
        unit_cell=prim_atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )
    # Reduction can change electron-count parity: four odd-Z FCC atoms are an
    # even-electron singlet, while their one-atom primitive cell is a doublet.
    system_prim = _system_with_valid_default_multiplicity(system_prim)
    # Attach symmetry to the primitive cell too.
    attach_symmetry(system_prim, symprec=symprec)

    return system_prim, sg_original


def _build_primitive_summary(
    system_in: PeriodicSystem,
    system_prim: PeriodicSystem,
    sg: SpaceGroup,
) -> str:
    """Human-readable summary of the cell reduction."""
    n_in = len(system_in.unit_cell)
    n_out = len(system_prim.unit_cell)
    ratio = n_in / n_out if n_out > 0 else 1.0
    L_in = np.asarray(system_in.lattice, dtype=float)
    L_out = np.asarray(system_prim.lattice, dtype=float)
    vol_in = float(abs(np.linalg.det(L_in)))
    vol_out = float(abs(np.linalg.det(L_out)))

    block = HeaderlessBlock(
        "Cell reduction (reduce_to_primitive=True)",
        [HeaderlessColumn("<")],
        body_indent=2,
    )
    block.add_row(
        f"space group      = {sg.international_symbol} "
        f"(No. {sg.number}), point group {sg.point_group}"
    )
    block.add_row(f"symmetry order   = {sg.order}")
    block.add_row(
        f"input cell       = {n_in} atoms, volume = {vol_in:.4f} bohr^3"
    )
    block.add_row(
        f"primitive cell   = {n_out} atoms, volume = {vol_out:.4f} bohr^3"
    )
    block.add_row(f"reduction factor = {ratio:.1f}x fewer atoms")
    if sg.equivalent_atoms:
        block.add_row(
            f"inequivalent     = {len(set(sg.equivalent_atoms))} atom type(s)"
        )
    return block.render(active_policy()) + "\n"


# ============================================================
# Helpers -- text output sections
# ============================================================


def _system_summary(system: PeriodicSystem, ecp_total_ncore: int = 0) -> str:
    """Print lattice + atoms in bohr."""
    L = np.asarray(system.lattice, dtype=float)
    dim = int(system.dim)
    if dim == 1:
        measure_label = "periodic length"
        measure_unit = "bohr"
        measure = float(np.linalg.norm(L[:, 0]))
    elif dim == 2:
        measure_label = "periodic area"
        measure_unit = "bohr^2"
        measure = float(np.linalg.norm(np.cross(L[:, 0], L[:, 1])))
    else:
        measure_label = "cell volume"
        measure_unit = "bohr^3"
        measure = float(abs(np.linalg.det(L)))
    periodicity = HeaderlessBlock(
        "Periodicity",
        [HeaderlessColumn("<")],
        body_indent=2,
    )
    # These are label:value rows, but their historical contract is a
    # *minimum* 14-character label field rather than a table-wide aligned
    # column.  A longer runtime label ("periodic length") must not push the
    # equals signs on the other rows to the right.
    periodicity.add_row(f"{'dimensionality':<14s} = {dim}D")
    active_axes = ", ".join(f"a{i + 1}" for i in range(dim))
    periodicity.add_row(f"{'active axes':<14s} = {active_axes}")
    periodicity.add_row(
        f"{measure_label:<14s} = {measure:.4f} {measure_unit}"
    )

    lattice = HeaderlessBlock(
        "Lattice (bohr)",
        [
            HeaderlessColumn("<", min_width=2),
            HeaderlessColumn("<", min_width=1),
            HeaderlessColumn(">", min_width=14),
            HeaderlessColumn(">", min_width=14),
            HeaderlessColumn(">", min_width=14),
        ],
        body_indent=2,
        gutter=1,
    )
    # ``system.lattice`` columns are the Cartesian lattice vectors (C++
    # ``PeriodicSystem.lattice``: "Columns = Cartesian lattice vectors"), so
    # a{i+1} is column i -- ``L[:, i]``. Printing the rows (``L[i, :]``) would
    # display the transpose, which agrees with the true vectors only for a
    # symmetric lattice matrix and is wrong for a skewed / triclinic cell. The
    # measure lines above (periodic length/area) already read columns.
    for i in range(3):
        lattice.add_row(
            f"a{i + 1}",
            "=",
            f"{L[0, i]:14.8f}",
            f"{L[1, i]:14.8f}",
            f"{L[2, i]:14.8f}",
        )
    if dim < 3:
        supercell_volume = float(abs(np.linalg.det(L)))
        lattice.footer(f"embedding volume = {supercell_volume:.4f} bohr^3")

    # Issue #445: lengths, angles and the nearest interatomic distance are
    # the orientation-SENSITIVE view of the cell. A row-fed (transposed)
    # lattice keeps the volume above bit-identical -- det(L) == det(L.T) --
    # and still runs, so the volume line cannot expose it; these numbers
    # can, and they are what a reviewer compares against literature
    # (h-BN: a = 4.732 bohr, gamma = 60.000 deg, B-N 2.732 bohr). Only the
    # periodic axes are reported; the synthesized axes of a 1-D / 2-D cell
    # are bookkeeping and carry no physical length or angle.
    cell = HeaderlessBlock(
        "Cell parameters",
        [HeaderlessColumn("<")],
        body_indent=2,
    )
    params = cell_parameters(L)
    axis_lengths = params.lengths[:dim]
    cell.add_row(
        f"{'lengths':<14s} = "
        + "  ".join(
            f"|a{i + 1}| = {length:.6f}"
            for i, length in enumerate(axis_lengths)
        )
        + " bohr"
    )
    if dim == 3:
        cell.add_row(
            f"{'angles':<14s} = alpha = {params.alpha:.4f}  "
            f"beta = {params.beta:.4f}  gamma = {params.gamma:.4f} deg"
        )
    elif dim == 2:
        cell.add_row(f"{'angles':<14s} = gamma = {params.gamma:.4f} deg")
    nn = nearest_neighbour_distance(system)
    cell.add_row(
        f"{'nearest pair':<14s} = {nn.distance_bohr:.6f} bohr "
        f"({nn.distance_angstrom:.4f} Angstrom), atoms "
        f"{nn.atom_i + 1}-{nn.atom_j + 1} @ {list(nn.image)}"
    )

    atoms = HeaderlessBlock(
        f"Atoms (bohr) -- {len(system.unit_cell)} in unit cell",
        [
            HeaderlessColumn(">", min_width=4),
            HeaderlessColumn(">", min_width=5, gutter_after=3),
            HeaderlessColumn(">", min_width=14),
            HeaderlessColumn(">", min_width=14),
            HeaderlessColumn(">", min_width=14),
        ],
        body_indent=2,
        gutter=2,
    )
    for i, atom in enumerate(system.unit_cell, start=1):
        x, y, z = atom.xyz
        atoms.add_row(
            i,
            f"Z={atom.Z:3d}",
            f"{x:14.8f}",
            f"{y:14.8f}",
            f"{z:14.8f}",
        )
    n_physical = int(system.n_electrons())
    ncore = int(ecp_total_ncore or 0)
    if ncore > 0:
        # An ECP cell fills the valence count (#88): say so where the old
        # header printed the physical count that no ECP driver uses.
        electrons = (
            f"n_electrons = {n_physical - ncore} valence "
            f"({n_physical} physical, {ncore} in ECP cores)  "
        )
    else:
        electrons = f"n_electrons = {n_physical}  "
    atoms.footer(electrons + f"multiplicity = {system.multiplicity}")
    return "\n\n".join(
        (
            periodicity.render(active_policy()),
            lattice.render(active_policy()),
            cell.render(active_policy()),
            atoms.render(active_policy()),
        )
    ) + "\n"


def _basis_summary(basis: BasisSet) -> str:
    block = HeaderlessBlock(
        "Basis",
        [HeaderlessColumn("<")],
        body_indent=2,
    )
    block.add_row(f"name    = {basis.name}")
    block.add_row(f"nbasis  = {basis.nbasis}")
    block.add_row(f"nshells = {basis.nshells}")
    return block.render(active_policy()) + "\n\n"


def _parity_hold_summary(result) -> str:
    """Parity-hold notice for the ``.out``, appended after the shared SCF
    trace (IID 344).

    Returns ``""`` -- the common case -- unless the executing driver tagged
    the result backend ``+PARITY_HELD`` (the dense-core hold class,
    :func:`vibeqc.pbc_gdf._gdf_backend_with_parity_hold`), so ordinary runs
    emit nothing and golden outputs are unchanged. A held run states the
    hold in the ``.out`` next to the energy it qualifies instead of only in
    the ``.err`` sidecar: the absolute energy is known-wrong versus the
    external parity reference while energy differences may still be usable.
    """
    backend = str(
        getattr(result, "runtime_backend", None)
        or getattr(result, "backend", "")
        or ""
    )
    if "+PARITY_HELD" not in backend:
        return ""
    return (
        "    PARITY HELD: the absolute energy above is held for external\n"
        "    parity (dense-core class; see the .err warning for the\n"
        f"    remediation). Executing backend: {backend}.\n"
        "    Energy differences between runs of this same backend may\n"
        "    still cancel the held offset; do not quote the absolute\n"
        "    energy.\n\n"
    )


def _smearing_summary(result) -> str:
    """Finite-temperature (smearing) block, appended after the shared SCF
    trace + energy-component breakdown.

    Only the smearing-specific quantities live here now: kBT, electronic
    temperature, entropy, Helmholtz free energy, and the Fermi level. The
    iteration table, the ``converged in N iterations`` line, and the
    energy-component breakdown are produced by the shared
    :func:`vibeqc.output.formats.scf_log.format_scf_trace`, identical to the
    molecular path. Returns ``""`` for a zero-temperature (non-smeared)
    result, which is the common case.
    """
    smearing_T = float(getattr(result, "smearing_temperature", 0.0))
    if smearing_T <= 0.0:
        return ""
    policy = (
        active_policy()
        .with_spec("energy", width=20, precision=10)
        .with_spec("temperature", width=20, precision=3)
        .with_spec("dimensionless", width=20, precision=10)
    )
    energy_unit = policy.unit_of("energy")
    temperature_unit = policy.unit_of("temperature")
    block = HeaderlessBlock(
        "Finite-temperature (smearing)",
        [
            HeaderlessColumn(">", min_width=20),
            HeaderlessColumn(">", min_width=1),
            HeaderlessColumn(">", min_width=20),
        ],
        body_indent=2,
        gutter=1,
    )
    block.add_row(
        f"kBT_smearing ({energy_unit})", "=", Quantity(smearing_T, "energy")
    )
    block.add_row(
        f"T_elec ({temperature_unit})",
        "=",
        Quantity(
            hartree_to_kelvin_temperature(smearing_T),
            "temperature",
        ),
    )
    block.add_row(
        "entropy S/kB",
        "=",
        Quantity(float(getattr(result, "entropy", 0.0)), "dimensionless"),
    )
    block.add_row(
        f"free_energy ({energy_unit})",
        "=",
        Quantity(float(getattr(result, "free_energy", result.energy)), "energy"),
    )
    fermi_alpha = getattr(result, "fermi_level_alpha", None)
    fermi_beta = getattr(result, "fermi_level_beta", None)
    if fermi_alpha is not None or fermi_beta is not None:
        if fermi_alpha is not None:
            block.add_row(
                f"fermi_level_alpha ({energy_unit})",
                "=",
                Quantity(float(fermi_alpha), "energy"),
            )
        if fermi_beta is not None:
            block.add_row(
                f"fermi_level_beta ({energy_unit})",
                "=",
                Quantity(float(fermi_beta), "energy"),
            )
    else:
        block.add_row(
            f"fermi_level ({energy_unit})",
            "=",
            Quantity(float(getattr(result, "fermi_level", 0.0)), "energy"),
        )
    return block.render(policy) + "\n\n"


def _linear_dependence_summary(result) -> str:
    """The ``.out`` block for what orthogonalisation actually discarded.

    Emitted whenever the driver populated it, INCLUDING when nothing was
    discarded: "0 directions dropped" is the observation that rules
    linear dependence out as a cause of a surprising periodic energy, and
    its absence is what left several post-mortems unable to distinguish
    an ill-conditioned basis from a broken operator.
    """
    from .linear_dependence import format_periodic_linear_dependence

    summary = getattr(result, "linear_dependence", None)
    if summary is None:
        return ""
    return (
        section_header("Linear dependence", width=56)
        + format_periodic_linear_dependence(summary)
        + "\n"
    )


def _periodic_eigenvalue_channels(result):
    """Return ``(spin, energies, occupations, max_occ)`` result channels.

    Occupations are the only reliable frontier classifier: periodic orbital
    energies have an arbitrary gauge, so their sign cannot determine how many
    states are filled. Older restricted results may supply ``n_electrons``;
    that count is converted to explicit integer occupations as a compatibility
    fallback.
    """
    energies = getattr(result, "mo_energies", None)
    occupations = getattr(result, "occupations", None)
    method_label = str(getattr(result, "method", "") or "").strip().lower()
    if (
        method_label in ("rohf", "roks")
        and energies is not None
        and occupations is not None
        and not (
            isinstance(occupations, (list, tuple)) and len(occupations) == 0
        )
    ):
        # ROHF/ROKS expose alpha/beta aliases for their one restricted spatial
        # orbital set. Prefer the actual 2/1/0 restricted occupations rather
        # than treating those aliases as independent spin channels.
        return [(None, energies, occupations, 2.0)]

    alpha = getattr(result, "mo_energies_alpha", None)
    beta = getattr(result, "mo_energies_beta", None)
    if alpha is not None or beta is not None:
        channels = []
        for spin, energies, occ_name in (
            ("alpha", alpha, "occupations_alpha"),
            ("beta", beta, "occupations_beta"),
        ):
            if energies is None:
                continue
            occupations = getattr(result, occ_name, None)
            if occupations is None:
                return []
            if isinstance(energies, (list, tuple)):
                if not isinstance(occupations, (list, tuple)):
                    continue
                if len(energies) == 0 or len(energies) != len(occupations):
                    continue
            channels.append((spin, energies, occupations, 1.0))
        if channels:
            return channels

    if energies is not None and occupations is not None and not (
        isinstance(occupations, (list, tuple)) and len(occupations) == 0
    ):
        return [(None, energies, occupations, 2.0)]

    if energies is None:
        return []
    if occupations is None or (
        isinstance(occupations, (list, tuple)) and len(occupations) == 0
    ):
        n_electrons = int(getattr(result, "n_electrons", 0) or 0)
        if n_electrons <= 0 or n_electrons % 2:
            return []
        n_occ = n_electrons // 2
        energy_blocks = (
            list(energies)
            if isinstance(energies, (list, tuple))
            else [np.asarray(energies)]
        )
        occupations = []
        for eps in energy_blocks:
            n_orb = len(np.asarray(eps))
            if n_occ > n_orb:
                return []
            occ = np.zeros(n_orb, dtype=float)
            occ[:n_occ] = 2.0
            occupations.append(occ)
        if not isinstance(energies, (list, tuple)):
            occupations = occupations[0]
    return [(None, energies, occupations, 2.0)]


def _band_summary(result) -> str:
    """Print band extrema and gap for multi-k periodic jobs.

    Scans all k-point eigenstates and reports the valence-band
    maximum (VBM), conduction-band minimum (CBM), and the
    associated direct/indirect band gaps across the mesh.
    """
    channels = _periodic_eigenvalue_channels(result)
    if not channels:
        return ""
    first_energies = channels[0][1]
    if not isinstance(first_energies, (list, tuple)) or len(first_energies) <= 1:
        return ""
    n_k = len(first_energies)
    if any(
        not isinstance(energies, (list, tuple)) or len(energies) != n_k
        for _, energies, _, _ in channels
    ):
        return ""
    restricted_open = str(
        getattr(result, "method", "") or ""
    ).strip().lower() in ("rohf", "roks")
    # Absent on the lightweight result shims used by unit fixtures; a result
    # that does not carry convergence state is treated as converged.
    converged = bool(getattr(result, "converged", True))
    for _, _, occupations, max_occ in channels:
        if not isinstance(occupations, (list, tuple)) or len(occupations) != n_k:
            return ""
        for occupation_block in occupations:
            occ = np.asarray(occupation_block, dtype=float)
            if restricted_open:
                distance_to_integer_class = np.min(
                    np.abs(occ[..., None] - np.array([0.0, 1.0, 2.0])),
                    axis=-1,
                )
                fractional = np.any(distance_to_integer_class > 1e-8)
            else:
                fractional = np.any(
                    (occ > 1e-8) & (occ < max_occ - 1e-8)
                )
            if fractional:
                block = HeaderlessBlock(
                    "Band extrema (multi-k)",
                    [HeaderlessColumn("<")],
                    body_indent=2,
                    annotation_gutter=2,
                )
                # "fractionally occupied / smeared" is a statement about the
                # material. A non-converged SCF returns last-iteration
                # eigenvalues, and the fill attached to them is an artefact of
                # an unconverged Fock, not an occupation class -- reporting it
                # as one is how the 2D slab V_ne(k) defect was read as "h-BN
                # is band-overlapping" (GitLab #106, #85). Say what actually
                # happened instead.
                if converged:
                    block.add_row(
                        "occupation class = fractionally occupied / smeared",
                        annotation=(
                            "zero-temperature band edges and metallicity "
                            "not classified"
                        ),
                    )
                else:
                    block.add_row(
                        "SCF not converged - band edges not classified",
                        annotation=(
                            "fractional frontier occupations here are an "
                            "artefact of the unconverged Fock"
                        ),
                    )
                return block.render(active_policy()) + "\n"
    # Gather occupied/virtual indices per k-point from occupation thresholds.
    vbm = -1e300  # highest occupied energy across all k
    cbm = +1e300  # lowest unoccupied energy across all k
    vbm_k = -1
    cbm_k = -1
    direct_gap_min = +1e300
    direct_gap_min_k = -1

    for k_idx in range(n_k):
        occupied_energies = []
        virtual_energies = []
        for _, energies, occupations, max_occ in channels:
            if not isinstance(occupations, (list, tuple)) or len(occupations) != n_k:
                return ""
            eps = np.asarray(energies[k_idx], dtype=float)
            occ = np.asarray(occupations[k_idx], dtype=float)
            if eps.shape != occ.shape:
                return ""
            occ_mask = occ > 1e-8
            # A singly occupied ROHF/ROKS spatial orbital is an integer
            # occupation class, not a thermally fractional virtual state.
            virt_mask = (
                occ < 1e-8
                if restricted_open
                else occ < max_occ - 1e-8
            )
            occupied_energies.extend(eps[occ_mask].tolist())
            virtual_energies.extend(eps[virt_mask].tolist())
        # The VBM and CBM are independent maxima over the whole mesh, so a k
        # point contributes whichever half it has. Under one global Fermi
        # level (#85) a band-overlap mesh can leave a k point with NO occupied
        # states -- every band there sits above mu -- and skipping the whole k
        # would then hide its virtual states from the CBM search. On the
        # two-k reproducer that reported the CBM as +0.80 Ha at the occupied k
        # instead of -0.20 Ha at the empty one: a 1.300000-Ha "direct" gap in
        # place of the true 0.300000-Ha indirect one.
        ho = float(np.max(occupied_energies)) if occupied_energies else None
        lu = float(np.min(virtual_energies)) if virtual_energies else None
        if ho is not None and ho > vbm:
            vbm = ho
            vbm_k = k_idx
        if lu is not None and lu < cbm:
            cbm = lu
            cbm_k = k_idx
        # The DIRECT gap is a per-k quantity and is undefined at a k point
        # that has no occupied or no virtual state; only that half is skipped.
        # Never clamped at zero. A negative per-k direct gap means the
        # occupation classification placed a higher state below a lower
        # one at this k; that is a defect to surface, not to hide
        # (CLAUDE.md section 7).
        if ho is None or lu is None:
            continue
        direct_gap = lu - ho
        if direct_gap < direct_gap_min:
            direct_gap_min = direct_gap
            direct_gap_min_k = k_idx

    if vbm_k < 0 or cbm_k < 0:
        return ""

    # Reported unclamped. When the CBM lies *below* the VBM, the band
    # ordering across the mesh is inverted and the truthful report is the
    # negative number: clamping it to 0.000000 papers over exactly the
    # defect the reader needs to see (CLAUDE.md section 7) and silently
    # defeats any regression test written against this line. The hBN
    # sto-3g slab-GDF reproducer printed 0.000000 Ha while its true
    # CBM - VBM was -0.1201748057 Ha (-3.270 eV).
    indirect_gap = cbm - vbm
    block = HeaderlessBlock(
        "Band extrema (multi-k)",
        [HeaderlessColumn("<")],
        body_indent=2,
        annotation_gutter=2,
    )

    def _energy_pair(value: float, *, width: int, precision: int) -> str:
        return (
            f"{render_energy(value, width=width, precision=precision, unit='Ha')}"
            " Ha  ("
            f"{render_energy(value, width=8, precision=3, unit='eV')} eV)"
        )

    k_ref = getattr(result, "kpoints", None)
    if k_ref is not None and isinstance(k_ref, (list, tuple)):
        _fmt_k = lambda k: ", ".join(f"{x:+.4f}" for x in k)
        if 0 <= vbm_k < len(k_ref):
            block.add_row(
                f"VBM     = {_energy_pair(vbm, width=14, precision=10)}",
                annotation=f"at k = [{_fmt_k(k_ref[vbm_k])}]",
            )
        else:
            block.add_row(
                f"VBM     = {_energy_pair(vbm, width=14, precision=10)}",
                annotation=f"at k_idx = {vbm_k}",
            )
        if 0 <= cbm_k < len(k_ref):
            block.add_row(
                f"CBM     = {_energy_pair(cbm, width=14, precision=10)}",
                annotation=f"at k = [{_fmt_k(k_ref[cbm_k])}]",
            )
        else:
            block.add_row(
                f"CBM     = {_energy_pair(cbm, width=14, precision=10)}",
                annotation=f"at k_idx = {cbm_k}",
            )
    else:
        block.add_row(
            f"VBM     = {_energy_pair(vbm, width=14, precision=10)}",
            annotation=f"(k_idx = {vbm_k})",
        )
        block.add_row(
            f"CBM     = {_energy_pair(cbm, width=14, precision=10)}",
            annotation=f"(k_idx = {cbm_k})",
        )
    gap_type = "direct" if vbm_k == cbm_k else "indirect"
    block.add_row(
        f"gap ({gap_type}) = "
        f"{_energy_pair(indirect_gap, width=10, precision=6)}",
        annotation=(
            "negative: band ordering inverted across the mesh"
            if indirect_gap < 0.0
            else None
        ),
    )
    if direct_gap_min_k >= 0:
        block.add_row(
            "direct gap (min) = "
            f"{_energy_pair(direct_gap_min, width=10, precision=6)}",
            annotation=f"(k_idx = {direct_gap_min_k})",
        )
    else:
        # No k point carries both an occupied and a virtual state, so the
        # direct gap is undefined everywhere on the mesh. Say that rather than
        # rendering the +1e300 sentinel as a number (#85).
        block.add_row(
            "direct gap (min) = n/a",
            annotation=(
                "no k-point has both an occupied and a virtual state"
            ),
        )
    return block.render(active_policy()) + "\n"


def _mo_summary(result, n_show: int = 20) -> str:
    """Print HOCO/LUCO (crystal orbitals) + nearest energies.

    This function is only called for periodic jobs, so the frontier
    orbitals are crystalline orbitals, not molecular orbitals.
    """
    channels = _periodic_eigenvalue_channels(result)
    if not channels:
        return ""
    restricted_open = str(
        getattr(result, "method", "") or ""
    ).strip().lower() in ("rohf", "roks")
    policy = active_policy().with_spec("energy", width=18, precision=10)
    energy_unit = policy.unit_of("energy")
    rendered = []
    for spin, energies, occupations, max_occ in channels:
        if isinstance(energies, (list, tuple)):
            if len(energies) == 0 or not isinstance(occupations, (list, tuple)):
                continue
            eps = np.asarray(energies[0], dtype=float)
            occ = np.asarray(occupations[0], dtype=float)
            n_k = len(energies)
            k_suffix = f" at k-point 0 (of {n_k})"
        else:
            eps = np.asarray(energies, dtype=float)
            occ = np.asarray(occupations, dtype=float)
            k_suffix = ""
        if eps.shape != occ.shape or not occ.size:
            continue
        if restricted_open:
            distance_to_integer_class = np.min(
                np.abs(occ[..., None] - np.array([0.0, 1.0, 2.0])),
                axis=-1,
            )
            fractional = bool(np.any(distance_to_integer_class > 1e-8))
        else:
            fractional = bool(
                np.any((occ > 1e-8) & (occ < max_occ - 1e-8))
            )
        occ_mask = occ > 1e-8
        virt_mask = (
            occ < 1e-8
            if restricted_open
            else occ < max_occ - 1e-8
        )
        if fractional:
            homo_idx = -1
            lumo_idx = eps.size
            fermi_level = (
                getattr(result, f"fermi_level_{spin}", None)
                if spin is not None
                else None
            )
            if fermi_level is None:
                fermi_level = getattr(result, "fermi_level", None)
            centre = (
                int(np.searchsorted(eps, float(fermi_level)))
                if fermi_level is not None
                else eps.size // 2
            )
        else:
            homo_idx = (
                int(np.where(occ_mask)[0][-1]) if np.any(occ_mask) else -1
            )
            lumo_idx = (
                int(np.where(virt_mask)[0][0])
                if np.any(virt_mask)
                else eps.size
            )
            centre = (
                max(homo_idx + 1, 0)
                if homo_idx >= 0
                else min(lumo_idx, eps.size)
            )
        spin_suffix = f" ({spin})" if spin is not None else ""
        title = (
            f"Crystal orbital energies{spin_suffix} ({energy_unit})"
            f"{k_suffix} -- sorted low -> high"
        )
        if fractional:
            title += "; fractional occupations, frontier labels omitted"
        block = HeaderlessBlock(
            title,
            [
                HeaderlessColumn(">", min_width=4),
                HeaderlessColumn(">", min_width=18),
                HeaderlessColumn("<", min_width=12),
            ],
            body_indent=2,
            gutter=3,
            annotation_gutter=1,
        )
        half = n_show // 2
        lo = max(0, centre - half)
        hi = min(eps.size, centre + half)
        for i in range(lo, hi):
            marker = "HOCO" if i == homo_idx else "LUCO" if i == lumo_idx else None
            block.add_row(
                i + 1,
                Quantity(eps[i], "energy"),
                f"occ={occ[i]:8.5f}",
                annotation=marker,
            )
        rendered.append(block.render(policy))
    return "\n".join(rendered) + ("\n" if rendered else "")


def _optimized_geometry_summary(opt_result) -> str:
    """Render the final periodic optimization verdict once for every route."""
    block = HeaderlessBlock(
        "Optimized geometry",
        [HeaderlessColumn("<")],
        body_indent=2,
    )
    smeared = (
        getattr(opt_result, "minimised_potential", "energy") == "free_energy"
        and getattr(opt_result, "free_energy", None) is not None
        and getattr(opt_result, "internal_energy", None) is not None
    )
    if smeared:
        # Finite electronic temperature: the optimizer minimised the Mermin
        # free energy F = E - T S (#545); report F, E and -TS so a reader
        # sees which surface the geometry is stationary on.
        free_energy = float(opt_result.free_energy)
        internal = float(opt_result.internal_energy)
        block.add_row(
            "F_final = "
            f"{render_energy_labeled(free_energy, width=0, precision=10, sign=True)}"
            "  (Mermin free energy, minimised)"
        )
        block.add_row(
            "E_final = "
            f"{render_energy_labeled(internal, width=0, precision=10, sign=True)}"
            "  (internal energy)"
        )
        block.add_row(
            "-TS     = "
            f"{render_energy_labeled(free_energy - internal, width=0, precision=10, sign=True)}"
            f"  (kT = {float(opt_result.smearing_temperature):.6g} Ha)"
        )
    else:
        block.add_row(
            "E_final = "
            f"{render_energy_labeled(opt_result.energy, width=0, precision=10, sign=True)}"
        )
    block.add_row(f"n_iter  = {int(opt_result.n_iter)}")
    block.add_row(f"converged = {bool(opt_result.converged)}")
    return block.render(active_policy()) + "\n\n"


def _terminal_scf_check_clause(trace, *, conv_tol_energy, conv_tol_grad) -> str:
    """Render the "Terminal check on the density returned" clause (#116).

    **This function must never raise.** It runs on the SCF *failure* path, so
    an exception here replaces the report of why a calculation failed with a
    traceback about the reporter. That is GitLab #748: five LiF cases in the
    2026-09-07 validation wave reported
    ``AttributeError: 'dict' object has no attribute 'delta_e'`` and nothing
    about their real convergence state, because every case reaching this
    branch failed the same way regardless of cause.

    Trace rows arrive in two shapes. The C++ drivers and the GDF / BIPOLE /
    Ewald routes emit ``SCFIteration`` records; the GPW / GAPW Python drivers
    emit plain dicts with the same key names (``periodic_gapw_j.py``,
    ``periodic_gapw_augment.py``, ``periodic_gapw_open_shell.py``). Most GPW
    routes are lifted back to records by ``periodic_gapw_runner_adapter``,
    but the GAPW multi-k route reaches the runner through a proxy that
    forwards the driver's attributes unchanged, so its dict rows arrive here
    raw.

    The two shapes do **not** imply two physics. On the GPW / GAPW density
    loops ``grad_norm`` holds the density change ``||D - D_prev||_F`` gated by
    ``conv_tol_density``, while ``periodic_gapw_cpp_host`` re-packs genuine
    ``||[F,DS]||`` commutator norms into the very same dict shape. Row shape
    therefore cannot decide which quantity is in hand, so the residual is
    named ``||[F,DS]||`` and compared against ``conv_tol_grad`` only for the
    record rows where both are known to hold, and reported under its own
    field name, with no tolerance claimed, otherwise. Reporting a density
    change as a commutator norm against a tolerance that does not gate it
    would be a wrong number under a wrong name.
    """
    try:
        rows = list(trace) if trace is not None else []
        if not rows:
            return ""
        last = rows[-1]
        is_record = not isinstance(last, dict)
        if isinstance(last, dict):
            delta_e = last.get("delta_e")
            residual = last.get("grad_norm")
        else:
            delta_e = getattr(last, "delta_e", None)
            residual = getattr(last, "grad_norm", None)
        parts = []
        if delta_e is not None:
            # |dE| against conv_tol_energy is meaningful for both families:
            # every periodic loop gates on the energy change.
            parts.append(
                f"|dE| = {abs(float(delta_e)):.3e} Ha against "
                f"conv_tol_energy {float(conv_tol_energy):.0e}"
            )
        if residual is not None:
            if is_record:
                clause = f"||[F,DS]|| = {float(residual):.3e}"
                if conv_tol_grad is not None:
                    clause += f" against conv_tol_grad {float(conv_tol_grad):.0e}"
                parts.append(clause)
            else:
                parts.append(
                    f"trace grad_norm = {float(residual):.3e} (the driver's "
                    "own residual field; this route's convergence gate is "
                    "not necessarily conv_tol_grad)"
                )
        if not parts:
            return ""
        return " Terminal check on the density returned: " + ", ".join(parts) + "."
    except Exception:  # noqa: BLE001 - a diagnostic must not mask the failure
        return ""


@contextmanager
def _gdf_displaced_ecp_centers(original_system, displaced_system, options):
    """Move home ECP projectors with their atoms for one serial SCF call.

    Native option objects are not copyable. Resolve every owner before any
    mutation, and restore the caller's metadata even if the SCF raises.
    """
    original = np.asarray([atom.xyz for atom in original_system.unit_cell])
    displaced = np.asarray([atom.xyz for atom in displaced_system.unit_cell])
    if original.shape != displaced.shape:
        raise ValueError("GDF displacement changed the atom count")
    changes, seen = [], set()
    for opts in options:
        if id(opts) in seen or not hasattr(opts, "ecp_home_centers"):
            continue
        seen.add(id(opts))
        centers = list(getattr(opts, "ecp_home_centers", ()) or ())
        if not centers:
            continue
        owners = []
        for center in centers:
            matches = np.flatnonzero(np.linalg.norm(original - center, axis=1) < 1e-10)
            if len(matches) != 1:
                raise ValueError("GDF ECP projector must belong to exactly one home atom")
            owners.append(int(matches[0]))
        changes.append((opts, centers, displaced[owners].tolist()))
    try:
        for opts, _, centers in changes:
            opts.ecp_home_centers = centers
        yield
    finally:
        for opts, centers, _ in changes:
            opts.ecp_home_centers = centers


def _relax_periodic_gdf_atoms(
    system: PeriodicSystem,
    basis_name: Union[str, BasisSet],
    rerun_gdf_scf,
    *,
    max_iter: int,
    conv_tol_grad: float,
):
    """Relax atomic positions on the GDF analytic-gradient objective.

    Mirrors :func:`vibeqc.bipole_optimize.relax_atoms` (scipy L-BFGS-B in
    fractional coordinates, system + basis rebuilt per candidate
    geometry, the same non-convergence penalty barrier and the same
    independent gradient gate) -- but the objective is
    ``rerun_gdf_scf(system, basis)``: the exact GDF driver call the
    runner's SCF dispatch executed, re-run with ``compute_gradient=True``.
    The relaxation therefore converges to a stationary point of the same
    surface the job reports (G-PBC-002 optimizer wiring); relaxing a GDF
    energy on BIPOLE forces lands on a stationary point of the wrong
    surface.

    One SCF per evaluation supplies energy *and* analytic gradient
    (``jac=True``), unlike the BIPOLE relaxer's separate objective /
    gradient callbacks.

    Fail-closed: an envelope the GDF gradient rejects surfaces the
    driver's ``NotImplementedError`` unchanged -- there is no silent
    fallback to another force objective.
    """
    from scipy.optimize import minimize

    from .bipole_optimize import (
        OptimizeResult,
        SCFNonConvergence,
        _atoms_to_flat,
        _flat_to_system,
    )
    from .molecular_optimize import _gradient_converged
    from .bipole_gradient import _recenter_basis_on_periodic_system

    template = (BasisSet(system.unit_cell_molecule(), basis_name)
                if isinstance(basis_name, str) else basis_name)

    n_atoms = len(system.unit_cell)
    # Same recoverable line-search barrier as relax_atoms: a probe
    # geometry whose SCF fails returns a finite penalty (never used as a
    # real objective value) so the line search backs off; only a failure
    # at the initial geometry aborts.
    _PENALTY_HA = 1.0e6
    _had_good_eval = {"ok": False}

    def _energy_and_gradient(x: np.ndarray):
        sys2 = _flat_to_system(system, x)
        basis2 = _recenter_basis_on_periodic_system(template, sys2)
        result = rerun_gdf_scf(sys2, basis2)
        if not bool(getattr(result, "converged", True)):
            if not _had_good_eval["ok"]:
                raise SCFNonConvergence(
                    "run_periodic_job(optimize=True): the GDF SCF did not "
                    "converge at the initial geometry inside the optimizer "
                    "re-run; refusing to relax against a non-converged "
                    "objective. Increase max_iter, loosen conv_tol_energy, "
                    "or adjust the convergence aids."
                )
            return _PENALTY_HA, np.zeros(3 * n_atoms)
        _had_good_eval["ok"] = True
        grad_cart = getattr(result, "gradient", None)
        if grad_cart is None:
            raise NotImplementedError(
                "run_periodic_job(optimize=True): the dispatched GDF "
                "driver returned no analytic gradient despite "
                "compute_gradient=True; the GDF relaxation objective "
                "cannot proceed."
            )
        grad_cart = np.asarray(grad_cart, dtype=float)
        # dE/d(frac) = lattice^T . dE/d(cart), row per atom -- the same
        # conversion relax_atoms applies.
        grad_frac = grad_cart @ np.asarray(sys2.lattice, dtype=float)
        # Objective/gradient consistency: for a Fermi-Dirac smeared SCF
        # the analytic gradient is dA/dR of the Mermin free energy
        # A = E - T.S (the surface the smeared SCF converges on), so
        # the scalar objective must be result.free_energy, not
        # result.energy -- otherwise L-BFGS-B line-searches a surface
        # whose derivative it was not given. At T = 0 the drivers set
        # free_energy == energy, so this is only exercised when
        # smearing is active.
        smear_T = float(getattr(result, "smearing_temperature", 0.0) or 0.0)
        e_obj = (
            float(result.free_energy) if smear_T > 0.0
            else float(result.energy)
        )
        return e_obj, grad_frac.ravel()

    x0 = _atoms_to_flat(system)
    res = minimize(
        _energy_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "gtol": float(conv_tol_grad)},
    )
    sys_opt = _flat_to_system(system, res.x)
    # Independent gradient gate (shared with relax_atoms /
    # optimize_molecule): scipy's success flag can trip on ftol at a
    # non-stationary geometry.
    converged, grad_max = _gradient_converged(
        bool(res.success), res.jac, float(conv_tol_grad)
    )
    write(
        f"\nAtomic relaxation: {res.nit} iters, "
        f"E = {res.fun:.8f} Ha, "
        f"max|grad| = {grad_max:.4e}, "
        f"converged={converged}\n"
    )
    return OptimizeResult(sys_opt, res.fun, res.jac, res.nit, converged)


def _is_lattice_matrix_set_like(obj) -> bool:
    return (
        hasattr(obj, "cells") and hasattr(obj, "blocks") and hasattr(obj, "set_block")
    )


def _single_gamma_block_for_periodic_artifact(block, *, result) -> np.ndarray | None:
    """Return the exact Gamma block of a one-k-point per-k container.

    A Gamma-only Bloch mesh carries a single k-point at k = 0, where the Bloch
    matrix *is* the real-space unit-cell matrix.  The multi-k drivers return
    that one block inside a per-k container, so without this the Gamma artifact
    path refuses a result it can represent exactly: the same cell writes its
    density with ``kpoints=None`` and is discarded with ``kpoints=(1, 1, 1)``
    (#679).

    Returns ``None`` unless the container holds exactly one block at exactly
    k = 0, so every other shape -- a true multi-k stack above all -- keeps its
    existing refusal.  The caller independently certifies D:S against the
    expected electron count, so an admitted block cannot pass a wrong density.
    """
    arr = np.asarray(block)
    if arr.ndim != 3 or arr.shape[0] != 1:
        return None
    kpoints = getattr(result, "kpoints_cart", None)
    if kpoints is None:
        return None
    k_cart = np.asarray(kpoints, dtype=float).reshape(-1, 3)
    if k_cart.shape[0] != 1 or not np.allclose(
        k_cart[0], 0.0, rtol=0.0, atol=1.0e-12
    ):
        return None
    return arr[0]


def _real_block_for_periodic_output(
    block,
    *,
    label: str,
    abs_tol: float = 1.0e-10,
    rel_tol: float = 1.0e-7,
) -> np.ndarray:
    """Return a real matrix block for real scalar output artifacts."""
    arr = np.asarray(block)
    if np.iscomplexobj(arr):
        if arr.size:
            max_imag = float(np.max(np.abs(arr.imag)))
            max_real = float(np.max(np.abs(arr.real)))
        else:
            max_imag = 0.0
            max_real = 0.0
        limit = max(float(abs_tol), float(rel_tol) * max(max_real, 1.0))
        if max_imag > limit:
            raise ValueError(
                f"{label} has a non-negligible imaginary component "
                f"(max|Im|={max_imag:.2e}, max|Re|={max_real:.2e}, "
                f"tolerance={limit:.2e}); periodic density output is "
                "real-only and the k-point density fold is not "
                "time-reversal consistent"
            )
        arr = arr.real
    return np.ascontiguousarray(arr, dtype=float)


def _system_with_valid_unit_cell_multiplicity(
    system: PeriodicSystem,
) -> PeriodicSystem:
    """Return ``system`` or an output-only copy with a valid unit-cell spin."""
    unit = system.unit_cell_molecule()
    charge = int(getattr(unit, "charge", system.charge))
    multiplicity = int(getattr(unit, "multiplicity", system.multiplicity))
    if charge == int(system.charge) and multiplicity == int(system.multiplicity):
        return system
    atoms = [
        Atom(int(atom.Z), [float(x) for x in atom.xyz])
        for atom in system.unit_cell
    ]
    return PeriodicSystem(
        int(system.dim),
        np.asarray(system.lattice, dtype=float),
        atoms,
        charge=charge,
        multiplicity=multiplicity,
    )


def _fold_per_k_density_for_periodic_output(
    basis: BasisSet,
    system: PeriodicSystem,
    D_per_k: Sequence[np.ndarray],
    kpoints_cart,
    weights,
    lat_opts,
):
    """Inverse-Bloch fold per-k AO density matrices for real output grids."""
    from ._vibeqc_core import compute_overlap_lattice

    D_set = compute_overlap_lattice(basis, system, lat_opts)
    kpts = np.asarray(kpoints_cart, dtype=float).reshape(-1, 3)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    if len(D_per_k) == 0:
        raise ValueError("SCF result has an empty density list")
    if kpts.shape[0] != len(D_per_k) or w_arr.shape[0] != len(D_per_k):
        raise ValueError(
            "per-k density output requires aligned density, k-point, and "
            f"weight lists; got {len(D_per_k)} densities, {kpts.shape[0]} "
            f"k-points, {w_arr.shape[0]} weights"
        )
    total_w = float(w_arr.sum())
    if total_w <= 0.0:
        raise ValueError("per-k density output requires positive k-point weights")
    if not np.isclose(total_w, 1.0, rtol=1.0e-9, atol=1.0e-12):
        w_arr = w_arr / total_w

    terms = _time_reversal_completed_density_terms(
        D_per_k,
        kpts,
        w_arr,
    )
    for cell_idx, cell in enumerate(D_set.cells):
        r_cart = np.asarray(cell.r_cart, dtype=float).reshape(3)
        P_g = np.zeros((basis.nbasis, basis.nbasis), dtype=np.complex128)
        for weight, k_cart, D in terms:
            phase = np.exp(-1j * float(np.dot(k_cart, r_cart)))
            P_g += float(weight) * phase * D
        D_set.set_block(
            cell_idx,
            _real_block_for_periodic_output(
                P_g,
                label=f"periodic density block g={tuple(cell.index)}",
            ),
        )
    return D_set


def _wrap_reciprocal_fractional(frac: np.ndarray) -> np.ndarray:
    """Wrap reciprocal fractional coordinates into [-0.5, 0.5)."""
    arr = np.asarray(frac, dtype=float)
    return arr - np.floor(arr + 0.5)


def _fractional_kpoints_for_output(
    system: PeriodicSystem,
    kpoints_cart: np.ndarray,
) -> np.ndarray:
    """Return reciprocal fractional k coordinates for output folding."""
    B = np.asarray(system.reciprocal_lattice(), dtype=float)
    return (np.linalg.pinv(B) @ np.asarray(kpoints_cart, dtype=float).T).T


def _time_reversal_completed_density_terms(
    D_per_k: Sequence[np.ndarray],
    kpoints_cart: np.ndarray,
    weights: np.ndarray,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return density-fold terms symmetrized over time reversal.

    Every k point is folded as the explicit conjugate pair

        w/2 exp(-i k.R) D(k) + w/2 exp(+i k.R) D(k)^*

    so each lattice block of the fold is 2 Re(...) -- real to machine
    precision by construction. This both materializes the implicit -k
    partner of a symmetry-reduced mesh and time-reversal-symmetrizes an
    explicit +/-k mesh: the converged per-k densities satisfy
    D(-k) = D(k)^* only up to the per-k lattice-truncation asymmetry of
    the Fock build (~1e-6 relative; see
    handovers/HANDOVER_GDF_FIT_SCREENING.md "Open findings"), so trusting
    the explicit partner leaves a spurious imaginary residue of that size
    in the folded blocks. The symmetrized fold equals the real part of
    the pristine fold, which is exact for the time-reversal-symmetric
    Hamiltonians vibe-qc ships (no magnetic field / spin-orbit coupling).
    """
    kpts = np.asarray(kpoints_cart, dtype=float).reshape(-1, 3)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    terms: list[tuple[float, np.ndarray, np.ndarray]] = []

    for weight, k_cart, D_k in zip(w_arr, kpts, D_per_k):
        D = np.asarray(D_k, dtype=np.complex128)
        D = 0.5 * (D + D.conj().T)
        half = 0.5 * float(weight)
        k = np.asarray(k_cart, dtype=float)
        terms.append((half, k, D))
        terms.append((half, -k, D.conj()))
    return terms


def _result_kpoints_cart(result) -> Optional[np.ndarray]:
    """Return result k-points in Cartesian bohr^-1 coordinates, if present."""
    kpts = getattr(result, "kpoints_cart", None)
    if kpts is None:
        kpts = getattr(result, "kpoints", None)
    if kpts is None:
        kmesh = getattr(result, "kmesh", None)
        if kmesh is not None:
            kpts = getattr(kmesh, "kpoints_cart", None)
            if kpts is None:
                kpts = getattr(kmesh, "kpoints", None)
    if kpts is None:
        kpts = getattr(result, "restart_kpoints", None)
    if kpts is None:
        return None
    arr = np.asarray(kpts, dtype=float)
    if arr.size == 0:
        return None
    return arr.reshape(-1, 3)


def _result_kpoint_weights(result) -> Optional[np.ndarray]:
    """Return result k-point weights, if present."""
    weights = getattr(result, "kpoint_weights", None)
    if weights is None:
        weights = getattr(result, "weights", None)
    if weights is None:
        kmesh = getattr(result, "kmesh", None)
        if kmesh is not None:
            weights = getattr(kmesh, "weights", None)
    if weights is None:
        weights = getattr(result, "restart_weights", None)
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _density_proxy_with_k_metadata(result, density) -> SimpleNamespace:
    """Proxy one spin density while preserving k metadata for output folds."""
    payload: dict[str, object] = {"density": density}
    for name in ("kpoints_cart", "kpoint_weights", "kpoints", "weights", "kmesh",
                 "restart_kpoints", "restart_weights"):
        if hasattr(result, name):
            payload[name] = getattr(result, name)
    return SimpleNamespace(**payload)


def _gamma_index_for_multi_k(result, n_items: int) -> int:
    """Find the actual Gamma point for Gamma-only writer fallbacks."""
    if n_items <= 0:
        raise ValueError("multi-k result has no k-point data")
    if n_items == 1:
        return 0
    kpts = _result_kpoints_cart(result)
    if kpts is None:
        raise ValueError(
            "multi-k Gamma-only output requires k-point metadata; "
            "cannot assume the first k-point is Gamma"
        )
    if kpts.shape[0] != n_items:
        raise ValueError(
            "multi-k Gamma-only output requires aligned k-point metadata; "
            f"got {n_items} data blocks and {kpts.shape[0]} k-points"
        )
    norms = np.linalg.norm(kpts, axis=1)
    gamma_idx = int(np.argmin(norms))
    if float(norms[gamma_idx]) > 1.0e-10:
        raise ValueError(
            "multi-k Gamma-only output requires an explicit Gamma k-point; "
            f"closest |k|={float(norms[gamma_idx]):.2e} bohr^-1"
        )
    return gamma_idx


def _density_lattice_set_for_output(
    basis: BasisSet,
    system: PeriodicSystem,
    result,
    lat_opts,
):
    """Return a template-aligned LatticeMatrixSet for legacy DOS analysis."""
    D_attr = getattr(result, "density", None)
    if D_attr is None:
        raise ValueError("SCF result has no .density attribute")
    if _is_lattice_matrix_set_like(D_attr):
        from ._vibeqc_core import compute_overlap_lattice

        # SCF drivers may converge a density on a shorter lattice cutoff
        # than the fixed legacy DOS template. Build that target first,
        # then zero-extend the converged density by lattice-cell key. Never
        # use positional alignment: equal-radius cells have no ordering
        # contract across independently constructed lattice sets.
        D_set = compute_overlap_lattice(basis, system, lat_opts)
        target_cells = list(D_set.cells)
        target_blocks = list(D_set.blocks)
        if len(target_cells) != len(target_blocks):
            raise ValueError(
                "density output template received misaligned cells and blocks"
            )
        target_keys = [
            tuple(int(value) for value in cell.index) for cell in target_cells
        ]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("density output template received duplicate cells")

        source_cells = list(D_attr.cells)
        source_blocks = list(D_attr.blocks)
        if len(source_cells) != len(source_blocks):
            raise ValueError(
                "density lattice-set output received misaligned cells and blocks"
            )

        block_by_cell: dict[tuple[int, int, int], np.ndarray] = {}
        for cell, block in zip(source_cells, source_blocks):
            key = tuple(int(value) for value in cell.index)
            if key in block_by_cell:
                raise ValueError(
                    "density lattice-set output received duplicate cell "
                    f"{key}"
                )
            real_block = _real_block_for_periodic_output(
                block,
                label=f"periodic density block g={key}",
            )
            expected_shape = (int(D_set.nbf), int(D_set.nbf))
            if real_block.shape != expected_shape:
                raise ValueError(
                    "density lattice-set output block "
                    f"g={key} has shape {real_block.shape}, expected "
                    f"{expected_shape}"
                )
            block_by_cell[key] = real_block

        source_only = set(block_by_cell).difference(target_keys)
        if source_only:
            missing = sorted(source_only)
            raise ValueError(
                "density output template omits converged lattice cells: "
                f"{missing}"
            )

        zero = np.zeros((int(D_set.nbf), int(D_set.nbf)), dtype=float)
        for i, key in enumerate(target_keys):
            D_set.set_block(i, block_by_cell.get(key, zero))
        return D_set

    from ._vibeqc_core import compute_overlap_lattice

    if isinstance(D_attr, (list, tuple)):
        kpoints_cart = _result_kpoints_cart(result)
        weights = _result_kpoint_weights(result)
        has_complex_blocks = any(np.iscomplexobj(np.asarray(d)) for d in D_attr)
        if has_complex_blocks and (kpoints_cart is None or weights is None):
            raise ValueError(
                "complex per-k density output requires k-point and weight "
                "metadata so the density can be inverse-Bloch folded; "
                "refusing to real-project individual k blocks"
            )
        if kpoints_cart is not None and weights is not None:
            return _fold_per_k_density_for_periodic_output(
                basis,
                system,
                D_attr,
                kpoints_cart,
                weights,
                lat_opts,
            )
        if len(D_attr) == 0:
            raise ValueError("SCF result has an empty density list")
        D_set = compute_overlap_lattice(basis, system, lat_opts)
        D_home = sum(
            _real_block_for_periodic_output(
                d,
                label=f"periodic density k-point {i}",
            )
            for i, d in enumerate(D_attr)
        ) / len(D_attr)
    else:
        D_set = compute_overlap_lattice(basis, system, lat_opts)
        D_home = _real_block_for_periodic_output(
            D_attr,
            label="periodic density matrix",
        )
    zero = np.zeros_like(D_home)
    for i in range(len(D_set)):
        D_set.set_block(i, D_home if i == 0 else zero)
    return D_set


def _exact_lattice_density_set_for_grid_artifact(
    basis: BasisSet,
    system: PeriodicSystem,
    density,
    *,
    label: str,
):
    """Copy the returned SCF lattice density without changing its cell set.

    Density grids are a representation of the converged SCF state, so their
    lattice-cell domain must come from that state rather than from an
    independently generated overlap template.  The copy projects only
    numerically negligible imaginary residue and otherwise preserves every
    returned cell and block in source order.
    """
    if not _is_lattice_matrix_set_like(density):
        raise ValueError(
            f"{label} is not a returned lattice-cell density; exact periodic "
            "density artifacts require LatticeMatrixSet cells and blocks and "
            "will not reconstruct them from orbitals or a surrogate template"
        )

    nbf = int(basis.nbasis)
    if int(getattr(density, "nbf", nbf)) != nbf:
        raise ValueError(
            f"{label} has basis dimension {density.nbf}, expected {nbf}"
        )
    cells = list(density.cells)
    source = list(density.blocks)
    if not cells:
        raise ValueError(f"{label} has no lattice cells")
    if len(cells) != len(source):
        raise ValueError(
            f"{label} has {len(cells)} cells but {len(source)} blocks"
        )
    lattice = np.asarray(system.lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError(
            "exact periodic density artifacts require a finite 3x3 "
            "column-vector lattice"
        )

    expected_shape = (nbf, nbf)
    seen: set[tuple[int, int, int]] = set()
    blocks: list[np.ndarray] = []
    for cell, block in zip(cells, source):
        index = np.asarray(cell.index, dtype=float)
        if index.shape != (3,) or not np.isfinite(index).all():
            raise ValueError(
                f"{label} lattice-cell index must be a finite three-vector"
            )
        key = tuple(int(value) for value in index)
        if not np.array_equal(index, np.asarray(key, dtype=float)):
            raise ValueError(f"{label} contains a non-integral lattice cell {key}")
        if key in seen:
            raise ValueError(f"{label} contains duplicate lattice cell {key}")
        seen.add(key)
        r_cart = np.asarray(cell.r_cart, dtype=float)
        expected_r_cart = lattice @ np.asarray(key, dtype=float)
        if (
            r_cart.shape != (3,)
            or not np.isfinite(r_cart).all()
            or not np.allclose(
                r_cart,
                expected_r_cart,
                rtol=1.0e-12,
                atol=1.0e-10,
            )
        ):
            raise ValueError(
                f"{label} lattice cell {key} has Cartesian translation "
                "inconsistent with the column-vector primitive lattice"
            )
        real_block = _real_block_for_periodic_output(
            block,
            label=f"{label} block g={key}",
        )
        if real_block.shape != expected_shape:
            raise ValueError(
                f"{label} block g={key} has shape {real_block.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(real_block).all():
            raise ValueError(f"{label} block g={key} contains non-finite values")
        blocks.append(real_block)

    from ._vibeqc_core import make_lattice_matrix_set

    return make_lattice_matrix_set(nbf, cells, blocks)


def _sum_exact_lattice_density_sets_for_grid_artifact(
    basis: BasisSet,
    *density_sets,
    label: str,
):
    """Add returned spin densities by exact lattice-cell key."""
    if not density_sets:
        raise ValueError("no exact spin density lattice sets supplied")

    ref_cells = list(density_sets[0].cells)
    ref_keys = [tuple(int(value) for value in cell.index) for cell in ref_cells]
    ref_key_set = set(ref_keys)
    if len(ref_key_set) != len(ref_keys):
        raise ValueError(f"{label} reference contains duplicate lattice cells")

    block_maps: list[dict[tuple[int, int, int], np.ndarray]] = []
    for spin_index, density_set in enumerate(density_sets):
        cells = list(density_set.cells)
        blocks = list(density_set.blocks)
        if len(cells) != len(blocks):
            raise ValueError(
                f"{label} channel {spin_index} has {len(cells)} cells but "
                f"{len(blocks)} blocks"
            )
        block_map = {
            tuple(int(value) for value in cell.index): np.asarray(block)
            for cell, block in zip(cells, blocks)
        }
        if len(block_map) != len(cells):
            raise ValueError(f"{label} channel {spin_index} has duplicate cells")
        if set(block_map) != ref_key_set:
            missing = sorted(ref_key_set.difference(block_map))
            extra = sorted(set(block_map).difference(ref_key_set))
            raise ValueError(
                f"{label} channels have different lattice-cell keys; "
                f"channel {spin_index} missing {missing} and adds {extra}"
            )
        block_maps.append(block_map)

    blocks = [
        _real_block_for_periodic_output(
            sum(block_map[key] for block_map in block_maps),
            label=f"{label} block g={key}",
        )
        for key in ref_keys
    ]
    if any(not np.isfinite(block).all() for block in blocks):
        raise ValueError(f"{label} contains non-finite summed density blocks")

    from ._vibeqc_core import make_lattice_matrix_set

    return make_lattice_matrix_set(int(basis.nbasis), ref_cells, blocks)


def _fit_exact_lattice_density_to_weighted_k_matrices(
    density_set,
    kpoints_cart: np.ndarray,
    kpoint_weights: np.ndarray,
    *,
    system: PeriodicSystem,
    bvk_mesh: Sequence[int],
    label: str,
) -> list[np.ndarray]:
    """Recover ``w_k P(k)`` from a returned BvK-periodic lattice density.

    This is a representation transform of the returned SCF density.  It uses
    one complete Born-von-Karman residue system (the exact inverse used by
    the BIPOLE Fock path), then refolds the recovered matrices onto *every*
    returned cell.  The transform is accepted only when that reconstruction
    agrees at a scale-aware floating-point tolerance; a truncated, corrupted,
    or non-BvK density therefore fails before any artifact is written.
    """
    kpts = np.asarray(kpoints_cart, dtype=float)
    weights = np.asarray(kpoint_weights, dtype=float).reshape(-1)
    if kpts.ndim != 2 or kpts.shape[1] != 3 or not np.isfinite(kpts).all():
        raise ValueError(
            f"{label} requires finite Cartesian k-points with shape (n_k, 3)"
        )
    if (
        weights.shape != (kpts.shape[0],)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError(
            f"{label} requires one finite positive weight per k-point"
        )
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{label} k-point weights sum to {float(weights.sum()):.16g}, "
            "not one"
        )
    mesh = tuple(int(value) for value in bvk_mesh)
    if len(mesh) != 3 or any(value < 1 for value in mesh):
        raise ValueError(
            f"{label} requires a verified three-axis BvK mesh; got {mesh}"
        )
    if int(np.prod(mesh)) != kpts.shape[0]:
        raise ValueError(
            f"{label} BvK mesh {mesh} contains {int(np.prod(mesh))} points "
            f"but the returned SCF metadata contains {kpts.shape[0]}"
        )
    uniform_weight = 1.0 / float(kpts.shape[0])
    if not np.allclose(weights, uniform_weight, rtol=0.0, atol=2.0e-13):
        raise ValueError(
            f"{label} BvK inversion requires uniform Monkhorst-Pack weights; "
            f"got {weights.tolist()}"
        )
    dim = max(0, min(3, int(getattr(system, "dim", 3))))
    lattice = np.asarray(system.lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError(f"{label} requires a finite 3x3 primitive lattice")
    fractional = (lattice.T @ kpts.T).T / (2.0 * np.pi)
    residues: list[np.ndarray] = []
    metadata_tolerance = 2.0e-10
    for axis, period in enumerate(mesh):
        wrapped = fractional[:, axis] - np.floor(fractional[:, axis])
        if axis >= dim:
            centered = ((fractional[:, axis] + 0.5) % 1.0) - 0.5
            if period != 1 or not np.allclose(
                centered,
                0.0,
                rtol=0.0,
                atol=metadata_tolerance,
            ):
                raise ValueError(
                    f"{label} has non-Gamma sampling on nonperiodic axis {axis}"
                )
            residues.append(np.zeros(kpts.shape[0], dtype=int))
            continue
        scaled = float(period) * fractional[:, axis]
        fractional_offsets = scaled - np.floor(scaled)
        distance_unshifted = np.minimum(
            fractional_offsets,
            1.0 - fractional_offsets,
        )
        is_unshifted = bool(np.all(distance_unshifted <= metadata_tolerance))
        is_half_shifted = bool(
            np.all(np.abs(fractional_offsets - 0.5) <= metadata_tolerance)
        )
        if not (is_unshifted or is_half_shifted):
            raise ValueError(
                f"{label} k-points are not a supported Monkhorst-Pack "
                f"character set on axis {axis}"
            )
        offset = 0.5 if is_half_shifted else 0.0
        residues.append(
            np.mod(np.rint(scaled - offset).astype(int), period)
        )
    residue_tuples = list(zip(*(axis.tolist() for axis in residues)))
    if len(set(residue_tuples)) != int(np.prod(mesh)):
        raise ValueError(
            f"{label} k-points do not contain every BvK residue of mesh "
            f"{mesh} exactly once"
        )
    cells = list(density_set.cells)
    blocks = [np.asarray(block, dtype=float) for block in density_set.blocks]
    if not cells or len(cells) != len(blocks):
        raise ValueError(f"{label} has misaligned or empty lattice blocks")
    expected_shape = (int(density_set.nbf), int(density_set.nbf))
    if any(block.shape != expected_shape for block in blocks):
        raise ValueError(f"{label} contains a block with the wrong basis shape")
    blocks_by_key = {
        tuple(int(value) for value in cell.index): block
        for cell, block in zip(cells, blocks)
    }
    if len(blocks_by_key) != len(cells):
        raise ValueError(f"{label} contains duplicate lattice cells")
    block_scale = max(1.0, max(float(np.max(np.abs(block))) for block in blocks))
    transpose_tolerance = 5.0e-11 * block_scale
    for key, block in blocks_by_key.items():
        inverse_key = tuple(-value for value in key)
        inverse_block = blocks_by_key.get(inverse_key)
        if inverse_block is None:
            raise ValueError(
                f"{label} lacks inverse lattice cell {inverse_key} required "
                "by a real time-reversal-symmetric density"
            )
        transpose_error = float(np.max(np.abs(inverse_block - block.T)))
        if transpose_error > transpose_tolerance:
            raise ValueError(
                f"{label} has a pre-refold residual in D(-g)=D(g).T at "
                f"cell {key}; max residual {transpose_error:.3e}"
            )

    from .pbc_bipole_common import bvk_torus_density_matrices

    recovered = [
        np.asarray(matrix, dtype=np.complex128)
        for matrix in bvk_torus_density_matrices(density_set, kpts, mesh)
    ]
    if len(recovered) != kpts.shape[0]:
        raise ValueError(
            f"{label} BvK inverse returned {len(recovered)} matrices for "
            f"{kpts.shape[0]} k-points"
        )
    for index, matrix in enumerate(recovered):
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(
                f"{label} recovered matrix k={index} is not finite with "
                f"shape {expected_shape}"
            )

    wrapped_fractional = fractional - np.floor(fractional)
    for index, (qpoint, matrix) in enumerate(zip(wrapped_fractional, recovered)):
        pair_distances = np.max(
            np.abs(
                ((wrapped_fractional + qpoint + 0.5) % 1.0) - 0.5
            ),
            axis=1,
        )
        pair_index = int(np.argmin(pair_distances))
        if float(pair_distances[pair_index]) > metadata_tolerance:
            raise ValueError(
                f"{label} k-point {index} has no time-reversal partner"
            )
        pair = recovered[pair_index]
        pair_scale = max(1.0, float(np.max(np.abs(matrix))))
        time_reversal_error = float(np.max(np.abs(pair - matrix.conj())))
        if time_reversal_error > 5.0e-10 * pair_scale:
            raise ValueError(
                f"{label} recovered k-point densities violate time reversal "
                f"at k={index}; max residual {time_reversal_error:.3e}"
            )
        matrix_scale = max(1.0, float(np.max(np.abs(matrix))))
        hermitian_error = float(np.max(np.abs(matrix - matrix.conj().T)))
        if hermitian_error > 5.0e-11 * matrix_scale:
            raise ValueError(
                f"{label} recovered matrix k={index} is not Hermitian; "
                f"max residual {hermitian_error:.3e}"
            )

    # The inverse/refold consists of O(n_k) complex sums.  This bound is
    # deliberately close to floating-point noise while allowing different
    # BLAS accumulation orders on supported platforms.
    tolerance = max(
        5.0e-12,
        256.0
        * np.finfo(float).eps
        * max(1, kpts.shape[0])
        * block_scale,
    )
    worst_error = 0.0
    worst_cell: tuple[int, int, int] | None = None
    for cell, actual in zip(cells, blocks):
        r_cart = np.asarray(cell.r_cart, dtype=float)
        predicted = np.zeros(expected_shape, dtype=float)
        for weight, kpoint, matrix in zip(weights, kpts, recovered):
            phase = float(np.dot(kpoint, r_cart))
            predicted += float(weight) * (
                np.cos(phase) * matrix.real + np.sin(phase) * matrix.imag
            )
        error = float(np.max(np.abs(predicted - actual)))
        if error > worst_error:
            worst_error = error
            worst_cell = tuple(int(value) for value in cell.index)
    if worst_error > tolerance:
        raise ValueError(
            f"{label} is not exactly representable by its returned BvK "
            f"k-point metadata: refold residual {worst_error:.3e} at cell "
            f"{worst_cell} exceeds {tolerance:.3e}"
        )
    return [float(weight) * matrix for weight, matrix in zip(weights, recovered)]


def _validated_periodic_density_electron_count(
    basis: BasisSet,
    system: PeriodicSystem,
    density_set,
    *,
    expected_electrons: float,
    tolerance: float = 5.0e-5,
) -> float:
    """Validate ``sum_g D(g):S(g)`` on the exact returned cell domain."""
    from ._vibeqc_core import compute_overlap_lattice_explicit

    cells = list(density_set.cells)
    overlap = compute_overlap_lattice_explicit(basis, system, cells)
    charge = float(
        sum(
            np.einsum(
                "ij,ij->",
                np.asarray(density_block, dtype=float),
                np.asarray(overlap_block, dtype=float),
            )
            for density_block, overlap_block in zip(
                density_set.blocks,
                overlap.blocks,
            )
        )
    )
    _require_artifact_electron_count(
        charge,
        expected_electrons,
        tolerance=tolerance,
        label="returned periodic density sum_g D(g):S(g)",
    )
    return charge


def _require_artifact_electron_count(
    actual: float,
    expected: float,
    *,
    tolerance: float,
    label: str,
) -> None:
    if not np.isfinite(float(expected)):
        raise ValueError(f"{label} expected electron count must be finite")
    if not np.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError(f"{label} tolerance must be finite and positive")
    if not np.isfinite(actual) or abs(float(actual) - float(expected)) > float(
        tolerance
    ):
        raise ValueError(
            f"{label}={float(actual):.10f}, expected {float(expected):.10f} "
            f"within {float(tolerance):.1e}"
        )


def _converged_periodic_density_grid_for_artifact(
    evaluate_at_radius,
    *,
    lattice_bohr: np.ndarray,
    grid_shape: tuple[int, int, int],
    analytic_electrons: float,
    label: str,
    max_image_radius: int = 3,
    charge_tolerance: float = 5.0e-3,
    image_l1_tolerance: float = 1.0e-5,
    image_max_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, object]:
    """Certify an artifact grid against AO-image and charge convergence."""
    lattice = np.asarray(lattice_bohr, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("periodic density certification requires a finite 3x3 lattice")
    volume = float(abs(np.linalg.det(lattice)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("periodic density certification requires nonzero cell volume")
    if len(grid_shape) != 3 or any(int(value) < 1 for value in grid_shape):
        raise ValueError("periodic density certification grid shape must be positive")
    if not np.isfinite(float(analytic_electrons)):
        raise ValueError("periodic density analytic electron count must be finite")
    if int(max_image_radius) < 2:
        raise ValueError(
            "periodic density certification requires max_image_radius >= 2"
        )
    for name, value in (
        ("charge_tolerance", charge_tolerance),
        ("image_l1_tolerance", image_l1_tolerance),
        ("image_max_tolerance", image_max_tolerance),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"periodic density {name} must be finite and positive")
    dV = volume / float(np.prod(grid_shape))
    previous: np.ndarray | None = None
    last_l1 = float("inf")
    last_max = float("inf")
    for radius in range(1, int(max_image_radius) + 1):
        rho_raw, metadata = evaluate_at_radius(radius)
        rho = np.asarray(rho_raw, dtype=float)
        if rho.shape != tuple(grid_shape) or not np.isfinite(rho).all():
            raise ValueError(
                f"{label} evaluator returned invalid grid shape {rho.shape}; "
                f"expected finite {tuple(grid_shape)}"
            )
        if previous is not None:
            delta = rho - previous
            last_l1 = float(np.sum(np.abs(delta)) * dV)
            last_max = float(np.max(np.abs(delta)))
            if (
                radius == int(max_image_radius)
                and last_l1 <= float(image_l1_tolerance)
                and last_max <= float(image_max_tolerance)
            ):
                grid_electrons = float(np.sum(rho) * dV)
                if (
                    not np.isfinite(grid_electrons)
                    or abs(grid_electrons - float(analytic_electrons))
                    > float(charge_tolerance)
                ):
                    raise ValueError(
                        f"{label} primitive-cell grid integral="
                        f"{grid_electrons:.10f}, expected "
                        f"{float(analytic_electrons):.10f} within "
                        f"{float(charge_tolerance):.1e}; decrease "
                        "density_spacing_bohr to refine the artifact grid"
                    )
                return rho, metadata
        previous = rho

    raise ValueError(
        f"{label} did not converge its periodic AO basis translations "
        f"through image radius {int(max_image_radius)}: successive-grid "
        f"L1={last_l1:.3e} electrons (limit "
        f"{float(image_l1_tolerance):.1e}), max|delta rho|={last_max:.3e} "
        f"electron/bohr^3 (limit {float(image_max_tolerance):.1e})"
    )


def _converged_lattice_density_grid_for_artifact(
    basis: BasisSet,
    system: PeriodicSystem,
    density_set,
    *,
    grid_shape: tuple[int, int, int],
    spacing_bohr: float,
    expected_electrons: float,
    max_image_radius: int = 3,
    charge_tolerance: float = 5.0e-3,
    image_l1_tolerance: float = 1.0e-5,
    image_max_tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate and certify a returned lattice density for an artifact.

    The finite AO-image sum is increased through radius three.  An artifact is
    emitted only when successive grids agree both pointwise and in integrated
    absolute density, and the final numerical primitive-cell integral agrees
    with the exact lattice trace on the same returned cells.
    """
    from .periodic_density import evaluate_periodic_density_on_grid

    analytic_electrons = _validated_periodic_density_electron_count(
        basis,
        system,
        density_set,
        expected_electrons=expected_electrons,
    )

    def _evaluate(radius: int) -> tuple[np.ndarray, None]:
        rho, resolved_shape = evaluate_periodic_density_on_grid(
            basis,
            system,
            density_set,
            grid_shape=grid_shape,
            spacing_bohr=spacing_bohr,
            ao_image_radius=radius,
        )
        if tuple(resolved_shape) != tuple(grid_shape):
            raise ValueError(
                "periodic density evaluator changed the requested artifact "
                f"grid shape from {tuple(grid_shape)} to {tuple(resolved_shape)}"
            )
        return rho, None

    rho, _ = _converged_periodic_density_grid_for_artifact(
        _evaluate,
        lattice_bohr=np.asarray(system.lattice, dtype=float),
        grid_shape=grid_shape,
        analytic_electrons=analytic_electrons,
        label="periodic density artifact",
        max_image_radius=max_image_radius,
        charge_tolerance=charge_tolerance,
        image_l1_tolerance=image_l1_tolerance,
        image_max_tolerance=image_max_tolerance,
    )
    return rho


def _converged_bvk_density_grid_for_artifact(
    basis: BasisSet,
    system: PeriodicSystem,
    density_set,
    kpoints_cart: np.ndarray,
    kpoint_weights: np.ndarray,
    bvk_mesh: Sequence[int],
    *,
    grid_shape: tuple[int, int, int],
    spacing_bohr: float,
    expected_electrons: float,
    max_image_radius: int = 3,
    charge_tolerance: float = 5.0e-3,
    image_l1_tolerance: float = 1.0e-5,
    image_max_tolerance: float = 1.0e-6,
    component_density_sets: Sequence[object] | None = None,
    overlap_per_k: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate a refold-certified returned BvK density via Bloch AOs."""
    from .periodic_density import evaluate_weighted_k_density_on_grid

    components = list(component_density_sets or (density_set,))
    weighted_components = [
        _fit_exact_lattice_density_to_weighted_k_matrices(
            component,
            kpoints_cart,
            kpoint_weights,
            system=system,
            bvk_mesh=bvk_mesh,
            label=(
                "returned periodic density"
                if len(components) == 1
                else f"returned periodic spin density channel {index}"
            ),
        )
        for index, component in enumerate(components)
    ]
    weighted_density = [
        sum(component[k_index] for component in weighted_components)
        for k_index in range(len(weighted_components[0]))
    ]
    if overlap_per_k is None:
        analytic_electrons = _validated_periodic_density_electron_count(
            basis, system, density_set, expected_electrons=expected_electrons,
        )
    else:
        # A complete BvK residue set represents the infinite periodic density;
        # its finite cell list is not a real-space overlap truncation. Use the
        # producer's accepted Bloch overlap for the analytic charge. The grid
        # still independently converges AO images and certifies this charge.
        overlaps = np.asarray(overlap_per_k)
        expected_shape = (len(weighted_density), int(basis.nbasis), int(basis.nbasis))
        if overlaps.shape != expected_shape or not np.isfinite(overlaps).all():
            raise ValueError("returned BvK density requires aligned finite Bloch overlaps")
        charge = sum(
            np.einsum("mn,nm->", density, overlap)
            for density, overlap in zip(weighted_density, overlaps)
        )
        if abs(charge.imag) > 5e-11:
            raise ValueError("returned BvK density has a complex electron count")
        analytic_electrons = float(charge.real)
        _require_artifact_electron_count(
            analytic_electrons, expected_electrons, tolerance=5e-5,
            label="returned periodic density sum_k w_k Tr(D(k) S(k))",
        )

    def _evaluate(radius: int) -> tuple[np.ndarray, None]:
        rho, resolved_shape = evaluate_weighted_k_density_on_grid(
            basis,
            system,
            weighted_density,
            kpoints_cart,
            grid_shape=grid_shape,
            spacing_bohr=spacing_bohr,
            ao_image_radius=radius,
        )
        if tuple(resolved_shape) != tuple(grid_shape):
            raise ValueError(
                "weighted-k density evaluator changed the requested artifact "
                f"grid shape from {tuple(grid_shape)} to {tuple(resolved_shape)}"
            )
        return rho, None

    rho, _ = _converged_periodic_density_grid_for_artifact(
        _evaluate,
        lattice_bohr=np.asarray(system.lattice, dtype=float),
        grid_shape=grid_shape,
        analytic_electrons=analytic_electrons,
        label="periodic BvK density artifact",
        max_image_radius=max_image_radius,
        charge_tolerance=charge_tolerance,
        image_l1_tolerance=image_l1_tolerance,
        image_max_tolerance=image_max_tolerance,
    )
    return rho


def _converged_gamma_density_grid_for_artifact(
    basis: BasisSet,
    system: PeriodicSystem,
    density: np.ndarray,
    overlap: np.ndarray,
    lattice_bohr: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    expected_electrons: float,
    max_image_radius: int = 3,
    charge_tolerance: float = 5.0e-3,
    image_l1_tolerance: float = 1.0e-5,
    image_max_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and certify an actual returned Gamma density matrix."""
    expected_shape = (int(basis.nbasis), int(basis.nbasis))
    D = _real_block_for_periodic_output(
        density,
        label="periodic Gamma density",
    )
    S = _real_block_for_periodic_output(
        overlap,
        label="periodic Gamma overlap",
    )
    if D.shape != expected_shape or S.shape != expected_shape:
        raise ValueError(
            "exact Gamma density artifacts require returned density and "
            f"overlap matrices of shape {expected_shape}; got D{D.shape}, "
            f"S{S.shape}"
        )
    if not np.isfinite(D).all() or not np.isfinite(S).all():
        raise ValueError(
            "exact Gamma density artifacts require finite returned density "
            "and overlap matrices"
        )
    analytic_electrons = float(np.einsum("ij,ij->", D, S))
    _require_artifact_electron_count(
        analytic_electrons,
        expected_electrons,
        tolerance=5.0e-5,
        label="returned Gamma density D:S",
    )

    def _evaluate(radius: int) -> tuple[np.ndarray, np.ndarray]:
        return _evaluate_density_matrix_on_lattice_grid(
            D,
            basis,
            lattice_bohr,
            grid_shape,
            system=system,
            ao_image_radius=radius,
        )

    rho, voxel_vectors = _converged_periodic_density_grid_for_artifact(
        _evaluate,
        lattice_bohr=lattice_bohr,
        grid_shape=grid_shape,
        analytic_electrons=analytic_electrons,
        label="Gamma density artifact",
        max_image_radius=max_image_radius,
        charge_tolerance=charge_tolerance,
        image_l1_tolerance=image_l1_tolerance,
        image_max_tolerance=image_max_tolerance,
    )
    return rho, np.asarray(voxel_vectors, dtype=float)


def _exact_periodic_density_grid_artifact(
    basis: BasisSet,
    system: PeriodicSystem,
    result,
    *,
    lattice_bohr: np.ndarray,
    grid_shape: tuple[int, int, int],
    spacing_bohr: float,
    gamma_only: bool,
    expected_electrons: float,
    bvk_mesh: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the returned SCF density without reconstructing its state."""
    lattice_arr = np.asarray(lattice_bohr, dtype=float)
    system_lattice = np.asarray(system.lattice, dtype=float)
    if (
        lattice_arr.shape != (3, 3)
        or not np.isfinite(lattice_arr).all()
        or not np.allclose(lattice_arr, system_lattice, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(
            "periodic density artifact lattice must equal the system's finite "
            "column-vector primitive lattice"
        )
    if not np.isfinite(float(expected_electrons)):
        raise ValueError("periodic density artifact electron count must be finite")
    voxel_vectors = lattice_arr.T / np.asarray(
        grid_shape,
        dtype=float,
    )[:, None]
    # A channel counts as present only when it carries data.  The
    # supercell-Gamma CCM results (real-gamma, four-center) are closed-shell
    # dataclasses that *declare* both spin fields as None; testing for the
    # attribute alone routed them into the open-shell branch, which then
    # discarded the LatticeMatrixSet in ``result.density`` in favour of two
    # Nones and refused the completed calculation (#679).
    has_alpha = getattr(result, "density_alpha", None) is not None
    has_beta = getattr(result, "density_beta", None) is not None
    if has_alpha != has_beta:
        raise ValueError(
            "periodic density artifact received only one spin density channel"
        )
    lattice_density = getattr(result, "density_lattice", None)
    lattice_alpha = getattr(result, "density_alpha_lattice", None)
    lattice_beta = getattr(result, "density_beta_lattice", None)
    has_lattice_alpha = lattice_alpha is not None
    has_lattice_beta = lattice_beta is not None
    if has_lattice_alpha != has_lattice_beta:
        raise ValueError(
            "periodic density artifact received only one returned spin "
            "lattice density channel"
        )
    if has_alpha:
        if lattice_density is not None:
            raise ValueError(
                "periodic density artifact received both restricted and "
                "spin-resolved returned lattice densities"
            )
        densities = (
            [lattice_alpha, lattice_beta]
            if has_lattice_alpha
            else [result.density_alpha, result.density_beta]
        )
        labels = ["periodic alpha density", "periodic beta density"]
    else:
        if has_lattice_alpha:
            raise ValueError(
                "periodic density artifact received spin lattice densities "
                "without spin-resolved SCF density channels"
            )
        returned = (
            lattice_density
            if lattice_density is not None
            else getattr(result, "density", None)
        )
        if returned is None:
            raise ValueError("SCF result has no returned periodic density")
        densities = [returned]
        labels = ["periodic density"]

    lattice_flags = [_is_lattice_matrix_set_like(item) for item in densities]
    if any(lattice_flags) and not all(lattice_flags):
        raise ValueError(
            "periodic density artifact received different alpha/beta density "
            "representations"
        )
    if all(lattice_flags):
        exact = [
            _exact_lattice_density_set_for_grid_artifact(
                basis,
                system,
                item,
                label=label,
            )
            for item, label in zip(densities, labels)
        ]
        total = exact[0]
        if len(exact) == 2:
            total = _sum_exact_lattice_density_sets_for_grid_artifact(
                basis,
                *exact,
                label="periodic spin density",
            )
        kpoints_cart = _result_kpoints_cart(result)
        weights = _result_kpoint_weights(result)
        # The BvK triple is metadata *about a Bloch mesh*.  A real-space
        # supercell-Gamma result (variant="real-gamma", "four-center") carries
        # no k-points at all -- the repetition count the runner resolved from
        # aiccm_lattice_extension describes its supercell, not a k-mesh -- so
        # a lone bvk_mesh there describes nothing and must not make the triple
        # look half-supplied.  The returned lattice density is already the
        # real-space representation these routes converged (#679).
        if kpoints_cart is None and weights is None:
            bvk_mesh = None
        metadata_count = sum(
            item is not None for item in (kpoints_cart, weights, bvk_mesh)
        )
        if metadata_count not in (0, 3):
            raise ValueError(
                "exact periodic lattice density artifacts require aligned "
                "kpoints_cart, kpoint_weights, and verified BvK mesh metadata"
            )
        if metadata_count == 3:
            assert kpoints_cart is not None and weights is not None
            # Legacy Gamma-local BIPOLE densities are a finite-support real-
            # space representation, not a BvK-periodic inverse fold.  Their
            # exact grid remains the direct returned-block contraction.
            gamma_local = bool(
                getattr(result, "density_lattice_mesh", None) is None
                and
                kpoints_cart.shape == (1, 3)
                and float(np.linalg.norm(kpoints_cart[0])) <= 1.0e-12
                and tuple(int(value) for value in bvk_mesh) == (1, 1, 1)
                and all(
                    tuple(int(value) for value in cell.index) == (0, 0, 0)
                    or not np.any(np.asarray(block) != 0.0)
                    for cell, block in zip(total.cells, total.blocks)
                )
            )
            if not gamma_local:
                return (
                    _converged_bvk_density_grid_for_artifact(
                        basis,
                        system,
                        total,
                        kpoints_cart,
                        weights,
                        bvk_mesh,
                        grid_shape=grid_shape,
                        spacing_bohr=spacing_bohr,
                        expected_electrons=expected_electrons,
                        component_density_sets=exact,
                        overlap_per_k=(
                            getattr(result, "overlap", None)
                            if getattr(result, "density_lattice_mesh", None)
                            == tuple(bvk_mesh) else None
                        ),
                    ),
                    voxel_vectors,
                )
        live_blocks = sum(
            bool(np.any(np.asarray(block) != 0.0)) for block in total.blocks
        )
        if metadata_count == 0 and live_blocks > 128:
            raise ValueError(
                "exact periodic density artifact has no verified BvK k-point "
                f"metadata and {live_blocks} nonzero returned lattice blocks; "
                "refusing the non-scalable direct grid evaluation"
            )
        return (
            _converged_lattice_density_grid_for_artifact(
                basis,
                system,
                total,
                grid_shape=grid_shape,
                spacing_bohr=spacing_bohr,
                expected_electrons=expected_electrons,
            ),
            voxel_vectors,
        )

    if not gamma_only:
        raise ValueError(
            "exact true-multi-k periodic density artifacts require a returned "
            "lattice-cell density; refusing to reconstruct one from per-k "
            "matrices"
        )
    matrices = [np.asarray(item) for item in densities]
    overlap = np.asarray(getattr(result, "overlap", None))
    gamma_blocks = [
        _single_gamma_block_for_periodic_artifact(item, result=result)
        for item in matrices
    ]
    gamma_overlap = _single_gamma_block_for_periodic_artifact(
        overlap, result=result
    )
    if all(block is not None for block in gamma_blocks) and (
        gamma_overlap is not None
    ):
        matrices = gamma_blocks
        overlap = gamma_overlap
    if any(matrix.ndim != 2 for matrix in matrices) or overlap.ndim != 2:
        raise ValueError(
            "exact Gamma density artifacts require returned rank-2 density "
            "and overlap matrices"
        )
    return _converged_gamma_density_grid_for_artifact(
        basis,
        system,
        sum(matrices),
        overlap,
        lattice_bohr,
        grid_shape,
        expected_electrons=expected_electrons,
    )


def _sum_lattice_density_sets_for_output(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts,
    *density_sets,
    label: str,
):
    """Add spin-resolved lattice density sets for real output artifacts."""
    if not density_sets:
        raise ValueError("no density lattice sets supplied")
    ref = density_sets[0]
    ref_cells = list(ref.cells)
    ref_blocks = list(ref.blocks)
    if len(ref_cells) != len(ref_blocks):
        raise ValueError("spin density lattice set has misaligned cells and blocks")
    ref_keys = [tuple(int(value) for value in cell.index) for cell in ref_cells]
    for other in density_sets[1:]:
        other_cells = list(other.cells)
        other_blocks = list(other.blocks)
        if len(other_cells) != len(other_blocks):
            raise ValueError(
                "spin density lattice set has misaligned cells and blocks"
            )
        other_keys = [
            tuple(int(value) for value in cell.index) for cell in other_cells
        ]
        if other_keys != ref_keys:
            raise ValueError(
                "spin density lattice sets have different keyed cell order"
            )

    from vibeqc.pbc_bipole_common import _copy_lattice_with_blocks

    blocks = []
    for i, parts in enumerate(zip(*(list(ds.blocks) for ds in density_sets))):
        cell = ref_cells[i]
        combined = sum(np.asarray(part) for part in parts)
        blocks.append(
            _real_block_for_periodic_output(
                combined,
                label=f"{label} block g={tuple(cell.index)}",
            )
        )
    # BUG 37 fix: the legacy template from compute_overlap_lattice may include
    # additional cells (e.g. in odd-electron DOS preparation).
    # fill_missing=True zero-fills those cells instead of raising.
    return _copy_lattice_with_blocks(
        basis,
        system,
        lat_opts,
        ref.cells,
        blocks,
        fill_missing=True,
    )


def _evaluate_density_matrix_on_lattice_grid(
    density: np.ndarray,
    basis: BasisSet,
    lattice_bohr: np.ndarray,
    shape: tuple[int, int, int],
    *,
    system: PeriodicSystem | None = None,
    ao_image_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a one-cell AO density matrix on a lattice-spanning grid.

    ``PeriodicSystem.lattice`` stores lattice vectors as columns. QVF grids
    store per-voxel vectors as rows, so the returned ``voxel_vectors`` are
    ``lattice.T / shape`` and ``shape[i] * voxel_vectors[i]`` exactly spans
    lattice vector ``i``.
    """
    L_bohr = np.asarray(lattice_bohr, dtype=float)
    if L_bohr.shape != (3, 3):
        raise ValueError("periodic QVF lattice grid requires a 3x3 lattice")
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    if min(nx, ny, nz) < 1:
        raise ValueError("periodic QVF lattice grid shape must be positive")
    voxel_vectors = L_bohr.T / np.array([nx, ny, nz], dtype=float)[:, None]
    D = _real_block_for_periodic_output(density, label="periodic QVF density")
    if D.ndim != 2:
        raise ValueError(
            "periodic QVF density matrix must be rank 2; "
            f"got shape {D.shape!r}"
        )

    n_points = nx * ny * nz
    rho_flat = np.empty(n_points, dtype=float)
    chunk_size = 200_000
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        linear = np.arange(start, stop, dtype=np.int64)
        ix = linear // (ny * nz)
        iy = (linear // nz) % ny
        iz = linear % nz
        frac = np.column_stack(
            [
                ix.astype(float) / float(nx),
                iy.astype(float) / float(ny),
                iz.astype(float) / float(nz),
            ]
        )
        points = frac @ L_bohr.T
        chi = _evaluate_ao_for_periodic_qvf(
            basis,
            system,
            points,
            image_radius=ao_image_radius,
        )
        rho_flat[start:stop] = np.einsum("mi,ij,mj->m", chi, D, chi)
    return rho_flat.reshape((nx, ny, nz)), voxel_vectors


def _evaluate_ao_for_periodic_qvf(
    basis: BasisSet,
    system: PeriodicSystem | None,
    points: np.ndarray,
    *,
    image_radius: int,
) -> np.ndarray:
    """Evaluate AO values with image sums on periodic axes for QVF grids."""
    from ._vibeqc_core import evaluate_ao

    if system is None or int(image_radius) <= 0:
        return evaluate_ao(basis, points)

    import itertools

    from .ewald_j import get_shifted_basis

    dim = max(0, min(3, int(getattr(system, "dim", 3))))
    ranges = [
        range(-int(image_radius), int(image_radius) + 1) if axis < dim else (0,)
        for axis in range(3)
    ]
    chi = evaluate_ao(basis, points)
    for shift in itertools.product(*ranges):
        if shift == (0, 0, 0):
            continue
        shifted = get_shifted_basis(basis, system, shift)
        chi += evaluate_ao(shifted, points)
    return chi


def _evaluate_orbital_on_lattice_grid(
    coefficients: np.ndarray,
    basis: BasisSet,
    lattice_bohr: np.ndarray,
    shape: tuple[int, int, int],
    *,
    system: PeriodicSystem | None = None,
    ao_image_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one AO coefficient vector on a lattice-spanning grid."""
    L_bohr = np.asarray(lattice_bohr, dtype=float)
    if L_bohr.shape != (3, 3):
        raise ValueError("periodic QVF lattice grid requires a 3x3 lattice")
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    if min(nx, ny, nz) < 1:
        raise ValueError("periodic QVF lattice grid shape must be positive")
    voxel_vectors = L_bohr.T / np.array([nx, ny, nz], dtype=float)[:, None]
    coeff = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    if coeff.shape[0] != int(basis.nbasis):
        raise ValueError(
            "periodic QVF orbital coefficient length "
            f"{coeff.shape[0]} does not match basis size {int(basis.nbasis)}"
        )

    n_points = nx * ny * nz
    psi_flat = np.empty(n_points, dtype=np.complex128)
    chunk_size = 200_000
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        linear = np.arange(start, stop, dtype=np.int64)
        ix = linear // (ny * nz)
        iy = (linear // nz) % ny
        iz = linear % nz
        frac = np.column_stack(
            [
                ix.astype(float) / float(nx),
                iy.astype(float) / float(ny),
                iz.astype(float) / float(nz),
            ]
        )
        points = frac @ L_bohr.T
        chi = _evaluate_ao_for_periodic_qvf(
            basis,
            system,
            points,
            image_radius=ao_image_radius,
        )
        psi_flat[start:stop] = chi @ coeff
    psi = _real_block_for_periodic_output(
        psi_flat.reshape((nx, ny, nz)),
        label="periodic QVF orbital grid",
    )
    return psi, voxel_vectors


def _aiccm_b_qvf_translations(mesh: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [
        (int(i), int(j), int(k))
        for i in range(int(mesh[0]))
        for j in range(int(mesh[1]))
        for k in range(int(mesh[2]))
    ]


def _aiccm_b_qvf_supercell_system(
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
) -> PeriodicSystem:
    """Build the visual BvK supercell used by χ-CCM-B QVF output."""
    lattice = np.asarray(system.lattice, dtype=float)
    mesh_arr = np.asarray(mesh, dtype=float)
    n_cells = int(np.prod(mesh))
    super_lattice = lattice * mesh_arr[None, :]
    atoms: list[Atom] = []
    for translation in _aiccm_b_qvf_translations(mesh):
        shift = lattice @ np.asarray(translation, dtype=float)
        for atom in system.unit_cell:
            position = np.asarray(atom.xyz, dtype=float) + shift
            atoms.append(Atom(int(atom.Z), position.tolist()))
    return PeriodicSystem(
        int(system.dim),
        super_lattice,
        atoms,
        charge=int(system.charge) * n_cells,
        multiplicity=n_cells * (int(system.multiplicity) - 1) + 1,
    )


def _aiccm_b_residue_density_blocks_for_qvf(
    density_like,
    mesh: tuple[int, int, int],
    *,
    label: str,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Fold lattice-set density aliases to one block per cyclic residue."""
    residues = _aiccm_b_qvf_translations(mesh)
    if not _is_lattice_matrix_set_like(density_like):
        if tuple(mesh) != (1, 1, 1):
            raise ValueError(
                f"{label} is not a lattice matrix set for χ-CCM-B mesh {mesh!r}"
            )
        return {
            (0, 0, 0): _real_block_for_periodic_output(
                density_like,
                label=label,
            )
        }

    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {
        residue: [] for residue in residues
    }
    mesh_arr = np.asarray(mesh, dtype=int)
    for cell, block in zip(density_like.cells, density_like.blocks):
        index = np.asarray(cell.index, dtype=int)
        residue = tuple(int(x) for x in np.mod(index, mesh_arr))
        if residue in buckets:
            buckets[residue].append(
                _real_block_for_periodic_output(
                    block,
                    label=f"{label} residue {residue}",
                )
            )
    missing = [residue for residue, blocks in buckets.items() if not blocks]
    if missing:
        raise ValueError(
            f"{label} is missing cyclic density residues {missing!r}"
        )
    return {
        residue: np.mean(np.stack(blocks, axis=0), axis=0)
        for residue, blocks in buckets.items()
    }


def _aiccm_b_density_alias_can_fold_directly(
    density_like,
    mesh: tuple[int, int, int],
) -> bool:
    if _is_lattice_matrix_set_like(density_like):
        return True
    return tuple(mesh) == (1, 1, 1) and np.asarray(density_like).ndim == 2


def _aiccm_b_effective_electrons_for_qvf(result) -> int:
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    value = getattr(result, "effective_n_electrons", None)
    if value is None and diagnostics is not None:
        value = getattr(diagnostics, "effective_electron_count", None)
    if value is None:
        raise TypeError(
            "χ-CCM-B QVF density folding requires an effective electron count"
        )
    return int(value)


def _aiccm_b_spin_occupations_for_qvf(
    system: PeriodicSystem,
    effective_electrons: int,
) -> tuple[int, int]:
    two_s = int(system.multiplicity) - 1
    alpha_twice = int(effective_electrons) + two_s
    beta_twice = int(effective_electrons) - two_s
    if alpha_twice < 0 or beta_twice < 0 or alpha_twice % 2 or beta_twice % 2:
        raise ValueError(
            "χ-CCM-B QVF density folding requires the effective electron "
            "count and multiplicity to define integer alpha/beta occupations"
        )
    return alpha_twice // 2, beta_twice // 2


def _aiccm_b_residue_density_blocks_from_k_for_qvf(
    result,
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
) -> dict[tuple[int, int, int], np.ndarray]:
    """Fold stored B k-density blocks to one block per cyclic residue."""
    effective_electrons = _aiccm_b_effective_electrons_for_qvf(result)
    _b_alpha, _b_beta = spin_densities(result)
    if _b_alpha is not None:
        n_alpha, n_beta = _aiccm_b_spin_occupations_for_qvf(
            system,
            effective_electrons,
        )
        alpha = _spin_density_blocks_per_k(result, "alpha", n_alpha)
        beta = _spin_density_blocks_per_k(result, "beta", n_beta)
        if len(alpha) != len(beta):
            raise ValueError("χ-CCM-B QVF alpha/beta density block counts differ")
        density_k = [a + b for a, b in zip(alpha, beta)]
    else:
        if int(system.multiplicity) != 1 or int(effective_electrons) % 2:
            raise ValueError(
                "χ-CCM-B QVF restricted density folding requires a singlet "
                "record with an even effective electron count"
            )
        density_k = _density_blocks_per_k(result, effective_electrons)

    kpoints_frac = np.asarray(getattr(result, "kpoints_frac"), dtype=float).reshape(
        -1,
        3,
    )
    weights = np.asarray(getattr(result, "kpoint_weights"), dtype=float).reshape(-1)
    if len(density_k) != kpoints_frac.shape[0] or weights.shape != (
        kpoints_frac.shape[0],
    ):
        raise ValueError(
            "χ-CCM-B QVF density folding requires matching density, k-point, "
            "and weight counts"
        )
    residues = _aiccm_b_qvf_translations(mesh)
    blocks = inverse_bloch_transform(density_k, kpoints_frac, residues, weights)
    imaginary_residual = float(np.max(np.abs(blocks.imag))) if blocks.size else 0.0
    if imaginary_residual > 1.0e-7:
        raise NotImplementedError(
            "χ-CCM-B QVF density folding would discard a non-real density "
            f"residue ({imaginary_residual:.3e})"
        )
    return {
        residue: np.ascontiguousarray(block, dtype=float)
        for residue, block in zip(residues, blocks.real)
    }


def _aiccm_b_full_density_matrix_for_qvf(
    result,
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
) -> np.ndarray:
    """Expand χ-CCM-B residue density blocks to a full BvK AO matrix."""

    def full_from_blocks(blocks_by_residue: dict[tuple[int, int, int], np.ndarray]):
        residues = _aiccm_b_qvf_translations(mesh)
        nbf = int(next(iter(blocks_by_residue.values())).shape[0])
        full = np.zeros((len(residues) * nbf, len(residues) * nbf), dtype=float)
        for origin_index, origin in enumerate(residues):
            row = slice(origin_index * nbf, (origin_index + 1) * nbf)
            for target_index, target in enumerate(residues):
                delta = tuple(
                    (int(target[axis]) - int(origin[axis])) % int(mesh[axis])
                    for axis in range(3)
                )
                col = slice(target_index * nbf, (target_index + 1) * nbf)
                full[row, col] = blocks_by_residue[delta]
        return full

    _fold_alpha, _fold_beta = spin_densities(result)
    if _fold_alpha is not None:
        if _aiccm_b_density_alias_can_fold_directly(
            _fold_alpha,
            mesh,
        ):
            alpha = _aiccm_b_residue_density_blocks_for_qvf(
                result.density_alpha,
                mesh,
                label="periodic QVF alpha density",
            )
            beta = _aiccm_b_residue_density_blocks_for_qvf(
                result.density_beta,
                mesh,
                label="periodic QVF beta density",
            )
            return full_from_blocks(alpha) + full_from_blocks(beta)
        return full_from_blocks(
            _aiccm_b_residue_density_blocks_from_k_for_qvf(
                result,
                system,
                mesh,
            )
        )
    density_like = getattr(result, "density", None)
    if _aiccm_b_density_alias_can_fold_directly(density_like, mesh):
        density = _aiccm_b_residue_density_blocks_for_qvf(
            density_like,
            mesh,
            label="periodic QVF density",
        )
    else:
        density = _aiccm_b_residue_density_blocks_from_k_for_qvf(
            result,
            system,
            mesh,
        )
    return full_from_blocks(density)


def _aiccm_b_qvf_gamma_orbital_grids(
    result,
    super_basis: BasisSet,
    super_system: PeriodicSystem,
    lattice_bohr: np.ndarray,
    shape: tuple[int, int, int],
    mesh: tuple[int, int, int],
) -> list[dict[str, object]]:
    """Return Γ-character HOMO/LUMO grids in the χ-CCM-B BvK supercell."""
    mo_coeffs = getattr(result, "mo_coeffs", None)
    if not isinstance(mo_coeffs, (list, tuple)) or not mo_coeffs:
        return []
    if hasattr(result, "mo_coeffs_alpha") or hasattr(result, "mo_coeffs_beta"):
        return []

    kfrac = getattr(result, "kpoints_frac", None)
    if kfrac is None:
        return []
    kfrac_arr = np.asarray(kfrac, dtype=float).reshape(-1, 3)
    if kfrac_arr.shape[0] != len(mo_coeffs):
        raise ValueError(
            "aiccm2026dev-b QVF orbital grids require aligned kpoints_frac "
            f"({kfrac_arr.shape[0]}) and mo_coeffs ({len(mo_coeffs)})"
        )
    wrapped = _wrap_reciprocal_fractional(kfrac_arr)
    gamma_idx = int(np.argmin(np.linalg.norm(wrapped, axis=1)))
    if float(np.linalg.norm(wrapped[gamma_idx])) > 1.0e-12:
        raise ValueError(
            "aiccm2026dev-b QVF orbital grids require an explicit Gamma "
            "character in the finite mesh"
        )

    C_gamma = np.asarray(mo_coeffs[gamma_idx], dtype=np.complex128)
    if C_gamma.ndim != 2:
        raise ValueError(
            "aiccm2026dev-b QVF Gamma coefficient block must be rank 2; "
            f"got shape {C_gamma.shape!r}"
        )
    n_ao_cell, n_bands = C_gamma.shape
    n_cells = int(np.prod(np.asarray(mesh, dtype=int)))
    if int(super_basis.nbasis) != n_cells * n_ao_cell:
        raise ValueError(
            "aiccm2026dev-b QVF supercell basis size does not match "
            f"mesh*AO count: {int(super_basis.nbasis)} != {n_cells}*{n_ao_cell}"
        )

    diag = getattr(result, "aiccm2026dev_b", None)
    n_electrons = getattr(result, "effective_n_electrons", None)
    if n_electrons is None and diag is not None:
        n_electrons = getattr(diag, "effective_electron_count", None)
    try:
        n_occ = int(n_electrons) // 2
    except Exception:
        n_occ = 0
    indices: list[int] = []
    if 0 < n_occ <= n_bands:
        indices.append(n_occ - 1)
    if n_occ < n_bands:
        indices.append(n_occ)
    if not indices and n_bands:
        indices.append(0)

    energies = getattr(result, "mo_energies", None)
    if isinstance(energies, (list, tuple)) and len(energies) > gamma_idx:
        eps_gamma = np.asarray(energies[gamma_idx], dtype=float).reshape(-1)
    elif energies is not None:
        eps_gamma = np.asarray(energies, dtype=float).reshape(-1)
    else:
        eps_gamma = np.zeros(n_bands, dtype=float)

    translations = _aiccm_b_qvf_translations(mesh)
    norm = 1.0 / np.sqrt(float(n_cells))
    out: list[dict[str, object]] = []
    for band_index in indices:
        column = np.array(C_gamma[:, band_index], dtype=np.complex128, copy=True)
        if column.size:
            pivot = int(np.argmax(np.abs(column)))
            if abs(column[pivot]) > 0.0:
                column *= np.exp(-1j * float(np.angle(column[pivot])))
        coeff = np.zeros(int(super_basis.nbasis), dtype=np.complex128)
        for cell_index, _translation in enumerate(translations):
            start = cell_index * n_ao_cell
            stop = start + n_ao_cell
            coeff[start:stop] = norm * column
        psi, voxel_vectors = _evaluate_orbital_on_lattice_grid(
            coeff,
            super_basis,
            lattice_bohr,
            shape,
            system=super_system,
        )
        if band_index == n_occ - 1:
            role = "HOMO"
        elif band_index == n_occ:
            role = "LUMO"
        else:
            role = f"band {band_index}"
        out.append(
            {
                "label": f"chi-CCM Gamma {role}",
                "data": psi,
                "origin": np.zeros(3, dtype=float),
                "span": voxel_vectors,
                "band_index": int(band_index),
                "energy_eh": (
                    float(eps_gamma[band_index])
                    if band_index < eps_gamma.shape[0]
                    else 0.0
                ),
                "occupation": 2.0 if band_index < n_occ else 0.0,
                "spin": "both",
                "component": "real",
            }
        )
    return out


def _json_safe_qvf_value(value):
    """Convert small diagnostic payloads to JSON-safe Python values."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return _json_safe_qvf_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe_qvf_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_safe_qvf_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_qvf_value(item) for item in value]
    return value


def _real_gamma_qvf_vendor_sections(result) -> list[dict[str, object]]:
    """QVF vendor metadata for the real-Γ direct-route control (EXPERIMENTAL).

    Carries the :class:`~vibeqc.periodic.ccm.real_gamma_runner.CCMRealGammaConvention`
    record verbatim -- the 2d convention fields (``exchange_q0`` +
    applicability, the D86 executed values, ``lattice_vector_convention``,
    the screened-exchange assembly triple for HSE-class runs).
    """
    convention = getattr(result, "real_gamma", None)
    if convention is None:
        return []
    payload = {
        "schema_version": 1,
        "method_selector": "real-gamma",
        "prose_name": "neutral fitted-torus real-Gamma control",
        # D89/D90 terminology, load-bearing: a representation control of
        # the neutral fitted-torus Hamiltonian; not the union-and-weight
        # Γ-CCM construction, and not construction evidence for it.
        "ccm_construction": "none (representation control)",
        "evaluation_representation": "real supercell-Gamma eigenproblem",
        "family": "neutral fitted-torus BvK control",
        "result_backend": getattr(result, "backend", None),
        "electronic_method": getattr(result, "functional", None),
        "experimental": True,
        "real_gamma_convention": _json_safe_qvf_value(convention),
    }
    return [
        {
            "id": "x_vibeqc_real_gamma_convention",
            "kind": "x_vibeqc.real_gamma_convention",
            "member": "convention",
            "label": "real-Gamma neutral fitted-torus convention",
            "payload": _json_safe_qvf_value(payload),
        }
    ]


def _neutral_bloch_qvf_vendor_sections(result) -> list[dict[str, object]]:
    """QVF vendor metadata for the neutral Γ-CCM Bloch producer (EXPERIMENTAL).

    The sibling of :func:`_real_gamma_qvf_vendor_sections`. The two producers
    evaluate the SAME neutral finite-BvK-torus Hamiltonian (ruling R1), so the
    ``ccm_construction`` label is deliberately byte-identical to the real-Γ
    one; only ``evaluation_representation`` differs.
    """
    convention = getattr(result, "neutral_bloch", None)
    if convention is None:
        return []
    payload = {
        "schema_version": 1,
        "method_selector": "neutral-bloch",
        "prose_name": "neutral fitted-torus Bloch producer",
        "ccm_construction": "none (representation control)",
        "evaluation_representation": (
            "Bloch representation: Gamma-centred BvK k-mesh eigenproblems "
            "(multi-k GDF)"
        ),
        "family": "neutral fitted-torus BvK control",
        "result_backend": getattr(result, "backend", None),
        "electronic_method": getattr(result, "functional", None),
        "experimental": True,
        "neutral_bloch_convention": _json_safe_qvf_value(convention),
    }
    return [
        {
            "id": "x_vibeqc_neutral_bloch_convention",
            "kind": "x_vibeqc.neutral_bloch_convention",
            "member": "convention",
            "label": "neutral fitted-torus Bloch convention",
            "payload": _json_safe_qvf_value(payload),
        }
    ]


def _aiccm_b_qvf_vendor_sections(result) -> list[dict[str, object]]:
    """Return first-party QVF vendor metadata sections for χ-CCM-B output."""
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    convention = getattr(diagnostics, "finite_torus_convention", None)
    if convention is None:
        return []
    exact_exchange_assembly = getattr(
        diagnostics,
        "exact_exchange_assembly",
        None,
    )
    if exact_exchange_assembly is None:
        raise RuntimeError(
            "χ-CCM-B QVF convention metadata requires the versioned "
            "exact-exchange assembly"
        )
    payload = {
        "schema_version": 2,
        "method_selector": "aiccm2026dev-b",
        "prose_name": "chi-CCM",
        "ccm_approach": convention.ccm_approach,
        "ccm_construction": convention.ccm_construction,
        "evaluation_representation": convention.evaluation_representation,
        # Retained verbatim from vendor schema v1.  The exact machine identity
        # is the evaluation_representation field above.
        "representation": "finite-character Gamma-centred character mesh",
        "family": "variational finite-BvK-torus CCM",
        "result_backend": getattr(result, "backend", None),
        "backend": getattr(diagnostics, "backend", None),
        "electronic_method": getattr(diagnostics, "electronic_method", None),
        "mesh": _json_safe_qvf_value(getattr(diagnostics, "mesh", None)),
        "n_cyclic_cells": getattr(diagnostics, "n_cyclic_cells", None),
        "finite_torus_convention": _json_safe_qvf_value(convention),
        "exact_exchange_assembly": _json_safe_qvf_value(
            exact_exchange_assembly
        ),
    }
    return [
        {
            "id": "x_vibeqc_aiccm2026dev_b_convention",
            "kind": "x_vibeqc.aiccm2026dev_b_convention",
            "member": "convention",
            "label": "chi-CCM-B finite-torus convention",
            "payload": _json_safe_qvf_value(payload),
        }
    ]


def _gamma_qvf_wannier_center_sections(
    localization_result,
) -> list[dict[str, object]]:
    """Wannier-centre overlay sections from a Γ-point localization.

    Takes a :class:`~vibeqc.periodic_localise.PeriodicWannierResult`.

    **Uses ``centers``, not ``centroids``.** Both are on the result and the
    difference matters here: ``centroids`` is the true ``<i|r|i>`` and is,
    per its own docstring, "faithful only for orbitals that do not wrap the
    cell boundary" -- the same cell-face truncation that keeps localized
    coefficients out of periodic QVF entirely. ``centers`` is the
    Mulliken-charge-weighted centre, approximate but PBC-robust, so it stays
    meaningful for exactly the orbitals a reader most wants to locate: the
    ones straddling a face.

    Descriptors only. No coefficients are emitted, which is what makes this
    section safe for periodic output at all.
    """
    centers = np.asarray(getattr(localization_result, "centers"), dtype=float)
    spreads = np.asarray(getattr(localization_result, "spreads"), dtype=float)
    centers = centers.reshape((-1, 3))
    spreads = spreads.reshape((-1,))
    if centers.shape[0] != spreads.shape[0]:
        raise ValueError(
            "Wannier centre count does not match spread count"
        )

    method = str(getattr(localization_result, "method", "") or "localized")
    bohr2_to_ang2 = _BOHR_TO_ANGSTROM * _BOHR_TO_ANGSTROM
    entries: list[dict[str, object]] = [
        {
            "center": [float(v) * _BOHR_TO_ANGSTROM for v in center_bohr],
            "spread": float(spread_bohr2) * bohr2_to_ang2,
            "label": f"Γ {method} Wannier {index + 1}",
        }
        for index, (center_bohr, spread_bohr2) in enumerate(
            zip(centers, spreads)
        )
    ]
    if not entries:
        return []
    return [
        {
            "id": "x_ccm_wannier_centers",
            "kind": "x_ccm.wannier_centers",
            "member": "centers",
            "label": f"Γ-point Wannier centres ({method})",
            "payload": {"centers": _json_safe_qvf_value(entries)},
        }
    ]


def _aiccm_b_qvf_wannier_center_sections(
    localization_result,
) -> list[dict[str, object]]:
    """Return vibe-view Wannier-centre overlay sections for χ-CCM-B output."""

    entries: list[dict[str, object]] = []
    bohr2_to_ang2 = _BOHR_TO_ANGSTROM * _BOHR_TO_ANGSTROM

    def append_block(block, *, prefix: str = "") -> None:
        centers = np.asarray(getattr(block, "centers_bohr"), dtype=float)
        spreads = np.asarray(getattr(block, "spreads_bohr2"), dtype=float)
        centers = centers.reshape((-1, 3))
        spreads = spreads.reshape((-1,))
        if centers.shape[0] != spreads.shape[0]:
            raise ValueError(
                "χ-CCM-B Wannier centre count does not match spread count"
            )
        for index, (center_bohr, spread_bohr2) in enumerate(zip(centers, spreads)):
            label = (
                f"{prefix}Wannier {index + 1}"
                if prefix
                else f"Wannier {index + 1}"
            )
            entries.append(
                {
                    "center": [
                        float(value) * _BOHR_TO_ANGSTROM
                        for value in center_bohr
                    ],
                    "spread": float(spread_bohr2) * bohr2_to_ang2,
                    "label": f"χ-CCM-B {label}",
                }
            )

    if hasattr(localization_result, "alpha"):
        append_block(localization_result.alpha, prefix="alpha ")
        beta = getattr(localization_result, "beta", None)
        if beta is not None:
            append_block(beta, prefix="beta ")
    else:
        append_block(localization_result)

    if not entries:
        return []
    return [
        {
            "id": "x_ccm_wannier_centers",
            "kind": "x_ccm.wannier_centers",
            "member": "centers",
            "label": "χ-CCM-B Wannier centres",
            "payload": {"centers": _json_safe_qvf_value(entries)},
        }
    ]


def _qvf_extensions_with(
    extensions: dict[str, object] | None,
    namespace: str,
) -> dict[str, object]:
    """Return a QVF extension map containing ``namespace``."""
    merged = dict(extensions or {})
    merged.setdefault(namespace, {"version": "1.0", "critical": False})
    return merged


# ============================================================
# Main entry point
# ============================================================


def _has_valid_mo_coeffs(result) -> bool:
    """Check whether result.mo_coeffs is non-empty (array or list)."""
    mc = getattr(result, "mo_coeffs", None)
    if mc is None:
        return False
    if isinstance(mc, (list, tuple)):
        return len(mc) > 0 and hasattr(mc[0], "size") and mc[0].size > 0
    return hasattr(mc, "size") and mc.size > 0


def _has_valid_sidecar_mo_coeffs(result) -> bool:
    """Return whether a restricted or unrestricted result has MO data."""
    if _has_valid_mo_coeffs(result):
        return True

    def _nonempty(value) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return bool(value) and hasattr(value[0], "size") and value[0].size > 0
        return hasattr(value, "size") and value.size > 0

    return _nonempty(getattr(result, "mo_coeffs_alpha", None)) and _nonempty(
        getattr(result, "mo_coeffs_beta", None)
    )


def _sidecar_mo_ao_dimension(result) -> Optional[int]:
    """AO dimension of a result's MO coefficients, or ``None`` if unreadable.

    vibe-qc's convention is ``mo_coeffs[n_ao, n_mo]`` (columns are orbitals);
    a multi-k result carries a list of per-k blocks that share the leading
    dimension. Open-shell results expose ``mo_coeffs_alpha`` instead.
    """
    for attr in ("mo_coeffs", "mo_coeffs_alpha"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) > 0:
            return int(shape[0])
    return None


def _mo_coeffs_span_unit_cell_basis(result, basis) -> bool:
    """Whether the result's MO coefficients are in the UNIT-CELL AO basis.

    The artefact writers pair MO coefficients with the unit cell's basis and
    structure, so a producer whose natural state spans a supercell must not
    reach them. The real-Γ CCM adapter is exactly that case: it retains the
    supercell-Γ MOs verbatim by design (its energy and density ARE folded per
    unit cell), so at a Born-von Karman torus larger than one cell its
    coefficient matrix is ``(N_c*n_ao)`` wide against an ``n_ao`` basis.
    Before this check the QVF ``wavefunction`` section shipped that matrix
    with metadata describing the unit cell -- ``n_mo * n_ao`` did not equal
    the number of coefficients written, and ``validate_qvf`` still passed the
    archive (GitLab #654). An unmeasurable result keeps the historical
    behaviour: this gate exists to catch a KNOWN mismatch, not to withdraw
    artefacts whose dimensions cannot be read.
    """
    n_ao = _sidecar_mo_ao_dimension(result)
    n_basis = int(getattr(basis, "nbasis", 0) or 0)
    if n_ao is None or n_basis <= 0:
        return True
    return n_ao == n_basis


def _should_prepare_periodic_qvf_wavefunction(
    output_qvf: bool, result, basis=None
) -> bool:
    """Gate QVF wavefunction data for restricted and unrestricted results.

    ``basis`` is the unit cell's :class:`BasisSet`; when given, a result whose
    MO coefficients do not span it is refused (see
    :func:`_mo_coeffs_span_unit_cell_basis`).
    """
    if not (bool(output_qvf) and _has_valid_sidecar_mo_coeffs(result)):
        return False
    if basis is None:
        return True
    return _mo_coeffs_span_unit_cell_basis(result, basis)


class _ResultECPProxy:
    """Expose runner-resolved ECP electron provenance to format writers.

    Periodic native result classes do not yet carry ``ecp_total_ncore``.
    The runner has already resolved the authoritative count from the basis
    sidecar before SCF, so keep this compatibility adapter in the runner
    rather than teaching output code to infer ECP metadata.
    """

    __slots__ = ("_inner", "ecp_total_ncore")

    def __init__(self, inner, ecp_total_ncore: int):
        self._inner = inner
        self.ecp_total_ncore = int(ecp_total_ncore)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _result_with_ecp_ncore(result, total_ncore: int):
    """Return a format-writer view with authoritative ECP provenance."""
    count = int(total_ncore or 0)
    if count < 0:
        raise ValueError("periodic ECP replaced-core count must be non-negative")
    return _ResultECPProxy(result, count)


def _gamma_proxy_for_multi_k(result) -> _GammaProxy:
    """Wrap a multi-k result to expose Γ-point (k=0) MOs for molden/etc."""
    mo_coeffs = result.mo_coeffs
    if isinstance(mo_coeffs, (list, tuple)):
        gamma_idx = _gamma_index_for_multi_k(result, len(mo_coeffs))
    else:
        gamma_idx = 0
    return _kpoint_proxy_for_multi_k(result, gamma_idx)


def _is_unrestricted_orbital_result(result) -> bool:
    """Whether a result carries split alpha/beta orbitals instead of MOs."""
    return getattr(result, "mo_coeffs", None) is None and (
        getattr(result, "mo_coeffs_alpha", None) is not None
    )


def _gamma_orbital_proxy(result):
    """Wrap the Γ block of any periodic result in single-k result shape.

    Restricted results delegate to :func:`_gamma_proxy_for_multi_k`.
    Unrestricted (UHF/UKS) periodic drivers store one ``(nbf, nmo)`` block
    per k-point under ``mo_coeffs_alpha`` / ``mo_coeffs_beta`` and carry no
    ``mo_coeffs`` at all, so they need their own slice: without it the
    Γ-only writers receive the whole per-k list and see a 3-D array.

    Both paths reach :func:`_gamma_index_for_multi_k`, which locates the
    real Γ entry from the result's k-point metadata and refuses a mesh that
    has none rather than exporting an arbitrary k-block as if it were Γ.
    """
    if not _is_unrestricted_orbital_result(result):
        return _gamma_proxy_for_multi_k(result)

    coeffs_alpha = result.mo_coeffs_alpha
    coeffs_beta = getattr(result, "mo_coeffs_beta", None)
    if coeffs_beta is None:
        raise ValueError(
            "unrestricted periodic Γ-only output requires both "
            "mo_coeffs_alpha and mo_coeffs_beta"
        )
    if isinstance(coeffs_alpha, (list, tuple)):
        if len(coeffs_beta) != len(coeffs_alpha):
            raise ValueError(
                "unrestricted periodic Γ-only output requires matching "
                f"alpha/beta k-point counts; got {len(coeffs_alpha)} alpha "
                f"and {len(coeffs_beta)} beta blocks"
            )
        gamma_idx = _gamma_index_for_multi_k(result, len(coeffs_alpha))
    else:
        gamma_idx = 0

    def _at_gamma(value):
        if isinstance(value, (list, tuple)):
            if gamma_idx >= len(value):
                raise ValueError(
                    "unrestricted periodic result has per-k metadata for "
                    f"fewer k-points than orbital blocks; requested index "
                    f"{gamma_idx}, got {len(value)} blocks"
                )
            return value[gamma_idx]
        return value

    def _optional_at_gamma(attr: str):
        value = getattr(result, attr, None)
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and not value:
            return None
        return _at_gamma(value)

    return _UnrestrictedGammaProxy(
        mo_coeffs_alpha=_at_gamma(coeffs_alpha),
        mo_coeffs_beta=_at_gamma(coeffs_beta),
        mo_energies_alpha=_at_gamma(result.mo_energies_alpha),
        mo_energies_beta=_at_gamma(result.mo_energies_beta),
        occupations_alpha=_optional_at_gamma("occupations_alpha"),
        occupations_beta=_optional_at_gamma("occupations_beta"),
        density_alpha=getattr(result, "density_alpha", None),
        density_beta=getattr(result, "density_beta", None),
        overlap=_overlap_at_kpoint(result, gamma_idx),
    )


def _qvf_wavefunction_proxy_for_multi_k(result, system) -> tuple[object, list[float]]:
    """Return a result proxy and fractional k point for QVF wavefunction.gto.

    QVF can carry one selected complex Bloch wavefunction. Prefer Γ when the
    SCF k list contains it, preserving older archives; otherwise use the first
    k point and record its reciprocal fractional coordinate.
    """
    mo_coeffs = getattr(result, "mo_coeffs", None)
    if not isinstance(mo_coeffs, (list, tuple)):
        return result, [0.0, 0.0, 0.0]
    n_items = len(mo_coeffs)
    if n_items <= 0:
        raise ValueError("multi-k result has no k-point data")
    kpts = _result_kpoints_cart(result)
    if kpts is None:
        if n_items == 1:
            return _kpoint_proxy_for_multi_k(result, 0), [0.0, 0.0, 0.0]
        raise ValueError(
            "multi-k QVF wavefunction output requires k-point metadata"
        )
    if kpts.shape[0] != n_items:
        raise ValueError(
            "multi-k QVF wavefunction output requires aligned k-point metadata; "
            f"got {n_items} data blocks and {kpts.shape[0]} k-points"
        )
    norms = np.linalg.norm(kpts, axis=1)
    gamma_idx = int(np.argmin(norms))
    idx = gamma_idx if float(norms[gamma_idx]) <= 1.0e-10 else 0
    frac = _wrap_reciprocal_fractional(
        _fractional_kpoints_for_output(system, kpts)
    )
    return _kpoint_proxy_for_multi_k(result, idx), [float(x) for x in frac[idx]]


def _overlap_at_kpoint(result, idx: int):
    """Return the single-k overlap block a Gamma-only writer can use as S.

    Periodic drivers carry ``overlap`` as one Hermitian block per k-point.
    The Molden writer needs S(k=0) as a metric when it re-expresses a
    degenerate block on a real basis; handing it the whole per-k list would
    silently drop it back to the identity metric.
    """
    overlap = getattr(result, "overlap", None)
    if isinstance(overlap, (list, tuple)):
        if idx >= len(overlap):
            return None
        return overlap[idx]
    return overlap


def _kpoint_proxy_for_multi_k(result, idx: int) -> _GammaProxy:
    """Wrap one k-point block of a multi-k result in single-k result shape."""
    mo_coeffs = result.mo_coeffs
    mo_energies = result.mo_energies
    occupations = getattr(result, "occupations", None)
    if isinstance(occupations, (list, tuple)):
        if len(occupations) == 0:
            occupations = None
        elif idx < len(occupations):
            occupations = occupations[idx]
        else:
            raise ValueError(
                "multi-k result has occupation metadata for fewer k-points "
                f"than orbital blocks; requested index {idx}, got "
                f"{len(occupations)} occupation blocks"
            )
    density = result.density
    return _GammaProxy(
        mo_coeffs=mo_coeffs[idx]
        if isinstance(mo_coeffs, (list, tuple))
        else mo_coeffs,
        mo_energies=mo_energies[idx]
        if isinstance(mo_energies, (list, tuple))
        else mo_energies,
        occupations=occupations,
        density=density[idx]
        if isinstance(density, (list, tuple))
        else density,
        overlap=_overlap_at_kpoint(result, idx),
    )


class _GammaProxy:
    """Duck-typed result wrapping Γ-point (k=0) of a multi-k result."""

    __slots__ = ("mo_coeffs", "mo_energies", "occupations", "density", "overlap")

    def __init__(self, *, mo_coeffs, mo_energies, occupations, density, overlap):
        self.mo_coeffs = mo_coeffs
        self.mo_energies = mo_energies
        self.occupations = occupations
        self.density = density
        self.overlap = overlap


class _UnrestrictedGammaProxy:
    """Duck-typed UHF/UKS result wrapping the Γ block of a periodic result.

    Mirrors :class:`_GammaProxy` for the split-spin attribute names the
    Molden writer dispatches on. The spin densities are passed through
    unsliced: they are real-space ``LatticeMatrixSet`` objects that already
    span the full k-mesh, not per-k lists.
    """

    __slots__ = (
        "mo_coeffs_alpha",
        "mo_coeffs_beta",
        "mo_energies_alpha",
        "mo_energies_beta",
        "occupations_alpha",
        "occupations_beta",
        "density_alpha",
        "density_beta",
        "overlap",
    )

    def __init__(
        self,
        *,
        mo_coeffs_alpha,
        mo_coeffs_beta,
        mo_energies_alpha,
        mo_energies_beta,
        occupations_alpha,
        occupations_beta,
        density_alpha,
        density_beta,
        overlap,
    ):
        self.mo_coeffs_alpha = mo_coeffs_alpha
        self.mo_coeffs_beta = mo_coeffs_beta
        self.mo_energies_alpha = mo_energies_alpha
        self.mo_energies_beta = mo_energies_beta
        self.occupations_alpha = occupations_alpha
        self.occupations_beta = occupations_beta
        self.density_alpha = density_alpha
        self.density_beta = density_beta
        self.overlap = overlap


class _GapwMultiKRunnerProxy:
    """Duck-typed proxy wrapping a ``GpwMultiKScfResult`` for the runner.

    ``GpwMultiKScfResult`` stores per-k data under ``mo_coeffs_k`` /
    ``mo_energies_k`` / ``occupations_k``, but the runner's output code
    (molden, MO summary, density) expects ``mo_coeffs`` / ``mo_energies`` /
    ``occupations`` as per-k list-shaped attributes. This proxy delegates
    those names to the ``*_k`` equivalents and passes all other attribute
    access through to the original result.
    """

    __slots__ = ("_inner", "guess_selection")

    def __init__(self, inner):
        self._inner = inner
        self.guess_selection = getattr(inner, "guess_selection", None)

    @property
    def mo_coeffs(self):
        return list(self._inner.mo_coeffs_k)

    @property
    def mo_energies(self):
        return list(self._inner.mo_energies_k)

    @property
    def occupations(self):
        return list(self._inner.occupations_k)

    @property
    def e_electronic(self) -> float:
        bd = getattr(self._inner, "breakdown", None)
        if bd is not None:
            return float(
                getattr(bd, "e_kinetic", 0.0)
                + getattr(bd, "e_nuclear_attraction", 0.0)
                + getattr(bd, "e_hartree", 0.0)
                + getattr(bd, "e_xc", 0.0)
                + getattr(bd, "e_hf_exchange", 0.0)
            )
        return 0.0

    @property
    def e_nuclear(self) -> float:
        bd = getattr(self._inner, "breakdown", None)
        if bd is not None:
            return float(getattr(bd, "e_nuclear_repulsion", 0.0))
        return 0.0

    @property
    def e_xc(self) -> float:
        bd = getattr(self._inner, "breakdown", None)
        if bd is not None:
            return float(getattr(bd, "e_xc", 0.0))
        return 0.0

    def __getattr__(self, name):
        # Delegate all other attribute lookups to the inner result.
        # Properties (mo_coeffs etc.) are handled by the class descriptors
        # and never reach __getattr__.
        return getattr(self._inner, name)


class _SemiempiricalKPointRunnerProxy:
    """Expose native k-point semiempirical results to runner summaries."""

    __slots__ = ("_inner", "_kmesh")

    def __init__(self, inner, kmesh):
        self._inner = inner
        self._kmesh = kmesh

    @property
    def mo_energies(self):
        return list(getattr(self._inner, "eps_per_k"))

    @property
    def occupations(self):
        return list(getattr(self._inner, "occupations_per_k"))

    @property
    def kpoints(self):
        return list(getattr(self._kmesh, "kpoints"))

    @property
    def converged(self):
        return bool(getattr(self._inner, "converged", True))

    @property
    def iterations(self):
        return getattr(self._inner, "n_iter", None)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _plan_periodic_semiempirical_method(
    method: str,
    system: PeriodicSystem,
    *,
    kpoints: object | None,
):
    """Return a validated semiempirical route plan, or ``None``."""
    from vibeqc.semiempirical.routes import (
        BOUNDARY_PERIODIC_GAMMA,
        BOUNDARY_PERIODIC_K,
        is_semiempirical_method,
        plan_periodic_semiempirical_route,
    )

    if not is_semiempirical_method(method):
        return None
    boundary = (
        BOUNDARY_PERIODIC_K
        if kpoints is not None
        else BOUNDARY_PERIODIC_GAMMA
    )
    return plan_periodic_semiempirical_route(
        method,
        system,
        boundary=boundary,
    )


def _periodic_shell_gamma_form_name(form) -> str | None:
    """Route-plan spelling of a native ``ShellGammaForm``, or None.

    The native enum is matched by identity rather than by parsing its repr:
    ``str(ShellGammaForm.KlopmanOhno)`` is ``"ShellGammaForm.KlopmanOhno"``,
    whose lowercased tail is ``"klopmanohno"`` and not the route plan's
    ``"klopman_ohno"``.  A result without the attribute (every non-GFN2
    periodic engine) reports None and adds no electrostatics citations.
    """
    if form is None:
        return None
    from vibeqc._vibeqc_core import semiempirical as _se

    for member, name in (
        (_se.ShellGammaForm.Elstner, "elstner"),
        (_se.ShellGammaForm.KlopmanOhno, "klopman_ohno"),
    ):
        if form == member:
            return name
    raise ValueError(f"unrecognised periodic shell gamma form: {form!r}")


def _run_periodic_semiempirical_engine(
    system: PeriodicSystem,
    route_plan,
    *,
    max_iter: int,
    conv_tol: float,
    kpoints: object | None = None,
    smearing_temperature_hartree: float | None = None,
):
    """Run one basis-free periodic semiempirical backend.

    ``smearing_temperature_hartree`` is the resolved electronic temperature
    (k_B T in Hartree). For GFN2-xTB, ``None`` leaves the native default-on
    frontier smearing in place; a numeric value (including 0.0 for exact
    zero-temperature Aufbau) is forwarded explicitly. Other backends treat
    ``None`` as 0.0.
    """
    method_key = route_plan.method_key
    resolved_smearing = float(
        0.0 if smearing_temperature_hartree is None else smearing_temperature_hartree
    )
    if method_key in ("dftb0", "scc_dftb"):
        from vibeqc._vibeqc_core import semiempirical as _se

        open_shell = route_plan.spin == "unrestricted"
        if route_plan.boundary == "periodic_k":
            if open_shell:
                raise NotImplementedError(
                    "full k-point periodic DFTB routes are closed-shell only; "
                    "unrestricted k-point DFTB/SCC-DFTB is not implemented."
                )
            from vibeqc.semiempirical.periodic import _as_bloch_kmesh

            kmesh = _as_bloch_kmesh(system, kpoints)

        from vibeqc.semiempirical.parameters import default_parameters

        params = default_parameters()
        # Loudly flag placeholder repulsive pairs before any number is
        # produced (issue #306): fixed-geometry differences stay valid,
        # absolute energies/EOS fits for such systems are not chemistry.
        from vibeqc.semiempirical.dftb0 import warn_placeholder_repulsives

        warn_placeholder_repulsives(
            params,
            [atom.Z for atom in system.unit_cell],
            route=f"periodic {method_key}",
        )
        if route_plan.boundary == "periodic_k":
            occupation_options = _se.KPointOccupationOptions()
            occupation_options.smearing_temperature = resolved_smearing
            if method_key == "dftb0":
                result = _se.run_dftb0_kpoints(
                    system,
                    params,
                    kmesh,
                    15.0,
                    occupation_options,
                )
            else:
                opts = _se.SCCOptions()
                opts.max_iter = int(max_iter)
                opts.conv_tol_charge = float(conv_tol)
                opts.use_diis = True
                result = _se.run_scc_dftb_kpoints(
                    system,
                    params,
                    kmesh,
                    opts,
                    15.0,
                    occupation_options,
                    # Issue #342: receive the record even when unconverged.
                    # The established output-stage guard downstream writes
                    # the fatal diagnostic into the .out, marks the .system
                    # manifest crashed, and then raises; an engine-level
                    # raise here would lose that evidence (the d70f4335b
                    # lesson).
                    allow_unconverged=True,
                )
            return _SemiempiricalKPointRunnerProxy(result, kmesh)

        if method_key == "dftb0":
            if open_shell:
                return _se.run_udftb0_gamma(system, params)
            return _se.run_dftb0_gamma(system, params)

        opts = _se.PeriodicSCCOptions()
        opts.max_iter = int(max_iter)
        opts.conv_tol_charge = float(conv_tol)
        opts.use_diis = True
        if open_shell:
            return _se.run_uscc_dftb_gamma(system, params, opts)
        return _se.run_scc_dftb_gamma(system, params, opts)

    if method_key == "gfn2_xtb":
        from vibeqc._vibeqc_core.semiempirical.xtb import (
            XTBSccOptions,
            run_gfn2_xtb_gamma,
        )
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        opts = XTBSccOptions()
        opts.max_iter = int(max_iter)
        opts.conv_tol_charge = float(conv_tol)
        if smearing_temperature_hartree is not None:
            # Assignment marks the temperature explicit: 0.0 keeps exact
            # zero-temperature Aufbau. None leaves the native default-on
            # frontier smearing (0.001 Ha) in place.
            opts.electronic_temperature = float(smearing_temperature_hartree)
        return run_gfn2_xtb_gamma(system, load_gfn2_params(), opts)

    if method_key == "pm6":
        from vibeqc.semiempirical.methods.periodic_pm6 import run_pm6_gamma

        return run_pm6_gamma(
            system,
            max_iter=int(max_iter),
            conv_tol=float(conv_tol),
        )

    if method_key in ("om1", "om2", "om3"):
        from vibeqc.semiempirical.methods.periodic_omx import run_omx_gamma

        return run_omx_gamma(
            system,
            variant=method_key,
            max_iter=int(max_iter),
            conv_tol=float(conv_tol),
        )

    raise ValueError(f"unknown periodic semiempirical method {method_key!r}")


def _periodic_semiempirical_attr(result: object, *names: str) -> object | None:
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return value
    return None


def _run_periodic_semiempirical_job(
    system: PeriodicSystem,
    *,
    route_plan,
    output: Union[str, os.PathLike],
    dry_run: bool,
    record_hostname: bool,
    citations: bool,
    write_xyz_file: bool,
    write_poscar_file: bool,
    write_xsf_structure_file: bool,
    write_cif_file: bool,
    write_molden_file: bool,
    write_density: bool,
    write_population_file: bool,
    output_qvf: bool,
    jk_method: Union[str, "PeriodicJKMethod"],
    aux_basis: Optional[str],
    smearing: Optional[SmearingOptions],
    smearing_temperature: Union[float, str, None],
    smearing_unit: str,
    smearing_method: str,
    smearing_metallic: Optional[bool],
    smearing_band_gap_hartree: Optional[float],
    dispersion: Optional[Union[str, bool, "D3BJParams"]],
    optimize: bool,
    optimize_max_iter: int,
    optimize_conv_tol_grad: float,
    optimize_cell_requested: bool,
    hessian: bool,
    tddft: bool,
    coop_cohp: bool,
    band_structure: object | None,
    qvf_wannier_centers: bool,
    kpoints: object | None,
    checkpoint_qvf: Union[str, os.PathLike, None],
    dft_plus_u: object | None,
    atomic_spins: object | None,
    spinlock: str | None,
    read_from: object | None,
    restart_from: Union[str, os.PathLike, None],
    functional: str | None,
    max_iter: int,
    conv_tol_energy: float,
):
    """Public ``run_periodic_job`` branch for basis-free SE methods."""
    from vibeqc.semiempirical.routes import BOUNDARY_PERIODIC_K

    method_key = route_plan.method_key
    full_k_route = route_plan.boundary == BOUNDARY_PERIODIC_K
    full_k_optimizer = full_k_route and method_key in ("dftb0", "scc_dftb")
    gfn2_gamma_route = method_key == "gfn2_xtb" and not full_k_route
    if functional is not None:
        raise ValueError(
            f"run_periodic_job: functional={functional!r} is not used with "
            f"basis-free semiempirical method={method_key!r}."
        )
    unsupported: list[str] = []
    if optimize and not full_k_optimizer:
        unsupported.append("optimize")
    if optimize_cell_requested and not full_k_optimizer:
        unsupported.append("optimize_cell")
    if hessian:
        unsupported.append("hessian")
    if tddft:
        unsupported.append("tddft")
    if coop_cohp:
        unsupported.append("coop_cohp")
    if band_structure is not None:
        unsupported.append("band_structure")
    if qvf_wannier_centers:
        unsupported.append("qvf_wannier_centers")
    if write_density:
        unsupported.append("write_density")
    jk_label = (
        jk_method.value if isinstance(jk_method, PeriodicJKMethod) else str(jk_method)
    ).strip().lower()
    if jk_label not in ("auto", ""):
        unsupported.append("jk_method")
    if aux_basis is not None:
        unsupported.append("aux_basis")
    smearing_requested = (
        smearing is not None
        or smearing_temperature is not _SMEARING_UNSET
        or smearing_metallic is not None
        or smearing_band_gap_hartree is not None
    )
    if smearing_requested and not full_k_route and not gfn2_gamma_route:
        unsupported.append("smearing")
    if full_k_route or gfn2_gamma_route:
        if smearing is not None:
            if not isinstance(smearing, SmearingOptions):
                raise TypeError(
                    "run_periodic_job: smearing must be a SmearingOptions "
                    "instance."
                )
            if smearing.flavor not in ("fermi-dirac", "mermin"):
                raise NotImplementedError(
                    "periodic semiempirical routes support only Fermi-Dirac / "
                    "Mermin smearing."
                )
        if smearing_metallic is not None or smearing_band_gap_hartree is not None:
            unsupported.append("smearing_auto")
    if dispersion not in (None, False):
        unsupported.append("dispersion")
    if (not full_k_route) and kpoints is not None:
        unsupported.append("kpoints")
    if checkpoint_qvf is not None:
        unsupported.append("checkpoint_qvf")
    if dft_plus_u:
        unsupported.append("dft_plus_u")
    if atomic_spins is not None:
        unsupported.append("atomic_spins")
    if spinlock is not None:
        unsupported.append("spinlock")
    if read_from is not None:
        unsupported.append("read_from")
    if restart_from is not None:
        unsupported.append("restart_from")
    if unsupported:
        opts = ", ".join(sorted(unsupported))
        route_desc = (
            "full k-point single-point routes"
            if full_k_route
            else "Gamma-point single-point routes"
        )
        raise NotImplementedError(
            "run_periodic_job: basis-free periodic semiempirical methods are "
            f"{route_desc}; unsupported option(s): {opts}."
        )

    if full_k_route and kpoints is None:
        raise ValueError(
            "full k-point periodic semiempirical routes require kpoints=."
        )
    if full_k_route:
        from vibeqc.semiempirical.periodic import _as_bloch_kmesh

        _as_bloch_kmesh(system, kpoints)
    if optimize_cell_requested and not optimize:
        raise ValueError(
            "run_periodic_job: optimize_cell=True requires optimize=True for "
            "basis-free periodic semiempirical routes."
        )

    smearing_temperature_hartree = 0.0
    smearing_source = "explicit"
    smearing_reason = ""
    if full_k_route:
        if smearing is not None:
            smearing_temperature_hartree = (
                float(smearing.temperature) if bool(smearing.enabled) else 0.0
            )
            smearing_source = smearing.source
            smearing_reason = smearing.reason
        else:
            smearing_resolution = resolve_smearing_temperature(
                0.0
                if smearing_temperature is _SMEARING_UNSET
                else smearing_temperature,
                unit=smearing_unit,
                method=smearing_method,
                metallic=False,
                band_gap_hartree=None,
                n_electrons=system.n_electrons(),
            )
            if smearing_resolution.method not in ("fermi-dirac", "mermin"):
                raise NotImplementedError(
                    "full k-point semiempirical DFTB routes currently support "
                    "only Fermi-Dirac / Mermin smearing."
                )
            smearing_temperature_hartree = float(smearing_resolution.temperature)
            smearing_source = smearing_resolution.source
            smearing_reason = smearing_resolution.reason
    elif gfn2_gamma_route:
        # Periodic GFN2-xTB smears by default. An explicit numeric 0.0 (or
        # None / "off") restores exact zero-temperature Aufbau; leaving the
        # argument unset (or "auto") selects the default width.
        if smearing is not None:
            smearing_temperature_hartree = (
                float(smearing.temperature) if bool(smearing.enabled) else 0.0
            )
            smearing_source = smearing.source
            smearing_reason = smearing.reason
        elif smearing_temperature is _SMEARING_UNSET or (
            isinstance(smearing_temperature, str)
            and str(smearing_temperature).strip().lower().replace("_", "-")
            == "auto"
        ):
            smearing_temperature_hartree = _GFN2_PERIODIC_DEFAULT_SMEARING_HA
            smearing_source = "auto"
            smearing_reason = (
                "periodic GFN2-xTB default frontier smearing; pass "
                "smearing_temperature=0 for exact zero-temperature Aufbau"
            )
        else:
            smearing_resolution = resolve_smearing_temperature(
                smearing_temperature,
                unit=smearing_unit,
                method=smearing_method,
                metallic=False,
                band_gap_hartree=None,
                n_electrons=system.n_electrons(),
            )
            if smearing_resolution.method not in ("fermi-dirac", "mermin"):
                raise NotImplementedError(
                    "periodic GFN2-xTB supports only Fermi-Dirac / Mermin "
                    "smearing."
                )
            smearing_temperature_hartree = float(smearing_resolution.temperature)
            smearing_source = smearing_resolution.source
            smearing_reason = smearing_resolution.reason
    # Finite-temperature full-k DFTB optimization minimises the Mermin free
    # energy F = E - T S with its own derivatives (maintainer decision on
    # GitLab #545, 2026-09-06): opt_energy, opt_derivatives and
    # opt_cell_derivatives below all run at the resolved temperature, and
    # the optimizer pairs grad F with F (semiempirical.periodic
    # ._minimised_potential). The former zero-temperature-only gate here
    # sent users who followed the #434 frontier guard's advice (supply a
    # smearing temperature) straight into a second refusal.
    if optimize:
        if int(optimize_max_iter) < 0:
            raise ValueError("run_periodic_job: optimize_max_iter must be non-negative.")
        if (
            not np.isfinite(float(optimize_conv_tol_grad))
            or float(optimize_conv_tol_grad) <= 0.0
        ):
            raise ValueError(
                "run_periodic_job: optimize_conv_tol_grad must be finite and positive."
            )

    output_stem = Path(os.fspath(output))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    out_path = stem_sibling(output_stem, ".out")
    basis_label = "<basis-free>"
    plan = OutputPlan.from_run_job_kwargs(
        output=output_stem,
        method=method_key,
        basis=basis_label,
        functional=None,
        write_molden_file=False,
        write_xyz=write_xyz_file,
        write_poscar=write_poscar_file,
        write_xsf_structure=write_xsf_structure_file,
        write_cif=write_cif_file,
        write_density_xsf=False,
        write_population=False,
        citations=citations,
        crash_dump=False,
        output_qvf=output_qvf,
        job_kind="periodic_scf",
    )
    if dry_run or is_dry_run_requested():
        dry_run_manifest(plan, record_hostname=record_hostname)
        return None

    t_start = time.perf_counter()
    _output_writer = OutputWriter(plan, record_hostname=record_hostname)
    _PERIODIC_OUTPUT_WRITER.set(_output_writer)
    # Re-register progress handler with THIS writer's manifest.
    install_progress_handler(
        lambda fields: _output_writer.update_progress(**fields)
    )
    try:
        result = _run_periodic_semiempirical_engine(
            system,
            route_plan,
            max_iter=max_iter,
            conv_tol=conv_tol_energy,
            kpoints=kpoints,
            smearing_temperature_hartree=smearing_temperature_hartree,
        )
        parameter_identity = getattr(result, "parameter_identity", None)
        parameter_sha256 = getattr(result, "parameter_sha256", None)
        if (parameter_identity is None) != (parameter_sha256 is None):
            raise RuntimeError(
                "periodic semiempirical result has incomplete parameter "
                "identity provenance"
            )
        if parameter_identity is not None:
            parameter_identity = str(parameter_identity)
            parameter_sha256 = str(parameter_sha256)
            if (
                not parameter_identity
                or len(parameter_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in parameter_sha256
                )
            ):
                raise RuntimeError(
                    "periodic semiempirical result has invalid parameter "
                    "identity provenance"
                )
            _output_writer.update_run_fields(
                {
                    "parameter_identity": parameter_identity,
                    "parameter_sha256": parameter_sha256,
                }
            )
        if full_k_route:
            # Issue #426: the measured band edges + gaps are result-struct
            # fields on every k-route record (convention documented on the
            # native KPointBandEdges); carry them into the .system [run]
            # section through the existing extension point, mirroring the
            # IID 344 backend/parity_held surfacing. NaN is not a valid
            # manifest float -- a non-finite value means the edge does not
            # exist in the model space (or an unconverged diagnostics
            # record measured nothing) and is recorded as the explicit
            # string "not-measured", never silently dropped.
            _gap_fields = {}
            for _gap_name in (
                "indirect_gap",
                "direct_gap",
                "valence_band_max",
                "conduction_band_min",
                "gap_above_fermi_manifold",
            ):
                _gap_value = getattr(result, _gap_name, None)
                if _gap_value is None:
                    continue
                _gap_value = float(_gap_value)
                _gap_fields[_gap_name] = (
                    _gap_value if math.isfinite(_gap_value) else "not-measured"
                )
            # The structural gapless flag travels with the numbers it
            # qualifies: a screen must be able to read "metallic" without
            # inferring it from the sign of a float (sec8-r4 F1).
            _is_metallic = getattr(result, "is_metallic", None)
            if _is_metallic is not None:
                _gap_fields["is_metallic"] = bool(_is_metallic)
            if _gap_fields:
                _output_writer.update_run_fields(_gap_fields)
        t_total = time.perf_counter() - t_start

        cite_block_text = ""
        bibtex_content = ""
        cite_manifest_rows: list[dict[str, Any]] = []
        if citations:
            try:
                citation_route_plan = replace(
                    route_plan,
                    electronic_temperature=smearing_temperature_hartree,
                )
                # The Bloch-periodic second-order kernel is an Ewald-split
                # lattice sum of a short-range shell gamma (issues
                # #296/#338), so its citation surface depends on the gamma
                # form the driver used and the dimensional Ewald channel the
                # cell selected -- neither of which the method key carries.
                # Attach them from the result that was actually produced.
                _gamma_form = _periodic_shell_gamma_form_name(
                    getattr(result, "gamma_form", None)
                )
                if _gamma_form is not None:
                    citation_route_plan = (
                        citation_route_plan
                        .with_periodic_electrostatics_runtime(
                            periodic_dimension=int(system.dim),
                            shell_gamma_form=_gamma_form,
                        )
                    )
                refs = load_default_database().assemble(
                    basis="",
                    functional=None,
                    **citation_route_plan.citation_assemble_kwargs,
                )
                cite_manifest_rows = citation_manifest_rows(refs)
                bibtex_content = format_bibtex(refs)
                _output_writer.dispatch_role(
                    "citations",
                    citations=refs,
                    raise_on_error=True,
                )
                cite_block_text = format_references_block(refs)
            except Exception as exc:
                warn(
                    f"citation emission failed: {type(exc).__name__}: {exc}",
                )

        nonconverged_error: str | None = None
        with OutputChannel.to_file(out_path):
            write(banner() + "\n\n")
            write(
                f"  Job: PERIODIC {method_key.upper()}  "
                "basis=<basis-free>\n"
            )
            write(
                "  Route: "
                + (
                    "full k-point periodic semiempirical"
                    if full_k_route
                    else "Gamma-point periodic semiempirical"
                )
                + "\n\n"
            )
            write(_system_summary(system))
            write(section_header("Semiempirical options", width=56))
            write(f"    method              = {method_key}\n")
            if full_k_route:
                write(f"    boundary            = {route_plan.boundary}\n")
                n_kpoints = _periodic_semiempirical_attr(result, "n_kpoints")
                if n_kpoints is not None:
                    write(f"    kpoints             = {int(n_kpoints)}\n")
                    write_kmesh_convention()
                write("    kpoint_occupations  = fermi-dirac\n")
                if smearing_temperature_hartree > 0.0 or smearing_source != "explicit":
                    write(f"    smearing_source     = {smearing_source}\n")
                    if smearing_reason:
                        write(f"    smearing_reason     = {smearing_reason}\n")
                write(
                    "    smearing_temperature = "
                    f"{smearing_temperature_hartree}\n"
                )
                if smearing_temperature_hartree > 0.0:
                    write(
                        "    smearing_temperature_K = "
                        f"{hartree_to_kelvin_temperature(smearing_temperature_hartree)}\n"
                    )
            elif gfn2_gamma_route and smearing_temperature_hartree > 0.0:
                write("    occupations         = fermi-dirac\n")
                write(f"    smearing_source     = {smearing_source}\n")
                if smearing_reason:
                    write(f"    smearing_reason     = {smearing_reason}\n")
                write(
                    "    smearing_temperature = "
                    f"{smearing_temperature_hartree}\n"
                )
                write(
                    "    smearing_temperature_K = "
                    f"{hartree_to_kelvin_temperature(smearing_temperature_hartree)}\n"
                )
            write(f"    max_iter            = {int(max_iter)}\n")
            write(f"    conv_tol            = {float(conv_tol_energy)}\n\n")
            write(
                "  Basis-free route: Molden orbitals and population "
                "analysis are inapplicable (no Gaussian AO wavefunction).\n"
            )
            skipped = []
            if skipped:
                write(
                    "  Basis-free route: skipped unavailable artefacts "
                    f"({', '.join(skipped)}).\n\n"
                )
            else:
                write("\n")

            write(section_header("Results", width=56))
            energy = float(getattr(result, "energy"))
            write(
                "    total energy        = "
                f"{render_energy_labeled(energy, width=0, precision=10, sign=True)}\n"
            )
            result_smearing_t = float(
                getattr(result, "smearing_temperature", 0.0) or 0.0
            )
            if full_k_route or result_smearing_t > 0.0:
                free_energy = float(getattr(result, "free_energy", energy))
                if free_energy != energy:
                    write(
                        "    free energy         = "
                        f"{render_energy_labeled(free_energy, width=0, precision=10, sign=True)}\n"
                    )
            converged = _periodic_semiempirical_attr(result, "converged")
            if converged is not None:
                write(f"    converged           = {bool(converged)}\n")
            n_iter = _periodic_semiempirical_attr(result, "n_iter", "iterations")
            if n_iter is not None:
                write(f"    iterations          = {int(n_iter)}\n")
            if converged is not None and not bool(converged):
                n_iter_text = (
                    f"{int(n_iter)} iterations"
                    if n_iter is not None
                    else "the allowed iterations"
                )
                nonconverged_error = (
                    f"{method_key.upper()} periodic semiempirical SCF did not "
                    f"converge after {n_iter_text}; refusing to mark the "
                    "calculation complete."
                )
            n_basis = _periodic_semiempirical_attr(result, "n_basis")
            if n_basis is not None:
                write(f"    semiempirical basis = {int(n_basis)} functions\n")
            if parameter_identity is not None:
                write(f"    parameter identity  = {parameter_identity}\n")
                write(f"    parameter sha256    = {parameter_sha256}\n")
            write("\n")
            if full_k_route:
                write(_smearing_summary(result))
                write(_band_summary(result))
                write(_mo_summary(result))
            elif gfn2_gamma_route:
                write(_smearing_summary(result))
            if nonconverged_error is not None:
                write(f"\n  FATAL: {nonconverged_error}\n")
                flush()

            if citations and cite_block_text:
                write_references_block(block=cite_block_text)
                flush()

        _output_writer.record(out_path, wall_time_s=t_total)
        if nonconverged_error is not None:
            raise RuntimeError(nonconverged_error)

        if write_xyz_file:
            _output_writer.dispatch_role(
                "geometry",
                only_format="extended-xyz",
                system=system,
                energy_ha=float(getattr(result, "energy")),
                comment=f"vibe-qc periodic {method_key}",
                raise_on_error=True,
            )
        if write_poscar_file:
            _output_writer.dispatch_role(
                "geometry",
                only_format="poscar",
                system=system,
                comment=f"vibe-qc periodic {method_key}",
                raise_on_error=True,
            )
        if write_xsf_structure_file:
            _output_writer.dispatch_role(
                "geometry",
                only_format="xsf",
                system=system,
                raise_on_error=True,
            )
        if write_cif_file:
            _output_writer.dispatch_role(
                "geometry",
                only_format="cif",
                system=system,
                comment=f"vibe-qc periodic {method_key}",
                raise_on_error=True,
            )

        if optimize:
            from vibeqc.semiempirical.periodic import (
                evaluate_periodic_energy_gradient,
                evaluate_periodic_kpoint_energy_gradient_stress,
                optimize_cell,
                optimize_periodic_positions,
            )
            from vibeqc.semiempirical.routes import (
                plan_periodic_semiempirical_route,
            )

            opt_properties = (
                ("energy", "gradient", "stress")
                if optimize_cell_requested
                else ("energy", "gradient")
            )
            opt_route_plan = plan_periodic_semiempirical_route(
                method_key,
                system,
                boundary=BOUNDARY_PERIODIC_K,
                properties=opt_properties,
            )

            def opt_energy(candidate):
                candidate_result = _run_periodic_semiempirical_engine(
                    candidate,
                    route_plan,
                    max_iter=max_iter,
                    conv_tol=conv_tol_energy,
                    kpoints=kpoints,
                    smearing_temperature_hartree=smearing_temperature_hartree,
                )
                if not bool(getattr(candidate_result, "converged", True)):
                    n_iter = _periodic_semiempirical_attr(
                        candidate_result, "n_iter", "iterations"
                    )
                    suffix = (
                        f" after {int(n_iter)} iterations"
                        if n_iter is not None
                        else ""
                    )
                    raise RuntimeError(
                        f"{method_key.upper()} full-k periodic optimization "
                        f"SCF did not converge{suffix}."
                    )
                if smearing_temperature_hartree > 0.0:
                    # Smeared run: the minimised potential is the Mermin
                    # free energy, the same surface the derivatives below
                    # differentiate (#545).
                    free_energy = getattr(candidate_result, "free_energy", None)
                    if free_energy is None:
                        raise RuntimeError(
                            f"{method_key.upper()} full-k periodic optimization "
                            "ran at finite temperature but the engine result "
                            "carries no free_energy"
                        )
                    return float(free_energy)
                return float(getattr(candidate_result, "energy"))

            def opt_derivatives(candidate):
                return evaluate_periodic_energy_gradient(
                    method_key,
                    candidate,
                    kpoints=kpoints,
                    cutoff_bohr=15.0,
                    max_iter=max_iter,
                    conv_tol_charge=conv_tol_energy,
                    smearing_temperature=smearing_temperature_hartree,
                    _return_result=True,
                )

            def opt_cell_derivatives(candidate):
                return evaluate_periodic_kpoint_energy_gradient_stress(
                    method_key,
                    candidate,
                    kpoints=kpoints,
                    cutoff_bohr=15.0,
                    max_iter=max_iter,
                    conv_tol_charge=conv_tol_energy,
                    smearing_temperature=smearing_temperature_hartree,
                )

            with OutputChannel.to_file(out_path, mode="a"):
                gradient_policy = active_policy().with_spec("gradient", width=0)
                gradient_tolerance = gradient_policy.render(
                    Quantity(float(optimize_conv_tol_grad), "gradient")
                )
                gradient_unit = gradient_policy.unit_of("gradient")
                write(section_header("Geometry optimization", width=56))
                write("    optimizer           = ASE BFGSLineSearch\n")
                write(
                    "    cell                = "
                    f"{'variable' if optimize_cell_requested else 'fixed'}\n"
                )
                write(f"    max_iter            = {int(optimize_max_iter)}\n")
                write(
                    "    gradient_tolerance  = "
                    f"{gradient_tolerance} {gradient_unit}\n\n"
                )
                flush()

            if optimize_cell_requested:
                opt_result = optimize_cell(
                    system,
                    opt_energy,
                    derivatives_fn=opt_cell_derivatives,
                    gradient_tolerance_ha_bohr=float(optimize_conv_tol_grad),
                    max_steps=int(optimize_max_iter),
                    return_result=True,
                    route_plan=opt_route_plan,
                )
            else:
                opt_result = optimize_periodic_positions(
                    system,
                    opt_energy,
                    derivatives_fn=opt_derivatives,
                    gradient_tolerance_ha_bohr=float(optimize_conv_tol_grad),
                    max_steps=int(optimize_max_iter),
                    route_plan=opt_route_plan,
                )
            if parameter_identity is not None and (
                opt_result.parameter_identity != parameter_identity
                or opt_result.parameter_sha256 != parameter_sha256
            ):
                raise RuntimeError(
                    "periodic semiempirical optimization used a different "
                    "immutable parameter snapshot than its initial energy"
                )
            with OutputChannel.to_file(out_path, mode="a"):
                write("\n" + _optimized_geometry_summary(opt_result))
                write(_system_summary(opt_result.system))
                if not opt_result.converged:
                    write(
                        "\n  FATAL: full-k periodic semiempirical geometry "
                        "optimization did not converge; refusing to mark the "
                        "calculation complete.\n"
                    )
                flush()
            _output_writer.record(out_path)
            if not opt_result.converged:
                raise RuntimeError(
                    "full-k periodic semiempirical geometry optimization did "
                    "not converge; refusing to mark the calculation complete."
                )
            if write_xyz_file:
                _output_writer.dispatch_role(
                    "geometry",
                    runtime_path=stem_sibling(output_stem, ".opt.xyz"),
                    runtime_format="extended-xyz",
                    runtime_description="Optimized periodic semiempirical geometry.",
                    system=opt_result.system,
                    energy_ha=float(opt_result.energy),
                    raise_on_error=True,
                )
            if write_poscar_file:
                _output_writer.dispatch_role(
                    "geometry",
                    runtime_path=stem_sibling(output_stem, ".opt.POSCAR"),
                    runtime_format="poscar",
                    runtime_description="Optimized periodic semiempirical POSCAR geometry.",
                    system=opt_result.system,
                    comment=f"vibe-qc optimized periodic {method_key}",
                    raise_on_error=True,
                )
            result = opt_result

        t_total = time.perf_counter() - t_start
        if citations and cite_block_text:
            _output_writer.set_citations(cite_manifest_rows)
        if output_qvf:
            from vibeqc.output.formats.qvf import (
                assemble_run_record,
                terminal_run_status,
            )

            # --- Population summary for QVF atom_properties ----------
            # Semiempirical methods carry native Mulliken charges when the
            # engine exposes them (DFTB SCC variants, GFN2-xTB).  Build a
            # population summary so the QVF archive carries at least
            # atom_properties rather than being an empty shell.
            _qvf_pop = None
            try:
                _native_charges = _periodic_semiempirical_attr(
                    result, "charges"
                )
                if _native_charges is not None:
                    from vibeqc.output.formats.population import (
                        compute_native_mulliken_population_summary,
                    )

                    _qvf_pop = compute_native_mulliken_population_summary(
                        _native_charges,
                        system.unit_cell_molecule(),
                        method_key,
                    )
            except Exception:
                pass  # optional enrichment; empty QVF is still valid

            # Symmetry data for structure.symmetry (semiempirical).
            _qvf_symmetry_se = None
            try:
                _sg = getattr(qvf_system, "symmetry", None)
                if _sg is not None:
                    _qvf_symmetry_se = {
                        "space_group_number": int(getattr(_sg, "number", 0)),
                        "space_group_symbol": str(
                            getattr(_sg, "international_symbol", "")
                        ),
                        "point_group": str(getattr(_sg, "point_group", "")),
                    }
            except Exception:
                pass

            qvf_system = getattr(result, "system", None) or system
            _output_writer.dispatch_role(
                "qvf",
                atomic=True,
                record_hostname=record_hostname,
                system=qvf_system,
                result=result,
                method=method_key,
                basis=basis_label,
                wall_seconds=t_total,
                bibtex_content=bibtex_content,
                population_summary=_qvf_pop,
                job_spec={
                    "job_type": "periodic",
                    "method": method_key,
                    "basis": basis_label or "",
                    **(
                        {
                            "options": {
                                "parameter_identity": parameter_identity,
                                "parameter_sha256": parameter_sha256,
                            }
                        }
                        if parameter_identity is not None
                        else {}
                    ),
                },
                symmetry_data=_qvf_symmetry_se,
                run_record=assemble_run_record(
                    plan, wall_seconds=t_total
                ),
                run_status=terminal_run_status(result),
                raise_on_error=True,
            )
        _output_writer.finish(wall_seconds=t_total)
        return result
    except Exception:
        _output_writer.crash(wall_seconds=time.perf_counter() - t_start)
        raise


def _finalize_periodic_checkpoint(
    _checkpointer: object,
    result: object,
    system: object,
    method_upper: str,
    basis_name: str,
    functional: object,
) -> None:
    """Single terminal-checkpoint finalize callsite (AST contract).

    ``tests/test_periodic_runner_bipole_callsites.py`` pins the module to
    exactly one direct ``_checkpointer.finalize`` call whose status
    argument is ``_checkpoint_terminal_run_status(result)``. Every
    terminal checkpoint (converged, the fail-closed SCF gate, and
    optimization) therefore routes through this helper and settles to the
    same status the QVF job-container finalizer derives: ``"failed"``
    only for a falsy ``result.converged``.
    """
    from .output.formats.qvf import (
        terminal_run_status as _checkpoint_terminal_run_status,
    )

    _final_system = getattr(result, "system", None) or system
    _checkpointer.finalize(
        _checkpoint_terminal_run_status(result),
        system=_final_system,
        result=result,
        method=method_upper,
        basis=basis_name,
        functional=functional,
    )


@_periodic_output_lifecycle
def run_periodic_job(
    system: PeriodicSystem,
    basis: Optional[BasisSet],
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    # AICCM front door (handovers/HANDOVER_AICCM_STANDARD_METHOD.md M1):
    # with method="aiccm", ``variant`` is mandatory and names the formulation
    # ("real-gamma" | "neutral-bloch" | "four-center" | "chi"); the SCF
    # reference is inferred (functional -> HF/KS; the system's multiplicity
    # and electron parity -> restricted/unrestricted) unless
    # ``scf_reference="rohf"`` / ``"roks"`` selects a restricted open-shell
    # reference explicitly. Both fail closed with any other ``method``.
    variant: Optional[str] = None,
    scf_reference: Optional[str] = None,
    # M4b (#778): the optional post-HF treatment on top of the variant's SCF.
    # ``None`` means SCF only. Orthogonal to ``variant``: the reference stays
    # whatever the variant and the D-3 inference chose, and the correlation
    # driver is picked to MATCH that construction, never to substitute
    # another one (ruling R1; maintainer 2026-09-08).
    correlation: Optional[str] = None,
    jk_method: Union[str, "PeriodicJKMethod"] = "auto",
    # Plane-wave grid cutoff (Hartree) for GPW / GAPW routes.
    # Default 300 Ha ≈ 600 Ry gives ~10⁻⁴ Ha grid convergence on
    # compact contracted Gaussians. Ignored for GDF / BIPOLE routes.
    cutoff_ha: float = 300.0,
    aux_basis: Optional[str] = None,
    # GDF backend selection for jk_method="gdf". ``None`` selects rsgdf
    # (the per-path default for the multi-k + Γ open-shell drivers AND, since
    # 2026-06-15, the Γ closed-shell RHF path: a plain dim=3 neutral Γ RHF
    # now routes through run_pbc_gdf_rhf with exxdiv='ewald' -- PySCF µHa
    # parity -- instead of the legacy molecular-limit gamma driver, which is
    # kept only for RKS / dim<3 / charged / smeared / symmetry runs). Set to
    # "compcell" / "rsgdf" / "mdf" to choose explicitly; "mdf" (Mixed
    # Density Fitting) closes the all-electron Gaussian-DF floor. All three
    # explicit values route Γ closed-shell RHF through run_pbc_gdf_rhf.
    # rsgdf_ke_cutoff controls the reciprocal-space auxiliary mesh used by the
    # range-separated GDF backend. rsgdf_tail_ke_cutoff optionally completes
    # the exact high-|G| shell for GDF and fitted aiccm2026dev-b routes; the
    # latter transport is diagnostic and does not by itself qualify D93.
    # mdf_ke_cutoff is the modest residual-mesh cutoff used only by
    # gdf_method="mdf".
    gdf_method: Optional[str] = None,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    mdf_ke_cutoff: float = 40.0,
    # Electron-repulsion representation for jk_method="aiccm2026dev-b":
    # corrected-gauge direct four-centre, pair-resolved RI, or RIJCOSX.
    aiccm_backend: str = "four_center",
    # Primary B-stream finite-size control: number of primitive lattice
    # vectors in each cyclic supercell direction.  The equivalent k net is
    # derived, never chosen independently.  ``aiccm_wigner_seitz_shells=s``
    # is the odd-cluster shorthand N=2s+1 (s complete layers per side).
    aiccm_lattice_extension: Optional[Union[int, Sequence[int]]] = None,
    aiccm_wigner_seitz_shells: Optional[Union[int, Sequence[int]]] = None,
    # B-stream-only space-group mode. ``diagnostic`` constructs and checks
    # the finite-torus subgroup and k orbits without changing the SCF;
    # ``integrals`` fails closed until petite-list parity is established.
    aiccm_symmetry: str = "off",
    aiccm_symmetry_require_full_group: bool = False,
    output: Union[str, os.PathLike] = "output",
    use_diis: bool = True,
    # Diagonalisation solver for the SCF procedure.
    # "dense"  -- NumPy/ScaLAPACK dense eigh (default).
    # "davidson" -- block-Davidson iterative solver.
    # "lobpcg" -- LOBPCG iterative solver (handled by Python SCF loop).
    solver: str = "dense",
    # Convergence strategy: "auto" fills any convergence knob the user
    # did not set, from a cheap pre-SCF classification of the system
    # (ionic / covalent / metallic / molecular-limit); "off" keeps the
    # plain defaults. Omitted (None) -> auto unless any explicit knob
    # below is given. The chosen strategy, its per-knob values, and the
    # classification reasons are stated in the .out file. v1 scope:
    # jk_method="bipole" and jk_method="gdf" (other routes fall back
    # to plain defaults).
    convergence: Optional[str] = None,
    # None = not given (auto strategy may choose); a float (incl. 0.0)
    # is an explicit user choice that auto never overrides.
    damping: Optional[float] = None,
    # Explicitly control the B selector's underlying SCF dynamic-damping
    # controller. ``None`` preserves the option-struct default; ``False``
    # disables adaptive damping updates. Other J/K selectors fail closed on an
    # explicit value until their dispatches implement the same contract.
    dynamic_damping: Optional[bool] = None,
    fmixing_percent: Optional[float] = None,
    fock_mixing: Optional[float] = None,
    density_mixer: Optional[str] = None,
    density_mixer_depth: int = 8,
    density_mixer_beta: float = 0.5,
    density_mixer_kerker: bool = False,
    kerker_k0: float = 1.5,
    kerker_strength: float = 1.0,
    kerker_cutoff_ha: float = 120.0,
    smearing: Optional[SmearingOptions] = None,
    smearing_temperature: Union[float, str, None] = _SMEARING_UNSET,  # type: ignore[assignment]
    smearing_unit: str = "hartree",
    smearing_method: str = "fermi-dirac",
    smearing_metallic: Optional[bool] = None,
    smearing_band_gap_hartree: Optional[float] = None,
    # BZ integration backend. None / "smearing" (temperature-broadening,
    # the default), or "gilat" (parameter-free Gilat-Raubenheimer net,
    # T=0 tetrahedron-family integrator). When not given (sentinel),
    # auto-reads from a KPoints.recommend() result's .bz_integration
    # attribute. Explicit user args win over KPoints metadata.
    bz_integration: Optional[str] = _BZ_INTEGRATION_UNSET,
    diis_start_iter: int = 2,
    diis_subspace_size: int = 8,
    max_iter: int = 80,
    conv_tol_energy: float = 1e-7,
    initial_guess: Union[str, InitialGuess] = "AUTO",
    write_molden_file: bool | None = None,
    trexio: bool | str | os.PathLike = False,
    trexio_backend: str = "hdf5",
    write_density: bool | None = False,
    density_spacing_bohr: float = 0.2,
    write_xyz_file: bool = True,
    write_poscar_file: bool = False,
    write_xsf_structure_file: bool = True,
    write_cif_file: bool = True,
    write_population_file: bool | None = None,
    citations: bool = True,
    dry_run: bool = False,
    memory_override: bool = False,
    gapw_molecular_limit: bool = False,
    record_hostname: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    # --- Dispersion ---------------------------------------------------
    dispersion: Optional[Union[str, bool, "D3BJParams"]] = None,
    dispersion_backend: str = "auto",
    dispersion_cutoff_bohr: float = 50.0,
    # --- BIPOLE-specific ---------------------------------------------
    # Direct-lattice cutoff (bohr) for BIPOLE Fock and nuclear sums.
    # ``cutoff_ha`` is a plane-wave grid cutoff and remains GPW/GAPW-only.
    bipole_cutoff_bohr: Optional[float] = None,
    bipole_nuclear_cutoff_bohr: Optional[float] = None,
    bipole_exact_zone_bohr: Optional[float] = None,
    # M5 production default for the erfc SR internal ket-image ball.
    # ``None`` restores the historical unpadded traversal.
    sr_image_precision: Optional[float] = 1e-6,
    # Separation-aware charge-pair Schwarz
    # screening for the BIPOLE SR erfc J/K build
    # (LatticeSumOptions.sr_range_screening). The M5 precision path
    # enables it automatically; this explicit flag also permits screened
    # historical/unpadded diagnostics. Ignored by non-BIPOLE routes.
    sr_range_screening: bool = False,
    ewald_omega: Optional[float] = None,
    ewald_precision: float = 1e-8,
    use_oda: bool = False,
    oda_trust_lambda_max: float = 1.0,
    use_mom: bool = False,
    # None = not given (auto strategy may choose); a float (incl. 0.0)
    # is an explicit user choice that auto never overrides.
    level_shift: Optional[float] = None,
    # --- Multi-k (GDF, RIJCOSX, and BIPOLE) -------------------------
    kpoints: Optional[
        Union[Tuple[int, int, int], List[int], int, "KPoints", "BlochKMesh"]
    ] = None,
    # --- Cell reduction -----------------------------------------------
    reduce_to_primitive: bool = False,
    symmetry_precision: float = 1e-4,
    symmetry: Union[bool, str] = False,
    symmetry_stabilize: bool = False,
    symmetry_reduce_fock: Optional[bool] = None,
    # Space-group reduction of the multi-k GDF exchange build (the driver's
    # ``ibz_native``). Exchange is the one n_k^2 term: under this flag K is
    # built only at the irreducible-wedge bras and symmetry-transported to
    # the rest of each star, so the dense Lpq cache holds n_IBZ x n_k blocks
    # instead of n_k^2. Everything else -- diagonalisation at every k, the
    # occupations, the energy expression, the result shape -- is unchanged,
    # so the reduction is exact rather than an approximation. Requires an
    # attached symmetry model (pass symmetry='attach'), a true multi-k
    # Monkhorst-Pack mesh, exact exchange, and a symmorphic space group;
    # every one of those fails closed rather than silently running full-BZ.
    symmetry_reduce_k: bool = False,
    # --- BIPOLE multipole far-field ----------------------------------
    # Fail-closed: the dormant Saunders 1992 quartet-level bipolar
    # far-field (G6) prototype does not preserve the exact route's
    # three-translation Fock domain, so an explicit True raises
    # NotImplementedError here and in every direct driver
    # (reject_bipole_quartet_far_field in pbc_bipole_common.py).
    # "Here" is unconditional as of #511: on jk_method='bipole' the
    # quartet-domain refusal fires, and on every other route the flag
    # is refused as a BIPOLE-only control rather than dropped in
    # silence (only the four direct BIPOLE drivers ever read it).
    # False -- the default -- selects the exact four-centre traversal,
    # the only supported route (the retired G1 cell-level path never
    # engages). The dipolar far-field accuracy remains uncertified:
    # the 2026-08-13 LiH measurement puts the exact-vs-far-field error
    # at 6.3e-2 Ha at 18 bohr.
    use_multipole_far_field: bool = False,
    multipole_l_max: int = 2,
    # --- BIPOLE exchange convention (option (b), 2026-06-10) ----------
    # None = auto: the Ewald exchange split (corrected gauge -- full
    # Bloch density, split K with the exxdiv G=0 correction, no
    # spheropole term) is ON for 3D Γ-only RHF BIPOLE runs and OFF
    # otherwise (multi-k pends the q!=0 LR-exchange channels). Pass
    # False to force the legacy Γ-locality gauge (needed e.g. to
    # combine the multipole far-field branch with Γ sampling, or for
    # the analytic-gradient preview). 'exchange_exxdiv' picks the K=0
    # convention: 'ewald' (probe-charge Madelung; PySCF-equivalent
    # default) or 'none'.
    use_exchange_ewald_split: Optional[bool] = None,
    exchange_exxdiv: str = "ewald",
    # --- Geometry optimization ----------------------------------------
    optimize: bool = False,
    optimize_max_iter: int = 30,
    optimize_conv_tol_grad: float = 1e-4,
    optimize_cell: bool = False,
    # QVF visualisation archive (v1).
    output_qvf: bool = True,
    # Optional vibe-view overlay: localize χ-CCM-B occupied orbitals and
    # embed x_ccm.wannier_centers in the QVF archive.
    qvf_wannier_centers: bool = False,
    # Opt-in live QVF checkpointing for vibe-view hot-reload. When
    # ``checkpoint_qvf`` is set, a running snapshot is atomically
    # (re)written there as the SCF climbs -- every ``checkpoint_every``
    # cycles -- carrying ``provenance.run_status="running"`` and a
    # monotonic ``provenance.checkpoint.seq``; the terminal snapshot is
    # labeled ``"converged"`` / ``"failed"``. ``checkpoint_every=0``
    # keeps the start + end frames but no per-iteration cadence.
    checkpoint_qvf: Optional[Union[str, os.PathLike]] = None,
    checkpoint_every: int = 0,
    # Harmonic vibrational frequencies via finite-difference Hessian.
    hessian: bool = False,
    # Partial Hessian: atom indices to hold fixed (e.g. bulk-like bottom
    # slab layers) so only the unfrozen atoms are displaced -- 6M instead
    # of 6N SCFs. Frozen atoms anchor the cell, so the frequencies are
    # vibrational-only (no gas-phase trans/rot). See HessianFDOptions.
    hessian_frozen_indices: Optional[List[int]] = None,
    # Pre-computed band structure to include in QVF (vibe-view bands plot).
    band_structure: Optional["BandStructure"] = None,
    # Compute COOP/COHP bonding analysis (requires output_qvf=True;
    # refused pre-SCF otherwise, IID 195).
    coop_cohp: bool = False,
    # TD-DFT excited states (Gamma-point only, TDA).
    tddft: bool = False,
    tddft_n_states: int = 5,
    # DOS/PDOS/COOP k-mesh dimensions for QVF output (default [8,8,8]).
    # For 1D/2D systems, consider e.g. [32,1,1] or [12,12,1].
    dos_kmesh: Optional[Sequence[int]] = None,
    # --- DFT+U (Dudarev rotationally-invariant) ------------------------
    # Shipped on main (see docs/user_guide/dft_plus_u.md):
    #   - RHF +U: Γ-only, via vibeqc.run_rhf_periodic_gamma
    #     (multi-k RHF +U raises NotImplementedError).
    #   - RKS +U: Γ and multi-k both supported, via vibeqc.run_rks_periodic.
    #     Multi-k (Increment 4c) uses the k-averaged AO occupation matrix
    #     and adds S(k) V_AO S(k) per k in cpp/src/periodic_scf.cpp.
    #   - UHF / UKS +U: Γ GPW and Γ/multi-k BIPOLE are supported via
    #     their open-shell drivers.
    #   - GDF +U: queued (GDF Γ-only driver has no +U hook yet).
    dft_plus_u: Optional[List["HubbardSite"]] = None,
    # --- ATOMSPIN: broken-symmetry magnetic seed (UHF/UKS only) -------
    # Per-atom spin tag in unit-cell atom order: +1 (majority alpha),
    # -1 (majority beta), 0 (unpolarised). Seeds an AFM / ferrimagnetic
    # g=0 spin pattern (Bloch-sums to a broken-symmetry D(k)). Supported
    # on the Γ UHF/UKS Ewald / GDF / BIPOLE drivers and the multi-k UHF/UKS
    # Ewald drivers.
    # Requires the effective SAD construction. See docs/roadmap.md Sec.G2.
    atomic_spins: Optional[List[int]] = None,
    # --- READ: restart from a prior SCF -------------------------------
    # Gamma periodic restarts use the prior g=0 cell density, projected onto
    # this cell's basis. Closed-shell multi-k GDF/GPW/GAPW restarts accept an
    # in-memory prior result and rebuild the per-k Bloch density blocks from
    # its coefficients/occupations. QVF sources must carry a complete all-k
    # Bloch restart payload; selected visualization orbitals do not suffice.
    read_from: Optional[object] = None,
    fragments: Optional[object] = None,
    # --- SPINLOCK: broken-symmetry magnetic convergence (UHF/UKS only) -
    # "pattern_hold" holds the seeded (ATOMSPIN) occupied set by maximum
    # overlap (MOM) for ``spinlock_iterations`` cycles, then releases --
    # protecting an AFM seed from collapsing to the symmetric solution.
    # "spin_schedule" runs a two-phase SCF: converge at locked
    # n_alpha-n_beta = ``spinlock_value`` for ``spinlock_iterations`` cycles,
    # then release to the multiplicity target. PATTERN_HOLD is supported on
    # the Γ-Ewald, GDF, BIPOLE and multi-k UKS Ewald drivers; SPIN_SCHEDULE
    # on the Γ-Ewald, Γ-GDF and Γ-BIPOLE drivers (unsupported
    # (driver, mode) pairs fail closed). See docs/roadmap.md Sec.G2.
    spinlock: Optional[str] = None,
    spinlock_value: int = 0,
    spinlock_iterations: int = 0,
    # Restart a Gamma GPW/GAPW RHF/RKS calculation from an .npz file written
    # by ``save_gpw_result``. Multi-k and open-shell routes fail closed because
    # the current archive contains one closed-shell density, not per-k or
    # per-spin blocks.
    restart_from: Optional[Union[str, os.PathLike]] = None,
):
    """Run a periodic SCF job and write the standard output files.

    Mirrors :func:`vibeqc.run_job` but for periodic systems.
    Gamma RHF/RKS GDF is the default when no k-mesh is specified.
    ROHF/GDF is available by explicit backend selection at Gamma and on a
    full Monkhorst-Pack mesh. Multi-k RHF/RKS GDF, multi-k
    RHF/RKS/UHF/UKS RIJCOSX, and all four BIPOLE methods are available via
    ``kpoints``. These routes accept a
    mesh tuple/list, a scalar mesh size, :class:`KPoints`, or a native
    :class:`BlochKMesh`.

    .. warning::

       ``kpoints=(N, N, N)`` builds the **Gamma-centred** mesh
       ``{0, 1/N, ..., (N-1)/N}`` per axis. ASE/GPAW read the same-looking
       argument as the **classical shifted** mesh, offset by half a step.
       The two agree exactly for **odd** N and are *disjoint* for **even**
       N (measured 2026-08-02: ``(3,3,3)``/``(5,5,5)`` share every point,
       ``(2,2,2)``/``(4,4,4)``/``(6,6,6)`` share none), so a cross-code
       comparison validated at an odd mesh can still be sampling a
       different Brillouin zone at an even one. The resolved convention
       and k-point count are printed in the ``.out``; pass
       ``VIBEQC_OUTPUT_LEVEL=verbose`` for the full k-list.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis.  For basis-free periodic semiempirical
        methods (`dftb0`, `scc_dftb`, `gfn2_xtb`, `pm6`, `om1`, `om2`,
        `om3`), pass `basis=None`; the runner dispatches before any
        Gaussian-basis setup.
    method
        ``"RHF"``, ``"ROHF"``, ``"ROKS"``, ``"UHF"``, ``"RKS"``, ``"UKS"``,
        or ``"aiccm"`` (case-insensitive). ``method="aiccm"`` is the front
        door of the experimental AICCM family: ``variant`` names the
        formulation, the SCF reference is inferred (see ``variant`` and
        ``scf_reference`` below), and the formulation runs on its own J/K
        route, so ``jk_method`` stays ``"auto"`` (any other value must equal
        that route, else a ``ValueError`` names both). Every AICCM job
        stamps ``[run].method_status = "experimental"`` plus
        ``aiccm_variant`` / ``aiccm_selector`` (plus ``aiccm_correlation``
        when one was requested) into the ``.system`` manifest
        and keeps its experimental warning. See ``docs/user_guide/aiccm.md``.
        Periodic ROHF/ROKS are maintained-preview 3D routes:
        ROHF runs on native GDF at Gamma or on full Monkhorst-Pack meshes
        (``jk_method="gdf"``), on Gamma-only GPW (``jk_method="gpw"``),
        or on the BIPOLE-route corrected-Ewald-exchange EWALD_3D engine
        (``jk_method="bipole"`` -- the AUTO default for ROHF -- at Gamma
        and on full Monkhorst-Pack meshes). ROKS runs through Gamma-only GPW
        for pure DFT, or through BIPOLE at Gamma and on full Monkhorst-Pack
        meshes for pure and global-hybrid functionals. Unsupported backends,
        multi-k GPW ROKS, range-separated ROKS,
        smearing, gradients, and response properties fail closed.
        Closed-shell RHF / RKS default to the Gamma or multi-k GDF path
        depending on ``kpoints``. With ``jk_method="bipole"`, the
        RHF/ROHF/RKS/UHF/UKS methods
        dispatch through the BIPOLE Gamma or multi-k route. With
        ``jk_method="rijcosx"``, Gamma RHF uses the dedicated RIJCOSX
        driver and true multi-k meshes use the GDF/COSX backend for
        RHF/RKS/UHF/UKS.
    functional
        XC functional for ``method="RKS"`` or ``method="UKS"``; with
        ``method="aiccm"`` its presence selects a Kohn-Sham reference.
    variant
        Required with ``method="aiccm"``, rejected otherwise. One of
        ``"real-gamma"`` (neutral Γ-CCM, real-Γ supercell representation;
        the former ``jk_method="real-gamma"``), ``"neutral-bloch"`` (the same
        neutral torus Hamiltonian in its Bloch representation through the
        ``periodic.ccm.ri`` producer, not plain unit-cell GDF),
        ``"four-center"`` (the union-and-weight 2014 lineage), or
        ``"chi"`` (χ-CCM; the former ``jk_method="aiccm2026dev-b"``, with the
        ``aiccm_*`` keywords as its options). ``"gamma"`` / ``"gamma-ccm"``
        are not variants (ruling R1). Every variant reads
        ``aiccm_lattice_extension`` for the BvK torus; ``kpoints`` stays an
        accepted alias for the torus mesh, never both.
    scf_reference
        With ``method="aiccm"`` only. ``None`` (default) infers the
        reference: ``functional`` decides HF versus KS, and the system is
        open-shell when ``system.multiplicity != 1`` or the electron count
        is odd (unrestricted), else restricted -- the parity rule the CCM
        library's ``run_ccm_scf`` applies. ``"rohf"`` / ``"roks"`` select
        the restricted open-shell references explicitly (``"rohf"`` rejects
        a ``functional``; ``"roks"`` requires one); whether the variant runs
        them is decided by that variant's own guard (none does today).
    correlation
        With ``method="aiccm"`` only, and optional: ``None`` (default) is an
        SCF-only run. ``"mp2"``, ``"ccsd"``, ``"dlpno-mp2"`` and
        ``"dlpno-ccsd"`` add a post-HF treatment on top of the variant's
        SCF. ``"mp2"`` and ``"ccsd"`` are wired on ``variant="four-center"``
        and ``variant="real-gamma"``; the two DLPNO treatments are
        ``"real-gamma"`` only, because DLPNO screens on a fitted reference
        and the bare four-centre operator has no RI decomposition. At their
        default zero truncations the DLPNO routes reproduce the canonical
        ones on the same reference to machine precision. ``"ccsd"`` and
        ``"dlpno-ccsd"`` are CCSD(T), and their citation keys record whether
        the triples actually ran. Open-shell clusters take the ``u`` siblings
        of every route automatically; the one refusal is ``"ccsd"`` on
        ``variant="four-center"``, whose only open-shell driver is
        neutral-only and so cannot build that arm's reference. The driver is chosen to MATCH the
        construction the SCF ran, never to substitute another one, so each
        arm cites its own lineage: ``four-center`` takes the bare
        ``run_ccm_mp2`` / ``run_ccm_ump2`` (``aiccm2026dev-a-mp2``) and
        ``real-gamma`` the neutral-RI correlation
        (``aiccm2026dev-a-ri-mp2``). They never share a citation row --
        handing a union-and-weight SCF to a neutral-RI driver would return a
        different construction's number (ruling R1). Both consume the SCF
        the front door already converged rather than building a second one.
        ``neutral-bloch`` refuses because its reference is the per-k Bloch
        representation while the drivers work in real-Gamma supercell space;
        the refusal names ``real-gamma`` as the exact substitute (same
        Hamiltonian, ruling R1). ``chi`` refuses as not yet wired. A KS
        reference refuses because MP2 needs an HF one. The bare four-centre
        drivers form the dense ``n_ref_ao**4`` AO tensor, so that arm is a
        small-cluster tool: no dimensionality guard, but size the torus
        deliberately. See ``docs/user_guide/aiccm.md``.
    output
        Path stem; produces ``{output}.out``, ``{output}.system``,
        ``{output}.molden``, ``{output}.xsf`` (when ``write_density``).
    band_structure
        Optional :class:`BandStructure` pre-computed by
        :func:`vibeqc.band_structure` (or ``_hcore``).
        When given together with ``output_qvf=True``, the band
        structure is embedded in the QVF archive so vibe-view can
        render an interactive Plotly band-structure plot.
        Compute it before calling this function -- the same workflow
        used for matplotlib plotting with
        :func:`vibeqc.plot.band_structure_figure`.
    coop_cohp
        When ``True``, compute COOP and COHP bonding analysis and
        embed ``dos.coop`` + ``dos.cohp`` sections in the QVF
        archive. Requires ``output_qvf=True``: the archive is the
        analysis's only sink, so ``coop_cohp=True`` with
        ``output_qvf=False`` raises ``ValueError`` before SCF
        rather than silently skipping the analysis. Uses the same
        DOS k-mesh as the total/projected DOS (``[8,8,8]``).
        The Hcore matrix (T + V) needed for COHP is computed
        independently within the DOS/COHP k-mesh block; no
        additional user input is required.
    tddft
        When True, compute TD-DFT vertical excitation energies via
        the Tamm-Dancoff approximation (TDA) at the Gamma point.
        Writes excitation energies, oscillator strengths, and
        dominant transitions to the .out file. Requires
        ``_has_valid_mo_coeffs(result)`` (true for Γ-only and
        multi-k results). Not embedded in QVF yet.
    tddft_n_states
        Number of excited states to compute when ``tddft=True``.
        Default 5.
    dos_kmesh
        Override the DOS/PDOS/COOP k-mesh dimensions. Default
        ``[8, 8, 8]``. For 1D systems use e.g. ``[32, 1, 1]``;
        for 2D use ``[12, 12, 1]``. Only used when
        ``output_qvf=True``.
    qvf_wannier_centers
        When True for ``jk_method="aiccm2026dev-b"``, localize the occupied
        finite-torus space with the B-owned Wannier gauge and embed an
        ``x_ccm.wannier_centers`` vendor section in the QVF archive for
        vibe-view's centre overlay. Requires ``output_qvf=True``.
    aux_basis
        Optional auxiliary basis for ``jk_method="gdf"`` and the RI-J part
        of ``jk_method="rijcosx"``. If omitted, vibe-qc chooses the current
        native-GDF default for ``basis.name``.
    gapw_molecular_limit
        Declare that ``jk_method="gapw"`` is describing an isolated molecule
        or atom in a vacuum-padded single-Gamma cell. Required for RHF/UHF so
        the method-aware one-centre default can select the validated fit-free
        analytic augmentation without guessing from geometry. Compact
        crystals should use GDF or BIPOLE. Rejected for RKS/UKS, whose
        method-aware default remains the block DFT functional. High-level
        GAPW HF geometry/cell optimization and Hessians fail closed until
        derivatives of the same fit-free energy are implemented.
    bipole_cutoff_bohr, bipole_nuclear_cutoff_bohr
        Direct-lattice cutoff radii in bohr for ``jk_method="bipole"``.
        ``bipole_cutoff_bohr`` controls the electronic BIPOLE Fock sums; when
        ``bipole_nuclear_cutoff_bohr`` is omitted, the BIPOLE route keeps the
        nuclear/Ewald real-space cutoff no longer than the electronic J/K
        cutoff so neutral-cell cancellation stays in the corrected Ewald
        gauge. Corrected-gauge RHF/RKS/UHF/UKS runs measure Bloch-overlap
        fold drift before the first Fock build. Drift above ``1e-2`` is the
        established unreliable-support regime and raises before SCF; increase
        this cutoff until the reported drift is below ``1e-4`` for
        quantitative work. Intermediate drift retains a truncation note.
        A home-only short-range image list is valid for the 3D Ewald split:
        its reciprocal term still supplies periodic coupling. Converge both
        halves of the split; the number of short-range images alone does not
        determine whether the Hamiltonian is periodic. Other backends have
        route-specific image semantics.
        These controls are separate from
        ``cutoff_ha``, which is a GPW/GAPW plane-wave grid cutoff.
    convergence
        Convergence-strategy selector. ``"auto"`` classifies the system
        from cheap pre-SCF signals (composition electronegativity
        spread, cell volume, vacuum axes, electron parity) into a
        profile -- ionic-insulator, covalent-insulator,
        metallic-candidate or molecular-limit -- and fills every
        convergence knob the user did not set (Fermi-Dirac smearing,
        FMIXING, level shift, damping) with profile defaults grounded
        in measured behaviour. MgO-class ionic cells get FMIXING 30 %
        with integer occupations; smearing is never selected automatically
        for an insulating profile because it can converge a wrong-energy
        metallic basin.
        ``"off"``/``"none"`` keeps the plain defaults. Omitted
        (``None``): auto applies **only when no explicit convergence
        knob is given** -- any explicit ``damping=`` /
        ``fmixing_percent=`` / ``fock_mixing=`` / ``level_shift=`` / smearing input
        switches to fully-manual mode and nothing is auto-filled.
        Either way the ``.out`` file carries a "Convergence strategy"
        block stating the mode (AUTO default / AUTO requested / manual
        / off), the per-knob values with their provenance, and the
        classification reasons. Explicit knobs are never overridden.
        Scope: applied on ``jk_method="bipole"`` and ``"gdf"``;
        other routes run with plain defaults and label the block
        accordingly.
    use_diis, damping, dynamic_damping, fmixing_percent, fock_mixing, diis_start_iter,
    diis_subspace_size, max_iter, conv_tol_energy
        SCF controls forwarded to the periodic driver.
        ``dynamic_damping`` is a χ-CCM-B-only override. ``None`` preserves
        that selector's option default, while ``False`` disables adaptive
        damping updates. Supplying it explicitly with any other ``jk_method``
        fails closed. It is separate from the generic ``convergence="auto"``
        explicit-knob detection; χ-CCM-B campaign callers pin their generic
        convergence controls independently.
        ``fmixing_percent`` mirrors CRYSTAL's ``FMIXING`` keyword:
        the percentage of the previous Fock/KS matrix mixed into the
        matrix diagonalised on the next cycle. It is separate from
        density damping.
        ``fock_mixing`` is the same knob on the fractional 0.0-1.0
        scale; pass only one spelling.
        ``density_mixer`` / ``density_mixer_kerker`` expose the periodic
        Anderson/Broyden/Kerker API surface on the closed-shell multi-k GDF
        and RIJCOSX routes. The lower-level Ewald RKS drivers also support
        these mixers directly. Other routes fail closed on active requests
        instead of silently ignoring them or changing electrostatic gauge.
        Compact 3D closed-shell RKS/GDF SCAN/r2SCAN-family jobs with an
        explicit ``kpoints=`` mesh select Anderson density mixing by default
        unless ``density_mixer=`` or ``convergence="off"`` is set.
        ``smearing`` accepts the new :class:`vibeqc.SmearingOptions`
        surface. The legacy ``smearing_temperature`` may be a numeric
        electronic ``k_B T``
        (interpreted via ``smearing_unit``), ``"auto"``, ``"metal"``,
        ``"small-gap"``, ``"debug"``, ``"none"`` / ``"off"``, or
        ``None``. ``smearing_method`` selects ``"fermi-dirac"`` (default),
        ``"mermin"`` (Mermin finite-temperature free-energy functional),
        ``"methfessel-paxton"`` or ``"marzari-vanderbilt"`` (all
        implemented). ``smearing_metallic`` and ``smearing_band_gap_hartree``
        guide the conservative ``"auto"`` guess.
    initial_guess
        Initial-guess selector. ``"AUTO"`` is the default; ``"SAD"``,
        ``"HCORE"``, ``"SAP"``, ``"PATOM"``, ``"HUECKEL"``, ``"MINAO"``,
        and ``"READ"`` are available where supported by the selected
        method and periodic Coulomb route. Unsupported combinations fail
        before SCF instead of silently changing the guess.
        AUTO resolution is *variant-aware*: ``method="aiccm"`` with
        ``variant="real-gamma"`` or ``"four-center"`` resolves AUTO to
        ``"HCORE"``, because their supercell-Gamma SCF loops implement only
        that guess. Passing ``initial_guess="SAD"`` to those two variants
        explicitly still fails before SCF -- the default moves, a request
        never does (#692).
    write_molden_file
        Emit ``{output}.molden`` of the Γ-point MOs (using the unit-cell
        molecule + basis as the molecular target). ``None`` (the default)
        enables it for exact single-Γ Gaussian-basis SCF routes and for
        Gamma-containing BIPOLE/GDF meshes whose result metadata locates that
        block. It is inapplicable for basis-free and shifted meshes. Explicit
        ``True`` on an inapplicable route fails before calculation.
    trexio, trexio_backend
        Write the periodic Gaussian SCF wavefunction, including all k-point
        and spin blocks, lattice and applied scalar ECP parameters. ``True``
        selects ``{output}.trexio.h5`` (HDF5) or ``{output}.trexio`` (text);
        a path selects an explicit target. Requires the optional ``trexio``
        extra. See :doc:`/user_guide/trexio`.
    write_population_file
        Emit the population-summary text/JSON pair. Exact single-Γ routes use
        the molecular analysis; multi-k BIPOLE uses its lattice-density
        population and χ-CCM uses its finite-torus population. ``None`` is
        capability-aware auto mode, ``True`` is a guaranteed request, and
        ``False`` disables the pair.
    write_density
        Emit ``{output}.xsf`` with the SCF density on a primitive-cell
        grid (XSF works for any lattice; cube is orthorhombic-only).
        The artefact evaluates a real-space lattice density, or the exact
        Gamma block of a Gamma-only mesh. On a *true* multi-k mesh the plain
        Gaussian routes (``gdf``, ``rijcosx``, ``gpw``) return per-k Bloch
        blocks only, and a lattice density is not reconstructed from them:
        an explicit ``True`` there is refused before the SCF rather than
        after it (#679). ``bipole`` and every ``method="aiccm"`` variant
        carry a lattice density at any mesh. ``None`` is capability-aware
        auto mode, ``True`` is a guaranteed request, and ``False`` (the
        default) disables it.
    density_spacing_bohr
        Grid spacing for the XSF density. Default 0.2 bohr.
    solver
        Diagonalisation solver.
        ``"dense"`` (default) uses NumPy/ScaLAPACK dense eigh.
        ``"davidson"`` uses the block-Davidson iterative solver
        (``opts.use_davidson = True``). ``"lobpcg"`` also sets
        ``use_davidson = True``; the Python SCF loop then detects
        the LOBPCG preference and dispatches through the Python
        solver stack instead of dense or Davidson diagonalisation.
    hessian
        When True, compute harmonic vibrational frequencies for
        the unit-cell molecule via finite-difference Hessian.
        Frequencies and IR intensities printed to .out and
        embedded in QVF for vibe-view. Default False.
        Cost: ~6N SCF evaluations for the unit cell.
    """
    enforce_runtime_pin_from_env()
    if str(trexio_backend).strip().lower() not in ("hdf5", "text"):
        raise ValueError("run_periodic_job: trexio_backend must be 'hdf5' or 'text'.")
    if trexio is not None and not isinstance(trexio, (bool, str, os.PathLike)):
        raise TypeError("run_periodic_job: trexio must be a boolean or target path.")
    if isinstance(trexio, (str, os.PathLike)) and not os.fspath(trexio):
        raise ValueError("run_periodic_job: trexio target path cannot be empty.")
    if trexio and basis is None:
        raise NotImplementedError("run_periodic_job: TREXIO export requires a Gaussian AO basis.")


    # Fail fast on a molecular system. Passing a Molecule here used to die
    # deep in the setup with "'Molecule' object has no attribute 'lattice'";
    # say what the user actually needs instead.
    if not hasattr(system, "lattice") or not hasattr(system, "unit_cell"):
        raise TypeError(
            f"run_periodic_job: system must be a PeriodicSystem (a cell "
            f"with a lattice); got {type(system).__name__}. Molecules have "
            f"no lattice or Brillouin zone -- k-points, band structures, "
            f"and DOS meshes are periodic-boundary-condition features. Use "
            f"vibeqc.run_job(...) for molecular calculations, or build a "
            f"PeriodicSystem (set .lattice / .unit_cell / .dim) for a "
            f"crystal."
        )

    semiempirical_route = _plan_periodic_semiempirical_method(
        method,
        system,
        kpoints=kpoints,
    )
    if semiempirical_route is not None:
        if trexio:
            raise NotImplementedError("run_periodic_job: TREXIO export requires a Gaussian AO wavefunction.")
        if fragments is not None:
            raise NotImplementedError("periodic FRAGMO fragment sources require an HF/KS SCF route")
        _basis_free_reason = (
            "basis-free periodic semiempirical results do not expose the "
            "Gaussian AO wavefunction required by this writer."
        )
        write_molden_file = _resolve_sidecar_request(
            write_molden_file,
            supported=False,
            option="write_molden_file",
            method=semiempirical_route.method_key,
            caller="run_periodic_job",
            unavailable_reason=_basis_free_reason,
        )
        write_population_file = _resolve_sidecar_request(
            write_population_file,
            supported=False,
            option="write_population_file",
            method=semiempirical_route.method_key,
            caller="run_periodic_job",
            unavailable_reason=_basis_free_reason,
        )
        return _run_periodic_semiempirical_job(
            system,
            route_plan=semiempirical_route,
            output=output,
            dry_run=dry_run,
            record_hostname=record_hostname,
            citations=citations,
            write_xyz_file=write_xyz_file,
            write_poscar_file=write_poscar_file,
            write_xsf_structure_file=write_xsf_structure_file,
            write_cif_file=write_cif_file,
            write_molden_file=write_molden_file,
            write_density=write_density,
            write_population_file=write_population_file,
            output_qvf=output_qvf,
            jk_method=jk_method,
            aux_basis=aux_basis,
            smearing=smearing,
            smearing_temperature=smearing_temperature,
            smearing_unit=smearing_unit,
            smearing_method=smearing_method,
            smearing_metallic=smearing_metallic,
            smearing_band_gap_hartree=smearing_band_gap_hartree,
            dispersion=dispersion,
            optimize=optimize,
            optimize_max_iter=optimize_max_iter,
            optimize_conv_tol_grad=optimize_conv_tol_grad,
            optimize_cell_requested=optimize_cell,
            hessian=hessian,
            tddft=tddft,
            coop_cohp=coop_cohp,
            band_structure=band_structure,
            qvf_wannier_centers=qvf_wannier_centers,
            kpoints=kpoints,
            checkpoint_qvf=checkpoint_qvf,
            dft_plus_u=dft_plus_u,
            atomic_spins=atomic_spins,
            spinlock=spinlock,
            read_from=read_from,
            restart_from=restart_from,
            functional=functional,
            max_iter=max_iter,
            conv_tol_energy=conv_tol_energy,
        )

    # --- AICCM front door (HANDOVER_AICCM_STANDARD_METHOD.md M1) ------------
    # method="aiccm" names the family; ``variant`` names the formulation and
    # maps onto the existing jk_method dispatch through one member; the SCF
    # reference is inferred (D-3) so ``method_upper`` is one of the six
    # spellings the rest of this function already handles. The caller's
    # jk_method is kept for the .out "user-requested" line.
    _jk_method_input = jk_method
    _aiccm_front_door = str(method).strip().lower() == "aiccm"
    _aiccm_variant: Optional[str] = None
    if _aiccm_front_door:
        _aiccm_variant = resolve_aiccm_variant(variant)
        _aiccm_member = AICCM_VARIANT_ROUTES[_aiccm_variant]
        if not jk_method_agrees_with_variant(jk_method, _aiccm_variant):
            raise ValueError(
                f"run_periodic_job: method='aiccm', variant={_aiccm_variant!r} "
                f"runs on jk_method={_aiccm_member.value!r}; got "
                f"jk_method={jk_method!r}. Leave jk_method='auto' (the "
                "default) or pass the variant's own route."
            )
        jk_method = _aiccm_member
        method_upper = _infer_aiccm_scf_reference(
            system, functional=functional, scf_reference=scf_reference
        )
        _aiccm_correlation = resolve_aiccm_correlation(correlation)
        if _aiccm_correlation is not None:
            # Only the four-centre arm is wired (M4b, #778). The other three
            # variants refuse rather than silently running SCF-only: a job
            # that asked for correlation and got an SCF number back would be
            # indistinguishable from one that did not ask.
            if _aiccm_variant == "neutral-bloch":
                # Not merely unwired: the correlation drivers work in the
                # real-Γ supercell space, and this variant's reference is
                # per-k Bloch. Ruling R1 makes the workaround exact rather
                # than approximate -- real-gamma is the SAME Hamiltonian in
                # the representation the drivers already speak -- so name it
                # instead of leaving the caller to find it.
                raise NotImplementedError(
                    "run_periodic_job: correlation="
                    f"{_aiccm_correlation!r} is not available on "
                    "variant='neutral-bloch': its reference is the Bloch "
                    "representation, while the correlation drivers work in "
                    "the real-Gamma supercell space. Use "
                    "variant='real-gamma', which is the same neutral "
                    "Hamiltonian in that representation (ruling R1), and "
                    "carries the same citation lineage."
                )
            if _aiccm_variant not in ("four-center", "real-gamma"):
                raise NotImplementedError(
                    "run_periodic_job: correlation="
                    f"{_aiccm_correlation!r} is wired for "
                    "variant='four-center' and variant='real-gamma'; got "
                    f"variant={_aiccm_variant!r}. The chi correlation "
                    "drivers are library-only for now "
                    "(vibeqc.periodic.chi.posthf)."
                )
            if method_upper not in ("RHF", "UHF"):
                # Post-HF needs an HF reference. A KS determinant would give a
                # number with no defined meaning here, so name the reason.
                raise NotImplementedError(
                    "run_periodic_job: correlation="
                    f"{_aiccm_correlation!r} requires a Hartree-Fock "
                    f"reference; the inferred reference is {method_upper!r}. "
                    "Drop functional= (or pass scf_reference=) so the "
                    "variant infers RHF/UHF."
                )
            if (
                _aiccm_correlation.startswith("dlpno-")
                and _aiccm_variant != "real-gamma"
            ):
                # DLPNO screens pair energies on a fitted (RI) reference and
                # every driver takes the neutral cderi. The bare four-centre
                # omega^sym has no RI decomposition -- that is precisely why
                # the bare arm is dense and small-cluster only -- so there is
                # no construction-matched four-centre DLPNO to dispatch to,
                # and no producer stamps such a label.
                raise NotImplementedError(
                    "run_periodic_job: correlation="
                    f"{_aiccm_correlation!r} needs a fitted (RI) reference "
                    "and is available on variant='real-gamma' only; got "
                    f"variant={_aiccm_variant!r}. The bare four-centre "
                    "operator has no RI decomposition, so there is no "
                    "four-centre DLPNO. Use correlation='mp2' or 'ccsd' "
                    "there, or variant='real-gamma'."
                )
            if (
                _aiccm_correlation == "ccsd"
                and method_upper != "RHF"
                and _aiccm_variant == "four-center"
            ):
                # Open-shell CCSD is refused on THIS arm only. run_ccm_uccsd
                # takes no ``method=``, so it cannot build the
                # union-and-weight reference the four-centre variant
                # converged on, and no producer stamps a bare four-centre
                # open-shell CCSD route. It is neutral-only -- which makes it
                # exactly right on real-gamma, where this pairing runs.
                raise NotImplementedError(
                    "run_periodic_job: correlation='ccsd' is closed-shell "
                    "only on variant='four-center'; the inferred reference "
                    f"is {method_upper!r}. run_ccm_uccsd is neutral-only and "
                    "cannot build the union-and-weight reference. Use "
                    "variant='real-gamma' (where open-shell CCSD runs), "
                    "correlation='mp2', or an even-electron singlet cluster."
                )
    else:
        _aiccm_correlation = None
        if variant is not None or scf_reference is not None or correlation is not None:
            raise ValueError(
                "run_periodic_job: variant=, scf_reference= and correlation= "
                "select an AICCM formulation and require method='aiccm'; got "
                f"method={method!r}, variant={variant!r}, "
                f"scf_reference={scf_reference!r}, "
                f"correlation={correlation!r}."
            )
        method_upper = method.upper()
    if method_upper not in ("RHF", "ROHF", "ROKS", "RKS", "UHF", "UKS"):
        # A leading "K" is the PySCF spelling for the multi-k driver class
        # (KRHF/KRKS/KUHF), and vibe-qc's own result type is named
        # PeriodicKRHFGDFResult, so `method="KRHF"` is a natural thing to
        # reach for. It has never been a run_periodic_job method: k-point
        # sampling is not a *method*, it is the `kpoints` argument, and every
        # method below is multi-k whenever kpoints is set. Name the
        # replacement rather than dead-ending on the unsupported list.
        if method_upper.startswith("K") and method_upper[1:] in (
            "RHF",
            "ROHF",
            "ROKS",
            "RKS",
            "UHF",
            "UKS",
        ):
            raise NotImplementedError(
                f"run_periodic_job: method={method!r} not supported; k-point "
                f"sampling is not a separate method. Use "
                f"method={method_upper[1:]!r} and pass the mesh explicitly, "
                f"e.g. kpoints=(4, 4, 4). Every method runs multi-k when "
                f"`kpoints` is set (the default is the Gamma point)."
            )
        raise NotImplementedError(
            f"run_periodic_job: method={method!r} not supported. "
            f"Supported: RHF, ROHF, ROKS, RKS, UHF, UKS."
        )
    if basis is None:
        raise TypeError(
            f"run_periodic_job: method={method_upper!r} requires a BasisSet. "
            "Pass basis=None only for basis-free periodic semiempirical "
            "methods such as 'scc_dftb' or 'pm6'."
        )
    if method_upper in ("RHF", "ROHF", "UHF") and functional is not None:
        raise ValueError(
            f"functional={functional!r} given but method={method_upper}; "
            f"use method='RKS' or 'UKS' for KS calculations"
        )
    if method_upper in ("ROKS", "RKS", "UKS") and functional is None:
        raise ValueError(f"method={method_upper!r} requires functional=...")

    # Keep the user's canonical request separate from the concrete kind that
    # the unified GuessEngine policy will execute. Route capabilities refine
    # AUTO after JK dispatch below. Every downstream route receives the enum,
    # so aliases, dotted spellings, and enum-valued inputs have one contract.
    _requested_initial_guess = coerce_initial_guess(initial_guess)
    _fragmo_request = _requested_initial_guess == InitialGuess.FRAGMO
    if _fragmo_request:
        if read_from is not None or restart_from is not None:
            raise ValueError("FRAGMO cannot be combined with a READ/restart source")
        from .guess_fragmo import _normalize_fragments, _validate_partition
        fragments = _normalize_fragments(fragments)
        _validate_partition(fragments, system.unit_cell_molecule())
        if reduce_to_primitive or symmetry is True or str(symmetry).lower() in ("auto", "reduce"):
            raise ValueError("FRAGMO atom/image ownership requires the input cell; disable primitive reduction")
        # Physical construction is recorded as FRAGMO; the existing validated
        # all-k density transport is READ throughout the concrete SCF routes.
        _requested_initial_guess = InitialGuess.READ
    elif fragments is not None:
        raise ValueError("fragments requires initial_guess='FRAGMO'")
    _resolved_initial_guess = resolve_initial_guess(
        system.unit_cell_molecule(),
        _requested_initial_guess,
        is_periodic=True,
        is_open_shell=method_upper in ("ROHF", "ROKS", "UHF", "UKS"),
    )
    _uses_external_xc = False
    _external_xc_grid_profile = None
    if method_upper in ("ROKS", "RKS", "UKS"):
        _functional_obj = Functional(
            str(functional),
            2 if method_upper in ("ROKS", "UKS") else 1,
        )
        _uses_external_xc = bool(
            getattr(_functional_obj, "is_external", False)
        )
        if _uses_external_xc:
            _external_xc_grid_profile = _negotiate_external_xc_grid_profile(
                _functional_obj
            )
    _uses_skala = str(functional).strip().lower() in {
        "skala",
        "skala-1.1",
        "skala-1.1-rev1",
    }
    _skala_dispersion_disabled = dispersion in (None, False) or (
        isinstance(dispersion, str)
        and dispersion.strip().lower() in ("", "none", "false")
    )
    if _uses_skala and not _skala_dispersion_disabled:
        raise NotImplementedError(
            "run_periodic_job: dispersion is not yet supported with periodic "
            "SKALA. The periodic output lifecycle cannot currently preserve "
            "the corrected energy and complete D3 citation provenance. Run "
            "bare experimental periodic SKALA, or use the documented "
            "molecular SKALA+D3(BJ) workflow."
        )
    if _uses_external_xc and (optimize or optimize_cell or hessian or tddft):
        _unsupported_external_xc = []
        if optimize:
            _unsupported_external_xc.append("optimize")
        if optimize_cell:
            _unsupported_external_xc.append("optimize_cell")
        if hessian:
            _unsupported_external_xc.append("hessian")
        if tddft:
            _unsupported_external_xc.append("tddft")
        raise NotImplementedError(
            "run_periodic_job: full-grid external XC functionals do not yet "
            "support nuclear/grid-response or pointwise response routes "
            "(requested: "
            + ", ".join(_unsupported_external_xc)
            + "). Run a fixed-geometry single point or use explicit "
            "whole-SCF finite differences."
        )
    if _uses_external_xc and (
        coop_cohp or dos_kmesh is not None or band_structure is not None
    ):
        _unsupported_external_properties = []
        if coop_cohp:
            _unsupported_external_properties.append("coop_cohp")
        if dos_kmesh is not None:
            _unsupported_external_properties.append("dos_kmesh")
        if band_structure is not None:
            _unsupported_external_properties.append("band_structure")
        raise NotImplementedError(
            "run_periodic_job: full-grid external XC functionals do not yet "
            "support Hamiltonian-derived periodic DOS, COOP/COHP, or band "
            "payloads (requested: "
            + ", ".join(_unsupported_external_properties)
            + "). The generic QVF reconstruction cannot include the "
            "nonlocal XC potential, so this request fails before SCF."
        )
    # Fail early for an input cell that is already odd. The same validation
    # runs after optional primitive reduction because reduction can change
    # electron-count parity.
    _validate_closed_shell_electron_count(system, method_upper)
    if atomic_spins is not None and method_upper not in ("UHF", "UKS"):
        # ATOMSPIN is an open-shell broken-symmetry seed; reject the
        # closed-shell misuse early (before the dry-run manifest / setup).
        raise ValueError(
            "atomic_spins (ATOMSPIN) is an open-shell broken-symmetry seed; "
            f"it requires method='UHF' or 'UKS', got {method_upper!r}."
        )
    if atomic_spins is not None and len(atomic_spins) != len(system.unit_cell):
        raise ValueError(
            "run_periodic_job: atomic_spins length "
            f"({len(atomic_spins)}) must match the input-cell atom count "
            f"({len(system.unit_cell)})."
        )
    if atomic_spins is not None and any(s not in (-1, 0, 1) for s in atomic_spins):
        raise ValueError("run_periodic_job: atomic_spins tags must be -1, 0 or 1")
    # SPINLOCK: resolve the string mode to a SpinlockMode (open-shell only).
    # The selected driver fails closed if it does not implement the mode.
    _spinlock_mode = SpinlockMode.OFF
    if spinlock is not None:
        if method_upper not in ("UHF", "UKS"):
            raise ValueError(
                "spinlock is an open-shell broken-symmetry convergence aid; "
                f"it requires method='UHF' or 'UKS', got {method_upper!r}."
            )
        _spinlock_map = {
            "spin_schedule": SpinlockMode.SPIN_SCHEDULE,
            "pattern_hold": SpinlockMode.PATTERN_HOLD,
            "off": SpinlockMode.OFF,
        }
        _spinlock_key = str(spinlock).strip().lower().replace("-", "_")
        if _spinlock_key not in _spinlock_map:
            raise ValueError(
                f"unknown spinlock={spinlock!r} (valid: 'spin_schedule', "
                "'pattern_hold')."
            )
        _spinlock_mode = _spinlock_map[_spinlock_key]
        if _spinlock_mode != SpinlockMode.OFF and int(spinlock_iterations) <= 0:
            raise ValueError(
                "spinlock requires spinlock_iterations > 0 (the number of "
                "locked/held SCF cycles before release)."
            )
    # READ restart validations, early (before the dry-run manifest / setup).
    _is_read_request = _resolved_initial_guess == InitialGuess.READ
    if optimize_cell and not optimize:
        raise ValueError(
            "run_periodic_job: optimize_cell=True requires optimize=True; "
            "the cell step is part of the geometry-optimization workflow."
        )
    _read_multik_request = _is_read_request and _is_multik_kpoints(kpoints)
    if read_from is not None and not _is_read_request:
        raise ValueError(
            "read_from is only used with initial_guess='read' (or 'moread' / "
            "'coread'); set initial_guess to request a READ restart."
        )
    if _read_multik_request:
        if read_from is None and not _fragmo_request:
            raise ValueError(
                "multi-k periodic initial_guess='read' needs an in-memory "
                "multi-k source result or a .qvf archive with all-k Bloch "
                "restart data."
            )
        if isinstance(read_from, (str, os.PathLike)):
            _read_suffix = Path(os.fspath(read_from)).suffix.lower()
            if _read_suffix != ".qvf":
                raise NotImplementedError(
                    "multi-k periodic READ restart from non-QVF file sources "
                    "is not implemented: a multi-k restart needs all per-k "
                    "complex Bloch coefficients and occupations."
                )

    # A PeriodicJKMethod member passed directly bypasses the string resolver
    # (and its DeprecationWarning). Outside the front door the three legacy
    # AICCM members warn through the same helper; NEUTRAL_BLOCH never had a
    # legacy spelling and is front-door-only, so it fails closed here.
    if (
        not _aiccm_front_door
        and isinstance(jk_method, PeriodicJKMethod)
        and jk_method in _AICCM_JK_METHODS
    ):
        if jk_method is PeriodicJKMethod.NEUTRAL_BLOCH:
            raise ValueError(
                "run_periodic_job: PeriodicJKMethod.NEUTRAL_BLOCH is reachable "
                "only through method='aiccm', variant='neutral-bloch' (the "
                "neutral Γ-CCM Bloch producer, ruling R1); it is not a "
                "jk_method for the RHF/ROHF/ROKS/UHF/RKS/UKS methods."
            )
        # (The deprecation is attributed to the first frame outside the
        # package; the stacklevel is the pre-3.12 fallback only.)
        _warn_legacy_aiccm_spelling(jk_method.value, jk_method, stacklevel=3)
    # Resolve the J/K method (AUTO -> concrete pick) and validate the
    # combination of method x lattice x basis. This is intentionally
    # done before any expensive setup so user errors fire early.
    resolved_jk = pick_jk_method(
        jk_method,
        lattice=np.asarray(system.lattice, dtype=float),
        basis_name=basis.name,
        n_atoms=len(system.unit_cell),
        scf_method=method_upper,
        dim=int(system.dim),
    )
    validate_jk_method(
        resolved_jk,
        lattice=np.asarray(system.lattice, dtype=float),
        basis_name=basis.name,
    )
    if restart_from is not None and resolved_jk not in (
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
    ):
        raise NotImplementedError(
            "run_periodic_job: restart_from is implemented only for GPW "
            "and GAPW .npz results. Use initial_guess='READ' with "
            "read_from= on a route that supports density restart; the "
            f"resolved {resolved_jk.value!r} route cannot consume "
            "restart_from safely."
        )
    if (
        resolved_jk == PeriodicJKMethod.GAPW
        and not isinstance(gapw_molecular_limit, (bool, np.bool_))
    ):
        raise TypeError(
            "run_periodic_job: gapw_molecular_limit must be bool; "
            f"got {type(gapw_molecular_limit).__name__}."
        )
    if (
        resolved_jk == PeriodicJKMethod.GAPW
        and method_upper in ("RKS", "UKS")
        and gapw_molecular_limit
    ):
        raise ValueError(
            "run_periodic_job: gapw_molecular_limit applies only to GAPW "
            "RHF/UHF. GAPW RKS/UKS resolves one_centre='auto' to the "
            "legacy block DFT functional."
        )
    if (
        resolved_jk == PeriodicJKMethod.GAPW
        and method_upper in ("RHF", "UHF")
        and not gapw_molecular_limit
    ):
        raise NotImplementedError(
            "run_periodic_job: GAPW RHF/UHF requires "
            "gapw_molecular_limit=True. The fit-free analytic one-centre "
            "default is validated only for a vacuum-padded single-Gamma "
            "molecule/atom, and the legacy block fallback is known wrong "
            "for H2O-class bonded cores. Use GDF/BIPOLE for compact "
            "crystals."
        )
    if (
        resolved_jk == PeriodicJKMethod.GAPW
        and method_upper in ("RHF", "UHF")
        and (optimize or optimize_cell or hessian)
    ):
        requested_derivatives = []
        if optimize:
            requested_derivatives.append("optimize")
        if optimize_cell:
            requested_derivatives.append("optimize_cell")
        if hessian:
            requested_derivatives.append("hessian")
        raise NotImplementedError(
            "run_periodic_job: GAPW RHF/UHF "
            f"{', '.join(requested_derivatives)} requires derivatives of "
            "the same fit-free analytic one-centre energy. Those derivatives "
            "are not implemented; the generic periodic optimizers use a "
            "different BIPOLE Hamiltonian and the generic Hessian uses the "
            "molecular unit-cell Hamiltonian."
        )
    if (
        dynamic_damping is not None
        and resolved_jk != PeriodicJKMethod.AICCM2026DEV_B
    ):
        raise NotImplementedError(
            "run_periodic_job: explicit dynamic_damping is currently "
            "implemented only for jk_method='aiccm2026dev-b'; other "
            "periodic dispatches do not all preserve this option"
        )
    # AICCM selector label for every guard message below: the front-door
    # form, or the legacy spelling with its front-door equivalent.
    _aiccm_label: Optional[str] = None
    if resolved_jk in _AICCM_JK_METHODS:
        _aiccm_variant_name = AICCM_VARIANT_OF[resolved_jk]
        if _aiccm_front_door:
            _aiccm_label = f"method='aiccm', variant={_aiccm_variant_name!r}"
        else:
            _legacy_spelling = (
                jk_method.value
                if isinstance(jk_method, PeriodicJKMethod)
                else str(jk_method).strip()
            )
            _aiccm_label = (
                f"jk_method={_legacy_spelling!r} (deprecated spelling of "
                f"method='aiccm', variant={_aiccm_variant_name!r})"
            )
        if resolved_jk in (
            PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
            PeriodicJKMethod.AICCM2026DEV_A,
        ) and _requested_initial_guess == InitialGuess.AUTO:
            _resolved_initial_guess = resolve_initial_guess(
                system.unit_cell_molecule(), _requested_initial_guess,
                is_periodic=True,
                is_open_shell=method_upper in ("ROHF", "ROKS", "UHF", "UKS"),
                supported=(InitialGuess.HCORE,),
            )
            if atomic_spins is not None:
                raise ValueError(
                    "run_periodic_job: atomic_spins requires SAD; "
                    "AUTO resolves to HCORE on this supercell-Gamma route"
                )
        # The real-Gamma and four-centre adapters execute supercell-Gamma
        # SCF loops whose seed is unconditionally the core Hamiltonian.  They
        # take no options object or caller-owned density, so accepting any
        # other selector here would report one guess while executing HCORE.
        # Keep this gate route-local: neutral-bloch and chi-B do receive the
        # unified options object and must retain their own capability checks.
        if (
            resolved_jk
            in (
                PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
                PeriodicJKMethod.AICCM2026DEV_A,
            )
            and _resolved_initial_guess != InitialGuess.HCORE
        ):
            _aiccm_guess_label = _resolved_initial_guess.name
            if _requested_initial_guess != _resolved_initial_guess:
                _aiccm_guess_label = (
                    f"{_requested_initial_guess.name} (resolved to "
                    f"{_resolved_initial_guess.name})"
                )
            raise NotImplementedError(
                f"run_periodic_job: {_aiccm_label} does not implement "
                f"initial_guess={_aiccm_guess_label!r}. Its "
                "supercell-Gamma SCF loop currently executes only "
                "initial_guess='HCORE'; this request fails before SCF "
                "instead of silently substituting HCORE. Omit "
                "initial_guess to get HCORE as this variant's default."
            )
        # One torus keyword for every variant (plan section 3.4): the
        # real-space aiccm_lattice_extension / aiccm_wigner_seitz_shells
        # control or the legacy Γ-centred kpoints alias, never both. The
        # Γ-CCM producers parse the torus from ``kpoints`` (the Γ-centred
        # mesh IS the BvK nrep), so hand them the extension as that mesh
        # HERE, before the k-derived sidecar flags below are computed; chi
        # keeps parsing the extension itself at dispatch.
        if (
            aiccm_lattice_extension is not None
            or aiccm_wigner_seitz_shells is not None
        ):
            if kpoints is not None:
                raise ValueError(
                    f"{_aiccm_label} accepts either the real-space "
                    "aiccm_lattice_extension/aiccm_wigner_seitz_shells "
                    "control or the legacy kpoints mesh alias, not both"
                )
            if resolved_jk != PeriodicJKMethod.AICCM2026DEV_B:
                kpoints = tuple(
                    int(n)
                    for n in cyclic_lattice_extension(
                        system,
                        aiccm_lattice_extension,
                        wigner_seitz_shells=aiccm_wigner_seitz_shells,
                    ).repetitions
                )
    _requested_bloch_kmesh = _runner_bloch_kmesh(system, kpoints)
    _requested_kmesh_size = _bloch_kmesh_full_size(_requested_bloch_kmesh)
    _requested_true_multik = kpoints is not None and _requested_kmesh_size > 1
    _requested_gamma_only = (
        kpoints is None
        or _gamma_kmesh_info(system, _requested_bloch_kmesh) is not None
    )
    if _uses_external_xc:
        _require_external_xc_zero_temperature_request(
            system,
            kpoints,
            smearing=smearing,
            smearing_temperature=smearing_temperature,
            smearing_unit=smearing_unit,
            smearing_method=smearing_method,
            smearing_metallic=smearing_metallic,
            smearing_band_gap_hartree=smearing_band_gap_hartree,
        )
    _jk_requested_label = (
        jk_method.value
        if isinstance(jk_method, PeriodicJKMethod)
        else str(jk_method)
    ).strip().lower()
    _jk_method_explicit = _jk_requested_label not in ("auto", "")

    # AUTO must never change the Coulomb Hamiltonian only after setup merely
    # to find a +U implementation. Resolve that execution plan now, before
    # BIPOLE capability guards and the dry-run manifest. In 3D, the exact
    # BIPOLE drivers support +U for RHF/RKS/UHF/UKS. A true low-dimensional
    # BIPOLE Coulomb model does not exist and therefore fails closed.
    _auto_bipole_dftu_fallback = bool(
        dft_plus_u
        and not _jk_method_explicit
        and resolved_jk not in (
            PeriodicJKMethod.BIPOLE,
            PeriodicJKMethod.GPW,
            PeriodicJKMethod.GAPW,
        )
        and method_upper in ("RHF", "RKS", "UHF", "UKS")
        and not (
            resolved_jk == PeriodicJKMethod.GDF
            and method_upper in ("RHF", "RKS")
            and _requested_true_multik
        )
        and not (
            resolved_jk == PeriodicJKMethod.RIJCOSX
            and method_upper in ("RHF", "RKS")
            and _requested_true_multik
        )
    )
    if _auto_bipole_dftu_fallback:
        if int(system.dim) != 3:
            raise NotImplementedError(
                "run_periodic_job: AUTO cannot preserve the selected "
                f"{resolved_jk.value!r} low-dimensional Coulomb Hamiltonian "
                "with DFT+U. The available BIPOLE +U implementation is "
                "3D-only, so this request fails before dry-run/SCF."
            )
        resolved_jk = PeriodicJKMethod.BIPOLE

    _external_gpw_supported = bool(
        _uses_external_xc
        and resolved_jk == PeriodicJKMethod.GPW
        and method_upper == "RKS"
    )
    _external_gapw_gamma_dispatch = bool(
        _uses_external_xc
        and resolved_jk == PeriodicJKMethod.GAPW
        and method_upper == "RKS"
        and _requested_gamma_only
        and not _requested_true_multik
    )
    if (
        _uses_external_xc
        and resolved_jk in (PeriodicJKMethod.GPW, PeriodicJKMethod.GAPW)
        and float(system.charge) != 0.0
    ):
        raise NotImplementedError(
            "run_periodic_job: charged-cell GPW/GAPW external XC remains "
            "gated. These drivers do not define a validated charged "
            "FFT-Poisson background convention and must not fill the neutral "
            "electron count. Use a neutral cell."
        )
    if _uses_external_xc and int(system.dim) == 2:
        raise NotImplementedError(
            "run_periodic_job: full-grid external XC is not yet wired to "
            "the exact difference-closed density domain of the vacuum-free "
            "2D slab routes. Use a 1D/3D supported route; 2D external XC "
            "remains gated."
        )
    if _uses_external_xc and method_upper == "ROKS":
        raise NotImplementedError(
            "run_periodic_job: periodic ROKS with a full-grid external XC "
            "provider remains gated. The maintained ROKS/BIPOLE and "
            "ROKS/GPW drivers do not yet expose the explicit periodic-"
            "lattice density domain; use periodic UKS or molecular ROKS."
        )
    if (
        _uses_external_xc
        and resolved_jk in (PeriodicJKMethod.GPW, PeriodicJKMethod.GAPW)
        and not (_external_gpw_supported or _external_gapw_gamma_dispatch)
    ):
        raise NotImplementedError(
            "Full-grid external XC through GPW/GAPW is currently accepted "
            "for GPW RKS on Gamma or an unreduced complete k mesh, and GAPW "
            "RKS at Gamma only. UKS/ROKS, GAPW multi-k, ECP Hamiltonians, "
            "and symmetry-reduced representative meshes remain gated."
        )
    if (
        _uses_external_xc
        and resolved_jk == PeriodicJKMethod.RIJCOSX
        and not _requested_true_multik
    ):
        raise NotImplementedError(
            "Full-grid external XC functionals are not available through "
            "the Gamma/single-k RIJCOSX route. That route currently uses "
            "the molecular XC grid and kernel, which cannot supply the "
            "periodic dimension and lattice metadata required by an "
            "external provider. Use a complete multi-k RIJCOSX mesh, "
            "jk_method='gdf', 'bipole', or the Ewald AO-grid route."
        )
    if (
        _uses_external_xc
        and resolved_jk == PeriodicJKMethod.NEUTRAL_BLOCH
    ):
        raise NotImplementedError(
            "run_periodic_job: full-grid external XC is not available through "
            "method='aiccm', variant='neutral-bloch'. The unit-cell atom-block "
            "provider partition has not been proven representation-invariant "
            "against the real-Gamma supercell grouping. Use "
            "variant='real-gamma' or jk_method='gdf' for an external "
            "functional."
        )
    if (
        _uses_external_xc
        and resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
    ):
        _require_aiccm2026dev_b_external_xc(
            _functional_obj,
            _resolve_aiccm2026dev_b_backend(aiccm_backend),
            system=system,
            where="run_periodic_job(jk_method='aiccm2026dev-b')",
        )
    if (
        _uses_external_xc
        and resolved_jk in (PeriodicJKMethod.GDF, PeriodicJKMethod.RIJCOSX)
        and bool(symmetry_reduce_k)
    ):
        raise NotImplementedError(
            "run_periodic_job: full-grid external XC requires the complete "
            "unreduced Monkhorst-Pack mesh on GDF/RIJCOSX routes; "
            "symmetry_reduce_k=True cannot preserve the inverse Bloch "
            "density fold. Pass symmetry_reduce_k=False."
        )

    # These checks must follow AUTO +U route planning. Otherwise a request
    # that resolves to BIPOLE only through the fallback can bypass the
    # finalized route's k-mesh and restart capabilities during dry-run.
    if _uses_external_xc and resolved_jk == PeriodicJKMethod.BIPOLE:
        from .pbc_bipole_common import validate_bipole_kmesh

        validate_bipole_kmesh(
            _requested_bloch_kmesh,
            driver="run_periodic_job external XC/BIPOLE",
            require_complete=True,
        )
        if use_exchange_ewald_split is False:
            raise NotImplementedError(
                "run_periodic_job: full-grid external XC through BIPOLE "
                "requires the corrected Ewald exchange gauge. Leave "
                "use_exchange_ewald_split enabled and use a complete "
                "Monkhorst-Pack mesh or its valid expandable IBZ reduction."
            )
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and _requested_kmesh_size == 1
        and not _requested_gamma_only
    ):
        raise NotImplementedError(
            "run_periodic_job: BIPOLE does not support a lone non-Gamma "
            "twist. Its one-point path is Gamma-specific; use the Gamma "
            "point or a complete Monkhorst-Pack mesh."
        )
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and use_exchange_ewald_split is True
        and _requested_true_multik
    ):
        _split_ir = np.asarray(
            getattr(_requested_bloch_kmesh, "ir_mapping", []), dtype=int
        ).reshape(-1)
        _split_mesh = tuple(
            int(x)
            for x in getattr(_requested_bloch_kmesh, "mesh", (1, 1, 1))
        )
        _split_stored = _bloch_kmesh_size(_requested_bloch_kmesh)
        if _split_ir.size == 0 and int(np.prod(_split_mesh)) != _split_stored:
            raise NotImplementedError(
                "run_periodic_job: use_exchange_ewald_split=True at multi-k "
                "requires a complete Monkhorst-Pack mesh carrying its BvK "
                "dimensions. An ad-hoc explicit k-point list cannot execute "
                "the corrected exchange gauge."
            )
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and method_upper in ("UHF", "UKS")
        and _spinlock_mode == SpinlockMode.SPIN_SCHEDULE
        and _requested_true_multik
    ):
        raise NotImplementedError(
            "run_periodic_job: BIPOLE SPIN_SCHEDULE is Gamma-only because "
            "its phase-2 restart cannot reconstruct the complete multi-k "
            "spin density. Use spinlock='pattern_hold' or a Gamma mesh."
        )

    if resolved_jk == PeriodicJKMethod.BIPOLE and solver != "dense":
        raise NotImplementedError(
            "run_periodic_job: jk_method='bipole' currently supports only "
            "solver='dense'. The direct BIPOLE SCF loops do not execute "
            f"solver={solver!r}, so the request fails before dry-run/SCF."
        )

    def _explicit_unit_kmesh(value: object) -> bool:
        if isinstance(value, (int, np.integer)):
            return int(value) == 1
        if isinstance(value, (list, tuple)):
            return tuple(int(x) for x in value) == (1, 1, 1)
        return False

    # Match the actual GPW/GAPW branch predicates. Ordinary explicit RKS
    # (1,1,1) keeps the historical multi-k implementation, while an external
    # provider's GAPW Gamma object uses the dedicated full-grid Gamma adapter.
    _gpw_multik_dispatch = bool(
        resolved_jk == PeriodicJKMethod.GPW
        and kpoints is not None
        and not (
            method_upper != "RKS" and _explicit_unit_kmesh(kpoints)
        )
    )
    _gapw_multik_dispatch = bool(
        resolved_jk == PeriodicJKMethod.GAPW
        and kpoints is not None
        and not (
            _external_gapw_gamma_dispatch
            or (method_upper != "RKS" and _explicit_unit_kmesh(kpoints))
        )
    )
    if restart_from is not None and (
        _gpw_multik_dispatch
        or _gapw_multik_dispatch
        or method_upper not in ("RHF", "RKS")
    ):
        raise NotImplementedError(
            "run_periodic_job: restart_from currently supplies one Gamma "
            "density and is implemented for GPW/GAPW RHF/RKS Gamma routes "
            "only. Multi-k and open-shell routes "
            "cannot consume that density safely; use their supported "
            "initial_guess='READ' plus read_from= interface where available."
        )
    if dft_plus_u and _gpw_multik_dispatch and method_upper != "RKS":
        raise NotImplementedError(
            "run_periodic_job: multi-k GPW DFT+U is currently implemented "
            "only for RKS; this request fails before dry-run/SCF."
        )
    if dft_plus_u and _gapw_multik_dispatch and method_upper != "RKS":
        raise NotImplementedError(
            "run_periodic_job: multi-k GAPW DFT+U is currently implemented "
            "only for RKS; this request fails before dry-run/SCF."
        )
    if (
        dft_plus_u
        and method_upper == "RKS"
        and (_gpw_multik_dispatch or _gapw_multik_dispatch)
        and _periodic_functional_needs_exchange(functional, open_shell=False)
    ):
        _backend = "GPW" if _gpw_multik_dispatch else "GAPW"
        raise NotImplementedError(
            f"run_periodic_job: multi-k {_backend} RKS DFT+U does not "
            "implement hybrid or range-separated exact exchange. Use a "
            "pure functional such as LDA/PBE, or select a backend with a "
            "per-k exchange builder; this request fails before dry-run/SCF."
        )
    if method_upper in ("ROHF", "ROKS"):
        _restricted_open_backends = (
            PeriodicJKMethod.GPW,
            PeriodicJKMethod.BIPOLE,
        )
        if method_upper == "ROHF":
            _restricted_open_backends += (PeriodicJKMethod.GDF,)
        if resolved_jk not in _restricted_open_backends:
            _available = "jk_method='gpw' and jk_method='bipole'"
            _gated = "GDF, GAPW, and the other"
            if method_upper == "ROHF":
                _available += ", plus jk_method='gdf'"
                _gated = "GAPW and the other"
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper} is currently "
                f"wired for {_available} "
                "(Gamma and full Monkhorst-Pack meshes) on the "
                "maintained-preview 3D routes"
                f"; got jk_method={resolved_jk.value!r}. {_gated} "
                "restricted-open-shell backends remain gated."
            )
        if int(system.dim) != 3:
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper}/"
                f"{resolved_jk.value} currently requires a "
                f"3D cell; got dim={system.dim}. The 1D/2D Coulomb gauges "
                "remain gated."
            )
        if resolved_jk == PeriodicJKMethod.BIPOLE:
            _kpoints_kind = getattr(kpoints, "kind", None)
            if _kpoints_kind is not None and str(_kpoints_kind) != "monkhorst-pack":
                raise NotImplementedError(
                    f"run_periodic_job: periodic {method_upper}/bipole "
                    "requires a complete Monkhorst-Pack mesh; explicit, "
                    "band-path, generalized-regular, and database k-point "
                    "lists do not carry the BvK torus needed by this route."
                )
            _ro_ir_mapping = np.asarray(
                getattr(_requested_bloch_kmesh, "ir_mapping", []), dtype=int
            ).reshape(-1)
            _ro_stored_nk = _bloch_kmesh_size(_requested_bloch_kmesh)
            _ro_mesh = tuple(
                int(x)
                for x in getattr(_requested_bloch_kmesh, "mesh", (1, 1, 1))
            )
            if _ro_ir_mapping.size == 0 and int(np.prod(_ro_mesh)) != _ro_stored_nk:
                raise NotImplementedError(
                    f"run_periodic_job: periodic {method_upper}/bipole "
                    "requires a complete Monkhorst-Pack mesh carrying its "
                    f"dimensions; got mesh={_ro_mesh} for {_ro_stored_nk} "
                    "stored k-points."
                )
            if _requested_kmesh_size == 1 and not _kmesh_contains_gamma(
                _requested_bloch_kmesh
            ):
                raise NotImplementedError(
                    f"run_periodic_job: periodic {method_upper}/bipole does "
                    "not support a lone non-Gamma twist. Use the Gamma point "
                    "or a complete Monkhorst-Pack mesh."
                )
        if (
            method_upper == "ROHF"
            and resolved_jk == PeriodicJKMethod.GPW
            and (not _requested_gamma_only or _requested_kmesh_size != 1)
        ):
            raise NotImplementedError(
                "run_periodic_job: periodic ROHF/GPW is Gamma-only "
                "with one k point. Multi-k restricted-open-shell GPW needs "
                "per-k exact exchange and remains gated; multi-k ROHF is "
                "available via jk_method='bipole'."
            )
        if (
            method_upper == "ROKS"
            and _requested_kmesh_size == 1
            and not _requested_gamma_only
        ):
            raise NotImplementedError(
                f"run_periodic_job: periodic ROKS/{resolved_jk.value} "
                "supports either the Gamma route or a true multi-k mesh. A "
                "single non-Gamma k point is not a Monkhorst-Pack mesh and "
                "has no maintained restricted-open-shell route."
            )
        if dft_plus_u:
            raise NotImplementedError(
                "run_periodic_job: DFT+U is not implemented for periodic "
                f"{method_upper}/{resolved_jk.value}."
            )
        if optimize or optimize_cell or hessian:
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper}/{resolved_jk.value} "
                "gradients, geometry "
                "optimization, cell optimization, and Hessians are not "
                "implemented."
            )
        if tddft:
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper}/{resolved_jk.value} "
                "TD-DFT is not "
                "implemented."
            )
        if coop_cohp:
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper}/{resolved_jk.value} "
                "COOP/COHP analysis is "
                "not validated."
            )
        if restart_from is not None:
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper}/{resolved_jk.value} "
                "restart_from archives are not implemented; the route "
                "requires paired alpha/beta densities in one restricted-"
                "orbital gauge. Use initial_guess='READ' with read_from= "
                "on the Gamma GPW route."
            )
        if solver != "dense":
            raise NotImplementedError(
                f"run_periodic_job: periodic {method_upper}/{resolved_jk.value} "
                "currently "
                "supports only "
                "solver='dense'."
            )
    # Molden orbitals: the writable object is the Gamma block, which a
    # Gamma-centred mesh of any subdivision carries. Both routes below hand
    # back the k-point metadata `_gamma_index_for_multi_k` needs to locate
    # that block instead of assuming the first k-point is Gamma.
    #
    # GDF was briefly excluded here. Its Gamma block left a degenerate
    # frontier orbital complex after global-phase removal, which the writer
    # refused -- and because `plan.py` declares `.molden` a *guaranteed*
    # artefact, that refusal aborted finalization rather than dropping a
    # sidecar. The cause was not the route: the residual tracks degeneracy
    # exactly on both routes (non-degenerate orbitals sit at 1e-16 while
    # the degenerate Ne 2p pair sits at 1.1e-05 on BIPOLE and 3.9e-03 on
    # GDF), because one global phase per column cannot undo a rotation
    # *between* columns. The writer now re-expresses degenerate blocks on a
    # real basis of their own span, which is exact, so both routes export
    # S-orthonormal real orbitals to machine precision. The optional-row ask
    # in HANDOVER_OUTPUT_LOGGER.md still stands: it makes a genuine refusal
    # drop the sidecar instead of failing the job.
    _multik_gamma_orbital_routes = (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.GDF,
        # The neutral Γ-CCM Bloch producer returns the production multi-k GDF
        # result (kpoints_cart present), so its Gamma block is locatable.
        PeriodicJKMethod.NEUTRAL_BLOCH,
    )
    _molden_sidecar_supported = _requested_gamma_only or (
        resolved_jk in _multik_gamma_orbital_routes
        and _kmesh_contains_gamma(_requested_bloch_kmesh)
    )
    _molden_sidecar_reason = (
        "periodic Molden export writes the Gamma-block orbitals, so the "
        "k-mesh must contain an exact Gamma point and the route must carry "
        "k-point metadata locating it. A shifted mesh has no Gamma block, "
        "and orbitals at k != 0 are complex -- Molden's [MO] block holds "
        "only real coefficients."
    )
    write_molden_file = _resolve_sidecar_request(
        write_molden_file,
        supported=_molden_sidecar_supported,
        option="write_molden_file",
        method=method_upper.lower(),
        caller="run_periodic_job",
        unavailable_reason=_molden_sidecar_reason,
    )
    # Population: BIPOLE analyses the real-space lattice density and the
    # full SCF k-mesh (compute_bipole_population_summary), while χ-CCM uses
    # its dedicated finite-torus density and overlap contractions
    # (compute_aiccm2026dev_b_population_summary). Both are crystal
    # populations and need no Gamma restriction. GDF uses accepted Gamma
    # density/SCF-overlap matrices. Other routes use molecular analysis of
    # a single Bloch block. These adapters remain restricted to Gamma.
    _population_sidecar_supported = _requested_gamma_only or resolved_jk in (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.AICCM2026DEV_B,
        # M5: the two supercell-Γ arms analyse their own cyclic cluster
        # (compute_ccm_population_summary) and fold to per-cell charges, so
        # like BIPOLE and χ they are crystal populations needing no Γ-only
        # restriction. Their nrep is a supercell count, not a Bloch mesh.
        PeriodicJKMethod.AICCM2026DEV_A,
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
    )
    _population_sidecar_reason = (
        "the selected multi-k route has only a molecular Gamma-block "
        "population proxy. Full-k population output is implemented for "
        "BIPOLE through its lattice-density analysis, for aiccm2026dev-b "
        "through its finite-torus analysis, and for the four-center and "
        "real-gamma variants through their cyclic-cluster analysis; other "
        "routes require an exact single-Gamma result."
    )
    write_population_file = _resolve_sidecar_request(
        write_population_file,
        supported=_population_sidecar_supported,
        option="write_population_file",
        method=method_upper.lower(),
        caller="run_periodic_job",
        unavailable_reason=_population_sidecar_reason,
    )
    # Density: the artifact path evaluates a real-space lattice density, or
    # the exact Gamma block of a Gamma-only mesh.  The torus and supercell
    # producers return a LatticeMatrixSet on any mesh. Bulk GDF can also
    # return its certified inverse-Bloch representation on a full MP mesh.
    # Other Gaussian routes return per-k Bloch blocks, and the artifact path
    # refuses to invent a lattice density from those. Refuse in the first
    # second rather than after the SCF, the way write_molden_file does: a
    # completed calculation must not be discarded at finalization (#679).
    _lattice_density_routes = (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.AICCM2026DEV_A,
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
        PeriodicJKMethod.AICCM2026DEV_B,
        PeriodicJKMethod.NEUTRAL_BLOCH,
    )
    _density_sidecar_supported = (
        not _requested_true_multik or resolved_jk in _lattice_density_routes
        or (resolved_jk == PeriodicJKMethod.GDF and int(system.dim) == 3)
    )
    _density_sidecar_reason = (
        "the selected route returns per-k Bloch density blocks on a true "
        "multi-k mesh, and the density artifact refuses to reconstruct a "
        "real-space lattice density from them. Use a Gamma-only mesh, or a "
        "route that returns a lattice density (bipole, or method='aiccm')."
    )
    write_density = _resolve_sidecar_request(
        write_density,
        supported=_density_sidecar_supported,
        option="write_density",
        method=method_upper.lower(),
        caller="run_periodic_job",
        unavailable_reason=_density_sidecar_reason,
    )
    if _read_multik_request and resolved_jk not in (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.GDF,
        PeriodicJKMethod.RIJCOSX,
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
    ):
        raise NotImplementedError(
            "multi-k periodic READ from in-memory results is wired for "
            "BIPOLE, GDF, RIJCOSX, GPW, and GAPW routes; the "
            f"selected jk_method={resolved_jk.value!r} has no per-k density "
            "restart hook yet."
        )
    if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
        from .periodic.exchange_convention import BVK_EWALD, exchange_q0_label

        _requested_b_exchange_q0 = exchange_q0_label(exchange_exxdiv)
        if _requested_b_exchange_q0 != BVK_EWALD:
            raise ValueError(
                "jk_method='aiccm2026dev-b' fixes the finite-torus "
                "exchange_q0 convention to 'bvk-ewald'; "
                f"exchange_exxdiv={exchange_exxdiv!r} requests "
                f"{_requested_b_exchange_q0!r}, which the χ-CCM-B drivers "
                "do not implement. Omit exchange_exxdiv or pass 'ewald'."
            )
        if method_upper not in ("RHF", "RKS", "UHF", "UKS"):
            raise NotImplementedError(
                "jk_method='aiccm2026dev-b' implements RHF, RKS, UHF, and "
                f"UKS; got method={method_upper!r}"
            )
        if dft_plus_u:
            raise NotImplementedError(
                "jk_method='aiccm2026dev-b' does not yet implement DFT+U"
            )
        if optimize:
            raise NotImplementedError(
                "jk_method='aiccm2026dev-b' does not yet implement analytic "
                "χ-CCM-B nuclear gradients, so periodic geometry optimization "
                "is disabled. Use aiccm2026dev_b_gradient_status(result) to "
                "inspect the declared finite-torus convention and open "
                "gradient terms."
            )
        if hessian:
            raise NotImplementedError(
                "jk_method='aiccm2026dev-b' does not yet implement χ-CCM-B "
                "force constants. The generic periodic Hessian path would "
                "differentiate a non-B unit-cell model, so it is disabled."
            )
        if coop_cohp:
            raise NotImplementedError(
                "run_periodic_job: aiccm2026dev-b COOP/COHP analysis is not "
                "implemented with the converged finite-character "
                "Hamiltonian. The generic QVF property path rebuilds a "
                "fixed-cutoff Ewald/HF-like surrogate, so this request "
                "fails before SCF."
            )
    if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA:
        # EXPERIMENTAL: the neutral Γ-CCM torus in its real-Γ supercell
        # representation via the per-unit-cell adapter
        # (periodic.ccm.real_gamma_runner). Ruling R1: a producer of the
        # neutral construction, NOT the union-and-weight four-centre lineage.
        if method_upper not in ("RHF", "RKS", "UHF", "UKS"):
            raise NotImplementedError(
                f"{_aiccm_label} implements RHF, RKS, UHF, and UKS; "
                f"got method={method_upper!r}. Double hybrids run via the "
                "API driver "
                "vibeqc.periodic.ccm.direct.run_ccm_double_hybrid_direct."
            )
        # The direct-torus loop forwards max_iter and conv_tol_energy only
        # (real_gamma_runner); it implements no damping, Fock mixing, density
        # mixing or level shift (structural zeros, recorded per D86). An
        # explicit request for one of those would be dropped silently, which
        # CLAUDE.md section 7 forbids, so it fails closed before SCF.
        _rg_unhonoured = {
            name: value
            for name, value in (
                ("damping", damping),
                ("fock_mixing", fock_mixing),
                ("fmixing_percent", fmixing_percent),
                ("density_mixer", density_mixer),
                ("level_shift", level_shift),
            )
            if value is not None
        }
        if _rg_unhonoured:
            raise NotImplementedError(
                f"{_aiccm_label}: the real-Γ direct-torus SCF implements no "
                "damping, Fock mixing, density mixing or level shift (its "
                "executed values are structural zeros, recorded per D86), so "
                "the explicit "
                + ", ".join(f"{k}={v!r}" for k, v in _rg_unhonoured.items())
                + " would be silently dropped. This request fails before "
                "SCF: drop the keyword, or use variant='chi' / "
                "jk_method='gdf', which honour it."
            )
        if int(system.dim) != 3:
            raise NotImplementedError(
                f"{_aiccm_label} requires a 3-D periodic system "
                f"(dim=3); got dim={int(system.dim)}. The direct-torus "
                "route fails closed for dim<3 by design (no gauge-"
                "consistent neutral-torus Hamiltonian; see "
                "vibeqc.periodic.ccm.direct)."
            )
        if int(system.charge) != 0:
            raise NotImplementedError(
                "jk_method='real-gamma' implements a neutral fitted-torus "
                "Hamiltonian and does not support charged cells"
            )
        if dft_plus_u:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement DFT+U"
            )
        if optimize and optimize_cell:
            raise NotImplementedError(
                f"{_aiccm_label}: variable-cell relaxation is not "
                "implemented -- the route has no analytic stress, and its "
                "relaxation is fixed-lattice (unit-cell positions only). "
                "Use optimize=True with optimize_cell=False."
            )
        # Smearing needs no extra guard here: _validate_smearing_dispatch
        # already refuses this route generically (zero-temperature Aufbau
        # supercell-Γ SCF), so a smeared optimize never reaches dispatch.
        if hessian:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement force constants. "
                "The generic periodic Hessian path would differentiate a "
                "different unit-cell model, so it is disabled."
            )
        # Smearing rejects through _validate_smearing_dispatch's generic
        # not-wired branch (zero-temperature Aufbau supercell-Γ SCF only).
    if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A:
        # EXPERIMENTAL: the union-and-weight/Wigner--Seitz four-centre Γ-CCM
        # construction (the 2014 lineage) through the per-unit-cell adapter
        # periodic.ccm.four_center_runner. Ruling R1: a DIFFERENT construction
        # from the two neutral producers, not a representation of theirs.
        from .periodic.ccm.four_center_runner import FOUR_CENTRE_METHODS

        if method_upper not in FOUR_CENTRE_METHODS:
            raise NotImplementedError(
                f"{_aiccm_label} implements "
                f"{', '.join(FOUR_CENTRE_METHODS)}; got "
                f"method={method_upper!r}. Post-HF on this line is "
                "library-only (vibeqc.periodic.ccm.mp2 / .ccsd)."
            )
        if int(system.dim) != 3:
            raise NotImplementedError(
                f"{_aiccm_label} requires a 3-D periodic system (dim=3); got "
                f"dim={int(system.dim)}. The Wigner-Seitz weights are derived "
                "from a 3-D supercell topology."
            )
        if dft_plus_u:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement DFT+U"
            )
        if optimize or optimize_cell:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement geometry optimization: "
                "the four-centre analytic gradients are dense small-cluster "
                "only and are not wired to the runner (D-7). Use "
                "variant='real-gamma' for fixed-lattice relaxation."
            )
        if hessian:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement force constants."
            )
        if method_upper in ("RKS", "UKS"):
            from .periodic_screened_exchange import (
                reject_unscreened_range_separated,
            )

            _fc_functional = Functional(
                str(functional or "pbe"),
                2 if method_upper == "UKS" else 1,
            )
            reject_unscreened_range_separated(
                _fc_functional,
                where=f"run_periodic_job({_aiccm_label})",
            )
            if (
                bool(getattr(_fc_functional, "is_external", False))
                and abs(
                    float(
                        getattr(_fc_functional, "hf_exchange_fraction", 0.0)
                    )
                )
                > 1.0e-15
            ):
                raise NotImplementedError(
                    f"run_periodic_job({_aiccm_label}): the literal "
                    "four-centre WSSC external-XC route currently supports "
                    "pure full-grid functionals only. External hybrids need "
                    "a validated exact-exchange convention on the WSSC "
                    "operator."
                )
        # The four-centre SCF loops forward no mixing control, exactly like
        # the direct-torus loops: refuse rather than drop (CLAUDE.md § 7).
        _fc_unhonoured = {
            name: value
            for name, value in (
                ("damping", damping),
                ("fock_mixing", fock_mixing),
                ("fmixing_percent", fmixing_percent),
                ("density_mixer", density_mixer),
                ("level_shift", level_shift),
            )
            if value is not None
        }
        if _fc_unhonoured:
            raise NotImplementedError(
                f"{_aiccm_label}: the four-centre SCF loops implement no "
                "damping, Fock mixing, density mixing or level shift (their "
                "executed values are structural zeros, recorded per D86), so "
                "the explicit "
                + ", ".join(f"{k}={v!r}" for k, v in _fc_unhonoured.items())
                + " would be silently dropped. This request fails before SCF."
            )
    if resolved_jk == PeriodicJKMethod.NEUTRAL_BLOCH:
        # EXPERIMENTAL: the neutral Γ-CCM torus in its Bloch representation,
        # through the per-unit-cell adapter periodic.ccm.neutral_bloch_runner
        # (ruling R1: the Fourier partner of variant='real-gamma', and a
        # different construction from the union-and-weight four-centre line).
        from .periodic.ccm.neutral_bloch_runner import NEUTRAL_BLOCH_METHODS
        from .periodic.exchange_convention import BVK_EWALD, exchange_q0_label

        if method_upper not in NEUTRAL_BLOCH_METHODS:
            raise NotImplementedError(
                f"{_aiccm_label} implements "
                f"{', '.join(NEUTRAL_BLOCH_METHODS)}; got "
                f"method={method_upper!r}. The Bloch producer has no "
                "restricted-open-shell entry in vibeqc.periodic.ccm.ri."
            )
        if int(system.dim) != 3:
            raise NotImplementedError(
                f"{_aiccm_label} requires a 3-D periodic system (dim=3); got "
                f"dim={int(system.dim)}. The neutral torus has no consistent "
                "Coulomb gauge below three dimensions "
                "(vibeqc.periodic.ccm.ri)."
            )
        if dft_plus_u:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement DFT+U"
            )
        if optimize or optimize_cell:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement geometry optimization: "
                "the multi-k GDF analytic gradient is not captured on this "
                "arm. Use variant='real-gamma' (fixed-lattice relaxation) or "
                "jk_method='gdf'."
            )
        if hessian:
            raise NotImplementedError(
                f"{_aiccm_label} does not implement force constants. The "
                "generic periodic Hessian path would differentiate a "
                "different unit-cell model, so it is disabled."
            )
        if exchange_q0_label(exchange_exxdiv) != BVK_EWALD:
            raise ValueError(
                f"{_aiccm_label} fixes the exchange-q0 convention to "
                "'BvK-ewald': the multi-k GDF producer applies "
                "exxdiv='ewald' unconditionally. "
                f"exchange_exxdiv={exchange_exxdiv!r} requests something "
                "else; omit it or pass 'ewald'."
            )
        if _is_read_request or read_from is not None or restart_from is not None:
            raise NotImplementedError(
                f"{_aiccm_label} has no per-k density restart hook: the "
                "producer rebuilds the torus fits from the geometry. Run the "
                "single point without a restart, or use jk_method='gdf'."
            )
        # The producer forwards the SCF options object and the fit controls,
        # nothing else. A convergence aid it would drop fails closed rather
        # than being silently ignored (CLAUDE.md section 7); the same rule
        # M1 applied to the real-Γ producer.
        _nb_unhonoured = {
            name: value
            for name, value in (
                ("damping", damping),
                ("fock_mixing", fock_mixing),
                ("fmixing_percent", fmixing_percent),
                ("density_mixer", density_mixer),
                ("level_shift", level_shift),
            )
            if value is not None
        }
        if _nb_unhonoured:
            raise NotImplementedError(
                f"{_aiccm_label}: the neutral Bloch producer executes the "
                "torus fits with the SCF defaults and forwards no mixing or "
                "level-shift control, so the explicit "
                + ", ".join(f"{k}={v!r}" for k, v in _nb_unhonoured.items())
                + " would be silently dropped. This request fails before "
                "SCF: drop the keyword, or run the same Gamma-centred mesh on "
                "jk_method='gdf', which honours it wherever its own guards "
                "allow (a level shift, for instance, is closed-shell only "
                "there)."
            )
    if resolved_jk == PeriodicJKMethod.BIPOLE and int(system.dim) < 3:
        # Fail fast at the API level: the supported exact Ewald-J split relies
        # on a 3-D Ewald/Madelung lattice sum that is undefined for a true
        # 1-D/2-D Coulomb problem.
        raise NotImplementedError(
            f"jk_method='bipole' requires a 3-D periodic system (dim=3); got "
            f"dim={int(system.dim)}. The exact Ewald-J route uses a 3-D "
            "Ewald/Madelung lattice sum that is not a low-dimensional Coulomb "
            "model. For a 1-D wire use jk_method='auto' or 'gdf'; for a 2-D "
            "surface use jk_method='auto'/'slab_ewald_2d', or explicit GDF "
            "for its supported closed-shell slab envelope."
        )
    if resolved_jk == PeriodicJKMethod.BIPOLE:
        if optimize_cell:
            raise NotImplementedError(
                "run_periodic_job: BIPOLE variable-cell optimization is "
                "unavailable. The historical strain convention and coupled "
                "atom/cell convergence were not certified on one terminal "
                "geometry. Use optimize=True with optimize_cell=False for "
                "fixed-cell atomic relaxation."
            )
        unsupported_post_scf = []
        if hessian:
            unsupported_post_scf.append("hessian")
        if tddft:
            unsupported_post_scf.append("tddft")
        if coop_cohp:
            unsupported_post_scf.append("coop_cohp")
        if unsupported_post_scf:
            raise NotImplementedError(
                "run_periodic_job: BIPOLE "
                + ", ".join(unsupported_post_scf)
                + " is not implemented with the converged periodic "
                "Hamiltonian. The previous generic post-SCF paths rebuilt "
                "a molecular or fixed-cutoff surrogate operator, so these "
                "requests now fail before SCF."
            )
        if (optimize or optimize_cell) and _is_read_request:
            raise NotImplementedError(
                "run_periodic_job: BIPOLE optimization cannot preserve a "
                "READ restart across displaced geometries. Run the restart "
                "single point first, then begin optimization from a "
                "geometry-defined SAD/HCORE/ATOMSPIN guess."
            )
        _dispersion_disabled = dispersion in (None, False) or (
            isinstance(dispersion, str)
            and dispersion.strip().lower() in ("", "none", "false")
        )
        if (optimize or optimize_cell) and not _dispersion_disabled:
            raise NotImplementedError(
                "run_periodic_job: BIPOLE optimization with periodic "
                "dispersion is not implemented. The optimizer does not yet "
                "differentiate or add the selected dispersion correction at "
                "each displaced geometry, so the combination fails before "
                "the initial SCF instead of optimizing a different surface."
            )
        _attached_symmetry_requested = bool(
            symmetry not in (False, None, "off", "false", "none")
            or reduce_to_primitive
            or getattr(system, "symmetry", None) is not None
        )
        if (optimize or optimize_cell) and (
            symmetry_stabilize
            or symmetry_reduce_fock is True
            or (
                symmetry_reduce_fock is None
                and _attached_symmetry_requested
            )
        ):
            raise NotImplementedError(
                "run_periodic_job: BIPOLE optimization with Fock symmetry "
                "stabilization/reduction is not implemented. Atomic or cell "
                "displacements can break the input symmetry, and the "
                "optimizer has no symmetry-constrained coordinate space."
            )
    if (optimize or optimize_cell) and resolved_jk not in (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.GDF,
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
    ):
        raise NotImplementedError(
            "run_periodic_job: geometry optimization is not wired to the "
            f"executed {resolved_jk.value!r} Hamiltonian. Only BIPOLE, the "
            "captured analytic-gradient GDF route, and the real-Γ "
            "direct-torus route (EXPERIMENTAL, parity-verified per run) "
            "preserve their single-point objective at every displaced "
            "geometry; this request fails before SCF instead of falling "
            "through to BIPOLE forces."
        )
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and (symmetry_stabilize or symmetry_reduce_fock is True)
        and symmetry in (False, None, "off", "false", "none")
        and getattr(system, "symmetry", None) is None
    ):
        raise ValueError(
            "run_periodic_job: explicit BIPOLE Fock symmetry requires an "
            "attached symmetry model. Pass symmetry='attach' (or attach "
            "validated operations to the PeriodicSystem) before requesting "
            "symmetry_stabilize/symmetry_reduce_fock."
        )
    # --- Space-group-reduced multi-k GDF exchange (symmetry_reduce_k) ---
    # Every precondition fails closed. A silently-ignored performance flag
    # is worse than a refusal: the run looks reduced, costs the full n_k^2,
    # and the memory line the flag exists to demonstrate would be a lie.
    if symmetry_reduce_k:
        if resolved_jk != PeriodicJKMethod.GDF:
            raise NotImplementedError(
                "run_periodic_job: symmetry_reduce_k is wired to the "
                "multi-k GDF exchange build (jk_method='gdf'); got "
                f"{resolved_jk.value!r}. The reduction acts on the "
                "per-(k_i,k_j) cderi cache, which only the fitted-GDF "
                "exchange builds -- RIJCOSX/COSX assembles K in real space "
                "and never touches it, and the plane-wave routes have no "
                "such cache at all."
            )
        if int(system.dim) != 3:
            raise NotImplementedError(
                "run_periodic_job: symmetry_reduce_k is a bulk (dim=3) "
                f"route; got dim={int(system.dim)}. The wedge comes from "
                "the 3-D space group of the crystal, and the slab / "
                "polymer GDF paths run their own truncated-metric gauge."
            )
        if not _requested_true_multik:
            raise NotImplementedError(
                "run_periodic_job: symmetry_reduce_k needs a true multi-k "
                "Monkhorst-Pack mesh to reduce; this job samples "
                f"{_requested_kmesh_size} k-point(s). At Gamma the wedge is "
                "the whole mesh and there is nothing to save."
            )
        if method_upper not in ("RHF", "RKS", "UHF", "UKS"):
            raise NotImplementedError(
                "run_periodic_job: symmetry_reduce_k is implemented on the "
                "closed-shell (RHF/RKS) and unrestricted (UHF/UKS) multi-k "
                f"GDF drivers; got method={method_upper!r}. The "
                "spin-restricted open-shell GDF driver has no ibz_native "
                "argument surface (see periodic_rohf_gdf.py)."
            )
        if symmetry in (False, None, "off", "false", "none") and (
            getattr(system, "symmetry", None) is None
        ):
            raise ValueError(
                "run_periodic_job: symmetry_reduce_k requires an attached "
                "symmetry model -- there is no irreducible wedge without "
                "one. Pass symmetry='attach' (or attach validated "
                "operations to the PeriodicSystem) alongside "
                "symmetry_reduce_k=True."
            )
        if optimize or optimize_cell:
            raise NotImplementedError(
                "run_periodic_job: symmetry_reduce_k cannot be combined "
                "with geometry optimization. The wedge-native SCF converges "
                "the k-transported exchange, whose analytic gradient is not "
                "the captured single-point objective (periodic_k_gdf.py "
                "rejects compute_gradient=True with ibz_native=True), and "
                "displacements can break the input symmetry outright."
            )
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and method_upper in ("UHF", "UKS")
        and (symmetry_stabilize or symmetry_reduce_fock is True)
    ):
        _broken_symmetry_controls = []
        if atomic_spins is not None:
            _broken_symmetry_controls.append("atomic_spins")
        if use_mom:
            _broken_symmetry_controls.append("use_mom")
        if _spinlock_mode != SpinlockMode.OFF:
            _broken_symmetry_controls.append("spinlock")
        if dft_plus_u:
            _broken_symmetry_controls.append("dft_plus_u")
        if _is_read_request:
            _broken_symmetry_controls.append("READ")
        if _resolved_initial_guess == InitialGuess.PATOM:
            _broken_symmetry_controls.append("PATOM")
        if _broken_symmetry_controls:
            raise NotImplementedError(
                "run_periodic_job: explicit BIPOLE Fock symmetry cannot "
                "be combined with an electronically broken or unverified "
                "state control ("
                + ", ".join(_broken_symmetry_controls)
                + "). Structural symmetry alone does not prove spin-density "
                "covariance; disable Fock symmetry for this state."
            )
    _bipole_cutoff_bohr = (
        None if bipole_cutoff_bohr is None else float(bipole_cutoff_bohr)
    )
    _bipole_nuclear_cutoff_bohr = (
        None
        if bipole_nuclear_cutoff_bohr is None
        else float(bipole_nuclear_cutoff_bohr)
    )
    if _bipole_cutoff_bohr is not None or _bipole_nuclear_cutoff_bohr is not None:
        if resolved_jk != PeriodicJKMethod.BIPOLE:
            raise NotImplementedError(
                "run_periodic_job: bipole_cutoff_bohr and "
                "bipole_nuclear_cutoff_bohr apply only to "
                "jk_method='bipole'. Use cutoff_ha for GPW/GAPW grid "
                "routes, rsgdf_ke_cutoff/rsgdf_tail_ke_cutoff for GDF, "
                "or select jk_method='bipole'."
            )
        if _bipole_cutoff_bohr is not None and (
            not np.isfinite(_bipole_cutoff_bohr)
            or _bipole_cutoff_bohr <= 0.0
        ):
            raise ValueError(
                "run_periodic_job: bipole_cutoff_bohr must be finite and positive "
                f"(bohr); got {bipole_cutoff_bohr!r}."
            )
        if (
            _bipole_nuclear_cutoff_bohr is not None
            and (
                not np.isfinite(_bipole_nuclear_cutoff_bohr)
                or _bipole_nuclear_cutoff_bohr <= 0.0
            )
        ):
            raise ValueError(
                "run_periodic_job: bipole_nuclear_cutoff_bohr must be "
                "finite and positive "
                f"(bohr); got {bipole_nuclear_cutoff_bohr!r}."
            )
        if _bipole_nuclear_cutoff_bohr is None:
            _bipole_nuclear_cutoff_bohr = _bipole_cutoff_bohr
    _bipole_exact_zone_bohr = (
        None if bipole_exact_zone_bohr is None else float(bipole_exact_zone_bohr)
    )
    if _bipole_exact_zone_bohr is not None:
        # BIPOLE-EXACT-ZONE increment 1: restricted exact bielectronic
        # zone (RHF/RKS drivers). Deeper validation (zone < cutoff,
        # corrected split + padded SR path, no SYM3b reduction) happens
        # in resolve_bipole_exact_zone inside the driver.
        if resolved_jk != PeriodicJKMethod.BIPOLE:
            raise NotImplementedError(
                "run_periodic_job: bipole_exact_zone_bohr applies only to "
                "jk_method='bipole'."
            )
        if method_upper not in ("RHF", "RKS", "UHF", "UKS"):
            raise NotImplementedError(
                "run_periodic_job: bipole_exact_zone_bohr is wired for "
                "the RHF/RKS/UHF/UKS BIPOLE drivers; ROHF/ROKS route to "
                "the EWALD_3D engine, which has no exact-zone knob. Omit "
                "it for those methods."
            )
        if _bipole_exact_zone_bohr <= 0.0:
            raise ValueError(
                "run_periodic_job: bipole_exact_zone_bohr must be positive "
                f"(bohr); got {bipole_exact_zone_bohr!r}."
            )
    _rijcosx_closed_shell_multik = (
        resolved_jk == PeriodicJKMethod.RIJCOSX
        and method_upper in ("RHF", "RKS")
        and _requested_true_multik
    )
    if (
        rsgdf_tail_ke_cutoff is not None
        and resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
    ):
        _resolve_fitted_rsgdf_tail(
            _resolve_aiccm2026dev_b_backend(aiccm_backend),
            gdf_method=(gdf_method or "rsgdf"),
            rsgdf_ke_cutoff=rsgdf_ke_cutoff,
            rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
            where="run_periodic_job",
        )
    if (
        rsgdf_tail_ke_cutoff is not None
        and resolved_jk != PeriodicJKMethod.GDF
        and resolved_jk != PeriodicJKMethod.AICCM2026DEV_B
        and not _rijcosx_closed_shell_multik
    ):
        raise NotImplementedError(
            "run_periodic_job: rsgdf_tail_ke_cutoff is currently supported only "
            "with jk_method='gdf', fitted jk_method='aiccm2026dev-b', or "
            "closed-shell true multi-k jk_method='rijcosx'. Other J/K routes "
            "would ignore the high-|G| RSGDF tail correction, so this "
            "combination fails closed."
        )
    _executed_jk_method = resolved_jk.value
    if not dft_plus_u:
        _dft_plus_u_route = "none"
    elif resolved_jk == PeriodicJKMethod.BIPOLE:
        _dft_plus_u_route = (
            f"bipole_{method_upper.lower()}_"
            + ("multi_k" if _requested_true_multik else "gamma")
        )
    elif resolved_jk == PeriodicJKMethod.GDF and _requested_true_multik:
        _dft_plus_u_route = f"gdf_{method_upper.lower()}_multi_k"
    elif resolved_jk == PeriodicJKMethod.RIJCOSX and _requested_true_multik:
        _dft_plus_u_route = f"rijcosx_{method_upper.lower()}_multi_k"
    elif resolved_jk == PeriodicJKMethod.GPW:
        _dft_plus_u_route = (
            f"gpw_{method_upper.lower()}_"
            + ("multi_k" if _gpw_multik_dispatch else "gamma")
        )
    elif resolved_jk == PeriodicJKMethod.GAPW:
        _dft_plus_u_route = (
            f"gapw_{method_upper.lower()}_"
            + ("multi_k" if _gapw_multik_dispatch else "gamma")
        )
    else:
        _dft_plus_u_route = "none"
    _dftu_native_routes = (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
    )
    _dftu_gdf_native_route = (
        dft_plus_u
        and resolved_jk == PeriodicJKMethod.GDF
        and method_upper in ("RHF", "RKS")
        and _requested_true_multik
    )
    if resolved_jk == PeriodicJKMethod.GDF and int(system.dim) == 2:
        unsupported_slab_gdf: list[str] = []
        if method_upper not in ("RHF", "RKS"):
            unsupported_slab_gdf.append("open-shell method")
        # optimize=True is supported since 2026-07-30 (G-PBC-002 § 6
        # rung 5): the slab dispatch below runs through the GDF
        # optimizer-objective capture, so the relaxation re-runs the
        # identical slab driver with compute_gradient=True. Only
        # variable-cell relaxation stays closed (no slab GDF analytic
        # stress exists).
        if optimize_cell:
            unsupported_slab_gdf.append(
                "variable-cell optimization (no slab GDF analytic stress)"
            )
        if hessian:
            unsupported_slab_gdf.append("Hessian")
        if tddft:
            unsupported_slab_gdf.append("TDDFT/response")
        if dft_plus_u:
            unsupported_slab_gdf.append("DFT+U")
        if symmetry_stabilize or symmetry_reduce_fock:
            unsupported_slab_gdf.append("SCF symmetry reduction")
        if restart_from is not None or read_from is not None or _is_read_request:
            unsupported_slab_gdf.append("density restart")
        if unsupported_slab_gdf:
            raise NotImplementedError(
                "run_periodic_job: slab GDF currently supports closed-shell "
                "RHF/RKS single-point and atomic-relaxation calculations "
                "on a full Gamma-centered mesh only; unsupported "
                "option(s): "
                + ", ".join(unsupported_slab_gdf)
                + "."
            )
    if resolved_jk == PeriodicJKMethod.GDF and optimize and optimize_cell:
        # No GDF analytic stress exists, and mixing a BIPOLE-stress cell
        # step into a GDF energy surface would relax the wrong objective
        # (the silent-objective-swap bug the GDF optimizer wiring fixes).
        raise NotImplementedError(
            "run_periodic_job: optimize_cell=True is not implemented on "
            "the GDF route -- there is no GDF analytic stress. For "
            "variable-cell relaxation use a route with a certified stress "
            "and coupled atom/cell convergence implementation."
        )
    if resolved_jk == PeriodicJKMethod.GDF and optimize:
        _gdf_dispersion_disabled = dispersion in (None, False) or (
            isinstance(dispersion, str)
            and dispersion.strip().lower() in ("", "none", "false")
        )
        if not _gdf_dispersion_disabled:
            raise NotImplementedError(
                "run_periodic_job: GDF optimization with periodic dispersion "
                "is not implemented. The captured optimizer reruns the bare "
                "GDF Hamiltonian and does not add or differentiate D3 at each "
                "candidate geometry, so this combination fails before SCF "
                "instead of optimizing a different surface."
            )
    _dftu_rijcosx_native_route = (
        dft_plus_u
        and resolved_jk == PeriodicJKMethod.RIJCOSX
        and method_upper in ("RHF", "RKS")
        and _requested_true_multik
    )
    if (
        dft_plus_u
        and _jk_method_explicit
        and resolved_jk not in _dftu_native_routes
        and not _dftu_gdf_native_route
        and not _dftu_rijcosx_native_route
    ):
        raise NotImplementedError(
            "run_periodic_job: dft_plus_u with explicit "
            f"jk_method={_jk_requested_label!r} is not wired. Supported "
            "explicit +U routes are jk_method='bipole', Gamma "
            "jk_method='gpw' for RHF/RKS/UHF/UKS, multi-k GPW RKS, "
            "closed-shell jk_method='gapw', and closed-shell multi-k "
            "jk_method='gdf'/'rijcosx'. Omit jk_method to let AUTO choose "
            "a supported backend, or choose one explicitly."
        )
    # --- Cell reduction (symmetry) ------------------------------------
    system_original_info: Optional[str] = None

    # Resolve the 'symmetry' kwarg into a reduction flag.
    _sym_val = str(symmetry).lower() if isinstance(symmetry, str) else ""
    _do_reduce = (
        reduce_to_primitive or symmetry is True or _sym_val in ("auto", "reduce")
    )
    _do_attach = _do_reduce or _sym_val == "attach"

    if _do_attach and system.symmetry is None:
        attach_symmetry(system, symprec=symmetry_precision)

    if _do_reduce:
        try:
            system_prim, sg_input = _reduce_system_to_primitive(
                system,
                symprec=symmetry_precision,
            )
        except ValueError:
            # Already primitive -- just attach symmetry and continue
            if system.symmetry is None:
                attach_symmetry(system, symprec=symmetry_precision)
        else:
            if system.symmetry is None:
                attach_symmetry(system, symprec=symmetry_precision)
            if dft_plus_u:
                raise NotImplementedError(
                    "run_periodic_job: dft_plus_u cannot be combined with "
                    "primitive-cell reduction yet. Hubbard atom indices "
                    "refer to the input cell, and no symmetry-aware mapping "
                    "to primitive equivalent sites is defined. Reduce the "
                    "structure first and specify +U on that primitive cell, "
                    "or disable reduction."
                )
            if atomic_spins is not None:
                raise NotImplementedError(
                    "run_periodic_job: atomic_spins cannot be combined with "
                    "primitive-cell reduction yet. The tags refer to input-"
                    "cell atom order, and no symmetry-aware mapping to "
                    "primitive equivalent sites is defined. Reduce the "
                    "structure first and tag that primitive cell, or disable "
                    "reduction."
                )
            system_original_info = _build_primitive_summary(
                system,
                system_prim,
                sg_input,
            )
            basis = BasisSet(system_prim.unit_cell_molecule(), basis.name)
            kpoints = _remap_kpoints_after_primitive_reduction(
                system,
                system_prim,
                kpoints,
            )
            system = system_prim

    # Cover an already-primitive odd-electron cell as well as the reduced one.
    # Lower-level SCF drivers read PeriodicSystem.multiplicity directly, not
    # the auto-repaired Molecule used to construct the basis.
    system = _system_with_valid_default_multiplicity(system)
    _validate_closed_shell_electron_count(system, method_upper)

    # These four BIPOLE drivers resolve to the 3D Ewald split. An empty
    # nonzero-image set in its short-range half is valid: periodic coupling
    # also comes from the reciprocal half. A home-only SR list is therefore
    # not evidence of a free-boundary Hamiltonian (#724). Direct-truncated
    # backends retain their own nonzero-image guards.
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and method_upper in ("RHF", "RKS", "UHF", "UKS")
        and int(system.dim) == 3
    ):
        if not np.isfinite(ewald_precision) or not 0.0 < ewald_precision < 1.0:
            raise ValueError(
                "run_periodic_job: ewald_precision must be finite and in "
                f"(0, 1); got {ewald_precision!r}."
            )
        if ewald_omega is not None and (
            not np.isfinite(ewald_omega) or ewald_omega <= 0.0
        ):
            raise ValueError(
                "run_periodic_job: ewald_omega must be finite and positive; "
                f"got {ewald_omega!r}."
            )

    # External-XC GPW/GAPW and AICCM adapters have only been validated for
    # all-electron Gaussian Hamiltonians. Resolve ECP data before the dry-run
    # manifest so queue preflight cannot certify a request that live dispatch
    # would reject (or, for real-Gamma, silently treat as all-electron).
    _external_route_ecp_data = None
    if _uses_external_xc and resolved_jk in (
        PeriodicJKMethod.GDF,
        PeriodicJKMethod.RIJCOSX,
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
        PeriodicJKMethod.AICCM2026DEV_A,
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
        PeriodicJKMethod.AICCM2026DEV_B,
    ):
        _external_route_ecp_data = _resolve_ecp_data(system, basis)
        _ecp_blocks, _ecp_centers, _eff_z, _total_ncore = (
            _external_route_ecp_data
        )
        from .pbc_bipole_common import reject_bipole_ecp_options

        try:
            reject_bipole_ecp_options(
                SimpleNamespace(
                    ecp_primitive_blocks=_ecp_blocks,
                    ecp_home_centers=_ecp_centers,
                    ecp_effective_charges=_eff_z,
                    ecp_total_ncore=_total_ncore,
                ),
                driver="run_periodic_job external XC preflight",
                basis=basis,
                system=system,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "run_periodic_job: ECP-bearing bases are not implemented with "
                f"full-grid external XC on jk_method={resolved_jk.value!r}. "
                "This external-XC adapter has only been validated for "
                "all-electron Gaussian densities; it must not silently combine "
                "an unverified pseudopotential Hamiltonian with the provider."
            ) from exc

    # --- Resolve the irreducible wedge (symmetry_reduce_k) -------------
    # Resolved HERE: after primitive reduction has settled `system` and
    # remapped `kpoints`, so the count belongs to the cell that runs, and
    # before the dry-run short-circuit, so `vq` submit preflight sizes a
    # reduced job at n_IBZ x n_k rather than n_k^2 (sizing it at the full
    # mesh is the same defect as aborting it there).
    #
    # Deliberately NOT inside the estimate's try/except: when the caller
    # asked for the reduction, an unresolvable wedge is a hard error, not
    # a swallowed exception that quietly reverts to the full-mesh number.
    _symmetry_reduce_k_ibz: Optional[int] = None
    if symmetry_reduce_k:
        _symmetry_reduce_k_ibz = _ibz_kpoint_count(system, kpoints)
        if _symmetry_reduce_k_ibz is None:
            raise ValueError(
                "run_periodic_job: symmetry_reduce_k=True but the "
                "irreducible wedge of the requested k-mesh could not be "
                "resolved. It needs an attached symmetry model (pass "
                "symmetry='attach') and a Monkhorst-Pack mesh spec the "
                "reducer understands (a mesh tuple, a scalar, or a "
                "KPoints/BlochKMesh carrying `mesh`); a custom k-list "
                "quadrature has no spglib stars."
            )

    if fmixing_percent is not None and fock_mixing is not None:
        raise ValueError(
            "run_periodic_job: pass either fmixing_percent= or fock_mixing=, "
            "not both"
        )
    if fock_mixing is not None:
        fock_mixing_value = float(fock_mixing)
        if not (0.0 <= fock_mixing_value < 1.0):
            raise ValueError(
                "run_periodic_job: fock_mixing must be in [0, 1); "
                f"got {fock_mixing}"
            )
        fmixing_percent = 100.0 * fock_mixing_value

    fock_mixing = 0.0
    if fmixing_percent is not None:
        fock_mixing = float(fmixing_percent) / 100.0
        if not (0.0 <= fock_mixing < 1.0):
            raise ValueError(
                "run_periodic_job: fmixing_percent must be in [0, 100); "
                f"got {fmixing_percent}"
            )

    output_stem = Path(os.fspath(output))
    _trexio_path = None
    if trexio:
        _trexio_path = (Path(os.fspath(trexio)) if isinstance(trexio, (str, os.PathLike))
                        else Path(str(output_stem) + (".trexio.h5" if str(trexio_backend).strip().lower() == "hdf5" else ".trexio")))
    out_path = stem_sibling(output_stem, ".out")
    molden_path = stem_sibling(output_stem, ".molden")
    xsf_path = stem_sibling(output_stem, ".xsf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if qvf_wannier_centers:
        if not output_qvf:
            raise ValueError(
                "qvf_wannier_centers=True requires output_qvf=True so the "
                "x_ccm.wannier_centers overlay has a container to write into"
            )
        if resolved_jk != PeriodicJKMethod.AICCM2026DEV_B and not _requested_gamma_only:
            raise NotImplementedError(
                "qvf_wannier_centers=True needs an explicit localization "
                "convention. Two are available: jk_method='aiccm2026dev-b' "
                "(finite torus), or any route sampled at Gamma only, where "
                "the occupied Bloch functions span the whole occupied space "
                "of the cell and localizing them is a well-defined Wannier "
                "construction for it. A multi-k run is neither: localizing "
                "only the Gamma orbitals would sample one k point and call "
                "the result a Wannier function. See "
                "handovers/HANDOVER_IBO.md section Milestone 3 for the "
                "multi-k scheme (Zicovich-Wilson, doi:10.1063/1.1415745), "
                "which is not implemented."
            )

    if coop_cohp and not output_qvf:
        # The COOP/COHP analysis runs inside the DOS/QVF block below and its
        # only sink is the QVF archive (dos.coop / dos.cohp). With QVF output
        # off that block never executes, so the request used to be dropped
        # silently: converged run, no .qvf, not one COOP/COHP line in the
        # .out (IID 195). Fail closed before SCF, the same way
        # qvf_wannier_centers does just above for the same reason.
        raise ValueError(
            "coop_cohp=True requires output_qvf=True so the dos.coop and "
            "dos.cohp sections have a container to write into; the analysis "
            "has no .out or sidecar rendering of its own"
        )

    _population_variant = (
        "bipole"
        if resolved_jk == PeriodicJKMethod.BIPOLE
        else (
            "aiccm2026dev-b"
            if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
            else "standard"
        )
    )
    from .guess import periodic_guess_capabilities, select_initial_guess
    _guess_route = {
        PeriodicJKMethod.AICCM2026DEV_A: "aiccm-hcore",
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA: "aiccm-hcore",
        PeriodicJKMethod.AICCM2026DEV_B: "bipole",
        PeriodicJKMethod.NEUTRAL_BLOCH: "gdf",
        PeriodicJKMethod.SLAB_EWALD_2D: "ewald",
        PeriodicJKMethod.DIRECT: "native",
        PeriodicJKMethod.FFT_POISSON: "native",
    }.get(resolved_jk, resolved_jk.value)
    selection = select_initial_guess(
        system.unit_cell_molecule(), _requested_initial_guess,
        is_periodic=True, is_open_shell=method_upper in ("UHF", "UKS", "ROHF", "ROKS"),
        supported=periodic_guess_capabilities(
            _guess_route, method_upper, dim=system.dim, multi_k=_requested_true_multik,
        ),
        atomic_spins=atomic_spins, restart_supplied=restart_from is not None,
        driver="run_periodic_job",
    )
    _resolved_initial_guess = selection.effective
    # ECP-dependent guess capabilities must be checked before any output plan
    # or citation assembly. Reuse this operator payload for SCF attachment.
    if _external_route_ecp_data is None:
        _external_route_ecp_data = _resolve_ecp_data(system, basis)
    from .guess import guess_ecp_context, validate_guess_ecp
    _guess_blocks, _guess_centers, _guess_charges, _guess_ncore = _external_route_ecp_data
    _guess_molecule = system.unit_cell_molecule()
    _guess_context = guess_ecp_context(SimpleNamespace(
        ecp_primitive_blocks=_guess_blocks, ecp_home_centers=_guess_centers,
        ecp_effective_charges=_guess_charges, ecp_total_ncore=_guess_ncore,
    ), molecule=_guess_molecule)
    validate_guess_ecp(_resolved_initial_guess, _guess_context, _guess_molecule)

    real_plan = OutputPlan.from_run_job_kwargs(
        output=output_stem,
        method=method_upper,
        basis=basis.name,
        functional=functional,
        write_molden_file=write_molden_file,
        write_trexio_file=_trexio_path,
        write_xyz=write_xyz_file,
        write_poscar=write_poscar_file,
        write_xsf_structure=write_xsf_structure_file,
        write_density_xsf=write_density,
        write_cif=write_cif_file,
        write_population=write_population_file,
        population_variant=_population_variant,
        citations=citations,
        crash_dump=False,
        output_qvf=output_qvf,
        job_kind="periodic_scf",
    )

    if resolved_jk in (
        PeriodicJKMethod.GDF,
        # The neutral Bloch producer executes the same multi-k KS drivers,
        # so an unsupported functional must fail here for it too.
        PeriodicJKMethod.NEUTRAL_BLOCH,
    ) and method_upper in (
        "RKS",
        "UKS",
        "ROKS",
    ):
        reject_periodic_gdf_unsupported_functional(
            Functional(
                str(functional or "pbe"),
                2 if method_upper == "UKS" else 1,
            ),
            where="run_periodic_job",
        )

    # --- use_multipole_far_field: honoured or refused, never dropped (#511)
    # This flag reaches a consumer on exactly one route: the four direct
    # BIPOLE drivers read it, and nothing else in run_periodic_job does.
    # Until #511 its only validation was the reject_bipole_quartet_far_field
    # call below, which sat *inside* the BIPOLE preflight branch -- so an
    # explicit True on any other jk_method reached neither a consumer nor a
    # guard. The job ran the exact traversal to completion and said nothing,
    # leaving the caller believing they had measured the far-field route.
    # That is the failure shape of #113 and L54.
    #
    # The route dispatch lives here, in one place, ahead of the BIPOLE
    # branch and therefore ahead of the dry-run manifest, so queue preflight
    # cannot certify a request live dispatch would ignore. BIPOLE keeps its
    # own, more specific refusal (the quartet-domain explanation); every
    # other route gets one that names the route it actually asked for.
    # reject_bipole_quartet_far_field stays the single owner of the
    # bool-or-None type contract, which now applies on every route rather
    # than on BIPOLE alone.
    from .pbc_bipole_common import reject_bipole_quartet_far_field

    if use_multipole_far_field and resolved_jk != PeriodicJKMethod.BIPOLE:
        raise NotImplementedError(
            "run_periodic_job: use_multipole_far_field is a BIPOLE-only "
            f"control, but jk_method resolved to {resolved_jk.value!r}, "
            "which has no far-field route and would have silently run the "
            "exact traversal instead. Drop the flag, or pass "
            "jk_method='bipole' (where True is itself still refused until "
            "the quartet dispatch preserves the three-translation Fock "
            "domain)."
        )
    reject_bipole_quartet_far_field(
        use_multipole_far_field,
        driver="run_periodic_job",
    )

    # Queue/preflight validation must reject the same BIPOLE envelopes as
    # execution. These checks deliberately run before the dry-run manifest.
    if resolved_jk == PeriodicJKMethod.BIPOLE:
        from .pbc_bipole_common import (
            reject_bipole_ecp_options,
            reject_bipole_fractional_legacy_gauge,
            reject_bipole_unsupported_ks_functional,
            validate_bipole_kmesh,
        )

        if method_upper in ("RKS", "UKS"):
            reject_bipole_unsupported_ks_functional(
                Functional(
                    str(functional or "pbe"),
                    2 if method_upper == "UKS" else 1,
                ),
                driver="run_periodic_job",
            )

        if exchange_exxdiv not in ("ewald", "none"):
            raise ValueError(
                "run_periodic_job: exchange_exxdiv must be 'ewald' or "
                f"'none'; got {exchange_exxdiv!r}."
            )
        if int(max_iter) < 1:
            raise ValueError(
                "run_periodic_job: BIPOLE max_iter must be at least 1."
            )
        if damping is not None and not (0.0 <= float(damping) < 1.0):
            raise ValueError(
                "run_periodic_job: BIPOLE damping must be in [0, 1); "
                f"got {damping!r}."
            )
        if sr_image_precision is not None:
            _preflight_sr_precision = float(sr_image_precision)
            if not (0.0 < _preflight_sr_precision < 1.0):
                raise ValueError(
                    "run_periodic_job: sr_image_precision must be in (0, 1) "
                    f"or None; got {sr_image_precision!r}."
                )
        if use_oda and use_diis:
            raise ValueError(
                "run_periodic_job: use_oda and use_diis are mutually exclusive."
            )
        if use_oda and not (0.0 < float(oda_trust_lambda_max) <= 1.0):
            raise ValueError(
                "run_periodic_job: oda_trust_lambda_max must be in (0, 1]; "
                f"got {oda_trust_lambda_max!r}."
            )
        if use_oda:
            raise NotImplementedError(
                "run_periodic_job: use_oda (BIPOLE ODA) is unavailable "
                "because a "
                "mixed line-search density has no single orbital "
                "representation (and the unrestricted slope omits one spin "
                "direction); use DIIS instead."
            )
        if _bipole_exact_zone_bohr is not None:
            if use_exchange_ewald_split is False:
                raise ValueError(
                    "run_periodic_job: bipole_exact_zone_bohr requires the "
                    "corrected Ewald exchange split; "
                    "use_exchange_ewald_split=False is incompatible."
                )
            if sr_image_precision is None:
                raise ValueError(
                    "run_periodic_job: bipole_exact_zone_bohr requires the "
                    "padded short-range image path; sr_image_precision=None "
                    "is incompatible."
                )
            if symmetry_stabilize or symmetry_reduce_fock is True:
                raise NotImplementedError(
                    "run_periodic_job: bipole_exact_zone_bohr is not wired "
                    "for explicit Fock symmetry stabilization or reduction."
                )
            _preflight_bipole_cutoff = float(
                _bipole_cutoff_bohr
                if _bipole_cutoff_bohr is not None
                else PeriodicRHFOptions().lattice_opts.cutoff_bohr
            )
            if not _bipole_exact_zone_bohr < _preflight_bipole_cutoff:
                raise ValueError(
                    "run_periodic_job: bipole_exact_zone_bohr must be "
                    "strictly below the BIPOLE operator cutoff "
                    f"({_bipole_exact_zone_bohr!r} vs "
                    f"{_preflight_bipole_cutoff!r})."
                )
        if method_upper in ("ROHF", "ROKS"):
            # These methods use the dedicated corrected-Ewald restricted-open
            # engine, not the four direct BIPOLE drivers. Validate every
            # BIPOLE-only control before dry-run so queue preflight cannot
            # certify a request that live dispatch will reject or ignore.
            # Note: use_multipole_far_field can no longer reach this list --
            # the #511 guard above refuses it on every route first. The row
            # is retained as the complete statement of what this engine does
            # not implement, not because it is the live refusal path.
            _unsupported_open_shell_knobs = {
                "use_oda": bool(use_oda),
                "use_mom": bool(use_mom),
                "use_multipole_far_field": bool(use_multipole_far_field),
                "symmetry_stabilize": bool(symmetry_stabilize),
                "symmetry_reduce_fock": symmetry_reduce_fock is True,
                "ewald_omega": ewald_omega is not None,
                "ewald_precision": float(ewald_precision) != 1.0e-8,
                "use_exchange_ewald_split": use_exchange_ewald_split is False,
                "exchange_exxdiv": exchange_exxdiv != "ewald",
                "fock_mixing/fmixing_percent": fock_mixing != 0.0,
            }
            _set_knobs = [
                key for key, requested in _unsupported_open_shell_knobs.items()
                if requested
            ]
            if _set_knobs:
                raise NotImplementedError(
                    f"run_periodic_job: periodic {method_upper}/bipole runs "
                    "on the corrected Ewald-exchange EWALD_3D engine, which "
                    "does not implement these explicitly requested options: "
                    + ", ".join(sorted(_set_knobs))
                    + ". Drop them or use a supported method/backend."
                )

        reject_bipole_ecp_options(
            SimpleNamespace(),
            driver="run_periodic_job",
            basis=basis,
            system=system,
        )
        _preflight_ecp = _external_route_ecp_data
        if (
            len(_preflight_ecp[0]) > 0
            or len(_preflight_ecp[1]) > 0
            or int(_preflight_ecp[3] or 0) != 0
        ):
            raise NotImplementedError(
                "run_periodic_job: ECP-bearing bases are not implemented on "
                "jk_method='bipole'."
            )
        _preflight_smearing = smearing
        _preflight_smearing_engaged = (
            smearing is not None
            or smearing_temperature is not _SMEARING_UNSET
            or smearing_metallic is not None
            or smearing_band_gap_hartree is not None
        )
        if not _preflight_smearing_engaged:
            _preflight_smearing = getattr(kpoints, "smearing", None)
            _preflight_smearing_engaged = _preflight_smearing is not None
        _preflight_temperature_arg = (
            0.0
            if smearing_temperature is _SMEARING_UNSET
            else smearing_temperature
        )
        if _preflight_smearing is not None:
            if _preflight_temperature_arg not in (0.0, None):
                raise ValueError(
                    "run_periodic_job: pass either smearing= or "
                    "smearing_temperature=, not both"
                )
            if not isinstance(_preflight_smearing, SmearingOptions):
                raise TypeError(
                    "run_periodic_job: smearing must be a SmearingOptions "
                    "instance"
                )
            _preflight_smearing_temperature = float(
                _preflight_smearing.temperature
            )
            _preflight_smearing_label = _preflight_smearing.flavor
        else:
            _preflight_smearing_resolution = resolve_smearing_temperature(
                _preflight_temperature_arg,
                unit=smearing_unit,
                method=smearing_method,
                metallic=smearing_metallic,
                band_gap_hartree=smearing_band_gap_hartree,
                n_electrons=system.n_electrons(),
            )
            _preflight_smearing_temperature = float(
                _preflight_smearing_resolution.temperature
            )
            _preflight_smearing_label = _preflight_smearing_resolution.method
        _preflight_smearing_label = str(
            _preflight_smearing_label
        ).strip().lower().replace("_", "-")
        _validate_smearing_dispatch(
            method=method_upper,
            jk_method=resolved_jk,
            smearing_temperature=_preflight_smearing_temperature,
            kpoints=kpoints,
        )
        if (
            _preflight_smearing_temperature > 0.0
            and _preflight_smearing_label not in ("fermi-dirac", "mermin")
        ):
            raise NotImplementedError(
                "run_periodic_job: BIPOLE currently implements only "
                "Fermi-Dirac occupations (including the Mermin free-energy "
                "label); the requested flavour would otherwise be executed "
                "as Fermi-Dirac."
            )
        _preflight_bz = bz_integration
        if _preflight_bz is _BZ_INTEGRATION_UNSET:
            _preflight_bz = getattr(kpoints, "bz_integration", None)
        if _preflight_bz is not None:
            _preflight_bz = str(_preflight_bz).strip().lower()
            if _preflight_bz not in ("smearing", "gilat"):
                raise ValueError(
                    "run_periodic_job: bz_integration must be None, "
                    f"'smearing', or 'gilat'; got {_preflight_bz!r}."
                )
        if method_upper == "RHF" and _preflight_bz == "gilat":
            raise NotImplementedError(
                "run_periodic_job: BIPOLE RHF does not implement "
                "bz_integration='gilat'; use RKS/UHF/UKS or omit it."
            )
        if method_upper in ("ROHF", "ROKS") and _preflight_bz == "gilat":
            raise NotImplementedError(
                "run_periodic_job: Gilat-Raubenheimer BZ integration is "
                f"not implemented for periodic {method_upper}."
            )
        if (
            _preflight_bz == "gilat"
            and _preflight_smearing_temperature > 0.0
        ):
            raise ValueError(
                "run_periodic_job: bz_integration='gilat' is a T=0 "
                "integrator and cannot be combined with finite-temperature "
                "smearing."
            )
        if _preflight_bz == "gilat":
            _gilat_ir = np.asarray(
                getattr(_requested_bloch_kmesh, "ir_mapping", []), dtype=int
            ).reshape(-1)
            _gilat_mesh = tuple(
                int(x)
                for x in getattr(
                    _requested_bloch_kmesh, "mesh", (1, 1, 1)
                )
            )
            _gilat_stored = _bloch_kmesh_size(_requested_bloch_kmesh)
            if (
                _gilat_ir.size == 0
                and int(np.prod(_gilat_mesh)) != _gilat_stored
            ):
                raise NotImplementedError(
                    "run_periodic_job: BIPOLE Gilat-Raubenheimer "
                    "integration requires a complete Monkhorst-Pack mesh "
                    "carrying its grid dimensions; arbitrary explicit or "
                    "band-path k-point lists are not regular integration "
                    "grids."
                )
        # Resolve the final BIPOLE convergence strategy before dry-run for
        # every single-point job, not only optimizations.  AUTO can introduce
        # finite-temperature occupations for metallic KS systems; the
        # fractional/legacy-gauge capability guard must see that same final
        # temperature that live dispatch installs on ``opts``.
        _preflight_convergence = resolve_convergence_strategy(
            system,
            method=method_upper,
            convergence=convergence,
            ks_driver_fock_mixing_floor=not bool(use_diis),
            explicit={
                "fock_mixing": (
                    fock_mixing if fmixing_percent is not None else None
                ),
                "level_shift": (
                    float(level_shift) if level_shift is not None else None
                ),
                "damping": float(damping) if damping is not None else None,
                "smearing_temperature": (
                    _preflight_smearing_temperature
                    if _preflight_smearing_engaged
                    else None
                ),
            },
        )
        _preflight_convergence = _filter_bipole_restricted_open_convergence(
            _preflight_convergence,
            method_upper,
        )
        _preflight_final_temperature = float(
            _preflight_convergence.value("smearing_temperature")
        )
        _validate_smearing_dispatch(
            method=method_upper,
            jk_method=resolved_jk,
            smearing_temperature=_preflight_final_temperature,
            kpoints=kpoints,
        )
        _preflight_fractional_occupations = bool(
            (
                method_upper in ("RKS", "UKS")
                and (
                    _preflight_final_temperature > 0.0
                    or _preflight_bz == "gilat"
                )
            )
            or (method_upper == "UHF" and _preflight_bz == "gilat")
        )
        if _preflight_fractional_occupations:
            # Fractional occupations need the corrected gauge.  It is active
            # by default for a complete MP/IBZ mesh, but an ad-hoc multi-k
            # list makes AUTO fall back to the nonstationary legacy gauge.
            validate_bipole_kmesh(
                _requested_bloch_kmesh,
                driver="run_periodic_job fractional occupations",
                require_complete=True,
            )
            reject_bipole_fractional_legacy_gauge(
                use_ewald_j_split=True,
                exchange_split_active=use_exchange_ewald_split is not False,
                fractional_occupations=True,
                driver="run_periodic_job",
            )
        if optimize:
            _, _, _optimizer_true_multik, _ = validate_bipole_kmesh(
                _requested_bloch_kmesh,
                driver="run_periodic_job optimize",
                require_complete=True,
            )
            if (
                _optimizer_true_multik
                and use_exchange_ewald_split is False
            ):
                raise NotImplementedError(
                    "run_periodic_job: multi-k BIPOLE optimization requires "
                    "the corrected Ewald exchange split. The legacy gauge "
                    "does not define a stationary geometry objective."
                )
            if (
                _preflight_final_temperature > 0.0
            ):
                raise NotImplementedError(
                    "run_periodic_job: finite-temperature BIPOLE "
                    "optimization is unavailable until optimization and "
                    "trajectory/QVF results distinguish the Mermin "
                    "free-energy objective from internal energy. Run a "
                    "fixed-geometry smeared SCF or optimize at T=0."
                )

    if resolved_jk == PeriodicJKMethod.GDF and optimize:
        # Resolve the same final temperature that live GDF dispatch installs
        # after AUTO convergence classification.  The analytic gradient is
        # dA/dR at finite T, but OptimizeResult and the optimized-geometry/QVF
        # schema currently expose its scalar only as generic energy.  Until
        # the objective kind is represented explicitly, fail before dry-run
        # rather than publishing A as E.
        _gdf_preflight_smearing = smearing
        _gdf_preflight_smearing_engaged = (
            smearing is not None
            or smearing_temperature is not _SMEARING_UNSET
            or smearing_metallic is not None
            or smearing_band_gap_hartree is not None
        )
        if not _gdf_preflight_smearing_engaged:
            _gdf_preflight_smearing = getattr(kpoints, "smearing", None)
            _gdf_preflight_smearing_engaged = (
                _gdf_preflight_smearing is not None
            )
        _gdf_preflight_temperature_arg = (
            0.0
            if smearing_temperature is _SMEARING_UNSET
            else smearing_temperature
        )
        if _gdf_preflight_smearing is not None:
            if _gdf_preflight_temperature_arg not in (0.0, None):
                raise ValueError(
                    "run_periodic_job: pass either smearing= or "
                    "smearing_temperature=, not both"
                )
            if not isinstance(_gdf_preflight_smearing, SmearingOptions):
                raise TypeError(
                    "run_periodic_job: smearing must be a SmearingOptions "
                    "instance"
                )
            _gdf_preflight_temperature = float(
                _gdf_preflight_smearing.temperature
            )
        else:
            _gdf_preflight_resolution = resolve_smearing_temperature(
                _gdf_preflight_temperature_arg,
                unit=smearing_unit,
                method=smearing_method,
                metallic=smearing_metallic,
                band_gap_hartree=smearing_band_gap_hartree,
                n_electrons=system.n_electrons(),
            )
            _gdf_preflight_temperature = float(
                _gdf_preflight_resolution.temperature
            )
        _gdf_preflight_convergence = resolve_convergence_strategy(
            system,
            method=method_upper,
            convergence=convergence,
            ks_driver_fock_mixing_floor=False,
            explicit={
                "fock_mixing": (
                    fock_mixing if fmixing_percent is not None else None
                ),
                "level_shift": (
                    float(level_shift) if level_shift is not None else None
                ),
                "damping": float(damping) if damping is not None else None,
                "smearing_temperature": (
                    _gdf_preflight_temperature
                    if _gdf_preflight_smearing_engaged
                    else None
                ),
            },
        )
        _gdf_preflight_final_temperature = float(
            _gdf_preflight_convergence.value("smearing_temperature")
        )
        if _gdf_preflight_final_temperature > 0.0:
            raise NotImplementedError(
                "run_periodic_job: finite-temperature GDF optimization is "
                "unavailable until optimization and trajectory/QVF results "
                "distinguish the Mermin free-energy objective from internal "
                "energy. Run a fixed-geometry smeared GDF SCF or optimize "
                "at T=0."
            )
        _gdf_preflight_bz = bz_integration
        if _gdf_preflight_bz is _BZ_INTEGRATION_UNSET:
            _gdf_preflight_bz = getattr(kpoints, "bz_integration", None)
        if _gdf_preflight_bz is not None:
            _gdf_preflight_bz = str(_gdf_preflight_bz).strip().lower()
        if _gdf_preflight_bz not in (None, "smearing"):
            raise NotImplementedError(
                "run_periodic_job: GDF optimization supports Aufbau or "
                "Fermi-Dirac occupations only; the selected BZ integrator "
                "has no analytic occupation-response gradient."
            )
        if gdf_method is not None and str(gdf_method).strip().lower() != "rsgdf":
            raise NotImplementedError(
                "run_periodic_job: GDF optimization currently supports "
                "gdf_method='rsgdf' only; MDF/compcell fit derivatives are "
                "not implemented."
            )
        if dft_plus_u:
            raise NotImplementedError(
                "run_periodic_job: GDF optimization does not differentiate "
                "the DFT+U energy term. Run a +U single point or optimize "
                "without Hubbard sites."
            )
        if method_upper in ("RKS", "UKS") and functional is not None:
            _gdf_opt_functional = Functional(
                str(functional), 2 if method_upper == "UKS" else 1
            )
            if bool(getattr(_gdf_opt_functional, "is_range_separated", False)):
                raise NotImplementedError(
                    "run_periodic_job: GDF optimization does not support "
                    "range-separated or screened hybrids because the "
                    "analytic fitted-exchange gradient is full-range only."
                )

        # Prove that live dispatch can capture a GDF driver whose analytic
        # gradient differentiates the same SCF objective.  Without this
        # preflight, dry-run could certify jobs that either fell onto the
        # uncaptured legacy-Gamma bridge or failed only when the optimizer
        # reran the converged SCF with compute_gradient=True.
        _gdf_opt_kmesh = _runner_bloch_kmesh(system, kpoints)
        _gdf_opt_gamma = _gamma_kmesh_info(system, _gdf_opt_kmesh) is not None
        _gdf_opt_full_size = _bloch_kmesh_full_size(_gdf_opt_kmesh)
        _gdf_opt_true_multik = _gdf_opt_full_size > 1
        _gdf_opt_dim = int(system.dim)
        _gdf_opt_density_mixer = _canonical_gdf_density_mixer(density_mixer)
        if _gdf_opt_dim == 1:
            raise NotImplementedError(
                "run_periodic_job: GDF optimization does not support dim=1 "
                "wire cells because the analytic wire GDF gradient is not "
                "implemented."
            )

        _gdf_opt_fmix_resolution = _gdf_preflight_convergence.knobs[
            "fock_mixing"
        ]
        _gdf_opt_shift_resolution = _gdf_preflight_convergence.knobs[
            "level_shift"
        ]
        _gdf_opt_fmix = float(_gdf_opt_fmix_resolution.value)
        _gdf_opt_shift = float(_gdf_opt_shift_resolution.value)
        _gdf_opt_neutral = (
            abs(
                float(sum(atom.Z for atom in system.unit_cell))
                - int(system.n_electrons())
            )
            <= 0.5
        )
        _gdf_opt_default_rhf_gamma = (
            method_upper == "RHF"
            and kpoints is None
            and gdf_method is None
            and _gdf_opt_dim == 3
            and int(system.n_electrons()) % 2 == 0
            and int(system.multiplicity) == 1
            and _gdf_opt_neutral
            and not symmetry_stabilize
            and not symmetry_reduce_fock
        )
        # Mirror the live GDF capability filter: AUTO-only aids that a chosen
        # exact-Gamma/open-shell driver cannot execute are zeroed there.  An
        # explicit nonzero request remains part of the requested Hamiltonian
        # and must be rejected rather than silently discarded.
        if _gdf_opt_shift_resolution.source == "auto" and (
            method_upper in ("ROHF", "UHF", "UKS")
            or (_gdf_opt_gamma and gdf_method is not None)
            or _gdf_opt_default_rhf_gamma
        ):
            _gdf_opt_shift = 0.0
        if _gdf_opt_fmix_resolution.source == "auto" and (
            method_upper == "ROHF"
            or (method_upper in ("UHF", "UKS") and kpoints is not None)
            or (
                method_upper == "RHF"
                and _gdf_opt_gamma
                and gdf_method is not None
            )
            or _gdf_opt_default_rhf_gamma
        ):
            _gdf_opt_fmix = 0.0

        # The bounded slab adapter validates the finalized convergence
        # controls, after AUTO selection, and supports only plain Fock DIIS.
        # Mirror that envelope here so dry-run never certifies a relaxation
        # whose first live SCF would reject one of these controls (and so a
        # live rejection cannot leave a started manifest behind).
        if _gdf_opt_dim == 2:
            from ._vibeqc_core import BlochKMesh as _BlochKMesh
            from .kpoints import KPoints as _KPoints

            if isinstance(kpoints, (_KPoints, _BlochKMesh)):
                raise NotImplementedError(
                    "run_periodic_job: slab GDF optimization currently "
                    "requires a full Gamma-centered tuple mesh; custom, "
                    "shifted, weighted, and symmetry-reduced mesh objects "
                    "remain fail-closed."
                )
            if rsgdf_tail_ke_cutoff is not None:
                raise NotImplementedError(
                    "run_periodic_job: slab GDF optimization does not yet "
                    "implement the bulk high-|G| tail correction."
                )
            if _gdf_preflight_bz is not None:
                raise NotImplementedError(
                    "run_periodic_job: slab GDF optimization currently "
                    "supports integer zero-temperature occupations only; "
                    "bz_integration must be omitted."
                )
            if _gdf_opt_fmix != 0.0:
                raise NotImplementedError(
                    "run_periodic_job: slab GDF optimization does not yet "
                    "implement fock_mixing; pass fock_mixing=0.0."
                )
            if _gdf_opt_shift != 0.0:
                raise NotImplementedError(
                    "run_periodic_job: slab GDF optimization does not yet "
                    "implement level_shift; pass level_shift=0.0."
                )
            _gdf_opt_mixer_key = (
                None
                if _gdf_opt_density_mixer is None
                else str(_gdf_opt_density_mixer).strip().lower()
            )
            if _gdf_opt_mixer_key not in (None, "", "none", "diis"):
                raise NotImplementedError(
                    "run_periodic_job: slab GDF optimization currently "
                    "supports Fock DIIS only; Anderson, Broyden, and Kerker "
                    "density mixing remain fail-closed."
                )

        if _gdf_opt_gamma and _gdf_opt_dim == 3:
            _gdf_opt_legacy_reasons: list[str] = []
            if not _gdf_opt_neutral:
                _gdf_opt_legacy_reasons.append("charged cell")
            if symmetry_stabilize or symmetry_reduce_fock:
                _gdf_opt_legacy_reasons.append("SCF symmetry reduction")
            if _gdf_opt_shift != 0.0:
                _gdf_opt_legacy_reasons.append("level shift")
            # Default-Gamma RHF and an explicit Gamma mesh without a density
            # mixer require the exact-Gamma fast path, which does not carry
            # Fock mixing.  RKS with kpoints=None uses the native UKS-singlet
            # GDF engine and legitimately supports this aid.
            if _gdf_opt_fmix != 0.0 and (
                (method_upper == "RHF" and kpoints is None)
                or (
                    method_upper in ("RHF", "RKS")
                    and kpoints is not None
                    and _gdf_opt_density_mixer is None
                )
            ):
                _gdf_opt_legacy_reasons.append("Fock mixing")
            if _gdf_opt_legacy_reasons:
                raise NotImplementedError(
                    "run_periodic_job: GDF optimization would use the "
                    "legacy Gamma molecular-limit GDF driver, which has no "
                    "analytic gradient (unsupported: "
                    + ", ".join(_gdf_opt_legacy_reasons)
                    + "). Remove those options or run a fixed-geometry "
                    "single point."
                )

        if _gdf_opt_true_multik:
            _gdf_opt_expanded = _expand_ibz_kmesh_to_full_bz(
                system, _gdf_opt_kmesh
            )
            _gdf_opt_effective_kmesh = (
                _gdf_opt_expanded
                if _gdf_opt_expanded is not None
                else _gdf_opt_kmesh
            )
            _gdf_opt_weights = np.asarray(
                getattr(_gdf_opt_effective_kmesh, "weights", []),
                dtype=float,
            ).reshape(-1)
            _gdf_opt_nk = int(_gdf_opt_weights.size)
            if (
                _gdf_opt_nk == 0
                or not np.all(np.isfinite(_gdf_opt_weights))
                or not np.allclose(
                    _gdf_opt_weights,
                    1.0 / _gdf_opt_nk,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise NotImplementedError(
                    "run_periodic_job: GDF optimization requires a uniform "
                    "full-BZ k-point mesh. Custom nonuniform weights have no "
                    "orbit-unfolded analytic exchange derivative."
                )

        # Pure RKS in 3D normally uses the cheaper Ewald-3D J/K loop, whose
        # use_compcell=False state has no fitted Lpq cache to differentiate.
        # Any explicit non-Gamma mesh enters that loop, even when Nk == 1.
        # An explicit Gamma mesh with an active Anderson/Broyden density mixer
        # does too (rather than using the pure-GDF Gamma fast path). Default
        # DIIS aliases are canonicalized above and retain the fast path. Cover
        # every dispatch shape here.
        _gdf_opt_rks_ewald3d_loop = _gdf_opt_true_multik or (
            kpoints is not None
            and (
                not _gdf_opt_gamma
                or _gdf_opt_density_mixer is not None
            )
        )
        if (
            method_upper == "RKS"
            and _gdf_opt_dim == 3
            and _gdf_opt_rks_ewald3d_loop
        ):
            _gdf_opt_alpha = float(
                Functional(str(functional or "lda"), 1).hf_exchange_fraction
            )
            if _gdf_opt_alpha == 0.0:
                if _gdf_opt_true_multik:
                    _gdf_opt_sampling = "multi-k"
                elif _gdf_opt_gamma:
                    _gdf_opt_sampling = "explicit-Gamma density-mixed"
                else:
                    _gdf_opt_sampling = "explicit non-Gamma"
                raise NotImplementedError(
                    "run_periodic_job: pure-functional "
                    f"{_gdf_opt_sampling} RKS/GDF optimization is unavailable "
                    "because live dispatch uses use_compcell=False, while "
                    "the analytic gradient requires the cached-Lpq GDF J/K "
                    "objective. Use a global hybrid, remove density_mixer "
                    "from an explicit Gamma calculation, or run a "
                    "fixed-geometry pure-DFT single point."
                )

    # Validate the caller's density-mixer request before dry-run can publish
    # a manifest.  The later call remains necessary because AUTO convergence
    # may select a density mixer after this boundary; this early call pins
    # dry/live parity for every explicit mixer and all of its parameters.
    _validate_density_mixer_dispatch(
        density_mixer=density_mixer,
        density_mixer_depth=density_mixer_depth,
        density_mixer_beta=density_mixer_beta,
        density_mixer_kerker=density_mixer_kerker,
        kerker_k0=kerker_k0,
        kerker_strength=kerker_strength,
        kerker_cutoff_ha=kerker_cutoff_ha,
        jk_method=resolved_jk,
        method=method_upper,
        kpoints=kpoints,
    )

    # A full-grid provider needs the discrete inverse Bloch transform of a
    # complete finite torus.  Validate object-form meshes before dry-run can
    # publish a manifest; the low-level adapter repeats this check at the
    # actual fold boundary.  Tuple/scalar requests materialize as the native
    # unreduced Monkhorst-Pack mesh here and are therefore covered too.
    if _uses_external_xc and resolved_jk in (
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GDF,
        PeriodicJKMethod.RIJCOSX,
    ):
        from .periodic_external_xc import _reject_reduced_kmesh

        _reject_reduced_kmesh(
            _runner_bloch_kmesh(system, kpoints),
            system,
        )

    # Dry-run short-circuit (Phase O5). Mirrors the molecular run_job
    # path: build the OutputPlan from current kwargs, write a one-shot
    # ``{output}.system`` with ``[outputs].status = "dry_run"``, print
    # the declared-artefacts summary, and return None without running
    # the SCF. Honours both the ``dry_run=True`` kwarg and the
    # ``VIBEQC_DRY_RUN=1`` env var that vq's submit pre-flight sets.
    if dry_run or is_dry_run_requested():
        _estimate_bytes: Optional[int] = None
        if is_dry_run_estimate_requested():
            try:
                _estimate_bytes = _periodic_gpw_gapw_dry_run_estimate_bytes(
                    system,
                    basis,
                    resolved_jk=resolved_jk,
                    method_upper=method_upper,
                    functional=functional,
                    cutoff_ha=cutoff_ha,
                    kpoints=kpoints,
                )
            except Exception:
                _estimate_bytes = None
            if _estimate_bytes is None:
                try:
                    _gdf_memory = _periodic_gdf_estimate(
                        system,
                        basis,
                        resolved_jk=resolved_jk,
                        method_upper=method_upper,
                        functional=functional,
                        kpoints=kpoints,
                        aux_basis=aux_basis,
                        n_ibz_kpoints=_symmetry_reduce_k_ibz,
                        diis_subspace_size=int(diis_subspace_size) if use_diis else 0,
                        compute_gradient=bool(optimize),
                    )
                    if _gdf_memory is not None:
                        _estimate_bytes = _gdf_memory.estimate.total_bytes
                except Exception:
                    _estimate_bytes = None
            if optimize:
                try:
                    _gradient_estimate_bytes = (
                        _periodic_xc_gradient_dry_run_estimate_bytes(
                            system,
                            basis,
                            method_upper=method_upper,
                            functional=functional,
                            lattice_cutoff_bohr=_bipole_cutoff_bohr,
                        )
                    )
                    if _gradient_estimate_bytes is not None:
                        _estimate_bytes = max(
                            int(_estimate_bytes or 0),
                            int(_gradient_estimate_bytes),
                        )
                except Exception:
                    pass
            if _uses_skala:
                try:
                    _skala_memory = _periodic_skala_memory_estimate(system)
                    _skala_workspace = _skala_memory.by_category[
                        "SKALA full-grid + model workspace"
                    ]
                    if _estimate_bytes is None:
                        _estimate_bytes = _skala_memory.total_bytes
                    else:
                        _estimate_bytes += int(
                            _skala_workspace
                            * _skala_memory.headroom_factor
                        )
                except Exception:
                    # Dry-run estimation is deliberately best effort. The
                    # live preflight below repeats the same deterministic
                    # calculation and will enforce it before model loading.
                    pass
        from .periodic.exchange_convention import exchange_q0_label

        _dry_run_fields = {
            "exchange_q0": exchange_q0_label(exchange_exxdiv),
            "jk_method_requested": _jk_requested_label,
            "jk_method_resolved": resolved_jk.value,
            "initial_guess_requested": (InitialGuess.FRAGMO if _fragmo_request else _requested_initial_guess).name,
            "initial_guess": (
                "FRAGMO" if _fragmo_request else
                "READ" if restart_from is not None else _resolved_initial_guess.name
            ),
            "initial_guess_transport": (
                "READ" if restart_from is not None else _resolved_initial_guess.name
            ),
            "jk_method_executed": _executed_jk_method,
            "dft_plus_u": bool(dft_plus_u),
            "dft_plus_u_route": _dft_plus_u_route,
        }
        if _uses_skala:
            from .skala import _run_provenance_fields

            _dry_run_fields.update(_run_provenance_fields())
            _dry_run_fields["skala_grid_policy"] = "skala"
        if resolved_jk in _AICCM_JK_METHODS:
            _dry_run_fields["method_status"] = "experimental"
            # M1: which formulation ran and through which selector surface,
            # the two facts a reader of the .system cannot recover once the
            # legacy spellings and the front door map onto the same member.
            _dry_run_fields["aiccm_variant"] = AICCM_VARIANT_OF[resolved_jk]
            _dry_run_fields["aiccm_selector"] = (
                "front-door" if _aiccm_front_door else "legacy-jk_method"
            )
            # M4b (#778): the post-HF treatment, recorded only when one was
            # asked for, so an SCF-only manifest is unchanged. It is the
            # REQUEST, not the outcome -- the driver has not run yet here.
            if _aiccm_correlation is not None:
                _dry_run_fields["aiccm_correlation"] = _aiccm_correlation
        _dry_manifest = ManifestUpdater(
            real_plan,
            record_hostname=record_hostname,
            wall_seconds=0.0,
            estimate_bytes=_estimate_bytes,
            extra_run_fields=_dry_run_fields,
        )
        _dry_manifest.mark_dry_run()
        print_dry_run_summary(real_plan)
        return None

    from .periodic.exchange_convention import exchange_q0_label

    _initial_run_fields = {
        "exchange_q0": exchange_q0_label(exchange_exxdiv),
        "jk_method_requested": _jk_requested_label,
        "jk_method_resolved": resolved_jk.value,
        "jk_method_executed": _executed_jk_method,
        "dft_plus_u": bool(dft_plus_u),
        "dft_plus_u_route": _dft_plus_u_route,
    }
    if _uses_skala:
        from .skala import _run_provenance_fields

        _initial_run_fields.update(_run_provenance_fields())
        _initial_run_fields["skala_grid_policy"] = "skala"
    if resolved_jk in _AICCM_JK_METHODS:
        # HANDOVER_AICCM_STANDARD_METHOD.md M0: one maturity stamp for every
        # AICCM job, written before compute so a dry-run, a crashed and a
        # completed .system all carry it (the end-of-run block only merges).
        # Vocabulary: the test-gate maturity states. The real-Γ-only
        # ``experimental_route`` key stays for one release (the manifest
        # contract is additive); this is the uniform marker.
        _initial_run_fields["method_status"] = "experimental"
        # M1: mirrored from the dry-run fields above.
        _initial_run_fields["aiccm_variant"] = AICCM_VARIANT_OF[resolved_jk]
        _initial_run_fields["aiccm_selector"] = (
            "front-door" if _aiccm_front_door else "legacy-jk_method"
        )
        # M4b (#778): mirrored from the dry-run fields above.
        if _aiccm_correlation is not None:
            _initial_run_fields["aiccm_correlation"] = _aiccm_correlation
    if resolved_jk == PeriodicJKMethod.GAPW:
        _initial_run_fields["gapw_one_centre_resolved"] = (
            "analytic" if method_upper in ("RHF", "UHF") else "block"
        )
        _initial_run_fields["gapw_molecular_limit_declared"] = bool(
            gapw_molecular_limit
        )
    _output_writer = OutputWriter(
        real_plan,
        record_hostname=record_hostname,
        extra_run_fields=_initial_run_fields,
    )
    _PERIODIC_OUTPUT_WRITER.set(_output_writer)

    plog = resolve_progress(progress, verbose=verbose)
    # Early status line so silent BIPOLE/GDF crashes leave at least one
    # diagnostic line in the .out file before any driver-specific setup.
    plog.info(
        f"Periodic {method_upper}/{resolved_jk.value} job starting  "
        f"(basis: {basis.name}, natoms: {len(system.unit_cell)}, "
        f"dim: {int(system.dim)})"
    )

    # --- Opt-in live QVF checkpointing (vibe-view hot-reload) -----------
    # Build the checkpointer + its QVF-only plan up front. The job's real
    # OutputPlan already owns the final archive; this narrower plan is only a
    # type-gate because checkpoint sections come from per-snapshot context.
    # When enabled, wrap ``plog`` so each SCF cycle
    # that lands a ``plog.iteration(...)`` also refreshes the checkpoint
    # QVF on the configured cadence. This covers the routes that stream
    # per-iteration through the shared logger -- the Ewald, GDF, BIPOLE, and
    # GPW drivers all take ``progress=plog`` and call ``plog.iteration(...)``
    # per SCF cycle. The GPW route (``periodic_gapw_j.run_periodic_rhf_gpw`` /
    # ``run_periodic_rks_gpw_multi_k`` + the open-shell UHF/UKS/multi-k
    # siblings in ``periodic_gapw_open_shell``) now threads the same
    # ``progress=`` handle into its Python SCF loop, so GPW jobs get
    # per-iteration cadence too. Molecular compiled-C++ SCFs now expose a
    # native diagnostics callback, but QVF checkpointing does not consume that
    # logging callback, so molecular jobs remain start + terminal frames only.
    from .output.checkpoint import (
        QvfCheckpointer as _QvfCheckpointer,
        wrap_progress_for_checkpoints as _wrap_progress_for_checkpoints,
    )

    _checkpointer = _QvfCheckpointer(
        checkpoint_qvf if output_qvf else None,
        checkpoint_every,
        plan=OutputPlan.from_run_job_kwargs(
            output=output_stem,
            method=method_upper,
            basis=basis.name,
            functional=functional,
            output_qvf=True,
            job_kind="periodic_scf",
        ),
    )
    if _checkpointer.enabled and checkpoint_every > 0:

        def _periodic_scf_checkpoint(_n: int, _fields: dict) -> None:
            _checkpointer.maybe_snapshot(
                _n,
                energy_eh=_fields.get("energy"),
                system=system,
                method=method_upper,
                basis=basis.name,
                functional=functional,
            )

        plog = _wrap_progress_for_checkpoints(plog, _periodic_scf_checkpoint)

    # --- Auto-read smearing / bz_integration from KPoints metadata -----
    # When kpoints is a KPoints.recommend() result with .smearing /
    # .bz_integration set, and the user did not explicitly pass those
    # args, auto-apply them.  Explicit user args always win (mirrors the
    # periodic_convergence_auto "explicit wins" contract).
    from .kpoints import KPoints as _KPoints

    _kpts_smearing = None
    _kpts_bz = None
    _kpts_uses_ml = False
    # [routes.numerics] keys naming the published k-point constructions this
    # job used, so the references block cites them (CLAUDE.md § 8). The SCF
    # mesh contributes its own key; an attached band structure contributes its
    # path convention. Plain Monkhorst-Pack and explicit lists contribute none.
    _numerics: List[str] = []
    if isinstance(kpoints, _KPoints):
        _kpts_smearing = getattr(kpoints, "smearing", None)
        _kpts_bz = getattr(kpoints, "bz_integration", None)
        _kpts_uses_ml = getattr(kpoints, "uses_ml_predictor", False)
        _numerics.extend(getattr(kpoints, "citation_numerics", ()) or ())
    _numerics.extend(
        getattr(getattr(band_structure, "kpath", None), "citation_numerics", ()) or ()
    )
    # Resolve bz_integration: explicit arg wins, else KPoints metadata.
    if bz_integration is _BZ_INTEGRATION_UNSET:
        bz_integration = _kpts_bz  # None / "smearing" / "gilat"
    # Validate bz_integration early.
    if bz_integration is not None:
        bz_integration = str(bz_integration).strip().lower()
        if bz_integration not in ("smearing", "gilat"):
            raise ValueError(
                "run_periodic_job: bz_integration must be None, 'smearing', "
                f"or 'gilat'; got {bz_integration!r}."
            )
    # GPW / GAPW never thread bz_integration into their occupation logic:
    # the GPW and GAPW drivers take no such argument, so the flag was
    # accepted and silently dropped -- the run used Fermi-Dirac while the
    # user asked for Gilat-Raubenheimer. Measured on H2/STO-3G, 12-bohr
    # box, (2,1,1), RKS/PBE: the energy is BITWISE identical with and
    # without the flag (-1.152074940754 either way), while the .out
    # recorded "smearing_method = fermi-dirac" and "bz_integration =
    # gilat" at once and the .references cited Gilat-Raubenheimer 1966 +
    # Gilat 1972 for numerics the run never performed. That is a
    # citation-integrity break (CLAUDE.md section 8), not just an
    # ergonomics wart. Refuse it here -- BEFORE the citation route is
    # registered below -- so the entry cannot fire for a backend that
    # does not honour the request. GDF, RIJCOSX and BIPOLE do honour it.
    if bz_integration == "gilat" and resolved_jk in (
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
    ):
        raise NotImplementedError(
            "run_periodic_job: bz_integration='gilat' is not implemented "
            f"for {resolved_jk}; the GPW/GAPW drivers do not thread it "
            "into their occupation logic, so honouring the request would "
            "require route-specific Gilat-Raubenheimer occupations. It is "
            "refused rather than silently ignored, because accepting it "
            "would run Fermi-Dirac while recording (and citing) "
            "Gilat-Raubenheimer. Use jk_method='gdf'/'rijcosx'/'bipole' "
            "for Gilat-Raubenheimer, or drop bz_integration on this route."
        )
    if bz_integration == "gilat":
        _numerics.append("gilat_raubenheimer")
    if method_upper in ("ROHF", "ROKS") and bz_integration == "gilat":
        raise NotImplementedError(
            "run_periodic_job: Gilat-Raubenheimer BZ integration is not "
            f"implemented for periodic {method_upper}; it needs "
            "restricted-open-shell fractional per-k occupations."
        )

    # Build the option object expected by the selected native driver.
    opts = (
        PeriodicKSOptions()
        if method_upper in ("ROKS", "RKS", "UKS")
        else PeriodicRHFOptions()
    )
    if method_upper in ("ROKS", "RKS", "UKS"):
        opts.functional = str(functional)
        if _external_xc_grid_profile is not None:
            opts.grid.atomic_grid_profile = _external_xc_grid_profile
    if resolved_jk == PeriodicJKMethod.BIPOLE:
        if sr_range_screening:
            # Separation-aware charge-pair Schwarz
            # screening for the SR erfc J/K build; cited via the
            # bipole_sr_range route below.
            opts.lattice_opts.sr_range_screening = True
        if _bipole_cutoff_bohr is not None:
            opts.lattice_opts.cutoff_bohr = _bipole_cutoff_bohr
        if _bipole_nuclear_cutoff_bohr is not None:
            opts.lattice_opts.nuclear_cutoff_bohr = _bipole_nuclear_cutoff_bohr
        elif opts.lattice_opts.nuclear_cutoff_bohr > opts.lattice_opts.cutoff_bohr:
            # BIPOLE's corrected Ewald gauge relies on neutral-cell
            # cancellation between V_ne, E_nn, and the matching electronic
            # J/K cell set.  The generic LatticeSumOptions default has a
            # longer nuclear cutoff (25 bohr) than electronic cutoff
            # (15 bohr), which overbinds molecular-limit hybrid RKS rows by
            # mHa.  Keep the public BIPOLE default coherent unless the user
            # explicitly requested a separate nuclear cutoff.
            opts.lattice_opts.nuclear_cutoff_bohr = opts.lattice_opts.cutoff_bohr
    # --- Auto-attach ECPs from pob-TZVP-REV2 CRYSTAL ECP blocks -------
    # When the basis carries inline ECP data (detected via Z+200 header
    # in CRYSTAL-format basis files), convert to libecpint inline-primitive
    # blocks and attach to the options object (Phase 14g).
    if _external_route_ecp_data is None:
        _ecp_blocks, _ecp_centers, _eff_z, _total_ncore = _resolve_ecp_data(
            system,
            basis,
        )
    else:
        _ecp_blocks, _ecp_centers, _eff_z, _total_ncore = (
            _external_route_ecp_data
        )
    _ecp_active = (
        len(_ecp_blocks) > 0
        or len(_ecp_centers) > 0
        or int(_total_ncore or 0) != 0
    )
    if _ecp_active and not _periodic_route_applies_ecp(
        resolved_jk, method_upper, system
    ):
        # Fail closed. Routes whose drivers build Hcore from bare nuclear
        # charges and partition the physical electron count would carry
        # the inline ECP fields set on ``opts`` below and silently ignore
        # them (measured 2026-09-05 on Ag fcc / pob-TZVP-rev2 before the
        # k-point GDF drivers consumed them: 188 electrons dispatched into
        # a 19-valence-electron basis). See
        # handovers/HANDOVER_BASIS_ECP_UNIFICATION.md.
        _ecp_atoms = sorted(
            {
                _z
                for _z, _q in zip(_system_atomic_numbers_in_order(system), _eff_z)
                if float(_q) != float(_z)
            }
        )
        raise NotImplementedError(
            f"run_periodic_job: basis {getattr(basis, 'name', basis)!r} "
            f"replaces core electrons on Z={_ecp_atoms} with an ECP, but "
            f"jk_method={resolved_jk.value!r} with method={method_upper} does "
            "not apply a periodic ECP: that driver's one-electron Hamiltonian "
            "uses bare nuclear charges and the physical electron count, so the "
            "run would be an all-electron calculation in a valence-only basis. "
            "Periodic ECPs run on jk_method='gdf' with RHF/RKS/UHF/UKS in 3D "
            "(and on the native direct drivers vibeqc.run_rks_periodic / "
            "run_rhf_periodic_gamma); use an all-electron basis elsewhere."
        )
    if _ecp_blocks:
        opts.ecp_primitive_blocks = _ecp_blocks
        opts.ecp_home_centers = _ecp_centers
        opts.ecp_effective_charges = _eff_z
        opts.ecp_total_ncore = _total_ncore
    # --- BUG 99 guard: refuse ECP-paired molecular bases without ECP ----
    # Molecular ECP-paired bases (dhf-*, *-PP, lanl*, etc.) pass through
    # the periodic runner without the molecular runner's auto-attach or
    # validation.  Guard here so a user who copies an ORCA-style input
    # into a periodic calculation does not get a silently wrong answer.
    _basis_name = str(getattr(basis, "name", "") or "").strip()
    if _basis_name:
        from .ecp_metadata import is_ecp_paired_basis as _is_ecp_paired

        if _is_ecp_paired(_basis_name):
            _has_ecp = bool(
                getattr(opts, "ecp_primitive_blocks", None)
                or getattr(opts, "ecp_centers", None)
            )
            if not _has_ecp:
                raise ValueError(
                    f"basis={_basis_name!r} is designed for use with an "
                    f"effective core potential (ECP), but no ECP centers "
                    f"or primitive blocks were configured on the periodic "
                    f"SCF options.\n"
                    f"Either configure ecp_primitive_blocks / ecp_centers "
                    f"on the options struct, or choose an all-electron "
                    f"basis set (e.g. def2-TZVPP, cc-pVTZ).\n"
                    f"See vibeqc.ecp_metadata.auto_ecp_centers() for "
                    f"automatic ECP-centre construction."
                )
    opts.use_diis = bool(use_diis)
    if dynamic_damping is not None:
        opts.dynamic_damping = bool(dynamic_damping)
    # --- Diagonalisation solver -----------------------------------------
    if solver not in ("dense", "davidson", "lobpcg"):
        raise ValueError(
            f"run_periodic_job: solver='{solver}' is not recognised. "
            f"Supported: 'dense', 'davidson', 'lobpcg'."
        )
    if solver != "dense":
        opts.use_davidson = True
    # --- Smearing: resolve any user-engaged smearing input first ------
    smearing_engaged = (
        smearing is not None
        or smearing_temperature is not _SMEARING_UNSET
        or smearing_metallic is not None
        or smearing_band_gap_hartree is not None
    )
    # Auto-read smearing from KPoints metadata when not explicitly given.
    if not smearing_engaged and _kpts_smearing is not None:
        smearing = _kpts_smearing
        smearing_engaged = True
    _smearing_temperature_arg = (
        0.0 if smearing_temperature is _SMEARING_UNSET else smearing_temperature
    )
    if smearing is not None:
        if _smearing_temperature_arg not in (0.0, None):
            raise ValueError(
                "run_periodic_job: pass either smearing= or "
                "smearing_temperature=, not both"
            )
        if not isinstance(smearing, SmearingOptions):
            raise TypeError(
                "run_periodic_job: smearing must be a SmearingOptions instance"
            )
        if smearing.enabled and smearing.flavor not in (
            "fermi-dirac",
            "mermin",
            "methfessel-paxton",
            "marzari-vanderbilt",
        ):
            raise NotImplementedError(
                "run_periodic_job: smearing flavor "
                f"{smearing.flavor!r} is not implemented"
            )
        smearing_temperature_hartree = float(smearing.temperature)
        smearing_method_label = smearing.flavor
        smearing_source = smearing.source
        smearing_reason = smearing.reason
    else:
        smearing_resolution = resolve_smearing_temperature(
            _smearing_temperature_arg,
            unit=smearing_unit,
            method=smearing_method,
            metallic=smearing_metallic,
            band_gap_hartree=smearing_band_gap_hartree,
            n_electrons=system.n_electrons(),
        )
        smearing_temperature_hartree = float(smearing_resolution.temperature)
        smearing_method_label = smearing_resolution.method
        smearing_source = smearing_resolution.source
        smearing_reason = smearing_resolution.reason

    # --- Automatic convergence strategy (transparency contract) -------
    # Explicit user knobs are never overridden; auto fills only unset
    # knobs, and the .out states the mode, the classification, and the
    # per-knob reasons. v1 applies auto on the BIPOLE and GDF routes --
    # other routes run with mode "off" unless the user set knobs
    # (mode "manual" labels them explicitly there too).
    _auto_supported = resolved_jk in (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.GDF,
    )
    _requested_convergence = convergence
    if (
        convergence is not None
        and str(convergence).strip().lower() == "auto"
        and not _auto_supported
    ):
        _requested_convergence = "off"
    elif convergence is None and not _auto_supported:
        _requested_convergence = "off"
    convergence_strategy = resolve_convergence_strategy(
        system,
        method=method_upper,
        convergence=_requested_convergence,
        # Only the BIPOLE KS drivers carry an in-driver FMIXING-30%
        # default for DFT functionals, and since the 2026-07-13 Gap-B
        # validation only when DIIS is off (under DIIS the mixing is
        # redundant damping that measurably slows convergence -- see
        # pbc_bipole_rks.py). The floor mirrors the driver so the
        # printed strategy matches what actually runs; GDF honours the
        # resolved value verbatim, so no floor there.
        ks_driver_fock_mixing_floor=(
            resolved_jk == PeriodicJKMethod.BIPOLE and not bool(use_diis)
        ),
        explicit={
            "fock_mixing": fock_mixing if fmixing_percent is not None else None,
            "level_shift": (float(level_shift) if level_shift is not None else None),
            "damping": float(damping) if damping is not None else None,
            "smearing_temperature": (
                smearing_temperature_hartree if smearing_engaged else None
            ),
        },
    )
    if _uses_external_xc:
        _external_smearing = convergence_strategy.knobs[
            "smearing_temperature"
        ]
        if float(_external_smearing.value) > 0.0:
            if _external_smearing.source != "auto":
                from .periodic_external_xc import (
                    _require_zero_temperature_external_xc,
                )

                _require_zero_temperature_external_xc(
                    float(_external_smearing.value),
                    where="run_periodic_job",
                )
            _external_knobs = dict(convergence_strategy.knobs)
            _external_knobs["smearing_temperature"] = KnobResolution(
                0.0,
                "auto",
                "capability-filtered to zero: periodic full-grid external "
                "XC is zero-temperature only until finite-iteration "
                "Mermin state coherence is implemented",
            )
            convergence_strategy = ConvergenceStrategy(
                mode=convergence_strategy.mode,
                classification=convergence_strategy.classification,
                knobs=_external_knobs,
            )
    _gdf_capability_knobs = dict(convergence_strategy.knobs)
    _gdf_capability_changed = False
    if resolved_jk == PeriodicJKMethod.GDF:
        # Mirror of the _gamma_default_pure_gdf_ok domain gate in the Γ GDF
        # dispatch below, minus the knob values this filter itself resolves.
        # When the default-Γ closed-shell RHF run qualifies for the
        # PySCF-µHa-validated run_pbc_gdf_rhf in every non-knob respect, an
        # AUTO-resolved fock-mixing / level-shift (e.g. the ionic-insulator
        # profile's FMIXING 30% on MgO-class cells) must not force the run
        # onto the legacy molecular-limit fallback, whose dense-core absolute
        # energies are PARITY_HELD (G-GDF-001 in HANDOVER_GATED_ITEMS.md).
        # Zero the auto knobs instead -- the parity route carries its own
        # convergence handling (DIIS + accelerators + the auto-sized
        # high-|G| tail). Explicit user knobs keep the legacy fallback.
        _gamma_pure_gdf_capable = (
            method_upper == "RHF"
            and kpoints is None
            and gdf_method is None
            and int(system.dim) == 3
            and int(system.n_electrons()) % 2 == 0
            and int(system.multiplicity) == 1
            and abs(
                float(sum(atom.Z for atom in system.unit_cell))
                - int(system.n_electrons())
            )
            <= 0.5
            and float(convergence_strategy.knobs["smearing_temperature"].value)
            <= 0.0
            and not symmetry_stabilize
            and not symmetry_reduce_fock
        )
        _level_shift_resolution = convergence_strategy.knobs["level_shift"]
        _gdf_level_shift_filter_reason = None
        if (
            _level_shift_resolution.source == "auto"
            and _level_shift_resolution.value != 0.0
        ):
            if method_upper in ("ROHF", "UHF", "UKS"):
                _gdf_level_shift_filter_reason = (
                    "capability-filtered to zero: open-shell GDF drivers do "
                    "not implement the level-shift operator"
                )
            elif _requested_gamma_only and gdf_method is not None:
                _gdf_level_shift_filter_reason = (
                    "capability-filtered to zero: the exact-Gamma GDF driver "
                    "selected by an explicit gdf_method does not implement "
                    "the level-shift operator"
                )
            elif _gamma_pure_gdf_capable:
                _gdf_level_shift_filter_reason = (
                    "capability-filtered to zero: the default-Gamma "
                    "closed-shell RHF run stays on the PySCF-parity "
                    "run_pbc_gdf_rhf driver, which does not implement the "
                    "level-shift operator (the legacy fallback's dense-core "
                    "absolute energies are parity-held)"
                )
        if _gdf_level_shift_filter_reason is not None:
            _gdf_capability_knobs["level_shift"] = KnobResolution(
                0.0,
                "auto",
                _gdf_level_shift_filter_reason,
            )
            _gdf_capability_changed = True

        _fock_mixing_resolution = convergence_strategy.knobs["fock_mixing"]
        _gdf_fock_mixing_filter_reason = None
        if (
            _fock_mixing_resolution.source == "auto"
            and _fock_mixing_resolution.value != 0.0
        ):
            if method_upper == "ROHF" or (
                method_upper in ("UHF", "UKS") and kpoints is not None
            ):
                _gdf_fock_mixing_filter_reason = (
                    "capability-filtered to zero: the selected open-shell "
                    "GDF driver does not implement Fock mixing"
                )
            elif (
                method_upper == "RHF"
                and _requested_gamma_only
                and gdf_method is not None
            ):
                _gdf_fock_mixing_filter_reason = (
                    "capability-filtered to zero: the closed-shell "
                    "exact-Gamma GDF driver selected by an explicit "
                    "gdf_method does not implement Fock mixing"
                )
            elif _gamma_pure_gdf_capable:
                _gdf_fock_mixing_filter_reason = (
                    "capability-filtered to zero: the default-Gamma "
                    "closed-shell RHF run stays on the PySCF-parity "
                    "run_pbc_gdf_rhf driver, which does not implement Fock "
                    "mixing (the legacy fallback's dense-core absolute "
                    "energies are parity-held)"
                )
        if _gdf_fock_mixing_filter_reason is not None:
            _gdf_capability_knobs["fock_mixing"] = KnobResolution(
                0.0,
                "auto",
                _gdf_fock_mixing_filter_reason,
            )
            _gdf_capability_changed = True
    if _gdf_capability_changed:
        # Auto selection must respect the capabilities of the chosen driver.
        # Leave unset/default requests at zero and state why in the strategy
        # block. Explicit unsupported requests remain fail-closed errors below.
        convergence_strategy = ConvergenceStrategy(
            mode=convergence_strategy.mode,
            classification=convergence_strategy.classification,
            knobs=_gdf_capability_knobs,
        )
    if resolved_jk == PeriodicJKMethod.BIPOLE:
        convergence_strategy = _filter_bipole_restricted_open_convergence(
            convergence_strategy,
            method_upper,
        )
    _conv_auto_note = None
    if (
        convergence is not None
        and str(convergence).strip().lower() == "auto"
        and not _auto_supported
    ):
        _conv_auto_note = (
            'convergence="auto" is wired for jk_method="bipole" and '
            'jk_method="gdf" only in this version; using plain defaults '
            "for this route"
        )

    _density_mixer_auto_note = None
    _density_mixer_params_default = (
        int(density_mixer_depth) == 8
        and float(density_mixer_beta) == 0.5
        and not bool(density_mixer_kerker)
        and float(kerker_k0) == 1.5
        and float(kerker_strength) == 1.0
        and float(kerker_cutoff_ha) == 120.0
    )
    _convergence_off_requested = (
        convergence is not None
        and str(convergence).strip().lower() in ("off", "none")
    )
    if (
        density_mixer is None
        and _density_mixer_params_default
        and not _convergence_off_requested
        and resolved_jk == PeriodicJKMethod.GDF
        and method_upper == "RKS"
        and kpoints is not None
        and _functional_is_scan_family(functional)
        and int(system.dim) == 3
        and int(system.multiplicity) == 1
        and int(system.n_electrons()) % 2 == 0
    ):
        _mgga_cls = classify_periodic_system(system)
        if _mgga_cls.profile in _COMPACT_MGGA_DENSITY_MIXER_PROFILES:
            density_mixer = "anderson"
            density_mixer_beta = 0.35
            _density_mixer_auto_note = (
                "compact periodic SCAN/r2SCAN RKS/GDF profile "
                f"'{_mgga_cls.profile}' uses density_mixer='anderson' "
                "with beta=0.35 by default; the GDF driver disables "
                "Fock-DIIS and FMIXING while density mixing is active. "
                "Pass density_mixer='diis' or convergence='off' to keep "
                "the Fock-DIIS route"
            )

    opts.damping = convergence_strategy.value("damping")
    opts.fock_mixing = convergence_strategy.value("fock_mixing")
    if resolved_jk in (
        PeriodicJKMethod.BIPOLE,
        PeriodicJKMethod.GDF,
        # opts reaches the multi-k GDF drivers through the producer, so an
        # explicit level shift is honoured rather than dropped.
        PeriodicJKMethod.NEUTRAL_BLOCH,
        PeriodicJKMethod.AICCM2026DEV_B,
    ):
        opts.level_shift = convergence_strategy.value("level_shift")
    _smear_res = convergence_strategy.knobs["smearing_temperature"]
    if not smearing_engaged and _smear_res.source == "auto" and _smear_res.value > 0.0:
        # Auto strategy turned smearing on: surface it through the same
        # .out lines the explicit smearing path uses.
        smearing_method_label = "fermi-dirac"
        smearing_source = "auto-strategy"
        smearing_reason = _smear_res.reason
    opts.smearing_temperature = convergence_strategy.value("smearing_temperature")
    _insulator_smearing_note = insulator_smearing_warning(
        system,
        opts.smearing_temperature,
        band_gap_hartree=smearing_band_gap_hartree,
        metallic=smearing_metallic,
    )
    if _insulator_smearing_note is not None:
        warnings.warn(_insulator_smearing_note, UserWarning, stacklevel=2)
    _validate_smearing_dispatch(
        method=method_upper,
        jk_method=resolved_jk,
        smearing_temperature=opts.smearing_temperature,
        kpoints=kpoints,
    )
    if (
        resolved_jk == PeriodicJKMethod.BIPOLE
        and float(opts.smearing_temperature) > 0.0
        and str(smearing_method_label).strip().lower().replace("_", "-")
        not in ("fermi-dirac", "mermin")
    ):
        raise NotImplementedError(
            "run_periodic_job: BIPOLE currently implements only "
            "Fermi-Dirac occupations (including the Mermin free-energy "
            "label). Methfessel-Paxton and Marzari-Vanderbilt requests "
            "would otherwise execute Fermi-Dirac while reporting a different "
            "method, so they fail before SCF."
        )
    _validate_density_mixer_dispatch(
        density_mixer=density_mixer,
        density_mixer_depth=density_mixer_depth,
        density_mixer_beta=density_mixer_beta,
        density_mixer_kerker=density_mixer_kerker,
        kerker_k0=kerker_k0,
        kerker_strength=kerker_strength,
        kerker_cutoff_ha=kerker_cutoff_ha,
        jk_method=resolved_jk,
        method=method_upper,
        kpoints=kpoints,
    )
    opts.diis_start_iter = int(diis_start_iter)
    opts.diis_subspace_size = int(diis_subspace_size)
    opts.max_iter = int(max_iter)
    opts.conv_tol_energy = float(conv_tol_energy)
    opts.initial_guess = _resolved_initial_guess

    if atomic_spins is not None and opts.initial_guess not in (
        InitialGuess.SAD,
        InitialGuess.AUTO,
    ):
        raise ValueError(
            "run_periodic_job: atomic_spins is a broken-symmetry SAD seed "
            "and can only be combined with initial_guess='SAD' or 'AUTO'; "
            f"got initial_guess={opts.initial_guess.name!r}."
        )

    if opts.initial_guess == InitialGuess.SAP:
        if int(system.dim) != 3:
            raise NotImplementedError(
                "run_periodic_job: initial_guess='SAP' currently requires "
                f"a 3-D periodic system; got dim={int(system.dim)}. The "
                "lattice SAP potential uses the 3-D Ewald split, so a 1-D "
                "or 2-D request cannot be substituted safely."
            )
    if resolved_jk == PeriodicJKMethod.GDF and method_upper == "ROHF":
        from .periodic_rohf_gdf import ROHF_GDF_GUESSES

        resolve_initial_guess(
            system.unit_cell_molecule(), _requested_initial_guess,
            is_periodic=True, is_open_shell=True, supported=ROHF_GDF_GUESSES,
        )

    # ATOMSPIN broken-symmetry seed (UHF/UKS only; the closed-shell misuse is
    # rejected above). Carried on the options struct so the open-shell
    # driver's guess step assembles the per-atom broken-symmetry density. The
    # GuessEngine validates the per-atom tag count against the SCF cell (so it
    # stays correct under cell reduction).
    if atomic_spins is not None:
        opts.atomic_spins = [int(s) for s in atomic_spins]

    # SPINLOCK (UHF/UKS only; resolved + validated above). Carried on the
    # options struct so the open-shell driver runs the two-phase schedule
    # (SPIN_SCHEDULE) or MOM-holds the seeded pattern (PATTERN_HOLD).
    if _spinlock_mode != SpinlockMode.OFF:
        opts.spinlock_mode = _spinlock_mode
        opts.spinlock_value = int(spinlock_value)
        opts.spinlock_iterations = int(spinlock_iterations)

    # READ restart. Gamma restarts resolve the prior g=0 cell density and put
    # it on the options struct. Closed-shell multi-k restarts keep the native
    # per-k density list separate so the selected multi-k driver can inject
    # D(k) directly instead of degrading to a Gamma density.
    _read_density_k_closed = None
    _read_density_k_open = None
    if _fragmo_request:
        from .guess_fragmo import resolve_periodic_fragmo_source
        read_from = resolve_periodic_fragmo_source(
            opts, system, basis, _runner_bloch_kmesh(system, kpoints), fragments)
    if opts.initial_guess == InitialGuess.READ and restart_from is None:
        _read_path = opts.read_path
        _read_obj = read_from
        if isinstance(read_from, (str, os.PathLike)):
            _read_path = os.fspath(read_from)
            _read_obj = None
        opts.read_path = _read_path

        if _read_multik_request and method_upper in ("UHF", "UKS", "ROHF", "ROKS"):
            from .guess_read import resolve_periodic_read_densities_k_open
            _read_density_k_open = resolve_periodic_read_densities_k_open(
                read_from=_read_obj, read_path=_read_path,
                basis=basis, system=system, kmesh=_runner_bloch_kmesh(system, kpoints),
            )
        elif _read_multik_request:
            from .guess_read import resolve_periodic_read_density_k_closed

            _read_density_k_closed = resolve_periodic_read_density_k_closed(
                read_from=_read_obj,
                read_path=_read_path,
                expected_n_k=_bloch_kmesh_full_size(
                    _runner_bloch_kmesh(system, kpoints)
                ),
                n_basis=basis.nbasis,
                basis=basis, system=system,
                kmesh=_runner_bloch_kmesh(system, kpoints),
            )
        else:
            from .guess_read import (
                resolve_periodic_read_densities_open,
                resolve_periodic_read_density_closed,
            )

            if method_upper in ("ROHF", "ROKS", "UHF", "UKS"):
                _da, _db = resolve_periodic_read_densities_open(
                    basis, read_path=_read_path, read_from=_read_obj
                )
                opts.read_density_alpha = _da
                opts.read_density_beta = _db
                _read_density_k_open = ([_da], [_db])
            else:
                opts.read_density = resolve_periodic_read_density_closed(
                    basis, read_path=_read_path, read_from=_read_obj
                )
                _read_density_k_closed = [opts.read_density]

    label = f"{method_upper}"
    if functional:
        label = f"{label} / {functional}"

    _periodic_memory = None
    try:
        if resolved_jk in (PeriodicJKMethod.GPW, PeriodicJKMethod.GAPW):
            _gpw_estimate = _periodic_gpw_gapw_estimate(
                system,
                basis,
                resolved_jk=resolved_jk,
                method_upper=method_upper,
                functional=functional,
                cutoff_ha=cutoff_ha,
                kpoints=kpoints,
            )
            if _gpw_estimate is not None:
                _periodic_memory = SimpleNamespace(
                    estimate=_gpw_estimate,
                    kind="generic",
                )
        elif resolved_jk in (PeriodicJKMethod.GDF, PeriodicJKMethod.RIJCOSX):
            _gdf_estimate = _periodic_gdf_estimate(
                system,
                basis,
                resolved_jk=resolved_jk,
                method_upper=method_upper,
                functional=functional,
                kpoints=kpoints,
                aux_basis=aux_basis,
                rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
                n_ibz_kpoints=_symmetry_reduce_k_ibz,
                diis_subspace_size=int(opts.diis_subspace_size) if opts.use_diis else 0,
                scf_options=opts, compute_gradient=bool(optimize),
            )
            if _gdf_estimate is not None:
                _gdf_estimate.kind = "gdf"
                _periodic_memory = _gdf_estimate
    except Exception:
        _periodic_memory = None

    if _uses_skala:
        _skala_estimate = _merge_periodic_skala_memory(
            None if _periodic_memory is None else _periodic_memory.estimate,
            system,
        )
        if _periodic_memory is None:
            _periodic_memory = SimpleNamespace(
                estimate=_skala_estimate,
                kind="generic",
            )

    t_job_start = time.perf_counter()

    hessian_result = None  # populated by hessian=True; consumed by QVF writer

    # Install the C++ → Python diagnostics bridge before any
    # computation, so VIBEQC_DIAG / vibeqc::diagnostic() calls in
    # the C++ core are forwarded into the .out.
    install_diagnostics_bridge()
    # Route C++ progress diagnostics into the .system manifest.
    install_progress_handler(
        lambda fields: _output_writer.update_progress(**fields)
    )

    # Auto-enable structured log when running under vq.
    _structured_target: Path | None = None
    if os.environ.get("VQ_WORKDIR"):
        _structured_target = stem_sibling(output_stem, ".scf.jsonl")

    with (
        _structured_log_ctx(_structured_target) as _slog,
        OutputChannel.to_file(out_path),
    ):
        # Structured log: banner + job_start records.
        _libs = library_versions()
        _fp = run_fingerprint(
            method=method_upper,
            basis=basis.name,
            functional=functional,
            molecule=system.unit_cell_molecule(),
        )
        _slog.emit(
            "banner",
            vibeqc_version=VIBEQC_VERSION,
            libint=_libs.get("libint", "unknown"),
            libxc=_libs.get("libxc", "unknown"),
            spglib=_libs.get("spglib", "unknown"),
            run_fingerprint=_fp,
        )
        _slog.emit(
            "job_start",
            method=method_upper,
            basis=basis.name,
            functional=functional if resolved_jk == PeriodicJKMethod.GPW else None,
            optimize=False,
            threads=int(get_num_threads()),
            n_atoms=int(len(system.unit_cell)),
            charge=int(system.charge),
            multiplicity=int(system.multiplicity),
            n_electrons=int(system.n_electrons()),
            output_stem=str(output_stem),
            periodic=True,
            jk_method=str(resolved_jk.name),
        )

        # --- Banner ---------------------------------------------------
        write(banner() + "\n\n")
        libs = library_versions()
        write(f"  Job: PERIODIC {label}  basis={basis.name}\n")
        write(f"  J/K method: {describe_jk_method(resolved_jk)}\n")
        if _aiccm_front_door:
            _ref_origin = "inferred" if scf_reference is None else "explicit"
            write(
                f"    (selected by method='aiccm', variant={_aiccm_variant!r}; "
                f"SCF reference {method_upper} {_ref_origin})\n"
            )
        elif resolved_jk in _AICCM_JK_METHODS:
            # Legacy spelling: DeprecationWarning is hidden by the default
            # Python filters outside __main__, so the .out carries the
            # pointer where the user will actually read it.
            write(
                f"    (user-requested: {_jk_requested_label!r}; deprecated "
                "spelling of method='aiccm', "
                f"variant={AICCM_VARIANT_OF[resolved_jk]!r})\n"
            )
        elif (
            _jk_method_input != "auto"
            and _jk_method_input != PeriodicJKMethod.AUTO
        ):
            write(f"    (user-requested: {_jk_method_input!r})\n")
        else:
            write(f"    (resolved from AUTO)\n")
        write("\n")
        write(_system_summary(system, ecp_total_ncore=_total_ncore))
        if system_original_info is not None:
            write(system_original_info)
        elif system.symmetry is not None:
            sg = system.symmetry
            write(
                f"  Symmetry: {sg.international_symbol} (No. {sg.number}), "
                f"point group {sg.point_group}, order {sg.order}\n\n"
            )
        write(_basis_summary(basis))
        write(section_header("SCF options", width=56))
        write(f"    use_diis            = {opts.use_diis}\n")
        write(f"    damping             = {opts.damping}\n")
        if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
            write(f"    dynamic_damping     = {opts.dynamic_damping}\n")
        if fmixing_percent is not None:
            write(f"    fmixing_percent     = {float(fmixing_percent)}\n")
        if opts.smearing_temperature > 0.0 or smearing_source != "explicit":
            write(f"    smearing_method      = {smearing_method_label}\n")
            write(f"    smearing_source      = {smearing_source}\n")
            if smearing_reason:
                write(f"    smearing_reason      = {smearing_reason}\n")
            write(f"    smearing_temperature = {opts.smearing_temperature}\n")
            if opts.smearing_temperature > 0.0:
                write(
                    "    smearing_temperature_K = "
                    f"{hartree_to_kelvin_temperature(opts.smearing_temperature)}\n"
                )
        if bz_integration is not None:
            write(f"    bz_integration      = {bz_integration}\n")
        write(f"    diis_start_iter     = {opts.diis_start_iter}\n")
        write(f"    diis_subspace_size  = {opts.diis_subspace_size}\n")
        write(f"    max_iter            = {opts.max_iter}\n")
        write(f"    conv_tol_energy     = {opts.conv_tol_energy}\n")
        if restart_from is not None:
            _effective_guess_label = "RESTART"
        elif _fragmo_request:
            _effective_guess_label = "FRAGMO"
        elif _requested_initial_guess == InitialGuess.AUTO:
            _effective_guess_label = (
                f"AUTO -> {_resolved_initial_guess.name}"
            )
        else:
            _effective_guess_label = _resolved_initial_guess.name
        write(f"    initial_guess       = {_effective_guess_label}\n")
        if opts.fock_mixing != 0.0 and fmixing_percent is None:
            write(f"    fock_mixing         = {opts.fock_mixing}\n")
        if resolved_jk == PeriodicJKMethod.BIPOLE:
            write(
                "    bipole_cutoff_bohr  = "
                f"{float(opts.lattice_opts.cutoff_bohr)}\n"
            )
            write(
                "    bipole_nuclear_cutoff_bohr = "
                f"{float(opts.lattice_opts.nuclear_cutoff_bohr)}\n"
            )
        if resolved_jk == PeriodicJKMethod.BIPOLE and (
            convergence_strategy.value("level_shift") != 0.0
        ):
            write(
                "    level_shift         = "
                f"{convergence_strategy.value('level_shift')}\n"
            )
        if resolved_jk in (
            PeriodicJKMethod.GDF,
            PeriodicJKMethod.NEUTRAL_BLOCH,
        ) or (
            resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
            and aiccm_backend.strip().lower().replace("-", "_") != "four_center"
        ):
            write(f"    aux_basis           = {aux_basis or '<auto>'}\n")
            write(f"    gdf_method          = {gdf_method or 'rsgdf'}\n")
            write(f"    rsgdf_ke_cutoff     = {float(rsgdf_ke_cutoff)}\n")
            if rsgdf_tail_ke_cutoff is not None:
                write(
                    "    rsgdf_tail_ke_cutoff = "
                    f"{float(rsgdf_tail_ke_cutoff)}\n"
                )
            write(f"    mdf_ke_cutoff       = {float(mdf_ke_cutoff)}\n")
            # State the space-group reduction the run is actually using:
            # the exchange bra count, the wedge it came from, and the
            # resulting cderi-pair saving. Without this the [memory] line
            # below reports a number the reader cannot account for.
            if symmetry_reduce_k and _symmetry_reduce_k_ibz is not None:
                _n_k_full = int(
                    getattr(
                        _periodic_memory,
                        "n_kpoints",
                        _requested_kmesh_size,
                    )
                )
                _n_ibz = int(_symmetry_reduce_k_ibz)
                write(
                    "    symmetry_reduce_k   = exchange bras "
                    f"{_n_k_full} -> {_n_ibz} (irreducible wedge)\n"
                )
                write(
                    "                          Lpq cderi pairs "
                    f"{_n_k_full * _n_k_full} -> {_n_ibz * _n_k_full} "
                    f"({_n_k_full / max(_n_ibz, 1):.1f}x fewer)\n"
                )
                write(
                    "                          exact: K is "
                    "symmetry-transported over each star\n"
                )
            write(
                "    OpenMP threads      = "
                f"{get_num_threads()}  (max for native parallel regions)\n"
            )
            write("\n")
        # --- Convergence-strategy transparency block -------------------
        # States whether the convergence aids were chosen automatically
        # (by default or by convergence="auto"), set manually, or left
        # plain -- with the classification evidence for auto choices.
        write(section_header("Convergence strategy", width=56))
        for _line in convergence_strategy.log_lines():
            write(f"    {_line}\n")
        if _conv_auto_note:
            write(f"    note: {_conv_auto_note}\n")
        if _density_mixer_auto_note:
            write(f"    note: {_density_mixer_auto_note}\n")
        if _insulator_smearing_note:
            # A real warning: normalised to the canonical "  WARNING: ..."
            # surface (survives --quiet, fires a structured event) rather
            # than the 4-space in-block echo it used to be.
            warn(_insulator_smearing_note, role="insulator_smearing")
        write("\n")
        plog.info(
            "convergence strategy: "
            f"{convergence_strategy.mode}"
            + (
                f" ({convergence_strategy.classification.profile})"
                if convergence_strategy.classification is not None
                else ""
            )
        )
        if _density_mixer_auto_note:
            plog.info(_density_mixer_auto_note)
        if _insulator_smearing_note:
            plog.warn(_insulator_smearing_note)

        if _periodic_memory is not None:
            from .memory import (
                check_memory,
                check_periodic_gdf_memory,
                format_memory_report,
            )

            _estimate = _periodic_memory.estimate
            _gdf_override = bool(os.environ.get("VIBEQC_GDF_MEMORY_OVERRIDE"))
            _override_requested = memory_override or (
                getattr(_periodic_memory, "kind", "") == "gdf" and _gdf_override
            )
            write(
                "  "
                + format_memory_report(
                    _estimate,
                    override_requested=_override_requested,
                ).replace("\n", "\n  ")
                + "\n\n"
            )
            flush()
            if getattr(_periodic_memory, "kind", "") == "gdf":
                check_periodic_gdf_memory(
                    _estimate,
                    n_kpoints=int(_periodic_memory.n_kpoints),
                    route_label=str(_periodic_memory.route_label),
                    allow_exceed=_override_requested,
                )
            else:
                check_memory(_estimate, allow_exceed=memory_override)

        # --- Restart from previous GPW/GAPW calculation ---------------
        restart_density = None
        if restart_from is not None:
            from .periodic_gapw_restart import load_gpw_result

            restart_path = Path(os.fspath(restart_from))
            if not restart_path.exists():
                raise FileNotFoundError(
                    f"restart_from={restart_from!r}: file not found"
                )
            data = load_gpw_result(str(restart_path))
            kind = data.get("kind", "")
            if kind not in ("gpw_scf", "gpw_multi_k_scf"):
                raise ValueError(
                    f"restart_from={restart_from!r}: unsupported kind "
                    f"{kind!r} (expected 'gpw_scf' or 'gpw_multi_k_scf')"
                )
            restart_density = np.asarray(data["density"], dtype=float)
            plog.info(
                f"Restart from {restart_path} ({kind}, "
                f"E = {render_energy_labeled(float(data.get('energy', 0)), width=0, precision=6)})"
            )
            write(f"    restart_from        = {restart_from}\n")
            write(
                f"    restart_energy      = {render_energy_labeled(float(data.get('energy', 0)), width=20, precision=10)}\n"
            )

        # GPW/GAPW consume a caller-owned Gamma density directly.  Both the
        # dedicated ``restart_from`` archive and an explicit READ/read_from
        # request therefore take precedence over construction by the selected
        # guess. Multi-k READ has its own per-k payload below.
        _gamma_route_initial_density = restart_density
        if (
            _gamma_route_initial_density is None
            and opts.initial_guess == InitialGuess.READ
            and not _read_multik_request
        ):
            if method_upper in ("ROHF", "ROKS", "UHF", "UKS"):
                _gamma_route_initial_density = (
                    np.asarray(opts.read_density_alpha),
                    np.asarray(opts.read_density_beta),
                )
            else:
                _gamma_route_initial_density = np.asarray(opts.read_density)
        _gamma_route_initial_guess = opts.initial_guess


        # --- SCF (dispatch on resolved jk_method) --------------------
        t0 = time.perf_counter()

        plog.banner(f"run_periodic_job  PERIODIC {label}  basis={basis.name}")
        plog.info(f"Output file: {out_path}")

        # Initial checkpoint frame: the input geometry, so a live viewer
        # has a structure to show the instant the SCF starts. Per-cycle
        # cadence frames follow via the wrapped ``plog`` (if enabled).
        if _checkpointer.enabled:
            _checkpointer.snapshot(
                system=system,
                method=method_upper,
                basis=basis.name,
                functional=functional,
            )

        def _record_dft_plus_u_route(route: str, executed_jk: str) -> None:
            nonlocal _dft_plus_u_route, _executed_jk_method
            _dft_plus_u_route = route
            _executed_jk_method = executed_jk
            write(f"    dft_plus_u_route    = {route}\n")

        # --- GDF optimizer objective capture (G-PBC-002) --------------
        # The GDF dispatch below records the exact driver + arguments the
        # SCF ran with, so optimize=True can relax on the identical GDF
        # objective (the same driver re-run with compute_gradient=True at
        # each candidate geometry) instead of silently switching to the
        # BIPOLE force surface. GDF branches without an analytic-gradient
        # capable driver (the legacy Γ molecular-limit fallback) leave
        # this empty and the optimize block fails closed.
        _gdf_opt_capture: dict = {}

        def _gdf_scf(_driver, *_args, **_kwargs):
            """Run a dispatched GDF driver and capture the exact call.

            The capture (driver + positional tail + keyword arguments) is
            what guarantees SCF-objective identity for the optimizer:
            the relaxation objective is byte-for-byte the same call, with
            only ``progress``/``compute_gradient`` overridden.
            """
            if (
                (_requested_true_multik or _ecp_active) and int(system.dim) == 3
                and (write_density or output_qvf)
            ):
                _kwargs["return_lattice_density"] = True
            _gdf_opt_capture["driver"] = _driver
            _gdf_opt_capture["args"] = _args
            _gdf_opt_capture["kwargs"] = _kwargs
            return _driver(system, basis, *_args, progress=plog, **_kwargs)

        # --- DFT+U interception (Increment 4d) ----------------------
        # When dft_plus_u is set, route to the appropriate +U-capable
        # driver. Closed-shell true multi-k GDF/RIJCOSX and GPW/GAPW are
        # native below. AUTO single-k jobs are resolved to BIPOLE before
        # dry-run, and explicit unsupported combinations fail in preflight.
        # Reaching the historical fallback below is therefore an internal
        # planning error, never permission to change Coulomb Hamiltonians.
        if (
            dft_plus_u
            and resolved_jk not in _dftu_native_routes
            and not _dftu_gdf_native_route
            and not _dftu_rijcosx_native_route
        ):
            raise RuntimeError(
                "internal DFT+U route-planning invariant violated: the "
                "requested backend reached dispatch without a native +U "
                "implementation or a preflight rejection"
            )
            from . import HubbardSite as _HubbardSite  # noqa: F401
            from . import (
                run_rhf_periodic_gamma as _run_rhf_periodic_gamma,
            )
            from . import (
                run_rks_periodic as _run_rks_periodic,
            )

            if method_upper == "UKS":
                # Open-shell UKS +U via the BIPOLE driver (Increment
                # 4d-bipole UKS). Same per-spin pattern as UHF, plus
                # the UKS XC contribution.
                from .pbc_bipole_uks import run_pbc_bipole_uks

                kmesh = _runner_bloch_kmesh(system, kpoints)
                _record_dft_plus_u_route(
                    "legacy_auto_bipole_uks_"
                    + ("multi_k" if kpoints is not None else "gamma"),
                    "bipole",
                )
                result = run_pbc_bipole_uks(
                    system,
                    basis,
                    kmesh,
                    opts,
                    functional=functional,
                    linear_dep_threshold=1e-7,
                    use_ewald_j_split=True,
                    ewald_omega=ewald_omega,
                    ewald_precision=ewald_precision,
                    use_oda=use_oda,
                    oda_trust_lambda_max=oda_trust_lambda_max,
                    use_mom=use_mom,
                    use_multipole_far_field=use_multipole_far_field,
                    multipole_l_max=multipole_l_max,
                    use_exchange_ewald_split=use_exchange_ewald_split,
                    exchange_exxdiv=exchange_exxdiv,
                    use_fock_symmetry=symmetry_stabilize,
                    use_fock_symmetry_reduce=symmetry_reduce_fock,
                    sr_image_precision=sr_image_precision,
                    exact_zone_bohr=_bipole_exact_zone_bohr,
                    progress=plog,
                    dft_plus_u=dft_plus_u,
                    bz_integration=bz_integration,
                             initial_density_k=_read_density_k_open,
                )
            elif method_upper == "UHF":
                # Open-shell UHF +U via the BIPOLE driver (Increment
                # 4d-bipole). Route directly to run_pbc_bipole_uhf
                # with the user-supplied kmesh.
                from .pbc_bipole_uhf import run_pbc_bipole_uhf

                kmesh = _runner_bloch_kmesh(system, kpoints)
                _record_dft_plus_u_route(
                    "legacy_auto_bipole_uhf_"
                    + ("multi_k" if kpoints is not None else "gamma"),
                    "bipole",
                )
                result = run_pbc_bipole_uhf(
                    system,
                    basis,
                    kmesh,
                    opts,
                    linear_dep_threshold=1e-7,
                    use_ewald_j_split=True,
                    ewald_omega=ewald_omega,
                    ewald_precision=ewald_precision,
                    use_oda=use_oda,
                    oda_trust_lambda_max=oda_trust_lambda_max,
                    use_mom=use_mom,
                    use_multipole_far_field=use_multipole_far_field,
                    multipole_l_max=multipole_l_max,
                    use_exchange_ewald_split=use_exchange_ewald_split,
                    exchange_exxdiv=exchange_exxdiv,
                    use_fock_symmetry=symmetry_stabilize,
                    use_fock_symmetry_reduce=symmetry_reduce_fock,
                    sr_image_precision=sr_image_precision,
                    exact_zone_bohr=_bipole_exact_zone_bohr,
                    progress=plog,
                    dft_plus_u=dft_plus_u,
                    bz_integration=bz_integration,
                             initial_density_k=_read_density_k_open,
                )
            elif method_upper == "RHF":
                if kpoints is not None and tuple(kpoints) != (1, 1, 1):
                    raise NotImplementedError(
                        f"DFT+U on multi-k periodic RHF (kpoints="
                        f"{kpoints!r}) is not yet wired. Increment 4c "
                        "covers the closed-shell-DFT (RKS) multi-k "
                        "path through cpp/src/periodic_scf.cpp; for "
                        "RHF you can run Γ-only today via "
                        "kpoints=None or kpoints=(1,1,1)."
                    )
                _record_dft_plus_u_route("legacy_auto_direct_rhf_gamma", "direct")
                # PeriodicRHFOptions field-by-field copy of the salient
                # convergence knobs; the +U-via-DIRECT path uses its
                # own Coulomb backend so jk_method-specific options
                # don't apply.
                rhf_opts = PeriodicRHFOptions()
                rhf_opts.max_iter = opts.max_iter
                rhf_opts.conv_tol_energy = opts.conv_tol_energy
                rhf_opts.conv_tol_grad = float(getattr(opts, "conv_tol_grad", 1e-6))
                rhf_opts.damping = float(getattr(opts, "damping", 0.5))
                rhf_opts.use_diis = opts.use_diis
                rhf_opts.diis_start_iter = opts.diis_start_iter
                rhf_opts.diis_subspace_size = opts.diis_subspace_size
                rhf_opts.lattice_opts = opts.lattice_opts
                rhf_opts.use_davidson = getattr(opts, "use_davidson", False)
                result = _run_rhf_periodic_gamma(
                    system,
                    basis,
                    rhf_opts,
                    dft_plus_u=dft_plus_u,
                )
            else:
                # RKS via the multi-k DIRECT_TRUNCATED driver. Now
                # supports arbitrary kmesh (Increment 4c) -- uses
                # the k-averaged AO occupation matrix and adds
                # S(k) V_AO S(k) per k.
                from .kpoints import KPoints

                _record_dft_plus_u_route(
                    "legacy_auto_direct_rks_"
                    + ("gamma" if kpoints is None else "multi_k"),
                    "direct",
                )
                ks_opts = PeriodicKSOptions()
                ks_opts.functional = functional or "lda"
                ks_opts.max_iter = opts.max_iter
                ks_opts.conv_tol_energy = opts.conv_tol_energy
                ks_opts.conv_tol_grad = float(getattr(opts, "conv_tol_grad", 1e-6))
                ks_opts.damping = float(getattr(opts, "damping", 0.5))
                ks_opts.use_diis = opts.use_diis
                ks_opts.diis_start_iter = opts.diis_start_iter
                ks_opts.diis_subspace_size = opts.diis_subspace_size
                ks_opts.lattice_opts = opts.lattice_opts
                ks_opts.use_davidson = getattr(opts, "use_davidson", False)
                if kpoints is None:
                    kmesh = KPoints.monkhorst_pack(system, (1, 1, 1))
                else:
                    kp = (
                        list(kpoints)
                        if isinstance(kpoints, (list, tuple))
                        else [kpoints, kpoints, kpoints]
                    )
                    kmesh = KPoints.monkhorst_pack(system, tuple(kp))
                result = _run_rks_periodic(
                    system,
                    basis,
                    kmesh,
                    ks_opts,
                    dft_plus_u=dft_plus_u,
                )
        elif resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
            if bz_integration not in (None, "smearing"):
                raise NotImplementedError(
                    "aiccm2026dev-b uses the exact finite cyclic-group sum; "
                    "alternative Brillouin-zone integration is not applicable"
                )
            if (
                aiccm_lattice_extension is not None
                or aiccm_wigner_seitz_shells is not None
            ):
                if kpoints is not None:
                    raise ValueError(
                        "aiccm2026dev-b accepts either the real-space "
                        "aiccm_lattice_extension/aiccm_wigner_seitz_shells "
                        "control or the legacy kpoints mesh alias, not both"
                    )
                _extension = cyclic_lattice_extension(
                    system,
                    aiccm_lattice_extension,
                    wigner_seitz_shells=aiccm_wigner_seitz_shells,
                )
                aiccm_mesh = _extension.repetitions
            elif kpoints is None:
                _extension = cyclic_lattice_extension(system)
                aiccm_mesh = _extension.repetitions
            elif isinstance(kpoints, (int, list, tuple)):
                _extension = cyclic_lattice_extension(system, mesh=kpoints)
                aiccm_mesh = _extension.repetitions
            elif getattr(kpoints, "mesh", None) is not None:
                if tuple(getattr(kpoints, "shift", (0, 0, 0))) != (0, 0, 0):
                    raise ValueError(
                        "aiccm2026dev-b requires a Gamma-centred k mesh "
                        "(shift=(0,0,0)) because it represents a cyclic cluster"
                    )
                _extension = cyclic_lattice_extension(system, mesh=tuple(kpoints.mesh))
                aiccm_mesh = _extension.repetitions
            else:
                raise TypeError(
                    "aiccm2026dev-b kpoints must be a cyclic mesh size, a mesh "
                    "tuple/list, or a Gamma-centred KPoints object"
                )
            write(f"    lattice_extension   = {aiccm_mesh}\n")
            write(
                "    WS half-extent      = "
                f"{_extension.wigner_seitz_half_extent} lattice vectors\n"
            )
            write(f"    equivalent k net    = {aiccm_mesh} (Gamma-centred)\n")
            write(f"    aiccm_backend       = {aiccm_backend}\n")
            write(f"    aiccm_symmetry      = {aiccm_symmetry}\n")
            if method_upper == "RHF":
                result = run_aiccm2026dev_b_rhf(
                    system,
                    basis,
                    aiccm_mesh,
                    opts,
                    backend=aiccm_backend,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                    fock_mixing=opts.fock_mixing,
                    symmetry_mode=aiccm_symmetry,
                    symmetry_precision=symmetry_precision,
                    symmetry_require_full_group=(aiccm_symmetry_require_full_group),
                    progress=plog,
                )
            elif method_upper == "RKS":
                result = run_aiccm2026dev_b_rks(
                    system,
                    basis,
                    functional or "pbe",
                    aiccm_mesh,
                    opts,
                    backend=aiccm_backend,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                    fock_mixing=opts.fock_mixing,
                    symmetry_mode=aiccm_symmetry,
                    symmetry_precision=symmetry_precision,
                    symmetry_require_full_group=(aiccm_symmetry_require_full_group),
                    progress=plog,
                )
            elif method_upper == "UHF":
                result = run_aiccm2026dev_b_uhf(
                    system,
                    basis,
                    aiccm_mesh,
                    opts,
                    backend=aiccm_backend,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                    fock_mixing=opts.fock_mixing,
                    symmetry_mode=aiccm_symmetry,
                    symmetry_precision=symmetry_precision,
                    symmetry_require_full_group=(aiccm_symmetry_require_full_group),
                    progress=plog,
                )
            else:  # UKS
                result = run_aiccm2026dev_b_uks(
                    system,
                    basis,
                    functional or "pbe",
                    aiccm_mesh,
                    opts,
                    backend=aiccm_backend,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                    fock_mixing=opts.fock_mixing,
                    symmetry_mode=aiccm_symmetry,
                    symmetry_precision=symmetry_precision,
                    symmetry_require_full_group=(aiccm_symmetry_require_full_group),
                    progress=plog,
                )
        elif resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA:
            # EXPERIMENTAL: neutral fitted-torus real-Γ control via the
            # per-unit-cell adapter (D89: a representation control, not the
            # union-and-weight Γ-CCM construction). One real supercell-Γ
            # SCF on the BvK torus defined by the Γ-centred mesh.
            from .periodic.exchange_convention import (
                BVK_EWALD,
                exchange_q0_label,
            )
            from .periodic.ccm.real_gamma_runner import run_real_gamma_scf

            if bz_integration not in (None, "smearing"):
                raise NotImplementedError(
                    "jk_method='real-gamma' uses the exact finite BvK-torus "
                    "sum; alternative Brillouin-zone integration is not "
                    "applicable"
                )
            # Mesh parsing mirrors aiccm2026dev-b: the k-mesh argument IS
            # the BvK nrep (Γ-centred required).
            if kpoints is None:
                _extension = cyclic_lattice_extension(system)
                _rg_mesh = _extension.repetitions
            elif isinstance(kpoints, (int, list, tuple)):
                _extension = cyclic_lattice_extension(system, mesh=kpoints)
                _rg_mesh = _extension.repetitions
            elif getattr(kpoints, "mesh", None) is not None:
                if tuple(getattr(kpoints, "shift", (0, 0, 0))) != (0, 0, 0):
                    raise ValueError(
                        "jk_method='real-gamma' requires a Gamma-centred "
                        "k mesh (shift=(0,0,0)): the mesh defines the BvK "
                        "torus (nrep), not a Bloch sampling"
                    )
                _extension = cyclic_lattice_extension(
                    system, mesh=tuple(kpoints.mesh))
                _rg_mesh = _extension.repetitions
            else:
                raise TypeError(
                    "jk_method='real-gamma' kpoints must be a BvK mesh "
                    "size, a mesh tuple/list, or a Gamma-centred KPoints "
                    "object"
                )
            # The direct drivers implement exxdiv='ewald' and None
            # (strict-zero-mode); map the runner's exchange_exxdiv label.
            _rg_exxdiv = (
                "ewald"
                if exchange_q0_label(exchange_exxdiv) == BVK_EWALD
                else None
            )
            write(f"    BvK torus (nrep)    = {_rg_mesh}\n")
            write("    route               = real-gamma (neutral "
                  "fitted-torus control; EXPERIMENTAL)\n")
            write(f"    exchange_exxdiv     = {_rg_exxdiv}\n")
            if _aiccm_correlation is not None:
                write(f"    correlation         = {_aiccm_correlation} "
                      "(neutral-RI, EXPERIMENTAL)\n")
            result = run_real_gamma_scf(
                system,
                str(basis.name),
                method_upper,
                _rg_mesh,
                initial_guess=_requested_initial_guess,
                functional=(functional if method_upper in ("RKS", "UKS")
                            else None),
                exxdiv=_rg_exxdiv,
                aux_basis=aux_basis,
                ke_cutoff=(float(rsgdf_ke_cutoff)
                           if rsgdf_ke_cutoff is not None else 200.0),
                max_iter=int(opts.max_iter),
                conv_tol=float(opts.conv_tol_energy),
                lat_opts=opts.lattice_opts,
                retain_cderi=_aiccm_correlation is not None,
                **(
                    {
                        "grid_options": opts.grid,
                        "becke_image_radius_bohr": float(
                            opts.becke_image_radius_bohr
                        ),
                    }
                    if method_upper in ("RKS", "UKS")
                    else {}
                ),
            )
            if _aiccm_correlation is not None and result.converged:
                # M4b (#778). The NEUTRAL-RI correlation, on the same
                # CCMSystem and -- the point of retain_cderi -- the identical
                # L array the reference converged on, so the correlation
                # cannot ride a different kernel. Construction-matched: this
                # variant IS the neutral construction, so it owes the
                # neutral-RI citation lineage, never the bare four-centre one
                # (ruling R1).
                from dataclasses import replace as _dc_replace

                _ccm = result.ccm_system
                _open_shell = method_upper == "UHF"
                if _aiccm_correlation == "dlpno-mp2":
                    # The DLPNO drivers already take (ccm, reference, cderi),
                    # so they need no on-reference wrapper: the retained L is
                    # handed straight in. At the default zero truncations
                    # these reproduce the canonical RI correlation.
                    if _open_shell:
                        from .periodic.ccm.dlpno_ump2 import (
                            ccm_dlpno_ump2 as _corr_on_ref,
                        )
                    else:
                        from .periodic.ccm.dlpno import (
                            ccm_dlpno_mp2 as _corr_on_ref,
                        )
                elif _aiccm_correlation == "dlpno-ccsd":
                    if _open_shell:
                        from .periodic.ccm.dlpno_uccsd import (
                            ccm_dlpno_uccsd as _corr_on_ref,
                        )
                    else:
                        from .periodic.ccm.dlpno_ccsd import (
                            ccm_dlpno_ccsd as _corr_on_ref,
                        )
                elif _aiccm_correlation == "ccsd":
                    from .periodic.ccm.ccsd import (
                        run_ccm_ri_ccsd_on_reference as _corr_on_ref,
                    )
                else:
                    from .periodic.ccm.mp2 import (
                        run_ccm_ri_mp2_on_reference as _corr_on_ref,
                    )
                if _aiccm_correlation.startswith("dlpno-"):
                    _corr = _corr_on_ref(
                        _ccm, result.ccm_result, cderi=result.cderi,
                    )
                else:
                    _corr = _corr_on_ref(
                        _ccm, result.ccm_result, result.cderi,
                    )
                _n_c = int(_ccm.n_cells)
                # The DLPNO MP2 results spell it e_corr, the canonical and
                # CCSD ones e_correlation. Read whichever this driver carries
                # rather than assuming, so a rename surfaces as an
                # AttributeError here instead of a silently wrong energy.
                _e_corr_total = getattr(
                    _corr, "e_correlation", None
                )
                if _e_corr_total is None:
                    _e_corr_total = _corr.e_corr
                result = _dc_replace(
                    result,
                    correlation=_corr,
                    e_correlation=float(_e_corr_total) / _n_c,
                    e_total_correlated=float(_corr.e_total) / _n_c,
                )
        elif resolved_jk == PeriodicJKMethod.AICCM2026DEV_A:
            # EXPERIMENTAL: the union-and-weight four-centre Γ-CCM
            # construction. The Γ-centred mesh IS the BvK torus (nrep); M1's
            # front-door normalisation already turned
            # aiccm_lattice_extension into that mesh.
            from .periodic.ccm.four_center_runner import (
                FOUR_CENTRE_WEIGHTING,
                run_four_center_scf,
            )

            if bz_integration not in (None, "smearing"):
                raise NotImplementedError(
                    "variant='four-center' evaluates the exact finite "
                    "cyclic-group sum; alternative Brillouin-zone "
                    "integration is not applicable"
                )
            if kpoints is None:
                _fc_extension = cyclic_lattice_extension(system)
            elif isinstance(kpoints, (int, list, tuple)):
                _fc_extension = cyclic_lattice_extension(system, mesh=kpoints)
            elif getattr(kpoints, "mesh", None) is not None:
                if tuple(getattr(kpoints, "shift", (0, 0, 0))) != (0, 0, 0):
                    raise ValueError(
                        "variant='four-center' requires a Gamma-centred "
                        "k mesh (shift=(0,0,0)): the mesh defines the cyclic "
                        "cluster (nrep), not a Bloch sampling"
                    )
                _fc_extension = cyclic_lattice_extension(
                    system, mesh=tuple(kpoints.mesh))
            else:
                raise TypeError(
                    "variant='four-center' kpoints must be a cyclic mesh "
                    "size, a mesh tuple/list, or a Gamma-centred KPoints "
                    "object"
                )
            _fc_mesh = _fc_extension.repetitions
            write(f"    cyclic cluster (nrep) = {_fc_mesh}\n")
            write("    route               = four-center (union-and-weight "
                  "Γ-CCM, 2014 lineage; EXPERIMENTAL)\n")
            write("    weighting           = aiccm2026dev-a (symmetric "
                  "BvK-torus four-centre; union12 is not reachable here)\n")
            if _aiccm_correlation is not None:
                write(f"    correlation         = {_aiccm_correlation} (bare "
                      "four-centre, EXPERIMENTAL)\n")
            result = run_four_center_scf(
                system,
                str(basis.name),
                method_upper,
                _fc_mesh,
                initial_guess=_requested_initial_guess,
                functional=(functional if method_upper in ("RKS", "UKS")
                            else None),
                max_iter=int(opts.max_iter),
                conv_tol=float(opts.conv_tol_energy),
                lat_opts=opts.lattice_opts,
                **(
                    {
                        "grid_options": opts.grid,
                        "becke_image_radius_bohr": float(
                            opts.becke_image_radius_bohr
                        ),
                    }
                    if method_upper in ("RKS", "UKS")
                    else {}
                ),
            )
            # ``result.converged`` gates this deliberately: run_ccm_mp2
            # raises on an unconverged reference, and that error would mask
            # the SCF failure that actually happened. An unconverged run
            # reports itself as unconverged, with no correlation block.
            if _aiccm_correlation is not None and result.converged:
                # M4b (#778). The BARE four-centre drivers, on the SAME
                # CCMSystem and the SAME converged reference the SCF above
                # produced -- construction-matched, so the correlation rides
                # the union-and-weight Hamiltonian the user asked for and no
                # R1/D89 conflation arises (maintainer ruling 2026-09-08).
                # These drivers carry no dimensionality guard; their cost is
                # the dense n_ref_ao**4 AO tensor, which is a sizing story
                # (CLAUDE.md § 15), not a refusal.
                _ccm = result.ccm_system
                if _aiccm_correlation == "ccsd":
                    # Closed-shell only; the front door already refused UHF.
                    from .periodic.ccm.ccsd import run_ccm_ccsd as _corr_driver
                elif method_upper == "RHF":
                    from .periodic.ccm.mp2 import run_ccm_mp2 as _corr_driver
                else:
                    from .periodic.ccm.ump2 import run_ccm_ump2 as _corr_driver
                _corr = _corr_driver(
                    _ccm,
                    result.ccm_result,
                    method=FOUR_CENTRE_WEIGHTING,
                )
                # Every driver in periodic.ccm returns TOTAL cyclic-cluster
                # energies; this adapter divides, exactly as it does for the
                # SCF energy above. The dataclass field comments saying "per
                # reference cell" are wrong in the same way the CCMKSResult
                # and CCMUHFResult ones were (module docstring).
                _n_c = int(_ccm.n_cells)
                # Aliased: this function imports ``replace`` from dataclasses
                # locally further down, which makes the bare name a local for
                # the whole body and unbound this early.
                from dataclasses import replace as _dc_replace

                result = _dc_replace(
                    result,
                    correlation=_corr,
                    e_correlation=float(_corr.e_correlation) / _n_c,
                    e_total_correlated=float(_corr.e_total) / _n_c,
                )
        elif resolved_jk == PeriodicJKMethod.NEUTRAL_BLOCH:
            # EXPERIMENTAL: the neutral Γ-CCM torus in its Bloch
            # representation. The Γ-centred mesh IS the BvK torus (nrep), not
            # a Bloch sampling choice; M1's front-door normalisation already
            # turned aiccm_lattice_extension into that mesh before the
            # k-derived artefact flags were computed, so the per-k result
            # below reaches the output stage as a genuine multi-k run.
            from .periodic.ccm.neutral_bloch_runner import run_neutral_bloch_scf

            if bz_integration not in (None, "smearing"):
                raise NotImplementedError(
                    "variant='neutral-bloch' evaluates the exact finite "
                    "BvK-torus Bloch sum; alternative Brillouin-zone "
                    "integration is not applicable"
                )
            if kpoints is None:
                _nb_extension = cyclic_lattice_extension(system)
            elif isinstance(kpoints, (int, list, tuple)):
                _nb_extension = cyclic_lattice_extension(system, mesh=kpoints)
            elif getattr(kpoints, "mesh", None) is not None:
                if tuple(getattr(kpoints, "shift", (0, 0, 0))) != (0, 0, 0):
                    raise ValueError(
                        "variant='neutral-bloch' requires a Gamma-centred "
                        "k mesh (shift=(0,0,0)): the mesh defines the BvK "
                        "torus (nrep), not a Bloch sampling"
                    )
                _nb_extension = cyclic_lattice_extension(
                    system, mesh=tuple(kpoints.mesh))
            else:
                raise TypeError(
                    "variant='neutral-bloch' kpoints must be a BvK mesh "
                    "size, a mesh tuple/list, or a Gamma-centred KPoints "
                    "object"
                )
            _nb_mesh = _nb_extension.repetitions
            write(f"    BvK torus (nrep)    = {_nb_mesh}\n")
            write("    route               = neutral-bloch (neutral Γ-CCM, "
                  "Bloch producer; EXPERIMENTAL)\n")
            write("    pair_symmetry       = off (production multi-k fits)\n")
            result = run_neutral_bloch_scf(
                system,
                str(basis.name),
                method_upper,
                _nb_mesh,
                functional=(functional if method_upper in ("RKS", "UKS")
                            else None),
                aux_basis=aux_basis,
                # Off on purpose: with the CCM pair-star reduction off the
                # arm executes the production multi-k Hamiltonian, so the
                # Fourier identity with variant='real-gamma' is a statement
                # about the construction, not about a fit approximation.
                symmetry=False,
                return_lattice_density=bool(write_density or output_qvf),
                options=opts,
                progress=plog,
                gdf_method=(gdf_method or "rsgdf"),
                rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                mdf_ke_cutoff=mdf_ke_cutoff,
            )
        elif resolved_jk == PeriodicJKMethod.GDF:
            # Closed-shell multi-k GDF and the legacy closed-shell Gamma
            # driver apply the shared Saunders-Hillier operator.  The pure
            # PBC-GDF Gamma driver and all open-shell GDF drivers do not yet
            # implement it, so reject those combinations instead of silently
            # accepting and dropping an explicit or auto-resolved shift.
            if float(opts.level_shift) != 0.0 and method_upper == "ROHF":
                raise NotImplementedError(
                    "run_periodic_job: level_shift is not implemented for "
                    "ROHF/GDF. The Roothaan effective-Fock driver supports "
                    "DIIS and density damping; pass level_shift=0.0."
                )
            if float(opts.level_shift) != 0.0 and method_upper in ("UHF", "UKS"):
                raise NotImplementedError(
                    "run_periodic_job: level_shift is not implemented for "
                    f"open-shell {method_upper}/GDF drivers. Use "
                    "jk_method='bipole' "
                    "for a shifted open-shell periodic SCF, or pass "
                    "level_shift=0.0."
                )
            if (
                float(opts.level_shift) != 0.0
                and _requested_gamma_only
                and gdf_method is not None
            ):
                raise NotImplementedError(
                    "run_periodic_job: Gamma-only GDF with an explicit gdf_method "
                    f"({gdf_method!r}) does not implement level_shift. Omit "
                    "gdf_method to use the level-shift-capable closed-shell "
                    "Gamma fallback, provide a non-Gamma k-point mesh, or pass "
                    "level_shift=0.0."
                )
            if float(opts.fock_mixing) != 0.0 and method_upper == "ROHF":
                raise NotImplementedError(
                    "run_periodic_job: fock_mixing is not implemented for "
                    "ROHF/GDF. The Roothaan effective-Fock driver supports "
                    "DIIS and density damping; pass fock_mixing=0.0."
                )
            if (
                float(opts.fock_mixing) != 0.0
                and method_upper in ("UHF", "UKS")
                and kpoints is not None
            ):
                raise NotImplementedError(
                    "run_periodic_job: fock_mixing is not implemented for "
                    f"open-shell {method_upper}/GDF k-mesh drivers. Omit "
                    "kpoints only if Gamma is intended, or pass "
                    "fock_mixing=0.0."
                )
            if method_upper == "ROHF":
                # The validated KROHF/GDF driver fixes the BvK exchange
                # convention and implements Pulay DIIS plus density damping.
                # BIPOLE-only controls and SCF transformations without a
                # KROHF/GDF implementation must fail here, not disappear in
                # argument forwarding.
                # Note: use_multipole_far_field can no longer reach this list
                # -- the #511 guard refuses it on every non-BIPOLE route long
                # before dispatch. multipole_l_max is still live here, and is
                # the reason the row below stays: the pair is only meaningful
                # together, so listing one without the other would read as a
                # claim that the other is supported.
                _unsupported_rohf_gdf_knobs = {
                    "use_oda": bool(use_oda),
                    "oda_trust_lambda_max": float(oda_trust_lambda_max) != 1.0,
                    "use_mom": bool(use_mom),
                    "use_multipole_far_field": bool(use_multipole_far_field),
                    "multipole_l_max": int(multipole_l_max) != 2,
                    "ewald_omega": ewald_omega is not None,
                    "ewald_precision": float(ewald_precision) != 1e-8,
                    "use_exchange_ewald_split": (
                        use_exchange_ewald_split is not None
                    ),
                    "exchange_exxdiv": exchange_exxdiv != "ewald",
                    "sr_image_precision": (
                        sr_image_precision is None
                        or float(sr_image_precision) != 1e-6
                    ),
                    "sr_range_screening": bool(sr_range_screening),
                    "symmetry_stabilize": bool(symmetry_stabilize),
                    "symmetry_reduce_fock": bool(symmetry_reduce_fock),
                }
                _set_rohf_gdf_knobs = [
                    key
                    for key, active in _unsupported_rohf_gdf_knobs.items()
                    if active
                ]
                if _set_rohf_gdf_knobs:
                    raise NotImplementedError(
                        "run_periodic_job: periodic ROHF/GDF does not "
                        "implement these explicitly requested options: "
                        + ", ".join(sorted(_set_rohf_gdf_knobs))
                        + ". Drop them or use a supported method/backend."
                    )
            if (
                float(opts.fock_mixing) != 0.0
                and method_upper == "RHF"
                and _requested_gamma_only
                and gdf_method is not None
            ):
                raise NotImplementedError(
                    "run_periodic_job: closed-shell Gamma-only RHF/GDF with "
                    f"an explicit gdf_method ({gdf_method!r}) does not "
                    "implement fock_mixing. Omit gdf_method to use the "
                    "Fock-mixing-capable Gamma fallback, provide a non-Gamma "
                    "k-point mesh, or pass fock_mixing=0.0."
                )
            # Default-Γ closed-shell RHF (no explicit gdf_method) now routes
            # through the PySCF-µHa-validated run_pbc_gdf_rhf (exxdiv='ewald'),
            # the same driver as the explicit-gdf_method and open-shell Γ
            # UHF/UKS paths -- but ONLY when the cell is in its supported
            # domain. The legacy run_rhf_periodic_gamma_gdf (molecular limit /
            # exxdiv=None) stays the fallback for everything it cannot do:
            # RKS, dim<3, charged cells, finite-T smearing, and the symmetry /
            # Fock-mixing convergence aids it threads. The gate mirrors
            # run_pbc_gdf_rhf's own preconditions (pbc_gdf.py ~l.563-601) and
            # the knobs that driver honours (DIIS + damping, NOT fock_mixing /
            # level_shift / smearing) so we never route a cell it would reject
            # or silently drop a convergence aid. It also matches the
            # kpoints=(1,1,1) routing in run_krhf so default-Γ and an explicit
            # Γ k-mesh never disagree. Pre-2026-06-15 the default fell to the
            # legacy driver, which computes the molecular limit and disagreed
            # with PySCF / the explicit path by the finite-size Madelung shift
            # (~5.8 mHa on H2/sto-3g/12-bohr).
            _gamma_q_nuc = float(sum(atom.Z for atom in system.unit_cell))
            _gamma_n_elec = int(system.n_electrons())
            _gamma_default_pure_gdf_ok = (
                method_upper == "RHF"
                and not _ecp_active
                and int(system.dim) == 3
                and _gamma_n_elec % 2 == 0
                and int(system.multiplicity) == 1
                and abs(_gamma_q_nuc - _gamma_n_elec) <= 0.5
                and float(opts.smearing_temperature) <= 0.0
                and float(opts.fock_mixing) == 0.0
                and float(getattr(opts, "level_shift", 0.0)) == 0.0
                and not symmetry_stabilize
                and not symmetry_reduce_fock
            )
            # Closed-shell Γ RKS can use the native pure-GDF KS engine via the
            # spin-unrestricted implementation in its singlet limit. This is
            # the only Γ KS path here whose Hartree J is the Lpq/GDF build; the
            # legacy run_rhf_periodic_gamma_gdf fallback uses the molecular-
            # limit Ewald-J bridge and is not a PySCF-GDF parity route for
            # condensed cells. Keep the gate narrow so we never drop knobs that
            # run_pbc_gdf_uks does not honour. The Γ UKS/GDF driver applies
            # Fock mixing itself, so ionic auto profiles can remain on the
            # pure-GDF path instead of falling back to the legacy bridge.
            _gamma_default_rks_gdf_ok = (
                method_upper == "RKS"
                and not _ecp_active
                and int(system.dim) == 3
                and _gamma_n_elec % 2 == 0
                and int(system.multiplicity) == 1
                and abs(_gamma_q_nuc - _gamma_n_elec) <= 0.5
                and float(opts.smearing_temperature) <= 0.0
                and float(getattr(opts, "level_shift", 0.0)) == 0.0
                and not symmetry_stabilize
                and not symmetry_reduce_fock
            )
            # The explicit slab-GDF route is deliberately bounded to
            # closed-shell RHF/RKS on a full Gamma-centered tuple mesh. AUTO
            # continues to select the independent direct SLAB_EWALD_2D route.
            # Gamma and multi-k both enter the same signed truncated-metric
            # driver so their Coulomb/exchange gauge cannot diverge.
            if _uses_external_xc:
                # External XC always uses the generic finite-torus driver,
                # including its N_k=1 Gamma limit.  The dedicated Gamma GDF
                # fast paths below own molecular/overlap density domains and
                # cannot satisfy the provider's difference-closed periodic
                # projection contract.
                external_kpoints = (
                    kpoints if kpoints is not None else (1, 1, 1)
                )
                plog.info(f"external-XC GDF kmesh = {external_kpoints}")
                write_kmesh_line(external_kpoints, system=system)
                if method_upper == "UKS":
                    result = _gdf_scf(
                        run_kuks_periodic_gdf,
                        external_kpoints,
                        opts,
                        initial_density_k=_read_density_k_open,
                        functional=functional,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        xc_density_domain=(
                            PeriodicXCDensityDomain.PERIODIC_LATTICE
                        ),
                    )
                elif method_upper == "RKS":
                    if dft_plus_u:
                        _record_dft_plus_u_route(
                            "gdf_rks_"
                            + (
                                "multi_k"
                                if _requested_true_multik
                                else "gamma"
                            ),
                            "gdf",
                        )
                    result = _gdf_scf(
                        run_krks_periodic_gdf,
                        external_kpoints,
                        opts,
                        functional=functional,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        ibz_native=False,
                        bz_integration=bz_integration,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        density_mixer_depth=density_mixer_depth,
                        density_mixer_beta=density_mixer_beta,
                        density_mixer_kerker=density_mixer_kerker,
                        kerker_k0=kerker_k0,
                        kerker_strength=kerker_strength,
                        kerker_cutoff_ha=kerker_cutoff_ha,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                        xc_density_domain=(
                            PeriodicXCDensityDomain.PERIODIC_LATTICE
                        ),
                    )
                else:
                    raise RuntimeError(
                        "internal external-XC GDF dispatch invariant: "
                        f"unexpected method {method_upper!r}"
                    )
            elif int(system.dim) == 2:
                from ._vibeqc_core import CoulombMethod as _CoulombMethod

                opts.lattice_opts.coulomb_method = (
                    _CoulombMethod.SLAB_EWALD_2D
                )
                if kpoints is None:
                    slab_gdf_kpoints = (1, 1, 1)
                elif isinstance(kpoints, int):
                    slab_gdf_kpoints = (int(kpoints), int(kpoints), 1)
                else:
                    slab_gdf_kpoints = kpoints
                plog.info(f"slab GDF kmesh = {slab_gdf_kpoints}")
                write_kmesh_line(slab_gdf_kpoints, system=system)
                # Dispatch through the GDF optimizer-objective capture
                # (_gdf_scf) so optimize=True relaxes on the identical
                # slab driver re-run with compute_gradient=True --
                # never on a different force surface (G-PBC-002 § 6
                # rung 5; the bulk multi-k wiring's pattern).
                if method_upper == "RKS":
                    result = _gdf_scf(
                        run_krks_periodic_gdf,
                        slab_gdf_kpoints,
                        opts,
                        functional=functional,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                    )
                else:
                    result = _gdf_scf(
                        run_krhf_periodic_gdf,
                        slab_gdf_kpoints,
                        opts,
                        functional=None,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                    )
            # Multi-k bulk GDF dispatch when a k-mesh is requested.
            elif kpoints is not None or (
                _ecp_active and method_upper in ("RHF", "RKS", "UHF", "UKS")
            ):
                # An ECP-bearing cell at the default Gamma mesh takes this
                # branch too (#88): the run_pbc_gdf_* fast paths and the
                # legacy Gamma driver below read no ECP field, so they ran
                # all-electron in a valence basis (measured 2026-09-04: Mg
                # RKS/PBE identical with and without the ECP; Ag UKS
                # dispatched 47 electrons as 24/23). Gamma is the one-point
                # mesh of the k-point GDF drivers, which build the Z_eff
                # frame, Bloch-sum V_ECP into Hcore(k) and fill the valence
                # count; _periodic_route_applies_ecp admits exactly these
                # four methods.
                _gdf_kmesh = kpoints if kpoints is not None else (1, 1, 1)
                plog.info(f"kmesh = {_gdf_kmesh}")
                write_kmesh_line(_gdf_kmesh, system=system)
                if bz_integration is not None:
                    write(f"    bz_integration     = {bz_integration}\n")
                if method_upper == "UKS":
                    if bz_integration == "gilat":
                        raise NotImplementedError(
                            "run_periodic_job: bz_integration='gilat' is "
                            "wired for closed-shell multi-k GDF (RHF/RKS) "
                            "only; open-shell GDF needs per-spin "
                            "Gilat-Raubenheimer occupations."
                        )
                    result = _gdf_scf(
                        run_kuks_periodic_gdf,
                        _gdf_kmesh,
                        opts,
                        initial_density_k=_read_density_k_open,
                        functional=functional,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        ibz_native=symmetry_reduce_k,
                    )
                elif method_upper == "UHF":
                    if bz_integration == "gilat":
                        raise NotImplementedError(
                            "run_periodic_job: bz_integration='gilat' is "
                            "wired for closed-shell multi-k GDF (RHF/RKS) "
                            "only; open-shell GDF needs per-spin "
                            "Gilat-Raubenheimer occupations."
                        )
                    result = _gdf_scf(
                        run_kuhf_periodic_gdf,
                        _gdf_kmesh,
                        opts,
                        initial_density_k=_read_density_k_open,
                        functional=None,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        ibz_native=symmetry_reduce_k,
                    )
                elif method_upper == "ROHF":
                    result = _gdf_scf(
                        run_krohf_periodic_gdf,
                        _gdf_kmesh,
                        opts,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        initial_density_k=_read_density_k_open,
                    )
                elif method_upper == "RKS":
                    if dft_plus_u:
                        _record_dft_plus_u_route("gdf_rks_multi_k", "gdf")
                    result = _gdf_scf(
                        run_krks_periodic_gdf,
                        _gdf_kmesh,
                        opts,
                        functional=functional,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        ibz_native=symmetry_reduce_k,
                        bz_integration=bz_integration,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        density_mixer_depth=density_mixer_depth,
                        density_mixer_beta=density_mixer_beta,
                        density_mixer_kerker=density_mixer_kerker,
                        kerker_k0=kerker_k0,
                        kerker_strength=kerker_strength,
                        kerker_cutoff_ha=kerker_cutoff_ha,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                    )
                else:
                    if dft_plus_u:
                        _record_dft_plus_u_route("gdf_rhf_multi_k", "gdf")
                    result = _gdf_scf(
                        run_krhf_periodic_gdf,
                        _gdf_kmesh,
                        opts,
                        functional=None,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "rsgdf"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        ibz_native=symmetry_reduce_k,
                        bz_integration=bz_integration,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        density_mixer_depth=density_mixer_depth,
                        density_mixer_beta=density_mixer_beta,
                        density_mixer_kerker=density_mixer_kerker,
                        kerker_k0=kerker_k0,
                        kerker_strength=kerker_strength,
                        kerker_cutoff_ha=kerker_cutoff_ha,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                    )
            elif method_upper == "ROHF":
                # Gamma is the one-point mesh of the same spin-restricted
                # open-shell GDF engine; keeping one path preserves the
                # Roothaan orbital gauge and BvK exchange convention.
                result = _gdf_scf(
                    run_krohf_periodic_gdf,
                    (1, 1, 1),
                    opts,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                        initial_density_k=_read_density_k_open,
                )
            elif method_upper == "UKS":
                # Γ open-shell UKS -- the pure-GDF driver on the rsgdf
                # path (µHa-validated; consistent with the multi-k route).
                result = _gdf_scf(
                    run_pbc_gdf_uks,
                    opts,
                    functional=functional,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                )
            elif method_upper == "UHF":
                result = _gdf_scf(
                    run_pbc_gdf_uhf,
                    opts,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                )
            elif _gamma_default_rks_gdf_ok:
                result = _gdf_scf(
                    run_pbc_gdf_uks,
                    opts,
                    functional=functional,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                )
            elif gdf_method is not None or _gamma_default_pure_gdf_ok:
                # Closed-shell Γ RHF -> the PySCF-µHa-validated run_pbc_gdf_rhf
                # (exxdiv='ewald'). Fires for an explicit gdf_method (e.g.
                # 'mdf') AND for the plain default (gdf_method=None) once the
                # domain gate above passes. Γ RKS with an explicit gdf_method
                # is wired through the singlet UKS/GDF path above when that
                # path can honour all requested knobs; otherwise keep the
                # historical fail-closed behaviour for explicit gdf_method.
                if method_upper == "RKS":
                    raise NotImplementedError(
                        "run_periodic_job: Γ RKS with an explicit gdf_method "
                        f"({gdf_method!r}, e.g. 'mdf') cannot be combined with "
                        "the requested convergence/symmetry options. Drop "
                        "unsupported knobs such as fmixing_percent/fock_mixing "
                        "or use method='UKS'."
                    )
                result = _gdf_scf(
                    run_pbc_gdf_rhf,
                    opts,
                    aux_basis=aux_basis,
                    gdf_method=(gdf_method or "rsgdf"),
                    rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                    rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    mdf_ke_cutoff=mdf_ke_cutoff,
                    exxdiv="ewald",
                )
            else:
                # Legacy Γ GDF driver (molecular limit / exxdiv=None): the
                # fallback for the cases run_pbc_gdf_rhf does not support -- RKS,
                # dim<3, charged cells, finite-T smearing, and symmetry /
                # Fock-mixing convergence aids. (Open-shell Γ already routed to
                # run_pbc_gdf_u{hf,ks} above.)
                if rsgdf_tail_ke_cutoff is not None:
                    raise NotImplementedError(
                        "run_periodic_job: rsgdf_tail_ke_cutoff requires the "
                        "pure PBC-GDF route. The requested Γ job falls back to "
                        "the legacy molecular-limit GDF driver, which has no "
                        "high-|G| tail correction."
                    )
                _reject_legacy_gamma_gdf(system, basis, "run_periodic_job")
                result = run_rhf_periodic_gamma_gdf(
                    system,
                    basis,
                    opts,
                    functional=(functional if method_upper == "RKS" else None),
                    aux_basis=aux_basis,
                    fock_mixing=opts.fock_mixing,
                    symmetry_stabilize=symmetry_stabilize,
                    symmetry_reduce_fock=symmetry_reduce_fock,
                    progress=plog,
                )
        elif resolved_jk == PeriodicJKMethod.SLAB_EWALD_2D:
            # dim=2 vacuum-free slab (rigorous Parry / de Leeuw-Perram gauge).
            # Route through lattice_opts.coulomb_method=SLAB_EWALD_2D: the
            # RHF/RKS dispatchers split Gamma vs multi-k on the kmesh
            # internally; UKS uses its slab drivers directly. The slab-normal
            # axis is non-periodic, so the k-mesh along it is always a single
            # Gamma point. See handovers/HANDOVER_SLAB_EWALD_2D.md.
            from ._vibeqc_core import CoulombMethod as _CoulombMethod
            from .periodic_rhf_dispatch import run_rhf_periodic_scf as _run_rhf_slab
            from .periodic_ks_dispatch import run_rks_periodic_scf as _run_rks_slab
            from .kpoints import KPoints, as_bloch_kmesh as _as_bloch

            if int(system.dim) != 2:
                raise NotImplementedError(
                    "jk_method='slab_ewald_2d' requires a dim=2 slab; got "
                    f"dim={int(system.dim)}."
                )
            opts.lattice_opts.coulomb_method = _CoulombMethod.SLAB_EWALD_2D
            if kpoints is None:
                _kp = [1, 1, 1]
            elif isinstance(kpoints, int):
                _kp = [int(kpoints), int(kpoints), 1]
            elif isinstance(kpoints, (list, tuple)):
                _seq = list(kpoints)
                _kp = (_seq + [1, 1, 1])[:3]
                _kp[2] = 1  # slab normal is non-periodic
            else:
                _kp = None  # already a KPoints / BlochKMesh
            slab_kmesh = (
                kpoints if _kp is None
                else KPoints.monkhorst_pack(system, tuple(_kp))
            )
            if method_upper == "RHF":
                result = _run_rhf_slab(system, basis, slab_kmesh, opts, progress=plog)
            elif method_upper == "RKS":
                result = _run_rks_slab(system, basis, slab_kmesh, opts, progress=plog)
            elif method_upper == "UKS":
                from .periodic_uks_ewald import (
                    run_uks_periodic_gamma_ewald2d as _uks_gamma_slab,
                )
                from .periodic_uks_multi_k_ewald import (
                    run_uks_periodic_multi_k_ewald3d as _uks_multik_slab,
                )
                _bm = _as_bloch(slab_kmesh)
                if len(_bm.kpoints) == 1 and np.allclose(_bm.kpoints[0], 0.0):
                    result = _uks_gamma_slab(system, basis, opts, progress=plog)
                else:
                    result = _uks_multik_slab(system, basis, _bm, opts, progress=plog)
            else:
                # UHF-on-slab is a follow-on; pick_jk_method already blocks it,
                # so this is defense-in-depth (CLAUDE.md Sec. 7 fail-closed).
                raise NotImplementedError(
                    "SLAB_EWALD_2D open-shell HF (UHF) on slabs is a follow-on; "
                    "use RHF, RKS, or UKS."
                )
        elif resolved_jk == PeriodicJKMethod.FFT_POISSON:
            # Unreachable: validate_jk_method (above) raises on FFT_POISSON,
            # retired as a user route (v0.13.0) because Γ-only EWALD_3D is
            # wrong on dense ionic crystals. Kept as a defense-in-depth
            # fail-closed should a future caller bypass validation. The
            # internal Γ-only drivers (run_r{h,k}f_periodic_gamma_ewald3d)
            # remain for dilute periodic + mechanics, fail-closed on dense
            # cells (CLAUDE.md Sec.7).
            raise ValueError(
                "jk_method='fft_poisson' (Γ-only EWALD_3D) is retired "
                "(v0.13.0). Use jk_method='gdf' (default), 'bipole', or "
                "'gpw'."
            )
        elif resolved_jk == PeriodicJKMethod.RIJCOSX:
            if not _requested_true_multik:
                if method_upper not in ("RHF", "RKS", "UHF", "UKS"):
                    raise NotImplementedError(
                        "run_periodic_job: Gamma RIJCOSX is implemented for "
                        "RHF and vacuum-padded RKS/UHF/UKS only. Other "
                        "methods use the true multi-k GDF/COSX backend; pass "
                        "a mesh with at least two k-points (got "
                        f"kpoints={kpoints!r})."
                    )
                if float(getattr(opts, "smearing_temperature", 0.0) or 0.0) > 0.0:
                    raise NotImplementedError(
                        "run_periodic_job: Gamma RIJCOSX RHF uses integer "
                        "occupations. Pass a true multi-k mesh to use the "
                        "GDF/COSX smearing path."
                    )
                if density_mixer not in (None, "", "none", "diis"):
                    raise NotImplementedError(
                        "run_periodic_job: density_mixer is wired on the "
                        "true multi-k RIJCOSX route only. Pass a mesh with "
                        "at least two k-points, or use the Gamma RIJCOSX RHF "
                        "driver with DIIS."
                    )
                if method_upper == "RKS":
                    # Vacuum-padded envelope only; the driver fails closed
                    # on tight cells (their XC needs the periodic-density
                    # grid).
                    from .periodic_rijcosx import run_periodic_rijcosx_rks

                    result = run_periodic_rijcosx_rks(
                        system,
                        basis,
                        opts,
                        functional=functional,
                        aux_basis=aux_basis,
                        grid_options=opts.grid,
                        progress=plog,
                    )
                elif method_upper == "UHF":
                    from .periodic_rijcosx import run_periodic_rijcosx_uhf

                    result = run_periodic_rijcosx_uhf(
                        system,
                        basis,
                        opts,
                        aux_basis=aux_basis,
                        progress=plog,
                    )
                elif method_upper == "UKS":
                    from .periodic_rijcosx import run_periodic_rijcosx_uks

                    result = run_periodic_rijcosx_uks(
                        system,
                        basis,
                        opts,
                        functional=functional,
                        aux_basis=aux_basis,
                        grid_options=opts.grid,
                        progress=plog,
                    )
                else:
                    from .periodic_rijcosx import run_periodic_rijcosx_rhf

                    result = run_periodic_rijcosx_rhf(
                        system,
                        basis,
                        opts,
                        aux_basis=aux_basis,
                        gdf_method=(gdf_method or "compcell"),
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        progress=plog,
                    )
            else:
                plog.info(f"kmesh = {kpoints}")
                write_kmesh_line(kpoints, system=system)
                write("    k_exchange         = cosx\n")
                if bz_integration is not None:
                    write(f"    bz_integration     = {bz_integration}\n")
                rijcosx_gdf_method = gdf_method or "rsgdf"
                write(f"    aux_basis           = {aux_basis or '<auto>'}\n")
                write(f"    gdf_method          = {rijcosx_gdf_method}\n")
                write(f"    rsgdf_ke_cutoff     = {float(rsgdf_ke_cutoff)}\n")
                if rsgdf_tail_ke_cutoff is not None:
                    write(
                        "    rsgdf_tail_ke_cutoff = "
                        f"{float(rsgdf_tail_ke_cutoff)}\n"
                    )
                write(f"    mdf_ke_cutoff       = {float(mdf_ke_cutoff)}\n")
                if method_upper == "UKS":
                    if bz_integration == "gilat":
                        raise NotImplementedError(
                            "run_periodic_job: bz_integration='gilat' is "
                            "wired for closed-shell multi-k GDF/RIJCOSX "
                            "(RHF/RKS) only; open-shell RIJCOSX needs "
                            "per-spin Gilat-Raubenheimer occupations."
                        )
                    if rsgdf_tail_ke_cutoff is not None:
                        raise NotImplementedError(
                            "run_periodic_job: multi-k UKS/RIJCOSX does not "
                            "yet accept rsgdf_tail_ke_cutoff. The open-shell "
                            "multi-k GDF tail route must be wired first."
                        )
                    result = run_kuks_periodic_gdf(
                        system,
                        basis,
                        kpoints,
                        opts,
                        initial_density_k=_read_density_k_open,
                        functional=functional,
                        aux_basis=aux_basis,
                        gdf_method=rijcosx_gdf_method,
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        k_exchange="cosx",
                        xc_density_domain=(
                            PeriodicXCDensityDomain.PERIODIC_LATTICE
                            if _uses_external_xc
                            else PeriodicXCDensityDomain.AUTO
                        ),
                        progress=plog,
                    )
                elif method_upper == "UHF":
                    if bz_integration == "gilat":
                        raise NotImplementedError(
                            "run_periodic_job: bz_integration='gilat' is "
                            "wired for closed-shell multi-k GDF/RIJCOSX "
                            "(RHF/RKS) only; open-shell RIJCOSX needs "
                            "per-spin Gilat-Raubenheimer occupations."
                        )
                    if rsgdf_tail_ke_cutoff is not None:
                        raise NotImplementedError(
                            "run_periodic_job: multi-k UHF/RIJCOSX does not "
                            "yet accept rsgdf_tail_ke_cutoff. The open-shell "
                            "multi-k GDF tail route must be wired first."
                        )
                    result = run_kuhf_periodic_gdf(
                        system,
                        basis,
                        kpoints,
                        opts,
                        initial_density_k=_read_density_k_open,
                        functional=None,
                        aux_basis=aux_basis,
                        gdf_method=rijcosx_gdf_method,
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        k_exchange="cosx",
                        progress=plog,
                    )
                elif method_upper == "RKS":
                    if dft_plus_u:
                        _record_dft_plus_u_route("rijcosx_rks_multi_k", "rijcosx")
                    result = run_krks_periodic_gdf(
                        system,
                        basis,
                        kpoints,
                        opts,
                        functional=functional,
                        aux_basis=aux_basis,
                        use_compcell=True,
                        k_exchange="cosx",
                        gdf_method=rijcosx_gdf_method,
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        bz_integration=bz_integration,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        density_mixer_depth=density_mixer_depth,
                        density_mixer_beta=density_mixer_beta,
                        density_mixer_kerker=density_mixer_kerker,
                        kerker_k0=kerker_k0,
                        kerker_strength=kerker_strength,
                        kerker_cutoff_ha=kerker_cutoff_ha,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                        xc_density_domain=(
                            PeriodicXCDensityDomain.PERIODIC_LATTICE
                            if _uses_external_xc
                            else PeriodicXCDensityDomain.AUTO
                        ),
                        progress=plog,
                    )
                else:
                    if dft_plus_u:
                        _record_dft_plus_u_route("rijcosx_rhf_multi_k", "rijcosx")
                    result = run_krhf_periodic_gdf(
                        system,
                        basis,
                        kpoints,
                        opts,
                        functional=None,
                        aux_basis=aux_basis,
                        use_compcell=True,
                        k_exchange="cosx",
                        gdf_method=rijcosx_gdf_method,
                        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
                        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                        mdf_ke_cutoff=mdf_ke_cutoff,
                        bz_integration=bz_integration,
                        fock_mixing=opts.fock_mixing,
                        density_mixer=density_mixer,
                        density_mixer_depth=density_mixer_depth,
                        density_mixer_beta=density_mixer_beta,
                        density_mixer_kerker=density_mixer_kerker,
                        kerker_k0=kerker_k0,
                        kerker_strength=kerker_strength,
                        kerker_cutoff_ha=kerker_cutoff_ha,
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                        progress=plog,
                    )
        elif resolved_jk == PeriodicJKMethod.DIRECT:
            # Dispatch DIRECT through the periodic_jk_direct wrapper.
            # AUTO never picks DIRECT (it diverges on tight ionic
            # crystals); the user has explicitly opted in here.
            # NOTE: this is a placeholder dispatch -- there's no DIRECT
            # SCF driver yet that builds J + K via jk_via_direct each
            # iter. It's a wrapper one would call directly outside the
            # periodic_runner SCF loop. Track the SCF-driver port in
            # docs/design_native_gdf.md (DIRECT loop variant).
            raise NotImplementedError(
                "DIRECT SCF driver is not wired into run_periodic_job "
                "yet. Use vibeqc.periodic_jk_direct.jk_via_direct(...) "
                "for one-shot J/K builds. AUTO picks GDF for SCF; opt "
                "in to DIRECT only for vacuum-padded debug studies."
            )
        elif resolved_jk == PeriodicJKMethod.GPW:
            # M2-full / M3a / M3b / M3d / M3e GPW SCF entry.
            # Γ-only: RHF, ROHF, ROKS, RKS, UHF, UKS via the closed-shell and
            # open-shell GPW drivers.
            # Multi-k: pure-DFT RKS, ROKS, and UKS via per-k drivers.
            if method_upper not in (
                "RHF", "ROHF", "ROKS", "RKS", "UHF", "UKS"
            ):
                raise NotImplementedError(
                    f"PeriodicJKMethod.GPW currently supports RHF, ROHF, ROKS, "
                    f"RKS, UHF, and UKS; got method={method!r}."
                )
            from .periodic_gapw_j import run_periodic_rhf_gpw
            from .periodic_gapw_runner_adapter import (
                gpw_result_to_runner_shape,
                gpw_uhf_result_to_runner_shape,
                gpw_uks_result_to_runner_shape,
            )

            is_dft = method_upper in ("ROKS", "RKS", "UKS")
            xc_for_gpw = functional if is_dft else None
            if is_dft and not xc_for_gpw:
                raise ValueError(
                    f"PeriodicJKMethod.GPW + method={method_upper!r} "
                    "requires a functional= argument (e.g. 'lda', "
                    "'pbe', 'b3lyp'). Got functional=None."
                )
            def _gpw_explicit_gamma_mesh(value) -> bool:
                if value is None:
                    return False
                if isinstance(value, (int, np.integer)):
                    return int(value) == 1
                if isinstance(value, (list, tuple)):
                    return tuple(int(x) for x in value) == (1, 1, 1)
                return False

            gpw_use_gamma_branch = (
                method_upper != "RKS"
                and _gpw_explicit_gamma_mesh(kpoints)
            )
            if kpoints is not None and not gpw_use_gamma_branch:
                # Multi-k GPW: pure-DFT RKS, ROKS, and UKS (LDA/GGA/meta-GGA;
                # hybrids raise from the drivers). Build a BlochKMesh and
                # dispatch.
                if method_upper not in ("RKS", "ROKS", "UKS"):
                    raise NotImplementedError(
                        f"PeriodicJKMethod.GPW multi-k through "
                        f"run_periodic_job supports RKS, ROKS, and UKS (pure "
                        f"DFT); got method={method!r}. Multi-k UHF / "
                        f"hybrids need per-k exact exchange and are not "
                        f"wired."
                    )
                if method_upper in ("ROKS", "UKS") and dft_plus_u:
                    raise NotImplementedError(
                        "run_periodic_job: dft_plus_u is not wired on the "
                        f"multi-k {method_upper} GPW driver yet. Use a "
                        "supported Gamma GPW route or multi-k BIPOLE for "
                        "open-shell +U."
                    )
                from ._vibeqc_core import monkhorst_pack as _mp
                from .periodic_gapw_j import run_periodic_rks_gpw_multi_k

                kp = (
                    list(kpoints)
                    if isinstance(kpoints, (list, tuple))
                    else [kpoints, kpoints, kpoints]
                )
                # Use symmetry-reduced k-mesh when the system has
                # symmetry attached (attach_symmetry was called
                # upstream by the reduce_to_primitive / symmetry
                # kwarg resolution).
                if _uses_external_xc:
                    kmesh = _runner_bloch_kmesh(system, kpoints)
                    kp = list(getattr(kmesh, "mesh", (1, 1, 1)))
                    plog.info(f"  GPW multi-k: mesh = {kp} (full BZ)")
                    write_kmesh_line(
                        kp,
                        suffix=" (full BZ; external XC)",
                        system=system,
                        kpoints_cart=kmesh.kpoints,
                    )
                elif system.symmetry is not None:
                    from .kpoints import KPoints

                    kmesh_obj = KPoints.monkhorst_pack(system, tuple(kp), symmetry=True)
                    kmesh = kmesh_obj.to_bloch_kmesh()
                    plog.info(
                        f"  GPW multi-k: mesh = {kp} "
                        f"(symmetry-reduced: {len(kmesh.kpoints)} k)"
                    )
                    write_kmesh_line(
                        kp,
                        suffix=f" (symmetry-reduced: {len(kmesh.kpoints)} k)",
                        system=system,
                        kpoints_cart=kmesh.kpoints,
                    )
                else:
                    kmesh = _mp(system, list(kp))
                    plog.info(f"  GPW multi-k: mesh = {kp}")
                    write_kmesh_line(kp, system=system)

                if method_upper in ("ROKS", "UKS"):
                    # Pure-DFT multi-k ROKS/UKS (LDA/GGA/meta-GGA; hybrids
                    # raise in the drivers). Smearing for restricted-open
                    # shell and non-GDF UKS is gated upstream; +U gated above.
                    from .periodic_gapw_open_shell import (
                        run_periodic_roks_gpw_multi_k,
                        run_periodic_uks_gpw_multi_k,
                    )
                    from .periodic_gapw_runner_adapter import (
                        gpw_roks_multik_result_to_runner_shape,
                        gpw_uks_multik_result_to_runner_shape,
                    )

                    multik_open_driver = (
                        run_periodic_roks_gpw_multi_k
                        if method_upper == "ROKS"
                        else run_periodic_uks_gpw_multi_k
                    )
                    gpw_result = multik_open_driver(
                        system,
                        basis,
                        kmesh,
                        initial_density_k=_read_density_k_open,
                        functional=functional,
                        cutoff_ha=cutoff_ha,
                        max_iter=opts.max_iter,
                        conv_tol_energy=opts.conv_tol_energy,
                        conv_tol_density=1e-7,
                        initial_guess=opts.initial_guess,
                        quiet=True,
                        progress=plog,
                                     atomic_spins=atomic_spins,
                    )
                    if (not opts.use_diis) or opts.damping != 0.0:
                        write(
                            "    note: use_diis/damping are not honored "
                            f"by the GPW multi-k {method_upper} driver (internal "
                            "per-k Pulay DIIS)\n"
                        )
                    multik_open_adapter = (
                        gpw_roks_multik_result_to_runner_shape
                        if method_upper == "ROKS"
                        else gpw_uks_multik_result_to_runner_shape
                    )
                    result = multik_open_adapter(
                        gpw_result, system, basis
                    )
                else:
                    if dft_plus_u:
                        _record_dft_plus_u_route("gpw_rks_multi_k", "gpw")
                    gpw_result = run_periodic_rks_gpw_multi_k(
                        system,
                        basis,
                        kmesh,
                        functional=functional,
                        cutoff_ha=cutoff_ha,
                        max_iter=opts.max_iter,
                        conv_tol_energy=opts.conv_tol_energy,
                        conv_tol_density=1e-7,
                        smearing_temperature=opts.smearing_temperature,
                        smearing_method=smearing_method_label,
                        fock_mixing=(
                            opts.fock_mixing if fmixing_percent is not None else None
                        ),
                        dft_plus_u_sites=dft_plus_u,
                        initial_density_k=_read_density_k_closed,
                        initial_guess=opts.initial_guess,
                        external_xc_grid_options=opts.grid,
                        external_xc_image_radius_bohr=float(
                            opts.becke_image_radius_bohr
                        ),
                        external_xc_lattice_options=opts.lattice_opts,
                        quiet=True,
                        progress=plog,
                    )
                    if getattr(gpw_result, "fock_mixing", 0.0) != 0.0:
                        write(
                            "    fock_mixing         = "
                            f"{float(gpw_result.fock_mixing):.3g} "
                            "(GPW multi-k)\n"
                        )
                    if (not opts.use_diis) or opts.damping != 0.0:
                        write(
                            "    note: use_diis/damping are not honored by "
                            "the GPW multi-k driver (internal scheme)\n"
                        )
                    result = _GapwMultiKRunnerProxy(gpw_result)
            elif method_upper in ("RHF", "RKS"):
                if dft_plus_u:
                    _record_dft_plus_u_route(
                        f"gpw_{method_upper.lower()}_gamma",
                        "gpw",
                    )
                gpw_result = run_periodic_rhf_gpw(
                    system,
                    basis,
                    cutoff_ha=cutoff_ha,
                    max_iter=opts.max_iter,
                    conv_tol_energy=opts.conv_tol_energy,
                    conv_tol_density=1e-7,
                    functional=xc_for_gpw,
                    use_diis=opts.use_diis,
                    damping=opts.damping,
                    diis_subspace_size=opts.diis_subspace_size,
                    diis_start_iter=opts.diis_start_iter,
                    smearing_temperature=opts.smearing_temperature,
                    smearing_method=smearing_method_label,
                    dft_plus_u_sites=dft_plus_u,
                    initial_guess=_gamma_route_initial_guess,
                    initial_density=_gamma_route_initial_density,
                    external_xc_grid_options=(
                        opts.grid if _uses_external_xc else None
                    ),
                    external_xc_image_radius_bohr=float(
                        opts.becke_image_radius_bohr
                        if _uses_external_xc
                        else 10.0
                    ),
                    external_xc_lattice_options=(
                        opts.lattice_opts if _uses_external_xc else None
                    ),
                    quiet=True,
                    progress=plog,
                )
                result = gpw_result_to_runner_shape(
                    gpw_result,
                    system,
                    basis,
                )
            elif method_upper in ("ROHF", "ROKS"):
                from .periodic_gapw_open_shell import (
                    run_periodic_rohf_gpw,
                    run_periodic_roks_gpw,
                )
                from .periodic_gapw_runner_adapter import (
                    gpw_rohf_result_to_runner_shape,
                )

                if opts.fock_mixing != 0.0:
                    raise NotImplementedError(
                        "run_periodic_job: fock_mixing/fmixing_percent is not "
                        f"implemented for periodic {method_upper}/GPW."
                    )
                restricted_open_driver = (
                    run_periodic_roks_gpw
                    if method_upper == "ROKS"
                    else run_periodic_rohf_gpw
                )
                gpw_result = restricted_open_driver(
                    system,
                    basis,
                    cutoff_ha=cutoff_ha,
                    max_iter=opts.max_iter,
                    conv_tol_energy=opts.conv_tol_energy,
                    conv_tol_grad=1e-7,
                    use_diis=opts.use_diis,
                    damping=opts.damping,
                    diis_subspace_size=opts.diis_subspace_size,
                    diis_start_iter=opts.diis_start_iter,
                    level_shift=float(level_shift or 0.0),
                    initial_guess=_gamma_route_initial_guess,
                    initial_density=_gamma_route_initial_density,
                    quiet=True,
                    progress=plog,
                    **(
                        {"functional": str(functional)}
                        if method_upper == "ROKS"
                        else {}
                    ),
                )
                result = gpw_rohf_result_to_runner_shape(gpw_result)
            elif method_upper == "UHF":
                from .periodic_gapw_open_shell import (
                    run_periodic_uhf_gpw,
                )

                if dft_plus_u:
                    _record_dft_plus_u_route("gpw_uhf_gamma", "gpw")
                gpw_result = run_periodic_uhf_gpw(
                    system,
                    basis,
                    cutoff_ha=cutoff_ha,
                    max_iter=opts.max_iter,
                    conv_tol_energy=opts.conv_tol_energy,
                    conv_tol_density=1e-7,
                    use_diis=opts.use_diis,
                    damping=opts.damping,
                    diis_subspace_size=opts.diis_subspace_size,
                    diis_start_iter=opts.diis_start_iter,
                    dft_plus_u_sites=dft_plus_u,
                    initial_guess=_gamma_route_initial_guess,
                    initial_density=_gamma_route_initial_density,
                    quiet=True,
                    progress=plog,
                                 atomic_spins=atomic_spins,
                )
                result = gpw_uhf_result_to_runner_shape(
                    gpw_result,
                    system,
                    basis,
                )
            else:  # method_upper == "UKS"
                from .periodic_gapw_open_shell import (
                    run_periodic_uks_gpw,
                )

                if dft_plus_u:
                    _record_dft_plus_u_route("gpw_uks_gamma", "gpw")
                gpw_result = run_periodic_uks_gpw(
                    system,
                    basis,
                    functional=xc_for_gpw,
                    cutoff_ha=cutoff_ha,
                    max_iter=opts.max_iter,
                    conv_tol_energy=opts.conv_tol_energy,
                    conv_tol_density=1e-7,
                    use_diis=opts.use_diis,
                    damping=opts.damping,
                    diis_subspace_size=opts.diis_subspace_size,
                    diis_start_iter=opts.diis_start_iter,
                    dft_plus_u_sites=dft_plus_u,
                    initial_guess=_gamma_route_initial_guess,
                    initial_density=_gamma_route_initial_density,
                    quiet=True,
                    progress=plog,
                                 atomic_spins=atomic_spins,
                )
                result = gpw_uks_result_to_runner_shape(
                    gpw_result,
                    system,
                    basis,
                )
        elif resolved_jk == PeriodicJKMethod.GAPW:
            # M3c GAPW all-electron SCF entry. Same dispatch structure
            # as the GPW block, but uses the per-atom augmentation
            # correction for all-electron accuracy. Multi-k RKS is
            # available via :func:`run_periodic_rks_gapw_multi_k`.
            #
            # The GAPW all-electron augmentation route's two open correctness
            # bugs are FIXED (2026-06-26): the Hartree Fock is now the exact
            # derivative of the energy (analytic J = dE_H/dD), and the bonded-
            # molecule overlapping-augmentation double-count is root-caused
            # (own-atom compensator + partition-of-unity). The run-level
            # GAPWExperimentalWarning that flagged those bugs is therefore
            # retired for jk_method='gapw'. The remaining caveats are accuracy,
            # not correctness -- the per-atom augmentation has a grid-convergent
            # absolute residual (hundreds of mHa for 2nd-row atoms at the
            # default N=24 grid). Hydrogen's no-core soft-basis residual is
            # fixed by retaining its full valence contraction. The remaining
            # caveat is documented in docs/user_guide/gapw.md (the .out
            # "(experimental)" label stays).
            if method_upper not in ("RHF", "RKS", "UHF", "UKS"):
                raise NotImplementedError(
                    f"PeriodicJKMethod.GAPW currently supports RHF, "
                    f"RKS, UHF, and UKS; got method={method!r}."
                )
            gapw_use_gamma_branch = _external_gapw_gamma_dispatch or (
                method_upper != "RKS" and _explicit_unit_kmesh(kpoints)
            )
            if kpoints is not None and not gapw_use_gamma_branch:
                # Multi-k GAPW: pure DFT only (RKS).
                if method_upper != "RKS":
                    raise NotImplementedError(
                        f"PeriodicJKMethod.GAPW multi-k only supports "
                        f"RKS (pure DFT) at v0.10.x; hybrid / HF "
                        f"multi-k is not wired. Got method={method!r}."
                    )
                from ._vibeqc_core import monkhorst_pack as _mp
                from .periodic_gapw_augment import (
                    run_periodic_rks_gapw_multi_k,
                )

                kp = (
                    list(kpoints)
                    if isinstance(kpoints, (list, tuple))
                    else [kpoints, kpoints, kpoints]
                )
                # Use symmetry-reduced k-mesh when the system has
                # symmetry attached.
                if system.symmetry is not None:
                    from .kpoints import KPoints

                    kmesh_obj = KPoints.monkhorst_pack(system, tuple(kp), symmetry=True)
                    kmesh = kmesh_obj.to_bloch_kmesh()
                    plog.info(
                        f"  GAPW multi-k: mesh = {kp} "
                        f"(symmetry-reduced: {len(kmesh.kpoints)} k)"
                    )
                    write_kmesh_line(
                        kp,
                        suffix=f" (symmetry-reduced: {len(kmesh.kpoints)} k)",
                        system=system,
                        kpoints_cart=kmesh.kpoints,
                    )
                else:
                    kmesh = _mp(system, list(kp))
                    plog.info(f"  GAPW multi-k: mesh = {kp}")
                    write_kmesh_line(kp, system=system)

                if dft_plus_u:
                    _record_dft_plus_u_route("gapw_rks_multi_k", "gapw")
                gapw_result = run_periodic_rks_gapw_multi_k(
                    system,
                    basis,
                    kmesh,
                    functional=functional,
                    cutoff_ha=cutoff_ha,
                    max_iter=opts.max_iter,
                    conv_tol_energy=opts.conv_tol_energy,
                    conv_tol_density=1e-7,
                    smearing_temperature=opts.smearing_temperature,
                    smearing_method=smearing_method_label,
                    dft_plus_u_sites=dft_plus_u,
                    initial_density_k=_read_density_k_closed,
                    initial_guess=opts.initial_guess,
                    quiet=True,
                )
                # Adapt the multi-k result to the runner duck-type.
                # GpwMultiKScfResult carries per-k data under
                # ``mo_coeffs_k`` / ``mo_energies_k`` / ``occupations_k``;
                # the runner output code looks for ``mo_coeffs``,
                # ``mo_energies``, and ``occupations`` as per-k lists.
                result = _GapwMultiKRunnerProxy(gapw_result)
            else:
                from .periodic_gapw_augment import (
                    run_periodic_rhf_gapw,
                    run_periodic_rks_gapw,
                    run_periodic_uhf_gapw,
                    run_periodic_uks_gapw,
                )
                from .periodic_gapw_runner_adapter import (
                    gpw_result_to_runner_shape,
                    gpw_uhf_result_to_runner_shape,
                    gpw_uks_result_to_runner_shape,
                )

                is_dft = method_upper in ("RKS", "UKS")
                xc_for_gapw = functional if is_dft else None
                if is_dft and not xc_for_gapw:
                    raise ValueError(
                        f"PeriodicJKMethod.GAPW + method={method_upper!r} "
                        "requires a functional= argument (e.g. 'lda', "
                        "'pbe', 'b3lyp'). Got functional=None."
                    )
                if method_upper in ("RHF", "RKS"):
                    if dft_plus_u:
                        _record_dft_plus_u_route(
                            f"gapw_{method_upper.lower()}_gamma",
                            "gapw",
                        )
                    gapw_result = run_periodic_rhf_gapw(
                        system,
                        basis,
                        cutoff_ha=cutoff_ha,
                        max_iter=opts.max_iter,
                        conv_tol_energy=opts.conv_tol_energy,
                        conv_tol_density=1e-7,
                        functional=xc_for_gapw,
                        use_diis=opts.use_diis,
                        damping=opts.damping,
                        diis_subspace_size=opts.diis_subspace_size,
                        diis_start_iter=opts.diis_start_iter,
                        smearing_temperature=opts.smearing_temperature,
                        smearing_method=smearing_method_label,
                        dft_plus_u_sites=dft_plus_u,
                        initial_guess=_gamma_route_initial_guess,
                        initial_density=_gamma_route_initial_density,
                        molecular_limit=gapw_molecular_limit,
                        memory_override=memory_override,
                        external_xc_grid_options=(
                            opts.grid if _uses_external_xc else None
                        ),
                        external_xc_image_radius_bohr=float(
                            opts.becke_image_radius_bohr
                            if _uses_external_xc
                            else 10.0
                        ),
                        external_xc_lattice_options=(
                            opts.lattice_opts if _uses_external_xc else None
                        ),
                        quiet=True,
                    )
                    result = gpw_result_to_runner_shape(
                        gapw_result,
                        system,
                        basis,
                    )
                elif method_upper == "UHF":
                    if dft_plus_u:
                        _record_dft_plus_u_route("gapw_uhf_gamma", "gapw")
                    gapw_result = run_periodic_uhf_gapw(
                        system,
                        basis,
                        cutoff_ha=cutoff_ha,
                        max_iter=opts.max_iter,
                        conv_tol_energy=opts.conv_tol_energy,
                        conv_tol_density=1e-7,
                        use_diis=opts.use_diis,
                        damping=opts.damping,
                        diis_subspace_size=opts.diis_subspace_size,
                        diis_start_iter=opts.diis_start_iter,
                        dft_plus_u_sites=dft_plus_u,
                        initial_guess=_gamma_route_initial_guess,
                        initial_density=_gamma_route_initial_density,
                        molecular_limit=gapw_molecular_limit,
                        memory_override=memory_override,
                        quiet=True,
                                      atomic_spins=atomic_spins,
                    )
                    result = gpw_uhf_result_to_runner_shape(
                        gapw_result,
                        system,
                        basis,
                    )
                else:  # method_upper == "UKS"
                    if dft_plus_u:
                        _record_dft_plus_u_route("gapw_uks_gamma", "gapw")
                    gapw_result = run_periodic_uks_gapw(
                        system,
                        basis,
                        functional=xc_for_gapw,
                        cutoff_ha=cutoff_ha,
                        max_iter=opts.max_iter,
                        conv_tol_energy=opts.conv_tol_energy,
                        conv_tol_density=1e-7,
                        use_diis=opts.use_diis,
                        damping=opts.damping,
                        diis_subspace_size=opts.diis_subspace_size,
                        diis_start_iter=opts.diis_start_iter,
                        dft_plus_u_sites=dft_plus_u,
                        initial_guess=_gamma_route_initial_guess,
                        initial_density=_gamma_route_initial_density,
                        memory_override=memory_override,
                        quiet=True,
                                      atomic_spins=atomic_spins,
                    )
                    result = gpw_uks_result_to_runner_shape(
                        gapw_result,
                        system,
                        basis,
                    )
        elif resolved_jk == PeriodicJKMethod.BIPOLE:
            # CRYSTAL-gauge Ewald J-split -- RHF/RKS/UHF/UKS, plus the
            # restricted-open-shell corrected-Ewald-exchange engine.
            # Use user-provided k-mesh if given, else default to Γ-only
            kmesh = _runner_bloch_kmesh(system, kpoints)
            if kpoints is not None:
                plog.info(f"  BIPOLE kmesh: {kpoints}")
            if bz_integration == "gilat" and method_upper not in (
                "RKS",
                "UHF",
                "UKS",
            ):
                raise NotImplementedError(
                    "run_periodic_job: bz_integration='gilat' is wired for "
                    "BIPOLE RKS/UHF/UKS; BIPOLE RHF/ROHF/ROKS still need "
                    "route-specific Gilat-Raubenheimer occupation support."
                )
            if bz_integration is not None and kpoints is not None:
                write(f"    bz_integration     = {bz_integration}\n")
            if dft_plus_u:
                _record_dft_plus_u_route(
                    f"bipole_{method_upper.lower()}_"
                    + ("multi_k" if _requested_true_multik else "gamma"),
                    "bipole",
                )

            if method_upper in ("ROHF", "ROKS"):
                # Restricted-open-shell BIPOLE route: the corrected
                # Ewald-exchange EWALD_3D engine (erfc K_SR + reciprocal
                # q = k-k' K_LR + BvK probe-charge G=0 correction) with
                # Roothaan's effective Fock per k. ROKS adds the
                # spin-polarised V_xc to each per-spin Fock and scales the
                # exchange by the functional's global fraction (a pure
                # functional builds no exchange at all). Γ runs on the
                # (1,1,1) mesh of the same driver, so there is one code
                # path. The engine fixes its exchange convention and Ewald
                # α internally; the BIPOLE knob surface below is therefore
                # rejected rather than silently ignored.
                if method_upper == "ROKS":
                    from .periodic_roks_multi_k_ewald import (
                        run_roks_periodic_multi_k_ewald3d,
                    )

                    result = run_roks_periodic_multi_k_ewald3d(
                        system,
                        basis,
                        kmesh,
                        opts,
                        sr_image_precision=sr_image_precision,
                        progress=plog,
                        initial_density_k=_read_density_k_open,
                    )
                else:
                    from .periodic_rohf_multi_k_ewald import (
                        run_rohf_periodic_multi_k_ewald3d,
                    )

                    result = run_rohf_periodic_multi_k_ewald3d(
                        system,
                        basis,
                        kmesh,
                        opts,
                        sr_image_precision=sr_image_precision,
                        progress=plog,
                        initial_density_k=_read_density_k_open,
                    )
            elif method_upper == "RHF":
                from .pbc_bipole import run_pbc_bipole_rhf

                result = run_pbc_bipole_rhf(
                    system,
                    basis,
                    kmesh,
                    opts,
                    linear_dep_threshold=1e-7,
                    use_ewald_j_split=True,
                    ewald_omega=ewald_omega,
                    ewald_precision=ewald_precision,
                    use_oda=use_oda,
                    oda_trust_lambda_max=oda_trust_lambda_max,
                    use_mom=use_mom,
                    use_multipole_far_field=use_multipole_far_field,
                    multipole_l_max=multipole_l_max,
                    use_exchange_ewald_split=use_exchange_ewald_split,
                    exchange_exxdiv=exchange_exxdiv,
                    use_fock_symmetry=symmetry_stabilize,
                    use_fock_symmetry_reduce=symmetry_reduce_fock,
                    sr_image_precision=sr_image_precision,
                    exact_zone_bohr=_bipole_exact_zone_bohr,
                    progress=plog,
                    dft_plus_u=dft_plus_u,
                    bz_integration=bz_integration,
                             initial_density_k=_read_density_k_closed,
                )
            elif method_upper == "UHF":
                from .pbc_bipole_uhf import run_pbc_bipole_uhf

                result = run_pbc_bipole_uhf(
                    system,
                    basis,
                    kmesh,
                    opts,
                    linear_dep_threshold=1e-7,
                    use_ewald_j_split=True,
                    ewald_omega=ewald_omega,
                    ewald_precision=ewald_precision,
                    use_oda=use_oda,
                    oda_trust_lambda_max=oda_trust_lambda_max,
                    use_mom=use_mom,
                    use_multipole_far_field=use_multipole_far_field,
                    multipole_l_max=multipole_l_max,
                    use_exchange_ewald_split=use_exchange_ewald_split,
                    exchange_exxdiv=exchange_exxdiv,
                    use_fock_symmetry=symmetry_stabilize,
                    use_fock_symmetry_reduce=symmetry_reduce_fock,
                    sr_image_precision=sr_image_precision,
                    exact_zone_bohr=_bipole_exact_zone_bohr,
                    progress=plog,
                    dft_plus_u=dft_plus_u,
                    bz_integration=bz_integration,
                             initial_density_k=_read_density_k_open,
                )
            elif method_upper == "RKS":
                from .pbc_bipole_rks import run_pbc_bipole_rks

                result = run_pbc_bipole_rks(
                    system,
                    basis,
                    kmesh,
                    opts,
                    functional=functional,
                    linear_dep_threshold=1e-7,
                    use_ewald_j_split=True,
                    ewald_omega=ewald_omega,
                    ewald_precision=ewald_precision,
                    use_oda=use_oda,
                    oda_trust_lambda_max=oda_trust_lambda_max,
                    use_mom=use_mom,
                    use_multipole_far_field=use_multipole_far_field,
                    multipole_l_max=multipole_l_max,
                    use_exchange_ewald_split=use_exchange_ewald_split,
                    exchange_exxdiv=exchange_exxdiv,
                    use_fock_symmetry=symmetry_stabilize,
                    use_fock_symmetry_reduce=symmetry_reduce_fock,
                    sr_image_precision=sr_image_precision,
                    exact_zone_bohr=_bipole_exact_zone_bohr,
                    progress=plog,
                    dft_plus_u=dft_plus_u,
                    xc_density_domain=(
                        PeriodicXCDensityDomain.PERIODIC_LATTICE
                        if _uses_external_xc
                        else PeriodicXCDensityDomain.AUTO
                    ),
                    # Forward the resolved GR selector explicitly; silently
                    # dropping it here was the historical BUG-PER-002 mode.
                    bz_integration=bz_integration,
                             initial_density_k=_read_density_k_closed,
                )
            elif method_upper == "UKS":
                from .pbc_bipole_uks import run_pbc_bipole_uks

                result = run_pbc_bipole_uks(
                    system,
                    basis,
                    kmesh,
                    opts,
                    functional=functional,
                    linear_dep_threshold=1e-7,
                    use_ewald_j_split=True,
                    ewald_omega=ewald_omega,
                    ewald_precision=ewald_precision,
                    use_oda=use_oda,
                    oda_trust_lambda_max=oda_trust_lambda_max,
                    use_mom=use_mom,
                    use_multipole_far_field=use_multipole_far_field,
                    multipole_l_max=multipole_l_max,
                    use_exchange_ewald_split=use_exchange_ewald_split,
                    exchange_exxdiv=exchange_exxdiv,
                    use_fock_symmetry=symmetry_stabilize,
                    use_fock_symmetry_reduce=symmetry_reduce_fock,
                    sr_image_precision=sr_image_precision,
                    exact_zone_bohr=_bipole_exact_zone_bohr,
                    progress=plog,
                    dft_plus_u=dft_plus_u,
                    xc_density_domain=(
                        PeriodicXCDensityDomain.PERIODIC_LATTICE
                        if _uses_external_xc
                        else PeriodicXCDensityDomain.AUTO
                    ),
                    bz_integration=bz_integration,
                             initial_density_k=_read_density_k_open,
                )
            else:
                # Defensive: method_upper is validated to RHF/RKS/UHF/UKS at
                # the top of run_periodic_job and never reassigned, so this is
                # unreachable today. The guard keeps the BIPOLE branch self-
                # consistent with the GPW/GAPW branches (which guard locally)
                # so widening the allowed-method set upstream can never silently
                # fall through to an UnboundLocalError on `result` below.
                raise NotImplementedError(
                    "run_periodic_job: PeriodicJKMethod.BIPOLE dispatch: "
                    f"method={method_upper!r} is not supported "
                    "(expected RHF, ROHF, ROKS, RKS, UHF, or UKS)."
                )
        else:
            raise NotImplementedError(
                f"Periodic JK method {resolved_jk.value!r} dispatch "
                f"is not yet implemented in v0.7.1-spike."
            )
        t_scf = time.perf_counter() - t0
        from dataclasses import is_dataclass, replace
        from .guess import GuessSelection
        executed_guess = getattr(result, "guess_selection", None)
        selection = GuessSelection(
            InitialGuess.FRAGMO if _fragmo_request else _requested_initial_guess,
            InitialGuess.FRAGMO if _fragmo_request else (
                executed_guess.effective if executed_guess is not None else _resolved_initial_guess),
            executed_guess.transport if executed_guess is not None else _resolved_initial_guess,
        )
        if (is_dataclass(result) and result.__dataclass_params__.frozen
            and "guess_selection" in result.__dataclass_fields__):
            result = replace(result, guess_selection=selection)
        else:
            result.guess_selection = selection

        # --- Write SCF trace + energies ------------------------------
        # Stamp live progress into the .system manifest so vq can
        # read iteration + energy from a single file.
        _trace = getattr(result, "scf_trace", None)
        if _trace:
            # #768: select the terminal row from any sequence, not only a
            # list. The GPW / GAPW drivers return ``scf_trace=tuple(...)``
            # (periodic_gapw_j.py, periodic_gapw_augment.py,
            # periodic_gapw_open_shell.py), and a tuple failed the ``list``
            # test, so ``_last`` became the whole trace: neither field branch
            # below matched it, ``_fields`` stayed empty, and every multi-k
            # GPW/GAPW run stamped no [progress] section at all. A result
            # that exposes a single row rather than a sequence still falls
            # through to the ``else`` branch.
            _last = (
                _trace[-1] if isinstance(_trace, (list, tuple)) else _trace
            )
            _fields = {}
            if hasattr(_last, "iter"):
                _fields["iteration"] = int(_last.iter)
                _fields["energy_eh"] = float(_last.energy)
                _fields["gradient_norm"] = float(_last.grad_norm)
                _fields["diis_subspace"] = int(_last.diis_subspace)
            elif isinstance(_last, dict):
                _fields["iteration"] = int(_last.get("iter", 0))
                _fields["energy_eh"] = float(_last.get("energy", 0))
                _fields["gradient_norm"] = float(_last.get("grad_norm", 0))
                _fields["diis_subspace"] = int(_last.get("diis_subspace", 0))
            if _fields:
                _fields["phase"] = "scf"
                _output_writer.update_progress(**_fields)
        # Shared with the molecular path: iteration table, the
        # "converged in N iterations" line, and the energy-component
        # breakdown all come from format_scf_trace so periodic and
        # molecular .out files read identically. Periodic-only smearing
        # quantities follow. ``label`` (RHF/RKS/...) is already in the
        # "Job: PERIODIC <label>" header above, so it is not repeated here.
        write_scf_trace(
            result,
            include_banner=False,
            include_properties=False,
            energy_label="Total energy",
            trailing="\n\n",
        )
        # IID 344: a parity-held result states the hold next to the energy
        # it qualifies ("" for ordinary runs, so the .out is unchanged).
        write(_parity_hold_summary(result))
        write(_smearing_summary(result))
        write(_linear_dependence_summary(result))
        write(_band_summary(result))
        write(_mo_summary(result))
        # M4b (#778): the correlation block sits right under the SCF trace,
        # so the two energies it relates are adjacent in the .out.
        if getattr(result, "correlation", None) is not None:
            write(section_header("AICCM correlation", width=56))
            # The treatment follows the driver's stamp, so a (T) that actually
            # ran is named and one that did not is not claimed.
            _corr_treatment = (
                result.correlation.backend.rsplit("-a-", 1)[-1]
                .replace("ri-", "")
                .upper()
            )
            write(f"    treatment           = {_corr_treatment} "
                  "(EXPERIMENTAL)\n")
            # The lineage follows the driver's own stamp rather than a
            # hard-coded word: the two wired arms are DIFFERENT constructions
            # (ruling R1) and printing one's name over the other's number is
            # the confusion this whole taxonomy exists to prevent.
            write("    lineage             = "
                  + ("neutral-RI (fitted torus)"
                     if "-ri-" in result.correlation.backend
                     else "bare four-centre (union-and-weight)")
                  + "\n")
            write("    reference           = the SCF above "
                  "(construction-matched)\n")
            write("    driver              = "
                  f"{type(result.correlation).__name__}\n")
            write("    citation route      = "
                  f"{result.correlation.backend}\n")
            write("    E(corr) / cell      = "
                  f"{result.e_correlation:.10f} Ha\n")
            write("    E(SCF+corr) / cell  = "
                  f"{result.e_total_correlated:.10f} Ha\n\n")
        if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
            _aiccm_diag = result.aiccm2026dev_b
            write(section_header("AICCM2026DEV-B invariants", width=56))
            write(f"    cyclic mesh         = {_aiccm_diag.mesh}\n")
            write(f"    cyclic cells        = {_aiccm_diag.n_cyclic_cells}\n")
            write(f"    electronic method   = {_aiccm_diag.electronic_method}\n")
            write(f"    integral backend    = {_aiccm_diag.backend}\n")
            write(f"    CCM approach        = {_aiccm_diag.ccm_approach}\n")
            write(f"    CCM construction    = {_aiccm_diag.ccm_construction}\n")
            write(
                "    evaluation repr.    = "
                f"{_aiccm_diag.evaluation_representation}\n"
            )
            write(
                "    coulomb kernel      = "
                f"{getattr(_aiccm_diag, 'coulomb_kernel', 'not-recorded')}\n"
            )
            write(
                "    exchange q=0        = "
                f"{getattr(_aiccm_diag, 'exchange_q0', 'not-recorded')}\n"
            )
            write(
                "    q=0 applicability   = "
                f"{getattr(_aiccm_diag, 'exchange_q0_applicability', 'not-recorded')}\n"
            )
            _aiccm_exchange = getattr(
                _aiccm_diag,
                "exact_exchange_assembly",
                None,
            )
            if _aiccm_exchange is not None:
                write(
                    "    EXX assembly schema = "
                    f"{_aiccm_exchange.schema}\n"
                )
                write(
                    "    screened EXX live   = "
                    f"{_aiccm_exchange.screened_exchange_applicability}\n"
                )
                write(
                    "    screened EXX build  = "
                    f"{_aiccm_exchange.screened_exchange_assembly}\n"
                )
            write(
                "    boundary model      = "
                f"{getattr(_aiccm_diag, 'boundary_model', 'not-recorded')}\n"
            )
            _aiccm_convention = getattr(_aiccm_diag, "finite_torus_convention", None)
            if _aiccm_convention is not None:
                write(
                    "    BvK Madelung cell   = "
                    f"{_aiccm_convention.bvk_madelung_supercell_repetitions}\n"
                )
            write(
                "    WS partition error  = "
                f"{_aiccm_diag.wigner_seitz_partition_error:.3e}\n"
            )
            write(
                "    idempotency error   = "
                f"{_aiccm_diag.density_idempotency_error:.3e}\n"
            )
            write(
                f"    electron-count err  = {_aiccm_diag.electron_count_error:.3e}\n"
            )
            write(
                "    inverse-Bloch Im    = "
                f"{_aiccm_diag.inverse_bloch_imaginary_residual:.3e}\n\n"
            )
            _symmetry_diag = getattr(result, "aiccm2026dev_b_symmetry", None)
            if _symmetry_diag is not None:
                _symmetry_plan = _symmetry_diag.plan
                write(section_header("AICCM2026DEV-B space-group diagnostic", width=56))
                write(
                    "    space group         = "
                    f"{_symmetry_plan.international_symbol} "
                    f"(No. {_symmetry_plan.space_group_number})\n"
                )
                write(
                    "    compatible ops      = "
                    f"{_symmetry_plan.n_operations_compatible}/"
                    f"{_symmetry_plan.n_operations_full}\n"
                )
                write(
                    "    full/irreducible k  = "
                    f"{_symmetry_plan.n_kpoints_full}/"
                    f"{_symmetry_plan.n_kpoints_irreducible}\n"
                )
                write("    acceleration        = diagnostic only (not applied)\n")
                write(
                    "    shell pairs         = "
                    f"{_symmetry_diag.n_unique_shell_pairs}/"
                    f"{_symmetry_diag.n_shell_pairs} unique\n"
                )
                if _symmetry_diag.n_shell_quartets is not None:
                    write(
                        "    shell quartets      = "
                        f"{_symmetry_diag.n_unique_shell_quartets}/"
                        f"{_symmetry_diag.n_shell_quartets} unique\n"
                    )
                else:
                    write("    shell quartets      = count skipped (>24 shells)\n")
                if _symmetry_diag.gamma_fock_residual is not None:
                    write(
                        "    Gamma Fock residual = "
                        f"{_symmetry_diag.gamma_fock_residual:.3e}\n"
                    )
                if _symmetry_diag.gamma_density_residual is not None:
                    write(
                        "    Gamma dens residual = "
                        f"{_symmetry_diag.gamma_density_residual:.3e}\n"
                    )
                write("\n")

        # --- Convergence-strategy post-SCF check ----------------------
        # The pre-SCF classification is a guess; the converged spectrum
        # is the truth. On auto-mode runs, compare and say so when they
        # disagree (generalizes the GDF conducting-state warning).
        if convergence_strategy.mode.startswith("auto") and bool(
            getattr(result, "converged", False)
        ):
            from .periodic_convergence_auto import (
                converged_gap_hartree,
                post_scf_profile_check,
            )

            _n_elec_chk = int(round(system.n_electrons()))
            _mult_chk = int(getattr(system, "multiplicity", 1) or 1)
            _n_a_chk = (_n_elec_chk + (_mult_chk - 1)) // 2
            _n_b_chk = _n_elec_chk - _n_a_chk
            _gap_chk = converged_gap_hartree(
                result,
                n_alpha=(
                    _n_a_chk
                    if method_upper in ("ROHF", "ROKS", "UHF", "UKS")
                    else _n_elec_chk // 2
                ),
                n_beta=(
                    _n_b_chk
                    if method_upper in ("ROHF", "ROKS", "UHF", "UKS")
                    else None
                ),
            )
            _profile_check = post_scf_profile_check(convergence_strategy, _gap_chk)
            if _profile_check is not None:
                _lvl, _msg = _profile_check
                write(section_header("Convergence strategy check", width=56))
                write(f"    {_lvl}: {_msg}\n\n")
                if _lvl == "warning":
                    warnings.warn(
                        f"run_periodic_job: {_msg}",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    plog.info(f"convergence strategy check: {_msg}")

        # --- Fail-closed SCF convergence gate --------------------------
        # Mirrors the molecular runner's post-SCF guard (run_job): a
        # periodic SCF that hit its iteration cap must fail loudly, not
        # return an unconverged energy stamped "complete". The 2026-08-12
        # CaO PBE GDF 200-iter exit-0 incident is the canonical failure
        # mode: run_periodic_job returned the capped result, the CLI
        # exited 0, and the .system manifest flipped to "complete", so
        # every downstream consumer of exit codes / manifest status
        # promoted an unconverged energy into the validation database.
        # Everything after this gate (dispersion, sidecars, QVF,
        # optimization) assumes a converged reference, so it is skipped
        # on the failure path exactly like the molecular route.
        if method_upper in ("RHF", "UHF", "RKS", "UKS", "ROHF", "ROKS") and not bool(
            getattr(result, "converged", False)
        ):
            _n_iter = int(getattr(result, "n_iter", 0))
            _energy = float(getattr(result, "energy", 0.0))
            # #116: name the numbers the terminal check measured, so a
            # withdrawal is auditable from the message alone. The terminal
            # trace row carries the exact-operator delta and commutator
            # norm of the density returned to the caller.
            _tol_g = None
            try:
                _tol_g = getattr(opts, "conv_tol_grad", None)
            except NameError:
                _tol_g = None
            # #748: this clause is built by a helper that cannot raise. It
            # used to read ``_last.delta_e`` directly, which is an
            # AttributeError on the dict rows the GAPW multi-k route
            # forwards, so the diagnostic destroyed the very failure report
            # it exists to provide.
            _terminal = _terminal_scf_check_clause(
                getattr(result, "scf_trace", None),
                conv_tol_energy=conv_tol_energy,
                conv_tol_grad=_tol_g,
            )
            _msg = (
                f"{method_upper} periodic SCF did not converge after "
                f"{_n_iter} iterations; refusing to treat the energy or "
                "post-SCF properties as a successful calculation."
                + _terminal
            )
            write(f"\n  FATAL: {_msg}\n")
            flush()
            try:
                _output_writer.crash(wall_seconds=t_scf)
            except Exception as _crash_exc:
                warn_output_failure(
                    _crash_exc,
                    _output_writer.manifest_path,
                    role="manifest_crash_status",
                    category=OutputFailureKind.manifest_recording,
                )
            if _checkpointer.enabled:
                _finalize_periodic_checkpoint(
                    _checkpointer,
                    result,
                    system,
                    method_upper,
                    basis.name,
                    functional,
                )
            raise RuntimeError(_msg)

        # --- Post-SCF dispersion (D3-BJ) -----------------------------
        # Mirrors the molecular runner's _DispersionAugmented wrapping
        # (see python/vibeqc/runner.py): the SCF result is left
        # untouched (energy = pure SCF) and a wrapper exposes the
        # dispersion piece via .e_dispersion / .energy_total. The
        # periodic lattice sum is handled by
        # vibeqc.dispersion_periodic.compute_d3bj_periodic.
        if dispersion is not None and dispersion is not False:
            from .dispersion_periodic import compute_d3bj_periodic
            from .runner import _DispersionAugmented, _resolve_dispersion

            d3_params = _resolve_dispersion(
                dispersion,
                functional if method_upper in ("ROKS", "RKS", "UKS") else None,
            )
            if d3_params is not None:
                disp = compute_d3bj_periodic(
                    system,
                    d3_params,
                    cutoff_bohr=dispersion_cutoff_bohr,
                    backend=dispersion_backend,
                )
                e_scf = float(getattr(result, "energy", 0.0))
                e_total = e_scf + float(disp.energy)
                write(
                    "\n"
                    + section_header("Dispersion correction (D3-BJ, periodic)")
                    + f"  {'backend':>10s} {disp.backend!s:>14s}\n"
                    f"  {'supercell':>10s} {disp.supercell!s:>14s}\n"
                    f"  {'s6':>10s} {d3_params.s6:14.6f}\n"
                    f"  {'s8':>10s} {d3_params.s8:14.6f}\n"
                    f"  {'a1':>10s} {d3_params.a1:14.6f}\n"
                    f"  {'a2':>10s} {d3_params.a2:14.6f}\n"
                    f"  {'E_disp':>10s} {render_energy_labeled(disp.energy, width=14, precision=8)}"
                    f"  ({disp.energy * 627.5094740631:+.4f} kcal/mol)\n"
                    f"  {'E_SCF':>10s} {render_energy_labeled(e_scf, width=14, precision=8)}\n"
                    f"  {'E_total':>10s} {render_energy_labeled(e_total, width=14, precision=8)}\n"
                )
                plog.info(
                    f"  E_disp = {disp.energy:+.6e} Ha "
                    f"({disp.backend}, supercell={disp.supercell})"
                )
                result = _DispersionAugmented(result, float(disp.energy), d3_params)

        if _trexio_path is not None:
            # Use physical SCF sampling where the driver retains it; the
            # resolved runner mesh supplies older result types' metadata.
            _tx_points = getattr(result, "restart_kpoints", None)
            if _tx_points is None:
                _tx_points = getattr(result, "kpoints_cart", None)
            _tx_weights = getattr(result, "restart_weights", None)
            if _tx_weights is None:
                _tx_weights = getattr(result, "kpoint_weights", None)
            _tx_mesh = getattr(result, "kmesh", None) or _runner_bloch_kmesh(system, kpoints)
            if _tx_points is None:
                _tx_points = _tx_mesh.kpoints
            if _tx_weights is None:
                _tx_weights = _tx_mesh.weights
            _output_writer.dispatch_role(
                "orbitals", only_format="trexio", result=result, basis=basis,
                molecule=system.unit_cell_molecule(), system=system,
                ecp_source=SimpleNamespace(
                    ecp_primitive_blocks=_ecp_blocks,
                    ecp_primitive_centers=_ecp_centers,
                    ecp_effective_charges=_eff_z,
                    ecp_total_ncore=_total_ncore,
                ),
                uses_ecp=bool(_ecp_blocks), trexio_backend=trexio_backend,
                trexio_kpoints=_tx_points, trexio_weights=_tx_weights,
                trexio_description=f"{method_upper}/{basis.name} periodic job {output_stem.name}.",
                raise_on_error=True,
            )

        # --- Molden ---------------------------------------------------
        _qvf_wf = None
        _qvf_bloch_wf = None
        if write_molden_file and _has_valid_sidecar_mo_coeffs(result):
            try:
                from .runner import _DispersionAugmented  # noqa: F401

                mol = system.unit_cell_molecule()
                # Molden carries one real orbital set, so export the Γ block.
                # Periodic drivers hand back per-k lists even for a Γ-only
                # run (restricted under ``mo_coeffs``, unrestricted under
                # ``mo_coeffs_alpha`` / ``_beta``); both shapes go through
                # the same Γ-locating proxy.
                molden_result = result
                if isinstance(
                    getattr(result, "mo_coeffs", None), (list, tuple)
                ) or isinstance(
                    getattr(result, "mo_coeffs_alpha", None), (list, tuple)
                ):
                    molden_result = _gamma_orbital_proxy(result)
                molden_result = _result_with_ecp_ncore(
                    molden_result,
                    _total_ncore,
                )
                _output_writer.dispatch_role(
                    "orbitals",
                    only_format="molden",
                    result=molden_result,
                    basis=basis,
                    molecule=mol,
                    title=(
                        f"{output_stem.name} -- Gamma-block orbitals, "
                        "home cell only (no lattice sum)"
                    ),
                    raise_on_error=True,
                )
                # Molden has no periodic representation: a viewer evaluates
                # these coefficients over home-cell basis functions with no
                # image sum, so the render is the home-cell truncation of
                # the Gamma crystalline orbital and clips wherever an
                # orbital straddles a cell face. Say so rather than let the
                # file read as a crystalline-orbital export. QVF can carry
                # the coefficients together with the periodic metadata and
                # torus-periodic grids, so it does not share this limitation.
                write(
                    f"  Gamma-point orbitals written to {molden_path.name}\n"
                    "    Molden is a molecular format: no lattice, no "
                    "k-points. Coefficients are\n"
                    "    evaluated over home-cell basis functions only, so "
                    "orbitals straddling a\n"
                    "    cell face appear clipped. For crystalline orbitals "
                    "use output_qvf=True\n"
                    "    and open the archive with vibe-view.\n"
                )
            except Exception as exc:
                warn(
                    f"molden write failed: {type(exc).__name__}: {exc}",
                    role="molden",
                )
                warn_writer_failure(
                    exc,
                    molden_path,
                    role="molden",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )
                # Downgrade the Molden plan row so finish() does not
                # raise IncompleteOutputError for a legitimate
                # physics-gated refusal (non-real Bloch orbitals).
                _output_writer.downgrade_planned("orbitals")
        if (
            output_qvf
            and _has_valid_sidecar_mo_coeffs(result)
            and not _mo_coeffs_span_unit_cell_basis(result, basis)
        ):
            # Recorded, not silent: the QVF wavefunction section pairs MO
            # coefficients with the unit cell's basis and structure, and this
            # result's orbitals span a larger AO space (the real-Γ producer's
            # supercell-Γ MOs at a torus above one cell). Omitting the section
            # is the same discipline the Molden sidecar already applies
            # through _molden_sidecar_supported (#654).
            write(
                "    QVF wavefunction    = omitted ("
                f"{_sidecar_mo_ao_dimension(result)} AO orbitals against a "
                f"{int(getattr(basis, 'nbasis', 0) or 0)}-function unit-cell "
                "basis; the archive carries the unit cell)\n"
            )
        if _should_prepare_periodic_qvf_wavefunction(output_qvf, result, basis):
            # Build the ``wavefunction.gto`` section from the Gamma-point
            # MO coefficients.  vibe-view resamples orbitals on its own grid,
            # so the raw coefficients are valuable even when the periodic
            # runner's own torus-periodic grid render (above) is the primary
            # delivery path.  Gamma-point coefficients are real and portable;
            # multi-k runs that include Gamma export the k=0 block.
            try:
                from vibeqc.output.formats.qvf import qvf_wf_data

                _wf_mol = system.unit_cell_molecule()
                if isinstance(getattr(result, "mo_coeffs", None), (list, tuple)):
                    _wf_result = _gamma_proxy_for_multi_k(result)
                elif isinstance(getattr(result, "mo_coeffs_alpha", None), (list, tuple)):
                    _wf_result = _gamma_orbital_proxy(result)
                else:
                    _wf_result = result
                _wf_result = _result_with_ecp_ncore(
                    _wf_result,
                    _total_ncore,
                )
                _qvf_wf = qvf_wf_data(_wf_result, basis, _wf_mol)
            except Exception as _qvf_wf_exc:
                warn_output_failure(
                    _qvf_wf_exc,
                    stem_sibling(output_stem, ".qvf"),
                    role="qvf_wavefunction_prep",
                    category=OutputFailureKind.compatibility_fallback,
                )
            if (
                method_upper in ("RHF", "UHF", "ROHF", "ROKS", "RKS", "UKS")
                and resolved_jk != PeriodicJKMethod.AICCM2026DEV_B
            ):
                try:
                    from vibeqc.output.formats.qvf import qvf_bloch_wf_data

                    _kpts_cart = _result_kpoints_cart(result)
                    if _kpts_cart is None:
                        raise ValueError("QVF READ payload requires source k-point metadata")
                    _kpts_frac = _wrap_reciprocal_fractional(
                        _fractional_kpoints_for_output(system, _kpts_cart)
                    )
                    _qvf_bloch_result = _result_with_ecp_ncore(
                        result,
                        _total_ncore,
                    )
                    _qvf_bloch_wf = qvf_bloch_wf_data(
                        _qvf_bloch_result,
                        basis,
                        system.unit_cell_molecule(),
                        k_points=_kpts_frac,
                        k_weights=_result_kpoint_weights(result),
                    )
                except Exception as _qvf_bloch_exc:
                    warn_output_failure(
                        _qvf_bloch_exc,
                        stem_sibling(output_stem, ".qvf"),
                        role="qvf_bloch_wavefunction_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

        # --- Population dump (Phase O6, periodic) ---------------------
        _qvf_pop = None
        _qvf_bond_orders = None
        _qvf_dipole = None
        # Periodic UHF/UKS results carry ``mo_coeffs_alpha`` / ``mo_coeffs_beta``
        # but not ``mo_coeffs``, so _has_valid_sidecar_mo_coeffs (which checks
        # both) gates the QVF population section for open-shell as well.
        if write_population_file and _has_valid_sidecar_mo_coeffs(result):
            try:
                from ._vibeqc_core import Molecule as _Mol  # noqa: F401

                mol_p = system.unit_cell_molecule()
                pop_result = result
                if resolved_jk == PeriodicJKMethod.BIPOLE:
                    from vibeqc.output.formats.population import (
                        compute_bipole_population_summary,
                        unsupported_population_summary,
                    )

                    try:
                        _bipole_pop_kmesh = (
                            kmesh
                            if "kmesh" in locals()
                            else _runner_bloch_kmesh(system, kpoints)
                        )
                        _qvf_pop = compute_bipole_population_summary(
                            result,
                            basis,
                            mol_p,
                            system,
                            lattice_options=opts.lattice_opts,
                            kmesh=_bipole_pop_kmesh,
                        )
                    except Exception as _bipole_pop_exc:
                        warn_output_failure(
                            _bipole_pop_exc,
                            stem_sibling(output_stem, ".population.txt"),
                            role="bipole_population_summary",
                            category=OutputFailureKind.compatibility_fallback,
                        )
                        _qvf_pop = unsupported_population_summary(
                            "periodic BIPOLE population properties are "
                            "unavailable because lattice-summed Mulliken "
                            "evaluation failed"
                        )
                elif resolved_jk == PeriodicJKMethod.GDF:
                    from vibeqc.output.formats.population import (
                        _compute_gdf_gamma_population_summary,
                        unsupported_population_summary,
                    )

                    try:
                        _qvf_pop = _compute_gdf_gamma_population_summary(
                            result, basis, mol_p,
                            nuclear_charges=_eff_z if _ecp_active else None,
                        )
                    except Exception as _gdf_pop_exc:
                        warn_output_failure(
                            _gdf_pop_exc,
                            stem_sibling(output_stem, ".population.txt"),
                            role="gdf_population_summary",
                            category=OutputFailureKind.compatibility_fallback,
                        )
                        _qvf_pop = unsupported_population_summary(
                            "periodic GDF populations require aligned accepted "
                            "Gamma density and overlap matrices"
                        )
                elif resolved_jk in (
                    PeriodicJKMethod.AICCM2026DEV_A,
                    PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
                ):
                    # M5. These results' orbitals span the supercell while the
                    # sidecar's molecule is the unit cell, and their
                    # density_alpha/_beta are None-valued FIELDS, so the
                    # molecular fallback below both compares the wrong objects
                    # and trips every hasattr open-shell test. Analyse the
                    # cyclic cluster directly instead.
                    from vibeqc.output.formats.population import (
                        compute_ccm_population_summary,
                        unsupported_population_summary,
                    )

                    try:
                        _qvf_pop = compute_ccm_population_summary(
                            result, mol_p,
                        )
                    except Exception as _ccm_pop_exc:
                        warn_output_failure(
                            _ccm_pop_exc,
                            stem_sibling(output_stem, ".population.txt"),
                            role="ccm_population_summary",
                            category=OutputFailureKind.compatibility_fallback,
                        )
                        _qvf_pop = unsupported_population_summary(
                            "periodic Gamma-CCM population properties are "
                            "unavailable because cyclic-cluster evaluation "
                            "failed"
                        )
                elif resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
                    from vibeqc.output.formats.population import (
                        compute_aiccm2026dev_b_population_summary,
                        unsupported_population_summary,
                    )

                    try:
                        _qvf_pop = compute_aiccm2026dev_b_population_summary(
                            result,
                            basis,
                            mol_p,
                            system,
                        )
                    except Exception as _aiccm_pop_exc:
                        warn_output_failure(
                            _aiccm_pop_exc,
                            stem_sibling(output_stem, ".population.txt"),
                            role="aiccm2026dev_b_population_summary",
                            category=OutputFailureKind.compatibility_fallback,
                        )
                        _qvf_pop = unsupported_population_summary(
                            "periodic AICCM2026DEV-B population properties "
                            "are unavailable because finite-torus population "
                            "evaluation failed"
                        )
                else:
                    # GPW / GAPW / RIJCOSX / legacy direct:
                    # extract a Gamma-point proxy so compute_population_summary
                    # sees a single real coefficient block.  Open-shell
                    # (UHF/UKS) results store ``mo_coeffs_alpha`` / ``_beta``
                    # lists; use _gamma_orbital_proxy which dispatches to
                    # the correct per-spin Gamma slice.
                    if isinstance(getattr(result, "mo_coeffs", None), (list, tuple)):
                        pop_result = _gamma_proxy_for_multi_k(result)
                    elif isinstance(getattr(result, "mo_coeffs_alpha", None), (list, tuple)):
                        pop_result = _gamma_orbital_proxy(result)
                _output_writer.dispatch_role(
                    "population",
                    result=pop_result,
                    basis=basis,
                    molecule=mol_p,
                    population_summary=_qvf_pop,
                    raise_on_error=True,
                )
                write(
                    f"  Population dump written to "
                    f"{output_stem.name}.population.txt + "
                    f"{output_stem.name}.population.json\n"
                )
                # Also compute summary for QVF atom_properties section.
                if output_qvf:
                    try:
                        if _qvf_pop is None:
                            from vibeqc.output.formats.population import (
                                compute_population_summary,
                            )

                            _qvf_pop = compute_population_summary(
                                pop_result,
                                basis,
                                mol_p,
                            )
                        # Extract bond-order table for bond_orders QVF section.
                        if _qvf_pop.mayer_bonds:
                            import math

                            _pos = [(a.xyz[0], a.xyz[1], a.xyz[2]) for a in mol_p.atoms]
                            _qvf_bond_orders = {
                                "method": "mayer",
                                "pairs": [
                                    {
                                        "i": int(i),
                                        "j": int(j),
                                        "order": float(order),
                                        "symbol_i": si,
                                        "symbol_j": sj,
                                        "distance_ang": float(
                                            math.dist(_pos[i], _pos[j]) * 0.529177210903
                                        ),
                                    }
                                    for (i, j, si, sj, order) in _qvf_pop.mayer_bonds
                                ],
                            }
                        # Extract dipole moment for manifest root metadata.
                        if _qvf_pop.dipole is not None:
                            _qvf_dipole = {
                                "total_debye": float(_qvf_pop.dipole["total_debye"]),
                                "vector_debye": [
                                    float(_qvf_pop.dipole["x_ebohr"]) * 2.541746473,
                                    float(_qvf_pop.dipole["y_ebohr"]) * 2.541746473,
                                    float(_qvf_pop.dipole["z_ebohr"]) * 2.541746473,
                                ],
                                "origin": [
                                    float(value)
                                    for value in _qvf_pop.dipole["origin_bohr"]
                                ],
                            }
                    except Exception as _qvf_pop_exc:
                        warn_output_failure(
                            _qvf_pop_exc,
                            stem_sibling(output_stem, ".qvf"),
                            role="qvf_population_prep",
                            category=OutputFailureKind.compatibility_fallback,
                        )
            except Exception as exc:
                warn(
                    f"population dump failed: {type(exc).__name__}: {exc}",
                    role="population",
                )
                warn_writer_failure(
                    exc,
                    stem_sibling(output_stem, ".population.txt"),
                    role="population_summary",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        # --- Geometry siblings (Phase O5) -----------------------------
        # Extended-XYZ (lattice in comment line), POSCAR, and the
        # structure-only XSF -- every viewer / chem-toolkit / fellow
        # QC code can consume at least one of these. Default-on with
        # opt-out via the matching kwarg; failures are best-effort
        # warnings so a finished SCF never gets dragged down by a
        # geometry writer.
        if write_xyz_file:
            xyz_path = stem_sibling(output_stem, ".xyz")
            try:
                energy_ha = float(getattr(result, "energy", float("nan")))
                if energy_ha != energy_ha:  # NaN sentinel
                    energy_ha = None
                _output_writer.dispatch_role(
                    "geometry",
                    only_format="extended-xyz",
                    system=system,
                    energy_ha=energy_ha,
                    raise_on_error=True,
                )
                write(
                    f"  Final geometry written to {xyz_path.name} "
                    f"(extended XYZ with lattice)\n"
                )
            except Exception as exc:
                warn(
                    f"extended-xyz write failed: {type(exc).__name__}: {exc}",
                    role="extended_xyz",
                )
                warn_writer_failure(
                    exc,
                    xyz_path,
                    role="extended_xyz",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        if write_poscar_file:
            poscar_path = stem_sibling(output_stem, ".POSCAR")
            try:
                _output_writer.dispatch_role(
                    "geometry",
                    only_format="poscar",
                    system=system,
                    comment=f"vibe-qc periodic {label} basis={basis.name}",
                    raise_on_error=True,
                )
                write(f"  Structure written to {poscar_path.name} (VASP-5 POSCAR)\n")
            except Exception as exc:
                warn(
                    f"POSCAR write failed: {type(exc).__name__}: {exc}",
                    role="poscar",
                )
                warn_writer_failure(
                    exc,
                    poscar_path,
                    role="poscar",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        if write_cif_file:
            cif_path = stem_sibling(output_stem, ".cif")
            try:
                _output_writer.dispatch_role(
                    "geometry",
                    only_format="cif",
                    system=system,
                    comment=f"vibe-qc periodic {label} basis={basis.name}",
                    raise_on_error=True,
                )
                write(f"  Structure written to {cif_path.name} (CIF, P 1)\n")
            except Exception as exc:
                warn(f"CIF write failed: {type(exc).__name__}: {exc}", role="cif")
                warn_writer_failure(
                    exc,
                    cif_path,
                    role="cif",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        # --- Citations (Phase O5b) ------------------------------------
        # Assemble the citation list for this periodic job (software +
        # libint + libxc-if-DFT + basis-set + functional + DIIS +
        # spglib + ECP-if-used) and emit {stem}.bibtex / .references
        # siblings plus a "## References" block in the .out so the
        # text log is self-contained. Failures are non-fatal -- the
        # SCF result is the load-bearing artefact, citations are
        # observability.
        cite_block_text: str | None = None
        _bibtex_content: str = ""
        # Full assembled provenance rows destined for the .system
        # manifest's [citations] section (CLAUDE.md Sec.8.3/Sec.8.5).
        # The upfront OutputWriter receives these once assembly completes.
        cite_manifest_rows: list[dict[str, Any]] = []
        if citations:
            try:
                _db = load_default_database()
                # Symmetry-unfolded IBZ exchange (Pisani/Dovesi star
                # transport): the driver flags it on the result when the
                # full-range or erfc-screened exchange arm ran with a
                # symmetry-reduced mesh.
                if getattr(
                    result, "used_kpoint_symmetry_unfolding", False
                ) and "kpoint_symmetry_unfolding" not in _numerics:
                    _numerics.append("kpoint_symmetry_unfolding")
                # FFT-Poisson backend uses FFTW3 (fftw_plan_* /
                # fftw_execute in cpp/src/fft_poisson.cpp). The native
                # EWALD_3D ``fft_poisson`` method now defaults to the
                # analytical AO-pair-FT Hartree J path; it only exercises
                # FFTW3 when the legacy grid backend is explicitly
                # restored for diagnostics.
                _j_ewald_backend = os.environ.get(
                    "VIBEQC_J_EWALD3D_BACKEND", "analytic_ft"
                ).lower()
                _uses_fftw = resolved_jk in (
                    PeriodicJKMethod.GPW,
                    PeriodicJKMethod.GAPW,
                ) or (
                    resolved_jk == PeriodicJKMethod.FFT_POISSON
                    and _j_ewald_backend == "grid"
                )
                _uses_ewald_ao_ft = (
                    resolved_jk == PeriodicJKMethod.FFT_POISSON
                    and _j_ewald_backend != "grid"
                )
                _uses_gpw = resolved_jk == PeriodicJKMethod.GPW
                _uses_gapw = resolved_jk == PeriodicJKMethod.GAPW
                # GDF citation gating. The Sun-Berkelbach 2017 + MD78
                # + HJO00 stack fires whenever the GDF path runs. The
                # rsgdf sub-route in the database is reserved for the
                # GDF chat to extend with Ye-Berkelbach 2021 -- until
                # then it carries the same row as ``gdf`` and the runner
                # always uses the ``gdf`` flag (the periodic-runner
                # surface doesn't currently expose which gdf_method ran).
                _aiccm_backend_name = aiccm_backend.strip().lower().replace("-", "_")
                # The real-Γ direct route executes the GDF stack: its neutral
                # supercell cderi is the multi-k GDF's own per-(k_a,k_b)
                # unit-cell fits (cderi_build="fold", periodic/ccm/direct.py ->
                # neutral.py build_lpq_bloch_native_fft) and its exchange-q0
                # seam is periodic_k_gdf._madelung_for_kmesh, so it owes the
                # GDF citations exactly as the chi RI backend does.
                _uses_gdf = resolved_jk in (
                    PeriodicJKMethod.GDF,
                    PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
                    # The Bloch producer IS the multi-k GDF stack on the
                    # torus mesh, so it owes the GDF citation.
                    PeriodicJKMethod.NEUTRAL_BLOCH,
                ) or (
                    resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
                    and _aiccm_backend_name != "four_center"
                )
                _uses_gdf_2d = (
                    resolved_jk == PeriodicJKMethod.GDF
                    and int(system.dim) == 2
                )
                _uses_bipole = resolved_jk == PeriodicJKMethod.BIPOLE or (
                    resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
                    and _aiccm_backend_name == "four_center"
                )
                # RIJCOSX acceleration citations (Neese 2009, Izsak-Neese
                # 2011, Helmich-Paris 2021) fire for the dedicated RIJCOSX
                # route (Gamma builders + the multi-k GDF/COSX backend)
                # and for the AICCM2026DEV_B rijcosx backend.
                _jk_acceleration = (
                    ("rijcosx",)
                    if resolved_jk == PeriodicJKMethod.RIJCOSX
                    or (
                        resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
                        and _aiccm_backend_name == "rijcosx"
                    )
                    else ()
                )
                # Cite the concrete guess that ran, not the user's spelling.
                # AUTO includes the resolved route capabilities. A runner-owned
                # GPW/GAPW restart density supersedes the selector and carries
                # no construction-method citation.
                _periodic_guess_routes = {
                    InitialGuess.SAD: "sad",
                    InitialGuess.SAP: "sap",
                    InitialGuess.HUECKEL: "huckel",
                    InitialGuess.PATOM: "patom",
                    InitialGuess.MINAO: "minao",
                }
                _scf_guess = (
                    None
                    if restart_from is not None
                    else _periodic_guess_routes.get(_resolved_initial_guess)
                )
                # Smearing flavour. Fermi-Dirac, Mermin, Methfessel-Paxton,
                # and Marzari-Vanderbilt are all implemented (unsupported
                # flavours raise NotImplementedError above, before any SCF
                # runs).  The citation route keys are the canonical spellings;
                # Fermi-Dirac is skipped by assemble() and its Mermin
                # citation comes from uses_smearing.
                _smearing_method = None
                if opts.smearing_temperature > 0.0:
                    _smearing_method = (
                        str(smearing_method_label).strip().lower().replace("-", "_")
                    )
                # COOP/COHP bonding analysis (Hughbanks-Hoffmann COOP,
                # Dronskowski-Bloechl COHP, LOBSTER AO projection). The
                # analysis itself runs in the DOS/QVF block further down
                # (it needs the lattice matrices assembled there), so gate
                # the citation on the exact conditions that gate that
                # block: coop_cohp requested, QVF output on, SCF
                # converged. Both routes fire together -- the runner
                # requests include_cohp, so COHP is computed alongside
                # COOP whenever the analysis runs.
                _props: list[str] = []
                if coop_cohp and output_qvf and result.converged:
                    _props = ["coop", "cohp"]
                if output_qvf and result.converged and resolved_jk == PeriodicJKMethod.GDF:
                    _props.append("mayer_bond_order")
                # Analytic forces on a real-Γ relaxation: the optimizer block
                # below steps run_ccm_direct_optimize on run_ccm_direct_gradient
                # (Pulay 1969 + Hellmann-Feynman via routes.drivers.gradient).
                # Decided from the request, like the molecular runner, because
                # this surface is emitted before the optimizer runs. BIPOLE and
                # GDF relaxations do not pass the flag today (pre-existing gap,
                # tracked separately; not widened in M0).
                _uses_rg_gradient = bool(
                    optimize
                    and result.converged
                    and resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA
                )
                # M4c (#778): when a correlation driver ran, the citation
                # owed is ITS stamped compound key, not the SCF's bare route
                # row -- the whole point of stamping per call is that one
                # result class serves two lineages. Falls back to the SCF
                # route whenever nothing correlated (the ordinary case).
                _corr_backend = str(
                    getattr(getattr(result, "correlation", None), "backend", "")
                    or ""
                )
                _refs = _db.assemble(
                    # The routes.methods key: the correlation driver's own
                    # stamp when one ran, else the chi line's bare key, the
                    # real-Γ producer's variant name (both neutral producers
                    # share the CCM lineage row, ruling R1), else the SCF method.
                    method=(
                        _corr_backend
                        if _corr_backend
                        else "aiccm2026dev-a"
                        if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A
                        else "aiccm2026dev-b"
                        if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B
                        else "real-gamma"
                        if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA
                        else "neutral-bloch"
                        if resolved_jk == PeriodicJKMethod.NEUTRAL_BLOCH
                        else method_upper
                    ),
                    basis=basis.name,
                    functional=functional,
                    extra_libraries=["trexio"] if _trexio_path is not None else [],
                    periodic=True,  # => spglib fires
                    uses_ecp=bool(_ecp_blocks),
                    uses_fftw_poisson=_uses_fftw,
                    uses_smearing=opts.smearing_temperature > 0.0,
                    # Saunders-Hillier level shift (constant, warm-up, or an
                    # explicit level_shift_schedule) is a cited convergence
                    # technique.
                    uses_level_shift=(
                        float(getattr(opts, "level_shift", 0.0) or 0.0) != 0.0
                        or any(
                            float(s) != 0.0
                            for s in (getattr(opts, "level_shift_schedule", None) or [])
                        )
                    ),
                    smearing_method=_smearing_method,
                    scf_guess=_scf_guess,
                    # Cite the accelerator this run actually used. Without
                    # this the default fires and every Anderson/Broyden/
                    # Kerker run cited Pulay DIIS instead
                    # (PERIODIC-CITES-DIIS-IT-DID-NOT-USE).
                    scf_accelerator=citation_scf_accelerator(
                        density_mixer, density_mixer_kerker
                    ),
                    dft_plus_u=bool(dft_plus_u),
                    uses_slab_ewald_2d=(
                        resolved_jk == PeriodicJKMethod.SLAB_EWALD_2D
                        or _uses_gdf_2d
                    ),
                    uses_gpw=_uses_gpw,
                    uses_gapw=_uses_gapw,
                    uses_gdf=_uses_gdf and not _uses_gdf_2d,
                    uses_gdf_2d=_uses_gdf_2d,
                    uses_bipole=_uses_bipole,
                    # Charge-pair Schwarz SR screening. M5
                    # enables it with the default padded image domain.
                    uses_bipole_sr_range=(
                        _uses_bipole
                        and bool(
                            getattr(
                                getattr(opts, "lattice_opts", None),
                                "sr_range_screening",
                                False,
                            )
                        )
                    ),
                    uses_ewald_ao_ft=_uses_ewald_ao_ft,
                    uses_gradient=_uses_rg_gradient,
                    uses_ml_kpredictor=_kpts_uses_ml,
                    acceleration=_jk_acceleration,
                    properties=_props,
                    numerics=_numerics,
                )
                cite_manifest_rows = citation_manifest_rows(_refs)
                _output_writer.dispatch_role(
                    "citations",
                    citations=_refs,
                    raise_on_error=True,
                )
                cite_block_text = format_references_block(_refs)
                _bibtex_content = format_bibtex(_refs)
                bibtex_path = stem_sibling(output_stem, ".bibtex")
                write(
                    f"  Citations written to {bibtex_path.name} + "
                    f"{stem_sibling(output_stem, '.references').name}\n"
                )
            except Exception as exc:
                warn(
                    f"citation emission failed: {type(exc).__name__}: {exc}",
                    role="citations",
                )
                warn_writer_failure(
                    exc,
                    stem_sibling(output_stem, ".bibtex"),
                    role="citations",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        if write_xsf_structure_file:
            xsf_struct_path = stem_sibling(output_stem, ".xsf")
            try:
                # When write_density=True we'd collide with the
                # density XSF below; route the structure-only XSF to
                # a different suffix in that case.
                if write_density:
                    xsf_struct_path = stem_sibling(output_stem, 
                        ".structure.xsf",
                    )
                _output_writer.dispatch_role(
                    "geometry",
                    only_format="xsf",
                    system=system,
                    raise_on_error=True,
                )
                write(
                    f"  Structure written to {xsf_struct_path.name} "
                    f"(XSF crystal block)\n"
                )
            except Exception as exc:
                warn(
                    f"XSF structure write failed: {type(exc).__name__}: {exc}",
                    role="xsf_structure",
                )
                warn_writer_failure(
                    exc,
                    xsf_struct_path,
                    role="xsf_structure",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        # --- Density XSF + grid for QVF -------------------------------
        # Compute the primitive-cell density grid once; reuse it for
        # both the XSF writer and the QVF archive (when requested).
        # The grid origin is (0,0,0) and the span is the lattice-
        # vector transpose (rows = spanning vectors), matching the XSF
        # convention.  All units are bohr -- write_qvf stores them
        # as-is in the grid descriptor (see _grid_descriptor).
        _qvf_volume_data = None
        _qvf_system = system
        _qvf_mo_data = None
        _qvf_extensions = None
        _qvf_vendor_json_sections = None
        if output_qvf and resolved_jk == PeriodicJKMethod.NEUTRAL_BLOCH:
            _qvf_vendor_json_sections = _neutral_bloch_qvf_vendor_sections(
                result)
            if _qvf_vendor_json_sections:
                _qvf_extensions = _qvf_extensions_with(
                    _qvf_extensions,
                    "x_vibeqc",
                )
        if (
            output_qvf
            and resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA
        ):
            _qvf_vendor_json_sections = _real_gamma_qvf_vendor_sections(result)
            if _qvf_vendor_json_sections:
                _qvf_extensions = _qvf_extensions_with(
                    _qvf_extensions,
                    "x_vibeqc",
                )
        if output_qvf and resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
            _qvf_vendor_json_sections = _aiccm_b_qvf_vendor_sections(result)
            if _qvf_vendor_json_sections:
                _qvf_extensions = _qvf_extensions_with(
                    _qvf_extensions,
                    "x_vibeqc",
                )
        if output_qvf and qvf_wannier_centers:
            try:
                if resolved_jk != PeriodicJKMethod.AICCM2026DEV_B:
                    # Gamma-only route (the gate above rejects multi-k):
                    # the occupied Bloch functions span the cell's whole
                    # occupied space, so localizing them is a well-defined
                    # Wannier construction for that cell.
                    from .periodic_localise import localise_periodic_gamma

                    center_sections = _gamma_qvf_wannier_center_sections(
                        localise_periodic_gamma(result, basis, system)
                    )
                else:
                    if hasattr(result, "mo_coeffs_alpha"):
                        from .periodic.chi.localization import (
                            localize_aiccm2026dev_b_unrestricted_occupied,
                        )

                        localization = (
                            localize_aiccm2026dev_b_unrestricted_occupied(
                                result,
                                system,
                                basis,
                            )
                        )
                    else:
                        from .periodic.chi.localization import (
                            localize_aiccm2026dev_b_occupied,
                        )

                        localization = localize_aiccm2026dev_b_occupied(
                            result,
                            system,
                            basis,
                        )
                    center_sections = _aiccm_b_qvf_wannier_center_sections(
                        localization,
                    )
                if center_sections:
                    _qvf_vendor_json_sections = list(
                        _qvf_vendor_json_sections or []
                    )
                    _qvf_vendor_json_sections.extend(center_sections)
                    _qvf_extensions = _qvf_extensions_with(
                        _qvf_extensions,
                        "x_ccm",
                    )
            except Exception as _qvf_wannier_exc:
                warn(
                    "χ-CCM-B QVF Wannier-centre overlay failed: "
                    f"{type(_qvf_wannier_exc).__name__}: {_qvf_wannier_exc}",
                    role="qvf_wannier_overlay",
                )
                warn_output_failure(
                    _qvf_wannier_exc,
                    stem_sibling(output_stem, ".qvf"),
                    role="qvf_wannier_centers",
                    category=OutputFailureKind.optional_artifact,
                )
        _needs_density_grid = write_density or output_qvf
        if _needs_density_grid:
            try:
                L_bohr = np.asarray(system.lattice, dtype=float)
                density_write_system = system
                shape = tuple(
                    max(
                        1,
                        int(
                            np.ceil(np.linalg.norm(L_bohr[:, i]) / density_spacing_bohr)
                        ),
                    )
                    for i in range(3)
                )
                if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
                    diag = getattr(result, "aiccm2026dev_b", None)
                    mesh = tuple(
                        int(value)
                        for value in getattr(diag, "mesh", (1, 1, 1))
                    )
                    super_system = _aiccm_b_qvf_supercell_system(system, mesh)
                    super_basis = BasisSet(
                        super_system.unit_cell_molecule(),
                        basis.name,
                    )
                    L_bohr = np.asarray(super_system.lattice, dtype=float)
                    shape = tuple(
                        max(
                            1,
                            int(
                                np.ceil(
                                    np.linalg.norm(L_bohr[:, i])
                                    / density_spacing_bohr
                                )
                            ),
                        )
                        for i in range(3)
                    )
                    density_matrix = _aiccm_b_full_density_matrix_for_qvf(
                        result,
                        system,
                        mesh,
                    )
                    if density_matrix.shape != (
                        int(super_basis.nbasis),
                        int(super_basis.nbasis),
                    ):
                        raise ValueError(
                            "aiccm2026dev-b QVF supercell density shape "
                            f"{density_matrix.shape!r} does not match "
                            f"supercell basis size {int(super_basis.nbasis)}"
                        )
                    rho, per_voxel = _evaluate_density_matrix_on_lattice_grid(
                        density_matrix,
                        super_basis,
                        L_bohr,
                        shape,
                        system=super_system,
                    )
                    _qvf_system = super_system
                    density_write_system = super_system
                    if output_qvf:
                        try:
                            _qvf_mo_data = _aiccm_b_qvf_gamma_orbital_grids(
                                result,
                                super_basis,
                                super_system,
                                L_bohr,
                                shape,
                                mesh,
                            )
                        except Exception as _qvf_mo_exc:
                            warn(
                                "periodic QVF orbital grid preparation "
                                f"failed: {type(_qvf_mo_exc).__name__}: "
                                f"{_qvf_mo_exc}",
                                role="qvf_orbital_grid_prep",
                            )
                            warn_output_failure(
                                _qvf_mo_exc,
                                stem_sibling(output_stem, ".qvf"),
                                role="qvf_orbital_grid_prep",
                                category=OutputFailureKind.compatibility_fallback,
                            )
                else:
                    rho, per_voxel = _exact_periodic_density_grid_artifact(
                        basis,
                        system,
                        result,
                        lattice_bohr=L_bohr,
                        grid_shape=shape,
                        spacing_bohr=density_spacing_bohr,
                        gamma_only=_requested_gamma_only,
                        expected_electrons=float(
                            int(system.n_electrons()) - int(_total_ncore or 0)
                        ),
                        bvk_mesh=tuple(
                            int(value)
                            for value in getattr(
                                _requested_bloch_kmesh,
                                "mesh",
                                (),
                            )
                        ),
                    )

                # --- XSF output ---
                if write_density:
                    _output_writer.dispatch_role(
                        "density",
                        only_format="xsf",
                        system=density_write_system,
                        data=rho,
                        name=f"{output_stem.name}_density",
                        raise_on_error=True,
                    )
                    write(f"  Density written to {xsf_path.name}\n")

                # --- Package for QVF ----------------------------------
                # QVF writes grid in bohr.  Origin is (0,0,0) for a
                # primitive-cell grid.  voxel_vectors are per-voxel
                # step vectors: lattice column / shape_i.
                if output_qvf:
                    _qvf_volume_data = {
                        "Electron density": (
                            rho,
                            np.zeros(3, dtype=float),
                            per_voxel,
                        ),
                    }
            except Exception as exc:
                warn(
                    f"density grid evaluation failed: {type(exc).__name__}: {exc}",
                    role="density_grid",
                )
                if write_density:
                    warn_writer_failure(
                        exc,
                        xsf_path,
                        role="density_grid",
                        category=OutputFailureKind.optional_artifact,
                        writer=_output_writer,
                    )
                if output_qvf:
                    warn_output_failure(
                        exc,
                        stem_sibling(output_stem, ".qvf"),
                        role="qvf_density_grid",
                        category=OutputFailureKind.compatibility_fallback,
                    )
        # --- Timing ---------------------------------------------------
        t_total = time.perf_counter() - t_job_start
        n_iter = int(getattr(result, "n_iter", 0))
        iter_avg = (t_scf / n_iter) if n_iter > 0 else float("nan")
        write(
            "\n"
            + _format_timing_summary(
                [
                    ("SCF total", t_scf, None),
                    ("SCF avg per iter", iter_avg, f"({n_iter} iters)"),
                    ("Job total", t_total, None),
                ],
                label_width=28,
                body_indent=2,
            )
            + "\n"
        )
        flush()

        # --- TD-DFT excited states (Gamma-point, TDA) -------------------
        if tddft and _has_valid_mo_coeffs(result):
            try:
                from vibeqc.tddft import (
                    run_tddft_tda_periodic as _run_td_periodic,
                )

                # Extract n_occ from the result.
                _n_occ = None
                _occ = getattr(result, "occupations", None)
                if _occ is not None:
                    if isinstance(_occ, (list, tuple)):
                        _occ = _occ[0]
                    _occ = np.asarray(_occ, dtype=float)
                    _n_occ = int(np.sum(_occ > 1.0 - 1e-8))
                if _n_occ is None:
                    _n_el = getattr(result, "n_electrons", 0)
                    _n_occ = _n_el // 2 if _n_el else 1

                _td = _run_td_periodic(
                    result,
                    basis,
                    n_occ=_n_occ,
                    n_states=tddft_n_states,
                    functional=functional,
                )
                _td_func = f" ({functional})" if functional else ""
                write(
                    f"\n  ## TD-DFT excited states"
                    f" (TDA{_td_func}, Gamma-point)\n"
                    f"  {'─' * 50}\n"
                )
                write(
                    f"  {'State':>6s}  {'E (eV)':>10s}"
                    f"  {'λ (nm)':>10s}  {'f_osc':>10s}"
                    f"  {'Dominant transition'}\n"
                )
                for _st in _td.states:
                    _dom_str = ", ".join(
                        f"{occ}→{virt} ({abs(amp):.3f})"
                        for occ, virt, amp in _st.dominant_amplitudes[:3]
                    )
                    write(
                        f"  {_st.index:>6d}  "
                        f"{_st.excitation_energy_ev:>10.4f}  "
                        f"{_st.wavelength_nm:>10.1f}  "
                        f"{_st.oscillator_strength:>10.4f}  "
                        f"{_dom_str}\n"
                    )
                write("\n")
                flush()
            except Exception as _td_per_exc:
                write(
                    f"\n  ## TD-DFT excited states\n"
                    f"  {'─' * 50}\n"
                    f"  FAILED: {type(_td_per_exc).__name__}:"
                    f" {_td_per_exc}\n"
                )
                flush()

        # --- References block (Phase O5b) -----------------------------
        # Embed the assembled citation list in the .out so the text
        # log is self-contained -- a user reading the .out doesn't have
        # to chase the .bibtex / .references siblings to know what to
        # cite. Same content that lives in the .references file, hard-
        # wrapped to match the SCF-trace layout. Emitted through the
        # citation printer's channel writer (a first-class logger op).
        if cite_block_text:
            write_references_block(block=cite_block_text)
            flush()

        # --- Harmonic vibrational analysis (finite-difference Hessian) ---
        if hessian:
            _scf_converged = bool(getattr(result, "converged", False))
            if not _scf_converged:
                write(
                    "\n"
                    + section_header("## Vibrational Frequencies")
                    + "  SKIPPED -- SCF did not converge.\n"
                )
                flush()
            else:
                try:
                    from ._vibeqc_core import (
                        RHFOptions,
                        RKSOptions,
                        UHFOptions,
                        UKSOptions,
                    )
                    from .hessian import (
                        HessianFDOptions,
                        compute_hessian_fd,
                        ir_intensities,
                    )

                    if method_upper == "RHF":
                        _hess_scf_opts = RHFOptions()
                    elif method_upper == "UHF":
                        _hess_scf_opts = UHFOptions()
                    elif method_upper == "RKS":
                        _hess_scf_opts = RKSOptions()
                        _hess_scf_opts.functional = str(functional)
                    elif method_upper == "UKS":
                        _hess_scf_opts = UKSOptions()
                        _hess_scf_opts.functional = str(functional)
                    else:
                        raise ValueError(
                            f"Hessian not available for method={method_upper}"
                        )

                    _hess_opts = HessianFDOptions(
                        include_dipole_derivatives=True,
                        frozen_indices=hessian_frozen_indices,
                    )
                    _uc_mol = system.unit_cell_molecule()
                    hessian_result = compute_hessian_fd(
                        _uc_mol,
                        basis.name,
                        method=method_upper,
                        scf_options=_hess_scf_opts,
                        hessian_options=_hess_opts,
                    )
                    write(
                        "\n"
                        + section_header("## Vibrational Frequencies")
                        + f"  Finite-difference Hessian (unit cell)"
                        f"  (step = {_hess_opts.step_bohr:.3f} bohr,"
                        f"  {hessian_result.n_displacements} displacements)\n"
                        f"  Imaginary modes: {hessian_result.imaginary_count}\n"
                        f"  Linear molecule: {hessian_result.is_linear}\n\n"
                    )

                    _n_skip = 5 if hessian_result.is_linear else 6
                    _freqs = hessian_result.frequencies_cm1
                    _ir = None
                    try:
                        _ir = ir_intensities(hessian_result)
                    except Exception as _ir_exc:
                        write(
                            f"  (warning: IR intensities not available: "
                            f"{type(_ir_exc).__name__}: {_ir_exc})\n"
                        )

                    # Headered column table via Table (content-sized rule),
                    # mirroring the molecular frequency table.
                    _freq_cols = [
                        Column("Mode", "<"),
                        Column("Freq/cm\u207b\u00b9", ">"),
                    ]
                    if _ir is not None:
                        _freq_cols.append(Column("IR/(km/mol)", ">"))
                    _freq_tbl = Table(_freq_cols)
                    for k in range(_n_skip, len(_freqs)):
                        _label = f"{k - _n_skip + 1}"
                        _freq = _freqs[k]
                        if _freq < 0:
                            _freq_str = f"{render_frequency(abs(_freq))}i"
                        else:
                            _freq_str = render_frequency(_freq)
                        if _ir is not None:
                            _freq_tbl.add_row(_label, _freq_str, f"{_ir[k]:.2f}")
                        else:
                            _freq_tbl.add_row(_label, _freq_str)
                    write(_freq_tbl.render() + "\n\n")
                    flush()
                except Exception as _hess_exc:
                    write(
                        "\n"
                        + section_header("## Vibrational Frequencies")
                        + f"  FAILED: {type(_hess_exc).__name__}: {_hess_exc}\n"
                    )
                    flush()
                    hessian_result = None

    _output_writer.record(out_path, wall_time_s=t_total)

    # Fill late-bound execution and finite-torus fields while preserving the
    # upfront ``status = running`` manifest and all dispatch outcomes.
    # IID 344: the executed low-level backend identity and the structured
    # parity-hold state are reproducibility facts of this run -- carried as
    # result-struct fields through the existing [run] extension point
    # (``update_run_fields``) so a held absolute energy is visible in the
    # .system manifest, not only in the .err sidecar.
    _executed_backend = str(
        getattr(result, "runtime_backend", None)
        or getattr(result, "backend", "")
        or ""
    )
    _manifest_run_fields = {
        "initial_guess_requested": (InitialGuess.FRAGMO if _fragmo_request else _requested_initial_guess).name,
        "initial_guess": getattr(result, "guess_selection", selection).effective.name,
        "initial_guess_transport": getattr(result, "guess_selection", selection).transport.name,
        "exchange_q0": exchange_q0_label(exchange_exxdiv),
        "jk_method_requested": _jk_requested_label,
        "jk_method_resolved": resolved_jk.value,
        "jk_method_executed": _executed_jk_method,
        "backend": _executed_backend,
        "parity_held": bool(
            getattr(
                result,
                "parity_held",
                "+PARITY_HELD" in _executed_backend,
            )
        ),
        "dft_plus_u": bool(dft_plus_u),
        "dft_plus_u_route": _dft_plus_u_route,
    }
    # M8 provenance: ``aiccm_correlation`` records what was ASKED for
    # ("ccsd"); this records what was actually CITED. The two differ in
    # exactly the information a consumer cannot re-derive from the request:
    # the lineage (``-ri-`` neutral fitted torus vs the bare four-centre row,
    # ruling R1) and whether the perturbative triples ran, since
    # ``compute_triples`` is a runtime flag. Written only when a correlation
    # actually produced a stamp, so an SCF-only run carries no key and an
    # absent one is never a claim.
    _aiccm_corr_result = getattr(result, "correlation", None)
    if _aiccm_corr_result is not None:
        _aiccm_corr_route = str(getattr(_aiccm_corr_result, "backend", "") or "")
        if _aiccm_corr_route:
            _manifest_run_fields["aiccm_correlation_route"] = _aiccm_corr_route
    if resolved_jk == PeriodicJKMethod.GAPW:
        _manifest_run_fields["gapw_one_centre_resolved"] = getattr(
            result,
            "one_centre",
            "analytic" if method_upper in ("RHF", "UHF") else "block",
        )
        _manifest_run_fields["gapw_molecular_limit_declared"] = bool(
            getattr(
                result,
                "molecular_limit_declared",
                gapw_molecular_limit,
            )
        )
    if resolved_jk == PeriodicJKMethod.AICCM2026DEV_B:
        _b_convention = getattr(result, "finite_torus_convention", None)
        _b_exchange_q0 = getattr(_b_convention, "exchange_q0", None)
        _manifest_run_fields["exchange_q0"] = (
            "not-recorded"
            if _b_exchange_q0 is None
            else exchange_q0_label(_b_exchange_q0)
        )
        _manifest_run_fields["ccm_approach"] = getattr(
            _b_convention,
            "ccm_approach",
            "not-recorded",
        )
        _manifest_run_fields["ccm_construction"] = getattr(
            _b_convention,
            "ccm_construction",
            "not-recorded",
        )
        _manifest_run_fields["evaluation_representation"] = getattr(
            _b_convention,
            "evaluation_representation",
            "not-recorded",
        )
        _manifest_run_fields["exchange_q0_applicability"] = getattr(
            _b_convention,
            "exchange_q0_applicability",
            "not-recorded",
        )
        _b_exact_exchange = getattr(
            result,
            "exact_exchange_assembly",
            None,
        )
        _manifest_run_fields["exact_exchange_assembly_schema"] = getattr(
            _b_exact_exchange,
            "schema",
            "not-recorded",
        )
        _manifest_run_fields["exact_exchange_resolver"] = getattr(
            _b_exact_exchange,
            "resolver",
            "not-recorded",
        )
        _manifest_run_fields["exact_exchange_c_full"] = getattr(
            _b_exact_exchange,
            "c_full",
            "not-recorded",
        )
        _manifest_run_fields["exact_exchange_c_sr"] = getattr(
            _b_exact_exchange,
            "c_sr",
            "not-recorded",
        )
        _manifest_run_fields["exact_exchange_omega_screen_bohr_inv"] = getattr(
            _b_exact_exchange,
            "omega_screen_bohr_inv",
            "not-recorded",
        )
        _manifest_run_fields["screened_exchange_applicability"] = getattr(
            _b_exact_exchange,
            "screened_exchange_applicability",
            "not-recorded",
        )
        _manifest_run_fields["screened_exchange_assembly"] = getattr(
            _b_exact_exchange,
            "screened_exchange_assembly",
            "not-recorded",
        )
    if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A:
        # EXPERIMENTAL four-centre construction. Unlike the two neutral
        # producers this records what was BUILT, so ccm_construction names the
        # union-and-weight lineage instead of "representation control", and
        # the weighting actually executed is recorded because the library
        # default (union12) differs from what the front door runs.
        _fc_convention = getattr(result, "four_center", None)
        _fc = lambda field: getattr(_fc_convention, field, "not-recorded")
        _manifest_run_fields["exchange_q0"] = _fc("exchange_q0")
        _manifest_run_fields["exchange_q0_applicability"] = _fc(
            "exchange_q0_applicability")
        _manifest_run_fields["ccm_construction"] = _fc("ccm_construction")
        _manifest_run_fields["ccm_four_centre_weighting"] = _fc("weighting")
        _manifest_run_fields["ccm_four_centre_contraction"] = _fc(
            "four_center_contraction")
        _manifest_run_fields["ccm_four_centre_builder"] = _fc("builder")
        _manifest_run_fields["evaluation_representation"] = (
            "real supercell-Gamma eigenproblem")
        _manifest_run_fields["lattice_vector_convention"] = _fc(
            "lattice_vector_convention")
        _manifest_run_fields["executed_fock_mixing"] = _fc(
            "executed_fock_mixing")
        _manifest_run_fields["executed_damping"] = _fc("executed_damping")
        _manifest_run_fields["executed_level_shift"] = _fc(
            "executed_level_shift")
        _manifest_run_fields["bvk_nrep"] = list(
            _fc("nrep")) if _fc_convention is not None else "not-recorded"
        _manifest_run_fields["bvk_n_cells"] = _fc("n_cells")
        _manifest_run_fields["experimental_route"] = True
    if resolved_jk == PeriodicJKMethod.NEUTRAL_BLOCH:
        # EXPERIMENTAL neutral Bloch producer: every value is READ off the
        # producer's record (D86), never re-derived here. ``ccm_construction``
        # is byte-identical to the real-Γ block below because the two
        # producers evaluate the same neutral Hamiltonian (ruling R1).
        _nb_convention = getattr(result, "neutral_bloch", None)
        _nb = lambda field: getattr(_nb_convention, field, "not-recorded")
        _manifest_run_fields["exchange_q0"] = (
            _nb("exchange_q0") or "not-applicable"
        )
        _manifest_run_fields["exchange_q0_applicability"] = _nb(
            "exchange_q0_applicability")
        _manifest_run_fields["ccm_construction"] = (
            "none (neutral fitted-torus representation control)")
        _manifest_run_fields["evaluation_representation"] = (
            "Bloch representation: Gamma-centred BvK k-mesh eigenproblems "
            "(multi-k GDF)")
        _manifest_run_fields["lattice_vector_convention"] = _nb(
            "lattice_vector_convention")
        _manifest_run_fields["executed_fock_mixing"] = _nb(
            "executed_fock_mixing")
        _manifest_run_fields["executed_level_shift"] = _nb(
            "executed_level_shift")
        _manifest_run_fields["bvk_nrep"] = list(
            _nb("nrep")) if _nb_convention is not None else "not-recorded"
        _manifest_run_fields["bvk_n_cells"] = _nb("n_cells")
        # The fit accounting the producer records: what makes the two
        # producers' reciprocal support auditable against each other.
        _manifest_run_fields["lpq_pair_symmetry"] = _nb("pair_symmetry")
        _manifest_run_fields["gdf_pair_builds"] = _nb("gdf_pair_builds")
        _manifest_run_fields["gdf_pair_total"] = _nb("gdf_pair_total")
        _manifest_run_fields["gdf_pair_reduction_factor"] = _nb(
            "gdf_pair_reduction_factor")
        _nb_tail = _nb("rsgdf_tail_ke_cutoff")
        _manifest_run_fields["rsgdf_tail_ke_cutoff_executed"] = (
            "none" if _nb_tail is None else _nb_tail)
        _manifest_run_fields["experimental_route"] = True
    if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA:
        # EXPERIMENTAL real-Γ control: record the RESULT-derived convention
        # (never the requested label -- for a screened hybrid the seam is
        # structurally inactive regardless of exchange_exxdiv) plus the D86
        # executed values and the lattice-vector convention.
        _rg_convention = getattr(result, "real_gamma", None)
        _rg = lambda field: getattr(_rg_convention, field, "not-recorded")
        _manifest_run_fields["exchange_q0"] = (
            _rg("exchange_q0") or "not-applicable"
        )
        _manifest_run_fields["exchange_q0_applicability"] = _rg(
            "exchange_q0_applicability")
        _manifest_run_fields["ccm_construction"] = (
            "none (neutral fitted-torus representation control)")
        _manifest_run_fields["evaluation_representation"] = (
            "real supercell-Gamma eigenproblem")
        _manifest_run_fields["lattice_vector_convention"] = _rg(
            "lattice_vector_convention")
        _manifest_run_fields["executed_fock_mixing"] = _rg(
            "executed_fock_mixing")
        _manifest_run_fields["executed_damping"] = _rg("executed_damping")
        _manifest_run_fields["executed_level_shift"] = _rg(
            "executed_level_shift")
        _manifest_run_fields["bvk_nrep"] = list(
            _rg("nrep")) if _rg_convention is not None else "not-recorded"
        _manifest_run_fields["bvk_n_cells"] = _rg("n_cells")
        _manifest_run_fields["cderi_build"] = _rg("cderi_build")
        _manifest_run_fields["experimental_route"] = True
        _rg_screened = getattr(_rg_convention, "screened_exchange", None)
        if _rg_screened is not None:
            c_full, c_sr, omega_screen = _rg_screened
            _manifest_run_fields["exact_exchange_c_full"] = c_full
            _manifest_run_fields["exact_exchange_c_sr"] = c_sr
            _manifest_run_fields["exact_exchange_omega_screen_bohr_inv"] = (
                omega_screen)
            _manifest_run_fields["screened_exchange_assembly"] = (
                "sr-direct (erfc-attenuated fitted kernel + sigma0 zero "
                "mode; no exchange-q0 seam)")
    _output_writer.update_run_fields(_manifest_run_fields)
    if cite_manifest_rows:
        # Feed the full assembled provenance into the .system manifest's
        # [citations] section, mirroring the molecular runner. Without
        # this the periodic .system ended with `count = 0` even though
        # the references were assembled and printed to .out / .bibtex.
        _output_writer.set_citations(cite_manifest_rows)

    # --- DOS (total + projected) for QVF embedding -------------------
    # Compute Fock lattice terms from the converged SCF density, then
    # Gaussian-broaden eigenvalues on a dense Monkhorst-Pack mesh.
    # Both total and projected DOS are serialized into the QVF archive
    # so vibe-view can render interactive side-by-side bands+DOS panels.
    _qvf_dos_data: Optional[dict[str, Any]] = None
    _qvf_pdos_data: Optional[dict[str, Any]] = None
    _qvf_coop_data: Optional[dict[str, Any]] = None
    _qvf_cohp_data: Optional[dict[str, Any]] = None
    if output_qvf and result.converged and resolved_jk == PeriodicJKMethod.GDF:
        try:
            from .periodic_gdf_properties import gdf_properties_from_result

            if dos_kmesh is not None:
                from ._vibeqc_core import monkhorst_pack

                requested_mesh = monkhorst_pack(system, list(dos_kmesh))
                requested_points = np.asarray(requested_mesh.kpoints)
                accepted_points = np.asarray(
                    getattr(result, "kpoints_cart", np.zeros((1, 3)))
                )
                if (requested_points.shape != accepted_points.shape
                        or not np.allclose(requested_points, accepted_points, atol=1e-12, rtol=0)):
                    raise NotImplementedError(
                        "GDF spectral properties currently require the accepted SCF k mesh; "
                        "a different dos_kmesh needs an evaluated target-k Hamiltonian."
                    )
            (
                _qvf_dos_data, _qvf_pdos_data, _qvf_coop_data,
                _qvf_cohp_data, _accepted_mayer, _property_reserved,
            ) = gdf_properties_from_result(result, system, basis, coop_cohp=coop_cohp)
            from .coop_cohp import _pair_metadata

            _bond_keys = [(a, b) for a in range(len(system.unit_cell))
                          for b in range(a + 1, len(system.unit_cell))
                          if _accepted_mayer[a, b] >= 0.05]
            _bond_pairs = _pair_metadata(system, _bond_keys)
            for _pair, (_a, _b) in zip(_bond_pairs, _bond_keys):
                _pair["order"] = float(_accepted_mayer[_a, _b])
            _qvf_bond_orders = {"method": "mayer", "pairs": _bond_pairs}
            _output_writer.update_run_fields({
                "property_hamiltonian": "accepted-scf",
                "property_kpoints": len(getattr(result, "kpoint_weights", [1.])),
                "property_reserved_peak_bytes": _property_reserved,
            })
        except Exception as _property_exc:
            warn_output_failure(
                _property_exc, stem_sibling(output_stem, ".qvf"),
                role="dos_computation", category=OutputFailureKind.optional_artifact,
            )
    if (
        output_qvf
        and result.converged
        and resolved_jk != PeriodicJKMethod.GDF
        and _qvf_periodic_property_payload_supported(
            resolved_jk,
            uses_external_xc=_uses_external_xc,
            ecp_active=_ecp_active,
        )
    ):
        try:
            system = _system_with_valid_unit_cell_multiplicity(system)

            from vibeqc._vibeqc_core import (
                LatticeSumOptions,
                build_jk_2e_real_space,
                build_fock_2e_real_space,
                compute_kinetic_lattice,
                compute_nuclear_lattice,
                compute_nuclear_lattice_ewald,
                compute_overlap_lattice,
                monkhorst_pack,
            )
            from vibeqc.bands import (
                _HARTREE_TO_EV,
                _density_of_states_from_terms,
                _projected_dos_from_terms,
                ao_groups_per_atom_l,
            )

            lat_opts = LatticeSumOptions()
            from vibeqc._vibeqc_core import CoulombMethod

            lat_opts.coulomb_method = CoulombMethod.EWALD_3D
            lat_opts.cutoff_bohr = 18.0

            # Overlap lattice for the legacy DOS lattice template.
            S_real = compute_overlap_lattice(basis, system, lat_opts)

            _is_unrestricted = is_open_shell_result(result)
            if _is_unrestricted:
                D_real_alpha = _density_lattice_set_for_output(
                    basis,
                    system,
                    _density_proxy_with_k_metadata(result, result.density_alpha),
                    lat_opts,
                )
                D_real_beta = _density_lattice_set_for_output(
                    basis,
                    system,
                    _density_proxy_with_k_metadata(result, result.density_beta),
                    lat_opts,
                )
                D_real = _sum_lattice_density_sets_for_output(
                    basis,
                    system,
                    lat_opts,
                    D_real_alpha,
                    D_real_beta,
                    label="periodic spin density",
                )
            else:
                D_real = _density_lattice_set_for_output(
                    basis,
                    system,
                    result,
                    lat_opts,
                )

            # Real-space Fock terms: T(g), V(g), and 2e J-K(g).
            # For 3D bulk, use Ewald-summed nuclear attraction to avoid
            # the conditionally convergent point-charge sum.
            T_real = compute_kinetic_lattice(basis, system, lat_opts)
            if system.dim == 3:
                from vibeqc._vibeqc_core import (
                    EwaldOptions,
                    GridOptions,
                    build_grid,
                )

                # Reuse the SCF Ewald a if available (BIPOLE result
                # stores it); otherwise compute a reasonable default.
                ewald_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
                if ewald_alpha is None:
                    from vibeqc.bipole_ext_el_pole import (
                        crystal_default_ewald_alpha,
                    )

                    V_cell_au = float(
                        abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
                    )
                    ewald_alpha = crystal_default_ewald_alpha(V_cell_au)

                # Molecular grid for Ewald V_ne integration.
                mol = system.unit_cell_molecule()
                grid_opts = GridOptions()
                grid_opts.n_radial = 75
                grid_opts.angular = "lebedev"
                grid_opts.lebedev_order = 29
                grid_opts.partition = "becke"
                grid = build_grid(mol, grid_opts)
                ewald_opts = EwaldOptions()
                ewald_opts.alpha = float(ewald_alpha)
                ewald_opts.real_cutoff_bohr = lat_opts.cutoff_bohr
                ewald_opts.tolerance = 1e-8
                V_real = compute_nuclear_lattice_ewald(
                    basis,
                    system,
                    grid,
                    lat_opts,
                    ewald_opts,
                )
            else:
                V_real = compute_nuclear_lattice(basis, system, lat_opts)
            F2e_real = build_fock_2e_real_space(
                basis,
                system,
                lat_opts,
                D_real,
                1.0,
                0.0,
            )
            fock_terms = [T_real, V_real, F2e_real]

            # For unrestricted, build a separate beta Fock channel.
            if _is_unrestricted:
                F_J_real = build_fock_2e_real_space(
                    basis,
                    system,
                    lat_opts,
                    D_real,
                    0.0,
                    0.0,
                )
                if method_upper == "UHF":
                    output_hf_exchange_fraction = 1.0
                elif method_upper in ("ROKS", "UKS") and functional:
                    output_hf_exchange_fraction = float(
                        Functional(functional, 2).hf_exchange_fraction
                    )
                else:
                    output_hf_exchange_fraction = 0.0

                if output_hf_exchange_fraction != 0.0:
                    from vibeqc.pbc_bipole_common import _copy_lattice_with_blocks

                    jk_alpha = build_jk_2e_real_space(
                        basis,
                        system,
                        lat_opts,
                        D_real_alpha,
                        0.0,
                    )
                    jk_beta = build_jk_2e_real_space(
                        basis,
                        system,
                        lat_opts,
                        D_real_beta,
                        0.0,
                    )
                    j_blocks = [
                        np.asarray(block, dtype=float).copy()
                        for block in F_J_real.blocks
                    ]
                    alpha_blocks = [
                        j - output_hf_exchange_fraction
                        * np.asarray(k, dtype=float)
                        for j, k in zip(j_blocks, jk_alpha.K.blocks)
                    ]
                    beta_blocks = [
                        j - output_hf_exchange_fraction
                        * np.asarray(k, dtype=float)
                        for j, k in zip(j_blocks, jk_beta.K.blocks)
                    ]
                    F2e_real_alpha = _copy_lattice_with_blocks(
                        basis,
                        system,
                        lat_opts,
                        F_J_real.cells,
                        alpha_blocks,
                        fill_missing=True,
                    )
                    F2e_real_beta = _copy_lattice_with_blocks(
                        basis,
                        system,
                        lat_opts,
                        F_J_real.cells,
                        beta_blocks,
                        fill_missing=True,
                    )
                else:
                    F2e_real_alpha = F_J_real
                    F2e_real_beta = F_J_real
                fock_terms_alpha = [T_real, V_real, F2e_real_alpha]
                fock_terms_beta = [T_real, V_real, F2e_real_beta]
                _n_spin = 2
            else:
                fock_terms_alpha = fock_terms
                fock_terms_beta = None
                _n_spin = 1

            # DOS k-mesh -- denser than the SCF mesh for smooth curves.
            _dos_mesh_ints = list(dos_kmesh) if dos_kmesh is not None else [8, 8, 8]
            dos_kmesh_obj = monkhorst_pack(system, _dos_mesh_ints)

            n_elec = system.n_electrons()
            sigma_ev = 0.05  # eV, Gaussian broadening
            sigma_ha = sigma_ev / _HARTREE_TO_EV

            if _is_unrestricted:
                mult = int(system.multiplicity)
                n_alpha = (n_elec + mult - 1) // 2
                n_beta = (n_elec - mult + 1) // 2
            else:
                n_alpha = n_elec // 2
                n_beta = 0

            # --- Total DOS -------------------------------------------------
            dos_result = _density_of_states_from_terms(
                fock_terms_alpha,
                S_real,
                dos_kmesh_obj,
                sigma=sigma_ha,
                n_grid=500,
                n_electrons_per_cell=(
                    2 * n_alpha if _is_unrestricted else n_elec
                ),
            )
            dos_result_beta = None
            if _is_unrestricted and fock_terms_beta is not None:
                dos_result_beta = _density_of_states_from_terms(
                    fock_terms_beta,
                    S_real,
                    dos_kmesh_obj,
                    sigma=sigma_ha,
                    energy_grid=dos_result.energies,
                    n_grid=500,
                    n_electrons_per_cell=2 * n_beta,
                )

            # Energies in eV, shifted to Fermi = 0 eV.
            e_fermi_values = [
                e for e in (
                    dos_result.e_fermi,
                    getattr(dos_result_beta, "e_fermi", None),
                )
                if e is not None
            ]
            e_fermi_ha = max(e_fermi_values) if e_fermi_values else 0.0
            energies_ev = (dos_result.energies - e_fermi_ha) * _HARTREE_TO_EV
            if dos_result_beta is not None:
                dos_arr = np.stack(
                    [
                        np.asarray(dos_result.dos, dtype=np.float64),
                        np.asarray(dos_result_beta.dos, dtype=np.float64),
                    ],
                    axis=0,
                )
            else:
                dos_arr = np.asarray(dos_result.dos, dtype=np.float64)
            # Convert DOS units: states / Hartree / cell -> states / eV / cell
            dos_arr = dos_arr / _HARTREE_TO_EV

            _qvf_dos_data = {
                "energies": energies_ev,
                "dos": dos_arr,
                "smearing": sigma_ev,
                "smearing_type": "gaussian",
                "fermi_energy_ev": float(e_fermi_ha * _HARTREE_TO_EV),
                "n_electrons": float(n_elec),
                "n_spin": _n_spin,
            }

            # --- Projected DOS --------------------------------------------
            groups = ao_groups_per_atom_l(system, basis)
            if groups:
                pdos_result = _projected_dos_from_terms(
                    fock_terms_alpha,
                    S_real,
                    dos_kmesh_obj,
                    groups,
                    sigma_ha,
                    dos_result.energies,
                    n_grid=500,
                    pad=5.0,
                    n_electrons_per_cell=(
                        2 * n_alpha if _is_unrestricted else n_elec
                    ),
                )
                pdos_result_beta = None
                if _is_unrestricted and fock_terms_beta is not None:
                    pdos_result_beta = _projected_dos_from_terms(
                        fock_terms_beta,
                        S_real,
                        dos_kmesh_obj,
                        groups,
                        sigma_ha,
                        dos_result.energies,
                        n_grid=500,
                        pad=5.0,
                        n_electrons_per_cell=2 * n_beta,
                    )
                # Build channel metadata: (atom_index, symbol, l, label)
                channels: list[dict[str, Any]] = []
                projections_list: list[np.ndarray] = []
                projections_beta_list: list[np.ndarray] = []
                for label in pdos_result.group_labels:
                    contrib = np.asarray(
                        pdos_result.contributions[label],
                        dtype=np.float64,
                    )
                    # Convert from states/Hartree -> states/eV
                    contrib = contrib / _HARTREE_TO_EV
                    projections_list.append(contrib)
                    if pdos_result_beta is not None:
                        contrib_beta = np.asarray(
                            pdos_result_beta.contributions[label],
                            dtype=np.float64,
                        )
                        projections_beta_list.append(
                            contrib_beta / _HARTREE_TO_EV
                        )
                    # Parse label like "Mg1-s" -> atom_index, symbol, l
                    # Use the same element-symbol table as bands.py.
                    from vibeqc.basis_crystal import _ELEMENT_SYMBOLS as _SYMS

                    parts = label.rsplit("-", 1)
                    atom_label = parts[0] if len(parts) == 2 else label
                    l_letter = parts[1] if len(parts) == 2 else "?"
                    # Extract atom index from "H1", "Mg2", etc.
                    atom_idx = 0
                    atom_symbol = label
                    for a_idx, atom in enumerate(system.unit_cell):
                        z = int(atom.Z)
                        sym = _SYMS[z] if 0 < z < len(_SYMS) else f"Z{z}"
                        expected = sym + str(a_idx + 1)
                        if expected == atom_label:
                            atom_idx = a_idx
                            atom_symbol = sym
                            break
                    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
                    l_val = l_map.get(l_letter.lower(), -1)
                    channels.append(
                        {
                            "atom_index": atom_idx,
                            "symbol": atom_symbol,
                            "l": l_val,
                            "label": label,
                        }
                    )

                projections_alpha = np.stack(projections_list, axis=0)
                if projections_beta_list:
                    projections = np.stack(
                        [
                            projections_alpha,
                            np.stack(projections_beta_list, axis=0),
                        ],
                        axis=0,
                    )
                else:
                    projections = projections_alpha

                _qvf_pdos_data = {
                    "energies": energies_ev,
                    "projections": projections,
                    "energies_units": "eV",
                    "n_spin": _n_spin,
                    "fermi_energy_ev": float(e_fermi_ha * _HARTREE_TO_EV),
                    "channels": channels,
                }

            # --- COOP/COHP bonding analysis --------------------------
            if coop_cohp:
                try:
                    from vibeqc.coop_cohp import compute_coop_cohp as _compute_coop_cohp

                    _cc_result = _compute_coop_cohp(
                        fock_terms_alpha,
                        S_real,
                        system,
                        basis,
                        dos_kmesh_obj,
                        include_cohp=True,
                        F_terms_beta=fock_terms_beta,
                        sigma=sigma_ha,
                        n_grid=500,
                        n_electrons_per_cell=n_elec,
                    )
                    # Energies in eV, shifted to Fermi = 0.
                    _cc_fermi_ha = _cc_result.fermi_energy
                    _cc_energies_ev = (
                        _cc_result.energies - _cc_fermi_ha
                    ) * _HARTREE_TO_EV

                    _qvf_coop_data = {
                        "energies": _cc_energies_ev,
                        "projections": np.asarray(_cc_result.coop, dtype=np.float64),
                        "integrated": np.asarray(
                            _cc_result.integrated_coop, dtype=np.float64
                        ),
                        "energies_units": "eV",
                        "n_spin": _n_spin,
                        "fermi_energy_ev": float(_cc_fermi_ha * _HARTREE_TO_EV),
                        "sigma_ev": float(sigma_ha * _HARTREE_TO_EV),
                        "pairs": _cc_result.pairs,
                    }

                    if _cc_result.cohp is not None:
                        _qvf_cohp_data = {
                            "energies": _cc_energies_ev,
                            "projections": np.asarray(
                                _cc_result.cohp, dtype=np.float64
                            ),
                            "integrated": np.asarray(
                                _cc_result.integrated_cohp, dtype=np.float64
                            ),
                            "energies_units": "eV",
                            "n_spin": _n_spin,
                            "fermi_energy_ev": float(_cc_fermi_ha * _HARTREE_TO_EV),
                            "sigma_ev": float(sigma_ha * _HARTREE_TO_EV),
                            "pairs": _cc_result.pairs,
                        }
                except Exception as _cc_exc:
                    warn_output_failure(
                        _cc_exc,
                        stem_sibling(output_stem, ".qvf"),
                        role="coop_cohp_computation",
                        category=OutputFailureKind.optional_artifact,
                    )

            # --- Periodic Mayer bond orders (k-space) -----------
            try:
                from vibeqc.basis_crystal import _ELEMENT_SYMBOLS as _SYMS
                from vibeqc.coop_cohp import periodic_mayer_bond_orders as _pmbo

                _bo_matrix = _pmbo(
                    fock_terms_alpha,
                    S_real,
                    system,
                    basis,
                    dos_kmesh_obj,
                    n_electrons_per_cell=n_elec,
                    F_terms_beta=fock_terms_beta,
                )
                _bo_pairs: list[dict[str, Any]] = []
                _atoms = list(system.unit_cell)
                _coords = np.asarray([np.asarray(a.xyz) for a in _atoms])
                for i in range(_bo_matrix.shape[0]):
                    for j in range(i + 1, _bo_matrix.shape[1]):
                        order = float(_bo_matrix[i, j])
                        if order < 0.05:
                            continue
                        zi = int(_atoms[i].Z)
                        zj = int(_atoms[j].Z)
                        sym_i = _SYMS[zi] if 0 < zi < len(_SYMS) else f"Z{zi}"
                        sym_j = _SYMS[zj] if 0 < zj < len(_SYMS) else f"Z{zj}"
                        dist = float(np.linalg.norm(_coords[i] - _coords[j]))
                        _bo_pairs.append(
                            {
                                "i": i,
                                "j": j,
                                "symbol_i": sym_i,
                                "symbol_j": sym_j,
                                "order": order,
                                "distance_ang": float(dist * 0.529177210903),
                            }
                        )
                if _bo_pairs:
                    _qvf_bond_orders = {
                        "method": "mayer",
                        "pairs": _bo_pairs,
                    }
            except Exception as _bo_exc:
                warn_output_failure(
                    _bo_exc,
                    stem_sibling(output_stem, ".qvf"),
                    role="bond_orders_computation",
                    category=OutputFailureKind.optional_artifact,
                )

        except Exception as _dos_exc:
            warn_output_failure(
                _dos_exc,
                stem_sibling(output_stem, ".qvf"),
                role="dos_computation",
                category=OutputFailureKind.optional_artifact,
            )

    # Keep the converged electronic result used to assemble the archive.
    # Geometry optimization replaces ``result`` with its own return object,
    # but the QVF wavefunction/property payloads above describe this SCF
    # result and must remain internally consistent with it.
    _qvf_archive_result = result

    # ---- Geometry optimization ---------------------------------------
    opt_result = None
    if optimize and result.converged:
        if _executed_jk_method not in (
            PeriodicJKMethod.BIPOLE.value,
            PeriodicJKMethod.GDF.value,
            PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA.value,
        ):
            raise NotImplementedError(
                "run_periodic_job: the executed periodic route "
                f"{_executed_jk_method!r} has no same-Hamiltonian optimizer. "
                "The job will not fall through to BIPOLE forces."
            )
        # The optimize+U guard from prior commits is lifted: all four
        # BIPOLE drivers now accept dft_plus_u= and the per-k Pulay
        # overlap-derivative term is plumbed through
        # bipole_optimize._compute_gradient.
        from .bipole_optimize import relax_atoms, relax_full

        plog.info("")
        plog.banner("Geometry optimization")
        basis_name = basis.name
        # Reconstruct kmesh from the options used during SCF.
        km = _runner_bloch_kmesh(system, kpoints)
        _bipole_opt_driver_kwargs = {
            "use_ewald_j_split": True,
            "ewald_omega": ewald_omega,
            "ewald_precision": ewald_precision,
            "use_oda": use_oda,
            "oda_trust_lambda_max": oda_trust_lambda_max,
            "use_mom": use_mom,
            "use_multipole_far_field": use_multipole_far_field,
            "multipole_l_max": multipole_l_max,
            "use_exchange_ewald_split": use_exchange_ewald_split,
            "exchange_exxdiv": exchange_exxdiv,
            "use_fock_symmetry": symmetry_stabilize,
            "use_fock_symmetry_reduce": symmetry_reduce_fock,
            "sr_image_precision": sr_image_precision,
            "exact_zone_bohr": _bipole_exact_zone_bohr,
            "bz_integration": bz_integration,
            "dft_plus_u": dft_plus_u,
        }
        _bipole_opt_nuclear_cutoff = float(
            getattr(
                opts.lattice_opts,
                "nuclear_cutoff_bohr",
                getattr(opts.lattice_opts, "cutoff_bohr", 8.0),
            )
        )

        # One ambient append-channel spans the whole relaxation: both the
        # per-step trace and final geometry block emit through output.write().
        # No file handle is threaded into the optimizer.
        # GDF-routed SCF (the GDF dispatch ran and captured its exact
        # driver call): relax on the same GDF analytic-gradient objective
        # the SCF produced, never on the BIPOLE force surface -- a
        # relaxation driven by forces from a different Hamiltonian
        # converges to a stationary point of the wrong surface. The
        # dft_plus_u legacy interception can execute a non-GDF driver
        # even when resolved_jk is GDF, so gate on the executed route.
        _gdf_routed = (
            resolved_jk == PeriodicJKMethod.GDF
            and _executed_jk_method == PeriodicJKMethod.GDF.value
        )

        with OutputChannel.to_file(out_path, mode="a"):
            write(section_header("Geometry optimization", width=56))
            flush()
            if resolved_jk == PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA:
                # EXPERIMENTAL: relax on the real-Γ direct-torus surface.
                # run_ccm_direct_optimize steps on the multi-k control
                # objective and verifies the direct-vs-control energy
                # parity at the initial AND final geometries, so the
                # relaxed stationary point is attributable to the direct
                # energy at a measured residual rather than by assumption
                # (that verification is this route's "same-Hamiltonian
                # optimizer" claim, checked per run instead of asserted).
                from .bipole_optimize import OptimizeResult
                from .periodic.ccm import CCMSystem
                from .periodic.ccm.direct import run_ccm_direct_optimize

                write(
                    "    force objective     = real-Γ direct-torus, "
                    "parity-verified against the multi-k control\n"
                )
                flush()
                _rg_opt = run_ccm_direct_optimize(
                    CCMSystem(system, _rg_mesh, str(basis.name)),
                    functional=(functional
                                if method_upper in ("RKS", "UKS") else None),
                    exxdiv=_rg_exxdiv,
                    conv_tol_grad=float(optimize_conv_tol_grad),
                    max_steps=int(optimize_max_iter),
                    aux_basis=aux_basis,
                )
                write(
                    f"    parity (initial)    = "
                    f"{_rg_opt.parity_initial.parity_residual_ha_per_cell:.2e}"
                    f" Ha/cell\n"
                    f"    parity (final)      = "
                    f"{_rg_opt.parity_final.parity_residual_ha_per_cell:.2e}"
                    f" Ha/cell\n"
                )
                flush()
                opt_result = OptimizeResult(
                    _rg_opt.system_opt,
                    float(_rg_opt.energy_per_cell),
                    np.asarray(_rg_opt.gradient, dtype=float).ravel(),
                    int(_rg_opt.n_steps),
                    bool(_rg_opt.converged),
                )
            elif _gdf_routed:
                if not _gdf_opt_capture:
                    # The Γ dispatch fell back to the legacy molecular-
                    # limit GDF driver, which has no analytic gradient.
                    # Fail closed instead of silently relaxing on BIPOLE
                    # forces (the objective-swap bug this wiring fixes).
                    raise NotImplementedError(
                        "run_periodic_job: optimize=True on the GDF route "
                        "requires a GDF driver with an analytic gradient; "
                        "this run fell back to the legacy Γ molecular-"
                        "limit GDF driver (run_rhf_periodic_gamma_gdf), "
                        "which has none. Drop the conflicting options so "
                        "the pure PBC-GDF driver runs (see the Γ GDF "
                        "dispatch preconditions), or use "
                        "jk_method='bipole' for a BIPOLE-objective "
                        "relaxation."
                    )
                write(
                    "    force objective     = GDF analytic gradient "
                    f"({_gdf_opt_capture['kwargs'].get('gdf_method', 'rsgdf')})\n"
                )
                flush()

                def _gdf_opt_rerun(sys2, basis2):
                    # The captured dispatch call, byte-identical except
                    # progress (silenced) and compute_gradient (on). An
                    # envelope the GDF gradient rejects surfaces the
                    # driver's NotImplementedError -- no silent fallback.
                    _kwargs = dict(_gdf_opt_capture["kwargs"])
                    _kwargs["compute_gradient"] = True
                    _option_objects = [*_gdf_opt_capture["args"], *_kwargs.values()]
                    with _gdf_displaced_ecp_centers(system, sys2, _option_objects):
                        return _gdf_opt_capture["driver"](
                            sys2,
                            basis2,
                            *_gdf_opt_capture["args"],
                            progress=False,
                            **_kwargs,
                        )

                opt_result = _relax_periodic_gdf_atoms(
                    system,
                    basis,
                    _gdf_opt_rerun,
                    max_iter=optimize_max_iter,
                    conv_tol_grad=optimize_conv_tol_grad,
                )
            elif optimize_cell and system.dim == 3:
                from .bipole_optimize import relax_cell_gradient

                # First relax atoms, then cell (gradient-based), then atoms again
                opt_result = relax_atoms(
                    system,
                    basis_name,
                    km,
                    method.upper(),
                    functional=functional,
                    max_iter=optimize_max_iter,
                    conv_tol_grad=optimize_conv_tol_grad,
                    scf_options=opts,
                    cutoff_bohr=float(getattr(opts.lattice_opts, "cutoff_bohr", 8.0)),
                    nuclear_cutoff_bohr=_bipole_opt_nuclear_cutoff,
                    **_bipole_opt_driver_kwargs,
                )
                _before_cell_system = opt_result.system
                opt_result = relax_cell_gradient(
                    _before_cell_system,
                    basis_name,
                    km,
                    method.upper(),
                    functional=functional,
                    max_iter=10,
                    scf_options=opts,
                    cutoff_bohr=float(getattr(opts.lattice_opts, "cutoff_bohr", 8.0)),
                    nuclear_cutoff_bohr=_bipole_opt_nuclear_cutoff,
                    **_bipole_opt_driver_kwargs,
                )
                from .bipole_optimize import _kmesh_for_system

                km = _kmesh_for_system(
                    km,
                    _before_cell_system,
                    opt_result.system,
                )
                opt_result = relax_atoms(
                    opt_result.system,
                    basis_name,
                    km,
                    method.upper(),
                    functional=functional,
                    max_iter=optimize_max_iter,
                    conv_tol_grad=optimize_conv_tol_grad,
                    scf_options=opts,
                    cutoff_bohr=float(getattr(opts.lattice_opts, "cutoff_bohr", 8.0)),
                    nuclear_cutoff_bohr=_bipole_opt_nuclear_cutoff,
                    **_bipole_opt_driver_kwargs,
                )
            else:
                opt_result = relax_atoms(
                    system,
                    basis,
                    km,
                    method.upper(),
                    functional=functional,
                    max_iter=optimize_max_iter,
                    conv_tol_grad=optimize_conv_tol_grad,
                    scf_options=opts,
                    cutoff_bohr=float(getattr(opts.lattice_opts, "cutoff_bohr", 8.0)),
                    nuclear_cutoff_bohr=_bipole_opt_nuclear_cutoff,
                    **_bipole_opt_driver_kwargs,
                )
            if opt_result is not None:
                opt_sys = opt_result.system
                write("\n" + _optimized_geometry_summary(opt_result))
                write(_system_summary(opt_sys))
                flush()
                # Write optimized geometry files as SEPARATE artefacts
                # ({stem}.opt.xyz / {stem}.opt.POSCAR), not overwriting the
                # SCF geometry. The writers do stem_sibling(stem, EXT),
                # which appends EXT unless the name already ends in exactly
                # EXT -- so the stem is passed already ending in
                # ".opt.<EXT>" and the writer's own derivation stays a
                # no-op, yielding the intended compound name. (Before issue
                # #254 the writers used Path.with_suffix, which *replaced*;
                # stem_sibling keeps that no-op deliberately, see its
                # module docstring.) Each is recorded in
                # the manifest (runtime-conditional, so not in real_plan;
                # mark_written appends it to [[outputs.files]]).
                if write_xyz_file:
                    try:
                        _output_writer.dispatch_role(
                            "geometry",
                            runtime_path=stem_sibling(output_stem, ".opt.xyz"),
                            runtime_format="extended-xyz",
                            runtime_description="Optimized periodic geometry.",
                            system=opt_sys,
                            energy_ha=opt_result.energy,
                            raise_on_error=True,
                        )
                    except Exception as _opt_xyz_exc:
                        warn_writer_failure(
                            _opt_xyz_exc,
                            stem_sibling(output_stem, ".opt.xyz"),
                            role="optimized_extended_xyz",
                            category=OutputFailureKind.optional_artifact,
                            writer=_output_writer,
                        )
                if write_poscar_file:
                    try:
                        _output_writer.dispatch_role(
                            "geometry",
                            runtime_path=stem_sibling(output_stem, ".opt.POSCAR"),
                            runtime_format="poscar",
                            runtime_description="Optimized periodic POSCAR geometry.",
                            system=opt_sys,
                            comment=f"vibe-qc optimized {label}",
                            raise_on_error=True,
                        )
                    except Exception as _opt_poscar_exc:
                        warn_writer_failure(
                            _opt_poscar_exc,
                            stem_sibling(output_stem, ".opt.POSCAR"),
                            role="optimized_poscar",
                            category=OutputFailureKind.optional_artifact,
                            writer=_output_writer,
                        )
                result = opt_result  # Return optimization result

    # --- QVF visualisation archive (v1) ----------------------------------
    # This is deliberately the final artefact write. In particular, periodic
    # optimization appends its geometry epilogue above, so run.record.log can
    # now contain the last byte and never needs a truncated marker.
    if output_qvf:
        try:
            from vibeqc.output.formats.qvf import (
                assemble_run_record as _assemble_run_record,
                terminal_run_status as _terminal_run_status,
                scf_history_from_result as _scf_history_from_result,
            )

            t_total = time.perf_counter() - t_job_start
            _qvf_scf_history = _scf_history_from_result(_qvf_archive_result)
            _qvf_run_record = _assemble_run_record(
                real_plan,
                wall_seconds=t_total,
                log_truncated=False,
            )
            # --- Symmetry data for QVF structure.symmetry ----------
            # Extract space group from the system if available.
            _qvf_symmetry = None
            try:
                _sg = getattr(_qvf_system, "symmetry", None)
                if _sg is not None:
                    _qvf_symmetry = {
                        "space_group_number": int(getattr(_sg, "number", 0)),
                        "space_group_symbol": str(
                            getattr(_sg, "international_symbol", "")
                        ),
                        "point_group": str(getattr(_sg, "point_group", "")),
                    }
            except Exception:
                pass
            _output_writer.dispatch_role(
                "qvf",
                atomic=True,
                record_hostname=record_hostname,
                system=_qvf_system,
                result=_qvf_archive_result,
                method=method_upper,
                basis=basis.name,
                functional=functional,
                jk_method=_jk_requested_label,
                jk_method_resolved=resolved_jk.value,
                jk_method_executed=_executed_jk_method,
                dft_plus_u=bool(dft_plus_u),
                dft_plus_u_route=_dft_plus_u_route,
                wall_seconds=t_total,
                volume_data=_qvf_volume_data,
                mo_data=_qvf_mo_data,
                wf_data=_qvf_wf,
                bloch_wf_data=_qvf_bloch_wf,
                hessian_result=hessian_result,
                band_structure=band_structure,
                population_summary=_qvf_pop,
                bond_orders_data=_qvf_bond_orders,
                dipole_moment_data=_qvf_dipole,
                dos_data=_qvf_dos_data,
                pdos_data=_qvf_pdos_data,
                coop_data=_qvf_coop_data,
                cohp_data=_qvf_cohp_data,
                scf_history_data=_qvf_scf_history,
                extensions=_qvf_extensions,
                vendor_json_sections=_qvf_vendor_json_sections,
                bibtex_content=_bibtex_content,
                job_spec={
                    "job_type": "periodic",
                    "method": method_upper,
                    "basis": basis.name,
                    "functional": functional,
                    "jk_method": resolved_jk.value,
                    "charge": int(system.charge),
                    "multiplicity": int(system.multiplicity),
                },
                symmetry_data=_qvf_symmetry,
                run_record=_qvf_run_record,
                # Stamp from the same result the archive describes, not the
                # post-optimization ``result``: run_status shares the
                # provenance block with scf_converged, and geometry
                # optimization rebinds ``result`` to its own object whose
                # ``converged`` means "geometry converged" (see the
                # _qvf_archive_result snapshot above).
                run_status=_terminal_run_status(_qvf_archive_result),
                raise_on_error=True,
            )
        except Exception as _qvf_exc:
            warn_writer_failure(
                _qvf_exc,
                stem_sibling(output_stem, ".qvf"),
                role="qvf_archive",
                category=OutputFailureKind.optional_artifact,
                writer=_output_writer,
            )

    # Terminal checkpoint frame: use the actual terminal result status so a
    # live viewer can stop watching without mistaking a non-converged SCF or
    # optimization for a converged one. Uses the final (optimized, if any)
    # system + result. The OutputWriter lifecycle wrapper also marks the
    # manifest crashed if terminal work raises.
    if _checkpointer.enabled:
        _finalize_periodic_checkpoint(
            _checkpointer,
            result,
            system,
            method_upper,
            basis.name,
            functional,
        )

    # Optimization appends to the log after the SCF-side record above.
    # Re-recording updates the existing outcome row and final hash. The public
    # lifecycle wrapper finishes the manifest after runtime ``.opt.*`` files.
    _output_writer.record(out_path)

    return result
