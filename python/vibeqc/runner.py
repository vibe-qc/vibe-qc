"""High-level "run a job" convenience -- classic QC-program workflow.

A single call writes the output text file, the molden orbital file, and
(for geometry optimization) a trajectory animation -- the kind of shape
users expect from Gaussian / ORCA / NWChem:

    from vibeqc import Atom, Molecule
    from vibeqc.runner import run_job

    mol = Molecule.from_xyz("h2o.xyz")
    run_job(mol, basis="6-31g*", method="rhf", output="h2o")
    # -> h2o.out, h2o.molden

    run_job(mol, basis="6-31g*", method="rks", functional="PBE",
            optimize=True, output="h2o_opt")
    # -> h2o_opt.out, h2o_opt.molden, h2o_opt.traj

All text output goes to ``{output}.out``. With the default
``write_molden_file=None``, the Molden file is written iff the route produces
format-ready Gaussian AO molecular orbitals; explicit ``True`` guarantees the
file or fails before calculation. Geometry optimization uses ASE's BFGS and
writes one frame per step to ``{output}.traj``.
"""

from __future__ import annotations

from copy import copy
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
import math
import os
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from ._initial_guess import coerce_initial_guess
from ._vibeqc_core import (
    Atom,
    BasisSet,
    D3BJParams,
    Functional,
    GridOptions,
    InitialGuess,
    MP2Options,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UMP2Options,
    UKSOptions,
    XCKind,
    get_num_threads,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
    set_num_threads,
)
from .spin_channels import is_open_shell_result

# ---- DFT integration-grid presets ---------------------------------------
#
# vibe-qc's legacy default grid (ProductGaussLegendre 17×36, no pruning,
# Becke partition) differs from ORCA's DefGrid3 (Lebedev-302 with NWChem
# pruning, Stratmann partition), causing systematic energy differences of
# 0.1–2.8 mHa across all functionals and basis sets (BUG 80). These presets
# let the caller select the grid resolution explicitly. The new default
# ("orca-defgrid3") matches ORCA's DefGrid3 to sub-0.01 mHa for first-row
# systems.

_GRID_LEVEL_PRESETS: dict[str, dict[str, object]] = {
    # vibe-qc's pinned SKALA-1.1 parity profile, matching Microsoft's default
    # PySCF inference/benchmark setup. This selects the complete PySCF 2.14.0
    # level-3 protocol in the core rather than composing the generic uniform-
    # grid controls: period-dependent radial/angular sizes, atom-specific
    # Treutler radii, radius-based NWChem pruning, and sqrt(Bragg-radius)-
    # adjusted Becke cells.
    "skala": {
        "atomic_grid_profile": "pyscf-level3",
    },
    # ORCA DefGrid3 *energy* equivalent: Lebedev-302 with NWChem (SG1-style)
    # pruning, Stratmann partition, 75 radial shells.  This does NOT
    # reproduce ORCA's grid construction (ORCA uses an OptM3 radial grid,
    # Lebedev-590 angular, adaptive pruning); it matches ORCA's DefGrid3
    # energies for first-row atoms to < 0.01 mHa, and the grid ladder in
    # #567 puts the residual grid dependence at 0.01 uEh on H2/STO-3G.
    "orca-defgrid3": {
        "n_radial": 75,
        "angular": "lebedev",
        "lebedev_order": 29,
        "angular_pruning": "nwchem",
        "partition": "stratmann",
    },
    # Dense grid for grid-converged benchmarks.  Lebedev-434 (order 35;
    # the 590-point rule is order 41), 99 radial shells.  ~20× the point
    # count of orca-defgrid3; sub-µHa accuracy on first-row atomization
    # energies.
    "fine": {
        "n_radial": 99,
        "angular": "lebedev",
        "lebedev_order": 35,
        "angular_pruning": "nwchem",
        "partition": "stratmann",
    },
    # Coarse grid for rapid screening / geometry pre-optimization.
    # Lebedev-194 (order 23), 50 radial shells.  ~0.5–1.0 mHa energy error
    # relative to orca-defgrid3; acceptable for forces but not energies.
    "coarse": {
        "n_radial": 50,
        "angular": "lebedev",
        "lebedev_order": 23,
        "angular_pruning": "nwchem",
        "partition": "stratmann",
    },
    # Legacy v0.15 default: product Gauss-Legendre 17×36, no pruning,
    # Becke partition.  Kept for backward-compatible parity with published
    # v0.15 results.  Produces energies 0.1–2.8 mHa different from ORCA.
    "legacy": {
        "n_radial": 75,
        "angular": "product",
        "n_theta": 17,
        "n_phi": 36,
        "angular_pruning": "none",
        "partition": "becke",
    },
}


def _apply_grid_level(grid: GridOptions, grid_level: str) -> None:
    """Apply a named grid-level preset to *grid* in-place."""
    preset = _GRID_LEVEL_PRESETS.get(grid_level)
    if preset is None:
        raise ValueError(
            f"Unknown grid_level={grid_level!r}. "
            f"Choose from: {', '.join(sorted(_GRID_LEVEL_PRESETS))}."
        )
    for key, val in preset.items():
        setattr(grid, key, val)


# Every public field of the C++ GridOptions struct (grid.hpp). Kept explicit so
# a new field added to the struct is noticed here rather than silently treated
# as "never customised".
_GRID_OPTION_FIELDS = (
    "atomic_grid_profile",
    "n_radial",
    "angular",
    "lebedev_order",
    "n_theta",
    "n_phi",
    "angular_pruning",
    "partition",
    "becke_k",
    "orca_angular_points",
    "vv10_grid_factor",
)


def grid_is_untouched(grid: Optional[GridOptions]) -> bool:
    """True when *grid* is ``None`` or still carries the C++ construction
    defaults in every public field, i.e. the caller never chose a grid.

    GitLab #663. ``RKSOptions()`` / ``UKSOptions()`` default-construct a live
    ``GridOptions`` whose defaults are field for field the ``legacy`` preset
    (product Gauss-Legendre 17x36, no pruning, Becke partition), and
    ``GridOptions`` carries neither an "unset" sentinel nor ``__eq__``.
    Deciding whether ``grid_level`` applies from the mere presence of an
    options object therefore ran every caller who passed one (an empty one to
    set ``max_iter``, or one materialised by a feature path) on the legacy
    grid, 0.1 to 2.8 mHa from the documented ``orca-defgrid3`` default, with
    no warning. The rule is now: the preset applies to an untouched grid; a
    grid the caller customised in any field wins. A grid deliberately set to
    the exact construction defaults is indistinguishable from an untouched
    one; the supported way to request that grid is ``grid_level="legacy"``.
    """
    if grid is None:
        return True
    reference = GridOptions()
    return all(
        getattr(grid, name) == getattr(reference, name)
        for name in _GRID_OPTION_FIELDS
    )


def ks_options_need_grid_default(options: object) -> bool:
    """Whether a ``grid_level`` preset applies to *options*: no options object,
    no grid on it (``ROKSOptions.grid`` defaults to ``None``), or an untouched
    grid (:func:`grid_is_untouched`). GitLab #663."""
    return options is None or grid_is_untouched(getattr(options, "grid", None))


def apply_ks_grid_default(options, grid_level: str = "orca-defgrid3") -> None:
    """Apply the mid-level KS grid policy, preserving custom grid fields.

    Low-level SCF wrappers deliberately do not call this for supplied options.
    Carry ``grid_level`` through nested mid-level calls: exact legacy defaults
    cannot otherwise be distinguished from an untouched GridOptions object.
    """
    if ks_options_need_grid_default(options):
        if options.grid is None:
            options.grid = GridOptions()
        _apply_grid_level(options.grid, grid_level)


from .banner import (
    VIBEQC_VERSION,
    banner,
    enforce_runtime_pin_from_env,
    library_versions,
)
from .composites import (
    Availability,
    CompositeRecipe,
    CompositeUnavailable,
    resolve_composite,
)
from .correlation_conventions import effective_electron_count
from .crash_dump import crash_dump_context, dump_on_failure
from .dispersion import compute_d3bj, d3bj_params_for
from .dispersion_d4 import compute_d4, dftd4_available
from .dispersion_d4_parameters import normalize_d4_key
from .ecp_metadata import (
    attach_inline_ecp_options_from_basis_sidecar,
    basis_sidecar_has_ecp_operator,
    basis_sidecar_replaces_core,
    ecp_centre_atom_indices,
    effective_nuclear_charges,
    molecular_options_request_ecp_operator,
    molecular_options_replace_core,
    molecular_result_has_ecp_operator,
    options_carry_ecp,
    reposition_ecp_centres,
    validate_ecp_required,
)
from .gcp import GCPDataMissing, compute_gcp
from .memory import (
    InsufficientMemoryError,
    available_memory_bytes,
    check_memory,
    estimate_memory,
    estimate_semiempirical_memory,
    format_memory_report,
)
from .naming.report import write_iupac_name
from .output import (
    Column,
    HeaderlessBlock,
    HeaderlessColumn,
    Level,
    ManifestUpdater,
    OutputChannel,
    OutputDocument,
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
    parse_write_cube_kwarg,
    print_dry_run_summary,
    record,
    render_energy_labeled,
    render_frequency,
    render_temperature,
    requested_mo_indices,
    section_header,
    warn,
    write,
)
from .output._cpp_diagnostics import (
    _progress_handler as _CPP_PROGRESS_HANDLER_SLOT,
)
from .output._errors import (
    OutputFailureKind,
    warn_output_failure,
    warn_writer_failure,
)
from .output._stem_paths import stem_sibling
from .output.citations import (
    format_references_block,
    load_default_database,
    write_references_block,
)
from .output.formats.perf import _format_timing_summary
from .output.plan import _resolve_sidecar_request
from .perf import PerfScope
from .perf import perf_log as _perf_log_ctx
from .progress import ProgressLogger, resolve_progress
from .rohf import ROHFOptions, run_rohf
from .roks import ROKSOptions, _refuse_plain_double_hybrid, run_roks
from .scf_log import write_scf_trace
from .output.formats.scf_log import (
    _HEADER as _SCF_TRACE_HEADER,
    _SEPARATOR as _SCF_TRACE_SEPARATOR,
    _format_iter_line as _format_scf_iter_line,
)
from .semiempirical.runner import (
    MOLECULAR_SEMIEMPIRICAL_METHODS,
    SEMIEMPIRICAL_METHOD_ALIASES,
    SEMIEMPIRICAL_METHODS,
    SemiempiricalResult as _SEMPR,
    normalise_semiempirical_method,
    _run_closed_shell_pm6_from_density,
    run_ccm as _run_ccm,
    run_semiempirical as _run_semiempirical,
)
from .semiempirical.routes import SemiempiricalRoutePlan
from .solvers import (
    CASCIOptions,
    CASPT2Options,
    CASSCFOptions,
    CC3Options,
    CC3Result,
    CCSDTOptions,
    CCSDTResult,
    CISDOptions,
    DMRGOptions,
    Hamiltonian,
    NEVPT2Options,
    SelectedCIOptions,
    SolverResult,
    TranscorrelatedOptions,
    V2RDMOptions,
    build_hamiltonian_mo,
    build_transcorrelated_hamiltonian,
    get_hf_orbital_provider,
    solve_dmrg,
    solve_selected_ci,
    solve_v2rdm,
)

_BOHR_TO_ANGSTROM = 0.529177210903
_EV_PER_HARTREE = 27.211386245988
_EV_PER_ANGSTROM_TO_HA_PER_BOHR = _BOHR_TO_ANGSTROM / _EV_PER_HARTREE
from .structured_log import (
    StructuredLog,
    run_fingerprint,
)
from .structured_log import (
    structured_log as _structured_log_ctx,
)

DispersionSpec = Optional[object]  # str functional name | D3BJParams | None


_ACTIVE_MOLECULAR_PROGRESS_HANDLER: ContextVar[
    Callable[[dict[str, object]], None] | None
] = ContextVar("vibeqc_molecular_progress_handler", default=None)
_ACTIVE_MOLECULAR_PROGRESS_PHASE: ContextVar[str] = ContextVar(
    "vibeqc_molecular_progress_phase",
    default="scf",
)


def _dispatch_molecular_progress(fields: dict[str, object]) -> None:
    """Forward a native row to the handler for this execution context."""
    handler = _ACTIVE_MOLECULAR_PROGRESS_HANDLER.get()
    if handler is not None:
        handler(fields)


@contextmanager
def _molecular_progress_scope(
    handler: Callable[[dict[str, object]], None],
    *,
    phase: str = "scf",
) -> Iterator[None]:
    """Install a nested- and thread-safe native molecular progress scope."""
    token = _ACTIVE_MOLECULAR_PROGRESS_HANDLER.set(handler)
    phase_token = _ACTIVE_MOLECULAR_PROGRESS_PHASE.set(phase)
    progress_token = _CPP_PROGRESS_HANDLER_SLOT.set(
        _dispatch_molecular_progress
    )
    try:
        yield
    finally:
        _ACTIVE_MOLECULAR_PROGRESS_PHASE.reset(phase_token)
        _ACTIVE_MOLECULAR_PROGRESS_HANDLER.reset(token)
        _CPP_PROGRESS_HANDLER_SLOT.reset(progress_token)


def _scf_progress_identity(iteration: int, energy: float) -> tuple[int, str]:
    """Stable exact identity, including NaN/Inf failure trajectories."""
    return int(iteration), float(energy).hex()


def _match_scf_attempt_subsequence(
    attempts: Sequence[int],
    attempt_rows: dict[int, Sequence[tuple[int, str]]],
    canonical_rows: Sequence[tuple[int, str]],
    *,
    energy_only: bool = False,
) -> list[int]:
    """Match canonical rows to the rightmost ordered whole attempts.

    Wrappers may concatenate the traces from several selected native SCF
    calls while omitting rejected stability candidates that ran between
    them. Each selected attempt must therefore contribute its complete
    trace, but selected attempts need not be contiguous in execution order.
    """
    canonical = list(canonical_rows)
    memo: dict[tuple[int, int], Optional[tuple[int, ...]]] = {}

    def match_before(
        attempt_stop: int,
        row_stop: int,
    ) -> Optional[tuple[int, ...]]:
        if row_stop == 0:
            return ()
        key = (attempt_stop, row_stop)
        if key in memo:
            return memo[key]
        for attempt_index in range(attempt_stop - 1, -1, -1):
            attempt = attempts[attempt_index]
            rows = list(attempt_rows.get(attempt, ()))
            row_start = row_stop - len(rows)
            if not rows or row_start < 0:
                continue
            expected = canonical[row_start:row_stop]
            if energy_only:
                rows_match = [row[1] for row in rows] == [
                    row[1] for row in expected
                ]
            else:
                rows_match = rows == expected
            if not rows_match:
                continue
            prefix = match_before(attempt_index, row_start)
            if prefix is not None:
                match = (*prefix, attempt)
                memo[key] = match
                return match
        memo[key] = None
        return None

    return list(match_before(len(attempts), len(canonical)) or ())


def _match_scf_rows_rightmost(
    attempts: Sequence[int],
    attempt_rows: dict[int, Sequence[tuple[int, str]]],
    canonical_rows: Sequence[tuple[int, str]],
    *,
    energy_only: bool = False,
) -> dict[int, tuple[int, tuple[int, str]]]:
    """Match a partial live trace to canonical rows without duplicates.

    The normal path above recognizes complete native attempts. This fallback
    handles an interrupted or malformed callback stream: retain any exact
    rows that did arrive, skip rejected-attempt extras, and let terminal
    replay supply only the missing canonical rows. Matching from the right
    preserves the selected final attempt when stability searches emitted
    earlier rejected candidates with an indistinguishable row.
    """
    live_rows = [
        (attempt, row)
        for attempt in attempts
        for row in attempt_rows.get(attempt, ())
    ]
    live_stop = len(live_rows)
    matched: dict[int, tuple[int, tuple[int, str]]] = {}
    for canonical_index in range(len(canonical_rows) - 1, -1, -1):
        expected = canonical_rows[canonical_index]
        for live_index in range(live_stop - 1, -1, -1):
            observed = live_rows[live_index][1]
            rows_match = (
                observed[1] == expected[1]
                if energy_only
                else observed == expected
            )
            if not rows_match:
                continue
            matched[canonical_index] = live_rows[live_index]
            live_stop = live_index
            break
    return matched


def _resolve_dispersion(
    dispersion: DispersionSpec,
    functional: Optional[str],
) -> Optional[D3BJParams]:
    """Normalize the ``dispersion=`` argument into a D3BJParams object,
    or None if no dispersion correction is requested.

    * ``None`` or empty string -> no dispersion.
    * ``True`` / ``"d3bj"`` -> use D3-BJ params for the current functional.
    * ``"pbe" / "b3lyp" / ...`` -> use D3-BJ params for that functional
      explicitly (useful for HF + D3-BJ).
    * :class:`D3BJParams` instance -> used as-is.
    """
    if dispersion is None or dispersion is False:
        return None
    if isinstance(dispersion, D3BJParams):
        return dispersion
    if isinstance(dispersion, str):
        key = dispersion.strip().lower()
        if key in ("", "none", "false"):
            return None
        if key in ("d3bj", "true", "yes", "on"):
            if not functional:
                raise ValueError(
                    "dispersion='d3bj' requires a DFT functional so we can "
                    "look up damping parameters; pass functional='pbe', "
                    "etc., or give a functional name directly as the "
                    "dispersion argument."
                )
            lookup = functional
        else:
            lookup = dispersion
        p = d3bj_params_for(lookup)
        if p is None:
            raise ValueError(
                f"dispersion={dispersion!r}: no D3-BJ parameters found "
                f"for functional {lookup!r} (neither builtin nor dftd3 "
                f"backend knows this name). Install the dftd3 backend "
                f"for the full Grimme parameter set: "
                f"`pip install dftd3` into your vibe-qc venv, or "
                f"`pip install -e '.[dispersion]'` from the repo checkout."
            )
        return p
    if dispersion is True:
        if not functional:
            raise ValueError(
                "dispersion=True requires a DFT functional; pass functional='pbe', etc."
            )
        return d3bj_params_for(functional)
    raise TypeError(
        f"dispersion must be None, bool, str, or D3BJParams; got "
        f"{type(dispersion).__name__}"
    )


class _DispersionAugmented:
    """Transparent wrapper exposing a D3(BJ) or D4 correction.

    The raw ``.energy`` of the wrapped result remains unchanged. Users opt in
    to the corrected total through :attr:`energy_total`; when the wrapped
    result is itself a post-HF wrapper, its ``energy_total`` is the base to
    which dispersion is added. Everything else (mo_energies, density, fock,
    converged, n_iter, scf_trace, ...) forwards to the wrapped result.

    We wrap rather than mutate because the C++ result classes have
    ``def_readonly`` fields -- they can't be extended in place from
    Python.
    """

    __slots__ = ("_scf", "e_dispersion", "dispersion_params", "method")

    def __init__(
        self, scf_result, e_dispersion: float, params: object, method: str = ""
    ):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "e_dispersion", e_dispersion)
        object.__setattr__(self, "dispersion_params", params)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    @property
    def e_scf(self) -> float:
        """The bare SCF energy (Hartree) -- alias of ``.energy``, named
        so energy-collection code reads unambiguously next to
        :attr:`e_dispersion` and :attr:`energy_total`."""
        return float(self._scf.energy)

    @property
    def energy_total(self) -> float:
        """Underlying method total plus dispersion energy (Hartree)."""
        base = getattr(self._scf, "energy_total", self._scf.energy)
        return float(base) + float(self.e_dispersion)

    def __repr__(self) -> str:
        return (
            f"{type(self._scf).__name__}(energy={self._scf.energy:.10f}, "
            f"e_dispersion={self.e_dispersion:+.6e}, "
            f"energy_total={self.energy_total:.10f})"
        )


class _FreeEnergyAugmented:
    """Transparent wrapper exposing the Mermin free energy ``A = E - T·S``.

    The raw ``.energy`` of the wrapped SCF result is the integer-occupation
    SCF total energy (unchanged).  The Mermin free energy is available as
    :attr:`free_energy` and the dimensionless electronic entropy as
    :attr:`smearing_entropy`.  Everything else forwards to the wrapped
    result transparently.

    We wrap rather than mutate because the C++ result classes have
    ``def_readonly`` fields -- they can't be extended in place from Python.
    """

    __slots__ = (
        "_scf",
        "free_energy",
        "smearing_entropy",
        "smearing_temperature",
        "smearing_method",
        "smearing_fermi_level",
    )

    def __init__(
        self,
        scf_result,
        free_energy: float,
        entropy: float,
        temperature: float,
        mu: float,
        method: str = "fermi-dirac",
    ):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "free_energy", free_energy)
        object.__setattr__(self, "smearing_entropy", entropy)
        object.__setattr__(self, "smearing_temperature", temperature)
        object.__setattr__(self, "smearing_method", method)
        object.__setattr__(self, "smearing_fermi_level", mu)

    def __getattr__(self, name: str):
        return getattr(self._scf, name)

    def e_scf(self) -> float:
        """The bare SCF energy (Hartree) -- unchanged integer-occupation value."""
        return float(getattr(self._scf, "energy", 0.0))

    @property
    def energy(self) -> float:
        """The raw SCF energy (unchanged).  Use :attr:`free_energy` for the
        Mermin free energy ``A = E - T·S``."""
        return float(getattr(self._scf, "energy", 0.0))

    def energy_total(self) -> float:
        """Total energy inclusive of dispersion, using the free energy as base."""
        base = getattr(self._scf, "energy_total", None)
        if base is not None:
            return float(base)
        return float(self.free_energy)

    def __repr__(self) -> str:
        return (
            f"{type(self._scf).__name__}("
            f"energy={self._scf.energy:.10f}, "
            f"free_energy={self.free_energy:.10f}, "
            f"smearing_entropy={self.smearing_entropy:.6e})"
        )


class _MP2Augmented:
    """Transparent wrapper exposing a post-SCF MP2 result without
    overriding the raw SCF ``.energy``. :attr:`mp2` is the C++
    ``MP2Result`` (``e_hf`` / ``e_os`` / ``e_ss`` / ``e_correlation`` /
    ``e_total``); :attr:`energy_total` is the MP2 total energy. Everything
    else (mo_energies, density, fock, converged, n_iter, scf_trace, ...)
    forwards to the underlying pybind11 SCF-result struct.

    Mirrors :class:`_DispersionAugmented`: we wrap rather than mutate
    because the C++ result classes have ``def_readonly`` fields and no
    ``__dict__`` (so ``result.mp2 = ...`` would raise).
    """

    __slots__ = ("_scf", "mp2", "method")

    def __init__(self, scf_result, mp2_result, method: str = ""):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "mp2", mp2_result)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    @property
    def energy_total(self) -> float:
        """SCF + MP2 correlation energy (Hartree)."""
        return float(self.mp2.e_total)

    def __repr__(self) -> str:
        return (
            f"{type(self._scf).__name__}+MP2("
            f"e_scf={float(self._scf.energy):.10f}, "
            f"e_corr={float(self.mp2.e_correlation):+.6e}, "
            f"e_mp2_total={float(self.mp2.e_total):.10f})"
        )


class _DLPNOMP2Augmented:
    """SCF result + DLPNO-MP2 result, attribute-forwarding wrapper.

    Mirrors :class:`_MP2Augmented` -- wraps rather than mutates because the
    C++ SCF result has ``def_readonly`` fields and no ``__dict__``.
    """

    __slots__ = ("_scf", "dlpno_mp2", "method")

    def __init__(self, scf_result, dlpno_result, method: str = ""):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "dlpno_mp2", dlpno_result)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    @property
    def energy_total(self) -> float:
        """SCF + DLPNO-MP2 correlation energy (Hartree)."""
        return float(self.dlpno_mp2.e_total)

    def __repr__(self) -> str:
        return (
            f"{type(self._scf).__name__}+DLPNO-MP2("
            f"e_scf={float(self._scf.energy):.10f}, "
            f"e_corr={float(self.dlpno_mp2.e_corr):+.6e}, "
            f"e_total={float(self.dlpno_mp2.e_total):.10f})"
        )


class _DLPNOUMP2Augmented:
    """UHF result + open-shell DLPNO-UMP2 result, attribute-forwarding wrapper."""

    __slots__ = ("_scf", "dlpno_ump2", "method")

    def __init__(self, scf_result, dlpno_result, method: str = ""):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "dlpno_ump2", dlpno_result)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    @property
    def energy_total(self) -> float:
        """UHF + DLPNO-UMP2 correlation energy (Hartree)."""
        return float(self.dlpno_ump2.e_total)

    def __repr__(self) -> str:
        return (
            f"{type(self._scf).__name__}+DLPNO-UMP2("
            f"e_scf={float(self._scf.energy):.10f}, "
            f"e_corr={float(self.dlpno_ump2.e_corr):+.6e}, "
            f"e_total={float(self.dlpno_ump2.e_total):.10f})"
        )


class _DLPNOCCSDAugmented:
    """SCF result + DLPNO-CCSD/UCCSD result, attribute-forwarding wrapper."""

    __slots__ = ("_scf", "dlpno_ccsd", "method")

    def __init__(self, scf_result, cc_result, method: str = ""):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "dlpno_ccsd", cc_result)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    @property
    def energy_total(self) -> float:
        """SCF + CCSD correlation (+ (T) if computed), Hartree."""
        return float(self.dlpno_ccsd.e_total)

    def __repr__(self) -> str:
        label = "DLPNO-UCCSD" if "uccsd" in self.method else "DLPNO-CCSD"
        return (
            f"{type(self._scf).__name__}+{label}("
            f"e_scf={float(self._scf.energy):.10f}, "
            f"e_corr={float(self.dlpno_ccsd.e_corr):+.6e}, "
            f"e_t={float(self.dlpno_ccsd.e_t):+.3e}, "
            f"e_total={float(self.dlpno_ccsd.e_total):.10f})"
        )


class _CCSDAugmented:
    """SCF result + CCSD/CCSD(T) result, attribute-forwarding wrapper.

    Mirrors :class:`_MP2Augmented`: wraps rather than mutates because the
    C++ SCF result has ``def_readonly`` fields and no ``__dict__``.
    :attr:`ccsd` is the C++ ``CCSDResult`` (``e_ccsd_correlation`` /
    ``e_ccsd`` / ``e_t`` / ``e_ccsd_t`` / ``cc_trace`` / ...).
    """

    __slots__ = ("_scf", "ccsd", "method")

    def __init__(self, scf_result, cc_result, method: str = ""):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "ccsd", cc_result)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    @property
    def energy_total(self) -> float:
        """SCF + CCSD (+ (T) when computed) total energy (Hartree)."""
        return float(self.ccsd.e_total)

    def __repr__(self) -> str:
        return (
            f"{type(self._scf).__name__}+CCSD("
            f"e_scf={float(self._scf.energy):.10f}, "
            f"e_corr={float(self.ccsd.e_ccsd_correlation):+.6e}, "
            f"e_t={float(self.ccsd.e_t):+.6e}, "
            f"e_total={float(self.ccsd.e_total):.10f})"
        )


class _OVGFAugmented:
    """Transparent wrapper exposing OVGF / GF2 quasiparticle results.

    :attr:`ovgf` is the list of :class:`vibeqc.propagator.QuasiparticleResult`
    (one per corrected orbital, carrying e_scf / e_qp / pole strength).  All SCF
    attributes forward to the underlying result (mirrors :class:`_MP2Augmented`;
    the C++ result has no ``__dict__``).
    """

    __slots__ = ("_scf", "ovgf", "method")

    def __init__(self, scf_result, ovgf_results, method: str = ""):
        object.__setattr__(self, "_scf", scf_result)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "ovgf", ovgf_results)

    def __getattr__(self, name):
        return getattr(self._scf, name)

    def __repr__(self) -> str:
        n = len(self.ovgf)
        return f"{type(self._scf).__name__}+OVGF({n} quasiparticle orbitals)"


if TYPE_CHECKING:
    from .cc import CCSDOptions
    from .mlip import MLIPOptions
    from .semiempirical.methods.msindo_ccm import CCMOptions
    from .thermo import ThermoOptions


Method = Literal[
    "rhf",
    "rks",
    "uhf",
    "uks",
    # Restricted open-shell HF (spin-pure single determinant; Roothaan
    # 1960).  Pure-Python driver on the JK seam --- see vibeqc.rohf.
    "rohf",
    # Restricted open-shell Kohn--Sham (spin-pure KS; same Roothaan
    # coupling + libxc XC) --- see vibeqc.roks.
    "roks",
    "auto",
    "selected_ci",
    "dmrg",
    "v2rdm",
    "transcorrelated_ci",
    "casci",
    "mrci",
    "casscf",
    # CASSCF/CASCI-referenced multireference PT2 (internally-contracted
    # CASPT2 default; strongly-contracted NEVPT2).  Reachable through
    # run_job(method=...) and documented in docs/user_guide/non_hf_solvers.md
    # + handovers/HANDOVER_MULTIREF.md; listed here so the public method type
    # matches the documented, dispatched surface.
    "nevpt2",
    "caspt2",
    "fci",
    "ci",
    "cisd",
    "ccsd",
    "ccsd(t)",
    "cc3",
    "ccsdt",
    "bccd",
    "bccd(t)",
    # Approximate coupled-cluster / coupled-pair variants of the canonical
    # closed-shell DF-CCSD kernel (CCSDOptions.cc_variant; cpp/src/ccsd.cpp):
    # CC2, CCD (T1 frozen), linearized LCCD / LCCSD, and Meyer's CEPA(0..3).
    # RHF reference only; also reachable through method='ci' with citype=.
    "cc2",
    "ccd",
    "lccd",
    "lccsd",
    "cepa(0)",
    "cepa(1)",
    "cepa(2)",
    "cepa(3)",
    "qcisd",
    "qcisd(t)",
    # Post-SCF Moller-Plesset (RHF reference + native C++ MP2; SCS/SOS via
    # the c_os/c_ss spin-component scaling -- see vibeqc.correlation).
    "mp2",
    "scs-mp2",
    "sos-mp2",
    # DLPNO-MP2 (RHF reference + vibeqc.dlpno.mp2 -- Foster-Boys LMOs, PAO
    # domains, semicanonical PNOs, coupled LMP2 residuals; Pinski 2015).
    "dlpno-mp2",
    # DLPNO-CCSD / CCSD(T) local solver (vibeqc.dlpno.ccsd_local_solver --
    # reduced-scaling per-pair PNO engine; full-domain == canonical CCSD;
    # pilot-anchored = FCI-anchored; Riplinger 2013).
    "dlpno-ccsd",
    "dlpno-ccsd(t)",
    # Green's-function quasiparticle IPs/EAs (RHF reference + the diagonal
    # second-order self-energy -- see vibeqc.propagator).
    "ovgf",
    # Composite 3c keywords (v0.9.0 -- see vibeqc.composites). The
    # dispatcher resolves these via _apply_composite before any of the
    # other Method literal branches see them.
    "hf-3c",
    "pbeh-3c",
    "b97-3c",
    "b3lyp-3c",
    "r2scan-3c",
    "wb97x-3c",
    "hse-3c",
    # Semiempirical methods.
    "dftb0",
    "scc_dftb",
    "pm6",
    "gfn2_xtb",
    "om1",
    "om2",
    "om3",
    "msindo",
    # MSINDO SECCM (needs ccm_options with the supercell lattice vectors).
    # ``ccm`` is the normalized compatibility key for public ``seccm``.
    "ccm",
    # Machine-learning interatomic potential -- external pre-trained
    # model (ACEsuit MACE). Routed through vibeqc.mlip.mace.
    "mace",
]

# Machine-learning interatomic potentials (external pre-trained models,
# routed through vibeqc.mlip). Basis-set-free like the semiempirical
# methods; unlike them, vibe-qc drives a third-party pre-trained forward
# pass (maintainer-approved CLAUDE.md Sec.10 extension). Kept as a module
# constant so the basis-skip sites in run_job stay in sync.
_MLIP_METHODS = frozenset({"mace"})

# Routes whose returned result exposes the AO-basis orbitals and density
# needed by both the Molden and population adapters. Post-SCF methods that
# retain their mean-field result resolve to one of these keys in
# ``_select_method``. Solver-only, semiempirical, and MLIP results do not.
_WAVEFUNCTION_SIDECAR_METHODS = frozenset(
    {"rhf", "uhf", "rks", "uks", "rohf", "roks"}
)

# Basis-free SCC engines that expose conventional net atomic Mulliken charges
# even though their minimal Slater-orbital basis cannot be sent through the
# Gaussian-GTO Molden/property writers. Their population sidecars therefore
# carry the native Mulliken section plus explicit unsupported markers for the
# analyses that need a Gaussian AO contract.
_NATIVE_MULLIKEN_SIDECAR_METHODS = frozenset(
    {"dftb0", "scc_dftb", "gfn2_xtb"}
)

# Post-SCF correlation methods: an RHF/UHF SCF runs first (resolved_method),
# but citations + the output label key off the *original* method so the
# correlation papers (routes.methods.{ccsd,ccsd(t),mp2,scs-mp2,sos-mp2}) fire.
_POSTSCF_CITE_METHODS = frozenset(
    {
        "cisd",
        "ccsd",
        "ccsd(t)",
        "cc3",
        "ccsdt",
        "ccsd[t]",
        "a-ccsd(t)",
        "bccd",
        "bccd(t)",
        "cc2",
        "ccd",
        "lccd",
        "lccsd",
        "cepa(0)",
        "cepa(1)",
        "cepa(2)",
        "cepa(3)",
        "qcisd",
        "qcisd(t)",
        "mp2",
        "scs-mp2",
        "sos-mp2",
        "rohf-mp2",
        "dlpno-mp2",
        "dlpno-ump2",
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
        "dlpno-uccsd",
        "dlpno-uccsd(t)",
        "ovgf",
    }
)


_CITYPE_SUPPORTED = {
    "cisd": "cisd",
    "ccsd": "ccsd",
    "ccsd(t)": "ccsd(t)",
    "cc3": "cc3",
    "ccsdt": "ccsdt",
    "bccd": "bccd",
    "bccd(t)": "bccd(t)",
    "cc2": "cc2",
    "ccd": "ccd",
    "lccd": "lccd",
    "lccsd": "lccsd",
    "cepa0": "cepa(0)",
    "cepa(0)": "cepa(0)",
    "cepa1": "cepa(1)",
    "cepa(1)": "cepa(1)",
    "cepa2": "cepa(2)",
    "cepa(2)": "cepa(2)",
    "cepa3": "cepa(3)",
    "cepa(3)": "cepa(3)",
    "qcisd": "qcisd",
    "qcisd(t)": "qcisd(t)",
}
_CITYPE_UNIMPLEMENTED = {
    "cepa": "CEPA",
    "cepa(n)": "CEPA(n)",
    "cepan": "CEPA(n)",
}

# Coupled-pair variants of the closed-shell DF-CCSD kernel: run_job method
# string -> CCSDOptions.cc_variant selector (cpp/src/ccsd.cpp).  All run an
# RHF SCF first and reuse the CCSD post-SCF machinery with the variant set
# and (T) off.
_CC_VARIANT_METHODS = {
    "cc2": "cc2",
    "ccd": "ccd",
    "lccd": "lccd",
    "lccsd": "lccsd",
    "cepa(0)": "cepa(0)",
    "cepa(1)": "cepa(1)",
    "cepa(2)": "cepa(2)",
    "cepa(3)": "cepa(3)",
    "qcisd": "qcisd",
    "qcisd(t)": "qcisd",
}
_CC_VARIANT_LABELS = {
    "cc2": "CC2",
    "ccd": "CCD",
    "lccd": "LCCD",
    "lccsd": "LCCSD",
    "cepa(0)": "CEPA(0)",
    "cepa(1)": "CEPA(1)",
    "cepa(2)": "CEPA(2)",
    "cepa(3)": "CEPA(3)",
    "qcisd": "QCISD",
    "qcisd(t)": "QCISD(T)",
}
_BCCD_METHODS = {
    "bccd": False,
    "bccd(t)": True,
}
_BCCD_LABELS = {
    "bccd": "BCCD",
    "bccd(t)": "BCCD(T)",
}


def _normalise_citype_selector(citype: object) -> str:
    if not isinstance(citype, str):
        raise ValueError(
            "citype= expects a string such as 'cisd', 'ccsd', or 'ccsd(t)'; "
            f"got {citype!r}."
        )
    key = citype.strip().lower().replace(" ", "").replace("_", "-")
    if key in _CITYPE_SUPPORTED:
        return _CITYPE_SUPPORTED[key]
    if key in _CITYPE_UNIMPLEMENTED:
        label = _CITYPE_UNIMPLEMENTED[key]
        raise NotImplementedError(
            f"citype={citype!r} selects {label}, which is a roadmap item but "
            "is not implemented yet. Currently supported selectors are "
            "citype='cisd', 'ccsd', 'ccsd(t)', 'cc3', 'ccsdt', 'bccd', 'bccd(t)', 'cc2', 'ccd', "
            "'lccd', 'lccsd', 'cepa(0)'..'cepa(3)', 'qcisd', and "
            "'qcisd(t)'."
        )
    raise ValueError(
        "citype= expects 'cisd', 'ccsd', 'ccsd(t)', 'cc3', 'ccsdt', 'bccd', 'bccd(t)', "
        "'cc2', 'ccd', 'lccd', 'lccsd', 'cepa(0)'..'cepa(3)', 'qcisd', "
        f"'qcisd(t)', or one of the named roadmap variants; got {citype!r}."
    )


def _apply_citype_selector(method: str, citype: object | None) -> str:
    if method == "ci" and citype is None:
        raise ValueError(
            "method='ci' requires citype='cisd', 'ccsd', 'ccsd(t)', 'cc3', 'ccsdt', "
            "'bccd', 'bccd(t)', or another supported CI/CC selector."
        )
    if citype is None:
        return method
    selected = _normalise_citype_selector(citype)
    if method not in ("auto", "rhf", "ci", selected):
        raise ValueError(
            f"citype={citype!r} conflicts with method={method!r}. Use "
            "method='ci', method='rhf', method='auto', or the matching "
            f"method={selected!r}."
        )
    return selected


_METHOD_ALIASES = SEMIEMPIRICAL_METHOD_ALIASES


def _normalise_method_alias(method: str) -> str:
    return normalise_semiempirical_method(method)


def _select_method(
    method: Method,
    molecule: Molecule,
    functional: Optional[str],
    ccsd_reference: str = "uhf",
    mp2_reference: str = "uhf",
) -> str:
    """Resolve ``method='auto'`` against molecule.multiplicity + functional.

    Heuristic wavefunction routing by electron count:
    - <=4 electrons -> FCI (exact for tiny systems)
    - <=8 electrons -> Selected-CI (systematically improvable)
    - >8 electrons -> HF (via SCF)

    ``ccsd`` and ``ccsd(t)`` run RHF SCF first, then post-SCF coupled-cluster."""
    if method == "auto":
        if functional:
            return "uks" if molecule.multiplicity > 1 else "rks"
        n_elec = molecule.n_electrons()
        if n_elec <= 4:
            return "fci"
        if n_elec <= 8:
            return "selected_ci"
        return "uhf" if molecule.multiplicity > 1 else "rhf"
    if method in ("ccsd", "ccsd(t)"):
        # Closed shell -> RHF + RCCSD; open shell -> UHF + spin-orbital
        # UCCSD by default, or ROHF + ROHF-CCSD when ccsd_reference='rohf'.
        if molecule.multiplicity > 1:
            return "rohf" if ccsd_reference == "rohf" else "uhf"
        return "rhf"
    if method in _BCCD_METHODS or method in _CC_VARIANT_METHODS:
        # Coupled-pair / QCI variants live in the closed-shell kernel only.
        if molecule.multiplicity > 1:
            raise NotImplementedError(
                f"method={method!r} requires a closed-shell (singlet) "
                "reference; open-shell coupled-pair/QCI variants are not "
                "implemented. Use method='ccsd' / 'ccsd(t)' (UHF or ROHF "
                "reference) for open-shell coupled cluster."
            )
        return "rhf"
    # Post-SCF MP2 (and its spin-component-scaled variants) runs the mean-field
    # SCF first -- RHF for closed shell, UHF for open shell (native run_ump2),
    # or ROHF when mp2_reference='rohf' (spin-pure semicanonical ROHF-MP2);
    # the post-SCF step keys off the original ``method`` downstream.
    if method in ("mp2", "scs-mp2", "sos-mp2"):
        if molecule.multiplicity > 1:
            return "rohf" if mp2_reference == "rohf" else "uhf"
        return "rhf"
    # DLPNO-MP2: closed shell -> RHF + DLPNO-MP2; open shell -> UHF +
    # DLPNO-UMP2 (auto-routed in the post-SCF dispatch).
    if method == "dlpno-mp2":
        return "uhf" if molecule.multiplicity > 1 else "rhf"
    # DLPNO-CCSD/(T): closed shell -> RHF + DLPNO-CCSD local solver;
    # open shell -> UHF + DLPNO-UCCSD(T) local solver (pilot opt-in).
    if method in ("dlpno-ccsd", "dlpno-ccsd(t)"):
        return "uhf" if molecule.multiplicity > 1 else "rhf"
    # OVGF (diagonal-self-energy) runs the mean-field SCF first -- RHF for closed
    # shell, UHF for open shell (the spin-resolved spin-orbital self-energy).
    if method == "ovgf":
        return "uhf" if molecule.multiplicity > 1 else "rhf"
    return method


def _apply_composite(
    method: str,
    *,
    basis: Optional[str],
    functional: Optional[str],
    dispersion,
    molecule: Molecule,
) -> tuple[str, str, Optional[str], object, Optional[CompositeRecipe]]:
    """Resolve a composite-3c ``method=`` keyword (``"hf-3c"``,
    ``"pbeh-3c"``, ...) into the (method, basis, functional, dispersion,
    recipe) tuple the rest of :func:`run_job` consumes.

    Non-composite ``method`` values pass through unchanged with
    ``recipe = None``. For a composite:

    * basis / functional / dispersion are taken from the recipe when
      the caller left them at defaults; an explicit caller basis wins
      with a warning (deliberate-experiment escape hatch).
    * D3-BJ recipes carry their own re-fit damping (``d3bj_damping``);
      it is passed straight through as a :class:`D3BJParams` so the
      dispersion module doesn't fall back to a wrong per-functional
      lookup.
    * Recipes flagged ``PENDING_*`` raise :class:`CompositeUnavailable`
      with a pointer at the gating roadmap item.
    """
    recipe = resolve_composite(method)
    if recipe is None:
        return method, basis or "", functional, dispersion, None

    if recipe.availability in (
        Availability.PENDING_F1,
        Availability.PENDING_F3,
        Availability.PENDING_ECP,
    ):
        raise CompositeUnavailable(
            f"method={method!r}: {recipe.availability.value}. "
            f"This composite depends on infrastructure not yet on "
            f"main. See docs/user_guide/composites.md Sec. Availability.\n"
            f"  Notes: {recipe.notes}"
        )

    resolved_basis = basis if basis else recipe.basis
    if basis and basis.lower() != recipe.basis.lower():
        import warnings

        warnings.warn(
            f"method={method!r} specifies basis={recipe.basis!r}, but "
            f"caller passed basis={basis!r}. Honouring the explicit "
            f"basis -- the published 3c parameters were fit at the "
            f"composite's native basis and may not transfer.",
            stacklevel=2,
        )
    resolved_functional = functional if functional else recipe.functional

    if dispersion is None:
        if recipe.dispersion == "d3bj" and recipe.d3bj_damping is not None:
            resolved_dispersion: object = D3BJParams(
                s6=recipe.d3bj_damping.s6,
                s8=recipe.d3bj_damping.s8,
                a1=recipe.d3bj_damping.a1,
                a2=recipe.d3bj_damping.a2,
                s9=recipe.d3bj_damping.s9,
            )
        else:
            resolved_dispersion = recipe.dispersion
    else:
        resolved_dispersion = dispersion

    # Pure-HF composites (recipe.functional is None -- only HF-3c today)
    # must route straight to the HF SCF driver. Handing "auto" to
    # _select_method with no functional would drop into the
    # wavefunction auto-ladder (<=4e -> FCI, <=8e -> selected-CI), so a
    # turnkey hf-3c on H2 silently ran fci(ndet=4) instead of HF
    # (audit F1.1). DFT composites keep "auto" -> _select_method picks
    # rks/uks from multiplicity.
    if resolved_functional is None:
        resolved_method = "uhf" if molecule.multiplicity > 1 else "rhf"
    else:
        resolved_method = "auto"
    return (
        resolved_method,
        resolved_basis,
        resolved_functional,
        resolved_dispersion,
        recipe,
    )


def _default_open_shell_dlpno_uhf_options(
    method: str,
    molecule: Molecule,
    uhf_options: Optional[UHFOptions],
) -> Optional[UHFOptions]:
    """Return UHF options for open-shell DLPNO-MP2's reference SCF.

    Allyl/cc-pVTZ, the release-paper M22 DLPNO-UMP2 case, reaches the
    final UHF commutator tail after the generic 100-cycle cap and then
    converges cleanly at cycle 224.  Only materialise a larger default
    when the caller left ``uhf_options`` unset; explicit options remain
    authoritative.
    """
    if uhf_options is not None:
        return uhf_options
    if method == "dlpno-mp2" and molecule.multiplicity > 1:
        opts = UHFOptions()
        opts.max_iter = max(int(opts.max_iter), 250)
        return opts
    return None


# SCFIteration.accelerator_step codes (cpp/include/vibeqc/rhf.hpp): the Fock
# extrapolation that actually reached the SCF in one iteration.
_ACCELERATOR_STEP_NONE = 0
_ACCELERATOR_STEP_DIIS = 1
_ACCELERATOR_STEP_KDIIS = 2
_ACCELERATOR_STEP_EDIIS = 3
_ACCELERATOR_STEP_ADIIS = 4


def _executed_accelerator_steps(result: object) -> set[int]:
    """Set of ``accelerator_step`` codes the SCF trace recorded, excluding
    iterations in which no extrapolation reached the Fock matrix. Empty when
    the result carries no trace or a trace without the field (Python-side
    drivers that construct ``SCFIteration`` without it)."""
    trace = getattr(result, "scf_trace", None) if result is not None else None
    if not trace:
        return set()
    steps: set[int] = set()
    for entry in trace:
        step = getattr(entry, "accelerator_step", None)
        if step is None:
            continue
        step = int(step)
        if step != _ACCELERATOR_STEP_NONE:
            steps.add(step)
    return steps


def _detect_scf_accelerator(
    resolved_method: str,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    *,
    options_used: object = None,
    result: object = None,
) -> Optional[str]:
    """Return the lower-cased SCFAccelerator route key (``"diis"`` /
    ``"ediis"`` / ``"ediis_diis"`` / ``"kdiis"`` / ``"adiis"`` /
    ``"adiis_diis"`` / ``"r_cdiis"`` / ``"ad_cdiis"``) the resolved SCF
    executed, or ``None`` when no options struct is in play (post-SCF /
    non-mean-field methods).

    The citation router uses this to fire the right SCF-accelerator
    references -- Hu-Yang 2010 for ADIIS, Kollmar 1997 for KDIIS,
    Kudin-Scuseria-Cancès 2002 for EDIIS, plus Garza-Scuseria 2012 for
    the EDIIS+DIIS hybrid -- instead of always crediting plain DIIS
    (Pulay 1980 / 1982).

    Two sources, in order of authority (GitLab #682):

    * ``options_used`` is the options struct the runner actually dispatched
      to the SCF (``run_job`` materialises ``RHFOptions()`` when the caller
      passes none, and a convergence retry may swap it). Reading the
      caller's struct instead made an explicit ``RHFOptions()`` cite the
      EDIIS+DIIS hybrid while the identical implicit run cited plain DIIS.
      The caller's struct is only the fallback when nothing was dispatched.
    * ``result.scf_trace[*].accelerator_step`` records the extrapolation
      that reached the Fock matrix in each iteration. For the switching
      hybrids (``EDIIS_DIIS`` / ``ADIIS_DIIS``) the hybrid is credited only
      when its energy-based branch was actually taken; a run whose switch
      metric stayed below the threshold executed plain DIIS and is cited
      as such. When the trace records no extrapolation at all (converged
      below ``diis_start_iter``, ``use_diis=False``, or a driver without the
      field) the configured accelerator is reported.
    """
    opts = options_used
    if opts is None:
        opts = {
            "rhf": rhf_options,
            "uhf": uhf_options,
            "rks": rks_options,
            "uks": uks_options,
        }.get(resolved_method)
    if opts is None:
        return None
    accel = getattr(opts, "scf_accelerator", None)
    if accel is None:
        return None
    name = (getattr(accel, "name", None) or str(accel)).strip().lower()
    steps = _executed_accelerator_steps(result)
    if not steps:
        return name
    if name == "ediis_diis":
        return "ediis_diis" if _ACCELERATOR_STEP_EDIIS in steps else "diis"
    if name == "adiis_diis":
        return "adiis_diis" if _ACCELERATOR_STEP_ADIIS in steps else "diis"
    return name


def _detect_uses_ecp(
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    *,
    result: object = None,
) -> bool:
    """Return True when any options struct carries a non-empty
    ``ecp_centers`` or inline primitive blocks -- the runtime signal that
    libecpint will build the V_ECP operator.

    ``result`` is authoritative when available. This catches ECP options
    materialised automatically from an orbital-basis sidecar rather than
    supplied by the caller. Before SCF, the explicit option fields remain the
    only available signal.
    """
    if result is not None:
        try:
            if molecular_result_has_ecp_operator(result):
                return True
        except ValueError:
            # Invalid provenance is conservatively ECP-active; the consumer
            # that needs the count will issue the precise validation error.
            return True
    for opts in (rhf_options, uhf_options, rks_options, uks_options):
        if molecular_options_request_ecp_operator(opts):
            return True
    return False


#: Criteria run when ``run_job(localize=...)`` is left at its default.
#: All three describe the same occupied space through different criteria and
#: are cheap next to the SCF that produced it; Boys is the outlier, and only
#: on aromatics, where its p=2 functional converges slowly (~89 sweeps on
#: benzene against IBO's 14). See handovers/HANDOVER_IBO.md.
DEFAULT_LOCALIZE_METHODS = ("ibo", "boys", "pipek-mezey")

_LOCALIZE_ALIASES = {
    "pm": "pipek-mezey",
    "pipek_mezey": "pipek-mezey",
    "pipek-mezey": "pipek-mezey",
    "foster-boys": "boys",
    "foster_boys": "boys",
    "boys": "boys",
    "ibo": "ibo",
}


def _resolve_localize_methods(localize: Any) -> tuple[str, ...]:
    """Normalise the ``localize=`` argument to a tuple of criteria.

    ``None`` / ``True`` -> the default set; ``False`` / ``"none"`` / an empty
    sequence -> nothing. A bare string or a sequence of strings selects
    specific criteria, accepting the usual spellings (``"pm"``,
    ``"pipek_mezey"``, ``"foster-boys"``, ...).
    """
    if localize is None or localize is True:
        return DEFAULT_LOCALIZE_METHODS
    if localize is False:
        return ()
    if isinstance(localize, str):
        if localize.strip().lower() in ("none", "off", ""):
            return ()
        localize = [localize]
    resolved: list[str] = []
    for raw in localize:
        key = str(raw).strip().lower()
        if key not in _LOCALIZE_ALIASES:
            raise ValueError(
                f"unknown localize criterion {raw!r}; expected one of "
                f"{sorted(set(_LOCALIZE_ALIASES.values()))}, or False to disable"
            )
        canonical = _LOCALIZE_ALIASES[key]
        if canonical not in resolved:
            resolved.append(canonical)
    return tuple(resolved)


def _detect_direct_scf(
    resolved_method: str,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    basis_name: str,
    molecule: Molecule,
) -> bool:
    """Return True when the SCF will use the direct (integral-driven)
    Fock build path -- either because ``scf_mode=DIRECT`` was explicitly
    set, or because AUTO resolved to DIRECT (basis larger than the
    auto threshold, and density_fit/cosx are off).

    Conservative: returns False when we can't determine the resolved
    mode (no options struct, or density_fit/cosx supersedes).
    """
    opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
    }.get(resolved_method)
    if opts is None:
        return False

    # density_fit + cosx supersede the four-index path -- direct SCF
    # screening is not used, so the direct-SCF references don't apply.
    if getattr(opts, "density_fit", False):
        return False

    scf_mode = getattr(opts, "scf_mode", None)
    if scf_mode is None:
        return False

    from ._vibeqc_core import SCFMode

    if scf_mode == SCFMode.DIRECT:
        return True
    if scf_mode == SCFMode.AUTO:
        # BUG 87: threshold lowered from 200 to 140.
        threshold = getattr(opts, "scf_mode_auto_threshold", 140)
        n_bf = BasisSet(molecule, basis_name).nbasis
        return n_bf > threshold
    # CONVENTIONAL -- not direct.
    return False


def _detect_acceleration(
    resolved_method: str,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
) -> list[str]:
    """Return the integral/exchange-acceleration technique keys engaged by
    the resolved SCF, for ``assemble(acceleration=...)``.

    * ``density_fit=True`` (RI-J Coulomb fitting) => ``["rij"]`` -- Whitten
      1973 / Dunlap 1979 / Eichkorn 1995 + 1997.
    * ``density_fit=True`` *and* ``cosx=True`` (RIJCOSX: RI-J Coulomb +
      chain-of-spheres exchange) => ``["rijcosx"]`` -- Neese 2009. COSX
      requires density fitting (the C++ JK dispatch rejects ``cosx`` without
      ``density_fit``), so a lone ``cosx`` flag still maps to RIJCOSX.

    Empty list for post-SCF / non-mean-field methods (no options struct in
    the rhf/uhf/rks/uks map) and for the conventional / direct four-index
    path (neither density_fit nor cosx set).
    """
    opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
    }.get(resolved_method)
    if opts is None:
        return []
    density_fit = bool(getattr(opts, "density_fit", False))
    cosx = bool(getattr(opts, "cosx", False))
    if cosx:  # COSX always pairs the RI-J Coulomb build -> RIJCOSX
        return ["rijcosx"]
    if density_fit:
        return ["rij"]
    return []


def _detect_soscf_trah(
    resolved_method: str,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
) -> tuple[bool, bool]:
    """Return ``(uses_soscf, uses_trah)`` for the resolved SCF.

    Both second-order convergers are armed by a *positive* threshold on the
    options struct (``soscf_threshold`` / ``trah_threshold``): the driver
    switches from Roothaan diagonalisation to the second-order step once the
    orbital-gradient norm drops below it (cpp/src/soscf.hpp, trah.hpp). A
    threshold of 0.0 (the default) means "never engage" -- no citation. Both
    False for methods with no options struct.
    """
    opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
    }.get(resolved_method)
    if opts is None:
        return (False, False)

    def _armed(attr: str) -> bool:
        try:
            return float(getattr(opts, attr, 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    return (_armed("soscf_threshold"), _armed("trah_threshold"))


def _detect_level_shift(
    resolved_method: str,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    rohf_options: Optional[ROHFOptions] = None,
    roks_options: Optional[ROKSOptions] = None,
) -> bool:
    """Return whether the resolved SCF runs with a Saunders-Hillier shift.

    True when the options struct carries a non-zero ``level_shift`` or a
    non-empty ``level_shift_schedule`` -- either arms the shift during
    iteration, so the Saunders-Hillier paper is cited. False for methods
    with no options struct.
    """
    opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
        "roks": roks_options,
    }.get(resolved_method)
    if opts is None:
        return False
    try:
        base = float(getattr(opts, "level_shift", 0.0) or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    return base != 0.0 or any(float(s) != 0.0 for s in schedule)


def _auto_open_shell_optimizer_trah(
    resolved_method: str,
    molecule: Molecule,
    *,
    optimize: bool,
    uhf_options: Optional[UHFOptions],
    uks_options: Optional[UKSOptions],
) -> tuple[Optional[UHFOptions], Optional[UKSOptions], bool]:
    """Use TRAH for molecular open-shell geometry optimizations by default.

    Stretched radicals can land in DIIS-stagnating regions after a perfectly
    valid optimizer step. TRAH is the shipped robust SCF route for that case.
    Only arm it when the caller has not explicitly selected SOSCF/TRAH/Newton.
    """
    if not optimize or molecule.multiplicity <= 1:
        return uhf_options, uks_options, False
    if resolved_method not in ("uhf", "uks"):
        return uhf_options, uks_options, False

    opts = uhf_options if resolved_method == "uhf" else uks_options
    if opts is None:
        opts = UHFOptions() if resolved_method == "uhf" else UKSOptions()

    def _positive(attr: str) -> bool:
        try:
            return float(getattr(opts, attr, 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    if any(
        _positive(attr)
        for attr in ("soscf_threshold", "trah_threshold", "newton_threshold")
    ):
        return uhf_options, uks_options, False

    opts.trah_threshold = 1.0
    if resolved_method == "uhf":
        return opts, uks_options, True
    return uhf_options, opts, True


def _copy_grid_options(src: GridOptions) -> GridOptions:
    dst = GridOptions()
    # Every public field (GitLab #663 made the list canonical); the copy used
    # to drop atomic_grid_profile and vv10_grid_factor, so a SKALA run that
    # entered the TRAH retry lost its atomic-grid profile.
    for attr in _GRID_OPTION_FIELDS:
        try:
            setattr(dst, attr, getattr(src, attr))
        except Exception:
            pass
    return dst


def _clone_mp2_like_options(src: object, cls: type) -> object:
    dst = cls()
    if src is None:
        return dst
    for attr in (
        "density_fit",
        "aux_basis",
        "c_os",
        "c_ss",
        "n_frozen_core",
        "use_float_intermediates",
        "report_ri_residual",
        "requested_memory_bytes",
        "memory_mode",
        "scratch_directory",
    ):
        if hasattr(dst, attr) and hasattr(src, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except Exception:
                pass
    return dst


def _correlated_memory_budget_bytes(explicit: Optional[int]) -> int:
    """Resolve the execution budget shared by preflight and native kernels.

    An explicit ``run_job(memory_budget_bytes=...)`` is an upper bound.  The
    scheduler/cgroup/host probe remains a second upper bound so a request can
    never silently exceed the process allocation.  A zero probe means the
    platform could not determine a limit; in that case only the explicit
    value is used.  Low-level MP2/CC callers can still set the corresponding
    option field directly.
    """
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int):
            raise ValueError("memory_budget_bytes must be a positive integer")
        if explicit <= 0:
            raise ValueError("memory_budget_bytes must be a positive integer")
        requested = explicit
    else:
        requested = 0
    available = int(available_memory_bytes())
    if requested and available:
        return min(requested, available)
    return requested or available


def _native_correlated_budget_bytes(process_budget: int, estimate: object) -> int:
    """Convert a process-wide cap into a native post-SCF workspace cap.

    ``MemoryEstimate.total_bytes`` applies its safety headroom to the whole
    phase, including the converged SCF matrices that stay live while MP2 or
    triples runs.  Native kernels report only their own integral/amplitude
    storage.  Giving them the full process allocation would let an automatic
    slab/tile planner consume the headroom and then make preflight reject its
    own plan.  Reserve the retained reference first and hand the remainder to
    the native planner.

    A return value of one byte is deliberate when the retained reference
    already exhausts the raw allowance: normal preflight rejects the job, and
    ``memory_override=True`` still cannot turn the native cap into an
    accidental unlimited (zero) budget.
    """
    if process_budget <= 0:
        return 0
    headroom = float(getattr(estimate, "headroom_factor", 1.0) or 1.0)
    if not math.isfinite(headroom) or headroom < 1.0:
        headroom = 1.0
    raw_allowance = int(process_budget / headroom)
    categories = getattr(estimate, "by_category", {}) or {}
    retained_reference = sum(
        int(categories.get(name, 0) or 0)
        for name in (
            "Fock + density + 1e",
            "MO workspace",
            "Open-shell UHF buffers",
            "BCCD persistent AO ERI",
        )
    )
    return max(1, raw_allowance - retained_reference)


def _check_native_correlated_budget(estimate: object, option: object) -> None:
    """Fail preflight when a native route cannot realize its own cap.

    Process admission and the native workspace contract are deliberately
    different: the former includes retained reference arrays and headroom,
    while the latter covers the MP2 or triples kernel.  A one-orbital slab or
    scalar triples floor can therefore fit the process allocation while still
    exceeding a tighter cap supplied directly on the method options.  Catch
    that mismatch before SCF rather than letting the native kernel fail late.
    """
    budget = int(getattr(option, "requested_memory_bytes", 0) or 0)
    if budget <= 0:
        return

    dims = getattr(estimate, "dims", {}) or {}
    if isinstance(option, (MP2Options, UMP2Options)):
        workspace = int(dims.get("mp2_workspace_bytes", 0) or 0)
        label = "MP2 occupied-orbital slab"
    elif bool(getattr(option, "compute_triples", False)):
        workspace = int(dims.get("triples_workspace_bytes", 0) or 0)
        label = "CCSD(T) triples live state"
    else:
        return

    if workspace > budget:
        raise InsufficientMemoryError(
            f"Requested native memory budget {budget} bytes is below the "
            f"minimum modeled {label} workspace of {workspace} bytes. "
            "Increase the method or run_job memory budget, reduce the active "
            "space, or select a route with a smaller irreducible live state."
        )


def _apply_correlated_process_budget(option: object, process_budget: int) -> bool:
    """Cap one prepared native option at the process-wide allowance."""

    if option is None or process_budget <= 0:
        return False
    current = int(getattr(option, "requested_memory_bytes", 0) or 0)
    resolved = min(current, process_budget) if current else process_budget
    if resolved == current:
        return False
    option.requested_memory_bytes = resolved
    return True


def _refine_native_correlated_budgets(
    options: list[object],
    process_budget: int,
    estimate: object,
) -> tuple[int, bool]:
    """Reserve reference/headroom bytes and cap every native planner.

    Returns the smallest allowance actually handed to a kernel and whether
    any prepared option changed. The caller re-estimates after a change so
    scheduler dry-runs and real admission see the same slab/tile plan.
    """

    if process_budget <= 0 or not options:
        return 0, False
    allowance = _native_correlated_budget_bytes(process_budget, estimate)
    changed = False
    resolved_budgets = []
    for option in options:
        current = int(getattr(option, "requested_memory_bytes", 0) or 0)
        resolved = min(current, allowance) if current else allowance
        resolved_budgets.append(resolved)
        if resolved != current:
            option.requested_memory_bytes = resolved
            changed = True
    return min(resolved_budgets), changed


def _auto_resolve_mp2_aux_basis(
    opts: object, basis: BasisSet, molecule: object = None
) -> None:
    if not bool(getattr(opts, "density_fit", False)):
        return
    if not getattr(opts, "aux_basis", ""):
        try:
            from .density_fitting import default_aux_basis_for

            opts.aux_basis = default_aux_basis_for(basis.name, kind="ri")
        except Exception:
            # Leave empty so the native MP2/UMP2 driver raises its explicit
            # aux_basis error with the same autodetection hint as direct
            # calls.
            return
    if molecule is not None:
        # #480: the native DF-MP2 driver builds its own auxiliary BasisSet
        # from opts.aux_basis, so the DensityFitting guard never sees it.
        # Refuse an uncovered element before any fitted correlation work.
        from .density_fitting import guard_aux_coverage_for_options

        guard_aux_coverage_for_options(
            opts, molecule, basis, route="run_job (DF-MP2 auxiliary)"
        )


def _clone_rks_options_for_trah_retry(
    opts: RKSOptions,
    result: object,
) -> RKSOptions:
    retry = RKSOptions()
    for attr in (
        "functional",
        "max_iter",
        "conv_tol_energy",
        "conv_tol_grad",
        "damping",
        "dynamic_damping",
        "dynamic_damping_min",
        "dynamic_damping_max",
        "fock_mixing",
        "use_diis",
        "diis_start_iter",
        "diis_subspace_size",
        "diis_restart_tau",
        "diis_adaptive_delta",
        "scf_accelerator",
        "ediis_diis_switch_threshold",
        "linear_dep_threshold",
        "ecp_centers",
        "ecp_library",
        "ecp_primitive_blocks",
        "ecp_primitive_centers",
        "ecp_effective_charges",
        "ecp_total_ncore",
        "level_shift",
        "level_shift_warmup_cycles",
        "level_shift_schedule",
        "quadratic_fallback_iter",
        "quadratic_fallback_shift",
        "quadratic_fallback_max_step",
        "newton_threshold",
        "newton_opts",
        "soscf_threshold",
        "soscf_opts",
        "trah_opts",
        "density_fit",
        "aux_basis",
        "scf_mode",
        "scf_mode_auto_threshold",
        "schwarz_threshold",
        "incremental_fock",
        "incremental_fock_reset_freq",
        "schwarz_threshold_loose",
        "schwarz_threshold_tighten_at",
        "cosx",
        "cosx_variant",
        "cosx_grid_level",
        "dft_plus_u_sites",
        "dft_plus_u_ao_groups",
        "use_davidson",
        "davidson",
        "davidson_min_dim",
        "cosx",
        "cosx_grid",
        "cosx_variant",
        "cosx_grid_level",
        "dft_plus_u_sites",
        "dft_plus_u_ao_groups",
    ):
        try:
            setattr(retry, attr, getattr(opts, attr))
        except Exception:
            pass
    retry.grid = _copy_grid_options(opts.grid)
    retry.cosx_grid = _copy_grid_options(opts.cosx_grid)
    retry.initial_guess = InitialGuess.READ
    retry.read_path = ""
    retry.read_density = np.asarray(getattr(result, "density"), dtype=float)
    retry.trah_threshold = max(float(getattr(opts, "trah_threshold", 0.0)), 1.0e-2)
    retry.max_iter = max(40, min(int(getattr(opts, "max_iter", 100)), 80))
    return retry


def _clone_rhf_options_for_sad_retry(opts: RHFOptions) -> RHFOptions:
    retry = RHFOptions()
    for attr in (
        "max_iter",
        "conv_tol_energy",
        "conv_tol_grad",
        "damping",
        "dynamic_damping",
        "dynamic_damping_min",
        "dynamic_damping_max",
        "fock_mixing",
        "use_diis",
        "diis_start_iter",
        "diis_subspace_size",
        "diis_restart_tau",
        "diis_adaptive_delta",
        "scf_accelerator",
        "ediis_diis_switch_threshold",
        "linear_dep_threshold",
        "ecp_centers",
        "ecp_library",
        "ecp_primitive_blocks",
        "ecp_primitive_centers",
        "ecp_effective_charges",
        "ecp_total_ncore",
        "level_shift",
        "level_shift_warmup_cycles",
        "level_shift_schedule",
        "quadratic_fallback_iter",
        "quadratic_fallback_shift",
        "quadratic_fallback_max_step",
        "newton_threshold",
        "newton_opts",
        "soscf_threshold",
        "soscf_opts",
        "trah_opts",
        "density_fit",
        "aux_basis",
        "scf_mode",
        "scf_mode_auto_threshold",
        "schwarz_threshold",
        "incremental_fock",
        "incremental_fock_reset_freq",
        "schwarz_threshold_loose",
        "schwarz_threshold_tighten_at",
        "use_davidson",
        "davidson",
        "davidson_min_dim",
        "cosx",
        "cosx_grid",
        "cosx_variant",
        "cosx_grid_level",
        "dft_plus_u_sites",
        "dft_plus_u_ao_groups",
    ):
        try:
            setattr(retry, attr, getattr(opts, attr))
        except Exception:
            pass
    retry.initial_guess = InitialGuess.SAD
    retry.read_path = ""
    retry.soscf_threshold = 0.0
    retry.trah_threshold = 0.0
    retry.max_iter = max(120, int(getattr(opts, "max_iter", 100)))
    return retry


def _rhf_tail_sad_retry_supported(
    *,
    resolved_method: str,
    molecule: Molecule,
    basis: BasisSet,
    rhf_options: Optional[RHFOptions],
    result: object,
) -> bool:
    if resolved_method != "rhf" or rhf_options is None:
        return False
    if molecule.multiplicity != 1 or molecule.n_electrons() % 2 != 0:
        return False
    if int(getattr(result, "n_iter", 0) or 0) < 50:
        return False
    if int(getattr(basis, "nbasis", 0) or 0) < 150 and len(list(molecule.atoms)) < 20:
        return False
    density = getattr(result, "density", None)
    if density is None:
        return False
    if np.asarray(density).shape != (basis.nbasis, basis.nbasis):
        return False
    initial_guess = getattr(rhf_options, "initial_guess", None)
    if initial_guess in (InitialGuess.SAD, InitialGuess.READ, InitialGuess.FRAGMO):
        return False

    def _positive(attr: str) -> bool:
        try:
            return float(getattr(rhf_options, attr, 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    if any(
        _positive(attr)
        for attr in ("newton_threshold", "soscf_threshold", "trah_threshold")
    ):
        return False
    if int(getattr(rhf_options, "quadratic_fallback_iter", 0) or 0) > 0:
        return False
    return True


def _rks_tail_trah_retry_supported(
    *,
    resolved_method: str,
    molecule: Molecule,
    basis: BasisSet,
    functional: Optional[str],
    rks_options: Optional[RKSOptions],
    result: object,
) -> bool:
    if resolved_method != "rks" or rks_options is None:
        return False
    if molecule.multiplicity != 1 or molecule.n_electrons() % 2 != 0:
        return False
    if int(getattr(result, "n_iter", 0) or 0) < 50:
        return False
    if int(getattr(basis, "nbasis", 0) or 0) < 150 and len(list(molecule.atoms)) < 20:
        return False
    density = getattr(result, "density", None)
    if density is None:
        return False
    if np.asarray(density).shape != (basis.nbasis, basis.nbasis):
        return False

    def _positive(attr: str) -> bool:
        try:
            return float(getattr(rks_options, attr, 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    if any(
        _positive(attr)
        for attr in ("newton_threshold", "soscf_threshold", "trah_threshold")
    ):
        return False
    if int(getattr(rks_options, "quadratic_fallback_iter", 0) or 0) > 0:
        return False

    try:
        xc = Functional(functional or rks_options.functional or "lda")
    except Exception:
        return False
    if bool(getattr(xc, "is_range_separated", False)):
        return False
    if bool(getattr(xc, "is_double_hybrid", False)):
        return False
    return xc.kind in (XCKind.LDA, XCKind.GGA)


def _memory_estimator_method(resolved_method: str, effective_method: str) -> str:
    """Method label to hand to the memory estimator."""
    resolved = str(resolved_method or "").lower()
    effective = str(effective_method or resolved).lower()
    active_space_methods = {
        "casci",
        "casscf",
        "caspt2",
        "nevpt2",
        "fci",
        "mrci",
        "ci",
    }
    if effective in ("mp2", "scs-mp2", "sos-mp2") and resolved == "uhf":
        return "ump2"
    if effective in ("ccsd", "ccsd(t)", "ccsd[t]", "a-ccsd(t)") and resolved in ("uhf", "rohf"):
        return {
            "ccsd": "uccsd",
            "ccsd(t)": "uccsd(t)",
            "ccsd[t]": "uccsd(t)",
            "a-ccsd(t)": "uccsd(t)",
        }[effective]
    if (
        effective in _POSTSCF_CITE_METHODS
        or effective in _CC_VARIANT_METHODS
        or effective in active_space_methods
    ):
        return effective
    return {"rohf": "uhf", "roks": "uks"}.get(resolved, resolved)


def _memory_reference_options(
    resolved_method: str,
    *,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    rohf_options: Optional[ROHFOptions],
    roks_options: Optional[ROKSOptions],
):
    return {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
        "roks": roks_options,
    }.get(resolved_method)


def _memory_estimator_options(
    estimator_method: str,
    *,
    resolved_method: str,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    rohf_options: Optional[ROHFOptions],
    roks_options: Optional[ROKSOptions],
    mp2_options: Optional[object],
    ump2_options: Optional[object],
    ccsd_options: Optional[object],
    cc3_options: Optional[CC3Options],
    ccsdt_options: Optional[CCSDTOptions],
    dlpno_options: Optional[object],
    dlpno_ccsd_options: Optional[object],
    caspt2_options: Optional[CASPT2Options],
    nevpt2_options: Optional[NEVPT2Options],
    active_space: Optional[tuple[int, int]],
    selected_ci_options: Optional[SelectedCIOptions] = None,
    tddft: bool = False,
    tddft_n_states: int = 5,
    tddft_type: str = "tda",
    functional: Optional[str] = None,
):
    """Options bundle for the pre-flight memory estimator."""
    estimator_method = str(estimator_method).lower()
    scf_options = _memory_reference_options(
        resolved_method,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
        rohf_options=rohf_options,
        roks_options=roks_options,
    )
    if estimator_method in ("rhf", "uhf", "rks", "uks"):
        if tddft:
            td_functional = functional
            if td_functional is None and scf_options is not None:
                td_functional = getattr(scf_options, "functional", None)
            return {
                "scf_options": scf_options,
                "tddft": True,
                "tddft_n_states": tddft_n_states,
                "tddft_type": tddft_type,
                "functional": td_functional,
            }
        if estimator_method in ("rks", "uks") and functional is not None:
            return {
                "scf_options": scf_options,
                "functional": functional,
            }
        return scf_options
    if resolved_method in ("rohf", "roks") and tddft:
        return {
            "scf_options": scf_options,
            "tddft": True,
            "tddft_n_states": tddft_n_states,
            "tddft_type": tddft_type,
            "functional": functional,
        }
    if estimator_method in ("mp2", "scs-mp2", "sos-mp2"):
        return {"scf_options": scf_options, "mp2_options": mp2_options}
    if estimator_method in ("ump2", "rohf-mp2"):
        return {"scf_options": scf_options, "ump2_options": ump2_options}
    if estimator_method == "ovgf":
        return {"scf_options": scf_options}
    if estimator_method in (
        "ccsd",
        "ccsd(t)",
        "ccsd[t]",
        "a-ccsd(t)",
        "uccsd",
        "uccsd(t)",
        "cc2",
        "ccd",
        "lccd",
        "lccsd",
        "cepa(0)",
        "cepa(1)",
        "cepa(2)",
        "cepa(3)",
        "qcisd",
        "qcisd(t)",
        "bccd",
        "bccd(t)",
    ):
        return {"scf_options": scf_options, "ccsd_options": ccsd_options}
    if estimator_method == "cc3":
        return {
            "scf_options": scf_options,
            "cc3_options": cc3_options,
            "active_space": active_space,
        }
    if estimator_method == "ccsdt":
        return {
            "scf_options": scf_options,
            "ccsdt_options": ccsdt_options,
            "active_space": active_space,
        }
    if estimator_method in ("dlpno-mp2", "dlpno-ump2"):
        option_key = (
            "dlpno_ump2_options"
            if estimator_method == "dlpno-ump2"
            else "dlpno_options"
        )
        return {"scf_options": scf_options, option_key: dlpno_options}
    if estimator_method in (
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
        "dlpno-uccsd",
        "dlpno-uccsd(t)",
    ):
        return {
            "scf_options": scf_options,
            "dlpno_ccsd_options": dlpno_ccsd_options,
        }
    if estimator_method in ("casci", "casscf"):
        return {"scf_options": scf_options, "active_space": active_space}
    if estimator_method == "caspt2":
        return {
            "scf_options": scf_options,
            "active_space": active_space,
            "caspt2_options": caspt2_options,
        }
    if estimator_method == "nevpt2":
        return {
            "scf_options": scf_options,
            "active_space": active_space,
            "nevpt2_options": nevpt2_options,
        }
    if estimator_method == "selected_ci":
        # The selected-CI estimator is a function of both: active_space bounds
        # how many determinants can exist, selected_ci_options.target_size
        # bounds how many the solver will actually hold. Withholding them is
        # half of issue #79 -- the estimate could not vary with target_size
        # because target_size never reached the estimator.
        return {
            "scf_options": scf_options,
            "active_space": active_space,
            "selected_ci_options": selected_ci_options,
        }
    if estimator_method in (
        "cisd",
        "dmrg",
        "v2rdm",
        "transcorrelated_ci",
        "mrci",
        "fci",
    ):
        return {"scf_options": scf_options}
    return None


def _apply_molecular_smearing(
    result,
    *,
    n_electrons: int,
    smearing_temperature: float,
    smearing_method: str,
) -> object:
    """Apply finite-temperature smearing post-SCF to a molecular result.

    Computes the Mermin free energy ``A = E - T·S`` from the converged
    integer-occupation SCF eigenvalues.  The wrapped result carries the
    original ``.energy`` (integer-Aufbau value) unchanged and exposes
    ``.free_energy``, ``.smearing_entropy``, ``.smearing_temperature``,
    and ``.smearing_fermi_level``.

    Returns the original result unchanged when ``smearing_temperature <= 0``
    or when the result has no ``mo_energies`` attribute (non-mean-field
    methods).
    """
    if smearing_temperature <= 0.0:
        return result
    mo_energies = getattr(result, "mo_energies", None)
    if mo_energies is None:
        return result

    from .smearing import SmearingOptions, apply_smearing

    import numpy as np

    eps = np.asarray(np.real(mo_energies), dtype=float)
    n_occ = max(0, (int(n_electrons) + 1) // 2)  # closed-shell Aufbau count

    smearing = SmearingOptions(
        temperature=float(smearing_temperature),
        flavor=str(smearing_method),
    )
    smear_result = apply_smearing(
        [eps],
        weights=[1.0],
        n_electrons_per_cell=float(n_electrons),
        n_occ_each=n_occ,
        smearing=smearing,
    )
    free_energy = float(getattr(result, "energy", 0.0)) + float(
        smear_result.free_energy_correction
    )
    return _FreeEnergyAugmented(
        result,
        free_energy=free_energy,
        entropy=float(smear_result.entropy),
        temperature=float(smearing_temperature),
        mu=float(smear_result.mu),
        method=str(smearing_method),
    )


# SCF initial-guess enum names -> citation-route keys. Only guesses with a
# defining-paper route appear. The traditional Hcore guess maps to nothing;
# AUTO is resolved to its effective kind before this table is consulted.
_SCF_GUESS_ROUTE_KEYS = {
    "sad": "sad",
    "sap": "sap",
    "hueckel": "huckel",
    "huckel": "huckel",
    "patom": "patom",
    "minao": "minao",
}


# Canonical normalisation now lives in ``vibeqc._initial_guess`` so the
# pure-Python ROHF/ROKS drivers can share it (this module imports them, so
# they cannot import it back).
_coerce_initial_guess = coerce_initial_guess


def _materialize_run_job_initial_guess(
    resolved_method: str,
    initial_guess: Optional[object],
    *,
    rhf_options: Optional[RHFOptions],
    uhf_options: Optional[UHFOptions],
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    rohf_options: Optional[ROHFOptions],
    roks_options: Optional[ROKSOptions],
    read_from: Optional[object],
    fragments: Optional[object],
) -> tuple[object, object, object, object, object, object]:
    if initial_guess is None and read_from is None and fragments is None:
        return (
            rhf_options,
            uhf_options,
            rks_options,
            uks_options,
            rohf_options,
            roks_options,
        )

    option_map = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
        "roks": roks_options,
    }
    factories = {
        "rhf": RHFOptions,
        "uhf": UHFOptions,
        "rks": RKSOptions,
        "uks": UKSOptions,
        "rohf": ROHFOptions,
        "roks": ROKSOptions,
    }
    if resolved_method not in factories:
        raise ValueError(
            "run_job(initial_guess=...) only applies to molecular SCF-backed "
            "methods (RHF/UHF/RKS/UKS and their MP2/CCSD/DLPNO references). "
            f"method resolved to {resolved_method!r}."
        )

    if initial_guess is None:
        opts = option_map[resolved_method]
        guess = getattr(opts, "initial_guess", None) if opts is not None else None
        if guess is None:
            raise ValueError(
                "read_from= and fragments= are only used with READ/FRAGMO. "
                "Pass initial_guess='read' or initial_guess='fragmo', or set "
                "the resolved method options' initial_guess accordingly."
            )
        initial_guess = guess

    guess = _coerce_initial_guess(initial_guess)

    opt = option_map[resolved_method] or factories[resolved_method]()
    opt.initial_guess = guess
    option_map[resolved_method] = opt

    return (
        option_map["rhf"],
        option_map["uhf"],
        option_map["rks"],
        option_map["uks"],
        option_map["rohf"],
        option_map["roks"],
    )


def _prepare_molecular_special_guess(
    method: str,
    options: object,
    molecule: Molecule,
    basis: BasisSet,
    *,
    read_from: Optional[object],
    fragments: Optional[object],
) -> None:
    from .guess import prepare_molecular_guess_source
    prepare_molecular_guess_source(
        method, options, molecule, basis, read_from=read_from, fragments=fragments,
    )


def _detect_scf_guess(
    resolved_method: str,
    molecule: Molecule,
    scf_options: object,
) -> Optional[str]:
    """Return the citation route for the initial guess that actually ran.

    Explicit selections map directly. ``AUTO`` is resolved through the same
    molecule-aware native policy as :class:`vibeqc::GuessEngine`, so a light
    closed shell that executes PATOM cites PATOM, a multi-atom open-shell or
    transition/f-block system that executes SAD cites SAD, and an isolated
    open-shell atom that executes PATOM cites PATOM (issue 668). Hcore has no
    single defining publication.
    """
    if resolved_method not in ("rhf", "uhf", "rks", "uks", "rohf", "roks"):
        return None
    guess = getattr(scf_options, "initial_guess", None)
    if guess is None:
        return None
    from .guess import select_initial_guess

    selection = select_initial_guess(
        molecule, guess,
        is_open_shell=resolved_method in ("uhf", "uks", "rohf", "roks"),
        atomic_spins=getattr(scf_options, "atomic_spins", None),
    )
    return _SCF_GUESS_ROUTE_KEYS.get(selection.effective.name.lower())


def _run_single_point(
    method: str,
    molecule: Molecule,
    basis: BasisSet,
    *,
    functional: Optional[str],
    rhf_options: Optional[RHFOptions] = None,
    uhf_options: Optional[UHFOptions] = None,
    rks_options: Optional[RKSOptions] = None,
    uks_options: Optional[UKSOptions] = None,
    rohf_options: Optional[ROHFOptions] = None,
    roks_options: Optional[ROKSOptions] = None,
    cisd_options: Optional[CISDOptions] = None,
    cc3_options: Optional[CC3Options] = None,
    ccsdt_options: Optional[CCSDTOptions] = None,
    selected_ci_options: Optional[SelectedCIOptions] = None,
    dmrg_options: Optional[DMRGOptions] = None,
    v2rdm_options: Optional[V2RDMOptions] = None,
    transcorrelated_options: Optional[TranscorrelatedOptions] = None,
    casci_options: Optional[CASCIOptions] = None,
    caspt2_options: Optional[CASPT2Options] = None,
    nevpt2_options: Optional[NEVPT2Options] = None,
    casscf_options: Optional[CASSCFOptions] = None,
    active_space: Optional[tuple[int, int]] = None,
    cas_reference: Optional[str] = None,
    mlip_options: Optional[MLIPOptions] = None,
    ccm_options: Optional["CCMOptions"] = None,
    solvent: object = None,
    dft_plus_u: Optional[List["HubbardSite"]] = None,
    read_from: Optional[object] = None,
    fragments: Optional[object] = None,
    nddo: bool = False,
    grid_level: str = "orca-defgrid3",
    _mrpt_gradient: bool = True,
    used_scf_options_out: Optional[dict] = None,
):
    """Single-point dispatcher. ``solvent`` (v0.9.0) reroutes through
    :func:`vibeqc.solvation.run_cpcm_scf` and returns the underlying
    SCF result with an ``e_solv`` / ``solvent_result`` attribute
    attached for downstream output writers."""
    # Implicit solvation composes with the mean-field SCFs only (CPCM via
    # run_cpcm_scf: rhf/uhf/rks/uks) plus the dedicated MSINDO COSMO route
    # below; rohf/roks are refused by the solvent gate below.
    # Every other method used to *silently ignore* ``solvent`` and return a
    # gas-phase energy -- a solvated CASSCF request would quietly compute in
    # vacuum with no warning. Refuse instead (same discipline and wording as
    # the _run_molecular_scf gate in molecular_optimize.py). This also
    # covers the FD-optimizer energy path (_evaluate_energy forwards
    # ``solvent`` here), which previously relied on the silent drop and
    # walked the gas-phase surface while claiming solvation.
    if solvent is not None and method not in (
        "rhf",
        "uhf",
        "rks",
        "uks",
        "msindo",
        "rohf",
        "roks",
    ):
        raise ValueError(
            f"Implicit solvation is not supported for method={method!r}: "
            "CPCM (run_cpcm_scf) composes with rhf, uhf, rks, and uks "
            "only, and COSMO with method='msindo'. Run the calculation in "
            "gas phase, or use a solvent-capable method with solvent."
        )
    method_opts = {
        "rhf": rhf_options or RHFOptions(),
        "uhf": uhf_options or UHFOptions(),
    }
    if method in ("rks", "uks"):
        opts = (rks_options if method == "rks" else uks_options) or (
            RKSOptions() if method == "rks" else UKSOptions()
        )
        opts.functional = functional or opts.functional or "lda"
        # Apply grid_level to an untouched grid, whether the options object
        # was default-constructed here or passed in by the caller (GitLab
        # #663: a passed RKSOptions() used to opt the run out of the default
        # silently). A grid the caller customised wins.
        _user_provided_opts = rks_options if method == "rks" else uks_options
        if ks_options_need_grid_default(_user_provided_opts):
            _resolved_grid_level = grid_level
            if (
                str(opts.functional).strip().lower()
                in {"skala", "skala-1.1", "skala-1.1-rev1"}
                and grid_level == "orca-defgrid3"
            ):
                _resolved_grid_level = "skala"
            _apply_grid_level(opts.grid, _resolved_grid_level)
        method_opts[method] = opts

    if method in ("rhf", "uhf", "rks", "uks"):
        # A bundled sidecar is an authoritative basis/ECP pairing, so attach
        # it before enforcing BUG 99. If attachment cannot configure the
        # molecule, validation still refuses the unsafe all-electron route.
        attach_inline_ecp_options_from_basis_sidecar(
            method_opts[method], molecule, basis
        )
        _basis_name = str(getattr(basis, "name", "") or "").strip()
        if _basis_name:
            validate_ecp_required(
                method_opts[method], molecule, _basis_name
            )
        # Density-fitted / RIJCOSX SCF needs a JK auxiliary basis. If the
        # caller enabled density_fit or cosx but left aux_basis empty,
        # auto-resolve the matching JKfit aux from the orbital basis --
        # the same convenience the CCSD RI path provides -- instead of
        # letting the C++ driver reject the run with a bare "density_fit=
        # true requires aux_basis" (release-paper RIJCOSX glycine gap).
        _scf_opts = method_opts[method]
        _wants_df = bool(getattr(_scf_opts, "density_fit", False)) or bool(
            getattr(_scf_opts, "cosx", False)
        )
        if _wants_df and not getattr(_scf_opts, "aux_basis", ""):
            from .density_fitting import default_aux_basis_for

            # basis here is a BasisSet; the resolver takes the orbital-basis
            # name string.
            _basis_name = getattr(basis, "name", basis)
            try:
                _scf_opts.aux_basis = default_aux_basis_for(
                    _basis_name, kind="jk"
                )
            except Exception:
                # No registered default (e.g. pob-* bases): leave empty so
                # the driver raises its explicit aux_basis error, which now
                # also names the auto-resolve fallback.
                pass

        # #480: refuse an auxiliary basis with no functions on a centre
        # that carries orbital functions, before any fitted SCF work.
        # This module binds the *bare* C++ SCF entry points -- it is
        # imported from vibeqc/__init__.py before the Python wrappers are
        # defined -- so the guard inside those wrappers never runs on the
        # run_job path. Check here instead.
        from .density_fitting import guard_aux_coverage_for_options

        guard_aux_coverage_for_options(
            _scf_opts, molecule, basis, route=f"run_job ({method} SCF)"
        )
        # Hand the options object the SCF will actually run with (ECP fields
        # auto-attached above) back to the caller: run_job needs it for every
        # electrostatic property, which must use Z - n_core on ECP atoms
        # (#642). The route kwarg may have been None, so the caller cannot
        # reconstruct this object itself.
        if used_scf_options_out is not None:
            used_scf_options_out["scf_options"] = _scf_opts

    if method in ("rhf", "uhf", "rks", "uks"):
        _prepare_molecular_special_guess(
            method,
            method_opts[method],
            molecule,
            basis,
            read_from=read_from,
            fragments=fragments,
        )

    if solvent is not None and method in ("rhf", "uhf", "rks", "uks"):
        if dft_plus_u:
            raise NotImplementedError(
                "DFT+U combined with implicit solvation (CPCM) is not yet "
                "supported. The +U Fock term and the solvation macro-"
                "iteration would need to be combined inside "
                "vibeqc.solvation.run_cpcm_scf; defer until Increment "
                "3+ or run +U without solvent."
            )
        # CPCM macro-iteration wrapper. The C++ result types
        # (RHFResult / UHFResult / RKSResult / UKSResult) are
        # pybind11 ``def_readonly`` -- Python attribute set on them
        # silently fails inside try/except. Wrap the inner result in
        # an attribute-forwarding proxy so callers can still write
        # ``result.energy`` / ``result.density`` / etc., and *also*
        # read the new solvation diagnostics (``result.solvent_result``,
        # ``result.e_solv``, ``result.energy_in_solvent``).
        from .solvation import run_cpcm_scf
        from .solvation.driver import _solvent_aware_scf_result

        sol = run_cpcm_scf(
            molecule,
            basis,
            method=method,
            solvent=solvent,
            options=method_opts[method],
        )
        return _solvent_aware_scf_result(sol)

    if dft_plus_u and method in ("rhf", "uhf", "rks", "uks"):
        from .dft_plus_u import _apply_dft_plus_u_to_options

        _apply_dft_plus_u_to_options(method_opts[method], basis, dft_plus_u)


    if method == "rohf":
        # Restricted open-shell HF (pure-Python Roothaan driver). Solvent
        # / DFT+U are not yet combined with the ROHF Fock -- gate clearly
        # rather than silently ignoring them (roadmap: handovers/HANDOVER_ROHF.md).
        if solvent is not None:
            raise NotImplementedError(
                "Implicit solvation (CPCM) with method='rohf' is not yet "
                "supported; run ROHF in gas phase or use uhf/rks. See "
                "handovers/HANDOVER_ROHF.md."
            )
        if dft_plus_u:
            raise NotImplementedError(
                "DFT+U with method='rohf' is not yet supported. See handovers/HANDOVER_ROHF.md."
            )
        _rohf_opts = rohf_options or ROHFOptions()
        if used_scf_options_out is not None:
            used_scf_options_out["scf_options"] = _rohf_opts
        _prepare_molecular_special_guess(
            method, _rohf_opts, molecule, basis,
            read_from=read_from, fragments=fragments,
        )
        return run_rohf(molecule, basis, _rohf_opts)

    if method == "roks":
        # Restricted open-shell KS (pure-Python Roothaan driver + libxc XC).
        if solvent is not None:
            raise NotImplementedError(
                "Implicit solvation (CPCM) with method='roks' is not yet "
                "supported; run ROKS in gas phase or use uks. See "
                "handovers/HANDOVER_ROHF.md."
            )
        if dft_plus_u:
            raise NotImplementedError(
                "DFT+U with method='roks' is not yet supported. See handovers/HANDOVER_ROHF.md."
            )
        _roks_opts = roks_options or ROKSOptions()
        if ks_options_need_grid_default(roks_options):
            # ROKSOptions is a pure-Python dataclass whose ``grid`` defaults
            # to None (the driver materialises a default GridOptions when it
            # sees None), while the C++ RKS/UKS options default to a live
            # GridOptions instance. _apply_grid_level mutates the grid
            # in-place, so a bare default ROKS run crashed with
            # ``AttributeError: 'NoneType' object has no attribute ...``
            # before ever reaching the driver (e.g. run_job(method="roks",
            # basis="def2-tzvp")). Materialise the grid first, mirroring
            # run_roks' own default construction.
            if _roks_opts.grid is None:
                _roks_opts.grid = GridOptions()
            _resolved_grid_level = (
                "skala"
                if str(functional or _roks_opts.functional).strip().lower()
                in {"skala", "skala-1.1", "skala-1.1-rev1"}
                and grid_level == "orca-defgrid3"
                else grid_level
            )
            _apply_grid_level(_roks_opts.grid, _resolved_grid_level)
        if used_scf_options_out is not None:
            used_scf_options_out["scf_options"] = _roks_opts
        _prepare_molecular_special_guess(
            method, _roks_opts, molecule, basis,
            read_from=read_from, fragments=fragments,
        )
        return run_roks(
            molecule,
            basis,
            _roks_opts,
            functional=functional,
        )

    if method == "rhf":
        return run_rhf(molecule, basis, method_opts["rhf"])
    if method == "uhf":
        return run_uhf(molecule, basis, method_opts["uhf"])
    if method == "rks":
        return run_rks(molecule, basis, method_opts["rks"])
    if method == "uks":
        return run_uks(molecule, basis, method_opts["uks"])

    # ── Non-mean-field wavefunction methods ──
    if method in (
        "cisd",
        "cc3",
        "ccsdt",
        "selected_ci",
        "dmrg",
        "v2rdm",
        "transcorrelated_ci",
        "casci",
        "mrci",
        "casscf",
        "nevpt2",
        "caspt2",
    ):
        # Starting orbitals for the determinant-solver family.  The
        # open-shell default is UHF natural orbitals (UNO-CAS -- Pulay &
        # Hamilton, J. Chem. Phys. 88, 4926 (1988)): one spin-restricted
        # orbital set ordered by descending occupation, so the
        # doubly-occupied core and the active window are well-defined.
        # (Plain "uhf" keeps only the a orbitals; the b space -- and hence
        # the core -- is then only approximately represented.)  Override
        # with cas_reference="rhf"|"uhf"|"uno"|"rohf".  "rohf" gives a
        # spin-pure restricted-open-shell reference (Roothaan 1960) -- a
        # clean, contamination-free alternative to UNO for the active space.
        if cas_reference is not None and cas_reference not in (
            "rhf",
            "uhf",
            "uno",
            "rohf",
        ):
            raise ValueError(
                f"cas_reference must be 'rhf', 'uhf', 'uno' or 'rohf', got "
                f"{cas_reference!r}"
            )
        if method == "cc3":
            if molecule.multiplicity != 1:
                raise ValueError(
                    "CC3 requires a closed-shell singlet RHF reference"
                )
            if cas_reference not in (None, "rhf"):
                raise ValueError(
                    "CC3 supports cas_reference='rhf' only; its ground-state "
                    "equations require canonical closed-shell orbitals"
                )
            hf_method = "rhf"
        elif method == "ccsdt":
            hf_method = cas_reference or (
                "rohf" if molecule.multiplicity > 1 else "rhf"
            )
        else:
            hf_method = cas_reference or (
                "uno" if molecule.multiplicity > 1 else "rhf"
            )
        _iterative_cc_reference_result = None
        # The caller's SCF options must reach the reference. An ECP given
        # only through rhf_options / uhf_options (no basis sidecar) lives
        # nowhere else, and dropping it is silent: the run completes on a
        # bare-Z Hamiltonian with the physical electron count (#740).
        _ref_scf_options = (
            uhf_options if str(hf_method).lower() in ("uhf", "uno")
            else rhf_options
        )
        if method in ("cc3", "ccsdt"):
            C, _iterative_cc_reference_result = get_hf_orbital_provider(
                molecule,
                basis,
                method=hf_method,
                _with_result=True,
                scf_options=_ref_scf_options,
            )
        else:
            C, _iterative_cc_reference_result = get_hf_orbital_provider(
                molecule,
                basis,
                method=hf_method,
                _with_result=True,
                scf_options=_ref_scf_options,
            )
        # The reference carries the ECP operator (both routes) and the
        # valence count; the determinant Hamiltonian is built from it so
        # the solver diagonalises the operator the mean field solved.
        H = build_hamiltonian_mo(
            molecule, basis, C, reference=_iterative_cc_reference_result
        )

        # Optional active-space truncation.  The CAS methods (casci / nevpt2 /
        # caspt2) are handled separately below via casci()'s own frozen-core
        # path; these determinant/DMRG/v2RDM backends get the same standard
        # CAS partition through Hamiltonian.active_space(), which dresses the
        # active one-electron term with the inactive mean field and folds the
        # constant inactive energy E_core into nuclear_repulsion.  (A bare
        # integral slice would drop both and report the active-only energy,
        # off by ~E_core Hartree -- see ACTIVE_SPACE.md.)
        if active_space is not None and method not in (
            "casci",
            "mrci",
            "casscf",
            "nevpt2",
            "caspt2",
        ):
            n_active, n_elec = active_space
            H = H.active_space(n_active, n_elec)

        if method == "transcorrelated_ci":
            if transcorrelated_options is None:
                transcorrelated_options = TranscorrelatedOptions()
            H = build_transcorrelated_hamiltonian(H, transcorrelated_options)
            return solve_selected_ci(H, selected_ci_options or SelectedCIOptions())

        if method == "selected_ci":
            return solve_selected_ci(H, selected_ci_options or SelectedCIOptions())

        if method == "cisd":
            from .solvers import cisd as _run_cisd

            ci_opts = cisd_options or CISDOptions()
            ci = _run_cisd(
                H.h1e,
                H.h2e,
                H.nelec,
                H.norb,
                nuclear_repulsion=H.nuclear_repulsion,
                ms2=H.ms2,
                max_excitation=ci_opts.max_excitation,
                nroots=ci_opts.nroots,
                max_det=ci_opts.max_det,
            )
            roots = ci.e_totals if len(ci.e_totals) > 1 else None
            return SolverResult(
                energy=ci.e_total,
                method=f"cisd(ndet={ci.n_det})",
                converged=ci.converged,
                n_iter=1,
                energy_trace=[ci.e_total],
                root_energies=roots,
                ci_coeffs=ci.ci_coeffs,
                ci_labels=ci.determinants,
            )

        if method == "cc3":
            from .cc import chemical_core_orbital_count
            from .solvers import cc3 as _run_cc3

            cc_opts = cc3_options or CC3Options()
            if cc_opts.n_frozen_core is None:
                cc_opts = replace(
                    cc_opts,
                    n_frozen_core=(
                        0
                        if active_space is not None
                        else chemical_core_orbital_count(molecule)
                    ),
                )
            cc_result = _run_cc3(H, cc_opts)
            cc_result.scf_trace = list(
                getattr(_iterative_cc_reference_result, "scf_trace", []) or []
            )
            return cc_result

        if method == "ccsdt":
            from .cc import chemical_core_orbital_count
            from .solvers import ccsdt as _run_ccsdt

            cc_opts = ccsdt_options or CCSDTOptions()
            if cc_opts.n_frozen_core is None:
                cc_opts = replace(
                    cc_opts,
                    n_frozen_core=(
                        0
                        if active_space is not None
                        else chemical_core_orbital_count(molecule)
                    ),
                )
            cc_result = _run_ccsdt(H, cc_opts)
            cc_result.scf_trace = list(
                getattr(_iterative_cc_reference_result, "scf_trace", []) or []
            )
            return cc_result

        if method == "dmrg":
            return solve_dmrg(H, dmrg_options or DMRGOptions())

        if method == "v2rdm":
            return solve_v2rdm(H, v2rdm_options or V2RDMOptions())

        if method in ("casci", "mrci", "casscf", "nevpt2", "caspt2"):
            from .solvers._casci import casci as _run_casci

            # Standard CAS partition: full MO integrals with an explicit frozen
            # core.  active_space=(n_active_orb, n_active_elec) selects the
            # lowest n_core orbitals as doubly-occupied core, n_active_orb
            # active orbitals, the rest virtual (matching PySCF mcscf.CASCI).
            # H here is the full, untruncated Hamiltonian.
            # H was built from the SCF reference, so H.nelec is already
            # n_electrons - ecp_total_ncore: the count the CAS partition must
            # divide. Taking molecule.n_electrons() here put the ECP core back
            # into the frozen-core block and asked the solver for orbitals the
            # valence basis does not have (#740).
            n_elec_total = int(H.nelec)
            if active_space is not None:
                n_active_orb, n_active_elec = active_space
                if (n_elec_total - n_active_elec) % 2 != 0:
                    raise ValueError(
                        "CAS active_space=(n_orb, n_elec) needs an even "
                        f"frozen-core electron count; got n_elec_total="
                        f"{n_elec_total}, n_active_elec={n_active_elec}."
                    )
                n_core = (n_elec_total - n_active_elec) // 2
            else:
                n_active_orb, n_active_elec, n_core = H.norb, n_elec_total, 0
            n_virt = H.norb - n_core - n_active_orb

            if method == "casscf":
                from .solvers._casscf import casscf as _run_casscf

                c_opts = casscf_options or CASSCFOptions()
                if (
                    getattr(c_opts, "pt2", None) is not None
                    and c_opts.ci_solver != "selected_ci"
                ):
                    raise ValueError(
                        "CASSCFOptions.pt2 is the Epstein-Nesbet PT2 "
                        "stage on a SELECTED wavefunction; it requires "
                        "ci_solver='selected_ci' (the exact-CI backend "
                        "has no external perturbers)"
                    )
                sc = _run_casscf(
                    H.h1e,
                    H.h2e,
                    n_active_elec=n_active_elec,
                    n_active_orb=n_active_orb,
                    n_core=n_core,
                    nuclear_repulsion=H.nuclear_repulsion,
                    ms2=molecule.multiplicity - 1,
                    nroots=c_opts.nroots,
                    weights=c_opts.weights,
                    orbital_step=c_opts.orbital_step,
                    active_orbitals=c_opts.active_orbitals,
                    spin_pure=c_opts.spin_pure,
                    ci_solver=c_opts.ci_solver,
                    selected_ci_options=c_opts.selected_ci_options,
                )
                label = f"casscf({n_active_elec}e,{n_active_orb}o)"
                if c_opts.ci_solver == "selected_ci":
                    label += "_selci"
                if sc.e_totals:
                    label += f"_sa{len(sc.e_totals)}"
                _sel_pt2 = None
                if getattr(c_opts, "pt2", None) is not None:
                    # SHCI perturbative stage on the converged selected
                    # wavefunction, in the converged orbital basis (the
                    # variational headline energy is unchanged; the PT2
                    # estimate surfaces on its own).
                    from .solvers._selected_ci import selected_ci_pt2

                    p = c_opts.pt2
                    _sel_pt2 = selected_ci_pt2(
                        sc.h1e_cas,
                        sc.h2e_cas,
                        n_active_elec,
                        n_active_orb,
                        n_core,
                        result=sc.cas,
                        eps2=p.eps2,
                        n_samples=p.n_samples,
                        sample_size=p.sample_size,
                        eps2_loose=p.eps2_loose,
                        seed=p.seed,
                    )
                # The converged CAS orbitals in the AO basis. Computed here
                # rather than inside the gradient branch below because they
                # are also what the natural orbitals are built from, and a
                # non-converged run still has orbitals worth looking at.
                C_conv = C @ sc.mo_rotation

                grad = None
                # The analytic CASSCF gradient runs whenever the solver
                # converged, not only when a gradient was asked for, and its
                # kernel has no ECP derivative (#740). Skip it on an ECP
                # reference so the *energy* is still returned; a caller who
                # actually wants the derivative still meets the explicit
                # refusal from the gradient routes themselves.
                _ecp_blocks_casscf_gradient = (
                    basis_sidecar_has_ecp_operator(molecule, basis)
                    or molecular_result_has_ecp_operator(
                        _iterative_cc_reference_result
                    )
                )
                if sc.converged and not _ecp_blocks_casscf_gradient:
                    # Analytic CASSCF gradient.  Single-state: exact (a
                    # variational energy needs no response term; #516).
                    # SA: uses SA-RDMs, not FD-validated.
                    from .gradient import compute_casscf_gradient
                    from .solvers._rdm import make_rdm12, make_rdm12_sa

                    _sa = (
                        c_opts.weights is not None and sc.cas.ci_coeffs_all is not None
                    )
                    if _sa:
                        rdm1_g, rdm2_g = make_rdm12_sa(
                            sc.cas.ci_coeffs_all,
                            sc.cas.determinants,
                            n_active_orb,
                            c_opts.weights,
                        )
                        _use_wz = False
                        _dets = None
                        _civec = None
                    else:
                        rdm1_g, rdm2_g = make_rdm12(
                            sc.cas.ci_coeffs, sc.cas.determinants, n_active_orb
                        )
                        _use_wz = c_opts.compute_wz
                        _dets = sc.cas.determinants
                        _civec = sc.cas.ci_coeffs
                    grad = compute_casscf_gradient(
                        molecule,
                        basis,
                        C_conv,
                        sc.h1e_cas,
                        sc.h2e_cas,
                        n_core=n_core,
                        n_active_orb=n_active_orb,
                        rdm1=rdm1_g,
                        rdm2=rdm2_g,
                        compute_wz=_use_wz,
                        determinants=_dets,
                        ci_coeffs=_civec,
                    )
                _cas_rdm1 = _active_rdm1(sc.cas)
                _nat = _cas_natural_orbitals(
                    C_conv,
                    _cas_rdm1,
                    n_core=n_core,
                    n_active_orb=n_active_orb,
                    n_elec_total=molecule.n_electrons(),
                )
                return SolverResult(
                    energy=sc.e_total,
                    method=label,
                    converged=sc.converged,
                    n_iter=sc.n_iter,
                    energy_trace=sc.energy_trace or [sc.e_total],
                    root_energies=list(sc.e_totals) or None,
                    root_s2=_root_s2_values(sc.cas, molecule.multiplicity - 1),
                    ci_coeffs=sc.cas.ci_coeffs,
                    ci_labels=sc.cas.determinants,
                    rdm1=_cas_rdm1,
                    selected_pt2=_sel_pt2,
                    gradient=grad,
                    natural_orbitals=_nat[0] if _nat else None,
                    natural_occupations=_nat[1] if _nat else None,
                )

            # CASSCF-referenced MR-PT2: supplying casscf_options switches the
            # nevpt2 / caspt2 reference from CASCI-on-HF-orbitals to a CASSCF
            # whose converged integrals + lowest-root CI vector seed the PT2 --
            # the standard CASSCF->CASPT2/NEVPT2 composition (mirrors the MRCI
            # pattern below).  Without casscf_options the historical
            # CASCI-on-HF reference is kept (matches the recorded OpenMolcas
            # `RASSCF CIonly` parity constants).
            h1_ref, h2_ref = H.h1e, H.h2e
            ref_suffix = ""
            ref_root_energies: list[float] = []
            _selected_ref = False
            _casscf_grad = None  # CASSCF gradient for MR-PT2 results
            _casscf_cm = None  # converged MO coefficients
            _casscf_rdm1 = None
            _casscf_rdm2 = None
            if method in ("nevpt2", "caspt2") and casscf_options is not None:
                from .solvers._casscf import casscf as _run_casscf

                c_opts = casscf_options
                if c_opts.active_orbitals is not None:
                    raise ValueError(
                        "CASSCF-referenced MR-PT2 does not support "
                        "active_orbitals window selection yet; use the "
                        "default lowest-core active window."
                    )
                if c_opts.ci_solver not in ("casci", "selected_ci"):
                    raise ValueError(
                        "casscf_options.ci_solver must be 'casci' or "
                        f"'selected_ci', got {c_opts.ci_solver!r}"
                    )
                _selected_ref = c_opts.ci_solver == "selected_ci"
                # The MS/XMS-CASPT2 model space re-solves its own multi-root
                # CASCI over the full active space, so it needs the exact CI
                # backend (a selected reference is single-state PT2 only).
                if (
                    _selected_ref
                    and method == "caspt2"
                    and caspt2_options is not None
                    and caspt2_options.multistate is not None
                ):
                    raise ValueError(
                        "selected-CI reference (ci_solver='selected_ci') is "
                        "not supported for multi-state CASPT2: the MS/XMS "
                        "model space re-solves a multi-root CASCI over the "
                        "full active space.  Use the exact CI backend for "
                        "multistate, or a single-state selected-reference "
                        "CASPT2."
                    )
                # spin_pure=None resolves inside casscf(): True for SA
                # references (nroots > 1) since 2026-06-11, covering the
                # multi-state CASPT2 composition (whose <S^2>-filtered
                # model space needs same-spin SA orbitals), and False for
                # single-state references.
                sc = _run_casscf(
                    H.h1e,
                    H.h2e,
                    n_active_elec=n_active_elec,
                    n_active_orb=n_active_orb,
                    n_core=n_core,
                    nuclear_repulsion=H.nuclear_repulsion,
                    ms2=molecule.multiplicity - 1,
                    nroots=c_opts.nroots,
                    weights=c_opts.weights,
                    orbital_step=c_opts.orbital_step,
                    spin_pure=c_opts.spin_pure,
                    ci_solver=c_opts.ci_solver,
                    selected_ci_options=c_opts.selected_ci_options,
                )
                if not sc.converged:
                    raise RuntimeError(
                        f"CASSCF reference for method={method!r} did not "
                        f"converge (|g| = {sc.grad_norm:.2e} after "
                        f"{sc.n_iter} macro-iterations)"
                    )
                cas = sc.cas
                h1_ref, h2_ref = sc.h1e_cas, sc.h2e_cas
                ref_suffix = "_casscf"
                ref_root_energies = list(sc.e_totals)
                _casscf_cm = C @ sc.mo_rotation
                if _mrpt_gradient:
                    # Compute CASSCF gradient for inclusion in MR-PT2 result.
                    from .gradient import compute_casscf_gradient
                    from .solvers._rdm import make_rdm12

                    rdm1_g, rdm2_g = make_rdm12(
                        sc.cas.ci_coeffs, sc.cas.determinants, n_active_orb
                    )
                    _casscf_rdm1 = rdm1_g
                    _casscf_rdm2 = rdm2_g
                    _casscf_grad = compute_casscf_gradient(
                        molecule,
                        basis,
                        _casscf_cm,
                        sc.h1e_cas,
                        sc.h2e_cas,
                        n_core=n_core,
                        n_active_orb=n_active_orb,
                        rdm1=rdm1_g,
                        rdm2=rdm2_g,
                        compute_wz=(
                            casscf_options.compute_wz if casscf_options else False
                        ),
                        determinants=sc.cas.determinants,
                        ci_coeffs=sc.cas.ci_coeffs,
                    )
            else:
                # casci_options is method="casci"-only; the CASCI built here
                # as the *reference* for nevpt2 / caspt2 / mrci keeps the
                # solver defaults (single root -- the PT2 / MRCI engines seed
                # from root 0; the MS-CASPT2 model space re-solves its own
                # multi-root CASCI inside ms_caspt2()).
                ci_opts = (
                    (casci_options or CASCIOptions())
                    if method == "casci"
                    else CASCIOptions()
                )
                cas = _run_casci(
                    H.h1e,
                    H.h2e,
                    n_active_elec=n_active_elec,
                    n_active_orb=n_active_orb,
                    n_core=n_core,
                    nuclear_repulsion=H.nuclear_repulsion,
                    ms2=molecule.multiplicity - 1,
                    nroots=ci_opts.nroots,
                    max_det=ci_opts.max_det,
                )

            if method == "casci":
                energy, label = cas.e_total, f"casci({n_active_elec}e,{n_active_orb}o)"
                if cas.nroots > 1:
                    # Multi-root CASCI: per-root energies + <S^2> reach
                    # SolverResult.root_energies / .root_s2 via the shared
                    # return below (headline ``energy`` stays root 0).
                    ref_root_energies = list(cas.e_totals)
            elif method == "mrci":
                from .solvers._mrci import mrci as _run_mrci

                # If casscf_options are provided, run CASSCF first for
                # an orbital-optimized reference.
                if casscf_options is not None:
                    from .solvers._casscf import casscf as _run_casscf

                    c_opts = casscf_options or CASSCFOptions()
                    if c_opts.ci_solver != "casci":
                        raise ValueError(
                            "CASSCF-referenced MRCI requires the exact CI "
                            "backend; ci_solver='selected_ci' is "
                            "method='casscf' only"
                        )
                    sc = _run_casscf(
                        H.h1e,
                        H.h2e,
                        n_active_elec=n_active_elec,
                        n_active_orb=n_active_orb,
                        n_core=n_core,
                        nuclear_repulsion=H.nuclear_repulsion,
                        ms2=molecule.multiplicity - 1,
                        nroots=c_opts.nroots,
                        weights=c_opts.weights,
                        orbital_step=c_opts.orbital_step,
                        active_orbitals=c_opts.active_orbitals,
                        spin_pure=c_opts.spin_pure,
                    )
                    h1, h2 = sc.h1e_cas, sc.h2e_cas
                else:
                    h1, h2 = H.h1e, H.h2e

                mc = _run_mrci(
                    h1,
                    h2,
                    n_active_elec=n_active_elec,
                    n_active_orb=n_active_orb,
                    n_core=n_core,
                    nuclear_repulsion=H.nuclear_repulsion,
                    ms2=molecule.multiplicity - 1,
                )
                label = f"mrci({n_active_elec}e,{n_active_orb}o)"
                if casscf_options is not None:
                    label += "_casscf"
                energy = mc.e_total
            elif method == "nevpt2":
                from .solvers._mrpt import nevpt2 as _run_nevpt2

                pt = _run_nevpt2(
                    cas,
                    h1_ref,
                    h2_ref,
                    n_core=n_core,
                    n_virt=n_virt,
                    selected_reference=_selected_ref,
                )
                energy = pt.e_total
                label = f"nevpt2({n_active_elec}e,{n_active_orb}o){ref_suffix}"
            else:  # caspt2 -- internally-contracted (default), un-gated
                from .solvers._mrpt import caspt2 as _run_caspt2

                cp = caspt2_options or CASPT2Options()
                if cp.nac_pair is not None and cp.multistate is None:
                    raise ValueError(
                        "caspt2_options.nac_pair requires "
                        "multistate='ms' or 'xms'"
                    )
                if cp.multistate is not None:
                    # MS/XMS-CASPT2 (Finley 1998 / Granovsky 2011 +
                    # Shiozaki 2011): multi-state effective Hamiltonian over
                    # the lowest nroots spin-pure CASCI roots of the
                    # reference basis (HF orbitals, or the converged
                    # SA-CASSCF basis when casscf_options are supplied).
                    from .solvers._ms_caspt2 import ms_caspt2 as _run_ms

                    if cp.multistate not in ("ms", "xms"):
                        raise ValueError(
                            "caspt2_options.multistate must be None, 'ms' "
                            f"or 'xms', got {cp.multistate!r}"
                        )
                    if cp.variant != "ic":
                        raise ValueError(
                            "multi-state CASPT2 requires variant='ic' "
                            f"(got {cp.variant!r})"
                        )
                    if cp.ipea != 0.0:
                        raise ValueError(
                            "multi-state CASPT2 supports ipea=0 only; the "
                            "canonical eight-class IPEA contraction is "
                            "currently implemented for single-state "
                            "IC-CASPT2 only"
                        )
                    ms_nroots = cp.nroots or (
                        casscf_options.nroots if casscf_options else 0
                    )
                    if ms_nroots < 2:
                        raise ValueError(
                            "multi-state CASPT2 needs >= 2 model states: "
                            "set caspt2_options.nroots (CASCI reference) "
                            "or casscf_options.nroots (SA-CASSCF reference)"
                        )
                    msr = _run_ms(
                        h1_ref,
                        h2_ref,
                        n_core,
                        n_virt,
                        nroots=ms_nroots,
                        nuclear_repulsion=H.nuclear_repulsion,
                        mode=cp.multistate,
                        n_frozen=cp.n_frozen,
                        imaginary=cp.imaginary,
                        ms2=molecule.multiplicity - 1,
                        n_act_elec=n_active_elec,
                    )
                    label = (
                        f"caspt2({n_active_elec}e,{n_active_orb}o)"
                        f"_{cp.multistate}{ms_nroots}{ref_suffix}"
                    )
                    # Resolve the SA weights the same way casscf() does:
                    # nroots > 1 with weights=None means equal averaging.
                    _sa_w = None
                    if casscf_options is not None:
                        _sa_w = casscf_options.weights
                        _c_nr = casscf_options.nroots
                        if _sa_w is None and _c_nr and _c_nr > 1:
                            _sa_w = [1.0 / _c_nr] * _c_nr

                    # Relaxed MS/XMS-CASPT2 gradient (SA-CASSCF Lagrangian
                    # z-vector + state Lagrangian; see _ms_caspt2_grad).
                    # Opt-in via compute_corr_grad=True, matching the
                    # single-state convention: the semi-numerical response
                    # costs O(n_pairs) extra multi-state solves, which
                    # energy-only runs must not pay.
                    _ms_grad = None
                    _want_ms_grad = getattr(cp, "compute_corr_grad", False)
                    if (
                        _want_ms_grad
                        and _casscf_grad is not None
                        and _casscf_cm is not None
                    ):
                        try:
                            from .gradient._ms_caspt2_grad import (
                                compute_ms_caspt2_gradient as _ms_grad_fn,
                            )

                            _ms_grad = np.asarray(
                                _ms_grad_fn(
                                    molecule,
                                    basis,
                                    _casscf_cm,
                                    h1_ref,
                                    h2_ref,
                                    n_core,
                                    n_active_orb,
                                    n_active_elec=n_active_elec,
                                    sa_weights=_sa_w,
                                    nroots=ms_nroots,
                                    mode=cp.multistate,
                                    target_root=0,
                                    ms2=molecule.multiplicity - 1,
                                )
                            )
                        except Exception as _ms_grad_err:
                            import warnings

                            warnings.warn(
                                "MS-CASPT2 gradient computation failed "
                                f"({_ms_grad_err}); the result carries "
                                "no gradient.",
                                UserWarning,
                            )
                            _ms_grad = None

                    _ms_nac = None
                    if cp.nac_pair is not None:
                        if len(cp.nac_pair) != 2:
                            raise ValueError(
                                "caspt2_options.nac_pair must contain two "
                                f"physical roots, got {cp.nac_pair!r}"
                            )
                        if (
                            casscf_options is None
                            or _casscf_cm is None
                            or _sa_w is None
                        ):
                            raise ValueError(
                                "MS/XMS-CASPT2 nonadiabatic coupling requires "
                                "an exact state-averaged CASSCF reference: pass "
                                "casscf_options with nroots >= 2"
                            )
                        if casscf_options.nroots != ms_nroots:
                            raise ValueError(
                                "MS/XMS-CASPT2 nonadiabatic coupling requires "
                                "the SA-CASSCF and multistate model spaces to "
                                f"have the same size ({casscf_options.nroots} "
                                f"!= {ms_nroots})"
                            )
                        if casscf_options.ci_solver != "casci":
                            raise ValueError(
                                "MS/XMS-CASPT2 nonadiabatic coupling requires "
                                "casscf_options.ci_solver='casci'"
                            )
                        if molecule.multiplicity != 1:
                            raise NotImplementedError(
                                "MS/XMS-CASPT2 nonadiabatic coupling is "
                                "currently verified for closed-shell singlets only"
                            )
                        if cp.engine != "auto":
                            raise NotImplementedError(
                                "MS/XMS-CASPT2 nonadiabatic coupling requires "
                                "the explicit engine selected by engine='auto'"
                            )
                        if cp.imaginary != 0.0 or cp.n_frozen != 0:
                            raise NotImplementedError(
                                "MS/XMS-CASPT2 nonadiabatic coupling is "
                                "currently verified only for unshifted calculations "
                                "with n_frozen=0"
                            )
                        if max(_sa_w) - min(_sa_w) > 1e-12:
                            raise NotImplementedError(
                                "MS/XMS-CASPT2 nonadiabatic coupling currently "
                                "requires equal SA-CASSCF weights"
                            )
                        from .gradient import compute_ms_caspt2_nac

                        _ms_nac = np.asarray(
                            compute_ms_caspt2_nac(
                                molecule,
                                basis,
                                _casscf_cm,
                                h1_ref,
                                h2_ref,
                                n_core,
                                n_active_orb,
                                n_active_elec=n_active_elec,
                                sa_weights=list(_sa_w),
                                nroots=ms_nroots,
                                state_pair=tuple(cp.nac_pair),
                                mode=cp.multistate,
                                ms2=molecule.multiplicity - 1,
                                fd_step=cp.nac_fd_step,
                                gap_tolerance=cp.nac_gap_tolerance,
                                casscf_orbital_step=casscf_options.orbital_step,
                                casscf_spin_pure=casscf_options.spin_pure,
                            )
                        )
                    return SolverResult(
                        energy=msr.e_total,
                        method=label,
                        converged=True,
                        n_iter=1,
                        energy_trace=[msr.e_total],
                        root_energies=list(msr.energies),
                        ci_coeffs=cas.ci_coeffs,
                        ci_labels=cas.determinants,
                        rdm1=_active_rdm1(cas),
                        gradient=_ms_grad,
                        nonadiabatic_pair=(
                            tuple(cp.nac_pair) if cp.nac_pair is not None else None
                        ),
                        nonadiabatic_coupling=_ms_nac,
                        multistate=dict(
                            mode=msr.mode,
                            nroots=msr.nroots,
                            heff=msr.heff,
                            heff_asym=msr.heff_asym,
                            mixing=msr.mixing,
                            ss_energies=list(msr.ss_energies),
                            ref_energies=list(msr.ref_energies),
                            e2_corr=list(msr.e2_corr),
                            xms_rotation=msr.xms_rotation,
                            s2_values=list(msr.s2_values),
                        ),
                    )
                pt = _run_caspt2(
                    cas,
                    h1_ref,
                    h2_ref,
                    n_core=n_core,
                    n_virt=n_virt,
                    variant=cp.variant,
                    ipea=cp.ipea,
                    imaginary=cp.imaginary,
                    n_frozen=cp.n_frozen,
                    selected_reference=_selected_ref,
                    engine=cp.engine,
                )
                energy = pt.e_total
                _eng_suffix = "_cases" if cp.engine == "cases" else ""
                label = (
                    f"caspt2({n_active_elec}e,{n_active_orb}o){ref_suffix}{_eng_suffix}"
                )

            # Optionally replace the reference gradient with the derivative
            # of the full correlated energy.  Single-state IC-CASPT2 uses its
            # analytic Lagrangian inside the exact supported envelope;
            # SC-NEVPT2 retains the relaxed full-energy FD route.
            if _casscf_grad is not None and _casscf_cm is not None:
                _use_corr = False
                if method == "caspt2" and caspt2_options is not None:
                    _use_corr = getattr(caspt2_options, "compute_corr_grad", False)
                elif method == "nevpt2" and nevpt2_options is not None:
                    _use_corr = getattr(nevpt2_options, "compute_corr_grad", False)
                if not _use_corr:
                    import warnings

                    _corr_hint = (
                        "the analytic IC-CASPT2 gradient in its supported "
                        "single-state envelope"
                        if method == "caspt2"
                        else "the full relaxed correlation gradient"
                    )
                    warnings.warn(
                        f"{method.upper()} gradient: compute_corr_grad=False; "
                        f"the returned gradient is the CASSCF gradient only. "
                        f"Set {method}_options.compute_corr_grad=True for "
                        f"{_corr_hint}.",
                        UserWarning,
                    )
                if _use_corr and method == "caspt2":
                    from .solvers._mrpt import _ic_caspt2_uses_direct_active_ci

                    cp = caspt2_options
                    unsupported: list[str] = []
                    if cp is None or cp.variant != "ic":
                        unsupported.append("variant must be 'ic'")
                    if cp is not None and cp.multistate is not None:
                        unsupported.append("single-state only")
                    if cp is not None and cp.ipea != 0.0:
                        unsupported.append("ipea must be 0")
                    if cp is not None and cp.imaginary != 0.0:
                        unsupported.append("imaginary shift must be 0")
                    if cp is not None and cp.engine != "auto":
                        unsupported.append("engine must be 'auto'")
                    if _selected_ref:
                        unsupported.append("exact-CI CASSCF reference required")
                    if (c_opts.nroots or 1) != 1:
                        unsupported.append("state-specific CASSCF required")
                    if molecule.multiplicity != 1:
                        unsupported.append("closed-shell singlet required")
                    if cp is not None and _ic_caspt2_uses_direct_active_ci(
                        h1_ref.shape[0],
                        n_core,
                        n_active_orb,
                        n_frozen=cp.n_frozen,
                        ipea=cp.ipea,
                        imaginary=cp.imaginary,
                    ):
                        unsupported.append("explicit small-space engine required")
                    if unsupported:
                        raise NotImplementedError(
                            "Analytic IC-CASPT2 gradients currently support "
                            "single-state, unshifted, explicit-engine "
                            "closed-shell CASSCF references; unsupported: "
                            + "; ".join(unsupported)
                            + ". Use a full-energy finite-difference gradient "
                            "for this CASPT2 variant."
                        )
                    if _casscf_rdm1 is None or _casscf_rdm2 is None:
                        raise RuntimeError(
                            "IC-CASPT2 analytic gradient requires the "
                            "converged CASSCF reference RDMs"
                        )
                    from .gradient._caspt2 import (
                        compute_ic_caspt2_analytic_gradient,
                    )

                    _casscf_grad = compute_ic_caspt2_analytic_gradient(
                        molecule,
                        basis,
                        _casscf_cm,
                        h1_ref,
                        h2_ref,
                        n_core,
                        n_active_orb,
                        n_active_elec,
                        rdm1=_casscf_rdm1,
                        rdm2=_casscf_rdm2,
                        ci_coeffs=sc.cas.ci_coeffs,
                        determinants=sc.cas.determinants,
                        n_frozen=cp.n_frozen,
                    )
                if _use_corr and method == "nevpt2":
                    _grad_fd = np.zeros_like(_casscf_grad)
                    _atoms_list = list(molecule.atoms)
                    _h = 0.001
                    for _a in range(len(_atoms_list)):
                        for _c in range(3):
                            _xyz_p = np.array(
                                [at.xyz for at in _atoms_list], dtype=float
                            )
                            _xyz_m = _xyz_p.copy()
                            _xyz_p[_a, _c] += _h
                            _xyz_m[_a, _c] -= _h
                            _mp = Molecule(
                                [
                                    Atom(int(at.Z), list(xyz))
                                    for at, xyz in zip(_atoms_list, _xyz_p)
                                ]
                            )
                            _mm = Molecule(
                                [
                                    Atom(int(at.Z), list(xyz))
                                    for at, xyz in zip(_atoms_list, _xyz_m)
                                ]
                            )
                            _bp = BasisSet(_mp, basis.name)
                            _bm = BasisSet(_mm, basis.name)

                            _cp_energy = (
                                replace(caspt2_options, compute_corr_grad=False)
                                if caspt2_options is not None
                                else None
                            )
                            _np_energy = (
                                replace(nevpt2_options, compute_corr_grad=False)
                                if nevpt2_options is not None
                                else None
                            )
                            _ep = _run_single_point(
                                method,
                                _mp,
                                _bp,
                                functional=functional,
                                rhf_options=rhf_options,
                                uhf_options=uhf_options,
                                rks_options=rks_options,
                                uks_options=uks_options,
                                rohf_options=rohf_options,
                                roks_options=roks_options,
                                cisd_options=cisd_options,
                                selected_ci_options=selected_ci_options,
                                dmrg_options=dmrg_options,
                                v2rdm_options=v2rdm_options,
                                transcorrelated_options=transcorrelated_options,
                                casci_options=casci_options,
                                caspt2_options=_cp_energy,
                                nevpt2_options=_np_energy,
                                casscf_options=casscf_options,
                                active_space=active_space,
                                cas_reference=cas_reference,
                                mlip_options=mlip_options,
                                ccm_options=ccm_options,
                                solvent=solvent,
                                dft_plus_u=dft_plus_u,
                                read_from=read_from,
                                fragments=fragments,
                                nddo=nddo,
                                _mrpt_gradient=False,
                            ).energy
                            _em = _run_single_point(
                                method,
                                _mm,
                                _bm,
                                functional=functional,
                                rhf_options=rhf_options,
                                uhf_options=uhf_options,
                                rks_options=rks_options,
                                uks_options=uks_options,
                                rohf_options=rohf_options,
                                roks_options=roks_options,
                                cisd_options=cisd_options,
                                selected_ci_options=selected_ci_options,
                                dmrg_options=dmrg_options,
                                v2rdm_options=v2rdm_options,
                                transcorrelated_options=transcorrelated_options,
                                casci_options=casci_options,
                                caspt2_options=_cp_energy,
                                nevpt2_options=_np_energy,
                                casscf_options=casscf_options,
                                active_space=active_space,
                                cas_reference=cas_reference,
                                mlip_options=mlip_options,
                                ccm_options=ccm_options,
                                solvent=solvent,
                                dft_plus_u=dft_plus_u,
                                read_from=read_from,
                                fragments=fragments,
                                nddo=nddo,
                                _mrpt_gradient=False,
                            ).energy
                            _grad_fd[_a, _c] = (_ep - _em) / (2.0 * _h)
                    _casscf_grad = _grad_fd

            return SolverResult(
                energy=energy,
                method=label,
                # Direct CI + non-iterative PT2: a returned result means the
                # eigensolve succeeded.
                converged=True,
                n_iter=1,
                energy_trace=[energy],
                root_energies=ref_root_energies or None,
                root_s2=(
                    _root_s2_values(cas, molecule.multiplicity - 1)
                    if ref_root_energies
                    else None
                ),
                # Reference-wavefunction diagnostics (the CASCI / CASSCF
                # reference for PT2 methods): leading configurations +
                # active natural occupations in the .out solver block.
                ci_coeffs=cas.ci_coeffs,
                ci_labels=cas.determinants,
                rdm1=_active_rdm1(cas),
                gradient=_casscf_grad,
            )

    if method == "fci":
        from scipy.linalg import eigh

        from .solvers import (
            build_hamiltonian_matrix_unrestricted,
            generate_determinants,
        )

        hf_method = "uhf" if molecule.multiplicity > 1 else "rhf"
        # See the CAS/determinant site above: a manual ECP reaches the
        # reference only through the caller's options (#740).
        C, _fci_reference = get_hf_orbital_provider(
            molecule,
            basis,
            method=hf_method,
            _with_result=True,
            scf_options=(uhf_options if hf_method == "uhf" else rhf_options),
        )
        ham = build_hamiltonian_mo(molecule, basis, C, reference=_fci_reference)

        if active_space is not None:
            # Standard CAS partition with a properly dressed frozen core
            # (effective one-electron term + E_core folded into
            # nuclear_repulsion); see Hamiltonian.active_space.
            n_active, n_elec = active_space
            ham = ham.active_space(n_active, n_elec)

        norb = ham.norb
        nelec = ham.nelec
        nalpha = (nelec + ham.ms2) // 2
        nbeta = (nelec - ham.ms2) // 2

        all_dets = generate_determinants(norb, nalpha, nbeta)
        H = build_hamiltonian_matrix_unrestricted(all_dets, ham.h1e, ham.h2e)
        evals, evecs = eigh(H)
        e_fci = evals[0] + ham.nuclear_repulsion

        return SolverResult(
            energy=e_fci,
            method=f"fci(ndet={len(all_dets)})",
            converged=True,
            n_iter=1,
            energy_trace=[e_fci],
            ci_coeffs=evecs[:, 0],
            ci_labels=all_dets,
        )

    # ── Semiempirical methods ──
    if method in MOLECULAR_SEMIEMPIRICAL_METHODS:
        if solvent is not None and method == "msindo":
            return _run_semiempirical(method, molecule, nddo=nddo, solvent=solvent)
        return _run_semiempirical(method, molecule, nddo=nddo)

    # ── Periodic MSINDO SECCM ──
    if method == "ccm":
        return _run_ccm(molecule, ccm_options)

    # ── Machine-learning interatomic potentials (MACE, ...) ──
    if method in _MLIP_METHODS:
        return _run_mlip(method, molecule, mlip_options)

    raise ValueError(
        f"Unknown method {method!r}; use 'rhf', 'uhf', 'rohf', 'rks', 'uks', "
        f"'roks', 'selected_ci', 'dmrg', 'v2rdm', 'transcorrelated_ci', 'casci', "
        f"'mrci', 'casscf', 'nevpt2', 'caspt2', 'fci', "
        f"'ci' with citype=, 'cisd', 'ccsd', 'ccsd(t)', 'bccd', 'bccd(t)', "
        f"'cc2', 'ccd', 'lccd', 'lccsd', 'cepa(0)'..'cepa(3)', "
        f"'qcisd', 'qcisd(t)', "
        f"'dftb0', 'scc_dftb', 'pm6', 'gfn2_xtb', "
        f"'om1', 'om2', 'om3', 'seccm', 'mace', or 'auto'"
    )


def _citation_manifest_rows(refs: Any) -> list[dict[str, Any]]:
    """Flatten an :class:`AssembledCitations` into ``.system``
    ``[citations]`` rows. Thin alias for
    :func:`vibeqc.output.citations.citation_manifest_rows`, kept so
    existing importers of the private runner helper stay working; the
    shared implementation also feeds ``run_periodic_job``."""
    from .output.citations import citation_manifest_rows

    return citation_manifest_rows(refs)


def _run_mlip(method: str, molecule: Molecule, mlip_options=None):
    """Run a machine-learning interatomic potential and return a
    result-like object (duck-typing the SCF-result interface).

    vibe-qc drives the external pre-trained model's forward pass; the
    returned ``.energy`` is the model's reference-shifted DFT-surface
    energy (Hartree), NOT a total electronic energy (``CLAUDE.md`` Sec.10 /
    energy-scale caveat -- do not mix with HF/DFT totals).

    ``mlip_options`` (:class:`vibeqc.mlip.MLIPOptions`) selects the model,
    device, and dtype, and carries the ASL acknowledgment. Selecting an
    ASL (academic, non-commercial) model without acknowledgment raises
    :class:`PermissionError` (see :mod:`vibeqc.mlip.mace`).
    """
    if method == "mace":
        from vibeqc.mlip.mace import MACEModel

        model = MACEModel(molecule, mlip_options)
        return _MLIPResult(
            model.energy(),
            model.gradient(),
            method,
            converged=True,
            n_iter=1,
            model_info=model.info,
            model_options=model.options,
        )
    raise ValueError(f"Unhandled MLIP method: {method!r}")


class _MLIPResult:
    """Machine-learning-interatomic-potential result wrapper -- duck-types
    the same interface as :class:`SemiempiricalResult` (energy, gradient(), method,
    converged, n_iter, scf_trace). ``energy`` is the model's
    reference-shifted DFT-surface energy (Hartree), not a total electronic
    energy."""

    def __init__(
        self,
        energy,
        gradient=None,
        method_name="",
        converged=True,
        n_iter=1,
        model_info=None,
        model_options=None,
    ):
        self.energy = energy
        self._gradient = gradient
        self.method = method_name
        self.converged = converged
        self.n_iter = n_iter
        self.scf_trace = []
        # MaceModelInfo (or None) -- provenance for the MLIP model actually
        # run; used by run_job for the per-model citation + .out/.system.
        self.model_info = model_info
        # Runtime choices are separate from the immutable model-registry row:
        # two jobs can use the same weights on different devices/dtypes.
        self.model_options = model_options
        if model_info is not None:
            self.loader = model_info.loader
            self.loader_arg = model_info.loader_arg
        if model_options is not None:
            self.device = model_options.device
            self.dtype = model_options.dtype

    def gradient(self):
        return self._gradient


# The DLPNO truncation knobs, ordered as the publications tabulate them --
# TCutPairs, TCutPNO, TCutMKN (Liakos, Sparta, Kesharwani, Martin & Neese,
# J. Chem. Theory Comput. 11, 1525 (2015), doi:10.1021/ct501129s, Table 1) --
# followed by vibe-qc's own additional cutoffs.
_DLPNO_THRESHOLD_ORDER = (
    "tcut_pairs",
    "tcut_pno",
    "tcut_mkn",
    "tcut_pairs_weak",
    "tcut_pno_weak",
    "tcut_pno_singles",
    "tcut_tno",
)


def _format_dlpno_thresholds(
    options,
    *,
    provenance=None,
    label_width: int = 24,
) -> str:
    """One ``.out`` row naming the truncation thresholds a DLPNO run resolved.

    Issue #417: none of ``tcut_pno`` / ``tcut_pairs`` / ``tcut_mkn`` reached
    any artifact -- not the ``.out``, not the ``.system`` -- so the truncation
    level a run actually used could not be checked against the run that
    produced it.  A named-preset claim in a paper ("TightPNO") was therefore
    unprovable from the run's own output.  The DLPNO blocks already reported
    the *consequences* of the thresholds (pairs kept and screened, average
    PNOs per pair, frozen-core count); this reports the thresholds themselves.

    Only the knobs the route's own option class carries are printed.  The six
    DLPNO option classes expose different subsets of them (issue #448), so a
    fixed list would either invent thresholds a route does not have or hide
    ones it does.  ``0.0`` is printed rather than omitted -- it is a
    meaningful setting (no truncation / full domains), not an absent one.
    """
    if options is None:
        return ""
    lines = []
    if provenance is not None:
        subset = bool(provenance.unsupported or provenance.inactive)
        preset = str(provenance.preset)
        convention = preset + (" (supported subset)" if subset else "")
        lines.append(
            f"  {'Threshold convention':<{label_width}s} = {convention}\n"
        )
    # A route may store a published cutoff that it does not actually consume
    # (UMP2's canonical-occupied mode is the current example). Report only
    # applied published/derived coordinates, plus real solver-specific
    # controls outside the named triple.
    settings = (
        _dlpno_threshold_settings(options, provenance)
        if provenance is not None
        else {
            name: float(getattr(options, name))
            for name in _DLPNO_THRESHOLD_ORDER
            if getattr(options, name, None) is not None
        }
    )
    parts = []
    for name in _DLPNO_THRESHOLD_ORDER:
        if name not in settings:
            continue
        # ``repr(float)`` is Python's shortest round-trip representation.
        # Recipe provenance must retain 3.33e-7 (and arbitrary custom
        # cutoffs) rather than silently rounding it to one decimal place.
        parts.append(f"{name} {settings[name]!r}")
    if parts:
        lines.append(
            f"  {'Truncation thresholds':<{label_width}s} = "
            + "  ".join(parts)
            + "\n"
        )
    if provenance is not None and provenance.unsupported:
        lines.append(
            f"  {'Unsupported cutoffs':<{label_width}s} = "
            + ", ".join(provenance.unsupported)
            + "\n"
        )
    if provenance is not None and provenance.inactive:
        lines.append(
            f"  {'Inactive cutoffs':<{label_width}s} = "
            + ", ".join(provenance.inactive)
            + "\n"
        )
    return "".join(lines)


def _format_dlpno_residual_domain(options, *, label_width: int = 24) -> str:
    """Disclose the local CC residual-contraction domain when applicable."""

    domain = str(getattr(options, "residual_domain", "") or "")
    if not domain:
        return ""
    return f"  {'Residual domain':<{label_width}s} = {domain}\n"


def _format_dlpno_pno_norm(options, *, label_width: int = 24) -> str:
    """Disclose the pair-density convention the PNO cut-off was applied to.

    ``tcut_pno`` is an occupation-number threshold, so it only means what the
    published presets intend under the convention they were calibrated
    against. Empty for routes that do not expose the choice. See
    :mod:`vibeqc.dlpno.pno_density` and issue #701.
    """

    norm = str(getattr(options, "pno_norm", "") or "")
    if not norm:
        return ""
    return f"  {'PNO density norm':<{label_width}s} = {norm}\n"


def _format_dlpno_pno_correction(options, *, label_width: int = 24) -> str:
    """Disclose whether the semicanonical MP2 PNO-truncation correction ran.

    It moves ``e_corr``: the correction for the iterated pairs' PNO
    truncation is folded into the CCSD correlation energy, so two runs that
    differ only in this setting print different energies. Recording it with
    the other recipe rows keeps that visible (the #417 lesson). Empty for
    routes that do not expose the choice.
    """

    flag = getattr(options, "pno_correction", None)
    if flag is None:
        return ""
    return f"  {'PNO MP2 correction':<{label_width}s} = {'on' if flag else 'off'}\n"


def _dlpno_threshold_settings(options, provenance) -> dict[str, float]:
    """Return every cutoff disclosed for the route in artifact order.

    ``provenance.applied`` owns the published triple and the derived MP2 weak
    cutoffs. Local CC routes additionally carry solver-specific singles/TNO
    cutoffs outside that published request. Keeping those settings beside the
    named provenance makes the human, JSONL, and TOML artifacts mutually
    reproducible without pretending they are Liakos preset coordinates.
    """

    applied = dict(provenance.applied)
    settings: dict[str, float] = {}
    for name in _DLPNO_THRESHOLD_ORDER:
        if name in applied:
            settings[name] = float(applied[name])
        elif name in ("tcut_pno_singles", "tcut_tno"):
            if name == "tcut_tno" and (
                not bool(getattr(options, "compute_triples", False))
                or str(getattr(options, "triples_mode", "")).lower()
                == "exact"
            ):
                continue
            value = getattr(options, name, None)
            if value is not None:
                settings[name] = float(value)
    return settings


def _dlpno_triples_provenance(options, result=None) -> tuple[str, str]:
    """Return the executed triples mode and an artifact-safe description."""

    if not bool(getattr(options, "compute_triples", False)):
        return "none", "disabled"
    if (
        result is not None
        and hasattr(result, "triples_executed")
        and not bool(result.triples_executed)
    ):
        return "none", "not executed for this result"
    raw_mode = getattr(options, "triples_mode", None)
    mode = str(raw_mode).strip().lower() if raw_mode is not None else "pilot"
    algorithms = {
        "local": "DLPNO-(T0), diagonal local occupied-Fock",
        "t0": "DLPNO-(T0), diagonal local occupied-Fock",
        "t1": "rotated-occupied (T1) compatibility correction",
        "t1-iterative": (
            "experimental dense spin-orbital generalization of Guo Eq. (2), "
            "iterative local-basis DLPNO-(T1)"
        ),
        "exact": "canonical (T) contraction on local amplitudes",
        "pilot": "dense correctness-pilot (T)",
    }
    return mode, algorithms.get(mode, f"unrecognized mode {mode!r}")


def _format_dlpno_triples(options, result=None, *, label_width: int = 24) -> str:
    """Human-readable triples provenance for a DLPNO-CCSD(T) block."""

    mode, algorithm = _dlpno_triples_provenance(options, result)
    if mode == "none" and algorithm == "disabled":
        return ""
    return (
        f"  {'Triples algorithm':<{label_width}s} = {algorithm} "
        f"(triples_mode={mode!r})\n"
    )


def _dlpno_manifest_fields(options, provenance) -> dict[str, object]:
    """Flat, TOML-safe *effective* threshold provenance for ``[run]``.

    The structured event retains both the complete named request and the
    applied subset.  The legacy flat manifest coordinates, however, describe
    the route that actually ran: unsupported and inactive coordinates are
    empty and are named separately instead of being assigned a false value.
    """
    settings = _dlpno_threshold_settings(options, provenance)
    triples_mode, triples_algorithm = _dlpno_triples_provenance(options)

    def _applied(name: str) -> object:
        return float(settings[name]) if name in settings else ""

    return {
        "dlpno_threshold_preset": str(provenance.preset),
        "dlpno_tcut_pairs": _applied("tcut_pairs"),
        "dlpno_tcut_pno": _applied("tcut_pno"),
        "dlpno_tcut_mkn": _applied("tcut_mkn"),
        "dlpno_tcut_pairs_weak": _applied("tcut_pairs_weak"),
        "dlpno_tcut_pno_weak": _applied("tcut_pno_weak"),
        "dlpno_tcut_pno_singles": _applied("tcut_pno_singles"),
        "dlpno_tcut_tno": _applied("tcut_tno"),
        "dlpno_threshold_unsupported": ",".join(provenance.unsupported),
        "dlpno_threshold_inactive": ",".join(provenance.inactive),
        "dlpno_residual_domain": str(
            getattr(options, "residual_domain", "") or ""
        ),
        "dlpno_pno_norm": str(getattr(options, "pno_norm", "") or ""),
        "dlpno_pno_correction": (
            ""
            if getattr(options, "pno_correction", None) is None
            else ("on" if getattr(options, "pno_correction") else "off")
        ),
        "dlpno_triples_mode": triples_mode,
        "dlpno_triples_algorithm": triples_algorithm,
    }


_FROZEN_CORE_ROUTE_METHODS = frozenset(
    {
        "mp2",
        "scs-mp2",
        "sos-mp2",
        "ccsd",
        "ccsd(t)",
        "bccd",
        "bccd(t)",
        "qcisd",
        "qcisd(t)",
        "cc2",
        "ccd",
        "lccd",
        "lccsd",
        "cepa(0)",
        "cepa(1)",
        "cepa(2)",
        "cepa(3)",
        "dlpno-mp2",
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
    }
)

# Result-taking correlation kernels retain an ECP one-electron operator in the
# reference orbitals and orbital energies, so a genuine zero-core model
# potential remains sound. Their open defect is narrower: they still derive
# occupied-space sizes from Molecule.n_electrons(), which is wrong only when
# the SCF removed a positive number of core electrons.
# Routes whose kernels still partition occupied orbitals from the physical
# electron count. The MP2, CC and DLPNO families left this set on
# 2026-09-05: they consume ``effective_electron_count`` and the ECP-aware
# frozen core (handovers/HANDOVER_BASIS_ECP_UNIFICATION.md, M2).
# Empty since 2026-09-07 (#740): OVGF partitions from
# ``effective_electron_count`` at both its closed- and open-shell sites, so no
# route is left that removes core electrons in the SCF and then counts the
# physical total afterwards.
_MOLECULAR_ECP_REPLACED_CORE_UNSUPPORTED_METHODS: frozenset[str] = frozenset()

# Routes that build their own one-electron Hamiltonian without the
# reference's ECP operator. The determinant family reached through
# ``build_hamiltonian_mo`` (CISD, CC3, CCSDT, selected CI, DMRG, v2RDM,
# transcorrelated CI, FCI) and the Python ROHF / ROKS drivers left this set
# on 2026-09-05: they now take T + V_ne(Z_eff) + V_ECP and the valence count
# from the reference.
#
# Empty since 2026-09-07 (#740). The CAS / MR-PT family was assumed to keep a
# separate integral path; it does not. ``casci`` / ``mrci`` / ``casscf`` /
# ``nevpt2`` / ``caspt2`` are handed ``H.h1e`` / ``H.h2e`` from the same
# ``build_hamiltonian_mo(..., reference=...)`` call the determinant family
# uses, so they already carried the ECP operator. Only the active-space
# partition still counted physical electrons, which is fixed at that site.
_MOLECULAR_ECP_OPERATOR_UNSUPPORTED_METHODS: frozenset[str] = frozenset()

# Public inventory retained for tests, diagnostics, and release auditing.
# Keeping it separate from ``_FROZEN_CORE_ROUTE_METHODS`` prevents membership
# here from making an unsupported top-level ``frozen_core=`` selector appear
# accepted.
_MOLECULAR_ECP_UNSUPPORTED_METHODS = (
    _MOLECULAR_ECP_REPLACED_CORE_UNSUPPORTED_METHODS
    | _MOLECULAR_ECP_OPERATOR_UNSUPPORTED_METHODS
)


def _options_activate_molecular_ecp(options: object) -> bool:
    """Whether selected SCF options can remove molecular core electrons.

    Inline primitive inputs carry their authoritative aggregate core count,
    so a zero-core model potential is safe here. Explicit XML centers do not
    carry that count until native ECP construction and are conservatively
    rejected by this fail-fast layer; the result-carried count remains the
    authoritative post-SCF guard.
    """
    return molecular_options_replace_core(options)


def _basis_sidecar_replaces_molecular_core(molecule, basis: object) -> bool:
    """Whether an automatic basis sidecar applies a positive ECP core."""
    return basis_sidecar_replaces_core(molecule, basis)


_MOLECULAR_ECP_DERIVATIVE_METHODS = frozenset(
    {"rhf", "uhf", "rks", "uks"}
)


def _refuse_molecular_ecp_derivative_request(
    method: str,
    resolved_method: str,
    molecule,
    basis: object,
    *,
    optimize: bool,
    geom_opt: Optional[str],
    hessian: bool,
    solvent: object,
    rhf_options: object = None,
    uhf_options: object = None,
    rks_options: object = None,
    uks_options: object = None,
    rohf_options: object = None,
    roks_options: object = None,
) -> None:
    """Reject molecular ECP derivative routes without an exact surface.

    ``resolved_method`` names the mean-field reference for result-taking
    post-HF methods, so the original ``method`` must decide whether the
    requested optimization or finite-difference Hessian is itself a
    supported mean-field route.
    """
    if not (optimize or geom_opt is not None or hessian):
        return

    route_method = resolved_method if method == "auto" else method
    if route_method in SEMIEMPIRICAL_METHODS or route_method in _MLIP_METHODS:
        return

    selected_options = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
        "roks": roks_options,
    }.get(resolved_method)
    has_ecp_operator = (
        basis_sidecar_has_ecp_operator(molecule, basis)
        or molecular_options_request_ecp_operator(selected_options)
    )
    if not has_ecp_operator:
        return

    if route_method not in _MOLECULAR_ECP_DERIVATIVE_METHODS:
        raise NotImplementedError(
            f"run_job(method={method!r}): only RHF/UHF/RKS/UKS mean-field "
            "ECP gradients are supported for geometry optimization and "
            "finite-difference Hessians. Run this post-HF calculation as a "
            "single point or use an all-electron basis."
        )

    if solvent is not None:
        from .solvation import resolve_solvent

        solvent_model = resolve_solvent(solvent)
        if solvent_model is not None and not solvent_model.is_gas_phase:
            raise NotImplementedError(
                "ECP+CPCM molecular gradients are not implemented; geometry "
                "optimization and finite-difference Hessians require that "
                "missing derivative. Use gas phase or an all-electron basis."
            )


def _refuse_molecular_ecp_correlated_route(
    method: str,
    resolved_method: str,
    molecule,
    basis: object,
    *,
    rhf_options: object = None,
    uhf_options: object = None,
    rks_options: object = None,
    uks_options: object = None,
    rohf_options: object = None,
    roks_options: object = None,
) -> None:
    """Reject ECP-backed molecular routes with an unsafe electron boundary.

    The pure-Python restricted open-shell and determinant-Hamiltonian drivers
    omit the ECP one-electron operator entirely. Result-taking correlation
    routes preserve that operator but still partition occupied orbitals from
    the Molecule's physical electron count. Both hazards stop before planning
    or SCF work, using the narrow predicate appropriate to each family.
    """
    route_method = resolved_method if method == "auto" else method
    if route_method not in _MOLECULAR_ECP_UNSUPPORTED_METHODS:
        return
    selected_options = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
        "roks": roks_options,
    }.get(resolved_method)
    if route_method in _MOLECULAR_ECP_OPERATOR_UNSUPPORTED_METHODS:
        if route_method in {"rohf", "roks"}:
            option_sources = (selected_options,)
        else:
            # These routes choose their own reference internally; fail closed
            # on any explicitly supplied molecular SCF ECP request rather
            # than silently ignoring a requested operator.
            option_sources = (
                rhf_options,
                uhf_options,
                rks_options,
                uks_options,
                rohf_options,
                roks_options,
            )
        unsupported_ecp = (
            any(
                molecular_options_request_ecp_operator(source)
                for source in option_sources
                if source is not None
            )
            or basis_sidecar_has_ecp_operator(molecule, basis)
        )
    else:
        unsupported_ecp = (
            _options_activate_molecular_ecp(selected_options)
            or _basis_sidecar_replaces_molecular_core(molecule, basis)
        )
    if not unsupported_ecp:
        return
    if route_method in {"rohf", "roks"}:
        raise NotImplementedError(
            f"run_job(method={method!r}): molecular ECP references are not "
            f"supported because the pure-Python {route_method.upper()} "
            "driver does not apply the ECP one-electron operator or "
            "effective electron count. Use an all-electron basis."
        )
    if route_method in _MOLECULAR_ECP_OPERATOR_UNSUPPORTED_METHODS:
        raise NotImplementedError(
            f"run_job(method={method!r}): molecular ECP references are not "
            "supported because this route rebuilds a bare-Z one-electron "
            "Hamiltonian and would omit the ECP operator. Use an "
            "all-electron basis."
        )
    raise NotImplementedError(
        f"run_job(method={method!r}): molecular ECP references are not "
        "supported because SCF removes replaced core electrons while the "
        "post-HF route still partitions orbitals from "
        "Molecule.n_electrons() (the physical count). Selecting "
        "frozen_core=False/'all-electron' does not repair that mismatch; "
        "use an all-electron basis until the effective electron count is "
        "plumbed into post-HF."
    )


def _options_have_explicit_frozen_core(options: object) -> bool:
    """Whether a correlated-route option object carries a user selection."""
    if options is None:
        return False
    if hasattr(options, "_n_frozen_explicit"):
        return bool(getattr(options, "_n_frozen_explicit"))
    if hasattr(options, "_n_frozen_core_explicit"):
        return bool(getattr(options, "_n_frozen_core_explicit"))
    if hasattr(options, "n_frozen_core"):
        value = getattr(options, "n_frozen_core")
        if value is None:
            return False
        try:
            return int(value) != -1
        except (TypeError, ValueError):
            return True
    if hasattr(options, "n_frozen"):
        return getattr(options, "n_frozen") is not None
    return False


def _route_frozen_core_request(requested: object, options: object) -> object:
    """Recover the selector whose provenance a route should disclose."""
    if requested is not None:
        return requested
    if not _options_have_explicit_frozen_core(options):
        return None
    if hasattr(options, "n_frozen_core"):
        return getattr(options, "n_frozen_core")
    return getattr(options, "n_frozen")


def _frozen_core_citation_entries(
    method: str,
    requested: object,
    options: object,
) -> tuple[str, ...]:
    """Return the source citation only for the published FC selector."""
    # CC3/CCSDT expose their frozen-core selector only on their dedicated
    # option objects (not through run_job(frozen_core=)), but their implicit
    # count is resolved from the same published table.  Keep citation
    # eligibility separate from the public-selector inventory so adding the
    # source does not accidentally broaden the accepted run_job API.
    if method not in _FROZEN_CORE_ROUTE_METHODS and method not in (
        "cc3",
        "ccsdt",
    ):
        return ()
    from .correlation_conventions import (
        _PUBLISHED_FROZEN_CORE_CITATION_KEY,
        _uses_published_frozen_core_selector,
    )

    route_request = _route_frozen_core_request(requested, options)
    if not _uses_published_frozen_core_selector(route_request):
        return ()
    return (_PUBLISHED_FROZEN_CORE_CITATION_KEY,)


def _dlpno_triples_citation_entries(
    method: str,
    options: object,
    result=None,
) -> tuple[str, ...]:
    """Return both Guo sources only when iterative local ``(T1)`` runs.

    The 2018 closed-shell Eq. (2) keeps occupied triples in the localised
    basis and iterates their off-diagonal occupied-Fock coupling. The 2020
    paper is the dedicated open-shell extension. The compatibility ``"t1"``
    routes instead diagonalise/rotate the occupied space. Static method routes
    therefore cannot cite either source without over-claiming T0,
    rotated/canonical, exact, pilot, or skipped executions.
    """
    if method != "dlpno-ccsd(t)" or options is None or result is None:
        return ()
    from .dlpno.uccsd_local_solver import LocalUCCSDOptions

    if not isinstance(options, LocalUCCSDOptions):
        return ()
    if not bool(getattr(result, "triples_executed", False)):
        return ()
    mode, _ = _dlpno_triples_provenance(options, result)
    if mode != "t1-iterative":
        return ()
    return (
        "guo_dlpno_t1_2018",
        "guo_openshell_dlpno_triples_2020",
    )


def _frozen_core_label(requested: object) -> tuple[str, str]:
    """Return stable human/machine provenance for a resolved FC request."""
    from .correlation_conventions import (
        PUBLISHED_FROZEN_CORE_CONVENTION,
        _uses_published_frozen_core_selector,
    )

    if _uses_published_frozen_core_selector(requested):
        explicit = requested is not None
        return (
            "ORCA 6.1 published count-only"
            + (" (explicit)" if explicit else " default"),
            f"{PUBLISHED_FROZEN_CORE_CONVENTION}-"
            + ("explicit" if explicit else "default"),
        )
    normalized = None
    if isinstance(requested, str):
        normalized = requested.strip().lower().replace("_", "-")
        normalized = "-".join(normalized.split())
    if requested is False or requested == 0 or (
        normalized in {"all-electron", "all-electrons", "all", "none", "off"}
    ):
        return "explicit all-electron", "all-electron-explicit"
    return "explicit orbital count", "explicit-orbital-count"


def _format_frozen_core(
    n_frozen: int,
    requested: object,
    *,
    label_width: int = 24,
) -> str:
    del label_width  # Kept for call-site symmetry with threshold formatting.
    label, _ = _frozen_core_label(requested)
    return f"  Frozen core orbitals = {int(n_frozen):d} ({label})\n"


def _prepare_canonical_correlation_options(
    method: str,
    effective_method: str,
    molecule: Molecule,
    *,
    mp2_options: object,
    ump2_options: object,
    ccsd_options: object,
    requested_frozen_core_count: int | None,
    use_rohf_mp2: bool,
    ccsd_compute_triples: bool | None,
    ccsd_triples_variant: str,
    density_fit: bool | None,
    aux_basis: str | None,
    orbital_basis_name: str,
) -> tuple[object, object, object]:
    """Clone and resolve canonical post-SCF policy before both preflights.

    Scheduler dry-run, real admission, and execution must consume the same
    scientific option objects.  Resource budgets and element-coverage guards
    are added later, after the real basis is built, but frozen core, triples,
    integral route, and auxiliary-basis identity are already fixed here.
    Caller-owned objects are never mutated.
    """

    prepared_mp2 = None
    prepared_ump2 = None
    prepared_ccsd = None

    if method in ("mp2", "scs-mp2", "sos-mp2") and not use_rohf_mp2:
        if molecule.multiplicity > 1:
            prepared_ump2 = _clone_mp2_like_options(
                ump2_options,
                UMP2Options,
            )
            if requested_frozen_core_count is not None:
                prepared_ump2.n_frozen_core = requested_frozen_core_count
        else:
            prepared_mp2 = _clone_mp2_like_options(
                mp2_options,
                MP2Options,
            )
            if requested_frozen_core_count is not None:
                prepared_mp2.n_frozen_core = requested_frozen_core_count

        # Fix the correlation auxiliary identity on the private execution
        # clone before scheduler dry-run.  Deferring this until the real
        # BasisSet exists makes a DF dry-run use _aux_basis_size's generic
        # 3*N fallback while real admission and execution use the concrete
        # RIfit basis.  The later _auto_resolve_mp2_aux_basis call remains as
        # the element-coverage guard on the constructed basis.
        prepared_route = (
            prepared_ump2
            if prepared_ump2 is not None
            else prepared_mp2
        )
        if bool(getattr(prepared_route, "density_fit", False)):
            resolved_route_aux = _resolve_route_aux_basis(
                aux_basis,
                getattr(prepared_route, "aux_basis", ""),
                route=(
                    "ump2_options"
                    if prepared_ump2 is not None
                    else "mp2_options"
                ),
            )
            if resolved_route_aux:
                prepared_route.aux_basis = resolved_route_aux
            else:
                try:
                    from .density_fitting import default_aux_basis_for

                    prepared_route.aux_basis = default_aux_basis_for(
                        orbital_basis_name,
                        kind="ri",
                    )
                except Exception:
                    # Preserve the native route's explicit missing-auxiliary
                    # diagnostic when no registered RIfit partner exists.
                    prepared_route.aux_basis = ""

    if (
        method in ("ccsd", "ccsd(t)")
        or effective_method in _BCCD_METHODS
        or effective_method in _CC_VARIANT_METHODS
    ):
        from .cc import (
            CCSDOptions,
            _copy_ccsd_options,
            _resolve_ccsd_frozen_core,
        )

        prepared_ccsd = (
            CCSDOptions()
            if ccsd_options is None
            else _copy_ccsd_options(ccsd_options)
        )
        if requested_frozen_core_count is not None:
            prepared_ccsd.n_frozen_core = requested_frozen_core_count
            prepared_ccsd._n_frozen_core_explicit = True
        else:
            _resolve_ccsd_frozen_core(prepared_ccsd, molecule)
        prepared_ccsd.compute_triples = bool(ccsd_compute_triples)
        prepared_ccsd.triples_variant = ccsd_triples_variant
        prepared_ccsd.cc_variant = (
            "bccd"
            if effective_method in _BCCD_METHODS
            else _CC_VARIANT_METHODS.get(effective_method, "ccsd")
        )
        prepared_ccsd.density_fit = _resolve_cc_density_fit(
            density_fit,
            ccsd_options,
        )
        if prepared_ccsd.density_fit:
            if aux_basis:
                prepared_ccsd.aux_basis = _resolve_route_aux_basis(
                    aux_basis,
                    getattr(prepared_ccsd, "aux_basis", ""),
                    route="ccsd_options",
                )
            prepared_ccsd.resolve_aux_basis(orbital_basis_name)

    return prepared_mp2, prepared_ump2, prepared_ccsd


def _prepare_dlpno_route_options(
    method: str,
    molecule: Molecule,
    *,
    dlpno_options: object,
    dlpno_ccsd_options: object,
    frozen_core: object,
    thresholds: object,
    aux_basis: str | None,
    orbital_basis_name: str,
) -> tuple[object, object]:
    """Clone and resolve the exact DLPNO options that execution will use.

    This materialization happens before scheduler dry-run and real admission,
    so the memory estimator sees the same spin route, frozen occupied space,
    named cutoffs, and triples switch as the solver.  Only the clone is
    changed: runner-level policy must never rewrite a caller-owned options
    object as a side effect.
    """
    prepared_mp2 = None
    prepared_ccsd = None
    selected = None
    expected: tuple[type, ...] = ()

    if method == "dlpno-mp2":
        if molecule.multiplicity > 1:
            from .dlpno.ump2 import DLPNOUMP2Options

            expected = (DLPNOUMP2Options,)
            selected = (
                dlpno_options
                if dlpno_options is not None
                else DLPNOUMP2Options()
            )
        else:
            from .dlpno.mp2 import DLPNOMP2Options

            expected = (DLPNOMP2Options,)
            selected = (
                dlpno_options
                if dlpno_options is not None
                else DLPNOMP2Options()
            )
    elif method in ("dlpno-ccsd", "dlpno-ccsd(t)"):
        if molecule.multiplicity > 1:
            from .dlpno.uccsd import DLPNOUCCSDPilotOptions
            from .dlpno.uccsd_local_solver import LocalUCCSDOptions

            expected = (DLPNOUCCSDPilotOptions, LocalUCCSDOptions)
            selected = (
                dlpno_ccsd_options
                if dlpno_ccsd_options is not None
                else LocalUCCSDOptions()
            )
        else:
            from .dlpno.ccsd import DLPNOCCSDPilotOptions
            from .dlpno.ccsd_local_solver import LocalCCSDOptions

            expected = (DLPNOCCSDPilotOptions, LocalCCSDOptions)
            selected = (
                dlpno_ccsd_options
                if dlpno_ccsd_options is not None
                else LocalCCSDOptions()
            )
    else:
        return prepared_mp2, prepared_ccsd

    if not isinstance(selected, expected):
        expected_names = " or ".join(cls.__name__ for cls in expected)
        option_name = (
            "dlpno_options"
            if method == "dlpno-mp2"
            else "dlpno_ccsd_options"
        )
        raise TypeError(
            f"run_job: {option_name}= must be {expected_names} for this "
            f"{'open' if molecule.multiplicity > 1 else 'closed'}-shell "
            f"{method} route; got {type(selected).__name__}."
        )

    caller_selected = (
        dlpno_options if method == "dlpno-mp2" else dlpno_ccsd_options
    )
    caller_explicit_frozen = _options_have_explicit_frozen_core(
        caller_selected
    )
    # ``dataclasses.replace`` copies only declared fields.  DLPNO option
    # policy also carries runner-supported provenance markers (and older
    # callers may still attach ``aux_basis`` dynamically), so use a shallow
    # object copy to preserve that public option state without mutating the
    # caller-owned instance.
    selected = copy(selected)

    from .correlation_conventions import resolve_frozen_core_count

    route_request = _route_frozen_core_request(
        frozen_core, caller_selected
    )
    selected.n_frozen = resolve_frozen_core_count(molecule, route_request)
    # Preserve provenance after resolving the sentinel to an integer. Without
    # this marker the prepared default would be mislabeled as an explicit
    # count by the execution-side disclosure helper.
    selected._n_frozen_explicit = bool(
        frozen_core is not None or caller_explicit_frozen
    )

    # Resolve the RI identity before scheduler dry-run and real admission.
    # A large explicit auxiliary basis can otherwise be sized as the generic
    # 3*N fallback and only reach execution after the memory gate. Preserve
    # conflict semantics between the top-level selector and route options.
    selected.aux_basis = _resolve_route_aux_basis(
        aux_basis,
        getattr(selected, "aux_basis", None),
        route=(
            "dlpno_options"
            if method == "dlpno-mp2"
            else "dlpno_ccsd_options"
        ),
    )
    if not selected.aux_basis:
        try:
            from .density_fitting import default_aux_basis_for

            selected.aux_basis = default_aux_basis_for(
                orbital_basis_name,
                kind="ri",
            )
        except Exception:
            # Some orbital bases have no registered RI partner. Execution
            # will issue the route-specific error; the estimator keeps its
            # conservative dimension bound in the meantime.
            selected.aux_basis = None

    if thresholds is not None:
        from .dlpno import apply_dlpno_thresholds

        apply_dlpno_thresholds(selected, thresholds)
    if method in ("dlpno-ccsd", "dlpno-ccsd(t)"):
        selected.compute_triples = method == "dlpno-ccsd(t)"
        prepared_ccsd = selected
    else:
        prepared_mp2 = selected
    return prepared_mp2, prepared_ccsd


def _format_mlip_provenance(info, options=None) -> str:
    """A self-documenting MLIP provenance block for the .out -- which
    external pre-trained model produced these numbers, its license, and
    how to read the (model-specific, reference-shifted) energy."""
    lic = info.license.upper()
    if lic == "MIT":
        lic_note = "MIT  [free for commercial + academic use]"
    elif lic == "ASL":
        lic_note = "ASL  [ACADEMIC, NON-COMMERCIAL use only]"
    else:
        lic_note = f"{info.license}  [unverified -- treated as academic-only]"
    elts = f", {info.elements} elements" if info.elements else ""
    body = [
        f"  Model:     {info.key}  ({info.domain}{elts})",
        f"  Loader:    {info.loader}(model={info.loader_arg!r})",
        f"  License:   {lic_note}",
        f"  Training:  {info.training_data}   |   Theory: {info.theory}",
        "  Energy:    model-specific reference-shifted scale -- NOT a",
        "             vibe-qc total energy, not comparable across models.",
    ]
    if options is not None:
        from vibeqc.mlip.mace import mace_cache_root

        body.extend(
            [
                f"  Runtime:   device={options.device}   dtype={options.dtype}",
                f"  Weights:   upstream on-demand cache {mace_cache_root()}",
            ]
        )
    if info.doi:
        body.append(f"  Reference: doi:{info.doi}  (cited in the .bibtex sibling)")
    body.append(
        "  vibe-qc drives ACEsuit MACE's pre-trained forward pass (CLAUDE.md Sec.10)."
    )
    # Card rules size to the widest content line (no hand-guessed width).
    rule = "  " + "-" * (max(len(line) for line in body) - 2)
    lines = [
        rule,
        "  MACE machine-learning interatomic potential",
        rule,
        *body,
        rule,
    ]
    return "\n".join(lines)


def _cas_natural_orbitals(C_conv, rdm1_active, n_core, n_active_orb, n_elec_total):
    """Natural orbitals + occupations for a converged CAS wavefunction.

    ``C_conv`` is the converged CAS orbital set in the AO basis, columns as
    orbitals. ``rdm1_active`` is the active-block spin-summed 1-RDM in that
    same orbital basis.

    Natural orbitals are the eigenvectors of the one-particle density
    matrix and their eigenvalues are the occupation numbers, per Löwdin,
    *Phys. Rev.* **97**, 1474 (1955), doi:10.1103/PhysRev.97.1474, § 4.
    Pulay & Hamilton, *J. Chem. Phys.* **88**, 4926 (1988),
    doi:10.1063/1.454704, § II is the practical reading of the occupancies:
    orbitals between 0.02 and 1.98 are the fractionally occupied ones that
    constitute the active space (their window, which that paper itself calls
    "somewhat arbitrary"; nothing here thresholds on it, the numbers are
    reported as computed).

    The inactive and virtual blocks are already natural: a doubly occupied
    orbital and an empty one are eigenvectors of the density with
    eigenvalues 2 and 0, so only the active block needs diagonalizing.
    Eigenvectors come back in ascending eigenvalue order from ``eigh``, and
    are reversed here so the active block reads high occupancy first, which
    is the ordering every natural-orbital listing uses and the one the
    "highest occupied" of HONO refers to.

    Returns ``(orbitals, occupations)``, or ``None`` if anything is missing.
    Diagnostics must never kill a converged run.
    """
    try:
        if C_conv is None or rdm1_active is None:
            return None
        C_conv = np.asarray(C_conv, dtype=np.float64)
        rdm1_active = np.asarray(rdm1_active, dtype=np.float64)
        n_ao, n_mo = C_conv.shape
        if rdm1_active.shape != (n_active_orb, n_active_orb):
            return None

        occ_active, U = np.linalg.eigh(rdm1_active)
        occ_active = occ_active[::-1]
        U = U[:, ::-1]

        orbitals = C_conv.copy()
        lo, hi = n_core, n_core + n_active_orb
        if hi > n_mo:
            return None
        orbitals[:, lo:hi] = C_conv[:, lo:hi] @ U

        occupations = np.zeros(n_mo, dtype=np.float64)
        occupations[:n_core] = 2.0
        occupations[lo:hi] = occ_active

        # The occupations are an electron count; if they do not add up, the
        # core/active partition was wrong and publishing them would state a
        # falsehood about the wavefunction.
        if abs(occupations.sum() - float(n_elec_total)) > 1e-6:
            return None
        return orbitals, occupations
    except Exception:
        return None


def _active_rdm1(cas):
    """Active-space 1-RDM of a CASCI result's lowest root.

    Feeds the natural-occupation diagnostic in the .out solver block;
    routed through solvers._rdm.make_rdm12 so large full-CAS spaces use
    the C++ kernel.  Diagnostics must never kill a converged run, so any
    failure degrades to None (the block is simply omitted).
    """
    try:
        from .solvers._rdm import make_rdm12

        return make_rdm12(cas.ci_coeffs, cas.determinants, cas.n_active_orb)[0]
    except Exception:
        return None


#: Skip the per-root <S^2> diagnostic above this determinant count: the
#: spin-orbital-dict evaluation is Python-side and would add tens of
#: seconds on multi-million-determinant direct-CI vectors.
_ROOT_S2_MAX_DET = 200_000


def _root_s2_values(cas, ms2):
    """Per-root <S^2> of a multi-root CASCI result, for the .out root table.

    Shows which spin sector each state-averaged root lives in, the
    visibility surface for the 2026-06-11 spin-pure SA default (spin-pure
    runs print S(S+1) rows; ``spin_pure=False`` ensembles expose their
    mixed-spin composition).  Same degrade-to-omission contract as
    :func:`_active_rdm1`: any failure (or an oversized determinant list)
    returns None and the column is omitted.
    """
    try:
        if cas.ci_coeffs_all is None or cas.n_det > _ROOT_S2_MAX_DET:
            return None
        from .solvers._ms_caspt2 import _s2_expectation

        return [
            float(
                _s2_expectation(
                    cas.ci_coeffs_all[:, k],
                    cas.determinants,
                    cas.n_active_orb,
                    ms2,
                )
            )
            for k in range(cas.ci_coeffs_all.shape[1])
        ]
    except Exception:
        return None


def _format_solver_trace(result: SolverResult) -> str:
    """Format a non-mean-field solver result for the .out file."""
    import numpy as np

    lines = []
    lines.append(f"  Method:            {result.method}")
    lines.append(f"  Total energy:      {render_energy_labeled(result.energy, width=16, precision=10)}")
    lines.append(f"  Converged:         {result.converged}")
    if getattr(result, "stop_reason", None):
        lines.append(f"  Stop reason:       {result.stop_reason}")
    lines.append(f"  Iterations/sweeps: {result.n_iter}")
    if result.pt2_correction is not None:
        e_var = result.energy - result.pt2_correction
        lines.append(f"  E(variational):    {render_energy_labeled(e_var, width=16, precision=10)}")
        lines.append(f"  E(PT2 correction): {render_energy_labeled(result.pt2_correction, width=16, precision=10)}")
    if result.bond_dim is not None:
        lines.append(f"  Bond dimension:    {result.bond_dim}")
    if result.truncation_error is not None:
        lines.append(f"  Truncation error:  {result.truncation_error:.2e}")
    if result.constraint_residual is not None:
        lines.append(f"  Constraint resid.: {result.constraint_residual:.2e}")
    if result.ci_labels is not None and result.ci_coeffs is not None:
        lines.append(f"  Determinants:      {len(result.ci_labels)}")
        c_abs = np.abs(result.ci_coeffs)
        n_show = min(8, len(c_abs))
        idx = np.argsort(-c_abs)
        lines.append("  Leading configurations:")
        norb = 0
        for rank in range(n_show):
            i = idx[rank]
            label = result.ci_labels[i]
            # Format SpinDet as a|...> b|...>, Det as |...>
            if (
                isinstance(label, tuple)
                and len(label) == 2
                and isinstance(label[0], tuple)
            ):
                a_str = "".join(
                    "1" if j in label[0] else "0"
                    for j in range(max(label[0]) + 1 if label[0] else 0)
                )
                b_str = "".join(
                    "1" if j in label[1] else "0"
                    for j in range(max(label[1]) + 1 if label[1] else 0)
                )
                lines.append(f"    |c_{rank}| = {c_abs[i]:.6f}  a|{a_str}> b|{b_str}>")
            else:
                norb = max(norb, max(label) + 1 if label else 0)
                occ_str = "".join("1" if j in label else "0" for j in range(norb))
                lines.append(f"    |c_{rank}| = {c_abs[i]:.6f}  |{occ_str}>")
    if result.root_energies is not None and len(result.root_energies) > 1:
        lines.append("  Root energies:")
        s2s = result.root_s2
        if s2s is not None and len(s2s) != len(result.root_energies):
            s2s = None
        for k, e_root in enumerate(result.root_energies):
            s2_note = f"   <S^2> = {s2s[k]:7.4f}" if s2s is not None else ""
            lines.append(f"    root {k}:  E = {render_energy_labeled(e_root, width=16, precision=10)}{s2_note}")
    if result.selected_pt2:
        lines.append(
            "  Epstein-Nesbet PT2 on the selected wavefunction"
            " (Sharma et al 2017; variational energy unchanged):"
        )
        for k, p in enumerate(result.selected_pt2):
            sig = f" +/- {p['stderr']:.8f}" if p.get("stderr") else ""
            npert = (
                f"   ({p['n_perturbers']} perturbers)"
                if p.get("n_perturbers") is not None
                else "   (semistochastic)"
            )
            lines.append(f"    root {k}:  E_PT2 = {render_energy_labeled(p['e_pt2'], width=16, precision=10)}{sig}")
            lines.append(f"             E_var+PT2 = {render_energy_labeled(p['e_total'], width=16, precision=10)}{npert}")
    if result.multistate is not None:
        ms = result.multistate
        nst = ms["nroots"]
        mode_label = "XMS-CASPT2" if ms["mode"] == "xms" else "MS-CASPT2"
        lines.append(f"  {mode_label} ({nst} states):")
        lines.append(
            "    model-space reference energies (Ha): "
            + " ".join(f"{e:16.10f}" for e in ms["ref_energies"])
        )
        if ms.get("s2_values"):
            lines.append(
                "    model-space <S^2> (spin-pure roots):  "
                + " ".join(f"{s:16.4f}" for s in ms["s2_values"])
            )
        lines.append(
            "    single-state CASPT2 diagonal (Ha):   "
            + " ".join(f"{e:16.10f}" for e in ms["ss_energies"])
        )
        if ms["mode"] == "xms":
            lines.append("    XMS model-space rotation U (column = rotated state):")
            for row in np.asarray(ms["xms_rotation"]):
                lines.append("      " + " ".join(f"{u:10.6f}" for u in row))
        lines.append("    effective Hamiltonian (symmetrized, Ha):")
        for row in np.asarray(ms["heff"]):
            lines.append("      " + " ".join(f"{h:16.10f}" for h in row))
        lines.append("    mixing (column k = multi-state root k):")
        for row in np.asarray(ms["mixing"]):
            lines.append("      " + " ".join(f"{c:10.6f}" for c in row))
    if result.rdm1 is not None:
        occ = np.linalg.eigvalsh(np.asarray(result.rdm1))[::-1]
        lines.append(
            "  Natural occupations (active): " + " ".join(f"{o:7.4f}" for o in occ)
        )
    if result.energy_trace:
        lines.append("  Energy trace:")
        for i, e in enumerate(result.energy_trace):
            lines.append(f"    iter {i + 1:4d}:  E = {render_energy_labeled(e, width=16, precision=10)}")
    return "\n".join(lines)


def _format_ccsdt_result(result: CCSDTResult) -> tuple[str, str]:
    """Render full-CCSDT summary and opt-in iteration detail semantically."""
    summary = OutputDocument(indent=2, label_width=27)
    summary.scalar("Method", "CCSDT (full iterative T1/T2/T3)")
    summary.scalar("Reference energy", Quantity(result.e_reference, "energy"))
    summary.scalar(
        "Correlation energy", Quantity(result.e_correlation, "energy_delta")
    )
    summary.scalar("Total energy", Quantity(result.energy, "energy"))
    summary.scalar("Converged", "yes" if result.converged else "no")
    summary.scalar("Iterations", str(result.n_iter))
    summary.scalar("Frozen core orbitals", str(result.n_frozen_core))
    summary.scalar(
        "S/D/T amplitudes",
        f"{result.n_singles}/{result.n_doubles}/{result.n_triples}",
    )
    summary.scalar("T1 norm", Quantity(result.t1_norm, "dimensionless"))
    summary.scalar("T2 norm", Quantity(result.t2_norm, "dimensionless"))
    summary.scalar("T3 norm", Quantity(result.t3_norm, "dimensionless"))
    summary.scalar(
        "Final residual RMS", Quantity(result.residual_rms, "dimensionless")
    )
    summary.scalar(
        "Final residual max", Quantity(result.residual_max, "dimensionless")
    )

    iterations = Table(
        [
            Column("Iter"),
            Column("E(CCSDT)"),
            Column("dE"),
            Column("RMS residual"),
            Column("Max residual"),
            Column("DIIS"),
        ],
        indent=2,
    )
    for step in result.ccsdt_trace:
        iterations.add_row(
            step.iteration,
            Quantity(step.energy, "energy"),
            Quantity(step.delta_energy, "energy_delta"),
            Quantity(step.residual_rms, "dimensionless"),
            Quantity(step.residual_max, "dimensionless"),
            step.diis_subspace,
        )
    policy = active_policy().with_spec(
        "dimensionless", precision=6, width=0, notation="e"
    )
    return summary.render(policy), iterations.render(policy)


def _format_cc3_result(result: CC3Result) -> tuple[str, str]:
    """Render the CC3 summary and opt-in iteration detail semantically."""
    summary = OutputDocument(indent=2, label_width=27)
    summary.scalar("Method", "CC3 (iterative T1/T2, approximate T3)")
    summary.scalar("Reference energy", Quantity(result.e_reference, "energy"))
    summary.scalar(
        "Correlation energy", Quantity(result.e_correlation, "energy_delta")
    )
    summary.scalar("Total energy", Quantity(result.energy, "energy"))
    summary.scalar("Converged", "yes" if result.converged else "no")
    summary.scalar("Iterations", str(result.n_iter))
    summary.scalar("Frozen core orbitals", str(result.n_frozen_core))
    summary.scalar(
        "S/D/T amplitudes",
        f"{result.n_singles}/{result.n_doubles}/{result.n_triples}",
    )
    summary.scalar("T1 norm", Quantity(result.t1_norm, "dimensionless"))
    summary.scalar("T2 norm", Quantity(result.t2_norm, "dimensionless"))
    summary.scalar("T3 norm", Quantity(result.t3_norm, "dimensionless"))
    summary.scalar(
        "Final residual RMS", Quantity(result.residual_rms, "dimensionless")
    )
    summary.scalar(
        "Final residual max", Quantity(result.residual_max, "dimensionless")
    )

    iterations = Table(
        [
            Column("Iter"),
            Column("E(CC3)"),
            Column("dE"),
            Column("RMS residual"),
            Column("Max residual"),
            Column("DIIS"),
        ],
        indent=2,
    )
    for step in result.cc3_trace:
        iterations.add_row(
            step.iteration,
            Quantity(step.energy, "energy"),
            Quantity(step.delta_energy, "energy_delta"),
            Quantity(step.residual_rms, "dimensionless"),
            Quantity(step.residual_max, "dimensionless"),
            step.diis_subspace,
        )
    policy = active_policy().with_spec(
        "dimensionless", precision=6, width=0, notation="e"
    )
    return summary.render(policy), iterations.render(policy)


def _geom_summary(molecule: Molecule) -> str:
    """Three-column per-atom geometry listing (bohr)."""
    block = HeaderlessBlock(
        "Atoms (bohr)",
        [
            HeaderlessColumn(">", min_width=4),
            HeaderlessColumn(">", min_width=5, gutter_after=3),
            HeaderlessColumn(">", min_width=14),
            HeaderlessColumn(">", min_width=14),
            HeaderlessColumn(">", min_width=14),
        ],
        gutter=2,
    )
    for i, atom in enumerate(molecule.atoms, start=1):
        block.add_row(
            i,
            f"Z={atom.Z:3d}",
            f"{atom.xyz[0]:14.8f}",
            f"{atom.xyz[1]:14.8f}",
            f"{atom.xyz[2]:14.8f}",
        )
    block.footer(
        f"charge={molecule.charge}  multiplicity={molecule.multiplicity}  "
        f"n_electrons={molecule.n_electrons()}"
    )
    return block.render(active_policy())


def _job_header(
    method: str, basis_name: str, functional: Optional[str], scf_reference: str = "RHF"
) -> str:
    label = f"{method.upper()}"
    if method in ("rks", "uks") and functional:
        label = f"{label} / {functional}"
    # Post-SCF methods name their mean-field reference (RHF for closed shell,
    # UHF for an open-shell MP2/UMP2 run).
    if method in (
        "cisd",
        "ccsd",
        "ccsd(t)",
        "cc3",
        "ccsdt",
        "ccsd[t]",
        "a-ccsd(t)",
        "cc2",
        "ccd",
        "lccd",
        "lccsd",
        "cepa(0)",
        "cepa(1)",
        "cepa(2)",
        "cepa(3)",
        "qcisd",
        "qcisd(t)",
        "mp2",
        "scs-mp2",
        "sos-mp2",
        "dlpno-mp2",
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
        "ovgf",
    ):
        label = f"{scf_reference.upper()} + {method.upper()}"
    return f"  Job: {label}  basis={basis_name}"


def _make_semiempirical_ase_calculator(
    molecule: Molecule,
    method: str,
):
    """Return an ASE Calculator that computes energy via the
    semiempirical dispatcher and forces via the shared gradient adapter."""
    try:
        from ase.calculators.calculator import Calculator
        from ase.units import Bohr, Hartree
    except ImportError:
        raise ImportError("Geometry optimization requires ASE: pip install ase")

    import numpy as np

    from ._vibeqc_core import Atom as _Atom
    from ._vibeqc_core import Molecule as _Molecule

    h = 0.001  # bohr

    class _SemiempiricalCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self._pm6_density = None
            self._scc_charges = None
            self._last_result = None

        def calculate(self, atoms, properties, system_changes):
            # Record self.atoms per the ASE Calculator contract. Without
            # it check_state() reports every query as changed, so cached
            # results are discarded (each get_* triggers a fresh solve)
            # and TrajectoryWriter finds no stored energy -- semiempirical
            # optimization trajectories lost their per-frame energies.
            Calculator.calculate(self, atoms, properties, system_changes)
            positions_bohr = atoms.positions / Bohr
            mol = _Molecule(
                [
                    _Atom(int(z), list(xyz))
                    for z, xyz in zip(atoms.numbers, positions_bohr)
                ],
                charge=molecule.charge,
                multiplicity=molecule.multiplicity,
            )

            if (
                method == "pm6"
                and molecule.multiplicity == 1
                and self._pm6_density is not None
            ):
                result = _run_closed_shell_pm6_from_density(
                    mol, self._pm6_density
                )
            elif method == "scc_dftb":
                result = _run_semiempirical(
                    method,
                    mol,
                    initial_charges=self._scc_charges,
                )
            else:
                result = _run_semiempirical(method, mol)
            if not bool(getattr(result, "converged", True)):
                from .ase import VibeQCSCFConvergenceError, _closest_pair_note

                _, geometry_note = _closest_pair_note(atoms)
                raise VibeQCSCFConvergenceError(method, result, geometry_note)
            if method == "pm6" and molecule.multiplicity == 1:
                self._pm6_density = getattr(result, "density", None)
            elif method == "scc_dftb":
                self._scc_charges = getattr(result, "mulliken_charges", None)
            self._last_result = result
            energy_ha = float(getattr(result, "energy", 0.0))
            self.results["energy"] = energy_ha * Hartree

            if "forces" in properties:
                gradient = None
                gradient_fn = getattr(result, "gradient", None)
                if callable(gradient_fn):
                    gradient = gradient_fn()

                if gradient is not None:
                    gradient_ha_bohr = np.asarray(gradient, dtype=float)
                    expected_shape = (len(atoms), 3)
                    if gradient_ha_bohr.shape != expected_shape:
                        raise ValueError(
                            "Semiempirical gradient shape "
                            f"{gradient_ha_bohr.shape} does not match "
                            f"{expected_shape}."
                        )
                    if not np.all(np.isfinite(gradient_ha_bohr)):
                        raise ValueError("Semiempirical gradient contains NaN/Inf.")
                    forces_ha_bohr = -gradient_ha_bohr
                else:
                    forces_ha_bohr = np.zeros((len(atoms), 3))
                    for i in range(len(atoms)):
                        for c in range(3):
                            pos_plus = positions_bohr.copy()
                            pos_plus[i, c] += h
                            mol_p = _Molecule(
                                [
                                    _Atom(int(z), list(xyz))
                                    for z, xyz in zip(atoms.numbers, pos_plus)
                                ],
                                charge=molecule.charge,
                                multiplicity=molecule.multiplicity,
                            )
                            e_p = float(
                                getattr(
                                    _run_semiempirical(method, mol_p), "energy", 0.0
                                )
                            )

                            pos_minus = positions_bohr.copy()
                            pos_minus[i, c] -= h
                            mol_m = _Molecule(
                                [
                                    _Atom(int(z), list(xyz))
                                    for z, xyz in zip(atoms.numbers, pos_minus)
                                ],
                                charge=molecule.charge,
                                multiplicity=molecule.multiplicity,
                            )
                            e_m = float(
                                getattr(
                                    _run_semiempirical(method, mol_m), "energy", 0.0
                                )
                            )

                            forces_ha_bohr[i, c] = (e_m - e_p) / (2.0 * h)
                self.results["forces"] = forces_ha_bohr * (Hartree / Bohr)

    return _SemiempiricalCalculator()


def _make_wavefunction_ase_calculator(
    molecule: Molecule,
    basis_name: str,
    method: str,
    *,
    functional=None,
    roks_options=None,
    grid_level="orca-defgrid3",
    cisd_options=None,
    cc3_options=None,
    ccsdt_options=None,
    selected_ci_options=None,
    dmrg_options=None,
    v2rdm_options=None,
    transcorrelated_options=None,
    casci_options=None,
    caspt2_options=None,
    nevpt2_options=None,
    casscf_options=None,
    active_space=None,
    cas_reference=None,
    dft_plus_u: Optional[List["HubbardSite"]] = None,
):
    """Return an ASE Calculator that computes energy via the wavefunction
    solver and forces via central finite differences (h = 0.001 bohr).

    When ``dft_plus_u`` is set, the per-geometry SCF (energy + each FD
    gradient probe) runs with +U via :func:`_run_single_point`'s
    ``dft_plus_u`` kwarg. The analytic +U gradient (Increment 3) is
    *not* used by this calculator -- FD is good enough for BFGS step
    direction and avoids re-plumbing the calculator's force-call
    contract. For a tight optimization with analytic +U gradient,
    call :func:`vibeqc.compute_gradient` (or sibling) directly with
    ``dft_plus_u=`` per-step in your own loop.
    """
    try:
        from ase.calculators.calculator import Calculator
        from ase.units import Bohr, Hartree
    except ImportError:
        raise ImportError("Geometry optimization requires ASE: pip install ase")

    import numpy as np

    from ._vibeqc_core import Atom, BasisSet
    from ._vibeqc_core import Molecule as _Molecule

    h = 0.001  # bohr -- central-finite-difference step

    class _WavefunctionCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms, properties, system_changes):
            # Record self.atoms per the ASE Calculator contract. Without
            # it check_state() reports every query as changed, so cached
            # results are discarded (each get_* triggers a fresh solve)
            # and TrajectoryWriter finds no stored energy -- wavefunction
            # optimization trajectories lost their per-frame energies.
            Calculator.calculate(self, atoms, properties, system_changes)
            # Convert ASE Atoms -> vibeqc Molecule
            positions_bohr = atoms.positions / Bohr
            mol = _Molecule(
                [
                    Atom(int(z), list(xyz))
                    for z, xyz in zip(atoms.numbers, positions_bohr)
                ],
                charge=molecule.charge,
                multiplicity=molecule.multiplicity,
            )
            basis = BasisSet(mol, basis_name)

            # Compute energy at current geometry.  This call and the two
            # FD-displaced calls below must pass the *same* solver options:
            # any mismatch makes the BFGS energy and its forces sample
            # different surfaces (pre-2026-06-12 the energy call dropped
            # casscf_options while the FD calls carried it).
            result = _run_single_point(
                method,
                mol,
                basis,
                functional=functional,
                roks_options=roks_options, grid_level=grid_level,
                cisd_options=cisd_options,
                cc3_options=cc3_options,
                ccsdt_options=ccsdt_options,
                selected_ci_options=selected_ci_options,
                dmrg_options=dmrg_options,
                v2rdm_options=v2rdm_options,
                transcorrelated_options=transcorrelated_options,
                casci_options=casci_options,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                casscf_options=casscf_options,
                active_space=active_space,
                cas_reference=cas_reference,
                dft_plus_u=dft_plus_u,
            )
            energy_ha = float(getattr(result, "energy", 0.0))
            self.results["energy"] = energy_ha * Hartree

            # Numerical forces via central differences
            if "forces" in properties:
                forces_ha_bohr = np.zeros((len(atoms), 3))
                for i in range(len(atoms)):
                    for c in range(3):
                        # +h displacement
                        pos_plus = positions_bohr.copy()
                        pos_plus[i, c] += h
                        mol_plus = _Molecule(
                            [
                                Atom(int(z), list(xyz))
                                for z, xyz in zip(atoms.numbers, pos_plus)
                            ],
                            charge=molecule.charge,
                            multiplicity=molecule.multiplicity,
                        )
                        basis_plus = BasisSet(mol_plus, basis_name)
                        result_plus = _run_single_point(
                            method,
                            mol_plus,
                            basis_plus,
                            functional=functional,
                            roks_options=roks_options, grid_level=grid_level,
                            cisd_options=cisd_options,
                            cc3_options=cc3_options,
                            ccsdt_options=ccsdt_options,
                            selected_ci_options=selected_ci_options,
                            dmrg_options=dmrg_options,
                            v2rdm_options=v2rdm_options,
                            transcorrelated_options=transcorrelated_options,
                            casci_options=casci_options,
                            caspt2_options=caspt2_options,
                            nevpt2_options=nevpt2_options,
                            casscf_options=casscf_options,
                            active_space=active_space,
                            cas_reference=cas_reference,
                            dft_plus_u=dft_plus_u,
                        )
                        e_plus = float(getattr(result_plus, "energy", 0.0))

                        # -h displacement
                        pos_minus = positions_bohr.copy()
                        pos_minus[i, c] -= h
                        mol_minus = _Molecule(
                            [
                                Atom(int(z), list(xyz))
                                for z, xyz in zip(atoms.numbers, pos_minus)
                            ],
                            charge=molecule.charge,
                            multiplicity=molecule.multiplicity,
                        )
                        basis_minus = BasisSet(mol_minus, basis_name)
                        result_minus = _run_single_point(
                            method,
                            mol_minus,
                            basis_minus,
                            functional=functional,
                            roks_options=roks_options, grid_level=grid_level,
                            cisd_options=cisd_options,
                            cc3_options=cc3_options,
                            ccsdt_options=ccsdt_options,
                            selected_ci_options=selected_ci_options,
                            dmrg_options=dmrg_options,
                            v2rdm_options=v2rdm_options,
                            transcorrelated_options=transcorrelated_options,
                            casci_options=casci_options,
                            caspt2_options=caspt2_options,
                            nevpt2_options=nevpt2_options,
                            casscf_options=casscf_options,
                            active_space=active_space,
                            cas_reference=cas_reference,
                            dft_plus_u=dft_plus_u,
                        )
                        e_minus = float(getattr(result_minus, "energy", 0.0))

                        # Central difference: F = -dE/dR
                        forces_ha_bohr[i, c] = -(e_plus - e_minus) / (2 * h)

                self.results["forces"] = forces_ha_bohr * (Hartree / Bohr)

    return _WavefunctionCalculator()


class _ASEGeometryOptimizationError(RuntimeError):
    """ASE exhausted the molecular geometry-optimization step budget."""

    def __init__(
        self,
        *,
        system: Molecule,
        n_steps: int,
        max_steps: int,
        final_max_force: float,
        target_max_force: float,
        final_max_displacement: float,
        trajectory_path: Path,
    ) -> None:
        self.system = system
        self.n_steps = n_steps
        self.max_steps = max_steps
        self.final_max_force = final_max_force
        self.target_max_force = target_max_force
        self.final_max_displacement = final_max_displacement
        self.trajectory_path = trajectory_path
        super().__init__(
            "ASE BFGS geometry optimization did not converge after "
            f"{n_steps} of {max_steps} allowed steps: final max force "
            f"{final_max_force:.6g} eV/A exceeds the {target_max_force:.6g} "
            "eV/A target; final maximum atom displacement in the last step "
            f"was {final_max_displacement:.6g} A. The last geometry remains "
            f"available in {trajectory_path.name}."
        )


def _optimize_geometry(
    molecule: Molecule,
    basis_name: str,
    *,
    roks_options=None,
    functional: Optional[str],
    trajectory_path: Path,
    fmax: float,
    max_steps: int,
    dispersion_params: Optional[D3BJParams] = None,
    method: str = "rhf",
    cisd_options=None,
    cc3_options=None,
    ccsdt_options=None,
    selected_ci_options=None,
    dmrg_options=None,
    v2rdm_options=None,
    transcorrelated_options=None,
    casci_options=None,
    caspt2_options=None,
    nevpt2_options=None,
    casscf_options=None,
    active_space=None,
    cas_reference=None,
    rhf_options: Optional[RHFOptions] = None,
    uhf_options: Optional[UHFOptions] = None,
    rks_options: Optional[RKSOptions] = None,
    uks_options: Optional[UKSOptions] = None,
    rohf_options: Optional[ROHFOptions] = None,
    mlip_options: Optional[MLIPOptions] = None,
    result_holder: Optional[list] = None,
    grid_level: str = "orca-defgrid3",
) -> Molecule:
    """Run an ASE/BFGS geometry optimization, writing frames to a .traj file.

    Returns a new Molecule at the optimized geometry. The SCF options
    apply to every intermediate geometry, not just the final point -- use
    them to bump ``max_iter`` / add damping for systems where BFGS
    sometimes lands on a hard-to-converge geometry (H-bonded clusters,
    near-degenerate states).

    For wavefunction methods (``cisd``, ``selected_ci``, ``dmrg``, ``v2rdm``,
    ``transcorrelated_ci``, ``casci``, ``casscf``), forces are computed
    via central finite differences because the solvers do not yet provide
    analytic gradients. The per-step calculator receives the same solver
    options (``casscf_options``, ``casci_options``, ``cas_reference``, ...)
    as the final single point, so the optimizer walks the same surface
    the reported final energy is evaluated on.
    """
    # ASE < 3.23 line-search optimizers import scipy names removed in
    # SciPy >= 1.14; restore the aliases before touching ase.optimize.
    from .ase_optimizers import ensure_ase_scipy_compat

    ensure_ase_scipy_compat()
    try:
        from ase import Atoms
        from ase.io.trajectory import Trajectory
        from ase.optimize import BFGSLineSearch
        from ase.units import Bohr
    except ImportError as exc:
        raise ImportError(
            "Geometry optimization requires ASE. Install it with "
            "`pip install ase` into your vibe-qc venv."
        ) from exc

    from .ase import VibeQC

    positions_ang = [[coord * Bohr for coord in atom.xyz] for atom in molecule.atoms]
    atoms = Atoms(
        numbers=[atom.Z for atom in molecule.atoms],
        positions=positions_ang,
    )

    # casci / casscf joined 2026-06-12: they previously fell through to the
    # mean-field VibeQC calculator below, so the optimizer silently walked
    # the RHF/RKS surface while the final single point ran the CAS solver.
    # nevpt2 / caspt2 / mrci / fci still fall through (routing them onto an
    # FD-on-PT2 surface is a cost/policy call for the maintainer).
    wavefunction_methods = {
        "cisd",
        "cc3",
        "ccsdt",
        "selected_ci",
        "dmrg",
        "v2rdm",
        "transcorrelated_ci",
        "casci",
        "casscf",
    }
    if method in MOLECULAR_SEMIEMPIRICAL_METHODS:
        atoms.calc = _make_semiempirical_ase_calculator(
            molecule,
            method,
        )
    elif method in _MLIP_METHODS:
        # MACE geometry optimization: ASE BFGS drives the MACE ASE
        # calculator directly (it works in eV/Angstrom natively, the units
        # ASE optimizers expect). run_job's early ASL gate already
        # validated the model; MACEModel re-checks (defense in depth).
        from vibeqc.mlip.mace import MACEModel

        atoms.calc = MACEModel(molecule, mlip_options).calculator
    elif method == "rohf":
        # ROHF has an analytic gradient (compute_rohf_gradient), so route it
        # through the analytic VibeQC calculator (restricted_open=True) for
        # full-speed optimisation -- not the finite-difference path.
        atoms.calc = VibeQC(
            basis=basis_name,
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            restricted_open=True,
            dispersion=dispersion_params,
            rohf_options=rohf_options,
        )
    elif method in wavefunction_methods or method == "roks":
        # ROKS + the gradient-less wavefunction solvers use the
        # finite-difference force path: ROKS's analytic XC-gradient term is
        # a later milestone (handovers/HANDOVER_ROHF.md M5b), so geometry optimisation
        # differentiates the (verified) energy numerically. functional is
        # threaded for ROKS.
        atoms.calc = _make_wavefunction_ase_calculator(
            molecule,
            basis_name,
            method,
            functional=functional,
            cisd_options=cisd_options,
            cc3_options=cc3_options,
            ccsdt_options=ccsdt_options,
            selected_ci_options=selected_ci_options,
            dmrg_options=dmrg_options,
            v2rdm_options=v2rdm_options,
            transcorrelated_options=transcorrelated_options,
            casci_options=casci_options,
            caspt2_options=caspt2_options,
            nevpt2_options=nevpt2_options,
            casscf_options=casscf_options,
            active_space=active_space,
            cas_reference=cas_reference,
            roks_options=roks_options,
            grid_level=grid_level,
        )
    else:
        atoms.calc = VibeQC(
            basis=basis_name,
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            functional=functional,
            dispersion=dispersion_params,
            rhf_options=rhf_options,
            uhf_options=uhf_options,
            rks_options=rks_options,
            uks_options=uks_options,
            # GitLab #663: the calculator applies its grid preset to an
            # untouched grid, and run_job's resolved level must be the one it
            # applies; without this a run_job(grid_level="legacy") optimisation
            # was re-defaulted to orca-defgrid3 here.
            grid_level=grid_level,
        )

    last_positions = atoms.positions.copy()
    final_max_displacement = 0.0

    def _capture_step_displacement() -> None:
        nonlocal last_positions, final_max_displacement
        displacement = atoms.positions - last_positions
        if len(displacement):
            final_max_displacement = float(
                np.max(np.linalg.norm(displacement, axis=1))
            )
        last_positions = atoms.positions.copy()

    with Trajectory(str(trajectory_path), "w", atoms) as traj:
        # BFGSLineSearch refuses uphill steps by construction -- essential
        # on flat / weakly-bound PESs (H-bonded clusters, dispersion-dominated
        # complexes) where plain BFGS's Hessian extrapolation routinely
        # pushes atoms apart. Works on covalent minima too, so we use it
        # uniformly. SCC-DFTB is more fragile because failed charge fixed
        # points make forces unusable; keep its quasi-Newton trial steps inside
        # the region where the SCC solve remains on a continuous branch.
        opt_kwargs = {"maxstep": 0.08} if method == "scc_dftb" else {}
        opt = BFGSLineSearch(atoms, logfile=None, **opt_kwargs)
        opt.attach(traj)
        opt.attach(_capture_step_displacement)
        converged = bool(opt.run(fmax=fmax, steps=max_steps))

    # Re-build a vibe-qc Molecule at the optimized geometry.
    from ._vibeqc_core import Atom as _Atom

    new_positions_bohr = atoms.positions / Bohr
    optimized = Molecule(
        [_Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, new_positions_bohr)],
        molecule.charge,
        molecule.multiplicity,
    )
    if not converged:
        forces = np.asarray(atoms.get_forces(), dtype=float)
        final_max_force = (
            float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0
        )
        raise _ASEGeometryOptimizationError(
            system=optimized,
            n_steps=int(getattr(opt, "nsteps", max_steps)),
            max_steps=max_steps,
            final_max_force=final_max_force,
            target_max_force=fmax,
            final_max_displacement=final_max_displacement,
            trajectory_path=trajectory_path,
        )
    if result_holder is not None:
        result_holder.append(getattr(atoms.calc, "_last_result", None))
    return optimized


def _resolve_optimizer_backend(requested: str) -> str:
    """Resolve ``optimizer_backend`` to ``"ase"``, ``"native"``, or ``"brent"``."""
    if requested == "ase":
        try:
            import ase  # noqa: F401
        except ImportError:
            raise ImportError(
                "optimizer_backend='ase' requires ASE. Install with "
                "`pip install ase` or use optimizer_backend='native' / 'brent'."
            ) from None
        return "ase"
    if requested == "native":
        return "native"
    if requested == "brent":
        return "brent"
    if requested != "auto":
        raise ValueError(
            f"optimizer_backend={requested!r} -- expected 'auto', 'ase', "
            f"'native', or 'brent'."
        )
    # "auto": prefer ASE if installed, otherwise native.
    try:
        import ase  # noqa: F401
    except ImportError:
        return "native"
    return "ase"


# Coupled cluster's own integral-route default is density fitting: the RI
# error (1e-5 to 1e-4 Ha) sits orders of magnitude below chemical accuracy
# while the conventional four-index route caps out around ~100 basis
# functions in core.  See docs/user_guide/ccsd.md.
_CC_DENSITY_FIT_DEFAULT = True


def _resolve_cc_density_fit(
    density_fit: Optional[bool],
    ccsd_options: object,
) -> bool:
    """The integral route the coupled-cluster step will actually run.

    Three inputs, one answer, computed identically at every site that
    needs it (the fail-early aux gate, the memory pre-flight, and the
    execution block) so the estimate and the run cannot disagree -- the
    BUG 117 failure mode.

    * ``ccsd_options.density_fit`` is the CC-level request and wins.
    * ``run_job(density_fit=...)``, when explicitly given, is a
      whole-job request and governs the CC step as well.  Before this
      existed the kwarg was wired onto the SCF option structs only, so
      ``density_fit=False`` ran DF-CCSD(T) without a word
      (CCSDT-SILENT-DF-DEFAULT: two published parity cells compared a
      silently density-fitted vibe-qc leg against a conventional ORCA
      leg, and the residual was read as method error).
    * Neither supplied: :data:`_CC_DENSITY_FIT_DEFAULT`.

    Two explicit requests that contradict each other raise rather than
    letting one silently win.
    """
    if ccsd_options is None:
        if density_fit is None:
            return _CC_DENSITY_FIT_DEFAULT
        return bool(density_fit)
    cc_level = bool(
        getattr(ccsd_options, "density_fit", _CC_DENSITY_FIT_DEFAULT)
    )
    if density_fit is not None and bool(density_fit) != cc_level:
        raise ValueError(
            f"run_job: density_fit={bool(density_fit)} contradicts "
            f"ccsd_options.density_fit={cc_level}. Both select the "
            "coupled-cluster integral route, so pass only one of them -- "
            "drop the run_job density_fit= kwarg to let ccsd_options "
            "decide, or drop ccsd_options.density_fit to let the "
            "whole-job kwarg decide."
        )
    return cc_level


def _resolve_route_aux_basis(
    aux_basis: Optional[str],
    options_aux: Optional[str],
    *,
    route: str,
) -> Optional[str]:
    """Merge run_job(aux_basis=...) with a route options' ``aux_basis``.

    Both name the same physical object -- the auxiliary basis for a
    correlated route (DLPNO-MP2, DLPNO-CCSD, DF-CCSD/LCCSD/...) -- so
    two explicit requests that disagree raise rather than letting one
    silently win: the same contract as :func:`_resolve_cc_density_fit`
    (BUG 113: a kwarg that was accepted but never reached the route).
    Returns ``None`` when neither side supplied one (the caller then
    auto-resolves from the orbital basis).
    """
    kw = (aux_basis or "").strip().lower()
    opt = (options_aux or "").strip().lower()
    if kw and opt and kw != opt:
        raise ValueError(
            f"run_job: aux_basis={aux_basis!r} contradicts {route} "
            f"aux_basis={options_aux!r}. Both name the same auxiliary "
            f"basis, so pass only one of them -- drop the run_job "
            f"aux_basis= kwarg to let {route} decide, or drop "
            f"{route}.aux_basis to let the whole-job kwarg decide."
        )
    return aux_basis or options_aux or None


def run_job(
    molecule: Molecule,
    *,
    basis: Optional[str] = None,
    fetch_from_bse: Optional[bool] = None,
    method: Method = "auto",
    functional: Optional[str] = None,
    initial_guess: Optional[object] = None,
    citype: Any = None,
    triples: Any = None,
    output: str | os.PathLike = "output",
    name_molecule: bool = True,
    optimize: bool = False,
    write_molden_file: bool | None = None,
    write_xyz_file: bool = True,
    write_population_file: bool | None = None,
    write_cube: Union[bool, str, int, list, tuple, None] = False,
    cube_spacing: float = 0.2,
    cube_padding: float = 4.0,
    citations: bool = True,
    dry_run: bool = False,
    fmax: float = 0.05,
    max_opt_steps: int = 200,
    optimizer_backend: str = "auto",
    # Uniform geomopt keywords (v0.14+). When geom_opt is set,
    # the new geomopt framework is used directly.
    geom_opt: Optional[str] = None,
    geom_coords: str = "cartesian",
    geom_target: str = "minimum",
    geom_hessian_init: str = "diagonal",
    geom_hessian_update: str = "none",
    geom_line_search: str = "backtracking",
    geom_conv_gmax: Optional[float] = None,
    geom_freeze: Optional[List[int]] = None,
    geom_opt_options: Optional[dict[str, Any]] = None,
    geom_restart: Optional[str] = None,
    geom_checkpoint: Optional[str] = None,
    memory_override: bool = False,
    memory_budget_bytes: Optional[int] = None,
    num_threads: Optional[int] = None,
    dispersion: DispersionSpec = None,
    solvent: object = None,
    # ``None`` means "not specified": the SCF stays on the four-index
    # path (the historical ``False`` default) and each post-SCF method
    # keeps its own documented default.  An explicit ``True``/``False``
    # is a request that governs the correlated route too -- see the
    # ``density_fit`` entry in the docstring below.
    density_fit: Optional[bool] = None,
    aux_basis: Optional[str] = None,
    cosx: bool = False,
    # DFT integration-grid resolution for RKS/UKS/ROKS.  Ignored for HF and
    # semiempirical methods.  Accepts:
    #   "orca-defgrid3" (DEFAULT) — Lebedev-302 + NWChem pruning, matches ORCA
    #   "fine"            — Lebedev-590 radial-99, grid-converged benchmarks
    #   "coarse"          — Lebedev-194 radial-50, rapid screening
    #   "legacy"          — v0.15 product Gauss-Legendre 17×36, Becke partition
    #   "skala"           — vibe-qc's pinned PySCF-2.14 level-3 parity profile
    #                         for Microsoft SKALA-1.1
    grid_level: str = "orca-defgrid3",
    record_hostname: bool = True,
    rhf_options: Optional[RHFOptions] = None,
    uhf_options: Optional[UHFOptions] = None,
    rks_options: Optional[RKSOptions] = None,
    uks_options: Optional[UKSOptions] = None,
    rohf_options: Optional[ROHFOptions] = None,
    roks_options: Optional[ROKSOptions] = None,
    cisd_options: Optional[CISDOptions] = None,
    cc3_options: Optional[CC3Options] = None,
    ccsdt_options: Optional[CCSDTOptions] = None,
    selected_ci_options: Optional[SelectedCIOptions] = None,
    dmrg_options: Optional[DMRGOptions] = None,
    v2rdm_options: Optional[V2RDMOptions] = None,
    transcorrelated_options: Optional[TranscorrelatedOptions] = None,
    casci_options: Optional[CASCIOptions] = None,
    caspt2_options: Optional[CASPT2Options] = None,
    nevpt2_options: Optional[NEVPT2Options] = None,
    casscf_options: Optional[CASSCFOptions] = None,
    ccsd_options: Optional["CCSDOptions"] = None,
    ccsd_reference: str = "uhf",
    mp2_reference: str = "uhf",
    mp2_options: Optional[MP2Options] = None,
    ump2_options: Optional[UMP2Options] = None,
    dlpno_options: Optional["DLPNOMP2Options"] = None,
    dlpno_ccsd_options: Optional[object] = None,
    frozen_core: object = None,
    dlpno_thresholds: object = None,
    active_space: Optional[tuple[int, int]] = None,
    cas_reference: Optional[str] = None,
    mlip_options: Optional[MLIPOptions] = None,
    ccm_options: Optional["CCMOptions"] = None,
    read_from: Optional[object] = None,
    fragments: Optional[object] = None,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    use_logging: bool = False,
    perf_log: Optional[Union[str, os.PathLike, bool]] = None,
    structured_log: Union[bool, str, os.PathLike, None] = False,
    crash_dump: Union[bool, str, os.PathLike, None] = True,
    # QVF visualisation archive (v1).
    output_qvf: bool = True,
    # TREXIO wavefunction container (#573). False (default) writes nothing;
    # True writes ``{output}.trexio.h5`` (HDF5) or ``{output}.trexio``
    # (text directory) per ``trexio_backend``; a path writes there.
    trexio: bool | str | os.PathLike = False,
    trexio_backend: str = "hdf5",
    # Opt-in live QVF checkpointing for vibe-view hot-reload. When
    # ``checkpoint_qvf`` is set, a running snapshot is atomically written
    # there (start frame + terminal frame), labeled
    # ``provenance.run_status`` = ``"running"`` -> ``"converged"`` /
    # ``"failed"`` with a monotonic ``provenance.checkpoint.seq``. The
    # molecular native callback is reserved for terminal/NDJSON/manifest
    # output, so a single-point job still emits start + terminal QVF frames
    # only; ``checkpoint_every`` is honoured on checkpoint-aware paths.
    checkpoint_qvf: Optional[Union[str, os.PathLike]] = None,
    checkpoint_every: int = 0,
    qtaim: bool = False,
    # Localized orbitals + IAO partial charges (QVF-bound, on by default).
    # Runs each named criterion on the converged closed-shell occupied set and
    # emits one ``wavefunction.gto`` section per criterion
    # (``wf_localized_ibo``, ``..._boys``, ``..._pipek_mezey``), which
    # vibe-view renders as a Lewis-structure view and lets the user switch
    # between. ``True``/``None`` selects the default set, ``False``/``"none"``
    # disables it, or pass a specific criterion or sequence of them.
    # See handovers/HANDOVER_IBO.md.
    localize: Union[bool, str, Sequence[str], None] = None,
    # TD-DFT excited states + Natural Transition Orbitals (opt-in, QVF-bound).
    tddft: bool = False,
    tddft_n_states: int = 5,
    tddft_type: Literal["tda", "casida"] = "tda",
    tddft_gradient: bool = False,
    tddft_spectrum: bool = False,
    tddft_molden: bool = False,
    nto: bool = False,
    hessian: bool = False,
    # Partial Hessian: atom indices to hold fixed so only the unfrozen
    # atoms are displaced (6M instead of 6N SCFs); the frequencies are
    # then vibrational-only (no gas-phase trans/rot). See HessianFDOptions.
    hessian_frozen_indices: Optional[List[int]] = None,
    # Thermochemistry (RRHO ideal-gas). When a Hessian is computed
    # (``hessian=True``), the harmonic frequencies feed an end-of-run
    # thermochemistry block (ZPE + thermal U/H/S/G at T, p). Pass a
    # ThermoOptions to set temperature / pressure / rotational symmetry
    # number; None uses the defaults (298.15 K, 1 atm, s=1).
    thermo_options: Optional["ThermoOptions"] = None,
    # Atomization energy (S E_atom - E_mol) for mean-field methods. When True,
    # free-atom ground-state references are computed at the same level (cached
    # per element) and an atomization block is written. Main-group H-Kr; see
    # vibeqc.atomization.
    atomization: bool = False,
    # DFT+U (Dudarev rotationally-invariant) -- Increment 2c.
    # List of HubbardSite objects; empty / None == no +U.
    dft_plus_u: Optional[List["HubbardSite"]] = None,
    # MSINDO NDDO mode (method="msindo"): the reference program's default mode
    # (separate parametrization + two-centre multipole 2e Fock).  Ignored for
    # other methods.
    nddo: bool = False,
    # Eigensolver selection for Fock diagonalisation.
    # "dense" (default) -- full dense diagonalisation via LAPACK.
    # "davidson" -- block-Davidson iterative solver (C++ backend).
    # "lobpcg" -- LOBPCG iterative solver (C++ backend).  For molecular SCF,
    #   currently falls back to the Davidson path because the C++ molecular
    #   SCF kernel only exposes a use_davidson switch; the Python-level
    #   LOBPCG stack is available to periodic drivers and ROHF.
    solver: str = "dense",
    # Finite-temperature smearing for the molecular SCF eigenvalues.
    # After the integer-occupation SCF converges, the eigenvalues are
    # re-occupied with the chosen smearing function and the Mermin free
    # energy ``A = E - T·S`` is reported.  For gapped molecules the
    # density change is negligible, so this post-SCF correction gives the
    # correct finite-temperature free energy at the integer-occupation
    # density (the standard approach in molecular quantum chemistry).
    # ``smearing_temperature`` is ``k_B T`` in Hartree; ``smearing_method``
    # selects ``"fermi-dirac"`` (default), ``"mermin"``,
    # ``"methfessel-paxton"``, or ``"marzari-vanderbilt"``.
    smearing_temperature: float = 0.0,
    smearing_method: str = "fermi-dirac",
    # Periodic-only trap parameters. Molecules have no lattice or Brillouin
    # zone; requesting a k-point feature here is a common mix-up with
    # run_periodic_job, so these raise a targeted error instead of Python's
    # bare "unexpected keyword argument". Always None on the molecular path.
    kpoints: object = None,
    dos_kmesh: object = None,
    jk_method: object = None,
    bz_integration: object = None,
    cutoff_ha: object = None,
) -> object:
    """Run a vibe-qc SCF job and write the standard output files.

    Parameters
    ----------
    molecule
        The :class:`Molecule` describing the system (bohr coordinates).
    basis
        libint-recognized basis-set name.
    fetch_from_bse
        When true, a basis name the bundled library does not carry is
        rendered from the Basis Set Exchange into the per-user cache and
        made resolvable for this process; see :mod:`vibeqc.basis_fetch`.
        Needs the optional ``[bse]`` extra. ``None`` (the default) consults
        ``$VIBEQC_FETCH_BSE``; with neither, an unbundled name refuses
        exactly as before. A bundled name is unaffected, and a bundled basis
        that is merely missing one of the molecule's elements is refused by
        the coverage guard rather than patched from BSE.
    method
        ``"rhf"``, ``"uhf"``, ``"rks"``, ``"uks"``, ``"auto"``, ``"ccsd"``,
        ``"ccsd(t)"``, ``"cc3"``, ``"ccsdt"``, ``"cc2"``, ``"bccd"``, ``"bccd(t)"``,
        ``"qcisd"``, or ``"qcisd(t)"``. CC2 and the coupled-pair/QCI variants
        require a closed-shell reference; CCSD picks restricted vs unrestricted
        from ``molecule.multiplicity``. CC3 iterates singles and doubles with
        approximate triples; CCSDT is the dense full iterative
        singles/doubles/triples solver. CC3 requires a closed-shell RHF
        reference.
    functional
        XC functional for RKS / UKS (e.g. ``"PBE"``, ``"B3LYP"``). Ignored
        for HF.
    initial_guess
        High-level SCF initial-guess selector for molecular RHF/UHF/RKS/UKS
        and their post-SCF reference SCFs. Accepts :class:`vibeqc.InitialGuess`
        or spellings such as ``"auto"``, ``"sad"``, ``"sap"``, ``"patom"``,
        ``"hcore"``, ``"huckel"``, ``"minao"``, ``"read"``, and
        ``"fragmo"``.
    citype
        AutoCI-style selector for single-reference CI/CC methods. Supported
        now: ``"cisd"``, ``"ccsd"``, ``"ccsd(t)"``, ``"cc3"``, ``"ccsdt"``, ``"bccd"``,
        ``"bccd(t)"``, ``"cc2"``, ``"ccd"``, ``"lccd"``, ``"lccsd"``,
        ``"cepa(0)"``..``"cepa(3)"``, ``"qcisd"``, and ``"qcisd(t)"``
        (the Brueckner/coupled-pair/QCI variants require a closed-shell
        RHF reference). CC3 also requires a closed-shell RHF reference.
    cisd_options
        Optional :class:`vibeqc.CISDOptions` controlling the high-level
        CISD route: ``max_excitation`` (2 for CISD, 1 for CIS),
        ``nroots``, and ``max_det``.
    cc3_options
        Optional :class:`vibeqc.CC3Options` controlling CC3 convergence
        thresholds, DIIS history, iteration limit, and frozen-core count.
        ``n_frozen_core=None`` uses the chemical-core default in
        :func:`run_job`.
    ccsdt_options
        Optional :class:`vibeqc.CCSDTOptions` controlling the full iterative
        CCSDT convergence thresholds, DIIS history, iteration limit, and
        frozen-core count. ``n_frozen_core=None`` uses the chemical-core
        default in :func:`run_job`.
    triples
        CCSD perturbative-triples selector. Use ``"none"`` for plain
        CCSD, ``"(t)"`` for the standard Raghavachari correction,
        ``"[t]"`` (equivalently ``"+T(CCSD)"``) for the fourth-order
        bracket correction CCSD[T], or ``"A-CCSD(T)"`` for the closed-shell
        asymmetric/Lambda triples correction.
    ccsd_options
        Optional :class:`vibeqc.CCSDOptions` controlling canonical CCSD
        iteration thresholds, density-fitting settings, frozen-core
        count, and triples memory mode. In ``run_job`` the effective
        triples calculation is selected by ``method`` / ``triples`` so
        output labels and citations stay aligned.
    mp2_options, ump2_options
        Optional :class:`vibeqc.MP2Options` / :class:`vibeqc.UMP2Options`
        for the native post-SCF MP2 step. ``mp2_options`` applies to
        closed-shell RMP2, ``ump2_options`` to open-shell UMP2. Set
        ``density_fit=True`` to run RI-MP2/RI-UMP2 through ``run_job``;
        if ``aux_basis`` is empty, the per-zeta RI auxiliary basis is
        auto-detected when available.
    frozen_core
        Uniform frozen-core selector for MP2, CCSD, and DLPNO MP2/CCSD
        routes. ``None`` (default), ``True``, ``"published"``, or
        ``"chemical"`` use the ORCA 6.1 Table 2.69 published element-count
        convention; ``False`` or ``"all-electron"`` correlate every
        occupied orbital; a non-negative integer freezes exactly that many
        lowest-energy spatial orbitals. Do not combine this selector with an
        explicit route-option frozen-core count.
    dlpno_thresholds
        Named DLPNO cutoff set: ``"normal"`` (the default), ``"loose"``,
        or ``"tight"``. The Liakos NormalPNO triple is vibe-qc's shared
        cross-route default. Each solver applies only cutoffs its algorithm
        consumes and reports unsupported or inactive components. A named
        selector cannot be combined with ``dlpno_options`` or
        ``dlpno_ccsd_options``; use
        :func:`vibeqc.dlpno.options_from_dlpno_thresholds` for a preset plus
        explicit overrides.
    output
        Path stem for the generated files.
    name_molecule
        If ``True`` (default), print the IUPAC name of *molecule*
        to the ``.out`` header and live progress log. ``{output}.out`` always; also
        ``{output}.molden`` unless disabled; and ``{output}.traj`` when
        ``optimize=True``.
    optimize
        Run a BFGS geometry optimization first (via ASE), then the final
        SCF on the optimized geometry. The trajectory is written for
        animation (openable with ASE-aware viewers).
    write_molden_file
        Emit ``{output}.molden`` at the converged geometry. ``None`` (the
        default) emits it when the selected route exposes a Gaussian AO
        wavefunction. Explicit ``True`` is a guarantee and fails before the
        calculation on an unsupported route; ``False`` disables it.
    write_population_file
        Emit ``{output}.population.{txt,json}``. It follows the same
        capability-aware ``None`` / guaranteed ``True`` / disabled ``False``
        contract as ``write_molden_file``.
    trexio, trexio_backend
        Also export the converged wavefunction as a TREXIO file
        (Posenitskiy et al. 2023; optional ``[trexio]`` extra).
        ``trexio=True`` writes ``{output}.trexio.h5`` with the HDF5 back
        end, or ``{output}.trexio`` (a directory of per-group text files)
        with ``trexio_backend="text"``; a path writes there instead.
        Written groups: metadata, nucleus, electron, basis, ao, mo,
        ao_1e_int, state. Requires a route that exposes a Gaussian AO
        wavefunction (fails before the calculation otherwise) and an
        all-electron run (an ECP run is refused because the ``ecp`` group
        is not written). See :doc:`/user_guide/trexio`.
    fmax, max_opt_steps
        Optimizer tolerance (eV/Å) and iteration limit. Ignored unless
        ``optimize=True``.
    memory_override
        If ``False`` (default), the driver estimates peak memory and
        aborts with :class:`InsufficientMemoryError` when the estimate
        exceeds the machine's available RAM. Set to ``True`` to
        proceed anyway -- at the risk of swap-thrashing or a system
        freeze.
    memory_budget_bytes
        Requested peak-memory budget in bytes for bounded correlated-method
        kernels. The effective value is capped by scheduler, cgroup, and host
        availability. ``None`` (default) uses that detected allocation.
        ``memory_override`` may bypass preflight admission, but it never
        disables an explicit execution budget.
    num_threads
        If set, pin the OpenMP thread count for the duration of the
        calculation. ``None`` (default) leaves the current setting in
        place -- which is usually "all cores" unless the environment
        variable ``OMP_NUM_THREADS`` is set or
        :func:`vibeqc.set_num_threads` was called earlier. The actual
        thread count used is recorded in the output log for
        reproducibility.
    dispersion
        Post-SCF D3(BJ) or D4 dispersion correction. Accepts:

        * ``None`` (default) -- no dispersion.
        * ``True`` or ``"d3bj"`` -- use D3-BJ params for the current DFT
          functional.
        * ``"d4"`` -- use D4 parameters for the current functional through
          the optional ``dftd4`` backend.
        * A functional name (``"pbe"``, ``"b3lyp"``, ...) -- use its D3-BJ
          params (useful for ``method="rhf"`` + ``"hf"`` dispersion, or
          for overriding the SCF functional in the damping lookup).
        * A :class:`D3BJParams` instance -- used directly.

        The energy correction is written to the ``.out`` file, added to
        the returned object as ``e_dispersion`` / ``energy_total`` (the
        raw SCF ``.energy`` is preserved untouched), and, when
        ``optimize=True``, added to the forces the optimizer sees.
        Routes through :func:`vibeqc.compute_d3bj` with ``backend="auto"``
        -- the reference ``dftd3`` backend is used when installed,
        otherwise the D1a framework stub. See
        :mod:`vibeqc.dispersion` for details.
    solvent
        v0.9.0 CPCM / COSMO implicit solvation. Accepts:

        * ``None`` (default) -- gas-phase SCF.
        * A preset name (``"water"``, ``"dmso"``, ``"acetonitrile"``,
          ``"chloroform"``, ``"benzene"``, ...) -- looks up the static
          dielectric e from :data:`vibeqc.SOLVENT_PRESETS`.
        * A numeric e (e.g. ``78.39``) -- custom dielectric.
        * A dict (``{"epsilon": 25.0, "variant": "cosmo", ...}``) or a
          :class:`vibeqc.SolventModel` instance for full control over
          cavity construction (Bondi radii, Lebedev order, switching
          width, max macro-iterations).

        Routes through :func:`vibeqc.run_cpcm_scf`; macro-iterates the
        apparent surface charge against the SCF density until
        ΔE_solv < 1e-6 Ha (typically 3-5 outer cycles). The total
        energy is the in-solvent value; the gas-phase reference is
        retained on ``result.solvent_result.e_gas``. See
        :doc:`/user_guide/solvation` for the full theory and the
        cavity / preset table.
    density_fit
        Use density fitting (RI-J) for the Coulomb build and, together
        with ``cosx=True``, the chain-of-spheres (COSX) exchange build
        (RIJCOSX).  When ``True``, the JK auxiliary basis is resolved
        from ``aux_basis`` or autodetected from the orbital basis name
        (e.g. ``"def2-svp"`` → ``"def2-svp-jk"``).  Reduces SCF wall
        time 10–50× for larger basis sets and cuts CPCM peak memory by
        avoiding the in-core four-index ERI tensor path.

        Default ``None`` -- *not specified*.  The SCF then runs the
        four-index path (as the historical ``False`` default did) and
        every post-SCF method keeps its own documented default, which
        for coupled cluster is **density fitting on** (see
        :doc:`/user_guide/ccsd`).

        An explicit ``True`` or ``False`` is a request about the whole
        job, not only the SCF: it selects the integral route for the
        coupled-cluster step too, so ``run_job(method="ccsd(t)",
        density_fit=False)`` runs conventional CCSD(T).  This is the
        only way to reach the canonical route from a payload, which has
        no ``ccsd_options=`` surface.  Passing both this kwarg and a
        ``ccsd_options=`` whose ``density_fit`` disagrees is an error
        rather than a silently-picked winner.
    aux_basis
        Auxiliary basis name for density fitting.  For the SCF layer
        this is the JK auxiliary (e.g. ``"def2-tzvp-jk"``); when empty
        and ``density_fit=True``, the matching per-zeta JKfit basis is
        autodetected from the orbital basis name.  Ignored by the SCF
        when ``density_fit=False``.

        The same kwarg also names the correlation auxiliary for every
        post-SCF route that uses one: canonical ``mp2``/``ump2``,
        ``dlpno-mp2``, ``dlpno-ccsd``, and the
        ``ccsd``/``lccsd``/... family when it runs density fitted. There it
        means the RI-fit basis (e.g.
        ``"cc-pvdz-ri"``), not the JKfit; when empty, those routes
        autodetect from the orbital basis or read their own options
        object.  An explicit value here and one on the route's options
        object must agree -- a contradiction raises rather than
        silently picking one.
    cosx
        Enable chain-of-spheres exchange (COSX) together with
        ``density_fit=True`` (RIJCOSX).  Semi-numerical exchange build
        on a quadrature grid; faster than RI-JK for large basis sets
        at the cost of a few μHa grid noise.  Default ``False``.
    grid_level
        DFT integration-grid resolution for RKS / UKS / ROKS (ignored for HF,
        semiempirical, and post-SCF methods).  Accepts a string preset:

        * ``"orca-defgrid3"`` (DEFAULT) -- Lebedev-302 with NWChem
          pruning + Stratmann partition, matches ORCA's DefGrid3 to
          < 0.01 mHa for first-row systems. 75 radial shells.
        * ``"fine"`` -- Lebedev-590 order-35 with 99 radial shells.
          Grid-converged benchmarks; ~20× the point count of the
          default.
        * ``"coarse"`` -- Lebedev-194 order-23 with 50 radial shells.
          Rapid screening and geometry pre-optimization; ~0.5 mHa
          energy error relative to the default.
        * ``"legacy"`` -- v0.15 default: product Gauss-Legendre
          17×36, no pruning, Becke partition.  Backward-compatible
          with published v0.15 results.
        * ``"skala"`` -- vibe-qc's pinned PySCF-2.14 level-3 atomic-grid
          parity profile for Microsoft SKALA-1.1.

        The preset is applied whenever the method's RKS / UKS / ROKS options
        carry an untouched grid: no options object, an options object whose
        ``grid`` was never customised (``RKSOptions()`` passed to set
        ``max_iter`` still runs on the documented grid, GitLab #663), or a
        ``ROKSOptions`` whose ``grid`` is ``None``.  A grid the caller
        customised in any field wins and ``grid_level`` leaves it alone.
        Pass ``grid_level="legacy"`` to reproduce v0.15 energies exactly;
        that is also the supported way to request the C++ construction
        defaults, which are indistinguishable from an untouched grid.  The
        preset is written into the options object you pass, so the object
        records the grid that ran; reuse it in a later call and that grid
        counts as chosen, i.e. a different ``grid_level`` there is ignored
        (unless the recorded preset was ``"legacy"``, which is
        indistinguishable from an untouched grid and is replaced).  Build a
        fresh options object per call, or set the grid yourself, when you
        need different grids from one script.
    record_hostname
        If ``False``, the per-job ``{output}.system`` manifest writes
        ``hostname = "<redacted>"`` instead of the live hostname. The
        ``VIBEQC_NO_HOSTNAME=1`` environment variable does the same
        thing globally. Use it for public examples, paper artifacts,
        and shared reproductions so machine names do not leak. Other
        manifest fields (CPU model, OS, memory, library versions) are
        not redacted; the redaction is scoped to the hostname only.
    rhf_options / uhf_options / rks_options / uks_options
        Optional override for the respective SCF options struct.
    read_from
        Prior result object, .qvf path, or .molden path (including ORCA's
        .molden.input suffix) for
        ``initial_guess="read"``; rejected unless the resolved guess is READ.
    fragments
        Fragment partition for ``initial_guess="fragmo"``; same format as the
        direct SCF wrappers accept.
    casci_options
        CASCI knobs for ``method="casci"`` (the active space itself comes
        from ``active_space=``). ``CASCIOptions(nroots=N)`` requests N CI
        roots: ``result.energy`` stays the ground root, per-root energies
        land in ``result.root_energies`` (per-root <S^2> in
        ``result.root_s2``) and the .out solver block prints the root
        table. Ignored by every other method -- the SA-CASSCF root count
        is ``casscf_options.nroots``, the MS-CASPT2 model-space size
        ``caspt2_options.nroots``.
    progress
        Live progress logger for long-running jobs. Default behavior
        is **ON** -- the job emits a banner, per-stage milestones, and
        a final summary to stdout (line-flushed) so the canonical
        ``nohup python LiH.py > LiH.log 2>&1 &`` + ``tail -f LiH.log``
        workflow shows progress in real time. The ``.out`` file is
        also line-buffered so ``tail -f output-LiH.out`` works without
        any extra setup.

        Pass ``progress=False`` to silence stdout (the ``.out`` file
        is still written normally -- only the live mirror is
        suppressed). Pass a :class:`vibeqc.ProgressLogger` instance
        for fully custom routing (tee to a persistent file, mute,
        thread one logger through nested calls).

        Set ``VIBEQC_LIVE_LOGGING=0`` in the environment to disable
        live progress globally -- useful for batch scripts that don't
        want to edit every input file. The env var only takes effect
        when ``progress`` is left at its default (``None``); explicit
        ``progress=True`` / ``progress=False`` / a ``ProgressLogger``
        instance always wins, so a debugging session can re-enable
        progress for one shell.

        Per-iteration progress for periodic SCFs (which run in
        Python) is streamed live through this same logger via the
        lower-level ``run_*_periodic_*`` entry points; molecular SCFs
        run in C++ and only emit a pre-SCF banner + post-SCF summary
        live, with the per-iteration trace landing in the ``.out``
        when the SCF returns.
    verbose
        Integer verbosity level (PySCF convention, 0..9, default
        4). Each level is a strict superset of the one below, so
        bumping ``verbose`` only adds output:

        * ``0`` -- silent (nothing live; ``.out`` is still written)
        * ``1`` -- banner + warnings + final SCF summary only
        * ``2`` -- add per-stage milestones + ``info()`` lines
        * ``3`` -- add per-stage timing on stage exit
        * ``4`` -- add per-iteration SCF rows (DEFAULT)
        * ``5`` -- add inline RSS-memory snapshots
        * ``6+`` -- phase-level wall-clock breakdown live (overlaps
          the post-mortem ``.perf`` log on purpose)

        Pass ``verbose=None`` (the default) to read the
        ``VIBEQC_VERBOSE`` env var; if unset, falls back to 4.
        Ignored when ``progress`` is a :class:`ProgressLogger`
        instance -- that logger's own level wins.
    use_logging
        If ``True``, route progress through
        ``logging.getLogger("vibeqc.run_job")`` instead of bare
        ``stdout`` writes. Banner / stage milestones land at
        ``INFO``; per-iteration SCF rows at ``DEBUG``; warnings at
        ``WARNING``. Composes naturally with stdlib handlers
        (``RotatingFileHandler``, syslog, ``dictConfig``)::

            import logging
            logging.basicConfig(level=logging.INFO)
            vq.run_job(..., use_logging=True)

        ``progress=False`` still wins as a hard kill switch -- the
        verbose-level gate runs *before* the logging call, so a
        silent run stays silent regardless of the active logging
        config. Ignored when ``progress`` is a pre-built
        :class:`ProgressLogger` instance.
    perf_log
        Optional path (or ``True`` to use ``{output}.perf``) to write
        a post-mortem performance / debug breakdown -- phase-level
        wall + CPU times, memory snapshots, parallelism flags. The
        live ``progress=`` log shows progress *during* the run; the
        perf log shows where the time went *afterwards*. Off by
        default. Pass an explicit path, ``True`` to emit alongside
        ``{output}.out``, or set the ``VIBEQC_PERFLOG=path`` env
        var (which wins when ``perf_log`` is left at ``None``).
        See :mod:`vibeqc.perf` for the full instrumentation surface.
    structured_log
        Optional NDJSON (one-JSON-record-per-line) log capturing
        every SCF transition -- banner, job_start, memory_estimate,
        per-iter rows, scf_converged, properties, job_end. Off by
        default. Pass ``True`` to emit ``{output}.scf.jsonl``,
        a path-like to write there explicitly, or set the
        ``VIBEQC_STRUCTURED_LOG=path`` env var (which wins when
        ``structured_log`` is left at ``False``). The format is
        stable: events are append-only, fields are never renamed
        or removed. See :mod:`vibeqc.structured_log` for the full
        event catalog.
    crash_dump
        Write a snapshot to ``{output}.dump`` (TOML) plus binary
        attachments (``.dump.density.npy``, ``.dump.fock.npy``,
        ``.dump.mo.npy``) when the SCF fails ungracefully -- raised
        exception (NaN, linear dependence, memory error). Default
        ``True``: post-mortem reproducibility costs nothing on
        success and saves a re-run on failure. Pass ``False`` (or
        set ``VIBEQC_NO_CRASH_DUMP=1`` in the environment) to
        disable. The dump is written alongside ``{output}.out`` in
        the standard ``output.*`` family -- re-attach the ``.dump``
        + ``.dump.density.npy`` to a bug report and the maintainer
        can reconstruct the failing state via
        :func:`vibeqc.load_dump`. See :mod:`vibeqc.crash_dump` for
        the dump format. The exception is always re-raised after
        the dump is written; ``crash_dump=True`` does *not* swallow
        failures.
    tddft
        When True, compute TD-DFT vertical excitation energies via
        the Tamm-Dancoff approximation (TDA, default) or the full
        Casida linear-response formalism (tddft_type="casida").
        Requires a converged SCF that produces MO coefficients
        (RHF/RKS/UHF/UKS).
    tddft_n_states
        Number of excited states to compute when ``tddft=True``.
        Default 5.
    tddft_type
        ``"tda"`` (Tamm-Dancoff, default) or ``"casida"`` (full
        Casida).  Casida includes the B matrix and is the complete
        linear-response TD-DFT; TDA is faster, Hermitian, and
        usually within 0.1-0.3 eV of Casida for valence states.
    tddft_gradient
        When True, compute the finite-difference nuclear gradient of the
        lowest excited state on the closed-shell RHF + CIS (TDA)
        surface, the one energy function wired into
        ``vibeqc.excited_gradient`` (state-tracked central differences,
        step 1e-3 Å), and write it to the .out labelled with that
        surface. Any other run (an RKS or open-shell reference, or
        ``tddft_type="casida"``) is refused before the calculation with
        the missing capability named (issue #570). Cost: ~6N additional
        SCF + CIS evaluations. Opt-in; not computed by default.
    tddft_n_states
        Number of excited states to compute when ``tddft=True``.
        Default 5.
    nto
        When True (and ``tddft=True``), compute Natural Transition
        Orbitals for each TD-DFT excited state and embed paired
        hole/particle NTOs into the QVF archive as
        ``wavefunction.gto`` sections with ``orbital_kind="natural"``.
        Requires ``output_qvf=True``.
    hessian
        When True, compute harmonic vibrational frequencies via
        finite-difference Hessian. Default False. Cost: ~6N SCF evals.
        Results printed to .out and embedded in QVF for vibe-view.

    Returns
    -------
    The SCF result object (RHFResult / UHFResult / RKSResult / UKSResult).
    """
    enforce_runtime_pin_from_env()

    # GitLab #663: whether grid_level applies is decided from the options'
    # GRID (ks_options_need_grid_default), never from whether an object was
    # passed. A caller who hands in RKSOptions() to set max_iter, and every
    # object a feature path materialises internally, receives the documented
    # grid; only a grid the caller customised is authoritative. Nothing
    # between here and the grid block below writes a grid field.
    _rks_grid_untouched = ks_options_need_grid_default(rks_options)
    _uks_grid_untouched = ks_options_need_grid_default(uks_options)
    _roks_grid_untouched = ks_options_need_grid_default(roks_options)

    # Fail fast on the wrong-runner call: a PeriodicSystem (crystal) handed
    # to the molecular driver. run_job computes isolated molecules; a
    # crystal needs the periodic driver. Detect by the lattice attribute a
    # PeriodicSystem carries and a Molecule does not (duck-typed so a bare
    # import cycle is avoided).
    if not isinstance(molecule, Molecule) and hasattr(molecule, "lattice"):
        raise TypeError(
            "run_job was given a periodic system (it has lattice vectors), "
            "but run_job computes isolated molecules. For a crystal use "
            "vibeqc.run_periodic_job(system, basis, ...) instead."
        )
    if not isinstance(molecule, Molecule):
        raise TypeError(
            "run_job: 'molecule' must be a vibeqc.Molecule; got "
            f"{type(molecule).__name__}."
        )

    # Opt-in on-demand basis rendering. Off unless the caller asks, either
    # here or through $VIBEQC_FETCH_BSE, because a run whose basis depends on
    # whether an optional 334 MB distribution happens to be installed is not
    # one you want by default.
    #
    # A bundled name is a complete no-op: no lookup, no cache entry, and
    # resolution is left exactly as it was. Only a name that does not resolve
    # at all triggers a render, and a basis that resolves but lacks one of
    # the molecule's elements is deliberately NOT a trigger -- that is the
    # coverage guard's refusal, and filling the gap from BSE would put two
    # sources under one basis name with no reviewed provenance for the join.
    from .basis_fetch import ensure_bases_available, fetch_requested_by_env

    _bse_rendered: tuple = ()
    _want_bse = (
        fetch_requested_by_env() if fetch_from_bse is None else bool(fetch_from_bse)
    )
    if _want_bse:
        # Announced through ``warnings``, not ``output.warn``: the basis has
        # to be resolvable before anything constructs a BasisSet, which is
        # long before the job's output channel exists, and ``record`` no-ops
        # while no channel is installed -- the notice would be silently
        # dropped. This is the same constraint and the same answer as
        # ``vibeqc._warn_if_basis_overlay_is_stale``, the other place that
        # has to say "your basis data is not what you assume" before there
        # is anywhere to say it. The durable record is the provenance file
        # written beside the rendered basis.
        import warnings as _warnings

        _bse_rendered = ensure_bases_available(
            (basis, aux_basis),
            elements=sorted({int(_a.Z) for _a in molecule.atoms}),
        )
        for _rendered in _bse_rendered:
            _warnings.warn(
                f"vibe-qc basis library: {_rendered.name!r} is not bundled. "
                "It was rendered from the Basis Set Exchange catalogue "
                f"{_rendered.bse_version} (BSE name {_rendered.bse_name!r}) "
                f"into {_rendered.g94_path}.\n"
                "Cite the originating publications from that file's header, "
                "and record the catalogue version with any result that uses "
                f"it; the provenance is in {_rendered.provenance_path}.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Periodic-only features requested on the molecular runner: fail with a
    # pointer instead of a bare TypeError. A molecule has no lattice, so
    # there is no Brillouin zone to sample -- k-points, band structures,
    # DOS meshes, PW cutoffs, and periodic J/K builders are all
    # run_periodic_job territory.
    _pbc_only = {
        "kpoints": kpoints,
        "dos_kmesh": dos_kmesh,
        "jk_method": jk_method,
        "bz_integration": bz_integration,
        "cutoff_ha": cutoff_ha,
    }
    _pbc_requested = [k for k, v in _pbc_only.items() if v is not None]
    if _pbc_requested:
        raise ValueError(
            f"run_job: {', '.join(_pbc_requested)} requested, but these are "
            f"periodic-boundary-condition (k-point) features and this is the "
            f"molecular runner -- a molecule has no lattice or Brillouin "
            f"zone, so band structures / DOS / k-meshes are not defined for "
            f"it. Build a PeriodicSystem and use vibeqc.run_periodic_job("
            f"...) for crystals, slabs, and wires."
        )

    output_stem = Path(os.fspath(output))
    out_path = stem_sibling(output_stem, ".out")
    molden_path = stem_sibling(output_stem, ".molden")
    traj_path = stem_sibling(output_stem, ".traj")

    # TREXIO artefact (#573): resolve the request to a path (or None) once,
    # so the plan, the .out notice, the citation route and the final write
    # all agree on the same target.
    _trexio_backend = str(trexio_backend).strip().lower()
    if _trexio_backend not in ("hdf5", "text"):
        raise ValueError(
            f"run_job: trexio_backend must be 'hdf5' or 'text'; got "
            f"{trexio_backend!r}."
        )
    if isinstance(trexio, bool) or trexio is None:
        _trexio_path = (
            stem_sibling(
                output_stem,
                ".trexio.h5" if _trexio_backend == "hdf5" else ".trexio",
            )
            if trexio
            else None
        )
    elif isinstance(trexio, (str, os.PathLike)):
        if not os.fspath(trexio):
            raise ValueError("run_job: trexio='' is not a valid target path.")
        _trexio_path = Path(os.fspath(trexio))
    else:
        raise TypeError(
            "run_job: trexio must be True, False, or a target path; got "
            f"{type(trexio).__name__}."
        )
    _trexio_requested = _trexio_path is not None

    # Resolve perf_log target. ``True`` -> emit to {output}.perf;
    # str/Path -> use that path verbatim; None -> defer to
    # VIBEQC_PERFLOG env var inside perf_log() (or no-op if unset).
    if perf_log is True:
        _perf_target: Optional[Path] = stem_sibling(output_stem, ".perf")
    elif perf_log is False or perf_log is None:
        _perf_target = None
    else:
        _perf_target = Path(os.fspath(perf_log))

    # Resolve structured_log target. Default OFF (False) -- only emit
    # when the caller opts in. Resolution mirrors perf_log:
    #   True             -> {output}.scf.jsonl
    #   str/PathLike     -> use verbatim
    #   False / None     -> defer to VIBEQC_STRUCTURED_LOG env var (or
    #                       leave disabled if the env var is unset)
    #   $VQ_WORKDIR set  -> auto-enable to {output}.scf.jsonl even when
    #                       structured_log=False (v0.24: always-on
    #                       machine-readable progress for vq)
    if structured_log is True:
        _structured_target: Optional[Path] = stem_sibling(output_stem, ".scf.jsonl")
    elif structured_log is False or structured_log is None:
        _structured_target = None
        # Auto-enable when running under vq — the daemon always sets
        # $VQ_WORKDIR.  Machine-readable progress lets vq status /
        # monitors show iteration + energy without parsing .out.
        if os.environ.get("VQ_WORKDIR"):
            _structured_target = stem_sibling(output_stem, ".scf.jsonl")
    else:
        _structured_target = Path(os.fspath(structured_log))

    # Resolve crash_dump target. Default ON: dump on failure unless the
    # caller explicitly opts out (or VIBEQC_NO_CRASH_DUMP=1). The
    # post-mortem snapshot is cheap on success (zero bytes written),
    # and saves a re-run on failure. Resolution:
    #   True             -> {output}.dump (default)
    #   str/PathLike     -> use verbatim as the stem
    #   False            -> disabled
    #   None             -> also default-on (treated as True)
    _crash_off_env = os.environ.get(
        "VIBEQC_NO_CRASH_DUMP",
        "",
    ).strip().lower() in ("1", "true", "yes", "on")
    if crash_dump is False or _crash_off_env:
        _crash_stem: Optional[Path] = None
    elif crash_dump is True or crash_dump is None:
        _crash_stem = output_stem
    else:
        _crash_stem = Path(os.fspath(crash_dump))

    # Default-ON progress: when the caller didn't pass a value,
    # check the env-var opt-out (VIBEQC_LIVE_LOGGING=0) and otherwise
    # turn it on. This lets every backgrounded ``python input.py >
    # log 2>&1 &`` show progress without the user having to learn
    # a new kwarg -- answering the "is my job stuck or actually
    # running?" question by default. Explicit progress= (True/False/
    # logger) bypasses the env var entirely.
    if progress is None:
        env = os.environ.get("VIBEQC_LIVE_LOGGING", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            progress = False
        else:
            progress = True

    # Verbose level resolution (v0.5.3). When the caller leaves
    # ``verbose=None`` we read VIBEQC_VERBOSE; otherwise the
    # explicit argument wins. Falls back to the package default
    # (level 4 -- banner + stages + per-iter SCF) when neither
    # is set. The env var is a parsing best-effort: a junk value
    # silently falls back to the default rather than raising,
    # since a typo in $VIBEQC_VERBOSE shouldn't kill someone's
    # overnight run.
    if verbose is None:
        env_verbose = os.environ.get("VIBEQC_VERBOSE", "").strip()
        if env_verbose:
            try:
                verbose_level = int(env_verbose)
            except ValueError:
                verbose_level = None  # type: ignore[assignment]
            else:
                verbose_level = max(0, verbose_level)
        else:
            verbose_level = None  # type: ignore[assignment]
    else:
        verbose_level = max(0, int(verbose))

    plog = resolve_progress(
        progress,
        verbose=verbose_level,
        use_logging=use_logging,
    )

    # Composite 3c keyword resolution (e.g. ``method="hf-3c"``). For
    # non-composite ``method`` values this is a pass-through. Composite
    # values get their (basis, functional, dispersion) tuple inferred
    # from the :mod:`vibeqc.composites` registry; raises
    # :class:`CompositeUnavailable` for recipes whose prerequisites
    # have not yet landed.
    method = _normalise_method_alias(method)
    method, basis, functional, dispersion, composite_recipe = _apply_composite(
        method,
        basis=basis,
        functional=functional,
        dispersion=dispersion,
        molecule=molecule,
    )
    if composite_recipe is not None and citype is not None:
        raise ValueError(
            "run_job: citype= cannot be combined with composite method "
            f"{composite_recipe.name!r}; choose an explicit basis/method pair."
        )
    method = _normalise_method_alias(_apply_citype_selector(method, citype))
    if nddo and method != "msindo":
        raise ValueError(
            f"run_job: nddo=True is only valid with method='msindo'; got "
            f"method={method!r}."
        )
    if ccm_options is not None and method != "ccm":
        raise ValueError(
            f"run_job: ccm_options= is only valid with method='seccm' "
            f"(legacy 'ccm'); got "
            f"method={method!r}."
        )
    # Semiempirical methods don't use a Gaussian basis set -- accept
    # any basis value (including None / empty).
    if method in SEMIEMPIRICAL_METHODS or method in _MLIP_METHODS:
        if not basis:
            basis = ""  # placeholder; semiempirical / MLIP code ignores it
    elif not basis:
        raise ValueError(
            "run_job: basis is required. Pass basis='def2-svp' (or "
            "another supported basis), or use a composite keyword "
            "(method='hf-3c', ...) which carries its own basis. "
            "Composite catalogue: "
            f"{[r.name for r in __import__('vibeqc').list_composites()]}"
        )

    resolved_method = _select_method(
        method, molecule, functional, ccsd_reference, mp2_reference
    )
    _refuse_molecular_ecp_derivative_request(
        method,
        resolved_method,
        molecule,
        basis,
        optimize=bool(optimize),
        geom_opt=geom_opt,
        hessian=bool(hessian),
        solvent=solvent,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
        rohf_options=rohf_options,
        roks_options=roks_options,
    )
    _refuse_molecular_ecp_correlated_route(
        method,
        resolved_method,
        molecule,
        basis,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
        rohf_options=rohf_options,
        roks_options=roks_options,
    )
    _uses_skala = False
    _uses_external_xc = False
    # PM7/UPM7 parameter records remain queryable for future implementation,
    # but the published feathered electrostatics are not executable yet.
    # Gate before sidecar/output planning so a rejected request leaves no
    # manifest or other job artefact behind.
    if resolved_method in ("pm7", "upm7"):
        SemiempiricalRoutePlan.from_request(
            resolved_method,
            boundary="molecule",
            charge=int(molecule.charge),
            multiplicity=int(molecule.multiplicity),
        )
    if resolved_method in ("rks", "uks", "roks"):
        _ks_options = {
            "rks": rks_options,
            "uks": uks_options,
            "roks": roks_options,
        }[resolved_method]
        _functional_name = (
            functional or getattr(_ks_options, "functional", "") or "lda"
        )
        # From this point onward one effective name must drive execution,
        # output metadata, dispersion damping, restart fingerprints, and
        # citations.  In particular, an options-only request such as
        # RKSOptions(functional="skala-1.1") must not execute SKALA while the
        # job is recorded as functional=None.
        functional = str(_functional_name)
        _functional_obj = Functional(
            functional,
            2 if resolved_method in ("uks", "roks") else 1,
        )
        _uses_external_xc = bool(
            getattr(_functional_obj, "is_external", False)
        )
        _uses_skala = functional.strip().lower() in {
            "skala",
            "skala-1.1",
            "skala-1.1-rev1",
        }
        if _uses_external_xc and (optimize or geom_opt is not None):
            raise NotImplementedError(
                "Geometry optimization with a full-grid external XC "
                "functional is unavailable: the provider supplies "
                "self-consistent energies and XC matrices but the host "
                "contract does not yet carry grid-coordinate, partition-"
                "weight, and coarse-center derivatives needed for forces. "
                "Use whole-SCF finite differences explicitly."
            )
        if _uses_external_xc and (hessian or tddft or tddft_gradient):
            raise NotImplementedError(
                "Hessian and TDDFT/response calculations are not implemented "
                "for full-grid external XC functionals because the provider "
                "contract does not expose a pointwise second functional "
                "derivative."
            )
        _pointwise_second_order = []
        if resolved_method in ("rks", "uks"):
            for _threshold in ("newton_threshold", "trah_threshold"):
                try:
                    if (
                        float(getattr(_ks_options, _threshold, 0.0) or 0.0)
                        > 0.0
                    ):
                        _pointwise_second_order.append(_threshold)
                except (TypeError, ValueError):
                    # The option validator owns malformed diagnostics.
                    pass
        if _uses_external_xc and _pointwise_second_order:
            raise NotImplementedError(
                "Molecular KS Newton/TRAH acceleration is not implemented "
                "for full-grid external XC functionals because those paths "
                "require a pointwise f_xc kernel; disable "
                + ", ".join(_pointwise_second_order)
                + "."
            )
        _refuse_plain_double_hybrid(
            _functional_obj,
            route=resolved_method.upper(),
        )

    if tddft_gradient:
        # Issue #570: the FD excited-state gradient block below is built from
        # the ONE energy function run_job wires into vibeqc.excited_gradient,
        # the closed-shell RHF reference with CIS (TDA) excitation energies
        # (make_hf_cis_energy_fn).  Before this gate an RKS/TDA, RKS/Casida
        # or RHF/Casida run was handed that HF-CIS S1 gradient under a header
        # that named no surface (H2/STO-3G, one root: all three printed the
        # RHF/TDA value 0.02279028 Ha/bohr).  Refuse before any calculation
        # starts, naming the energy function that is missing, rather than
        # differentiate a surface the run did not compute.
        _tdg_missing = None
        if resolved_method != "rhf":
            _tdg_missing = (
                f"an excited-state energy function for a {resolved_method.upper()} "
                f"reference"
            )
        elif tddft_type != "tda":
            _tdg_missing = (
                "an excited-state energy function for Casida (full linear-"
                "response) excitation energies"
            )
        elif int(molecule.multiplicity) != 1:
            _tdg_missing = (
                "an excited-state energy function for an open-shell reference"
            )
        if _tdg_missing is not None:
            raise NotImplementedError(
                "tddft_gradient=True: the finite-difference excited-state "
                "gradient is wired for a closed-shell RHF reference with CIS "
                f"(TDA) excitation energies only; this run needs {_tdg_missing}, "
                "which is not implemented. Refusing rather than reporting the "
                "HF-CIS S1 gradient for a different surface (issue #570)."
            )
        if (
            basis_sidecar_has_ecp_operator(molecule, basis)
            or molecular_options_request_ecp_operator(rhf_options)
        ):
            raise NotImplementedError(
                "tddft_gradient=True: the closed-shell HF-CIS finite-"
                "difference energy function does not yet preserve a "
                "molecular ECP Hamiltonian across displaced geometries. "
                "Refusing before SCF or output creation rather than "
                "reporting an all-electron excited-state gradient."
            )
    _wavefunction_sidecars_supported = (
        resolved_method in _WAVEFUNCTION_SIDECAR_METHODS
    )
    _population_sidecar_supported = (
        _wavefunction_sidecars_supported
        or resolved_method in _NATIVE_MULLIKEN_SIDECAR_METHODS
    )
    if resolved_method in SEMIEMPIRICAL_METHODS:
        _sidecar_unavailable_reason = (
            "the basis-free semiempirical result does not expose a Gaussian "
            "AO wavefunction for these writers."
        )
    elif resolved_method in _MLIP_METHODS:
        _sidecar_unavailable_reason = (
            "MACE is an interatomic potential and produces no electronic "
            "wavefunction or AO density."
        )
    else:
        _sidecar_unavailable_reason = (
            "the returned solver result does not expose format-ready "
            "mean-field orbitals and AO density."
        )
    write_molden_file = _resolve_sidecar_request(
        write_molden_file,
        supported=_wavefunction_sidecars_supported,
        option="write_molden_file",
        method=resolved_method,
        caller="run_job",
        unavailable_reason=_sidecar_unavailable_reason,
    )
    if _trexio_requested and not _wavefunction_sidecars_supported:
        # A TREXIO request is a guarantee, like write_molden_file=True:
        # refuse before the calculation rather than discover at the end
        # that the route exposes no Gaussian AO wavefunction to write.
        raise NotImplementedError(
            f"run_job: trexio={trexio!r} is not supported for "
            f"method={resolved_method!r}: {_sidecar_unavailable_reason} "
            "Pass trexio=False."
        )
    write_population_file = _resolve_sidecar_request(
        write_population_file,
        supported=_population_sidecar_supported,
        option="write_population_file",
        method=resolved_method,
        caller="run_job",
        unavailable_reason=(
            "the method exposes neither a Gaussian AO density nor validated "
            "method-native atomic populations for this writer."
            if resolved_method in SEMIEMPIRICAL_METHODS
            else _sidecar_unavailable_reason
        ),
    )

    # Fail-fast solvent gate (the dispatcher gate in _run_single_point is
    # the backstop). Two cases only run_job can catch cleanly:
    #  * post-SCF methods (mp2 / ccsd / dlpno / ovgf / coupled-pair
    #    families) resolve to a mean-field reference, so the dispatcher
    #    would run a CPCM reference SCF -- but the correlation step then
    #    builds its reported total from the bare reference energy, not the
    #    in-solvent total, and the output claims an "in-solvent" result it
    #    does not compute. Refuse until a validated post-SCF-in-solvent
    #    composition exists.
    #  * unsupported methods combined with optimize=True would burn the
    #    whole gas-phase optimization before the final single point raises.
    # rohf/roks pass through to their dedicated NotImplementedError gates
    # (handovers/HANDOVER_ROHF.md).
    if solvent is not None:
        _solvent_capable = ("rhf", "uhf", "rks", "uks", "msindo")
        if resolved_method in _solvent_capable and method not in (
            *_solvent_capable,
            "auto",
        ):
            raise ValueError(
                f"Implicit solvation is not supported for method={method!r}: "
                "CPCM composes with the mean-field SCF reference only; the "
                f"post-SCF {method} treatment in solvent is not implemented. "
                "Run the calculation in gas phase, or use rhf/uhf/rks/uks "
                "(CPCM) or msindo (COSMO) with solvent."
            )
        if resolved_method not in (*_solvent_capable, "rohf", "roks"):
            _auto_hint = (
                f" (method='auto' selected {resolved_method!r} for this system;"
                " pass a mean-field method explicitly)"
                if method == "auto"
                else ""
            )
            raise ValueError(
                "Implicit solvation is not supported for "
                f"method={resolved_method!r}{_auto_hint}: CPCM "
                "(run_cpcm_scf) composes with rhf, uhf, rks, and uks only, "
                "and COSMO with method='msindo'. Run the calculation in "
                "gas phase, or use a solvent-capable method with solvent."
            )

    _effective_method = method
    _ccsd_compute_triples: Optional[bool] = None
    if triples is not None and method not in (
        "ccsd",
        "ccsd(t)",
        "bccd",
        "bccd(t)",
        "qcisd",
        "qcisd(t)",
    ):
        raise ValueError(
            "run_job: triples= is only valid with method='ccsd', "
            "method='ccsd(t)', 'bccd', 'bccd(t)', 'qcisd', or "
            "'qcisd(t)'. The other approximate/coupled-pair variants "
            "(cc2, ccd, lccd, "
            "lccsd, cepa(n)) do not define a (T) correction."
        )
    if ccsd_options is not None and method not in (
        "ccsd",
        "ccsd(t)",
        "bccd",
        "bccd(t)",
        "qcisd",
        "qcisd(t)",
        *_CC_VARIANT_METHODS,
    ):
        raise ValueError(
            "run_job: ccsd_options= is only valid with method='ccsd', "
            "'ccsd(t)', 'bccd', 'bccd(t)', 'qcisd', 'qcisd(t)', or a "
            "approximate coupled-cluster/coupled-pair/QCI variant "
            "(cc2, ccd, lccd, lccsd, "
            "cepa(0)..cepa(3), qcisd)."
        )
    # ---- Fail-early CC aux-basis validation (BUG 111) -------------------
    # CCSD(T) with density_fit=True (the default) requires an RI auxiliary
    # basis.  For orbital bases with no registered default (sto-ng, 3-21g,
    # 6-31g*, ...) this check must run *before* SCF to avoid wasting a
    # converged SCF on a calculation that cannot proceed.
    #
    # The post-SCF block repeats the same resolve_aux_basis call for
    # defence-in-depth; this early gate catches the failure at input-
    # validation time with a clear, actionable error message.
    if method in (
        "ccsd", "ccsd(t)", "bccd", "bccd(t)", "qcisd", "qcisd(t)",
        *_CC_VARIANT_METHODS,
    ):
        _df_for_cc = _resolve_cc_density_fit(density_fit, ccsd_options)
        if _df_for_cc:
            # An explicit run_job(aux_basis=...) kwarg counts here: it
            # reaches the CC route (BUG 113), so it also satisfies this
            # gate for orbital bases with no registered default RI aux.
            _aux = _resolve_route_aux_basis(
                aux_basis,
                (
                    getattr(ccsd_options, "aux_basis", "")
                    if ccsd_options is not None
                    else ""
                ),
                route="ccsd_options",
            )
            if not _aux:
                from .density_fitting import default_aux_basis_for

                try:
                    default_aux_basis_for(basis, kind="ri")
                except NotImplementedError as exc:
                    raise ValueError(
                        f"CCSD(T) with orbital basis {basis!r} requires an RI "
                        f"auxiliary basis, but no default auxiliary is "
                        f"registered for {basis!r}. STO-3G is a minimal basis "
                        f"set with no standard RI-J or RI-C auxiliary pairing."
                        f"\n\n"
                        f"Options:\n"
                        f"  1. Use a larger orbital basis that has a "
                        f"registered auxiliary (e.g., def2-SVP, cc-pVDZ)\n"
                        f"  2. Specify an explicit auxiliary basis: "
                        f"CCSDOptions(aux_basis='def2-svp-ri')\n"
                        f"  3. If you need {basis!r} explicitly, use a "
                        f"non-density-fitted CCSD route: "
                        f"CCSDOptions(density_fit=False)"
                    ) from exc
    if ccsdt_options is not None and method != "ccsdt":
        raise ValueError(
            "run_job: ccsdt_options= is only valid with method='ccsdt' "
            "or method='ci', citype='ccsdt'."
        )
    if cc3_options is not None and method != "cc3":
        raise ValueError(
            "run_job: cc3_options= is only valid with method='cc3' "
            "or method='ci', citype='cc3'."
        )
    if (
        method == "cc3"
        and active_space is not None
        and cc3_options is not None
        and cc3_options.n_frozen_core not in (None, 0)
    ):
        raise ValueError(
            "run_job: use either active_space= or "
            "CC3Options(n_frozen_core=...), not both."
        )
    if (
        method == "ccsdt"
        and active_space is not None
        and ccsdt_options is not None
        and ccsdt_options.n_frozen_core not in (None, 0)
    ):
        raise ValueError(
            "run_job: use either active_space= or "
            "CCSDTOptions(n_frozen_core=...), not both."
        )
    if (mp2_options is not None or ump2_options is not None) and method not in (
        "mp2",
        "scs-mp2",
        "sos-mp2",
    ):
        raise ValueError(
            "run_job: mp2_options=/ump2_options= are only valid with "
            "method='mp2', 'scs-mp2', or 'sos-mp2'."
        )
    if frozen_core is not None and method not in _FROZEN_CORE_ROUTE_METHODS:
        raise ValueError(
            "run_job: frozen_core= is only valid with MP2, CCSD/coupled-pair, "
            "or DLPNO-MP2/CCSD routes."
        )
    _route_frozen_options = (
        (ump2_options if molecule.multiplicity > 1 else mp2_options)
        if method in ("mp2", "scs-mp2", "sos-mp2")
        else (
            ccsd_options
            if method in _CC_VARIANT_METHODS
            or method in _BCCD_METHODS
            or method in ("ccsd", "ccsd(t)")
            else (
                dlpno_options
                if method == "dlpno-mp2"
                else (
                    dlpno_ccsd_options
                    if method in ("dlpno-ccsd", "dlpno-ccsd(t)")
                    else None
                )
            )
        )
    )
    if (
        frozen_core is not None
        and _options_have_explicit_frozen_core(_route_frozen_options)
    ):
        raise ValueError(
            "run_job: frozen_core= contradicts the explicit frozen-core "
            "count on the route options. Pass only one selector."
        )
    if frozen_core is not None:
        from .correlation_conventions import resolve_frozen_core_count

        _requested_frozen_core_count = resolve_frozen_core_count(
            molecule, frozen_core
        )
    else:
        _requested_frozen_core_count = None
    if dlpno_thresholds is not None and method not in (
        "dlpno-mp2",
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
    ):
        raise ValueError(
            "run_job: dlpno_thresholds= is only valid with DLPNO-MP2 or "
            "DLPNO-CCSD/(T)."
        )
    if dlpno_thresholds is not None and (
        dlpno_options is not None or dlpno_ccsd_options is not None
    ):
        raise ValueError(
            "run_job: dlpno_thresholds= cannot be combined with "
            "dlpno_options= or dlpno_ccsd_options=. Use "
            "vibeqc.dlpno.options_from_dlpno_thresholds(...) for a named "
            "preset plus explicit overrides."
        )
    if dlpno_thresholds is not None:
        # Resolve names while input validation is still side-effect free.
        # Waiting until the post-SCF route applies the preset would waste a
        # converged reference calculation before reporting a typo.
        from .dlpno import resolve_dlpno_thresholds

        _requested_dlpno_thresholds = resolve_dlpno_thresholds(
            dlpno_thresholds
        )
    else:
        _requested_dlpno_thresholds = None
    _ccsd_triples_variant = "(t)"
    if method in ("ccsd", "ccsd(t)", "bccd", "bccd(t)", "qcisd", "qcisd(t)"):
        from .cc import resolve_triples_variant

        _tv = (
            resolve_triples_variant(triples)
            if triples is not None
            else (
                "(t)"
                if method in ("ccsd(t)", "bccd(t)", "qcisd(t)")
                else "none"
            )
        )
        _ccsd_compute_triples = _tv != "none"
        if _tv != "none":
            _ccsd_triples_variant = _tv
        if method.startswith("bccd") and _tv in ("[t]", "a-ccsd(t)"):
            raise NotImplementedError(
                "triples='[t]' / CCSD+T(CCSD) and triples='A-CCSD(T)' "
                "are implemented for CCSD only. Use triples='(t)' for "
                "the standard BCCD(T) "
                "correction."
            )
        if method.startswith("qcisd") and _tv in ("[t]", "a-ccsd(t)"):
            raise NotImplementedError(
                "triples='[t]' / CCSD+T(CCSD) and triples='A-CCSD(T)' "
                "are implemented for CCSD only. Use triples='(t)' for "
                "the standard QCISD(T) "
                "correction."
            )
        if _tv in ("[t]", "a-ccsd(t)") and molecule.multiplicity > 1:
            raise NotImplementedError(
                "triples='[t]' and triples='A-CCSD(T)' are implemented in "
                "the closed-shell kernel only; the open-shell (T) route is "
                "the standard Raghavachari correction (triples='(t)')."
            )
        if method.startswith("qcisd"):
            _effective_method = {
                "none": "qcisd",
                "(t)": "qcisd(t)",
            }[_tv]
        elif method.startswith("bccd"):
            _effective_method = {
                "none": "bccd",
                "(t)": "bccd(t)",
            }[_tv]
        else:
            _effective_method = {
                "none": "ccsd",
                "(t)": "ccsd(t)",
                "[t]": "ccsd[t]",
                "a-ccsd(t)": "a-ccsd(t)",
            }[_tv]
    elif method in _CC_VARIANT_METHODS:
        _ccsd_compute_triples = False
    # method='dlpno-mp2' auto-routes an open-shell reference to the
    # UHF-based DLPNO-UMP2 path; the effective label drives the header,
    # output block and citation route.
    if method == "dlpno-mp2" and molecule.multiplicity > 1:
        _effective_method = "dlpno-ump2"
        uhf_options = _default_open_shell_dlpno_uhf_options(
            method,
            molecule,
            uhf_options,
        )
    # method='dlpno-ccsd'/'dlpno-ccsd(t)' auto-route an open-shell reference
    # to the UHF-based DLPNO-UCCSD(T) local solver; the effective label drives
    # the header, output block and citation route. Saitow 2017 is retained as
    # qualified open-shell CC background, not as a claim that this independent
    # UHF-space solver implements its single-spatial-orbital construction.
    if method in ("dlpno-ccsd", "dlpno-ccsd(t)") and molecule.multiplicity > 1:
        _effective_method = (
            "dlpno-uccsd(t)" if method == "dlpno-ccsd(t)" else "dlpno-uccsd"
        )

    # method in {mp2, scs-mp2, sos-mp2} with mp2_reference='rohf' on an
    # open-shell system runs an ROHF SCF + semicanonical ROHF-MP2; the
    # effective label drives the header + the rohf-mp2 citation route
    # (Roothaan + Knowles RMP2 + RI), distinct from the UHF-reference UMP2.
    _use_rohf_mp2 = (
        method in ("mp2", "scs-mp2", "sos-mp2")
        and molecule.multiplicity > 1
        and mp2_reference == "rohf"
    )
    if _use_rohf_mp2:
        _effective_method = "rohf-mp2"
        if mp2_options is not None or ump2_options is not None:
            raise ValueError(
                "run_job: mp2_options=/ump2_options= control the native "
                "RMP2/UMP2 kernels and are not used with mp2_reference='rohf'."
            )
    elif method in ("mp2", "scs-mp2", "sos-mp2"):
        if molecule.multiplicity > 1 and mp2_options is not None:
            raise ValueError(
                "run_job: mp2_options= controls closed-shell RMP2; use "
                "ump2_options= for open-shell UMP2."
            )
        if molecule.multiplicity == 1 and ump2_options is not None:
            raise ValueError(
                "run_job: ump2_options= controls open-shell UMP2; use "
                "mp2_options= for closed-shell RMP2."
            )

    # Materialize the post-SCF policy before either memory-estimate call.
    # Both VIBEQC_DRY_RUN_ESTIMATE (scheduler placement) and the real
    # preflight must size exactly the option object that execution consumes.
    # The helper clones caller objects before applying selectors or triples.
    (
        _dlpno_options_for_run,
        _dlpno_ccsd_options_for_run,
    ) = _prepare_dlpno_route_options(
        method,
        molecule,
        dlpno_options=dlpno_options,
        dlpno_ccsd_options=dlpno_ccsd_options,
        frozen_core=frozen_core,
        thresholds=_requested_dlpno_thresholds,
        aux_basis=aux_basis,
        orbital_basis_name=str(basis),
    )
    _rohf_mp2_options_for_estimate = None
    if _use_rohf_mp2:
        _rohf_mp2_options_for_estimate = _clone_mp2_like_options(
            None, UMP2Options
        )
        # ROHF-MP2 is unconditionally RI.  Materialize the exact auxiliary
        # identity before scheduler dry-run so dry admission, real admission,
        # and execution all consume the same prepared route policy.
        _rohf_mp2_options_for_estimate.density_fit = True
        _rohf_mp2_options_for_estimate.aux_basis = str(aux_basis or "")
        if not _rohf_mp2_options_for_estimate.aux_basis:
            try:
                from .density_fitting import default_aux_basis_for

                _rohf_mp2_options_for_estimate.aux_basis = (
                    default_aux_basis_for(str(basis), kind="ri")
                )
            except Exception:
                # Preserve run_rohf_mp2's explicit missing-auxiliary
                # diagnostic when no registered RIfit partner exists.
                _rohf_mp2_options_for_estimate.aux_basis = ""
        if _requested_frozen_core_count is not None:
            _rohf_mp2_options_for_estimate.n_frozen_core = (
                _requested_frozen_core_count
            )
    (
        _mp2_options_for_run,
        _ump2_options_for_run,
        _ccsd_options_for_run,
    ) = _prepare_canonical_correlation_options(
        method,
        _effective_method,
        molecule,
        mp2_options=mp2_options,
        ump2_options=ump2_options,
        ccsd_options=ccsd_options,
        requested_frozen_core_count=_requested_frozen_core_count,
        use_rohf_mp2=_use_rohf_mp2,
        ccsd_compute_triples=_ccsd_compute_triples,
        ccsd_triples_variant=_ccsd_triples_variant,
        density_fit=density_fit,
        aux_basis=aux_basis,
        orbital_basis_name=str(basis),
    )
    _frozen_core_citation_request = frozen_core
    if method in ("mp2", "scs-mp2", "sos-mp2"):
        _frozen_core_citation_options = (
            _rohf_mp2_options_for_estimate
            if _use_rohf_mp2
            else (
                _ump2_options_for_run
                if molecule.multiplicity > 1
                else _mp2_options_for_run
            )
        )
    elif method == "dlpno-mp2":
        _frozen_core_citation_options = _dlpno_options_for_run
    elif method in ("dlpno-ccsd", "dlpno-ccsd(t)"):
        _frozen_core_citation_options = _dlpno_ccsd_options_for_run
    elif method == "cc3":
        _frozen_core_citation_options = cc3_options
        # An active-space Hamiltonian fixes this partition to all-electron;
        # the implicit Table 2.69 selector is not used in that mode.
        if active_space is not None:
            _frozen_core_citation_request = 0
    elif method == "ccsdt":
        _frozen_core_citation_options = ccsdt_options
        if active_space is not None:
            _frozen_core_citation_request = 0
    else:
        _frozen_core_citation_options = _ccsd_options_for_run
    _frozen_core_source_entries = _frozen_core_citation_entries(
        method,
        _frozen_core_citation_request,
        _frozen_core_citation_options,
    )
    # Triples citations are result-derived after execution. In particular, a
    # local (T1) request whose pair space is fully screened must not cite an
    # algorithm that never ran.
    _dlpno_triples_source_entries: tuple[str, ...] = ()

    # Resolve the process allocation before either scheduler dry-run or real
    # admission. Prepared route options are private clones, so applying the
    # bound here cannot mutate caller-owned objects. A second refinement after
    # the first estimate reserves the retained SCF reference and safety
    # headroom; both paths perform that same re-plan below.
    _correlated_budget_bytes = _correlated_memory_budget_bytes(
        memory_budget_bytes
    )
    _native_budget = 0
    _native_options_for_run = [
        option
        for option in (
            _mp2_options_for_run,
            _ump2_options_for_run,
            _rohf_mp2_options_for_estimate,
            (
                _ccsd_options_for_run
                if _ccsd_triples_variant != "a-ccsd(t)"
                else None
            ),
        )
        if option is not None
    ]
    for _native_option in _native_options_for_run:
        _apply_correlated_process_budget(
            _native_option,
            _correlated_budget_bytes,
        )

    (
        rhf_options,
        uhf_options,
        rks_options,
        uks_options,
        rohf_options,
        roks_options,
    ) = _materialize_run_job_initial_guess(
        resolved_method,
        initial_guess,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
        rohf_options=rohf_options,
        roks_options=roks_options,
        read_from=read_from,
        fragments=fragments,
    )
    uhf_options, uks_options, _auto_open_shell_trah = _auto_open_shell_optimizer_trah(
        resolved_method,
        molecule,
        optimize=bool(optimize),
        uhf_options=uhf_options,
        uks_options=uks_options,
    )

    # Early SECCM gate (fail fast, clean "unavailable" -- not a silent gap):
    # the MSINDO cyclic-cluster route is single-point only through run_job.
    # gradients (finite-difference) + geometry optimisation live in
    # msindo_ccm.ccm_gradient_fd / ccm_optimize; analytic CCM gradients and the
    # run_job optimiser hook are not wired up.
    if resolved_method == "ccm" and optimize:
        raise NotImplementedError(
            "method='seccm' geometry optimisation is not available through "
            "run_job. Use vibeqc.semiempirical.methods.msindo_ccm.ccm_optimize "
            "(finite-difference, fixed Wigner-Seitz) for relaxed CCM geometries."
        )

    # ROKS geometry optimisation runs on finite-difference forces (routed
    # to the wavefunction ASE calculator below). The ROKS Hessian instead
    # finite-differences the *analytic* gradient (compute_hessian_fd),
    # which needs the molecular XC-gradient term ROKS doesn't have yet, so
    # gate it. ROHF has an analytic gradient (compute_rohf_gradient), so
    # the ROHF Hessian / frequencies are supported.
    if resolved_method == "roks" and hessian:
        raise NotImplementedError(
            "method='roks': the Hessian finite-differences the analytic "
            "ROKS gradient (needs the molecular XC-gradient term), not yet "
            "implemented (single-point energies + FD geometry optimisation "
            "work; ROHF Hessians work). Track: handovers/HANDOVER_ROHF.md milestone M5b."
        )

    # Early MLIP gate (fail fast, before any output / SCF machinery): an
    # academic-only (ASL) MACE model needs explicit acknowledgment, and the
    # model must cover every element in the system -- else a config error
    # would otherwise surface as a crashed SCF. Both checks run again inside
    # MACEModel (defense in depth).
    if resolved_method in _MLIP_METHODS:
        from vibeqc.mlip import MLIPOptions, resolve_model
        from vibeqc.mlip.mace import element_coverage_error, enforce_academic_license

        _mlopts = mlip_options or MLIPOptions()
        _minfo = resolve_model(_mlopts.model)
        enforce_academic_license(_minfo, _mlopts)
        _ecov = element_coverage_error(_minfo, molecule)
        if _ecov is not None:
            raise ValueError(_ecov)

    # Composite recipes may request D4 dispersion; the D4 path is
    # wired through compute_d4 in the post-SCF block below. A D4
    # request short-circuits the D3-BJ resolution.
    use_d4 = isinstance(dispersion, str) and dispersion.strip().lower() == "d4"

    # Build the declarative OutputPlan once. It drives BOTH the
    # dry-run pre-flight (below) and the real run's manifest
    # lifecycle (the OutputWriter constructed before the SCF). One
    # source of truth for "what files will this job produce".
    _cube_req_plan = parse_write_cube_kwarg(write_cube)
    _output_plan = OutputPlan.from_run_job_kwargs(
        output=output_stem,
        method=resolved_method,
        basis=basis,
        functional=functional,
        optimize=optimize,
        write_molden_file=write_molden_file,
        write_xyz=write_xyz_file,
        citations=citations,
        write_population=write_population_file,
        write_cube_density=_cube_req_plan.density,
        cube_mo_labels=tuple(_cube_req_plan.mo_labels),
        perf_log=perf_log,
        structured_log=structured_log,
        crash_dump=crash_dump,
        output_qvf=output_qvf,
        write_trexio_file=_trexio_path,
    )

    # Materialize one internal KS options object before density fitting,
    # solver selection, DFT+U, or an optimizer can consume it. Previously
    # those feature paths could each create a bare options object and bypass
    # _run_single_point's orca-defgrid3 default, silently running on the
    # legacy ProductGL/Becke surface (issue #251). A caller-provided grid
    # remains authoritative; a caller-provided object with an untouched grid
    # does not (issue #663).
    _skala_grid_policy = "explicit-options"
    if resolved_method == "rks" and _rks_grid_untouched:
        if rks_options is None:
            rks_options = RKSOptions()
        _resolved_grid_level = (
            "skala"
            if _uses_skala and grid_level == "orca-defgrid3"
            else grid_level
        )
        _apply_grid_level(rks_options.grid, _resolved_grid_level)
        _skala_grid_policy = _resolved_grid_level
    elif resolved_method == "uks" and _uks_grid_untouched:
        if uks_options is None:
            uks_options = UKSOptions()
        _resolved_grid_level = (
            "skala"
            if _uses_skala and grid_level == "orca-defgrid3"
            else grid_level
        )
        _apply_grid_level(uks_options.grid, _resolved_grid_level)
        _skala_grid_policy = _resolved_grid_level
    elif resolved_method == "roks" and _roks_grid_untouched:
        if roks_options is None:
            roks_options = ROKSOptions()
        if roks_options.grid is None:
            roks_options.grid = GridOptions()
        _resolved_grid_level = (
            "skala"
            if _uses_skala and grid_level == "orca-defgrid3"
            else grid_level
        )
        _apply_grid_level(roks_options.grid, _resolved_grid_level)
        _skala_grid_policy = _resolved_grid_level

    # Wire density_fit / aux_basis / cosx onto the SCF option structs so
    # the SCF drivers and the CPCM JKBuilder resolver can read them via
    # getattr.  Materialise defaults when the user passed None so the
    # attributes have a home.  This wiring MUST happen before every
    # ``estimate_memory`` call site (the dry-run estimate below and the
    # real-run pre-flight): BUG 117 — when it ran only just before the
    # SCF, a ``run_job(..., density_fit=True)`` job was estimated on the
    # un-wired options, took the direct-SCF estimator branch, and a
    # def2-QZVPP DF job preflighted at ~1 GB while the driver
    # materialised the tens-of-GB three-index tensor (BH9 campaign:
    # 515/822 jobs OOM-killed).
    if density_fit or cosx:
        if rhf_options is None:
            rhf_options = RHFOptions()
        rhf_options.density_fit = True
        rhf_options.cosx = bool(cosx)
        if aux_basis:
            rhf_options.aux_basis = aux_basis
        if uhf_options is None:
            uhf_options = UHFOptions()
        uhf_options.density_fit = True
        uhf_options.cosx = bool(cosx)
        if aux_basis:
            uhf_options.aux_basis = aux_basis
        if rks_options is None:
            rks_options = RKSOptions()
        rks_options.density_fit = True
        rks_options.cosx = bool(cosx)
        if aux_basis:
            rks_options.aux_basis = aux_basis
        if uks_options is None:
            uks_options = UKSOptions()
        uks_options.density_fit = True
        uks_options.cosx = bool(cosx)
        if aux_basis:
            uks_options.aux_basis = aux_basis

    # Dry-run short-circuit (Phase O3). When ``dry_run=True`` or the
    # ``VIBEQC_DRY_RUN`` env var is set, write a one-shot
    # ``{output}.system`` with ``[outputs].status = "dry_run"``,
    # print the declared-artefacts summary to stdout, and return
    # ``None`` without running the SCF. This is the pre-flight path
    # that ``vq submit`` uses to learn which files a job will
    # produce. Run before any heavyweight setup (basis-set
    # construction, memory estimate, perf tracker) so the
    # short-circuit is genuinely cheap.
    if dry_run or is_dry_run_requested():
        # Make sure the parent directory exists so we don't fail on a
        # freshly-typed `-o /tmp/new-subdir/foo` path.
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        # Opt-in peak-memory estimate for memory-aware `vq submit auto`
        # RAM-fit placement (VIBEQC_DRY_RUN_ESTIMATE). Default-off so the
        # cheap output-discovery dry-run stays cheap; only under the flag
        # do we build the basis + estimate. Mirrors the real-run estimate
        # below (search `estimate_memory(`). Mean-field methods keep their
        # lightweight default options; active-space methods must pass the CAS
        # partition so scheduler preflight sees the real post-SCF workspace.
        # Best-effort: any failure (unestimable method, basis build error,
        # ...) leaves estimate_bytes=None so the dry-run never breaks.
        _estimate_bytes: Optional[int] = None
        if is_dry_run_estimate_requested() and resolved_method not in _MLIP_METHODS:
            try:
                if resolved_method in SEMIEMPIRICAL_METHODS:
                    _estimate_bytes = estimate_semiempirical_memory(
                        molecule,
                        method=resolved_method,
                        nddo=nddo,
                        solvent=solvent,
                        optimize=optimize,
                        max_steps=max_opt_steps,
                        ccm_options=ccm_options,
                    ).total_bytes
                else:
                    _estimate_method = _memory_estimator_method(
                        resolved_method,
                        _effective_method,
                    )
                    _dry_basis = BasisSet(molecule, basis)
                    _dry_estimator_options = _memory_estimator_options(
                        _estimate_method,
                        resolved_method=resolved_method,
                        rhf_options=rhf_options,
                        uhf_options=uhf_options,
                        rks_options=rks_options,
                        uks_options=uks_options,
                        rohf_options=rohf_options,
                        roks_options=roks_options,
                        mp2_options=_mp2_options_for_run,
                        ump2_options=(
                            _rohf_mp2_options_for_estimate
                            if _use_rohf_mp2
                            else _ump2_options_for_run
                        ),
                        ccsd_options=_ccsd_options_for_run,
                        cc3_options=cc3_options,
                        ccsdt_options=ccsdt_options,
                        dlpno_options=_dlpno_options_for_run,
                        dlpno_ccsd_options=_dlpno_ccsd_options_for_run,
                        caspt2_options=caspt2_options,
                        nevpt2_options=nevpt2_options,
                        active_space=active_space,
                        selected_ci_options=selected_ci_options,
                        tddft=tddft,
                        tddft_n_states=tddft_n_states,
                        tddft_type=tddft_type,
                        functional=functional,
                    )
                    _dry_estimate = estimate_memory(
                        molecule,
                        _dry_basis,
                        method=_estimate_method,
                        options=_dry_estimator_options,
                    )
                    _native_budget, _budget_changed = (
                        _refine_native_correlated_budgets(
                            _native_options_for_run,
                            _correlated_budget_bytes,
                            _dry_estimate,
                        )
                    )
                    if _budget_changed:
                        _dry_estimate = estimate_memory(
                            molecule,
                            _dry_basis,
                            method=_estimate_method,
                            options=_dry_estimator_options,
                        )
                    _estimate_bytes = _dry_estimate.total_bytes
            except Exception:
                _estimate_bytes = None
        if _uses_skala:
            from .skala import _run_provenance_fields

            _dry_run_fields = _run_provenance_fields()
            _dry_run_fields["skala_grid_policy"] = _skala_grid_policy
            _dry_manifest = ManifestUpdater(
                _output_plan,
                record_hostname=record_hostname,
                wall_seconds=0.0,
                estimate_bytes=_estimate_bytes,
                extra_run_fields=_dry_run_fields,
            )
            _dry_manifest.mark_dry_run()
            print_dry_run_summary(_output_plan)
        else:
            dry_run_manifest(
                _output_plan,
                record_hostname=record_hostname,
                estimate_bytes=_estimate_bytes,
            )
        return None

    # Resolve dispersion up front so a bad spec fails before we touch
    # the filesystem. d3_params is None iff dispersion is disabled.
    # A D4 request (composite recipes r^2SCAN-3c / wB97X-3c) short-
    # circuits the D3-BJ path -- D4 is applied in the post-SCF block.
    if use_d4:
        d3_params = None
    else:
        d3_params = _resolve_dispersion(dispersion, functional)

    # Pin the thread count if the caller asked for one; otherwise leave
    # the current state alone. We always query at the end so the output
    # reports what was actually used (which may differ from what the
    # user asked for -- e.g. on a single-core build or OpenMP-disabled
    # environment).
    if num_threads is not None:
        set_num_threads(int(num_threads))
    threads_in_use = get_num_threads()

    # Ensure the parent directory exists so users can give names like
    # "runs/water" without pre-creating "runs/".
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct the OutputWriter -- the per-job coordinator that owns
    # ``{output}.system`` for the rest of the run. This writes the
    # manifest immediately with the ``[plan]`` section + ``[outputs]
    # .status = "running"`` *before* the SCF starts, so a ``vq``
    # daemon polling the workspace sees the declared output set and a
    # liveness signal from the first moment. ``finish()`` /
    # ``crash()`` at the exit paths flip the status; the end-of-job
    # record-sweep fills ``[[outputs.files]]`` with the artefacts
    # that actually landed. Replaces the bare end-of-job
    # ``write_system_manifest`` call (pre-v1.0 dispatch-overhaul).
    _extra_run: dict[str, Any] = {}
    if _uses_skala:
        from .skala import _run_provenance_fields

        _extra_run.update(_run_provenance_fields())
        _extra_run["skala_grid_policy"] = _skala_grid_policy
    if resolved_method in ("rohf", "roks"):
        # Initial running-manifest value.  Fractional-shell ROKS may select
        # the Roothaan fractional-shell convention after its frontier
        # degeneracy is known; the converged result overwrites this late-bound
        # field below.
        _extra_run["rohf_canonicalization"] = "guest-saunders"
    _output_writer = OutputWriter(
        _output_plan,
        record_hostname=record_hostname,
        extra_run_fields=_extra_run if _extra_run else None,
    )

    # [basis_library] -- which basis library actually answered, plus any set
    # rendered on demand. Recorded on every run, not only a fetching one:
    # vibe-qc prefers a build overlay over the committed tree, and a result
    # produced against a stale overlay is otherwise indistinguishable from
    # one produced against current data.
    try:
        from .basis_fetch import library_root as _library_root

        _resolved_library = _library_root()
        _output_writer.set_basis_library(
            resolved_root=str(_resolved_library) if _resolved_library else "",
            fetched=[_r.manifest_row() for _r in _bse_rendered],
        )
    except Exception:  # pragma: no cover - provenance must never fail a run
        pass

    # Opt-in live QVF checkpointing (vibe-view hot-reload). Disabled unless
    # ``checkpoint_qvf`` is set. The native molecular callback below feeds
    # terminal/NDJSON/manifest observability, not checkpoint snapshots, so
    # this still emits only atomic start + terminal lifecycle frames.
    from .output.checkpoint import QvfCheckpointer as _QvfCheckpointer

    _checkpointer = _QvfCheckpointer(
        checkpoint_qvf if output_qvf else None,
        checkpoint_every,
        plan=_output_plan,
    )
    # Initial checkpoint frame: the input geometry, so a live viewer has a
    # structure to show from the first moment the job is submitted.
    if _checkpointer.enabled:
        _checkpointer.snapshot(
            molecule=molecule,
            method=resolved_method,
            basis=basis,
            functional=functional,
        )

    t_job_start = time.perf_counter()

    # ``OutputChannel.to_file`` opens the .out line-buffered so a
    # `tail -f` shows new lines as the SCF / property pipeline emits
    # them. We also call ``flush()`` explicitly after major blocks for
    # the same reason -- line buffering only kicks in on newlines.
    # Installing it as the ambient channel lets any module downstream
    # emit into the .out via ``vibeqc.output.write`` without being
    # handed a file object.
    #
    # The perf-log context manager wraps the whole body so PerfScope
    # entries from anywhere downstream -- periodic SCF stages, gradient
    # calls, anything that calls ``with PerfScope("..."): ...`` --
    # accumulate into the same tracker. When ``_perf_target`` is None
    # (and ``$VIBEQC_PERFLOG`` isn't set), the context manager is a
    # cheap no-op: PerfScope.__enter__/__exit__ become a single
    # ContextVar lookup with no I/O.
    hessian_result = None
    _optimized_single_point_result = None

    # Install the C++ → Python diagnostics bridge before any
    # computation, so VIBEQC_DIAG / vibeqc::diagnostic() calls in
    # the C++ core are forwarded into the .out (at their tagged
    # verbosity level).  Idempotent; cheap enough to call every job.
    install_diagnostics_bridge()
    # Native molecular SCFs emit one ``progress`` diagnostic after each
    # completed iteration. Keep exact (iteration, energy) identities so the
    # compatibility replay below can fill traces from non-callback drivers
    # without duplicating rows that were already written live.
    _cpp_scf_callbacks: list[tuple[int, str]] = []
    _cpp_scf_attempt_rows: dict[int, list[tuple[int, str]]] = {}
    _cpp_scf_attempt_phases: dict[int, str] = {}
    _live_cpp_scf_attempt_rows: dict[int, list[tuple[int, str]]] = {}
    _cpp_scf_attempt = 0
    # GitLab #482: has the live .out iteration table been opened?
    _scf_rows_streamed = False
    _cpp_scf_last_iteration = 0
    _cpp_scf_last_phase: str | None = None
    _headline_progress_fields: dict[str, object] | None = None

    with (
        ExitStack() as _progress_stack,
        _perf_log_ctx(_perf_target),
        _structured_log_ctx(_structured_target) as _slog,
        crash_dump_context(_crash_stem) as _crash_target,
        OutputChannel.to_file(out_path) as _out_channel,
        PerfScope("run_job.total"),
    ):
        def _handle_cpp_scf_progress(fields: dict[str, object]) -> None:
            nonlocal _cpp_scf_attempt
            nonlocal _cpp_scf_last_iteration
            nonlocal _cpp_scf_last_phase

            iteration = int(fields["iter"])
            energy = float(fields["energy"])
            delta_e = (
                None
                if iteration <= 1 or fields.get("dE") is None
                else float(fields["dE"])
            )
            gradient = (
                None
                if fields.get("grad") is None
                else float(fields["grad"])
            )
            diis = int(fields.get("diis", 0))
            phase = _ACTIVE_MOLECULAR_PROGRESS_PHASE.get()

            # A new native SCF invocation restarts its cycle counter. This is
            # common for unrestricted stability searches, multi-guess runs,
            # and convergence retries. Preserve every completed cycle, but
            # label the attempt so consumers can distinguish the sequences.
            if (
                _cpp_scf_attempt == 0
                or phase != _cpp_scf_last_phase
                or iteration <= _cpp_scf_last_iteration
            ):
                _cpp_scf_attempt += 1
                _cpp_scf_attempt_phases[_cpp_scf_attempt] = phase
            _cpp_scf_last_iteration = iteration
            _cpp_scf_last_phase = phase
            identity = _scf_progress_identity(iteration, energy)
            _cpp_scf_callbacks.append(identity)
            _cpp_scf_attempt_rows.setdefault(_cpp_scf_attempt, []).append(
                identity
            )

            # Structured durability is independent of the presentation sink:
            # caller-supplied progress wrappers may consume or raise from
            # iteration(). Emit first, at the callback's wall-clock time.
            structured_before = _slog.n_emitted
            try:
                _slog.emit(
                    "scf_iter",
                    iter=iteration,
                    energy=energy,
                    dE=delta_e,
                    grad_norm=gradient,
                    diis_subspace=diis,
                    phase=phase,
                    attempt=_cpp_scf_attempt,
                    wall_s=float(time.perf_counter() - t_job_start),
                )
            except Exception:
                # A broken optional NDJSON sink must not suppress the live
                # manifest heartbeat or the user's terminal stream.
                pass
            if _slog.n_emitted > structured_before:
                _live_cpp_scf_attempt_rows.setdefault(
                    _cpp_scf_attempt,
                    [],
                ).append(identity)

            # Preserve the manifest heartbeat before invoking arbitrary user
            # progress code. Use documented stable names, not wire aliases.
            manifest_fields: dict[str, object] = {
                "phase": phase,
                "iteration": iteration,
                "attempt": _cpp_scf_attempt,
                "energy_eh": energy,
                "diis_subspace": diis,
            }
            if gradient is not None:
                manifest_fields["gradient_norm"] = gradient
            try:
                _output_writer.update_progress(**manifest_fields)
            except Exception:
                # Structured and terminal sinks are independent. A transient
                # manifest rewrite failure must not suppress the completed
                # native cycle from either one.
                pass

            # ProgressLogger owns only human formatting here. Temporarily
            # shadow the ambient StructuredLog because the canonical row was
            # already emitted directly above; this also makes duck-typed sinks
            # unable to suppress or duplicate the machine-readable record.
            try:
                with _structured_log_ctx(enabled=False):
                    plog.iteration(
                        iteration,
                        energy=energy,
                        dE=delta_e,
                        grad=gradient,
                        diis=diis,
                        attempt=_cpp_scf_attempt,
                    )
            except Exception:
                pass

            # Stream the row into the .out as it happens (GitLab #482).
            #
            # The .out is the artifact an operator opens first, and for a
            # walltime-killed job it used to end at the memory-estimate
            # banner: the iteration table was rendered from result.scf_trace
            # only after the native SCF returned, so a process killed inside
            # the SCF never reached it. The structured sidecar kept its rows
            # (#25) but is not written unless structured_log is on.
            #
            # write() + flush() reaches disk mid-run and survives SIGTERM --
            # verified by killing a uracil-dimer RHF at 14 s and recovering
            # 17 rows -- so this needs nothing new from vibeqc.output; the
            # .out was never "assembled at job end", the runner simply never
            # wrote here. write_scf_trace is told to close the table rather
            # than re-emit it, so a completed run's bytes are unchanged.
            #
            # Only the primary SCF phase streams. Helper SCFs (optimizer
            # steps, atomization fragments, Hessian displacements) would
            # interleave dozens of tables into the document.
            try:
                if phase == "scf":
                    nonlocal _scf_rows_streamed
                    if not _scf_rows_streamed:
                        write(_SCF_TRACE_HEADER + "\n"
                              + _SCF_TRACE_SEPARATOR + "\n")
                        _scf_rows_streamed = True
                    write(
                        _format_scf_iter_line(
                            {
                                "iter": iteration,
                                "energy": energy,
                                "delta_e": delta_e if delta_e is not None else 0.0,
                                "grad_norm": gradient if gradient is not None else 0.0,
                                "diis_subspace": diis,
                            }
                        )
                        + "\n"
                    )
                    flush()
            except Exception:
                # Never let the .out mirror break a running SCF.
                pass

        # Keep a safe fallback scope active for every native molecular SCF
        # launched by this job, including optimizer, atomization, Hessian,
        # and property helpers. Narrower scopes below replace the phase label
        # while preserving nested restoration.
        _progress_stack.enter_context(
            _molecular_progress_scope(
                _handle_cpp_scf_progress,
                phase="helper.scf",
            )
        )

        # Structured log: banner + job_start records. Always safe to
        # emit -- _slog.emit is a no-op when the log is disabled.
        _libs = library_versions()
        _fp = run_fingerprint(
            method=resolved_method,
            basis=basis,
            functional=functional,
            molecule=molecule,
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
            method=resolved_method,
            basis=basis,
            functional=functional,
            optimize=bool(optimize),
            threads=int(threads_in_use),
            n_atoms=int(len(list(molecule.atoms))),
            charge=int(molecule.charge),
            multiplicity=int(molecule.multiplicity),
            n_electrons=int(molecule.n_electrons()),
            output_stem=str(output_stem),
        )
        write(banner() + "\n\n")
        # Post-SCF methods resolve their SCF to rhf/uhf; the *header* should
        # still name the correlation method (RHF + MP2/CCSD), so display the
        # original keyword for those (but keep "auto" -> resolved everywhere
        # else).
        _header_method = (
            _effective_method
            if _effective_method in _POSTSCF_CITE_METHODS
            else resolved_method
        )
        # resolved_method is the mean-field SCF a post-SCF method ran on
        # (rhf/uhf/rohf); name it as the reference in the header. ROHF is the
        # reference for CCSD(T) with ccsd_reference="rohf".
        _scf_ref = (
            resolved_method if resolved_method in ("rhf", "uhf", "rohf") else "rhf"
        )
        write(
            _job_header(_header_method, basis, functional, scf_reference=_scf_ref)
            + "\n"
        )
        write(_geom_summary(molecule) + "\n")
        if name_molecule:
            write_iupac_name(molecule)
        write(f"  Threads: {threads_in_use}  (OpenMP shared-memory parallelism)\n\n")
        if _auto_open_shell_trah:
            write(
                "  Auto convergence profile: open-shell geometry optimization "
                "uses TRAH (trah_threshold=1.0)\n\n"
            )
        flush()
        plog.banner(
            f"run_job  {_job_header(_header_method, basis, functional, scf_reference=_scf_ref).strip()}"
        )
        plog.info(f"Threads: {threads_in_use}  (OpenMP shared-memory parallelism)")
        plog.info(f"Output file: {out_path}")

        # ── Solver selection ───────────────────────────────────────────────
        if solver not in ("dense", "davidson", "lobpcg"):
            raise ValueError(
                f"run_job: solver='{solver}' is not recognised. "
                f"Supported: 'dense', 'davidson', 'lobpcg'."
            )
        if solver != "dense" and resolved_method in (
            "rhf",
            "uhf",
            "rks",
            "uks",
        ):
            # Materialise default options for the molecular C++ SCF path.
            # "davidson" -> use_davidson=True (C++ Davidson backend).
            # "lobpcg" -> use_davidson=True for now (C++ molecular SCF
            #   doesn't expose LOBPCG directly; periodic / ROHF drivers
            #   use the Python solver stack via _scf_diagonalize).
            if resolved_method == "rhf":
                if rhf_options is None:
                    rhf_options = RHFOptions()
                rhf_options.use_davidson = True
            elif resolved_method == "uhf":
                if uhf_options is None:
                    uhf_options = UHFOptions()
                uhf_options.use_davidson = True
            elif resolved_method == "rks":
                if rks_options is None:
                    rks_options = RKSOptions()
                rks_options.use_davidson = True
            elif resolved_method == "uks":
                if uks_options is None:
                    uks_options = UKSOptions()
                uks_options.use_davidson = True

        t_opt = 0.0
        _traj_frames = []
        _traj_energies = []
        # Whether the ASE BFGS backend actually drove the relaxation. Only
        # then is the ASE citation honest -- the native / brent / geomopt
        # (RFO / FIRE / GDIIS / ...) backends use no ASE optimizer, so they
        # must not pull the larsen_ase_2017 BFGS row (CLAUDE.md § 8: attribute,
        # never claim). Set True inside the ASE branch below.
        _used_ase_optimizer = False
        _used_geomopt_optimizer = False
        # Citation gate: True unless a coupled-cluster step actually ran on
        # the canonical (non-DF) integral route, in which case the DF-CCSD
        # assembly citation (DePrince-Sherrill 2013) is dropped from the
        # method bundle (assemble(cc_density_fit=...); CLAUDE.md § 8:
        # attribute, never claim). Set in the CCSD block below.
        _cc_used_density_fit = True
        # #701: which pair density the PNO occupation-number cut was applied
        # to. Empty for routes that do not build PNOs; the resolved canonical
        # value feeds [routes.pno_norm].
        _dlpno_pno_norm = ""
        _cc_open_shell_used = False
        _cc_used_fno = False
        _mp2_used_density_fit = False
        _mp2_memory_mode_used = ""
        _cc_triples_memory_mode_used = ""
        if optimize:
            # Increment 3 -- analytic +U gradient via the
            # variational F = F_HF + S V_AO S Fock contribution.
            # The ASE _WavefunctionCalculator path is plumbed below.
            # For dft_plus_u, pre-populate the relevant method-options
            # struct's dft_plus_u_sites / dft_plus_u_ao_groups fields
            # once at the initial geometry -- the AO-group indices are
            # geometry-invariant (depend only on shell layout per atom
            # Z + basis name), so the same translation is valid at
            # every optimization step.
            if dft_plus_u and resolved_method in (
                "rhf",
                "uhf",
                "rks",
                "uks",
            ):
                from .dft_plus_u import _apply_dft_plus_u_to_options

                # Materialise a defaulted options struct if the user
                # didn't pass one, so the C++ binding sees a real
                # object with the dft_plus_u_sites field populated.
                if resolved_method == "rhf":
                    if rhf_options is None:
                        rhf_options = RHFOptions()
                    _opts_for_dft_plus_u = rhf_options
                elif resolved_method == "uhf":
                    if uhf_options is None:
                        uhf_options = UHFOptions()
                    _opts_for_dft_plus_u = uhf_options
                elif resolved_method == "rks":
                    if rks_options is None:
                        rks_options = RKSOptions()
                    _opts_for_dft_plus_u = rks_options
                else:  # uks
                    if uks_options is None:
                        uks_options = UKSOptions()
                    _opts_for_dft_plus_u = uks_options
                _basis_for_dft_plus_u = BasisSet(molecule, basis)
                _apply_dft_plus_u_to_options(
                    _opts_for_dft_plus_u,
                    _basis_for_dft_plus_u,
                    dft_plus_u,
                )
            # #643: ECP centres are absolute coordinates on the SCF options.
            # Attach the basis sidecar's ECP for the START geometry now (the
            # SCF wrappers would otherwise attach lazily inside the first
            # single point, after the optimizer has already taken a step),
            # remember which atom each centre sits on, and after the
            # optimizer returns move the centres onto the optimized geometry
            # so the final single point and the FD Hessian differentiate the
            # ECP Hamiltonian of the geometry they actually run at. The
            # optimizers themselves follow the centres per visited geometry
            # (ase.VibeQC, molecular_optimize).
            _ecp_opt_opts = {
                "rhf": rhf_options,
                "uhf": uhf_options,
                "rks": rks_options,
                "uks": uks_options,
            }.get(resolved_method)
            _ecp_opt_indices = None
            if _ecp_opt_opts is not None and basis:
                attach_inline_ecp_options_from_basis_sidecar(
                    _ecp_opt_opts, molecule, BasisSet(molecule, basis)
                )
                if options_carry_ecp(_ecp_opt_opts):
                    _ecp_opt_indices = ecp_centre_atom_indices(
                        _ecp_opt_opts, molecule
                    )

            def _ecp_follow_optimized_geometry(final_molecule) -> None:
                if _ecp_opt_indices is not None:
                    reposition_ecp_centres(
                        _ecp_opt_opts, final_molecule, _ecp_opt_indices
                    )

            _opt_backend = _resolve_optimizer_backend(optimizer_backend)

            # Common gradient tolerance (eV/Angstrom -> Ha/bohr).
            _conv_grad = fmax * _EV_PER_ANGSTROM_TO_HA_PER_BOHR

            # Semi-empirical and MLIP methods must use ASE -- the native
            # L-BFGS-B and Brent backends drive SCF+gradient through the
            # molecular ab-initio code path, not semi-empirical drivers.
            _se_or_mlip = (
                resolved_method in SEMIEMPIRICAL_METHODS or resolved_method in _MLIP_METHODS
            )
            if geom_opt is not None and _se_or_mlip:
                raise ValueError(
                    "run_job(geom_opt=...) uses the native molecular geomopt "
                    "provider and is only available for Gaussian-basis "
                    "electronic-structure methods. Semi-empirical and MLIP "
                    "methods must use optimizer_backend='ase'."
                )
            if _se_or_mlip and _opt_backend != "ase":
                if optimizer_backend == "auto":
                    raise ImportError(
                        "run_job(optimize=True) for semi-empirical and MLIP "
                        "methods requires ASE. Install ASE or set "
                        "optimize=False; the native and brent optimizers use "
                        "Gaussian-basis wavefunction providers and cannot run "
                        f"method={resolved_method!r}."
                    )
                raise ValueError(
                    "run_job(optimize=True) for semi-empirical and MLIP methods "
                    "requires optimizer_backend='ase'; native and brent "
                    "optimizers use Gaussian-basis wavefunction providers and "
                    f"cannot run method={resolved_method!r}."
                )

            # ---- geomopt-native path (v0.14+) ----------------------------------
            if geom_opt is not None and not _se_or_mlip:
                _used_geomopt_optimizer = True
                write(
                    f"  Geometry optimization (geomopt/{geom_opt})\n"
                    f"  target = {geom_target}, coords = {geom_coords}\n"
                    f"  line_search = {geom_line_search}, "
                    f"hessian = {geom_hessian_init}/{geom_hessian_update}\n"
                    f"  gmax = {_conv_grad:.2e} Ha/bohr "
                    f"({fmax:.4g} eV/A), max_steps = {max_opt_steps}\n\n"
                )
                flush()
                with (
                    plog.stage(
                        "geometry_optimization",
                        detail=f"{geom_opt}, gmax={_conv_grad:.1e} Ha/bohr",
                    ),
                    PerfScope("geometry_optimization"),
                    _molecular_progress_scope(
                        _handle_cpp_scf_progress,
                        phase="geom-opt.scf",
                    ),
                ):
                    t0 = time.perf_counter()
                    from .geomopt import (
                        ConvergencePolicy,
                        MolecularSCFProvider,
                        run_geomopt,
                    )

                    _conv_pol = ConvergencePolicy(
                        gmax=geom_conv_gmax
                        if geom_conv_gmax is not None
                        else _conv_grad
                    )
                    _provider = MolecularSCFProvider(
                        basis,
                        method=resolved_method,
                        functional=functional,
                        rhf_options=rhf_options,
                        uhf_options=uhf_options,
                        rks_options=rks_options,
                        uks_options=uks_options,
                        rohf_options=rohf_options, roks_options=roks_options,
                        # GitLab #663: the provider applies its grid preset to
                        # an untouched grid; forward run_job's level so an
                        # explicit "legacy" is not re-defaulted to
                        # orca-defgrid3 (the SKALA auto-selection already
                        # marked the grid as chosen when it applied).
                        grid_level=grid_level,
                        dispersion_params=d3_params,
                        solvent=solvent,
                        cisd_options=cisd_options,
                        selected_ci_options=selected_ci_options,
                        dmrg_options=dmrg_options,
                        v2rdm_options=v2rdm_options,
                        transcorrelated_options=transcorrelated_options,
                        casci_options=casci_options,
                        caspt2_options=caspt2_options,
                        nevpt2_options=nevpt2_options,
                        casscf_options=casscf_options,
                        active_space=active_space,
                        cas_reference=cas_reference,
                    )
                    _output_writer.update_progress(phase="geom-opt", iteration=0)
                    _opt_result = run_geomopt(
                        molecule,
                        _provider,
                        geom_opt=geom_opt,
                        geom_coords=geom_coords,
                        geom_target=geom_target,
                        geom_hessian_init=geom_hessian_init,
                        geom_hessian_update=geom_hessian_update,
                        geom_line_search=geom_line_search,
                        geom_conv=_conv_pol,
                        geom_max_iter=max_opt_steps,
                        geom_freeze=geom_freeze,
                        geom_opt_options=geom_opt_options,
                        geom_restart=geom_restart,
                        geom_checkpoint=geom_checkpoint,
                        record_trajectory=bool(output_qvf),
                        progress=False,
                    )
                    t_opt = time.perf_counter() - t0
                molecule = _opt_result.system
                _ecp_follow_optimized_geometry(molecule)
                _traj_frames = _opt_result.trajectory_frames or []
                _traj_energies = _opt_result.trajectory_energies or []

                write("  Optimized geometry\n")
                write(_geom_summary(molecule) + "\n\n")
                flush()

            elif _opt_backend == "ase":
                _used_ase_optimizer = True
                _output_writer.update_progress(phase="geom-opt", iteration=0)
                write(f"  Geometry optimization (ASE/BFGS) -> {traj_path.name}\n")
                write(f"  fmax = {fmax} eV/A, max_steps = {max_opt_steps}\n\n")
                flush()
                t0 = time.perf_counter()
                _ase_result_holder = []
                try:
                    with (
                        plog.stage(
                            "geometry_optimization",
                            detail=(
                                f"BFGS, fmax={fmax} eV/A, "
                                f"max_steps={max_opt_steps}"
                            ),
                        ),
                        PerfScope("geometry_optimization"),
                        _molecular_progress_scope(
                            _handle_cpp_scf_progress,
                            phase="geom-opt.scf",
                        ),
                    ):
                        molecule = _optimize_geometry(
                            molecule,
                            basis,
                            functional=functional,
                            trajectory_path=traj_path,
                            fmax=fmax,
                            max_steps=max_opt_steps,
                            dispersion_params=d3_params,
                            method=resolved_method,
                            cisd_options=cisd_options,
                            cc3_options=cc3_options,
                            ccsdt_options=ccsdt_options,
                            selected_ci_options=selected_ci_options,
                            dmrg_options=dmrg_options,
                            v2rdm_options=v2rdm_options,
                            transcorrelated_options=transcorrelated_options,
                            casci_options=casci_options,
                            caspt2_options=caspt2_options,
                            nevpt2_options=nevpt2_options,
                            casscf_options=casscf_options,
                            active_space=active_space,
                            cas_reference=cas_reference,
                            rhf_options=rhf_options,
                            uhf_options=uhf_options,
                            rks_options=rks_options,
                            uks_options=uks_options,
                            rohf_options=rohf_options,
                            mlip_options=mlip_options,
                            result_holder=_ase_result_holder,
                            grid_level=grid_level,
                            roks_options=roks_options,
                        )
                        _ecp_follow_optimized_geometry(molecule)
                except Exception as _opt_exc:
                    t_opt = time.perf_counter() - t0
                    _nonconverged = isinstance(
                        _opt_exc, _ASEGeometryOptimizationError
                    )
                    if _nonconverged:
                        molecule = _opt_exc.system
                        _ecp_follow_optimized_geometry(molecule)
                        _event = "geometry_optimization_nonconverged"
                        _event_fields = {
                            "optimizer": "ase_bfgs",
                            "converged": False,
                            "n_steps": int(_opt_exc.n_steps),
                            "max_steps": int(_opt_exc.max_steps),
                            "final_max_force_ev_per_angstrom": float(
                                _opt_exc.final_max_force
                            ),
                            "target_max_force_ev_per_angstrom": float(
                                _opt_exc.target_max_force
                            ),
                            "final_max_displacement_angstrom": float(
                                _opt_exc.final_max_displacement
                            ),
                        }
                    else:
                        _event = "geometry_optimization_failed"
                        _event_fields = {
                            "optimizer": "ase_bfgs",
                            "exception_type": type(_opt_exc).__name__,
                            "exception": str(_opt_exc),
                        }
                    record(
                        _event,
                        f"\n  FATAL: {_opt_exc}\n",
                        level=Level.QUIET,
                        wall_s=float(t_opt),
                        **_event_fields,
                    )
                    flush()

                    # A capped run is scientifically unsuccessful, but its
                    # last geometry and complete trajectory are valuable
                    # diagnostics and must survive the nonzero return.
                    if _nonconverged and write_xyz_file:
                        try:
                            _output_writer.dispatch_role(
                                "geometry",
                                only_format="xyz",
                                molecule=molecule,
                                energy_ha=None,
                                raise_on_error=True,
                            )
                            write(
                                "  Last non-converged geometry written to "
                                f"{stem_sibling(output_stem, '.xyz').name}\n"
                            )
                            flush()
                        except Exception as _xyz_exc:
                            warn_writer_failure(
                                _xyz_exc,
                                stem_sibling(output_stem, ".xyz"),
                                role="nonconverged_geometry",
                                category=OutputFailureKind.optional_artifact,
                                writer=_output_writer,
                            )
                    if traj_path.is_file():
                        try:
                            _output_writer.record(traj_path)
                        except Exception as _traj_exc:
                            warn_writer_failure(
                                _traj_exc,
                                traj_path,
                                role="nonconverged_trajectory",
                                category=OutputFailureKind.manifest_recording,
                                writer=_output_writer,
                            )
                    try:
                        _output_writer.crash(
                            wall_seconds=time.perf_counter() - t_job_start
                        )
                    except Exception as _crash_exc:
                        warn_output_failure(
                            _crash_exc,
                            _output_writer.manifest_path,
                            role="manifest_crash_status",
                            category=OutputFailureKind.manifest_recording,
                        )
                    _checkpointer.finalize(
                        "failed",
                        molecule=molecule,
                        method=resolved_method,
                        basis=basis,
                        functional=functional,
                    )
                    raise
                t_opt = time.perf_counter() - t0
                if _ase_result_holder:
                    _optimized_single_point_result = _ase_result_holder[0]
                write("  Optimized geometry\n")
                write(_geom_summary(molecule) + "\n\n")
                flush()

                # Read trajectory frames for QVF archive.
                _traj_frames = []
                _traj_energies = []
                try:
                    from ase.io.trajectory import Trajectory
                    from ase.units import Bohr, Hartree

                    from ._vibeqc_core import Atom

                    traj = Trajectory(str(traj_path))
                    for atoms in traj:
                        pos_bohr = atoms.positions / Bohr
                        frame_mol = Molecule(
                            [
                                Atom(int(z), list(xyz))
                                for z, xyz in zip(atoms.numbers, pos_bohr)
                            ],
                            molecule.charge,
                            molecule.multiplicity,
                        )
                        _traj_frames.append(frame_mol)
                        try:
                            e_eV = atoms.get_potential_energy()
                            if e_eV is not None and e_eV == e_eV:
                                _traj_energies.append(float(e_eV) / Hartree)
                        except Exception as _traj_e_exc:
                            # ASE calculator may not expose energies
                            # per frame; compatibility shim -- never
                            # crash on this.
                            warn_output_failure(
                                _traj_e_exc,
                                traj_path,
                                role="trajectory_frame_energy",
                                category=OutputFailureKind.compatibility_fallback,
                            )
                except Exception as _traj_read_exc:
                    # The traj file was written by ASE just above but
                    # we can't read it back -- log it (the QVF archive
                    # will lose its trajectory section but the SCF
                    # itself is fine).
                    warn_output_failure(
                        _traj_read_exc,
                        traj_path,
                        role="trajectory_read_for_qvf",
                        category=OutputFailureKind.optional_artifact,
                    )
                    _traj_frames = []
                    _traj_energies = []
            elif _opt_backend == "native":
                # Native scipy L-BFGS-B backend.
                from .molecular_optimize import optimize_molecule as _native_opt

                write(
                    f"  Geometry optimization (native L-BFGS-B)\n"
                    f"  gtol = {_conv_grad:.2e} Ha/bohr, max_steps = {max_opt_steps}\n\n"
                )
                flush()
                with (
                    plog.stage(
                        "geometry_optimization",
                        detail=f"L-BFGS-B, gtol={_conv_grad:.1e} Ha/bohr",
                    ),
                    PerfScope("geometry_optimization"),
                    _molecular_progress_scope(
                        _handle_cpp_scf_progress,
                        phase="geom-opt.scf",
                    ),
                ):
                    t0 = time.perf_counter()
                    _output_writer.update_progress(phase="geom-opt", iteration=0)
                    _opt_result = _native_opt(
                        molecule,
                        basis,
                        method=resolved_method,
                        functional=functional,
                        rhf_options=rhf_options,
                        uhf_options=uhf_options,
                        rks_options=rks_options,
                        uks_options=uks_options,
                        rohf_options=rohf_options, roks_options=roks_options,
                        cisd_options=cisd_options,
                        selected_ci_options=selected_ci_options,
                        dmrg_options=dmrg_options,
                        v2rdm_options=v2rdm_options,
                        transcorrelated_options=transcorrelated_options,
                        casci_options=casci_options,
                        caspt2_options=caspt2_options,
                        nevpt2_options=nevpt2_options,
                        casscf_options=casscf_options,
                        active_space=active_space,
                        cas_reference=cas_reference,
                        max_iter=max_opt_steps,
                        conv_tol_grad=_conv_grad,
                        dispersion_params=d3_params,
                        solvent=solvent,
                        record_trajectory=bool(output_qvf),
                        progress=False,
                        grid_level=grid_level,
                    )
                    t_opt = time.perf_counter() - t0
                molecule = _opt_result.system
                _ecp_follow_optimized_geometry(molecule)
                _traj_frames = _opt_result.trajectory_frames or None
                _traj_energies = _opt_result.trajectory_energies or None

                write("  Optimized geometry\n")
                write(_geom_summary(molecule) + "\n\n")
                flush()

            elif _opt_backend == "brent":
                # Brent steepest-descent + line-search backend.
                from .molecular_optimize import optimize_molecule_brent as _brent_opt

                write(
                    f"  Geometry optimization (Brent steepest-descent)\n"
                    f"  gtol = {_conv_grad:.2e} Ha/bohr, max_steps = {max_opt_steps}\n\n"
                )
                flush()
                with (
                    plog.stage(
                        "geometry_optimization",
                        detail=f"Brent, gtol={_conv_grad:.1e} Ha/bohr",
                    ),
                    PerfScope("geometry_optimization"),
                    _molecular_progress_scope(
                        _handle_cpp_scf_progress,
                        phase="geom-opt.scf",
                    ),
                ):
                    t0 = time.perf_counter()
                    _output_writer.update_progress(phase="geom-opt", iteration=0)
                    _opt_result = _brent_opt(
                        molecule,
                        basis,
                        method=resolved_method,
                        functional=functional,
                        rhf_options=rhf_options,
                        uhf_options=uhf_options,
                        rks_options=rks_options,
                        uks_options=uks_options,
                        rohf_options=rohf_options, roks_options=roks_options,
                        cisd_options=cisd_options,
                        selected_ci_options=selected_ci_options,
                        dmrg_options=dmrg_options,
                        v2rdm_options=v2rdm_options,
                        transcorrelated_options=transcorrelated_options,
                        casci_options=casci_options,
                        caspt2_options=caspt2_options,
                        nevpt2_options=nevpt2_options,
                        casscf_options=casscf_options,
                        active_space=active_space,
                        cas_reference=cas_reference,
                        max_iter=max_opt_steps,
                        conv_tol_grad=_conv_grad,
                        dispersion_params=d3_params,
                        solvent=solvent,
                        record_trajectory=bool(output_qvf),
                        progress=False,
                        grid_level=grid_level,
                    )
                    t_opt = time.perf_counter() - t0
                molecule = _opt_result.system
                _ecp_follow_optimized_geometry(molecule)
                _traj_frames = _opt_result.trajectory_frames or None
                _traj_energies = _opt_result.trajectory_energies or None
                write("  Optimized geometry\n")
                write(_geom_summary(molecule) + "\n\n")
                flush()
            else:
                raise ValueError(f"Unknown optimizer_backend={_opt_backend!r}")

        # Semiempirical and MLIP methods don't use a Gaussian basis; skip
        # BasisSet construction. Semiempirical routes still allocate valence
        # matrices, SCC histories, and route-specific scratch, so estimate
        # those without a basis object.
        if resolved_method in SEMIEMPIRICAL_METHODS:
            basis_obj = None
            estimate = estimate_semiempirical_memory(
                molecule,
                method=resolved_method,
                nddo=nddo,
                solvent=solvent,
                optimize=optimize,
                max_steps=max_opt_steps,
                ccm_options=ccm_options,
            )
        elif resolved_method in _MLIP_METHODS:
            basis_obj = None
            estimate = None
        else:
            with PerfScope("basis_set_construction"):
                basis_obj = BasisSet(molecule, basis)

            if method in ("mp2", "scs-mp2", "sos-mp2") and not _use_rohf_mp2:
                if molecule.multiplicity > 1:
                    if _ump2_options_for_run is None:
                        raise RuntimeError(
                            "internal error: UMP2 policy was not materialized"
                        )
                    _auto_resolve_mp2_aux_basis(
                        _ump2_options_for_run, basis_obj, molecule
                    )
                else:
                    if _mp2_options_for_run is None:
                        raise RuntimeError(
                            "internal error: MP2 policy was not materialized"
                        )
                    _auto_resolve_mp2_aux_basis(
                        _mp2_options_for_run, basis_obj, molecule
                    )

            if (
                method in ("ccsd", "ccsd(t)")
                or _effective_method in _BCCD_METHODS
                or _effective_method in _CC_VARIANT_METHODS
            ):
                if _ccsd_options_for_run is None:
                    raise RuntimeError(
                        "internal error: CCSD policy was not materialized"
                    )
                # Estimate the route that will actually run. A DF and a
                # conventional CCSD(T) have wildly different footprints,
                # so an estimate taken on the wrong one is worse than no
                # estimate (BUG 117).
                _ccsd_options_for_run.density_fit = _resolve_cc_density_fit(
                    density_fit, ccsd_options
                )
                if _ccsd_options_for_run.density_fit:
                    # BUG 113: the run_job aux_basis kwarg reaches the CC
                    # route. Stamp it on the pre-flight copy before
                    # resolve_aux_basis so the memory estimate and the
                    # execution both model the same auxiliary basis.
                    if aux_basis:
                        _ccsd_options_for_run.aux_basis = (
                            _resolve_route_aux_basis(
                                aux_basis,
                                getattr(_ccsd_options_for_run, "aux_basis", ""),
                                route="ccsd_options",
                            )
                        )
                    _ccsd_options_for_run.resolve_aux_basis(basis)

            # Memory pre-flight
            _estimate_method = _memory_estimator_method(
                resolved_method,
                _effective_method,
            )
            est_opts = _memory_estimator_options(
                _estimate_method,
                resolved_method=resolved_method,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                rohf_options=rohf_options,
                roks_options=roks_options,
                mp2_options=_mp2_options_for_run,
                ump2_options=(
                    _rohf_mp2_options_for_estimate
                    if _use_rohf_mp2
                    else _ump2_options_for_run
                ),
                ccsd_options=_ccsd_options_for_run or ccsd_options,
                cc3_options=cc3_options,
                ccsdt_options=ccsdt_options,
                dlpno_options=_dlpno_options_for_run,
                dlpno_ccsd_options=_dlpno_ccsd_options_for_run,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                active_space=active_space,
                selected_ci_options=selected_ci_options,
                tddft=tddft,
                tddft_n_states=tddft_n_states,
                tddft_type=tddft_type,
                functional=functional,
            )
            estimate = estimate_memory(
                molecule,
                basis_obj,
                method=_estimate_method,
                options=est_opts,
            )
            if estimate is not None and _native_options_for_run:
                # The public budget is process-wide and includes retained SCF
                # matrices plus the estimator's safety headroom. Native
                # planners own only the post-SCF workspace, so reserve those
                # two pieces before choosing an occupied slab or triples tile.
                _native_budget, _budget_changed = (
                    _refine_native_correlated_budgets(
                        _native_options_for_run,
                        _correlated_budget_bytes,
                        estimate,
                    )
                )
                if _budget_changed:
                    # Re-plan with the method-owned allowance. This is the
                    # estimate shown to the user and admitted below.
                    estimate = estimate_memory(
                        molecule,
                        basis_obj,
                        method=_estimate_method,
                        options=est_opts,
                    )
                # An explicit method cap remains an execution contract even
                # when the platform cannot report a process-wide allocation.
                # Always validate it before SCF.
                for _native_option in _native_options_for_run:
                    _check_native_correlated_budget(estimate, _native_option)
        if estimate is not None:
            _memory_admission_bytes = (
                _correlated_budget_bytes
                if _correlated_budget_bytes
                and (
                    _mp2_options_for_run is not None
                    or _ump2_options_for_run is not None
                    or _ccsd_options_for_run is not None
                    or _dlpno_options_for_run is not None
                    or _dlpno_ccsd_options_for_run is not None
                    or _use_rohf_mp2
                )
                else None
            )
            write(
                "  "
                + format_memory_report(
                    estimate,
                    override_requested=memory_override,
                    available=_memory_admission_bytes,
                ).replace("\n", "\n  ")
                + "\n\n"
            )
            flush()
            _slog.emit(
                "memory_estimate",
                total_gb=float(estimate.total_gb),
                raw_total_bytes=int(estimate.raw_total_bytes),
                headroom_factor=float(estimate.headroom_factor),
                requested_budget_bytes=(
                    int(_memory_admission_bytes)
                    if _memory_admission_bytes is not None
                    else None
                ),
                native_workspace_budget_bytes=(
                    int(_native_budget)
                    if _memory_admission_bytes is not None and _native_budget
                    else None
                ),
                by_category={k: int(v) for k, v in estimate.by_category.items()},
                phase_peaks={
                    k: int(v) for k, v in estimate.phase_peaks.items()
                },
            )
            check_memory(
                estimate,
                allow_exceed=memory_override,
                available=_memory_admission_bytes,
            )

        if resolved_method in _MLIP_METHODS:
            plog.info(
                f"Evaluating {resolved_method.upper()} pre-trained MLIP "
                "(single forward pass; no SCF)."
            )
        else:
            plog.info(
                f"Starting molecular SCF ({resolved_method.upper()}"
                + (f" / {functional}" if functional else "")
                + ")."
            )
        # Memory snapshot before SCF: lets the perf log show RSS
        # growth caused by the integral / Fock build, separate from
        # the basis-set / pre-flight overhead.
        from .perf import active_tracker as _at

        _tracker = _at()
        if _tracker is not None:
            _tracker.snapshot_memory("start_of_scf")
        t0 = time.perf_counter()
        _scf_progress_start = len(_cpp_scf_callbacks)
        _headline_attempt_floor = _cpp_scf_attempt
        # A large-RKS TRAH recovery starts from the terminal density of the
        # first attempt by setting the retry options to READ.  Keep the
        # original physical guess route for citations: that density still
        # descends from SAP/SAD/etc., while the retry options remain
        # authoritative for convergence-method reporting.
        _scf_guess_citation_override_active = False
        _scf_guess_citation_override: Optional[str] = None
        _scf_options_used = {
            "rhf": rhf_options,
            "uhf": uhf_options,
            "rks": rks_options,
            "uks": uks_options,
            "rohf": rohf_options,
            "roks": roks_options,
        }.get(resolved_method)
        # Crash-dump scope: if the C++ SCF raises (NaN, linear
        # dependence, memory error, ...), we want a snapshot of the
        # last known state on disk before the exception bubbles up
        # and the user has nothing to debug from. The state we can
        # capture from outside the C++ call is limited (no live
        # density / Fock) -- we record the geometry, options, and
        # phase. Any deeper introspection needs a future C++ hook.
        try:
            # density_fit / cosx / aux_basis wiring happened above, before
            # the dry-run short-circuit and the memory pre-flight (BUG 117).
            with (
                PerfScope(f"scf.{resolved_method}"),
                _molecular_progress_scope(_handle_cpp_scf_progress),
            ):
                # SCF options the single point actually ran with (ECP fields
                # attached), for the Z_eff-aware properties below (#642).
                _used_scf_options: dict = {}
                if (
                    resolved_method == "pm6"
                    and _optimized_single_point_result is not None
                ):
                    result = _optimized_single_point_result
                else:
                    result = _run_single_point(
                        resolved_method,
                        molecule,
                        basis_obj,
                        functional=functional,
                        rhf_options=rhf_options,
                        uhf_options=uhf_options,
                        rks_options=rks_options,
                        uks_options=uks_options,
                        rohf_options=rohf_options,
                        roks_options=roks_options,
                        cisd_options=cisd_options,
                        cc3_options=cc3_options,
                        ccsdt_options=ccsdt_options,
                        selected_ci_options=selected_ci_options,
                        dmrg_options=dmrg_options,
                        v2rdm_options=v2rdm_options,
                        transcorrelated_options=transcorrelated_options,
                        casci_options=casci_options,
                        caspt2_options=caspt2_options,
                        nevpt2_options=nevpt2_options,
                        casscf_options=casscf_options,
                        active_space=active_space,
                        cas_reference=cas_reference,
                        mlip_options=mlip_options,
                        ccm_options=ccm_options,
                        solvent=solvent,
                        dft_plus_u=dft_plus_u,
                        read_from=read_from,
                        fragments=fragments,
                        nddo=nddo,
                        grid_level=grid_level,
                        used_scf_options_out=_used_scf_options,
                    )
            _scf_options_used = (
                _used_scf_options.get("scf_options") or _scf_options_used
            )
            # Record the canonicalisation actually selected after the
            # frontier-shell occupation model is known.  This matters for
            # ROKS: a symmetry-degenerate fractional shell retains the
            # symmetry-preserving Roothaan fractional-shell spectrum.
            _canonicalization = getattr(result, "rohf_canonicalization", None)
            if _canonicalization is not None:
                _output_writer.update_run_fields(
                    {"rohf_canonicalization": str(_canonicalization)}
                )

            # Bind semiempirical parameter provenance to the exact native
            # result before any post-SCF processing.  Do not reconstruct it
            # from a mutable parameter builder after execution.
            _parameter_identity = getattr(result, "parameter_identity", None)
            _parameter_sha256 = getattr(result, "parameter_sha256", None)
            if (_parameter_identity is None) != (_parameter_sha256 is None):
                raise RuntimeError(
                    "molecular semiempirical result has incomplete parameter "
                    "identity provenance"
                )
            if _parameter_identity is not None:
                _parameter_identity = str(_parameter_identity)
                _parameter_sha256 = str(_parameter_sha256)
                if (
                    not _parameter_identity
                    or len(_parameter_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in _parameter_sha256
                    )
                ):
                    raise RuntimeError(
                        "molecular semiempirical result has invalid parameter "
                        "identity provenance"
                    )
            _prov = getattr(result, "parameter_provenance", None)
            if _prov is not None:
                _output_writer.update_run_fields(_prov)
        except BaseException as _scf_exc:
            t_scf = time.perf_counter() - t0
            record(
                "scf_failed",
                f"\n  SCF FAILED: {type(_scf_exc).__name__}: {_scf_exc}\n",
                exception_type=type(_scf_exc).__name__,
                exception=str(_scf_exc),
                wall_s=float(t_scf),
            )
            flush()
            if _crash_target is not None:
                _opts_for_dump = (
                    {
                        "rhf": rhf_options,
                        "uhf": uhf_options,
                        "rks": rks_options,
                        "uks": uks_options,
                    }.get(resolved_method)
                    if resolved_method in ("rhf", "uhf", "rks", "uks")
                    else None
                )
                dump_on_failure(
                    _crash_target,
                    _scf_exc,
                    {
                        "phase": f"scf.{resolved_method}",
                        "n_iters_completed": (
                            len(_cpp_scf_callbacks) - _scf_progress_start
                        ),
                    },
                    options=_opts_for_dump,
                    molecule=molecule,
                )
                write(
                    f"  Crash dump written to "
                    f"{stem_sibling(_crash_target, '.dump').name}\n"
                )
                flush()
            # Flip the manifest status to "crashed" before the
            # exception propagates -- a ``vq`` daemon polling
            # ``{output}.system`` then sees the job aborted rather
            # than mistaking a stale "running" for a live job.
            try:
                _output_writer.crash(wall_seconds=t_scf)
            except Exception as _crash_exc:
                # Manifest bookkeeping must never mask the real SCF
                # exception we're about to re-raise.
                warn_output_failure(
                    _crash_exc,
                    _output_writer.manifest_path,
                    role="manifest_crash_status",
                    category=OutputFailureKind.manifest_recording,
                )
            # Label the checkpoint QVF "failed" so a live viewer stops
            # watching (mirrors the OutputWriter.crash status flip).
            _checkpointer.finalize(
                "failed",
                molecule=molecule,
                method=resolved_method,
                basis=basis,
                functional=functional,
            )
            # Re-raise: crash_dump=True does NOT swallow the failure;
            # it just makes the failure debuggable.
            raise
        t_scf = time.perf_counter() - t0
        if (
            not bool(getattr(result, "converged", False))
            and basis_obj is not None
            and _rhf_tail_sad_retry_supported(
                resolved_method=resolved_method,
                molecule=molecule,
                basis=basis_obj,
                rhf_options=rhf_options,
                result=result,
            )
        ):
            _retry_source_selection = getattr(result, "guess_selection", None)
            if _retry_source_selection is None:
                from .guess import select_initial_guess
                _retry_source_selection = select_initial_guess(
                    molecule, rhf_options.initial_guess,
                )
            retry_opts = _clone_rhf_options_for_sad_retry(rhf_options)
            record(
                "scf_retry",
                "\n  Auto convergence profile: large closed-shell RHF "
                "returned non-converged; restarting once from the SAD "
                "initial guess.\n",
                method="rhf",
                reason="large_rhf_tail_nonconvergence",
                initial_n_iter=int(getattr(result, "n_iter", 0)),
                initial_energy=float(getattr(result, "energy", 0.0)),
                retry_initial_guess="sad",
                retry_max_iter=int(retry_opts.max_iter),
            )
            flush()
            t_retry0 = time.perf_counter()
            _retry_progress_start = len(_cpp_scf_callbacks)
            try:
                with (
                    PerfScope("scf.rhf.sad_retry"),
                    _molecular_progress_scope(
                        _handle_cpp_scf_progress,
                        phase="scf.rhf.sad_retry",
                    ),
                ):
                    result = run_rhf(molecule, basis_obj, retry_opts)
                    from ._vibeqc_core import GuessSelection
                    result.guess_selection = GuessSelection(
                        _retry_source_selection.requested,
                        InitialGuess.SAD, InitialGuess.SAD,
                    )
            except BaseException as _retry_exc:
                t_scf += time.perf_counter() - t_retry0
                record(
                    "scf_retry_failed",
                    f"\n  SCF RETRY FAILED: {type(_retry_exc).__name__}: {_retry_exc}\n",
                    exception_type=type(_retry_exc).__name__,
                    exception=str(_retry_exc),
                    wall_s=float(t_scf),
                )
                flush()
                if _crash_target is not None:
                    dump_on_failure(
                        _crash_target,
                        _retry_exc,
                        {
                            "phase": "scf.rhf.sad_retry",
                            "n_iters_completed": (
                                len(_cpp_scf_callbacks) - _retry_progress_start
                            ),
                        },
                        options=retry_opts,
                        molecule=molecule,
                    )
                try:
                    _output_writer.crash(wall_seconds=t_scf)
                except Exception as _crash_exc:
                    warn_output_failure(
                        _crash_exc,
                        _output_writer.manifest_path,
                        role="manifest_crash_status",
                        category=OutputFailureKind.manifest_recording,
                    )
                _checkpointer.finalize(
                    "failed",
                    molecule=molecule,
                    method=resolved_method,
                    basis=basis,
                    functional=functional,
                )
                raise
            t_retry = time.perf_counter() - t_retry0
            t_scf += t_retry
            rhf_options = retry_opts
            _scf_options_used = retry_opts
            record(
                "scf_retry_done",
                "  Auto convergence profile: RHF SAD restart "
                f"{'converged' if bool(getattr(result, 'converged', False)) else 'did not converge'} "
                f"in {int(getattr(result, 'n_iter', 0))} iterations; "
                f"E = {render_energy_labeled(float(getattr(result, 'energy', 0.0)), width=0, precision=10)}.\n",
                method="rhf",
                converged=bool(getattr(result, "converged", False)),
                n_iter=int(getattr(result, "n_iter", 0)),
                energy=float(getattr(result, "energy", 0.0)),
                wall_s=float(t_retry),
            )
            flush()
        if (
            not bool(getattr(result, "converged", False))
            and basis_obj is not None
            and _rks_tail_trah_retry_supported(
                resolved_method=resolved_method,
                molecule=molecule,
                basis=basis_obj,
                functional=functional,
                rks_options=rks_options,
                result=result,
            )
        ):
            _scf_guess_citation_override = _detect_scf_guess(
                resolved_method,
                molecule,
                _scf_options_used,
            )
            _scf_guess_citation_override_active = True
            retry_opts = _clone_rks_options_for_trah_retry(rks_options, result)
            # Ordinary RKS now preflights its bounded grid batches, but this
            # conditional TRAH recovery route still constructs a dense f_xc
            # kernel. Recheck the route-specific peak before arming it so a
            # calculation admitted on the batched estimate cannot fail later
            # with an unannounced whole-grid allocation.
            retry_estimate = estimate_memory(
                molecule,
                basis_obj,
                method="rks",
                options=retry_opts,
            )
            record(
                "memory_estimate",
                "\n  Auto convergence profile: dense TRAH retry memory "
                "preflight\n  "
                + format_memory_report(
                    retry_estimate,
                    override_requested=memory_override,
                ).replace("\n", "\n  ")
                + "\n\n",
                phase="scf.rks.trah_retry",
                total_gb=float(retry_estimate.total_gb),
                raw_total_bytes=int(retry_estimate.raw_total_bytes),
                headroom_factor=float(retry_estimate.headroom_factor),
                by_category={
                    key: int(value)
                    for key, value in retry_estimate.by_category.items()
                },
            )
            flush()
            check_memory(retry_estimate, allow_exceed=memory_override)
            record(
                "scf_retry",
                "\n  Auto convergence profile: large closed-shell RKS "
                "returned non-converged; retrying from the final density "
                "with TRAH (trah_threshold=1.0e-2).\n",
                method="rks",
                reason="large_rks_tail_nonconvergence",
                initial_n_iter=int(getattr(result, "n_iter", 0)),
                initial_energy=float(getattr(result, "energy", 0.0)),
                trah_threshold=float(retry_opts.trah_threshold),
                retry_max_iter=int(retry_opts.max_iter),
            )
            flush()
            t_retry0 = time.perf_counter()
            _retry_progress_start = len(_cpp_scf_callbacks)
            try:
                with (
                    PerfScope("scf.rks.trah_retry"),
                    _molecular_progress_scope(
                        _handle_cpp_scf_progress,
                        phase="scf.rks.trah_retry",
                    ),
                ):
                    _retry_source_selection = getattr(result, "guess_selection", None)
                    if _retry_source_selection is None:
                        from .guess import select_initial_guess
                        _retry_source_selection = select_initial_guess(
                            molecule, rks_options.initial_guess,
                        )
                    result = run_rks(molecule, basis_obj, retry_opts)
                    from ._vibeqc_core import GuessSelection
                    result.guess_selection = GuessSelection(
                        _retry_source_selection.requested,
                        _retry_source_selection.effective, InitialGuess.READ,
                    )
            except BaseException as _retry_exc:
                t_scf += time.perf_counter() - t_retry0
                record(
                    "scf_retry_failed",
                    f"\n  SCF RETRY FAILED: {type(_retry_exc).__name__}: {_retry_exc}\n",
                    exception_type=type(_retry_exc).__name__,
                    exception=str(_retry_exc),
                    wall_s=float(t_scf),
                )
                flush()
                if _crash_target is not None:
                    dump_on_failure(
                        _crash_target,
                        _retry_exc,
                        {
                            "phase": "scf.rks.trah_retry",
                            "n_iters_completed": (
                                len(_cpp_scf_callbacks) - _retry_progress_start
                            ),
                        },
                        options=retry_opts,
                        molecule=molecule,
                    )
                try:
                    _output_writer.crash(wall_seconds=t_scf)
                except Exception as _crash_exc:
                    warn_output_failure(
                        _crash_exc,
                        _output_writer.manifest_path,
                        role="manifest_crash_status",
                        category=OutputFailureKind.manifest_recording,
                    )
                _checkpointer.finalize(
                    "failed",
                    molecule=molecule,
                    method=resolved_method,
                    basis=basis,
                    functional=functional,
                )
                raise
            t_retry = time.perf_counter() - t_retry0
            t_scf += t_retry
            rks_options = retry_opts
            _scf_options_used = retry_opts
            record(
                "scf_retry_done",
                "  Auto convergence profile: RKS TRAH retry "
                f"{'converged' if bool(getattr(result, 'converged', False)) else 'did not converge'} "
                f"in {int(getattr(result, 'n_iter', 0))} iterations; "
                f"E = {render_energy_labeled(float(getattr(result, 'energy', 0.0)), width=0, precision=10)}.\n",
                method="rks",
                converged=bool(getattr(result, "converged", False)),
                n_iter=int(getattr(result, "n_iter", 0)),
                energy=float(getattr(result, "energy", 0.0)),
                wall_s=float(t_retry),
            )
            flush()
        if _tracker is not None:
            _tracker.snapshot_memory("end_of_scf")
            # Also record the per-iteration trace so the perf log
            # carries the same info as the .out's SCF trace, in
            # tabular form. The C++ result.scf_trace is the canonical
            # source for these numbers.
            for step in getattr(result, "scf_trace", []) or []:
                # wall_s == 0.0 is the C++ "unmeasured" sentinel; pass
                # None so the perf table renders "--" instead of
                # fabricating a 0.000 s measurement (BUG87-A).
                _step_wall = float(getattr(step, "wall_s", 0.0))
                _tracker.add_scf_iter(
                    iter=int(step.iter),
                    energy=float(step.energy),
                    dE=float(step.delta_e) if step.iter > 1 else None,
                    grad=float(step.grad_norm),
                    diis=int(step.diis_subspace),
                    wall_s=_step_wall if _step_wall > 0.0 else None,
                )
        # Structured-log compatibility replay. Native RHF/UHF/RKS/UKS rows
        # now land live through the diagnostics callback; other result
        # producers may still expose only a terminal ``scf_trace``. Match at
        # full round-trip precision and replay only rows not already durable.
        _result_scf_trace = list(getattr(result, "scf_trace", []) or [])
        _canonical_cpp_rows = [
            _scf_progress_identity(int(step.iter), float(step.energy))
            for step in _result_scf_trace
        ]
        _headline_attempts = [
            attempt
            for attempt in _cpp_scf_attempt_rows
            if (
                attempt > _headline_attempt_floor
                and (
                    _cpp_scf_attempt_phases.get(attempt) == "scf"
                    or _cpp_scf_attempt_phases.get(
                        attempt,
                        "",
                    ).startswith("scf.")
                )
            )
        ]
        # Some wrappers, notably CPCM, concatenate several completed native
        # SCF traces into the result's canonical trace. Stability searches may
        # insert rejected candidates between selected traces. Match the
        # rightmost ordered subsequence of complete headline attempts before
        # falling back to terminal replay.
        _selected_cpp_scf_attempts = _match_scf_attempt_subsequence(
            _headline_attempts,
            _cpp_scf_attempt_rows,
            _canonical_cpp_rows,
        )
        _selected_cpp_rows_match_energy_only = False
        _partial_callback_matches: dict[
            int,
            tuple[int, tuple[int, str]],
        ] = {}
        if not _selected_cpp_scf_attempts and _canonical_cpp_rows:
            # RIJCOSX runs several native grid segments. Each segment emits
            # its own 1..N live cycle numbers, while the C++ staging wrapper
            # renumbers the returned canonical trace globally. The energies
            # remain an exact ordered fingerprint, so use them to recognize
            # those already-durable rows without replaying duplicates.
            _selected_cpp_scf_attempts = _match_scf_attempt_subsequence(
                _headline_attempts,
                _cpp_scf_attempt_rows,
                _canonical_cpp_rows,
                energy_only=True,
            )
            _selected_cpp_rows_match_energy_only = bool(
                _selected_cpp_scf_attempts
            )
        if not _selected_cpp_scf_attempts and _canonical_cpp_rows:
            # A killed or malformed callback stream can expose only a prefix
            # (or sparse subset) of an otherwise valid native attempt. Keep
            # those durable exact rows and replay just the missing canonical
            # suffix instead of duplicating the prefix after return.
            _partial_callback_matches = _match_scf_rows_rightmost(
                _headline_attempts,
                _cpp_scf_attempt_rows,
                _canonical_cpp_rows,
            )
            _partial_energy_matches = _match_scf_rows_rightmost(
                _headline_attempts,
                _cpp_scf_attempt_rows,
                _canonical_cpp_rows,
                energy_only=True,
            )
            if len(_partial_energy_matches) > len(
                _partial_callback_matches
            ):
                _partial_callback_matches = _partial_energy_matches
                _selected_cpp_rows_match_energy_only = True
            for _canonical_index in sorted(_partial_callback_matches):
                _matched_attempt = _partial_callback_matches[
                    _canonical_index
                ][0]
                if _matched_attempt not in _selected_cpp_scf_attempts:
                    _selected_cpp_scf_attempts.append(_matched_attempt)
        _selected_cpp_scf_attempt = (
            _selected_cpp_scf_attempts[0]
            if len(_selected_cpp_scf_attempts) == 1
            else None
        )
        if _partial_callback_matches:
            _selected_cpp_row_context: list[
                tuple[int, str, tuple[int, str]] | None
            ] = [None] * len(_canonical_cpp_rows)
            for _canonical_index, (
                _matched_attempt,
                _matched_identity,
            ) in _partial_callback_matches.items():
                _selected_cpp_row_context[_canonical_index] = (
                    _matched_attempt,
                    _cpp_scf_attempt_phases[_matched_attempt],
                    _matched_identity,
                )
            # Missing rows inherit the nearest matched attempt only for
            # provenance; their canonical identity cannot be removed from
            # the durable-live multiset, so the replay below still emits it.
            for _canonical_index, _context in enumerate(
                _selected_cpp_row_context
            ):
                if _context is not None:
                    continue
                _neighbor = next(
                    (
                        candidate
                        for candidate in reversed(
                            _selected_cpp_row_context[:_canonical_index]
                        )
                        if candidate is not None
                    ),
                    None,
                )
                if _neighbor is None:
                    _neighbor = next(
                        (
                            candidate
                            for candidate in _selected_cpp_row_context[
                                _canonical_index + 1 :
                            ]
                            if candidate is not None
                        ),
                        None,
                    )
                if _neighbor is not None:
                    _selected_cpp_row_context[_canonical_index] = (
                        _neighbor[0],
                        _neighbor[1],
                        _canonical_cpp_rows[_canonical_index],
                    )
        else:
            _selected_cpp_row_context = [
                (
                    attempt,
                    _cpp_scf_attempt_phases[attempt],
                    row,
                )
                for attempt in _selected_cpp_scf_attempts
                for row in _cpp_scf_attempt_rows[attempt]
            ]
        _unmatched_live_cpp_rows = {
            attempt: list(_live_cpp_scf_attempt_rows.get(attempt, []))
            for attempt in _selected_cpp_scf_attempts
        }
        for _step_index, step in enumerate(_result_scf_trace):
            _step_identity = _scf_progress_identity(
                int(step.iter),
                float(step.energy),
            )
            _replay_context = (
                _selected_cpp_row_context[_step_index]
                if _step_index < len(_selected_cpp_row_context)
                else None
            )
            if _replay_context is not None:
                _attempt, _phase, _expected_identity = _replay_context
                _identity_matches = _expected_identity == _step_identity
                if _selected_cpp_rows_match_energy_only:
                    _identity_matches = (
                        _expected_identity[1] == _step_identity[1]
                    )
                if _identity_matches:
                    try:
                        _unmatched_live_cpp_rows[_attempt].remove(
                            _expected_identity
                        )
                    except ValueError:
                        pass
                    else:
                        continue
                _replay_fields: dict[str, object] = {
                    "phase": _phase,
                    "attempt": _attempt,
                }
            else:
                _replay_fields = {"phase": "scf"}
            try:
                _slog.emit(
                    "scf_iter",
                    iter=int(step.iter),
                    energy=float(step.energy),
                    dE=(float(step.delta_e) if step.iter > 1 else None),
                    grad_norm=float(step.grad_norm),
                    diis_subspace=int(step.diis_subspace),
                    **_replay_fields,
                )
            except Exception:
                # Compatibility replay obeys the same optional-sink contract
                # as live delivery: a failed NDJSON writer cannot fail SCF.
                pass
        # Stamp the final SCF progress into the .system manifest so
        # vq status / monitors can see iteration + energy from a
        # single file read.
        _trace = getattr(result, "scf_trace", None)
        if _trace:
            _last = _trace[-1]
            _headline_progress_fields = dict(
                phase="scf",
                iteration=int(_last.iter),
                energy_eh=float(_last.energy),
                gradient_norm=float(_last.grad_norm),
                diis_subspace=int(_last.diis_subspace),
                **(
                    {"selected_attempt": _selected_cpp_scf_attempt}
                    if _selected_cpp_scf_attempt is not None
                    else {}
                ),
            )
            _output_writer.update_progress(**_headline_progress_fields)
        _executed_guess_selection = getattr(result, "guess_selection", None)
        if _executed_guess_selection is not None:
            requested = _executed_guess_selection.requested.name
            effective = _executed_guess_selection.effective.name
            transport = _executed_guess_selection.transport.name
            label = requested if requested == effective else f"{requested} -> {effective}"
            if transport != effective:
                label += f" (transport: {transport})"
            write(f"  initial_guess = {label}\n", Level.QUIET)
            _output_writer.update_run_fields({
                "initial_guess_requested": requested,
                "initial_guess": effective,
                "initial_guess_transport": transport,
            })
        # Non-convergence path: write a crash dump too, since "ran
        # to max_iter" is a real failure mode. We don't raise -- the
        # SCF returned a result and the user can inspect it -- but we
        # leave the .dump alongside the .out so the diagnostic recipe
        # is the same as the exception path.
        if not bool(getattr(result, "converged", False)) and _crash_target is not None:
            _nonconv_kind = "Solver" if isinstance(result, SolverResult) else "SCF"
            _nonconv_phase = _nonconv_kind.lower()
            dump_state = {
                "phase": f"{_nonconv_phase}.{resolved_method} (max_iter)",
                "scf_trace": list(getattr(result, "scf_trace", []) or []),
                "n_iters_completed": int(getattr(result, "n_iter", 0)),
            }
            for k_state, k_attr in (
                ("density", "density"),
                ("density_alpha", "density_alpha"),
                ("density_beta", "density_beta"),
                ("fock", "fock"),
                ("fock_alpha", "fock_alpha"),
                ("fock_beta", "fock_beta"),
                ("mo_coeffs", "mo_coeffs"),
                ("mo_coeffs_alpha", "mo_coeffs_alpha"),
                ("mo_coeffs_beta", "mo_coeffs_beta"),
            ):
                v = getattr(result, k_attr, None)
                if v is not None:
                    dump_state[k_state] = v
            _opts_for_dump = (
                {
                    "rhf": rhf_options,
                    "uhf": uhf_options,
                    "rks": rks_options,
                    "uks": uks_options,
                }.get(resolved_method)
                if resolved_method in ("rhf", "uhf", "rks", "uks")
                else None
            )
            dump_on_failure(
                _crash_target,
                None,
                dump_state,
                phase=f"{_nonconv_phase}.{resolved_method} (max_iter)",
                options=_opts_for_dump,
                molecule=molecule,
            )
            write(
                f"\n  {_nonconv_kind} did not converge -- crash dump written to "
                f"{stem_sibling(_crash_target, '.dump').name}\n"
            )
            flush()

        # Per-analysis success flags from the .out properties block, so the
        # citation assembler below can gate best-effort analyses (Hirshfeld)
        # on whether they actually computed + surfaced for this job. Stays
        # empty on the solver-trace / MLIP branches (no properties block),
        # which correctly suppresses the Hirshfeld citation there.
        _prop_status: dict[str, bool] = {}

        # Check if we got a SolverResult (non-mean-field method)
        if isinstance(result, SolverResult):
            _output_writer.update_progress(
                phase="solver",
                iteration=int(getattr(result, "n_iter", 0)),
                energy_eh=float(getattr(result, "energy", 0.0)),
                method=str(getattr(result, "method", "unknown")),
            )
            write("\n" + section_header("Non-mean-field solver result"))
            if isinstance(result, CC3Result):
                _cc3_summary, _cc3_iterations = _format_cc3_result(result)
                write(_cc3_summary + "\n", Level.QUIET)
                write("\n" + _cc3_iterations + "\n", Level.VERBOSE)
            elif isinstance(result, CCSDTResult):
                _ccsdt_summary, _ccsdt_iterations = _format_ccsdt_result(result)
                write(_ccsdt_summary + "\n", Level.QUIET)
                write("\n" + _ccsdt_iterations + "\n", Level.VERBOSE)
            else:
                write(_format_solver_trace(result) + "\n")
        elif getattr(result, "model_info", None) is not None:
            # MLIP (method="mace"): a single pre-trained forward pass -- there
            # is NO SCF here, no Fock matrix, no DIIS. Emitting the empty SCF
            # iteration table (||[F,DS]|| / DIIS columns, "converged in 1
            # iterations") would misrepresent what MACE does, so print the
            # single-shot energy instead, then the model provenance block
            # (model + license + energy-scale caveat) so the .out is self-
            # documenting about the external model behind these numbers.
            write(
                "\n"
                + section_header(
                    "Single-point energy -- pre-trained MLIP forward pass (no SCF)"
                )
            )
            write(f"  E = {render_energy_labeled(float(result.energy), width=0, precision=10)}\n")
            # _format_mlip_provenance opens with its own separator rule.
            write(
                _format_mlip_provenance(
                    result.model_info,
                    getattr(result, "model_options", None),
                )
                + "\n"
            )
        else:
            _scf_energy_label = (
                "SCF energy"
                if d3_params is not None or use_d4 or composite_recipe is not None
                else "Total energy"
            )
            # A solvated run's headline energy is the in-solvent total
            # (IID 148): label the row accordingly so it is never read as
            # the gas-phase SCF total.  The SCF component rows above it
            # remain the in-field SCF decomposition; the solvation block
            # below reconciles the two explicitly.
            if getattr(result, "energy_in_solvent", None) is not None:
                _scf_energy_label = "In-solvent total"
            # A finite-temperature semiempirical result (SCC-DFTB retry, GFN2
            # ladder rung) reports the Mermin free energy
            # E_free = E_internal - T*S, not the internal energy: label
            # it so a smeared result is never mistaken for a zero-T total.
            if float(getattr(result, "electronic_temperature", 0.0) or 0.0) > 0.0:
                _scf_energy_label = "Mermin free energy"
            # Per-atom nuclear charges for every electrostatic property of
            # this run: Z - n_core on ECP atoms (both routes), bare Z
            # otherwise (#642). The optimizer path runs its single points
            # on the materialised route kwargs, so fall back to those.
            _nuclear_charges_used = effective_nuclear_charges(
                molecule, _scf_options_used
            )
            write_scf_trace(
                result,
                molecule=molecule,
                basis=basis_obj,
                include_banner=False,
                nuclear_charges=_nuclear_charges_used,
                # The rows were streamed live as the native SCF produced
                # them (#482); close the table rather than repeat it.
                iterations_already_streamed=_scf_rows_streamed,
                # A failed SCF has no valid derived properties.  Besides
                # contradicting the fail-closed error below, evaluating the
                # default Hirshfeld/Mayer/dipole block can repeat a large
                # grid + SAD build immediately before the result is rejected.
                include_properties=bool(getattr(result, "converged", False)),
                property_status=_prop_status,
                energy_label=_scf_energy_label,
                trailing="\n",
            )
            if _parameter_identity is not None:
                write(
                    "\n"
                    + section_header("## Semiempirical parameter identity")
                    + f"  Identity = {_parameter_identity}\n"
                    + f"  SHA-256  = {_parameter_sha256}\n"
                )
            # Finite-temperature semiempirical result: the total above is the
            # Mermin free energy at the executed temperature.  Surface the temperature,
            # E_internal vs E_free, and the -T*S term so the convention is
            # explicit in the .out.
            _e_temperature = float(getattr(result, "electronic_temperature", 0.0) or 0.0)
            if _e_temperature > 0.0:
                _e_internal = getattr(result, "e_internal", None)
                _entropy = getattr(result, "entropy", None)
                _mermin_lines = [
                    f"  Electronic temperature (Fermi-Dirac): {_e_temperature:.6f} Ha"
                    f"  (~{_e_temperature * 315775.13:.0f} K)"
                ]
                if _e_internal is not None:
                    _mermin_lines.append(
                        "  E_free (Mermin) = E_internal - T*S = "
                        f"{render_energy_labeled(float(result.energy), width=16, precision=10)}"
                    )
                    _mermin_lines.append(
                        "  E_internal                       = "
                        f"{render_energy_labeled(float(_e_internal), width=16, precision=10)}"
                    )
                if _entropy is not None:
                    _mermin_lines.append(
                        f"  -T*S = {-float(_e_temperature) * float(_entropy):.10f} Ha"
                        f"  (S = {float(_entropy):.6e})"
                    )
                write(
                    "\n"
                    + section_header("## Mermin free energy (finite electronic temperature)")
                    + "\n".join(_mermin_lines)
                    + "\n"
                )
            # MSINDO COSMO (run_job(method="msindo", solvent=...)) -- surface the
            # solvation energy + gas reference alongside the in-solvent total.
            _esolv = getattr(result, "e_solv", None)
            if _esolv is not None:
                _Hk = 627.509474063
                _eg = getattr(result, "e_gas", None)
                _energy_in_solvent = getattr(result, "energy_in_solvent", None)
                _solvent_variant = getattr(result, "solvent_variant", None)
                _solvent_result = getattr(result, "solvent_result", None)
                if _eg is None and _solvent_result is not None:
                    _eg = getattr(_solvent_result, "e_gas", None)
                if _energy_in_solvent is None and _solvent_result is not None:
                    _energy_in_solvent = getattr(_solvent_result, "energy", None)
                if _energy_in_solvent is None and _eg is not None:
                    _energy_in_solvent = float(_eg) + float(_esolv)
                if _solvent_variant is None and _solvent_result is not None:
                    _solvent_variant = getattr(_solvent_result, "solvent_variant", None)
                _solvent_label = (
                    "COSMO"
                    if str(_solvent_variant or "").strip().lower() == "cosmo"
                    else "CPCM"
                )
                # IID 148: decompose the in-solvent total exactly.  ``e_solv``
                # is the electrostatic polarisation energy (1/2 q.V_tot), which
                # is NOT in-solvent minus gas: the solute density also relaxes
                # in the field, so the two differ by the polarisation of the
                # wavefunction.  Print both so the block self-checks.
                _shift_lines = []
                if _eg is not None and _energy_in_solvent is not None:
                    _shift = float(_energy_in_solvent) - float(_eg)
                    _shift_lines.append(
                        f"  Total solvation shift   = {render_energy_labeled(_shift, width=16, precision=10)}"
                        f"  (in-solvent - gas-phase; "
                        f"includes solute polarisation "
                        f"{render_energy_labeled(_shift - float(_esolv), width=0, precision=10)})\n"
                    )
                write(
                    "\n"
                    + section_header(f"## Implicit solvation ({_solvent_label})")
                    + (
                        f"  Gas-phase total        = {render_energy_labeled(_eg, width=16, precision=10)}\n"
                        if _eg is not None
                        else ""
                    )
                    + (
                        f"  In-solvent total       = {render_energy_labeled(_energy_in_solvent, width=16, precision=10)}\n"
                        if _energy_in_solvent is not None
                        else ""
                    )
                    + f"  Solvation energy (1/2 q.V_tot) = {render_energy_labeled(_esolv, width=16, precision=10)}"
                    f"  ({_esolv * _Hk:9.2f} kcal/mol)\n"
                    + "".join(_shift_lines)
                    + "\n"
                )
                _slog.emit(
                    "e_solv", value=float(_esolv), value_kcal=float(_esolv * _Hk)
                )
            # MSINDO reports an atomization/binding energy (E - S atomic
            # reference, ATENG) directly -- surface it, the semiempirical
            # analogue of the mean-field atomization block.
            _be = getattr(result, "binding_energy", None)
            if _be is not None:
                _Hk = 627.509474063
                record(
                    "binding_energy",
                    "\n" + section_header("## Atomization / binding energy")
                    + f"  Binding energy (E - S ATENG) = {render_energy_labeled(_be, width=16, precision=10)}"
                    f"  ({_be * _Hk:9.2f} kcal/mol)\n\n",
                    value=float(_be),
                    value_kcal=float(_be * _Hk),
                )
        flush()
        if resolved_method in _MLIP_METHODS:
            plog.info(
                "MLIP energy evaluated (single forward pass, no SCF); "
                f"E = {render_energy_labeled(float(getattr(result, 'energy', 0.0)), width=0, precision=10)}"
            )
        else:
            _converged_payload: dict[str, object] = {
                "n_iter": int(getattr(result, "n_iter", 0)),
                "energy": float(getattr(result, "energy", 0.0)),
                "converged": bool(getattr(result, "converged", False)),
            }
            if _selected_cpp_scf_attempt is not None:
                _converged_payload["selected_attempt"] = (
                    _selected_cpp_scf_attempt
                )
            elif len(_selected_cpp_scf_attempts) > 1:
                _converged_payload["selected_attempts"] = (
                    _selected_cpp_scf_attempts
                )
            _slog.emit("scf_converged", **_converged_payload)
            with _structured_log_ctx(enabled=False):
                plog.converged(
                    n_iter=int(_converged_payload["n_iter"]),
                    energy=float(_converged_payload["energy"]),
                    converged=bool(_converged_payload["converged"]),
                )
        # Spin-squared diagnostic for unrestricted wavefunctions (BUG 77).
        # Emitted to the structured log whenever the result carries the
        # C++-computed <S^2> value.  The human-readable .out line is handled
        # by output/formats/scf_log.py; the structured event is the
        # machine-readable counterpart.
        _s2 = getattr(result, "s_squared", None)
        _s2_ideal = getattr(result, "s_squared_ideal", None)
        _spin_diagnostic = None
        if _s2 is not None and _s2_ideal is not None:
            _spin_diagnostic = {
                "s_squared": float(_s2),
                "s_squared_ideal": float(_s2_ideal),
                "s_squared_deviation": float(_s2) - float(_s2_ideal),
            }
            _slog.emit(
                "spin_squared",
                value=_spin_diagnostic["s_squared"],
                ideal=_spin_diagnostic["s_squared_ideal"],
                deviation=_spin_diagnostic["s_squared_deviation"],
            )
            _output_writer.update_run_fields(_spin_diagnostic)

        # Apply molecular smearing post-SCF: compute Mermin free energy
        # A = E - T·S from the converged eigenvalues. Only engaged for
        # mean-field methods (which carry mo_energies).
        if smearing_temperature > 0.0:
            result = _apply_molecular_smearing(
                result,
                n_electrons=effective_electron_count(molecule, result),
                smearing_temperature=smearing_temperature,
                smearing_method=smearing_method,
            )

        _scf_route_nonconverged = resolved_method in (
            "rhf",
            "uhf",
            "rks",
            "uks",
            "rohf",
            "roks",
        ) and not bool(getattr(result, "converged", False))
        _semiempirical_nonconverged = (
            resolved_method in SEMIEMPIRICAL_METHODS
            and not bool(getattr(result, "converged", False))
        )
        if _scf_route_nonconverged or _semiempirical_nonconverged:
            if _semiempirical_nonconverged:
                _method_label = str(
                    getattr(result, "method", resolved_method)
                ).upper()
                _msg = (
                    f"{_method_label} semiempirical SCF did not converge after "
                    f"{int(getattr(result, 'n_iter', 0))} iterations; refusing "
                    "to treat the energy or derived properties as a successful "
                    "calculation."
                )
                _event = "semiempirical_scf_nonconverged"
            else:
                _msg = (
                    f"{resolved_method.upper()} SCF did not converge after "
                    f"{int(getattr(result, 'n_iter', 0))} iterations; refusing "
                    "to treat the energy or post-SCF properties as a successful "
                    "calculation."
                )
                _event = "scf_nonconverged"
            record(
                _event,
                f"\n  FATAL: {_msg}\n",
                method=resolved_method,
                n_iter=int(getattr(result, "n_iter", 0)),
                energy=float(getattr(result, "energy", 0.0)),
            )
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
            _checkpointer.finalize(
                "failed",
                molecule=molecule,
                method=resolved_method,
                basis=basis,
                functional=functional,
            )
            raise RuntimeError(_msg)

        if (
            isinstance(result, SolverResult)
            and resolved_method in ("v2rdm", "cc3", "ccsdt")
            and not bool(getattr(result, "converged", False))
        ):
            _msg = (
                f"{result.method} did not converge after {result.n_iter} "
                "iterations; refusing to treat the solver energy as a "
                "successful calculation."
            )
            record(
                "solver_nonconverged",
                f"\n  FATAL: {_msg}\n",
                method=str(result.method),
                n_iter=int(result.n_iter),
                energy=float(result.energy),
            )
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
            _checkpointer.finalize(
                "failed",
                molecule=molecule,
                method=resolved_method,
                basis=basis,
                functional=functional,
            )
            raise RuntimeError(_msg)

        # ---- SCF diagnostics: unresolved internal instability ---------
        # The molecular UHF/UKS drivers run a post-convergence internal
        # stability analysis (cpp/src/uhf.cpp; Lehtola 2020 Sec. 10 /
        # Seeger-Pople 1977) and escape detected saddles by
        # rotate-and-reconverge restarts. RHF/RKS carry the VERDICT-only
        # restricted-stability surface (issue #144; Seeger-Pople 1977 /
        # Bauernschmitt-Ahlrichs 1996) — they detect but never escape.
        # When the returned solution is STILL internally unstable (escape
        # failed, retries exhausted, a deliberate state-targeting run, or
        # a restricted route whose verdict-only surface cannot escape),
        # the energy is an excited SCF solution -- warn loudly on top of
        # the .out verdict line.
        if bool(getattr(result, "internal_instability", False)):
            _lam_stab = float(getattr(result, "stability_eigenvalue", 0.0))
            _stab_msg = (
                "SCF INTERNAL INSTABILITY -- the converged solution is "
                "not a local minimum (lowest orbital-rotation Hessian "
                f"eigenvalue {_lam_stab:.3e}); the reported energy is an "
                "EXCITED SCF solution. Variational comparisons (e.g. "
                "against ROHF) are invalid for this run."
            )
            write("\n")
            warn(_stab_msg, role="scf_internal_instability")
            write("\n")
            flush()
            import warnings as _warnings

            _warnings.warn("  WARNING: " + _stab_msg)
        elif getattr(result, "stability_checked", False) and not getattr(
            result, "stability_analysis_converged", False
        ):
            # GitLab #566: the stability analysis ran and returned NO VERDICT
            # (its Davidson exhausted the budget). internal_instability is the
            # fail-closed conjunction, so it is False here -- the same value a
            # certified-stable solution carries. Without this branch the run
            # is silently indistinguishable from a verified-stable one at the
            # warning surface; only the .out carries the UNVERIFIED line, and
            # a consumer reading warnings sees nothing at all.
            _lam_stab = float(getattr(result, "stability_eigenvalue", 0.0))
            _stab_msg = (
                "SCF STABILITY UNVERIFIED -- the internal-stability analysis "
                "did not converge, so this solution is neither certified "
                f"stable nor shown unstable (last eigenvalue estimate "
                f"{_lam_stab:.3e}, not a verdict). Treat a variational "
                "comparison against another method as unsupported for this "
                "run; re-run with a larger stability budget for a verdict."
            )
            write("\n")
            warn(_stab_msg, role="scf_stability_unverified")
            write("\n")
            flush()
            import warnings as _warnings

            _warnings.warn("  WARNING: " + _stab_msg)

        # ---- SCF diagnostics: "possibly conducting state" warning -----
        # If the converged HOMO-LUMO gap is below 1e-4 Ha (~3 meV) the
        # Aufbau-filled ground state is unreliable -- the user should
        # enable Fermi-Dirac smearing.
        _CONDUCTING_GAP_THRESHOLD_HA = 1.0e-4
        has_mo_energies = hasattr(result, "mo_energies")
        if has_mo_energies and bool(getattr(result, "converged", False)):
            eps = result.mo_energies
            # Determine occupation count for this method.
            n_elec = effective_electron_count(molecule, result)
            if method in ("rhf", "rks"):
                n_occ = n_elec // 2
            elif method in ("rohf", "roks"):
                # ROHF/ROKS: n_alpha occupied orbitals (closed + open).
                n_occ = (n_elec + (molecule.multiplicity - 1)) // 2
            else:
                n_occ = None
            if n_occ is not None and n_occ > 0 and n_occ < len(eps):
                gap = float(eps[n_occ] - eps[n_occ - 1])
                if gap < _CONDUCTING_GAP_THRESHOLD_HA:
                    _ev_factor = 27.211386245988
                    # The HOMO-LUMO gap is a deliberate dual Ha/eV display
                    # (like the orbital table), kept hardcoded so it does not
                    # become "X eV (X eV)" under eV output units.
                    _gap_msg = (
                        f"POSSIBLY CONDUCTING STATE -- "
                        f"HOMO-LUMO gap = {gap:.6f} Ha ({gap * _ev_factor:.4f} eV). "
                        f"Consider Fermi-Dirac smearing (smearing_options=...)."
                    )
                    # The surrounding blank lines preserve the historical
                    # spacing; warn() emits the "  WARNING: ..." line itself
                    # (at QUIET, plus a structured warning event).
                    write("\n")
                    warn(_gap_msg, role="conducting_state")
                    write("\n")
                    flush()
                    import warnings as _warnings

                    _warnings.warn("  WARNING: " + _gap_msg)

        # Post-SCF Coupled-Cluster (CCSD / CCSD(T)), Brueckner CCD, and
        # closed-shell kernel variants (CCD / LCCD / LCCSD / CEPA(n) /
        # QCISD).  The closed-shell DF kernel is anchored against the
        # in-repo spin-orbital SGWB-1991 reference
        # (tests/test_ccsd_anchor.py, tests/test_cc_variants.py); dense
        # O(N^6) pilot, small molecules only (see handovers/HANDOVER_CCSD.md).
        if (
            method in ("ccsd", "ccsd(t)")
            or method in _BCCD_METHODS
            or method in _CC_VARIANT_METHODS
        ) and bool(getattr(result, "converged", False)):
            from .cc import (
                CCSDOptions,
                _copy_ccsd_options,
                chemical_core_orbital_count,
            )
            from .cc import run_bccd as _run_bccd_py
            from .cc import run_ccsd as _run_ccsd_py
            from .cc import run_rohf_ccsd as _run_rohf_ccsd_py
            from .cc import run_uccsd as _run_uccsd_py

            if _ccsd_options_for_run is not None:
                # The preflight copy already carries the exact frozen-core,
                # memory-budget, triples, and integral-route choices modeled
                # by estimate_memory(). Copy once more so execution-side
                # normalization cannot mutate the estimator input.
                cc_opts = _copy_ccsd_options(_ccsd_options_for_run)
            elif ccsd_options is None:
                cc_opts = CCSDOptions()
                # Default frozen core: one chemical-core count per atom
                # (noble-gas cores; H2O -> 1, CO2 -> 3).
                cc_opts.n_frozen_core = chemical_core_orbital_count(
                    molecule, result
                )
                # Internal default, not user intent: keep the .out /
                # structured-log provenance label on "(chemical-core
                # default)" (the CCSDOptions.n_frozen_core setter marks
                # every write as explicit).
                cc_opts._n_frozen_core_explicit = False
            else:
                cc_opts = _copy_ccsd_options(ccsd_options)
                if (
                    getattr(cc_opts, "fno", False)
                    and int(cc_opts.n_frozen_core) == 0
                    and not getattr(cc_opts, "_n_frozen_core_explicit", False)
                ):
                    # run_job's CCSD(T) default is frozen-core. Keep the
                    # FNO convenience path on the same convention unless the
                    # caller uses the lower-level run_fno_ccsd API directly.
                    cc_opts.n_frozen_core = chemical_core_orbital_count(
                        molecule, result
                    )
                    # Same as above: a runner-applied default stays
                    # labelled "(chemical-core default)".
                    cc_opts._n_frozen_core_explicit = False
            _variant_method = (
                _effective_method
                if (
                    _effective_method in _CC_VARIANT_METHODS
                    or _effective_method in _BCCD_METHODS
                )
                else method
            )
            cc_opts.compute_triples = bool(_ccsd_compute_triples)
            cc_opts.triples_variant = _ccsd_triples_variant
            # Coupled-pair / QCI variants: select the kernel variant;
            # _select_method guaranteed a closed-shell RHF reference.
            cc_opts.cc_variant = (
                "bccd"
                if _variant_method in _BCCD_METHODS
                else _CC_VARIANT_METHODS.get(_variant_method, "ccsd")
            )
            # Same resolution the aux gate and the memory pre-flight used;
            # everything downstream (the DF- label, the structured-log
            # provenance, the citation bundle) reads it off cc_opts.
            cc_opts.density_fit = _resolve_cc_density_fit(
                density_fit, ccsd_options
            )
            if cc_opts.density_fit:
                # Defence in depth: _ccsd_options_for_run usually already
                # carries the kwarg (stamped in the pre-flight block), but
                # when it is None this branch builds cc_opts fresh and the
                # kwarg must still reach the route (BUG 113).
                if aux_basis:
                    cc_opts.aux_basis = _resolve_route_aux_basis(
                        aux_basis,
                        getattr(cc_opts, "aux_basis", ""),
                        route="ccsd_options",
                    )
                cc_opts.resolve_aux_basis(basis)

            # Open-shell references (multiplicity > 1) route to the UHF
            # reference and the spin-orbital UCCSD(T) kernel by default.
            # ccsd_reference='rohf' selects an ROHF SCF + ROHF-CCSD(T)
            # (spin-pure restricted-open-shell reference, same kernel).
            # Closed shell uses the RHF kernel. Both share CCSDOptions /
            # CCSDResult and the same chemical-core default.
            _open_shell = molecule.multiplicity > 1
            _cc_open_shell_used = _open_shell
            _use_rohf = _open_shell and ccsd_reference == "rohf"
            # The DF- prefix is provenance: only claim it when density
            # fitting is actually on. density_fit=False runs the canonical
            # exact four-index integral route.
            _df_prefix = "DF-" if cc_opts.density_fit else ""
            _cc_used_density_fit = bool(cc_opts.density_fit)
            _cc_used_fno = bool(getattr(cc_opts, "fno", False))
            if _variant_method in _BCCD_METHODS:
                _cc_label = _df_prefix + _BCCD_LABELS[_variant_method]
            elif _variant_method in _CC_VARIANT_METHODS:
                _cc_label = _df_prefix + _CC_VARIANT_LABELS[_variant_method]
            else:
                if _ccsd_triples_variant == "a-ccsd(t)":
                    _cc_label = _df_prefix + "A-CCSD(T)"
                else:
                    _t_suffix = ""
                    if cc_opts.compute_triples:
                        _t_suffix = (
                            "[T]"
                            if _ccsd_triples_variant == "[t]"
                            else "(T)"
                        )
                    _cc_label = _df_prefix + (
                        "ROHF-CCSD"
                        if _use_rohf
                        else ("UCCSD" if _open_shell else "CCSD")
                    ) + _t_suffix
            _cc_fc_request = _route_frozen_core_request(
                frozen_core, cc_opts
            )
            _cc_fc_label, _cc_fc_convention = _frozen_core_label(
                _cc_fc_request
            )
            # Machine-readable integral-route declaration for the L7
            # convention gate (#23): a DF-vs-conventional parity pair must
            # be refusable from the .system manifest alone. Written BEFORE
            # the CC step runs, through the documented late-bound [run]
            # extension point (OutputWriter.update_run_fields; precedent:
            # #344 runtime_backend, #426 band edges), so even a run whose
            # CC step crashes still records the route it selected. A
            # conventional run records exactly False / exactly "".
            _output_writer.update_run_fields(
                {
                    "cc_density_fit": _cc_used_density_fit,
                    "cc_auxiliary_basis": str(cc_opts.aux_basis or ""),
                    "cc_algorithm": _cc_label,
                    "frozen_core_orbitals": int(cc_opts.n_frozen_core),
                    "frozen_core_convention": _cc_fc_convention,
                }
            )
            with (
                plog.stage("ccsd", detail=_cc_label),
                PerfScope("ccsd"),
            ):
                if _variant_method in _BCCD_METHODS:
                    cc_result = _run_bccd_py(molecule, basis_obj, result, cc_opts)
                elif _use_rohf:
                    cc_result = _run_rohf_ccsd_py(molecule, basis_obj, result, cc_opts)
                elif _open_shell:
                    cc_result = _run_uccsd_py(molecule, basis_obj, result, cc_opts)
                else:
                    cc_result = _run_ccsd_py(molecule, basis_obj, result, cc_opts)

            _cc_triples_memory_mode_used = str(
                getattr(cc_result, "triples_memory_mode_used", "") or ""
            )

            _slog.emit(
                "ccsd_converged",
                cc_variant=str(getattr(cc_opts, "cc_variant", "ccsd")),
                converged=bool(cc_result.converged),
                n_iter=int(cc_result.n_iter),
                n_frozen_core=int(cc_opts.n_frozen_core),
                frozen_core_convention=_cc_fc_convention,
                algorithm=_cc_label,
                density_fit=bool(cc_opts.density_fit),
                aux_basis=str(cc_opts.aux_basis or ""),
                e_ccsd_correlation=float(cc_result.e_ccsd_correlation),
                e_ccsd=float(cc_result.e_ccsd),
                e_t=float(cc_result.e_t) if cc_opts.compute_triples else None,
                e_total=float(cc_result.e_total),
                t1_norm=float(cc_result.t1_norm),
                t2_norm=float(cc_result.t2_norm),
                triples_memory_mode=_cc_triples_memory_mode_used or None,
                triples_tile_size=int(
                    getattr(cc_result, "triples_tile_size_used", 0) or 0
                ),
                triples_threads=int(
                    getattr(cc_result, "triples_threads_used", 0) or 0
                ),
                triples_workspace_bytes=int(
                    getattr(cc_result, "triples_workspace_bytes", 0) or 0
                ),
                triples_disk_bytes=int(
                    getattr(cc_result, "triples_disk_bytes", 0) or 0
                ),
                brueckner_iterations=(
                    int(cc_result.brueckner_iterations)
                    if hasattr(cc_result, "brueckner_iterations")
                    else None
                ),
                brueckner_t1_norm=(
                    float(cc_result.brueckner_t1_norm)
                    if hasattr(cc_result, "brueckner_t1_norm")
                    else None
                ),
            )

            # Write the CC iteration table.  _cc_kind is the plain method
            # label ("CCSD", "CCSD(T)", "CCD", "CEPA(1)", ...) used in the
            # block header and the energy lines below.
            _is_bccd = _variant_method in _BCCD_METHODS
            _is_pair_variant = _variant_method in _CC_VARIANT_METHODS or _is_bccd
            if _is_bccd:
                _cc_kind = _BCCD_LABELS[_variant_method]
                _base_cc_kind = "BCCD"
                _block_title = "Brueckner Coupled-Cluster " + _cc_kind
            elif _variant_method in _CC_VARIANT_METHODS:
                _cc_kind = _CC_VARIANT_LABELS[_variant_method]
                _base_cc_kind = (
                    "QCISD" if _variant_method == "qcisd(t)" else _cc_kind
                )
                _block_title = (
                    "Approximate Coupled-Cluster CC2"
                    if _variant_method == "cc2"
                    else "Coupled-Pair / Coupled-Cluster " + _cc_kind
                )
            else:
                if not cc_opts.compute_triples:
                    _cc_kind = "CCSD"
                elif _ccsd_triples_variant == "[t]":
                    _cc_kind = "CCSD[T]"
                elif _ccsd_triples_variant == "a-ccsd(t)":
                    _cc_kind = "A-CCSD(T)"
                else:
                    _cc_kind = "CCSD(T)"
                _base_cc_kind = "CCSD"
                _block_title = "Coupled-Cluster " + _cc_kind
            write("\n" + section_header(_block_title))
            _e_col = (
                "E(" + _base_cc_kind + ")"
                if _is_pair_variant
                else "E(CCSD)"
            )
            # The iteration trace is a headered column table; Table sizes the
            # rule to the content (same convention as the CCSDT/CC3 blocks),
            # retiring the hand-guessed 78-dash rules.
            _cc_tbl = Table(
                [
                    Column("Iter", "<"),
                    Column(_e_col),
                    Column("dE"),
                    Column("|R1|"),
                    Column("|R2|"),
                    Column("DIIS"),
                ],
                indent=2,
            )
            for step in cc_result.cc_trace:
                _cc_tbl.add_row(
                    int(step.iter),
                    Quantity(float(step.energy), "energy"),
                    Quantity(
                        float(step.delta_e) if step.iter > 1 else 0.0,
                        "energy_delta",
                    ),
                    Quantity(float(step.r1_norm), "dimensionless"),
                    Quantity(float(step.r2_norm), "dimensionless"),
                    int(step.diis_subspace),
                )
            _cc_policy = active_policy().with_spec(
                "dimensionless", precision=6, width=0, notation="e"
            )
            write(_cc_tbl.render(_cc_policy) + "\n")
            _conv_kind = _base_cc_kind
            if cc_result.converged:
                write(
                    f"  {_conv_kind} converged in {cc_result.n_iter} iterations\n"
                )
            else:
                write(f"  {_conv_kind} DID NOT CONVERGE\n")
            write(
                f"  Frozen core orbitals = {int(cc_opts.n_frozen_core):d}"
                f" ({_cc_fc_label})\n"
            )
            write(f"  Algorithm            = {_cc_label}\n")
            write(
                "  Density fitting       = "
                f"{'on' if bool(cc_opts.density_fit) else 'off'}\n"
            )
            if bool(cc_opts.density_fit):
                write(f"  RI auxiliary basis    = {cc_opts.aux_basis}\n")
            if hasattr(cc_result, "brueckner_iterations"):
                write(
                    "  Brueckner iterations = "
                    f"{cc_result.brueckner_iterations:d}; "
                    f"final CCSD |T1| = {cc_result.brueckner_t1_norm:.6e}\n"
                )
            # FNO summary (only present on an FNOCCSDResult).
            if hasattr(cc_result, "n_virtual_kept"):
                write(
                    f"  FNO: kept {cc_result.n_virtual_kept}/"
                    f"{cc_result.n_virtual_total} virtual natural orbitals; "
                    f"delta-MP2 correction = {cc_result.delta_mp2:+.6e} Ha\n"
                )
            write(
                f"  E({_conv_kind} correlation) = "
                f"{render_energy_labeled(cc_result.e_ccsd_correlation, width=16, precision=10)}\n"
                f"  E({_conv_kind})             = {render_energy_labeled(cc_result.e_ccsd, width=16, precision=10)}\n"
            )
            if cc_opts.compute_triples:
                _e_t4 = float(getattr(cc_result, "e_t4", 0.0))
                _e_t5 = float(getattr(cc_result, "e_t5_st", 0.0))
                if _ccsd_triples_variant == "[t]":
                    write(
                        f"  E[T] correction      = {render_energy_labeled(cc_result.e_t, width=16, precision=10)}\n"
                        f"    (omitted 5th-order singles-triples E_ST = "
                        f"{render_energy_labeled(_e_t5, width=0, precision=10, sign=True)})\n"
                        f"  E({_cc_kind})           = "
                        f"{render_energy_labeled(cc_result.e_ccsd_t, width=16, precision=10)}\n"
                    )
                elif _ccsd_triples_variant == "a-ccsd(t)":
                    write(
                        f"  E(A-T) correction    = {render_energy_labeled(cc_result.e_t, width=16, precision=10)}\n"
                    )
                    if hasattr(cc_result, "lambda_residual_norm"):
                        lambda_doc = OutputDocument(indent=2, label_width=20)
                        lambda_doc.scalar(
                            "Lambda residual =",
                            Quantity(
                                cc_result.lambda_residual_norm,
                                "dimensionless",
                            ),
                        )
                        lambda_policy = active_policy().with_spec(
                            "dimensionless",
                            precision=6,
                            width=16,
                            notation="e",
                        )
                        write(
                            lambda_doc.render(lambda_policy) + "\n",
                            Level.VERBOSE,
                        )
                    write(
                        f"  E({_cc_kind})        = "
                        f"{render_energy_labeled(cc_result.e_ccsd_t, width=16, precision=10)}\n"
                    )
                else:
                    write(
                        f"  E(T) correction      = {render_energy_labeled(cc_result.e_t, width=16, precision=10)}\n"
                    )
                    if _e_t4 != 0.0 or _e_t5 != 0.0:
                        write(
                            f"    (4th-order E[T] = {render_energy_labeled(_e_t4, width=0, precision=10, sign=True)}, "
                            f"5th-order E_ST = {render_energy_labeled(_e_t5, width=0, precision=10, sign=True)})\n"
                        )
                    write(
                        f"  E({_cc_kind})           = "
                        f"{render_energy_labeled(cc_result.e_ccsd_t, width=16, precision=10)}\n"
                    )
            write(
                f"  T1 norm              = {cc_result.t1_norm:16.10f}\n"
                f"  T2 norm              = {cc_result.t2_norm:16.10f}\n"
            )
            write("\n")
            flush()

            # Expose the CC result without mutating the pybind11 SCF
            # struct (def_readonly fields, no __dict__).
            result = _CCSDAugmented(result, cc_result, method=_effective_method)

        # Post-SCF Moller-Plesset MP2 (+ spin-component-scaled SCS / SOS).
        # RHF-reference only (closed-shell guarded early). The native C++ MP2
        # carries the c_os / c_ss spin scaling; the factors are taken from the
        # single source of truth in vibeqc.correlation so the run_job variants
        # and the general/MSINDO kernels never drift. The unscaled e_os / e_ss
        # are always reported so a user can re-derive any variant from one run.
        if method in ("mp2", "scs-mp2", "sos-mp2") and bool(
            getattr(result, "converged", False)
        ):
            from .correlation import _MP2_SCALES

            _os_s, _ss_s = _MP2_SCALES[method]
            # Closed shell -> restricted MP2; open shell -> native UMP2 (UHF
            # reference). Both carry the c_os/c_ss spin-component scaling and
            # report e_hf / e_correlation / e_total; the spin channels differ
            # (RMP2: e_os/e_ss; UMP2: e_ab opposite-spin, e_aa+e_bb same-spin).
            _open = molecule.multiplicity > 1
            _mp2_route_options = (
                None
                if _use_rohf_mp2
                else (
                    _ump2_options_for_run
                    if _open
                    else _mp2_options_for_run
                )
            )
            _mp2_fc_request = _route_frozen_core_request(
                frozen_core, _mp2_route_options
            )
            from .correlation_conventions import resolve_frozen_core_count

            # Resolved against the converged reference so an ECP's removed
            # cores are not frozen a second time (published selector only;
            # an explicit integer passes through unchanged).
            _mp2_n_frozen = resolve_frozen_core_count(
                molecule, _mp2_fc_request, reference=result
            )
            _mp2_fc_label, _mp2_fc_convention = _frozen_core_label(
                _mp2_fc_request
            )
            _output_writer.update_run_fields(
                {
                    "frozen_core_orbitals": int(_mp2_n_frozen),
                    "frozen_core_convention": _mp2_fc_convention,
                }
            )
            if _use_rohf_mp2:
                # Open-shell, mp2_reference='rohf': spin-pure semicanonical
                # ROHF-MP2 (pure-Python kernel, vibeqc.cc.run_rohf_mp2). The
                # c_os/c_ss scaling applies to the doubles; the Knowles singles
                # term (which UMP2 lacks) rides in e_corr and is reported below.
                from .cc import run_rohf_mp2
                from .density_fitting import default_aux_basis_for

                _detail = "ROHF-" + method.upper()
                _rohf_mp2_aux = str(
                    getattr(
                        _rohf_mp2_options_for_estimate,
                        "aux_basis",
                        "",
                    )
                    or ""
                )
                if not _rohf_mp2_aux:
                    _rohf_mp2_aux = default_aux_basis_for(
                        basis_obj.name,
                        kind="ri",
                    )
                _mp2_used_density_fit = True
                with plog.stage("mp2", detail=_detail), PerfScope("mp2"):
                    mp2_result = run_rohf_mp2(
                        molecule,
                        basis_obj,
                        result,
                        n_frozen_core=_mp2_n_frozen,
                        aux_basis=_rohf_mp2_aux,
                        os_scale=_os_s,
                        ss_scale=_ss_s,
                    )
                _e_os = float(mp2_result.e_os)
                _e_ss = float(mp2_result.e_ss)
            else:
                if _open:
                    from ._vibeqc_core import run_ump2 as _run_ump2

                    _mp2_opts = _ump2_options_for_run or UMP2Options()
                else:
                    from ._vibeqc_core import run_mp2 as _run_mp2

                    _mp2_opts = _mp2_options_for_run or MP2Options()
                _mp2_opts.c_os = _os_s
                _mp2_opts.c_ss = _ss_s
                _mp2_opts.n_frozen_core = int(_mp2_n_frozen)
                _mp2_used_density_fit = bool(getattr(_mp2_opts, "density_fit", False))
                _detail = (
                    ("RI-" if _mp2_used_density_fit else "")
                    + ("U" if _open else "")
                    + method.upper()
                )
                with plog.stage("mp2", detail=_detail), PerfScope("mp2"):
                    if _open:
                        mp2_result = _run_ump2(molecule, basis_obj, result, _mp2_opts)
                    else:
                        mp2_result = _run_mp2(molecule, basis_obj, result, _mp2_opts)
                _mp2_memory_mode_used = str(
                    getattr(mp2_result, "memory_mode_used", "") or ""
                )
                # Uniform opposite-/same-spin view across R/U (Grimme channels:
                # opposite = ab, same = aa+bb).
                _e_os = float(mp2_result.e_ab if _open else mp2_result.e_os)
                _e_ss = float(
                    (mp2_result.e_aa + mp2_result.e_bb) if _open else mp2_result.e_ss
                )

            _slog.emit(
                "mp2_done",
                variant=method,
                open_shell=_open,
                n_frozen_core=int(_mp2_n_frozen),
                frozen_core_convention=_mp2_fc_convention,
                c_os=float(_os_s),
                c_ss=float(_ss_s),
                density_fit=bool(_mp2_used_density_fit),
                aux_basis=(
                    str(getattr(_mp2_opts, "aux_basis", ""))
                    if not _use_rohf_mp2
                    else _rohf_mp2_aux
                ),
                memory_mode=_mp2_memory_mode_used or None,
                workspace_bytes=int(
                    getattr(mp2_result, "workspace_bytes", 0) or 0
                ),
                disk_bytes=int(getattr(mp2_result, "disk_bytes", 0) or 0),
                e_os=_e_os,
                e_ss=_e_ss,
                e_correlation=float(mp2_result.e_correlation),
                e_total=float(mp2_result.e_total),
            )

            _ref = "ROHF" if _use_rohf_mp2 else ("UHF" if _open else "RHF")
            write("\n" + section_header("Moller-Plesset MP2 (" + _detail + ")"))
            if _use_rohf_mp2:
                _algo_note = (
                    "RI semicanonical ROHF-MP2; "
                    f"aux_basis={_rohf_mp2_aux}"
                )
            else:
                if _mp2_used_density_fit:
                    _aux = str(getattr(_mp2_opts, "aux_basis", "") or "<auto>")
                    _algo_note = (
                        f"RI-{'U' if _open else ''}MP2; "
                        f"aux_basis={_aux}, "
                        f"memory={_mp2_memory_mode_used or 'legacy'}"
                    )
                else:
                    _storage_label = (
                        "integral-direct"
                        if _mp2_memory_mode_used in {"direct", "disk"}
                        else "in-core conventional"
                    )
                    _algo_note = (
                        f"{_storage_label} canonical "
                        f"{'U' if _open else ''}MP2; "
                        "no RI auxiliary basis, "
                        f"memory={_mp2_memory_mode_used or 'legacy'}"
                    )
            write(f"  Algorithm              = {_algo_note}\n")
            write(
                f"  Frozen core orbitals = {_mp2_n_frozen:d}"
                f" ({_mp2_fc_label})\n"
            )
            write(
                f"  E({_ref} reference)       = {render_energy_labeled(mp2_result.e_hf, width=16, precision=10)}\n"
                f"  E(OS, opposite-spin)   = {render_energy_labeled(_e_os, width=16, precision=10)}"
                f"   (c_os = {_os_s:.4f})\n"
                f"  E(SS, same-spin)       = {render_energy_labeled(_e_ss, width=16, precision=10)}"
                f"   (c_ss = {_ss_s:.4f})\n"
            )
            if _use_rohf_mp2:
                # The semicanonical ROHF reference violates Brillouin, so the
                # singles term is non-zero and is part of the correlation.
                write(
                    f"  E(singles, Brillouin)  = "
                    f"{render_energy_labeled(float(mp2_result.e_singles), width=16, precision=10)}\n"
                )
            write(
                f"  E(MP2 correlation)     = {render_energy_labeled(mp2_result.e_correlation, width=16, precision=10)}\n"
            )
            _total_label = f"E({method.upper()} total)"
            write(f"  {_total_label:<22s} = {render_energy_labeled(mp2_result.e_total, width=16, precision=10)}\n")
            write("\n")
            flush()

            # Expose the MP2 result for programmatic access. The C++ SCF
            # result has no __dict__, so we wrap it in a transparent proxy
            # (mirrors the dispersion path): result.mp2 carries the
            # MP2Result, result.energy_total is the MP2 total, and every
            # SCF attribute still forwards through.
            result = _MP2Augmented(result, mp2_result, method=_effective_method)

        # Post-SCF DLPNO-MP2 (vibeqc.dlpno.mp2): Foster-Boys-localised
        # occupieds, PAO domains with semicanonical virtual bases, PNO
        # compression with the semicanonical truncation correction, and the
        # coupled LMP2 residual iteration (Pinski et al., JCP 143, 034108
        # (2015)). Closed-shell RHF reference only for now. The RI fitting
        # basis resolves per the correlation ("ri") aux family of the
        # orbital basis; pass DLPNOMP2Options via dlpno_options to control
        # thresholds / frozen core.
        if method == "dlpno-mp2" and bool(getattr(result, "converged", False)):
            from .density_fitting import (
                DensityFitting as _PyDF,
            )
            from .density_fitting import (
                default_aux_basis_for as _aux_for,
                make_checked_aux_basis as _make_checked_aux,
            )

            # BUG 113: the run_job aux_basis kwarg reaches the route.
            # Priority: explicit kwarg > options-level aux_basis >
            # auto-resolve from the orbital basis (a disagreement between
            # the two explicit sources raises in _resolve_route_aux_basis).
            _aux_name = (
                _resolve_route_aux_basis(
                    aux_basis,
                    getattr(_dlpno_options_for_run, "aux_basis", None),
                    route="dlpno_options",
                )
                or _aux_for(basis_obj.name, kind="ri")
            )
            # #480: refuse an aux basis with no functions on a centre
            # that carries orbital functions, before any fitting work.
            # Not every DLPNO route builds a DensityFitting (the
            # local_df branch wraps the aux in a SimpleNamespace), so the
            # check belongs at construction rather than downstream.
            _aux_obj = _make_checked_aux(
                molecule,
                _aux_name,
                orbital_basis=basis_obj,
                orbital_basis_name=basis_obj.name,
                route="run_job (DLPNO RI auxiliary)",
            )

            if molecule.multiplicity > 1:
                # Open-shell: UHF-reference DLPNO-UMP2 (spin-channel resolved
                # aa/bb/ab). Auto-routed from method='dlpno-mp2'.
                from .dlpno.ump2 import run_dlpno_ump2

                _u_opts = _dlpno_options_for_run
                from .correlation_conventions import resolve_frozen_core_count
                from .dlpno import (
                    apply_dlpno_thresholds,
                    describe_dlpno_thresholds,
                )

                _u_thresholds = (
                    apply_dlpno_thresholds(
                        _u_opts, _requested_dlpno_thresholds
                    )
                    if _requested_dlpno_thresholds is not None
                    else describe_dlpno_thresholds(_u_opts)
                )
                _u_fc_request = _route_frozen_core_request(
                    frozen_core, _u_opts
                )
                _u_n_frozen = resolve_frozen_core_count(
                    molecule, _u_fc_request, reference=result
                )
                _u_opts.n_frozen = int(_u_n_frozen)
                _u_fc_label, _u_fc_convention = _frozen_core_label(
                    _u_fc_request
                )
                _output_writer.update_run_fields(
                    {
                        **_dlpno_manifest_fields(_u_opts, _u_thresholds),
                        "frozen_core_orbitals": int(_u_n_frozen),
                        "frozen_core_convention": _u_fc_convention,
                    }
                )
                with (
                    plog.stage("dlpno-ump2", detail=_aux_name),
                    PerfScope("dlpno_ump2"),
                ):
                    _df = _PyDF(basis_obj, _aux_obj, aux_basis_name=_aux_name)
                    ur = run_dlpno_ump2(molecule, basis_obj, result, _df, _u_opts)

                _n_pairs = ur.n_pairs_aa + ur.n_pairs_bb + ur.n_pairs_ab
                _slog.emit(
                    "dlpno_ump2_done",
                    aux_basis=_aux_name,
                    converged=bool(ur.converged),
                    localise=str(ur.localise),
                    n_frozen=int(ur.n_frozen),
                    frozen_core_convention=_u_fc_convention,
                    threshold_preset=str(_u_thresholds.preset),
                    threshold_requested=dict(_u_thresholds.requested),
                    threshold_applied=dict(_u_thresholds.applied),
                    threshold_settings=_dlpno_threshold_settings(
                        _u_opts, _u_thresholds
                    ),
                    threshold_unsupported=list(_u_thresholds.unsupported),
                    threshold_inactive=list(_u_thresholds.inactive),
                    n_pairs=int(_n_pairs),
                    n_screened=int(ur.n_screened),
                    e_aa=float(ur.e_aa),
                    e_bb=float(ur.e_bb),
                    e_ab=float(ur.e_ab),
                    e_correlation=float(ur.e_corr),
                    e_total=float(ur.e_total),
                )
                write(
                    "\n"
                    + section_header(
                        "DLPNO-UMP2 (UHF reference; RI: "
                        + _aux_name
                        + ")"
                    )
                )
                write(
                    f"  E(UHF reference)       = {render_energy_labeled(ur.e_hf, width=16, precision=10)}\n"
                    + _format_dlpno_thresholds(
                        _u_opts, provenance=_u_thresholds, label_width=22
                    )
                    + _format_frozen_core(
                        ur.n_frozen, _u_fc_request, label_width=22
                    )
                    + f"  localise / screened    = {ur.localise} / {ur.n_screened:d}"
                    "\n"
                    f"  pairs (aa/bb/ab)       = "
                    f"{ur.n_pairs_aa:d} / {ur.n_pairs_bb:d} / {ur.n_pairs_ab:d}\n"
                    f"  E(aa same-spin)        = {render_energy_labeled(ur.e_aa, width=16, precision=10)}\n"
                    f"  E(bb same-spin)        = {render_energy_labeled(ur.e_bb, width=16, precision=10)}\n"
                    f"  E(ab opposite-spin)    = {render_energy_labeled(ur.e_ab, width=16, precision=10)}\n"
                    f"  E(DLPNO-UMP2 corr)     = {render_energy_labeled(ur.e_corr, width=16, precision=10)}\n"
                    f"  E(DLPNO-UMP2 total)    = {render_energy_labeled(ur.e_total, width=16, precision=10)}\n"
                )
                if not ur.converged:
                    warn("coupled LMP2 iteration did not converge",
                         method="dlpno_ump2")
                write("\n")
                flush()
                result = _DLPNOUMP2Augmented(result, ur, method=_effective_method)
            else:
                from .dlpno.mp2 import run_dlpno_mp2
                from types import SimpleNamespace

                _dlpno_opts = _dlpno_options_for_run
                from .correlation_conventions import resolve_frozen_core_count
                from .dlpno import (
                    apply_dlpno_thresholds,
                    describe_dlpno_thresholds,
                )

                _dlpno_threshold_provenance = (
                    apply_dlpno_thresholds(
                        _dlpno_opts, _requested_dlpno_thresholds
                    )
                    if _requested_dlpno_thresholds is not None
                    else describe_dlpno_thresholds(_dlpno_opts)
                )
                _dlpno_fc_request = _route_frozen_core_request(
                    frozen_core, _dlpno_opts
                )
                _dlpno_n_frozen = resolve_frozen_core_count(
                    molecule, _dlpno_fc_request, reference=result
                )
                _dlpno_opts.n_frozen = int(_dlpno_n_frozen)
                _dlpno_fc_label, _dlpno_fc_convention = _frozen_core_label(
                    _dlpno_fc_request
                )
                _output_writer.update_run_fields(
                    {
                        **_dlpno_manifest_fields(
                            _dlpno_opts, _dlpno_threshold_provenance
                        ),
                        "frozen_core_orbitals": int(_dlpno_n_frozen),
                        "frozen_core_convention": _dlpno_fc_convention,
                    }
                )
                with plog.stage("dlpno-mp2", detail=_aux_name), PerfScope("dlpno_mp2"):
                    if bool(getattr(_dlpno_opts, "local_df", False)):
                        _df = SimpleNamespace(
                            orbital_basis=basis_obj,
                            aux_basis=_aux_obj,
                            aux_basis_name=_aux_name,
                            n_orb=int(basis_obj.nbasis),
                            n_aux=int(_aux_obj.nbasis),
                        )
                    else:
                        _df = _PyDF(basis_obj, _aux_obj, aux_basis_name=_aux_name)
                    dlpno_result = run_dlpno_mp2(
                        molecule, basis_obj, result, _df, _dlpno_opts
                    )

                _n_kept = dlpno_result.n_pairs
                _n_scr = dlpno_result.n_pairs_screened
                _avg_pno = (
                    sum(dlpno_result.pno_per_pair.values()) / _n_kept
                    if _n_kept
                    else 0.0
                )
                _slog.emit(
                    "dlpno_mp2_done",
                    aux_basis=_aux_name,
                    converged=bool(dlpno_result.converged),
                    n_iter=int(dlpno_result.n_iter),
                    n_frozen=int(dlpno_result.n_frozen),
                    frozen_core_convention=_dlpno_fc_convention,
                    threshold_preset=str(
                        _dlpno_threshold_provenance.preset
                    ),
                    threshold_requested=dict(
                        _dlpno_threshold_provenance.requested
                    ),
                    threshold_applied=dict(
                        _dlpno_threshold_provenance.applied
                    ),
                    threshold_settings=_dlpno_threshold_settings(
                        _dlpno_opts, _dlpno_threshold_provenance
                    ),
                    threshold_unsupported=list(
                        _dlpno_threshold_provenance.unsupported
                    ),
                    threshold_inactive=list(
                        _dlpno_threshold_provenance.inactive
                    ),
                    n_pairs=int(_n_kept),
                    n_pairs_screened=int(_n_scr),
                    avg_pno_per_pair=float(_avg_pno),
                    e_corr_iterated=float(dlpno_result.e_corr_iterated),
                    e_pno_correction=float(dlpno_result.e_pno_correction),
                    e_distant=float(dlpno_result.e_distant),
                    e_correlation=float(dlpno_result.e_corr),
                    e_total=float(dlpno_result.e_total),
                )

                write(
                    "\n"
                    + section_header(
                        "DLPNO-MP2 (Pinski 2015; RI: " + _aux_name + ")"
                    )
                )
                write(
                    f"  E(RHF reference)       = {render_energy_labeled(dlpno_result.e_hf, width=16, precision=10)}\n"
                    + _format_dlpno_thresholds(
                        _dlpno_opts,
                        provenance=_dlpno_threshold_provenance,
                        label_width=22,
                    )
                    + _format_frozen_core(
                        dlpno_result.n_frozen,
                        _dlpno_fc_request,
                        label_width=22,
                    )
                    + f"  pairs kept / screened  = {_n_kept:d} / {_n_scr:d}"
                    "\n"
                    f"  avg PNOs per pair      = {_avg_pno:14.1f}\n"
                    f"  E(iterated pairs)      = {render_energy_labeled(dlpno_result.e_corr_iterated, width=16, precision=10)}\n"
                    f"  E(PNO truncation corr) = {render_energy_labeled(dlpno_result.e_pno_correction, width=16, precision=10)}\n"
                    f"  E(distant-pair est.)   = {render_energy_labeled(dlpno_result.e_distant, width=16, precision=10)}\n"
                    f"  E(DLPNO-MP2 corr)      = {render_energy_labeled(dlpno_result.e_corr, width=16, precision=10)}\n"
                    f"  E(DLPNO-MP2 total)     = {render_energy_labeled(dlpno_result.e_total, width=16, precision=10)}\n"
                )
                if not dlpno_result.converged:
                    warn("coupled LMP2 iteration did not converge",
                         method="dlpno_mp2")
                write("\n")
                flush()

                result = _DLPNOMP2Augmented(
                    result, dlpno_result, method=_effective_method
                )

        # Post-SCF DLPNO-CCSD / CCSD(T).
        #   dlpno-ccsd    -> the reduced-scaling local solver (M3c,
        #                   vibeqc.dlpno.ccsd_local_solver): pair-PNO
        #                   amplitudes, with each residual contracted in the
        #                   atom-based extended PAO domain and projected back,
        #                   full-domain == canonical CCSD; FCI-anchored.
        #   dlpno-ccsd(t) -> the same local solver + a rotated-occupied (T1)
        #                   compatibility correction on the
        #                   converged amplitudes (vibeqc.dlpno.triples_local),
        #                   whose full-domain limit equals canonical CCSD(T)
        #                   to machine precision while finite TNO domains
        #                   remain truncated. The spin-orbital generalization
        #                   of Guo et al. (2018) Eq. (2) is the distinct
        #                   open-shell ``t1-iterative`` route; Guo et al.
        #                   (2020) is its open-shell method-family source.
        #                   The O(N^6) correctness pilot is reachable by
        #                   passing a DLPNOCCSDPilotOptions.
        if (
            method in ("dlpno-ccsd", "dlpno-ccsd(t)")
            and molecule.multiplicity == 1
            and bool(getattr(result, "converged", False))
        ):
            from .density_fitting import (
                DensityFitting as _PyDF,
            )
            from .density_fitting import (
                default_aux_basis_for as _aux_for,
                make_checked_aux_basis as _make_checked_aux,
            )

            _is_t = method == "dlpno-ccsd(t)"
            # Honor an explicit aux_basis from run_job or the DLPNO options;
            # else auto-resolve from the orbital basis. Minimal / Pople bases
            # (sto-3g, 6-31g*, ...) ship no default RI aux, so an explicit
            # aux_basis is the only route to DLPNO on those (BUG 113: the
            # run_job kwarg now reaches the route, not only the SCF layer).
            _aux_name = (
                _resolve_route_aux_basis(
                    aux_basis,
                    getattr(_dlpno_ccsd_options_for_run, "aux_basis", None),
                    route="dlpno_ccsd_options",
                )
                or _aux_for(basis_obj.name, kind="ri")
            )
            # #480: refuse an aux basis with no functions on a centre
            # that carries orbital functions, before any fitting work.
            # Not every DLPNO route builds a DensityFitting (the
            # local_df branch wraps the aux in a SimpleNamespace), so the
            # check belongs at construction rather than downstream.
            _aux_obj = _make_checked_aux(
                molecule,
                _aux_name,
                orbital_basis=basis_obj,
                orbital_basis_name=basis_obj.name,
                route="run_job (DLPNO RI auxiliary)",
            )
            with plog.stage("dlpno-ccsd", detail=_aux_name), PerfScope("dlpno_ccsd"):
                _df = _PyDF(basis_obj, _aux_obj, aux_basis_name=_aux_name)
                from .dlpno.ccsd import DLPNOCCSDPilotOptions, run_dlpno_ccsd_pilot
                from .dlpno.ccsd_local_solver import (
                    LocalCCSDOptions,
                    run_local_dlpno_ccsd,
                )

                # The O(N^6) correctness pilot is opt-in (pass a
                # DLPNOCCSDPilotOptions); otherwise both variants run the
                # reduced-scaling local solver, with the local DLPNO-(T) for
                # dlpno-ccsd(t).
                if isinstance(
                    _dlpno_ccsd_options_for_run, DLPNOCCSDPilotOptions
                ):
                    _cc_opts = _dlpno_ccsd_options_for_run
                    from .correlation_conventions import (
                        resolve_frozen_core_count,
                    )
                    from .dlpno import (
                        apply_dlpno_thresholds,
                        describe_dlpno_thresholds,
                    )

                    _cc_thresholds = (
                        apply_dlpno_thresholds(
                            _cc_opts, _requested_dlpno_thresholds
                        )
                        if _requested_dlpno_thresholds is not None
                        else describe_dlpno_thresholds(_cc_opts)
                    )
                    _dlpno_cc_fc_request = _route_frozen_core_request(
                        frozen_core, _cc_opts
                    )
                    _dlpno_cc_n_frozen = resolve_frozen_core_count(
                        molecule, _dlpno_cc_fc_request, reference=result
                    )
                    _cc_opts.n_frozen = int(_dlpno_cc_n_frozen)
                    _, _dlpno_cc_fc_convention = _frozen_core_label(
                        _dlpno_cc_fc_request
                    )
                    _output_writer.update_run_fields(
                        {
                            **_dlpno_manifest_fields(
                                _cc_opts, _cc_thresholds
                            ),
                            "frozen_core_orbitals": int(
                                _dlpno_cc_n_frozen
                            ),
                            "frozen_core_convention": (
                                _dlpno_cc_fc_convention
                            ),
                        }
                    )
                    if _is_t:
                        _cc_opts.compute_triples = True
                    cc_result = run_dlpno_ccsd_pilot(
                        molecule, basis_obj, result, _df, _cc_opts
                    )
                    _engine = "pilot (Riplinger 2013; O(N^6), max_nbf-capped)"
                    if _is_t:
                        _triples_mode, _triples_algorithm = (
                            _dlpno_triples_provenance(_cc_opts)
                        )
                        _note = (
                            f"  note: {_triples_algorithm} "
                            "(max_nbf-capped), explicitly selected; the default "
                            "dlpno-ccsd(t) uses the local (T).\n"
                        )
                    else:
                        _note = (
                            "  note: O(N^6) correctness pilot "
                            "(max_nbf-capped), explicitly selected; the default "
                            "uses the reduced-scaling local solver.\n"
                        )
                else:
                    _cc_opts = _dlpno_ccsd_options_for_run
                    from .correlation_conventions import (
                        resolve_frozen_core_count,
                    )
                    from .dlpno import (
                        apply_dlpno_thresholds,
                        describe_dlpno_thresholds,
                    )

                    _cc_thresholds = (
                        apply_dlpno_thresholds(
                            _cc_opts, _requested_dlpno_thresholds
                        )
                        if _requested_dlpno_thresholds is not None
                        else describe_dlpno_thresholds(_cc_opts)
                    )
                    _dlpno_cc_fc_request = _route_frozen_core_request(
                        frozen_core, _cc_opts
                    )
                    _dlpno_cc_n_frozen = resolve_frozen_core_count(
                        molecule, _dlpno_cc_fc_request, reference=result
                    )
                    _cc_opts.n_frozen = int(_dlpno_cc_n_frozen)
                    _, _dlpno_cc_fc_convention = _frozen_core_label(
                        _dlpno_cc_fc_request
                    )
                    _output_writer.update_run_fields(
                        {
                            **_dlpno_manifest_fields(
                                _cc_opts, _cc_thresholds
                            ),
                            "frozen_core_orbitals": int(
                                _dlpno_cc_n_frozen
                            ),
                            "frozen_core_convention": (
                                _dlpno_cc_fc_convention
                            ),
                        }
                    )
                    if _is_t:
                        _cc_opts.compute_triples = True
                    cc_result = run_local_dlpno_ccsd(
                        molecule, basis_obj, result, _df, _cc_opts
                    )
                    _engine = "local reduced-scaling (M3c)"
                    if _is_t:
                        _triples_mode, _triples_algorithm = (
                            _dlpno_triples_provenance(_cc_opts)
                        )
                        _amplitude_state = (
                            "converged amplitudes"
                            if cc_result.converged
                            else "final non-converged amplitudes"
                        )
                        _note = (
                            "  note: reduced-scaling local solver; "
                            f"{_triples_algorithm} on the {_amplitude_state}. "
                            "Finite local domains remain approximate.\n"
                        )
                    else:
                        _note = (
                            "  note: reduced-scaling local solver -- full-domain "
                            "limit == canonical CCSD (FCI-anchored).\n"
                        )

            _avg_pno = (
                sum(cc_result.pno_per_pair.values()) / cc_result.n_pairs
                if cc_result.n_pairs
                else 0.0
            )
            # TNO-domain diagnostics exist only on the local solver's result
            # and only once a triples domain was actually built: the pilot
            # engine and triples_mode="exact" both leave the map empty.
            _tno_sizes = list(getattr(cc_result, "tno_per_triple", {}).values())
            _n_collapsed = int(
                getattr(cc_result, "n_degenerate_tno_triples", 0)
            )
            # coupling_radius drops whole occupied triple keys: the
            # occupied-side analogue of the TNO truncation above.
            _n_key_screened = int(
                getattr(cc_result, "n_triple_keys_screened", 0)
            )
            _triples_mode, _triples_algorithm = _dlpno_triples_provenance(
                _cc_opts, cc_result
            )
            _output_writer.update_run_fields(
                {
                    "dlpno_triples_mode": _triples_mode,
                    "dlpno_triples_algorithm": _triples_algorithm,
                }
            )
            _dlpno_triples_source_entries = _dlpno_triples_citation_entries(
                method, _cc_opts, cc_result
            )
            if _is_t and _triples_mode == "none":
                _note = f"  note: {_triples_algorithm}.\n"
            _dlpno_pno_norm = str(getattr(_cc_opts, "pno_norm", "") or "")
            _slog.emit(
                "dlpno_ccsd_done",
                variant=method,
                engine=_engine,
                aux_basis=_aux_name,
                converged=bool(cc_result.converged),
                n_iter=int(cc_result.n_iter),
                n_frozen=int(cc_result.n_frozen),
                frozen_core_convention=_dlpno_cc_fc_convention,
                threshold_preset=str(_cc_thresholds.preset),
                threshold_requested=dict(_cc_thresholds.requested),
                threshold_applied=dict(_cc_thresholds.applied),
                threshold_settings=_dlpno_threshold_settings(
                    _cc_opts, _cc_thresholds
                ),
                threshold_unsupported=list(_cc_thresholds.unsupported),
                threshold_inactive=list(_cc_thresholds.inactive),
                residual_domain=str(
                    getattr(_cc_opts, "residual_domain", "") or ""
                ),
                # #701: tcut_pno is an occupation-number threshold, so the
                # density it was applied to is part of the recipe.
                pno_norm=_dlpno_pno_norm,
                # The semicanonical MP2 correction for the iterated pairs' PNO
                # truncation moves e_corr, so it is part of the recipe too.
                pno_correction=getattr(_cc_opts, "pno_correction", None),
                e_pno_correction=float(
                    getattr(cc_result, "e_pno_correction", 0.0)
                ),
                # #689: the extended/full contraction runs once per distinct
                # (coupling set, extended domain) group; the group count and
                # the mean contraction dimension size a wave honestly.
                avg_residual_domain=float(
                    getattr(cc_result, "avg_residual_domain", 0.0)
                ),
                n_residual_domains=int(
                    getattr(cc_result, "n_residual_domains", 0)
                ),
                triples_mode=_triples_mode,
                triples_algorithm=_triples_algorithm,
                n_pairs=int(cc_result.n_pairs),
                avg_pno_per_pair=float(_avg_pno),
                n_tno_triples=len(_tno_sizes),
                avg_tno_per_triple=float(getattr(cc_result, "avg_tno", 0.0)),
                min_tno_per_triple=int(min(_tno_sizes)) if _tno_sizes else 0,
                n_degenerate_tno_triples=_n_collapsed,
                n_triple_keys_screened=_n_key_screened,
                t1_norm=float(cc_result.t1_norm),
                e_ccsd_correlation=float(cc_result.e_corr),
                e_t=float(cc_result.e_t),
                e_total=float(cc_result.e_total),
            )

            write(
                "\n"
                + section_header(
                    "DLPNO-"
                    + ("CCSD(T)" if _is_t else "CCSD")
                    + " "
                    + _engine
                    + " (RI: "
                    + _aux_name
                    + ")"
                )
            )
            write(
                f"  {'E(RHF reference)':<24s} = {render_energy_labeled(cc_result.e_hf, width=16, precision=10)}\n"
                + _format_dlpno_thresholds(
                    _cc_opts, provenance=_cc_thresholds, label_width=24
                )
                + _format_dlpno_residual_domain(_cc_opts, label_width=24)
                + _format_dlpno_pno_norm(_cc_opts, label_width=24)
                + _format_dlpno_pno_correction(_cc_opts, label_width=24)
                + _format_frozen_core(
                    cc_result.n_frozen,
                    _dlpno_cc_fc_request,
                    label_width=24,
                )
                + _format_dlpno_triples(
                    _cc_opts, cc_result, label_width=24
                )
                + f"  {'pairs / avg PNOs':<24s} = {cc_result.n_pairs:d} / {_avg_pno:.1f}"
                "\n"
                + (
                    f"  {'E(PNO truncation corr)':<24s} = "
                    f"{render_energy_labeled(float(getattr(cc_result, 'e_pno_correction', 0.0)), width=16, precision=10)}\n"
                    if getattr(_cc_opts, "pno_correction", False)
                    else ""
                )
                + f"  {'E(CCSD correlation)':<24s} = {render_energy_labeled(cc_result.e_corr, width=16, precision=10)}\n"
            )
            if _is_t and _tno_sizes:
                _screened_note = (
                    f"   ({_n_key_screened:d} screened by coupling_radius)"
                    if _n_key_screened
                    else ""
                )
                write(
                    f"  {'triples / avg TNOs':<24s} = {len(_tno_sizes):d} /"
                    f" {float(getattr(cc_result, 'avg_tno', 0.0)):.1f}"
                    f"   (smallest domain: {min(_tno_sizes):d})"
                    f"{_screened_note}\n"
                )
            if _is_t:
                write(
                    f"  {'E((T) correction)':<24s} = {render_energy_labeled(cc_result.e_t, width=16, precision=10)}\n"
                    f"  {'E(DLPNO-CCSD(T) corr)':<24s} = "
                    f"{render_energy_labeled((cc_result.e_corr + cc_result.e_t), width=16, precision=10)}\n"
                    f"  {'E(DLPNO-CCSD(T) total)':<24s} = "
                    f"{render_energy_labeled(cc_result.e_total, width=16, precision=10)}\n"
                )
            else:
                write(
                    f"  {'E(DLPNO-CCSD total)':<24s} = {render_energy_labeled(cc_result.e_total, width=16, precision=10)}\n"
                )
            write(_note)
            if not cc_result.converged:
                warn("DLPNO-CCSD iteration did not converge", method="dlpno_ccsd")
            # The historical residual_domain="pair" path over-correlates under
            # PNO truncation (HANDOVER_OPEN_BUGS_V015.md, 2026-08-03). Keep a
            # loud note when a caller deliberately opts back into it; the
            # paper's extended domain is the production default.
            if (
                isinstance(_cc_opts, LocalCCSDOptions)
                and getattr(_cc_opts, "residual_domain", "extended") == "pair"
                and getattr(_cc_opts, "tcut_pno", 0.0) > 0.0
            ):
                write(
                    "  note: residual_domain=\"pair\" (legacy opt-in) "
                    "over-correlates "
                    "under PNO truncation.\n"
                    "        Use the default residual_domain=\"extended\" "
                    "for the production local solver.\n"
                )
            if _n_collapsed:
                # A triple excitation promotes three electrons into three
                # distinct virtuals, so a domain below three cannot host one
                # and contributes exactly zero rather than approximately. The
                # energy alone cannot show this, hence the explicit warning.
                warn(
                    f"tcut_tno truncated {_n_collapsed} of {len(_tno_sizes)} "
                    "triples below three virtuals; those contribute exactly "
                    "zero to (T), so part of the correction is missing rather "
                    "than approximated. Lower tcut_tno (0 keeps the full "
                    "domain).",
                    method="dlpno_ccsd",
                )
            write("\n")
            flush()

            result = _DLPNOCCSDAugmented(result, cc_result, method=_effective_method)

        # Open-shell DLPNO-CCSD/(T): UHF-reference spin-orbital UCCSD(T).
        #
        # The production path is the reduced-scaling local solver (M3c,
        # vibeqc.dlpno.uccsd_local_solver): per-pair PNO residual evaluation,
        # full-domain == canonical UCCSD (pilot-anchored). The O(N^6)
        # correctness pilot is opt-in (pass a DLPNOUCCSDPilotOptions);
        # otherwise both variants run the local solver, with the local
        # DLPNO-(T) for dlpno-ccsd(t).
        if (
            method in ("dlpno-ccsd", "dlpno-ccsd(t)")
            and molecule.multiplicity > 1
            and bool(getattr(result, "converged", False))
        ):
            from .density_fitting import (
                DensityFitting as _PyDF,
            )
            from .density_fitting import (
                default_aux_basis_for as _aux_for,
                make_checked_aux_basis as _make_checked_aux,
            )
            from .dlpno.uccsd import (
                DLPNOUCCSDPilotOptions,
                run_dlpno_uccsd_pilot,
            )
            from .dlpno.uccsd_local_solver import (
                LocalUCCSDOptions,
                run_local_dlpno_uccsd,
            )

            _is_t = method == "dlpno-ccsd(t)"
            # Honor an explicit aux_basis from run_job or the DLPNO options;
            # else auto-resolve from the orbital basis (see the closed-shell
            # path; BUG 113 -- the run_job kwarg now reaches the route).
            _aux_name = (
                _resolve_route_aux_basis(
                    aux_basis,
                    getattr(_dlpno_ccsd_options_for_run, "aux_basis", None),
                    route="dlpno_ccsd_options",
                )
                or _aux_for(basis_obj.name, kind="ri")
            )
            # #480: refuse an aux basis with no functions on a centre
            # that carries orbital functions, before any fitting work.
            # Not every DLPNO route builds a DensityFitting (the
            # local_df branch wraps the aux in a SimpleNamespace), so the
            # check belongs at construction rather than downstream.
            _aux_obj = _make_checked_aux(
                molecule,
                _aux_name,
                orbital_basis=basis_obj,
                orbital_basis_name=basis_obj.name,
                route="run_job (DLPNO RI auxiliary)",
            )

            # The O(N^6) correctness pilot is opt-in (pass a
            # DLPNOUCCSDPilotOptions); otherwise both variants run the
            # reduced-scaling local solver, with the local DLPNO-(T) for
            # dlpno-ccsd(t).
            if isinstance(
                _dlpno_ccsd_options_for_run, DLPNOUCCSDPilotOptions
            ):
                _cc_opts = _dlpno_ccsd_options_for_run
                from .correlation_conventions import resolve_frozen_core_count
                from .dlpno import (
                    apply_dlpno_thresholds,
                    describe_dlpno_thresholds,
                )

                _cc_thresholds = (
                    apply_dlpno_thresholds(
                        _cc_opts, _requested_dlpno_thresholds
                    )
                    if _requested_dlpno_thresholds is not None
                    else describe_dlpno_thresholds(_cc_opts)
                )
                _dlpno_cc_fc_request = _route_frozen_core_request(
                    frozen_core, _cc_opts
                )
                _dlpno_cc_n_frozen = resolve_frozen_core_count(
                    molecule, _dlpno_cc_fc_request, reference=result
                )
                _cc_opts.n_frozen = int(_dlpno_cc_n_frozen)
                _, _dlpno_cc_fc_convention = _frozen_core_label(
                    _dlpno_cc_fc_request
                )
                _output_writer.update_run_fields(
                    {
                        **_dlpno_manifest_fields(_cc_opts, _cc_thresholds),
                        "frozen_core_orbitals": int(_dlpno_cc_n_frozen),
                        "frozen_core_convention": _dlpno_cc_fc_convention,
                    }
                )
                if _is_t:
                    _cc_opts.compute_triples = True
                with plog.stage("dlpno-uccsd", detail=_aux_name), PerfScope("dlpno_uccsd"):
                    _df = _PyDF(basis_obj, _aux_obj, aux_basis_name=_aux_name)
                    cc_result = run_dlpno_uccsd_pilot(
                        molecule, basis_obj, result, _df, _cc_opts
                    )
                _engine = "pilot (spin-orbital; O(N^6), max_nbf-capped)"
                _triples_mode, _triples_algorithm = (
                    _dlpno_triples_provenance(_cc_opts)
                )
                _note = (
                    "  note: O(N^6) spin-orbital correctness pilot "
                    "(max_nbf-capped)"
                    + (
                        f" using {_triples_algorithm}"
                        if _is_t
                        else ""
                    )
                    + "; the default route uses the reduced-scaling local "
                    "solver.\n"
                )
            else:
                _cc_opts = _dlpno_ccsd_options_for_run
                from .correlation_conventions import resolve_frozen_core_count
                from .dlpno import (
                    apply_dlpno_thresholds,
                    describe_dlpno_thresholds,
                )

                _cc_thresholds = (
                    apply_dlpno_thresholds(
                        _cc_opts, _requested_dlpno_thresholds
                    )
                    if _requested_dlpno_thresholds is not None
                    else describe_dlpno_thresholds(_cc_opts)
                )
                _dlpno_cc_fc_request = _route_frozen_core_request(
                    frozen_core, _cc_opts
                )
                _dlpno_cc_n_frozen = resolve_frozen_core_count(
                    molecule, _dlpno_cc_fc_request, reference=result
                )
                _cc_opts.n_frozen = int(_dlpno_cc_n_frozen)
                _, _dlpno_cc_fc_convention = _frozen_core_label(
                    _dlpno_cc_fc_request
                )
                _output_writer.update_run_fields(
                    {
                        **_dlpno_manifest_fields(_cc_opts, _cc_thresholds),
                        "frozen_core_orbitals": int(_dlpno_cc_n_frozen),
                        "frozen_core_convention": _dlpno_cc_fc_convention,
                    }
                )
                if _is_t:
                    _cc_opts.compute_triples = True
                with plog.stage("dlpno-uccsd", detail=_aux_name), PerfScope("dlpno_uccsd"):
                    _df = _PyDF(basis_obj, _aux_obj, aux_basis_name=_aux_name)
                    cc_result = run_local_dlpno_uccsd(
                        molecule, basis_obj, result, _df, _cc_opts
                    )
                _engine = "local reduced-scaling (M3c)"
                if _is_t:
                    _triples_mode, _triples_algorithm = (
                        _dlpno_triples_provenance(_cc_opts)
                    )
                    _note = (
                        "  note: reduced-scaling local solver; "
                        f"{_triples_algorithm} on the converged amplitudes. "
                        "Finite local domains remain approximate.\n"
                    )
                else:
                    _note = (
                        "  note: reduced-scaling local solver -- full-domain "
                        "limit == canonical UCCSD (pilot-anchored).\n"
                    )

            _avg_pno = float(
                getattr(cc_result, "avg_pno",
                        sum(getattr(cc_result, "pno_per_pair", {}).values()) / max(getattr(cc_result, "n_pairs", 1), 1))
            )
            _avg_pao = float(getattr(cc_result, "avg_pao", 0.0))
            _avg_singles = float(getattr(cc_result, "avg_singles_domain", 0.0))
            # TNO-domain diagnostics exist only on the local solver's result
            # and only once a triples domain was actually built.
            _tno_sizes = list(getattr(cc_result, "tno_per_triple", {}).values())
            _n_collapsed = int(
                getattr(cc_result, "n_degenerate_tno_triples", 0)
            )
            _n_triples = int(getattr(cc_result, "n_triples", 0))
            _n_key_screened = int(
                getattr(cc_result, "n_screened_triples", 0)
            )
            _triples_mode, _triples_algorithm = _dlpno_triples_provenance(
                _cc_opts, cc_result
            )
            if _triples_mode == "t1-iterative":
                # The UCCSD amplitudes come from the local M3c solver, but
                # this opt-in triples contraction currently materialises
                # dense spin-orbital source tensors under max_nbf. Do not
                # label the combined calculation as wholly reduced-scaling.
                _engine = (
                    "local CCSD (M3c) + experimental dense iterative "
                    "triples"
                )
                _note = (
                    "  note: reduced-scaling local UCCSD phase; "
                    f"{_triples_algorithm} on the converged amplitudes. "
                    "The triples phase is dense and max_nbf-capped.\n"
                )
            _output_writer.update_run_fields(
                {
                    "dlpno_triples_mode": _triples_mode,
                    "dlpno_triples_algorithm": _triples_algorithm,
                }
            )
            _dlpno_triples_source_entries = _dlpno_triples_citation_entries(
                method, _cc_opts, cc_result
            )
            if _is_t and _triples_mode == "none":
                _note = f"  note: {_triples_algorithm}.\n"
            _slog.emit(
                "dlpno_uccsd_done",
                variant=method,
                engine=_engine,
                aux_basis=_aux_name,
                converged=bool(cc_result.converged),
                n_iter=int(cc_result.n_iter),
                localise=str(getattr(cc_result, "localise", "")),
                n_frozen=int(cc_result.n_frozen),
                frozen_core_convention=_dlpno_cc_fc_convention,
                threshold_preset=str(_cc_thresholds.preset),
                threshold_requested=dict(_cc_thresholds.requested),
                threshold_applied=dict(_cc_thresholds.applied),
                threshold_settings=_dlpno_threshold_settings(
                    _cc_opts, _cc_thresholds
                ),
                threshold_unsupported=list(_cc_thresholds.unsupported),
                threshold_inactive=list(_cc_thresholds.inactive),
                residual_domain=str(
                    getattr(_cc_opts, "residual_domain", "") or ""
                ),
                triples_mode=_triples_mode,
                triples_algorithm=_triples_algorithm,
                n_pairs=int(cc_result.n_pairs),
                avg_pno_per_pair=_avg_pno,
                n_tno_triples=len(_tno_sizes),
                avg_tno_per_triple=float(getattr(cc_result, "avg_tno", 0.0)),
                min_tno_per_triple=int(min(_tno_sizes)) if _tno_sizes else 0,
                n_degenerate_tno_triples=_n_collapsed,
                n_triple_keys_screened=_n_key_screened,
                t1_norm=float(cc_result.t1_norm),
                e_ccsd_correlation=float(cc_result.e_corr),
                e_t=float(cc_result.e_t),
                e_total=float(cc_result.e_total),
            )
            write(
                "\n"
                + section_header(
                    "DLPNO-U"
                    + ("CCSD(T)" if _is_t else "CCSD")
                    + " "
                    + _engine
                    + " (RI: "
                    + _aux_name
                    + ")"
                )
            )
            write(
                f"  {'E(UHF reference)':<24s} = {render_energy_labeled(cc_result.e_hf, width=16, precision=10)}\n"
                + _format_dlpno_thresholds(
                    _cc_opts, provenance=_cc_thresholds, label_width=24
                )
                + _format_dlpno_residual_domain(_cc_opts, label_width=24)
                + _format_dlpno_pno_norm(_cc_opts, label_width=24)
                + _format_frozen_core(
                    cc_result.n_frozen,
                    _dlpno_cc_fc_request,
                    label_width=24,
                )
                + _format_dlpno_triples(
                    _cc_opts, cc_result, label_width=24
                )
                + f"  {'pairs / avg PNOs':<24s} = {cc_result.n_pairs:d} / {_avg_pno:.1f}"
            )
            if _avg_pao > 0:
                write(f"   PAO: {_avg_pao:.1f}")
            if _avg_singles > 0:
                write(f"   singles: {_avg_singles:.1f}")
            write("\n")
            write(
                f"  {'E(UCCSD correlation)':<24s} = {render_energy_labeled(cc_result.e_corr, width=16, precision=10)}\n"
            )
            if _is_t and _tno_sizes:
                _screened_note = (
                    f"   ({_n_key_screened:d} screened by coupling_radius)"
                    if _n_key_screened
                    else ""
                )
                write(
                    f"  {'triples / avg TNOs':<24s} = {_n_triples:d} /"
                    f" {float(getattr(cc_result, 'avg_tno', 0.0)):.1f}"
                    f"   (smallest domain: {min(_tno_sizes):d})"
                    f"{_screened_note}\n"
                )
            if _is_t:
                write(
                    f"  {'E((T) correction)':<24s} = {render_energy_labeled(cc_result.e_t, width=16, precision=10)}\n"
                    f"  {'E(DLPNO-UCCSD(T) corr)':<24s} = "
                    f"{render_energy_labeled((cc_result.e_corr + cc_result.e_t), width=16, precision=10)}\n"
                    f"  {'E(DLPNO-UCCSD(T) total)':<24s} = "
                    f"{render_energy_labeled(cc_result.e_total, width=16, precision=10)}\n"
                )
            else:
                write(
                    f"  {'E(DLPNO-UCCSD total)':<24s} = {render_energy_labeled(cc_result.e_total, width=16, precision=10)}\n"
                )
            write(_note)
            if not cc_result.converged:
                warn("DLPNO-UCCSD iteration did not converge", method="dlpno_uccsd")
            if _n_collapsed:
                warn(
                    f"tcut_tno truncated {_n_collapsed} of {len(_tno_sizes)} "
                    "triples below three virtuals; those contribute exactly "
                    "zero to (T), so part of the correction is missing rather "
                    "than approximated. Lower tcut_tno (0 keeps the full "
                    "domain).",
                    method="dlpno_uccsd",
                )
            write("\n")
            flush()
            result = _DLPNOCCSDAugmented(result, cc_result, method=_effective_method)

        # Post-SCF OVGF / GF2 Green's-function quasiparticle IPs / EAs. Closed
        # shell uses the spatial diagonal second-order self-energy; open shell
        # (UHF reference) uses the spin-resolved spin-orbital self-energy
        # (vibeqc.propagator) -- the electron-propagator analogue of UMP2.
        _ovgf_ok = method == "ovgf" and bool(getattr(result, "converged", False))
        if _ovgf_ok and molecule.multiplicity != 1:
            import numpy as _np

            from ._vibeqc_core import compute_eri as _compute_eri
            from .propagator import (
                unrestricted_quasiparticle_energies as _u_qp_energies,
            )

            EV = 27.211386245988
            Ca = _np.asarray(result.mo_coeffs_alpha)
            Cb = _np.asarray(result.mo_coeffs_beta)
            ea_mo = _np.asarray(result.mo_energies_alpha)
            eb_mo = _np.asarray(result.mo_energies_beta)
            nbf = len(ea_mo)
            # The spin-orbital tensor is (2.n_bf)⁴ -- guard tighter than the
            # closed-shell n_bf⁴ path.
            if 2 * nbf > 80:
                raise NotImplementedError(
                    f"open-shell method='ovgf' builds the full spin-orbital MO "
                    f"two-electron tensor (2.n_bf={2 * nbf}); it is limited to "
                    f"2.n_bf <= 80 (n_bf <= 40). Use a smaller basis."
                )
            # Valence count on an ECP reference (#740): the self-energy is
            # built over the orbitals the SCF actually filled.
            n_elec = effective_electron_count(molecule, result)
            na = (n_elec + molecule.multiplicity - 1) // 2
            nb = n_elec - na
            # a/b frontier orbitals: spin-HOMO (IP) + spin-LUMO (EA) per channel.
            orbs = []
            for _sp, _no in (("alpha", na), ("beta", nb)):
                if _no >= 1:
                    orbs.append((_sp, _no - 1))
                if _no < nbf:
                    orbs.append((_sp, _no))
            with plog.stage("ovgf", detail="open-shell GF2"), PerfScope("ovgf"):
                eri_ao = _np.asarray(_compute_eri(basis_obj))
                res = _u_qp_energies(
                    eri_ao, Ca, Cb, ea_mo, eb_mo, na, nb, orbs, iterate=True
                )

            def _occ(sp, ix):
                return ix < (na if sp == "alpha" else nb)

            ips = [-q.eps_qp * EV for sp, q in res if _occ(sp, q.orbital)]
            eas = [-q.eps_qp * EV for sp, q in res if not _occ(sp, q.orbital)]
            first_ip = min(ips) if ips else None
            ea_val = max(eas) if eas else None
            _slog.emit(
                "ovgf_done",
                open_shell=True,
                orbitals=[[sp, q.orbital] for sp, q in res],
                eps_scf=[float(q.eps_scf) for sp, q in res],
                eps_qp=[float(q.eps_qp) for sp, q in res],
                pole_strength=[float(q.pole_strength) for sp, q in res],
                ip_first_ev=(None if first_ip is None else float(first_ip)),
                ea_ev=(None if ea_val is None else float(ea_val)),
            )
            write(
                "\n"
                + section_header(
                    "OVGF / GF2 quasiparticle energies (open-shell, "
                    "spin-resolved second-order self-energy)"
                )
            )
            # Headered column table: Table sizes the rule to the content,
            # retiring the hand-guessed 78-dash rules. Columns carry eV in
            # their headers; the cells are plain 4-decimal numbers.
            _qp_tbl = Table(
                [
                    Column("spin"),
                    Column("MO"),
                    Column("occ"),
                    Column("Koopmans(eV)"),
                    Column("OVGF (eV)"),
                    Column("pole Z"),
                ],
                indent=2,
            )
            for sp, q in res:
                _qp_tbl.add_row(
                    sp,
                    int(q.orbital),
                    "occ" if _occ(sp, q.orbital) else "vir",
                    Quantity(float(q.eps_scf * EV), "dimensionless"),
                    Quantity(float(q.eps_qp * EV), "dimensionless"),
                    Quantity(float(q.pole_strength), "dimensionless"),
                )
            _qp_policy = active_policy().with_spec(
                "dimensionless", precision=4, width=0
            )
            write(_qp_tbl.render(_qp_policy) + "\n")
            if first_ip is not None:
                write(f"  OVGF first ionization potential = {first_ip:.4f} eV\n")
            if ea_val is not None:
                write(
                    f"  OVGF electron affinity (lowest virtual) = "
                    f"{ea_val:.4f} eV"
                    f"   (small-basis EA -- qualitative only)\n"
                )
            write("\n")
            flush()
            result = _OVGFAugmented(result, res, method=_effective_method)
        elif _ovgf_ok:
            import numpy as _np

            from ._vibeqc_core import compute_eri as _compute_eri
            from .propagator import quasiparticle_energies as _qp_energies

            C = _np.asarray(result.mo_coeffs)
            eps_mo = _np.asarray(result.mo_energies)
            nbf = len(eps_mo)
            # The diagonal self-energy needs the full MO 2e tensor (n_bf⁴); guard
            # large systems rather than OOM on the transform.
            if nbf > 100:
                raise NotImplementedError(
                    f"method='ovgf' builds the full MO two-electron tensor "
                    f"(n_bf={nbf}); it is currently limited to small/medium "
                    f"systems (n_bf <= 100). Use a smaller basis, or the "
                    f"vibeqc.propagator API with on-demand integral blocks."
                )
            nocc = effective_electron_count(molecule, result) // 2
            with plog.stage("ovgf", detail="GF2 self-energy"), PerfScope("ovgf"):
                eri_ao = _np.asarray(_compute_eri(basis_obj))
                eri_mo = _np.einsum(
                    "pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True
                )
                # Outer valence: up to 3 highest occupied + LUMO.
                lo = max(0, nocc - 3)
                orbs = (
                    list(range(lo, nocc + 1)) if nocc < nbf else list(range(lo, nocc))
                )
                qp = _qp_energies(eps_mo, nocc, eri_mo, orbs, iterate=True)

            EV = 27.211386245988
            _homo_q = next(q for q in qp if q.orbital == nocc - 1)
            _lumo_q = next((q for q in qp if q.orbital == nocc), None)
            _slog.emit(
                "ovgf_done",
                orbitals=orbs,
                eps_scf=[float(q.eps_scf) for q in qp],
                eps_qp=[float(q.eps_qp) for q in qp],
                pole_strength=[float(q.pole_strength) for q in qp],
                ip_homo_ev=float(-_homo_q.eps_qp * EV),
                ea_lumo_ev=(None if _lumo_q is None else float(-_lumo_q.eps_qp * EV)),
            )
            write(
                "\n"
                + section_header(
                    "OVGF / GF2 quasiparticle energies (diagonal "
                    "second-order self-energy)"
                )
            )
            # Headered column table; same convention as the spin-resolved
            # block above (content-sized rule, eV carried by the headers).
            _qp_tbl = Table(
                [
                    Column("MO"),
                    Column("label"),
                    Column("Koopmans(eV)"),
                    Column("OVGF (eV)"),
                    Column("pole Z"),
                ],
                indent=2,
            )
            for q in qp:
                lbl = (
                    "HOMO"
                    if q.orbital == nocc - 1
                    else "LUMO"
                    if q.orbital == nocc
                    else ""
                )
                _qp_tbl.add_row(
                    int(q.orbital),
                    lbl,
                    Quantity(float(q.eps_scf * EV), "dimensionless"),
                    Quantity(float(q.eps_qp * EV), "dimensionless"),
                    Quantity(float(q.pole_strength), "dimensionless"),
                )
            _qp_policy = active_policy().with_spec(
                "dimensionless", precision=4, width=0
            )
            write(_qp_tbl.render(_qp_policy) + "\n")
            write(
                f"  OVGF ionization potential (HOMO) = {-_homo_q.eps_qp * EV:.4f} eV\n"
            )
            if _lumo_q is not None:
                # EA = -eps_qp(LUMO). The diagonal GF2 self-energy gives only
                # qualitative EAs in a small/non-augmented basis (non-variational
                # virtuals; no diffuse functions for a bound anion) -- flagged so
                # users don't read a positive value as a stable anion.
                write(
                    f"  OVGF electron affinity   (LUMO) = "
                    f"{-_lumo_q.eps_qp * EV:.4f} eV"
                    f"   (small-basis EA -- qualitative only)\n"
                )
            if 2 * nbf <= 60:
                # Renormalized GF2 (full third-order self-energy + geometric
                # screening) -- trims the bare-GF2 overcorrection back toward
                # experiment. Small systems only (the third-order contractions
                # scale steeply). See vibeqc.propagator (renormalized second-
                # order GF; lineage Cederbaum 1975 / von Niessen 1984).
                from .propagator import renormalized_quasiparticle_energies as _ren_qp

                _ren = _ren_qp(
                    eri_ao,
                    C,
                    C,
                    eps_mo,
                    eps_mo,
                    nocc,
                    nocc,
                    [("alpha", nocc - 1)],
                    iterate=True,
                )[0][1]
                write(
                    f"  GF2(renorm) ionization potential (HOMO) = "
                    f"{-_ren.eps_qp * EV:.4f} eV"
                    f"   (renormalized; pole Z={_ren.pole_strength:.3f})\n"
                )
            write("\n")
            flush()
            result = _OVGFAugmented(result, qp, method=_effective_method)

        # Atomization energy (S E_atom - E_mol) at the molecule's mean-field
        # level. Free-atom ground-state references are computed at the same
        # level (cached per element). Mean-field methods only (a post-SCF
        # method's atomization would need atomic post-SCF references too).
        if (
            atomization
            and method in ("rhf", "uhf", "rks", "uks", "rohf", "roks")
            and bool(getattr(result, "converged", False))
        ):
            try:
                from .atomization import atomization_energy as _atomization

                with _molecular_progress_scope(
                    _handle_cpp_scf_progress,
                    phase="atomization.scf",
                ):
                    _atm = _atomization(
                        molecule,
                        float(result.energy),
                        method,
                        basis,
                        functional=functional,
                        grid_level=grid_level,
                        grid_options=getattr(
                            {"rks": rks_options, "uks": uks_options,
                             "roks": roks_options}.get(method), "grid", None
                        ),
                    )
                _Hk = 627.509474063
                _slog.emit(
                    "atomization",
                    method=method,
                    functional=functional,
                    e_molecule=_atm.e_molecule,
                    e_atoms_sum=_atm.e_atoms_sum,
                    atomization=_atm.atomization,
                    atomization_per_atom=_atm.atomization_per_atom,
                    atomic_energies={
                        str(z): e for z, e in _atm.atomic_energies.items()
                    },
                )
                _flabel = f"/{functional}" if functional else ""
                write("\n" + section_header("## Atomization energy"))
                write(
                    f"  Free-atom references ({method.upper()}{_flabel}/{basis}):\n"
                )
                for _z in sorted(_atm.atomic_energies):
                    write(
                        f"    Z = {_z:<3d}  E = {render_energy_labeled(_atm.atomic_energies[_z], width=16, precision=10)}\n"
                    )
                write(
                    f"  S E(atoms)               = {render_energy_labeled(_atm.e_atoms_sum, width=16, precision=10)}\n"
                    f"  E(molecule)              = {render_energy_labeled(_atm.e_molecule, width=16, precision=10)}\n"
                    f"  Atomization energy       = {render_energy_labeled(_atm.atomization, width=16, precision=10)}"
                    f"  ({_atm.atomization_kcal:9.2f} kcal/mol)\n"
                    f"  per atom (n={_atm.n_atoms})           = "
                    f"{_atm.atomization_per_atom * _Hk:9.2f} kcal/mol\n\n"
                )
                flush()
            except Exception as _atm_exc:
                write(
                    f"\n  (warning: atomization not available: "
                    f"{type(_atm_exc).__name__}: {_atm_exc})\n\n"
                )
                flush()

        # Capture a correlated method total before additive corrections wrap
        # the result. MP2, CC, and DLPNO wrappers preserve ``energy`` as the
        # SCF component and expose the method total through ``energy_total``.
        # Keeping this pre-correction value avoids both dropping correlation
        # and double-counting dispersion in the terminal machine records.
        _has_post_scf_total = not isinstance(result, SolverResult) and hasattr(
            result, "energy_total"
        )
        _method_total_before_corrections = float(
            getattr(result, "energy_total", getattr(result, "energy", 0.0))
        )

        # Post-SCF dispersion (D3-BJ) as an additive correction.
        # Captured in a local so the composite-total sum below is
        # independent of whether ``result`` got wrapped in
        # _DispersionAugmented -- a SolverResult is never wrapped (it has
        # no .e_dispersion attribute), and reading the term back off the
        # result then silently dropped D3-BJ from the composite total
        # (audit F1.2).
        e_d3bj = 0.0
        if d3_params is not None:
            disp = compute_d3bj(molecule, d3_params)
            e_d3bj = float(disp.energy)
            e_scf = float(getattr(result, "energy", 0.0))
            e_total = e_scf + e_d3bj
            # Name the variant so the log distinguishes the two-body
            # correction from the ATM-inclusive one (s9 != 0).
            _d3_label = "D3-BJ(ATM)" if d3_params.s9 else "D3-BJ"
            # For a solvated job ``result.energy`` is the in-solvent total
            # (IID 148), so label the base row accordingly rather than
            # calling it the gas-phase SCF energy.
            _base_label = (
                "E_SCF(in-solv)"
                if getattr(result, "energy_in_solvent", None) is not None
                else "E_SCF"
            )
            write(
                "\n"
                + section_header(f"Dispersion correction ({_d3_label})")
                + f"  {'s6':>10s} {d3_params.s6:14.6f}\n"
                + f"  {'s8':>10s} {d3_params.s8:14.6f}\n"
                + f"  {'a1':>10s} {d3_params.a1:14.6f}\n"
                + f"  {'a2':>10s} {d3_params.a2:14.6f}\n"
                + f"  {'s9':>10s} {d3_params.s9:14.6f}\n"
                + f"  {'E_disp':>10s} {render_energy_labeled(disp.energy, width=14, precision=8)}"
                f"  ({disp.energy * 627.5094740631:+.4f} kcal/mol)\n"
                + f"  {_base_label:>10s} {render_energy_labeled(e_scf, width=14, precision=8)}\n"
                + f"  {'E_total':>10s} {render_energy_labeled(e_total, width=14, precision=8)}\n"
            )
            # Wrap the SCF result so callers can access both components
            # without breaking existing accesses to mo_coeffs, fock, etc.
            if not isinstance(result, SolverResult):
                result = _DispersionAugmented(
                    result, e_d3bj, d3_params, method=_effective_method
                )

        # ---- Composite 3c post-SCF corrections (D4 + gCP + SRB) -------
        # Only runs when a composite recipe is active. Each piece is
        # additive and logged separately; the composite total is the
        # SCF energy + every correction.
        e_d4 = 0.0
        e_gcp = 0.0
        e_srb = 0.0
        if use_d4:
            if not dftd4_available():
                raise ImportError(
                    "composite recipe requests D4 dispersion but the "
                    "optional 'dftd4' package is not installed.\n  "
                    "Install via `pip install -e '.[dispersion]'` or "
                    "`pip install dftd4`."
                )
            # dftd4 ships per-composite damping keyed on the composite
            # name (e.g. 'r2scan-3c', 'wb97x-3c') -- distinct from the
            # parent-functional damping. Prefer the composite name.
            d4_func = (
                composite_recipe.name
                if composite_recipe is not None
                else (functional or "hf")
            )
            d4_result = compute_d4(
                molecule,
                d4_func,
                charge=float(molecule.charge),
            )
            e_d4 = float(d4_result.energy)
            write(
                "\n"
                + section_header("Dispersion correction (D4)")
                + f"  {'functional':>10s} {d4_result.functional!s:>14s}\n"
                f"  {'E_disp':>10s} {render_energy_labeled(e_d4, width=14, precision=8)}"
                f"  ({e_d4 * 627.5094740631:+.4f} kcal/mol)\n"
            )
            # Match the D3(BJ) public result contract: keep ``energy`` as the
            # uncorrected SCF reference and expose the D4-inclusive method
            # total explicitly. Composite recipes add gCP/SRB below, so their
            # aggregate result needs a separate all-corrections contract.
            if composite_recipe is None and not isinstance(result, SolverResult):
                result = _DispersionAugmented(
                    result, e_d4, d4_result, method=_effective_method
                )

        if composite_recipe is not None and composite_recipe.gcp_basis:
            # Damped composites (pbeh-3c / hse-3c / r2scan-3c) carry a
            # GCPDamping; pass it through as (dmp_scal, dmp_exp). Undamped
            # recipes (hf-3c, b3lyp-3c) leave it None. b97-3c has
            # gcp_basis=None (SRB-only) so this block is skipped for it.
            _gcp_damping = (
                (
                    composite_recipe.gcp_damping.dmp_scal,
                    composite_recipe.gcp_damping.dmp_exp,
                )
                if composite_recipe.gcp_damping is not None
                else None
            )
            try:
                gcp_result = compute_gcp(
                    molecule,
                    composite_recipe.gcp_basis,
                    variant=composite_recipe.gcp_variant,
                    damping=_gcp_damping,
                )
                e_gcp = float(gcp_result.energy)
                # The four fit constants are per (method, basis); surface which
                # set was used so a shared parent basis (pbeh-3c / b3lyp-3c /
                # hse-3c all on def2-mSVP) is not mistaken for a shared fit.
                _gcp_variant = gcp_result.params.variant or "(basis default)"
                write(
                    "\n"
                    + section_header("Geometric counterpoise correction (gCP)")
                    + f"  {'basis':>10s} {gcp_result.basis_name!s:>14s}\n"
                    f"  {'params':>10s} {_gcp_variant!s:>14s}\n"
                    f"  {'E_gCP':>10s} {render_energy_labeled(e_gcp, width=14, precision=8)}"
                    f"  ({e_gcp * 627.5094740631:+.4f} kcal/mol)\n"
                )
            except GCPDataMissing as exc:
                # gCP data table not bundled for this basis -- log and
                # continue with E_gCP = 0 so the composite total is
                # explicitly marked incomplete rather than silently
                # wrong. Same posture as the dispersion builtin-stub.
                write(
                    "\n"
                    + section_header("Geometric counterpoise correction (gCP)")
                    + f"  SKIPPED -- {exc}\n"
                )

        if composite_recipe is not None and composite_recipe.sr_mod is not None:
            srb = composite_recipe.sr_mod
            atomic_numbers = [int(a.Z) for a in molecule.atoms]
            positions = [list(a.xyz) for a in molecule.atoms]
            e_srb = float(srb.evaluate(atomic_numbers, positions))
            write(
                "\n"
                + section_header(str(srb.name))
                + f"  {'E_SRB':>10s} {render_energy_labeled(e_srb, width=14, precision=8)}"
                f"  ({e_srb * 627.5094740631:+.4f} kcal/mol)\n"
            )

        if composite_recipe is not None:
            composite_total = (
                float(getattr(result, "energy", 0.0)) + e_d3bj + e_d4 + e_gcp + e_srb
            )
            write(
                "\n"
                + section_header(f"Composite total ({composite_recipe.name})")
                + f"  {'E_SCF':>10s} "
                f"{render_energy_labeled(float(getattr(result, 'energy', 0.0)), width=14, precision=8)}\n"
                f"  {'E_disp':>10s} "
                f"{render_energy_labeled(e_d3bj + e_d4, width=14, precision=8)}\n"
                f"  {'E_gCP':>10s} {render_energy_labeled(e_gcp, width=14, precision=8)}\n"
                f"  {'E_SRB':>10s} {render_energy_labeled(e_srb, width=14, precision=8)}\n"
                f"  {'E_total':>10s} {render_energy_labeled(composite_total, width=14, precision=8)}\n"
            )

        has_mos = hasattr(result, "mo_energies") or hasattr(result, "mo_energies_alpha")
        if write_molden_file and has_mos:
            # Molden is intentionally queued rather than written here.  A
            # large text wavefunction can be hundreds of megabytes; issue
            # #28 captured a carrier timeout in this writer that discarded
            # every compact post-SCF artefact still waiting behind it.
            write(
                f"\n  Molecular orbitals queued for final export to "
                f"{molden_path.name}\n"
            )
            flush()
        if _trexio_requested and has_mos:
            write(
                f"\n  TREXIO wavefunction queued for final export to "
                f"{_trexio_path.name} ({_trexio_backend} back end)\n"
            )
            flush()

        # Population dump (Phase O6). Always-on for molecular runs by
        # default -- cheap to compute (the SCF density is in memory),
        # cheap to write, and downstream tooling shouldn't have to
        # screen-scrape the .out for charges + bond orders + dipole.
        # The matching block stays in .out for human reading; the
        # .txt + .json siblings are the machine-readable form.
        _population_summary = None
        _population_written = False
        if write_population_file and (
            has_mos or getattr(result, "mulliken_charges", None) is not None
        ):
            pop_txt = output_stem.parent / (output_stem.name + ".population.txt")
            try:
                with (
                    plog.stage("write_population", detail=str(pop_txt.name)),
                    PerfScope("write_population"),
                ):
                    if getattr(result, "mulliken_charges", None) is not None:
                        from vibeqc.output.formats.population import (
                            compute_native_mulliken_population_summary,
                        )

                        _population_summary = (
                            compute_native_mulliken_population_summary(
                                result.mulliken_charges,
                                molecule,
                                str(getattr(result, "method", resolved_method)),
                            )
                        )
                    else:
                        from vibeqc.output.formats.population import (
                            compute_population_summary,
                        )

                        _population_summary = compute_population_summary(
                            result,
                            basis_obj,
                            molecule,
                            nuclear_charges=_nuclear_charges_used,
                        )
                    _output_writer.dispatch_role(
                        "population",
                        result=result,
                        basis=basis_obj,
                        molecule=molecule,
                        population_summary=_population_summary,
                        raise_on_error=True,
                    )
                    _population_written = True
                write(
                    f"  Population dump written to "
                    f"{pop_txt.name} + "
                    f"{output_stem.name}.population.json\n"
                )
                flush()
            except Exception as _pop_exc:
                write(
                    f"  (warning: population dump failed: "
                    f"{type(_pop_exc).__name__}: {_pop_exc})\n"
                )
                flush()
                warn_writer_failure(
                    _pop_exc,
                    pop_txt,
                    role="population_summary",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        # Volumetric cubes (Phase O6). Opt-in via write_cube=: True /
        # "density" / "homo" / "lumo" / list-of-those-or-int-indices.
        # Each cube is wrapped in its own try/except so a single
        # grid-evaluation failure on one MO doesn't block the others.
        _cube_req = parse_write_cube_kwarg(write_cube)
        if _cube_req and has_mos:
            if _cube_req.density:
                try:
                    with (
                        plog.stage("write_cube_density"),
                        PerfScope("write_cube_density"),
                    ):
                        cube_path = _output_writer.dispatch_role(
                            "density",
                            only_format="cube",
                            result=result,
                            basis=basis_obj,
                            molecule=molecule,
                            cube_spacing=cube_spacing,
                            cube_padding=cube_padding,
                            raise_on_error=True,
                        )[0]
                    write(f"  Density cube written to {cube_path.name}\n")
                    flush()
                except Exception as _cube_exc:
                    write(
                        f"  (warning: density cube write failed: "
                        f"{type(_cube_exc).__name__}: {_cube_exc})\n"
                    )
                    flush()
                    warn_writer_failure(
                        _cube_exc,
                        stem_sibling(output_stem, ".density.cube"),
                        role="density_cube",
                        category=OutputFailureKind.optional_artifact,
                        writer=_output_writer,
                    )
            if _cube_req.mo_labels:
                try:
                    _mo_targets = requested_mo_indices(
                        list(_cube_req.mo_labels),
                        result,
                        molecule,
                    )
                except Exception as _exc:
                    write(
                        f"  (warning: MO label resolution failed: "
                        f"{type(_exc).__name__}: {_exc})\n"
                    )
                    warn_writer_failure(
                        _exc,
                        stem_sibling(output_stem, ".cube"),
                        role="mo_label_resolution",
                        category=OutputFailureKind.optional_artifact,
                        writer=_output_writer,
                    )
                    _mo_targets = []
                for _idx, _name in _mo_targets:
                    try:
                        with (
                            plog.stage(f"write_cube_mo_{_name}"),
                            PerfScope(f"write_cube_mo_{_name}"),
                        ):
                            _mo_expected = output_stem.parent / (
                                output_stem.name + f".{_name}.cube"
                            )
                            _mo_path = _output_writer.dispatch_role(
                                "orbital_vol",
                                only_path=_mo_expected,
                                result=result,
                                basis=basis_obj,
                                molecule=molecule,
                                mo_index=_idx,
                                display_name=_name,
                                cube_spacing=cube_spacing,
                                cube_padding=cube_padding,
                                raise_on_error=True,
                            )[0]
                        write(f"  MO cube written to {_mo_path.name}\n")
                        flush()
                    except Exception as _cube_exc:
                        write(
                            f"  (warning: MO {_name} cube write failed: "
                            f"{type(_cube_exc).__name__}: {_cube_exc})\n"
                        )
                        flush()
                        warn_writer_failure(
                            _cube_exc,
                            stem_sibling(output_stem, f".{_name}.cube"),
                            role=f"mo_cube_{_name}",
                            category=OutputFailureKind.optional_artifact,
                            writer=_output_writer,
                        )

        # Citations (Phase O5b). Assemble the reference list for this
        # job (software + libint + libxc-if-DFT + basis-set +
        # functional + DIIS + dispersion-if-used + ECP-if-used) and
        # emit {stem}.bibtex + .references siblings. The matching
        # "## References" block is appended to the .out a few lines
        # below so the text log carries the same references. Failures
        # are non-fatal: a routing miss or a writer crash leaves a
        # warning in the .out and the SCF result is unaffected.
        cite_block_text: str | None = None
        _refs = None
        if citations:
            try:
                _cite_db = load_default_database()
                _dispersion_key = None
                _dispersion_params_key = None
                if d3_params is not None:
                    # The database routes "d3bj" -> Grimme 2010 + 2011.
                    _dispersion_key = "d3bj"
                elif use_d4:
                    # "d4" -> Caldeweyher 2019 (method paper), plus --
                    # for parametrizations fit outside that paper --
                    # the damping-parameter fit paper via
                    # routes.dispersion_params["d4:<key>"]. The key
                    # mirrors the damping lookup in the post-SCF D4
                    # block above: composite name when a 3c recipe is
                    # active, else the SCF functional ("hf" for pure
                    # Hartree-Fock + D4).
                    _dispersion_key = "d4"
                    _dispersion_params_key = normalize_d4_key(
                        composite_recipe.name
                        if composite_recipe is not None
                        else (functional or "hf")
                    )
                # Detect direct SCF usage for citation routing.
                _uses_direct = _detect_direct_scf(
                    resolved_method,
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                    basis,
                    molecule,
                )
                # SCF accelerator from the options the SCF actually ran with
                # and the extrapolation steps its trace recorded (#682) --
                # fires the matching DIIS / EDIIS / ADIIS / KDIIS citation
                # instead of crediting whatever the caller's struct said.
                _scf_accel = _detect_scf_accelerator(
                    resolved_method,
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                    options_used=_scf_options_used,
                    result=result,
                )
                # ECP detection -- libecpint citation only fires when
                # at least one options struct carries an ecp_centers
                # list (manual ECPCenter recipe or auto_ecp_centers).
                _uses_ecp = _detect_uses_ecp(
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                    result=result,
                )
                # CPCM solvation citation fires when run_job was
                # invoked with a non-None ``solvent``; vibe-qc on main
                # only ships CPCM-style models, so the variant key
                # defaults to "cpcm".
                _uses_cpcm = solvent is not None
                # For composite-3c jobs (hf-3c / pbeh-3c / ...) route
                # citations by the *composite keyword*, not the
                # resolved mean-field method -- the composite carries
                # its own defining paper (+ gCP / D3) in
                # routes.methods. ``composite_recipe`` is None for
                # non-composite jobs, in which case the resolved
                # method name is the right routing key.
                # Post-SCF correlation methods run an RHF/UHF SCF first
                # (resolved_method = "rhf"/"uhf"), but their citation key is
                # the *original* method -- routes.methods.{ccsd,mp2,scs-mp2,...}
                # carry the correlation papers (Moller-Plesset, Grimme SCS,
                # Jung SOS, Purvis-Bartlett ...). Routing on resolved_method
                # would silently drop those references.
                _mp2_cite_method = None
                if (
                    _mp2_used_density_fit
                    and not _use_rohf_mp2
                    and method in ("mp2", "scs-mp2", "sos-mp2")
                ):
                    _mp2_cite_method = {
                        "mp2": "ri-mp2",
                        "scs-mp2": "scs-ri-mp2",
                        "sos-mp2": "sos-ri-mp2",
                    }[method]
                _cite_method = (
                    composite_recipe.name
                    if composite_recipe is not None
                    else _mp2_cite_method
                    if _mp2_cite_method is not None
                    else _effective_method
                    if _effective_method in _POSTSCF_CITE_METHODS
                    else resolved_method
                )
                # MLIP engines (method="mace") evaluate no Gaussian
                # integrals and run no SCF -- suppress the always-on
                # libint + DIIS routes so the references reflect what
                # actually ran. The MACE / e3nn / PyTorch citations come
                # in via routes.methods[<mlip>] instead.
                _is_mlip = resolved_method in _MLIP_METHODS
                # INDO-family engines (MSINDO molecular + SECCM) use
                # analytic Slater (STO) integrals, not libint Gaussians -- so the
                # always-on libint integral citation must not fire (they do use
                # Pulay DIIS, so uses_scf stays on).
                _is_indo = resolved_method in ("msindo", "ccm")
                # The job-wide citation assembly stays here, but the concrete
                # SECCM method, dimensional kernel, temperature, integral/SCF
                # flags, and method-specific extras have one owner: the
                # runtime route plan. Falling back to the legacy values keeps
                # model-like test doubles without a validated plan usable.
                _citation_route_kwargs: dict[str, object] = {
                    "method": _cite_method,
                    "uses_integrals": not (_is_mlip or _is_indo),
                    "uses_scf": not _is_mlip,
                    "electronic_temperature": float(
                        getattr(result, "electronic_temperature", 0.0) or 0.0
                    ),
                    "seccm_dimension": None,
                    "seccm_electrostatics_kernel": None,
                }
                _semiempirical_citation_plan = (
                    getattr(result, "route_plan", None)
                    if resolved_method in SEMIEMPIRICAL_METHODS
                    else None
                )
                if isinstance(
                    _semiempirical_citation_plan,
                    SemiempiricalRoutePlan,
                ):
                    _citation_route_kwargs.update(
                        _semiempirical_citation_plan.citation_assemble_kwargs
                    )
                _semiempirical_extra_entries = tuple(
                    _citation_route_kwargs.pop("extra_entries", ())
                )
                # Per-model MLIP foundation-model citation: the model
                # actually run (result.model_info) decides which foundation
                # paper to cite (MPA-0 -> Batatia 2024; OFF23 -> Kovács
                # 2023). The MACE *method* paper fires via the static
                # routes.methods.mace route.
                _mlip_extra: list[str] = []
                if _is_mlip:
                    _mi = getattr(result, "model_info", None)
                    if _mi is not None and getattr(_mi, "citation", ""):
                        _mlip_extra.append(_mi.citation)
                # UNO-CAS starting reference (Pulay-Hamilton 1988): cite when
                # an open-shell determinant-solver job resolved to the UHF
                # natural-orbital reference (the open-shell default; see the
                # cas_reference resolution in _run_single_point).
                if (
                    resolved_method
                    in (
                        "cisd",
                        "selected_ci",
                        "dmrg",
                        "v2rdm",
                        "transcorrelated_ci",
                        "casci",
                        "mrci",
                        "casscf",
                        "nevpt2",
                        "caspt2",
                    )
                    and (
                        cas_reference or ("uno" if molecule.multiplicity > 1 else "rhf")
                    )
                    == "uno"
                ):
                    _mlip_extra.append("pulay_hamilton_uno_1988")
                # Spin-pure ROHF starting reference (Roothaan 1960): cite when
                # a determinant-solver job used cas_reference="rohf".
                if (
                    resolved_method
                    in (
                        "cisd",
                        "selected_ci",
                        "dmrg",
                        "v2rdm",
                        "transcorrelated_ci",
                        "casci",
                        "mrci",
                        "casscf",
                        "nevpt2",
                        "caspt2",
                    )
                    and cas_reference == "rohf"
                ):
                    _mlip_extra.append("roothaan_rohf_1960")
                # Multi-state CASPT2: the effective-Hamiltonian formalism
                # (Finley 1998) fires for both modes; the extended (XMS)
                # rotation adds Granovsky 2011 + Shiozaki 2011.
                if (
                    resolved_method == "caspt2"
                    and caspt2_options is not None
                    and caspt2_options.multistate is not None
                ):
                    _mlip_extra.append("finley_ms_caspt2_1998")
                    if caspt2_options.multistate == "xms":
                        _mlip_extra.append("granovsky_xmcqdpt2_2011")
                        _mlip_extra.append("shiozaki_xms_caspt2_2011")
                    # Relaxed MS/XMS gradient: the SA-MCSCF state-specific
                    # gradient formalism (Stalring 2001) + the (MS-)CASPT2
                    # analytic-gradient Lagrangian (Celani-Werner 2003).
                    if getattr(result, "gradient", None) is not None:
                        _mlip_extra.append("stalring_sa_mcscf_gradient_2001")
                        _mlip_extra.append("celani_werner_ms_caspt2_gradient_2003")
                    if (
                        getattr(result, "nonadiabatic_coupling", None)
                        is not None
                    ):
                        _mlip_extra.append("park_shiozaki_ms_caspt2_nac_2017")
                elif (
                    resolved_method == "caspt2"
                    and caspt2_options is not None
                    and caspt2_options.compute_corr_grad
                    and getattr(result, "gradient", None) is not None
                ):
                    # Single-state internally-contracted CASPT2 analytic
                    # Lagrangian (Celani-Werner 2003).
                    _mlip_extra.append("celani_werner_ms_caspt2_gradient_2003")
                # Closed-shell (seniority-zero, DOCI) selected-CI basis:
                # spin_restricted=True walks doubly-occupied spatial
                # orbitals coupled through g_iiaa (GitLab #639).  The DOCI
                # reference fires only on that basis; the default
                # unrestricted basis is the CIPSI/SHCI citations alone.
                if (
                    resolved_method in ("selected_ci", "transcorrelated_ci")
                    and selected_ci_options is not None
                    and getattr(selected_ci_options, "spin_restricted", False)
                ):
                    _mlip_extra.append("bytautas_doci_2011")
                # Selected-CI CASSCF backend: the CI inside the macro-
                # iteration is the CIPSI selection algorithm (the static
                # routes only fire for method="selected_ci").
                if (
                    resolved_method == "casscf"
                    and casscf_options is not None
                    and casscf_options.ci_solver == "selected_ci"
                ):
                    _mlip_extra.append("huron_malrieu_cipsi_1973")
                    _mlip_extra.append("holmes_tubman_umrigar_shci_2016")
                    # The Epstein-Nesbet PT2 stage on the selected
                    # wavefunction cites the semistochastic-HCI paper
                    # (the routes.methods key fires only for a literal
                    # selected_ci_pt2 method; this hook is the live
                    # path for the CASSCF composition).
                    if getattr(casscf_options, "pt2", None) is not None:
                        _mlip_extra.append("sharma_semistochastic_hci_2017")
                # Storage algorithms are selected at runtime from the
                # requested memory budget, so their implementation papers
                # cannot be attached statically to every MP2/CCSD(T) route.
                # Add only the algorithms the native result says actually ran.
                if _mp2_memory_mode_used in {"direct", "disk"}:
                    if _mp2_used_density_fit:
                        _mlip_extra.append("bintrim_direct_df_mp2_2022")
                    else:
                        _mlip_extra.append("head_gordon_direct_mp2_1988")
                    if _mp2_memory_mode_used == "disk":
                        _mlip_extra.append("frisch_semidirect_mp2_1990")
                if (
                    _cc_used_density_fit
                    and not _cc_open_shell_used
                    and _cc_triples_memory_mode_used
                    in {"blocked", "direct", "disk"}
                ):
                    _mlip_extra.append("gyevi_nagy_direct_ccsdt_2020")
                if _cc_used_fno:
                    _mlip_extra.append("taube_bartlett_fno_2008")
                # Integral/exchange acceleration (RI-J Coulomb fitting /
                # RIJCOSX chain-of-spheres exchange) from the density_fit /
                # cosx flags on the resolved options struct.
                _accel = _detect_acceleration(
                    resolved_method,
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                )
                # Second-order convergers (SOSCF / TRAH) cite their defining
                # papers when armed by a positive threshold.
                _uses_soscf, _uses_trah = _detect_soscf_trah(
                    resolved_method,
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                )
                # Saunders-Hillier level shift (opts.level_shift or an
                # explicit level_shift_schedule) is a cited convergence
                # technique.
                _uses_level_shift = _detect_level_shift(
                    resolved_method,
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                    rohf_options,
                    roks_options,
                )
                # Map the effective SCF initial guess (including molecule-aware
                # AUTO resolution) to its defining-paper citation route.
                if _executed_guess_selection is not None:
                    _scf_guess = _SCF_GUESS_ROUTE_KEYS.get(
                        _executed_guess_selection.effective.name.lower())
                elif _scf_guess_citation_override_active:
                    _scf_guess = _scf_guess_citation_override
                else:
                    _scf_guess = _detect_scf_guess(
                        resolved_method,
                        molecule,
                        _scf_options_used,
                    )
                # Analytic-gradient + vibrational-analysis drivers. The
                # Hessian block (further below) runs a finite-difference
                # Hessian *of the analytic atomic gradient*, but only when the
                # SCF converged -- so the Hessian and gradient driver
                # citations gate on convergence. Geometry optimisation also
                # evaluates the analytic gradient at every step. Neither
                # applies to MLIP engines, whose forces come from the model
                # (cited via routes.methods[<mlip>]), not Pulay/Hellmann-
                # Feynman theory.
                _uses_basis_free = (
                    resolved_method in SEMIEMPIRICAL_METHODS
                    or resolved_method in _MLIP_METHODS
                )
                _uses_hessian = (
                    bool(hessian)
                    and bool(getattr(result, "converged", False))
                    and not _uses_basis_free
                )
                _uses_gradient = (
                    bool(optimize) or _uses_hessian
                ) and not _uses_basis_free
                # Cite only population analyses that actually succeeded for
                # this job. This is section-specific: Gaussian AO routes often
                # compute the full stack, while method-native semiempirical
                # output may truthfully contain Mulliken charges plus explicit
                # unsupported markers for every other analysis. Hirshfeld also
                # retains the `.out` success gate because its grid build there
                # is independent from the later sidecar computation.
                _props: list[str] = []
                if _population_written and _population_summary is not None:
                    _population_routes = (
                        (
                            "mulliken",
                            "mulliken",
                            bool(_population_summary.mulliken_atoms),
                        ),
                        (
                            "loewdin",
                            "lowdin_charges",
                            bool(_population_summary.loewdin_atoms),
                        ),
                        (
                            "hirshfeld",
                            "hirshfeld",
                            bool(_population_summary.hirshfeld_atoms)
                            and bool(_prop_status.get("hirshfeld")),
                        ),
                        (
                            "mayer",
                            "mayer_bond_order",
                            True,
                        ),
                        (
                            "wiberg",
                            "wiberg",
                            True,
                        ),
                        (
                            "npa",
                            "npa",
                            bool(_population_summary.npa_atoms),
                        ),
                    )
                    for _section, _route, _has_required_rows in _population_routes:
                        if (
                            _has_required_rows
                            and not _population_summary.errors.get(_section)
                        ):
                            _props.append(_route)
                if nto:
                    _props.append("nto")
                if qtaim:
                    # The "qtaim" route and its bader_qtaim_1985 entry have
                    # both existed since QTAIM landed; nothing ever appended
                    # the key, so an opt-in topological analysis cited Bader
                    # nowhere.
                    _props.append("qtaim")
                for _loc_method in _resolve_localize_methods(localize):
                    _props.append(
                        {"pipek-mezey": "pipek_mezey", "boys": "foster_boys"}
                        .get(_loc_method, _loc_method)
                    )
                _refs = _cite_db.assemble(
                    cc_density_fit=_cc_used_density_fit,
                    basis=basis,
                    functional=functional,
                    dispersion=_dispersion_key,
                    dispersion_params=_dispersion_params_key,
                    scf_accelerator=_scf_accel,
                    # ASE is cited only when its BFGS optimizer actually ran;
                    # the native / brent / geomopt backends do not use ASE.
                    uses_ase=_used_ase_optimizer,
                    # Geometry-optimization algorithm + coordinate system. Only
                    # the uniform geomopt path (geom_opt set) carries a
                    # keyword-selected optimizer / coordinate citation; the
                    # legacy ase/native/brent backends leave these None.
                    geom_optimizer=(
                        geom_opt if _used_geomopt_optimizer else None
                    ),
                    geom_coords=(
                        geom_coords if _used_geomopt_optimizer else None
                    ),
                    uses_ecp=_uses_ecp,
                    uses_cpcm=_uses_cpcm,
                    # MSINDO COSMO uses its GEPOL cavity (+ Silla 1991); the
                    # HF/DFT Lebedev path keeps the default "cpcm" bundle.
                    solvent_variant=(
                        "cosmo_gepol"
                        if (_uses_cpcm and resolved_method == "msindo")
                        else None
                    ),
                    # #701: the pair-density convention the PNO cut used.
                    # "legacy" is vibe-qc's own and has no route row.
                    pno_norm=_dlpno_pno_norm or None,
                    direct_scf=_uses_direct,
                    dft_plus_u=bool(dft_plus_u),
                    acceleration=_accel,
                    uses_soscf=_uses_soscf,
                    uses_trah=_uses_trah,
                    uses_scf_stability=bool(
                        getattr(result, "stability_checked", False)
                    ),
                    uses_level_shift=_uses_level_shift,
                    scf_guess=_scf_guess,
                    uses_gradient=_uses_gradient,
                    uses_hessian=_uses_hessian,
                    uses_tddft=tddft,
                    tddft_variant=tddft_type if tddft else None,
                    properties=_props,
                    # TREXIO (Posenitskiy 2023) is cited only when the
                    # artefact is actually written (routes.libraries.trexio).
                    extra_libraries=(
                        ("trexio",) if (_trexio_requested and has_mos) else ()
                    ),
                    extra_entries=(
                        *_mlip_extra,
                        *_semiempirical_extra_entries,
                        *_frozen_core_source_entries,
                        *_dlpno_triples_source_entries,
                    ),
                    **_citation_route_kwargs,
                )
                # Feed the full assembled provenance into the .system
                # manifest's [citations] section (CLAUDE.md Sec.8.3/Sec.8.5),
                # before writing the user-facing siblings so a brittle
                # .bibtex write can't deprive .system of the record.
                _output_writer.set_citations(_citation_manifest_rows(_refs))
                _output_writer.dispatch_role(
                    "citations",
                    citations=_refs,
                    raise_on_error=True,
                )
                cite_block_text = format_references_block(_refs)
                _bib_name = stem_sibling(output_stem, ".bibtex").name
                _ref_name = stem_sibling(output_stem, ".references").name
                write(f"  Citations written to {_bib_name} + {_ref_name}\n")
                flush()
            except Exception as _cite_exc:
                write(
                    f"  (warning: citation emission failed: "
                    f"{type(_cite_exc).__name__}: {_cite_exc})\n"
                )
                flush()
                warn_writer_failure(
                    _cite_exc,
                    stem_sibling(output_stem, ".bibtex"),
                    role="citations",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        # Final geometry as a plain XYZ sibling (Phase O3). ``molecule``
        # at this point is the geometry the SCF actually ran on -- the
        # optimised one when ``optimize=True``, otherwise the input.
        # Energy in Hartree is appended to the comment line so ASE /
        # Open Babel can recover the SCF result from the .xyz alone.
        if write_xyz_file:
            xyz_path = stem_sibling(output_stem, ".xyz")
            try:
                energy_ha = float(getattr(result, "energy", float("nan")))
            except (TypeError, ValueError):
                energy_ha = float("nan")
            try:
                with (
                    plog.stage("write_xyz", detail=str(xyz_path.name)),
                    PerfScope("write_xyz"),
                ):
                    _output_writer.dispatch_role(
                        "geometry",
                        only_format="xyz",
                        molecule=molecule,
                        energy_ha=(None if energy_ha != energy_ha else energy_ha),
                        raise_on_error=True,
                    )
                write(f"  Final geometry written to {xyz_path.name}\n")
                flush()
            except Exception as _xyz_exc:
                # Geometry writer is best-effort -- never let a `.xyz`
                # failure tank a finished SCF. The user can always
                # re-run with ``write_xyz_file=False``.
                write(
                    f"  (warning: .xyz writer failed: "
                    f"{type(_xyz_exc).__name__}: {_xyz_exc})\n"
                )
                flush()
                warn_writer_failure(
                    _xyz_exc,
                    xyz_path,
                    role="final_xyz_geometry",
                    category=OutputFailureKind.optional_artifact,
                    writer=_output_writer,
                )

        # Structured log: properties event. Best-effort -- the helpers
        # raise on basis-set shapes vibe-qc's Python side can't
        # parse; a failure here must never tank the run, so we wrap
        # each section. The event only fires when at least one
        # property was successfully computed.
        if _slog.enabled and has_mos:
            _props_payload: dict[str, Any] = {}
            try:
                from .properties import (
                    dipole_moment as _dipole,
                )
                from .properties import (
                    loewdin_charges as _low,
                )
                from .properties import (
                    mulliken_charges as _mul,
                )

                _props_payload["mulliken"] = [
                    float(x)
                    for x in _mul(
                        result, basis_obj, molecule,
                        nuclear_charges=_nuclear_charges_used,
                    )
                ]
                _props_payload["loewdin"] = [
                    float(x)
                    for x in _low(
                        result, basis_obj, molecule,
                        nuclear_charges=_nuclear_charges_used,
                    )
                ]
                _dip = _dipole(
                    result, basis_obj, molecule,
                    nuclear_charges=_nuclear_charges_used,
                )
                _props_payload["dipole"] = {
                    "x": float(_dip.x),
                    "y": float(_dip.y),
                    "z": float(_dip.z),
                    "total": float(_dip.total),
                    "total_debye": float(_dip.total_debye),
                }
            except Exception as _props_exc:
                # Properties are best-effort; the trace already
                # carries a degraded properties block, the structured
                # log just skips the event. Surface a warning so the
                # user knows the structured-log `properties` event was
                # dropped.
                warn_output_failure(
                    _props_exc,
                    stem_sibling(output_stem, ".out"),
                    role="structured_log_properties",
                    category=OutputFailureKind.compatibility_fallback,
                )
            if _props_payload:
                _slog.emit("properties", **_props_payload)

        if optimize:
            if _opt_backend == "ase" and traj_path.is_file():
                write(f"  Optimization trajectory written to {traj_path.name}\n")
            flush()

        # The returned result keeps ``energy`` as the bare SCF component for
        # backward compatibility. Terminal machine records instead use the
        # complete method total, while ``e_scf`` preserves that component
        # explicitly whenever a post-SCF method or additive correction is
        # present.
        _corr_fields: dict = {}
        _e_corr_sum = e_d3bj + e_d4 + e_gcp + e_srb
        _correction_active = (
            d3_params is not None
            or use_d4
            or e_gcp != 0.0
            or e_srb != 0.0
        )
        _e_scf_bare = float(getattr(result, "energy", 0.0))
        _terminal_energy = _method_total_before_corrections + _e_corr_sum
        if _has_post_scf_total:
            _corr_fields["e_scf"] = _e_scf_bare
        if _correction_active:
            # ``result.energy`` is the bare pre-correction energy in every
            # case: the _DispersionAugmented wrapper forwards ``.energy``
            # to the underlying SCF result, and SolverResults are never
            # wrapped. Same value the .out D3 block prints as E_SCF.
            _corr_fields.update(
                e_scf=_e_scf_bare,
                e_dispersion=float(e_d3bj + e_d4),
                e_total=_terminal_energy,
            )
            if e_gcp != 0.0 or e_srb != 0.0:
                _corr_fields["e_gcp"] = float(e_gcp)
                _corr_fields["e_srb"] = float(e_srb)
        _terminal_progress_fields = None
        if _has_post_scf_total or _correction_active:
            _terminal_progress_fields = dict(
                phase="solver" if _has_post_scf_total else "post-scf",
                energy_eh=_terminal_energy,
                method=str(_effective_method),
            )
        # --- Harmonic vibrational analysis (finite-difference Hessian) ---
        _thermo_result_for_qvf = None
        if hessian:
            _scf_converged = bool(getattr(result, "converged", False))
            if not _scf_converged:
                write(
                    "\n"
                    + section_header("## Vibrational Frequencies")
                    + "  SKIPPED -- SCF did not converge.\n"
                )
                flush()
            elif (
                resolved_method in SEMIEMPIRICAL_METHODS
                or resolved_method in _MLIP_METHODS
            ):
                write(
                    "\n  ## Vibrational Frequencies\n"
                    "  SKIPPED -- finite-difference Hessians are currently "
                    "available only for Gaussian-basis molecular methods.\n\n"
                )
                flush()
            else:
                with (
                    plog.stage("hessian_fd", detail="finite-difference Hessian"),
                    PerfScope("hessian_fd"),
                    _molecular_progress_scope(
                        _handle_cpp_scf_progress,
                        phase="hessian.scf",
                    ),
                ):
                    try:
                        from .hessian import (
                            HessianFDOptions,
                            compute_hessian_fd,
                            ir_intensities,
                        )

                        # The options the SCF ran with carry the auto-
                        # attached ECP fields even when the route kwarg was
                        # None; the FD Hessian must differentiate the same
                        # Hamiltonian and use Z - n_core in its dipole rows
                        # (#576, #642).
                        _hess_scf_opts = _scf_options_used or {
                            "rhf": rhf_options,
                            "uhf": uhf_options,
                            "rks": rks_options,
                            "uks": uks_options,
                            "rohf": rohf_options,
                        }.get(resolved_method)
                        _hess_opts = HessianFDOptions(
                            include_dipole_derivatives=True,
                            frozen_indices=hessian_frozen_indices,
                        )
                        hessian_result = compute_hessian_fd(
                            molecule,
                            basis,
                            method=resolved_method.upper(),
                            scf_options=_hess_scf_opts,
                            hessian_options=_hess_opts,
                            grid_level=grid_level,
                        )
                        write(
                            "\n"
                            + section_header("## Vibrational Frequencies")
                            + f"  Finite-difference Hessian"
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
                        # The frequency table through the document-layer
                        # Table primitive: columns + rule size to content,
                        # instead of the old hardcoded "-" * 52 rule and
                        # fixed 6/10/12 widths. The IR column is present only
                        # when intensities computed. Freq magnitudes render
                        # via render_frequency (cm-1, policy-aware); imaginary
                        # modes keep the trailing "i", so the column is text.
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
                                _freq_tbl.add_row(
                                    _label, _freq_str, f"{_ir[k]:.2f}"
                                )
                            else:
                                _freq_tbl.add_row(_label, _freq_str)
                        write(_freq_tbl.render() + "\n\n")
                        flush()

                        # Thermochemistry (RRHO ideal gas) from the harmonic
                        # frequencies -- ZPE + thermal U/H/S/G at T, p. Reuses
                        # the general vibeqc.thermo engine; method-agnostic
                        # (any reference that produced the Hessian).
                        _thermo_result_for_qvf = None
                        try:
                            from .thermo import (
                                ThermoOptions as _ThermoOptions,
                            )
                            from .thermo import (
                                compute_thermochemistry as _compute_thermo,
                            )

                            _topts = thermo_options or _ThermoOptions()
                            _thermo = _compute_thermo(
                                molecule, hessian_result, options=_topts
                            )
                            _thermo_result_for_qvf = _thermo
                            _Hced = 627.509474063  # Hartree -> kcal/mol
                            _slog.emit(
                                "thermochemistry",
                                temperature=float(_topts.temperature),
                                pressure=float(_topts.pressure),
                                symmetry_number=int(_topts.symmetry_number),
                                rotor_type=_thermo.rotor_type,
                                zpe=float(_thermo.zpe),
                                u_thermal=float(_thermo.u_thermal),
                                h_thermal=float(_thermo.h_thermal),
                                g_thermal=float(_thermo.g_thermal),
                                s_total=float(_thermo.s_total),
                                e_scf=float(getattr(result, "energy", float("nan"))),
                            )
                            _e0 = float(getattr(result, "energy", float("nan")))
                            write(
                                section_header("## Thermochemistry (RRHO ideal gas)")
                                + f"  T = {render_temperature(_topts.temperature, precision=2)} K"
                                f"   p = {_topts.pressure:.0f} Pa"
                                f"   s_rot = {_topts.symmetry_number}"
                                f"   rotor = {_thermo.rotor_type}\n"
                            )
                            if _thermo.n_imaginary_modes_excluded:
                                write(
                                    f"  (excluded {_thermo.n_imaginary_modes_excluded}"
                                    f" imaginary mode(s) from the partition function)\n"
                                )
                            write(
                                f"  Zero-point energy        = {render_energy_labeled(_thermo.zpe, width=16, precision=10)}"
                                f"  ({_thermo.zpe * _Hced:9.3f} kcal/mol)\n"
                                f"  Thermal corr. to U       = {render_energy_labeled(_thermo.u_thermal, width=16, precision=10)}\n"
                                f"  Thermal corr. to H       = {render_energy_labeled(_thermo.h_thermal, width=16, precision=10)}\n"
                                f"  Thermal corr. to G       = {render_energy_labeled(_thermo.g_thermal, width=16, precision=10)}\n"
                                f"  Total entropy S          = {render_energy_labeled(_thermo.s_total, width=16, precision=10)}/K"
                                f"  ({_thermo.s_total * _Hced * 1000:8.3f} cal/mol/K)\n"
                            )
                            if _e0 == _e0:  # not NaN
                                write(
                                    f"  E(elec) + ZPE            = {render_energy_labeled(_e0 + _thermo.zpe, width=16, precision=10)}\n"
                                    f"  H = E(elec) + H_corr     = {render_energy_labeled(_e0 + _thermo.h_thermal, width=16, precision=10)}\n"
                                    f"  G = E(elec) + G_corr     = {render_energy_labeled(_e0 + _thermo.g_thermal, width=16, precision=10)}\n"
                                )
                            write("\n")
                            flush()
                        except Exception as _thermo_exc:
                            write(
                                f"  (warning: thermochemistry not available: "
                                f"{type(_thermo_exc).__name__}: {_thermo_exc})\n\n"
                            )
                            flush()
                    except Exception as _hess_exc:
                        write(
                            "\n"
                            + section_header("## Vibrational Frequencies")
                            + f"  FAILED: {type(_hess_exc).__name__}: {_hess_exc}\n"
                        )
                        flush()
                        hessian_result = None

        # TD-DFT excited states (output to .out; NTOs handled in QVF below).
        _tddft_result = None
        _qvf_uvvis_data = None
        _is_uhf = False  # set by TD-DFT block below
        if tddft and has_mos and bool(getattr(result, "converged", False)):
            try:
                from vibeqc.tddft import (
                    run_tddft_casida as _run_tddft_casida,
                )
                from vibeqc.tddft import (
                    run_tddft_casida_uhf as _run_tddft_casida_uhf,
                )
                from vibeqc.tddft import (
                    run_tddft_tda as _run_tddft_tda,
                )
                from vibeqc.tddft import (
                    run_tddft_tda_uhf as _run_tddft_tda_uhf,
                )

                _is_uhf = bool(
                    hasattr(result, "mo_coeffs_alpha")
                    and not hasattr(result, "mo_coeffs")
                )
                _is_rohf = bool(
                    hasattr(result, "mo_coeffs")
                    and not hasattr(result, "mo_coeffs_alpha")
                    and molecule.multiplicity > 1
                )
                _density_ao = (
                    np.asarray(result.density, dtype=float)
                    if getattr(result, "density", None) is not None
                    else None
                )
                if _is_uhf:
                    _n_el = effective_electron_count(molecule, result)
                    _mult = molecule.multiplicity
                    _n_occ_a = int(getattr(result, "n_occ_a", 0) or 0)
                    _n_occ_b = int(getattr(result, "n_occ_b", 0) or 0)
                    if _n_occ_a <= 0 and _n_occ_b <= 0:
                        _n_occ_a = (_n_el + (_mult - 1)) // 2
                        _n_occ_b = _n_el - _n_occ_a
                    _uhf_tddft_runner = (
                        _run_tddft_casida_uhf
                        if tddft_type == "casida"
                        else _run_tddft_tda_uhf
                    )
                    _tddft_result = _uhf_tddft_runner(
                        molecule,
                        basis_obj,
                        np.asarray(result.mo_energies_alpha, dtype=float),
                        np.asarray(result.mo_energies_beta, dtype=float),
                        np.asarray(result.mo_coeffs_alpha, dtype=float),
                        np.asarray(result.mo_coeffs_beta, dtype=float),
                        _n_occ_a,
                        _n_occ_b,
                        n_states=tddft_n_states,
                        functional=functional,
                        density_alpha_ao=np.asarray(
                            result.density_alpha,
                            dtype=float,
                        ),
                        density_beta_ao=np.asarray(
                            result.density_beta,
                            dtype=float,
                        ),
                    )
                elif _is_rohf:
                    from vibeqc.tddft import (
                        run_tddft_tda_rohf as _run_tddft_tda_rohf,
                    )

                    _n_el = effective_electron_count(molecule, result)
                    _mult = molecule.multiplicity
                    _n_docc = (_n_el - (_mult - 1)) // 2
                    _n_socc = _mult - 1
                    _tddft_result = _run_tddft_tda_rohf(
                        molecule,
                        basis_obj,
                        np.asarray(result.mo_energies, dtype=float),
                        np.asarray(result.mo_coeffs, dtype=float),
                        _n_docc,
                        _n_socc,
                        n_states=tddft_n_states,
                    )
                elif tddft_type == "casida":
                    _tddft_result = _run_tddft_casida(
                        molecule,
                        basis_obj,
                        np.asarray(result.mo_energies, dtype=float),
                        np.asarray(result.mo_coeffs, dtype=float),
                        effective_electron_count(molecule, result) // 2,
                        n_states=tddft_n_states,
                        functional=functional,
                        density_ao=_density_ao,
                    )
                else:
                    _tddft_result = _run_tddft_tda(
                        molecule,
                        basis_obj,
                        np.asarray(result.mo_energies, dtype=float),
                        np.asarray(result.mo_coeffs, dtype=float),
                        effective_electron_count(molecule, result) // 2,
                        n_states=tddft_n_states,
                        functional=functional,
                        density_ao=_density_ao,
                    )
                # Emit to .out file.
                _td_method = _tddft_result.method
                _td_func = f" ({functional})" if functional else ""
                write(
                    f"\n  ## TD-DFT excited states"
                    f" ({_td_method}{_td_func})\n"
                    f"  {'─' * 50}\n"
                )
                # Excited-states table via the document-layer Table
                # primitive: columns + rule size to content instead of the
                # old fixed 6/10/10/10 widths. The trailing dominant-
                # transition column is free text, left-aligned.
                _es_tbl = Table([
                    Column("State", ">"),
                    Column("E (eV)", ">"),
                    Column("λ (nm)", ">"),
                    Column("f_osc", ">"),
                    Column("Dominant transition", "<"),
                ])
                for _st in _tddft_result.states:
                    _dom_str = ", ".join(
                        f"{occ}→{virt} ({abs(amp):.3f})"
                        for occ, virt, amp in _st.dominant_amplitudes[:3]
                    )
                    _es_tbl.add_row(
                        _st.index,
                        f"{_st.excitation_energy_ev:.4f}",
                        f"{_st.wavelength_nm:.1f}",
                        f"{_st.oscillator_strength:.4f}",
                        _dom_str,
                    )
                write(_es_tbl.render() + "\n\n")
                flush()
                # Package excitation energies for QVF spectra.uvvis.
                _qvf_uvvis_data = {
                    "energies_ev": [
                        float(_st.excitation_energy_ev)
                        for _st in _tddft_result.states
                    ],
                    "intensities": [
                        float(_st.oscillator_strength)
                        for _st in _tddft_result.states
                    ],
                    "wavelength_nm": [
                        float(_st.wavelength_nm)
                        for _st in _tddft_result.states
                    ],
                    "method": _td_method,
                }
                # UV/Vis spectrum (Gaussian-broadened stick spectrum).
                if tddft_spectrum and _tddft_result.states:
                    _emin = min(
                        _st.excitation_energy_ev for _st in _tddft_result.states
                    )
                    _emax = max(
                        _st.excitation_energy_ev for _st in _tddft_result.states
                    )
                    _pad = 1.0  # eV
                    _sigma = 0.15  # eV (typical vibronic broadening)
                    _npts = 200
                    _egrid = np.linspace(max(0, _emin - _pad), _emax + _pad, _npts)
                    _spec = np.zeros(_npts)
                    for _st in _tddft_result.states:
                        _spec += _st.oscillator_strength * np.exp(
                            -0.5 * ((_egrid - _st.excitation_energy_ev) / _sigma) ** 2
                        )
                    _spec_max = float(_spec.max()) if _spec.max() > 0 else 1.0
                    _spec /= _spec_max  # normalise to 1
                    write(
                        f"\n  ## UV/Vis spectrum (Gaussian, σ={_sigma} eV)\n"
                        f"  {'─' * 50}\n"
                    )
                    write(
                        f"  {'E (eV)':>10s}  {'Intensity':>10s}  "
                        f"{'├' + '─' * 40 + '┤'}\n"
                    )
                    _step = max(1, _npts // 40)
                    for _k in range(0, _npts, _step):
                        _bar = "█" * max(0, min(40, int(_spec[_k] * 40)))
                        write(f"  {_egrid[_k]:>10.4f}  {_spec[_k]:>10.4f}  {_bar}\n")
                    write("\n")
                    flush()
            except Exception as _tddft_exc:
                write(
                    f"\n  ## TD-DFT excited states\n"
                    f"  {'─' * 50}\n"
                    f"  FAILED: {type(_tddft_exc).__name__}:"
                    f" {_tddft_exc}\n"
                )
                flush()
                _tddft_result = None
                _qvf_uvvis_data = None

        # FD excited-state nuclear gradient (opt-in, costly).
        if tddft_gradient and _tddft_result is not None and _tddft_result.states:
            try:
                from vibeqc.excited_gradient import (
                    cis_state_gradients_fd as _cis_grad_fd,
                )
                from vibeqc.excited_gradient import (
                    make_hf_cis_energy_fn as _make_hf_cis_energy_fn,
                )

                _mol = molecule
                _Z = [int(a.Z) for a in _mol.atoms]
                _coords = np.array([a.xyz for a in _mol.atoms], dtype=float)
                _fn = _make_hf_cis_energy_fn(
                    _Z,
                    basis,
                    charge=_mol.charge,
                    spin="singlet" if _mol.multiplicity == 1 else "triplet",
                    n_states=min(tddft_n_states, 10),
                )
                _grads = _cis_grad_fd(_fn, _coords, states=[1])
                _grad1 = _grads.get(1)
                if _grad1 is not None:
                    # The surface is part of the result (issue #570): the
                    # gate above admits only the RHF + CIS (TDA) run, and
                    # the block says so rather than leaving the reader to
                    # assume it is the method the job was asked for.
                    write(
                        f"\n  ## Excited-state gradient (S₁, RHF/CIS(TDA) "
                        f"surface, FD, step=1e-3 Å)\n"
                        f"  {'─' * 50}\n"
                        f"  Surface: RHF ground state + CIS (TDA) excitation "
                        f"energy, state-tracked central differences\n"
                    )
                    write(
                        f"  {'Atom':>6s}  {'dE/dx':>12s}"
                        f"  {'dE/dy':>12s}  {'dE/dz':>12s}\n"
                    )
                    for _k, _g in enumerate(_grad1):
                        write(
                            f"  {_k + 1:>6d}  "
                            f"{_g[0]:>12.8f}  "
                            f"{_g[1]:>12.8f}  "
                            f"{_g[2]:>12.8f}\n"
                        )
                    write("\n")
                    flush()
            except Exception as _tdgrad_exc:
                write(
                    f"\n  ## Excited-state gradient\n"
                    f"  {'─' * 50}\n"
                    f"  FAILED: {type(_tdgrad_exc).__name__}:"
                    f" {_tdgrad_exc}\n"
                )
                flush()

        # Terminal timing follows every helper that can launch another native
        # SCF. No live iteration row can therefore appear after the claimed
        # job total or its terminal progress summary. The post-convergence
        # internal-stability analysis remains a distinct measured phase
        # (issue #205), not SCF-loop work.
        t_total = time.perf_counter() - t_job_start
        n_iter = getattr(result, "n_iter", 0)
        _t_stability = float(
            getattr(result, "stability_wall_s", 0.0) or 0.0
        )
        if not math.isfinite(_t_stability) or _t_stability <= 0.0:
            _t_stability = 0.0
        _loop_n_iter = int(n_iter)
        if (
            _t_stability > 0.0
            and int(getattr(result, "n_stability_restarts", 0) or 0) > 0
        ):
            _loop_n_iter = int(
                getattr(result, "n_iter_before_stability", 0) or n_iter
            )
        # A retried SCF (SAD / TRAH tail) folds the discarded attempt's own
        # analysis into ``t_scf``; only the returned solution's phase is
        # attributable, so clamp rather than report a negative loop wall.
        _t_scf_loop = max(t_scf - _t_stability, 0.0)
        iter_avg = (
            (_t_scf_loop / _loop_n_iter)
            if _loop_n_iter > 0
            else float("nan")
        )
        if _t_stability > 0.0:
            from .perf import active_tracker as _at_phase

            _phase_tracker = _at_phase()
            if _phase_tracker is not None:
                _t_stability_cpu = float(
                    getattr(result, "stability_cpu_s", 0.0) or 0.0
                )
                if (
                    not math.isfinite(_t_stability_cpu)
                    or _t_stability_cpu < 0.0
                ):
                    _t_stability_cpu = 0.0
                _phase_tracker.add_scope(
                    f"scf.{resolved_method}.stability_analysis",
                    _t_stability,
                    _t_stability_cpu,
                )
        timing_rows: list[tuple[str, float, Optional[str]]] = []
        if optimize:
            timing_rows.append(("Geometry optimization", t_opt, None))
        if resolved_method in _MLIP_METHODS:
            # MLIP: a single forward pass, not an SCF -- honest phase label,
            # no per-iteration average.
            timing_rows.append(("MLIP energy evaluation", t_scf, None))
        elif isinstance(result, SolverResult):
            timing_rows.append(("Solver total", t_scf, None))
            timing_rows.append(
                ("Solver avg. per iteration", iter_avg, f"({n_iter} iters)")
            )
        else:
            timing_rows.append(("SCF total", t_scf, None))
            if _t_stability > 0.0:
                timing_rows.append(
                    (
                        "SCF stability analysis",
                        _t_stability,
                        "(post-convergence)",
                    )
                )
            timing_rows.append(
                (
                    "SCF avg. per iteration",
                    iter_avg,
                    f"({_loop_n_iter} iters)",
                )
            )
        timing_rows.append(("Job total", t_total, None))
        write(
            "\n"
            + _format_timing_summary(timing_rows, label_width=32)
            + "\n"
        )
        write(
            f"  Used {threads_in_use} OpenMP thread"
            f"{'s' if threads_in_use != 1 else ''}.\n"
        )
        flush()

        # "## References" block (Phase O5b) -- the citation footer, emitted
        # LAST so it follows every post-SCF results block (vibrational
        # analysis, thermochemistry, TD-DFT) instead of appearing mid-report
        # ahead of them. Embeds the assembled citation list so the .out is
        # self-contained; goes through the citation printer's own channel
        # writer (mirrors write_scf_trace). For a plain SCF job the post-SCF
        # blocks above are all skipped, so this lands in the same place the
        # earlier emission did -- the .out is byte-identical there.
        if cite_block_text:
            write_references_block(block=cite_block_text)
            flush()

        # This is deliberately the final structured event. Optional
        # atomization, Hessian, and property helpers above may launch their
        # own native SCFs; keeping job_end here preserves append-order truth
        # and includes their wall time.
        _slog.emit(
            "job_end",
            total_wall_s=float(time.perf_counter() - t_job_start),
            scf_wall_s=float(t_scf),
            opt_wall_s=float(t_opt),
            n_iter=int(getattr(result, "n_iter", 0)),
            converged=bool(getattr(result, "converged", False)),
            energy=_terminal_energy,
            out_path=str(out_path),
            **_corr_fields,
        )

    # --- QVF visualisation archive (v1) ----------------------------------
    if output_qvf:
        try:
            from vibeqc.output.formats.qvf import (
                scf_history_from_result as _scf_history_from_result,
            )

            _qvf_path_for_warn = stem_sibling(output_stem, ".qvf")
            _qvf_scf_history = _scf_history_from_result(result)
            _bibtex = None
            if _refs is not None:
                try:
                    from vibeqc.output.citations.bibtex import format_bibtex

                    _bibtex = format_bibtex(_refs)
                except Exception as _bib_exc:
                    warn_output_failure(
                        _bib_exc,
                        _qvf_path_for_warn,
                        role="qvf_bibtex_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )
            _pop = _population_summary
            _qvf_bond_orders = None
            _qvf_dipole = None
            if write_population_file and has_mos:
                try:
                    from vibeqc.output.formats.population import (
                        compute_population_summary,
                    )

                    if _pop is None:
                        _pop = compute_population_summary(
                            result,
                            basis_obj,
                            molecule,
                            nuclear_charges=_nuclear_charges_used,
                        )
                    # Extract bond-order table for bond_orders QVF section.
                    if _pop.mayer_bonds:
                        # ``math`` is imported at module scope; a
                        # function-local re-import here made ``math`` a
                        # local name for all of run_job() and turned every
                        # earlier use into an UnboundLocalError.
                        _pos = [(a.xyz[0], a.xyz[1], a.xyz[2]) for a in molecule.atoms]
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
                                for (i, j, si, sj, order) in _pop.mayer_bonds
                            ],
                        }
                    # Extract dipole moment for manifest root metadata.
                    if _pop.dipole is not None:
                        _qvf_dipole = {
                            "total_debye": float(_pop.dipole["total_debye"]),
                            "vector_debye": [
                                float(_pop.dipole["x_ebohr"]) * 2.541746473,
                                float(_pop.dipole["y_ebohr"]) * 2.541746473,
                                float(_pop.dipole["z_ebohr"]) * 2.541746473,
                            ],
                            "origin": [
                                float(value)
                                for value in _pop.dipole["origin_bohr"]
                            ],
                        }
                except Exception as _qpop_exc:
                    warn_output_failure(
                        _qpop_exc,
                        _qvf_path_for_warn,
                        role="qvf_population_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            _qvf_vol_data = None
            _qvf_spin_data = None
            if has_mos and _cube_req.density:
                try:
                    from vibeqc.output.formats.qvf import qvf_density_data

                    _qvf_vol_data = qvf_density_data(
                        result,
                        basis_obj,
                        molecule,
                        spacing=cube_spacing,
                        padding=cube_padding,
                    )
                except Exception as _qvol_exc:
                    warn_output_failure(
                        _qvol_exc,
                        _qvf_path_for_warn,
                        role="qvf_density_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            # Spin density (volume.spin) for unrestricted wavefunctions.
            if has_mos and _cube_req.density and getattr(result, "density_alpha", None) is not None:
                try:
                    from vibeqc.cube import (
                        CubeGrid,
                        _density_on_grid,
                        make_uniform_grid,
                    )
                    from vibeqc.properties import _real_if_hermitian

                    _grid: CubeGrid = make_uniform_grid(
                        molecule,
                        spacing=cube_spacing,
                        padding=cube_padding,
                    )
                    _D_spin = _real_if_hermitian(
                        np.asarray(result.density_alpha)
                        - np.asarray(result.density_beta),
                        what="QVF spin density matrix",
                    )
                    _rho_spin = _density_on_grid(_D_spin, basis_obj, _grid)
                    _origin = np.asarray(_grid.origin, dtype=np.float64)
                    _span = np.asarray(_grid.voxel_vectors, dtype=np.float64)
                    _qvf_spin_data = {"Spin density": (_rho_spin, _origin, _span)}
                except Exception as _qspin_exc:
                    warn_output_failure(
                        _qspin_exc,
                        _qvf_path_for_warn,
                        role="qvf_spin_density_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            # Distance-based bond table for QVF bonds section.
            _qvf_bonds = None
            try:
                # Simple covalent-radii-based bond detection.
                # Radii from Cordero et al., Dalton Trans. (2008).
                _covalent_radii = {
                    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76,
                    7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
                    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
                    16: 1.05, 17: 1.02, 18: 1.06,
                    19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53,
                    24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
                    29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
                    34: 1.20, 35: 1.20, 36: 1.16,
                    37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64,
                    42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39,
                    47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39,
                    52: 1.38, 53: 1.39, 54: 1.40,
                    55: 2.44, 56: 2.15, 57: 2.07, 72: 1.75, 73: 1.70,
                    74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36,
                    79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48,
                    84: 1.40, 85: 1.50, 86: 1.50,
                }
                _atoms = list(molecule.atoms)
                _pos = np.array([a.xyz for a in _atoms], dtype=float)
                _bonds_list = []
                for _i in range(len(_atoms)):
                    for _j in range(_i + 1, len(_atoms)):
                        _d = float(np.linalg.norm(_pos[_i] - _pos[_j]))
                        _r_i = _covalent_radii.get(int(_atoms[_i].Z), 1.5)
                        _r_j = _covalent_radii.get(int(_atoms[_j].Z), 1.5)
                        _threshold = (_r_i + _r_j) * 1.3  # 30% tolerance
                        if _d < _threshold and _d > 0.4:  # exclude non-bonded
                            _bonds_list.append((_i, _j, 1.0))  # order=1 for now
                if _bonds_list:
                    _qvf_bonds = _bonds_list
            except Exception:
                pass  # optional; non-critical

            _qvf_mo_list = None
            if has_mos and _cube_req.mo_labels:
                try:
                    from vibeqc.output.formats.qvf import qvf_mo_data

                    _indices = requested_mo_indices(
                        _cube_req.mo_labels, result, molecule
                    )
                    _qvf_mo_list = qvf_mo_data(
                        result,
                        basis_obj,
                        molecule,
                        [int(i) for i, _ in _indices],
                        spacing=cube_spacing,
                        padding=cube_padding,
                    )
                except Exception as _qmo_exc:
                    warn_output_failure(
                        _qmo_exc,
                        _qvf_path_for_warn,
                        role="qvf_mo_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            # Package the wavefunction for the QVF archive so vibe-view
            # can resample any MO on demand (wavefunction.gto section).
            # Mirrors the periodic runner; gated on write_molden_file so
            # the .molden sidecar and the embedded basis+MO stay in sync.
            _qvf_wf = None
            if getattr(result, "natural_orbitals", None) is not None:
                # A correlated solver (CASCI/CASSCF) has no mean-field
                # orbitals, so the branch below never fires for it and the
                # archive would carry no wavefunction at all. Its natural
                # orbitals are a complete, orthonormal set with genuine
                # occupation numbers, which is the more informative object
                # anyway: the fractional occupancies are the multireference
                # character the calculation was run to find.
                try:
                    from vibeqc.output.formats.qvf import qvf_natural_wf_data

                    _qvf_wf = qvf_natural_wf_data(
                        result.natural_orbitals,
                        result.natural_occupations,
                        basis_obj,
                        molecule,
                    )
                except Exception as _qwf_exc:
                    warn_output_failure(
                        _qwf_exc,
                        _qvf_path_for_warn,
                        role="qvf_natural_wavefunction_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )
            elif write_molden_file and has_mos:
                try:
                    from vibeqc.output.formats.qvf import qvf_wf_data

                    _qvf_wf = qvf_wf_data(result, basis_obj, molecule)
                except Exception as _qwf_exc:
                    warn_output_failure(
                        _qwf_exc,
                        _qvf_path_for_warn,
                        role="qvf_wavefunction_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            # QTAIM topological analysis (opt-in, QVF-bound).
            _qvf_qtaim = None
            if qtaim and has_mos and bool(getattr(result, "converged", False)):
                try:
                    from vibeqc.qtaim import qtaim_analysis, qtaim_result_to_qvf

                    _qtaim_result = qtaim_analysis(
                        result,
                        basis_obj,
                        molecule,
                    )
                    _qvf_qtaim = qtaim_result_to_qvf(_qtaim_result)
                except Exception as _qtaim_exc:
                    warn_output_failure(
                        _qtaim_exc,
                        _qvf_path_for_warn,
                        role="qvf_qtaim_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            # IAO + IBO chemical-interpretation analysis (opt-in, QVF-bound).
            # Closed-shell only: the IAO charge formula below assumes a
            # doubly-occupied reference (Knizia eq 3, gamma = 2 sum_i |i><i|).
            _qvf_localized_wf = None
            _qvf_iao_charges = None
            _localize_methods = _resolve_localize_methods(localize)
            if (
                _localize_methods
                and has_mos
                and basis_obj is not None
                and bool(getattr(result, "converged", False))
                and not is_open_shell_result(result)
            ):
                try:
                    from vibeqc.iao import (
                        analyse_localization,
                        iao_unsupported_reason,
                    )
                    from vibeqc.output.formats.qvf import qvf_localized_wf_data

                    # Recomputed rather than reusing the citation block's
                    # ``_uses_ecp``: that name is bound inside a conditional
                    # branch further up and is not guaranteed to exist here.
                    _iao_block = iao_unsupported_reason(
                        molecule,
                        uses_ecp=_detect_uses_ecp(
                            rhf_options,
                            uhf_options,
                            rks_options,
                            uks_options,
                            result=result,
                        ),
                    )
                    if _iao_block is not None:
                        raise ValueError(_iao_block)

                    _n_occ = effective_electron_count(molecule, result) // 2
                    _mo = np.asarray(result.mo_coeffs)
                    if _mo.shape[0] != basis_obj.nbasis:
                        _mo = _mo.T
                    _occ_block = _mo[:, :_n_occ]
                    # Shared across criteria: the dipole integrals every
                    # centroid needs, and Boys needs outright.
                    from vibeqc import compute_dipole

                    _dip = compute_dipole(basis_obj)
                    _dipoles = np.stack(
                        [
                            np.asarray(_dip.x),
                            np.asarray(_dip.y),
                            np.asarray(_dip.z),
                        ],
                        axis=-1,
                    )
                    _qvf_localized_wf = []
                    for _method in _localize_methods:
                        _loc = analyse_localization(
                            molecule,
                            basis_obj,
                            _occ_block,
                            method=_method,
                            dipoles=_dipoles,
                        )
                        _qvf_localized_wf.append(
                            (
                                f"wf_localized_{_method.replace('-', '_')}",
                                qvf_localized_wf_data(
                                    _loc.coefficients,
                                    basis_obj,
                                    method=_method,
                                    charges=_loc.charges,
                                    atom_populations=_loc.atom_populations,
                                    centroids=_loc.centroids,
                                    n_centres=_loc.n_centres,
                                    reference_basis=_loc.reference_basis,
                                ),
                            )
                        )
                        # IAO charges are a property of the occupied space and
                        # so are identical for every criterion; take the first.
                        # They are also surfaced in atom_properties, next to
                        # the Mulliken / Loewdin / Hirshfeld charges a user
                        # compares them against.
                        if _qvf_iao_charges is None:
                            _qvf_iao_charges = _loc.charges
                except Exception as _ibo_exc:
                    warn_output_failure(
                        _ibo_exc,
                        _qvf_path_for_warn,
                        role="qvf_localized_wf_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            # TD-DFT excited states + NTO analysis (opt-in, QVF-bound).
            # TD-DFT computation runs before this block (inside the with-open
            # context so it can write to the .out file).  Here we only compute
            # NTOs and package them for QVF.
            _qvf_nto = None
            if tddft and _tddft_result is not None and nto and _tddft_result.states:
                try:
                    from vibeqc.output.formats.qvf import (
                        qvf_wf_data as _qvf_wf_data,
                    )
                    from vibeqc.tddft import (
                        compute_nto as _compute_nto,
                    )
                    from vibeqc.tddft import (
                        compute_nto_uhf as _compute_nto_uhf,
                    )

                    _gs_wf = _qvf_wf_data(result, basis_obj, molecule)
                    _shells = _gs_wf["basis"] if _gs_wf is not None else []
                    _pure = bool(_gs_wf.get("pure", True)) if _gs_wf else True
                    _n_basis = basis_obj.nbasis
                    _nto_list: list[dict[str, object]] = []

                    if _is_uhf:
                        # UHF: alpha and beta NTOs for each state.
                        _n_occ_a = int(getattr(result, "n_occ_a", 0))
                        _n_occ_b = int(getattr(result, "n_occ_b", 0))
                        for _state in _tddft_result.states:
                            _nto_uhf = _compute_nto_uhf(
                                _state,
                                np.asarray(result.mo_coeffs_alpha, dtype=float),
                                np.asarray(result.mo_coeffs_beta, dtype=float),
                                _n_occ_a,
                                _n_occ_b,
                            )
                            for _spin, _nto_res in _nto_uhf.items():
                                _hole_coeffs = np.ascontiguousarray(
                                    np.asarray(
                                        _nto_res.hole_orbitals_ao,
                                        dtype=np.float64,
                                    ).T
                                )
                                _n_hole = int(_hole_coeffs.shape[0])
                                _part_coeffs = np.ascontiguousarray(
                                    np.asarray(
                                        _nto_res.particle_orbitals_ao,
                                        dtype=np.float64,
                                    ).T
                                )
                                _n_part = int(_part_coeffs.shape[0])
                                _nto_list.append(
                                    {
                                        "hole": {
                                            "basis": _shells,
                                            "structure_ref": "structure",
                                            "pure": _pure,
                                            "mo_metadata": {
                                                "n_mo": _n_hole,
                                                "n_ao": _n_basis,
                                                "spin": "restricted",
                                                "orbital_kind": "natural",
                                                "energies": [0.0] * _n_hole,
                                                "occupations": [
                                                    _nto_res.hole_weights[i]
                                                    for i in range(_n_hole)
                                                ],
                                            },
                                            "mo_coefficients": _hole_coeffs,
                                        },
                                        "electron": {
                                            "basis": _shells,
                                            "structure_ref": "structure",
                                            "pure": _pure,
                                            "mo_metadata": {
                                                "n_mo": _n_part,
                                                "n_ao": _n_basis,
                                                "spin": "restricted",
                                                "orbital_kind": "natural",
                                                "energies": [0.0] * _n_part,
                                                "occupations": [
                                                    _nto_res.particle_weights[i]
                                                    for i in range(_n_part)
                                                ],
                                            },
                                            "mo_coefficients": _part_coeffs,
                                        },
                                        "state_index": int(_state.index),
                                        "excitation_energy_ev": float(
                                            _state.excitation_energy_ev
                                        ),
                                        "spin": _spin,
                                    }
                                )
                    else:
                        # Restricted: single-spin NTOs.
                        _n_occ = effective_electron_count(molecule, result) // 2
                        for _state in _tddft_result.states:
                            _nto_res = _compute_nto(
                                _state,
                                np.asarray(result.mo_coeffs, dtype=float),
                                _n_occ,
                            )
                            _hole_coeffs = np.ascontiguousarray(
                                np.asarray(
                                    _nto_res.hole_orbitals_ao, dtype=np.float64
                                ).T
                            )
                            _n_hole = int(_hole_coeffs.shape[0])
                            _hole_wf: dict[str, object] = {
                                "basis": _shells,
                                "structure_ref": "structure",
                                "pure": _pure,
                                "mo_metadata": {
                                    "n_mo": _n_hole,
                                    "n_ao": _n_basis,
                                    "spin": "restricted",
                                    "orbital_kind": "natural",
                                    "energies": [0.0] * _n_hole,
                                    "occupations": [
                                        _nto_res.hole_weights[i] for i in range(_n_hole)
                                    ],
                                },
                                "mo_coefficients": _hole_coeffs,
                            }
                            _part_coeffs = np.ascontiguousarray(
                                np.asarray(
                                    _nto_res.particle_orbitals_ao, dtype=np.float64
                                ).T
                            )
                            _n_part = int(_part_coeffs.shape[0])
                            _electron_wf: dict[str, object] = {
                                "basis": _shells,
                                "structure_ref": "structure",
                                "pure": _pure,
                                "mo_metadata": {
                                    "n_mo": _n_part,
                                    "n_ao": _n_basis,
                                    "spin": "restricted",
                                    "orbital_kind": "natural",
                                    "energies": [0.0] * _n_part,
                                    "occupations": [
                                        _nto_res.particle_weights[i]
                                        for i in range(_n_part)
                                    ],
                                },
                                "mo_coefficients": _part_coeffs,
                            }
                            _nto_list.append(
                                {
                                    "hole": _hole_wf,
                                    "electron": _electron_wf,
                                    "state_index": int(_state.index),
                                    "excitation_energy_ev": float(
                                        _state.excitation_energy_ev
                                    ),
                                }
                            )
                    _qvf_nto = _nto_list
                    # Write NTO molden files for visualisation.
                    if tddft_molden and _nto_list:
                        try:
                            for _entry in _nto_list:
                                _si = _entry["state_index"]
                                _hole_wf = _entry["hole"]
                                _electron = _entry["electron"]
                                _e_ev = _entry.get("excitation_energy_ev", 0)
                                _spin_sfx = (
                                    f"_{_entry['spin']}" if "spin" in _entry else ""
                                )
                                for _role, _wf, _sfx in [
                                    ("hole", _hole_wf, "hole"),
                                    ("electron", _electron, "electron"),
                                ]:
                                    _mo_c = np.asarray(
                                        _wf["mo_coefficients"]
                                    ).T  # (n_ao, n_mo)
                                    _n_mo = _mo_c.shape[1]
                                    _fake = type(
                                        "_FakeResult",
                                        (),
                                        {
                                            "mo_coeffs": _mo_c,
                                            "mo_energies": [0.0] * _n_mo,
                                        },
                                    )()
                                    _path = stem_sibling(output_stem, 
                                        f".nto_S{_si}{_spin_sfx}_{_sfx}.molden"
                                    )
                                    _output_writer.dispatch_role(
                                        "orbitals",
                                        runtime_path=_path,
                                        runtime_format="nto-molden",
                                        runtime_description=(
                                            "Natural transition orbital "
                                            f"{_role} Molden file for state {_si}."
                                        ),
                                        molecule=molecule,
                                        basis=basis_obj,
                                        result=_fake,
                                        raise_on_error=True,
                                    )
                        except Exception as _nto_mol_exc:
                            # This block runs after the .out channel has
                            # closed, so the warning cannot land in the
                            # .out file. It never did: the old code called
                            # ``f.write`` on the already-closed handle,
                            # which raised ValueError, which the enclosing
                            # handler below then reported as a
                            # "qvf_tddft_nto_prep" failure -- masking the
                            # real cause. Route it through the non-fatal
                            # output-failure taxonomy, which needs no
                            # channel and records the true role.
                            warn_output_failure(
                                _nto_mol_exc,
                                stem_sibling(output_stem, ".nto.molden"),
                                role="nto_molden",
                                category=OutputFailureKind.compatibility_fallback,
                            )
                except Exception as _tddft_exc:
                    warn_output_failure(
                        _tddft_exc,
                        _qvf_path_for_warn,
                        role="qvf_tddft_nto_prep",
                        category=OutputFailureKind.compatibility_fallback,
                    )

            from vibeqc.output.formats.qvf import (
                assemble_run_record,
                terminal_run_status,
            )

            # The .out channel closed above, so the on-disk log is the
            # complete run log; embed it (+ the driving script) so the
            # archive is a self-contained record of the calculation.
            _qvf_run_record = assemble_run_record(
                _output_plan, wall_seconds=t_total
            )
            _qvf_path = _output_writer.dispatch_role(
                "qvf",
                atomic=True,
                record_hostname=record_hostname,
                molecule=molecule,
                result=result,
                method=resolved_method,
                basis=basis,
                functional=functional,
                population_summary=_pop,
                bibtex_content=_bibtex,
                wall_seconds=t_total,
                run_record=_qvf_run_record,
                run_status=terminal_run_status(result),
                trajectory_frames=_traj_frames if _traj_frames else None,
                trajectory_energies=_traj_energies if _traj_energies else None,
                volume_data=_qvf_vol_data,
                spin_data=_qvf_spin_data,
                mo_data=_qvf_mo_list,
                wf_data=_qvf_wf,
                hessian_result=hessian_result,
                scf_history_data=_qvf_scf_history,
                bond_orders_data=_qvf_bond_orders,
                dipole_moment_data=_qvf_dipole,
                spin_diagnostic_data=_spin_diagnostic,
                qtaim_data=_qvf_qtaim,
                wf_localized_data=_qvf_localized_wf,
                iao_charges=_qvf_iao_charges,
                nto_data=_qvf_nto,
                uvvis_data=_qvf_uvvis_data,
                bonds_data=_qvf_bonds,
                job_spec={
                    "job_type": "molecular",
                    "method": resolved_method,
                    "basis": str(basis),
                    "functional": functional,
                    **(
                        {
                            "options": {
                                "parameter_identity": _parameter_identity,
                                "parameter_sha256": _parameter_sha256,
                            }
                        }
                        if _parameter_identity is not None
                        else {}
                    ),
                },
                thermochemistry_data=(
                    {
                        "zpve_eh": float(_thermo_result_for_qvf.zpe),
                        "enthalpy_eh": float(
                            getattr(result, "energy", 0.0)
                            + _thermo_result_for_qvf.h_thermal
                        ),
                        "entropy_cal_mol_k": float(
                            _thermo_result_for_qvf.s_total * 627.509474063 * 1000
                        ),
                        "gibbs_free_energy_eh": float(
                            getattr(result, "energy", 0.0)
                            + _thermo_result_for_qvf.g_thermal
                        ),
                        "temperature_k": float(_thermo_result_for_qvf.temperature),
                        "pressure_atm": float(
                            _thermo_result_for_qvf.pressure / 101325.0
                        ),
                    }
                    if _thermo_result_for_qvf is not None
                    else None
                ),
                raise_on_error=True,
            )[0]
        except Exception as _qvf_exc:
            warn_writer_failure(
                _qvf_exc,
                stem_sibling(output_stem, ".qvf"),
                role="qvf_archive",
                category=OutputFailureKind.optional_artifact,
                writer=_output_writer,
            )

    # Commit coordinator-owned streams before the potentially very large
    # Molden text wavefunction. Single-shot compact artefacts were recorded
    # inline by dispatch_role; these streams close with the contexts above and
    # need this explicit sweep. If a scheduler kills Molden export, the live
    # manifest can therefore still direct a harvester to every settled compact
    # result (issue #28).
    #
    # The manifest lifecycle owns its own self-excluded outcome row: hashing
    # it here would necessarily become stale when a later record() / finish()
    # rewrite changes the file.
    if _terminal_progress_fields is not None:
        _output_writer.update_progress(**_terminal_progress_fields)
    elif _headline_progress_fields is not None:
        _output_writer.update_progress(**_headline_progress_fields)

    _stream_roles = {"log", "perf", "structured", "trajectory"}
    for _pf in _output_plan.files:
        if _pf.role not in _stream_roles:
            continue
        if _pf.path.is_file():
            try:
                _output_writer.record(_pf.path)
            except Exception as _rec_exc:
                # record() stats + hashes the file; a brittle write
                # at wrap-up must never tank a finished job. We do
                # surface it though -- a manifest with stale rows
                # confuses ``vq fetch``.
                warn_writer_failure(
                    _rec_exc,
                    _pf.path,
                    role=f"manifest_record_{_pf.role}",
                    category=OutputFailureKind.manifest_recording,
                    writer=_output_writer,
                )

    # Keep the potentially hundreds-of-megabytes text wavefunction last.
    # Population, citations, geometry, the closed output streams, and the
    # atomic QVF archive have all reached disk and the manifest by this
    # boundary.
    if _trexio_requested and has_mos:
        # Compact binary (or small text) container; written before the
        # potentially very large Molden text so a Molden failure cannot
        # take it down. The adapter refuses ECP runs (the `ecp` group is
        # not written in this increment) rather than emit an incomplete
        # Hamiltonian, so the ECP signal is passed explicitly.
        with plog.stage("write_trexio", detail=str(_trexio_path.name)):
            _output_writer.dispatch_role(
                "orbitals",
                only_format="trexio",
                molecule=molecule,
                basis=basis_obj,
                result=result,
                trexio_backend=_trexio_backend,
                trexio_description=(
                    f"{resolved_method.upper()}/{basis} job "
                    f"{output_stem.name}."
                ),
                uses_ecp=_detect_uses_ecp(
                    rhf_options,
                    uhf_options,
                    rks_options,
                    uks_options,
                    result=result,
                ),
                raise_on_error=True,
            )

    if write_molden_file and has_mos:
        with plog.stage("write_molden", detail=str(molden_path.name)):
            _output_writer.dispatch_role(
                "orbitals",
                only_format="molden",
                molecule=molecule,
                basis=basis_obj,
                result=result,
                title=str(output_stem.name),
                raise_on_error=True,
            )

    # Finalise only after every guaranteed plan row, including Molden, was
    # recorded. Non-converged molecular SCF routes raise before this point so
    # failed wavefunctions cannot be mistaken for publishable outputs.
    _final_wall_seconds = time.perf_counter() - t_job_start
    _output_writer.finish(wall_seconds=_final_wall_seconds)

    if _tddft_result is not None:
        try:
            result.tddft = _tddft_result
        except AttributeError:
            # pybind11 result types (RHFResult, RKSResult, ...) don't
            # carry __dict__ and object.__setattr__ also fails;
            # dynamic_attr() on the C++ side is the real fix.
            # For now, the TDDFT result is still accessible through
            # the QVF output; we just can't attach it to the SCF result.
            pass

    # Terminal checkpoint frame: label the checkpoint QVF "converged" so a
    # live viewer knows the job settled (final geometry + converged SCF).
    if _checkpointer.enabled:
        _checkpointer.finalize(
            "converged",
            molecule=molecule,
            result=result,
            method=resolved_method,
            basis=basis,
            functional=functional,
        )

    # The human terminal summary is last: optional QVF, Molden, manifest, and
    # checkpoint finalization have all settled before this flushed line.
    _reported_wall_seconds = time.perf_counter() - t_job_start
    plog.info(
        f"Job total {_reported_wall_seconds:.2f}s -- output written to {out_path}"
    )

    return result


__all__ = ["run_job"]
