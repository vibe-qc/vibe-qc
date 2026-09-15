"""Pre-flight memory estimator + budget enforcement.

Goal
----

Gaussian-basis and semiempirical vibe-qc drivers covered by this module can
estimate their peak memory *before* starting the numerical driver, compare
against the machine's available RAM, and abort with a helpful message if the
calculation would otherwise thrash-to-disk or crash the system. The user can
opt in to running anyway by passing ``memory_override=True`` to ``run_job``
(or by calling the explicit ``check_memory`` API with ``allow_exceed=True`` in
a low-level workflow).

Public API
----------

.. autoclass:: MemoryEstimate
.. autoclass:: InsufficientMemoryError
.. autofunction:: estimate_memory
.. autofunction:: estimate_neb_memory
.. autofunction:: check_memory
.. autofunction:: available_memory_bytes

The estimators are deliberately conservative -- they return a peak
upper bound, not a measured footprint.  Sequential phases are represented
as peaks rather than summed as if every transient were live at once. The
output feeds two things:

1. A summary written near the top of each supported ``run_job`` text output
   (the "vibe-qc estimates this calculation will require X GB" block).
2. The pre-flight abort -- ``run_job`` calls ``check_memory`` with the
   estimate before constructing the SCF driver.

For example, a raw 12.4 GiB peak appears as 18.6 GB after the default 1.5x
headroom factor (rendered by ``MemoryEstimate.format``):

    vibe-qc estimates this calculation will require ~18.6 GB of memory:
        ERI tensor       11.60 GB
        Fock + density   409.6 MB
        AO evaluation    307.2 MB
        DIIS history     102.4 MB
    Available on this machine: 119.8 GB. Proceeding.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import comb
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Optional

from ._vibeqc_core import (
    BasisSet,
    Functional,
    MOLECULAR_XC_GRID_BATCH_SIZE,
    MOLECULAR_XC_GRID_MAX_WORKERS,
    MOLECULAR_XC_KERNEL_MAX_WORKERS,
    Molecule,
)

if TYPE_CHECKING:
    from .semiempirical.routes import SemiempiricalRoutePlan

__all__ = [
    "MemoryEstimate",
    "InsufficientMemoryError",
    "BatchMemoryPlan",
    "estimate_memory",
    "estimate_neb_memory",
    "check_memory",
    "available_memory_bytes",
    "format_memory_report",
    "plan_batch_memory",
    "estimate_semiempirical_memory",
    "estimate_periodic_multik_gdf",
    "estimate_periodic_xc_value",
    "estimate_periodic_xc_gradient",
    "estimate_periodic_gpw_gapw",
    "estimate_skala_xc_memory",
    "check_periodic_gdf_memory",
]


# Compact multi-k GPW evaluates Bloch AOs in point batches.  This is the
# maximum size of one complex (n_batch, n_basis) AO table; the driver imports
# the same constant so the execution path and preflight describe one bound.
_COMPACT_BLOCH_AO_BATCH_BYTES = 64 * 1024**2

# Preserve the iteration-invariant cache for small compact jobs when every
# k-point table fits under one fixed cap. Larger jobs use the batch bound above.
_COMPACT_BLOCH_AO_CACHE_BYTES = 1024**3

# SKALA-1.1's measured first-derivative/autograd peak is 6,680 float64
# elements per fine-grid point.  Keep the multiplication explicit: the exact
# calibration is 53,440 bytes/point, not the nearby but larger ``53 KiB``.
_SKALA_MODEL_FIRST_DERIVATIVE_ELEMENTS_PER_POINT = 6680
_SKALA_MODEL_FIRST_DERIVATIVE_BYTES_PER_POINT = (
    _SKALA_MODEL_FIRST_DERIVATIVE_ELEMENTS_PER_POINT * 8
)
_SKALA_DEFAULT_MODEL_CHUNK_TARGET_POINTS = 8192
_SKALA_FUNCTIONAL_NAMES = frozenset(
    ("skala", "skala-1.1", "skala-1.1-rev1")
)

# Full-grid storage which is live at the external C++/Python callback
# boundary.  Each callback input has fifteen float64 values per point
# (coordinates 3, final/raw weights 2, and spin rho/grad/tau 10), and each
# result has ten float64 feature adjoints.  Charge one complete copy of both
# sides in C++ and Python.  The final +1 is Grid::atomic_weights, newly needed
# by SKALA but not part of the ordinary immutable-grid allowance below.
_EXTERNAL_XC_FULL_GRID_FLOAT64_VALUES_PER_POINT = 2 * (15 + 10) + 1

# Grid::atom_of_point is copied from the source grid into ExternalXCInput.
# The source copy is already part of the ordinary immutable-grid allowance.
_EXTERNAL_XC_FULL_GRID_INDEX_BYTES_PER_POINT = 4


# ----------------------------------------------------------------------
# Raw probe of the machine
# ----------------------------------------------------------------------

_CGROUP_ROOT = "/sys/fs/cgroup"
_CGROUP_V1_MEMORY_ROOT = "/sys/fs/cgroup/memory"
_CGROUP_UNLIMITED_THRESHOLD_BYTES = 1 << 60


def _read_ascii_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="ascii") as f:
            return f.read().strip()
    except (OSError, UnicodeError):
        return None


def _cgroup_directories(root: str, group_path: str) -> list[str]:
    """Return the current cgroup directory followed by its ancestors."""
    root = os.path.abspath(root)
    relative = os.path.normpath("/" + group_path).lstrip("/")
    current = os.path.abspath(os.path.join(root, relative))
    try:
        if os.path.commonpath((root, current)) != root:
            return []
    except ValueError:
        return []

    directories = []
    while True:
        directories.append(current)
        if current == root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return directories


def _cgroup_remaining_bytes(
    root: str,
    group_path: str,
    *,
    limit_file: str,
    usage_file: str,
) -> int | None:
    """Return the tightest finite remaining allowance in a cgroup tree."""
    remaining = []
    for directory in _cgroup_directories(root, group_path):
        raw_limit = _read_ascii_file(os.path.join(directory, limit_file))
        if raw_limit is None or raw_limit == "max":
            continue
        raw_usage = _read_ascii_file(os.path.join(directory, usage_file))
        if raw_usage is None:
            continue
        try:
            limit = int(raw_limit)
            usage = int(raw_usage)
        except ValueError:
            continue
        # Linux cgroup v1 represents "unlimited" with a very large integer.
        if limit < 0 or limit >= _CGROUP_UNLIMITED_THRESHOLD_BYTES:
            continue
        # Zero is reserved by the public API for an unknown probe. A known,
        # exhausted cgroup therefore reports one byte and fails closed.
        remaining.append(max(1, limit - max(0, usage)))
    return min(remaining) if remaining else None


def _linux_cgroup_available_memory_bytes() -> int | None:
    """Return memory remaining under Linux cgroup v2 or v1 constraints."""
    memberships = _read_ascii_file("/proc/self/cgroup")
    if memberships is None:
        return None

    remaining = []
    for line in memberships.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, group_path = fields
        if hierarchy == "0" and controllers == "":
            available = _cgroup_remaining_bytes(
                _CGROUP_ROOT,
                group_path,
                limit_file="memory.max",
                usage_file="memory.current",
            )
        elif "memory" in controllers.split(","):
            available = _cgroup_remaining_bytes(
                _CGROUP_V1_MEMORY_ROOT,
                group_path,
                limit_file="memory.limit_in_bytes",
                usage_file="memory.usage_in_bytes",
            )
        else:
            continue
        if available is not None:
            remaining.append(available)
    return min(remaining) if remaining else None


def _macos_available_memory_bytes() -> int:
    """Return the number of bytes macOS reports as reclaimable.

    Parses ``vm_stat`` for free + inactive + speculative + purgeable pages
    and multiplies by the page size from ``sysctl vm.pagesize``.  This is the
    same set of pages the system considers available for new allocations
    before paging out; it matches what ``psutil.virtual_memory().available``
    reports on Darwin.

    Returns 0 on any failure so callers treat it as "unknown" rather than
    "no memory".
    """
    try:
        import subprocess as _sp
    except ImportError:
        return 0

    try:
        # Page size — default to 16384 (Apple Silicon's 16 KiB) if the
        # sysctl call fails; the resulting byte count is wrong but
        # non-zero, so the caller at least gets a positive number.
        page_size = 16384
        ps = _sp.run(
            ["sysctl", "-n", "vm.pagesize"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if ps.returncode == 0 and ps.stdout.strip():
            try:
                page_size = int(ps.stdout.strip())
            except ValueError:
                pass

        vm = _sp.run(
            ["vm_stat"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if vm.returncode != 0:
            return 0

        counts: dict[str, int] = {}
        for line in vm.stdout.splitlines():
            line = line.strip().rstrip(".")
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().strip('"')
                try:
                    counts[key] = int(val.strip())
                except ValueError:
                    pass

        free = counts.get("Pages free", 0)
        inactive = counts.get("Pages inactive", 0)
        speculative = counts.get("Pages speculative", 0)
        purgeable = counts.get("Pages purgeable", 0)
        # wired and active are not reclaimable without swapping.
        return (free + inactive + speculative + purgeable) * page_size
    except (OSError, ValueError, _sp.TimeoutExpired):
        return 0


def _orchestrator_memory_limit_bytes() -> int | None:
    """Return the tightest memory limit declared by an orchestrator.

    Reads ``VIBEQC_MEMORY_LIMIT_BYTES`` (explicit bytes) and ``VQ_MEM_MB``
    (vq scheduler allocation in MB).  Returns ``None`` when neither is set.
    """
    limits: list[int] = []

    raw_bytes = os.environ.get("VIBEQC_MEMORY_LIMIT_BYTES")
    if raw_bytes is not None:
        try:
            limit = int(raw_bytes)
            if limit > 0:
                limits.append(limit)
        except (ValueError, TypeError):
            pass

    raw_mb = os.environ.get("VQ_MEM_MB")
    if raw_mb is not None:
        try:
            limit_mb = int(raw_mb)
            if limit_mb > 0:
                limits.append(limit_mb * 1024 * 1024)
        except (ValueError, TypeError):
            pass

    return min(limits) if limits else None


def available_memory_bytes():
    """Best-effort available-RAM probe for the current process.

    The host-wide probe prefers ``psutil`` when installed and falls back to
    platform-specific mechanisms. On Linux, a finite cgroup v2 or v1 memory
    allowance bounds that host value so scheduler and container allocations
    are respected.

    An explicit ``VIBEQC_MEMORY_LIMIT_BYTES`` or ``VQ_MEM_MB`` environment
    variable takes priority over every probe, so orchestrators (``vq submit``,
    Docker, SLURM, …) can directly declare the job's memory budget regardless
    of platform or containerisation.

    Returns ``0`` only when no probe succeeds, in which case the
    caller should treat the result as "unknown" rather than "no
    memory".
    """
    # ── Explicit override for container/orchestrator memory limits ──
    orch_limit = _orchestrator_memory_limit_bytes()
    if orch_limit is not None:
        # On Linux a cgroup may also be active; return the tighter bound.
        if sys.platform.startswith("linux"):
            cg = _linux_cgroup_available_memory_bytes()
            if cg is not None:
                return min(orch_limit, cg)
        return orch_limit

    host_available = 0

    # Preferred: psutil if present (optional dep).
    try:
        import psutil  # type: ignore[import-not-found]

        host_available = int(psutil.virtual_memory().available)
    except ImportError:
        pass

    # Linux: /proc/meminfo has MemAvailable since kernel 3.14.
    if host_available <= 0 and sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        host_available = kb * 1024
                        break
        except OSError:
            pass

    # macOS: parse ``vm_stat`` for pages the kernel considers reclaimable.
    # ``free + inactive + speculative + purgeable`` approximates the same
    # class of pages that the "Memory Pressure" gauge uses (and mirrors
    # what ``psutil`` reports as ``available`` on Darwin).  Only the total
    # physical page count (SC_PHYS_PAGES) was available before, which is
    # nearly useless as a budget cap — it always returns the full installed
    # RAM and masks every real capacity limit.
    if host_available <= 0 and sys.platform == "darwin":
        host_available = _macos_available_memory_bytes()

    # Final fallback: total installed RAM via sysconf.  This is a blunt
    # instrument — it reports *total* memory, not *available* — so it
    # cannot catch a bad_alloc when the host is busy.  Use it only when
    # every real probe has failed.
    if host_available <= 0:
        try:
            if hasattr(os, "sysconf"):
                page = os.sysconf("SC_PAGE_SIZE")
                n_pages = os.sysconf("SC_PHYS_PAGES")
                if page > 0 and n_pages > 0:
                    host_available = int(page) * int(n_pages)
        except (ValueError, OSError):
            pass

    if sys.platform.startswith("linux"):
        cgroup_available = _linux_cgroup_available_memory_bytes()
        if cgroup_available is not None:
            if host_available > 0:
                host_available = min(host_available, cgroup_available)
            else:
                host_available = cgroup_available

    return max(0, host_available)


# ----------------------------------------------------------------------
# MemoryEstimate -- carrier + formatter
# ----------------------------------------------------------------------

_GB = 1024**3
_DEFAULT_MEMORY_HEADROOM = 1.5
_PYTHON_RUNTIME_FLOOR_BYTES = 100 * 1024**2
_CCSD_VVVV_INCORE_BUDGET_BYTES = 2 * 1024**3
_CCSD_VVVV_TILE_ROWS = 2048
_DF_RESIDENT_BLOCKS = 2
_DF_CONSTRUCTION_SAFETY_FACTOR = 2.0

# Default DLPNO pair screening makes the retained strong-pair list linear in
# system size.  Across 41 completed S22 DLPNO-CCSD(T)/TightPNO profiles the
# largest observed list contained 8.83 pairs per active occupied orbital.  Ten
# gives the pre-headroom estimate a 13% count margin; the job-wide 1.5x factor
# remains the final protection against allocator and library overhead.  A
# tighter-than-profiled pair cutoff, non-pair residual domain, or pure-Python
# residual route always charges the full pair list.
_DLPNO_STRONG_PAIRS_PER_OCC = 10
_DLPNO_PROFILED_PAIR_CUTOFF = 5.0e-5
_DLPNO_PROFILED_PNO_CUTOFF = 1.0e-8
_DLPNO_TRIPLES_TILED_THRESHOLD_BYTES = 512 * 1024**2
_DLPNO_COMPOSITION_BOUND_DETAIL = (
    "composition-level bound; geometry-dependent pair/PNO locality is not modeled"
)
# GNU OpenMP worker stacks plus thread-local libint/Eigen/BLAS scratch remain
# live beside every numerical phase.  The 16-MiB bound per active worker is
# pinned by the profiled OMP=20/48/64 S22 inventory; treating it as a peer of
# the 100-MiB Python floor underpredicts small, many-thread calculations.
_DLPNO_NATIVE_WORKER_BYTES = 16 * 1024**2


def _omp_max_threads() -> int:
    """Return the OpenMP worker count the next native region will use.

    Querying the compiled runtime keeps preflight aligned with
    ``omp_get_max_threads()`` after :func:`vibeqc.set_num_threads`, including
    ``run_job(num_threads=...)``.  Do not impose an estimator-only many-core
    cap: the native triples planner has none, so doing so would undercount
    aggregate worker scratch on hosts with more than 128 OpenMP workers.
    """
    try:
        from ._vibeqc_core import get_num_threads

        return max(1, int(get_num_threads()))
    except Exception:
        # Import-only documentation builds may not have a native extension.
        raw = os.environ.get("OMP_NUM_THREADS", "").strip()
        try:
            return max(1, int(raw)) if raw else max(1, os.cpu_count() or 1)
        except ValueError:
            return max(1, os.cpu_count() or 1)


def _memory_headroom_factor() -> float:
    """Safety factor applied to raw estimates.

    HPC jobs routinely carry OS, Python, libint, MPI, filesystem, and allocator
    overhead beyond the arrays we can count directly. Keep the default
    conservative and let site wrappers tune it through an environment variable.
    """
    raw = os.environ.get("VIBEQC_MEMORY_HEADROOM")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MEMORY_HEADROOM
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("VIBEQC_MEMORY_HEADROOM must be a float >= 1.0") from exc
    if value < 1.0:
        raise ValueError("VIBEQC_MEMORY_HEADROOM must be >= 1.0")
    return value


def _add_python_runtime_floor(by_category: dict[str, int]) -> None:
    by_category["Python runtime + NumPy overhead"] = max(
        int(by_category.get("Python runtime + NumPy overhead", 0)),
        _PYTHON_RUNTIME_FLOOR_BYTES,
    )


def _memory_estimate(by_category: dict[str, int]) -> "MemoryEstimate":
    if by_category:
        _add_python_runtime_floor(by_category)
    return MemoryEstimate(by_category=by_category)


@dataclass
class MemoryEstimate:
    """Peak memory estimate for a calculation.

    All byte counts are integers (pre-headroom). ``total_bytes``
    multiplies the raw peak by ``headroom_factor`` so the headline figure
    already carries a safety margin. Most estimators are a single phase and
    therefore use the sum of ``by_category``. Sequential post-HF estimators
    populate ``phase_peaks``; their raw peak is the largest phase instead of
    the physically impossible sum of SCF, integral-build, solver, and
    perturbative-correction transients.

    ``dims`` carries key scale dimensions (``n_basis``, ``n_occ``,
    ``n_vir``, ``n_aux``, ``n_threads``) so the formatted output can
    show why the estimate is the size it is. ``safety_factor`` records
    the largest estimator-specific safety factor already folded into a
    category; it is separate from the job-wide ``headroom_factor``.
    Estimators should populate these fields with the values they used.
    """

    by_category: dict[str, int] = field(default_factory=dict)
    headroom_factor: float = field(default_factory=_memory_headroom_factor)
    dims: dict[str, int] = field(default_factory=dict)
    safety_factor: float = 1.0
    category_details: dict[str, str] = field(default_factory=dict)
    phase_peaks: dict[str, int] = field(default_factory=dict)

    def _with_dims(self, **kwargs: int) -> "MemoryEstimate":
        """Attach key scale dimensions for the formatted output.

        Typical call::

            est._with_dims(n_basis=236, n_occ=24, n_vir=212, n_aux=708)

        Returns self so callers can chain it on the return line.
        """
        self.dims.update(kwargs)
        return self

    @property
    def raw_total_bytes(self) -> int:
        if self.phase_peaks:
            return max(self.phase_peaks.values(), default=0)
        return sum(self.by_category.values())

    @property
    def total_bytes(self) -> int:
        return int(self.raw_total_bytes * self.headroom_factor)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / _GB

    def format(
        self,
        available: Optional[int] = None,
        *,
        status: Literal[
            "Proceeding", "ABORTING", "Proceeding (override)"
        ] = "Proceeding",
    ) -> str:
        """Render the standard memory-report block. ``available`` is the
        current process's available RAM in bytes; if ``None`` we probe live."""
        if available is None:
            available = available_memory_bytes()

        # Longest category label sets the column width.
        label_width = max(
            (len(k) for k in self.by_category),
            default=0,
        )
        label_width = max(label_width, 12)

        # Headline precision adapts to magnitude so tiny calcs don't
        # collapse to "~0.0 GB" and huge ones don't overflow columns.
        if self.total_gb >= 10:
            headline = f"~{self.total_gb:.1f} GB"
        elif self.total_gb >= 0.1:
            headline = f"~{self.total_gb:.2f} GB"
        else:
            mb = self.total_bytes / (1024**2)
            headline = f"~{mb:.1f} MB"
        lines = [
            f"vibe-qc estimates this calculation will require {headline} of memory:",
        ]

        # Show key scaling dimensions when available.
        if self.dims:
            dim_parts = []
            for key in (
                "n_basis",
                "n_occ",
                "n_vir",
                "n_frozen_core",
                "n_aux",
                "n_threads",
            ):
                if key in self.dims:
                    dim_parts.append(f"{key}={self.dims[key]}")
            if dim_parts:
                lines.append(f"  Dimensions: {', '.join(dim_parts)}")

        if self.phase_peaks:
            lines.append("  Peak phases (maximum used; phases are not additive):")
            phase_width = max(len(label) for label in self.phase_peaks)
            for label, size in self.phase_peaks.items():
                if size >= _GB:
                    value_str = f"{size / _GB:6.2f} GB"
                else:
                    value_str = f"{size / (1024**2):6.1f} MB"
                lines.append(f"    {label:<{phase_width}s} {value_str}")

        for label, size in self.by_category.items():
            if size >= _GB:
                gb = size / _GB
                value_str = f"{gb:6.2f} GB"
            else:
                mb = size / (1024 * 1024)
                if mb >= 0.95:
                    value_str = f"{mb:6.1f} MB"
                else:
                    value_str = f"{mb:6.2f} MB"
            detail = self.category_details.get(label)
            if detail:
                value_str += f" ({detail})"
            lines.append(f"    {label:<{label_width}s} {value_str}")
        # Trailing status line
        if available <= 0:
            lines.append("(available memory could not be probed on this platform).")
            lines.append(f"{status}.")
        else:
            avail_gb = available / _GB
            lines.append(f"Available on this machine: {avail_gb:.1f} GB. {status}.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


# ----------------------------------------------------------------------
# Enforcement
# ----------------------------------------------------------------------


class InsufficientMemoryError(MemoryError):
    """Raised by ``check_memory`` when the estimate exceeds available
    RAM and the caller did not request an override."""


def check_memory(
    estimate: MemoryEstimate,
    *,
    allow_exceed: bool = False,
    available: Optional[int] = None,
) -> None:
    """Abort if ``estimate.total_bytes > available`` and
    ``allow_exceed`` is false.

    ``available`` may be passed in for reproducible tests; leave
    ``None`` to probe the live machine.
    """
    avail = available if available is not None else available_memory_bytes()
    if avail <= 0:
        # No probe worked -- we can't enforce; the caller's risk.
        return
    if estimate.total_bytes <= avail:
        return
    if allow_exceed:
        return

    # ── Build the abort message ──
    if avail >= _GB:
        alloc_str = f"{avail / _GB:.1f} GB"
    else:
        alloc_str = f"{avail / (1024**2):.0f} MB"
    est_str = f"{estimate.total_gb:.1f} GB" if estimate.total_gb >= 1.0 else f"{estimate.total_bytes / (1024**2):.0f} MB"
    msg = (
        f"Estimated peak memory {est_str} exceeds effective allocation "
        f"{alloc_str} — refusing to proceed. "
    )
    if os.environ.get("VQ_MEM_MB"):
        msg += (
            "Either increase the scheduler memory request "
            "(``vq submit --mem-mb=N``) or use a smaller basis set."
        )
    else:
        msg += (
            "Either pass ``memory_override=True`` to ``run_job`` or call "
            "``check_memory(..., allow_exceed=True)`` in a low-level "
            "workflow to proceed anyway, or use a smaller basis / "
            "density fitting / integral-direct SCF to shrink the estimate."
        )
    msg += "\n\n" + estimate.format(available=avail, status="ABORTING") + "\n\n"
    msg += (
        "InsufficientMemoryError: Pass ``memory_override=True`` to ``run_job`` "
        "or call ``check_memory(..., allow_exceed=True)`` in a low-level "
        "workflow to proceed anyway. To shrink the estimate: "
        "use a smaller basis; enable density fitting (``density_fit=True`` + "
        "an ``aux_basis``); or the integral-direct SCF "
        "(``options.scf_mode = SCFMode.DIRECT``), which avoids the in-core "
        "4-index ERI "
        "tensor. (Post-HF / DLPNO methods inherit this from their SCF "
        "reference options.)"
    )
    raise InsufficientMemoryError(msg)


def estimate_neb_memory(
    per_image_estimate: MemoryEstimate | int,
    *,
    n_images: int,
    n_jobs: int,
    n_atoms: int = 0,
    n_basis: int = 0,
    open_shell: bool = False,
    finite_difference_evaluations: int = 1,
    warm_start: bool = True,
) -> MemoryEstimate:
    """Peak-memory estimate for one NEB outer iteration.

    ``n_images`` is the number of intermediate images. ``per_image_estimate``
    is the raw peak for a single image evaluator (usually the molecular
    :func:`estimate_memory` result for the SCF method). The estimate charges
    the image workers that can be live at once, the coordinate/force band
    arrays, density warm-start storage retained across outer iterations, and
    the additional finite-difference gradient scratch used by periodic NEB.
    """
    image_count = max(1, int(n_images))
    if isinstance(per_image_estimate, MemoryEstimate):
        per_image_bytes = int(per_image_estimate.raw_total_bytes)
        headroom = float(per_image_estimate.headroom_factor)
    else:
        per_image_bytes = int(per_image_estimate)
        headroom = 1.2
    per_image_bytes = max(0, per_image_bytes)

    requested_jobs = int(n_jobs)
    if requested_jobs < 0:
        worker_count = min(image_count, max(1, os.cpu_count() or 1))
    elif requested_jobs == 0:
        worker_count = min(image_count, max(1, os.cpu_count() or 1))
    else:
        worker_count = min(image_count, requested_jobs)
    worker_count = max(1, worker_count)

    atoms = max(0, int(n_atoms))
    basis_size = max(0, int(n_basis))
    spin_channels = 2 if open_shell else 1
    fd_evaluations = max(1, int(finite_difference_evaluations))

    by_cat: dict[str, int] = {
        "NEB parallel image workers": worker_count * per_image_bytes,
        "NEB band coordinates/forces": (
            (image_count + 2) * max(1, atoms) * 3 * 8 * 6
        ),
    }

    if warm_start and basis_size > 0:
        by_cat["NEB warm-start density cache"] = (
            (image_count + 2) * spin_channels * basis_size * basis_size * 8
        )
    elif warm_start:
        by_cat["NEB warm-start density cache"] = 0

    if fd_evaluations > 1:
        matrix_bytes = spin_channels * basis_size * basis_size * 8
        geometry_bytes = max(1, atoms) * 3 * 8
        by_cat["NEB finite-difference gradient scratch"] = (
            worker_count * fd_evaluations * (matrix_bytes + geometry_bytes)
        )

    return MemoryEstimate(by_category=by_cat, headroom_factor=headroom)


def _semiempirical_atoms(system) -> list:
    atoms = getattr(system, "atoms", None)
    if atoms is None:
        atoms = getattr(system, "unit_cell", None)
    return list(atoms or [])


def _semiempirical_orbital_count(z: int, method_family: str) -> int:
    if z <= 2:
        return 1
    if method_family == "msindo" and z > 10:
        return 9
    return 4


def _semiempirical_route_plan(
    system,
    method: str | SemiempiricalRoutePlan,
    *,
    nddo: bool,
    solvent: str | None,
    ccm_options,
) -> SemiempiricalRoutePlan:
    from vibeqc.semiempirical.routes import (
        BOUNDARY_MOLECULE,
        BOUNDARY_PERIODIC_GAMMA,
        BOUNDARY_SECCM_DIRECT_TORUS,
        SemiempiricalRoutePlan,
        plan_periodic_semiempirical_route,
    )

    is_periodic = hasattr(system, "unit_cell") and hasattr(system, "lattice")
    if is_periodic and not nddo and solvent is None and ccm_options is None:
        return plan_periodic_semiempirical_route(method, system)
    if isinstance(method, SemiempiricalRoutePlan):
        plan = method
        expected_boundaries = (
            {BOUNDARY_PERIODIC_GAMMA}
            if is_periodic
            else {BOUNDARY_MOLECULE, BOUNDARY_SECCM_DIRECT_TORUS}
        )
        if plan.boundary not in expected_boundaries:
            expected = ", ".join(sorted(expected_boundaries))
            raise ValueError(
                "semiempirical memory route/system boundary mismatch: "
                f"plan has boundary={plan.boundary!r}, expected {expected}."
            )
        if nddo and plan.variant != "nddo":
            raise ValueError("nddo=True does not match the supplied route plan.")
        if solvent is not None and plan.variant != "cosmo":
            raise ValueError("solvent does not match the supplied route plan.")
        if ccm_options is not None and plan.boundary != BOUNDARY_SECCM_DIRECT_TORUS:
            raise ValueError("ccm_options does not match the supplied route plan.")
        return plan

    boundary = BOUNDARY_PERIODIC_GAMMA if is_periodic else None
    return SemiempiricalRoutePlan.from_request(
        method,
        boundary=boundary,
        charge=int(getattr(system, "charge", 0) or 0),
        multiplicity=int(getattr(system, "multiplicity", 1) or 1),
        nddo=nddo,
        solvent=solvent,
        ccm_options=ccm_options,
    )


def estimate_semiempirical_memory(
    system,
    *,
    method: str | SemiempiricalRoutePlan,
    nddo: bool = False,
    solvent: str | None = None,
    optimize: bool = False,
    max_steps: int = 0,
    ccm_options=None,
) -> MemoryEstimate:
    """Peak-memory estimate for basis-free semiempirical routes.

    The estimate validates or consumes a :class:`SemiempiricalRoutePlan`, then
    counts valence-sized dense matrices and route-specific scratch without
    constructing a Gaussian :class:`BasisSet`. It is intentionally
    conservative and cheap enough for dry-run placement.
    """
    from vibeqc.semiempirical.routes import (
        BOUNDARY_PERIODIC_GAMMA,
        BOUNDARY_SECCM_DIRECT_TORUS,
    )

    plan = _semiempirical_route_plan(
        system,
        method,
        nddo=nddo,
        solvent=solvent,
        ccm_options=ccm_options,
    )
    is_periodic = plan.boundary == BOUNDARY_PERIODIC_GAMMA
    periodic_dim = max(0, int(getattr(system, "dim", 0) or 0)) if is_periodic else 0
    atoms = _semiempirical_atoms(system)
    n_atoms = max(1, len(atoms))
    z_values = [
        int(getattr(atom, "Z", getattr(atom, "number", 0)))
        for atom in atoms
    ]
    n_orb = max(
        1,
        sum(
            _semiempirical_orbital_count(z, plan.method_family)
            for z in z_values
        ),
    )
    spin_channels = 2 if plan.spin == "unrestricted" else 1
    matrix_bytes = n_orb * n_orb * 8
    vector_bytes = n_orb * 8
    coord_bytes = n_atoms * 3 * 8

    by_cat: dict[str, int] = {
        "Semiempirical valence matrices": (8 + 3 * spin_channels) * matrix_bytes,
        "Semiempirical SCF eigensolver scratch": (
            6 * matrix_bytes + 4 * vector_bytes
        ),
        "Semiempirical DIIS history": 8 * spin_channels * 2 * matrix_bytes,
        "Semiempirical atom-pair scratch": max(1, n_atoms * n_atoms) * 128,
    }

    if plan.scc in {"atomic", "shell"}:
        by_cat["Semiempirical charge/SCC state"] = (
            12 * n_atoms * 8 + 4 * matrix_bytes
        )
    if plan.method_family == "gfn2":
        by_cat["GFN2 shell/AES multipole state"] = (
            24 * n_atoms * 8 + 6 * matrix_bytes
        )

    if is_periodic:
        image_count = max(1, 3 ** max(1, periodic_dim))
        by_cat["Periodic semiempirical image/neighbour buffers"] = (
            image_count * (n_atoms * (3 * 8 + 16) + (2 + spin_channels) * matrix_bytes)
        )

    fd_route = plan.variant in {"pm6", "upm6", "om1", "om2", "om3"} or optimize
    if fd_route:
        by_cat["Semiempirical finite-difference gradient scratch"] = (
            2 * (coord_bytes + (4 + spin_channels) * matrix_bytes)
            + n_atoms * 3 * 8
        )
    if is_periodic and optimize:
        strain_components = max(1, periodic_dim * periodic_dim)
        by_cat["Periodic semiempirical stress/strain scratch"] = (
            2 * strain_components * (coord_bytes + (4 + spin_channels) * matrix_bytes)
            + 3 * 3 * 8
        )

    if optimize:
        steps = max(1, int(max_steps) if max_steps is not None else 1)
        by_cat["Semiempirical optimizer coordinates/history"] = (
            min(steps, 50) * n_atoms * 3 * 8 * 4
        )

    if plan.variant == "cosmo":
        n_segments = max(24, 60 * n_atoms)
        by_cat["MSINDO COSMO cavity A matrix"] = n_segments * n_segments * 8
        by_cat["MSINDO COSMO B matrix"] = n_segments * n_orb * n_orb * 8
        by_cat["MSINDO COSMO surface vectors"] = 8 * n_segments * 8

    if plan.boundary == BOUNDARY_SECCM_DIRECT_TORUS:
        translations = (
            getattr(ccm_options, "translations", None) if ccm_options else None
        )
        dim = len(translations or [])
        image_count = max(1, 3 ** max(1, dim))
        by_cat["MSINDO SECCM image-cell buffers"] = (
            image_count * (n_atoms * (3 * 8 + 16) + 2 * matrix_bytes)
        )
        if bool(getattr(ccm_options, "madelung", False)):
            by_cat["MSINDO SECCM Madelung workspace"] = image_count * matrix_bytes

    if plan.variant == "nddo":
        by_cat["MSINDO NDDO parameter scratch"] = 2 * matrix_bytes

    return _memory_estimate(by_cat)


def estimate_periodic_multik_gdf(
    *,
    n_basis: int,
    n_aux: int,
    n_kpoints: int,
    need_k_pairs: bool,
    open_shell: bool = False,
    n_ibz_kpoints: Optional[int] = None,
    diis_subspace_size: int = 8,
) -> MemoryEstimate:
    """Peak memory for a dense multi-k periodic GDF SCF.

    The dominant allocation is the in-core density-fitting factor cache:
    one ``complex128`` ``(n_aux, n_basis, n_basis)`` ``Lpq`` block per stored
    ``(k_i, k_j)`` pair. HF / hybrid exchange needs every ``n_kpoints**2``
    pair; pure DFT (and J-only) keeps just the ``n_kpoints`` diagonal pairs.
    The cache is held dense in RAM -- streaming is not yet implemented (see
    the :mod:`vibeqc.periodic_k_gdf` module docstring) -- so for a
    paper-grade cell this term is what OOM-kills the run before SCF iter 1
    (the NiO/def2-SVP KUKS ``(4,4,4)`` exit-137 case).

    ``n_ibz_kpoints`` is the size of the irreducible wedge when the run
    requests the space-group-reduced exchange build (``ibz_native=True`` on
    the drivers, ``symmetry_reduce_k=True`` on
    :func:`vibeqc.run_periodic_job`). The saving is on the **bra** index
    only -- ``K(k_i)`` is built for the wedge representatives and
    symmetry-transported to the rest of the star, while the ket sum still
    runs over the whole Brillouin zone -- so the cache holds
    ``n_ibz_kpoints * n_kpoints + n_kpoints - n_ibz_kpoints`` blocks
    rather than ``n_kpoints**2``. Hartree also needs diagonal factors
    outside the exchange wedge, as in the production factor builder.
    Without exchange-wedge accounting the
    preflight would charge a reduced run at the full-mesh number and abort
    calculations that fit. Measured 2026-08-14 on NaCl primitive /
    def2-SVP / def2-svp-jk (nao 33, naux 240) at ``(4,4,4)``: the Lpq term
    is 15.95 GiB unreduced against 2.21 GiB at ``n_IBZ = 8``.

    The wedge never reduces the diagonal-only cache, so ``n_ibz_kpoints``
    is ignored when ``need_k_pairs`` is False -- matching the driver, which
    passes ``bra_rows`` only on the exchange path.

    Parameters are plain integers so the estimate needs no built core and is
    unit-testable in isolation. ``complex128`` is 16 bytes/element.
    Accelerator history uses the EDIIS/DIIS hybrid upper bound: both F/error
    and F/density histories stay live, as does EDIIS's previous return.
    Pass zero depth when acceleration is disabled. This estimate does not
    include a fitting source's workspace or an XC grid/cache.
    """
    n = int(n_basis)
    n_k = int(n_kpoints)
    depth = int(diis_subspace_size)
    if depth != diis_subspace_size or depth < 0:
        raise ValueError("estimate_periodic_multik_gdf: diis_subspace_size must be a nonnegative integer")
    if need_k_pairs:
        n_bra = n_k if n_ibz_kpoints is None else int(n_ibz_kpoints)
        if n_bra < 1 or n_bra > n_k:
            raise ValueError(
                "estimate_periodic_multik_gdf: n_ibz_kpoints must be in "
                f"[1, n_kpoints]; got {n_ibz_kpoints!r} for n_kpoints={n_k}"
            )
        n_pairs = n_bra * n_k + n_k - n_bra
    else:
        n_pairs = n_k
    by_cat: dict[str, int] = {}
    by_cat["GDF Lpq factor cache"] = n_pairs * int(n_aux) * n * n * 16
    # Per-k complex one-electron + SCF-loop buffers (S, Hcore, X, MO, and per
    # spin channel a Fock + density). Sub-dominant to the Lpq cache but real.
    spin = 2 if open_shell else 1
    per_k_matrices = 4 + 2 * spin
    by_cat["per-k complex buffers"] = per_k_matrices * int(n_kpoints) * n * n * 16
    # MultiKPeriodicSCFAccelerator keeps both hybrid histories warm. Each
    # history owns two full-mesh arrays per iteration. EDIIS also retains
    # its previous extrapolate, and a push copies the next two arrays before
    # evicting the oldest. A wedge reduces none of this full-mesh storage.
    by_cat["SCF accelerator history peak"] = (
        (4 * depth + 3) * spin * n_k * n * n * 16 if depth else 0
    )
    # The k-point exchange team shares this budget; it is not multiplied by
    # the number of workers. Pure Hartree uses smaller auxiliary panels.
    by_cat["GDF contraction workspace"] = (32 if need_k_pairs else 1) * 1024**2
    return _memory_estimate(by_cat)


def estimate_periodic_gdf_source(
    *, n_basis: int, n_aux: int, n_kpoints: int, n_reciprocal_upper: int,
    native_workspace_bytes: int = 64 * 1024**2,
) -> MemoryEstimate:
    """SR/LR source peak for one complete q group, excluding its cache.

    The runtime can subdivide the group to fit a smaller admitted budget.
    This estimate keeps the complete group as a conservative upper bound.
    Only reciprocal vectors scale with G; AO products remain native panels.
    Query the linked complex LAPACK workspace just as the source does.
    """
    import numpy as np
    from scipy.linalg.lapack import get_lapack_funcs

    values = (n_basis, n_aux, n_kpoints, n_reciprocal_upper, native_workspace_bytes)
    if any(int(v) != v or v <= 0 for v in values):
        raise ValueError('periodic GDF source dimensions and workspace must be positive integers')
    n, a, k, g, native = map(int, values)
    heevd, query = get_lapack_funcs(('heevd', 'heevd_lwork'), dtype=np.complex128)
    work, integer_work, real_work, info = query(a, compute_v=1, lower=1)
    if info:
        raise RuntimeError(f'periodic GDF LAPACK workspace query failed ({info})')
    eigen_work = (16*int(np.ceil(work.real)) + heevd.int_dtype.itemsize*int(integer_work)
                  + 8*int(real_work))
    metric, tensor = 16*a*a, 16*k*a*n*n
    source = 4096 + 128*(a+k) + max(
        4096 + 128*512,
        tensor + 2*metric + native,
        tensor + 2*metric + eigen_work,
        2*tensor + 3*metric,
    )
    return _memory_estimate({
        'GDF SR/LR source and whitening workspace': source,
        'GDF reciprocal vectors and binding copy': 2*24*g,
    })


def estimate_periodic_xc_value(
    *, n_basis: int, n_atoms: int, n_grid_points: int, n_cells: int,
    functional_kind: str, open_shell: bool = False,
    n_threads: int | None = None,
    n_active_cells: int | None = None, n_bra_cells: int | None = None,
) -> MemoryEstimate:
    """Semilocal periodic XC peak, including retained grid and lattice blocks.

    Charge all density cells as potentially active. The native kernel may
    screen more tightly, but never allocates larger numerical batches than
    this bound. Python inverse-Bloch construction can hold both its block
    list and the native copy; output potentials coexist with that density.
    """
    n, cells, points = int(n_basis), int(n_cells), int(n_grid_points)
    if min(n, cells, points, int(n_atoms)) < 0:
        raise ValueError("Periodic XC dimensions must be nonnegative")
    if not n or not cells or not points:
        return MemoryEstimate(by_category={})
    active = cells if n_active_cells is None else int(n_active_cells)
    bras = active if n_bra_cells is None else int(n_bra_cells)
    if not 1 <= bras <= active <= cells:
        raise ValueError("Periodic XC image counts must satisfy 1 <= bra <= active <= density")
    if n_threads is None:
        from ._vibeqc_core import get_num_threads
        n_threads = get_num_threads()
    threads = max(1, int(n_threads))
    need_grad = str(functional_kind).upper().rsplit('.', 1)[-1] != 'LDA'
    per_point = 8 * ((4 if need_grad else 1) * active * n
                     + threads * (6 * n + 20) + 64)
    fixed = 8 * threads * n * n * (6 if open_shell else 4)
    workspace = 256 * 1024**2
    if fixed + per_point > workspace:
        raise MemoryError("Periodic XC value cannot fit one grid point in its workspace")
    batch = min(points, 4096, (workspace - fixed) // per_point)
    spin = 2 if open_shell else 1
    return _memory_estimate({
        'Periodic-XC value numerical workspace': fixed + batch * per_point,
        'Periodic-XC value input grid': 44 * points + 32 * int(n_atoms),
        'Periodic-XC value density and potential blocks': 3 * spin * cells * n * n * 8,
        # Pair records (12 bytes) and growing per-output-cell index lists
        # (up to 8 bytes per pair), plus cell maps and vector/matrix headers.
        'Periodic-XC value cell-pair metadata': 20 * active * bras + 512 * spin * cells,
    })._with_dims(n_basis=n, n_cells=cells, n_grid_points=points,
                 n_threads=threads, batch_points=batch)


def estimate_periodic_xc_gradient(
    *,
    n_basis: int,
    n_atoms: int,
    n_grid_points: int,
    n_cells: int,
    functional_kind: str,
    open_shell: bool = False,
    n_threads: int | None = None,
    n_active_cells: int | None = None,
    n_bra_cells: int | None = None,
) -> MemoryEstimate:
    """Peak periodic XC force memory, including bounded native grid batches.

    The input quadrature and accepted density stay live across all batches.
    AO tables and pair scratch are sized by the native workspace planner;
    atom accumulators and image-pair metadata are separate retained costs.
    """
    n = max(0, int(n_basis))
    n_atoms = max(0, int(n_atoms))
    n_pts = max(0, int(n_grid_points))
    n_cells = max(1, int(n_cells))
    if n == 0 or n_pts == 0:
        return MemoryEstimate(by_category={})
    active = n_cells if n_active_cells is None else int(n_active_cells)
    bras = active if n_bra_cells is None else int(n_bra_cells)
    if not 1 <= bras <= active <= n_cells:
        raise ValueError("Periodic XC image counts must satisfy 1 <= bra <= active <= density")

    kind = str(functional_kind or "").strip().upper()
    is_mgga = "MGGA" in kind or "META" in kind
    # Unknown semilocal kinds are charged as GGA rather than undercounted.
    need_hess = is_mgga or kind not in ("", "LDA")

    from ._vibeqc_core import periodic_xc_gradient_batch_size

    threads = _periodic_xc_gradient_thread_count(n_threads, n_cells)
    # The native allocator reserves omp_max_threads, even if there are fewer
    # active cell pairs. Preserve that bound for shifted-grid temporaries too.
    if n_threads is None:
        from ._vibeqc_core import get_num_threads
        threads = max(1, int(get_num_threads()))
    else:
        threads = max(1, int(n_threads))
    batch_points = min(n_pts, periodic_xc_gradient_batch_size(
        n, active, need_hess, bool(open_shell), threads,
    ))
    matrix_bytes = batch_points * n * 8
    vector_bytes = batch_points * 8
    matrix_1e_bytes = n * n * 8

    by_cat: dict[str, int] = {}

    # Home-cell reference tables plus one entry per density cell in chi_h /
    # dchi_h / hess_h. The home-cell entry is copied into the cell vector, so
    # peak memory includes both the reference object and the vector slot.
    ao_table_count = 10 if need_hess else 4
    by_cat["Periodic-XC gradient AO/gradient/Hessian tables"] = (
        (active + 1) * ao_table_count * matrix_bytes
    )

    # Density blocks are already present when the gradient is called, but they
    # contribute to the same peak. UKS carries alpha and beta cell matrices.
    spin_channels = 2 if open_shell else 1
    by_cat["Periodic-XC gradient density blocks"] = (
        spin_channels * n_cells * matrix_1e_bytes
    )

    if open_shell:
        scratch_matrices = 18 if need_hess else 6
    else:
        scratch_matrices = 10 if need_hess else 4
    by_cat["Periodic-XC gradient pair scratch"] = (
        threads * scratch_matrices * matrix_bytes
    )

    if open_shell:
        if is_mgga:
            grid_vectors = 31
        elif need_hess:
            grid_vectors = 25
        else:
            grid_vectors = 7
    else:
        if is_mgga:
            grid_vectors = 14
        elif need_hess:
            grid_vectors = 11
        else:
            grid_vectors = 4
    by_cat["Periodic-XC gradient grid vectors"] = (
        (max(grid_vectors, 40 if open_shell else 24) + 4 + 3 * threads)
        * vector_bytes
    )
    by_cat["Periodic-XC gradient input grid"] = n_pts * 44 + n_atoms * 32
    by_cat["Periodic-XC gradient cell-pair metadata"] = active * bras * 16 + n_cells * 384

    by_cat["Periodic-XC gradient thread buffers"] = (
        (threads + 2) * max(1, n_atoms) * 3 * 8 + n * (32 * threads + 8)
    )

    return _memory_estimate(by_cat)


_LEBEDEV_ORDER_POINTS = {
    3: 6,
    5: 14,
    7: 26,
    9: 38,
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    19: 146,
    21: 170,
    23: 194,
    25: 230,
    27: 266,
    29: 302,
    31: 350,
    35: 434,
    41: 590,
    47: 770,
    53: 974,
}

# Exact order/point pairs in the molecular grid's native dispatch table
# (``kLebedevTiers`` in cpp/src/lebedev_data.cpp). ``build_grid`` matches
# ``lebedev_order`` against that table exactly and throws on a miss, so this
# tuple has to list precisely those 13 orders: a narrower guard would make
# the estimator reject grids the native builder accepts. Orders 5 and 7 feed
# the sparse regions of ORCA-style grids, and 15/27/31 back the PySCF
# level-3 profile's 86/266/350-point tiers. Keep it ascending so the
# nearest-tier search below breaks ties toward the lower tier, matching
# ``lebedev_tier_for_points``. The remaining orders in
# ``_LEBEDEV_ORDER_POINTS`` are reachable only through the SciPy-backed
# periodic GAPW grids, which do not go through this table.
_MOLECULAR_LEBEDEV_ORDERS = (5, 7, 11, 15, 17, 23, 27, 29, 31, 35, 41, 47, 53)
_MOLECULAR_LEBEDEV_POINT_TIERS = tuple(
    _LEBEDEV_ORDER_POINTS[order] for order in _MOLECULAR_LEBEDEV_ORDERS
)


def estimate_periodic_gpw_gapw(
    *,
    n_basis: int,
    n_grid_points: int,
    route: str,
    functional_kind: str | None = None,
    open_shell: bool = False,
    n_kpoints: int = 1,
    compact_multik: bool = False,
    n_soft_basis: int | None = None,
    n_atoms: int = 0,
    augmentation_active: bool | None = None,
    analytic_eri_one_centre: bool = False,
    lmax: int = 3,
    n_radial: int = 80,
    lebedev_order: int = 17,
) -> MemoryEstimate:
    """Peak memory for the FFT-grid GPW/GAPW periodic routes.

    The estimate mirrors the large iteration-invariant caches in
    ``periodic_gapw_j`` / ``periodic_gapw_augment``: the full-basis AO
    collocation table, reciprocal mesh, density/Poisson/XC grid workspaces,
    bounded compact multi-k Bloch AO batches, and GAPW's soft-basis plus
    analytic augmentation caches. ``analytic_eri_one_centre`` additionally
    charges the retained hard and soft four-index ERI tensors used by the
    fit-free HF default. Inputs are plain integers so vq dry-runs can place
    jobs without building a plane-wave grid or entering the SCF loop.
    """
    route_key = str(route or "").strip().lower()
    if route_key not in ("gpw", "gapw"):
        raise ValueError("estimate_periodic_gpw_gapw: route must be 'gpw' or 'gapw'")

    n = max(0, int(n_basis))
    n_pts = max(0, int(n_grid_points))
    if n == 0 or n_pts == 0:
        return MemoryEstimate(by_category={})

    n_k = max(1, int(n_kpoints))
    n_soft = n if n_soft_basis is None else max(0, int(n_soft_basis))
    n_atoms = max(0, int(n_atoms))
    spin = 2 if open_shell else 1
    grid_real = n_pts * 8
    grid_complex = n_pts * 16
    matrix_real = n * n * 8
    matrix_complex = n * n * 16
    is_compact = bool(compact_multik and n_k > 1)
    # Both spin branches of the compact route stream their Bloch AO tables
    # (closed shell since d17aae954; open shell since issue #89), so the
    # bounded model applies to each.
    is_bounded_compact = bool(is_compact)

    by_cat: dict[str, int] = {}
    if not is_bounded_compact:
        by_cat["GPW full collocation cache"] = n_pts * n * 8
    by_cat["GPW reciprocal mesh cache"] = n_pts * 3 * 8

    # The smooth J path keeps a density grid, a Poisson-potential grid and FFT
    # work arrays. UKS/compact routes can hold both spin densities locally.
    by_cat["GPW density/Poisson grids"] = (
        (spin + 3) * grid_real + 2 * grid_complex
    )
    by_cat["GPW contraction/projection scratch"] = (
        spin * matrix_real + 2 * matrix_real + grid_real
    )

    kind = str(functional_kind or "").strip().upper()
    has_xc = bool(kind)
    is_mgga = "MGGA" in kind or "META" in kind
    needs_gradient = has_xc and (is_mgga or kind != "LDA")
    if has_xc:
        if open_shell:
            grid_vectors = 10 if not needs_gradient else 24
            if is_mgga:
                grid_vectors = 34
        else:
            grid_vectors = 6 if not needs_gradient else 15
            if is_mgga:
                grid_vectors = 22
        by_cat["GPW XC grid workspace"] = grid_vectors * grid_real
    if is_mgga:
        if is_bounded_compact:
            # One k-point's chi/u/FFT/gradient/contraction complex tables are
            # live at a time. Spectral derivatives require the complete grid,
            # but the peak no longer carries an n_k multiplier.
            by_cat["GPW compact multi-k meta-GGA AO/FFT scratch"] = (
                spin * 5 * n_pts * n * 16
            )
        else:
            by_cat["GPW meta-GGA AO-gradient scratch"] = (
                spin * (3 * n_pts * n * 8 + n_pts * n * 16)
            )

    if is_bounded_compact:
        one_bloch_table = n_pts * n * 16
        all_bloch_tables = n_k * one_bloch_table
        if all_bloch_tables <= _COMPACT_BLOCH_AO_CACHE_BYTES:
            bloch_storage = all_bloch_tables
            by_cat["GPW compact multi-k Bloch AO cache"] = bloch_storage
            # Dense density contraction can hold the cached chi table plus
            # one D contraction and its conjugated table for the active k.
            bloch_scratch = 2 * one_bloch_table
        else:
            batch_points = max(
                1,
                _COMPACT_BLOCH_AO_BATCH_BYTES // (n * 16),
            )
            bloch_storage = min(one_bloch_table, batch_points * n * 16)
            by_cat["GPW compact multi-k Bloch AO batch"] = bloch_storage
            bloch_scratch = 2 * bloch_storage
        by_cat["GPW compact multi-k projection scratch"] = (
            bloch_scratch
            + n_k * spin * matrix_complex
            + spin * grid_complex
        )

    if route_key == "gapw":
        active = True if augmentation_active is None else bool(augmentation_active)
        if active and analytic_eri_one_centre:
            by_cat["GAPW fit-free hard/soft ERI cache"] = 8 * (
                n**4 + n_soft**4
            )
        if active:
            by_cat["GAPW soft collocation cache"] = n_pts * n_soft * 8

            n_comp = max(1, int(lmax) + 1) ** 2
            n_rad = max(1, int(n_radial))
            n_ang = _LEBEDEV_ORDER_POINTS.get(int(lebedev_order), int(lebedev_order))
            n_ang = max(1, n_ang)
            n_atom_grid = n_rad * n_ang
            soft_matrix_real = n_soft * n_soft * 8

            by_cat["GAPW analytic augmentation cache"] = n_atoms * (
                n_atom_grid * (n + n_soft) * 8
                + n_comp * matrix_real
                + n_comp * n_pts * 8
                + n_comp * n_atom_grid * 8
                + 6 * n_atom_grid * 8
                + soft_matrix_real
            )

    return _memory_estimate(by_cat)


def _periodic_xc_gradient_thread_count(
    n_threads: int | None,
    n_cells: int,
) -> int:
    if n_threads is None:
        try:
            from ._vibeqc_core import get_num_threads

            n_threads = int(get_num_threads())
        except Exception:
            n_threads = os.cpu_count() or 1
    n_pairs_upper = max(1, int(n_cells) * int(n_cells))
    return max(1, min(int(n_threads), n_pairs_upper))


def check_periodic_gdf_memory(
    estimate: MemoryEstimate,
    *,
    n_kpoints: int,
    route_label: str,
    allow_exceed: bool = False,
    available: Optional[int] = None,
) -> None:
    """Fail-early gate for the dense multi-k GDF ``Lpq`` cache.

    Raises :class:`InsufficientMemoryError` -- with periodic-specific
    remedies rather than the molecular ``check_memory`` text -- when the
    estimate exceeds available RAM. A no-op when RAM cannot be probed
    (``available`` is 0, treated as "unknown") or ``allow_exceed`` is set,
    mirroring :func:`check_memory`'s fail-open-on-unknown contract.
    """
    avail = available if available is not None else available_memory_bytes()
    if avail <= 0 or estimate.total_bytes <= avail or allow_exceed:
        return

    msg = estimate.format(available=avail, status="ABORTING") + "\n\n"
    msg += (
        f"InsufficientMemoryError: the {route_label} GDF route holds its "
        f"density-fitting data dense in RAM for {int(n_kpoints)} k-point(s) "
        "(per-k-pair Lpq factors and/or the dense AO-pair FT bundle of the "
        "rsgdf fit; streaming / Schwarz-screened chunking is opt-in, see "
        "handovers/HANDOVER_GDF_FIT_SCREENING.md and the periodic_k_gdf "
        "module docstring). To fit the calculation: use a coarser "
        "Monkhorst-Pack mesh or the Gamma-only GDF path; on an "
        "exact-exchange run in a symmorphic space group, request the "
        "space-group-reduced exchange build "
        "(run_periodic_job(symmetry='attach', symmetry_reduce_k=True), "
        "which stores n_IBZ x n_k Lpq blocks instead of n_k^2); "
        "choose a smaller "
        "orbital / auxiliary basis; lower rsgdf_ke_cutoff (an accuracy "
        "knob -- re-check parity); or switch to the plane-wave GPW route "
        "(jk_method='gpw'), which does not materialise per-k-pair DF factors. "
        "Set VIBEQC_GDF_MEMORY_OVERRIDE=1 to attempt the run anyway."
    )
    raise InsufficientMemoryError(msg)


def format_memory_report(
    estimate: MemoryEstimate,
    *,
    override_requested: bool = False,
    available: Optional[int] = None,
) -> str:
    """One-stop formatter for the run_job output block.

    Decides the status string based on whether the estimate fits and
    whether an override was requested.
    """
    avail = available if available is not None else available_memory_bytes()
    if avail <= 0 or estimate.total_bytes <= avail:
        status = "Proceeding"
    elif override_requested:
        status = "Proceeding (override)"
    else:
        # Caller should have already invoked check_memory and aborted;
        # reaching here with status=ABORTING is a "report what happened".
        status = "ABORTING"
    return estimate.format(available=avail, status=status)



# ----------------------------------------------------------------------
# Batch runner memory budgeting
# ----------------------------------------------------------------------

@dataclass
class BatchMemoryPlan:
    """Result of planning batch parallelism from per-child memory estimates.

    Batch runners (``run_batch.py`` and similar) that launch multiple
    vibe-qc child processes inside a single PBS job must account for the
    fact that Torque applies ``PBS -l mem=`` as a **per-process rlimit**,
    while the PBS allocation is **per-node** (shared by all children).

    ``plan_batch_memory`` computes a safe parallelism level by summing
    per-child estimates, adding overhead, and comparing against the
    available allocation. When the sum exceeds the allocation it falls
    back to sequential execution.

    .. versionadded:: 0.15.x
    """

    n_parallel: int
    """Safe number of children to run concurrently."""

    mode: Literal["parallel", "sequential"]
    """``"parallel"`` when multiple children fit; ``"sequential"`` when
    the total estimated memory exceeds the allocation."""

    total_estimated_mb: float
    """Sum of all per-child estimates (with per-child headroom) × batch
    overhead factor, in megabytes."""

    per_child_mb: list[float]
    """Each child's estimate (with its own headroom) in megabytes."""

    overhead_factor: float
    """Batch overhead factor applied on top of per-child headroom."""

    available_mb: int
    """Available memory allocation in megabytes (the PBS request)."""

    max_parallel_requested: Optional[int]
    """The ``max_parallel`` value the caller passed in, or ``None``."""


def plan_batch_memory(
    estimates: list[MemoryEstimate],
    *,
    available_mb: int | None = None,
    overhead_factor: float = 1.2,
    max_parallel: int | None = None,
) -> BatchMemoryPlan:
    """Determine safe parallelism for a batch of vibe-qc calculations.

    Parameters
    ----------
    estimates
        Per-child :class:`MemoryEstimate` results, one for each case in
        the batch. Each estimate already carries its own headroom factor.
    available_mb
        PBS memory allocation in megabytes. When ``None``, we probe the
        live machine via :func:`available_memory_bytes`. **Always pass
        this explicitly when the scheduler applies PBS ``mem=`` as
        a per-process rlimit** (see BUG 118).
    overhead_factor
        Safety factor applied to the sum of per-child estimates. Default
        ``1.2`` (20% overhead for process memory, shared libraries, and
        kernel buffers).
    max_parallel
        Upper bound on concurrent children (e.g. CPU count). When
        ``None``, the only limit is memory.

    Returns
    -------
    BatchMemoryPlan
        A plan with ``n_parallel``, ``mode``, and the per-child breakdown.
        Use ``plan.n_parallel`` to cap your worker pool; check
        ``plan.mode`` to decide between parallel and sequential dispatch.

    Notes
    -----
    When Torque applies PBS ``mem=`` as a per-process rlimit,
    each child sees the **full** PBS allocation as its limit, not
    ``PBS_request / n_children``. For such configurations, a conservative strategy is to
    run sequentially until BUG 118's ``scheduler_mem_directive = "omit"``
    is deployed. This function accounts for that by comparing the sum of
    all children against the PBS allocation — if the sum exceeds it,
    ``mode`` will be ``"sequential"``.
    """
    if not estimates:
        return BatchMemoryPlan(
            n_parallel=0,
            mode="sequential",
            total_estimated_mb=0.0,
            per_child_mb=[],
            overhead_factor=overhead_factor,
            available_mb=available_mb or 0,
            max_parallel_requested=max_parallel,
        )

    # Each child's estimate already carries its own headroom factor.
    per_child_mb = [est.total_bytes / (1024**2) for est in estimates]
    total_mb = sum(per_child_mb) * overhead_factor

    if available_mb is None:
        avail_bytes = available_memory_bytes()
        avail_mb = max(avail_bytes, 0) // (1024**2)
    else:
        avail_mb = available_mb

    # Determine safe parallelism.
    if avail_mb <= 0:
        # Can't probe — caller's risk; allow all.
        n_fit = len(estimates)
    else:
        # How many children fit within the available allocation?
        # Each child consumes per_child_mb[i] * overhead_factor of the
        # node-level budget.
        cumulative = 0.0
        n_fit = 0
        child_overhead_mb = [m * overhead_factor for m in per_child_mb]
        for cm in sorted(child_overhead_mb):
            if cumulative + cm <= avail_mb:
                cumulative += cm
                n_fit += 1
            else:
                break

    # At least 1 (sequential fallback).
    n_fit = max(n_fit, 1)
    if max_parallel is not None:
        n_fit = min(n_fit, max_parallel)

    mode: Literal["parallel", "sequential"] = (
        "sequential" if n_fit <= 1 else "parallel"
    )

    return BatchMemoryPlan(
        n_parallel=n_fit,
        mode=mode,
        total_estimated_mb=total_mb,
        per_child_mb=per_child_mb,
        overhead_factor=overhead_factor,
        available_mb=avail_mb,
        max_parallel_requested=max_parallel,
    )


# ----------------------------------------------------------------------
# Per-driver estimators
# ----------------------------------------------------------------------


def _n_basis(molecule: Molecule, basis: BasisSet) -> int:
    return int(basis.nbasis)


def _n_shells(basis: BasisSet) -> int:
    value = getattr(basis, "nshells", None)
    if value is not None:
        return int(value)
    try:
        return int(len(basis.shells()))
    except Exception:
        return 0


def _diis_subspace_size(options) -> int:
    return int(getattr(options, "diis_subspace_size", 8)) if options is not None else 8


def _enum_name(value) -> str:
    """Return a stable upper-case name for pybind11 enums and strings."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    text = str(value)
    return text.rsplit(".", 1)[-1].upper()


def _uses_density_fit(options) -> bool:
    return (
        bool(getattr(options, "density_fit", False)) if options is not None else False
    )


def _uses_cosx(options) -> bool:
    return bool(getattr(options, "cosx", False)) if options is not None else False


def _uses_active_cosx(options) -> bool:
    """Whether the molecular driver actually constructs a COSX builder."""
    return _uses_density_fit(options) and _uses_cosx(options)


def _uses_direct_scf(n_basis: int, options) -> bool:
    """Mirror the SCF-mode dispatch used by the molecular drivers.

    ``options is None`` means the driver builds a default ``RHFOptions`` /
    ``UHFOptions``, whose ``scf_mode`` default is **AUTO** (threshold 140 bf,
    see cpp/include/vibeqc/rhf.hpp). So a large default-options SCF runs the
    integral-direct path -- it does NOT materialise the in-core 4-index ERI.
    Treat ``None`` / an unset ``scf_mode`` as that AUTO default rather than
    assuming the dense ERI tensor; otherwise the estimator over-counts a
    large default job by ``n_basis**4`` and spuriously aborts it (e.g. a
    DLPNO job's RHF reference on n-octane/cc-pVTZ: ~436 GB phantom ERI
    tensor, while the AUTO->direct SCF actually runs in a few GB)."""
    if _uses_density_fit(options):
        return False
    mode_name = (
        _enum_name(getattr(options, "scf_mode", None)) if options is not None
        else ""
    )
    if mode_name == "DIRECT":
        return True
    if mode_name == "AUTO" or not mode_name:
        # AUTO is the driver default; an absent options object inherits it.
        # BUG 87: threshold lowered from 200 to 140.
        threshold = (
            int(getattr(options, "scf_mode_auto_threshold", 140))
            if options is not None else 140
        )
        return n_basis > threshold
    return False  # explicit CONVENTIONAL: the in-core 4-index path


def _option_value(options, key: str, default=None):
    """Read an estimator option from a dict or an options object."""
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(key, default)
    return getattr(options, key, default)


def _scf_estimator_options(options):
    """Return the SCF-options object from an optional estimator bundle."""
    if isinstance(options, dict) and "scf_options" in options:
        return options.get("scf_options")
    return options


def _looks_like_scf_options(options) -> bool:
    if options is None or isinstance(options, dict):
        return False
    if type(options).__name__ in {
        "RHFOptions",
        "UHFOptions",
        "RKSOptions",
        "UKSOptions",
        "ROHFOptions",
        "ROKSOptions",
    }:
        return True
    return any(
        hasattr(options, attr)
        for attr in ("scf_mode", "scf_mode_auto_threshold", "cosx")
    )


def _post_reference_options(options):
    """Return SCF reference options from a post-SCF estimator bundle."""
    if isinstance(options, dict):
        return options.get("scf_options")
    if _looks_like_scf_options(options):
        return options
    return None


def _post_method_options(options, key: str):
    """Return method-specific post-SCF options from a bundle or bare object."""
    if isinstance(options, dict):
        return options.get(key)
    if _looks_like_scf_options(options):
        return None
    return options


def _posthf_uses_density_fit(method_options, *, default: bool = False) -> bool:
    if method_options is None:
        return bool(default)
    return bool(getattr(method_options, "density_fit", default))


def _tddft_estimator_options(options) -> Optional[dict]:
    """Return TDDFT post-SCF options when the estimate should include them."""
    if not isinstance(options, dict):
        return None
    if not bool(options.get("tddft", False)):
        return None
    return options


def _aux_basis_size(
    molecule: Molecule,
    n_basis: int,
    options,
    *,
    orbital_basis_name: str = "",
) -> int:
    """Best-effort auxiliary basis size for DF/COSX estimates.

    The estimator is allowed to be conservative, but it should still
    reflect the requested algorithm. If an aux basis name is available,
    use the same local resolver the drivers use. When the caller enabled
    density fitting but left ``aux_basis`` empty, mirror the driver's
    auto-resolution (``default_aux_basis_for(<orbital basis>, kind="jk")``
    in ``runner._run_single_point``) so the estimate and the execution
    agree on ``n_aux`` (BUG 117). Only when no concrete aux basis can be
    resolved fall back to the published dimension bound.
    """
    aux_name = str(getattr(options, "aux_basis", "") or "")
    if not aux_name and orbital_basis_name:
        try:
            from .density_fitting import default_aux_basis_for

            aux_name = default_aux_basis_for(orbital_basis_name, kind="jk")
        except Exception:
            # No registered default (e.g. pob-* bases): the driver will
            # reject the run; keep the conservative bound meanwhile.
            aux_name = ""
    if aux_name:
        try:
            from .aux_basis import make_aux_basis_set

            return int(make_aux_basis_set(molecule, aux_name=aux_name).nbasis)
        except Exception:
            pass
    # Eichkorn, Treutler, Öhm, Häser & Ahlrichs, Chem. Phys. Lett. 240,
    # 283 (1995), Eq. 26: N_J < 3 N_BF — the auxiliary set is "typically
    # twice as large as the CGTO basis" and "should not exceed 3 N_BF".
    # Charge the upper bound of that design rule when unresolvable.
    return max(1, 3 * n_basis)


def _mean_field_matrix_bytes(n_basis: int, *, open_shell: bool = False) -> int:
    factor = 12 if open_shell else 8
    return factor * n_basis * n_basis * 8


def _native_density_fitting_storage(
    n_aux: int,
    n_basis: int,
) -> tuple[int, int, int, int]:
    """Return AO-DF tensor sizes and the native object's phase bounds.

    The C++ constructor retains ``T_flat_``, ``B_per_P_``, and ``L_``.  At
    the end of its unpacking loop the raw ``T``, solved ``B_flat``, Coulomb
    metric ``V``, Eigen ``LLT`` storage, and member ``L_`` still share scope,
    giving a four-three-centre/three-metric construction peak.
    """
    three_index = n_aux * n_basis * n_basis * 8
    metric = n_aux * n_aux * 8
    construction = 4 * three_index + 3 * metric
    resident = 2 * three_index + metric
    return three_index, metric, construction, resident


def _df_tensor_layout(n_basis: int) -> tuple[int, float]:
    """Return resident tensor blocks and construction safety for SCF AO DF.

    Eichkorn et al., Chem. Phys. Lett. 240, 283-290 (1995), section 4.3,
    doi:10.1016/0009-2614(95)00621-A, stores the three-center integrals
    for every auxiliary function with the AO pair as a combined index. The
    native implementation uses the full square extent, ``n_aux * n_orb^2``.

    The C++ SCF ``DensityFitting`` constructor (cpp/src/df.cpp) holds
    FOUR such extents simultaneously at every basis size: the raw
    ``Eri3D T``, the packed ``T_flat_`` copy, the triangular-solve
    product ``B_flat``, and the per-P unpacked ``B_per_P_`` — ``T`` and
    ``B_flat`` only leave scope after ``B_per_P_`` is fully built.
    ``T_flat_`` + ``B_per_P_`` (two blocks) stay resident across the SCF.
    Two resident blocks times the 2x construction-overlap factor covers
    the four simultaneous extents.  At that same construction point the
    Coulomb metric, Eigen LLT storage, and copied lower-triangular member are
    three simultaneous ``n_aux^2`` matrices.

    BUG 117: this construction peak is size-independent, so there is no
    small-basis threshold. The previous ``n_basis > 500`` gate kept a
    "validated" 1x small-basis baseline that measurement disproved
    (BH9 wave 2026-08-06, ``/usr/bin/time -l`` on converged production
    SCFs: UKS n_bf 261 true peak 1.088 GB vs 0.516 GB estimated, 2.11x;
    RKS n_bf 1059 true peak 31.5 GB vs 12.02 GB estimated, 2.62x).
    """
    del n_basis  # the four-extent construction peak has no size gate
    return _DF_RESIDENT_BLOCKS, _DF_CONSTRUCTION_SAFETY_FACTOR


def _add_jk_storage(
    by_cat: dict[str, int],
    molecule: Molecule,
    basis: BasisSet,
    n_basis: int,
    options,
    *,
    open_shell: bool = False,
) -> Optional[tuple[int, int, float]]:
    """Add the dominant two-electron storage for the selected JK path."""
    basis_name = str(getattr(basis, "name", "") or "")
    if _uses_active_cosx(options):
        n_aux = _aux_basis_size(
            molecule, n_basis, options, orbital_basis_name=basis_name
        )
        n_grid = _grid_points(molecule, getattr(options, "cosx_grid", None))
        # COSXJKBuilder embeds the same C++ ``DensityFitting`` as the
        # plain-DF path (cpp/src/jk_builder.cpp), so its RI-J tensors
        # carry the identical four-extent construction peak (BUG 117).
        _, _, construction, _ = _native_density_fitting_storage(
            n_aux,
            n_basis,
        )
        by_cat["RI-J tensors"] = construction
        by_cat["COSX grid workspace"] = max(1, n_grid) * n_basis * 8
    elif _uses_density_fit(options):
        n_aux = _aux_basis_size(
            molecule, n_basis, options, orbital_basis_name=basis_name
        )
        n_blocks, safety_factor = _df_tensor_layout(n_basis)
        by_cat["DF three-index tensors"] = int(
            n_aux * n_basis * n_basis * n_blocks * 8 * safety_factor
        )
        by_cat["DF metric/workspace"] = 3 * n_aux * n_aux * 8
        return n_aux, n_blocks, safety_factor
    elif _uses_direct_scf(n_basis, options):
        n_shell = max(0, _n_shells(basis))
        shell_pairs = n_shell * (n_shell + 1) // 2
        by_cat["Direct-SCF shell-pair scratch"] = (
            shell_pairs * 1024
        )
    else:
        by_cat["ERI tensor"] = n_basis**4 * 8
    return None


def _annotate_df_tensor_estimate(
    estimate: MemoryEstimate,
    n_basis: int,
    layout: Optional[tuple[int, int, float]],
) -> None:
    if layout is None:
        return
    n_aux, n_blocks, safety_factor = layout
    estimate._with_dims(n_basis=n_basis, n_aux=n_aux)
    estimate.safety_factor = max(estimate.safety_factor, safety_factor)
    estimate.category_details["DF three-index tensors"] = (
        f"n_orb^2={n_basis}^2 x n_aux={n_aux} x n_blocks={n_blocks} "
        f"x 8 bytes x {safety_factor:g}x safety"
    )


def _molecular_lebedev_point_count(grid) -> int:
    """Return the densest active molecular Lebedev angular tier."""
    orca_points = tuple(
        int(value) for value in (getattr(grid, "orca_angular_points", ()) or ())
    )
    if len(orca_points) == 5:
        # The native builder selects the nearest bundled tier for each ORCA
        # region. Memory uses the densest region as a conservative upper bound.
        return max(
            min(
                _MOLECULAR_LEBEDEV_POINT_TIERS,
                key=lambda supported: abs(supported - requested),
            )
            for requested in orca_points
        )

    order = int(getattr(grid, "lebedev_order", 29))
    if order not in _MOLECULAR_LEBEDEV_ORDERS:
        available = ", ".join(str(value) for value in _MOLECULAR_LEBEDEV_ORDERS)
        raise ValueError(
            f"memory estimate: unsupported molecular Lebedev order {order} "
            f"(bundled tiers: {available})"
        )
    return _LEBEDEV_ORDER_POINTS[order]


def _grid_dimensions(options) -> tuple[int, int, int]:
    """Conservative factors for the molecular DFT grid-point count.

    Product grids return ``n_radial x n_theta x n_phi``. Lebedev grids return
    ``n_radial x n_angular x 1``, using the exact unpruned tier or the densest
    active pruning tier. Falls back to vibe-qc's default 75 x 17 x 36 product
    grid when ``options`` is absent.
    """
    if isinstance(options, dict) and "scf_options" in options:
        options = options.get("scf_options")
    n_radial, n_theta, n_phi = 75, 17, 36
    grid = options
    if options is not None and not hasattr(options, "n_radial"):
        grid = getattr(options, "grid", None)
    if grid is not None and hasattr(grid, "n_radial"):
        g = grid
        n_radial = int(getattr(g, "n_radial", n_radial))
        if _enum_name(getattr(g, "angular", None)) == "LEBEDEV":
            n_theta = _molecular_lebedev_point_count(g)
            n_phi = 1
        else:
            n_theta = int(getattr(g, "n_theta", n_theta))
            n_phi = int(getattr(g, "n_phi", n_phi))
    return max(0, n_radial), max(0, n_theta), max(0, n_phi)


def _grid_points(molecule: Molecule, options) -> int:
    grid = _grid_options_for_point_count(options)
    profile = _enum_name(getattr(grid, "atomic_grid_profile", None))
    if profile.replace("_", "").replace("-", "") == "PYSCFLEVEL3":
        return sum(_atomic_grid_point_counts(molecule, options))
    n_radial, n_theta, n_phi = _grid_dimensions(options)
    return n_radial * n_theta * n_phi * len(molecule.atoms)


def _grid_options_for_point_count(options):
    """Extract one native GridOptions object from estimator input."""
    options = _scf_estimator_options(options)
    if options is None:
        return None
    if hasattr(options, "n_radial"):
        return options
    return _option_value(options, "grid", None)


def _atomic_grid_point_counts(
    molecule: Molecule,
    options,
) -> tuple[int, ...]:
    """Return exact raw-grid point counts in atom order without a grid build."""
    from ._vibeqc_core import grid_atomic_point_counts

    grid = _grid_options_for_point_count(options)
    counts = (
        grid_atomic_point_counts(molecule)
        if grid is None
        else grid_atomic_point_counts(molecule, grid)
    )
    result = tuple(int(count) for count in counts)
    if len(result) != len(molecule.atoms) or any(
        count <= 0 for count in result
    ):
        raise RuntimeError(
            "native grid point-count plan did not return one positive block "
            "per atom"
        )
    return result


def _functional_kind_name(options, *, open_shell: bool = False) -> str:
    name = _option_value(options, "functional", None)
    if name is None:
        return "GGA"
    try:
        func = Functional(str(name), 2 if open_shell else 1)
    except Exception:
        return "GGA"
    return _enum_name(getattr(func, "kind", None)) or "GGA"


def _functional_is_external(options, *, open_shell: bool = False) -> bool:
    """Whether ``options`` selects a registered full-grid XC provider.

    Functional construction consults the native provider registry and does
    not load the provider's optional runtime.  In particular, detecting a
    registered SKALA alias here does not import PyTorch or fetch a model.
    """
    name = _option_value(options, "functional", None)
    if name is None:
        name = _option_value(
            _scf_estimator_options(options),
            "functional",
            None,
        )
    if name is None:
        return False
    try:
        func = Functional(str(name), 2 if open_shell else 1)
    except Exception:
        return False
    return bool(getattr(func, "is_external", False))


def _functional_is_skala(options, *, open_shell: bool = False) -> bool:
    """Whether ``options`` selects one of the registered SKALA aliases."""
    name = _option_value(options, "functional", None)
    if name is None:
        name = _option_value(
            _scf_estimator_options(options),
            "functional",
            None,
        )
    if str(name or "").strip().lower() not in _SKALA_FUNCTIONAL_NAMES:
        return False
    return _functional_is_external(options, open_shell=open_shell)


def _external_xc_bridge_peak_bytes(n_grid_points: int) -> int:
    """Full-grid native/Python provider bridge, excluding model internals."""
    n_grid_points = int(n_grid_points)
    if n_grid_points < 0:
        raise ValueError("n_grid_points must be non-negative")
    return n_grid_points * (
        _EXTERNAL_XC_FULL_GRID_FLOAT64_VALUES_PER_POINT * 8
        + _EXTERNAL_XC_FULL_GRID_INDEX_BYTES_PER_POINT
    )


def _skala_atom_aligned_chunk_points(
    atomic_grid_sizes: Sequence[int],
    *,
    target_points: int = _SKALA_DEFAULT_MODEL_CHUNK_TARGET_POINTS,
) -> int:
    """Return the largest complete-atom model chunk in grid points.

    This mirrors :func:`vibeqc.skala.plan_atom_grid_chunks` without importing
    the adapter (or PyTorch). Atom blocks are stably sorted by size, equal-size
    blocks are grouped into homogeneous calls up to ``target_points``, and no
    block is split. Homogeneous calls make SKALA's padded point count equal
    the real point count. A single oversized atom is therefore one oversized
    model call. Explicit block sizes keep the preflight exact for
    element-specific and radius-pruned grids.
    """
    if not isinstance(target_points, Integral) or isinstance(
        target_points, bool
    ):
        raise TypeError("target_points must be a positive integer")
    target_points = int(target_points)
    if target_points <= 0:
        raise ValueError("target_points must be positive")

    raw_sizes = tuple(atomic_grid_sizes)
    if not raw_sizes:
        return 0
    if not all(
        isinstance(size, Integral) and not isinstance(size, bool)
        for size in raw_sizes
    ):
        raise TypeError("atomic_grid_sizes must contain integers")
    sizes = tuple(int(size) for size in raw_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("atomic_grid_sizes must all be positive")

    peak_points = 0
    sorted_sizes = sorted(sizes)
    group_start = 0
    while group_start < len(sorted_sizes):
        atom_points = sorted_sizes[group_start]
        group_stop = group_start + 1
        while (
            group_stop < len(sorted_sizes)
            and sorted_sizes[group_stop] == atom_points
        ):
            group_stop += 1
        group_atoms = group_stop - group_start
        atoms_per_chunk = max(1, target_points // atom_points)
        peak_points = max(
            peak_points,
            atom_points * min(group_atoms, atoms_per_chunk),
        )
        group_start = group_stop
    return peak_points


def _skala_xc_peak_bytes(
    atomic_grid_sizes: Sequence[int],
    *,
    target_points: int = _SKALA_DEFAULT_MODEL_CHUNK_TARGET_POINTS,
) -> int:
    """External bridge plus SKALA first-derivative peak, in bytes.

    This parameter-level helper is shared-ready for molecular and periodic
    route planners.  It deliberately excludes the source grid's ordinary
    coordinates/final weights/owner storage, which each caller already
    charges as part of its quadrature, but includes SKALA's additional raw
    atomic weight field.
    """
    atomic_grid_sizes = tuple(atomic_grid_sizes)
    chunk_points = _skala_atom_aligned_chunk_points(
        atomic_grid_sizes,
        target_points=target_points,
    )
    n_grid_points = sum(int(size) for size in atomic_grid_sizes)
    if n_grid_points == 0:
        return 0
    full_grid_bytes = _external_xc_bridge_peak_bytes(n_grid_points)
    model_bytes = (
        chunk_points * _SKALA_MODEL_FIRST_DERIVATIVE_BYTES_PER_POINT
    )
    return full_grid_bytes + model_bytes


def estimate_skala_xc_memory(
    *,
    atomic_grid_sizes: Sequence[int],
) -> MemoryEstimate:
    """Peak memory of the full-grid SKALA callback/model phase.

    The parameter-only surface lets molecular and periodic route planners
    account for the same immutable model contract without importing PyTorch
    or loading the checkpoint.  The normal process-runtime floor is included
    when this estimate stands alone; callers merging it into an existing
    route estimate should add only the named workspace category.
    """

    peak = _skala_xc_peak_bytes(atomic_grid_sizes)
    if peak == 0:
        return MemoryEstimate(by_category={})
    return _memory_estimate({"SKALA full-grid + model workspace": peak})


def _functional_work_vectors(options, *, open_shell: bool = False) -> int:
    """Approximate libxc work-vector count per grid point."""
    kind = _functional_kind_name(options, open_shell=open_shell)
    spin = 2 if open_shell else 1
    if "MGGA" in kind or "META" in kind:
        return 28 if spin == 1 else 48
    if kind == "LDA":
        return 8 if spin == 1 else 14
    # Unknown semilocal functionals are charged as GGA rather than LDA.
    return 16 if spin == 1 else 30


def _functional_needs_vv10(options, *, open_shell: bool = False) -> bool:
    """Whether the selected functional carries a VV10 double-grid term."""
    name = _option_value(options, "functional", None)
    if name is None:
        return False
    try:
        func = Functional(str(name), 2 if open_shell else 1)
    except Exception:
        return False
    return bool(getattr(func, "needs_vv10", False))


def _functional_has_exact_exchange(options, *, open_shell: bool = False) -> bool:
    """Whether an RKS/UKS reference executes a DF exchange gradient."""
    scf_options = _scf_estimator_options(options)
    name = _option_value(scf_options, "functional", None)
    if name is None:
        name = _option_value(options, "functional", None)
    if name is None:
        return False
    try:
        func = Functional(str(name), 2 if open_shell else 1)
    except Exception:
        # The driver will reject an unknown functional. Until then, keep the
        # memory estimate fail-closed instead of assuming pure DFT.
        return True
    return bool(
        float(getattr(func, "hf_exchange_fraction", 0.0)) != 0.0
        or getattr(func, "is_range_separated", False)
    )


def _reference_uses_python_density_fitting(options) -> bool:
    """Whether the reference builds the Python ``DensityFitting`` object.

    ROHF and ROKS share the open-shell estimator shape with UHF/UKS, but their
    JK resolver owns the Python DF implementation.  Its constructor retains
    one more auxiliary-metric peer than the native object.
    """

    scf_options = _scf_estimator_options(options)
    return type(scf_options).__name__ in {"ROHFOptions", "ROKSOptions"}


def _uks_stability_response_supported(options) -> bool:
    """Whether default UKS stability owns a complete response model."""
    scf_options = _scf_estimator_options(options)
    for source in (scf_options, options):
        if _option_value(source, "dft_plus_u_sites", None):
            return False
        if _option_value(source, "dft_plus_u", None):
            return False

    name = _option_value(scf_options, "functional", None)
    if name is None:
        name = _option_value(options, "functional", None)
    if name is None:
        return True
    try:
        func = Functional(str(name), 2)
    except Exception:
        # The driver will reject an unknown functional. Until then, retain
        # the conservative stability allocation rather than underestimate.
        return True
    kind = _enum_name(getattr(func, "kind", None)) or "GGA"
    cosx_requested = any(
        bool(_option_value(source, "cosx", False))
        for source in (scf_options, options)
    )
    density_fit = any(
        bool(_option_value(source, "density_fit", False))
        for source in (scf_options, options)
    )
    hybrid_cosx = (
        density_fit
        and cosx_requested
        and float(func.hf_exchange_fraction) != 0.0
    )
    return not (
        "MGGA" in kind
        or "META" in kind
        or bool(getattr(func, "is_range_separated", False))
        or bool(getattr(func, "needs_vv10", False))
        or hybrid_cosx
    )


def _dft_xc_requires_dense_ao_tables(options) -> bool:
    """Whether molecular XC still owns whole-grid AO tables.

    Ordinary native RKS/UKS energy and Fock quadrature is pointwise and uses
    bounded grid slices.  Three callers still retain the complete AO table:
    the pure-Python ROKS route, TDDFT response setup, and the molecular
    Newton/TRAH XC-kernel builders.  SOSCF is intentionally absent because
    its diagonal Hessian approximation does not construct an XC kernel.
    """
    if bool(_option_value(options, "tddft", False)):
        return True

    scf_options = _scf_estimator_options(options)
    if scf_options is None:
        return False
    if type(scf_options).__name__ == "ROKSOptions":
        return True
    for threshold in ("newton_threshold", "trah_threshold"):
        try:
            if float(_option_value(scf_options, threshold, 0.0) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            # An invalid option will be rejected by the driver. Until then,
            # fail conservatively rather than underestimating its route.
            return True
    return False


def _dft_xc_matrix_table_count(kind: str, *, open_shell: bool) -> int:
    """Peak batch-local ``n_grid x n_basis`` tables in XC matrix builds.

    The meta-GGA counts include the three derivative-density contractions
    for RKS and both sets of three for UKS.  They replace the old synthetic
    ``tau`` AO table: tau itself is a scalar grid vector, while the real
    basis-sized intermediates are ``dchi * D``.
    """
    is_mgga = "MGGA" in kind or "META" in kind
    needs_gradient = kind != "LDA"
    if open_shell:
        if is_mgga:
            # chi + 3 dchi + chi*Da/Db + 3*(dchi*Da/Db)
            return 12
        if needs_gradient:
            # chi + 3 dchi + chi*Da/Db
            return 6
        # chi + chi*Da/Db
        return 3
    if is_mgga:
        # chi + 3 dchi + chi*D + 3*(dchi*D)
        return 8
    if needs_gradient:
        # chi + 3 dchi + chi*D (the flow table is a later peer peak)
        return 5
    # chi + chi*D
    return 2


def _dft_xc_gradient_table_count(kind: str, *, open_shell: bool) -> int:
    """Conservative AO-table peak of the batched analytic XC gradient."""
    if kind == "LDA":
        # chi + 3 dchi, plus one (RKS) or two (UKS) density contractions.
        return 6 if open_shell else 5
    # GGA/meta-GGA need chi + 3 dchi + 6 Hessians. Their directional
    # Hessian and density/flow contractions bring the observed peak to 13
    # tables for RKS and 15 for UKS. Meta-GGA tau vectors are charged in the
    # separate O(n_pts) libxc/vector term below.
    return 15 if open_shell else 13


def _dft_xc_dense_vector_count(kind: str, *, open_shell: bool) -> int:
    """Conservative full-grid scalar-vector peak of dense XC consumers."""
    if kind == "LDA":
        return 11 if open_shell else 4
    # The GGA second-order factories retain reference-density fields and an
    # f_xc cache in addition to apply-local perturbation vectors. These floors
    # also safely cover the dense meta-GGA ROKS/TDDFT consumers.
    return 57 if open_shell else 19


def _molecular_xc_batch_workers(n_points: int, *, kernel: bool = False) -> int:
    """Mirror the native bounded OpenMP batch planner exactly."""
    n_points = max(0, int(n_points))
    if n_points == 0:
        return 1
    batch = int(MOLECULAR_XC_GRID_BATCH_SIZE)
    batches = (n_points + batch - 1) // batch
    cap = (
        int(MOLECULAR_XC_KERNEL_MAX_WORKERS)
        if kernel
        else int(MOLECULAR_XC_GRID_MAX_WORKERS)
    )
    return max(1, min(batches, cap, _omp_max_threads()))


def _rhf_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    include_df_exchange_gradient: bool = True,
) -> MemoryEstimate:
    scf_options = _scf_estimator_options(options)
    n = _n_basis(molecule, basis)
    by_cat: dict[str, int] = {}

    df_layout = _add_jk_storage(by_cat, molecule, basis, n, scf_options)
    if df_layout is not None and include_df_exchange_gradient:
        n_aux = df_layout[0]
        n_occ = _clamped_occ(molecule.n_electrons() // 2, n)
        # The fused closed-shell DF J/K gradient retains B_M + eta while
        # resident/weighted DF tensors and metric workspaces are all live.
        # Its source peak is the constructor bound plus two Q x o x o arrays.
        by_cat["DF exchange-gradient occupied factors"] = (
            2 * n_aux * n_occ * n_occ * 8
        )

    # One-electron matrices (S, T, V, Hcore, F, D, X=S^{-1/2}, scratch).
    # ~8 * n^2 x 8 covers all core SCF-loop matrices.
    by_cat["Fock + density + 1e"] = _mean_field_matrix_bytes(n)

    # DIIS extrapolation holds (F, error) pairs across iterations.
    diis = _diis_subspace_size(scf_options)
    by_cat["DIIS history"] = diis * 2 * n * n * 8

    # MO coefficient + eigenvalue arrays + transform buffers.
    by_cat["MO workspace"] = 4 * n * n * 8 + n * 8

    est = _memory_estimate(by_cat)
    _annotate_df_tensor_estimate(est, n, df_layout)
    _add_tddft_response_estimate(est, molecule, basis, options)
    return est


def _uhf_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    include_df_exchange_gradient: bool = True,
) -> MemoryEstimate:
    scf_options = _scf_estimator_options(options)
    n = _n_basis(molecule, basis)
    by_cat: dict[str, int] = {}
    python_df_reference = _reference_uses_python_density_fitting(scf_options)
    df_layout = _add_jk_storage(
        by_cat,
        molecule,
        basis,
        n,
        scf_options,
        open_shell=True,
    )
    python_df_n_aux = None
    if python_df_reference:
        if df_layout is not None:
            python_df_n_aux = df_layout[0]
        elif _uses_active_cosx(scf_options):
            python_df_n_aux = _aux_basis_size(
                molecule,
                n,
                scf_options,
                orbital_basis_name=str(getattr(basis, "name", "") or ""),
            )
    if python_df_n_aux is not None:
        by_cat["Python DF constructor extra metric peer"] = (
            python_df_n_aux * python_df_n_aux * 8
        )
    if df_layout is not None and include_df_exchange_gradient:
        n_aux = df_layout[0]
        n_electrons = molecule.n_electrons()
        ms2 = int(molecule.multiplicity) - 1
        n_alpha = _clamped_occ((n_electrons + ms2) // 2, n)
        n_beta = _clamped_occ(n_electrons - n_alpha, n)
        three_index = n_aux * n * n * 8
        metric = n_aux * n_aux * 8
        constructor = 4 * three_index + (
            4 if python_df_reference else 3
        ) * metric
        largest_q = n_aux * max(n_alpha * n_alpha, n_beta * n_beta) * 8
        gradient_peak = 3 * three_index + 2 * metric + 2 * largest_q
        gradient_delta = max(0, gradient_peak - constructor)
        if gradient_delta:
            by_cat["DF exchange-gradient occupied-factor excess"] = (
                gradient_delta
            )
    by_cat["Fock + density + 1e"] = _mean_field_matrix_bytes(n, open_shell=True)
    base = _memory_estimate(by_cat)
    _annotate_df_tensor_estimate(base, n, df_layout)
    if python_df_n_aux is not None and df_layout is None:
        base._with_dims(n_basis=n, n_aux=python_df_n_aux)
    diis = _diis_subspace_size(scf_options)
    base.by_category["DIIS history"] = diis * 4 * n * n * 8
    base.by_category["MO workspace"] = 8 * n * n * 8 + 2 * n * 8
    # Open-shell adds D_alpha, D_beta, F_alpha, F_beta; rough 2x on
    # the density and Fock buffers.
    extra = 4 * n * n * 8
    base.by_category["Open-shell UHF buffers"] = extra
    _add_tddft_response_estimate(base, molecule, basis, options, open_shell=True)
    return base


def _dft_xc_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    open_shell: bool = False,
) -> int:
    """Extra memory beyond the HF baseline for XC integration.

    Ordinary native RKS/UKS evaluates AO values, derivatives, density
    contractions, and Fock projections in bounded grid slices.  The grid
    coordinates/weights remain O(n_pts), while conservative scalar libxc
    work vectors are batch-local. Analytic XC gradients use the same native
    batch bound; their GGA/meta-GGA AO Hessians are included as a separate
    peak rather than being silently omitted from a single-point-shaped
    estimate.

    ROKS, TDDFT, and the Newton/TRAH XC-kernel fallback still retain dense
    whole-grid AO tables and therefore keep the historical O(n_pts*n_basis)
    preflight scaling.
    """
    n = _n_basis(molecule, basis)
    skala_atomic_grid_sizes: tuple[int, ...] | None = None
    if _functional_is_skala(options, open_shell=open_shell):
        skala_atomic_grid_sizes = _atomic_grid_point_counts(molecule, options)
        n_pts = sum(skala_atomic_grid_sizes)
    else:
        n_pts = _grid_points(molecule, options)
    if n_pts == 0:
        return 0
    n_radial, n_theta, n_phi = _grid_dimensions(options)
    n_angular = max(1, n_theta * n_phi)
    kind = _functional_kind_name(options, open_shell=open_shell)
    batch_points = min(n_pts, int(MOLECULAR_XC_GRID_BATCH_SIZE))
    dense_tables = _dft_xc_requires_dense_ao_tables(options)
    batch_matrix_table_count = _dft_xc_matrix_table_count(
        kind,
        open_shell=open_shell,
    )
    matrix_points = n_pts if dense_tables else batch_points
    matrix_table_count = batch_matrix_table_count
    if dense_tables:
        # The second-order XC builders retain a copy of the dense AO tables
        # alongside the driver's source tables, their cached flow table(s),
        # and apply-local density/flow contractions. ROKS and TDDFT also use
        # dense Python-side consumers. Charge a conservative peer-table
        # floor rather than the old eight-table allowance, which missed the
        # simultaneous source + builder copies.
        matrix_table_count = max(16 if open_shell else 12, matrix_table_count)
    matrix_tables = matrix_table_count * matrix_points * n * 8

    gradient_tables = (
        _dft_xc_gradient_table_count(kind, open_shell=open_shell)
        * batch_points
        * n
        * 8
    )
    # The immutable grid remains proportional to the complete quadrature.
    # Grid::atom_of_point stores one 32-bit owner index per point. Each native
    # XC slice also copies its coordinates, weights, and owner indices while
    # its AO tables are live, so include that bounded peer allocation in the
    # active XC phase rather than silently losing it from the plateau.
    becke_weights = n_pts * 8
    point_owners = n_pts * 4
    angular_points = 4 * max(1, n_radial) * n_angular * 8
    coordinates = 3 * n_pts * 8
    batch_grid_copy = batch_points * (4 * 8 + 4)

    # Semilocal density/libxc vectors are batch-local with the AO tables.
    # Dense legacy consumers retain their full-grid vectors too.
    vector_points = n_pts if dense_tables else batch_points
    batch_vector_count = _functional_work_vectors(
        options,
        open_shell=open_shell,
    )
    vector_count = batch_vector_count
    if dense_tables:
        vector_count = max(
            vector_count,
            _dft_xc_dense_vector_count(kind, open_shell=open_shell),
        )
    libxc_work = (
        vector_points
        * vector_count
        * 8
    )

    # First-order SCF now evaluates independent grid batches concurrently.
    # Every worker owns its AO/libxc tables and one private projected result;
    # completed results are reduced immediately in grid order, so no more
    # than ``workers`` copies are live. Analytic gradients remain one-batch
    # and are a separate sequential phase.
    scf_workers = _molecular_xc_batch_workers(n_pts)
    projected_matrices = (
        (2 if kind == "LDA" else 4)
        if open_shell
        else (1 if kind == "LDA" else 2)
    )
    scf_worker_peak = (
        batch_matrix_table_count * batch_points * n * 8
        + batch_vector_count * batch_points * 8
        + projected_matrices * n * n * 8
        + batch_grid_copy
    )
    scf_parallel_peak = scf_workers * scf_worker_peak
    gradient_peak = (
        gradient_tables
        + batch_vector_count * batch_points * 8
        + batch_grid_copy
    )
    if dense_tables:
        # Newton/TRAH/ROKS/TDDFT may retain their whole-grid state while the
        # ordinary SCF Fock build owns its concurrent batch workspaces.
        dense_peak = matrix_tables + libxc_work + batch_grid_copy
        xc_peak = dense_peak + scf_parallel_peak
    else:
        xc_peak = max(scf_parallel_peak, gradient_peak)

    # A registered external provider is a distinct sequential phase and
    # retains the complete descriptor/adjoint bridge. SKALA additionally
    # evaluates its calibrated atom-aligned first-derivative model chunk;
    # another provider must not silently inherit that model-specific charge.
    # Compare these peer phases with the bounded AO projection rather than
    # summing workspaces which are not live together. Registry inspection does
    # not import Torch or load any model artifact.
    if _functional_is_external(options, open_shell=open_shell):
        external_peak = _external_xc_bridge_peak_bytes(n_pts)
        if skala_atomic_grid_sizes is not None:
            external_peak = _skala_xc_peak_bytes(
                skala_atomic_grid_sizes,
            )
        xc_peak = max(
            xc_peak,
            external_peak,
        )

    # VV10 is the deliberate non-pointwise exception. The native two-pass
    # path retains rho/sigma and the two output potentials over the whole
    # grid while compute_vv10 owns its active-point record (eleven doubles
    # plus an index, rounded to twelve double-sized slots). No n_basis-sized
    # table is retained across batches.
    if _functional_needs_vv10(options, open_shell=open_shell):
        vv10_kernel_peak = 16 * n_pts * 8
        if dense_tables:
            # Once a second-order builder exists, its retained dense AO/f_xc
            # cache overlaps the following iteration's VV10 kernel phase.
            xc_peak += vv10_kernel_peak
        else:
            # Kernel phase: rho + sigma + the result potentials + Pt records.
            # Projection phase: two retained potentials + batch AO/libxc work.
            vv10_projection_peak = 2 * n_pts * 8 + xc_peak
            xc_peak = max(vv10_kernel_peak, vv10_projection_peak)
    # Post-convergence UKS internal stability analysis (default-on for
    # LDA/GGA/global-hybrid GGA; skipped when the complete response is not
    # available). The batched f_xc
    # builder keeps whole-grid per-point state across Davidson matvecs
    # — GGA: reference grad-rho (6) + v_sigma (3) + the 15 polarised
    # fxc pieces; LDA: 3 v2rho2 pieces — plus its own copies of the
    # grid coordinates (3 doubles/point) and weights (1). Its apply scratch
    # is larger than the first-order SCF batch and has an independent cap.
    stability_vectors = 0
    if open_shell:
        stability = _option_value(options, "stability_check", None)
        if stability is None:
            stability = _option_value(
                _scf_estimator_options(options), "stability_check", None)
        if stability is None:
            # Driver default: UKSOptions.stability_check is true.
            stability = True
        if (
            not bool(stability)
            or not _uks_stability_response_supported(options)
        ):
            stability_vectors = 0
        elif kind == "LDA":
            stability_vectors = (3 + 4) * n_pts * 8
        else:
            stability_vectors = (24 + 4) * n_pts * 8
    if stability_vectors:
        stability_workers = _molecular_xc_batch_workers(n_pts, kernel=True)
        stability_worker_peak = (
            10 * batch_points * n * 8
            + 5 * n * n * 8
            + 60 * batch_points * 8
            + batch_grid_copy
        )
        xc_peak = max(
            xc_peak,
            stability_vectors + stability_workers * stability_worker_peak,
        )
    return (
        xc_peak
        + becke_weights
        + point_owners
        + angular_points
        + coordinates
    )


def _clamped_occ(value: int, n_basis: int) -> int:
    return max(0, min(int(n_basis), int(value)))


def _add_tddft_response_estimate(
    est: MemoryEstimate,
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    open_shell: bool = False,
) -> None:
    """Add dense molecular TDDFT/TDA/CIS response-memory categories.

    The current TDDFT implementation materializes the full AO ERI tensor,
    transforms dense MO response blocks, builds dense A (and, for Casida, B)
    matrices, then diagonalizes dense response matrices. This helper mirrors
    those allocation shapes so preflight and vq RAM placement see the post-SCF
    peak instead of only the underlying SCF reference.
    """
    td_opts = _tddft_estimator_options(options)
    if td_opts is None:
        return

    n = _n_basis(molecule, basis)
    if n <= 0:
        return
    tddft_type = str(td_opts.get("tddft_type", "tda") or "tda").strip().lower()
    is_casida = tddft_type == "casida"
    n_states = max(1, int(td_opts.get("tddft_n_states", 5) or 5))
    functional = td_opts.get("functional")

    if open_shell:
        n_elec = molecule.n_electrons()
        ms2 = int(molecule.multiplicity) - 1
        n_alpha = _clamped_occ((n_elec + ms2) // 2, n)
        n_beta = _clamped_occ(n_elec - n_alpha, n)
        n_virt_a = max(0, n - n_alpha)
        n_virt_b = max(0, n - n_beta)
        n_pair_a = n_alpha * n_virt_a
        n_pair_b = n_beta * n_virt_b
        n_pair = n_pair_a + n_pair_b
        if n_pair <= 0:
            return

        # UHF TDA builds same-spin ovov/oovv blocks and one cross-spin block;
        # UHF Casida reuses that path and transforms a second set for B.
        tda_blocks = 2 * n_pair_a**2 + 2 * n_pair_b**2 + n_pair_a * n_pair_b
        casida_extra_blocks = n_pair_a**2 + n_pair_b**2 + n_pair_a * n_pair_b
        block_elems = tda_blocks + (casida_extra_blocks if is_casida else 0)
        est.by_category["TDDFT ovov/oovv blocks"] = block_elems * 8

        pair2 = n_pair**2
    else:
        n_occ = _clamped_occ(molecule.n_electrons() // 2, n)
        n_virt = max(0, n - n_occ)
        n_pair = n_occ * n_virt
        if n_pair <= 0:
            return
        pair2 = n_pair**2
        est.by_category["TDDFT ovov/oovv blocks"] = 2 * pair2 * 8

    est.by_category["TDDFT AO ERI tensor"] = n**4 * 8
    response_matrix_count = 2 if is_casida else 1
    est.by_category["TDDFT A/B response matrices"] = (
        response_matrix_count * pair2 * 8
    )
    # Dense eigensolvers and Casida square-root products hold several
    # additional n_pair x n_pair arrays transiently.
    eig_workspace_count = 6 if is_casida else 2
    est.by_category["TDDFT eigensolver workspace"] = eig_workspace_count * pair2 * 8
    est.by_category["TDDFT state vectors"] = min(n_states, n_pair) * n_pair * 8
    if functional is not None:
        est.by_category["TDDFT f_xc kernel blocks"] = pair2 * 8


def _rks_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    est = _rhf_estimate(
        molecule,
        basis,
        options,
        include_df_exchange_gradient=_functional_has_exact_exchange(options),
    )
    extra = _dft_xc_estimate(molecule, basis, options)
    if extra:
        est.by_category["DFT grid + chi"] = extra
    return est


def _uks_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    est = _uhf_estimate(
        molecule,
        basis,
        options,
        include_df_exchange_gradient=_functional_has_exact_exchange(
            options,
            open_shell=True,
        ),
    )
    extra = _dft_xc_estimate(
        molecule,
        basis,
        options,
        open_shell=True,
    )
    if extra:
        est.by_category["DFT grid + chi"] = extra
    return est


def _frozen_core_count(
    options,
    max_frozen: int,
    *,
    molecule: Molecule | None = None,
    published_default: bool = False,
) -> int:
    """Return and validate the correlated-method frozen-core count.

    ``n_frozen_core`` is the public CC spelling. ``n_frozen`` is accepted as
    a compatibility alias for lightweight estimator option objects and older
    correlated routes. Frozen occupied orbitals do not become virtuals.  The
    bound is the execution route's largest valid count; silently clamping an
    invalid published or explicit request would make preflight describe a
    different calculation and defer the real diagnostic until after SCF.
    """
    value = getattr(options, "n_frozen_core", None) if options is not None else None
    if value is None and options is not None:
        value = getattr(options, "n_frozen", None)
    implicit = value is None
    if value is not None:
        try:
            implicit = int(value) == -1
        except (TypeError, ValueError):
            pass
    if (
        options is not None
        and hasattr(options, "_n_frozen_core_explicit")
        and not getattr(options, "_n_frozen_core_explicit")
    ):
        implicit = True
    if published_default and implicit:
        if molecule is None:
            raise ValueError("published frozen-core default requires a molecule")
        from .correlation_conventions import resolve_frozen_core_count

        value = resolve_frozen_core_count(molecule, None)
    elif implicit:
        value = 0
    try:
        frozen = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "frozen-core orbital count must be an integer or the published "
            "default sentinel"
        ) from exc
    if frozen < 0:
        raise ValueError("frozen-core orbital count must be non-negative")
    maximum = max(0, int(max_frozen))
    if frozen > maximum:
        raise ValueError(
            f"frozen-core orbital count {frozen} exceeds this route's "
            f"maximum valid count {maximum}"
        )
    return frozen


def _closed_shell_occ_vir(
    molecule: Molecule,
    n_basis: int,
    *,
    n_frozen_core: int = 0,
) -> tuple[int, int]:
    n_occ_total = _clamped_occ(molecule.n_electrons() // 2, n_basis)
    frozen = max(0, min(n_occ_total, int(n_frozen_core)))
    return max(1, n_occ_total - frozen), max(1, n_basis - n_occ_total)


def _open_shell_occ_vir(
    molecule: Molecule,
    n_basis: int,
    *,
    n_frozen_core: int = 0,
) -> tuple[int, int, int, int]:
    n_elec = molecule.n_electrons()
    mult = int(molecule.multiplicity)
    n_alpha = _clamped_occ((n_elec + mult - 1) // 2, n_basis)
    n_beta = _clamped_occ(n_elec - n_alpha, n_basis)
    frozen = max(0, min(n_alpha, n_beta, int(n_frozen_core)))
    return (
        max(1, n_alpha - frozen),
        max(1, n_beta - frozen),
        max(1, n_basis - n_alpha),
        max(1, n_basis - n_beta),
    )


def _posthf_memory_mode(method_options) -> str:
    mode = str(getattr(method_options, "memory_mode", "auto") or "auto")
    mode = mode.strip().lower().replace("_", "-")
    if mode in {"in-core", "inmemory", "in-memory"}:
        return "incore"
    if mode in {"semi-direct", "semidirect", "disk-backed"}:
        return "disk"
    if mode not in {"auto", "incore", "direct", "disk"}:
        # The numerical driver reports the invalid option. Until then the
        # estimator fails conservatively with the legacy in-core shape.
        return "incore"
    return mode


def _requested_workspace_bytes(method_options) -> int:
    try:
        return max(
            0,
            int(getattr(method_options, "requested_memory_bytes", 0) or 0),
        )
    except (TypeError, ValueError):
        return 0


def _exact_mp2_direct_bytes_per_i(
    n_basis: int,
    n_vir_bra: int,
    n_occ_ket: int,
    n_vir_ket: int,
) -> int:
    """Peak of the adjacent AO-to-MO transform stages for one bra ``i``.

    This mirrors ``exact_direct_bytes_per_i`` in the native RMP2/UMP2
    kernels. Each predecessor is released only after its successor has been
    formed, so the live peak is the largest of I1+I2, I2+I3, and I3+panel.
    """
    i1 = n_basis**3 * 8
    i2 = n_vir_bra * n_basis**2 * 8
    i3 = n_vir_bra * n_occ_ket * n_basis * 8
    panel = n_vir_bra * n_occ_ket * n_vir_ket * 8
    return max(i1 + i2, i2 + i3, i3 + panel)


def _occupied_i_slab_plan(
    n_occ_bra: int,
    bytes_per_i: int,
    resident_bytes: int,
    construction_bytes: int,
    budget_bytes: int,
) -> tuple[int, int, int]:
    """Mirror the native occupied-``i`` slab planner.

    Returns ``(occupied_count, slab_bytes, full_workspace_bytes)``. An
    impossible non-zero cap is represented by the irreducible one-``i``
    peak exceeding that cap; the runner's admission check then fails before
    the native kernel would issue its equivalent route-specific error.
    """
    target = 64 * 1024**2 if budget_bytes == 0 else max(
        0, budget_bytes - resident_bytes
    )
    count = max(
        1,
        min(
            max(1, n_occ_bra),
            target // max(1, bytes_per_i),
        ),
    )
    slab_bytes = count * bytes_per_i
    return count, slab_bytes, max(
        construction_bytes,
        resident_bytes + slab_bytes,
    )


def _category_sum(estimate: MemoryEstimate, names) -> int:
    return sum(int(estimate.by_category.get(name, 0)) for name in names)


def _reference_result_bytes(estimate: MemoryEstimate) -> int:
    """Arrays retained by an RHF/UHF result while post-HF work executes."""
    return _category_sum(
        estimate,
        (
            "Fock + density + 1e",
            "MO workspace",
            "Open-shell UHF buffers",
        ),
    )


def _python_density_fitting_storage(
    n_aux: int,
    n_basis: int,
) -> tuple[int, int, int, int]:
    """Return AO-DF tensor sizes and the Python object's phase bounds.

    :class:`vibeqc.density_fitting.DensityFitting` retains the raw and
    metric-orthogonalised three-centre tensors together with ``V``, scipy's
    Cholesky storage, and the explicit lower triangle.  Its constructor also
    creates integral/binding and linear-algebra peers.  Keep this source-level
    inventory in one place so every pure-Python correlated route prices the
    same object it actually constructs.
    """
    three_index = n_aux * n_basis * n_basis * 8
    metric = n_aux * n_aux * 8
    construction = 4 * three_index + 4 * metric
    resident = 2 * three_index + 3 * metric
    return three_index, metric, construction, resident


def _set_posthf_phase_peaks(
    estimate: MemoryEstimate,
    reference_categories: set[str],
    post_phases: dict[str, int],
) -> None:
    """Record mutually exclusive reference and correlated phase peaks.

    The historical category breakdown remains available for diagnostics and
    structured logs. The headline uses the maximum phase, so the synthetic
    Python baseline and already-freed SCF/build scratch are not added to every
    correlated allocation.
    """
    runtime = int(
        estimate.by_category.get("Python runtime + NumPy overhead", 0)
    )
    scf_peak = _category_sum(
        estimate,
        (
            name
            for name in reference_categories
            if name != "Python runtime + NumPy overhead"
        ),
    )
    phases: dict[str, int] = {}
    if runtime:
        phases["Runtime baseline"] = runtime
    if scf_peak:
        phases["SCF reference"] = scf_peak
    phases.update(
        (label, max(0, int(size)))
        for label, size in post_phases.items()
        if int(size) > 0
    )
    estimate.phase_peaks = phases


def _mp2_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    ref_options = _post_reference_options(options)
    mp2_options = _post_method_options(options, "mp2_options")
    est = _rhf_estimate(molecule, basis, ref_options)
    reference_categories = set(est.by_category)
    n = _n_basis(molecule, basis)
    n_occ_total = _clamped_occ(molecule.n_electrons() // 2, n)
    n_frozen = _frozen_core_count(
        mp2_options,
        max(0, n_occ_total - 1),
        molecule=molecule,
        published_default=True,
    )
    n_occ, n_vir = _closed_shell_occ_vir(
        molecule,
        n,
        n_frozen_core=n_frozen,
    )
    n_ov = n_occ * n_vir
    fixed_mo = n * n * 8
    ovov_bytes = n_ov * n_ov * 8
    uses_df = _posthf_uses_density_fit(mp2_options, default=False)
    n_aux = _aux_basis_size(molecule, n, mp2_options) if uses_df else 0
    budget = _requested_workspace_bytes(mp2_options)
    requested_mode = _posthf_memory_mode(mp2_options)

    exact_incore = fixed_mo + (
        n**4
        + n_occ * n**3
        + n_occ * n_vir * n * n
        + n_occ * n_vir * n_occ * n
        + n_occ * n_vir * n_occ * n_vir
    ) * 8
    exact_per_i = _exact_mp2_direct_bytes_per_i(n, n_vir, n_occ, n_vir)
    (
        three_index_bytes,
        metric_bytes,
        native_df_construction,
        native_df_resident,
    ) = _native_density_fitting_storage(n_aux, n)
    b_mo_bytes = n_aux * n_ov * 8
    df_construction = fixed_mo + native_df_construction
    df_resident = fixed_mo + native_df_resident + b_mo_bytes
    df_transform_scratch = (
        min(n_aux, _omp_max_threads())
        * n_occ
        * (n + n_vir)
        * 8
    )
    df_transform = df_resident + df_transform_scratch
    df_irreducible = max(df_construction, df_transform)
    df_contraction = df_resident + ovov_bytes
    if bool(getattr(mp2_options, "use_float_intermediates", False)):
        df_contraction += b_mo_bytes // 2 + ovov_bytes // 2
    incore_required = (
        max(df_irreducible, df_contraction)
        if uses_df
        else exact_incore
    )
    mode = requested_mode
    if mode == "auto":
        mode = "incore" if budget == 0 or incore_required <= budget else "direct"

    reference_result = _reference_result_bytes(est)
    phases: dict[str, int] = {}
    selected_workspace = incore_required if mode == "incore" else 0
    est.by_category["MP2 owned MO coefficient slices"] = fixed_mo
    if uses_df:
        # DensityFitting still has an eager construction phase: raw T,
        # flattened T, solved B, and unpacked B overlap with three metric
        # matrices. Direct MP2 bounds OVOV, not this three-index build.
        est.by_category["DF-MP2 three-index/B tensors"] = 2 * three_index_bytes
        est.by_category["DF-MP2 metric/workspace"] = metric_bytes
        est.by_category["DF-MP2 aux-OV workspace"] = b_mo_bytes
        est.by_category["DF-MP2 construction overlap"] = df_construction
        est.by_category["DF-MP2 threaded MO-transform scratch"] = (
            df_transform_scratch
        )
        phases["DF-MP2 integral build"] = reference_result + df_construction
        phases["DF-MP2 MO transformation"] = reference_result + df_transform
        if mode == "incore":
            est.by_category["OVOV MO tensor"] = ovov_bytes
            if bool(getattr(mp2_options, "use_float_intermediates", False)):
                est.by_category["DF-MP2 float GEMM scratch"] = (
                    b_mo_bytes // 2 + ovov_bytes // 2
                )
            phases["MP2 incore correlation"] = (
                reference_result + df_contraction
            )
        else:
            bytes_per_i = n_vir * n_occ * n_vir * 8
            block, slab_bytes, workspace = _occupied_i_slab_plan(
                n_occ,
                bytes_per_i,
                df_resident,
                df_irreducible,
                budget,
            )
            label = (
                "DF-MP2 disk-backed occupied-orbital slab"
                if mode == "disk"
                else "DF-MP2 direct occupied-orbital slab"
            )
            est.by_category[label] = slab_bytes
            est.category_details[label] = (
                f"mode={mode}, occupied-i block={block}/{n_occ}, "
                f"bytes-per-i={bytes_per_i}, "
                f"requested={budget} bytes"
            )
            est.dims["mp2_block_size"] = block
            selected_workspace = max(selected_workspace, workspace)
            phases[f"MP2 {mode} correlation"] = reference_result + workspace
        if bool(getattr(mp2_options, "report_ri_residual", False)):
            block, slab_bytes, workspace = _occupied_i_slab_plan(
                n_occ,
                exact_per_i,
                fixed_mo,
                0,
                budget,
            )
            label = "MP2 RI-residual direct occupied-orbital slab"
            est.by_category[label] = slab_bytes
            est.category_details[label] = (
                f"occupied-i block={block}/{n_occ}, "
                f"bytes-per-i={exact_per_i}, requested={budget} bytes"
            )
            phases["MP2 RI residual diagnostic"] = (
                reference_result + workspace
            )
            selected_workspace = max(selected_workspace, workspace)
    else:
        if mode == "incore":
            # The category name is shared with the conventional SCF phase
            # when present. The phase model charges one AO tensor during the
            # MP2 transform, not two simultaneous copies across SCF + MP2.
            if "ERI tensor" not in est.by_category:
                est.by_category["MP2 AO ERI tensor"] = n**4 * 8
            est.by_category["MP2 AO->MO I1 scratch"] = n_occ * n**3 * 8
            est.by_category["MP2 AO->MO I2 scratch"] = (
                n_occ * n_vir * n * n * 8
            )
            est.by_category["MP2 AO->MO I3 scratch"] = (
                n_occ * n_vir * n_occ * n * 8
            )
            est.by_category["OVOV MO tensor"] = ovov_bytes
            phases["MP2 incore correlation"] = reference_result + exact_incore
        else:
            block, slab_bytes, workspace = _occupied_i_slab_plan(
                n_occ,
                exact_per_i,
                fixed_mo,
                0,
                budget,
            )
            label = (
                "MP2 disk-backed occupied-orbital slab"
                if mode == "disk"
                else "MP2 direct occupied-orbital slab"
            )
            est.by_category[label] = slab_bytes
            est.category_details[label] = (
                f"mode={mode}, occupied-i block={block}/{n_occ}, "
                f"bytes-per-i={exact_per_i}, "
                f"requested={budget} bytes"
            )
            est.dims["mp2_block_size"] = block
            selected_workspace = max(selected_workspace, workspace)
            phases[f"MP2 {mode} correlation"] = reference_result + workspace
    _set_posthf_phase_peaks(est, reference_categories, phases)
    est._with_dims(
        n_basis=n,
        n_occ=n_occ,
        n_vir=n_vir,
        n_frozen_core=n_frozen,
        mp2_workspace_bytes=selected_workspace,
    )
    if mode == "disk":
        est.dims["mp2_disk_bytes"] = ovov_bytes
    if uses_df:
        est.dims["n_aux"] = n_aux
    return est


def _ump2_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    ref_options = _post_reference_options(options)
    ump2_options = _post_method_options(options, "ump2_options")
    est = _uhf_estimate(molecule, basis, ref_options)
    reference_categories = set(est.by_category)
    n = _n_basis(molecule, basis)
    n_elec = molecule.n_electrons()
    n_alpha_total = _clamped_occ(
        (n_elec + int(molecule.multiplicity) - 1) // 2,
        n,
    )
    n_beta_total = _clamped_occ(n_elec - n_alpha_total, n)
    n_frozen = _frozen_core_count(
        ump2_options,
        max(0, min(n_beta_total, n_alpha_total - 1)),
        molecule=molecule,
        published_default=True,
    )
    # Native UMP2 admits n_frozen == n_beta_total for a high-spin reference:
    # the beta-containing channels disappear while alpha-alpha correlation
    # remains. Do not clamp that valid zero-active-beta boundary back to one.
    n_alpha = max(0, n_alpha_total - n_frozen)
    n_beta = max(0, n_beta_total - n_frozen)
    n_vir_a = max(0, n - n_alpha_total)
    n_vir_b = max(0, n - n_beta_total)
    fixed_mo = 2 * n * n * 8
    channel_specs: list[tuple[str, int, int, int, int]] = []
    if n_alpha >= 2 and n_vir_a >= 2:
        channel_specs.append(("alpha-alpha", n_alpha, n_vir_a, n_alpha, n_vir_a))
    if n_beta >= 2 and n_vir_b >= 2:
        channel_specs.append(("beta-beta", n_beta, n_vir_b, n_beta, n_vir_b))
    if n_alpha >= 1 and n_beta >= 1 and n_vir_a >= 1 and n_vir_b >= 1:
        channel_specs.append(("alpha-beta", n_alpha, n_vir_a, n_beta, n_vir_b))
    largest_ovov = max(
        (
            no_bra * nv_bra * no_ket * nv_ket * 8
            for _, no_bra, nv_bra, no_ket, nv_ket in channel_specs
        ),
        default=0,
    )
    uses_df = _posthf_uses_density_fit(ump2_options, default=False)
    n_aux = _aux_basis_size(molecule, n, ump2_options) if uses_df else 0
    budget = _requested_workspace_bytes(ump2_options)
    requested_mode = _posthf_memory_mode(ump2_options)
    channel_workspaces = [
        (
            no_bra * n**3
            + no_bra * nv_bra * n * n
            + no_bra * nv_bra * no_ket * n
            + no_bra * nv_bra * no_ket * nv_ket
        )
        * 8
        for _, no_bra, nv_bra, no_ket, nv_ket in channel_specs
    ]
    largest_transform = max(channel_workspaces, default=0)
    exact_incore = fixed_mo + n**4 * 8 + largest_transform

    (
        three_index_bytes,
        metric_bytes,
        native_df_construction,
        native_df_resident,
    ) = _native_density_fitting_storage(n_aux, n)
    b_mo_bytes = n_aux * (
        n_alpha * n_vir_a + n_beta * n_vir_b
    ) * 8
    df_construction = fixed_mo + native_df_construction
    df_resident = fixed_mo + native_df_resident + b_mo_bytes
    transform_workers = min(n_aux, _omp_max_threads())
    alpha_output = n_aux * n_alpha * n_vir_a * 8
    alpha_transform = (
        fixed_mo
        + native_df_resident
        + alpha_output
        + transform_workers * n_alpha * (n + n_vir_a) * 8
    )
    beta_transform = (
        df_resident
        + transform_workers * n_beta * (n + n_vir_b) * 8
    )
    df_transform = max(alpha_transform, beta_transform)
    df_transform_scratch = max(
        transform_workers * n_alpha * (n + n_vir_a) * 8,
        transform_workers * n_beta * (n + n_vir_b) * 8,
    )
    df_irreducible = max(df_construction, df_transform)
    df_contraction = df_resident + largest_ovov
    incore_required = (
        max(df_irreducible, df_contraction)
        if uses_df
        else exact_incore
    )
    mode = requested_mode
    if mode == "auto":
        mode = "incore" if budget == 0 or incore_required <= budget else "direct"

    def panel_plan(
        *,
        density_fitted: bool,
        resident: int,
        construction: int,
    ) -> tuple[int, int, int]:
        largest_block = 0
        largest_slab = 0
        largest_workspace = max(resident, construction)
        for _, no_bra, nv_bra, no_ket, nv_ket in channel_specs:
            bytes_per_i = (
                nv_bra * no_ket * nv_ket * 8
                if density_fitted
                else _exact_mp2_direct_bytes_per_i(
                    n, nv_bra, no_ket, nv_ket
                )
            )
            block, slab, workspace = _occupied_i_slab_plan(
                no_bra,
                bytes_per_i,
                resident,
                construction,
                budget,
            )
            largest_block = max(largest_block, block)
            largest_slab = max(largest_slab, slab)
            largest_workspace = max(largest_workspace, workspace)
        return largest_block, largest_slab, largest_workspace

    reference_result = _reference_result_bytes(est)
    phases: dict[str, int] = {}
    selected_workspace = incore_required if mode == "incore" else 0
    est.by_category["UMP2 owned MO coefficient slices"] = fixed_mo
    if uses_df:
        est.by_category["DF-UMP2 three-index/B tensors"] = 2 * three_index_bytes
        est.by_category["DF-UMP2 metric/workspace"] = metric_bytes
        est.by_category["DF-UMP2 aux-OV workspace"] = b_mo_bytes
        est.by_category["DF-UMP2 construction overlap"] = df_construction
        est.by_category["DF-UMP2 threaded MO-transform scratch"] = (
            df_transform_scratch
        )
        phases["DF-UMP2 integral build"] = reference_result + df_construction
        phases["DF-UMP2 MO transformation"] = reference_result + df_transform
        if mode == "incore":
            est.by_category["UMP2 largest OVOV channel"] = largest_ovov
            phases["UMP2 incore correlation"] = (
                reference_result + df_contraction
            )
        else:
            block, slab_bytes, workspace = panel_plan(
                density_fitted=True,
                resident=df_resident,
                construction=df_irreducible,
            )
            label = (
                "DF-UMP2 disk-backed occupied-orbital slab"
                if mode == "disk"
                else "DF-UMP2 direct occupied-orbital slab"
            )
            est.by_category[label] = slab_bytes
            est.category_details[label] = (
                f"mode={mode}, largest occupied-i block={block}, "
                f"requested={budget} bytes"
            )
            est.dims["mp2_block_size"] = block
            selected_workspace = max(selected_workspace, workspace)
            phases[f"UMP2 {mode} correlation"] = reference_result + workspace
        if bool(getattr(ump2_options, "report_ri_residual", False)):
            block, slab_bytes, workspace = panel_plan(
                density_fitted=False,
                resident=fixed_mo,
                construction=0,
            )
            label = "UMP2 RI-residual direct occupied-orbital slab"
            est.by_category[label] = slab_bytes
            est.category_details[label] = (
                f"largest occupied-i block={block}, requested={budget} bytes"
            )
            phases["UMP2 RI residual diagnostic"] = (
                reference_result + workspace
            )
            selected_workspace = max(selected_workspace, workspace)
    else:
        if mode == "incore":
            if "ERI tensor" not in est.by_category:
                est.by_category["UMP2 AO ERI tensor"] = n**4 * 8
            est.by_category["UMP2 largest AO->MO scratch"] = largest_transform
            est.by_category["UMP2 largest OVOV channel"] = largest_ovov
            phases["UMP2 incore correlation"] = reference_result + exact_incore
        else:
            block, slab_bytes, workspace = panel_plan(
                density_fitted=False,
                resident=fixed_mo,
                construction=0,
            )
            label = (
                "UMP2 disk-backed occupied-orbital slab"
                if mode == "disk"
                else "UMP2 direct occupied-orbital slab"
            )
            est.by_category[label] = slab_bytes
            est.category_details[label] = (
                f"mode={mode}, largest occupied-i block={block}, "
                f"requested={budget} bytes"
            )
            est.dims["mp2_block_size"] = block
            selected_workspace = max(selected_workspace, workspace)
            phases[f"UMP2 {mode} correlation"] = (
                reference_result + workspace
            )
    _set_posthf_phase_peaks(est, reference_categories, phases)
    est._with_dims(
        n_basis=n,
        n_occ=n_alpha + n_beta,
        n_vir=n_vir_a + n_vir_b,
        n_frozen_core=n_frozen,
        mp2_workspace_bytes=selected_workspace,
    )
    if mode == "disk":
        est.dims["mp2_disk_bytes"] = largest_ovov
    if uses_df:
        est.dims["n_aux"] = n_aux
    return est


def _rohf_mp2_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    """Dense pure-Python semicanonical ROHF-MP2 accuracy-oracle route.

    ``run_rohf_mp2`` does not call the bounded native UMP2 implementation.
    It constructs a fresh RI substrate, transforms the complete active MO
    square, semicanonicalizes alpha and beta copies, and then calls
    ``dlpno._ccsd_ref.run_ref_ump2``.  That reference kernel deliberately
    materializes four spatial ``Vpair`` tensors and a dense spin-orbital ERI
    through NumPy mesh/index arrays.  Model that actual setup rather than
    inheriting the native UMP2 occupied-i slab planner.
    """
    ref_options = _post_reference_options(options)
    method_options = _post_method_options(options, "ump2_options")
    # A ROHF result carries common orbitals but separate alpha/beta Fock and
    # density state. The UHF-shaped reference accounting is conservative for
    # those retained arrays while preserving the correct open-shell scale.
    est = _uhf_estimate(molecule, basis, ref_options)
    reference_categories = set(est.by_category)
    reference_result = _reference_result_bytes(est)

    n = _n_basis(molecule, basis)
    n_elec = molecule.n_electrons()
    n_alpha_total = _clamped_occ(
        (n_elec + int(molecule.multiplicity) - 1) // 2,
        n,
    )
    n_beta_total = _clamped_occ(n_elec - n_alpha_total, n)
    n_frozen = _frozen_core_count(
        method_options,
        n_beta_total,
        molecule=molecule,
        published_default=True,
    )
    n_mo = max(1, n - n_frozen)
    n_alpha = max(1, n_alpha_total - n_frozen)
    n_beta = max(0, n_beta_total - n_frozen)
    n_occ = n_alpha + n_beta
    n_vir = max(1, n_mo - n_alpha) + max(1, n_mo - n_beta)

    # run_rohf_mp2 always uses RI. Resolve its default RIFIT basis when the
    # estimator has not been handed a method-specific auxiliary name.
    aux_options = method_options
    aux_name = str(getattr(method_options, "aux_basis", "") or "")
    if not aux_name:
        try:
            from .density_fitting import default_aux_basis_for

            aux_name = default_aux_basis_for(basis.name, kind="ri")
        except Exception:
            aux_name = ""
    if aux_name:
        try:
            n_aux = int(BasisSet(molecule, aux_name, require_all_atoms=False).nbasis)
        except Exception:
            n_aux = _aux_basis_size(molecule, n, aux_options)
    else:
        n_aux = _aux_basis_size(molecule, n, aux_options)

    _, _, df_construction, df_resident = _python_density_fitting_storage(
        n_aux,
        n,
    )

    # During the second spin semicanonicalisation, the original B_mo, the
    # completed alpha factor, the einsum temporary, and the beta output
    # coexist.  The input and semicanonical alpha/beta Fock matrices coexist
    # as well.
    transformed_b = 4 * n_aux * n_mo * n_mo * 8
    semicanonical_matrices = (4 * n_mo * n_mo + 2 * n_mo) * 8
    spatial_vpair = 4 * n_mo**4 * 8

    spin_orbitals = 2 * n_mo
    spin_eri = spin_orbitals**4 * 8
    # _spin_orbital_eri_uhf owns p/q/r/s, their four spin remainders,
    # their four spatial quotients (twelve int64 spin^4 arrays), then boolean
    # masks and advanced-index temporaries. Thirteen peer tensor extents in
    # addition to the destination is a conservative explicit bound.
    spin_index_setup = 13 * spin_eri

    oovv = n_occ * n_occ * n_vir * n_vir
    ov = n_occ * n_vir
    amplitude_workspace = (2 * oovv + 2 * ov) * 8

    est.by_category["ROHF-MP2 DF construction overlap"] = df_construction
    est.by_category["ROHF-MP2 retained DF substrate"] = df_resident
    est.by_category["ROHF-MP2 semicanonical B/F copies"] = (
        transformed_b + semicanonical_matrices
    )
    est.by_category["ROHF-MP2 four spatial Vpair tensors"] = spatial_vpair
    est.by_category["ROHF-MP2 spin-orbital ERI tensor"] = spin_eri
    est.by_category["ROHF-MP2 spin-index mesh/mask temporaries"] = (
        spin_index_setup
    )
    est.by_category["ROHF-MP2 denominator/amplitude tensors"] = (
        amplitude_workspace
    )
    est.category_details["ROHF-MP2 spin-index mesh/mask temporaries"] = (
        "dense pure-Python accuracy oracle; requested-memory native "
        "occupied-i slab modes do not apply"
    )

    retained_setup = df_resident + transformed_b + semicanonical_matrices
    phases = {
        "ROHF-MP2 DF construction": reference_result + df_construction,
        "ROHF-MP2 dense spin-orbital setup": (
            reference_result
            + retained_setup
            + spatial_vpair
            + spin_eri
            + spin_index_setup
        ),
        # Reordering temporarily owns the old and new spin ERIs. The later
        # denominator/amplitude phase owns one ERI; these are sequential.
        "ROHF-MP2 dense amplitude contraction": (
            reference_result
            + retained_setup
            + max(2 * spin_eri, spin_eri + amplitude_workspace)
        ),
    }
    _set_posthf_phase_peaks(est, reference_categories, phases)
    est._with_dims(
        n_basis=n,
        n_occ=n_occ,
        n_vir=n_vir,
        n_frozen_core=n_frozen,
        n_aux=n_aux,
    )
    return est


def _ovgf_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    """Memory estimate for the dense OVGF/GF2 runner path."""
    ref_options = _post_reference_options(options)
    n = _n_basis(molecule, basis)
    open_shell = int(getattr(molecule, "multiplicity", 1) or 1) != 1
    if open_shell:
        est = _uhf_estimate(molecule, basis, ref_options)
    else:
        est = _rhf_estimate(molecule, basis, ref_options)

    ao_eri_bytes = n**4 * 8
    est.by_category["OVGF AO ERI tensor"] = ao_eri_bytes

    if open_shell:
        n_spin = 2 * n
        n_alpha, n_beta, n_vir_a, n_vir_b = _open_shell_occ_vir(molecule, n)
        n_occ_spin = n_alpha + n_beta
        n_vir_spin = n_vir_a + n_vir_b
        est.by_category["OVGF spin-block MO tensors"] = 3 * ao_eri_bytes
        est.by_category["OVGF spin-orbital tensor"] = n_spin**4 * 8
        est.by_category["OVGF Dyson spin work slices"] = (
            max(
                n_occ_spin * n_vir_spin * n_vir_spin,
                n_vir_spin * n_occ_spin * n_occ_spin,
            )
            * 4
            * 8
        )
        est._with_dims(n_basis=n, n_occ=n_occ_spin, n_vir=n_vir_spin)
        return est

    n_occ, n_vir = _closed_shell_occ_vir(molecule, n)
    est.by_category["OVGF MO ERI tensor"] = ao_eri_bytes
    est.by_category["OVGF AO->MO transform scratch"] = ao_eri_bytes
    est.by_category["OVGF Dyson work slices"] = (
        max(n_occ * n_vir * n_occ, n_vir * n_occ * n_vir) * 4 * 8
    )
    if 2 * n <= 60:
        est.by_category["OVGF renormalized spin-orbital tensor"] = (2 * n) ** 4 * 8
    est._with_dims(n_basis=n, n_occ=n_occ, n_vir=n_vir)
    return est


def _ccsd_triples_workspace(
    n_occ: int,
    n_vir: int,
    cc_options,
    *,
    open_shell: bool,
    uses_df: bool,
    n_aux: int,
) -> tuple[str, int, int, int, int, int, int]:
    """Mirror the native triples planners.

    Returns ``(mode, tile, threads, peak, retained, work_list, disk_extent)``.
    ``peak`` has the same full-peak meaning as native
    ``triples_workspace_bytes``: retained numerical inputs plus aggregate
    scratch for every active worker.
    """
    runtime_threads = _omp_max_threads()
    try:
        requested_threads = int(
            getattr(cc_options, "triples_max_threads", 0) or 0
        )
    except (TypeError, ValueError):
        requested_threads = 0
    if requested_threads > 0:
        runtime_threads = min(runtime_threads, requested_threads)
    raw_mode = str(
        getattr(cc_options, "triples_memory_mode", "auto") or "auto"
    ).strip().lower()
    mode = {
        "low": "blocked",
    }.get(raw_mode, raw_mode)
    if mode not in {"auto", "fast", "blocked", "direct", "disk"}:
        raise ValueError(
            "triples_memory_mode must be one of auto, fast, blocked "
            "(legacy: low), direct, or disk; got " + repr(raw_mode)
        )

    budget = _requested_workspace_bytes(cc_options)
    if open_shell:
        if mode == "disk":
            raise ValueError(
                "UCCSD(T) triples_memory_mode='disk' cannot spill the dense "
                "spin-orbital n^4 integral representation; use 'direct'"
            )
        triples_count = (
            n_occ * (n_occ - 1) * (n_occ - 2) // 6
            if n_occ >= 3
            else 1
        )
        threads = max(1, min(runtime_threads, triples_count))
        retained = (
            (n_occ + n_vir) ** 4
            + n_occ * n_vir
            + n_occ * n_occ * n_vir * n_vir
            + n_occ
            + n_vir
        ) * 8
        fast_per_thread = 3 * n_vir**3 * 8

        def open_fast(candidate_threads: int) -> int:
            return retained + candidate_threads * fast_per_thread

        if mode == "auto":
            fast_peak = open_fast(threads)
            if budget == 0 or fast_peak <= budget:
                return "fast", n_vir, threads, fast_peak, retained, 0, 0
            mode = "direct"
        if mode == "fast":
            for candidate_threads in range(threads, 0, -1):
                peak = open_fast(candidate_threads)
                if budget == 0 or peak <= budget:
                    return (
                        "fast",
                        n_vir,
                        candidate_threads,
                        peak,
                        retained,
                        0,
                        0,
                    )
            return "fast", n_vir, 1, open_fast(1), retained, 0, 0

        # The open-shell kernel aliases legacy blocked/low to its scalar
        # integral-direct implementation and reports that realised mode.
        return "direct", 0, threads, retained, retained, 0, 0

    try:
        requested_tile = max(
            0,
            int(getattr(cc_options, "triples_tile_size", 0) or 0),
        )
    except (TypeError, ValueError):
        requested_tile = 0
    requested_tile = min(n_vir, requested_tile)
    work_entries = max(1, n_occ * (n_occ + 1) * (n_occ + 2) // 6)
    threads = max(1, min(runtime_threads, work_entries))
    work_bytes = work_entries * 3 * 8

    # T1, T2, f_ov, eps_o/v, ov_ov, and oo_ov survive every closed-shell
    # triples strategy. Fast/canonical routes retain ov_vv; DF blocked/direct
    # retain B_ov+B_vv, while disk releases those factors after spilling.
    base_retained = (
        2 * n_occ * n_vir
        + 2 * n_occ * n_occ * n_vir * n_vir
        + n_occ**3 * n_vir
        + n_occ
        + n_vir
    ) * 8
    ov_vv_bytes = n_occ * n_vir**3 * 8
    factor_bytes = n_aux * (n_occ * n_vir + n_vir * n_vir) * 8
    fast_retained = base_retained + ov_vv_bytes
    low_retained = base_retained + (
        factor_bytes if uses_df else ov_vv_bytes
    )
    disk_retained = base_retained

    fast_per_thread = (4 * n_vir**3 + n_vir**2) * 8
    blocked_per_thread_tile = 2 * n_vir * n_vir * 8
    factor_row_bytes = (
        (n_occ * n_vir + n_vir * n_vir) * 8 if uses_df else 0
    )
    disk_bytes = 32 + n_aux * factor_row_bytes if uses_df else 0

    def fits(peak: int) -> bool:
        return budget == 0 or peak <= budget

    def fast_plan(candidate_threads: int):
        return fast_retained + work_bytes + candidate_threads * fast_per_thread

    def blocked_plan(candidate_threads: int, candidate_tile: int):
        return (
            low_retained
            + work_bytes
            + candidate_threads * candidate_tile * blocked_per_thread_tile
        )

    def direct_plan() -> int:
        return low_retained + work_bytes

    def disk_plan(candidate_threads: int, candidate_tile: int) -> int:
        streamed = (
            disk_retained
            + work_bytes
            + candidate_threads
            * (
                candidate_tile * blocked_per_thread_tile
                + factor_row_bytes
            )
        )
        # The source factors are resident during the spill transition.
        return max(low_retained, streamed)

    def find_blocked() -> tuple[int, int, int] | None:
        for candidate_threads in range(threads, 0, -1):
            tile = requested_tile or n_vir
            if budget > 0:
                fixed = low_retained + work_bytes
                if fixed > budget:
                    continue
                tile = min(
                    tile,
                    (budget - fixed)
                    // max(1, candidate_threads * blocked_per_thread_tile),
                )
            elif requested_tile == 0:
                tile = min(64, n_vir)
            if tile < 1:
                continue
            peak = blocked_plan(candidate_threads, tile)
            if fits(peak):
                return tile, candidate_threads, peak
        return None

    def find_disk() -> tuple[int, int, int] | None:
        if not uses_df:
            return None
        for candidate_threads in range(threads, 0, -1):
            tile = requested_tile or min(64, n_vir)
            if budget > 0:
                fixed = (
                    disk_retained
                    + work_bytes
                    + candidate_threads * factor_row_bytes
                )
                if fixed > budget:
                    continue
                tile = min(
                    tile,
                    (budget - fixed)
                    // max(1, candidate_threads * blocked_per_thread_tile),
                )
            if tile < 1:
                continue
            peak = disk_plan(candidate_threads, tile)
            if fits(peak):
                return tile, candidate_threads, peak
        return None

    if mode == "auto":
        fast_peak = fast_plan(threads)
        if budget == 0 and fast_peak <= 512 * 1024**2:
            return (
                "fast", n_vir, threads, fast_peak,
                fast_retained, work_bytes, 0,
            )
        if budget > 0 and fits(fast_peak):
            return (
                "fast", n_vir, threads, fast_peak,
                fast_retained, work_bytes, 0,
            )
        blocked = find_blocked()
        if blocked is not None:
            tile, candidate_threads, peak = blocked
            return (
                "blocked", tile, candidate_threads, peak,
                low_retained, work_bytes, 0,
            )
        direct_peak = direct_plan()
        if fits(direct_peak):
            return (
                "direct", 0, threads, direct_peak,
                low_retained, work_bytes, 0,
            )
        disk = find_disk()
        if disk is not None:
            tile, candidate_threads, peak = disk
            return (
                "disk", tile, candidate_threads, peak,
                disk_retained, work_bytes, disk_bytes,
            )
        # Preserve the smallest irreducible peak so admission fails with a
        # truthful number instead of turning an impossible cap into zero.
        candidates = [("direct", 0, threads, direct_peak, low_retained, 0)]
        if uses_df:
            disk_peak = disk_plan(1, 1)
            candidates.append(
                ("disk", 1, 1, disk_peak, disk_retained, disk_bytes)
            )
        name, tile, candidate_threads, peak, retained, extent = min(
            candidates, key=lambda item: item[3]
        )
        return (
            name, tile, candidate_threads, peak,
            retained, work_bytes, extent,
        )

    if mode == "fast":
        for candidate_threads in range(threads, 0, -1):
            peak = fast_plan(candidate_threads)
            if fits(peak):
                return (
                    "fast", n_vir, candidate_threads, peak,
                    fast_retained, work_bytes, 0,
                )
        return (
            "fast", n_vir, 1, fast_plan(1),
            fast_retained, work_bytes, 0,
        )
    if mode == "blocked":
        blocked = find_blocked()
        if blocked is not None:
            tile, candidate_threads, peak = blocked
            return (
                "blocked", tile, candidate_threads, peak,
                low_retained, work_bytes, 0,
            )
        return (
            "blocked", 1, 1, blocked_plan(1, 1),
            low_retained, work_bytes, 0,
        )
    if mode == "disk":
        if not uses_df:
            raise ValueError(
                "CCSD(T) triples_memory_mode='disk' requires density fitting"
            )
        disk = find_disk()
        if disk is not None:
            tile, candidate_threads, peak = disk
            return (
                "disk", tile, candidate_threads, peak,
                disk_retained, work_bytes, disk_bytes,
            )
        return (
            "disk", 1, 1, disk_plan(1, 1),
            disk_retained, work_bytes, disk_bytes,
        )

    # Integral-direct triples recomputes scalar virtual tuples. Its only heap
    # scratch beyond retained inputs is the shared occupied-triple work list.
    return (
        "direct", 0, threads, direct_plan(),
        low_retained, work_bytes, 0,
    )


def _ccsd_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    open_shell: bool = False,
    triples: bool = False,
    suppress_option_triples: bool = False,
) -> MemoryEstimate:
    ref_options = _post_reference_options(options)
    cc_options = _post_method_options(options, "ccsd_options")
    est = (
        _uhf_estimate(molecule, basis, ref_options)
        if open_shell
        else _rhf_estimate(molecule, basis, ref_options)
    )
    reference_categories = set(est.by_category)
    n = _n_basis(molecule, basis)
    uses_df = _posthf_uses_density_fit(cc_options, default=True)
    fno_enabled = bool(getattr(cc_options, "fno", False))
    fno_kept_estimate: int | None = None
    if open_shell:
        if fno_enabled:
            raise ValueError("FNO-CCSD requires a closed-shell reference")
        n_elec = molecule.n_electrons()
        n_alpha_total = _clamped_occ(
            (n_elec + int(molecule.multiplicity) - 1) // 2,
            n,
        )
        n_beta_total = _clamped_occ(n_elec - n_alpha_total, n)
        n_frozen = _frozen_core_count(
            cc_options,
            max(0, min(n_beta_total, n_alpha_total - 1)),
            molecule=molecule,
            published_default=True,
        )
        # Native UCCSD, like UMP2 and the local open-shell solver, permits
        # the frozen core to empty the beta active space while alpha occupied
        # pairs remain. Preserve zero here rather than forcing a phantom beta
        # occupied orbital into the preflight dimensions.
        n_alpha = max(0, n_alpha_total - n_frozen)
        n_beta = max(0, n_beta_total - n_frozen)
        n_vir_a = max(0, n - n_alpha_total)
        n_vir_b = max(0, n - n_beta_total)
        n_occ = n_alpha + n_beta
        n_vir = n_vir_a + n_vir_b
    else:
        n_occ_total = _clamped_occ(molecule.n_electrons() // 2, n)
        n_frozen = _frozen_core_count(
            cc_options,
            max(0, n_occ_total - 1),
            molecule=molecule,
            published_default=True,
        )
        n_occ, n_vir = _closed_shell_occ_vir(
            molecule,
            n,
            n_frozen_core=n_frozen,
        )
        full_n_vir = n_vir
        if fno_enabled:
            if not uses_df:
                raise ValueError("FNO-CCSD requires density_fit=True")
            keep_fraction = getattr(cc_options, "fno_keep_fraction", None)
            if keep_fraction is None:
                # Threshold selection is data-dependent; retaining the full
                # virtual count is the conservative final-space estimate.
                fno_kept_estimate = full_n_vir
            else:
                fraction = float(keep_fraction)
                if not 0.0 < fraction <= 1.0:
                    raise ValueError("FNO fno_keep_fraction must be in (0, 1]")
                fno_kept_estimate = max(
                    1,
                    min(full_n_vir, int(round(fraction * full_n_vir))),
                )
            # The dense full-space MP2-NO setup is modeled separately below.
            # Native CC and triples receive only the selected FNO virtuals.
            n_vir = fno_kept_estimate
    n_ov = n_occ * n_vir
    n_oovv = n_occ * n_occ * n_vir * n_vir

    est.by_category["CCSD T1/T2 amplitudes"] = (4 * n_ov + 2 * n_oovv) * 8
    diis_vectors = max(1, int(getattr(cc_options, "diis_subspace_size", 6) or 6))
    est.by_category["CCSD residual/DIIS amplitudes"] = (
        2
        * diis_vectors
        * (n_ov + n_oovv)
        * 8
    )
    est.by_category["CCSD D1/D2 intermediates"] = (
        2 * n_oovv  # tau and tau-tilde
        + n_occ * n_vir
        + 2 * n_vir * n_vir
        + 2 * n_occ * n_occ
        + n_occ**4
        + 3 * n_ov * n_ov  # W1/W2/WX ring intermediates
        + 2 * n_oovv  # particle-particle M buffer + pair-exchange half
    ) * 8

    common_solver_categories = [
        "CCSD T1/T2 amplitudes",
        "CCSD residual/DIIS amplitudes",
        "CCSD D1/D2 intermediates",
    ]
    integral_categories: list[str] = []
    build_categories: list[str] = []
    cc_build_phases: dict[str, int] = {}
    n_aux = 0
    if open_shell:
        # The unrestricted kernel retains a dense antisymmetrized integral
        # tensor over the combined active spin-orbital window. This n^4 term
        # is the real UCCSD memory wall and must not be approximated by the
        # smaller restricted spatial blocks.
        n_spin = n_occ + n_vir
        est.by_category["UCCSD SpinOrbitalIntegrals"] = n_spin**4 * 8
        integral_categories.append("UCCSD SpinOrbitalIntegrals")
        n_a = n_alpha + n_vir_a
        n_b = n_beta + n_vir_b
        if uses_df:
            n_aux = _aux_basis_size(molecule, n, cc_options)
            (
                _,
                _,
                df_construction,
                df_resident,
            ) = _native_density_fitting_storage(n_aux, n)
            b_a = n_aux * n_a**2 * 8
            b_b = n_aux * n_b**2 * 8
            b_so = n_aux * n_spin**2 * 8
            b_transforms = b_a + b_b + b_so
            transform_workers = min(n_aux, _omp_max_threads())
            transform_peak = max(
                df_resident
                + b_a
                + transform_workers * n_a * (n + n_a) * 8,
                df_resident
                + b_a
                + b_b
                + transform_workers * n_b * (n + n_b) * 8,
                df_resident + b_transforms,
            )
            p_block = max(1, min(n_spin, 2048 // max(1, n_spin)))
            chem_tile = p_block * n_spin**3 * 8
            est.by_category["DF-UCCSD integral-build scratch"] = (
                df_resident + b_transforms + chem_tile
            )
            est.by_category["DF-UCCSD construction overlap"] = (
                df_construction
            )
            est.by_category["DF-UCCSD threaded MO-transform peak"] = (
                transform_peak
            )
            build_categories.append("DF-UCCSD integral-build scratch")
            cc_build_phases["CC DF construction"] = df_construction
            cc_build_phases["CC DF MO transformation"] = transform_peak
        else:
            # AO ERI plus the aa, bb, and ab spatial-channel transforms are
            # transient peers of the destination spin-orbital tensor.
            est.by_category["UCCSD canonical integral-build scratch"] = (
                n**4 + n_a**4 + n_b**4 + n_a**2 * n_b**2
            ) * 8
            build_categories.append("UCCSD canonical integral-build scratch")
    elif uses_df:
        n_aux = _aux_basis_size(molecule, n, cc_options)
        (
            _,
            _,
            df_construction,
            df_resident,
        ) = _native_density_fitting_storage(n_aux, n)
        b_ov = n_aux * n_occ * n_vir * 8
        b_vv = n_aux * n_vir * n_vir * 8
        b_oo = n_aux * n_occ * n_occ * 8
        est.by_category["DF-CCSD auxiliary integrals"] = b_ov + b_vv
        est.category_details["DF-CCSD auxiliary integrals"] = (
            "B_ov+B_vv retained by the CCSD solve; the native AO DF object "
            "and B_oo are build-only"
        )
        transform_workers = min(n_aux, _omp_max_threads())
        retained_factors = 0
        transform_peak = 0
        for n_left, n_right, output in (
            (n_occ, n_vir, b_ov),
            (n_vir, n_vir, b_vv),
            (n_occ, n_occ, b_oo),
        ):
            transform_scratch = (
                transform_workers * n_left * (n + n_right) * 8
            )
            transform_peak = max(
                transform_peak,
                df_resident
                + retained_factors
                + output
                + transform_scratch,
            )
            retained_factors += output
        est.by_category["DF-CCSD construction overlap"] = df_construction
        est.by_category["DF-CCSD threaded MO-transform peak"] = transform_peak
        est.by_category["DF-CCSD build-only DF substrate/B_oo"] = (
            df_resident + b_oo
        )
        build_categories.append("DF-CCSD build-only DF substrate/B_oo")
        cc_build_phases["CC DF construction"] = df_construction
        cc_build_phases["CC DF MO transformation"] = transform_peak
        est.by_category["DF-CCSD MO integral blocks"] = (
            n_ov * n_ov
            + n_occ**4
            + n_occ**3 * n_vir
            + n_oovv
            + n_occ * n_vir**3
        ) * 8
        vvvv_elems = n_vir**4
        vvvv_bytes = vvvv_elems * 8
        if vvvv_bytes <= _CCSD_VVVV_INCORE_BUDGET_BYTES:
            # V.vv_vv plus the W_abef work matrix are both resident on the
            # in-core path. B_vv itself is counted in the auxiliary category.
            est.by_category["DF-CCSD VVVV/tile scratch"] = 2 * vvvv_bytes
        else:
            rows = max(1, _CCSD_VVVV_TILE_ROWS)
            a_block = max(1, min(n_vir, rows // max(1, n_vir)))
            est.by_category["DF-CCSD VVVV/tile scratch"] = (
                2 * a_block * n_vir**3 + n_aux * a_block * n_vir
            ) * 8
        integral_categories.extend(
            (
                "DF-CCSD auxiliary integrals",
                "DF-CCSD MO integral blocks",
                "DF-CCSD VVVV/tile scratch",
            )
        )
    else:
        n_corr = n_occ + n_vir
        canonical_blocks = (
            n_ov * n_ov
            + n_occ**4
            + n_occ**3 * n_vir
            + n_oovv
            + n_occ * n_vir**3
            + n_vir**4
        ) * 8
        # build_integral_blocks_canonical first calls
        # eri_mo_pair_transform. At that function's return boundary the AO
        # ERI, both half-transform matrices, Wt, and the returned transpose
        # coexist. The caller then retains W while extracting every IntegralBlocks
        # member, so model that second source-lifetime peak separately.
        pair_transform_peak = (
            n**4
            + 2 * n * n * n_corr * n_corr
            + 2 * n_corr**4
        ) * 8
        block_extraction_peak = n_corr**4 * 8 + canonical_blocks
        est.by_category["CCSD canonical MO integral blocks"] = canonical_blocks
        est.by_category["CCSD canonical VVVV residual scratch"] = (
            n_vir**4 * 8
        )
        est.by_category["CCSD canonical AO-to-MO transform peak"] = (
            pair_transform_peak
        )
        est.by_category["CCSD canonical block extraction peak"] = (
            block_extraction_peak
        )
        est.category_details["CCSD canonical AO-to-MO transform peak"] = (
            "AO ERI + H + H-transpose + W-transpose + returned W at the "
            "eri_mo_pair_transform return boundary"
        )
        integral_categories.extend(
            (
                "CCSD canonical MO integral blocks",
                "CCSD canonical VVVV residual scratch",
            )
        )
        cc_build_phases["CC canonical AO-to-MO pair transform"] = (
            pair_transform_peak
        )
        cc_build_phases["CC canonical MO block extraction"] = (
            block_extraction_peak
        )

    compute_triples = triples or (
        not suppress_option_triples
        and bool(getattr(cc_options, "compute_triples", False))
    )
    triples_categories: list[str] = []
    triples_threads = _omp_max_threads()
    if compute_triples:
        (
            mode,
            tile,
            triples_threads,
            triples_peak,
            retained,
            work_bytes,
            disk_bytes,
        ) = _ccsd_triples_workspace(
            n_occ,
            n_vir,
            cc_options,
            open_shell=open_shell,
            uses_df=uses_df,
            n_aux=n_aux,
        )
        est.by_category["CCSD(T) triples image buffers"] = max(
            0,
            triples_peak - retained - work_bytes,
        )
        est.by_category["CCSD(T) retained amplitudes/integrals"] = retained
        est.category_details["CCSD(T) triples image buffers"] = (
            f"mode={mode}, virtual tile={tile}/{n_vir}, "
            f"threads={triples_threads}, requested="
            f"{_requested_workspace_bytes(cc_options)} bytes, "
            f"full modeled peak={triples_peak} bytes"
        )
        est.dims["triples_tile_size"] = tile
        est.dims["triples_workspace_bytes"] = triples_peak
        if disk_bytes:
            est.dims["triples_disk_bytes"] = disk_bytes
        triples_categories.append("CCSD(T) triples image buffers")

        # Work list: ~n_occ^3/6 entries (unordered i<=j<=k loop),
        # 3 indices per entry at 8 bytes each (std::array<int,3>).
        est.by_category["CCSD(T) triples work list"] = work_bytes
        triples_categories.append("CCSD(T) triples work list")

    reference_result = _reference_result_bytes(est)
    phases: dict[str, int] = {
        label: reference_result + peak
        for label, peak in cc_build_phases.items()
    }
    if build_categories:
        phases["CC integral build"] = reference_result + _category_sum(
            est,
            integral_categories + build_categories,
        )
    phases["CC amplitude solve"] = reference_result + _category_sum(
        est,
        common_solver_categories + integral_categories,
    )
    if triples_categories:
        # Residual/DIIS and VVVV contraction scratch have gone out of scope;
        # amplitudes and the integral blocks used by (T) remain resident.
        phases["CCSD(T) correction"] = reference_result + triples_peak
    _set_posthf_phase_peaks(est, reference_categories, phases)

    # Attach key scaling dimensions for the formatted output.
    est._with_dims(
        n_basis=n,
        n_occ=n_occ,
        n_vir=n_vir,
        n_threads=triples_threads,
        n_frozen_core=n_frozen,
    )
    if uses_df:
        est.dims["n_aux"] = n_aux

    # FNO orchestration precedes the native final-space CC call.  Its Python
    # MP2 natural-orbital setup is a separate dense phase: the full-space
    # arrays remain live while the optional truncated delta-MP2 arrays are
    # built.  run_fno_ccsd releases all of them before entering native CC, so
    # this setup is a peer phase rather than native retained storage.
    if fno_enabled:
        fno_kept = int(fno_kept_estimate or n_vir)

        (
            _,
            _,
            fno_df_construction,
            fno_df_resident,
        ) = _python_density_fitting_storage(n_aux, n)
        full_b_ov = n_aux * n_occ * full_n_vir * 8
        full_oovv = n_occ * n_occ * full_n_vir * full_n_vir * 8

        # g, d, and t remain live while ``2*t - t.T`` allocates both its
        # multiplication temporary and result. Four full tensors persist
        # through natural-orbital selection; five is the construction peak.
        full_mp2_persistent = 4 * full_oovv
        full_mp2_tensors = 5 * full_oovv
        # D, U, f_vv, semicanonical eigensolver/rotation products, and the
        # retained C_vir_fno. This is lower-order than oovv but material for
        # aggressive truncation and small occupied spaces.
        no_matrices = (
            4 * full_n_vir * full_n_vir
            + n * full_n_vir
            + 3 * fno_kept * fno_kept
            + full_n_vir * fno_kept
            + n * fno_kept
        ) * 8

        truncated_b_ov = 0
        delta_mp2_tensors = 0
        truncated_transform_scratch = 0
        if bool(getattr(cc_options, "fno_delta_mp2", True)):
            truncated_b_ov = n_aux * n_occ * fno_kept * 8
            truncated_transform_scratch = n_aux * n * fno_kept * 8
            truncated_oovv = (
                n_occ * n_occ * fno_kept * fno_kept * 8
            )
            # g_t, d_t, and t_t coexist with both the multiplication
            # temporary and subtraction result in ``2*t_t - t_t.T``.
            delta_mp2_tensors = 5 * truncated_oovv

        est.by_category["FNO DF construction overlap"] = fno_df_construction
        est.by_category["FNO retained DF substrate"] = fno_df_resident
        est.by_category["FNO full-space B_ov"] = full_b_ov
        est.by_category["FNO full-space MP2-NO tensors"] = full_mp2_tensors
        est.by_category["FNO natural-orbital matrices"] = no_matrices
        if truncated_b_ov:
            est.by_category["FNO truncated B_ov"] = truncated_b_ov
            est.by_category["FNO truncated MO-transform scratch"] = (
                truncated_transform_scratch
            )
            est.by_category["FNO delta-MP2 tensors"] = delta_mp2_tensors

        fno_live_setup = (
            fno_df_resident
            + full_b_ov
            + full_mp2_persistent
            + no_matrices
            + truncated_b_ov
            + delta_mp2_tensors
        )
        fno_transform_setup = (
            fno_df_resident
            + full_b_ov
            + full_mp2_persistent
            + no_matrices
            + truncated_b_ov
            + truncated_transform_scratch
        )
        fno_full_mp2_setup = (
            fno_df_resident + full_b_ov + full_mp2_tensors
        )
        est.phase_peaks["FNO dense MP2 natural-orbital setup"] = (
            reference_result
            + max(
                fno_df_construction,
                fno_full_mp2_setup,
                fno_transform_setup,
                fno_live_setup,
            )
        )
        est.category_details["FNO full-space MP2-NO tensors"] = (
            "g,d,t,tt remain live while optional truncated B_ov/g_t/d_t/t_t "
            "and its energy-expression temporary are built; all setup arrays "
            "are released before native final-space CC"
        )
        est.dims["fno_virtual_kept_estimate"] = fno_kept
        est.dims["fno_virtual_total"] = full_n_vir
    return est


def _ccsd_t_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    return _ccsd_estimate(molecule, basis, options, triples=True)


def _bccd_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    """BCCD with its persistent outer Brueckner AO-ERI tensor."""
    return _bccd_common_estimate(molecule, basis, options, triples=False)


def _bccd_t_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    """BCCD(T) with its persistent outer Brueckner AO-ERI tensor."""
    return _bccd_common_estimate(molecule, basis, options, triples=True)


def _bccd_common_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    triples: bool,
) -> MemoryEstimate:
    """Add the Python Brueckner driver's AO ERI to every inner-CC phase.

    ``run_bccd`` constructs the full exact AO tensor once for the orbital
    optimization reference builds. It remains live while every density-fitted
    or canonical inner CC probe and the final BCCD(T) kernel executes.
    """
    est = _ccsd_estimate(molecule, basis, options, triples=triples)
    ao_eri = _n_basis(molecule, basis) ** 4 * 8
    est.by_category["BCCD persistent AO ERI"] = ao_eri
    est.category_details["BCCD persistent AO ERI"] = (
        "outer Brueckner reference tensor; remains live across inner CC "
        "probes and the final BCCD(T) call"
    )
    for label in tuple(est.phase_peaks):
        if label not in {"Runtime baseline", "SCF reference"}:
            est.phase_peaks[label] += ao_eri
    return est


def _accsd_t_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    """Dense Python/Lambda A-CCSD(T) route.

    A-CCSD(T) does not execute the bounded standard-(T) planner. It eagerly
    builds every spatial CC integral block, including ``vvvv``, solves the
    Lambda equations with NumPy tensor intermediates, and calls a separate
    dense native Lambda-triples contraction. Model those phases explicitly so
    a requested standard-(T) mode cannot make this route look bounded.
    """
    est = _ccsd_estimate(
        molecule,
        basis,
        options,
        suppress_option_triples=True,
    )
    cc_options = _post_method_options(options, "ccsd_options")
    n = _n_basis(molecule, basis)
    n_occ_total = _clamped_occ(molecule.n_electrons() // 2, n)
    n_frozen = _frozen_core_count(
        cc_options,
        max(0, n_occ_total - 1),
        molecule=molecule,
        published_default=True,
    )
    no, nv = _closed_shell_occ_vir(
        molecule,
        n,
        n_frozen_core=n_frozen,
    )
    uses_df = _posthf_uses_density_fit(cc_options, default=True)
    n_aux = _aux_basis_size(molecule, n, cc_options) if uses_df else 0
    reference_result = _reference_result_bytes(est)

    dense_blocks = (
        2 * no * no * nv * nv
        + no * nv**3
        + no**3 * nv
        + no**4
        + nv**4
    ) * 8
    amplitude_elements = no * nv + no * no * nv * nv
    diis_vectors = max(
        0,
        int(getattr(cc_options, "diis_subspace_size", 6) or 0),
    )
    # Lambda multiplier/step/residual arrays and flattened DIIS histories.
    lambda_workspace = (12 + 2 * diis_vectors) * amplitude_elements * 8
    # cs_ccsd_residual_vjp retains the complete tensor-level reverse-mode
    # graph until the reverse traversal finishes. Count every materialized
    # node value plus one equally sized gradient per node; transpose views are
    # excluded because they share storage. Two largest extents cover the
    # contribution/accumulation boundary while an old gradient remains live.
    lambda_ad_node_elements = (
        9 * nv**4
        + 7 * no**4
        + 94 * no * no * nv * nv
        + 20 * nv * nv
        + 18 * no * no
        + 24 * no * nv
    )
    lambda_ad_workspace = (
        2 * lambda_ad_node_elements
        + 2 * max(nv**4, no**4, no * no * nv * nv)
    ) * 8
    active_threads = max(1, min(_omp_max_threads(), no**3))
    lambda_triples_buffers = active_threads * 3 * nv**3 * 8
    lambda_triples_integrals = (
        no * nv**3 + no**3 * nv + no * no * nv * nv
    ) * 8

    est.by_category["A-CCSD(T) dense integral blocks"] = dense_blocks
    est.by_category["A-CCSD(T) Lambda/DIIS workspace"] = lambda_workspace
    est.by_category["A-CCSD(T) Lambda reverse-AD graph"] = (
        lambda_ad_workspace
    )
    est.by_category["A-CCSD(T) Lambda triples image buffers"] = (
        lambda_triples_buffers
    )
    # The original T/L amplitudes remain live while Python creates Fortran
    # call packs and pybind materializes its dense Eigen arguments.  The same
    # two call-boundary peers exist for f_ov.  For the three integral blocks,
    # Python's packs and pybind's casters coexist with a third copy in the C++
    # IntegralBlocks object; the original V dictionary is already part of
    # retained_integrals below.
    lambda_triples_amplitude_call_copies = (
        4 * amplitude_elements + 2 * no * nv
    ) * 8
    est.by_category["A-CCSD(T) Lambda amplitude/call copies"] = (
        lambda_triples_amplitude_call_copies
    )
    est.by_category["A-CCSD(T) Lambda integral/call copies"] = (
        3 * lambda_triples_integrals
    )

    if uses_df:
        _, _, df_construction, df_resident = _python_density_fitting_storage(
            n_aux,
            n,
        )
        b_ov = n_aux * no * nv * 8
        b_vv = n_aux * nv * nv * 8
        b_oo = n_aux * no * no * 8
        transformed_factors = b_ov + b_vv + b_oo
        # ``DensityFitting.mo_transform`` first builds a Q x AO x right-MO
        # half transform and then the returned Q x left-MO x right-MO array.
        # The three calls in ``_run_accsd_t_from_mos`` are sequential, but
        # their earlier return values remain live while each later call runs.
        df_transform_peak = max(
            df_resident + n_aux * n * nv * 8 + b_ov,
            df_resident + b_ov + n_aux * n * nv * 8 + b_vv,
            df_resident
            + b_ov
            + b_vv
            + n_aux * n * no * 8
            + b_oo,
        )
        retained_integrals = df_resident + transformed_factors + dense_blocks
        est.by_category["A-CCSD(T) DF construction overlap"] = (
            df_construction
        )
        est.by_category["A-CCSD(T) DF MO-transform peak"] = (
            df_transform_peak
        )
        est.by_category["A-CCSD(T) retained DF/factor tensors"] = (
            df_resident + transformed_factors
        )
        est.phase_peaks["A-CCSD(T) dense integral build"] = reference_result + max(
            df_construction,
            df_transform_peak,
            retained_integrals,
        )
    else:
        ao_eri = n**4 * 8
        retained_integrals = dense_blocks
        # The optimized five-operand AO-to-MO contractions retain the AO ERI
        # and can materialize two AO-sized einsum peers while the previously
        # completed block values in the dictionary remain live.
        exact_transform_peak = 3 * ao_eri
        est.by_category["A-CCSD(T) exact integral-transform peak"] = (
            exact_transform_peak
        )
        est.phase_peaks["A-CCSD(T) dense integral build"] = (
            reference_result + exact_transform_peak + dense_blocks
        )

    common_solver_workspace = sum(
        est.by_category[name]
        for name in (
            "CCSD T1/T2 amplitudes",
            "CCSD residual/DIIS amplitudes",
            "CCSD D1/D2 intermediates",
        )
    )
    # The Python V dictionary remains live while the bridge creates a complete
    # Fortran-order argument pack. The pybind dense-Eigen caster and the C++
    # IntegralBlocks constructor each own another complete six-block set.
    # Native compute_residuals additionally materializes Wabef.
    spatial_bridge_copies = 3 * dense_blocks
    spatial_wabef = nv**4 * 8
    est.by_category["A-CCSD(T) spatial solver bridge copies"] = (
        spatial_bridge_copies
    )
    est.by_category["A-CCSD(T) spatial solver Wabef scratch"] = spatial_wabef
    est.phase_peaks["A-CCSD(T) spatial solver bridge"] = (
        reference_result
        + retained_integrals
        + spatial_bridge_copies
        + common_solver_workspace
        + spatial_wabef
    )
    est.phase_peaks["A-CCSD(T) Lambda solve"] = (
        reference_result
        + retained_integrals
        + lambda_workspace
        + lambda_ad_workspace
    )
    est.phase_peaks["A-CCSD(T) Lambda triples"] = (
        reference_result
        + retained_integrals
        + 2 * amplitude_elements * 8
        + lambda_triples_amplitude_call_copies
        + 3 * lambda_triples_integrals
        + lambda_triples_buffers
    )
    est.category_details["A-CCSD(T) Lambda triples image buffers"] = (
        "separate dense Lambda route; triples_memory_mode/requested_memory_bytes "
        "do not select the standard-(T) bounded planner"
    )
    est._with_dims(
        n_basis=n,
        n_occ=no,
        n_vir=nv,
        n_threads=active_threads,
        n_frozen_core=n_frozen,
    )
    if uses_df:
        est.dims["n_aux"] = n_aux
    return est


def _uccsd_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    return _ccsd_estimate(molecule, basis, options, open_shell=True)


def _uccsd_t_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    return _ccsd_estimate(molecule, basis, options, open_shell=True, triples=True)


def _dlpno_active_counts(
    molecule: Molecule,
    n_basis: int,
    method_options,
    *,
    open_shell: bool = False,
) -> tuple[int, int, int, int, int]:
    """Return active DLPNO dimensions under the execution convention.

    Every molecular DLPNO executor resolves ``n_frozen=None`` through the
    shared published count-only convention.  The estimator must do the same:
    treating that sentinel as zero makes standalone estimates and runner
    admission describe a larger, different correlation problem.
    """
    if open_shell:
        n_elec = molecule.n_electrons()
        multiplicity = int(molecule.multiplicity)
        n_alpha = _clamped_occ((n_elec + multiplicity - 1) // 2, n_basis)
        n_beta = _clamped_occ(n_elec - n_alpha, n_basis)
        frozen = _frozen_core_count(
            method_options,
            n_beta,
            molecule=molecule,
            published_default=True,
        )
        n_vir_a = max(0, n_basis - n_alpha)
        n_vir_b = max(0, n_basis - n_beta)
        n_occ = max(1, n_alpha + n_beta - 2 * frozen)
        n_vir = max(1, n_vir_a + n_vir_b)
    else:
        n_occ_raw = _clamped_occ(molecule.n_electrons() // 2, n_basis)
        frozen = _frozen_core_count(
            method_options,
            max(0, n_occ_raw - 1),
            molecule=molecule,
            published_default=True,
        )
        n_occ = max(1, n_occ_raw - frozen)
        n_vir = max(1, n_basis - n_occ_raw)
    avg_pno = min(n_vir, max(8, (n_vir + 2) // 3))
    try:
        tcut_pno = float(getattr(method_options, "tcut_pno", 1.0e-7))
    except (TypeError, ValueError):
        tcut_pno = 1.0e-7
    residual_domain = str(
        getattr(method_options, "residual_domain", "extended") or "extended"
    ).strip().lower()
    if tcut_pno < _DLPNO_PROFILED_PNO_CUTOFF or residual_domain != "pair":
        # Below the tightest profiled cutoff, or when residuals expand beyond
        # the pair PNO basis, no measured truncation bound exists.  Charge the
        # full virtual dimension rather than extrapolating the S22 heuristic.
        avg_pno = n_vir
    avg_pao = min(n_basis, max(avg_pno * 2, (2 * n_vir + 2) // 3))
    return n_occ, n_vir, avg_pao, avg_pno, frozen


def _dlpno_pair_count(n_occ: int, *, open_shell: bool = False) -> int:
    if open_shell:
        return max(1, n_occ * n_occ)
    return max(1, n_occ * (n_occ + 1) // 2)


def _dlpno_strong_pair_count(
    n_occ: int,
    method_options,
    *,
    open_shell: bool = False,
) -> int:
    """Estimate the retained strong-pair list used by local CCSD.

    ``run_local_dlpno_ccsd`` first screens the full MP2 pair list, then keeps
    static PNO/DF state only for strong pairs.  The screened list is the
    linear-scaling object described by Riplinger and Neese, JCP 138, 034106
    (2013), rather than the full quadratic pair inventory.  Open-shell local
    CCSD has no matching production-profile inventory yet, so it keeps the
    conservative full-list estimate.
    """
    full_pairs = _dlpno_pair_count(n_occ, open_shell=open_shell)
    if open_shell:
        return full_pairs
    try:
        tcut_pairs = float(getattr(method_options, "tcut_pairs", 1.0e-4))
    except (TypeError, ValueError):
        tcut_pairs = 1.0e-4
    residual_domain = str(
        getattr(method_options, "residual_domain", "extended") or "extended"
    ).strip().lower()
    use_cpp_kernel = bool(getattr(method_options, "use_cpp_kernel", True))
    if (
        tcut_pairs < _DLPNO_PROFILED_PAIR_CUTOFF
        or residual_domain != "pair"
        or not use_cpp_kernel
    ):
        return full_pairs
    return min(full_pairs, _DLPNO_STRONG_PAIRS_PER_OCC * n_occ)


def _dlpno_triples_virtual_tile(n_vir: int) -> int:
    """Mirror the native DLPNO spatial-triples dense/tiled decision."""
    requested = 0
    raw = os.environ.get("VIBEQC_TRIPLES_TILE_SIZE", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 0
    if requested <= 0 and n_vir**3 * 2 * 8 > (
        _DLPNO_TRIPLES_TILED_THRESHOLD_BYTES
    ):
        requested = 64
    if requested > 0 and n_vir > requested:
        return max(1, requested)
    return n_vir


def _dlpno_mp2_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    open_shell: bool = False,
) -> MemoryEstimate:
    ref_options = _post_reference_options(options)
    method_options = _post_method_options(
        options,
        "dlpno_options" if not open_shell else "dlpno_ump2_options",
    )
    if method_options is None and open_shell:
        method_options = _post_method_options(options, "dlpno_options")
    if method_options is None:
        if open_shell:
            from .dlpno.ump2 import DLPNOUMP2Options

            method_options = DLPNOUMP2Options()
        else:
            from .dlpno.mp2 import DLPNOMP2Options

            method_options = DLPNOMP2Options()
    est = (
        _uhf_estimate(molecule, basis, ref_options)
        if open_shell
        else _rhf_estimate(molecule, basis, ref_options)
    )
    reference_categories = set(est.by_category)
    n = _n_basis(molecule, basis)
    n_aux = _aux_basis_size(molecule, n, method_options)
    n_occ, n_vir, avg_pao, avg_pno, n_frozen = _dlpno_active_counts(
        molecule,
        n,
        method_options,
        open_shell=open_shell,
    )
    n_pairs = _dlpno_pair_count(n_occ, open_shell=open_shell)
    pair_domains = (
        n_pairs * (n * avg_pao + avg_pao * avg_pno) * 8
    )
    pair_lists = n_pairs * 1024
    pair_amplitudes = (
        n_pairs * 4 * avg_pno * avg_pno * 8
    )
    # B_i/B_j, K/T/D eigensolver inputs, and their peer copies are live while
    # the current pair is added to the persistent pair dictionary.
    pair_build_scratch = (
        2 * n_aux * avg_pao
        + n * avg_pao
        + 6 * avg_pao * avg_pao
    ) * 8
    est.by_category["DLPNO PAO coefficients/domains"] = pair_domains
    est.category_details["DLPNO PAO coefficients/domains"] = (
        _DLPNO_COMPOSITION_BOUND_DETAIL
    )
    est.by_category["DLPNO pair lists"] = pair_lists
    est.by_category["DLPNO-MP2 PNO pair amplitudes"] = pair_amplitudes
    est.by_category["DLPNO-MP2 pair-build scratch"] = pair_build_scratch

    reference_result = _reference_result_bytes(est)
    runtime_floor = int(
        est.by_category.get("Python runtime + NumPy overhead", 0)
    )
    live_reference = runtime_floor + reference_result
    persistent_pair_state = pair_domains + pair_lists + pair_amplitudes

    # The closed-shell coupled-LMP2 solver caches every ordered overlap link
    # S_pq = V_p.T @ S @ V_q reached by the dense occupied Fock couplings.
    # With the full retained pair graph this is O(n_occ^3) link matrices.  No
    # threshold supplies a rigorous upper bound below the full virtual rank,
    # so admission uses n_vir rather than the empirical average PNO size.
    # The native binding also coexists with Python Fortran-order packs, Eigen
    # input copies, the C++ T_new vector, and its Python return copies.
    iteration_pno_bound = n_vir
    iteration_matrix = iteration_pno_bound * iteration_pno_bound * 8
    if open_shell:
        lmp2_link_count = 0
        lmp2_link_cache = 0
        lmp2_iteration_copies = (
            2 * n_pairs * iteration_matrix
            + (8 * iteration_pno_bound * iteration_pno_bound
               + 4 * n_vir * iteration_pno_bound) * 8
        )
        est.by_category[
            "DLPNO-UMP2 coupled-iteration workspace"
        ] = lmp2_iteration_copies
    else:
        coupled_occupied = str(
            getattr(method_options, "localise", "boys") or "boys"
        ).strip().lower() != "none"
        lmp2_link_count = (
            n_pairs * 2 * max(0, n_occ - 1)
            if coupled_occupied
            else 0
        )
        lmp2_link_cache = lmp2_link_count * iteration_matrix
        packed_pair_inputs = n_pairs * (
            2 * iteration_pno_bound * iteration_pno_bound
            + n * iteration_pno_bound
            + iteration_pno_bound
        ) * 8
        returned_amplitudes = 2 * n_pairs * iteration_matrix
        copied_global_matrices = 2 * (n_occ * n_occ + n * n) * 8
        iteration_scratch = (
            8 * iteration_pno_bound * iteration_pno_bound
            + 2 * n * iteration_pno_bound
        ) * 8
        lmp2_iteration_copies = (
            2 * packed_pair_inputs
            + returned_amplitudes
            + copied_global_matrices
            + iteration_scratch
        )
        est.by_category[
            "DLPNO-MP2 LMP2 overlap-link cache"
        ] = lmp2_link_cache
        est.by_category[
            "DLPNO-MP2 native iteration bridge/copies"
        ] = lmp2_iteration_copies
        est.category_details[
            "DLPNO-MP2 LMP2 overlap-link cache"
        ] = (
            f"ordered-link bound={lmp2_link_count}, "
            f"PNO-rank bound={iteration_pno_bound}"
        )

    # The closed-shell local-DF route deliberately does not construct a
    # global DensityFitting object.  It caches raw three-centre tensors and
    # Coulomb metrics by unique fit-domain atom tuple instead.  A full fit
    # buffer yields exactly one global-size cache entry; otherwise use the
    # fail-closed upper bound of one distinct, full-auxiliary domain per
    # retained pair.  PAO size cannot upper-bound fit size because fit_buffer
    # can extend a compact PAO atom set across most or all auxiliary centres.
    local_df = (
        not open_shell
        and bool(getattr(method_options, "local_df", False))
    )
    if local_df:
        try:
            fit_buffer = float(getattr(method_options, "fit_buffer", 4.0))
        except (TypeError, ValueError):
            fit_buffer = 4.0
        if fit_buffer >= 1.0e8:
            n_fit_domains = 1
        else:
            n_fit_domains = n_pairs
        fit_dim_bound = n_aux

        local_three_index = fit_dim_bound * n * n * 8
        local_metric = fit_dim_bound * fit_dim_bound * 8
        cached_local_df = n_fit_domains * (
            local_three_index + local_metric
        )
        # compute_3c_eri/compute_2c_eri construction plus the two M[P,a]
        # transforms and dense metric solve for the current pair.  The cache
        # itself is accounted separately because earlier unique domains stay
        # resident until pair setup finishes.
        local_df_scratch = (
            local_three_index
            + 3 * local_metric
            + 4 * fit_dim_bound * avg_pao * 8
            + 4 * avg_pao * avg_pao * 8
        )
        est.by_category[
            "DLPNO-MP2 local-DF cached domain integrals"
        ] = cached_local_df
        est.by_category[
            "DLPNO-MP2 local-DF construction/solve scratch"
        ] = local_df_scratch
        phases = {
            "DLPNO-MP2 local-DF pair setup": (
                live_reference
                + cached_local_df
                + persistent_pair_state
                + max(pair_build_scratch, local_df_scratch)
            ),
            "DLPNO-MP2 coupled-LMP2 iteration": (
                live_reference
                + persistent_pair_state
                + lmp2_link_cache
                + lmp2_iteration_copies
            ),
        }
    else:
        # DensityFitting retains raw and metric-orthogonalised AO three-index
        # tensors plus V, the scipy Cholesky storage, and the explicit lower
        # triangle.  Construction reaches additional solve/eigensolver peers.
        # This is materially larger than the single tensor charged by the old
        # estimator and is live beside every transformed factor and pair.
        (
            three_index,
            _,
            df_construction,
            df_resident,
        ) = _python_density_fitting_storage(n_aux, n)
        est.by_category[
            "DLPNO-MP2 DF construction overlap"
        ] = df_construction
        est.by_category[
            "DLPNO-MP2 retained DF substrate"
        ] = df_resident

        if open_shell:
            n_elec = molecule.n_electrons()
            multiplicity = int(molecule.multiplicity)
            n_alpha = _clamped_occ(
                (n_elec + multiplicity - 1) // 2, n
            )
            n_beta = _clamped_occ(n_elec - n_alpha, n)
            n_occ_a = max(0, n_alpha - n_frozen)
            n_occ_b = max(0, n_beta - n_frozen)
            n_vir_a = max(0, n - n_alpha)
            n_vir_b = max(0, n - n_beta)
            transformed_factors = n_aux * (
                n_occ_a * n_vir_a + n_occ_b * n_vir_b
            ) * 8
            transform_scratch = (
                n_aux * n * max(n_vir_a, n_vir_b) * 8
            )
        else:
            # run_dlpno_mp2 asks for B[P,i,mu], so the second transform is
            # the identity and its n_aux*n_basis^2 temporary is explicit.
            transformed_factors = n_aux * n_occ * n * 8
            transform_scratch = three_index
        est.by_category[
            "DLPNO-MP2 transformed DF factors"
        ] = transformed_factors
        est.by_category[
            "DLPNO-MP2 MO-transform scratch"
        ] = transform_scratch
        phases = {
            "DLPNO-MP2 DF construction": (
                live_reference + df_construction
            ),
            "DLPNO-MP2 DF transformation": (
                live_reference
                + df_resident
                + transformed_factors
                + transform_scratch
            ),
            "DLPNO-MP2 pair setup": (
                live_reference
                + df_resident
                + transformed_factors
                + persistent_pair_state
                + pair_build_scratch
            ),
        }
        if open_shell:
            phases["DLPNO-UMP2 coupled iteration"] = (
                live_reference
                + df_resident
                + transformed_factors
                + persistent_pair_state
                + lmp2_iteration_copies
            )
        else:
            phases["DLPNO-MP2 coupled-LMP2 iteration"] = (
                live_reference
                + df_resident
                + persistent_pair_state
                + lmp2_link_cache
                + lmp2_iteration_copies
            )

    _set_posthf_phase_peaks(est, reference_categories, phases)
    est._with_dims(
        n_basis=n,
        n_occ=n_occ,
        n_vir=n_vir,
        n_frozen_core=n_frozen,
        n_aux=n_aux,
        n_pairs=n_pairs,
        avg_pno=avg_pno,
        lmp2_link_count_bound=lmp2_link_count,
        lmp2_pno_rank_bound=iteration_pno_bound,
    )
    if local_df:
        est.dims["local_fit_dimension_bound"] = fit_dim_bound
        est.dims["n_local_fit_domains_estimate"] = n_fit_domains
    return est


def _dlpno_ccsd_pilot_estimate(
    molecule: Molecule,
    basis: BasisSet,
    ref_options,
    method_options,
    *,
    open_shell: bool,
    triples: bool,
) -> MemoryEstimate:
    """Estimate the dense Python DLPNO-CCSD correctness pilots.

    The pilot classes share a user-facing method label with the local
    solvers, but deliberately do not share their memory scaling.  Both build
    and retain a full antisymmetrised spin-orbital ERI tensor.  Its NumPy
    constructor also materialises four mesh grids, eight spin/spatial index
    grids, advanced-index results, and masks at the same time.  Treating a
    pilot as a streamed one-pair solver can therefore under-admit it by orders
    of magnitude when callers deliberately raise ``max_nbf``.
    """
    est = (
        _uhf_estimate(molecule, basis, ref_options)
        if open_shell
        else _rhf_estimate(molecule, basis, ref_options)
    )
    reference_categories = set(est.by_category)
    n = _n_basis(molecule, basis)
    n_aux = _aux_basis_size(molecule, n, method_options)
    n_occ, n_vir, avg_pao, avg_pno, n_frozen = _dlpno_active_counts(
        molecule,
        n,
        method_options,
        open_shell=open_shell,
    )
    del avg_pao

    # Each UHF spin channel drops the same frozen spatial core and therefore
    # retains n-n_frozen spatial MOs.  On RHF, n_occ/n_vir are already spatial.
    n_mo = n - n_frozen if open_shell else n_occ + n_vir
    n_spin = 2 * n_mo
    n_pairs = (
        n_occ * (n_occ - 1) // 2
        if open_shell
        else n_occ * (n_occ + 1) // 2
    )

    # This dense Python pilot constructs the shared DensityFitting object.
    _, _, df_construction, df_resident = _python_density_fitting_storage(
        n_aux,
        n,
    )
    est.by_category["DLPNO pilot DF construction overlap"] = df_construction
    est.by_category["DLPNO pilot retained DF substrate"] = df_resident

    # Closed-shell keeps B_eng, three contiguous occupied/virtual slices, and
    # B_half for pair-PNO construction. Open-shell keeps independent B_a/B_b.
    transformed_factors = (
        2 * n_aux * n_mo * n_mo * 8
        if open_shell
        else n_aux
        * (2 * n_mo * n_mo + n_occ * n)
        * 8
    )
    est.by_category["DLPNO pilot transformed DF factors"] = transformed_factors

    spin_eri = n_spin**4 * 8
    spatial_eri_build = (4 if open_shell else 1) * n_mo**4 * 8
    # _spin_orbital_eri[_uhf] has twelve int64 n_spin^4 index grids live,
    # plus the destination, masks, advanced-index values, and arithmetic
    # temporaries. Seventeen peer spin tensors beyond the retained result is a
    # conservative inventory of the source as written, not an empirical fit.
    spin_build_temporaries = 17 * spin_eri + spatial_eri_build
    est.by_category["DLPNO pilot retained spin ERI"] = spin_eri
    est.by_category["DLPNO pilot spin-ERI build temporaries"] = (
        spin_build_temporaries
    )
    est.category_details["DLPNO pilot spin-ERI build temporaries"] = (
        "dense NumPy mesh/index/mask construction; not the local pair kernel"
    )

    # Pair spaces remain material for small pilots even though the N^4 tensor
    # dominates once max_nbf is raised.  Bound each pair by the estimator's
    # retained PNO dimension and include its coordinates, amplitudes, and
    # eigensolver peer matrices.
    pair_state = n_pairs * (
        n_vir * avg_pno + 4 * avg_pno * avg_pno
    ) * 8
    est.by_category["DLPNO pilot PNO pair state"] = pair_state

    no = n_occ if open_shell else 2 * n_occ
    nv = n_vir if open_shell else 2 * n_vir
    singles = no * nv
    doubles = no * no * nv * nv
    try:
        diis_size = max(0, int(getattr(method_options, "diis_size", 6)))
    except (TypeError, ValueError):
        diis_size = 6
    diis_history = 2 * diis_size * (singles + doubles) * 8
    est.by_category["DLPNO pilot dense DIIS history"] = diis_history

    if open_shell:
        spatial_blocks = 0
    else:
        # Exact extents returned by dlpno._ccsd_cs._blocks.
        spatial_blocks = (
            (n_occ * n_vir) ** 2
            + n_occ * n_vir**3
            + n_occ**3 * n_vir
            + n_occ**4
            + n_vir**4
            + n_occ**2 * n_vir**2
        ) * 8
    est.by_category["DLPNO pilot dense spatial integral blocks"] = spatial_blocks

    # The open pilot evaluates the full spin-orbital residual; the closed
    # pilot uses a smaller spatial residual but expands full spin amplitudes
    # for (T). Twelve full tensor peers safely cover the retained/updated
    # amplitudes and simultaneous residual intermediates on either route.
    dense_solver = 12 * spin_eri + 4 * (singles + doubles) * 8
    est.by_category["DLPNO pilot dense residual/amplitude workspace"] = (
        dense_solver
    )

    compute_triples = triples or bool(
        getattr(method_options, "compute_triples", False)
    )
    triples_peak = 0
    triples_threads = _omp_max_threads()
    triples_tile = 0
    localized_triples_rebuild = 0
    canonical_triples_factors = 0
    if compute_triples:
        # Both pilots pass spin-orbital tensors to the UCCSD(T)-shaped native
        # correction, even for an RHF reference.
        (
            triples_mode,
            triples_tile,
            triples_threads,
            triples_peak,
            _triples_retained,
            _triples_work,
            _triples_disk,
        ) = _ccsd_triples_workspace(
            no,
            nv,
            method_options,
            open_shell=True,
            uses_df=False,
            n_aux=0,
        )
        est.by_category["DLPNO pilot dense spin-orbital triples"] = triples_peak
        est.category_details["DLPNO pilot dense spin-orbital triples"] = (
            f"mode={triples_mode}, virtual tile={triples_tile}/{nv}, "
            f"threads={triples_threads}"
        )
        localise = str(
            getattr(method_options, "localise", "none") or "none"
        ).strip().lower()
        if localise != "none":
            # Localized pilots rotate occupieds back to the canonical gauge
            # for (T), rebuild B and a second full spin ERI, and do so while
            # the original ERI, pair dictionaries, and DIIS histories remain
            # reachable in the outer solver frame.
            canonical_triples_factors = (
                (2 if open_shell else 1) * n_aux * n_mo * n_mo * 8
            )
            localized_triples_rebuild = (
                canonical_triples_factors
                + spin_eri
                + spin_build_temporaries
            )
            est.by_category[
                "DLPNO pilot localized-(T) spin-ERI rebuild"
            ] = localized_triples_rebuild
            est.category_details[
                "DLPNO pilot localized-(T) spin-ERI rebuild"
            ] = (
                "second canonical-gauge B/ERI construction while the "
                "localized ERI and solver histories remain live"
            )

    reference_result = _reference_result_bytes(est)
    runtime = int(est.by_category.get("Python runtime + NumPy overhead", 0))
    live_reference = runtime + reference_result
    retained_substrate = df_resident + transformed_factors
    phases = {
        "DLPNO pilot DF construction": live_reference + df_construction,
        "DLPNO pilot dense spin-ERI construction": (
            live_reference
            + retained_substrate
            + spin_eri
            + spin_build_temporaries
        ),
        "DLPNO pilot dense CCSD solve": (
            live_reference
            + retained_substrate
            + spin_eri
            + spatial_blocks
            + pair_state
            + diis_history
            + dense_solver
        ),
    }
    if triples_peak:
        if localized_triples_rebuild:
            phases["DLPNO pilot localized-(T) ERI rebuild"] = (
                live_reference
                + retained_substrate
                + spin_eri
                + spatial_blocks
                + pair_state
                + diis_history
                + localized_triples_rebuild
            )
        phases["DLPNO pilot dense (T) correction"] = (
            live_reference
            + retained_substrate
            + spatial_blocks
            + pair_state
            + diis_history
            + (
                spin_eri + canonical_triples_factors
                if localized_triples_rebuild
                else 0
            )
            + triples_peak
        )
    _set_posthf_phase_peaks(est, reference_categories, phases)

    est._with_dims(
        n_basis=n,
        n_occ=n_occ,
        n_vir=n_vir,
        n_frozen_core=n_frozen,
        n_aux=n_aux,
        n_threads=triples_threads,
        n_pairs=n_pairs,
        n_strong_pairs_estimate=n_pairs,
        avg_pno=avg_pno,
    )
    if triples_peak:
        est.dims["triples_tile_size"] = triples_tile
    return est


def _dlpno_ccsd_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
    *,
    open_shell: bool = False,
    triples: bool = False,
) -> MemoryEstimate:
    ref_options = _post_reference_options(options)
    method_options = _post_method_options(options, "dlpno_ccsd_options")
    if method_options is None:
        if open_shell:
            from .dlpno.uccsd_local_solver import LocalUCCSDOptions

            method_options = LocalUCCSDOptions()
        else:
            from .dlpno.ccsd_local_solver import LocalCCSDOptions

            method_options = LocalCCSDOptions()
        method_options.compute_triples = bool(triples)
    from .dlpno.ccsd import DLPNOCCSDPilotOptions
    from .dlpno.uccsd import DLPNOUCCSDPilotOptions

    if isinstance(
        method_options,
        (DLPNOCCSDPilotOptions, DLPNOUCCSDPilotOptions),
    ):
        return _dlpno_ccsd_pilot_estimate(
            molecule,
            basis,
            ref_options,
            method_options,
            open_shell=open_shell,
            triples=triples,
        )
    est = (
        _uhf_estimate(molecule, basis, ref_options)
        if open_shell
        else _rhf_estimate(molecule, basis, ref_options)
    )
    reference_categories = set(est.by_category)
    n = _n_basis(molecule, basis)
    n_aux = _aux_basis_size(molecule, n, method_options)
    n_occ, n_vir, avg_pao, avg_pno, n_frozen = _dlpno_active_counts(
        molecule,
        n,
        method_options,
        open_shell=open_shell,
    )
    n_pairs = _dlpno_pair_count(n_occ, open_shell=open_shell)
    n_strong_pairs = _dlpno_strong_pair_count(
        n_occ,
        method_options,
        open_shell=open_shell,
    )
    runtime_threads = _omp_max_threads()
    native_worker_state = runtime_threads * _DLPNO_NATIVE_WORKER_BYTES
    est.by_category["DLPNO native worker stacks/scratch"] = native_worker_state

    # The production local solvers construct the shared Python
    # DensityFitting object before transforming their occupied/virtual blocks.
    _, _, df_construction, df_resident = _python_density_fitting_storage(
        n_aux,
        n,
    )
    est.by_category["DLPNO DF construction overlap"] = df_construction
    est.by_category["DLPNO DF auxiliary integrals"] = df_resident

    # One PAO/PNO domain is assembled at a time.  U_ij and the static local
    # DF blocks below persist for every retained strong pair.
    pair_setup = (
        n * avg_pao
        + avg_pao * avg_pno
        + 2 * n_aux * avg_pao
        + 4 * avg_pao * avg_pao
    ) * 8
    est.by_category["DLPNO PAO coefficients/domains"] = pair_setup
    est.category_details["DLPNO PAO coefficients/domains"] = (
        _DLPNO_COMPOSITION_BOUND_DETAIL
    )
    est.by_category["DLPNO pair lists"] = n_pairs * 1024

    # ``DensityFitting.mo_transform`` creates a temporary
    # (n_aux,n_basis,n_right) tensor while its output is formed.  Earlier
    # outputs remain live across the sequential calls.  The unrestricted
    # solver additionally retains all six spin-specific outputs after packing
    # them into the three combined spin-orbital blocks, so those are part of
    # its steady-state solver footprint rather than transient scratch.
    if open_shell:
        n_elec = molecule.n_electrons()
        multiplicity = int(molecule.multiplicity)
        n_alpha = _clamped_occ((n_elec + multiplicity - 1) // 2, n)
        n_beta = _clamped_occ(n_elec - n_alpha, n)
        n_occ_a = max(0, n_alpha - n_frozen)
        n_occ_b = max(0, n_beta - n_frozen)
        n_vir_a = max(0, n - n_alpha)
        n_vir_b = max(0, n - n_beta)
        transform_shapes = (
            (n_occ_a, n_occ_a),
            (n_occ_b, n_occ_b),
            (n_occ_a, n_vir_a),
            (n_occ_b, n_vir_b),
            (n_vir_a, n_vir_a),
            (n_vir_b, n_vir_b),
        )
        retained_spin_factors = 0
        df_transform_peak = 0
        for n_left, n_right in transform_shapes:
            output = n_aux * n_left * n_right * 8
            scratch = n_aux * n * n_right * 8
            df_transform_peak = max(
                df_transform_peak,
                retained_spin_factors + scratch + output,
            )
            retained_spin_factors += output
        combined_factors = n_aux * (
            n_occ * n_occ + n_occ * n_vir + n_vir * n_vir
        ) * 8
        global_factors = retained_spin_factors + combined_factors
        df_transform_peak = max(df_transform_peak, global_factors)
    else:
        transform_shapes = (
            (n_occ, n_vir),
            (n_vir, n_vir),
            (n_occ, n_occ),
        )
        global_factors = 0
        df_transform_peak = 0
        for n_left, n_right in transform_shapes:
            output = n_aux * n_left * n_right * 8
            scratch = n_aux * n * n_right * 8
            df_transform_peak = max(
                df_transform_peak,
                global_factors + scratch + output,
            )
            global_factors += output
    est.by_category["DLPNO global MO DF factors"] = global_factors
    est.by_category["DLPNO MO DF transformation peak"] = df_transform_peak

    # pdata retains B_ov_L, B_oo_L, and B_vv_L for every strong pair.  The
    # lower-order term covers U_ij, the pair amplitudes, K/Fock blocks, and
    # conservative peer copies made while each dictionary entry is built.
    pair_df_state = n_strong_pairs * n_aux * (
        n_occ * avg_pno + n_occ * n_occ + avg_pno * avg_pno
    ) * 8
    pair_amplitudes = n_strong_pairs * 4 * avg_pno * avg_pno * 8
    pair_maps = n_strong_pairs * (
        n_vir * avg_pno
        + 3 * n_occ * n_occ
        + 3 * n_occ * avg_pno
    ) * 8
    est.by_category["DLPNO-CCSD auxiliary pair workspace"] = pair_df_state
    est.by_category["DLPNO-CCSD T1/T2 pair amplitudes"] = pair_amplitudes
    est.by_category["DLPNO-CCSD pair-domain maps/Fock blocks"] = pair_maps

    try:
        diis_size = max(0, int(getattr(method_options, "diis_size", 6)))
    except (TypeError, ValueError):
        diis_size = 6
    pair_diis = (
        (2 * diis_size + 4)
        * (n_strong_pairs * avg_pno * avg_pno + n_occ * avg_pno)
        * 8
    )
    est.by_category["DLPNO-CCSD pair DIIS/copy history"] = pair_diis

    # The Python loop calls the native residual for one target pair and drops
    # its R1/R2/integral blocks before advancing.  Charge the larger of the
    # historical one-pair polynomial and an explicit local-block inventory;
    # multiplying either by the pair count is the N^5.7 defect fixed here.
    residual_heuristic = (
        12 * avg_pno**4 + 8 * n_occ * avg_pno * avg_pno
    ) * 8
    residual_blocks = (
        8 * n_occ * n_occ * avg_pno * avg_pno
        + 2 * n_occ**4
        + 2 * n_occ**3 * avg_pno
        + 2 * n_occ * avg_pno**3
    ) * 8
    pair_residual = max(residual_heuristic, residual_blocks)
    residual_domain = str(
        getattr(method_options, "residual_domain", "extended") or "extended"
    ).strip().lower()
    use_cpp_kernel = bool(getattr(method_options, "use_cpp_kernel", True))
    if open_shell:
        # LocalUCCSDOptions carries neither residual_domain nor
        # use_cpp_kernel, and the open-shell local solver is untouched by
        # #689: it still calls dlpno_uccsd_pair_residual once per occupied
        # and once per pair with persistent per-pair blocks, so it keeps the
        # conservative per-pair inventory.
        pair_residual *= n_strong_pairs
        residual_detail = (
            "persistent per-pair integral blocks on non-targeted route"
        )
    elif residual_domain == "pair":
        residual_is_streamed = use_cpp_kernel
        if not residual_is_streamed:
            # The NumPy pair route retains its integral blocks in pdata for
            # every strong pair.  It does not have the compiled target-pair
            # lifetime that permits a single streamed residual workspace.
            pair_residual *= n_strong_pairs
        residual_detail = (
            "one streamed target-pair workspace; not multiplied by pair count"
            if residual_is_streamed
            else "persistent per-pair integral blocks on non-targeted route"
        )
    else:
        # Extended/full domains (#689): the residual is contracted once per
        # distinct (coupling set, extended atom domain) group in a basis of
        # n_ext <= n_vir virtuals, and every pair in the group reads its own
        # block.  A compact molecule has one group at n_ext = n_vir, which is
        # the bound charged here: the group's persistent DF factors plus the
        # compiled kernel's transient integral blocks and intermediates (the
        # (ae|bf) ladder is held in core below the kernel's 2 GiB budget and
        # regenerated in 2048-row tiles above it).  Extended systems hold
        # more groups of smaller domains; their per-group factors are not
        # multiplied out here.
        n_ext = n_vir
        group_factors = n_aux * (
            n_occ * n_ext + n_ext * n_ext + n_occ * n_occ
        ) * 8
        vvvv_bytes = n_ext**4 * 8
        ladder = (
            2 * vvvv_bytes
            if vvvv_bytes <= _CCSD_VVVV_INCORE_BUDGET_BYTES
            else 2 * 2048 * n_ext * n_ext * 8
        )
        two_index_blocks = (
            10 * n_occ * n_occ * n_ext * n_ext
            + 2 * n_occ * n_ext**3
            + 2 * n_occ**3 * n_ext
            + 2 * n_occ**4
        ) * 8
        if use_cpp_kernel:
            pair_residual = group_factors + ladder + two_index_blocks
            residual_detail = (
                "one streamed extended-domain group workspace at "
                "n_ext = n_vir (compact-molecule bound); extended systems "
                "hold more groups of smaller domains"
            )
        else:
            # The NumPy transcription retains dense chemist-notation blocks
            # per group, (ab|cd) included.
            pair_residual = (
                group_factors + vvvv_bytes + ladder + two_index_blocks
            )
            residual_detail = (
                "persistent dense extended-domain integral blocks per group "
                "(NumPy fallback), one group at n_ext = n_vir charged"
            )
    est.by_category["DLPNO-CCSD pair residual/intermediates"] = pair_residual
    est.category_details["DLPNO-CCSD pair residual/intermediates"] = (
        residual_detail
    )
    est.category_details["DLPNO-CCSD auxiliary pair workspace"] = (
        f"retained strong-pair estimate={n_strong_pairs}/{n_pairs}"
    )

    compute_triples = triples or bool(
        getattr(method_options, "compute_triples", False)
    )
    triples_peak = 0
    triples_threads = runtime_threads
    triples_tile = 0
    if compute_triples:
        n_triples = max(1, n_occ * (n_occ + 1) * (n_occ + 2) // 6)
        triples_tile = _dlpno_triples_virtual_tile(n_vir)

        # The native contraction keeps only W/Wd for one occupied triple per
        # OpenMP worker.  Large virtual spaces tile the leading virtual index.
        triples_buffers = (
            2 * triples_threads * triples_tile * n_vir * n_vir * 8
        )
        triples_work = n_triples * 3 * 8
        est.by_category["DLPNO triples T_ijk^abc workspace"] = triples_buffers
        est.by_category["DLPNO triples occupied work list"] = triples_work

        # The default (T1), tcut_tno=0 compact-molecule route expands local
        # amplitudes once, rotates them once, then calls the native spatial
        # contraction.  These retained arrays and the C++ argument copies are
        # peers of the streamed worker buffers, not one copy per triple.
        triples_python_amplitudes = (
            2 * n_occ * n_occ * n_vir * n_vir
            + 2 * n_occ * n_vir
        ) * 8
        triples_factors = n_aux * (
            n_occ * n_vir + n_occ * n_occ + n_vir * n_vir
        ) * 8
        triples_call_copies = (
            n_occ * n_occ * n_vir * n_vir
            + n_occ * n_vir
            + n_aux
            * (n_occ * n_vir + n_occ * n_occ + n_vir * n_vir)
            + n_occ
            + n_vir
        ) * 8
        triples_integral_blocks = (
            2 * n_occ * n_occ * n_vir * n_vir
            + n_occ**4
            + n_occ**3 * n_vir
            + n_occ * n_vir**3
        ) * 8
        vvvv_bytes = n_vir**4 * 8
        if vvvv_bytes <= _CCSD_VVVV_INCORE_BUDGET_BYTES:
            triples_integral_blocks += vvvv_bytes
        est.by_category["DLPNO triples retained amplitudes/DF factors"] = (
            triples_python_amplitudes + triples_factors
        )
        est.by_category["DLPNO triples native argument copies"] = (
            triples_call_copies
        )
        est.by_category["DLPNO triples integral blocks"] = (
            triples_integral_blocks
        )
        est.category_details["DLPNO triples T_ijk^abc workspace"] = (
            f"two streamed buffers per worker; virtual tile="
            f"{triples_tile}/{n_vir}, threads={triples_threads}"
        )
        triples_peak = (
            triples_buffers
            + triples_work
            + triples_python_amplitudes
            + triples_factors
            + triples_call_copies
            + triples_integral_blocks
        )

    reference_result = _reference_result_bytes(est)
    runtime_floor = int(
        est.by_category.get("Python runtime + NumPy overhead", 0)
    )
    live_runtime = runtime_floor + native_worker_state
    persistent_pair_state = (
        df_resident
        + global_factors
        + est.by_category["DLPNO pair lists"]
        + pair_df_state
        + pair_amplitudes
        + pair_maps
        + pair_diis
    )
    phases = {
        "DLPNO DF construction": (
            live_runtime + reference_result + df_construction
        ),
        "DLPNO MO DF transformation": (
            live_runtime
            + reference_result
            + df_resident
            + df_transform_peak
        ),
        "DLPNO pair setup/CCSD solve": (
            live_runtime
            + reference_result
            + persistent_pair_state
            + max(pair_setup, pair_residual)
        ),
    }
    if triples_peak:
        phases["DLPNO (T) correction"] = (
            live_runtime
            + reference_result
            + persistent_pair_state
            + triples_peak
        )
    _set_posthf_phase_peaks(est, reference_categories, phases)

    est._with_dims(
        n_basis=n,
        n_occ=n_occ,
        n_vir=n_vir,
        n_frozen_core=n_frozen,
        n_aux=n_aux,
        n_threads=runtime_threads,
        n_pairs=n_pairs,
        n_strong_pairs_estimate=n_strong_pairs,
        avg_pno=avg_pno,
    )
    if triples_tile:
        est.dims["triples_tile_size"] = triples_tile
    return est


def _dlpno_mp2_closed_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    return _dlpno_mp2_estimate(molecule, basis, options)


def _dlpno_ump2_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    return _dlpno_mp2_estimate(molecule, basis, options, open_shell=True)


def _dlpno_ccsd_closed_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    return _dlpno_ccsd_estimate(molecule, basis, options)


def _dlpno_ccsd_t_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    return _dlpno_ccsd_estimate(molecule, basis, options, triples=True)


def _dlpno_uccsd_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    return _dlpno_ccsd_estimate(molecule, basis, options, open_shell=True)


def _dlpno_uccsd_t_estimate(
    molecule: Molecule,
    basis: BasisSet,
    options,
) -> MemoryEstimate:
    return _dlpno_ccsd_estimate(
        molecule,
        basis,
        options,
        open_shell=True,
        triples=True,
    )


def _wavefunction_estimate(molecule, basis, options=None):
    """Memory estimate for determinant CI-like wavefunction methods."""
    n = _n_basis(molecule, basis)
    est = _rhf_estimate(molecule, basis, _post_reference_options(options))
    n_elec = max(0, molecule.n_electrons())
    try:
        n_det = max(1, comb(max(0, 2 * n), min(n_elec, max(0, 2 * n))))
    except ValueError:
        n_det = 1
    est.by_category["CI determinant vector"] = 3 * n_det * 8
    est.by_category["CI sigma/residual workspace"] = max(n_det * 16, n**4 * 8)
    est.by_category["MO integral transform workspace"] = n * n * n * n * 8
    return est


def _selected_ci_excitations_per_determinant(
    n_orb: int,
    n_elec: int,
    ms2: int,
    spin_restricted: bool,
) -> int:
    """Distinct singles + doubles one determinant can generate.

    Mirrors the enumeration in
    ``SelectedCISolver._select_candidates_{restricted,unrestricted}`` exactly,
    so the estimate tracks the code rather than a textbook count.  The
    restricted kernel walks a closed-shell (seniority-zero) determinant,
    which couples only through single pair moves ``i^2 -> a^2`` over
    *spatial* orbitals (``n_occ * n_vir`` candidates; a two-pair move has
    zero coupling and is not enumerated, GitLab #639); the unrestricted
    kernel walks ``(alpha_occ, beta_occ)`` and adds the same-spin and
    opposite-spin blocks separately.
    """
    n_orb = max(0, int(n_orb))
    n_elec = max(0, int(n_elec))
    if spin_restricted:
        n_occ = min(n_elec // 2, n_orb)
        n_vir = max(0, n_orb - n_occ)
        return n_occ * n_vir

    n_alpha = min(max(0, (n_elec + ms2) // 2), n_orb)
    n_beta = min(max(0, n_elec - n_alpha), n_orb)
    vir_alpha = max(0, n_orb - n_alpha)
    vir_beta = max(0, n_orb - n_beta)
    singles = n_alpha * vir_alpha + n_beta * vir_beta
    doubles = (
        comb(n_alpha, 2) * comb(vir_alpha, 2)
        + comb(n_beta, 2) * comb(vir_beta, 2)
        + (n_alpha * vir_alpha) * (n_beta * vir_beta)
    )
    return singles + doubles


def _selected_ci_estimate(molecule, basis, options=None):
    """Peak memory for the CIPSI-style selected-CI solver.

    Selected CI is defined by *not* holding the full CI space: the
    variational space is an iteratively selected subset capped at
    ``SelectedCIOptions.target_size`` determinants (Huron, Malrieu & Rancurel,
    J. Chem. Phys. 58, 5745 (1973), the CIPSI construction; the heat-bath
    variant vibe-qc implements follows Holmes, Tubman & Umrigar, JCTC 12, 3674
    (2016)).  Sizing it with the untruncated determinant count -- which is what
    ``_wavefunction_estimate`` does, and what this route used to do -- prices
    the one thing the algorithm exists to avoid: it produced a byte-identical
    324364.8 GB estimate at ``target_size`` 6, 20, 50, 100, 200 and 400 for an
    N2/cc-pVDZ CAS(6e,6o) job whose *entire* active space is 400 determinants,
    ~13 orders of magnitude too large, and refused every rung (issue #79).

    The working set the solver actually allocates, per
    ``SelectedCISolver.solve``:

    * The variational space.  ``solve`` appends the selected candidates, then
      trims back to ``target_size`` whenever the list passes
      ``target_size * selection_growth_factor``.  ``H_mat`` is built at the top
      of the next iteration, i.e. always on a post-trim space, so
      ``int(target_size * selection_growth_factor)`` bounds every dense matrix
      the solver forms -- and the space itself cannot exceed the determinant
      count of the (active) space it is selecting from.
    * ``build_hamiltonian_matrix`` materialises that space densely,
      ``ndet x ndet`` float64, and ``scipy_eigh`` adds eigenvectors plus a
      LAPACK workspace of the same order (Davidson, used above
      ``davidson_threshold``, is strictly cheaper but still takes the dense
      ``H_mat``).
    * The candidate dictionary from the selection step, one
      ``_SelectionCandidate`` per *distinct* determinant reachable by a single
      or double excitation from the generators -- deduplicated against both the
      dict and the current space, and bounded by the space's own size.

    ``active_space`` matters twice: it sets both the orbital window the
    excitations are generated in and the hard ceiling on how many determinants
    can exist at all.  The exact/full-CI estimators (``_fci_estimate``,
    ``_casscf_estimate``, ``_wavefunction_estimate``) are deliberately left
    alone -- they size solvers that really do hold the whole space.
    """
    n = _n_basis(molecule, basis)
    est = _rhf_estimate(molecule, basis, _post_reference_options(options))
    method_options = _post_method_options(options, "selected_ci_options")
    active_space = _option_value(options, "active_space", None)

    if active_space is not None:
        n_orb = max(1, int(active_space[0]))
        n_elec = max(0, int(active_space[1]))
    else:
        n_orb = max(1, n)
        n_elec = max(0, int(molecule.n_electrons()))

    multiplicity = int(getattr(molecule, "multiplicity", 1) or 1)
    ms2 = multiplicity - 1
    # The S_z-block count, i.e. the unrestricted (alpha_occ, beta_occ) basis.
    # The spin-restricted kernel walks spin-summed closed-shell configurations,
    # of which there are strictly fewer, so this is an upper bound for both
    # bases -- the safe direction for a pre-flight ceiling.
    n_space_total = _active_determinant_count(n_orb, n_elec, multiplicity)

    # Defaults mirror SelectedCIOptions; a caller that passes no options gets
    # the same working set the solver would build for them.
    target_size = int(_option_value(method_options, "target_size", 100) or 100)
    growth = float(
        _option_value(method_options, "selection_growth_factor", 2.0) or 2.0
    )
    # The default is read from the option class so the pre-flight cannot drift
    # from the solver (it did: the estimate assumed the restricted basis after
    # #107 flipped the solver default to the unrestricted one).
    from .solvers._selected_ci import SelectedCIOptions as _SelectedCIOptions

    spin_restricted = bool(
        _option_value(
            method_options, "spin_restricted", _SelectedCIOptions.spin_restricted
        )
    )

    n_work = max(1, int(max(1, target_size) * max(1.0, growth)))
    n_work = min(n_work, n_space_total)

    per_det = _selected_ci_excitations_per_determinant(
        n_orb, n_elec, ms2, spin_restricted
    )
    n_candidates = min(n_space_total, n_work * max(0, per_det))

    # A candidate is a Python object, not a float64 slot: measured at 476 B for
    # a 7-orbital determinant (dict entry + _SelectionCandidate instance + the
    # occupation tuple, tracemalloc over 2e5 entries).  Rounded up and made
    # linear in the occupation so wider determinants are not under-counted.
    candidate_bytes = 384 + 16 * n_elec

    # H_mat, the eigenvectors scipy_eigh returns, and the LAPACK workspace of
    # the same order.
    est.by_category["Selected-CI variational Hamiltonian"] = 3 * n_work * n_work * 8
    est.by_category["Selected-CI determinant space"] = n_work * candidate_bytes
    est.by_category["Selected-CI candidate buffer"] = n_candidates * candidate_bytes
    # build_hamiltonian_mo transforms the *whole* MO set before the active-space
    # partition, so this term is basis-sized even for a small CAS.
    est.by_category["Selected-CI MO integral transform"] = n**4 * 8 * 3
    return est


def _ccsdt_excitation_count(
    norb: int,
    nalpha: int,
    nbeta: int,
    rank: int,
) -> int:
    """Number of spin-preserving determinants at one excitation rank."""
    total = 0
    for rank_alpha in range(rank + 1):
        rank_beta = rank - rank_alpha
        if not (
            0 <= rank_alpha <= nalpha
            and rank_alpha <= norb - nalpha
            and 0 <= rank_beta <= nbeta
            and rank_beta <= norb - nbeta
        ):
            continue
        total += (
            comb(nalpha, rank_alpha)
            * comb(norb - nalpha, rank_alpha)
            * comb(nbeta, rank_beta)
            * comb(norb - nbeta, rank_beta)
        )
    return total


def _ccsdt_estimate(molecule, basis, options=None):
    """Peak estimate for the dense determinant-space full-CCSDT route."""
    n_basis = _n_basis(molecule, basis)
    reference_estimator = (
        _uhf_estimate if int(molecule.multiplicity) > 1 else _rhf_estimate
    )
    est = reference_estimator(
        molecule,
        basis,
        _post_reference_options(options),
    )
    method_options = _post_method_options(options, "ccsdt_options")
    active_space = _option_value(options, "active_space", None)

    n_elec_total = max(0, int(molecule.n_electrons()))
    if active_space is not None:
        n_active_orb = int(active_space[0])
        n_active_elec = int(active_space[1])
    else:
        n_frozen = _option_value(method_options, "n_frozen_core", None)
        if n_frozen is None:
            from .cc import chemical_core_orbital_count

            n_frozen = chemical_core_orbital_count(molecule)
        n_frozen = max(0, int(n_frozen))
        n_active_orb = max(1, n_basis - n_frozen)
        n_active_elec = max(0, n_elec_total - 2 * n_frozen)

    ms2 = max(0, int(molecule.multiplicity) - 1)
    if (n_active_elec + ms2) % 2:
        nalpha = (n_active_elec + 1) // 2
    else:
        nalpha = (n_active_elec + ms2) // 2
    nbeta = n_active_elec - nalpha
    nalpha = min(max(0, nalpha), n_active_orb)
    nbeta = min(max(0, nbeta), n_active_orb)

    n_amplitudes = sum(
        _ccsdt_excitation_count(n_active_orb, nalpha, nbeta, rank)
        for rank in (1, 2, 3)
    )
    max_state_rank = min(
        7,
        n_active_elec,
        2 * n_active_orb - n_active_elec,
    )
    n_state_determinants = sum(
        _ccsdt_excitation_count(n_active_orb, nalpha, nbeta, rank)
        for rank in range(max_state_rank + 1)
    )
    diis_size = int(_option_value(method_options, "diis_subspace_size", 6))

    est.by_category["CCSDT MO integral transform workspace"] = (
        3 * n_basis**4 * 8
    )
    est.by_category["CCSDT active MO integrals"] = (
        2 * n_active_orb**4 * 8 + n_active_orb**2 * 8
    )
    est.by_category["CCSDT excitation metadata"] = n_amplitudes * 320
    est.by_category["CCSDT amplitudes and DIIS"] = n_amplitudes * (
        8 * (8 + 2 * max(1, diis_size)) + 1
    )
    # Python determinant maps dominate this exact-projection implementation.
    # The two-body action can transiently reach two ranks above exp(T)|0>;
    # six simultaneous maps at 160 bytes/entry is a conservative peak model.
    est.by_category["CCSDT determinant-state dictionaries"] = (
        6 * n_state_determinants * 160
    )
    return est


def _cc3_estimate(molecule, basis, options=None):
    """Conservative peak estimate for dense determinant-space CC3.

    CC3 stores DIIS vectors for singles and doubles only, but its transformed
    Hamiltonian commutators visit the same determinant ranks as the full
    CCSDT projection.  Reusing the CCSDT estimate is therefore conservative
    and avoids understating the dominant Python determinant-map workspace.
    """
    ccsdt_options = {
        "scf_options": _post_reference_options(options),
        "ccsdt_options": _post_method_options(options, "cc3_options"),
        "active_space": _option_value(options, "active_space", None),
    }
    est = _ccsdt_estimate(molecule, basis, ccsdt_options)
    est.by_category = {
        label.replace("CCSDT", "CC3"): value
        for label, value in est.by_category.items()
    }
    return est


def _casscf_estimate(molecule, basis, options=None):
    n = _n_basis(molecule, basis)
    est = _rhf_estimate(molecule, basis, _post_reference_options(options))
    active_space = _option_value(options, "active_space", None)
    if active_space is None:
        active_orb = min(n, max(2, molecule.n_electrons() // 2))
        active_elec = min(molecule.n_electrons(), 2 * active_orb)
    else:
        active_orb, active_elec = int(active_space[0]), int(active_space[1])
    n_det = _active_determinant_count(active_orb, active_elec, molecule.multiplicity)
    est.by_category["CASSCF CI vector"] = 3 * n_det * 8
    est.by_category["CASSCF 4-RDM storage"] = active_orb**8 * 8
    est.by_category["CASSCF MO integral transform"] = n**4 * 8 * 3
    return est


def _fci_estimate(molecule, basis, options=None):
    n = _n_basis(molecule, basis)
    est = _rhf_estimate(molecule, basis, _post_reference_options(options))
    n_elec = max(0, molecule.n_electrons())
    try:
        n_det = max(1, comb(max(0, 2 * n), min(n_elec, max(0, 2 * n))))
    except ValueError:
        n_det = 1
    est.by_category["FCI determinant vector"] = 3 * n_det * 8
    est.by_category["FCI sigma/residual workspace"] = max(n_det * 16, n**4 * 8)
    est.by_category["FCI MO integral transform"] = n**4 * 8 * 3
    return est


def _active_determinant_count(
    n_active_orb: int,
    n_active_elec: int,
    multiplicity: int,
) -> int:
    """Determinants in the spin block used by the CAS reference."""
    ms2 = int(multiplicity) - 1
    if (n_active_elec + ms2) % 2 != 0:
        return max(1, comb(2 * n_active_orb, n_active_elec))
    n_alpha = (n_active_elec + ms2) // 2
    n_beta = n_active_elec - n_alpha
    if not (0 <= n_alpha <= n_active_orb and 0 <= n_beta <= n_active_orb):
        return max(1, comb(2 * n_active_orb, n_active_elec))
    return max(1, comb(n_active_orb, n_alpha) * comb(n_active_orb, n_beta))


def _caspt2_label_count(
    n_orb: int,
    n_core: int,
    n_active_orb: int,
    n_frozen: int,
) -> int:
    """Number of raw internally-contracted CASPT2 labels.

    Mirrors ``solvers._mrpt._ic_caspt2_label_count`` without importing the
    solver stack from the lightweight pre-flight estimator.
    """
    inactive = max(0, n_core - max(0, n_frozen))
    secondary = max(0, n_orb - n_core - n_active_orb)
    occ = inactive + n_active_orb
    virt = n_active_orb + secondary
    n_exc = occ * virt - n_active_orb
    return max(0, n_exc + n_exc * n_exc)


def _mrpt2_orbital_pair_count(
    n_orb: int,
    n_core: int,
    n_active_orb: int,
) -> int:
    """Non-redundant CASSCF/MRPT2 orbital-rotation pair count."""
    inactive = max(0, int(n_core))
    active = max(0, int(n_active_orb))
    secondary = max(0, int(n_orb) - inactive - active)
    return inactive * active + inactive * secondary + active * secondary


def _pt2_gradient_requested(options, option_key: str) -> bool:
    """Whether the requested MRPT2 route computes a correlation gradient."""
    pt2_options = _option_value(options, option_key, None)
    if pt2_options is None:
        return False
    return bool(_option_value(pt2_options, "compute_corr_grad", False))


def _pt2_fd_worker_bytes(nmo: int, *, use_cpp_transform: bool) -> int:
    """Mirror gradient._parallel_helpers.estimate_pt2_fd_worker_bytes."""
    tensor_bytes = max(1, int(nmo)) ** 4 * 8
    multiplier = 10 if use_cpp_transform else 7
    matrix_bytes = max(1, int(nmo)) ** 2 * 8
    return max(256 * 1024**2, multiplier * tensor_bytes + 12 * matrix_bytes)


def _pt2_fd_worker_count(
    *,
    npr: int,
    nmo: int,
    use_cpp_transform: bool,
) -> int:
    """Estimate the runtime PT2 FD process count for memory preflight."""
    npr = int(npr)
    if npr <= 4:
        return 1

    def _positive_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    forced = _positive_env("VIBEQC_PT2_GRADIENT_N_JOBS")
    if forced is not None:
        return max(1, min(npr, forced))

    cpu_limit = max(1, min(npr, os.cpu_count() or 1))
    hard_cap = _positive_env("VIBEQC_PT2_GRADIENT_MAX_JOBS") or 4
    worker_bytes = _pt2_fd_worker_bytes(nmo, use_cpp_transform=use_cpp_transform)
    available = available_memory_bytes()
    if available <= 0:
        memory_limit = 1
    else:
        reserve = max(2 * _GB, available // 4)
        usable = max(0, available - reserve)
        memory_limit = max(1, usable // worker_bytes)
    return max(1, min(cpu_limit, hard_cap, int(memory_limit)))


def _add_pt2_gradient_fd_estimate(
    est: MemoryEstimate,
    *,
    n_orb: int,
    n_core: int,
    n_active_orb: int,
) -> None:
    """Account for CASPT2/NEVPT2 Z-vector dE2/dk finite-difference workers."""
    npr = _mrpt2_orbital_pair_count(n_orb, n_core, n_active_orb)
    if npr <= 0:
        return
    use_cpp_transform = False
    try:
        from . import _vibeqc_core as _core

        use_cpp_transform = hasattr(_core, "transform_4index_mo")
    except Exception:
        use_cpp_transform = False
    n_workers = _pt2_fd_worker_count(
        npr=npr,
        nmo=n_orb,
        use_cpp_transform=use_cpp_transform,
    )
    worker_bytes = _pt2_fd_worker_bytes(
        n_orb,
        use_cpp_transform=use_cpp_transform,
    )
    est.by_category["PT2 gradient FD workers"] = n_workers * worker_bytes


def _caspt2_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    """Memory estimate for the internally-contracted CASPT2 pilot route.

    The production IC-CASPT2 path materializes active-CI external states and
    their generalized-Fock images in Python dictionaries before solving the
    signature-block response system. For release-sized active spaces this
    dominates the small ``n_basis**4`` MO-ERI estimate by orders of magnitude;
    account for it so runs fail before the host watchdog kills them.
    """
    est = _rhf_estimate(molecule, basis, None)
    n_orb = _n_basis(molecule, basis)
    n_elec_total = molecule.n_electrons()
    active_space = _option_value(options, "active_space", None)
    if active_space is None:
        est.by_category["MO ERI tensor + solver"] = (
            n_orb * n_orb * n_orb * n_orb * 8 * 3
        )
        return est

    n_active_orb, n_active_elec = (int(active_space[0]), int(active_space[1]))
    if n_active_orb <= 0 or n_active_elec < 0:
        est.by_category["MO ERI tensor + solver"] = (
            n_orb * n_orb * n_orb * n_orb * 8 * 3
        )
        return est
    if (n_elec_total - n_active_elec) % 2 != 0:
        est.by_category["MO ERI tensor + solver"] = (
            n_orb * n_orb * n_orb * n_orb * 8 * 3
        )
        return est

    caspt2_options = _option_value(options, "caspt2_options", None)
    n_core = max(0, (n_elec_total - n_active_elec) // 2)
    n_frozen = int(_option_value(caspt2_options, "n_frozen", 0) or 0)
    n_labels = _caspt2_label_count(n_orb, n_core, n_active_orb, n_frozen)
    n_det = _active_determinant_count(
        n_active_orb,
        n_active_elec,
        molecule.multiplicity,
    )

    est.by_category["CASPT2 MO integral workspace"] = n_orb**4 * 8 * 3
    est.by_category["CASPT2 contracted labels"] = max(1, n_labels) * 256

    # Each selected contracted candidate can carry an active-CI dictionary,
    # and the current block builder also materializes Fock-applied images.
    # Python dict/tuple/int/float overhead dwarfs the raw float payload; 2 KiB
    # per active determinant entry is conservative but matches the observed
    # N2/cc-pVDZ CAS(6,6) watchdog kill scale.
    est.by_category["CASPT2 active-CI state workspace"] = (
        max(1, n_labels) * max(1, n_det) * 2048
    )

    # Dense per-signature overlap/H0/coupling blocks and their
    # metric-orthonormalized transforms. The direct active-CI route avoids the
    # full n_labels^2 matrix, so keep this term below the state workspace while
    # still exposing the response-system cost in the report.
    est.by_category["CASPT2 response blocks"] = max(1, n_labels) * 512 * 8
    if _pt2_gradient_requested(options, "caspt2_options"):
        _add_pt2_gradient_fd_estimate(
            est,
            n_orb=n_orb,
            n_core=n_core,
            n_active_orb=n_active_orb,
        )
    return est


def _nevpt2_estimate(molecule: Molecule, basis: BasisSet, options) -> MemoryEstimate:
    """Memory estimate for NEVPT2, including optional Z-vector FD workers."""
    est = _rhf_estimate(molecule, basis, None)
    n_orb = _n_basis(molecule, basis)
    est.by_category["MO ERI tensor + solver"] = n_orb**4 * 8 * 3

    active_space = _option_value(options, "active_space", None)
    if active_space is None or not _pt2_gradient_requested(options, "nevpt2_options"):
        return est
    n_active_orb, n_active_elec = (int(active_space[0]), int(active_space[1]))
    n_elec_total = molecule.n_electrons()
    if n_active_orb <= 0 or n_active_elec < 0:
        return est
    if (n_elec_total - n_active_elec) % 2 != 0:
        return est
    n_core = max(0, (n_elec_total - n_active_elec) // 2)
    _add_pt2_gradient_fd_estimate(
        est,
        n_orb=n_orb,
        n_core=n_core,
        n_active_orb=n_active_orb,
    )
    return est


_ESTIMATORS = {
    "rhf": _rhf_estimate,
    "uhf": _uhf_estimate,
    "rks": _rks_estimate,
    "uks": _uks_estimate,
    "mp2": _mp2_estimate,
    "scs-mp2": _mp2_estimate,
    "sos-mp2": _mp2_estimate,
    "ump2": _ump2_estimate,
    "rohf-mp2": _rohf_mp2_estimate,
    "ovgf": _ovgf_estimate,
    "ccsd": _ccsd_estimate,
    "ccsd(t)": _ccsd_t_estimate,
    "ccsd[t]": _ccsd_t_estimate,
    "a-ccsd(t)": _accsd_t_estimate,
    "uccsd": _uccsd_estimate,
    "uccsd(t)": _uccsd_t_estimate,
    "bccd": _bccd_estimate,
    "bccd(t)": _bccd_t_estimate,
    "cc2": _ccsd_estimate,
    "ccd": _ccsd_estimate,
    "lccd": _ccsd_estimate,
    "lccsd": _ccsd_estimate,
    "cepa(0)": _ccsd_estimate,
    "cepa(1)": _ccsd_estimate,
    "cepa(2)": _ccsd_estimate,
    "cepa(3)": _ccsd_estimate,
    "qcisd": _ccsd_estimate,
    "qcisd(t)": _ccsd_t_estimate,
    "cc3": _cc3_estimate,
    "ccsdt": _ccsdt_estimate,
    "dlpno-mp2": _dlpno_mp2_closed_estimate,
    "dlpno-ump2": _dlpno_ump2_estimate,
    "dlpno-ccsd": _dlpno_ccsd_closed_estimate,
    "dlpno-ccsd(t)": _dlpno_ccsd_t_estimate,
    "dlpno-uccsd": _dlpno_uccsd_estimate,
    "dlpno-uccsd(t)": _dlpno_uccsd_t_estimate,
    "cisd": _wavefunction_estimate,
    "selected_ci": _selected_ci_estimate,
    "dmrg": _wavefunction_estimate,
    "v2rdm": _wavefunction_estimate,
    "transcorrelated_ci": _wavefunction_estimate,
    "casci": _casscf_estimate,
    "mrci": _wavefunction_estimate,
    "casscf": _casscf_estimate,
    "nevpt2": _nevpt2_estimate,
    "caspt2": _caspt2_estimate,
    "fci": _fci_estimate,
}


def estimate_memory(
    molecule: Molecule,
    basis: BasisSet,
    *,
    method: str,
    options=None,
) -> MemoryEstimate:
    """Peak memory estimate for ``method``.

    Parameters
    ----------
    molecule
        The :class:`Molecule` about to be run.
    basis
        The :class:`BasisSet` paired with the molecule.
    method
        Molecular SCF, MP2, CC/CI, DLPNO, CAS, and MRPT2 method label.
        Case-insensitive. Periodic drivers use route-specific estimator
        helpers rather than this molecular entry point.
    options
        The matching ``*Options`` struct, or ``None`` for defaults.
        DIIS history size and DFT grid dimensions are read from it
        when available.
    """
    key = method.lower()
    if key not in _ESTIMATORS:
        raise ValueError(
            f"estimate_memory: unknown method {method!r}. Known: {sorted(_ESTIMATORS)}"
        )
    return _ESTIMATORS[key](molecule, basis, options)
