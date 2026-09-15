"""Per-job system manifest -- runtime environment snapshot for ``run_job``.

A ``run_job(output="x")`` call writes ``x.out`` (the human-readable text
log) and, since v0.5.1, ``x.system`` (a TOML manifest pinning *which
machine produced these numbers*). The two files are siblings: the
``.out`` carries the chemistry, the ``.system`` carries the hardware,
linked-library, validation-boundary, and runtime context needed to
interpret a wall-time figure or reproduce a calculation on a different
box.

Without the manifest, an ``.out`` says ``SCF total: 0.015 s`` with no
indication whether that's an Apple M2 Pro at 12 OMP threads or an
8-year-old Xeon at 1 thread -- making generated outputs much less
useful for newcomers comparing their own runs.

Public API
----------

``system_info(*, record_hostname=True)``
    Collect the runtime environment as a nested dict shaped like the
    TOML output. Pure-Python; never raises (every probe falls back to
    ``"unknown"`` on failure). The hostname *and* the Python executable
    path can be redacted via the ``record_hostname=False`` kwarg or the
    ``VIBEQC_NO_HOSTNAME=1`` environment variable -- engineering's
    bundled docs runs use the latter so machine names and home paths
    don't leak into the public bundle.

``write_system_manifest(out_path, wall_seconds, basename, *, record_hostname=True)``
    Render ``system_info()`` plus per-run fields (``wall_seconds``,
    ``basename``, ISO timestamp) as TOML and write to
    ``out_path.with_suffix('.system')``. Returns the path written.

The TOML shape is fixed and machine-readable -- every documented section
+ key is always present, even when the underlying probe couldn't
resolve a value (you get ``"unknown"`` rather than a missing key). This
keeps downstream parsers simple.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...banner import VIBEQC_VERSION, build_info, codename_for_version
from .._text_safety import toml_escape_str


__all__ = [
    "system_info",
    "write_system_manifest",
]


_UNKNOWN = "unknown"


def _safe(callable_, default=_UNKNOWN):
    """Run ``callable_()`` and return its result, swallowing any
    exception and returning ``default`` instead. Probes must never
    raise -- a missing CPU model shouldn't abort a calculation."""
    try:
        result = callable_()
    except Exception:
        return default
    return result if result else default


def _cpu_model() -> str:
    """Best-effort CPU model string. On macOS, ``platform.processor()``
    returns generic strings like ``"arm"`` or ``"i386"`` -- useless for
    distinguishing M1 vs M2 vs M3 Pro/Max -- so we prefer the
    ``sysctl machdep.cpu.brand_string`` probe there. On Linux,
    ``/proc/cpuinfo`` is the authority. ``platform.processor()`` is
    only used as a last-resort fallback (and only when it returns
    something more specific than the platform-generic value).
    Falls back to ``"unknown"``."""
    if sys.platform == "darwin":
        def _sysctl():
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
        out = _safe(_sysctl, default="")
        if out:
            return out

    if sys.platform.startswith("linux"):
        def _cpuinfo():
            with open("/proc/cpuinfo", "r", encoding="ascii",
                      errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
            return ""
        out = _safe(_cpuinfo, default="")
        if out:
            return out

    proc = _safe(platform.processor, default="")
    if proc and proc != "unknown":
        return proc
    return _UNKNOWN


def _os_pretty() -> str:
    """Human-friendly OS name + version (``"macOS 14.4"``,
    ``"Ubuntu 22.04"``). Falls back to ``"<system> <release>"``."""
    if sys.platform == "darwin":
        ver = _safe(lambda: platform.mac_ver()[0], default="")
        if ver:
            return f"macOS {ver}"
    if sys.platform.startswith("linux"):
        def _osrelease():
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                fields: dict[str, str] = {}
                for line in f:
                    if "=" in line:
                        k, v = line.rstrip("\n").split("=", 1)
                        fields[k] = v.strip().strip('"')
            return fields.get("PRETTY_NAME") or fields.get("NAME") or ""
        out = _safe(_osrelease, default="")
        if out:
            return out
    sysname = _safe(platform.system, default="")
    rel = _safe(platform.release, default="")
    return (f"{sysname} {rel}".strip()) or _UNKNOWN


def _omp_threads() -> int:
    """OpenMP thread count vibe-qc would actually use for this job --
    queried via the C++ ``get_num_threads()`` so it agrees with the
    figure the SCF driver actually saw. Falls back to 0 (sentinel:
    "unknown") if the native module isn't importable."""
    try:
        from ..._vibeqc_core import get_num_threads
        return int(get_num_threads())
    except Exception:
        return 0


def _libecpint_version() -> str:
    try:
        from ..._vibeqc_core import libecpint_version
        return str(libecpint_version())
    except Exception:
        return _UNKNOWN


def _fftw3_version() -> str:
    try:
        from ..._vibeqc_core import fftw3_version
        return str(fftw3_version())
    except Exception:
        return _UNKNOWN


_GIB = 1024 ** 3


def _total_memory_bytes() -> int:
    """Best-effort total RAM probe. psutil first (when available),
    then platform-specific (sysctl on macOS, /proc/meminfo on Linux,
    sysconf cross-platform). Returns 0 when no probe works."""
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass

    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            return int(out)
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired, OSError, ValueError):
            pass

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass

    try:
        if hasattr(os, "sysconf"):
            page = os.sysconf("SC_PAGE_SIZE")
            n_pages = os.sysconf("SC_PHYS_PAGES")
            if page > 0 and n_pages > 0:
                return int(page) * int(n_pages)
    except (ValueError, OSError):
        pass

    return 0


def _memory_gb() -> tuple[float, float]:
    """``(total_gb, available_gb)``. Reuses the existing best-effort
    probe in :mod:`vibeqc.memory` for *available* memory so we don't
    duplicate platform handling. Total is its own probe (psutil ->
    ``sysctl hw.memsize`` on macOS -> ``/proc/meminfo`` on Linux ->
    ``sysconf`` fallback). Returns ``(0.0, 0.0)`` if nothing succeeds --
    the field stays present in the manifest, the value just signals
    "could not probe" the same way the memory-estimator block does."""
    total = _total_memory_bytes()
    try:
        from ...memory import available_memory_bytes
        avail = available_memory_bytes()
    except Exception:
        avail = 0
    return (total / _GIB, avail / _GIB)


def _cpu_cores() -> tuple[int, int]:
    """``(physical_cores, logical_cores)``. Logical from
    :func:`os.cpu_count`; physical from ``sysctl hw.physicalcpu`` on
    macOS, ``/proc/cpuinfo`` core-id deduplication on Linux, or
    ``logical`` (== physical with HT off) as a last resort."""
    logical = int(_safe(lambda: os.cpu_count() or 0, default=0))

    physical = 0
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            physical = int(out)
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired, OSError, ValueError):
            physical = 0
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="ascii",
                      errors="replace") as f:
                # Each (physical id, core id) pair counts a physical
                # core. Hyper-threaded siblings share both fields.
                seen: set[tuple[str, str]] = set()
                phys_id = ""
                core_id = ""
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
                    elif line.strip() == "":
                        if phys_id or core_id:
                            seen.add((phys_id, core_id))
                            phys_id = core_id = ""
                if phys_id or core_id:
                    seen.add((phys_id, core_id))
                physical = len(seen)
        except OSError:
            physical = 0

    if physical <= 0:
        physical = logical

    return (physical, logical)


def _hostname_redacted() -> bool:
    """Honor ``VIBEQC_NO_HOSTNAME=1`` (or any non-empty, non-"0"
    value) as a global opt-out -- orthogonal to the function-call
    kwarg. Either lever produces ``hostname = "<redacted>"``."""
    val = os.environ.get("VIBEQC_NO_HOSTNAME", "").strip().lower()
    return val not in ("", "0", "false", "no")


def system_info(*, record_hostname: bool = True) -> dict[str, Any]:
    """Collect the runtime environment as a nested dict.

    The returned dict has the same shape as the on-disk ``.system``
    TOML -- top-level keys are sections (``vibeqc``, ``host``, ``cpu``,
    ``memory``, ``python``, ``libraries``, ``validation``), each mapping
    to the keys documented in :mod:`vibeqc.system_info`.
    ``vibeqc.git_sha`` prefers the full immutable commit ID and falls back to
    the abbreviated ID only when full build provenance is unavailable.

    Parameters
    ----------
    record_hostname
        If ``False`` (or if the ``VIBEQC_NO_HOSTNAME=1`` env var is
        set), the ``host.hostname`` field is set to ``"<redacted>"``
        and the ``python.executable`` path is home-relativised
        (``/Users/<user>/...`` -> ``~/...``) so the home directory is
        not exposed.  Both fields are always present so the TOML shape
        stays stable for parsers; we never *omit* either.
    """
    info = build_info()
    if info:
        git_sha = info.get("sha_full") or info.get("sha") or _UNKNOWN
        git_branch = info.get("branch") or _UNKNOWN
    else:
        git_sha = _UNKNOWN
        git_branch = _UNKNOWN
    is_release = bool(info.get("is_release")) if info else False

    _redacting = not record_hostname or _hostname_redacted()
    if _redacting:
        hostname = "<redacted>"
    else:
        hostname = _safe(socket.gethostname)

    arch = _safe(platform.machine, default=_UNKNOWN)
    os_name = _safe(platform.system, default=_UNKNOWN)
    os_release = _safe(platform.release, default=_UNKNOWN)

    physical, logical = _cpu_cores()
    total_gb, avail_gb = _memory_gb()

    py_version = _safe(platform.python_version, default=_UNKNOWN)
    py_impl = _safe(platform.python_implementation, default=_UNKNOWN)
    py_exe = _safe(lambda: sys.executable, default=_UNKNOWN)
    # When redacting, home-relativise the exe path so the same opt-out
    # that suppresses the hostname also prevents /Users/<user>/... from
    # leaking via the python.executable field (Sec.12 -- same lever, same
    # guarantee, same bundled-docs contract).
    if _redacting and py_exe != _UNKNOWN:
        try:
            _home_str = str(Path.home())
        except Exception:
            _home_str = ""
        if _home_str and py_exe.startswith(_home_str):
            py_exe = "~" + py_exe[len(_home_str):]

    # Linked-library versions: source from banner.library_versions so
    # the manifest matches the banner's "linked:" line exactly.
    # libecpint is part of the linked-native-deps story since the v0.8.0
    # banner / libecpint coupled fix surfaced it on the banner alongside
    # libint / libxc / spglib (see docs/release_v0_8_0_prep.md Sec. 488-501).
    from ...banner import library_versions
    lv = library_versions()
    libs = {
        "libint":    lv.get("libint", _UNKNOWN),
        "libxc":     lv.get("libxc", _UNKNOWN),
        "spglib":    lv.get("spglib", _UNKNOWN),
        "libecpint": lv.get("libecpint", _libecpint_version()),
        # FFTW3 -- required since the FFT-Poisson long-range Hartree
        # solver landed; second consumer is the GAPW route (v0.10.x).
        # Banner + manifest coverage per CLAUDE.md Sec. 6.
        "fftw3":     lv.get("fftw3", _fftw3_version()),
    }

    return {
        "vibeqc": {
            "version":    VIBEQC_VERSION,
            "codename":   codename_for_version(VIBEQC_VERSION) or "",
            "git_sha":    git_sha,
            "git_branch": git_branch,
            "is_release": is_release,
        },
        "host": {
            "hostname":   hostname,
            "os":         os_name,
            "os_release": os_release,
            "os_pretty":  _os_pretty(),
            "arch":       arch,
        },
        "cpu": {
            "model":            _cpu_model(),
            "physical_cores":   physical,
            "logical_cores":    logical,
            "omp_threads_used": _omp_threads(),
        },
        "memory": {
            "total_gb":     round(total_gb, 2),
            "available_gb": round(avail_gb, 2),
        },
        "python": {
            "version":        py_version,
            "implementation": py_impl,
            "executable":     py_exe,
        },
        "libraries": libs,
        "validation": {
            "external_programs_policy": (
                "External QC programs are validation references only."
            ),
            "execution_boundary": (
                "Run external programs out-of-process and parse their "
                "outputs; do not import them as vibe-qc backends."
            ),
            "native_backend_policy": (
                "vibe-qc runtime methods execute vibe-qc-owned native "
                "or Python code."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Hand-rolled TOML emitter
# ---------------------------------------------------------------------------
#
# The manifest is fixed-shape (keys and types are known statically), so a
# general TOML emitter would be overkill. We hand-format with a tiny
# value-quoting helper. Round-tripping is verified in
# tests/test_system_manifest.py via stdlib ``tomllib.loads``.

def _toml_str(s: str) -> str:
    """Quote a string as a TOML basic string. Escapes ``\\``, ``"``, the
    whitespace + C0 controls, and the bidi / zero-width / BOM format
    controls (U+202E etc.) per the TOML 1.0 spec. Delegates to the shared
    output text-safety helper so the dangerous-character set is defined in
    exactly one place -- see :mod:`vibeqc.output._text_safety`."""
    return toml_escape_str(s)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Always emit a decimal point so the value parses as float, not int.
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    return _toml_str(str(v))


# Section ordering for the emitted file -- fixed so diffs are stable
# across runs and across machines. Within a section, keys retain the
# dict-insertion order from system_info() above.
_SECTION_ORDER = (
    "vibeqc", "host", "cpu", "memory", "python", "libraries",
    "validation", "run",
)


def _format_manifest(info: dict[str, Any]) -> str:
    """Render the info dict as the on-disk TOML manifest."""
    lines = [
        "# vibe-qc system manifest -- written alongside output-<job>.out by run_job(...).",
        "# Captures the runtime environment so generated calculation outputs are",
        "# reproducible and wall-time numbers are interpretable.",
        "# External QC programs are validation references only: run them",
        "# out-of-process and parse their outputs.",
        "",
    ]
    for section in _SECTION_ORDER:
        if section not in info:
            continue
        lines.append(f"[{section}]")
        for key, value in info[section].items():
            lines.append(f"{key:<14s} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def write_system_manifest(
    out_path: os.PathLike | str,
    wall_seconds: float,
    basename: str,
    *,
    record_hostname: bool = True,
) -> Path:
    """Write the per-job system manifest next to ``out_path``.

    The manifest path is ``out_path.with_suffix('.system')`` -- pass
    either the ``.out`` file path or the bare stem; both produce the
    same target.

    Parameters
    ----------
    out_path
        The text-output path (or stem) the calculation produced.
        ``write_system_manifest`` writes to the sibling ``.system``
        path. The parent directory must already exist (``run_job``
        creates it).
    wall_seconds
        Total job wall-clock time in seconds. Recorded in the
        ``[run]`` section so a reader can pair the wall-time with the
        host hardware.
    basename
        Path stem identifying the job (e.g. ``"input-h2o-rhf"``).
        Recorded as ``run.basename`` so the manifest is self-
        identifying without needing to read its filename.
    record_hostname
        Forwarded to :func:`system_info`. ``False`` (or
        ``VIBEQC_NO_HOSTNAME=1`` in the env) writes
        ``hostname = "<redacted>"``.

    Returns
    -------
    pathlib.Path
        The path of the written ``.system`` file.
    """
    info = system_info(record_hostname=record_hostname)
    info["run"] = {
        "timestamp_iso": _dt.datetime.now().astimezone().isoformat(
            timespec="seconds",
        ),
        "wall_seconds":  float(wall_seconds),
        "basename":      str(basename),
        "pid":           int(os.getpid()),
    }
    target = Path(os.fspath(out_path)).with_suffix(".system")
    target.write_text(_format_manifest(info), encoding="utf-8")
    return target
