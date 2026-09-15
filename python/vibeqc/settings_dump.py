"""Introspect every user-tunable vibe-qc option struct.

Two entry points:

* :func:`print_settings` -- pretty-print to stdout.
* :func:`format_settings` -- same content, returned as a string.

Both accept either no argument (dump every option struct's defaults)
or a single instance of one of the recognised option structs (dump
only that struct, with a ``*`` marker on each attribute the user
has modified relative to the default-constructed instance).

Also surfaces the runtime environment variables vibe-qc reads --
see :data:`_ENV_VARS` for the authoritative list.

This module is a settings-introspection helper -- it does not
affect any SCF or property kernel.
"""

from __future__ import annotations

import dataclasses
import io
import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Option struct registry
# ----------------------------------------------------------------------
#
# The C++-bound option structs (RHFOptions, ...) all expose a no-arg
# ``__init__`` that fills every member with the C++ default. The Python
# dataclasses (CPHFOptions, HessianFDOptions, ThermoOptions) likewise
# default-construct to the documented defaults. So a fresh ``cls()``
# call is the canonical "default" we diff against.


# Resolved lazily so that this module imports cleanly even if a
# downstream user has built a partial extension.
def _registered_option_classes() -> List[type]:
    """Return the list of recognised option struct classes, in the
    order :func:`print_settings` walks them when called with no args."""
    from . import _vibeqc_core as _core  # type: ignore[attr-defined]
    from .cphf import CPHFOptions
    from .hessian import HessianFDOptions
    from .thermo import ThermoOptions

    return [
        _core.RHFOptions,
        _core.UHFOptions,
        _core.RKSOptions,
        _core.UKSOptions,
        _core.PeriodicSCFOptions,
        _core.PeriodicRHFOptions,
        _core.PeriodicKSOptions,
        _core.LatticeSumOptions,
        _core.EwaldOptions,
        _core.GridOptions,
        _core.GradientOptions,
        HessianFDOptions,
        CPHFOptions,
        ThermoOptions,
        _core.D3BJParams,
        _core.DavidsonOptions,
        _core.LOBPCGOptions,
    ]


# Environment variables vibeqc reads at runtime. Each entry is
# (name, one-line description). Values are pulled from os.environ at
# format time; the "default" column shows ``<unset>`` because that's
# what the library sees in the absence of user override.
_ENV_VARS: Sequence[Tuple[str, str]] = (
    ("OMP_NUM_THREADS", "OpenMP thread count (default: hardware-concurrency)"),
    (
        "LIBINT_DATA_PATH",
        "libint G94 basis library directory (auto-set to bundled basis_library/)",
    ),
    (
        "VIBEQC_ECP_SHARE_DIR",
        "libecpint XML library share dir (auto-set to bundled ecp_library/)",
    ),
    (
        "VIBEQC_FILTERED_BASIS_DIR",
        "override path to the filtered/custom basis directory",
    ),
    (
        "VIBEQC_LIVE_LOGGING",
        "live SCF progress logging in run_job (set to '0' to disable)",
    ),
    ("VIBEQC_NO_HOSTNAME", "redact hostname in run_job's .system manifest"),
    (
        "VIBEQC_NO_CRASH_DUMP",
        "suppress automatic crash-dump files on unhandled failure",
    ),
    ("VIBEQC_PERFLOG", "per-section wall-time perf log (v0.5.2+)"),
    (
        "VIBEQC_PERIODIC_XC_MAX_ESTIMATED_GB",
        "periodic XC build memory guardrail (estimated GB before bail-out)",
    ),
    ("VIBEQC_STRUCTURED_LOG", "structured/JSON SCF event log path (used by run_job)"),
    ("VIBEQC_VERBOSE", "verbose runtime diagnostics"),
    ("VIBEQC_BUILD_BRANCH", "build provenance -- git branch baked into the wheel"),
    ("VIBEQC_BUILD_SHA", "build provenance -- git SHA baked into the wheel"),
    ("VIBEQC_BUILD_TAG", "build provenance -- git tag baked into the wheel"),
)


# ----------------------------------------------------------------------
# Value formatting + nested-option flattening
# ----------------------------------------------------------------------


def _is_option_struct(value: Any) -> bool:
    """Recognise nested option structs so we can flatten ``grid.n_radial``-
    style dotted entries into the parent's table.

    Anything from ``vibeqc._vibeqc_core`` whose class name ends in
    ``Options`` is treated as an option struct even if it isn't in the
    registry yet -- that way new C++ Options classes auto-flatten with
    a stable representation until someone adds them to
    :func:`_registered_option_classes`. Pybind enum values
    (``InitialGuess.SAD``, ``SCFAccelerator.DIIS``, ...) live in the
    same module but must NOT be flattened -- their ``dir()`` returns
    the other enum members of the same type, which would recurse
    forever. They are detected via the duck-typed ``name``/``value``
    pair already used by ``_fmt_value``.
    """
    try:
        registered = tuple(_registered_option_classes())
    except Exception:
        return False
    if isinstance(value, registered):
        return True
    # Reject enums (have a `.name` and `.value` pair) even if their
    # module matches.
    if hasattr(value, "name") and hasattr(value, "value"):
        return False
    cls = type(value)
    if getattr(cls, "__module__", "") != "vibeqc._vibeqc_core":
        return False
    name = getattr(cls, "__name__", "")
    return name.endswith("Options") or name == "D3BJParams"


def _fmt_value(value: Any) -> str:
    """Render a single attribute value into one display token."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        # Compact, but never lossy at the precision users care about.
        # Typical defaults -- 1e-8, 1e-6, 0.5, 0.1 -- render cleanly via
        # %g; uncommon high-precision overrides survive via repr fallback.
        text = f"{value:g}"
        # Round-trip safety: if %g's compact form doesn't reproduce the
        # original double, fall back to repr (e.g. 1e-12 stays as repr).
        try:
            if float(text) != value:
                return repr(value)
        except (TypeError, ValueError):
            return repr(value)
        return text
    if isinstance(value, str):
        return repr(value)
    # Enum (e.g. InitialGuess.SAD, CoulombMethod.DIRECT_TRUNCATED) --
    # str() gives the qualified form; cheap & informative.
    if hasattr(value, "name") and hasattr(value, "value"):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return repr(value)


def _values_equal(a: Any, b: Any) -> bool:
    """Equality check that tolerates nested option structs (compared
    field-by-field) and the few container types we encounter
    (``ecp_centers`` is a list of ECPCenter)."""
    if _is_option_struct(a) and _is_option_struct(b):
        return _flatten_attrs(a) == _flatten_attrs(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    try:
        return a == b
    except Exception:
        return False


def _attribute_names(obj: Any) -> List[str]:
    """Public, non-callable attribute names in declaration-stable order.

    For dataclasses we use ``dataclasses.fields()`` so the declared
    order is preserved. For the C++-bound structs ``dir()`` is sorted
    alphabetically by pybind11; we accept that ordering for the
    bound structs since their attribute names are short and the
    alphabetic order is at least stable across Python versions.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [f.name for f in dataclasses.fields(obj)]
    return sorted(
        a
        for a in dir(obj)
        if not a.startswith("_") and not callable(getattr(obj, a, None))
    )


def _flatten_attrs(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """Walk an option struct, returning a flat list of
    ``(dotted_name, value)`` pairs. Nested option structs are
    recursed into; everything else is left as a single entry."""
    out: List[Tuple[str, Any]] = []
    for name in _attribute_names(obj):
        value = getattr(obj, name)
        full = f"{prefix}{name}"
        if _is_option_struct(value):
            out.extend(_flatten_attrs(value, prefix=f"{full}."))
        else:
            out.append((full, value))
    return out


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

_HEADER_RULE = "=" * 70
_SECTION_RULE = "-" * 70


def _render_struct(
    cls: type,
    instance: Optional[Any],
    out: io.StringIO,
) -> None:
    """Render one option struct's attribute table.

    If ``instance`` is None, only defaults are shown (no ``*`` markers,
    'current' column collapses to the same value as 'default').
    """
    default = cls()
    target = instance if instance is not None else default

    cur = _flatten_attrs(target)
    dflt = dict(_flatten_attrs(default))

    out.write(f"{_HEADER_RULE}\n")
    out.write(f"{cls.__name__}\n")
    out.write(f"{_HEADER_RULE}\n")
    if not cur:
        out.write("  (no public attributes)\n\n")
        return

    name_width = max(len(name) for name, _ in cur)
    name_width = max(name_width, len("attribute"))
    out.write(
        f"  {'':1} {'attribute':<{name_width}}  {'current':<24}  {'default':<24}\n"
    )
    out.write(f"  {_SECTION_RULE}\n")
    for name, value in cur:
        default_value = dflt.get(name, value)
        modified = instance is not None and not _values_equal(value, default_value)
        marker = "*" if modified else " "
        cur_str = _fmt_value(value)
        dflt_str = _fmt_value(default_value)
        out.write(f"  {marker} {name:<{name_width}}  {cur_str:<24}  {dflt_str:<24}\n")
    out.write("\n")


def _render_env(out: io.StringIO) -> None:
    """Render the runtime-environment table -- shows current value vs
    ``<unset>``. The ``*`` marker fires on any name that is currently
    set in the process environment."""
    out.write(f"{_HEADER_RULE}\n")
    out.write("Environment variables\n")
    out.write(f"{_HEADER_RULE}\n")
    name_width = max(len(name) for name, _ in _ENV_VARS)
    name_width = max(name_width, len("variable"))
    out.write(f"  {'':1} {'variable':<{name_width}}  {'current':<28}  description\n")
    out.write(f"  {_SECTION_RULE}\n")
    for name, description in _ENV_VARS:
        if name in os.environ:
            current = os.environ[name]
            marker = "*"
            # Quote so trailing whitespace / empty strings are visible.
            cur_str = repr(current)
        else:
            marker = " "
            cur_str = "<unset>"
        out.write(f"  {marker} {name:<{name_width}}  {cur_str:<28}  {description}\n")
    out.write("\n")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _render_solver(out: io.StringIO, solver: Optional[str]) -> None:
    """Render the eigensolver selection line."""
    out.write(f"{_HEADER_RULE}\n")
    out.write("Solver\n")
    out.write(f"{_HEADER_RULE}\n")
    current = solver if solver else "dense"
    default = "dense"
    marker = "*" if solver and solver != "dense" else " "
    out.write(f"  {marker} solver                     {current:<24}  {default:<24}\n")
    out.write("\n")


def format_settings(
    opts: Optional[Any] = None,
    *,
    solver: Optional[str] = None,
) -> str:
    """Return a formatted settings dump as a string.

    Parameters
    ----------
    opts:
        If ``None`` (default), every recognised option struct is
        listed at its default values, followed by the env-var table.
        If an instance of one of the recognised option structs (e.g.
        :class:`vibeqc.RHFOptions`) is passed, only that struct's
        table is emitted, with a ``*`` next to every attribute whose
        value differs from the default.
    solver:
        Optional eigensolver keyword (``"dense"``, ``"davidson"``,
        or ``"lobpcg"``).  When provided and not ``"dense"``, a
        solver selection line is rendered at the top of the output
        with a ``*`` marker.  When ``None`` (default), the solver
        line always shows ``"dense"`` (the default).

    Raises
    ------
    TypeError
        ``opts`` is not ``None`` and not an instance of one of the
        recognised option structs.
    """
    out = io.StringIO()
    out.write("vibe-qc settings\n")
    out.write("(values prefixed with '*' have been modified from the default)\n\n")

    if opts is None:
        _render_solver(out, solver)
        for cls in _registered_option_classes():
            _render_struct(cls, instance=None, out=out)
        _render_env(out)
        return out.getvalue()

    registered = tuple(_registered_option_classes())
    if not isinstance(opts, registered):
        names = ", ".join(c.__name__ for c in registered)
        raise TypeError(
            f"format_settings: opts must be one of {{{names}}} or None; "
            f"got {type(opts).__name__}"
        )
    _render_struct(type(opts), instance=opts, out=out)
    return out.getvalue()


def print_settings(
    opts: Optional[Any] = None,
    *,
    solver: Optional[str] = None,
) -> None:
    """Pretty-print every user-adjustable vibe-qc option to stdout.

    Thin wrapper around :func:`format_settings`. See that function
    for the calling convention.
    """
    text = format_settings(opts, solver=solver)
    # ``end=""`` because format_settings already appends a trailing newline.
    print(text, end="")


__all__ = ["format_settings", "print_settings"]
