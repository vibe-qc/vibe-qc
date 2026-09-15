"""Crash / abort dump for vibe-qc SCF failures.

When an SCF blows up (raised exception, NaN in the density, linear
dependence, max-iter exceeded), the user's job has already done a lot
of work that's about to be thrown away. Without a dump, the only
record is the exception text the driver re-raises. Re-running from
scratch to debug -- recompute the basis, redo five iterations of SCF,
re-trigger the failure -- wastes 90% of the original wall time.

The crash dump is the equivalent of Gaussian's "link death" output:
on failure, snapshot the last known SCF state to disk so the user can
attach two files (``output.dump`` + ``output.dump.density.npy``) plus
their input script to a bug report and the maintainer can reconstruct
the exact failing state without rerunning.

Output layout
-------------

For ``out_stem = "output-h2o"`` we write up to four files:

* ``output-h2o.dump`` -- TOML-formatted snapshot with sections
  ``[crash]`` (when, phase, exception text, n_iters_completed),
  ``[scf.last_iter]`` (iter, energy, dE, grad_norm, diis_subspace),
  ``[geometry]`` (atom-by-atom Z + xyz + charge + multiplicity),
  ``[options]`` (the SCF options struct), ``[hint]`` (heuristic
  ``likely_cause`` based on the exception type).

* ``output-h2o.dump.density.npy`` -- last-iteration density matrix
  (only when present in ``state``).

* ``output-h2o.dump.fock.npy`` -- last-iteration Fock matrix
  (only when present in ``state``).

* ``output-h2o.dump.mo.npy`` -- current MO coefficients
  (only when present in ``state``).

The TOML is hand-rolled (matches the convention in
:mod:`vibeqc.system_info`); round-tripping is verified in the test
suite via stdlib :mod:`tomllib`.

Public surface
--------------

.. autofunction:: dump_on_failure
.. autofunction:: load_dump
.. autofunction:: classify_failure
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .._errors import OutputFailureKind, warn_output_failure
from .._text_safety import toml_escape_str
from .._stem_paths import stem_sibling


__all__ = [
    "active_crash_dump_stem",
    "classify_failure",
    "crash_dump_context",
    "dump_on_failure",
    "load_dump",
]


# ---------------------------------------------------------------------------
# Heuristic: map an exception to a one-line "likely_cause" hint.
# ---------------------------------------------------------------------------

_HINTS: tuple[tuple[str, str], ...] = (
    ("nan in density",
     "DIIS instability -- try damping=0.5, level_shift=0.5, "
     "or DIIS=False to fall back to plain Roothaan iterations."),
    ("nan",
     "Numerical blow-up -- likely DIIS or quasi-Newton step ran into "
     "an indefinite Hessian. Try a smaller initial step or disable "
     "the quadratic fallback."),
    ("linear depend",
     "Severe basis-set linear dependence -- switch to canonical "
     "orthogonalisation or remove diffuse functions. See "
     "vibeqc.linear_dependence.check_linear_dependence."),
    ("singular",
     "Singular Fock or overlap matrix -- same root cause as linear "
     "dependence. Try canonical_orthog=True in the options struct."),
    ("max_iter",
     "SCF did not converge in max_iter steps -- try damping=0.3, "
     "level_shift=0.5, or increase max_iter. Run again with "
     "structured_log=True to inspect convergence behavior."),
    ("converge",
     "SCF did not converge -- try damping=0.3, level_shift=0.5, or "
     "increase max_iter."),
    ("memory",
     "Out-of-memory during SCF -- try a smaller basis, fewer threads, "
     "or set memory_override=False to abort earlier with a clearer "
     "estimate."),
    ("ecp",
     "ECP backend failure -- verify the ECP library matches the basis "
     "(libecpint XML library 'ecp10mdf' / 'ecp28mdf' / ...)."),
)


def classify_failure(exc: BaseException, *, phase: Optional[str] = None) -> str:
    """Return a one-line ``likely_cause`` string for the dump's
    ``[hint]`` section. Best-effort: matches keywords in the exception
    message and type name; falls back to a generic hint when no
    keyword fires.
    """
    try:
        text = str(exc).lower()
    except Exception:
        text = ""
    type_name = type(exc).__name__.lower()
    haystack = f"{type_name} {text} {(phase or '').lower()}"
    for needle, hint in _HINTS:
        if needle in haystack:
            return hint
    return (
        "No specific hint matched. Inspect the [scf.last_iter] block "
        "and the .density.npy attachment. Re-run with "
        "structured_log=True for an iter-by-iter trace."
    )


# ---------------------------------------------------------------------------
# State-collection helpers.
# ---------------------------------------------------------------------------

def _atom_geometry(molecule: Any) -> list[dict[str, Any]]:
    """Per-atom geometry dump as a list of inline-table-ready dicts."""
    rows: list[dict[str, Any]] = []
    if molecule is None:
        return rows
    for atom in getattr(molecule, "atoms", []) or []:
        xyz = list(getattr(atom, "xyz", (0.0, 0.0, 0.0)))
        rows.append({
            "Z":  int(getattr(atom, "Z", 0)),
            "x":  float(xyz[0]) if len(xyz) > 0 else 0.0,
            "y":  float(xyz[1]) if len(xyz) > 1 else 0.0,
            "z":  float(xyz[2]) if len(xyz) > 2 else 0.0,
        })
    return rows


def _options_dict(options: Any) -> dict[str, Any]:
    """Best-effort dump of an options struct.

    The C++ option classes (``RHFOptions`` etc.) are ``def_readwrite``
    pybind11 wrappers -- we walk public attributes and keep only
    primitive values that round-trip through TOML cleanly.
    """
    if options is None:
        return {}
    out: dict[str, Any] = {}
    for name in dir(options):
        if name.startswith("_"):
            continue
        try:
            value = getattr(options, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (bool, int, float, str)):
            out[name] = value
    return out


def _last_iter_dict(scf_trace: Any) -> dict[str, Any]:
    """Extract the final SCFIteration record (or whatever shape the
    trace carries) into a plain dict. Empty dict when the trace is
    empty or absent."""
    try:
        trace = list(scf_trace) if scf_trace is not None else []
    except TypeError:
        return {}
    if not trace:
        return {}
    last = trace[-1]
    fields = ("iter", "energy", "delta_e", "grad_norm", "diis_subspace")
    out: dict[str, Any] = {}
    for f in fields:
        if hasattr(last, f):
            out[f] = getattr(last, f)
        elif isinstance(last, dict) and f in last:
            out[f] = last[f]
    # Normalize numeric fields.
    if "iter" in out:
        out["iter"] = int(out["iter"])
    if "diis_subspace" in out:
        out["diis_subspace"] = int(out["diis_subspace"])
    for k in ("energy", "delta_e", "grad_norm"):
        if k in out:
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                out.pop(k, None)
    return out


# ---------------------------------------------------------------------------
# TOML emit -- same hand-rolled pattern as system_info.
# ---------------------------------------------------------------------------

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
        # TOML 1.0 has no NaN/Inf in tables -- round-trip safety: emit
        # them as ``nan`` / ``inf`` literals (which are valid TOML floats
        # per spec), but ``tomllib.loads`` returns Python floats with
        # the right finiteness. ``-inf`` is valid too.
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    return _toml_str(str(v))


def _format_simple_section(title: str, data: dict[str, Any]) -> list[str]:
    """Render a single ``[title]`` section with key = value lines."""
    lines = [f"[{title}]"]
    for key, value in data.items():
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    return lines


def _format_atoms_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render the geometry as a TOML array of inline tables."""
    lines = ["[geometry]"]
    if not rows:
        lines.append("atoms = []")
        lines.append("")
        return lines
    lines.append("atoms = [")
    for r in rows:
        lines.append(
            f"  {{ Z = {int(r['Z'])}, "
            f"x = {_toml_value(r['x'])}, "
            f"y = {_toml_value(r['y'])}, "
            f"z = {_toml_value(r['z'])} }},"
        )
    lines.append("]")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# dump_on_failure -- the entry point.
# ---------------------------------------------------------------------------

def dump_on_failure(
    out_stem: Union[str, os.PathLike],
    exc: Optional[BaseException],
    state: dict[str, Any],
    *,
    phase: Optional[str] = None,
    options: Any = None,
    molecule: Any = None,
) -> Path:
    """Write ``{out_stem}.dump`` (TOML) plus binary attachments capturing
    the last known SCF state.

    Parameters
    ----------
    out_stem
        Path stem identifying the failing job -- typically the same
        stem ``run_job`` was called with. ``.dump`` and friends are
        appended.
    exc
        The exception that triggered the dump. May be ``None`` -- used
        when the dump fires from a non-exceptional max-iter path.
    state
        Dictionary carrying as much of the last SCF state as the
        caller has on hand. Recognized keys (all optional):

        * ``"scf_trace"`` -- list of SCFIteration-like records; the
          last entry is rendered into ``[scf.last_iter]``.
        * ``"density"`` / ``"density_alpha"`` / ``"density_beta"`` --
          numpy arrays, side-loaded as ``.dump.density.npy`` /
          ``.dump.density_alpha.npy`` / ``.dump.density_beta.npy``.
        * ``"fock"`` / ``"fock_alpha"`` / ``"fock_beta"`` -- numpy
          arrays, side-loaded similarly.
        * ``"mo_coeffs"`` / ``"mo_coeffs_alpha"`` / ``"mo_coeffs_beta"`` --
          numpy arrays, side-loaded as ``.dump.mo.npy`` /
          ``.dump.mo_alpha.npy`` / ``.dump.mo_beta.npy``.
        * ``"phase"`` -- overridden by the ``phase`` kwarg when set.
        * ``"n_iters_completed"`` -- explicit override; otherwise
          inferred from the trace length.

    phase
        Short string identifying where the failure happened
        (e.g. ``"scf_iteration_5"``, ``"basis_set_construction"``).
        When ``None``, derived from the last trace entry or set to
        ``"unknown"``.
    options
        SCF options struct. Walked via ``dir()`` and only
        primitive-typed attributes survive.
    molecule
        :class:`Molecule` (or shape-compatible) -- used to dump the
        ``[geometry]`` section.

    Returns
    -------
    pathlib.Path
        Path to the written ``.dump`` file (the binary attachments live
        next to it as ``.dump.density.npy``, ``.dump.fock.npy``, ...).

    Notes
    -----
    Never raises. A dump-write failure uses the shared structured output
    warning path and returns the (would-be) path so the caller's re-raise path
    is unchanged.
    """
    stem = Path(os.fspath(out_stem))
    dump_path = stem_sibling(stem, ".dump")

    # The parent must exist before either the binary attachments or the TOML
    # body is written.  A custom nested ``crash_dump=`` stem previously made
    # the directory only after the attachment loop, leaving an apparently
    # valid dump whose density/Fock/MO siblings had all failed.
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc2:
        warn_output_failure(
            exc2,
            dump_path,
            "crash_dump",
            category=OutputFailureKind.optional_artifact,
        )
        return dump_path

    # --- Render the TOML body -------------------------------------------

    trace = state.get("scf_trace")
    last = _last_iter_dict(trace)
    n_iters_completed = state.get("n_iters_completed")
    if n_iters_completed is None:
        try:
            n_iters_completed = len(list(trace)) if trace is not None else 0
        except TypeError:
            n_iters_completed = 0

    resolved_phase = phase or state.get("phase") or (
        f"scf_iteration_{int(last['iter'])}" if "iter" in last else "unknown"
    )

    crash_section = {
        "when": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase": str(resolved_phase),
        "exception_type": type(exc).__name__ if exc is not None else "",
        "exception": str(exc) if exc is not None else "",
        "n_iters_completed": int(n_iters_completed),
    }

    lines: list[str] = [
        "# vibe-qc crash dump -- written when an SCF fails ungracefully.",
        "# Pair this file with the .dump.density.npy attachment and your",
        "# input script to reproduce the exact failing state via",
        "# vibeqc.load_dump('output.dump').",
        "",
    ]
    lines.extend(_format_simple_section("crash", crash_section))

    if last:
        lines.extend(_format_simple_section("scf.last_iter", last))

    lines.extend(_format_atoms_table(_atom_geometry(molecule)))
    if molecule is not None:
        lines.extend(_format_simple_section("molecule", {
            "charge": int(getattr(molecule, "charge", 0)),
            "multiplicity": int(getattr(molecule, "multiplicity", 1)),
            "n_atoms": int(len(list(getattr(molecule, "atoms", []) or []))),
        }))

    opts_dict = _options_dict(options)
    if opts_dict:
        lines.extend(_format_simple_section("options", opts_dict))

    hint_text = classify_failure(exc, phase=resolved_phase) if exc is not None \
        else "SCF reached max_iter without converging -- try damping or " \
             "level_shift adjustments."
    lines.extend(_format_simple_section("hint", {"likely_cause": hint_text}))

    # --- Side-load binary attachments -----------------------------------

    attachments: list[tuple[str, Any]] = []
    for key, suffix in (
        ("density",       "density"),
        ("density_alpha", "density_alpha"),
        ("density_beta",  "density_beta"),
        ("fock",          "fock"),
        ("fock_alpha",    "fock_alpha"),
        ("fock_beta",     "fock_beta"),
        ("mo_coeffs",     "mo"),
        ("mo_coeffs_alpha", "mo_alpha"),
        ("mo_coeffs_beta",  "mo_beta"),
    ):
        arr = state.get(key)
        if arr is None:
            continue
        attachments.append((suffix, arr))

    written_paths: list[str] = []
    for suffix, arr in attachments:
        side_path = dump_path.with_name(f"{dump_path.name}.{suffix}.npy")
        try:
            import numpy as np
            np.save(side_path, np.asarray(arr))
            written_paths.append(side_path.name)
        except Exception as exc2:
            warn_output_failure(
                exc2,
                side_path,
                "crash_dump_attachment",
                category=OutputFailureKind.optional_artifact,
            )

    if written_paths:
        lines.extend(_format_simple_section("attachments", {
            "files": ", ".join(written_paths),
        }))

    # --- Write the .dump itself -----------------------------------------

    try:
        dump_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc2:
        warn_output_failure(
            exc2,
            dump_path,
            "crash_dump",
            category=OutputFailureKind.optional_artifact,
        )

    return dump_path


# ---------------------------------------------------------------------------
# load_dump -- read a dump file back, re-attaching numpy arrays.
# ---------------------------------------------------------------------------

def load_dump(path: Union[str, os.PathLike]) -> dict[str, Any]:
    """Parse a ``.dump`` TOML file and load its sibling ``.npy``
    attachments.

    Returns a nested dict matching the TOML sections, with an extra
    ``"arrays"`` key carrying a ``{name: numpy.ndarray}`` map of every
    sibling ``.dump.<name>.npy`` file present.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover -- Python <3.11
        import tomli as tomllib  # type: ignore[import-not-found]

    p = Path(os.fspath(path))
    with open(p, "rb") as fh:
        parsed: dict[str, Any] = tomllib.load(fh)

    arrays: dict[str, Any] = {}
    try:
        import numpy as np
        prefix = p.name + "."
        for sibling in p.parent.iterdir():
            if not sibling.name.startswith(prefix):
                continue
            if not sibling.name.endswith(".npy"):
                continue
            label = sibling.name[len(prefix):-len(".npy")]
            try:
                arrays[label] = np.load(sibling)
            except Exception:
                continue
    except ImportError:
        pass

    parsed["arrays"] = arrays
    return parsed


# ---------------------------------------------------------------------------
# Crash-dump context (contextvar plumbing) -- let downstream entry points
# fire a dump on max-iter without taking an explicit kwarg.
# ---------------------------------------------------------------------------

_active_stem: contextvars.ContextVar[Optional[Path]] = (
    contextvars.ContextVar("vibeqc_crash_dump_stem", default=None)
)


def active_crash_dump_stem() -> Optional[Path]:
    """The currently registered crash-dump stem, or ``None`` if no
    :func:`crash_dump_context` block is open. Lower-level SCF entry
    points consult this on max-iter to decide whether to side-load a
    dump."""
    return _active_stem.get()


@contextmanager
def crash_dump_context(
    out_stem: Union[str, os.PathLike, None],
    *,
    enabled: bool = True,
) -> Iterator[Optional[Path]]:
    """Register a crash-dump stem for the duration of the block.

    Parameters
    ----------
    out_stem
        Path stem to use for any dump emitted from inside the block.
        ``None`` disables -- the contextvar is left unset.
    enabled
        When ``False``, behaves the same as ``out_stem=None``.

    Notes
    -----
    The companion env var ``VIBEQC_NO_CRASH_DUMP=1`` *also* disables
    even when an explicit stem is passed -- matches the
    ``VIBEQC_NO_HOSTNAME`` opt-out spelling. ``run_job`` calls
    :func:`crash_dump_context` automatically with the resolved opt-out
    semantics; advanced callers using this directly should likewise
    honor the env var, or let :func:`dump_on_failure` callers do so.
    """
    env_off = os.environ.get("VIBEQC_NO_CRASH_DUMP", "").strip().lower()
    if env_off in ("1", "true", "yes", "on"):
        enabled = False

    target: Optional[Path] = None
    if enabled and out_stem is not None:
        target = Path(os.fspath(out_stem))

    token = _active_stem.set(target)
    try:
        yield target
    finally:
        _active_stem.reset(token)
