"""Structured (NDJSON) machine-readable log for vibe-qc jobs.

Companion to the human-readable ``{output}.out`` and the post-mortem
``{output}.perf``: when ``run_job(structured_log=True)`` is set the job
also writes ``{output}.scf.jsonl`` -- one JSON record per line, format-
stable, line-flushed -- so dashboards and analysis scripts can ingest
convergence data without screen-scraping the text log.

Every record carries an ``"event"`` field plus a per-event payload.
New events can be appended over time, but existing event names and
field names are never renamed or removed. NDJSON shape (one JSON
object per line, no enclosing array) means each line is independently
valid -- ``tail -f`` works, and a partial write at crash-time still
leaves earlier records parseable.

Three ways to enable, all of which feed the same logger under the
hood (parallel to :mod:`vibeqc.perf`):

* **Environment variable** -- common case for one-off jobs::

      VIBEQC_STRUCTURED_LOG=output.scf.jsonl python my-calc.py

* **Programmatic context manager** -- wrap a region::

      import vibeqc as vq
      with vq.structured_log("output.scf.jsonl"):
          vq.run_rhf(mol, basis, opts)

* **``run_job`` kwarg** -- one-shot::

      vq.run_job(mol, basis="cc-pVDZ", method="rks",
                 functional="pbe", structured_log=True)

The ``run_job`` kwarg accepts ``True`` (writes ``{output}.scf.jsonl``),
a path-like (writes there), or ``False`` / ``None`` (off).

Public surface
--------------

.. autoclass:: StructuredLog
.. autofunction:: structured_log
.. autofunction:: active_structured_log
.. autofunction:: emit
.. autofunction:: run_fingerprint
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .._errors import OutputFailureKind, warn_output_failure


__all__ = [
    "StructuredLog",
    "active_structured_log",
    "emit",
    "run_fingerprint",
    "structured_log",
]


# ---------------------------------------------------------------------------
# JSON encoding -- numpy / Path / non-finite floats.
# ---------------------------------------------------------------------------

def _json_default(value: Any) -> Any:
    """``json.dumps(default=...)`` hook for non-stdlib types.

    Numpy scalars and arrays are converted to plain Python ints / floats
    / lists so the output stays portable. ``Path`` becomes a string.
    Anything else falls back to ``str(value)`` so the logger never
    raises on an unknown type -- a logging failure must never tank the
    calculation.
    """
    # Numpy is optional at the surface but always available in vibe-qc;
    # import lazily so the module loads cleanly in a pure-stdlib test
    # harness too.
    try:
        import numpy as np
        if isinstance(value, np.generic):
            # Scalars: bool/int/float/complex from numpy -> Python.
            return _coerce_finite(value.item())
        if isinstance(value, np.ndarray):
            return [_json_default(x) if not _json_safe(x) else x
                    for x in value.tolist()]
    except ImportError:
        pass

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    return str(value)


def _coerce_finite(value: Any) -> Any:
    """Map ±Inf and NaN to ``None`` so the encoded JSON is strict.

    ``json.dumps(allow_nan=True)`` would emit literal ``NaN`` / ``Infinity``
    tokens -- which is convenient for Python round-trip but rejected by
    strict consumers (``jq`` first and foremost). The acceptance bar
    for this module says ``jq -c '.' file.jsonl`` succeeds for every
    line, so we coerce to ``null`` instead. Callers that want a
    sentinel string can pre-substitute themselves before calling
    ``emit``.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _json_safe(value: Any) -> bool:
    """Whether ``value`` is already a stdlib type ``json.dumps`` handles
    natively (without triggering ``default=``). Used inside the numpy
    array path to avoid recursing on plain Python floats -- but we
    still need to coerce non-finite floats."""
    return isinstance(value, (str, int, bool, type(None)))


def _scrub(value: Any) -> Any:
    """Recursively walk a payload and coerce non-finite floats / numpy
    scalars in place. Lets the JSON encoder stay strict while callers
    happily pass numpy arrays of floats with the occasional NaN."""
    if isinstance(value, float):
        return _coerce_finite(value)
    if isinstance(value, dict):
        return {str(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return _coerce_finite(value.item())
        if isinstance(value, np.ndarray):
            return _scrub(value.tolist())
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# StructuredLog -- the writer.
# ---------------------------------------------------------------------------

class StructuredLog:
    """One-record-per-line NDJSON writer with line-flushed I/O.

    Construct via :func:`structured_log` (the context manager) for the
    common case. Direct construction is supported for tests / advanced
    callers that want to manage the lifetime themselves -- call
    :meth:`close` when done.

    Parameters
    ----------
    path
        Destination NDJSON file. Truncated on construction so a fresh
        run doesn't accumulate alongside last week's tail. The parent
        directory is created if missing.
    enabled
        If ``False``, all :meth:`emit` calls are no-ops and the file
        is *not* opened. Lets callers pass through a ``StructuredLog``
        unconditionally without wrapping every ``emit`` in an
        ``if log:`` guard.

    Notes
    -----
    Each :meth:`emit` call:

    * stamps ``timestamp`` (ISO-8601 with timezone) onto the record,
    * encodes via :func:`json.dumps` with ``allow_nan=False`` (jq-
      compatible),
    * writes a single line + ``\\n``, then ``flush()`` so a ``tail -f``
      sees records as they land.

    A logging failure (disk full, permissions, ...) is caught and surfaced
    through the shared structured output-warning path; the calculation
    continues. This matches :class:`vibeqc.ProgressLogger` and
    :class:`vibeqc.PerfTracker`.
    """

    __slots__ = ("_path", "_fh", "_enabled", "_n_emitted", "_once_keys")

    def __init__(
        self,
        path: Union[str, os.PathLike, None] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._enabled = bool(enabled and path is not None)
        self._path: Optional[Path] = (
            Path(os.fspath(path)) if path is not None else None
        )
        self._fh = None
        self._n_emitted = 0
        self._once_keys: set[str] = set()
        if self._enabled and self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self._path, "w", encoding="utf-8",
                                buffering=1)
            except OSError as exc:
                warn_output_failure(
                    exc,
                    self._path,
                    "structured_log_open",
                    category=OutputFailureKind.optional_artifact,
                )
                self._enabled = False
                self._fh = None

    # ----- API -----------------------------------------------------------

    def emit(self, event: str, **payload: Any) -> None:
        """Write one record with ``"event": event`` plus ``payload``.

        ``payload`` keys overwrite ``"event"`` if a caller passes
        ``event=...`` as a kwarg -- last-write-wins, matching ``dict``
        construction.
        """
        if not self._enabled or self._fh is None:
            return
        record = {
            "event":     str(event),
            "timestamp": _dt.datetime.now().astimezone().isoformat(
                timespec="microseconds",
            ),
        }
        record.update(_scrub(payload))
        try:
            line = json.dumps(record, default=_json_default,
                              allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            # Last-resort: stringify the whole payload so the line
            # survives. Better a degraded record than a missing event.
            record_safe = {
                "event":     str(event),
                "timestamp": record["timestamp"],
                "_emit_error": str(exc),
                "_repr":     repr(payload),
            }
            line = json.dumps(record_safe, allow_nan=False,
                              separators=(",", ":"))
        try:
            self._fh.write(line + "\n")
            self._fh.flush()
            self._n_emitted += 1
        except OSError as exc:
            warn_output_failure(
                exc,
                self._path,
                "structured_log_write",
                category=OutputFailureKind.optional_artifact,
            )

    def _claim_once(self, key: str) -> bool:
        """Claim a semantic ``once_key`` for this log's lifetime."""
        if key in self._once_keys:
            return False
        self._once_keys.add(key)
        return True

    def _has_once(self, key: str) -> bool:
        """Whether ``key`` is already claimed in this log."""
        return key in self._once_keys

    def close(self) -> None:
        """Flush + close the underlying file handle. Safe to call
        multiple times. After ``close()``, :meth:`emit` is a no-op."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            finally:
                self._fh = None
                self._enabled = False

    # ----- properties ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def n_emitted(self) -> int:
        """Number of records successfully written so far."""
        return self._n_emitted

    # ----- context-manager protocol --------------------------------------

    def __enter__(self) -> "StructuredLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ContextVar plumbing -- parallel to vibeqc.perf._active.
# ---------------------------------------------------------------------------

_active: contextvars.ContextVar[Optional[StructuredLog]] = (
    contextvars.ContextVar("vibeqc_structured_log", default=None)
)


def active_structured_log() -> Optional[StructuredLog]:
    """The currently active :class:`StructuredLog`, or ``None`` if no
    :func:`structured_log` block is open. Used by lower-level entry
    points (e.g. periodic SCF iteration loops) to emit per-iter rows
    without taking an explicit logger argument."""
    return _active.get()


def emit(event: str, **payload: Any) -> None:
    """Module-level convenience: route to the currently active log.

    No-op if no log is active. Equivalent to::

        log = vibeqc.active_structured_log()
        if log is not None:
            log.emit(event, **payload)

    Most call sites should prefer this over threading a logger object.
    """
    log = _active.get()
    if log is not None:
        log.emit(event, **payload)


# ---------------------------------------------------------------------------
# structured_log() context manager.
# ---------------------------------------------------------------------------

@contextmanager
def structured_log(
    path: Union[str, os.PathLike, None] = None,
    *,
    enabled: Optional[bool] = None,
) -> Iterator[StructuredLog]:
    """Activate a fresh :class:`StructuredLog` for the duration of the
    block; close on exit.

    Resolution order for the destination path:

      1. Explicit ``path`` argument when not ``None``.
      2. ``$VIBEQC_STRUCTURED_LOG`` env var.
      3. No file written -- the yielded :class:`StructuredLog` is
         disabled, every ``emit`` is a no-op.

    Parameters
    ----------
    path
        Output path or ``None`` to defer to the env var.
    enabled
        Force enable / disable, overriding both the path and the env
        var. ``None`` (default) means "enabled iff a path resolved".

    Example
    -------
    ::

        import vibeqc as vq
        with vq.structured_log("h2o.scf.jsonl") as log:
            vq.run_rhf(mol, basis, opts)
        # h2o.scf.jsonl now contains banner + scf_iter + ... records.
    """
    target: Optional[Path] = None
    if path is not None:
        target = Path(os.fspath(path))
    else:
        env = os.environ.get("VIBEQC_STRUCTURED_LOG", "").strip()
        if env:
            target = Path(env)

    if enabled is False:
        target = None

    log = StructuredLog(target, enabled=(enabled if enabled is not None
                                         else target is not None))
    token = _active.set(log)
    try:
        yield log
    finally:
        _active.reset(token)
        log.close()


# ---------------------------------------------------------------------------
# run_fingerprint -- stable hash of a job's identifying inputs.
# ---------------------------------------------------------------------------

def run_fingerprint(
    *,
    method: str,
    basis: str,
    functional: Optional[str],
    molecule: Any,
    extras: Optional[dict[str, Any]] = None,
) -> str:
    """A short hex digest summarizing a job's identifying inputs.

    Two runs that share the same method + basis + functional + atoms +
    charge + multiplicity produce the same fingerprint -- useful as an
    opaque "did the user re-run the same calculation?" identifier in
    the structured log's banner record. The hash is deliberately
    coarse: it does *not* capture every SCF option, so two runs with
    different ``conv_tol_energy`` will share a fingerprint. That keeps
    the identifier interpretable as "what was being calculated", not
    "what numerical knobs were turned".

    Returned as the first 16 hex characters of a SHA-256 digest.
    """
    parts: list[str] = [
        f"method={str(method).lower()}",
        f"basis={str(basis).lower()}",
        f"functional={str(functional or '').lower()}",
        f"charge={int(getattr(molecule, 'charge', 0))}",
        f"multiplicity={int(getattr(molecule, 'multiplicity', 1))}",
    ]
    atoms = list(getattr(molecule, "atoms", []) or [])
    for atom in atoms:
        z = int(getattr(atom, "Z", 0))
        xyz = tuple(getattr(atom, "xyz", (0.0, 0.0, 0.0)))
        # Round to 1e-8 bohr so float-formatting noise doesn't perturb
        # the hash; that's well below any meaningful chemistry resolution.
        parts.append(
            f"atom Z={z} "
            f"x={float(xyz[0]):.8f} y={float(xyz[1]):.8f} z={float(xyz[2]):.8f}"
        )
    if extras:
        for key in sorted(extras):
            parts.append(f"{key}={extras[key]}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
