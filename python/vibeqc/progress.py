"""Live progress logging for vibe-qc SCF and post-SCF stages.

Designed to defeat output buffering (every write is flushed) so that
the typical remote-job pattern just works::

    nohup python my_calc.py > job.log 2>&1 &
    tail -f job.log

... and you see each SCF iteration land as it happens, instead of a
long silence followed by a wall of text at the end.

Every SCF entry point in :mod:`vibeqc` accepts a ``progress=`` keyword
argument that is one of:

* ``True`` -- a fresh :class:`ProgressLogger` writing to ``sys.stdout``.
* ``False`` / ``None`` -- silent (the historical behavior).
* a :class:`ProgressLogger` instance -- fully under caller control,
  including a tee to a persistent ``.out`` file.

Verbosity (v0.5.3)
------------------

The amount of detail is controlled by an integer ``verbose`` level
following the PySCF convention (0..9). Each level is a strict
superset of the one below, so increasing ``verbose`` only adds
output:

=====  ==========================================================
level  what is emitted
=====  ==========================================================
0      silent -- every method is a no-op
1      banner + warnings + final SCF status only
2      add per-stage start lines + ``info()`` milestones
3      add per-stage timing on stage exit (current pre-v0.5.3
       default behavior)
4      add per-iteration SCF rows (DEFAULT)
5      add inline memory snapshots
6+     debug -- phase-level wall-clock breakdown (overlaps the
       post-mortem ``.perf`` log)
=====  ==========================================================

Logging integration (v0.5.3)
----------------------------

Pass ``use_logging=True`` to route every emit through
:mod:`logging` instead of the bare stream. Banners, milestones,
and ``converged()`` summaries land at ``INFO`` on the
``vibeqc.run_job`` logger; per-iteration SCF rows at ``DEBUG``;
warnings at ``WARNING``. This composes naturally with stdlib
handlers -- ``RotatingFileHandler``, syslog, ``dictConfig`` -- so
the canonical example::

    import logging
    logging.basicConfig(level=logging.INFO)
    vq.run_job(mol, basis="6-31g*", method="rhf", output="x",
               use_logging=True)

... mirrors progress through the user's logging stack rather than
``sys.stdout``. The verbose-level gate still applies before the
logging call, so ``verbose=2`` + ``use_logging=True`` will not
emit per-iteration ``DEBUG`` records.

Public surface
--------------

.. autoclass:: ProgressLogger
   :members:

.. autofunction:: resolve_progress
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator, Optional, Union

__all__ = ["ProgressLogger", "resolve_progress"]


_INDENT = "  "

#: Default verbose level when neither caller nor environment
#: override is set. Matches PySCF's convention (4 = "verbose"):
#: banner + stage milestones + per-iteration SCF rows.
DEFAULT_VERBOSE: int = 4

#: Logger name used when ``use_logging=True``. Users can attach
#: their own handlers via ``logging.getLogger("vibeqc.run_job")``.
_LOGGER_NAME: str = "vibeqc.run_job"


def _coerce_verbose(verbose: Union[int, bool, None]) -> int:
    """Normalize the ``verbose`` argument into a non-negative int.

    * ``True`` -> :data:`DEFAULT_VERBOSE` (back-compat with the
      pre-v0.5.3 boolean-only API).
    * ``False`` -> 0 (silent -- same back-compat).
    * ``None`` -> :data:`DEFAULT_VERBOSE`.
    * ``int`` -> clamped at 0.
    """
    if verbose is None:
        return DEFAULT_VERBOSE
    if verbose is True:
        return DEFAULT_VERBOSE
    if verbose is False:
        return 0
    return max(0, int(verbose))


class ProgressLogger:
    """Per-stage progress emitter for long-running calculations.

    Writes plain-ASCII, line-flushed output to ``stream`` (default
    ``sys.stdout``) and, optionally, to ``log_path`` (truncated on
    construction, appended-and-flushed thereafter). Every write goes
    through ``flush()`` immediately so ``tail -f`` against a redirected
    stdout shows the run unfolding in real time.

    Parameters
    ----------
    stream
        Where to write live progress. Default ``sys.stdout``.
        Ignored when ``use_logging=True``.
    log_path
        Optional path. Truncated on construction; appended-to with a
        flush after each write. Useful when callers want both an
        interactive stdout view and a persistent on-disk record.
        Honored regardless of ``use_logging``.
    verbose
        Integer verbosity level (0..9, default 4). Higher values
        add more detail; level 0 makes every method a no-op. The
        level table is in the module docstring. ``True`` / ``False``
        are accepted for back-compat (``True`` -> 4, ``False`` -> 0).
    use_logging
        If ``True``, route emits through ``logging.getLogger(
        "vibeqc.run_job")`` instead of the stream. Composes with
        stdlib handlers (rotating files, syslog, dictConfig) -- no
        special integration required. The ``log_path`` tee still
        applies if set.

    Notes
    -----
    The current implementation deliberately stays plain ASCII -- the
    goal is the ``nohup`` + ``tail -f`` workflow, where color codes
    would only inject control bytes that no log viewer asked for. TTY
    detection is exposed via :attr:`is_tty` for downstream tooling
    that wants to layer color or progress bars on top.
    """

    def __init__(
        self,
        stream: Optional[IO[str]] = None,
        log_path: Optional[Union[str, os.PathLike]] = None,
        verbose: Union[int, bool, None] = DEFAULT_VERBOSE,
        use_logging: bool = False,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._log_path: Optional[Path] = (
            Path(log_path) if log_path is not None else None
        )
        self._level = _coerce_verbose(verbose)
        self._use_logging = bool(use_logging)
        self._logger = (
            logging.getLogger(_LOGGER_NAME) if self._use_logging else None
        )
        self._t_start = time.perf_counter()

        if self._log_path is not None and self._level > 0:
            # Truncate so a fresh run doesn't accumulate alongside last
            # week's tail. Subsequent writes are open(..., 'a') so an
            # external 'tail -f' sees lines appear as they're written.
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "w", encoding="utf-8") as fh:
                fh.write("")

    # ----- low-level write -------------------------------------------------

    def _emit(self, line: str, log_level: int = logging.INFO) -> None:
        """Push ``line`` to whichever sink is active.

        Stream mode (default): write + flush to ``self._stream`` and
        (if configured) ``self._log_path``.

        Logging mode (``use_logging=True``): hand the line to
        ``logging.getLogger("vibeqc.run_job").log(log_level, ...)``,
        which lets users plug in stdlib handlers (rotating files,
        syslog, dictConfig). The ``log_path`` tee still applies so
        callers can have both a stdlib-managed log stream AND a
        verbatim per-job ``.out`` companion.
        """
        if self._level <= 0:
            return
        if self._use_logging:
            try:
                # logging.<level> already handles its own newlines.
                self._logger.log(log_level, line.lstrip())
            except Exception:
                # A logging failure must never tank the calculation.
                pass
        else:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except Exception:
                pass
        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            except Exception:
                pass

    # ----- public API ------------------------------------------------------

    def info(self, message: str) -> None:
        """Emit a one-line informational message, indented.

        Visible at verbose level 2 and above.
        """
        if self._level < 2:
            return
        self._emit(f"{_INDENT}{message}", logging.INFO)

    def warn(self, message: str) -> None:
        """Emit a one-line warning, prefixed with ``WARN:``.

        Visible at verbose level 1 and above (warnings should not
        be hidden by quiet runs short of full silence).
        """
        if self._level < 1:
            return
        self._emit(f"{_INDENT}WARN: {message}", logging.WARNING)

    def banner(self, title: str) -> None:
        """Section banner: blank line + indented title + dashed rule.

        Visible at verbose level 1 and above. Under ``use_logging``
        the three lines emit as three INFO records.
        """
        if self._level < 1:
            return
        self._emit("", logging.INFO)
        self._emit(f"{_INDENT}{title}", logging.INFO)
        self._emit(f"{_INDENT}{'-' * max(len(title), 32)}", logging.INFO)

    def write_raw(self, text: str) -> None:
        """Write ``text`` exactly as given (newlines preserved), with a
        single flush. Used by callers that want to splice in a
        pre-formatted block (e.g. the geometry table from
        :mod:`vibeqc.scf_log`).

        Visible at verbose level 2 and above.
        """
        if self._level < 2:
            return
        if self._use_logging:
            # Preserve internal newlines as one logging record per
            # line, so handlers that prefix each record stay sane.
            for line in text.splitlines() or [""]:
                try:
                    self._logger.log(logging.INFO, line)
                except Exception:
                    pass
        else:
            try:
                self._stream.write(text)
                if not text.endswith("\n"):
                    self._stream.write("\n")
                self._stream.flush()
            except Exception:
                pass
        if self._log_path is not None:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(text)
                    if not text.endswith("\n"):
                        fh.write("\n")
                    fh.flush()
            except Exception:
                pass

    @contextmanager
    def stage(
        self, name: str, *, detail: Optional[str] = None,
    ) -> Iterator[None]:
        """Context manager: emit ``[name]`` on enter, ``[name] done
        (X.XXs)`` on exit -- regardless of which branch ran inside.

        The start line emits at verbose level 2; the timing
        ``done`` line emits at level 3 (so ``verbose=2`` shows
        which stages ran but suppresses per-stage wall-time noise,
        and ``verbose>=3`` adds the timing detail).

        Parameters
        ----------
        name
            Short stage identifier (printed in brackets).
        detail
            Optional one-line addendum appended to the start line
            (e.g. ``"4x4x4 -> 8 IBZ k-points"``).
        """
        if self._level < 2:
            yield
            return
        suffix = f" - {detail}" if detail else ""
        self._emit(f"{_INDENT}[{name}]{suffix}", logging.INFO)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._level >= 3:
                dt = time.perf_counter() - t0
                self._emit(
                    f"{_INDENT}[{name}] done ({_fmt_seconds(dt)})",
                    logging.INFO,
                )

    def iteration(self, n: int, **fields: Any) -> None:
        """One SCF iteration line.

        Recognized fields: ``energy`` (Ha), ``dE`` (Ha),
        ``grad`` (``||[F,DS]||``), ``diis`` (subspace dim). Any other
        key/value pair is appended at the end. Wall time since logger
        construction is appended automatically.

        At ``n == 1``, ``dE`` is rendered as a placeholder ``--`` so
        the column layout matches :func:`vibeqc.scf_log.format_scf_trace`.

        Visible at verbose level 4 and above. Under ``use_logging``
        emits at ``DEBUG`` (per-iter detail belongs below INFO so
        ``logging.basicConfig(level=INFO)`` keeps a quiet stream
        while ``level=DEBUG`` exposes the trace).

        As a side effect (regardless of verbose level), the same
        iteration is emitted to the currently active
        :class:`vibeqc.StructuredLog` (if any) as a ``scf_iter``
        event with the same field names -- funnelling every
        Python-driven SCF's per-iter trace into the structured log
        without each driver having to wire the event itself. The
        structured log is its own opt-in surface, so silencing the
        text trace via ``verbose=0`` does not also silence the
        machine-readable record.
        """
        # Snapshot the recognized SCF-iter fields up-front so they
        # survive the text formatter's fields.pop() consumption AND
        # are available even when verbose<4 silences the text body.
        # Cost when no structured log is active is one
        # contextvars.ContextVar.get() inside the emit() free
        # function -- the same gate perf_log uses.
        wall = time.perf_counter() - self._t_start
        _energy = fields.get("energy")
        _dE = fields.get("dE")
        _grad = fields.get("grad")
        _diis = fields.get("diis")

        if self._level >= 4:
            cells = [f"iter {n:4d}"]
            e = fields.pop("energy", None)
            if e is not None:
                cells.append(f"E = {float(e):18.10f} Ha")
            dE = fields.pop("dE", None)
            if dE is not None:
                cells.append(
                    "dE =     --   " if n <= 1 else f"dE = {float(dE):+.3e}"
                )
            grad = fields.pop("grad", None)
            if grad is not None:
                cells.append(f"||[F,DS]|| = {float(grad):.3e}")
            diis = fields.pop("diis", None)
            if diis is not None:
                cells.append(
                    f"DIIS={int(diis):2d}" if int(diis) > 0 else "DIIS= -"
                )
            for key, value in fields.items():
                cells.append(f"{key}={value}")
            cells.append(f"[{_fmt_seconds(wall)}]")
            self._emit(_INDENT + "  ".join(cells), logging.DEBUG)

        # Structured log: same iteration as a machine-readable
        # ``scf_iter`` record. ``dE`` follows the same null-on-first-
        # iter convention as the runner's molecular path so consumers
        # see one shape across both backends. Independent of verbose
        # level -- the structured log is gated by its own opt-in.
        try:
            from .structured_log import emit as _emit_structured
            _emit_structured(
                "scf_iter",
                iter=int(n),
                energy=(float(_energy) if _energy is not None else None),
                dE=(float(_dE) if (_dE is not None and n > 1) else None),
                grad_norm=(float(_grad) if _grad is not None else None),
                diis_subspace=(int(_diis) if _diis is not None else 0),
                wall_s=float(wall),
            )
        except Exception:
            # A logging failure must never tank the calculation.
            pass

    def energy_decomposition(self, n: int, **components: float) -> None:
        """Per-iteration energy decomposition line.

        Emits an indented row of ``key=value`` pairs right under the
        :meth:`iteration` summary so every SCF iteration shows the
        breakdown into kinetic / nuclear-attraction / Hartree-J /
        XC / nuclear-repulsion / Madelung-fix contributions. This
        matters for cross-code comparison (e.g. against PySCF.pbc):
        if the totals match but a component disagrees, the bug is
        localised to that one operator.

        Recognized keys (printed in this order if present):
        ``E_kin``, ``E_ne``, ``E_J``, ``E_xc``, ``E_K`` (HF exchange),
        ``E_nuc``, ``E_madelung``. Any other key/value pair is
        appended at the end. The accompanying ``E_total`` is already
        printed by :meth:`iteration`, so it isn't repeated here.

        Visible at verbose level 4+ (the same gate as
        :meth:`iteration`). Also forwards every component to the
        currently active :class:`vibeqc.StructuredLog` (if any) as
        an ``scf_energy_components`` event so post-hoc tools have
        machine-readable access to the decomposition without parsing
        text.
        """
        # Snapshot for structured-log emission below -- survives the
        # text formatter's components.pop() consumption.
        snap = dict(components)

        if self._level >= 4:
            cells = [f"     "]  # align under "iter NNNN"
            order = ("E_kin", "E_ne", "E_J", "E_xc", "E_K",
                     "E_nuc", "E_madelung")
            for key in order:
                v = components.pop(key, None)
                if v is not None:
                    cells.append(f"{key}={float(v):+15.8f}")
            for key, value in components.items():
                cells.append(f"{key}={float(value):+15.8f}")
            if len(cells) > 1:  # only emit if any component was passed
                self._emit(_INDENT + "  ".join(cells), logging.DEBUG)

        try:
            from .structured_log import emit as _emit_structured
            _emit_structured(
                "scf_energy_components",
                iter=int(n),
                **{k: (float(v) if v is not None else None)
                   for k, v in snap.items()},
            )
        except Exception:
            pass

    def converged(
        self, *, n_iter: int, energy: float, converged: bool,
    ) -> None:
        """Final SCF status line; called once after the iteration loop.

        Visible at verbose level 1 and above (the final summary is
        almost always wanted -- only a fully silent ``verbose=0``
        suppresses it).

        Also routes a ``scf_converged`` event to the currently
        active :class:`vibeqc.StructuredLog` (if any) -- same funnel
        pattern as :meth:`iteration`, independent of verbose level
        so the structured log's opt-in stays the only gate on the
        machine-readable record.
        """
        if self._level >= 1:
            status = "converged" if converged else "NOT converged"
            self._emit(
                f"{_INDENT}SCF {status} in {n_iter} iterations; "
                f"E = {energy:.10f} Ha",
                logging.INFO,
            )
        try:
            from .structured_log import emit as _emit_structured
            _emit_structured(
                "scf_converged",
                n_iter=int(n_iter),
                energy=float(energy),
                converged=bool(converged),
            )
        except Exception:
            pass

    def memory(self, label: str, rss_mib: float) -> None:
        """Inline RSS-memory snapshot.

        Visible at verbose level 5 and above. Pairs with the
        post-mortem ``.perf`` log's ``Memory snapshots`` section --
        same data, but live.
        """
        if self._level < 5:
            return
        self._emit(
            f"{_INDENT}[memory] {label}: {float(rss_mib):.1f} MiB",
            logging.INFO,
        )

    def debug(self, message: str) -> None:
        """Phase-level wall-clock breakdown / debug detail.

        Visible at verbose level 6 and above. Overlaps the
        post-mortem ``.perf`` log on purpose -- the perf log shows
        the same numbers when the run is over; ``verbose>=6``
        streams them live for users debugging an in-flight job.
        Under ``use_logging`` emits at ``DEBUG`` so a stdlib
        ``RotatingFileHandler`` set to ``DEBUG`` captures the
        trace without lifting the bar for INFO-only handlers.
        """
        if self._level < 6:
            return
        self._emit(f"{_INDENT}[debug] {message}", logging.DEBUG)

    @property
    def enabled(self) -> bool:
        """``True`` iff :attr:`level` is non-zero (i.e. anything
        emits). Kept as the legacy alias for ``verbose=True/False``
        callers that predate the integer level."""
        return self._level > 0

    @property
    def level(self) -> int:
        """The integer verbosity level this logger was constructed
        with (0..9)."""
        return self._level

    @property
    def use_logging(self) -> bool:
        """Whether emits route through :mod:`logging` (``True``) or
        the raw stream (``False``)."""
        return self._use_logging

    @property
    def stream(self) -> IO[str]:
        return self._stream

    @property
    def is_tty(self) -> bool:
        """Whether the underlying stream is a TTY. Useful for callers
        that want to layer color / spinners over the plain output."""
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False


_NULL_LOGGER: Optional["ProgressLogger"] = None


def _null_logger() -> "ProgressLogger":
    """Singleton no-op logger reused when callers pass ``progress=False``."""
    global _NULL_LOGGER
    if _NULL_LOGGER is None:
        _NULL_LOGGER = ProgressLogger(stream=sys.stdout, verbose=0)
    return _NULL_LOGGER


def resolve_progress(
    progress: Union[bool, "ProgressLogger", None],
    *,
    verbose: Union[int, bool, None] = None,
    use_logging: bool = False,
) -> "ProgressLogger":
    """Normalize the ``progress=`` argument every SCF entry point takes.

    * ``True`` -> a fresh :class:`ProgressLogger` writing to stdout
      at the requested ``verbose`` level (default 4).
    * ``False`` or ``None`` -> a shared, silent no-op logger.
    * :class:`ProgressLogger` -> returned unchanged (``verbose`` /
      ``use_logging`` kwargs are ignored -- the caller-supplied
      logger is fully under the caller's control).

    This indirection lets every SCF driver accept any of the four
    forms via a one-line normalisation at the top of the function,
    and threads the ``verbose=N`` knob through without each entry
    point having to construct its own logger by hand.

    A fifth form is a **transparent wrapper** around a
    :class:`ProgressLogger` -- e.g. the live-checkpointing proxy from
    :func:`vibeqc.output.checkpoint.wrap_progress_for_checkpoints`, which
    forwards every method to a real logger but also fires a QVF snapshot on
    each ``iteration``. Such a wrapper is not a ``ProgressLogger`` instance,
    but it *is* a caller-supplied sink: any object exposing a callable
    ``iteration`` is returned unchanged so the wrapper survives each
    driver's ``resolve_progress`` normalisation (otherwise the per-iteration
    checkpoint hook would be silently stripped before the SCF loop ran).
    """
    if isinstance(progress, ProgressLogger):
        return progress
    if progress is True:
        level = (
            DEFAULT_VERBOSE if verbose is None else _coerce_verbose(verbose)
        )
        return ProgressLogger(
            stream=sys.stdout, verbose=level, use_logging=use_logging,
        )
    if progress is False or progress is None:
        return _null_logger()
    # Duck-typed progress sink (a transparent wrapper such as the checkpoint
    # proxy): return it unchanged so its ``iteration`` override keeps firing.
    if callable(getattr(progress, "iteration", None)):
        return progress
    # Anything else falls through to a silent no-op. A *falsy* value (0, "",
    # []) reads as "off" and is silently accepted. A *truthy* unrecognized
    # value is almost always a mistake -- e.g. an OutputChannel (persistent
    # output is ambient and is not a live progress sink), or a typo.
    # Silently nulling it drops all progress with no signal; warn instead.
    if progress:
        import warnings

        warnings.warn(
            f"resolve_progress: progress={progress!r} is not a bool, None, "
            "ProgressLogger, or an object with a callable 'iteration'; "
            "treating it as silent (no progress). If you meant to tee "
            "output to a stream, that is a different argument.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _null_logger()


def _fmt_seconds(s: float) -> str:
    """Compact wall-clock formatter: ms / s / m+s, never negative width."""
    if s < 1.0:
        return f"{s * 1e3:6.1f}ms"
    if s < 60.0:
        return f"{s:6.2f}s"
    m, sec = divmod(s, 60.0)
    return f"{int(m)}m{sec:05.2f}s"
