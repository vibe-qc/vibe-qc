"""The ambient output channel: where a module's text goes.

Before this module, any code that wanted to contribute a line to the
``.out`` file had to be handed the file object. ``runner.py`` and
``periodic_runner.py`` each opened one inside a long ``with`` block and
threaded ``f`` to every write site, so a writer could only emit text if
it lived close enough to that handle to see it. The IUPAC naming call
was inlined in the SCF runner for exactly that reason: it had nothing to
do with SCF, it just needed ``f``.

A channel inverts that. A module imports :func:`write` and calls it. The
runner decides, once, where the text lands::

    # any module, at any depth
    from vibeqc.output import write, flush

    write(f"  IUPAC name: {iupac}\\n")
    flush()

    # the runner, once
    with OutputChannel.to_file(out_path):
        ...

Outside an active channel :func:`write` is a no-op, so importing this
module opens nothing, and a library call or a unit test that emits text
with no channel installed neither raises nor writes a file.

Streaming, not buffering
------------------------

The channel is not a post-hoc formatter that accumulates a document and
renders it at the end. Every :func:`write` pushes immediately, and files
are opened line-buffered (``buffering=1``), so ``tail -f job.out`` shows
lines as the SCF emits them. That is the behaviour the hand-threaded
``f.write(...)`` / ``f.flush()`` pairs had, preserved exactly.

Verbosity
---------

Each message carries a :class:`Level` (``QUIET`` / ``STANDARD`` /
``VERBOSE`` / ``DEBUG``) and each channel a threshold; a message emits when
``level <= channel.level``. An untagged :func:`write` is ``STANDARD`` and
the channel default is ``STANDARD``, so the ``.out`` keeps exactly its
historical content and only opts into more (``VERBOSE`` / ``DEBUG``) or
less (``QUIET``) when the threshold is moved, via ``VIBEQC_OUTPUT_LEVEL``,
:func:`output_level`, or :meth:`OutputChannel.set_level`. This is the
``.out``'s own verbosity axis, distinct from ``ProgressLogger``, which
gates the *live terminal* stream.

Relationship to the rest of ``vibeqc.output``
---------------------------------------------

* :class:`~vibeqc.output.writer.OutputWriter` coordinates *which files a
  job produces* and owns the ``.system`` manifest. It is not a text
  stream and this module does not touch it.
* :mod:`vibeqc.output.document` decides *how a number renders*.
* This module decides *where the resulting text goes*.

Thread safety
-------------

The active channel lives in a :class:`contextvars.ContextVar`, matching
the pattern already used by ``structured_log._active``,
``perf.active_tracker`` and ``crash_dump.active_crash_dump_stem``. A
``ContextVar`` is per-context, not per-OS-thread: code running in a
thread spawned with :class:`threading.Thread` sees the *default*
(``None``) rather than the caller's channel, and its writes would be
silently dropped.

Parallelism is OpenMP inside C++ nearly everywhere, so this is sound by
default. The one Python thread site is the corrected-Ewald exchange k
loop (``periodic_corrected_exchange.k_space_terms_all_k``), which farms
over a ``ThreadPoolExecutor``; it is admissible only because nothing on
that call path emits. The guard test in
``tests/test_output_channel.py`` holds every Python thread site to that
standard through a reviewed allowlist, so a thread that *does* need to
emit fails there rather than losing its output silently. The allowlist
is the settled mechanism (2026-08-11); thread sites remain responsible
for emitting nothing unless explicitly allowlisted with a documented
reason.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import sys
import os
from pathlib import Path
from types import TracebackType
from typing import IO, Iterator, Optional, Sequence, Type, Union


__all__ = [
    "Level",
    "OutputChannel",
    "section",
    "section_header",
    "active_channel",
    "flush",
    "output_level",
    "note",
    "record",
    "strict_output",
    "warn",
    "write",
]


class Level(enum.IntEnum):
    """Verbosity band a message is tagged with, and the threshold a channel
    is set to. A message emits when ``message.level <= channel.level``.

    So a higher number means "needs a more verbose channel to appear". The
    bands, from always-shown to diagnostic-only:

    * ``QUIET`` -- essential results that must survive even ``--quiet``:
      the final energy, the converged/not-converged verdict, fatal errors.
    * ``STANDARD`` -- the normal ``.out`` content (the default). An
      untagged ``write(text)`` is ``STANDARD``.
    * ``VERBOSE`` -- extra detail a user opts into: per-shell basis dumps,
      full orbital tables, per-stage timings.
    * ``DEBUG`` -- developer diagnostics: intermediate norms, gauge checks.

    The channel's threshold defaults to ``STANDARD``, so ``QUIET`` and
    ``STANDARD`` messages appear and ``VERBOSE`` / ``DEBUG`` are suppressed
    until the channel is raised. Setting the channel to ``QUIET`` shows
    only ``QUIET`` messages; to ``DEBUG`` shows everything.
    """

    QUIET = 0
    STANDARD = 1
    VERBOSE = 2
    DEBUG = 3


_channel: contextvars.ContextVar[Optional["OutputChannel"]] = contextvars.ContextVar(
    "vibeqc_output_channel", default=None
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_LEVEL_NAMES = {lvl.name.lower(): lvl for lvl in Level}


def _level_from_env() -> Level:
    """Resolve ``VIBEQC_OUTPUT_LEVEL`` (quiet/standard/verbose/debug or 0-3).

    Unset or unrecognised -> ``STANDARD``, so the ``.out`` keeps its
    current content by default.
    """
    raw = os.environ.get("VIBEQC_OUTPUT_LEVEL", "").strip().lower()
    if not raw:
        return Level.STANDARD
    if raw in _LEVEL_NAMES:
        return _LEVEL_NAMES[raw]
    try:
        return Level(int(raw))
    except (ValueError, TypeError):
        return Level.STANDARD


# The default level a channel adopts when none is passed to its factory.
# Read once at import; ``output_level()`` overrides at runtime.
_DEFAULT_LEVEL = _level_from_env()


def output_level(level: Optional[Union["Level", int, str]] = None) -> "Level":
    """Get or set the *default* channel level (returns the resulting value).

    This is the level newly-constructed channels adopt when their factory
    is not given an explicit ``level=``. An already-open channel keeps the
    level it was built with; use :meth:`OutputChannel.set_level` for that.
    Accepts a :class:`Level`, an int 0-3, or a name
    (``"quiet"``/``"standard"``/``"verbose"``/``"debug"``). ``None`` reads.
    """
    global _DEFAULT_LEVEL
    if level is not None:
        _DEFAULT_LEVEL = _coerce_level(level)
    return _DEFAULT_LEVEL


def _coerce_level(level: Union["Level", int, str]) -> "Level":
    if isinstance(level, Level):
        return level
    if isinstance(level, str):
        key = level.strip().lower()
        if key in _LEVEL_NAMES:
            return _LEVEL_NAMES[key]
        raise ValueError(
            f"unknown output level {level!r}; expected one of "
            f"{', '.join(_LEVEL_NAMES)} or 0-3"
        )
    return Level(int(level))


def _strict_from_env() -> bool:
    return os.environ.get("VIBEQC_STRICT_OUTPUT", "").strip().lower() in _TRUTHY


# Evaluated once at import so the hot path is a bare attribute read, not an
# environ lookup per write. ``strict_output()`` flips it at runtime for a
# test or a CI job that wants the strict behaviour without a subprocess.
_STRICT = _strict_from_env()


def strict_output(enabled: Optional[bool] = None) -> bool:
    """Get or set strict mode. Returns the resulting value.

    In strict mode, :func:`write` / :func:`flush` **raise** when no
    channel is active instead of silently discarding. Off by default so
    production stays lenient (a writer must be callable from a notebook or
    a library import); on under ``VIBEQC_STRICT_OUTPUT=1`` or an explicit
    ``strict_output(True)`` so a test or CI job catches a "forgot to
    install a channel" bug -- the silent-loss failure mode a no-op write
    otherwise hides. Passing ``None`` (the default) only reads.
    """
    global _STRICT
    if enabled is not None:
        _STRICT = bool(enabled)
    return _STRICT


def active_channel() -> Optional["OutputChannel"]:
    """Return the channel currently installed, or ``None``."""
    return _channel.get()


def write(text: str, level: "Level" = Level.STANDARD) -> None:
    """Write ``text`` to the active channel, if its threshold allows.

    ``level`` tags the message's verbosity band (default
    :attr:`Level.STANDARD`, so an untagged call is normal ``.out``
    content). The active channel emits it only when
    ``level <= channel.level``; a ``VERBOSE`` line is silent on a
    ``STANDARD`` channel, a ``QUIET`` line survives even a ``QUIET``
    channel.

    A no-op when no channel is active -- deliberately, so a writer module
    stays callable from a test, a notebook, or a library context where
    nobody has opened an ``.out`` file. Under strict mode (see
    :func:`strict_output`) the no-op becomes a ``RuntimeError`` instead,
    so a lost-output bug fails loudly where it is cheap to catch.
    """
    channel = _channel.get()
    if channel is not None:
        channel.write(text, level)
    elif _STRICT:
        raise RuntimeError(
            "vibeqc.output.write() called with no active OutputChannel "
            "(VIBEQC_STRICT_OUTPUT is on). Install one with "
            "`with OutputChannel.to_file(path):` or drop strict mode."
        )


def flush() -> None:
    """Flush the active channel. A no-op when none is active (raises under
    strict mode, mirroring :func:`write`)."""
    channel = _channel.get()
    if channel is not None:
        channel.flush()
    elif _STRICT:
        raise RuntimeError(
            "vibeqc.output.flush() called with no active OutputChannel "
            "(VIBEQC_STRICT_OUTPUT is on)."
        )


def record(
    event: str,
    human: Optional[str] = None,
    *,
    level: "Level" = Level.STANDARD,
    once_key: Optional[str] = None,
    **fields: object,
) -> None:
    """Emit one fact to both surfaces at once.

    ``human`` text goes to the ``.out`` channel (via :func:`write`, at
    ``level``) and a structured ``event`` -- carrying ``fields`` -- goes to
    the active :class:`~vibeqc.output.StructuredLog`. One call, so the
    human-readable line and the machine-readable record cannot drift out of
    sync the way two separate ``write(...)`` + ``emit(...)`` calls can (the
    ``.out``-says-one-thing, ``.scf.jsonl``-says-another bug this exists to
    prevent). Each surface no-ops independently when it is inactive -- no
    channel installed, or no structured log -- so ``record`` is safe to call
    anywhere, like :func:`write`.

    ``human=None`` emits only the structured event (a machine-only record);
    pass a string to also write the human line. Include the trailing
    newline in ``human`` yourself, as with :func:`write`.

    ``once_key`` coalesces a repeated semantic fact for the lifetime of the
    active output channel and structured log. The first call wins; later
    calls with the same key emit neither surface, so the human and machine
    records stay in lockstep. A fresh channel/log starts a fresh scope. Use a
    stable semantic key rather than the rendered message, and use distinct
    keys for severities that must be allowed to escalate independently.
    """
    channel = None
    structured = None
    if once_key is not None:
        if not isinstance(once_key, str):
            raise TypeError("once_key must be a string or None")

        channel = _channel.get()
        from .formats.structured_log import active_structured_log

        structured = active_structured_log()
        if channel is not None and channel._has_once(once_key):
            return
        if structured is not None and structured._has_once(once_key):
            return

    if human is not None:
        write(human, level)
    # Lazy import keeps this core module from importing the formats layer at
    # load time; emit() is a no-op when no structured log is active.
    from .formats.structured_log import emit as _emit

    _emit(event, **fields)
    if once_key is not None:
        if channel is not None:
            channel._claim_once(once_key)
        if structured is not None:
            structured._claim_once(once_key)


def warn(
    message: str,
    *,
    once_key: Optional[str] = None,
    **fields: object,
) -> None:
    """Emit a warning: a ``  WARNING: {message}`` line **at QUIET** (so it
    survives even ``--quiet``) and a structured ``warning`` event.

    The line format matches the ``write("  WARNING: ...")`` the runners
    already scatter by hand, so migrating those to :func:`warn` is
    byte-neutral at the default level while adding the machine-collectable
    event and the survives-quiet guarantee. Extra ``fields`` ride on the
    structured event (e.g. ``warn("CIF write failed", role="cif")``).
    """
    record(
        "warning",
        f"  WARNING: {message}\n",
        level=Level.QUIET,
        once_key=once_key,
        message=message,
        **fields,
    )


def note(
    message: str,
    *,
    once_key: Optional[str] = None,
    **fields: object,
) -> None:
    """A non-warning aside: a ``  Note: {message}`` line at ``STANDARD`` and
    a structured ``note`` event. For information a user wants but that is
    not a problem (a fallback taken, a default filled in)."""
    record(
        "note",
        f"  Note: {message}\n",
        level=Level.STANDARD,
        once_key=once_key,
        message=message,
        **fields,
    )


def section_header(
    title: str,
    *,
    width: Optional[int] = None,
    rule: str = "-",
    indent: int = 2,
) -> str:
    """Return a ``title`` + underline-rule header string (no side effects).

    ``width=None`` sizes the rule to the title; an int fixes it. Shared by
    :func:`section` and by string-building callers that cannot use the
    context manager, so the header shape lives in one place instead of the
    57 hand-drawn ``"  " + "-" * N`` rules the runners used to carry.
    """
    pad = " " * indent
    n = len(title) if width is None else width
    return f"{pad}{title}\n{pad}{rule * n}\n"


@contextlib.contextmanager
def section(
    title: str,
    *,
    level: "Level" = Level.STANDARD,
    width: Optional[int] = None,
    rule: str = "-",
    indent: int = 2,
) -> Iterator[None]:
    """Emit a section header (title + underline rule) through the active
    channel, then yield for the block body.

    The body is the ordinary ``write()`` calls inside the ``with`` block;
    they follow the header. ``width=None`` sizes the rule to the title, an
    int fixes it (the runners' historical blocks used ``width=56`` / ``52``
    / ``54``, so migrating them stays byte-identical). The header carries
    ``level``; tag the body writes to match to gate a whole section to
    ``VERBOSE`` / ``DEBUG``.

    Example::

        with section("Convergence strategy", width=56):
            for line in strategy.log_lines():
                write(f"    {line}\\n")
    """
    write(section_header(title, width=width, rule=rule, indent=indent), level)
    yield


class OutputChannel:
    """A set of sinks that text is broadcast to, installed by ``with``.

    Construct through the factories rather than directly:
    :meth:`to_file`, :meth:`to_stdout`, :meth:`tee`, :meth:`to_stream`,
    :meth:`null`.

    The channel is only active inside its ``with`` block. Entering
    installs it in the :class:`~contextvars.ContextVar`; leaving restores
    whatever was there before, so channels nest. Only sinks the channel
    opened itself are closed on exit: ``sys.stdout`` and caller-supplied
    streams are left alone.

    ``write`` and ``flush`` remain available for ordinary channel control,
    but the channel is not passed down as a file handle. Persistent output is
    ambient; live terminal output is independently owned by
    :class:`vibeqc.progress.ProgressLogger`.
    """

    def __init__(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        mode: str = "w",
        encoding: str = "utf-8",
        stdout: bool = False,
        stream: Optional[IO[str]] = None,
        level: Optional[Union["Level", int, str]] = None,
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a'; got {mode!r}")
        self._path = Path(path) if path is not None else None
        self._mode = mode
        self._encoding = encoding
        self._stdout = stdout
        self._stream = stream
        self._level = _DEFAULT_LEVEL if level is None else _coerce_level(level)
        self._file: Optional[IO[str]] = None
        self._token: Optional[contextvars.Token] = None
        self._once_keys: set[str] = set()
        self._entered = False
        self._closed = False

    # -- factories ----------------------------------------------------

    @classmethod
    def to_file(
        cls,
        path: Union[str, Path],
        *,
        mode: str = "w",
        encoding: str = "utf-8",
        level: Optional[Union["Level", int, str]] = None,
    ) -> "OutputChannel":
        """Line-buffered file sink. ``mode="a"`` appends to an existing file."""
        return cls(path=path, mode=mode, encoding=encoding, level=level)

    @classmethod
    def to_stdout(
        cls, *, level: Optional[Union["Level", int, str]] = None
    ) -> "OutputChannel":
        """Live terminal sink."""
        return cls(stdout=True, level=level)

    @classmethod
    def tee(
        cls,
        path: Union[str, Path],
        *,
        mode: str = "w",
        encoding: str = "utf-8",
        level: Optional[Union["Level", int, str]] = None,
    ) -> "OutputChannel":
        """File and stdout at once."""
        return cls(path=path, mode=mode, encoding=encoding, stdout=True, level=level)

    @classmethod
    def to_stream(
        cls, stream: IO[str], *, level: Optional[Union["Level", int, str]] = None
    ) -> "OutputChannel":
        """An arbitrary caller-owned stream. Useful in tests (``StringIO``)."""
        return cls(stream=stream, level=level)

    @classmethod
    def null(cls) -> "OutputChannel":
        """Discards everything. The behaviour of no channel at all."""
        return cls()

    # -- properties ---------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def active(self) -> bool:
        return self._entered and not self._closed

    @property
    def level(self) -> "Level":
        """The verbosity threshold: messages with ``level <= this`` emit."""
        return self._level

    def set_level(self, level: Union["Level", int, str]) -> None:
        """Raise or lower this channel's threshold at runtime."""
        self._level = _coerce_level(level)

    def writable(self) -> bool:  # stream protocol
        return True

    def _claim_once(self, key: str) -> bool:
        """Claim ``key`` in this channel's semantic-event scope.

        Returns ``True`` only for the first claim. Kept private because
        callers express coalescing through ``record(..., once_key=...)``;
        the channel, not feature code, owns the suppression state.
        """
        if key in self._once_keys:
            return False
        self._once_keys.add(key)
        return True

    def _has_once(self, key: str) -> bool:
        """Whether ``key`` is already claimed in this channel."""
        return key in self._once_keys

    # -- stream protocol ----------------------------------------------

    def _sinks(self) -> Iterator[IO[str]]:
        if self._file is not None:
            yield self._file
        if self._stdout:
            # Resolved per write, not captured at __enter__: pytest's
            # capsys swaps sys.stdout after the channel is installed.
            yield sys.stdout
        if self._stream is not None:
            yield self._stream

    def write(self, text: str, level: "Level" = Level.STANDARD) -> int:
        """Broadcast ``text`` to every sink if ``level <= self.level``.

        Returns ``len(text)`` whether or not the text was emitted, so a
        suppressed message still satisfies the ``io.TextIOBase.write``
        contract that ``print(..., file=channel)`` and some stream
        consumers expect (they check the returned count). ``level``
        defaults to :attr:`Level.STANDARD`, so a bare ``channel.write(s)``
        -- including ``print(..., file=channel)`` -- is normal content.
        """
        if self._closed:
            raise RuntimeError("write() on a closed OutputChannel")
        if int(level) > int(self._level):
            return len(text)  # below the channel threshold: suppressed
        for sink in self._sinks():
            sink.write(text)
        if self._stdout:
            # The file is line-buffered; stdout may be block-buffered when
            # piped, so push it explicitly to keep the tee live.
            sys.stdout.flush()
        return len(text)

    def writelines(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        if self._closed:
            return
        for sink in self._sinks():
            sink.flush()

    def close(self) -> None:
        """Close sinks this channel opened. Idempotent."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._closed = True

    # -- context manager ----------------------------------------------

    def __enter__(self) -> "OutputChannel":
        if self._entered:
            raise RuntimeError("OutputChannel is not reentrant")
        if self._closed:
            raise RuntimeError("OutputChannel has already been closed")
        if self._path is not None:
            # buffering=1 is line buffering, so `tail -f` sees each line
            # as it is emitted rather than at the next 8 KiB boundary.
            self._file = open(
                self._path, self._mode, encoding=self._encoding, buffering=1
            )
        self._entered = True
        self._token = _channel.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # Reset the ContextVar before closing, so a sink failure on close
        # cannot leave a dead channel installed for the rest of the run.
        if self._token is not None:
            _channel.reset(self._token)
            self._token = None
        self._entered = False
        self.close()
        return None  # never suppress an exception
