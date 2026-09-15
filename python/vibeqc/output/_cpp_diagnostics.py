"""Bridge: route C++ diagnostics into ``vibeqc.output``.

C++ kernels call ``vibeqc::diagnostic()`` or the ``VIBEQC_DIAG``
macro.  This module installs a callback that forwards those messages
into the active :class:`OutputChannel` via :func:`vibeqc.output.write`,
so C++ diagnostics appear in the ``.out`` file and obey the same
verbosity level as Python-side output.

Usage
-----
Call :func:`install_diagnostics_bridge` once, before any computation
that may emit C++ diagnostics.  The runners do this automatically; a
user exercising the C++ core directly can call it manually::

    from vibeqc.output._cpp_diagnostics import install_diagnostics_bridge
    install_diagnostics_bridge()

After installation the C++ verbosity threshold tracks
:func:`vibeqc.output.output_level` automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps

from .channel import Level
from .channel import output_level as _py_output_level
from .channel import write as _channel_write

_log = logging.getLogger("vibeqc.output._cpp_diagnostics")

_bridge_installed = False

# Optional callback for progress-tagged diagnostics. Each execution context
# owns its handler so concurrent jobs cannot route native rows into another
# job's terminal, structured log, or manifest.
_progress_handler: ContextVar[
    Callable[[dict[str, object]], None] | None
] = ContextVar("vibeqc_cpp_progress_handler", default=None)


def _diag_callback(tag: str, level_int: int, msg: str) -> None:
    """The callback registered with C++.  Routes ``(tag, level_int, msg)``
    into the ambient ``OutputChannel``.  A trailing newline is appended
    automatically so C++ call sites never need to include one.

    When ``tag == "progress"`` and a progress handler is installed, the
    message is parsed as ``key=value`` pairs and forwarded to the handler
    for live terminal, structured-log, and manifest updates **regardless of
    the channel verbosity level**. Progress tracking is always-on
    infrastructure rather than an ambient ``.out`` message."""
    try:
        # A handled progress message is consumed by the live-output funnel.
        # With no handler, keep the historical silent behavior; the native
        # result trace remains available to direct callers after return.
        if tag == "progress" and _progress_handler.get() is not None:
            _forward_progress(msg)
        elif tag != "progress":
            _channel_write(f"[C++/{tag}] {msg}\n", Level(level_int))
        else:
            _channel_write(f"[C++/{tag}] {msg}\n", Level.DEBUG)
    except Exception:
        _log.debug(
            "C++ diagnostic dropped: channel write failed", exc_info=True
        )


def _forward_progress(msg: str) -> None:
    """Parse ``key=value`` pairs from a progress message and forward
    to the registered progress handler."""
    fields: dict[str, object] = {"phase": "scf"}
    for token in msg.split():
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        try:
            fields[key] = int(val)
        except ValueError:
            try:
                fields[key] = float(val)
            except ValueError:
                fields[key] = val
    handler = _progress_handler.get()
    if handler is not None:
        try:
            handler(fields)
        except Exception:
            _log.debug("progress handler failed", exc_info=True)


def _sync_level_to_cpp(level: Level | None = None) -> None:
    """Push the current Python output level to the C++ diagnostics threshold."""
    try:
        from .._vibeqc_core import (  # type: ignore[attr-defined]
            _set_diagnostics_level,
            _get_diagnostics_level,
        )
    except ImportError:
        return  # compiled core not available (e.g. docs build)
    lvl = level if level is not None else _py_output_level()
    try:
        current = _get_diagnostics_level()
        if current != int(lvl):
            _set_diagnostics_level(int(lvl))
    except Exception:
        _log.debug("Failed to sync C++ diagnostics level", exc_info=True)


# ---- patched output_level --------------------------------------------------

_original_output_level = _py_output_level


@wraps(_original_output_level)
def _patched_output_level(level: Level | int | str | None = None) -> Level:
    """Drop-in replacement for :func:`vibeqc.output.output_level` that
    also syncs the C++ diagnostics threshold."""
    result = _original_output_level(level)
    if level is not None:
        _sync_level_to_cpp(result)
    return result


def install_diagnostics_bridge() -> bool:
    """Register the C++ diagnostics callback and link the verbosity level.

    Idempotent — safe to call more than once.  Returns ``True`` on the
    first successful installation, ``False`` on subsequent calls or when
    the compiled core is unavailable.

    Called automatically by :func:`vibeqc.runner.run_job` and
    :func:`vibeqc.periodic_runner.run_periodic_job`; ordinary users do
    not need to call this explicitly.
    """
    global _bridge_installed

    if _bridge_installed:
        return False

    try:
        from .._vibeqc_core import _set_diagnostics_sink  # type: ignore[attr-defined]
    except ImportError:
        _log.debug("_vibeqc_core not available; C++ diagnostics bridge skipped")
        return False

    try:
        # Register the Python callback as the C++ sink.
        _set_diagnostics_sink(_diag_callback)

        # Sync the current Python level to C++.
        _sync_level_to_cpp()

        # Patch ``vibeqc.output.output_level`` so raising the Python
        # verbosity also raises the C++ threshold.  We must patch both
        # ``channel.output_level`` (the canonical definition) and
        # ``vibeqc.output.output_level`` (the re-export — ``__init__.py``
        # captures a reference at import time that would otherwise stay
        # stale).
        from . import channel as _chan
        import vibeqc.output as _out

        _chan.output_level = _patched_output_level  # type: ignore[attr-defined]
        _out.output_level = _patched_output_level  # type: ignore[attr-defined]

        _bridge_installed = True
        _log.debug("C++ diagnostics bridge installed")
        return True
    except Exception:
        _log.debug("C++ diagnostics bridge installation failed", exc_info=True)
        return False


def install_progress_handler(
    handler: Callable[[dict[str, object]], None] | None,
) -> None:
    """Register a callback that receives parsed progress messages from C++.

    When a C++ kernel emits ``VIBEQC_DIAG("progress", ...)`` with
    ``key=value`` pairs, the bridge parses the message and calls
    ``handler(fields)``.  Fields are typed: ``iter`` → int,
    ``energy`` / ``dE`` / ``grad`` → float, and ``diis`` → int.

    High-level runners use the callback to flush the terminal progress row,
    structured ``scf_iter`` event, and ``[progress]`` manifest heartbeat at
    the instant a native iteration completes.

    Call with ``None`` to clear.
    """
    _progress_handler.set(handler)
