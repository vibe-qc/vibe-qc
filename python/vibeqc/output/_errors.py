"""Structured non-fatal output-failure diagnostics.

Optional artefact writers (QVF, cube, population, citations, manifest
recording) may fail without aborting a successful SCF.  When they do,
the failure must be **visible** -- via a structured warning, an entry
in the ``.system`` manifest's ``[[outputs.files]]`` (with
``written=false`` and an ``error`` key), and a log message -- rather
than silently disappearing into ``except Exception: pass``.

Categories
----------
Each recorded failure carries a ``category`` tag drawn from
:class:`OutputFailureKind`:

* ``"optional_artifact"`` -- the writer produced no file at all
  (QVF, cube, population, bibtex, .xyz, ...).
* ``"manifest_recording"`` -- the file was written but manifest
  bookkeeping (hash / stat / TOML rewrite) failed.
* ``"cleanup"`` -- post-job cleanup / finalization failed.
* ``"compatibility_fallback"`` -- a best-effort compatibility shim
  (ASE trajectory energy probe, optional sub-section probe for QVF).
* ``"unrecoverable"`` -- something that should have been fatal but got
  caught by a broad handler (treated as a bug; surfaced loudly).

Public API
----------
``warn_output_failure(exc, path, role, *, category, manifest)``
    Emit the failure to warnings + optional manifest.

``OutputFailureRecord``
    Lightweight dataclass for structured error information.

Stability
---------
This module's name starts with an underscore: the failure-handling
plumbing is a private implementation detail of the output layer.
Callers inside :mod:`vibeqc.runner` / :mod:`vibeqc.periodic_runner`
import it directly; external code should not depend on the name.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .manifest import ManifestUpdater  # noqa: F401
    from .writer import OutputWriter  # noqa: F401


_log = logging.getLogger("vibeqc.output")


class OutputFailureKind(Enum):
    """Categorisation of a non-fatal output failure."""

    #: The optional writer produced no output file.
    optional_artifact = "optional_artifact"

    #: The file was written but manifest bookkeeping failed.
    manifest_recording = "manifest_recording"

    #: Post-job cleanup / finalization failed.
    cleanup = "cleanup"

    #: Compatibility shim -- best-effort, expected to fail on
    #: some platform / version combinations.
    compatibility_fallback = "compatibility_fallback"

    #: Something that should have been fatal but got caught.
    #: Treated as a bug; logged at ERROR.
    unrecoverable = "unrecoverable"


@dataclass(frozen=True)
class OutputFailureRecord:
    """Immutable record of a non-fatal output failure."""

    path: Path
    role: str
    category: OutputFailureKind
    exc_type: str
    exc_message: str

    def warning_message(self) -> str:
        return (
            f"vibe-qc output [{self.category.value}]: "
            f"{self.role} on {self.path.name} failed with "
            f"{self.exc_type}: {self.exc_message}"
        )

    def manifest_error_string(self) -> str:
        """Short ``ExceptionType: message`` string for the manifest
        ``[[outputs.files]].error`` cell."""
        return f"{self.exc_type}: {self.exc_message}"


def warn_output_failure(
    exc: BaseException,
    path: Path,
    role: str,
    *,
    category: OutputFailureKind,
    manifest: Optional["ManifestUpdater"] = None,
) -> OutputFailureRecord:
    """Record and surface a non-fatal output failure.

    Parameters
    ----------
    exc:
        The caught exception.
    path:
        Path the writer was targeting.
    role:
        Human-readable role name (e.g. ``"qvf_archive"``,
        ``"population_summary"``, ``"manifest_record"``).
    category:
        One of the :class:`OutputFailureKind` values.
    manifest:
        If provided, the failure is also recorded in the manifest so
        downstream consumers (``vq fetch``, ``.system`` readers) can
        see that this artefact was *declared* but not *written*.

    Returns
    -------
    OutputFailureRecord
        The structured record (also returned so callers can assert in
        tests).
    """
    record = OutputFailureRecord(
        path=Path(path),
        role=role,
        category=category,
        exc_type=type(exc).__name__,
        exc_message=str(exc),
    )

    # Always emit a structured warning so the user sees it on stderr
    # (covers the case where logging is unconfigured) and a logger
    # entry so structured-log consumers see it too.
    warnings.warn(record.warning_message(), stacklevel=3)
    if category is OutputFailureKind.unrecoverable:
        _log.error(record.warning_message(), exc_info=exc)
    else:
        _log.warning(record.warning_message())

    # Optionally record in the manifest so the failure is visible
    # to ``vq`` / downstream tooling, not just the console.
    if manifest is not None:
        manifest.mark_failed(record.path, record.manifest_error_string())

    return record


def warn_writer_failure(
    exc: BaseException,
    path: Path,
    role: str,
    *,
    category: OutputFailureKind,
    writer: Optional["OutputWriter"] = None,
) -> OutputFailureRecord:
    """Convenience: like :func:`warn_output_failure` but accepts an
    :class:`~vibeqc.output.writer.OutputWriter` instead of a raw
    manifest. Use from molecular ``runner.run_job``."""
    manifest = writer._manifest if writer is not None else None  # type: ignore[attr-defined]
    return warn_output_failure(
        exc, Path(path), role, category=category, manifest=manifest,
    )
