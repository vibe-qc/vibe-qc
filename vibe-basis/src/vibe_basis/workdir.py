"""Session working-directory resolution for vibe-basis.

When vibe-basis runs under the vibe-queue (vq) daemon, ``$VQ_WORKDIR``
is set to a clean per-job scratch directory (outside any git checkout).
This module resolves that directory and provides a convenience for
creating per-iteration subdirectories.

The contract matches the vq agent-interaction protocol
(vibe-queue/docs/agent_interaction.md):

    import os
    workdir = os.environ["VQ_WORKDIR"]

    per_iter_dir = os.path.join(workdir, f"iter_{i:04d}")
    os.makedirs(per_iter_dir, exist_ok=True)

When running outside vq (``$VQ_WORKDIR`` unset), ``resolve_workdir``
falls back to a temporary directory (``tempfile.mkdtemp``) so that
optimization scratch never lands in the git checkout even in local
development.

Public API
----------
``resolve_workdir() -> Path``
    Return the absolute session working directory. Respects
    ``$VQ_WORKDIR``; falls back to a new ``tempfile.mkdtemp``.

``iter_dir(workdir, index) -> Path``
    Create (if needed) and return ``workdir / f"iter_{index:04d}"``.
    Convenience for per-iteration subdirectories.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def resolve_workdir() -> Path:
    """Return the absolute session working directory.

    Resolution order:

    1. ``$VQ_WORKDIR`` — set by the vq daemon when dispatching a job.
       An absolute path to a clean per-job scratch directory outside
       any git checkout (e.g. ``/var/lib/vq/users/<uid>/workdirs/<jobid>/``
       on remote compute hosts).
    2. ``tempfile.mkdtemp(prefix="vb-")`` — for local development and
       tests. Ensures vibration-basis never writes scratch into the
       git checkout even when running outside vq.

    Returns
    -------
    Path
        An existing, writable directory.
    """
    if "VQ_WORKDIR" in os.environ:
        p = Path(os.environ["VQ_WORKDIR"]).resolve()
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
        return p

    return Path(tempfile.mkdtemp(prefix="vb-"))


def iter_dir(workdir: str | Path, index: int, *, mkdir: bool = True) -> Path:
    """Return ``workdir / f"iter_{index:04d}"``, creating it if requested.

    Parameters
    ----------
    workdir
        The session working directory (e.g. from :func:`resolve_workdir`).
    index
        Iteration number (0-based).
    mkdir
        If True (default), create the directory (including parents) if
        it doesn't exist.

    Returns
    -------
    Path
    """
    d = Path(workdir) / f"iter_{index:04d}"
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = ["resolve_workdir", "iter_dir"]
