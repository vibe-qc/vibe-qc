"""Explicit byte-read evidence for resource preflight, not whole-process tracing.

Uninstrumented reads and imported code are not certified by this context.
Callers must use the returned bytes, not reopen the file after recording it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path

_reads: ContextVar[list[str] | None] = ContextVar("resource_input_reads", default=None)


@contextmanager
def resource_input_reads():
    """Collect input bytes read through :func:`read_resource_input` in this scope."""
    token = _reads.set([])
    try:
        yield
    finally:
        _reads.reset(token)


def read_resource_input(path: str | Path) -> bytes:
    """Read bytes once and bind exactly those returned bytes to the active scope."""
    reads = _reads.get()
    if reads is None:
        raise RuntimeError("read_resource_input requires resource_input_reads")
    data = Path(path).read_bytes()
    reads.append(hashlib.sha256(data).hexdigest())
    return data


def resource_evidence(features: dict) -> dict:
    """Return resolved values and honest, explicitly instrumented input coverage."""
    result = {"schema": "vibeqc.resource-evidence/1",
              "input_coverage": "explicit_reads", "input_sha256": sorted(set(_reads.get() or []))}
    result.update({key: value for key, value in features.items() if value is not None})
    return result
