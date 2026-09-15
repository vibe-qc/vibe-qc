"""``vibeqc.structured_log`` -- backward-compat shim.

The implementation moved to
:mod:`vibeqc.output.formats.structured_log` as part of the pre-v1.0
output-module refactor (see ``docs/design_output_module.md`` for the
trajectory). This module re-exports the same public API verbatim so
callers that still import from ``vibeqc.structured_log`` keep working
without changes -- the move is internal-organisation only; the
contract is unchanged.

Going forward, new code should import from
``vibeqc.output.formats.structured_log`` (or the top-level
``vibeqc.output`` re-exports). This shim stays in place for the
v0.8.x -> v1.0 transition; whether it's removed at v1.0 or kept
forever is a separate decision the maintainer makes at release time.
"""

from __future__ import annotations

from .output.formats.structured_log import (
    StructuredLog,
    active_structured_log,
    emit,
    run_fingerprint,
    structured_log,
)


__all__ = [
    "StructuredLog",
    "active_structured_log",
    "emit",
    "run_fingerprint",
    "structured_log",
]
