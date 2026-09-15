"""``vibeqc.perf`` -- backward-compat shim.

The implementation moved to :mod:`vibeqc.output.formats.perf` as
part of the pre-v1.0 output-module refactor (see
``docs/design_output_module.md``). This module re-exports the same
public API so existing callers keep working without changes -- the
move is internal-organisation only; the contract is unchanged.

Going forward, new code should import from
``vibeqc.output.formats.perf`` (or the top-level ``vibeqc.output``
re-exports once those land). This shim stays in place for the
v0.8.x -> v1.0 transition.
"""

from __future__ import annotations

from .output.formats.perf import (
    PerfScope,
    PerfTracker,
    active_tracker,
    format_perf_report,
    perf_log,
)


__all__ = [
    "PerfScope",
    "PerfTracker",
    "active_tracker",
    "format_perf_report",
    "perf_log",
]
