"""``vibeqc.crash_dump`` -- backward-compat shim.

The implementation moved to :mod:`vibeqc.output.formats.crash_dump`
as part of the pre-v1.0 output-module refactor (see
``docs/design_output_module.md``). This module re-exports the same
public API; every existing import keeps working.
"""

from __future__ import annotations

from .output.formats.crash_dump import (
    active_crash_dump_stem,
    classify_failure,
    crash_dump_context,
    dump_on_failure,
    load_dump,
)


__all__ = [
    "active_crash_dump_stem",
    "classify_failure",
    "crash_dump_context",
    "dump_on_failure",
    "load_dump",
]
