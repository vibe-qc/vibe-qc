"""``vibeqc.system_info`` -- backward-compat shim.

The implementation moved to :mod:`vibeqc.output.formats.system_info`
as part of the pre-v1.0 output-module refactor (see
``docs/design_output_module.md``). This module re-exports the same
public API; every existing import keeps working. New code should
prefer importing from ``vibeqc.output.formats.system_info`` or the
``vibeqc.output`` top-level once those re-exports land.
"""

from __future__ import annotations

from .output.formats.system_info import (
    _format_manifest,
    system_info,
    write_system_manifest,
)


__all__ = [
    "_format_manifest",
    "system_info",
    "write_system_manifest",
]
