"""``vibeqc.scf_log`` -- backward-compat shim.

The implementation moved to :mod:`vibeqc.output.formats.scf_log` as
part of the pre-v1.0 output-module refactor (see
``docs/design_output_module.md``). This module re-exports the same
public API; every existing import keeps working.
"""

from __future__ import annotations

from .output.formats.scf_log import (
    format_basis_summary,
    format_scf_trace,
    log_scf_trace,
    write_scf_trace,
)


__all__ = [
    "format_scf_trace",
    "write_scf_trace",
    "format_basis_summary",
    "log_scf_trace",
]
