"""``vibeqc.io.trajectory`` -- backward-compat shim.

The optimisation / vibrational trajectory writers' implementation
moved to :mod:`vibeqc.output.formats.trajectory` as part of the
pre-v1.0 output-module refactor (see
``docs/design_output_module.md``). This module re-exports the same
public API; ``from vibeqc.io.trajectory import ...`` and
``from vibeqc.io import ...`` (via ``io/__init__.py``) both continue
to work unchanged.
"""

from __future__ import annotations

from ..output.formats.trajectory import (
    normal_mode_trajectory,
    write_opt_trajectory,
    write_xyz_trajectory,
)


__all__ = [
    "write_xyz_trajectory",
    "write_opt_trajectory",
    "normal_mode_trajectory",
]
