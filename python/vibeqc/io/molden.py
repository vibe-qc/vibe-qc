"""``vibeqc.io.molden`` -- backward-compat shim.

The molden writer's implementation moved to
:mod:`vibeqc.output.formats.molden` as part of the pre-v1.0
output-module refactor (see ``docs/design_output_module.md``). This
module re-exports the same public function (``write_molden``); the
``vibeqc.io.__init__`` re-export still works through the shim, so
``from vibeqc.io.molden import write_molden`` and
``from vibeqc.io import write_molden`` both keep functioning.
"""

from __future__ import annotations

from ..output.formats.molden import write_molden


__all__ = ["write_molden"]
