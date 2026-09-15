"""Spike-local re-export — see ``vibeqc.properties.hirshfeld_charges``.

The classical Hirshfeld implementation that started life in this spike
was promoted into production in ``python/vibeqc/properties.py`` on
2026-05-18 (sibling of ``mulliken_charges`` / ``loewdin_charges``).
This module is kept as a thin re-export so the spike's other drivers
(``compare_charges.py``, ``d4_with_hirshfeld_charges.py``,
``sensitivity_estimate.py``) keep working without rewrites.

New code should import from the production location:

    from vibeqc.properties import hirshfeld_charges, HirshfeldResult
    # or, top-level:
    from vibeqc import hirshfeld_charges, HirshfeldResult
"""

from __future__ import annotations

from vibeqc.properties import HirshfeldResult, hirshfeld_charges

__all__ = ["HirshfeldResult", "hirshfeld_charges"]
