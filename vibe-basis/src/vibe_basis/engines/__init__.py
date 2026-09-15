"""Concrete :class:`~vibe_basis.engine.EnergyEngine` implementations.

Tier 1 only -- every engine here drives an **external** program through
a transport and needs nothing from vibe-qc. The primary engine
(``VibeQcEngine``) is tier 2 and lives in
``python/vibeqc/basis_optimization/``; see ``vibe_basis.engine`` for why
that inversion is deliberate.
"""

from .crystal23 import Crystal23Engine
from .gpaw import GpawEngine

__all__ = ["Crystal23Engine", "GpawEngine"]
