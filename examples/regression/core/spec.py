"""Per-system / per-method / per-reference dataclasses.

Re-exported from :mod:`vibeqc_spec`, the model's home since it ships in the
vibe-qc wheel for ``vqfetch``. Existing ``from ..core.spec import ...``
imports keep working and see the same classes.
"""
from __future__ import annotations

from vibeqc_spec import *  # noqa: F401,F403
from vibeqc_spec import __all__  # noqa: F401
