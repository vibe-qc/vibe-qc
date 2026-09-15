"""Python-side basis helpers for semiempirical methods.

Provides convenience functions for querying the minimal STO-NG basis
that DFTB0 constructs from the parameter set.

Stage 1: delegates to C++ ``SemiempiricalBasis`` (not yet directly
exposed via pybind11 -- the basis is built inside ``run_dftb0``).
"""

from __future__ import annotations
