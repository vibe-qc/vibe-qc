"""Analytically solvable model substrates for validating the embedding
Green's-function machinery before wiring the full 3D bulk Green function
(open design question 4)."""

from __future__ import annotations

from .tight_binding_1d import SemiInfiniteChain1D

__all__ = ["SemiInfiniteChain1D"]
