"""Optimiser registry -- shared factory for all geometry optimisers.

This module is deliberately minimal (no imports from sibling modules)
so every optimiser can use :func:`register` without circular imports.
"""

from __future__ import annotations

from typing import Any, Callable

_OPTIMIZERS: dict[str, Callable[..., Any]] = {}


def register(name: str):
    """Decorator that registers an optimiser function under *name*."""

    def deco(fn):
        _OPTIMIZERS[name] = fn
        return fn

    return deco


def resolve(name: str) -> Callable[..., Any]:
    """Return the optimiser function for *name* (``"sd"``, ``"cg"``, ...)."""
    try:
        return _OPTIMIZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown geom_opt={name!r}.  Available: {sorted(_OPTIMIZERS)}"
        ) from None
