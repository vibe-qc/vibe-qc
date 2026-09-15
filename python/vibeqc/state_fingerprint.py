"""Stable-state fingerprint for open-shell SCF runs (BUG 88).

Computes a deterministic SHA-256 hash identifying the converged electronic
state from energy, <S^2>, electron counts, and (optionally) spin populations.
Two runs converging to the same state produce identical fingerprints.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


def compute_state_fingerprint(
    *,
    energy: float,
    n_alpha: int,
    n_beta: int,
    s_squared: float,
    spin_populations: Optional[np.ndarray] = None,
    molecule_symbols: Optional[list[str]] = None,
) -> str:
    parts: list[str] = [
        f"n_alpha={n_alpha}",
        f"n_beta={n_beta}",
        f"s2={s_squared:.6f}",
        f"energy={energy:.8f}",
    ]
    if molecule_symbols:
        parts.append("atoms=" + ",".join(molecule_symbols))
    if spin_populations is not None:
        pop_parts = ",".join(
            f"{float(p):.6f}" for p in np.asarray(spin_populations)
        )
        parts.append(f"spin_pop={pop_parts}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
