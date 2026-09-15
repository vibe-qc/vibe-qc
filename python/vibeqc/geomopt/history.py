"""Restart and checkpoint infrastructure (Phase 6 -- production hardening).

:class:`HistoryManager` records every accepted step (geometry, energy,
gradient) during an optimisation run.

:class:`RestartSerializer` saves/loads the full optimiser state to a
checkpoint file so a crashed or interrupted optimisation can resume
from the last accepted step without re-running SCF evaluations.

Format
------
Checkpoints are written as JSON with binary blobs base64-encoded for
portability.  For large systems (>500 atoms) an HDF5 variant can be
added (the protocol is format-agnostic via ``serialize``/``deserialize``
methods).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .state import GeomOptResult, OptimizerState

# ---------------------------------------------------------------------------
# History manager
# ---------------------------------------------------------------------------


@dataclass
class HistoryRecord:
    """One accepted step in the optimisation history."""

    iteration: int
    energy: float
    gradient_flat: np.ndarray
    x_flat: np.ndarray
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        gmax = (
            float(np.max(np.abs(self.gradient_flat)))
            if self.gradient_flat.size
            else 0.0
        )
        return (
            f"HistoryRecord(iter={self.iteration}, E={self.energy:.8f}, "
            f"max|g|={gmax:.4e})"
        )


class HistoryManager:
    """Records per-step history during an optimisation run.

    Call :meth:`record` after each accepted step.  The history is
    a simple list of :class:`HistoryRecord` objects -- no disk I/O.
    Use :class:`RestartSerializer` for persistence.

    Attributes
    ----------
    records : list[HistoryRecord]
        Chronological per-step records.
    best : HistoryRecord or None
        The lowest-energy record seen so far.
    """

    def __init__(self, max_records: int = 1000):
        self._records: list[HistoryRecord] = []
        self._best: Optional[HistoryRecord] = None
        self._max_records = max_records

    @property
    def records(self) -> list[HistoryRecord]:
        return list(self._records)

    @property
    def best(self) -> Optional[HistoryRecord]:
        return self._best

    def record(
        self, iteration: int, energy: float, gradient: np.ndarray, x: np.ndarray
    ) -> None:
        """Record one accepted step."""
        rec = HistoryRecord(
            iteration=iteration,
            energy=energy,
            gradient_flat=np.asarray(gradient, dtype=float).ravel().copy(),
            x_flat=np.asarray(x, dtype=float).ravel().copy(),
        )
        self._records.append(rec)
        if self._best is None or energy < self._best.energy:
            self._best = rec

        # Trim if over budget
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    def last(self) -> Optional[HistoryRecord]:
        """Most recent record, or None if empty."""
        return self._records[-1] if self._records else None

    def to_state(self) -> dict[str, Any]:
        """Export history to a JSON-serialisable dict."""
        recs = []
        for r in self._records:
            recs.append(
                {
                    "iteration": r.iteration,
                    "energy": r.energy,
                    "gradient": _array_to_b64(r.gradient_flat),
                    "x": _array_to_b64(r.x_flat),
                    "timestamp": r.timestamp,
                }
            )
        return {"records": recs}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "HistoryManager":
        """Restore a HistoryManager from a JSON dict."""
        mgr = cls()
        for r in state.get("records", []):
            mgr._records.append(
                HistoryRecord(
                    iteration=r["iteration"],
                    energy=r["energy"],
                    gradient_flat=_b64_to_array(r["gradient"]),
                    x_flat=_b64_to_array(r["x"]),
                    timestamp=r.get("timestamp", time.time()),
                )
            )
        if mgr._records:
            mgr._best = min(mgr._records, key=lambda r: r.energy)
        return mgr


# ---------------------------------------------------------------------------
# Restart serializer
# ---------------------------------------------------------------------------


class RestartSerializer:
    """Save and load optimisation checkpoints.

    A checkpoint captures:
    * The current optimiser parameters (x vector).
    * The current energy and gradient.
    * The iteration count and evaluation count.
    * The full step history.
    * Optimiser-specific state (e.g. L-BFGS s/y lists, Hessian model, trust radius).
    * The convergence policy used.
    * Metadata (timestamp, optimizer name, method).

    Checkpoints are written atomically (write to temp file, then rename)
    to avoid corruption on crash.

    Parameters
    ----------
    checkpoint_path : str or Path
        File path for the checkpoint (``.json`` extension recommended).
    """

    def __init__(self, checkpoint_path: str | os.PathLike):
        self._path = str(checkpoint_path)

    @property
    def path(self) -> str:
        return self._path

    def exists(self) -> bool:
        return os.path.isfile(self._path)

    def save(self, state: dict[str, Any]) -> None:
        """Atomically write *state* dict to the checkpoint file."""
        serialised = _serialise_state(state)
        tmp = self._path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(serialised, fh, indent=2)
        os.replace(tmp, self._path)

    def load(self) -> dict[str, Any]:
        """Load the checkpoint dict.  Raises FileNotFoundError if absent."""
        with open(self._path) as fh:
            return _deserialise_state(json.load(fh))

    def delete(self) -> None:
        """Remove the checkpoint file (no-op if absent)."""
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _array_to_b64(arr: np.ndarray) -> str:
    """Encode a numpy array as base64."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_array(b64: str) -> np.ndarray:
    """Decode a base64-encoded numpy array."""
    return np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False)


_NP_SCALARS = (np.integer, np.floating, np.bool_)


class _StateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return _array_to_b64(obj)
        if isinstance(obj, _NP_SCALARS):
            return obj.item()
        return super().default(obj)


def _serialise_state(state: dict[str, Any]) -> dict[str, Any]:
    """Recursively serialise a state dict for JSON."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _serialise_state(v)
        elif isinstance(v, list):
            out[k] = [
                _serialise_state(e)
                if isinstance(e, dict)
                else _array_to_b64(e)
                if isinstance(e, np.ndarray)
                else e.item()
                if isinstance(e, _NP_SCALARS)
                else e
                for e in v
            ]
        elif isinstance(v, np.ndarray):
            out[k] = _array_to_b64(v)
        elif isinstance(v, _NP_SCALARS):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _deserialise_state(state: dict[str, Any]) -> dict[str, Any]:
    """Recursively deserialise a JSON state dict back to numpy arrays."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _deserialise_state(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [_deserialise_state(e) for e in v]
        elif isinstance(v, list) and v and isinstance(v[0], str):
            # Try to decode as arrays; if fails, keep as strings
            try:
                out[k] = [_b64_to_array(s) for s in v]
            except Exception:
                out[k] = v
        elif isinstance(v, str) and len(v) > 100 and v[0] not in '"{[':
            # Heuristic: long strings are likely base64 arrays
            try:
                out[k] = _b64_to_array(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def build_restart_state(
    state: OptimizerState,
    *,
    optimizer: str = "",
    method: str = "",
    convergence_policy: Optional[Any] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a restart state dict from the current optimiser state.

    Parameters
    ----------
    state : OptimizerState
        Current optimiser working state.
    optimizer : str
        Optimiser name (e.g. ``"bfgs"``).
    method : str
        Electronic-structure method (e.g. ``"rhf"``).
    convergence_policy : ConvergencePolicy or None
    extra : dict or None
        Optimiser-specific state (e.g. L-BFGS s/y lists, Hessian matrix,
        trust radius).
    """
    restart: dict[str, Any] = {
        "x": state.x,
        "energy": state.energy,
        "gradient": state.gradient,
        "iteration": state.iteration,
        "n_energy_evals": state.n_energy_evals,
        "optimizer": optimizer,
        "method": method,
        "timestamp": time.time(),
        "convergence": (
            {
                "gmax": convergence_policy.gmax,
                "grms": convergence_policy.grms,
                "dmax": convergence_policy.dmax,
                "drms": convergence_policy.drms,
                "ediff": convergence_policy.ediff,
            }
            if convergence_policy is not None
            else None
        ),
    }
    if extra:
        restart["extra"] = extra
    return restart


def restore_optimizer_state(
    restart: dict[str, Any],
) -> tuple[OptimizerState, dict[str, Any]]:
    """Restore an OptimizerState from a restart dict.

    Returns (state, extra) where *extra* is the optimiser-specific
    state (Hessian, s/y lists, trust radius, etc.).
    """
    state = OptimizerState(
        x=restart["x"],
        energy=restart["energy"],
        gradient=restart["gradient"],
        iteration=restart["iteration"],
        n_energy_evals=restart.get("n_energy_evals", 0),
    )
    return state, restart.get("extra", {})
