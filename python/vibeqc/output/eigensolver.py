"""Portable diagnostics for operator solves; no file I/O in solver modules."""
from __future__ import annotations


def eigensolver_diagnostics(result):
    """JSON-compatible diagnostics and frame labels; unavailable data stay null."""
    def values(array):
        return None if array is None else [float(x) for x in array]
    metadata = result.operator_metadata
    operator_record = None if metadata is None else {
        "kind": metadata.kind,
        "basis_id": metadata.basis_id,
        "generation": int(metadata.generation),
        "spin": metadata.spin,
        "kpoint": values(metadata.kpoint),
        "symmetry_block": metadata.symmetry_block,
        "occupations": values(metadata.occupations),
    }
    if metadata is not None and (metadata.frame_id is not None or metadata.build_id is not None
                                 or metadata.role != "physical"):
        operator_record.update(frame_id=metadata.frame_id, build_id=metadata.build_id,
                               density_id=metadata.density_id, role=metadata.role,
                               modification=metadata.modification)
    return {
        "converged": bool(result.converged),
        "eigenvalues": values(result.eigenvalues),
        "residuals": values(result.residuals),
        "projected_residuals": values(result.projected_residuals),
        "residual_frame": result.residual_frame,
        "orthogonality_error": result.orthogonality_error,
        "retained_rank": result.retained_rank,
        "selection_verified": result.selection_verified,
        "iterations": int(result.n_iter),
        "operator_vector_applications": result.n_matvec,
        "operator_kind": getattr(metadata, "kind", None),
        "operator_metadata": operator_record,
    }


def write_solver_failure(path, error):
    """Atomically save a refused solve to an explicit diagnostic NPZ path.

    This writes only the exception's bounded snapshots and finite JSON status.
    It neither replays the operator nor creates restart/QVF/accepted-state data.
    Load with ``numpy.load(path, allow_pickle=False)``; ``diagnostics_json`` is
    a JSON string. An I/O failure propagates, preserving an existing archive.
    """
    import json
    import os
    from pathlib import Path
    import tempfile
    import numpy as np
    from ..solvers.operators import SolverConvergenceError

    if not isinstance(error, SolverConvergenceError):
        raise TypeError("Expected a refused operator solve")
    metadata = json.dumps(dict(error.diagnostics), allow_nan=False, sort_keys=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, diagnostics_json=np.array(metadata), **error.arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
