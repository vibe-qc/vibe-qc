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
