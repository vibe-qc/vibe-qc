"""Restricted source admission for representative finite-torus pair builds.

The caller supplies active and frozen occupied coefficients in an explicit
translation gauge. Their separate projectors must match the corresponding
source spaces; matching the union would permit an unintended core/active
rotation. Translation covariance is checked on the resulting pair spaces by
``TranslationPairSpaces`` before any factor transforms are evaluated.

This admission neither certifies factor covariance nor changes ``same_source``.
The pair builder audits arbitrary providers at every translated placement;
its sealed home-auxiliary implementation has constructive covariance.
Overlap and Fock may be dense snapshots or exact immutable cyclic operators.
Occupied projectors are compared and hashed in streamed AO row blocks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..correlation_conventions import effective_electron_count
from ..periodic.chi.pno import _validated_pair_mesh
from .source import CorrelationSource, _array_identity, detect_reference_kind
from .translation import _error, _readonly, _real_matrix
from .translation_operators import (
    CyclicAOOperator, matrix_apply, matrix_record, matrix_snapshot,
)
from .translation_projectors import BlochOccupiedProjectors, reference_orbitals, projector_error
from .translation_storage import (TranslationOccupiedColumns, occupied_input,
    occupied_orthogonality_error, occupied_projector_rows)

__all__ = ["TranslationSourceInputs", "admit_translation_source"]


@dataclass(frozen=True)
class TranslationSourceInputs:
    """Private tensor snapshots plus the source's unchanged factor provider.

    This object does not copy potentially compact factor storage or promise
    immutability of an external provider. Numerical input snapshots and
    conventions are immutable; provenance is an independent record for the
    builder to extend. Cyclic overlap/Fock operators retain their immutable
    home-row representation instead of materializing a square AO matrix.
    """

    overlap: np.ndarray | CyclicAOOperator
    fock: np.ndarray | CyclicAOOperator
    occupied: np.ndarray
    frozen_occupied: np.ndarray
    factors: Any
    source_kind: str
    source_identity: str
    same_source: bool
    conventions: Mapping[str, Any]
    n_electrons: int
    n_frozen: int
    reference_energy: float
    provenance: dict[str, Any]


def _require_orthonormal(coefficients, overlap, tolerance, name):
    if occupied_orthogonality_error(coefficients, overlap) > tolerance:
        raise ValueError(f"{name} must be S-orthonormal")


def _tensor_record(array):
    if type(array) is TranslationOccupiedColumns:
        return array.record()
    if array.size == 0:
        # No frozen bands is a valid (n_ao, 0) coefficient matrix. A zero
        # dimension cannot be cast through the shared memoryview hasher.
        return {"shape": list(array.shape), "sha256": sha256(b"").hexdigest()}
    return _array_identity(array)


def _projector_error(left, right, other_left, other_right, block_size):
    """Maximum AO-entry projector difference with one row block in memory.

    Each projector is factored as ``left @ right``. The entrywise norm is
    deliberately the same admission criterion as a dense AO projector;
    occupied-space angles or Frobenius norms would change its tolerance.
    """
    error = 0.0
    for start in range(0, left.shape[0], block_size):
        stop = start + block_size
        difference = left[start:stop] @ right - other_left[start:stop] @ other_right
        error = max(error, _error(difference))
    return error


def _projector_record(left, right, block_size):
    """Hash row-ordered AO projector entries without retaining the matrix."""
    digest = sha256()
    for start in range(0, left.shape[0], block_size):
        block = np.ascontiguousarray(left[start:start + block_size] @ right, dtype="<f8")
        if not np.isfinite(block).all():
            raise FloatingPointError("non-finite occupied projector")
        digest.update(memoryview(block).cast("B"))
    return {"shape": [left.shape[0], right.shape[1]], "sha256": digest.hexdigest()}


def admit_translation_source(
    source, mesh, occupied, *, frozen_occupied=None, tolerance=1e-10,
):
    """Validate a real restricted finite-torus source and its occupied split.

    ``frozen_occupied`` determines the number of frozen source columns. Its
    projector must match ``reference.mo_coeffs[:, :n_frozen]`` or the sealed
    compact frozen projector; the active projector independently matches
    the remaining occupied columns or the compact active projector. Both
    spaces contain complete bands of the requested finite translation group.
    Electron counts include the reference's existing ECP core metadata.

    This helper validates source consistency. The pair-space constructor
    additionally requires a positive metric, symmetric/covariant Fock and
    overlap, and the actual phase-free cell-major occupied gauge. It must be
    called before the retained factor provider is asked to transform tensors.
    """
    if not isinstance(source, CorrelationSource):
        raise TypeError("source must be a CorrelationSource")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("source tolerance must be a positive finite real number")
    tolerance = float(tolerance)
    mesh = _validated_pair_mesh(mesh)
    if source.conventions.kmesh is None or _validated_pair_mesh(source.conventions.kmesh) != mesh:
        raise ValueError("source kmesh must match the requested translation mesh")
    normalization = source.conventions.bloch_normalisation
    if normalization not in {
        "real-bvk-supercell", "unnormalised-ao-bloch-sums-uniform-weights",
    }:
        raise ValueError("source must carry an admitted finite-torus normalization")
    reference = source.reference
    if (source.reference_kind != "rhf" or detect_reference_kind(reference) != "rhf"
            or hasattr(reference, "mo_coeffs_alpha") or hasattr(reference, "mo_coeffs_beta")):
        raise ValueError("translation pair construction requires a restricted RHF source")
    if not bool(getattr(reference, "converged", False)):
        raise ValueError("translation pair construction requires a converged reference")
    energy = getattr(reference, "energy", None)
    if (isinstance(energy, (bool, np.bool_)) or not isinstance(energy, Real)
            or not np.isfinite(energy)):
        raise ValueError("reference energy must be a finite real number")
    if source.overlap is None:
        raise ValueError("translation source requires an explicit AO overlap")
    S = matrix_snapshot(source.overlap, mesh, name="source overlap")
    F = matrix_snapshot(reference.fock, mesh, name="source Fock")
    if (type(S) is CyclicAOOperator) != (type(F) is CyclicAOOperator):
        raise ValueError("source overlap and Fock must both use cyclic operators or both use dense matrices")
    orbitals = reference_orbitals(reference)
    compact = type(orbitals) is BlochOccupiedProjectors
    reference_coefficients = None if compact else _real_matrix(orbitals, "source orbitals")
    C = occupied_input(occupied, mesh, "active occupied coefficients")
    nbf = getattr(source.basis, "nbasis", None)
    if (isinstance(nbf, (bool, np.bool_)) or not isinstance(nbf, Integral)
            or nbf <= 0 or S.shape != (nbf, nbf) or F.shape != S.shape
            or (orbitals.active.shape[0] != nbf if compact else reference_coefficients.shape[0] != nbf)
            or C.shape[0] != nbf):
        raise ValueError("source tensors must match its AO basis dimensions")
    if source.molecule.multiplicity != 1:
        raise ValueError("restricted translation source must be a singlet")
    n_electrons = effective_electron_count(source.molecule, reference)
    if n_electrons <= 0 or n_electrons % 2:
        raise ValueError("restricted translation source needs a positive even electron count")
    n_occ = n_electrons // 2
    if (n_occ != orbitals.n_occupied if compact else n_occ > reference_coefficients.shape[1]) or n_occ >= nbf:
        raise ValueError("source electron count must leave occupied and virtual AO spaces")
    frozen = (np.empty((nbf, 0)) if frozen_occupied is None
              else occupied_input(frozen_occupied, mesh, "frozen occupied coefficients"))
    n_cells = prod(mesh)
    if (nbf % n_cells or frozen.shape[0] != nbf or C.shape[1] == 0
            or C.shape[1] % n_cells or frozen.shape[1] % n_cells
            or C.shape[1] + frozen.shape[1] != n_occ):
        raise ValueError("active and frozen source spaces require complete bands matching the electron count")
    n_frozen = frozen.shape[1]
    if compact:
        orbitals.validate(mesh, S, F, n_occ, n_frozen, tolerance)
    else:
        source_occupied = reference_coefficients[:, :n_occ]
        _require_orthonormal(source_occupied, S, tolerance, "source occupied orbitals")
    _require_orthonormal(C, S, tolerance, "active occupied coefficients")
    _require_orthonormal(frozen, S, tolerance, "frozen occupied coefficients")
    if occupied_orthogonality_error(C, S, frozen) > tolerance:
        raise ValueError("active and frozen occupied spaces must be orthogonal")
    # Exhaustively accumulate all AO entries over occupied column tiles.
    # Keep C.T S, including admitted within-tolerance metric asymmetry.
    block_size = nbf // n_cells
    errors, projector_records = [], []
    for selected, first, last, target in (
        (C, n_frozen, n_occ, orbitals.active if compact else None),
        (frozen, 0, n_frozen, orbitals.frozen if compact else None),
    ):
        error, digest = 0., sha256()
        for start in range(0, nbf, block_size):
            rows = np.arange(start, start+block_size)
            actual = occupied_projector_rows(selected, S, rows)
            expected = (target.block(rows, np.arange(nbf)) if compact else
                        occupied_projector_rows(source_occupied[:, first:last], S, rows))
            error = max(error, _error(actual-expected))
            digest.update(memoryview(np.ascontiguousarray(actual, dtype='<f8')).cast('B'))
        errors.append(error)
        projector_records.append(dict(shape=[nbf, nbf], sha256=digest.hexdigest()))
    active_error, frozen_error = errors
    if active_error > tolerance:
        raise ValueError("active occupied projector does not match the source active space")
    if frozen_error > tolerance:
        raise ValueError("frozen occupied projector does not match the source frozen space")
    conventions = dict(source.conventions.as_record())
    conventions["kmesh"] = mesh
    provenance = {
        "schema": "vibeqc.translation-source-inputs/v2",
        "source_kind": str(source.kind), "source_identity": str(source.identity),
        "source_provenance": deepcopy(source.provenance),
        "conventions": dict(conventions),
        "n_electrons": n_electrons, "n_frozen": n_frozen,
        "ecp_total_ncore": int(getattr(reference, "ecp_total_ncore", 0) or 0),
        "reference_energy": float(energy),
        "overlap": matrix_record(S), "fock": matrix_record(F),
        "source_orbitals": orbitals.record() if compact else _tensor_record(reference_coefficients),
        "active_occupied": _tensor_record(C),
        "frozen_occupied": _tensor_record(frozen),
        "projector_evaluation": ("streamed-frame-against-home-row-bloch-projectors" if compact
                                 else "streamed-row-blocks-C-times-C-transpose-S"),
        "active_projector_error": active_error, "frozen_projector_error": frozen_error,
        "active_projector": (matrix_record(orbitals.active) if compact
                             else projector_records[0]),
        "frozen_projector": (matrix_record(orbitals.frozen) if compact
                             else projector_records[1]),
    }
    return TranslationSourceInputs(
        overlap=S, fock=F, occupied=_occupied_snapshot(occupied, C, mesh),
        frozen_occupied=_occupied_snapshot(frozen_occupied, frozen, mesh), factors=source.factors,
        source_kind=str(source.kind), source_identity=str(source.identity),
        same_source=bool(source.conventions.same_source),
        conventions=MappingProxyType(conventions), n_electrons=n_electrons,
        n_frozen=n_frozen, reference_energy=float(energy), provenance=provenance,
    )


def _occupied_snapshot(original, dense, mesh):
    from .translation_storage import TranslationOccupiedColumns
    if type(original) is TranslationOccupiedColumns:
        if original.mesh != tuple(mesh):
            raise ValueError("selected occupied storage mesh mismatch")
        return original
    return _readonly(dense)
